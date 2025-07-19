"""
MedBillGuard Agent Server - HTTP server for agent management and monitoring.

Provides REST API endpoints for health checks, metrics collection, agent registration,
and workflow execution. Designed for production deployment with K8s integration.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from config.env_config import config, check_required_config

# Import agent-related components
from agents.medical_bill_agent import MedicalBillAgent
from agents.agent_registry import AgentRegistry, TaskCapability, AgentCapabilities, AgentStatus
from agents.interfaces import AgentContext, AgentResult, ModelHint
from agents.router_agent import RouterAgent, RoutingStrategy, RoutingRequest
from agents.redis_state import RedisStateManager
from agents.tools.enhanced_router_agent import EnhancedRouterAgent
# from agents.simple_router import SimpleDocumentRouter, DocumentType  # Removed, file deleted
from security.oauth2_endpoints import oauth2_router
from security.auth_middleware import auth_manager

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
try:
    request_count = Counter('medbillguard_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
    request_duration = Histogram('medbillguard_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
    active_agents = Gauge('medbillguard_active_agents', 'Number of active agents', ['agent_type'])
    analysis_count = Counter('medbillguard_analysis_total', 'Total medical bill analyses', ['verdict', 'agent_id'])
    analysis_duration = Histogram('medbillguard_analysis_duration_seconds', 'Analysis duration', ['agent_id'])
    logger.info("Prometheus metrics initialized successfully")
except ValueError as e:
    logger.warning(f"Prometheus metrics already registered: {e}")
    # Get existing metrics or set to None
    from prometheus_client import REGISTRY
    
    request_count = None
    request_duration = None
    active_agents = None
    analysis_count = None
    analysis_duration = None
    
    for collector in REGISTRY._collector_to_names.keys():
        if hasattr(collector, '_name'):
            if collector._name == 'medbillguard_requests_total':
                request_count = collector
            elif collector._name == 'medbillguard_request_duration_seconds':
                request_duration = collector
            elif collector._name == 'medbillguard_active_agents':
                active_agents = collector
            elif collector._name == 'medbillguard_analysis_total':
                analysis_count = collector
            elif collector._name == 'medbillguard_analysis_duration_seconds':
                analysis_duration = collector
    
    logger.info(f"Using existing metrics: request_count={request_count is not None}, analysis_count={analysis_count is not None}")
except Exception as e:
    logger.error(f"Failed to initialize Prometheus metrics: {e}")
    # Set all metrics to None if initialization fails
    request_count = None
    request_duration = None
    active_agents = None
    analysis_count = None
    analysis_duration = None

# Add this helper function after the imports and before the app_state definition
async def ensure_agent_registration():
    """Ensure all agents are properly registered. Called on startup and health checks."""
    try:
        if not app_state["registry"]:
            logger.warning("Registry not available for registration")
            return False
            
        # In the new on-demand system, create and register medical agent if needed
        if not app_state.get("medical_agent"):
            logger.info("Creating medical agent for registration")
            
            # Create medical agent on-demand
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OpenAI API key not found, cannot create medical agent")
                return False
                
            app_state["medical_agent"] = MedicalBillAgent(
                redis_url=config.redis_url,
                openai_api_key=openai_api_key
            )
        
        # Check if medical agent is registered
        agent_status = await app_state["registry"].get_agent_status(app_state["medical_agent"].agent_id)
        
        if not agent_status or agent_status.status.value not in ["online", "degraded"]:
            logger.info("Medical agent not registered or offline, registering now")
            
            # Create capabilities for medical agent
            capabilities = AgentCapabilities(
                supported_tasks=[
                    TaskCapability.MEDICAL_ANALYSIS,
                    TaskCapability.RATE_VALIDATION,
                    TaskCapability.DUPLICATE_DETECTION,
                    TaskCapability.PROHIBITED_DETECTION,
                    TaskCapability.CONFIDENCE_SCORING
                ],
                max_concurrent_requests=config.max_workers * 2,
                preferred_model_hints=[ModelHint.STANDARD, ModelHint.PREMIUM],
                processing_time_ms_avg=config.timeout_seconds * 1000,
                cost_per_request_rupees=0.50, # Placeholder
                confidence_threshold=config.ocr_confidence_threshold,
                supported_document_types=config.supported_formats,
                supported_languages=config.ocr_languages
            )
            
            # Register the agent
            registration_success = await app_state["registry"].register_agent(
                agent=app_state["medical_agent"],
                capabilities=capabilities
            )
            
            if registration_success:
                logger.info("Medical agent successfully registered", agent_id=app_state["medical_agent"].agent_id)
                return True
            else:
                logger.error("Failed to register medical agent")
                return False
        else:
            logger.debug("Medical agent already registered", status=agent_status.status.value)
            return True
            
    except Exception as e:
        logger.error("Error during agent registration", error=str(e), exc_info=True)
        return False

async def start_background_registration_monitor():
    """Background task to monitor and re-register agents periodically."""
    while not app_state["shutdown_event"].is_set():
        try:
            await asyncio.sleep(120)  # Check every 2 minutes
            
            if app_state["registry"] and app_state["medical_agent"]:
                await ensure_agent_registration()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Background registration monitor error", error=str(e))
            await asyncio.sleep(60)  # Wait 1 minute before retrying

# Add this coroutine to send heartbeats
async def send_agent_heartbeat():
    while not app_state["shutdown_event"].is_set():
        try:
            if app_state.get("registry") and app_state.get("medical_agent"):
                await app_state["registry"]._update_agent_status(
                    app_state["medical_agent"].agent_id,
                    AgentStatus.ONLINE
                )
        except Exception as e:
            logger.error("Failed to send agent heartbeat", error=str(e))
        await asyncio.sleep(120)  # 2 minutes


# Global state
app_state = {
    "registry": None,  # Keep for compatibility, but not used in simple routing
    "simple_router": None,  # New clean router
    "redis_manager": None,
    "startup_time": None,
    "shutdown_event": asyncio.Event()
}

# In-memory conversation storage
conversations = {}


# Request/Response models
class HealthResponse(BaseModel):
    status: str
    timestamp: float
    uptime_seconds: float
    version: str = "1.0.0"
    components: Dict[str, str]


class AnalysisRequest(BaseModel):
    file_content: str = Field(..., description="Base64 encoded file content")
    doc_id: str = Field(..., description="Unique document identifier")
    user_id: str = Field(..., description="User identifier")
    language: str = Field(default="english", description="Document language")
    state_code: Optional[str] = Field(default=None, description="State code for regional rates")
    insurance_type: str = Field(default="cghs", description="Insurance type")
    file_format: str = Field(default="pdf", description="File format")
    query: Optional[str] = Field(default=None, description="Optional user query about the document")


class AnalysisResponse(BaseModel):
    success: bool
    doc_id: str
    analysis_complete: bool
    verdict: str
    total_bill_amount: float
    total_overcharge: float


class EnhancedAnalysisRequest(BaseModel):
    file_content: str = Field(..., description="Base64 encoded file content")
    doc_id: str = Field(..., description="Unique document identifier")
    user_id: str = Field(..., description="User identifier")
    language: str = Field(default="english", description="Document language")
    file_format: Optional[str] = Field(default=None, description="File format hint")
    routing_strategy: str = Field(default="capability_based", description="Routing strategy")
    priority: str = Field(default="normal", description="Processing priority")
    query: Optional[str] = Field(default=None, description="Optional query to ask about the document")


class EnhancedAnalysisResponse(BaseModel):
    success: bool
    doc_id: str
    document_type: str
    processing_stages: Dict[str, Any]
    final_result: Dict[str, Any]
    total_processing_time_ms: int
    error: Optional[str] = None
    query_response: Optional[str] = None


class MetricsResponse(BaseModel):
    timestamp: float
    active_agents: int
    total_analyses: int
    average_confidence: float
    uptime_seconds: float


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: float = Field(..., description="Unix timestamp")


class ChatRequest(BaseModel):
    doc_id: str = Field(..., description="Document identifier for context")
    user_id: str = Field(..., description="User identifier")
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(default=[], description="Previous conversation history")


class ChatResponse(BaseModel):
    success: bool
    doc_id: str
    message: str
    conversation_id: str
    timestamp: float


def create_database_tables_startup():
    """Create database tables during FastAPI startup using synchronous approach to avoid event loop issues"""
    try:
        logger.info("Starting database table creation process during FastAPI startup...")
        
        # Check if DATABASE_URL is available
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning("DATABASE_URL not found in environment - using in-memory fallback")
            return False
        
        logger.info(f"Database URL configured: {db_url[:50]}...")
        
        # Import database components - use synchronous approach for FastAPI lifespan
        from database.models import Base
        from sqlalchemy import create_engine, text
        
        # Create a synchronous engine for table creation during startup
        # Convert async PostgreSQL URL to sync URL
        sync_db_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
        logger.info(f"Using synchronous database URL for startup: {sync_db_url[:50]}...")
        
        # Create synchronous engine
        sync_engine = create_engine(sync_db_url)
        logger.info("Synchronous database engine created for startup")
        
        # Create tables using synchronous approach
        logger.info("Creating database tables using synchronous approach...")
        Base.metadata.create_all(sync_engine)
        logger.info("Database tables created successfully")
        
        # Verify table creation with a simple query
        logger.info("Verifying table creation...")
        with sync_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection verified")
        
        # Close the synchronous engine
        sync_engine.dispose()
        logger.info("Synchronous database engine disposed")
        
        return True
    except Exception as e:
        logger.error(f"Error creating database tables during FastAPI startup: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage FastAPI application lifespan."""
    # Startup
    logger.info("Starting MedBillGuard Agent Server")
    
    # Initialize app_state with None values first
    app_state["redis_manager"] = None
    app_state["registry"] = None
    app_state["router"] = None
    app_state["enhanced_router"] = None
    app_state["medical_agent"] = None
    app_state["shutdown_event"] = asyncio.Event()
    
    # Initialize Redis Manager with retry logic
    for attempt in range(3):
        try:
            app_state["redis_manager"] = RedisStateManager(config.redis_url)
            await app_state["redis_manager"].connect()
            logger.info("Redis connected successfully", attempt=attempt + 1)
            break
        except Exception as e:
            logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                await asyncio.sleep(5)  # Wait 5 seconds before retry
            else:
                logger.error("All Redis connection attempts failed, continuing without Redis")
                app_state["redis_manager"] = None
    
    # Initialize Agent Registry with retry logic
    if app_state["redis_manager"]:
        for attempt in range(3):
            try:
                app_state["registry"] = AgentRegistry(redis_url=config.redis_url)
                await app_state["registry"].start()
                logger.info("Agent registry initialized successfully", attempt=attempt + 1)
                break
            except Exception as e:
                logger.warning(f"Agent registry initialization attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(5)  # Wait 5 seconds before retry
                else:
                    logger.error("All agent registry initialization attempts failed")
                    app_state["registry"] = None
    
    # Initialize Enhanced Router
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            app_state["enhanced_router"] = EnhancedRouterAgent(
                registry=app_state.get("registry"),
                redis_url=config.redis_url,
                openai_api_key=openai_api_key
            )
            logger.info("Enhanced router initialized successfully")
        else:
            logger.warning("OpenAI API key not found, enhanced router not initialized")
            app_state["enhanced_router"] = None
    except Exception as e:
        logger.error(f"Enhanced router initialization failed: {e}")
        app_state["enhanced_router"] = None
    
    # Initialize OAuth2 Manager
    try:
        await auth_manager.initialize()
        logger.info("OAuth2 manager initialized")
    except Exception as e:
        logger.warning(f"OAuth2 manager initialization failed: {e}")
    
    # Note: Agents are now created on-demand by SimpleDocumentRouter
    # No need for complex initialization and registration logic
    
    # Start background tasks
    try:
        asyncio.create_task(update_metrics_background())
        asyncio.create_task(start_background_registration_monitor())
        logger.info("Background tasks started successfully")
    except Exception as e:
        logger.warning(f"Background tasks failed to start: {e}")
    
    # Initialize database tables
    try:
        logger.info("Starting database table creation during FastAPI startup...")
        create_database_tables_startup()
        logger.info("Database table creation completed during FastAPI startup")
    except Exception as e:
        logger.error(f"Database table creation failed during FastAPI startup: {e}")
        logger.info("Continuing with in-memory fallback for bill storage")
    
    # Record startup time
    app_state["startup_time"] = time.time()
    logger.info("Server startup completed successfully")

    # Ensure the medical agent is registered at startup
    await ensure_agent_registration()

    # Attach the live agent instance to the in-memory registration
    if app_state.get("registry") and app_state.get("medical_agent"):
        agent_id = app_state["medical_agent"].agent_id
        reg = await app_state["registry"].get_agent_status(agent_id)
        if reg:
            # Attach the agent instance to the registration
            reg.agent_instance = app_state["medical_agent"]
            # Update the cache with the modified registration
            app_state["registry"]._agent_cache[agent_id] = reg
            logger.info(f"Agent instance attached successfully for {agent_id}")
        else:
            logger.error(f"Failed to get agent status for {agent_id}, cannot attach agent instance")
    else:
        logger.error("Registry or medical_agent not available for agent instance attachment")

    # Start the heartbeat background task
    heartbeat_task = asyncio.create_task(send_agent_heartbeat())

    yield
    
    # On shutdown, signal and cancel the heartbeat task
    app_state["shutdown_event"].set()
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    # Shutdown
    logger.info("Shutting down MedBillGuard Agent Server")
    
    # Signal shutdown to background tasks
    app_state["shutdown_event"].set()
    
    # Cleanup Redis
    try:
        if app_state["redis_manager"]:
            await app_state["redis_manager"].close()
            logger.info("Redis connection closed")
    except Exception as e:
        logger.warning(f"Redis cleanup failed: {e}")
    
    # Cleanup Registry
    try:
        if app_state["registry"]:
            await app_state["registry"].stop()
            logger.info("Agent registry cleaned up")
    except Exception as e:
        logger.warning(f"Registry cleanup failed: {e}")
    
    logger.info("Server shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="MedBillGuard Agent Server",
    description="Production-ready multi-agent system for medical bill analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include OAuth2 router
app.include_router(oauth2_router)

# Initialize auth manager on startup
@app.on_event("startup")
async def startup_event():
    await auth_manager.initialize()

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware to collect request metrics."""
    start_time = time.time()
    method = request.method
    path = request.url.path
    
    # Skip metrics collection if metrics are not initialized
    if not all([request_count, request_duration]):
        response = await call_next(request)
        return response
    
    try:
        response = await call_next(request)
        
        # Record metrics
        status_code = response.status_code
        duration = time.time() - start_time
        
        # Update metrics safely
        if request_count and hasattr(request_count, 'labels'):
            request_count.labels(
                method=method,
                endpoint=path,
                status=str(status_code)
            ).inc()
        
        if request_duration and hasattr(request_duration, 'labels'):
            request_duration.labels(
                method=method,
                endpoint=path
            ).observe(duration)
        
        return response
        
    except Exception as e:
        # Record error metrics
        if request_count and hasattr(request_count, 'labels'):
            request_count.labels(
                method=method,
                endpoint=path,
                status="500"
            ).inc()
        
        raise e


# Health check endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for K8s readiness probe."""
    current_time = time.time()
    uptime = current_time - app_state["startup_time"] if app_state["startup_time"] else 0
    
    # Check component health
    components = {}
    
    try:
        # Redis health
        if app_state["redis_manager"]:
            await app_state["redis_manager"].ping()
            components["redis"] = "healthy"
        else:
            components["redis"] = "unhealthy"
    except Exception:
        components["redis"] = "unhealthy"
    
    try:
        # RouterAgent health check for cold-poke functionality
        from agents.router_agent import router_agent
        if router_agent:
            router_health = await router_agent.healthz()
            components["router_agent"] = router_health.get("status", "unknown")
            # Add detailed router health info as JSON string to satisfy HealthResponse model
            import json
            router_details = {
                "ttl_seconds": router_health.get("ttl_seconds", 0),
                "active_workflows": router_health.get("active_workflows", 0),
                "agent_registry_healthy": router_health.get("agent_registry_status", {}).get("healthy", False),
                "issues_count": len(router_health.get("issues", []))
            }
            components["router_agent_details"] = json.dumps(router_details)
        else:
            components["router_agent"] = "unhealthy"
    except Exception as e:
        logger.error("RouterAgent health check failed", error=str(e))
        components["router_agent"] = "unhealthy"
    
    # Determine overall status
    status = "healthy" if all(c == "healthy" for c in components.values()) else "unhealthy"
    
    return HealthResponse(
        status=status,
        timestamp=current_time,
        uptime_seconds=uptime,
        components=components
    )


@app.get("/health/liveness")
async def liveness_probe():
    """Simple liveness probe for K8s."""
    return {"status": "alive", "timestamp": time.time()}


@app.get("/health/readiness")
async def readiness_probe():
    """Readiness probe for K8s - checks if ready to serve traffic."""
    try:
        # For local development without Redis, we're ready if we can serve fallback analysis
        if not app_state["redis_manager"] and not app_state["registry"]:
            logger.info("Readiness check: Redis and registry not available, using fallback mode")
            return {"status": "ready", "timestamp": time.time(), "mode": "fallback"}
        
        # Production mode: check Redis and registry
        if app_state["redis_manager"]:
            await app_state["redis_manager"].ping()
        
        # Check if medical agent is registered, if not try to register it
        if app_state["registry"] and app_state["medical_agent"]:
            online_agents = await app_state["registry"].list_online_agents()
            medical_agent_online = any(
                a.agent_id == app_state["medical_agent"].agent_id 
                for a in online_agents
            )
            
            if not medical_agent_online:
                logger.warning("Medical agent not registered, attempting re-registration")
                registration_success = await ensure_agent_registration()
                if registration_success:
                    logger.info("Medical agent successfully re-registered during readiness check")
                else:
                    logger.error("Failed to re-register medical agent during readiness check")
        
        return {"status": "ready", "timestamp": time.time(), "mode": "full"}
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        # For local development, still return ready if basic components are available
        return {"status": "ready", "timestamp": time.time(), "mode": "fallback"}


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary", response_model=MetricsResponse)
async def metrics_summary():
    """Get aggregated metrics summary."""
    current_time = time.time()
    uptime = current_time - app_state["startup_time"] if app_state["startup_time"] else 0
    
    # Get total analysis count across all labels
    try:
        total_analyses = 0
        if analysis_count and hasattr(analysis_count, 'collect'):
            for sample in analysis_count.collect()[0].samples:
                total_analyses += sample.value
    except Exception:
        total_analyses = 0
    
    # Get active agents count
    try:
        active_agents_count = 0
        if active_agents and hasattr(active_agents, '_value'):
            active_agents_count = int(active_agents._value)
    except Exception:
        active_agents_count = 0
    
    return MetricsResponse(
        timestamp=current_time,
        active_agents=active_agents_count,
        total_analyses=int(total_analyses),  # Cast to int to avoid validation error
        average_confidence=0.85,  # Default placeholder
        uptime_seconds=uptime
    )


# Debug endpoints
@app.get("/debug/redis-keys")
async def debug_redis_keys():
    """Debug endpoint to check Redis keys and agent data."""
    try:
        if not app_state.get("redis_manager"):
            return {"error": "Redis manager not available"}
        
        redis_client = app_state["redis_manager"].redis_client
        if not redis_client:
            return {"error": "Redis client not connected"}
        
        # Get all keys
        all_keys = await redis_client.keys("*")
        all_keys = [key.decode() if isinstance(key, bytes) else key for key in all_keys]
        
        # Get agent registry keys specifically
        agent_keys = await redis_client.keys("agent_registry:*")
        agent_keys = [key.decode() if isinstance(key, bytes) else key for key in agent_keys]
        
        # Get agent data
        agent_data = {}
        for key in agent_keys:
            try:
                value = await redis_client.get(key)
                if value:
                    agent_data[key] = value.decode() if isinstance(value, bytes) else value
            except Exception as e:
                agent_data[key] = f"Error: {str(e)}"
        
        return {
            "total_keys": len(all_keys),
            "all_keys": all_keys,
            "agent_registry_keys": agent_keys,
            "agent_data": agent_data
        }
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}


@app.get("/debug/agent-cache")
async def debug_agent_cache():
    """Debug endpoint to check agent registry cache."""
    try:
        if not app_state.get("registry"):
            return {"error": "Agent registry not available"}
        
        registry = app_state["registry"]
        
        # Force cache refresh
        await registry._refresh_cache()
        
        # Get cache contents
        cache_data = {}
        for agent_id, registration in registry._agent_cache.items():
            agent_dict = asdict(registration)
            if hasattr(registration, "agent_instance"):
                agent_dict["agent_instance"] = registration.agent_instance
            else:
                agent_dict["agent_instance"] = None
            cache_data[agent_id] = agent_dict
        
        return {
            "cache_size": len(registry._agent_cache),
            "cache_last_update": registry._cache_last_update,
            "cache_data": cache_data
        }
    except Exception as e:
        return {"error": f"Cache debug failed: {str(e)}"}


@app.get("/debug/discover-test")
async def debug_discover_test():
    """Debug endpoint to test agent discovery."""
    try:
        if not app_state.get("registry"):
            return {"error": "Agent registry not available"}
        
        registry = app_state["registry"]
        
        # Test discovery
        from agents.agent_registry import TaskCapability
        from agents.base_agent import ModelHint
        
        required_capabilities = [TaskCapability.DOCUMENT_PROCESSING, TaskCapability.RATE_VALIDATION]
        
        discovered_agents = await registry.discover_agents(
            required_capabilities=required_capabilities,
            model_hint=ModelHint.STANDARD,
            max_agents=5
        )
        
        agents_info = []
        for agent in discovered_agents:
            agents_info.append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "status": agent.status.value,
                "capabilities": [task.value for task in agent.capabilities.supported_tasks],
                "model_hints": [hint.value for hint in agent.capabilities.preferred_model_hints]
            })
        
        return {
            "required_capabilities": [cap.value for cap in required_capabilities],
            "discovered_count": len(discovered_agents),
            "agents": agents_info
        }
    except Exception as e:
        return {"error": f"Discovery test failed: {str(e)}"}


@app.get("/debug/app-state")
async def debug_app_state():
    """Debug endpoint to check application state."""
    try:
        state_info = {}
        for key, value in app_state.items():
            if value is None:
                state_info[key] = "None"
            else:
                state_info[key] = {
                    "type": type(value).__name__,
                    "available": True
                }
        
        return {
            "app_state": state_info,
            "redis_connected": app_state.get("redis_manager") is not None,
            "registry_available": app_state.get("registry") is not None,
            "router_available": app_state.get("router") is not None
        }
    except Exception as e:
        return {"error": f"App state debug failed: {str(e)}"}


@app.get("/debug/test-simple")
async def test_simple():
    """Simple test endpoint."""
    return {"status": "working", "message": "Server is responding"}


@app.get("/debug/test-router-availability")
async def test_router_availability():
    """Test endpoint to verify RouterAgent is working with cold-poke functionality."""
    try:
        from agents.router_agent import router_agent
        if not router_agent:
            return {
                "status": "error",
                "message": "RouterAgent not available"
            }
        
        # Test router health with cold-poke functionality
        health_result = await router_agent.healthz()
        
        # Test routing decision capability
        from agents.router_agent import RoutingRequest, RoutingStrategy
        from agents.interfaces import TaskCapability, ModelHint
        
        test_routing_request = RoutingRequest(
            doc_id="test-123",
            user_id="test-user",
            task_type="medical_bill_analysis",
            required_capabilities=[TaskCapability.MEDICAL_ANALYSIS, TaskCapability.RATE_VALIDATION],
            model_hint=ModelHint.COST_OPTIMIZED,
            routing_strategy=RoutingStrategy.CAPABILITY_BASED,
            max_agents=1,
            timeout_seconds=30,
            priority=5,
            metadata={"test": True}
        )
        
        routing_decision = await router_agent.route_request(test_routing_request)
        
        return {
            "status": "success",
            "message": "RouterAgent is working with cold-poke functionality",
            "router_health": health_result,
            "test_routing": {
                "selected_agents": [agent.agent_id for agent in routing_decision.selected_agents],
                "routing_strategy": routing_decision.routing_strategy.value,
                "confidence": routing_decision.confidence,
                "estimated_cost": routing_decision.estimated_cost_rupees,
                "estimated_time_ms": routing_decision.estimated_time_ms
            },
            "cold_poke_status": {
                "health_ttl_seconds": health_result.get("ttl_seconds", 0),
                "active_workflows": health_result.get("active_workflows", 0),
                "agent_registry_healthy": health_result.get("agent_registry_status", {}).get("healthy", False)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"RouterAgent test failed: {str(e)}"
        }


@app.get("/debug/test-openai")
async def test_openai():
    """Test OpenAI API connection."""
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return {"status": "error", "message": "OpenAI API key not found in environment"}
        
        # Test OpenAI API call
        import openai
        client = openai.AsyncOpenAI(api_key=openai_api_key)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello' if you can read this."}],
            max_tokens=10,
            temperature=0.3
        )
        
        return {
            "status": "success",
            "message": "OpenAI API is working",
            "response": response.choices[0].message.content,
            "api_key_length": len(openai_api_key),
            "api_key_prefix": openai_api_key[:7] + "..." if len(openai_api_key) > 7 else "too_short"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"OpenAI API test failed: {str(e)}",
            "api_key_configured": openai_api_key is not None,
            "api_key_length": len(openai_api_key) if openai_api_key else 0
        }


@app.get("/debug/tesseract-info")
async def tesseract_diagnostic():
    """Diagnostic endpoint to check Tesseract OCR availability."""
    try:
        import pytesseract
        from PIL import Image
        
        # Check Tesseract version
        try:
            version = pytesseract.get_tesseract_version()
            version_str = str(version)
        except Exception as e:
            return {
                "success": False,
                "error": f"Tesseract not available: {e}",
                "timestamp": time.time()
            }
        
        # Check available languages
        try:
            languages = pytesseract.get_languages()
        except Exception as e:
            languages = f"Error getting languages: {e}"
        
        # Create a simple test image with text
        try:
            # Create a simple test image
            test_image = Image.new('RGB', (300, 100), color='white')
            
            # Try to draw some text (we'll use a simple approach)
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(test_image)
            try:
                # Try to use default font
                font = ImageFont.load_default()
                draw.text((10, 30), "TEST OCR 123", fill='black', font=font)
            except Exception:
                # Fallback: draw without font
                draw.text((10, 30), "TEST OCR 123", fill='black')
            
            # Test OCR on this simple image
            try:
                ocr_result = pytesseract.image_to_string(test_image).strip()
                ocr_success = len(ocr_result) > 0
                ocr_confidence = None
                
                # Try to get confidence data
                try:
                    data = pytesseract.image_to_data(test_image, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    ocr_confidence = sum(confidences) / len(confidences) if confidences else 0
                except Exception as conf_e:
                    ocr_confidence = f"Error getting confidence: {conf_e}"
                
            except Exception as ocr_e:
                ocr_result = f"OCR failed: {ocr_e}"
                ocr_success = False
                ocr_confidence = None
                
        except Exception as img_e:
            ocr_result = f"Image creation failed: {img_e}"
            ocr_success = False
            ocr_confidence = None
        
        return {
            "success": True,
            "tesseract_version": version_str,
            "available_languages": languages,
            "ocr_test": {
                "success": ocr_success,
                "result": ocr_result,
                "confidence": ocr_confidence
            },
            "timestamp": time.time()
        }
        
    except ImportError as e:
        return {
            "success": False,
            "error": f"Required modules not available: {e}",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {e}",
            "timestamp": time.time()
        }


# Analysis endpoints
@app.post("/analyze-fallback", response_model=AnalysisResponse)
async def analyze_fallback(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Fallback analysis endpoint that bypasses the router entirely."""
    start_time = time.time()
    
    logger.info(
        "Starting fallback analysis",
        doc_id=request.doc_id,
        user_id=request.user_id,
        file_format=request.file_format
    )
    
    try:
        # Decode base64 content
        try:
            import base64
            content = base64.b64decode(request.file_content).decode('utf-8')
            logger.info(f"Decoded content: {content[:100]}...")
        except Exception as e:
            logger.warning(f"Failed to decode base64 content: {e}")
            content = "Medical bill content"
        
        # Simple mock analysis for testing
        processing_time = time.time() - start_time
        
        # Handle user query if provided
        query_response = None
        if request.query:
            try:
                # Store document context for chat
                conversation_id = f"{request.doc_id}_{request.user_id}"
                if conversation_id not in conversations:
                    conversations[conversation_id] = {
                        "doc_id": request.doc_id,
                        "user_id": request.user_id,
                        "messages": [],
                        "document_context": content,
                        "created_at": time.time()
                    }
                else:
                    conversations[conversation_id]["document_context"] = content
                
                # Get OpenAI client
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Generate response to user query
                query_prompt = f"""You are a medical bill analysis assistant. A user has uploaded a medical bill and asked a question about it.

Document Content:
{content[:1500]}...

User Question: {request.query}

Please provide a helpful, informative response about their medical bill. Focus on:
- Answering their specific question if possible
- Providing relevant context about medical bills
- Explaining any medical procedures or charges mentioned
- Being clear about what you can and cannot determine from the document
- Keeping the response conversational and user-friendly

Response:"""
                
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": query_prompt}],
                    max_tokens=400,
                    temperature=0.7
                )
                
                query_response = response.choices[0].message.content
                
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                query_response = "I'm having trouble processing your question right now. Please try using the chat feature after analysis is complete."
        
        # Create a simple analysis result with debug data
        result = AnalysisResponse(
            success=True,
            doc_id=request.doc_id,
            analysis_complete=True,
            verdict="ok",
            total_bill_amount=1500.0,
            total_overcharge=0.0,
            confidence_score=0.85,
            red_flags=[],
            recommendations=[
                "Fallback analysis completed successfully",
                "No significant overcharges detected",
                "This is a simple mock analysis for testing purposes"
            ],
            processing_time_seconds=processing_time,
            
            # Include debug data for frontend visibility
            document_type="medical_bill",
            agent_type="fallback_analyzer",
            routing_confidence=1.0,
            ocr_text=content,
            raw_text=content,
            rawText=content,
            debug_data={
                "ocrText": content,
                "processingStats": {
                    "ocr_confidence": 95.0,
                    "characters_extracted": len(content),
                    "processing_time_ms": processing_time * 1000,
                    "method": "fallback_decoding"
                },
                "extractedLineItems": [
                    {"description": "Consultation Fee", "amount": 800.0},
                    {"description": "CBC Test", "amount": 400.0},
                    {"description": "ECG", "amount": 300.0}
                ],
                "aiAnalysis": "Fallback analysis completed successfully with decoded content",
                "analysisMethod": "fallback",
                "documentType": "medical_bill",
                "extractionMethod": "base64_decode"
            },
            query_response=query_response
        )
        
        logger.info(
            "Fallback analysis completed successfully",
            doc_id=request.doc_id,
            processing_time_seconds=processing_time
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Fallback analysis failed: {str(e)}", doc_id=request.doc_id)
        raise HTTPException(status_code=500, detail=f"Fallback analysis failed: {str(e)}")
    
    finally:
        # Background task for cleanup
        background_tasks.add_task(cleanup_analysis_artifacts, request.doc_id)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_medical_bill(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze a document using the ENHANCED document router."""
    start_time = time.time()
    
    logger.info(
        "Starting document analysis with enhanced router",
        doc_id=request.doc_id,
        user_id=request.user_id,
        file_format=request.file_format
    )
    
    try:
        # Check if enhanced router is available
        if not app_state.get("enhanced_router"):
            raise HTTPException(status_code=503, detail="Enhanced document router not available")
        
        # Decode file content
        import base64
        file_content_bytes = base64.b64decode(request.file_content)
        
        # Create AgentContext for enhanced router
        context = AgentContext(
            doc_id=request.doc_id,
            user_id=request.user_id,
            correlation_id=f"enhanced_analysis_{request.doc_id}_{int(time.time())}",
            model_hint=ModelHint.STANDARD,
            start_time=time.time(),
            metadata={
                "file_format": request.file_format,
                "language": request.language,
                "task_type": "enhanced_analysis"
            }
        )

        # Prepare task data for enhanced router
        task_data = {
            "file_content": file_content_bytes,
            "language": request.language,
            "file_format": request.file_format,
            "routing_strategy": "capability_based",
            "priority": "normal",
            "metadata": {
                "state_code": request.state_code,
                "insurance_type": request.insurance_type
            }
        }

        # Execute enhanced analysis workflow
        enhanced_result = await app_state["enhanced_router"].process_task(
            context=context,
            task_data=task_data
        )

        if not enhanced_result.get("success"):
            raise Exception(enhanced_result.get("error", "Enhanced analysis failed"))

        # Extract final result from the domain agent
        agent_result = enhanced_result.get("final_result", {}).get("domain_analysis", {})
        
        # Extract OCR text from various sources
        ocr_text = enhanced_result.get("final_result", {}).get("raw_text", "")

        # Always store document context for chat
        conversation_id = f"{request.doc_id}_{request.user_id}"
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                "doc_id": request.doc_id,
                "user_id": request.user_id,
                "messages": [],
                "document_context": ocr_text,
                "created_at": time.time()
            }
        else:
            conversations[conversation_id]["document_context"] = ocr_text
        
        # Handle user query if provided
        query_response = None
        if request.query and ocr_text:
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                query_prompt = f"""You are a medical bill analysis assistant. A user has uploaded a medical bill and asked a question about it.

Document Content:
{ocr_text[:1500]}...

User Question: {request.query}

Please provide a helpful, informative response about their medical bill. Focus on:
- Answering their specific question if possible
- Providing relevant context about medical bills
- Explaining any medical procedures or charges mentioned
- Being clear about what you can and cannot determine from the document
- Keeping the response conversational and user-friendly

Response:"""
                response = await client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": query_prompt}],
                    max_tokens=400,
                    temperature=0.7
                )
                query_response = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                query_response = "I'm having trouble processing your question right now."

        # Map enhanced result to AnalysisResponse
        result = AnalysisResponse(
            success=agent_result.get("success", False),
            doc_id=request.doc_id,
            analysis_complete=agent_result.get("analysis_complete", False),
            verdict=agent_result.get("verdict", "unknown"),
            total_bill_amount=agent_result.get("total_bill_amount", 0.0),
            total_overcharge=agent_result.get("total_overcharge", 0.0),
            confidence_score=agent_result.get("confidence_score", 0.0),
            red_flags=agent_result.get("red_flags", []),
            recommendations=agent_result.get("recommendations", []),
            processing_time_seconds=time.time() - start_time,
            
            document_type=enhanced_result.get("document_type"),
            agent_type=enhanced_result.get("final_result", {}).get("agent_used"),
            routing_confidence=enhanced_result.get("final_result", {}).get("routing_confidence"),
            
            ocr_text=ocr_text,
            raw_text=ocr_text,
            rawText=ocr_text,
            
            debug_data=agent_result.get("debug_data", {}),
            analysis=agent_result.get("confidence_analysis", {}),
            document_processing=agent_result.get("document_processing", {}),
            rate_validation=agent_result.get("rate_validation", {}),
            duplicate_detection=agent_result.get("duplicate_detection", {}),
            prohibited_detection=agent_result.get("prohibited_detection", {}),
            analysis_summary=agent_result.get("analysis_summary", {}),
            error=agent_result.get("error"),
            query_response=query_response
        )
        
        # Update metrics
        if analysis_count:
            analysis_count.labels(
                verdict=result.verdict,
                agent_id=result.agent_type or "unknown"
            ).inc()
        
        logger.info(
            "Document analysis completed successfully",
            doc_id=request.doc_id,
            verdict=result.verdict,
            confidence_score=result.confidence_score,
            document_type=result.document_type,
            processing_time_seconds=time.time() - start_time
        )
        
        return result
            
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(
            "Medical bill analysis failed",
            doc_id=request.doc_id,
            error=str(e),
            processing_time=processing_time,
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Background task for cleanup
        background_tasks.add_task(cleanup_analysis_artifacts, request.doc_id)


async def cleanup_analysis_artifacts(doc_id: str):
    """Clean up temporary artifacts from analysis."""
    try:
        # Cleanup temporary files or cache entries
        logger.debug(f"Cleaning up artifacts for doc_id: {doc_id}")
        # Add specific cleanup logic here if needed
    except Exception as e:
        logger.warning(f"Failed to cleanup artifacts for {doc_id}: {e}")


@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Chat with the agent about an analyzed document."""
    try:
        # Get conversation ID
        conversation_id = f"{request.doc_id}_{request.user_id}"
        
        # Initialize conversation if not exists
        if conversation_id not in conversations:
            conversations[conversation_id] = {
                "doc_id": request.doc_id,
                "user_id": request.user_id,
                "messages": [],
                "document_context": None,
                "created_at": time.time()
            }
        
        conversation = conversations[conversation_id]
        
        # Add user message to conversation
        user_message = {
            "role": "user",
            "content": request.message,
            "timestamp": time.time()
        }
        conversation["messages"].append(user_message)
        
        # Get document context from database
        document_context = ""
        try:
            from database.bill_chat_context import get_bill_by_id, get_user_bills
            from database.models import db_manager
            
            # Try to get the specific bill by doc_id
            async with db_manager.get_async_session() as session:
                bill = await get_bill_by_id(session, request.doc_id)
            
            # If specific bill not found, try to get the most recent bill for this user
            if not bill and request.user_id:
                logger.info(
                    "Specific bill not found, trying to get most recent bill",
                    doc_id=request.doc_id,
                    user_id=request.user_id
                )
                async with db_manager.get_async_session() as session:
                    user_bills = await get_user_bills(session, request.user_id, limit=1)
                    if user_bills and len(user_bills) > 0:
                        bill = user_bills[0]
                        logger.info(
                            "Using most recent bill for chat context",
                            doc_id=getattr(bill, 'id', 'unknown'),
                            user_id=request.user_id
                        )
            
            if bill:
                # Extract relevant information from the bill analysis
                context_parts = []
                
                # Add basic bill information
                context_parts.append(f"Bill filename: {bill.filename}")
                context_parts.append(f"Analysis status: {bill.status}")
                
                # Extract analysis results
                if hasattr(bill, 'raw_analysis') and bill.raw_analysis:
                    raw_analysis = bill.raw_analysis
                    
                    # Debug log the raw_analysis structure
                    logger.info(
                        "Raw analysis structure in chat context",
                        doc_id=request.doc_id,
                        raw_analysis_type=type(raw_analysis),
                        raw_analysis_keys=list(raw_analysis.keys()) if isinstance(raw_analysis, dict) else "Not a dict"
                    )
                    
                    # Extract final result information
                    if isinstance(raw_analysis, dict) and 'final_result' in raw_analysis:
                        final_result = raw_analysis['final_result']
                        
                        # Debug log the final_result structure
                        logger.info(
                            "Final result structure in chat context",
                            doc_id=request.doc_id,
                            final_result_type=type(final_result),
                            final_result_keys=list(final_result.keys()) if isinstance(final_result, dict) else "Not a dict",
                            has_line_items='line_items' in final_result if isinstance(final_result, dict) else False
                        )
                        
                        # If line_items exists, log its structure
                        if isinstance(final_result, dict) and 'line_items' in final_result:
                            line_items = final_result['line_items']
                            logger.info(
                                "Line items structure in chat context",
                                doc_id=request.doc_id,
                                line_items_type=type(line_items),
                                line_items_count=len(line_items) if isinstance(line_items, list) else "Not a list",
                                first_item=line_items[0] if isinstance(line_items, list) and line_items else "No items"
                            )
                        
                        # Add verdict and confidence
                        if 'verdict' in final_result:
                            context_parts.append(f"Analysis verdict: {final_result['verdict']}")
                        if 'confidence' in final_result:
                            context_parts.append(f"Confidence level: {final_result['confidence']}%")
                        
                        # Add total amount if available, else compute from line items
                        total_amount = None
                        if 'total_amount' in final_result:
                            total_amount = final_result['total_amount']
                        elif 'total_bill_amount' in final_result:
                            total_amount = final_result['total_bill_amount']
                        
                        # If still missing, compute from line items if available
                        if total_amount is None:
                            # Try to find line_items (already extracted below, but repeat for clarity)
                            possible_line_items = None
                            if 'line_items' in final_result and final_result['line_items']:
                                possible_line_items = final_result['line_items']
                            elif 'results' in final_result and isinstance(final_result['results'], dict) and 'line_items' in final_result['results']:
                                possible_line_items = final_result['results']['line_items']
                            # Compute sum if possible
                            if possible_line_items and isinstance(possible_line_items, list):
                                try:
                                    total_amount = sum(float(item.get('amount', item.get('cost', 0)) or 0) for item in possible_line_items)
                                except Exception as e:
                                    logger.warning(f"Failed to compute total_amount from line_items: {e}")
                        if total_amount is not None:
                            context_parts.append(f"Total bill amount: ₹{total_amount}")
                        
                        # Add overcharge information
                        if 'overcharge_amount' in final_result:
                            context_parts.append(f"Suspected overcharge: ₹{final_result['overcharge_amount']}")
                        
                        # Add line items/medicines information
                        line_items = None
                        
                        # Check multiple possible locations for line items
                        if 'line_items' in final_result and final_result['line_items']:
                            line_items = final_result['line_items']
                            logger.info("Found line_items directly in final_result", count=len(line_items))
                        elif 'results' in final_result and isinstance(final_result['results'], dict):
                            results = final_result['results']
                            if 'line_items' in results and results['line_items']:
                                line_items = results['line_items']
                                logger.info("Found line_items in final_result.results", count=len(line_items))
                            elif 'debug_line_items' in results and results['debug_line_items']:
                                line_items = results['debug_line_items']
                                logger.info("Found debug_line_items in final_result.results", count=len(line_items))
                        
                        # Also check raw_analysis for line_items
                        if not line_items and 'results' in raw_analysis and isinstance(raw_analysis['results'], dict):
                            results = raw_analysis['results']
                            if 'line_items' in results and results['line_items']:
                                line_items = results['line_items']
                                logger.info("Found line_items in raw_analysis.results", count=len(line_items))
                            elif 'debug_line_items' in results and results['debug_line_items']:
                                line_items = results['debug_line_items']
                                logger.info("Found debug_line_items in raw_analysis.results", count=len(line_items))
                        
                        # Check structured_results if available
                        if not line_items and hasattr(bill, 'structured_results') and bill.structured_results:
                            structured = bill.structured_results
                            logger.info({
                                "event": "Debug structured_results",
                                "structured_type": str(type(structured)),
                                "structured_keys": list(structured.keys()) if isinstance(structured, dict) else "Not a dict",
                                "structured_is_str": isinstance(structured, str),
                                "structured_length": len(str(structured)),
                                "structured_preview": str(structured)[:200] + "..." if len(str(structured)) > 200 else str(structured)
                            })
                            
                            if isinstance(structured, dict):
                                if 'line_items' in structured and structured['line_items']:
                                    line_items = structured['line_items']
                                    logger.info("Found line_items in structured_results", count=len(line_items))
                                # Also check for line_items at the end of the structured results string
                                elif isinstance(structured, str) and 'line_items' in structured:
                                    # Try to parse the JSON if it's a string
                                    try:
                                        import json
                                        parsed = json.loads(structured)
                                        if 'line_items' in parsed:
                                            line_items = parsed['line_items']
                                            logger.info("Found line_items in parsed structured_results string", count=len(line_items))
                                    except Exception as e:
                                        logger.warning(f"Failed to parse structured_results as JSON: {e}")
                                
                                # Additional check for domain_analysis which might contain line_items
                                elif 'domain_analysis' in structured and isinstance(structured['domain_analysis'], dict):
                                    domain = structured['domain_analysis']
                                    logger.info({
                                        "event": "Debug domain_analysis",
                                        "domain_type": str(type(domain)),
                                        "domain_keys": list(domain.keys()) if isinstance(domain, dict) else "Not a dict",
                                        "domain_preview": str(domain)[:200] + "..." if len(str(domain)) > 200 else str(domain)
                                    })
                                    
                                    if 'line_items' in domain:
                                        line_items = domain['line_items']
                                        logger.info("Found line_items in structured_results.domain_analysis", count=len(line_items))
                                    elif 'line_items_ai' in domain:
                                        line_items = domain['line_items_ai']
                                        logger.info("Found line_items_ai in structured_results.domain_analysis", count=len(line_items))
                                    elif 'results' in domain and isinstance(domain['results'], dict):
                                        results = domain['results']
                                        logger.info({
                                            "event": "Debug domain_results",
                                            "results_type": str(type(results)),
                                            "results_keys": list(results.keys()) if isinstance(results, dict) else "Not a dict",
                                            "results_preview": str(results)[:200] + "..." if len(str(results)) > 200 else str(results)
                                        })
                                        
                                        if 'line_items' in results:
                                            line_items = results['line_items']
                                            logger.info("Found line_items in structured_results.domain_analysis.results", count=len(line_items))
                                        elif 'line_items_ai' in results:
                                            line_items = results['line_items_ai']
                                            logger.info("Found line_items_ai in structured_results.domain_analysis.results", count=len(line_items))
                                            
                            # Try to parse the structured_results if it's a string
                            elif isinstance(structured, str):
                                try:
                                    import json
                                    parsed = json.loads(structured)
                                    logger.info({
                                        "event": "Parsed structured_results string",
                                        "parsed_type": str(type(parsed)),
                                        "parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else "Not a dict",
                                        "parsed_preview": str(parsed)[:200] + "..." if len(str(parsed)) > 200 else str(parsed)
                                    })
                                    
                                    if isinstance(parsed, dict):
                                        if 'line_items' in parsed and parsed['line_items']:
                                            line_items = parsed['line_items']
                                            logger.info("Found line_items in parsed structured_results", count=len(line_items))
                                        elif 'line_items_ai' in parsed and parsed['line_items_ai']:
                                            line_items = parsed['line_items_ai']
                                            logger.info("Found line_items_ai in parsed structured_results", count=len(line_items))
                                        elif 'domain_analysis' in parsed and isinstance(parsed['domain_analysis'], dict):
                                            domain = parsed['domain_analysis']
                                            if 'line_items' in domain:
                                                line_items = domain['line_items']
                                                logger.info("Found line_items in parsed structured_results.domain_analysis", count=len(line_items))
                                            elif 'line_items_ai' in domain:
                                                line_items = domain['line_items_ai']
                                                logger.info("Found line_items_ai in parsed structured_results.domain_analysis", count=len(line_items))
                                            elif 'results' in domain and isinstance(domain['results'], dict):
                                                results = domain['results']
                                                if 'line_items' in results:
                                                    line_items = results['line_items']
                                                    logger.info("Found line_items in parsed structured_results.domain_analysis.results", count=len(line_items))
                                                elif 'line_items_ai' in results:
                                                    line_items = results['line_items_ai']
                                                    logger.info("Found line_items_ai in parsed structured_results.domain_analysis.results", count=len(line_items))
                                except Exception as e:
                                    logger.warning(f"Failed to parse structured_results as JSON: {e}")
                            
                            # Also check raw_analysis for line_items
                            if not line_items and hasattr(bill, 'raw_analysis') and bill.raw_analysis:
                                raw = bill.raw_analysis
                                logger.info({
                                    "event": "Debug raw_analysis",
                                    "raw_type": str(type(raw)),
                                    "raw_keys": list(raw.keys()) if isinstance(raw, dict) else "Not a dict",
                                    "raw_preview": str(raw)[:200] + "..." if len(str(raw)) > 200 else str(raw)
                                })
                                
                                if isinstance(raw, dict):
                                    if 'line_items' in raw and raw['line_items']:
                                        line_items = raw['line_items']
                                        logger.info("Found line_items in raw_analysis", count=len(line_items))
                                    elif 'final_result' in raw and isinstance(raw['final_result'], dict):
                                        final = raw['final_result']
                                        if 'line_items' in final and final['line_items']:
                                            line_items = final['line_items']
                                            logger.info("Found line_items in raw_analysis.final_result", count=len(line_items))
                                        elif 'domain_analysis' in final and isinstance(final['domain_analysis'], dict):
                                            domain = final['domain_analysis']
                                            if 'line_items' in domain and domain['line_items']:
                                                line_items = domain['line_items']
                                                logger.info("Found line_items in raw_analysis.final_result.domain_analysis", count=len(line_items))
                                            elif 'results' in domain and isinstance(domain['results'], dict) and 'line_items' in domain['results']:
                                                line_items = domain['results']['line_items']
                                                logger.info("Found line_items in raw_analysis.final_result.domain_analysis.results", count=len(line_items))
                        
                        if line_items:
                            context_parts.append(f"Number of line items: {len(line_items)}")
                            
                            # Add details about first few medicines/items
                            medicines = []
                            for i, item in enumerate(line_items[:5]):  # First 5 items
                                if isinstance(item, dict):
                                    item_name = item.get('name', item.get('description', f'Item {i+1}'))
                                    item_amount = item.get('amount', item.get('cost', 'N/A'))
                                    medicines.append(f"{item_name}: ₹{item_amount}")
                            
                            if medicines:
                                context_parts.append(f"Sample medicines/items: {', '.join(medicines)}")
                        else:
                            logger.warning("No line items found in any location", doc_id=request.doc_id)
                        
                        # Add any specific findings or recommendations
                        if 'findings' in final_result:
                            context_parts.append(f"Key findings: {final_result['findings']}")
                        if 'recommendations' in final_result:
                            context_parts.append(f"Recommendations: {final_result['recommendations']}")
                
                # Create document context string
                if context_parts:
                    document_context = f"Document Context: {' | '.join(context_parts)}"
                    
                    # Update conversation context for future use
                    conversation["document_context"] = document_context
                    
                    logger.info(
                        "Retrieved bill context for chat",
                        doc_id=request.doc_id,
                        context_length=len(document_context)
                    )
                else:
                    logger.warning(
                        "No meaningful context found in bill analysis",
                        doc_id=request.doc_id
                    )
            else:
                logger.warning(
                    "No bill found for chat context",
                    doc_id=request.doc_id
                )
                
        except Exception as e:
            logger.error(
                "Failed to retrieve bill context for chat",
                doc_id=request.doc_id,
                error=str(e)
            )
        
        # Prepare chat history for OpenAI
        chat_history = []
        
        # Add system message with context
        system_message = f"""You are a helpful medical bill analysis assistant. You're discussing a medical bill document with the user.

{document_context}

Guidelines:
- Be helpful and informative about medical bills and healthcare costs
- If discussing specific charges, refer to the document context if available
- Provide clear explanations about medical procedures and their typical costs
- Help users understand their bills and identify potential issues
- Keep responses conversational and user-friendly
- If you don't have specific information about their document, be honest about limitations
"""
        
        chat_history.append({"role": "system", "content": system_message})
        
        # Add conversation history
        for msg in conversation["messages"][-10:]:  # Keep last 10 messages for context
            chat_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get OpenAI client
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Generate response
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=chat_history,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_response = response.choices[0].message.content
        
        # Add assistant response to conversation
        assistant_message = {
            "role": "assistant",
            "content": assistant_response,
            "timestamp": time.time()
        }
        conversation["messages"].append(assistant_message)
        
        return ChatResponse(
            success=True,
            doc_id=request.doc_id,
            message=assistant_response,
            conversation_id=conversation_id,
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            doc_id=request.doc_id,
            message="I'm having trouble processing your message right now. Please try again.",
            conversation_id=f"{request.doc_id}_{request.user_id}",
            timestamp=time.time()
        )


@app.post("/analyze-enhanced", response_model=EnhancedAnalysisResponse)
async def analyze_document_enhanced(request: EnhancedAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Enhanced document analysis using the new Generic OCR + Router architecture.
    
    This endpoint demonstrates the new architecture:
    1. Generic OCR Tool extracts raw data from any document type
    2. Document Type Classifier determines document type and requirements
    3. Enhanced Router Agent makes intelligent routing decisions
    4. Specialized Domain Agents handle domain-specific analysis
    """
    start_time = time.time()
    
    try:
        logger.info(
            "Starting enhanced document analysis",
            doc_id=request.doc_id,
            user_id=request.user_id,
            language=request.language,
            routing_strategy=request.routing_strategy
        )
        
        # Validate enhanced router availability
        if not app_state["enhanced_router"]:
            raise HTTPException(
                status_code=503, 
                detail="Enhanced router not available. Please check server configuration."
            )
        
        # Prepare task data for enhanced router
        import base64
        file_content = base64.b64decode(request.file_content)
        
        task_data = {
            "file_content": file_content,
            "language": request.language,
            "file_format": request.file_format,
            "routing_strategy": request.routing_strategy,
            "priority": request.priority,
            "metadata": {}
        }
        
        # Execute enhanced analysis workflow
        # Create proper AgentContext for enhanced router
        context = AgentContext(
            doc_id=request.doc_id,
            user_id=request.user_id,
            correlation_id=f"enhanced_analysis_{request.doc_id}_{int(time.time())}",
            model_hint=ModelHint.STANDARD,
            start_time=time.time(),
            metadata={
                "file_format": request.file_format,
                "routing_strategy": request.routing_strategy,
                "priority": request.priority,
                "language": request.language,
                "task_type": "enhanced_analysis"
            }
        )
        
        result = await app_state["enhanced_router"].process_task(
            context=context,
            task_data=task_data
        )
        
        processing_time = time.time() - start_time
        
        # Record metrics for enhanced analysis
        analysis_count.labels(
            verdict=result.get("final_result", {}).get("verdict", "unknown"),
            agent_id="enhanced_router"
        ).inc()
        
        analysis_duration.labels(
            agent_id="enhanced_router"
        ).observe(processing_time)
        
        logger.info(
            "Enhanced document analysis completed",
            doc_id=request.doc_id,
            document_type=result.get("document_type", "unknown"),
            processing_time=processing_time,
            success=result.get("success", False)
        )
        
        # Save analysis results to database for chat context (if analysis was successful)
        if result.get("success", False):
            try:
                from database.bill_chat_context import save_bill_analysis
                from database.models import db_manager
                import hashlib
                
                # Calculate file hash for deduplication
                file_hash = hashlib.sha256(file_content).hexdigest()
                
                # Get database session
                async with db_manager.get_async_session() as session:
                    await save_bill_analysis(
                        session=session,
                        user_id=request.user_id,
                        doc_id=request.doc_id,
                        filename=getattr(request, 'filename', f"document_{request.doc_id}.pdf"),
                        file_hash=file_hash,
                        file_size=len(file_content),
                        content_type=request.file_format or "application/pdf",
                        analysis_type="medical",
                        raw_analysis=result,
                        structured_results=result.get("final_result", {}),
                        status="completed"
                    )
                    
                logger.info(
                    "Bill analysis saved for chat context",
                    doc_id=request.doc_id,
                    user_id=request.user_id
                )
                    
            except Exception as e:
                logger.warning(
                    "Failed to save bill analysis for chat context",
                    doc_id=request.doc_id,
                    error=str(e)
                )
        
        # Handle user query if provided
        query_response = None
        if request.query and result.get("success", False):
            try:
                # Extract OCR text from the result
                ocr_text = result.get("final_result", {}).get("raw_text", "")
                
                if ocr_text:
                    from openai import AsyncOpenAI
                    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                    
                    query_prompt = f"""You are a medical bill analysis assistant. A user has uploaded a medical bill and asked a question about it.

Document Content:
{ocr_text[:1500]}...

User Question: {request.query}

Please provide a helpful, informative response about their medical bill. Focus on:
- Answering their specific question if possible
- Providing relevant context about medical bills
- Explaining any medical procedures or charges mentioned
- Being clear about what you can and cannot determine from the document
- Keeping the response conversational and user-friendly

Response:"""
                    
                    response = await client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": query_prompt}],
                        max_tokens=400,
                        temperature=0.7
                    )
                    
                    query_response = response.choices[0].message.content
                    
                    logger.info(
                        "Query processed successfully",
                        doc_id=request.doc_id,
                        query_length=len(request.query)
                    )
                else:
                    logger.warning(
                        "No OCR text available for query processing",
                        doc_id=request.doc_id
                    )
                    query_response = "I couldn't find any text in the document to answer your question."
            except Exception as e:
                logger.error(
                    "Query processing error",
                    doc_id=request.doc_id,
                    error=str(e)
                )
                query_response = "I'm having trouble processing your question right now."
        
        return EnhancedAnalysisResponse(
            success=result.get("success", False),
            doc_id=result.get("doc_id", request.doc_id),
            document_type=result.get("document_type", "unknown"),
            processing_stages=result.get("processing_stages", {}),
            final_result=result.get("final_result", {}),
            total_processing_time_ms=result.get("total_processing_time_ms", int(processing_time * 1000)),
            error=result.get("error"),
            query_response=query_response
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Enhanced analysis failed: {str(e)}"
        
        logger.error(
            "Enhanced document analysis failed",
            doc_id=request.doc_id,
            error=error_msg,
            processing_time=processing_time,
            exc_info=True
        )
        
        raise HTTPException(status_code=500, detail=error_msg)


# Bill management endpoints
@app.get("/bills")
async def get_user_bills(user_id: str, limit: int = 50):
    """Get user's bill analysis history for chat context."""
    try:
        from database.bill_chat_context import get_user_bills
        from database.models import db_manager
        
        async with db_manager.get_async_session() as session:
            bills = await get_user_bills(session, user_id, limit)
            
        # Convert to response format
        bills_data = []
        for bill in bills:
            bills_data.append({
                "id": str(bill.id),
                "filename": bill.filename,
                "created_at": bill.created_at.isoformat() if hasattr(bill.created_at, 'isoformat') else str(bill.created_at),
                "status": bill.status,
                "analysis_type": bill.analysis_type,
                "total_amount": getattr(bill, 'total_amount', 0),
                "suspected_overcharges": getattr(bill, 'suspected_overcharges', 0),
                "confidence_level": getattr(bill, 'confidence_level', 0)
            })
            
        return {
            "bills": bills_data,
            "total": len(bills_data)
        }
        
    except Exception as e:
        logger.error("Failed to retrieve user bills", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bills")


@app.get("/bills/{doc_id}")
async def get_bill_details(doc_id: str):
    """Get detailed bill analysis for chat context."""
    try:
        from database.bill_chat_context import get_bill_by_id
        from database.models import db_manager
        
        async with db_manager.get_async_session() as session:
            bill = await get_bill_by_id(session, doc_id)
            
        if not bill:
            raise HTTPException(status_code=404, detail="Bill not found")
            
        return {
            "id": str(bill.id),
            "filename": bill.filename,
            "created_at": bill.created_at.isoformat() if hasattr(bill.created_at, 'isoformat') else str(bill.created_at),
            "status": bill.status,
            "analysis_type": bill.analysis_type,
            "raw_analysis": bill.raw_analysis,
            "structured_results": bill.structured_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve bill details", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve bill details")


# Agent management endpoints
@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    try:
        if not app_state["registry"]:
            return {"agents": [], "total": 0, "message": "Registry not available"}
        
        online_agents = await app_state["registry"].list_online_agents()
        
        agents_info = []
        for agent in online_agents:
            agents_info.append({
                "agent_id": agent.agent_id,
                "name": agent.name,
                "status": agent.status.value,
                "capabilities": [cap.value for cap in agent.capabilities.supported_tasks],
                "max_concurrent_requests": agent.capabilities.max_concurrent_requests,
                "estimated_cost": agent.capabilities.cost_per_request_rupees,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None
            })
        
        return {"agents": agents_info, "total": len(agents_info)}
        
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agents")


@app.get("/agents/{agent_id}")
async def get_agent_details(agent_id: str):
    """Get detailed information about a specific agent."""
    try:
        if not app_state["registry"]:
            raise HTTPException(status_code=503, detail="Registry not available")
        
        agents = await app_state["registry"].discover_agents([])
        agent = next((a for a in agents if a.agent_id == agent_id), None)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "status": agent.status.value,
            "capabilities": {
                "supported_tasks": [cap.value for cap in agent.capabilities.supported_tasks],
                "max_concurrent_requests": agent.capabilities.max_concurrent_requests,
                "cost_per_request_rupees": agent.capabilities.cost_per_request_rupees,
                "processing_time_ms_avg": agent.capabilities.processing_time_ms_avg,
                "confidence_threshold": agent.capabilities.confidence_threshold,
                "supported_document_types": agent.capabilities.supported_document_types,
                "supported_languages": agent.capabilities.supported_languages,
                "preferred_model_hints": [hint.value for hint in agent.capabilities.preferred_model_hints]
            },
            "registration_time": agent.registration_time.isoformat() if agent.registration_time else None,
            "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
            "current_requests": agent.current_requests,
            "performance_metrics": {
                "total_requests": agent.total_requests,
                "total_errors": agent.total_errors,
                "avg_response_time_ms": agent.avg_response_time_ms,
                "success_rate": (agent.total_requests - agent.total_errors) / max(agent.total_requests, 1)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent details", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agent details")


# Background tasks
async def update_metrics_background():
    """Update Prometheus metrics periodically."""
    while not app_state["shutdown_event"].is_set():
        try:
            if app_state["registry"]:
                online_agents = await app_state["registry"].list_online_agents()
                
                # Update agent count metrics
                agent_counts = {}
                for agent in online_agents:
                    agent_type = "medical_bill" if "medical_bill" in agent.agent_id else "other"
                    agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1
                
                for agent_type, count in agent_counts.items():
                    active_agents.labels(agent_type=agent_type).set(count)
                
                # Set zero for agent types with no active agents
                all_types = ["medical_bill", "router", "other"]
                for agent_type in all_types:
                    if agent_type not in agent_counts:
                        active_agents.labels(agent_type=agent_type).set(0)
            else:
                # No registry available - set all agent counts to 0
                all_types = ["medical_bill", "router", "other"]
                for agent_type in all_types:
                    if active_agents:
                        active_agents.labels(agent_type=agent_type).set(0)
        
        except Exception as e:
            logger.error("Error updating metrics", error=str(e))
        
        await asyncio.sleep(30)  # Update every 30 seconds


# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal", signal=signum)
    app_state["shutdown_event"].set()


if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Get configuration from config object
    host = config.host
    port = config.port
    debug = config.debug if hasattr(config, 'debug') else False
    workers = config.max_workers
    log_level = config.log_level
    
    logging.basicConfig(level=getattr(logging, log_level))
    
    logger.info(
        "Starting server",
        host=host,
        port=port,
        debug=debug,
        workers=workers,
        log_level=log_level
    )
    
    # Run server
    uvicorn.run(
        "agents.server:app",
        host=host,
        port=port,
        log_level=log_level.lower(),
        reload=debug,
        workers=1 if debug else workers,
        access_log=True
    )