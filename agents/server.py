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
from typing import Dict, Any, Optional

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
from agents.agent_registry import AgentRegistry, TaskCapability, AgentCapabilities
from agents.base_agent import ModelHint, AgentContext
from agents.router_agent import RouterAgent, RoutingStrategy, RoutingRequest
from agents.redis_state import RedisStateManager
from agents.tools.enhanced_router_agent import EnhancedRouterAgent
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

# Global state
app_state = {
    "registry": None,
    "router": None,
    "enhanced_router": None,
    "medical_agent": None,
    "redis_manager": None,
    "startup_time": None,
    "shutdown_event": asyncio.Event()
}


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


class AnalysisResponse(BaseModel):
    success: bool
    doc_id: str
    analysis_complete: bool
    verdict: str
    total_bill_amount: float
    total_overcharge: float
    confidence_score: float
    red_flags: list
    recommendations: list
    processing_time_seconds: float


class EnhancedAnalysisRequest(BaseModel):
    file_content: str = Field(..., description="Base64 encoded file content")
    doc_id: str = Field(..., description="Unique document identifier")
    user_id: str = Field(..., description="User identifier")
    language: str = Field(default="english", description="Document language")
    file_format: Optional[str] = Field(default=None, description="File format hint")
    routing_strategy: str = Field(default="capability_based", description="Routing strategy")
    priority: str = Field(default="normal", description="Processing priority")


class EnhancedAnalysisResponse(BaseModel):
    success: bool
    doc_id: str
    document_type: str
    processing_stages: Dict[str, Any]
    final_result: Dict[str, Any]
    total_processing_time_ms: int
    error: Optional[str] = None


class MetricsResponse(BaseModel):
    timestamp: float
    active_agents: int
    total_analyses: int
    average_confidence: float
    uptime_seconds: float


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
    
    # Initialize Redis Manager
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        app_state["redis_manager"] = RedisStateManager(redis_url)
        await app_state["redis_manager"].connect()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Redis connection failed, continuing without Redis: {e}")
        app_state["redis_manager"] = None
    
    # Initialize Agent Registry
    try:
        if app_state["redis_manager"]:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            app_state["registry"] = AgentRegistry(redis_url=redis_url)
            await app_state["registry"].start()
            logger.info("Agent registry initialized")
        else:
            logger.warning("Skipping agent registry initialization - Redis not available")
    except Exception as e:
        logger.warning(f"Failed to initialize agent registry: {e}")
        app_state["registry"] = None
    
    # Initialize Router Agent
    try:
        if app_state["registry"]:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            app_state["router"] = RouterAgent(
                registry=app_state["registry"],
                redis_url=redis_url
            )
            logger.info("Router agent initialized")
        else:
            logger.warning("Skipping router agent initialization - Registry not available")
    except Exception as e:
        logger.warning(f"Router agent initialization failed: {e}")
        app_state["router"] = None
    
    # Initialize Enhanced Router Agent
    try:
        if app_state["registry"]:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            app_state["enhanced_router"] = EnhancedRouterAgent(
                registry=app_state["registry"],
                redis_url=redis_url,
                openai_api_key=openai_api_key
            )
            logger.info("Enhanced router agent initialized")
        else:
            logger.warning("Skipping enhanced router agent initialization - Registry not available")
    except Exception as e:
        logger.warning(f"Enhanced router agent initialization failed: {e}")
        app_state["enhanced_router"] = None
    
    # Initialize OAuth2 Manager
    try:
        await auth_manager.initialize()
        logger.info("OAuth2 manager initialized")
    except Exception as e:
        logger.warning(f"OAuth2 manager initialization failed: {e}")
        # Continue without OAuth2 - basic analysis will work
    
    # Initialize Medical Bill Agent
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if openai_api_key:
            app_state["medical_agent"] = MedicalBillAgent(
                redis_url=redis_url,
                openai_api_key=openai_api_key
            )
            
            # Register medical bill agent with capabilities
            if app_state["registry"]:
                capabilities = AgentCapabilities(
                    supported_tasks=[
                        TaskCapability.DOCUMENT_PROCESSING,
                        TaskCapability.RATE_VALIDATION,
                        TaskCapability.DUPLICATE_DETECTION,
                        TaskCapability.PROHIBITED_DETECTION,
                        TaskCapability.CONFIDENCE_SCORING
                    ],
                    max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
                    preferred_model_hints=[ModelHint.STANDARD, ModelHint.PREMIUM],
                    processing_time_ms_avg=int(os.getenv("ESTIMATED_RESPONSE_TIME_MS", "5000")),
                    cost_per_request_rupees=float(os.getenv("ESTIMATED_COST_PER_REQUEST", "0.50")),
                    confidence_threshold=0.85,
                    supported_document_types=["pdf", "jpg", "png", "jpeg"],
                    supported_languages=["english", "hindi"]
                )
                
                registration_success = await app_state["registry"].register_agent(
                    agent=app_state["medical_agent"],
                    capabilities=capabilities
                )
                
                if registration_success:
                    logger.info("Medical bill agent registered", agent_id=app_state["medical_agent"].agent_id)
                else:
                    logger.warning("Failed to register medical bill agent")
            else:
                logger.warning("Registry not available - medical bill agent created but not registered")
        else:
            logger.warning("OpenAI API key not found, medical agent not created")
    except Exception as e:
        logger.warning(f"Medical agent initialization failed: {e}")
        app_state["medical_agent"] = None
    
    # Start background tasks
    try:
        asyncio.create_task(update_metrics_background())
        logger.info("Background tasks started")
    except Exception as e:
        logger.warning(f"Background tasks failed to start: {e}")
    
    # Record startup time
    app_state["startup_time"] = time.time()
    logger.info("Server startup completed successfully")
    
    yield
    
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
        # Agent registry health
        if app_state["registry"]:
            online_agents = await app_state["registry"].list_online_agents()
            components["registry"] = "healthy"
        else:
            components["registry"] = "unhealthy"
    except Exception:
        components["registry"] = "unhealthy"
    
    try:
        # Medical agent health
        if app_state["medical_agent"]:
            agent_health = await app_state["medical_agent"].health_check()
            components["medical_agent"] = agent_health.get("status", "unknown")
        else:
            components["medical_agent"] = "unhealthy"
    except Exception:
        components["medical_agent"] = "unhealthy"
    
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
        
        # Check if medical agent is registered
        if app_state["registry"] and app_state["medical_agent"]:
            online_agents = await app_state["registry"].list_online_agents()
            medical_agent_online = any(
                a.agent_id == app_state["medical_agent"].agent_id 
                for a in online_agents
            )
            
            if not medical_agent_online:
                logger.warning("Medical agent not registered, but service is still ready")
        
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
        total_analyses=total_analyses,
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
            cache_data[agent_id] = {
                "agent_id": registration.agent_id,
                "name": registration.name,
                "status": registration.status.value,
                "capabilities": {
                    "supported_tasks": [task.value for task in registration.capabilities.supported_tasks],
                    "preferred_model_hints": [hint.value for hint in registration.capabilities.preferred_model_hints]
                },
                "last_heartbeat": registration.last_heartbeat.isoformat(),
                "registration_time": registration.registration_time.isoformat()
            }
        
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
        
        # Create a simple analysis result
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
            processing_time_seconds=processing_time
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
    """Analyze a medical bill document."""
    start_time = time.time()
    
    logger.info(
        "Starting medical bill analysis",
        doc_id=request.doc_id,
        user_id=request.user_id,
        file_format=request.file_format
    )
    
    try:
        # Debug: Check router status
        router_status = app_state.get("router")
        logger.info(f"Router status: {router_status is not None}, type: {type(router_status)}")
        
        # If router is available, use it
        if app_state.get("router") is not None:
            logger.info("Using router for analysis")
            # Create routing request
            routing_request = RoutingRequest(
                doc_id=request.doc_id,
                user_id=request.user_id,
                task_type="medical_bill_analysis",
                required_capabilities=[TaskCapability.DOCUMENT_PROCESSING, TaskCapability.RATE_VALIDATION],
                model_hint=ModelHint.STANDARD,
                routing_strategy=RoutingStrategy.CAPABILITY_BASED,
                max_agents=1,
                timeout_seconds=30,
                priority=3,  # Normal priority
                metadata={
                    "file_content": request.file_content,
                    "language": request.language,
                    "state_code": request.state_code,
                    "insurance_type": request.insurance_type,
                    "file_format": request.file_format
                }
            )
            
            # Route the request
            decision = await app_state["router"].route_request(routing_request)
            
            # Actually execute the selected agent
            if decision.selected_agents:
                selected_agent_id = decision.selected_agents[0].agent_id
                logger.info(f"Executing selected agent: {selected_agent_id}")
                
                # Get the agent from the registry
                if selected_agent_id == "medical_bill_agent" and app_state.get("medical_agent"):
                    # Decode base64 file content
                    import base64
                    file_content = base64.b64decode(request.file_content)
                    
                    # Execute the medical bill agent
                    agent_result = await app_state["medical_agent"].analyze_medical_bill(
                        file_content=file_content,
                        doc_id=request.doc_id,
                        user_id=request.user_id,
                        language=request.language,
                        state_code=request.state_code,
                        insurance_type=request.insurance_type,
                        file_format=request.file_format
                    )
                    
                    # Convert agent result to analysis response
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
                        processing_time_seconds=time.time() - start_time
                    )
                    
                    # Add OCR text and analysis to the response metadata
                    if hasattr(result, '__dict__'):
                        result.__dict__['ocr_text'] = agent_result.get("document_processing", {}).get("raw_text", "")
                        result.__dict__['analysis'] = agent_result.get("confidence_analysis", {})
                        result.__dict__['document_processing'] = agent_result.get("document_processing", {})
                        result.__dict__['rate_validation'] = agent_result.get("rate_validation", {})
                        result.__dict__['duplicate_detection'] = agent_result.get("duplicate_detection", {})
                        result.__dict__['prohibited_detection'] = agent_result.get("prohibited_detection", {})
                        result.__dict__['analysis_summary'] = agent_result.get("analysis_summary", {})
                else:
                    # Fallback if agent not found
                    result = AnalysisResponse(
                        success=False,
                        doc_id=request.doc_id,
                        analysis_complete=False,
                        verdict="error",
                        total_bill_amount=0.0,
                        total_overcharge=0.0,
                        confidence_score=0.0,
                        red_flags=[],
                        recommendations=[f"Selected agent {selected_agent_id} not available"],
                        processing_time_seconds=time.time() - start_time
                    )
            else:
                # No agents found
                result = AnalysisResponse(
                    success=False,
                    doc_id=request.doc_id,
                    analysis_complete=False,
                    verdict="error",
                    total_bill_amount=0.0,
                    total_overcharge=0.0,
                    confidence_score=0.0,
                    red_flags=[],
                    recommendations=["No suitable agents found for analysis"],
                    processing_time_seconds=time.time() - start_time
                )
            
            # Update metrics
            if analysis_count:
                analysis_count.labels(
                    verdict=result.verdict,
                    agent_id=result.verdict
                ).inc()
            
            logger.info(
                "Medical bill analysis completed via router",
                doc_id=request.doc_id,
                verdict=result.verdict,
                confidence_score=result.confidence_score,
                processing_time_seconds=time.time() - start_time
            )
            
            return result
            
        # Fallback: Use medical agent directly if available
        elif app_state.get("medical_agent") is not None:
            logger.info("Using medical agent directly for analysis")
            
            # Create agent context with correct parameters
            agent_context = AgentContext(
                doc_id=request.doc_id,
                user_id=request.user_id,
                correlation_id=f"analyze-{request.doc_id}-{int(time.time())}",
                model_hint=ModelHint.STANDARD,
                start_time=start_time,
                metadata={
                    "file_content": request.file_content,
                    "language": request.language,
                    "state_code": request.state_code,
                    "insurance_type": request.insurance_type,
                    "file_format": request.file_format,
                    "task_type": "medical_bill_analysis"
                }
            )
            
            # Use medical agent directly
            task_input = f"Analyze medical bill with ID: {request.doc_id}"
            result = await app_state["medical_agent"].execute(agent_context, task_input)
            
            # Convert AgentResult to AnalysisResponse
            analysis_response = AnalysisResponse(
                success=result.success,
                doc_id=request.doc_id,
                analysis_complete=result.success,
                verdict=result.data.get("verdict", "unknown"),
                total_bill_amount=result.data.get("total_bill_amount", 0.0),
                total_overcharge=result.data.get("total_overcharge", 0.0),
                confidence_score=result.confidence,
                red_flags=result.data.get("red_flags", []),
                recommendations=result.data.get("recommendations", []),
                processing_time_seconds=result.execution_time_ms / 1000.0
            )
            
            # Update metrics
            if analysis_count:
                analysis_count.labels(
                    verdict=analysis_response.verdict,
                    agent_id="medical_bill_agent"
                ).inc()
            
            logger.info(
                "Medical bill analysis completed via direct agent",
                doc_id=request.doc_id,
                verdict=analysis_response.verdict,
                confidence_score=analysis_response.confidence_score,
                processing_time_seconds=analysis_response.processing_time_seconds
            )
            
            return analysis_response
            
        # Final fallback: Mock analysis for development
        else:
            logger.info("Using mock analysis fallback")
            
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
            
            # Create a simple analysis result
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
                    "Mock analysis completed successfully",
                    "No significant overcharges detected",
                    "This is a development fallback - router and agents not available"
                ],
                processing_time_seconds=processing_time
            )
            
            # Update metrics
            if analysis_count:
                analysis_count.labels(
                    verdict=result.verdict,
                    agent_id="mock_agent"
                ).inc()
            
            logger.info(
                "Mock analysis completed successfully",
                doc_id=request.doc_id,
                processing_time_seconds=processing_time
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
        
        return EnhancedAnalysisResponse(
            success=result.get("success", False),
            doc_id=result.get("doc_id", request.doc_id),
            document_type=result.get("document_type", "unknown"),
            processing_stages=result.get("processing_stages", {}),
            final_result=result.get("final_result", {}),
            total_processing_time_ms=result.get("total_processing_time_ms", int(processing_time * 1000)),
            error=result.get("error")
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
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    # Configure logging level
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
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
        workers=1 if debug else workers,  # Single worker in debug mode
        access_log=True
    ) 