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
from agents.simple_router import SimpleDocumentRouter, DocumentType
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
        if not app_state["registry"] or not app_state["medical_agent"]:
            logger.warning("Registry or medical agent not available for registration")
            return False
        
        # Check if medical agent is registered
        agent_status = await app_state["registry"].get_agent_status(app_state["medical_agent"].agent_id)
        
        if not agent_status or agent_status.status.value not in ["online", "degraded"]:
            logger.info("Medical agent not registered or offline, registering now")
            
            # Create capabilities for medical agent
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



# Global state
app_state = {
    "registry": None,  # Keep for compatibility, but not used in simple routing
    "simple_router": None,  # New clean router
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
    
    # Initialize Redis Manager with retry logic
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    for attempt in range(3):
        try:
            app_state["redis_manager"] = RedisStateManager(redis_url)
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
                app_state["registry"] = AgentRegistry(redis_url=redis_url)
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
    
        # Initialize Simple Document Router (Railway-optimized)
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            app_state["simple_router"] = SimpleDocumentRouter(
                redis_url=redis_url,
                openai_api_key=openai_api_key
            )
            logger.info("Simple document router initialized successfully")
        else:
            logger.warning("OpenAI API key not found, router not initialized")
            app_state["simple_router"] = None
    except Exception as e:
        logger.error(f"Simple router initialization failed: {e}")
        app_state["simple_router"] = None
    
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
        # Simple router health
        if app_state["simple_router"]:
            router_health = await app_state["simple_router"].health_check()
            components["simple_router"] = router_health.get("status", "unknown")
        else:
            components["simple_router"] = "unhealthy"
    except Exception as e:
        logger.error("Simple router health check failed", error=str(e))
        components["simple_router"] = "unhealthy"
    
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


@app.get("/debug/test-router-availability")
async def test_router_availability():
    """Test endpoint to verify simple document router is working."""
    try:
        if not app_state.get("simple_router"):
            return {
                "status": "error",
                "message": "Simple router not available"
            }
        
        # Test router health
        health_result = await app_state["simple_router"].health_check()
        
        # Test document type detection
        test_content = "This is a medical bill from Apollo Hospital for patient treatment and medicines."
        routing_decision = await app_state["simple_router"].route_document(
            file_content=test_content,
            doc_id="test-123",
            user_id="test-user",
            filename="test_bill.pdf"
        )
        
        return {
            "status": "success",
            "message": "Simple router is working",
            "router_health": health_result,
            "test_routing": {
                "document_type": routing_decision.document_type,
                "agent_type": routing_decision.agent_type,
                "confidence": routing_decision.confidence
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Router test failed: {str(e)}"
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
    """Analyze a document using the simple document router."""
    start_time = time.time()
    
    logger.info(
        "Starting document analysis",
        doc_id=request.doc_id,
        user_id=request.user_id,
        file_format=request.file_format
    )
    
    try:
        # Check if simple router is available
        if not app_state.get("simple_router"):
            raise HTTPException(status_code=503, detail="Document router not available")
        
        # Decode file content for analysis
        import base64
        file_content_bytes = base64.b64decode(request.file_content)
        file_content_text = file_content_bytes.decode('utf-8', errors='ignore')
        
        # Route document to appropriate agent
        routing_decision = await app_state["simple_router"].route_document(
            file_content=file_content_text,
            doc_id=request.doc_id,
            user_id=request.user_id,
            filename=f"{request.doc_id}.{request.file_format}"
        )
        
        logger.info(
            "Document routing completed",
            doc_type=routing_decision.document_type,
            agent_type=routing_decision.agent_type,
            confidence=routing_decision.confidence
        )
        
        # Execute analysis using the routed agent
        agent_result = await app_state["simple_router"].execute_analysis(
            routing_decision=routing_decision,
            file_content=file_content_bytes,
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
        
        # Add document analysis metadata
        if hasattr(result, '__dict__'):
            result.__dict__['document_type'] = routing_decision.document_type
            result.__dict__['agent_type'] = routing_decision.agent_type
            result.__dict__['routing_confidence'] = routing_decision.confidence
            
            # OCR text from various sources
            ocr_text = (agent_result.get("document_processing", {}).get("raw_text", "") or 
                       agent_result.get("debug_data", {}).get("ocrText", "") or
                       agent_result.get("raw_text", ""))
            
            result.__dict__['ocr_text'] = ocr_text
            result.__dict__['raw_text'] = ocr_text  # Frontend also looks for this field
            result.__dict__['rawText'] = ocr_text   # Alternative field name
            
            # Include debug data for frontend
            result.__dict__['debug_data'] = agent_result.get("debug_data", {
                "ocrText": ocr_text,
                "processingStats": agent_result.get("document_processing", {}).get("processing_stats", {}),
                "extractedLineItems": agent_result.get("document_processing", {}).get("line_items", []),
                "aiAnalysis": agent_result.get("ai_analysis_notes", ""),
                "analysisMethod": agent_result.get("analysis_method", "standard"),
                "documentType": routing_decision.document_type,
                "extractionMethod": "document_processor"
            })
            
            result.__dict__['analysis'] = agent_result.get("confidence_analysis", {})
            result.__dict__['document_processing'] = agent_result.get("document_processing", {})
            result.__dict__['rate_validation'] = agent_result.get("rate_validation", {})
            result.__dict__['duplicate_detection'] = agent_result.get("duplicate_detection", {})
            result.__dict__['prohibited_detection'] = agent_result.get("prohibited_detection", {})
            result.__dict__['analysis_summary'] = agent_result.get("analysis_summary", {})
        
        # Update metrics
        if analysis_count:
            analysis_count.labels(
                verdict=result.verdict,
                agent_id=routing_decision.agent_type
            ).inc()
        
        logger.info(
            "Document analysis completed successfully",
            doc_id=request.doc_id,
            verdict=result.verdict,
            confidence_score=result.confidence_score,
            document_type=routing_decision.document_type,
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