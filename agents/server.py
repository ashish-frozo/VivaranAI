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

from agents.medical_bill_agent import MedicalBillAgent
from agents.agent_registry import AgentRegistry, TaskCapability, AgentCapabilities
from agents.base_agent import ModelHint
from agents.router_agent import RouterAgent, RoutingStrategy, RoutingRequest
from agents.redis_state import RedisStateManager
from agents.tools.enhanced_router_agent import EnhancedRouterAgent

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
request_count = Counter('medbillguard_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('medbillguard_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_agents = Gauge('medbillguard_active_agents', 'Number of active agents', ['agent_type'])
analysis_count = Counter('medbillguard_analysis_total', 'Total medical bill analyses', ['verdict', 'agent_id'])
analysis_duration = Histogram('medbillguard_analysis_duration_seconds', 'Analysis duration', ['agent_id'])

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
    """Application lifespan management."""
    # Startup
    logger.info("Starting MedBillGuard Agent Server")
    app_state["startup_time"] = time.time()
    
    try:
        # Initialize Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/1")
        app_state["redis_manager"] = RedisStateManager(redis_url)
        
        # Connect and test Redis connection
        await app_state["redis_manager"].connect()
        await app_state["redis_manager"].ping()
        logger.info("Redis connection established", redis_url=redis_url)
        
        # Initialize agent registry
        app_state["registry"] = AgentRegistry(redis_url=redis_url)
        logger.info("Agent registry initialized")
        
        # Initialize router agent
        app_state["router"] = RouterAgent(
            registry=app_state["registry"],
            redis_url=redis_url
        )
        logger.info("Router agent initialized")
        
        # Get OpenAI API key for enhanced router and medical agent
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize enhanced router agent with OCR and classification
        app_state["enhanced_router"] = EnhancedRouterAgent(
            registry=app_state["registry"],
            redis_url=redis_url,
            openai_api_key=openai_api_key
        )
        logger.info("Enhanced router agent initialized")
        
        # Initialize medical bill agent
        app_state["medical_agent"] = MedicalBillAgent(
            redis_url=redis_url,
            openai_api_key=openai_api_key
        )
        
        # Register medical bill agent with capabilities (updated parameters)
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
            raise Exception("Failed to register medical bill agent")
        
        # Start background tasks
        asyncio.create_task(update_metrics_background())
        
        logger.info("MedBillGuard Agent Server started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start server", error=str(e), exc_info=True)
        raise
    
    # Shutdown
    logger.info("Shutting down MedBillGuard Agent Server")
    app_state["shutdown_event"].set()
    
    try:
        # Deregister agent
        if app_state["registry"] and app_state["medical_agent"]:
            await app_state["registry"].deregister_agent(app_state["medical_agent"].agent_id)
            logger.info("Agent deregistered")
        
        # Close connections
        if app_state["redis_manager"]:
            await app_state["redis_manager"].disconnect()
        
        logger.info("Shutdown completed")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


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


# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Record metrics
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


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
        await app_state["redis_manager"].ping()
        components["redis"] = "healthy"
    except Exception:
        components["redis"] = "unhealthy"
    
    try:
        # Agent registry health
        online_agents = await app_state["registry"].list_online_agents()
        components["registry"] = "healthy"
    except Exception:
        components["registry"] = "unhealthy"
    
    try:
        # Medical agent health
        agent_health = await app_state["medical_agent"].health_check()
        components["medical_agent"] = agent_health.get("status", "unknown")
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
        # Quick health checks
        await app_state["redis_manager"].ping()
        
        # Check if medical agent is registered
        online_agents = await app_state["registry"].list_online_agents()
        medical_agent_online = any(
            a.agent_id == app_state["medical_agent"].agent_id 
            for a in online_agents
        )
        
        if not medical_agent_online:
            raise HTTPException(status_code=503, detail="Medical agent not ready")
        
        return {"status": "ready", "timestamp": time.time()}
        
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary", response_model=MetricsResponse)
async def metrics_summary():
    """Human-readable metrics summary."""
    current_time = time.time()
    uptime = current_time - app_state["startup_time"] if app_state["startup_time"] else 0
    
    try:
        online_agents = await app_state["registry"].list_online_agents()
        active_count = len(online_agents)
    except Exception:
        active_count = 0
    
    return MetricsResponse(
        timestamp=current_time,
        active_agents=active_count,
        total_analyses=int(analysis_count._value.sum()),
        average_confidence=0.85,  # This would be calculated from real data
        uptime_seconds=uptime
    )


# Analysis endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_medical_bill(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Analyze medical bill through the multi-agent system."""
    start_time = time.time()
    
    try:
        logger.info(
            "Starting medical bill analysis",
            doc_id=request.doc_id,
            user_id=request.user_id,
            file_format=request.file_format
        )
        
        # Route through the router agent for optimal agent selection
        routing_request = RoutingRequest(
            task_type="medical_bill_analysis",
            required_capabilities=[TaskCapability.DOCUMENT_PROCESSING],
            task_data=request.dict(),
            user_id=request.user_id,
            priority="normal"
        )
        
        routing_decision = await app_state["router"].route_request(
            routing_request,
            strategy=RoutingStrategy.PERFORMANCE_OPTIMIZED
        )
        
        if not routing_decision.success:
            raise HTTPException(
                status_code=503,
                detail=f"No suitable agent available: {routing_decision.reason}"
            )
        
        # Execute analysis
        import base64
        file_content = base64.b64decode(request.file_content)
        
        result = await app_state["medical_agent"].analyze_medical_bill(
            file_content=file_content,
            doc_id=request.doc_id,
            user_id=request.user_id,
            language=request.language,
            state_code=request.state_code,
            insurance_type=request.insurance_type,
            file_format=request.file_format
        )
        
        processing_time = time.time() - start_time
        
        # Record metrics
        analysis_count.labels(
            verdict=result.get("verdict", "unknown"),
            agent_id=routing_decision.selected_agent_id
        ).inc()
        
        analysis_duration.labels(
            agent_id=routing_decision.selected_agent_id
        ).observe(processing_time)
        
        logger.info(
            "Medical bill analysis completed",
            doc_id=request.doc_id,
            verdict=result.get("verdict"),
            processing_time=processing_time,
            selected_agent=routing_decision.selected_agent_id
        )
        
        return AnalysisResponse(
            success=result["success"],
            doc_id=result["doc_id"],
            analysis_complete=result.get("analysis_complete", False),
            verdict=result.get("verdict", "unknown"),
            total_bill_amount=result.get("total_bill_amount", 0.0),
            total_overcharge=result.get("total_overcharge", 0.0),
            confidence_score=result.get("confidence_score", 0.0),
            red_flags=result.get("red_flags", []),
            recommendations=result.get("recommendations", []),
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Analysis failed: {str(e)}"
        
        logger.error(
            "Medical bill analysis failed",
            doc_id=request.doc_id,
            error=error_msg,
            processing_time=processing_time,
            exc_info=True
        )
        
        raise HTTPException(status_code=500, detail=error_msg)


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
        result = await app_state["enhanced_router"].process_task(
            context=None,  # Enhanced router will create its own context
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