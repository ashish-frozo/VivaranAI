"""
Metrics and Health Check Server for MedBillGuard Agents.

Provides /metrics endpoint for Prometheus scraping and /healthz endpoint 
for Kubernetes readiness/liveness probes with self-test capability.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, Response, status
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import structlog

from agents.base_agent import (
    AGENT_EXECUTION_TIME,
    OPENAI_COST_TOTAL,
    AGENT_REQUESTS_TOTAL,
    ACTIVE_AGENTS
)
from agents.redis_state import state_manager

logger = structlog.get_logger(__name__)

# FastAPI app instance
app = FastAPI(
    title="MedBillGuard Agent Metrics",
    description="Prometheus metrics and health checks for agent infrastructure",
    version="1.0.0"
)

# Global health status
health_status = {
    "startup_time": datetime.utcnow().isoformat(),
    "last_health_check": None,
    "redis_healthy": False,
    "selftest_enabled": False
}


@app.on_event("startup")
async def startup_event():
    """Initialize metrics server on startup."""
    logger.info("Starting metrics server")
    
    # Test Redis connection
    try:
        await state_manager.connect()
        stats = await state_manager.get_stats()
        if "error" not in stats:
            health_status["redis_healthy"] = True
            logger.info("Redis connection verified", stats=stats)
        else:
            logger.warning("Redis connection degraded", error=stats.get("error"))
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        health_status["redis_healthy"] = False
    
    health_status["last_health_check"] = datetime.utcnow().isoformat()
    logger.info("Metrics server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down metrics server")


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns all agent metrics in Prometheus exposition format.
    """
    try:
        # Generate Prometheus metrics
        metrics_data = generate_latest()
        
        # Add custom agent metrics if needed
        custom_metrics = await _get_custom_metrics()
        
        response_data = metrics_data.decode('utf-8')
        if custom_metrics:
            response_data += f"\n{custom_metrics}"
        
        return Response(
            content=response_data,
            media_type=CONTENT_TYPE_LATEST
        )
        
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e), exc_info=True)
        return Response(
            content="# Failed to generate metrics\n",
            media_type=CONTENT_TYPE_LATEST,
            status_code=500
        )


@app.get("/healthz")
async def health_check(selftest: bool = False):
    """
    Kubernetes health check endpoint.
    
    Args:
        selftest: If true, performs comprehensive self-test including Redis connectivity
        
    Returns:
        200 OK if healthy, 503 Service Unavailable if unhealthy
    """
    current_time = datetime.utcnow().isoformat()
    health_status["last_health_check"] = current_time
    
    health_data = {
        "status": "healthy",
        "timestamp": current_time,
        "startup_time": health_status["startup_time"],
        "checks": {}
    }
    
    status_code = status.HTTP_200_OK
    
    try:
        # Basic health checks
        health_data["checks"]["server"] = "healthy"
        
        # Redis connectivity check
        if selftest or health_status["selftest_enabled"]:
            redis_status = await _check_redis_health()
            health_data["checks"]["redis"] = redis_status
            
            if redis_status != "healthy":
                health_data["status"] = "degraded"
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            # Use cached Redis status
            health_data["checks"]["redis"] = "healthy" if health_status["redis_healthy"] else "degraded"
        
        # Agent metrics check
        metrics_status = await _check_metrics_health()
        health_data["checks"]["metrics"] = metrics_status
        
        if metrics_status != "healthy":
            health_data["status"] = "degraded"
            # Don't fail health check for metrics issues
        
        # Overall status
        if health_data["status"] != "healthy":
            logger.warning("Health check degraded", health_data=health_data)
        
        return Response(
            content=health_data,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e), exc_info=True)
        
        return Response(
            content={
                "status": "unhealthy",
                "timestamp": current_time,
                "error": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@app.get("/healthz/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Returns 200 OK when the service is ready to accept traffic.
    """
    try:
        # Check if Redis is accessible
        redis_healthy = await _check_redis_health()
        
        if redis_healthy == "healthy":
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            logger.warning("Readiness check failed - Redis unhealthy")
            return Response(
                content={"status": "not_ready", "reason": "redis_unhealthy"},
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return Response(
            content={"status": "not_ready", "error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@app.get("/healthz/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Returns 200 OK if the service is alive and should not be restarted.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": _get_uptime_seconds()
    }


@app.get("/stats")
async def get_stats():
    """
    Agent statistics endpoint for monitoring and debugging.
    
    Returns detailed statistics about agent performance and Redis state.
    """
    try:
        stats = {
            "server": {
                "status": "running",
                "startup_time": health_status["startup_time"],
                "uptime_seconds": _get_uptime_seconds(),
                "last_health_check": health_status["last_health_check"]
            }
        }
        
        # Get Redis statistics
        try:
            redis_stats = await state_manager.get_stats()
            stats["redis"] = redis_stats
        except Exception as e:
            stats["redis"] = {"error": str(e)}
        
        # Get Prometheus metrics summary
        stats["metrics"] = await _get_metrics_summary()
        
        return stats
        
    except Exception as e:
        logger.error("Failed to get stats", error=str(e), exc_info=True)
        return Response(
            content={"error": str(e)},
            status_code=500
        )


# Helper functions

async def _check_redis_health() -> str:
    """Check Redis connectivity and return status."""
    try:
        if not state_manager.redis_client:
            await state_manager.connect()
        
        await state_manager.redis_client.ping()
        health_status["redis_healthy"] = True
        return "healthy"
        
    except Exception as e:
        logger.warning("Redis health check failed", error=str(e))
        health_status["redis_healthy"] = False
        return "unhealthy"


async def _check_metrics_health() -> str:
    """Check metrics collection health."""
    try:
        # Try to generate metrics
        metrics_data = generate_latest()
        
        if metrics_data:
            return "healthy"
        else:
            return "degraded"
            
    except Exception as e:
        logger.warning("Metrics health check failed", error=str(e))
        return "unhealthy"


async def _get_custom_metrics() -> str:
    """Generate custom agent metrics."""
    try:
        custom_metrics = []
        
        # Add Redis connection status
        redis_status = 1 if health_status["redis_healthy"] else 0
        custom_metrics.append(f"agent_redis_connected {redis_status}")
        
        # Add server uptime
        uptime = _get_uptime_seconds()
        custom_metrics.append(f"agent_server_uptime_seconds {uptime}")
        
        # Add current timestamp
        timestamp = int(time.time())
        custom_metrics.append(f"agent_server_timestamp {timestamp}")
        
        return "\n".join(custom_metrics)
        
    except Exception as e:
        logger.error("Failed to generate custom metrics", error=str(e))
        return ""


async def _get_metrics_summary() -> Dict[str, Any]:
    """Get summary of current metrics."""
    try:
        # This would typically query the Prometheus metrics registry
        # For now, return basic info
        return {
            "total_requests": "Available via /metrics",
            "active_agents": "Available via /metrics", 
            "execution_time": "Available via /metrics",
            "cost_tracking": "Available via /metrics"
        }
    except Exception as e:
        return {"error": str(e)}


def _get_uptime_seconds() -> float:
    """Calculate server uptime in seconds."""
    try:
        startup_time = datetime.fromisoformat(health_status["startup_time"].replace('Z', '+00:00'))
        current_time = datetime.utcnow().replace(tzinfo=startup_time.tzinfo)
        uptime = (current_time - startup_time).total_seconds()
        return uptime
    except Exception:
        return 0.0


# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "metrics_server:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    ) 