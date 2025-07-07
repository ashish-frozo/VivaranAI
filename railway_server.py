#!/usr/bin/env python3
"""
Railway-optimized server for MedBillGuardAgent
Handles Railway-specific configurations and deployment requirements
"""

import os
import sys
import asyncio
import uvicorn
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_railway_environment():
    """Setup Railway-specific environment variables"""
    
    # Railway provides PORT automatically
    port = int(os.getenv("PORT", "8001"))
    
    # Set Railway-specific defaults
    os.environ.setdefault("HOST", "0.0.0.0")
    os.environ.setdefault("PORT", str(port))
    os.environ.setdefault("ENVIRONMENT", "production")
    os.environ.setdefault("DEBUG", "false")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    # Railway database URLs (set via Railway dashboard)
    if not os.getenv("DATABASE_URL"):
        logger.warning("DATABASE_URL not set - using SQLite fallback")
        os.environ["DATABASE_URL"] = "sqlite:///./railway_medbillguard.db"
    
    if not os.getenv("ASYNC_DATABASE_URL"):
        db_url = os.getenv("DATABASE_URL", "")
        if db_url.startswith("postgresql://"):
            async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        else:
            async_db_url = "sqlite+aiosqlite:///./railway_medbillguard.db"
        os.environ["ASYNC_DATABASE_URL"] = async_db_url
    
    # Railway Redis URL (set via Railway dashboard)
    if not os.getenv("REDIS_URL"):
        logger.warning("REDIS_URL not set - Redis features will be disabled")
    
    # JWT Secret (set via Railway dashboard)
    if not os.getenv("JWT_SECRET_KEY"):
        logger.warning("JWT_SECRET_KEY not set - using default")
        os.environ["JWT_SECRET_KEY"] = "railway-default-secret-key"
    
    # Performance settings for Railway
    os.environ.setdefault("MAX_WORKERS", "4")
    os.environ.setdefault("TIMEOUT_SECONDS", "30")
    os.environ.setdefault("API_RATE_LIMIT", "100")
    os.environ.setdefault("FORCE_HTTPS", "true")
    
    logger.info(f"Railway environment setup complete - Port: {port}")
    return port

def main():
    """Main entry point for Railway deployment"""
    try:
        # Setup Railway environment
        port = setup_railway_environment()
        
        # Import the FastAPI app
        from agents.server import app
        
        # Add Railway-specific health check
        @app.get("/railway/health")
        async def railway_health():
            return {"status": "healthy", "service": "MedBillGuardAgent", "platform": "Railway"}
        
        # Add Railway-specific info endpoint
        @app.get("/railway/info")
        async def railway_info():
            return {
                "service": "MedBillGuardAgent",
                "platform": "Railway",
                "environment": os.getenv("ENVIRONMENT", "production"),
                "version": "1.0.0",
                "database": "connected" if os.getenv("DATABASE_URL") else "fallback",
                "redis": "connected" if os.getenv("REDIS_URL") else "disabled"
            }
        
        # Start the server
        logger.info("Starting MedBillGuardAgent on Railway...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
            access_log=True,
            loop="asyncio"
        )
        
    except Exception as e:
        logger.error(f"Failed to start Railway server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 