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
    
    # Import config object after path setup
    from config.env_config import config

    # Railway provides PORT automatically, which overrides the config
    port = int(os.getenv("PORT", config.server.port))
    config.server.port = port
    
    # Set Railway-specific defaults in the config object
    config.app.environment = "production"
    config.app.debug = False
    config.logging.level = "INFO"
    
    # Railway database URLs (set via Railway dashboard)
    config.databases.audit_db.url = os.getenv("DATABASE_URL", config.databases.audit_db.url)
    
    # Railway Redis URL (set via Railway dashboard)
    config.databases.redis.url = os.getenv("REDIS_URL", config.databases.redis.url)
    
    # JWT Secret (set via Railway dashboard)
    config.security.jwt.secret_key = os.getenv("JWT_SECRET_KEY", config.security.jwt.secret_key)
    
    # Performance settings for Railway
    config.server.workers = int(os.getenv("MAX_WORKERS", config.server.workers))
    config.server.timeout = int(os.getenv("TIMEOUT_SECONDS", config.server.timeout))
    
    logger.info(f"Railway environment setup complete - Port: {port}")
    return port

async def run_database_migrations():
    """Run database migrations using Alembic"""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        logger.info("Running database migrations with Alembic...")
        alembic_ini_path = Path(__file__).parent / "alembic.ini"
        
        if not alembic_ini_path.exists():
            logger.error(f"Alembic configuration not found at {alembic_ini_path}")
            return False
        
        # Run alembic upgrade head
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Database migrations completed successfully")
            logger.debug(f"Migration output: {result.stdout}")
            return True
        else:
            logger.error(f"Database migration failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running database migrations: {e}")
        return False

def main():
    """Main entry point for Railway deployment"""
    try:
        # Setup Railway environment
        port = setup_railway_environment()
        
        # Run database migrations if DATABASE_URL is set
        if os.getenv("DATABASE_URL"):
            asyncio.run(run_database_migrations())
            logger.info("Database setup complete")
        else:
            logger.warning("DATABASE_URL not set, skipping migrations")
        
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
        from config.env_config import config
        uvicorn.run(
            app,
            host=config.server.host,
            port=config.server.port,
            log_level=config.logging.level.lower(),
            access_log=True,
            loop="asyncio"
        )
        
    except Exception as e:
        logger.error(f"Failed to start Railway server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 