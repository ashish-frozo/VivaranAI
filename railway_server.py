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

async def create_database_tables():
    """Create database tables using SQLAlchemy metadata"""
    try:
        logger.info("Starting database table creation process...")
        
        # Check if DATABASE_URL is available
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.error("DATABASE_URL not found in environment")
            return False
        
        logger.info(f"Database URL configured: {db_url[:50]}...")
        
        # Import and initialize database components
        from database.models import create_tables, db_manager
        
        logger.info("Initializing database manager...")
        if not db_manager.async_engine:
            db_manager.initialize_async()
            logger.info("Database manager initialized")
        
        logger.info("Creating database tables...")
        await create_tables()
        logger.info("Database tables created successfully")
        
        # Verify table creation
        logger.info("Verifying table creation...")
        async with db_manager.get_async_session() as session:
            result = await session.execute("SELECT 1")
            logger.info("Database connection verified")
        
        return True
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main entry point for Railway deployment"""
    try:
        # Setup Railway environment
        port = setup_railway_environment()
        
        # Create database tables if DATABASE_URL is set
        if os.getenv("DATABASE_URL"):
            asyncio.run(create_database_tables())
            logger.info("Database setup complete")
        else:
            logger.warning("DATABASE_URL not set, skipping database setup")
        
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