#!/usr/bin/env python3
"""
Railway-specific startup script for MedBillGuard Agent Server.
Handles Railway cold starts, agent registration, and ensures system availability.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any

import structlog
import uvicorn

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = structlog.get_logger(__name__)


async def wait_for_services():
    """Wait for required services to be available."""
    logger.info("Waiting for required services to be available...")
    
    # Wait for Redis if configured
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        import redis.asyncio as redis
        
        max_retries = 10
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                redis_client = redis.from_url(redis_url)
                await redis_client.ping()
                await redis_client.close()
                logger.info("Redis connection successful", attempt=attempt + 1)
                break
            except Exception as e:
                logger.warning(f"Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error("Max Redis connection attempts reached")
                    return False
    
    # Check OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return False
    
    logger.info("All required services are available")
    return True


async def pre_warm_system():
    """Pre-warm the system to reduce cold start impact."""
    logger.info("Pre-warming system components...")
    
    try:
        # Import and initialize core components
        from agents.agent_registry import AgentRegistry
        from agents.medical_bill_agent import MedicalBillAgent
        from agents.redis_state import RedisStateManager
        
        # Pre-create instances (they'll be recreated in main app)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        # Test Redis connection
        redis_manager = RedisStateManager(redis_url)
        await redis_manager.connect()
        await redis_manager.ping()
        await redis_manager.close()
        
        logger.info("System pre-warming completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"System pre-warming failed: {e}")
        return False


async def startup_checks():
    """Perform comprehensive startup checks."""
    logger.info("Starting Railway-specific startup checks...")
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = ["REDIS_URL", "MAX_CONCURRENT_REQUESTS", "ESTIMATED_RESPONSE_TIME_MS"]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        logger.error(f"Missing required environment variables: {missing_required}")
        return False
    
    # Log configuration
    logger.info("Configuration check:")
    for var in required_vars + optional_vars:
        value = os.getenv(var)
        if var in ["OPENAI_API_KEY"]:
            # Don't log sensitive values
            logger.info(f"  {var}: {'✓ Set' if value else '✗ Not set'}")
        else:
            logger.info(f"  {var}: {value or 'Not set'}")
    
    # Wait for services
    if not await wait_for_services():
        return False
    
    # Pre-warm system
    if not await pre_warm_system():
        logger.warning("System pre-warming failed, continuing anyway")
    
    # Initialize database tables
    if not await create_database_tables():
        logger.warning("Database table creation failed, using in-memory fallback")
    
    logger.info("Startup checks completed successfully")
    return True


async def create_database_tables():
    """Create database tables during Railway startup with enhanced diagnostics"""
    try:
        logger.info("Starting database table creation process during Railway startup...")
        
        # Check if DATABASE_URL is available
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning("DATABASE_URL not found in environment - using in-memory fallback")
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
        logger.error(f"Error creating database tables during Railway startup: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main():
    """Main entry point for Railway deployment."""
    logger.info("Starting MedBillGuard Agent Server on Railway")
    
    # Run startup checks
    if not asyncio.run(startup_checks()):
        logger.error("Startup checks failed, exiting")
        sys.exit(1)
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Railway-specific configuration
    railway_config = {
        "host": host,
        "port": port,
        "workers": 1,  # Single worker for Railway
        "timeout_keep_alive": 65,  # Railway requires this
        "timeout_graceful_shutdown": 10,
        "log_level": "info",
        "access_log": True,
        "use_colors": False,  # Railway doesn't support colors
        "loop": "asyncio",
        "reload": False,  # Never reload in production
    }
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Configuration: {railway_config}")
    
    # Import the app after startup checks
    from agents.server import app
    
    # Run the server
    try:
        uvicorn.run(app, **railway_config)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 