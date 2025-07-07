#!/usr/bin/env python3
"""
Railway Database Migration Script
Handles database migrations for Railway deployment
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_migrations():
    """Run database migrations for Railway"""
    try:
        # Import database models and create tables
        from database.models import Base, engine
        from sqlalchemy import text
        
        logger.info("Starting Railway database migrations...")
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
            
        # Check database connectivity
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info(f"Database connectivity check: {result.scalar()}")
            
        logger.info("Railway database migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        sys.exit(1)

def main():
    """Main entry point for Railway migrations"""
    try:
        logger.info("Railway Migration Script - Starting...")
        
        # Check if DATABASE_URL is set
        if not os.getenv("DATABASE_URL"):
            logger.warning("DATABASE_URL not set - skipping migrations")
            return
        
        # Run migrations
        asyncio.run(run_migrations())
        
    except Exception as e:
        logger.error(f"Migration script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 