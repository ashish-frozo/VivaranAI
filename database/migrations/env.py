"""
Database Migration Environment for VivaranAI

This file configures Alembic to work with our SQLAlchemy models
and supports both synchronous and asynchronous database operations.
"""

import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from alembic import context

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import our models and database configuration
from database.models import Base, db_config
from config.env_config import config as app_config

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def get_database_url():
    """Get database URL from environment configuration"""
    # Try to get from environment first
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    
    # Fall back to config
    try:
        return app_config.database_url if hasattr(app_config, 'database_url') else db_config.database_url
    except:
        return "postgresql://localhost/vivaranai"

def get_async_database_url():
    """Get async database URL from environment configuration"""
    # Try to get from environment first
    async_db_url = os.getenv("ASYNC_DATABASE_URL")
    if async_db_url:
        return async_db_url
    
    # Fall back to config
    try:
        return app_config.async_database_url if hasattr(app_config, 'async_database_url') else db_config.async_database_url
    except:
        return "postgresql+asyncpg://localhost/vivaranai"

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Override the sqlalchemy.url in alembic.ini
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_database_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

async def run_async_migrations() -> None:
    """Run migrations in async mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_async_database_url()
    
    async_engine = create_async_engine(
        get_async_database_url(),
        echo=False,
        future=True,
    )
    
    async with async_engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    
    await async_engine.dispose()

def do_run_migrations(connection):
    """Helper function to run migrations with connection"""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()

# Determine if we're running in async mode
if os.getenv("ASYNC_MIGRATIONS", "false").lower() == "true":
    import asyncio
    asyncio.run(run_async_migrations())
elif context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
