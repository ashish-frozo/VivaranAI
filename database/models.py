"""
Database Models and Connection Management for VivaranAI Production

Implements:
- SQLAlchemy models for audit logs, users, API keys
- PostgreSQL connection pooling with async support
- Database migrations with Alembic
- Audit trail for all operations
"""

import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, DateTime, Boolean, Text, JSON, 
    ForeignKey, Index, func, BigInteger, Numeric, create_engine, select
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.pool import QueuePool
import structlog

logger = structlog.get_logger(__name__)

# Database configuration
class DatabaseConfig:
    """Database configuration"""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql://localhost/vivaranai")
        self.async_database_url = os.getenv(
            "ASYNC_DATABASE_URL", 
            self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        )
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))
        self.echo = os.getenv("DB_ECHO", "false").lower() == "true"

db_config = DatabaseConfig()

# Database base
Base = declarative_base()

# Enums
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    API_CLIENT = "api_client"
    VIEWER = "viewer"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AuditAction(str, Enum):
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ANALYZE = "analyze"
    EXPORT = "export"

# Models
class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default=UserRole.USER)
    
    # Authentication
    hashed_password = Column(String(255), nullable=True)  # For local auth
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("BillAnalysis", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_role", "role"),
        Index("idx_users_created_at", "created_at"),
    )

class APIKey(Base):
    """API Key model"""
    __tablename__ = "api_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Permissions
    role = Column(String(50), nullable=False, default=UserRole.API_CLIENT)
    permissions = Column(JSON, nullable=False, default=list)
    
    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=100)
    rate_limit_per_hour = Column(Integer, default=1000)
    rate_limit_per_day = Column(Integer, default=10000)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(BigInteger, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index("idx_api_keys_hash", "key_hash"),
        Index("idx_api_keys_user_id", "user_id"),
        Index("idx_api_keys_active", "is_active"),
    )

class BillAnalysis(Base):
    """Bill Analysis model"""
    __tablename__ = "bill_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Analysis details
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(100), nullable=False)
    
    # Analysis results
    status = Column(String(50), nullable=False, default=AnalysisStatus.PENDING)
    analysis_type = Column(String(100), nullable=False)  # medical_bill, insurance_claim, etc.
    
    # Results
    total_amount = Column(Numeric(10, 2), nullable=True)
    suspected_overcharges = Column(Numeric(10, 2), nullable=True)
    accuracy_score = Column(Numeric(5, 4), nullable=True)
    confidence_level = Column(Numeric(5, 4), nullable=True)
    
    # Detailed results
    raw_analysis = Column(JSON, nullable=True)
    structured_results = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Processing details
    processing_time = Column(Numeric(10, 3), nullable=True)  # seconds
    agent_used = Column(String(100), nullable=True)
    ai_model_used = Column(String(100), nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="analyses")
    
    # Indexes
    __table_args__ = (
        Index("idx_analyses_user_id", "user_id"),
        Index("idx_analyses_status", "status"),
        Index("idx_analyses_created_at", "created_at"),
        Index("idx_analyses_file_hash", "file_hash"),
        Index("idx_analyses_type", "analysis_type"),
    )

class AuditLog(Base):
    """Audit Log model"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event details
    action = Column(String(50), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255), nullable=True)
    
    # Request details
    endpoint = Column(String(255), nullable=True)
    method = Column(String(10), nullable=True)
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Response details
    status_code = Column(Integer, nullable=True)
    response_time = Column(Numeric(10, 3), nullable=True)
    
    # Additional context
    details = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_action", "action"),
        Index("idx_audit_logs_resource", "resource_type"),
        Index("idx_audit_logs_created_at", "created_at"),
        Index("idx_audit_logs_ip", "ip_address"),
    )

class SystemMetrics(Base):
    """System Metrics model"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric details
    metric_name = Column(String(255), nullable=False)
    metric_value = Column(Numeric(15, 6), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    
    # Labels/Tags
    labels = Column(JSON, nullable=True)
    
    # Timestamp
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indexes
    __table_args__ = (
        Index("idx_metrics_name", "metric_name"),
        Index("idx_metrics_timestamp", "timestamp"),
        Index("idx_metrics_type", "metric_type"),
    )

# Database engine and session management
class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
    def initialize_sync(self):
        """Initialize synchronous database connection"""
        self.engine = create_engine(
            db_config.database_url,
            poolclass=QueuePool,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_timeout=db_config.pool_timeout,
            pool_recycle=db_config.pool_recycle,
            echo=db_config.echo
        )
        
        self.session_factory = sessionmaker(
            bind=self.engine,
            expire_on_commit=False
        )
        
        logger.info("Synchronous database connection initialized")
    
    def initialize_async(self):
        """Initialize asynchronous database connection"""
        self.async_engine = create_async_engine(
            db_config.async_database_url,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_timeout=db_config.pool_timeout,
            pool_recycle=db_config.pool_recycle,
            echo=db_config.echo
        )
        
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Asynchronous database connection initialized")
    
    def get_session(self):
        """Get synchronous database session"""
        if not self.session_factory:
            self.initialize_sync()
        return self.session_factory()
    
    def get_async_session(self):
        """Get asynchronous database session"""
        if not self.async_session_factory:
            self.initialize_async()
        return self.async_session_factory()
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()

# Dependency injection helpers
def get_db():
    """Get database session for dependency injection"""
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()

async def get_async_db():
    """Get async database session for dependency injection"""
    async with db_manager.get_async_session() as session:
        yield session

# Repository classes
class UserRepository:
    """User repository for database operations"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(self, user_data: Dict[str, Any]) -> User:
        """Create a new user"""
        user = User(**user_data)
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user
    
    async def get_user_by_id(self, user_id: uuid.UUID) -> Optional[User]:
        """Get user by ID"""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def update_user(self, user_id: uuid.UUID, updates: Dict[str, Any]) -> Optional[User]:
        """Update user"""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user:
            for key, value in updates.items():
                setattr(user, key, value)
            await self.session.commit()
            await self.session.refresh(user)
        return user
    
    async def delete_user(self, user_id: uuid.UUID) -> bool:
        """Delete user"""
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        if user:
            await self.session.delete(user)
            await self.session.commit()
            return True
        return False

class BillAnalysisRepository:
    """Bill Analysis repository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_analysis(self, analysis_data: Dict[str, Any]) -> BillAnalysis:
        """Create new bill analysis"""
        analysis = BillAnalysis(**analysis_data)
        self.session.add(analysis)
        await self.session.commit()
        await self.session.refresh(analysis)
        return analysis
    
    async def get_analysis_by_id(self, analysis_id: uuid.UUID) -> Optional[BillAnalysis]:
        """Get analysis by ID"""
        result = await self.session.execute(
            select(BillAnalysis).where(BillAnalysis.id == analysis_id)
        )
        return result.scalar_one_or_none()
    
    async def update_analysis(self, analysis_id: uuid.UUID, updates: Dict[str, Any]) -> Optional[BillAnalysis]:
        """Update analysis"""
        result = await self.session.execute(
            select(BillAnalysis).where(BillAnalysis.id == analysis_id)
        )
        analysis = result.scalar_one_or_none()
        if analysis:
            for key, value in updates.items():
                setattr(analysis, key, value)
            await self.session.commit()
            await self.session.refresh(analysis)
        return analysis
    
    async def get_user_analyses(self, user_id: uuid.UUID, limit: int = 50) -> List[BillAnalysis]:
        """Get user's analyses"""
        result = await self.session.execute(
            select(BillAnalysis)
            .where(BillAnalysis.user_id == user_id)
            .order_by(BillAnalysis.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

class AuditLogRepository:
    """Audit Log repository"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_audit_log(self, log_data: Dict[str, Any]) -> AuditLog:
        """Create audit log entry"""
        audit_log = AuditLog(**log_data)
        self.session.add(audit_log)
        await self.session.commit()
        return audit_log
    
    async def get_audit_logs(
        self, 
        user_id: Optional[uuid.UUID] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs with filters"""
        query = select(AuditLog)
        
        if user_id:
            query = query.where(AuditLog.user_id == user_id)
        if action:
            query = query.where(AuditLog.action == action)
        if resource_type:
            query = query.where(AuditLog.resource_type == resource_type)
        
        query = query.order_by(AuditLog.created_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()

# Database initialization
async def create_tables():
    """Create all database tables"""
    if not db_manager.async_engine:
        db_manager.initialize_async()
    
    async with db_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database tables created successfully")

async def drop_tables():
    """Drop all database tables"""
    if not db_manager.async_engine:
        db_manager.initialize_async()
    
    async with db_manager.async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.info("Database tables dropped successfully") 