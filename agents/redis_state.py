"""
Redis State Management - Hash-based state management for agent coordination.

Provides atomic operations for document state sharing between agents with TTL management.
Uses single Redis hash per document for atomic multi-field updates and simpler housekeeping.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

import redis.asyncio as redis
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DocumentState:
    """Structured document state for Redis storage."""
    doc_id: str
    ocr_text: str
    line_items: List[Dict[str, Any]]
    document_metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    
    def to_redis_hash(self) -> Dict[str, str]:
        """Convert to Redis hash format (all values as strings)."""
        return {
            "doc_id": self.doc_id,
            "ocr_text": self.ocr_text,
            "line_items": json.dumps(self.line_items),
            "document_metadata": json.dumps(self.document_metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_redis_hash(cls, doc_id: str, hash_data: Dict[str, str]) -> "DocumentState":
        """Create DocumentState from Redis hash data."""
        return cls(
            doc_id=doc_id,
            ocr_text=hash_data.get("ocr_text", ""),
            line_items=json.loads(hash_data.get("line_items", "[]")),
            document_metadata=json.loads(hash_data.get("document_metadata", "{}")),
            created_at=hash_data.get("created_at", ""),
            updated_at=hash_data.get("updated_at", "")
        )


@dataclass
class AgentResultCache:
    """Cache structure for agent execution results."""
    agent_id: str
    success: bool
    data: Dict[str, Any]
    execution_time_ms: int
    cost_rupees: float
    confidence: float
    error: Optional[str]
    timestamp: str
    
    def to_redis_value(self) -> str:
        """Convert to JSON string for Redis storage."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_redis_value(cls, value: str) -> "AgentResultCache":
        """Create from Redis JSON string."""
        data = json.loads(value)
        return cls(**data)


class RedisStateManager:
    """
    Redis-based state management for multi-agent coordination.
    
    Key patterns:
    - doc:{doc_id} -> Hash with document state
    - doc:{doc_id}:results:{agent_id} -> Agent execution results
    - doc:{doc_id}:lock -> Coordination locks
    - file_hash:{sha256} -> File content hash cache
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/1"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # TTL configurations (in seconds)
        self.DOC_STATE_TTL = 24 * 60 * 60  # 24 hours
        self.LINE_ITEMS_TTL = 6 * 60 * 60   # 6 hours
        self.AGENT_RESULTS_TTL = 6 * 60 * 60  # 6 hours
        self.FILE_HASH_TTL = 24 * 60 * 60   # 24 hours
        self.LOCK_TTL = 5 * 60              # 5 minutes
    
    async def connect(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis state manager connected successfully")
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e), exc_info=True)
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis state manager disconnected")
    
    async def ping(self):
        """Test Redis connection."""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        return await self.redis_client.ping()
    
    # Document State Management
    
    async def store_document_state(
        self,
        doc_id: str,
        ocr_text: str,
        line_items: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Store complete document state atomically.
        
        Args:
            doc_id: Document identifier
            ocr_text: Extracted text from OCR
            line_items: Parsed line items
            metadata: Document metadata
            
        Returns:
            True if stored successfully
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            now = datetime.utcnow().isoformat()
            
            doc_state = DocumentState(
                doc_id=doc_id,
                ocr_text=ocr_text,
                line_items=line_items,
                document_metadata=metadata,
                created_at=now,
                updated_at=now
            )
            
            hash_key = f"doc:{doc_id}"
            hash_data = doc_state.to_redis_hash()
            
            # Atomic hash set with TTL
            pipe = self.redis_client.pipeline()
            pipe.hset(hash_key, mapping=hash_data)
            pipe.expire(hash_key, self.DOC_STATE_TTL)
            await pipe.execute()
            
            logger.info(
                "Document state stored",
                doc_id=doc_id,
                ocr_text_length=len(ocr_text),
                line_items_count=len(line_items),
                ttl_hours=self.DOC_STATE_TTL / 3600
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to store document state",
                doc_id=doc_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_document_state(self, doc_id: str) -> Optional[DocumentState]:
        """
        Retrieve complete document state.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            DocumentState if found, None otherwise
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            hash_key = f"doc:{doc_id}"
            hash_data = await self.redis_client.hgetall(hash_key)
            
            if not hash_data:
                logger.debug("Document state not found", doc_id=doc_id)
                return None
            
            # Convert bytes keys/values to strings
            str_hash_data = {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v
                for k, v in hash_data.items()
            }
            
            doc_state = DocumentState.from_redis_hash(doc_id, str_hash_data)
            
            logger.debug(
                "Document state retrieved",
                doc_id=doc_id,
                line_items_count=len(doc_state.line_items)
            )
            
            return doc_state
            
        except Exception as e:
            logger.error(
                "Failed to retrieve document state",
                doc_id=doc_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def update_document_field(
        self,
        doc_id: str,
        field_updates: Dict[str, Any]
    ) -> bool:
        """
        Update specific fields in document state atomically.
        
        Args:
            doc_id: Document identifier
            field_updates: Fields to update
            
        Returns:
            True if updated successfully
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            hash_key = f"doc:{doc_id}"
            
            # Prepare updates with JSON serialization for complex fields
            redis_updates = {}
            for field, value in field_updates.items():
                if isinstance(value, (dict, list)):
                    redis_updates[field] = json.dumps(value)
                else:
                    redis_updates[field] = str(value)
            
            # Add updated timestamp
            redis_updates["updated_at"] = datetime.utcnow().isoformat()
            
            # Atomic update
            await self.redis_client.hset(hash_key, mapping=redis_updates)
            
            logger.info(
                "Document state updated",
                doc_id=doc_id,
                fields_updated=list(field_updates.keys())
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to update document state",
                doc_id=doc_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    # Agent Results Caching
    
    async def cache_agent_result(
        self,
        doc_id: str,
        agent_id: str,
        result_data: Dict[str, Any]
    ) -> bool:
        """
        Cache agent execution result for coordination.
        
        Args:
            doc_id: Document identifier
            agent_id: Agent identifier
            result_data: Agent result data
            
        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            result_cache = AgentResultCache(
                agent_id=agent_id,
                success=result_data.get("success", False),
                data=result_data.get("data", {}),
                execution_time_ms=result_data.get("execution_time_ms", 0),
                cost_rupees=result_data.get("cost_rupees", 0.0),
                confidence=result_data.get("confidence", 1.0),
                error=result_data.get("error"),
                timestamp=datetime.utcnow().isoformat()
            )
            
            cache_key = f"doc:{doc_id}:results:{agent_id}"
            await self.redis_client.setex(
                cache_key,
                self.AGENT_RESULTS_TTL,
                result_cache.to_redis_value()
            )
            
            logger.info(
                "Agent result cached",
                doc_id=doc_id,
                agent_id=agent_id,
                success=result_cache.success,
                confidence=result_cache.confidence
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cache agent result",
                doc_id=doc_id,
                agent_id=agent_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_agent_result(
        self,
        doc_id: str,
        agent_id: str
    ) -> Optional[AgentResultCache]:
        """
        Retrieve cached agent result.
        
        Args:
            doc_id: Document identifier
            agent_id: Agent identifier
            
        Returns:
            AgentResultCache if found, None otherwise
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            cache_key = f"doc:{doc_id}:results:{agent_id}"
            cached_data = await self.redis_client.get(cache_key)
            
            if not cached_data:
                return None
            
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode()
            
            result_cache = AgentResultCache.from_redis_value(cached_data)
            
            logger.debug(
                "Agent result retrieved from cache",
                doc_id=doc_id,
                agent_id=agent_id
            )
            
            return result_cache
            
        except Exception as e:
            logger.error(
                "Failed to retrieve agent result",
                doc_id=doc_id,
                agent_id=agent_id,
                error=str(e),
                exc_info=True
            )
            return None
    
    # File Hash Caching (24h cache for duplicate processing)
    
    async def cache_file_result(
        self,
        file_hash: str,
        result_data: Dict[str, Any]
    ) -> bool:
        """
        Cache processing result by file SHA-256 hash.
        
        Args:
            file_hash: SHA-256 hash of file content
            result_data: Processing result to cache
            
        Returns:
            True if cached successfully
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            cache_key = f"file_hash:{file_hash}"
            cache_data = json.dumps(result_data)
            
            await self.redis_client.setex(
                cache_key,
                self.FILE_HASH_TTL,
                cache_data
            )
            
            logger.info(
                "File result cached by hash",
                file_hash=file_hash[:16] + "...",
                ttl_hours=self.FILE_HASH_TTL / 3600
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to cache file result",
                file_hash=file_hash[:16] + "...",
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_cached_file_result(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result by file hash.
        
        Args:
            file_hash: SHA-256 hash of file content
            
        Returns:
            Cached result if found, None otherwise
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            cache_key = f"file_hash:{file_hash}"
            cached_data = await self.redis_client.get(cache_key)
            
            if not cached_data:
                return None
            
            if isinstance(cached_data, bytes):
                cached_data = cached_data.decode()
            
            result_data = json.loads(cached_data)
            
            logger.info(
                "File result retrieved from hash cache",
                file_hash=file_hash[:16] + "..."
            )
            
            return result_data
            
        except Exception as e:
            logger.error(
                "Failed to retrieve cached file result",
                file_hash=file_hash[:16] + "...",
                error=str(e),
                exc_info=True
            )
            return None
    
    # Coordination Locks (for agent sequencing when needed)
    
    async def acquire_lock(
        self,
        lock_key: str,
        lock_value: str,
        ttl_seconds: int = None
    ) -> bool:
        """
        Acquire distributed lock for coordination.
        
        Args:
            lock_key: Lock identifier
            lock_value: Unique value for this lock holder
            ttl_seconds: Lock TTL (defaults to 5 minutes)
            
        Returns:
            True if lock acquired
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        ttl = ttl_seconds or self.LOCK_TTL
        
        try:
            # SET with NX (only if not exists) and EX (expiration)
            result = await self.redis_client.set(
                f"lock:{lock_key}",
                lock_value,
                nx=True,
                ex=ttl
            )
            
            acquired = result is True
            
            if acquired:
                logger.debug(
                    "Lock acquired",
                    lock_key=lock_key,
                    lock_value=lock_value,
                    ttl_seconds=ttl
                )
            else:
                logger.debug(
                    "Lock acquisition failed - already held",
                    lock_key=lock_key
                )
            
            return acquired
            
        except Exception as e:
            logger.error(
                "Failed to acquire lock",
                lock_key=lock_key,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def release_lock(self, lock_key: str, lock_value: str) -> bool:
        """
        Release distributed lock (only if we own it).
        
        Args:
            lock_key: Lock identifier
            lock_value: Expected lock value
            
        Returns:
            True if lock released
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        # Lua script for atomic check-and-delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        try:
            result = await self.redis_client.eval(
                lua_script,
                1,
                f"lock:{lock_key}",
                lock_value
            )
            
            released = result == 1
            
            if released:
                logger.debug(
                    "Lock released",
                    lock_key=lock_key,
                    lock_value=lock_value
                )
            else:
                logger.debug(
                    "Lock release failed - not owned or expired",
                    lock_key=lock_key
                )
            
            return released
            
        except Exception as e:
            logger.error(
                "Failed to release lock",
                lock_key=lock_key,
                error=str(e),
                exc_info=True
            )
            return False
    
    # Utility Methods
    
    async def cleanup_expired_documents(self) -> int:
        """
        Clean up expired document keys (housekeeping).
        
        Returns:
            Number of keys cleaned up
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            # Find all document keys
            doc_keys = await self.redis_client.keys("doc:*")
            expired_count = 0
            
            for key in doc_keys:
                if isinstance(key, bytes):
                    key = key.decode()
                
                # Check if key has TTL
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # No TTL set
                    # Set default TTL
                    await self.redis_client.expire(key, self.DOC_STATE_TTL)
                elif ttl == -2:  # Key doesn't exist
                    expired_count += 1
            
            if expired_count > 0:
                logger.info(
                    "Cleaned up expired document keys",
                    expired_count=expired_count
                )
            
            return expired_count
            
        except Exception as e:
            logger.error(
                "Failed to cleanup expired documents",
                error=str(e),
                exc_info=True
            )
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis state manager statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        
        try:
            # Count different key types
            doc_keys = await self.redis_client.keys("doc:*")
            lock_keys = await self.redis_client.keys("lock:*")
            hash_keys = await self.redis_client.keys("file_hash:*")
            
            # Get memory usage
            info = await self.redis_client.info("memory")
            
            return {
                "document_keys": len(doc_keys),
                "lock_keys": len(lock_keys),
                "file_hash_keys": len(hash_keys),
                "memory_used_bytes": info.get("used_memory", 0),
                "memory_used_human": info.get("used_memory_human", "0B"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(
                "Failed to get Redis stats",
                error=str(e),
                exc_info=True
            )
            return {"error": str(e)}


# Global instance for easy import
state_manager = RedisStateManager() 