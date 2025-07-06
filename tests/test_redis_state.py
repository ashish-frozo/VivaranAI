"""
Unit tests for Redis State Management.

Tests hash-based document state management, agent result caching,
file hash caching, and coordination locks.
"""

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import pytest
import structlog

from agents.redis_state import (
    RedisStateManager,
    DocumentState,
    AgentResultCache,
    state_manager
)


@pytest.fixture
def mock_redis():
    """Mock Redis async client with all required methods."""
    client = AsyncMock()
    
    # Basic connection methods
    client.ping.return_value = True
    client.close.return_value = None
    
    # Hash operations
    client.hset.return_value = True
    client.hgetall.return_value = {}
    client.expire.return_value = True
    
    # String operations
    client.set.return_value = True
    client.setex.return_value = True
    client.get.return_value = None
    
    # Pipeline operations
    pipeline_mock = AsyncMock()
    pipeline_mock.hset.return_value = pipeline_mock
    pipeline_mock.expire.return_value = pipeline_mock
    pipeline_mock.execute.return_value = [True, True]
    client.pipeline.return_value = pipeline_mock
    
    # Utility operations
    client.keys.return_value = []
    client.ttl.return_value = -1
    client.eval.return_value = 1
    client.info.return_value = {
        "used_memory": 1024,
        "used_memory_human": "1.0K"
    }
    
    return client


@pytest.fixture
def redis_manager(mock_redis):
    """RedisStateManager instance with mocked Redis client."""
    manager = RedisStateManager()
    manager.redis_client = mock_redis
    return manager


@pytest.fixture
def sample_document_state():
    """Sample document state for testing."""
    return DocumentState(
        doc_id="test_doc_123",
        ocr_text="Sample OCR text content",
        line_items=[
            {"item": "Consultation", "amount": 500.0, "type": "service"},
            {"item": "Blood Test", "amount": 200.0, "type": "test"}
        ],
        document_metadata={
            "file_name": "test_bill.pdf",
            "pages": 2,
            "language": "en"
        },
        created_at="2024-01-01T12:00:00Z",
        updated_at="2024-01-01T12:00:00Z"
    )


@pytest.fixture
def sample_agent_result():
    """Sample agent result for testing."""
    return {
        "success": True,
        "data": {"analysis": "completed", "confidence": 0.95},
        "execution_time_ms": 1500,
        "cost_rupees": 2.50,
        "confidence": 0.95,
        "error": None
    }


class TestDocumentState:
    """Test DocumentState dataclass functionality."""
    
    def test_document_state_creation(self, sample_document_state):
        """Test DocumentState creation and basic properties."""
        doc_state = sample_document_state
        
        assert doc_state.doc_id == "test_doc_123"
        assert doc_state.ocr_text == "Sample OCR text content"
        assert len(doc_state.line_items) == 2
        assert doc_state.document_metadata["file_name"] == "test_bill.pdf"
        assert doc_state.created_at == "2024-01-01T12:00:00Z"
    
    def test_to_redis_hash(self, sample_document_state):
        """Test conversion to Redis hash format."""
        doc_state = sample_document_state
        redis_hash = doc_state.to_redis_hash()
        
        assert redis_hash["doc_id"] == "test_doc_123"
        assert redis_hash["ocr_text"] == "Sample OCR text content"
        assert redis_hash["created_at"] == "2024-01-01T12:00:00Z"
        assert redis_hash["updated_at"] == "2024-01-01T12:00:00Z"
        
        # Verify JSON serialization
        line_items = json.loads(redis_hash["line_items"])
        assert len(line_items) == 2
        assert line_items[0]["item"] == "Consultation"
        
        metadata = json.loads(redis_hash["document_metadata"])
        assert metadata["file_name"] == "test_bill.pdf"
    
    def test_from_redis_hash(self):
        """Test creation from Redis hash data."""
        hash_data = {
            "doc_id": "test_doc_456",
            "ocr_text": "Another OCR text",
            "line_items": json.dumps([{"item": "X-Ray", "amount": 300.0}]),
            "document_metadata": json.dumps({"file_name": "xray.pdf"}),
            "created_at": "2024-01-02T10:00:00Z",
            "updated_at": "2024-01-02T10:30:00Z"
        }
        
        doc_state = DocumentState.from_redis_hash("test_doc_456", hash_data)
        
        assert doc_state.doc_id == "test_doc_456"
        assert doc_state.ocr_text == "Another OCR text"
        assert len(doc_state.line_items) == 1
        assert doc_state.line_items[0]["item"] == "X-Ray"
        assert doc_state.document_metadata["file_name"] == "xray.pdf"
    
    def test_from_redis_hash_empty_data(self):
        """Test creation from empty Redis hash data."""
        hash_data = {}
        
        doc_state = DocumentState.from_redis_hash("empty_doc", hash_data)
        
        assert doc_state.doc_id == "empty_doc"
        assert doc_state.ocr_text == ""
        assert doc_state.line_items == []
        assert doc_state.document_metadata == {}


class TestAgentResultCache:
    """Test AgentResultCache dataclass functionality."""
    
    def test_agent_result_cache_creation(self):
        """Test AgentResultCache creation."""
        cache = AgentResultCache(
            agent_id="test_agent",
            success=True,
            data={"result": "success"},
            execution_time_ms=1000,
            cost_rupees=1.50,
            confidence=0.9,
            error=None,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        assert cache.agent_id == "test_agent"
        assert cache.success is True
        assert cache.data["result"] == "success"
        assert cache.execution_time_ms == 1000
        assert cache.cost_rupees == 1.50
        assert cache.confidence == 0.9
        assert cache.error is None
    
    def test_to_redis_value(self):
        """Test conversion to Redis value (JSON string)."""
        cache = AgentResultCache(
            agent_id="test_agent",
            success=True,
            data={"result": "success"},
            execution_time_ms=1000,
            cost_rupees=1.50,
            confidence=0.9,
            error=None,
            timestamp="2024-01-01T12:00:00Z"
        )
        
        redis_value = cache.to_redis_value()
        
        # Verify it's valid JSON
        parsed = json.loads(redis_value)
        assert parsed["agent_id"] == "test_agent"
        assert parsed["success"] is True
        assert parsed["data"]["result"] == "success"
        assert parsed["cost_rupees"] == 1.50
    
    def test_from_redis_value(self):
        """Test creation from Redis value (JSON string)."""
        redis_value = json.dumps({
            "agent_id": "test_agent",
            "success": False,
            "data": {"error": "failed"},
            "execution_time_ms": 500,
            "cost_rupees": 0.0,
            "confidence": 0.0,
            "error": "Processing failed",
            "timestamp": "2024-01-01T12:00:00Z"
        })
        
        cache = AgentResultCache.from_redis_value(redis_value)
        
        assert cache.agent_id == "test_agent"
        assert cache.success is False
        assert cache.data["error"] == "failed"
        assert cache.error == "Processing failed"


class TestRedisStateManagerInitialization:
    """Test RedisStateManager initialization and connection."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        manager = RedisStateManager()
        
        assert manager.redis_url == "redis://localhost:6379/1"
        assert manager.redis_client is None
        assert manager.DOC_STATE_TTL == 24 * 60 * 60  # 24 hours
        assert manager.LINE_ITEMS_TTL == 6 * 60 * 60   # 6 hours
        assert manager.AGENT_RESULTS_TTL == 6 * 60 * 60  # 6 hours
        assert manager.FILE_HASH_TTL == 24 * 60 * 60   # 24 hours
        assert manager.LOCK_TTL == 5 * 60              # 5 minutes
    
    def test_initialization_custom_url(self):
        """Test initialization with custom Redis URL."""
        manager = RedisStateManager("redis://custom:6379/2")
        assert manager.redis_url == "redis://custom:6379/2"
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis):
        """Test successful Redis connection."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = RedisStateManager()
            await manager.connect()
            
            assert manager.redis_client == mock_redis
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test Redis connection failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = RedisStateManager()
            
            with pytest.raises(Exception, match="Connection failed"):
                await manager.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_redis):
        """Test Redis disconnection."""
        manager = RedisStateManager()
        manager.redis_client = mock_redis
        
        await manager.disconnect()
        mock_redis.close.assert_called_once()


class TestDocumentStateManagement:
    """Test document state management operations."""
    
    @pytest.mark.asyncio
    async def test_store_document_state_success(self, redis_manager, mock_redis):
        """Test successful document state storage."""
        doc_id = "test_doc_123"
        ocr_text = "Sample OCR text"
        line_items = [{"item": "Test", "amount": 100.0}]
        metadata = {"file_name": "test.pdf"}
        
        result = await redis_manager.store_document_state(
            doc_id, ocr_text, line_items, metadata
        )
        
        assert result is True
        
        # Verify pipeline operations
        mock_redis.pipeline.assert_called_once()
        pipeline = mock_redis.pipeline.return_value
        pipeline.hset.assert_called_once()
        pipeline.expire.assert_called_once()
        pipeline.execute.assert_called_once()
        
        # Check the hash data passed to hset
        hset_call = pipeline.hset.call_args
        assert hset_call[0][0] == f"doc:{doc_id}"  # Key
        hash_data = hset_call[1]["mapping"]
        assert hash_data["doc_id"] == doc_id
        assert hash_data["ocr_text"] == ocr_text
        assert json.loads(hash_data["line_items"]) == line_items
        assert json.loads(hash_data["document_metadata"]) == metadata
    
    @pytest.mark.asyncio
    async def test_store_document_state_failure(self, redis_manager, mock_redis):
        """Test document state storage failure."""
        # Make pipeline execution fail
        pipeline = mock_redis.pipeline.return_value
        pipeline.execute.side_effect = Exception("Redis error")
        
        result = await redis_manager.store_document_state(
            "test_doc", "text", [], {}
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_document_state_success(self, redis_manager, mock_redis):
        """Test successful document state retrieval."""
        doc_id = "test_doc_123"
        
        # Mock Redis hash data
        hash_data = {
            b"doc_id": b"test_doc_123",
            b"ocr_text": b"Sample OCR text",
            b"line_items": b'[{"item": "Test", "amount": 100.0}]',
            b"document_metadata": b'{"file_name": "test.pdf"}',
            b"created_at": b"2024-01-01T12:00:00Z",
            b"updated_at": b"2024-01-01T12:00:00Z"
        }
        mock_redis.hgetall.return_value = hash_data
        
        doc_state = await redis_manager.get_document_state(doc_id)
        
        assert doc_state is not None
        assert doc_state.doc_id == doc_id
        assert doc_state.ocr_text == "Sample OCR text"
        assert len(doc_state.line_items) == 1
        assert doc_state.line_items[0]["item"] == "Test"
        assert doc_state.document_metadata["file_name"] == "test.pdf"
        
        mock_redis.hgetall.assert_called_once_with(f"doc:{doc_id}")
    
    @pytest.mark.asyncio
    async def test_get_document_state_not_found(self, redis_manager, mock_redis):
        """Test document state retrieval when not found."""
        mock_redis.hgetall.return_value = {}
        
        doc_state = await redis_manager.get_document_state("nonexistent_doc")
        
        assert doc_state is None
    
    @pytest.mark.asyncio
    async def test_get_document_state_error(self, redis_manager, mock_redis):
        """Test document state retrieval with Redis error."""
        mock_redis.hgetall.side_effect = Exception("Redis error")
        
        doc_state = await redis_manager.get_document_state("test_doc")
        
        assert doc_state is None
    
    @pytest.mark.asyncio
    async def test_update_document_field_success(self, redis_manager, mock_redis):
        """Test successful document field update."""
        doc_id = "test_doc_123"
        field_updates = {
            "ocr_text": "Updated OCR text",
            "line_items": [{"item": "Updated", "amount": 200.0}],
            "simple_field": "simple_value"
        }
        
        result = await redis_manager.update_document_field(doc_id, field_updates)
        
        assert result is True
        
        # Verify hset was called with correct data
        mock_redis.hset.assert_called_once()
        hset_call = mock_redis.hset.call_args
        assert hset_call[0][0] == f"doc:{doc_id}"  # Key
        
        mapping = hset_call[1]["mapping"]
        assert mapping["ocr_text"] == "Updated OCR text"
        assert json.loads(mapping["line_items"]) == [{"item": "Updated", "amount": 200.0}]
        assert mapping["simple_field"] == "simple_value"
        assert "updated_at" in mapping  # Timestamp should be added
    
    @pytest.mark.asyncio
    async def test_update_document_field_error(self, redis_manager, mock_redis):
        """Test document field update with Redis error."""
        mock_redis.hset.side_effect = Exception("Redis error")
        
        result = await redis_manager.update_document_field("test_doc", {"field": "value"})
        
        assert result is False


class TestAgentResultCaching:
    """Test agent result caching operations."""
    
    @pytest.mark.asyncio
    async def test_cache_agent_result_success(self, redis_manager, mock_redis):
        """Test successful agent result caching."""
        doc_id = "test_doc_123"
        agent_id = "test_agent"
        sample_result = {
            "success": True,
            "data": {"analysis": "completed"},
            "execution_time_ms": 1500,
            "cost_rupees": 2.50,
            "confidence": 0.95
        }
        
        result = await redis_manager.cache_agent_result(doc_id, agent_id, sample_result)
        
        assert result is True
        mock_redis.setex.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_agent_result_error(self, redis_manager, mock_redis):
        """Test agent result caching with Redis error."""
        mock_redis.setex.side_effect = Exception("Redis error")
        
        result = await redis_manager.cache_agent_result("doc", "agent", sample_agent_result)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_agent_result_success(self, redis_manager, mock_redis):
        """Test successful agent result retrieval."""
        doc_id = "test_doc_123"
        agent_id = "test_agent"
        
        # Mock cached data
        cached_data = json.dumps({
            "agent_id": agent_id,
            "success": True,
            "data": {"result": "cached"},
            "execution_time_ms": 1000,
            "cost_rupees": 1.50,
            "confidence": 0.9,
            "error": None,
            "timestamp": "2024-01-01T12:00:00Z"
        })
        mock_redis.get.return_value = cached_data.encode()
        
        result_cache = await redis_manager.get_agent_result(doc_id, agent_id)
        
        assert result_cache is not None
        assert result_cache.agent_id == agent_id
        assert result_cache.success is True
        assert result_cache.data["result"] == "cached"
        assert result_cache.cost_rupees == 1.50
        
        expected_key = f"doc:{doc_id}:results:{agent_id}"
        mock_redis.get.assert_called_once_with(expected_key)
    
    @pytest.mark.asyncio
    async def test_get_agent_result_not_found(self, redis_manager, mock_redis):
        """Test agent result retrieval when not found."""
        mock_redis.get.return_value = None
        
        result_cache = await redis_manager.get_agent_result("doc", "agent")
        
        assert result_cache is None
    
    @pytest.mark.asyncio
    async def test_get_agent_result_error(self, redis_manager, mock_redis):
        """Test agent result retrieval with Redis error."""
        mock_redis.get.side_effect = Exception("Redis error")
        
        result_cache = await redis_manager.get_agent_result("doc", "agent")
        
        assert result_cache is None


class TestFileHashCaching:
    """Test file hash caching operations."""
    
    @pytest.mark.asyncio
    async def test_cache_file_result_success(self, redis_manager, mock_redis):
        """Test successful file result caching."""
        file_hash = "abc123def456"
        result_data = {"processed": True, "cost": 1.50}
        
        result = await redis_manager.cache_file_result(file_hash, result_data)
        
        assert result is True
        
        # Verify setex was called
        mock_redis.setex.assert_called_once()
        setex_call = mock_redis.setex.call_args
        
        expected_key = f"file_hash:{file_hash}"
        assert setex_call[0][0] == expected_key
        assert setex_call[0][1] == redis_manager.FILE_HASH_TTL
        assert json.loads(setex_call[0][2]) == result_data
    
    @pytest.mark.asyncio
    async def test_cache_file_result_error(self, redis_manager, mock_redis):
        """Test file result caching with Redis error."""
        mock_redis.setex.side_effect = Exception("Redis error")
        
        result = await redis_manager.cache_file_result("hash", {"data": "value"})
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_cached_file_result_success(self, redis_manager, mock_redis):
        """Test successful cached file result retrieval."""
        file_hash = "abc123def456"
        cached_data = {"processed": True, "cost": 1.50}
        
        mock_redis.get.return_value = json.dumps(cached_data).encode()
        
        result = await redis_manager.get_cached_file_result(file_hash)
        
        assert result == cached_data
        
        expected_key = f"file_hash:{file_hash}"
        mock_redis.get.assert_called_once_with(expected_key)
    
    @pytest.mark.asyncio
    async def test_get_cached_file_result_not_found(self, redis_manager, mock_redis):
        """Test cached file result retrieval when not found."""
        mock_redis.get.return_value = None
        
        result = await redis_manager.get_cached_file_result("nonexistent_hash")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_cached_file_result_error(self, redis_manager, mock_redis):
        """Test cached file result retrieval with Redis error."""
        mock_redis.get.side_effect = Exception("Redis error")
        
        result = await redis_manager.get_cached_file_result("hash")
        
        assert result is None


class TestCoordinationLocks:
    """Test coordination lock operations."""
    
    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, redis_manager, mock_redis):
        """Test successful lock acquisition."""
        lock_key = "test_lock"
        lock_value = "unique_value_123"
        
        mock_redis.set.return_value = True
        
        result = await redis_manager.acquire_lock(lock_key, lock_value)
        
        assert result is True
        
        # Verify set was called with correct parameters
        mock_redis.set.assert_called_once_with(
            f"lock:{lock_key}",
            lock_value,
            nx=True,
            ex=redis_manager.LOCK_TTL
        )
    
    @pytest.mark.asyncio
    async def test_acquire_lock_already_held(self, redis_manager, mock_redis):
        """Test lock acquisition when already held."""
        mock_redis.set.return_value = None  # Lock already exists
        
        result = await redis_manager.acquire_lock("test_lock", "value")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_acquire_lock_custom_ttl(self, redis_manager, mock_redis):
        """Test lock acquisition with custom TTL."""
        lock_key = "test_lock"
        lock_value = "unique_value"
        custom_ttl = 120  # 2 minutes
        
        mock_redis.set.return_value = True
        
        result = await redis_manager.acquire_lock(lock_key, lock_value, custom_ttl)
        
        assert result is True
        
        # Verify custom TTL was used
        mock_redis.set.assert_called_once_with(
            f"lock:{lock_key}",
            lock_value,
            nx=True,
            ex=custom_ttl
        )
    
    @pytest.mark.asyncio
    async def test_acquire_lock_error(self, redis_manager, mock_redis):
        """Test lock acquisition with Redis error."""
        mock_redis.set.side_effect = Exception("Redis error")
        
        result = await redis_manager.acquire_lock("test_lock", "value")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_release_lock_success(self, redis_manager, mock_redis):
        """Test successful lock release."""
        lock_key = "test_lock"
        lock_value = "unique_value_123"
        
        mock_redis.eval.return_value = 1  # Successfully deleted
        
        result = await redis_manager.release_lock(lock_key, lock_value)
        
        assert result is True
        
        # Verify eval was called with correct Lua script
        mock_redis.eval.assert_called_once()
        eval_call = mock_redis.eval.call_args
        assert "redis.call" in eval_call[0][0]  # Lua script
        assert eval_call[0][1] == 1  # Number of keys
        assert eval_call[0][2] == f"lock:{lock_key}"  # Key
        assert eval_call[0][3] == lock_value  # Value
    
    @pytest.mark.asyncio
    async def test_release_lock_not_owned(self, redis_manager, mock_redis):
        """Test lock release when not owned."""
        mock_redis.eval.return_value = 0  # Not deleted (not owned)
        
        result = await redis_manager.release_lock("test_lock", "wrong_value")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_release_lock_error(self, redis_manager, mock_redis):
        """Test lock release with Redis error."""
        mock_redis.eval.side_effect = Exception("Redis error")
        
        result = await redis_manager.release_lock("test_lock", "value")
        
        assert result is False


class TestUtilityMethods:
    """Test utility methods."""
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_documents(self, redis_manager, mock_redis):
        """Test cleanup of expired document keys."""
        # Mock keys with mixed TTL states
        mock_redis.keys.return_value = [
            b"doc:key1",
            b"doc:key2",
            b"doc:key3"
        ]
        
        # Mock TTL responses: -1 (no TTL), 3600 (has TTL), -2 (expired)
        mock_redis.ttl.side_effect = [-1, 3600, -2]
        
        result = await redis_manager.cleanup_expired_documents()
        
        assert result == 1  # One expired key
        
        # Verify keys were checked
        mock_redis.keys.assert_called_once_with("doc:*")
        assert mock_redis.ttl.call_count == 3
        
        # Verify expire was called for key without TTL
        mock_redis.expire.assert_called_once_with(
            "doc:key1", 
            redis_manager.DOC_STATE_TTL
        )
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_documents_error(self, redis_manager, mock_redis):
        """Test cleanup with Redis error."""
        mock_redis.keys.side_effect = Exception("Redis error")
        
        result = await redis_manager.cleanup_expired_documents()
        
        assert result == 0
    
    @pytest.mark.asyncio
    async def test_get_stats_success(self, redis_manager, mock_redis):
        """Test successful statistics retrieval."""
        # Mock keys counts
        mock_redis.keys.side_effect = [
            [b"doc:1", b"doc:2"],        # 2 document keys
            [b"lock:1"],                 # 1 lock key
            [b"file_hash:1", b"file_hash:2", b"file_hash:3"]  # 3 file hash keys
        ]
        
        # Mock memory info
        mock_redis.info.return_value = {
            "used_memory": 2048,
            "used_memory_human": "2.0K"
        }
        
        stats = await redis_manager.get_stats()
        
        assert stats["document_keys"] == 2
        assert stats["lock_keys"] == 1
        assert stats["file_hash_keys"] == 3
        assert stats["memory_used_bytes"] == 2048
        assert stats["memory_used_human"] == "2.0K"
        assert "timestamp" in stats
        
        # Verify all keys calls were made
        assert mock_redis.keys.call_count == 3
        mock_redis.info.assert_called_once_with("memory")
    
    @pytest.mark.asyncio
    async def test_get_stats_error(self, redis_manager, mock_redis):
        """Test statistics retrieval with Redis error."""
        mock_redis.keys.side_effect = Exception("Redis error")
        
        stats = await redis_manager.get_stats()
        
        assert "error" in stats
        assert "Redis error" in stats["error"]


class TestGlobalStateManager:
    """Test global state manager instance."""
    
    def test_global_state_manager_instance(self):
        """Test that global state_manager instance is created."""
        assert state_manager is not None
        assert isinstance(state_manager, RedisStateManager)
        assert state_manager.redis_url == "redis://localhost:6379/1"
    
    @pytest.mark.asyncio
    async def test_global_state_manager_usage(self, mock_redis):
        """Test using global state manager instance."""
        # Temporarily replace the global instance's client
        original_client = state_manager.redis_client
        state_manager.redis_client = mock_redis
        
        try:
            # Test a simple operation
            result = await state_manager.store_document_state(
                "global_test_doc",
                "test OCR",
                [{"item": "test"}],
                {"file": "test.pdf"}
            )
            
            assert result is True
            mock_redis.pipeline.assert_called_once()
            
        finally:
            # Restore original client
            state_manager.redis_client = original_client


# Integration test
class TestRedisStateManagerIntegration:
    """Integration tests for RedisStateManager."""
    
    @pytest.mark.asyncio
    async def test_full_document_lifecycle(self, mock_redis):
        """Test complete document state lifecycle."""
        manager = RedisStateManager()
        manager.redis_client = mock_redis
        
        doc_id = "integration_test_doc"
        
        # Store document state
        store_result = await manager.store_document_state(
            doc_id,
            "Integration test OCR",
            [{"item": "Integration Test", "amount": 100.0}],
            {"file_name": "integration.pdf"}
        )
        assert store_result is True
        
        # Mock retrieval data
        hash_data = {
            b"doc_id": doc_id.encode(),
            b"ocr_text": b"Integration test OCR",
            b"line_items": b'[{"item": "Integration Test", "amount": 100.0}]',
            b"document_metadata": b'{"file_name": "integration.pdf"}',
            b"created_at": b"2024-01-01T12:00:00Z",
            b"updated_at": b"2024-01-01T12:00:00Z"
        }
        mock_redis.hgetall.return_value = hash_data
        
        # Retrieve document state
        doc_state = await manager.get_document_state(doc_id)
        assert doc_state is not None
        assert doc_state.doc_id == doc_id
        assert doc_state.ocr_text == "Integration test OCR"
        
        # Update document fields
        update_result = await manager.update_document_field(
            doc_id,
            {"ocr_text": "Updated OCR text"}
        )
        assert update_result is True
        
        # Cache agent result
        agent_result = {
            "success": True,
            "data": {"analysis": "complete"},
            "execution_time_ms": 2000,
            "cost_rupees": 3.00,
            "confidence": 0.95
        }
        cache_result = await manager.cache_agent_result(doc_id, "test_agent", agent_result)
        assert cache_result is True 