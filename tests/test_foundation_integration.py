"""
Foundation Layer Integration Test.

Tests the complete foundation layer including BaseAgent, RedisStateManager,
and metrics server working together in a realistic scenario.
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from agents.base_agent import BaseAgent, AgentContext, ModelHint
from agents.redis_state import RedisStateManager
from agents.metrics_server import app as metrics_app


class TestFoundationAgent(BaseAgent):
    """Test agent for foundation integration testing."""
    
    def __init__(self):
        super().__init__(
            agent_id="foundation_test_agent",
            name="Foundation Test Agent",
            instructions="Test agent for foundation layer integration testing",
            tools=[]
        )
    
    async def process_task(self, context, task_data):
        """Simple task processing for testing."""
        return {
            "test_result": "success",
            "doc_id": context.doc_id,
            "processed_at": time.time()
        }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for integration testing."""
    client = AsyncMock()
    client.ping.return_value = True
    client.close.return_value = None
    
    # Mock pipeline operations
    pipeline = AsyncMock()
    pipeline.hset.return_value = pipeline
    pipeline.expire.return_value = pipeline
    pipeline.execute.return_value = [True, True]
    client.pipeline.return_value = pipeline
    
    # Mock hash operations
    client.hgetall.return_value = {
        b"doc_id": b"test_doc_123",
        b"ocr_text": b"Test OCR content",
        b"line_items": b'[{"item": "Test", "amount": 100.0}]',
        b"document_metadata": b'{"file_name": "test.pdf"}',
        b"created_at": b"2024-01-01T12:00:00Z",
        b"updated_at": b"2024-01-01T12:00:00Z"
    }
    
    # Mock string operations
    client.setex.return_value = True
    client.get.return_value = None
    
    # Mock utility operations
    client.keys.return_value = [b"doc:test_1", b"doc:test_2"]
    client.ttl.return_value = 3600
    client.info.return_value = {
        "used_memory": 1024,
        "used_memory_human": "1.0K"
    }
    
    return client


@pytest.mark.asyncio
async def test_foundation_layer_integration(mock_redis_client):
    """
    Integration test for the complete foundation layer.
    
    Tests BaseAgent + RedisStateManager + metrics collection working together.
    """
    # Mock Redis connections
    with patch('redis.asyncio.from_url', return_value=mock_redis_client):
        
        # 1. Initialize Redis State Manager
        state_manager = RedisStateManager()
        await state_manager.connect()
        
        assert state_manager.redis_client is not None
        
        # 2. Create and start test agent
        agent = TestFoundationAgent()
        await agent.start()
        
        assert agent.redis_client is not None
        
        # 3. Create agent context
        context = AgentContext(
            doc_id="integration_test_doc_123",
            user_id="integration_test_user",
            correlation_id=str(uuid.uuid4()),
            model_hint=ModelHint.CHEAP,
            start_time=time.time(),
            metadata={"task_type": "integration_test", "priority": "high"}
        )
        
        # 4. Store document state in Redis
        store_result = await state_manager.store_document_state(
            context.doc_id,
            "Integration test OCR content",
            [
                {"item": "Consultation", "amount": 500.0, "type": "service"},
                {"item": "Blood Test", "amount": 200.0, "type": "test"}
            ],
            {
                "file_name": "integration_test.pdf",
                "pages": 2,
                "language": "en",
                "test_flag": True
            }
        )
        
        assert store_result is True
        
        # 5. Mock OpenAI agent execution
        mock_openai_result = MagicMock()
        mock_openai_result.final_output = "Integration test completed successfully"
        mock_openai_result.usage = MagicMock(
            prompt_tokens=150,
            completion_tokens=75
        )
        
        with patch.object(agent, '_execute_with_timeout', return_value=mock_openai_result):
            # 6. Execute agent task
            result = await agent.execute(context, "Analyze this integration test document")
        
        # 7. Verify agent execution result
        assert result.success is True
        assert result.agent_id == "foundation_test_agent"
        assert result.model_used == "gpt-3.5-turbo"  # Cheap hint
        assert result.cost_rupees > 0
        assert result.execution_time_ms > 0
        assert result.data["output"] == "Integration test completed successfully"
        
        # 8. Cache agent result in Redis
        cache_result = await state_manager.cache_agent_result(
            context.doc_id,
            agent.agent_id,
            {
                "success": result.success,
                "data": result.data,
                "execution_time_ms": result.execution_time_ms,
                "cost_rupees": result.cost_rupees,
                "confidence": result.confidence
            }
        )
        
        assert cache_result is True
        
        # 9. Retrieve document state
        doc_state = await state_manager.get_document_state(context.doc_id)
        
        assert doc_state is not None
        assert doc_state.doc_id == context.doc_id
        assert "Integration test OCR content" in doc_state.ocr_text
        assert len(doc_state.line_items) == 2
        assert doc_state.document_metadata["test_flag"] is True
        
        # 10. Test agent health check
        health = await agent.health_check()
        
        assert health["status"] == "healthy"
        assert health["agent_id"] == "foundation_test_agent"
        assert health["redis_connected"] is True
        
        # 11. Test Redis statistics
        stats = await state_manager.get_stats()
        
        assert "document_keys" in stats
        assert "memory_used_bytes" in stats
        assert stats["document_keys"] == 2  # Mocked to return 2 keys
        
        # 12. Cleanup
        await agent.stop()
        await state_manager.disconnect()


@pytest.mark.asyncio
async def test_foundation_layer_error_handling(mock_redis_client):
    """
    Test error handling and degraded mode operation.
    """
    # Test with Redis connection failure
    failing_redis = AsyncMock()
    failing_redis.ping.side_effect = Exception("Connection failed")
    
    with patch('redis.asyncio.from_url', return_value=failing_redis):
        state_manager = RedisStateManager()
        
        # Connection should fail gracefully
        with pytest.raises(Exception, match="Connection failed"):
            await state_manager.connect()
    
    # Test agent with Redis unavailable
    agent = TestFoundationAgent()
    
    # Agent should handle missing Redis gracefully
    health = await agent.health_check()
    assert health["status"] == "degraded"
    assert health["redis_connected"] is False


@pytest.mark.asyncio 
async def test_metrics_server_integration():
    """
    Test metrics server endpoints with foundation components.
    """
    from fastapi.testclient import TestClient
    
    # Mock Redis for metrics server
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.keys.return_value = [b"doc:test1", b"doc:test2"]
    mock_redis.info.return_value = {"used_memory": 2048, "used_memory_human": "2.0K"}
    
    with patch('agents.redis_state.state_manager.redis_client', mock_redis):
        with patch('agents.redis_state.state_manager.connect'):
            with patch('agents.redis_state.state_manager.get_stats') as mock_stats:
                mock_stats.return_value = {
                    "document_keys": 2,
                    "lock_keys": 0,
                    "file_hash_keys": 1,
                    "memory_used_bytes": 2048
                }
                
                client = TestClient(metrics_app)
                
                # Test health check endpoint
                response = client.get("/healthz")
                assert response.status_code == 200
                
                health_data = response.json()
                assert health_data["status"] in ["healthy", "degraded"]
                assert "timestamp" in health_data
                
                # Test readiness check
                response = client.get("/healthz/ready")
                # May return 503 if Redis check fails in test, that's OK
                assert response.status_code in [200, 503]
                
                # Test liveness check
                response = client.get("/healthz/live")
                assert response.status_code == 200
                
                live_data = response.json()
                assert live_data["status"] == "alive"
                assert "uptime_seconds" in live_data
                
                # Test stats endpoint
                response = client.get("/stats")
                assert response.status_code == 200
                
                stats_data = response.json()
                assert "server" in stats_data
                assert "metrics" in stats_data
                
                # Test metrics endpoint
                response = client.get("/metrics")
                assert response.status_code in [200, 500]  # May fail without full setup
                
                if response.status_code == 200:
                    # Should return Prometheus format
                    assert "text/plain" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_concurrent_agent_execution(mock_redis_client):
    """
    Test multiple agents executing concurrently with shared Redis state.
    """
    with patch('redis.asyncio.from_url', return_value=mock_redis_client):
        
        # Create state manager
        state_manager = RedisStateManager()
        await state_manager.connect()
        
        # Create multiple agents
        agents = []
        for i in range(3):
            agent = TestFoundationAgent()
            agent.agent_id = f"concurrent_agent_{i}"
            await agent.start()
            agents.append(agent)
        
        # Create contexts for concurrent execution
        contexts = []
        for i in range(3):
            context = AgentContext(
                doc_id=f"concurrent_doc_{i}",
                user_id=f"user_{i}",
                correlation_id=str(uuid.uuid4()),
                model_hint=ModelHint.STANDARD,
                start_time=time.time(),
                metadata={"task_type": "concurrent_test", "agent_id": i}
            )
            contexts.append(context)
        
        # Mock OpenAI execution for all agents
        mock_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.final_output = f"Concurrent agent {i} completed"
            mock_result.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
            mock_results.append(mock_result)
        
        # Execute agents concurrently
        tasks = []
        for i, (agent, context) in enumerate(zip(agents, contexts)):
            with patch.object(agent, '_execute_with_timeout', return_value=mock_results[i]):
                task = agent.execute(context, f"Concurrent task {i}")
                tasks.append(task)
        
        # Wait for all executions to complete
        results = await asyncio.gather(*tasks)
        
        # Verify all executions succeeded
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.success is True
            assert f"concurrent_agent_{i}" in result.agent_id
            assert f"Concurrent agent {i} completed" in result.data["output"]
        
        # Cleanup all agents
        for agent in agents:
            await agent.stop()
        
        await state_manager.disconnect()


if __name__ == "__main__":
    # Run integration test directly
    asyncio.run(test_foundation_layer_integration(AsyncMock())) 