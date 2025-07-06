"""
Unit tests for BaseAgent foundation class.

Tests OpenAI SDK integration, OTEL tracing, Redis state management,
cost tracking, and CPU timeout enforcement.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import pytest
import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agents.base_agent import (
    BaseAgent,
    AgentContext,
    AgentResult, 
    ModelHint,
    CPUTimeoutError,
    OPENAI_COSTS
)


class MockAgent(BaseAgent):
    """Mock agent implementation for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(
            agent_id="test_agent",
            name="Test Agent",
            instructions="Test agent for unit tests",
            **kwargs
        )
    
    async def process_task(self, context: AgentContext, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock task processing."""
        return {
            "result": "mock_result",
            "processed_data": task_data.get("input", "default"),
            "confidence": 0.95
        }


@pytest.fixture
def memory_span_exporter():
    """In-memory span exporter for testing OTEL traces."""
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)
    return exporter


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.close.return_value = None
    return mock_client


@pytest.fixture
def agent_context():
    """Sample agent context for testing."""
    return AgentContext(
        doc_id="doc123",
        user_id="user456",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "analysis", "priority": "high"}
    )


@pytest.fixture
def mock_openai_agent():
    """Mock OpenAI Agent."""
    agent = MagicMock()
    agent.run.return_value = MagicMock(
        final_output="Mock agent response",
        usage=MagicMock(
            prompt_tokens=100,
            completion_tokens=50
        )
    )
    return agent


class TestBaseAgentInitialization:
    """Test BaseAgent initialization and basic properties."""
    
    def test_agent_initialization_basic(self):
        """Test basic agent initialization."""
        agent = MockAgent()
        
        assert agent.agent_id == "test_agent"
        assert agent.name == "Test Agent"
        assert agent.instructions == "Test agent for unit tests"
        assert agent.default_model == "gpt-4o"
        assert agent.redis_url == "redis://localhost:6379/1"
        assert agent.tools == []
        assert agent.redis_client is None
    
    def test_agent_initialization_with_tools(self):
        """Test agent initialization with tools."""
        def mock_tool():
            pass
        
        agent = MockAgent(tools=[mock_tool])
        assert len(agent.tools) == 1
        assert agent.tools[0] == mock_tool
    
    def test_agent_initialization_custom_config(self):
        """Test agent initialization with custom configuration."""
        agent = MockAgent(
            redis_url="redis://custom:6379/2",
            default_model="gpt-3.5-turbo"
        )
        
        assert agent.redis_url == "redis://custom:6379/2"
        assert agent.default_model == "gpt-3.5-turbo"


class TestModelSelection:
    """Test model selection logic."""
    
    def test_select_model_cheap(self):
        """Test cheap model selection."""
        agent = MockAgent()
        model = agent.select_model(ModelHint.CHEAP)
        assert model == "gpt-3.5-turbo"
    
    def test_select_model_standard(self):
        """Test standard model selection."""
        agent = MockAgent()
        model = agent.select_model(ModelHint.STANDARD)
        assert model == "gpt-4o"
    
    def test_select_model_premium(self):
        """Test premium model selection."""
        agent = MockAgent()
        model = agent.select_model(ModelHint.PREMIUM)
        assert model == "gpt-4"
    
    def test_select_model_default(self):
        """Test default model selection for unknown hint."""
        agent = MockAgent(default_model="custom-model")
        model = agent.select_model("unknown_hint")
        assert model == "custom-model"


class TestAgentLifecycle:
    """Test agent startup and shutdown lifecycle."""
    
    @pytest.mark.asyncio
    async def test_start_success(self, mock_redis):
        """Test successful agent startup."""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            agent = MockAgent()
            await agent.start()
            
            assert agent.redis_client == mock_redis
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_redis_failure(self):
        """Test agent startup with Redis connection failure."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Connection failed")
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            agent = MockAgent()
            
            with pytest.raises(Exception, match="Connection failed"):
                await agent.start()
    
    @pytest.mark.asyncio
    async def test_stop(self, mock_redis):
        """Test agent shutdown."""
        agent = MockAgent()
        agent.redis_client = mock_redis
        
        await agent.stop()
        mock_redis.close.assert_called_once()


class TestCostCalculation:
    """Test OpenAI API cost calculation."""
    
    def test_calculate_cost_gpt4o(self):
        """Test cost calculation for GPT-4o."""
        agent = MockAgent()
        
        mock_result = MagicMock()
        mock_result.usage = MagicMock(
            prompt_tokens=1000,  # 1K tokens
            completion_tokens=500  # 0.5K tokens
        )
        
        cost = agent._calculate_cost(mock_result, "gpt-4o")
        
        # Expected: (1 * 0.42) + (0.5 * 1.26) = 0.42 + 0.63 = 1.05
        expected_cost = round((1000/1000) * 0.42 + (500/1000) * 1.26, 4)
        assert cost == expected_cost
    
    def test_calculate_cost_gpt35(self):
        """Test cost calculation for GPT-3.5."""
        agent = MockAgent()
        
        mock_result = MagicMock()
        mock_result.usage = MagicMock(
            prompt_tokens=2000,  # 2K tokens
            completion_tokens=1000  # 1K tokens
        )
        
        cost = agent._calculate_cost(mock_result, "gpt-3.5-turbo")
        
        # Expected: (2 * 0.08) + (1 * 0.17) = 0.16 + 0.17 = 0.33
        expected_cost = round((2000/1000) * 0.08 + (1000/1000) * 0.17, 4)
        assert cost == expected_cost
    
    def test_calculate_cost_no_usage(self):
        """Test cost calculation when usage is not available."""
        agent = MockAgent()
        
        mock_result = MagicMock()
        mock_result.usage = None
        
        cost = agent._calculate_cost(mock_result, "gpt-4o")
        assert cost == 0.0
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation for unknown model."""
        agent = MockAgent()
        
        mock_result = MagicMock()
        mock_result.usage = MagicMock(
            prompt_tokens=1000,
            completion_tokens=500
        )
        
        cost = agent._calculate_cost(mock_result, "unknown-model")
        assert cost == 0.0  # No cost for unknown model


class TestAgentExecution:
    """Test agent execution with full observability."""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent_context, memory_span_exporter, mock_redis):
        """Test successful agent execution."""
        agent = MockAgent()
        agent.redis_client = mock_redis
        
        # Mock OpenAI agent creation and execution
        mock_openai_agent = MagicMock()
        mock_result = MagicMock()
        mock_result.final_output = "Test response"
        mock_result.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        
        with patch.object(agent, '_create_openai_agent', return_value=mock_openai_agent):
            with patch.object(agent, '_execute_with_timeout', return_value=mock_result):
                result = await agent.execute(agent_context, "Test task input")
        
        # Verify result structure
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.agent_id == "test_agent"
        assert result.model_used == "gpt-4o"  # Standard hint -> GPT-4o
        assert result.cost_rupees > 0
        assert result.execution_time_ms > 0
        assert result.data["output"] == "Test response"
        
        # Verify spans were created
        spans = memory_span_exporter.get_finished_spans()
        assert len(spans) > 0
        
        execution_span = next(
            (span for span in spans if span.name == "test_agent.execute"), 
            None
        )
        assert execution_span is not None
        assert execution_span.attributes["agent.id"] == "test_agent"
        assert execution_span.attributes["doc.id"] == agent_context.doc_id
    
    @pytest.mark.asyncio
    async def test_execute_cpu_timeout(self, agent_context, mock_redis):
        """Test agent execution with CPU timeout."""
        agent = MockAgent()
        agent.redis_client = mock_redis
        
        # Mock CPU timeout
        with patch.object(agent, '_execute_with_timeout', side_effect=CPUTimeoutError("CPU timeout")):
            result = await agent.execute(agent_context, "Test task input")
        
        assert result.success is False
        assert result.error == "Agent execution timeout: CPU timeout"
        assert result.execution_time_ms == 150  # Timeout at 150ms
        assert result.cost_rupees == 0.0
    
    @pytest.mark.asyncio
    async def test_execute_general_error(self, agent_context, mock_redis):
        """Test agent execution with general error."""
        agent = MockAgent()
        agent.redis_client = mock_redis
        
        # Mock general error
        with patch.object(agent, '_execute_with_timeout', side_effect=ValueError("Test error")):
            result = await agent.execute(agent_context, "Test task input")
        
        assert result.success is False
        assert "Agent execution error: Test error" in result.error
        assert result.cost_rupees == 0.0


class TestCPUTimeoutEnforcement:
    """Test CPU time slice enforcement."""
    
    @pytest.mark.asyncio
    async def test_cpu_timeout_detection(self):
        """Test CPU timeout detection mechanism."""
        agent = MockAgent()
        
        # Mock time.process_time to simulate CPU usage
        start_time = 0.0
        
        def mock_process_time():
            nonlocal start_time
            start_time += 0.2  # Simulate 200ms CPU time each call
            return start_time
        
        mock_openai_agent = MagicMock()
        
        with patch('time.process_time', side_effect=mock_process_time):
            with patch('asyncio.to_thread') as mock_to_thread:
                mock_to_thread.return_value = asyncio.sleep(0.1)  # Short task
                
                with pytest.raises(CPUTimeoutError, match="CPU time slice exceeded"):
                    await agent._execute_with_timeout(
                        mock_openai_agent,
                        "test input",
                        0.0
                    )
    
    @pytest.mark.asyncio
    async def test_cpu_timeout_within_limit(self):
        """Test execution within CPU time limit."""
        agent = MockAgent()
        
        # Mock time.process_time to simulate normal CPU usage
        def mock_process_time():
            return 0.1  # 100ms CPU time (within 150ms limit)
        
        mock_openai_agent = MagicMock()
        mock_result = MagicMock()
        
        async def mock_task_result():
            await asyncio.sleep(0.01)  # Short async task
            return mock_result
        
        with patch('time.process_time', side_effect=mock_process_time):
            with patch('asyncio.to_thread', return_value=mock_task_result()):
                result = await agent._execute_with_timeout(
                    mock_openai_agent,
                    "test input",
                    0.0
                )
                
                assert result == mock_result


class TestHealthCheck:
    """Test agent health check functionality."""
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, mock_redis):
        """Test health check when agent is healthy."""
        agent = MockAgent()
        agent.redis_client = mock_redis
        
        health = await agent.health_check()
        
        assert health["agent_id"] == "test_agent"
        assert health["name"] == "Test Agent"
        assert health["status"] == "healthy"
        assert health["redis_connected"] is True
        assert health["tools_count"] == 0
        assert "timestamp" in health
        
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_redis_disconnected(self):
        """Test health check when Redis is disconnected."""
        agent = MockAgent()
        agent.redis_client = None
        
        health = await agent.health_check()
        
        assert health["status"] == "degraded"
        assert health["redis_connected"] is False
    
    @pytest.mark.asyncio
    async def test_health_check_redis_error(self, mock_redis):
        """Test health check when Redis ping fails."""
        agent = MockAgent()
        agent.redis_client = mock_redis
        mock_redis.ping.side_effect = Exception("Redis error")
        
        health = await agent.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "Redis error" in health["error"]


class TestOpenAIAgentCreation:
    """Test OpenAI Agent creation and tool integration."""
    
    def test_create_openai_agent_no_tools(self):
        """Test OpenAI agent creation without tools."""
        agent = MockAgent()
        
        with patch('agents.base_agent.OpenAIAgent') as mock_openai_agent_class:
            openai_agent = agent._create_openai_agent("gpt-4o")
            
            mock_openai_agent_class.assert_called_once_with(
                name="Test Agent",
                instructions="Test agent for unit tests",
                tools=[],
                model="gpt-4o"
            )
    
    def test_create_openai_agent_with_tools(self):
        """Test OpenAI agent creation with tools."""
        def mock_tool():
            pass
        
        mock_tool._tool_schema = {"type": "function"}
        
        agent = MockAgent(tools=[mock_tool])
        
        with patch('agents.base_agent.OpenAIAgent') as mock_openai_agent_class:
            with patch('agents.base_agent.function_tool') as mock_function_tool:
                mock_function_tool.return_value = {"wrapped": "tool"}
                
                openai_agent = agent._create_openai_agent("gpt-4o")
                
                mock_function_tool.assert_called_once_with(mock_tool)
                mock_openai_agent_class.assert_called_once()


class TestResultParsing:
    """Test agent result parsing."""
    
    @pytest.mark.asyncio
    async def test_parse_agent_result_with_final_output(self):
        """Test parsing result with final_output attribute."""
        agent = MockAgent()
        
        mock_result = MagicMock()
        mock_result.final_output = "Test final output"
        
        parsed = await agent._parse_agent_result(mock_result)
        
        assert parsed == {"output": "Test final output"}
    
    @pytest.mark.asyncio
    async def test_parse_agent_result_without_final_output(self):
        """Test parsing result without final_output attribute."""
        agent = MockAgent()
        
        mock_result = "Direct string result"
        
        parsed = await agent._parse_agent_result(mock_result)
        
        assert parsed == {"output": "Direct string result"}


# Integration test
class TestBaseAgentIntegration:
    """Integration tests for BaseAgent."""
    
    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, memory_span_exporter):
        """Test complete agent lifecycle from start to execution."""
        # Use real Redis mock that supports all operations
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.close.return_value = None
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            agent = MockAgent()
            
            # Start agent
            await agent.start()
            assert agent.redis_client is not None
            
            # Create context
            context = AgentContext(
                doc_id="integration_test_doc",
                user_id="integration_test_user",
                correlation_id="integration_test_corr",
                model_hint=ModelHint.CHEAP,
                start_time=time.time(),
                metadata={"task_type": "integration_test"}
            )
            
            # Mock successful execution
            mock_result = MagicMock()
            mock_result.final_output = "Integration test result"
            mock_result.usage = MagicMock(prompt_tokens=50, completion_tokens=25)
            
            with patch.object(agent, '_execute_with_timeout', return_value=mock_result):
                result = await agent.execute(context, "Integration test input")
            
            # Verify execution result
            assert result.success is True
            assert result.agent_id == "test_agent"
            assert result.model_used == "gpt-3.5-turbo"  # Cheap hint
            assert result.data["output"] == "Integration test result"
            
            # Stop agent
            await agent.stop()
            mock_redis.close.assert_called_once()
            
            # Verify tracing
            spans = memory_span_exporter.get_finished_spans()
            execution_spans = [s for s in spans if s.name == "test_agent.execute"]
            assert len(execution_spans) >= 1 