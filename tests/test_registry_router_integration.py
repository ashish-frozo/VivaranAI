"""
Registry & Router Integration Tests.

Tests the complete Agent Registry and Router system including:
- Agent registration and discovery
- Routing strategies and decision making
- Multi-agent workflow orchestration
- Load balancing and fallback handling
- End-to-end coordination scenarios
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from agents.base_agent import BaseAgent, AgentContext, ModelHint
from agents.agent_registry import (
    AgentRegistry, 
    AgentStatus, 
    TaskCapability, 
    AgentCapabilities, 
    AgentRegistration
)
from agents.router_agent import (
    RouterAgent,
    RoutingStrategy,
    WorkflowType,
    RoutingRequest,
    RoutingDecision,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowResult
)
from agents.redis_state import RedisStateManager


class TestDocumentProcessingAgent(BaseAgent):
    """Mock document processing agent for testing."""
    
    def __init__(self):
        super().__init__(
            agent_id="document_processor",
            name="Document Processing Agent",
            instructions="Processes medical bills and extracts structured data",
            tools=[]
        )
    
    async def process_task(self, context, task_data):
        """Mock document processing."""
        return {
            "extracted_text": "Mock extracted medical bill text",
            "line_items": [
                {"item": "Consultation", "amount": 500},
                {"item": "Lab Test", "amount": 200}
            ],
            "total_amount": 700
        }


class TestRateValidationAgent(BaseAgent):
    """Mock rate validation agent for testing."""
    
    def __init__(self):
        super().__init__(
            agent_id="rate_validator",
            name="Rate Validation Agent", 
            instructions="Validates medical charges against CGHS/ESI rates",
            tools=[]
        )
    
    async def process_task(self, context, task_data):
        """Mock rate validation."""
        return {
            "validation_results": [
                {"item": "Consultation", "billed": 500, "max_allowed": 300, "overcharge": 200},
                {"item": "Lab Test", "billed": 200, "max_allowed": 150, "overcharge": 50}
            ],
            "total_overcharge": 250,
            "confidence": 0.92
        }


@pytest.fixture
async def mock_redis_client():
    """Mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.ping.return_value = "PONG"
    mock_client.keys.return_value = []
    mock_client.get.return_value = None
    mock_client.setex.return_value = True
    mock_client.delete.return_value = 1
    return mock_client


@pytest.fixture
async def agent_registry(mock_redis_client):
    """Create test agent registry."""
    with patch('redis.asyncio.from_url', return_value=mock_redis_client):
        registry = AgentRegistry()
        await registry.start()
        yield registry
        await registry.stop()


@pytest.fixture
async def router_agent(agent_registry):
    """Create test router agent."""
    router = RouterAgent(registry=agent_registry)
    await router.start()
    yield router
    await router.stop()


@pytest.fixture
def test_agents():
    """Create test agents with capabilities."""
    doc_agent = TestDocumentProcessingAgent()
    rate_agent = TestRateValidationAgent()
    
    return {
        "document_processor": doc_agent,
        "rate_validator": rate_agent
    }


@pytest.fixture
def agent_capabilities():
    """Define capabilities for test agents."""
    return {
        "document_processor": AgentCapabilities(
            supported_tasks=[TaskCapability.DOCUMENT_PROCESSING, TaskCapability.OCR_EXTRACTION],
            max_concurrent_requests=5,
            preferred_model_hints=[ModelHint.STANDARD, ModelHint.PREMIUM],
            processing_time_ms_avg=2000,
            cost_per_request_rupees=2.5,
            confidence_threshold=0.85,
            supported_document_types=["pdf", "image"],
            supported_languages=["en", "hi"]
        ),
        "rate_validator": AgentCapabilities(
            supported_tasks=[TaskCapability.RATE_VALIDATION, TaskCapability.DATA_VALIDATION],
            max_concurrent_requests=10,
            preferred_model_hints=[ModelHint.STANDARD],
            processing_time_ms_avg=1500,
            cost_per_request_rupees=1.8,
            confidence_threshold=0.90,
            supported_document_types=["json"],
            supported_languages=["en"]
        )
    }


@pytest.mark.asyncio
async def test_agent_registration_and_discovery(agent_registry, test_agents, agent_capabilities):
    """
    Test agent registration and capability-based discovery.
    """
    # Register test agents
    for agent_id, agent in test_agents.items():
        await agent.start()
        success = await agent_registry.register_agent(
            agent=agent,
            capabilities=agent_capabilities[agent_id]
        )
        assert success, f"Failed to register agent: {agent_id}"
    
    # Test discovery by capabilities
    discovered_agents = await agent_registry.discover_agents(
        required_capabilities=[TaskCapability.DOCUMENT_PROCESSING],
        model_hint=ModelHint.STANDARD,
        max_agents=2
    )
    
    assert len(discovered_agents) == 1
    assert discovered_agents[0].agent_id == "document_processor"
    assert discovered_agents[0].status == AgentStatus.ONLINE
    
    # Test multi-capability discovery
    multi_cap_agents = await agent_registry.discover_agents(
        required_capabilities=[TaskCapability.RATE_VALIDATION, TaskCapability.DATA_VALIDATION],
        model_hint=ModelHint.STANDARD,
        max_agents=5
    )
    
    assert len(multi_cap_agents) == 1
    assert multi_cap_agents[0].agent_id == "rate_validator"
    
    # Cleanup
    for agent in test_agents.values():
        await agent.stop()


@pytest.mark.asyncio
async def test_routing_strategies(router_agent, agent_registry, test_agents, agent_capabilities):
    """
    Test different routing strategies for agent selection.
    """
    # Register agents
    for agent_id, agent in test_agents.items():
        await agent.start()
        await agent_registry.register_agent(
            agent=agent,
            capabilities=agent_capabilities[agent_id]
        )
    
    # Test cost-optimized routing
    cost_request = RoutingRequest(
        doc_id="test_doc_1",
        user_id="test_user",
        task_type="validation",
        required_capabilities=[TaskCapability.RATE_VALIDATION],
        model_hint=ModelHint.STANDARD,
        routing_strategy=RoutingStrategy.COST_OPTIMIZED,
        max_agents=1,
        timeout_seconds=30,
        priority=5,
        metadata={}
    )
    
    cost_decision = await router_agent.route_request(cost_request)
    assert cost_decision.success
    assert len(cost_decision.selected_agents) == 1
    assert cost_decision.selected_agents[0].agent_id == "rate_validator"
    assert cost_decision.routing_strategy == RoutingStrategy.COST_OPTIMIZED
    
    # Cleanup
    for agent in test_agents.values():
        await agent.stop()


@pytest.mark.asyncio
async def test_sequential_workflow_execution(router_agent, agent_registry, test_agents, agent_capabilities):
    """
    Test sequential multi-agent workflow execution.
    """
    # Register agents
    for agent_id, agent in test_agents.items():
        await agent.start()
        await agent_registry.register_agent(
            agent=agent,
            capabilities=agent_capabilities[agent_id]
        )
    
    # Define sequential workflow: Document Processing → Rate Validation
    workflow = WorkflowDefinition(
        workflow_id="medical_bill_analysis",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="step_1_extract",
                agent_id="document_processor",
                task_input="Extract data from medical bill PDF",
                dependencies=[],
                timeout_seconds=30,
                metadata={"priority": "high"}
            ),
            WorkflowStep(
                step_id="step_2_validate",
                agent_id="rate_validator",
                task_input="Validate extracted charges against CGHS rates",
                dependencies=["step_1_extract"],
                timeout_seconds=30,
                metadata={"reference": "cghs_2023"}
            )
        ],
        max_parallel_steps=1,
        total_timeout_seconds=120,
        failure_strategy="fail_fast",
        metadata={"workflow_type": "medical_analysis"}
    )
    
    # Execute workflow
    context = AgentContext(
        doc_id="medical_bill_123",
        user_id="patient_456",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "workflow_execution"}
    )
    
    # Mock the agent execution to return results
    with patch.object(router_agent, '_simulate_agent_execution') as mock_exec:
        mock_exec.side_effect = [
            # Step 1: Document processing result
            type('AgentResult', (), {
                'success': True,
                'data': {
                    'extracted_text': 'Medical bill content',
                    'line_items': [{'item': 'Consultation', 'amount': 500}]
                },
                'agent_id': 'document_processor',
                'task_type': 'document_processing',
                'execution_time_ms': 2000,
                'model_used': 'gpt-4o',
                'cost_rupees': 2.5,
                'confidence': 0.9
            }),
            # Step 2: Rate validation result
            type('AgentResult', (), {
                'success': True,
                'data': {
                    'validation_results': [{'item': 'Consultation', 'overcharge': 200}],
                    'total_overcharge': 200
                },
                'agent_id': 'rate_validator',
                'task_type': 'rate_validation',
                'execution_time_ms': 1500,
                'model_used': 'gpt-4o',
                'cost_rupees': 1.8,
                'confidence': 0.95
            })
        ]
        
        result = await router_agent.execute_workflow(workflow, context)
    
    # Verify workflow execution
    assert result.success
    assert result.workflow_id == "medical_bill_analysis"
    assert len(result.step_results) == 2
    
    # Verify step execution order and results
    step_1_result = result.step_results["step_1_extract"]
    assert step_1_result.success
    assert step_1_result.agent_id == "document_processor"
    assert "extracted_text" in step_1_result.data
    
    step_2_result = result.step_results["step_2_validate"]
    assert step_2_result.success
    assert step_2_result.agent_id == "rate_validator"
    assert "validation_results" in step_2_result.data
    
    # Verify cost and time calculations
    assert result.total_cost_rupees == 4.3  # 2.5 + 1.8
    assert result.total_execution_time_ms > 0
    
    # Cleanup
    for agent in test_agents.values():
        await agent.stop()
