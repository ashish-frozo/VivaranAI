"""
RouterAgent Multi-Vertical Tests.

Tests the RouterAgent's multi-vertical, pack-driven architecture including:
- Multi-domain document routing
- Pack-driven agent selection
- Cold-poke functionality and health checks
- Capability-based routing across verticals
"""

import asyncio
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from agents.base_agent import BaseAgent, AgentContext, ModelHint
from agents.interfaces import TaskCapability
from agents.agent_registry import (
    AgentRegistry, 
    AgentStatus, 
    AgentCapabilities, 
    AgentRegistration
)
from agents.router_agent import (
    RouterAgent,
    RoutingStrategy,
    RoutingRequest,
    RoutingDecision,
    WorkflowType,
    WorkflowStep,
    WorkflowDefinition
)
from agents.medical_bill_agent import MedicalBillAgent
from agents.loan_risk_agent import LoanRiskAgent


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
async def medical_bill_agent():
    """Create a mock MedicalBillAgent for testing."""
    agent = AsyncMock(spec=MedicalBillAgent)
    agent.agent_id = "medical_bill_agent"
    agent.name = "Medical Bill Analysis Agent"
    agent.instructions = "Analyzes medical bills for overcharges"
    agent.process_task = AsyncMock(return_value={
        "document_type": "medical_bill",
        "total_amount": 5000,
        "line_items": [{"item": "Consultation", "amount": 500}],
        "overcharge_amount": 200,
        "confidence": 0.95
    })
    return agent


@pytest.fixture
async def loan_risk_agent():
    """Create a mock LoanRiskAgent for testing."""
    agent = AsyncMock(spec=LoanRiskAgent)
    agent.agent_id = "loan_risk_agent"
    agent.name = "Loan Risk Analysis Agent"
    agent.instructions = "Analyzes loan applications for risk assessment"
    agent.process_task = AsyncMock(return_value={
        "document_type": "loan_application",
        "applicant_name": "John Smith",
        "loan_amount": 25000,
        "risk_assessment": {
            "overall_risk_level": "low",
            "approval_recommendation": "approve"
        },
        "confidence": 0.92
    })
    return agent


@pytest.fixture
def agent_capabilities():
    """Define capabilities for test agents."""
    return {
        "medical_bill_agent": AgentCapabilities(
            supported_tasks=[
                TaskCapability.MEDICAL_ANALYSIS, 
                TaskCapability.RATE_VALIDATION,
                TaskCapability.OVERCHARGE_DETECTION
            ],
            max_concurrent_requests=5,
            preferred_model_hints=[ModelHint.STANDARD, ModelHint.PREMIUM],
            processing_time_ms_avg=3000,
            cost_per_request_rupees=3.5,
            confidence_threshold=0.85,
            supported_document_types=["medical_bill", "hospital_invoice", "pharmacy_receipt"],
            supported_languages=["en", "hi"]
        ),
        "loan_risk_agent": AgentCapabilities(
            supported_tasks=[
                TaskCapability.LOAN_ANALYSIS,
                TaskCapability.RISK_ASSESSMENT,
                TaskCapability.DUPLICATE_DETECTION
            ],
            max_concurrent_requests=3,
            preferred_model_hints=[ModelHint.STANDARD],
            processing_time_ms_avg=2500,
            cost_per_request_rupees=2.8,
            confidence_threshold=0.80,
            supported_document_types=["loan_application", "loan_agreement", "credit_report"],
            supported_languages=["en"]
        )
    }


@pytest.mark.asyncio
async def test_multi_vertical_agent_registration(agent_registry, medical_bill_agent, loan_risk_agent, agent_capabilities):
    """Test registration of agents from multiple verticals."""
    # Register medical bill agent
    await agent_registry.register_agent(
        agent=medical_bill_agent,
        capabilities=agent_capabilities["medical_bill_agent"]
    )
    
    # Register loan risk agent
    await agent_registry.register_agent(
        agent=loan_risk_agent,
        capabilities=agent_capabilities["loan_risk_agent"]
    )
    
    # Verify both agents are registered
    agents = await agent_registry.get_all_agents()
    assert len(agents) == 2
    
    # Check medical bill agent registration
    medical_agent = await agent_registry.get_agent_by_id("medical_bill_agent")
    assert medical_agent.agent_id == "medical_bill_agent"
    assert medical_agent.status == AgentStatus.READY
    
    # Check loan risk agent registration
    loan_agent = await agent_registry.get_agent_by_id("loan_risk_agent")
    assert loan_agent.agent_id == "loan_risk_agent"
    assert loan_agent.status == AgentStatus.READY
    
    # Check capabilities were registered correctly
    medical_capabilities = await agent_registry.get_agent_capabilities("medical_bill_agent")
    assert TaskCapability.MEDICAL_ANALYSIS in medical_capabilities.supported_tasks
    assert "medical_bill" in medical_capabilities.supported_document_types
    
    loan_capabilities = await agent_registry.get_agent_capabilities("loan_risk_agent")
    assert TaskCapability.LOAN_ANALYSIS in loan_capabilities.supported_tasks
    assert "loan_application" in loan_capabilities.supported_document_types


@pytest.mark.asyncio
async def test_medical_document_routing(router_agent, agent_registry, medical_bill_agent, loan_risk_agent, agent_capabilities):
    """Test routing of medical documents to the medical bill agent."""
    # Register both agents
    await agent_registry.register_agent(
        agent=medical_bill_agent,
        capabilities=agent_capabilities["medical_bill_agent"]
    )
    
    await agent_registry.register_agent(
        agent=loan_risk_agent,
        capabilities=agent_capabilities["loan_risk_agent"]
    )
    
    # Create a routing request for a medical document
    routing_request = RoutingRequest(
        doc_id="medical_doc_123",
        user_id="patient_456",
        task_type="medical_bill_analysis",
        required_capabilities=[TaskCapability.MEDICAL_ANALYSIS, TaskCapability.RATE_VALIDATION],
        model_hint=ModelHint.STANDARD,
        routing_strategy=RoutingStrategy.CAPABILITY_BASED,
        max_agents=1,
        timeout_seconds=30,
        priority=5,
        metadata={"document_type": "medical_bill"}
    )
    
    # Route the request
    routing_decision = await router_agent.route_request(routing_request)
    
    # Verify routing decision
    assert routing_decision.success
    assert len(routing_decision.selected_agents) == 1
    assert routing_decision.selected_agents[0].agent_id == "medical_bill_agent"
    assert routing_decision.confidence > 0.8
    assert routing_decision.routing_strategy == RoutingStrategy.CAPABILITY_BASED


@pytest.mark.asyncio
async def test_loan_document_routing(router_agent, agent_registry, medical_bill_agent, loan_risk_agent, agent_capabilities):
    """Test routing of loan documents to the loan risk agent."""
    # Register both agents
    await agent_registry.register_agent(
        agent=medical_bill_agent,
        capabilities=agent_capabilities["medical_bill_agent"]
    )
    
    await agent_registry.register_agent(
        agent=loan_risk_agent,
        capabilities=agent_capabilities["loan_risk_agent"]
    )
    
    # Create a routing request for a loan document
    routing_request = RoutingRequest(
        doc_id="loan_doc_789",
        user_id="customer_101",
        task_type="loan_application_analysis",
        required_capabilities=[TaskCapability.LOAN_ANALYSIS, TaskCapability.RISK_ASSESSMENT],
        model_hint=ModelHint.STANDARD,
        routing_strategy=RoutingStrategy.CAPABILITY_BASED,
        max_agents=1,
        timeout_seconds=30,
        priority=5,
        metadata={"document_type": "loan_application"}
    )
    
    # Route the request
    routing_decision = await router_agent.route_request(routing_request)
    
    # Verify routing decision
    assert routing_decision.success
    assert len(routing_decision.selected_agents) == 1
    assert routing_decision.selected_agents[0].agent_id == "loan_risk_agent"
    assert routing_decision.confidence > 0.8
    assert routing_decision.routing_strategy == RoutingStrategy.CAPABILITY_BASED


@pytest.mark.asyncio
async def test_document_type_based_routing(router_agent, agent_registry, medical_bill_agent, loan_risk_agent, agent_capabilities):
    """Test routing based on document type."""
    # Register both agents
    await agent_registry.register_agent(
        agent=medical_bill_agent,
        capabilities=agent_capabilities["medical_bill_agent"]
    )
    
    await agent_registry.register_agent(
        agent=loan_risk_agent,
        capabilities=agent_capabilities["loan_risk_agent"]
    )
    
    # Create a routing request with document_type in metadata
    routing_request = RoutingRequest(
        doc_id="doc_type_test_123",
        user_id="user_456",
        task_type="document_analysis",
        required_capabilities=[],  # No specific capabilities required
        model_hint=ModelHint.STANDARD,
        routing_strategy=RoutingStrategy.DOCUMENT_TYPE_BASED,
        max_agents=1,
        timeout_seconds=30,
        priority=5,
        metadata={"document_type": "loan_agreement"}
    )
    
    # Route the request
    routing_decision = await router_agent.route_request(routing_request)
    
    # Verify routing decision
    assert routing_decision.success
    assert len(routing_decision.selected_agents) == 1
    assert routing_decision.selected_agents[0].agent_id == "loan_risk_agent"
    assert routing_decision.confidence > 0.8
    assert routing_decision.routing_strategy == RoutingStrategy.DOCUMENT_TYPE_BASED


@pytest.mark.asyncio
async def test_cold_poke_functionality(router_agent):
    """Test RouterAgent's cold-poke functionality via healthz check."""
    # Call the healthz method to test cold-poke
    health_result = await router_agent.healthz()
    
    # Verify health check response
    assert "status" in health_result
    assert health_result["status"] in ["healthy", "unhealthy", "degraded"]
    assert "timestamp" in health_result
    assert "ttl_seconds" in health_result
    assert health_result["ttl_seconds"] > 0
    
    # Verify agent registry status is included
    assert "agent_registry_status" in health_result
    assert "healthy" in health_result["agent_registry_status"]
    
    # Verify active workflows are tracked
    assert "active_workflows" in health_result
    assert isinstance(health_result["active_workflows"], int)
    
    # Verify issues list exists
    assert "issues" in health_result
    assert isinstance(health_result["issues"], list)


@pytest.mark.asyncio
async def test_multi_agent_workflow(router_agent, agent_registry, medical_bill_agent, loan_risk_agent, agent_capabilities):
    """Test multi-agent workflow execution across verticals."""
    # Register both agents
    await agent_registry.register_agent(
        agent=medical_bill_agent,
        capabilities=agent_capabilities["medical_bill_agent"]
    )
    
    await agent_registry.register_agent(
        agent=loan_risk_agent,
        capabilities=agent_capabilities["loan_risk_agent"]
    )
    
    # Define a cross-vertical workflow (unusual but tests the capability)
    workflow = WorkflowDefinition(
        workflow_id="cross_vertical_analysis",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[
            WorkflowStep(
                step_id="step_1_medical",
                agent_id="medical_bill_agent",
                task_input="Analyze medical expenses for loan applicant",
                dependencies=[],
                timeout_seconds=30,
                metadata={"priority": "high"}
            ),
            WorkflowStep(
                step_id="step_2_loan",
                agent_id="loan_risk_agent",
                task_input="Assess loan risk including medical expenses",
                dependencies=["step_1_medical"],
                timeout_seconds=30,
                metadata={"reference": "medical_expenses"}
            )
        ],
        max_parallel_steps=1,
        total_timeout_seconds=120,
        failure_strategy="fail_fast",
        metadata={"workflow_type": "cross_vertical_analysis"}
    )
    
    # Create context for workflow execution
    context = AgentContext(
        doc_id="cross_vertical_123",
        user_id="customer_456",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "cross_vertical_analysis"}
    )
    
    # Mock the agent execution to return results
    with patch.object(router_agent, '_execute_agent_step') as mock_exec:
        mock_exec.side_effect = [
            # Step 1: Medical bill analysis result
            {
                "success": True,
                "data": {
                    "document_type": "medical_bill",
                    "total_amount": 5000,
                    "overcharge_amount": 200
                },
                "agent_id": "medical_bill_agent",
                "execution_time_ms": 2000,
                "cost_rupees": 3.5,
                "confidence": 0.95
            },
            # Step 2: Loan risk analysis result
            {
                "success": True,
                "data": {
                    "document_type": "loan_application",
                    "loan_amount": 25000,
                    "risk_assessment": {
                        "overall_risk_level": "low",
                        "approval_recommendation": "approve"
                    }
                },
                "agent_id": "loan_risk_agent",
                "execution_time_ms": 1500,
                "cost_rupees": 2.8,
                "confidence": 0.92
            }
        ]
        
        # Execute the workflow
        result = await router_agent.execute_workflow(workflow, context)
    
    # Verify workflow execution
    assert result.success
    assert result.workflow_id == "cross_vertical_analysis"
    assert len(result.step_results) == 2
    
    # Verify step execution order and results
    step_1_result = result.step_results["step_1_medical"]
    assert step_1_result.success
    assert step_1_result.agent_id == "medical_bill_agent"
    
    step_2_result = result.step_results["step_2_loan"]
    assert step_2_result.success
    assert step_2_result.agent_id == "loan_risk_agent"
    
    # Verify cost and time calculations
    assert result.total_cost_rupees == pytest.approx(6.3)  # 3.5 + 2.8
    assert result.total_execution_time_ms > 0
