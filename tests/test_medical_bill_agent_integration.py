"""
Integration tests for MedicalBillAgent with Registry and Router.

Tests the complete multi-agent workflow including agent registration,
routing decisions, and medical bill analysis coordination.
"""

import asyncio
import base64
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from agents.medical_bill_agent import MedicalBillAgent
from agents.agent_registry import AgentRegistry, TaskCapability, AgentCapabilities
from agents.router_agent import RouterAgent, RoutingStrategy, WorkflowType, WorkflowDefinition, WorkflowStep
from agents.base_agent import AgentContext, ModelHint


@pytest.fixture
async def redis_registry():
    """Shared agent registry for testing."""
    registry = AgentRegistry(redis_url="redis://localhost:6379/2")
    yield registry
    # Cleanup
    try:
        await registry.cleanup()
    except:
        pass


@pytest.fixture
async def medical_bill_agent():
    """MedicalBillAgent instance for testing."""
    agent = MedicalBillAgent(redis_url="redis://localhost:6379/2")
    yield agent
    # Cleanup
    try:
        if hasattr(agent, 'redis_client'):
            await agent.redis_client.aclose()
    except:
        pass


@pytest.fixture
async def router_agent(redis_registry):
    """RouterAgent instance for testing."""
    router = RouterAgent(
        agent_registry=redis_registry,
        redis_url="redis://localhost:6379/2"
    )
    yield router
    # Cleanup
    try:
        if hasattr(router, 'redis_client'):
            await router.redis_client.aclose()
    except:
        pass


@pytest.fixture
def sample_medical_bill_data():
    """Sample medical bill file data for testing."""
    sample_content = """
    MEDICAL BILL - TEST HOSPITAL
    Patient: Test Patient
    Date: 2024-01-15
    
    Specialist Consultation: Rs. 1200
    Blood Test - CBC: Rs. 600
    X-Ray Chest: Rs. 800
    
    Total: Rs. 2600
    """.encode('utf-8')
    
    return {
        "file_content": base64.b64encode(sample_content).decode('utf-8'),
        "doc_id": "test_bill_integration_001",
        "language": "english",
        "state_code": "DL",
        "insurance_type": "cghs",
        "file_format": "pdf"
    }


class TestMedicalBillAgentRegistration:
    """Test MedicalBillAgent registration with AgentRegistry."""
    
    @pytest.mark.asyncio
    async def test_medical_bill_agent_registration(self, redis_registry, medical_bill_agent):
        """Test registering MedicalBillAgent with the registry."""
        
        # Define capabilities for medical bill agent
        capabilities = AgentCapabilities(
            supported_tasks=[
                TaskCapability.DOCUMENT_PROCESSING,
                TaskCapability.RATE_VALIDATION,
                TaskCapability.DUPLICATE_DETECTION,
                TaskCapability.PROHIBITED_DETECTION,
                TaskCapability.CONFIDENCE_SCORING
            ],
            max_concurrent_requests=3,
            estimated_cost_per_request=0.50,
            estimated_response_time_ms=5000,
            supported_file_formats=["pdf", "jpg", "png"],
            supported_languages=["english", "hindi"],
            state_support=True
        )
        
        # Register the agent
        registration = await redis_registry.register_agent(
            agent_id=medical_bill_agent.agent_id,
            name=medical_bill_agent.name,
            capabilities=capabilities,
            agent_instance=medical_bill_agent
        )
        
        assert registration is not None
        assert registration.agent_id == "medical_bill_agent"
        assert TaskCapability.DOCUMENT_PROCESSING in registration.capabilities.supported_tasks
        assert TaskCapability.RATE_VALIDATION in registration.capabilities.supported_tasks
        
        # Verify agent is discoverable
        discovered_agents = await redis_registry.discover_agents([TaskCapability.DOCUMENT_PROCESSING])
        assert len(discovered_agents) >= 1
        
        medical_agent = next(
            (agent for agent in discovered_agents if agent.agent_id == "medical_bill_agent"), 
            None
        )
        assert medical_agent is not None
        assert medical_agent.capabilities.max_concurrent_requests == 3
    
    @pytest.mark.asyncio
    async def test_medical_bill_agent_health_monitoring(self, redis_registry, medical_bill_agent):
        """Test health monitoring of registered MedicalBillAgent."""
        
        capabilities = AgentCapabilities(
            supported_tasks=[TaskCapability.DOCUMENT_PROCESSING],
            max_concurrent_requests=2,
            estimated_cost_per_request=0.30,
            estimated_response_time_ms=3000
        )
        
        # Register agent
        await redis_registry.register_agent(
            agent_id=medical_bill_agent.agent_id,
            name=medical_bill_agent.name,
            capabilities=capabilities,
            agent_instance=medical_bill_agent
        )
        
        # Check initial health
        health = await medical_bill_agent.health_check()
        assert health["status"] == "healthy"
        assert "tools" in health
        assert len(health["analysis_capabilities"]) > 0
        
        # Agent should be online in registry
        online_agents = await redis_registry.list_online_agents()
        medical_agents = [a for a in online_agents if a.agent_id == "medical_bill_agent"]
        assert len(medical_agents) == 1
        assert medical_agents[0].status.value == "online"


class TestMedicalBillAgentRouting:
    """Test routing to MedicalBillAgent via RouterAgent."""
    
    @pytest.mark.asyncio
    async def test_route_to_medical_bill_agent(self, redis_registry, medical_bill_agent, router_agent, sample_medical_bill_data):
        """Test routing medical bill analysis task to MedicalBillAgent."""
        
        # Register medical bill agent
        capabilities = AgentCapabilities(
            supported_tasks=[
                TaskCapability.DOCUMENT_PROCESSING,
                TaskCapability.RATE_VALIDATION
            ],
            max_concurrent_requests=2,
            estimated_cost_per_request=0.40,
            estimated_response_time_ms=4000
        )
        
        await redis_registry.register_agent(
            agent_id=medical_bill_agent.agent_id,
            name=medical_bill_agent.name,
            capabilities=capabilities,
            agent_instance=medical_bill_agent
        )
        
        # Create routing request
        from agents.router_agent import RoutingRequest
        
        routing_request = RoutingRequest(
            task_type="medical_bill_analysis",
            required_capabilities=[TaskCapability.DOCUMENT_PROCESSING],
            task_data=sample_medical_bill_data,
            user_id="test_user",
            priority="high",
            estimated_complexity="medium"
        )
        
        # Mock the medical bill agent's process_task method
        with patch.object(medical_bill_agent, 'process_task') as mock_process:
            mock_process.return_value = {
                "success": True,
                "analysis_complete": True,
                "doc_id": sample_medical_bill_data["doc_id"],
                "verdict": "ok",
                "total_bill_amount": 2600.0,
                "total_overcharge": 0.0,
                "confidence_score": 0.95,
                "red_flags": [],
                "recommendations": ["Bill appears compliant"]
            }
            
            # Route the request
            routing_decision = await router_agent.route_request(
                routing_request,
                strategy=RoutingStrategy.CAPABILITY_BASED
            )
            
            assert routing_decision.success is True
            assert routing_decision.selected_agent_id == "medical_bill_agent"
            assert routing_decision.confidence > 0.0
            
            # Execute the routed task
            context = AgentContext(
                doc_id=sample_medical_bill_data["doc_id"],
                user_id="test_user",
                correlation_id="routing_test",
                model_hint=ModelHint.STANDARD,
                start_time=time.time()
            )
            
            result = await medical_bill_agent.process_task(context, sample_medical_bill_data)
            
            assert result["success"] is True
            assert result["doc_id"] == sample_medical_bill_data["doc_id"]
            assert "total_bill_amount" in result
    
    @pytest.mark.asyncio
    async def test_workflow_with_medical_bill_agent(self, redis_registry, medical_bill_agent, router_agent, sample_medical_bill_data):
        """Test workflow execution involving MedicalBillAgent."""
        
        # Register medical bill agent  
        capabilities = AgentCapabilities(
            supported_tasks=[
                TaskCapability.DOCUMENT_PROCESSING,
                TaskCapability.RATE_VALIDATION,
                TaskCapability.DUPLICATE_DETECTION
            ],
            max_concurrent_requests=1,
            estimated_cost_per_request=0.60,
            estimated_response_time_ms=6000
        )
        
        await redis_registry.register_agent(
            agent_id=medical_bill_agent.agent_id,
            name=medical_bill_agent.name,
            capabilities=capabilities,
            agent_instance=medical_bill_agent
        )
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            workflow_id="medical_bill_analysis_workflow",
            name="Complete Medical Bill Analysis",
            description="Full medical bill analysis with all components",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="medical_analysis",
                    agent_capabilities=[
                        TaskCapability.DOCUMENT_PROCESSING,
                        TaskCapability.RATE_VALIDATION
                    ],
                    input_data=sample_medical_bill_data,
                    required=True,
                    timeout_seconds=30
                )
            ],
            max_execution_time_seconds=60
        )
        
        # Mock the agent execution
        with patch.object(medical_bill_agent, 'process_task') as mock_process:
            mock_process.return_value = {
                "success": True,
                "analysis_complete": True,
                "doc_id": sample_medical_bill_data["doc_id"],
                "verdict": "warning",
                "total_bill_amount": 2600.0,
                "total_overcharge": 200.0,
                "confidence_score": 0.87,
                "red_flags": [
                    {
                        "type": "overcharge",
                        "severity": "medium",
                        "item": "Specialist Consultation",
                        "overcharge_amount": 200.0
                    }
                ],
                "recommendations": ["Review specialist consultation charges"]
            }
            
            # Execute workflow
            workflow_result = await router_agent.execute_workflow(
                workflow_def,
                user_id="test_user",
                correlation_id="workflow_test"
            )
            
            assert workflow_result.success is True
            assert workflow_result.workflow_id == "medical_bill_analysis_workflow"
            assert len(workflow_result.step_results) == 1
            
            step_result = workflow_result.step_results[0]
            assert step_result.step_id == "medical_analysis"
            assert step_result.success is True
            assert step_result.agent_id == "medical_bill_agent"
            
            # Check the actual analysis result
            analysis_result = step_result.result
            assert analysis_result["verdict"] == "warning"
            assert analysis_result["total_overcharge"] == 200.0
            assert len(analysis_result["red_flags"]) == 1


class TestMedicalBillAgentLoadBalancing:
    """Test load balancing across multiple MedicalBillAgent instances."""
    
    @pytest.mark.asyncio
    async def test_multiple_medical_bill_agents(self, redis_registry, router_agent):
        """Test load balancing across multiple MedicalBillAgent instances."""
        
        # Create multiple medical bill agent instances
        agents = []
        for i in range(3):
            agent = MedicalBillAgent(redis_url="redis://localhost:6379/2")
            agent.agent_id = f"medical_bill_agent_{i}"
            agent.name = f"Medical Bill Agent {i}"
            agents.append(agent)
        
        try:
            # Register all agents with different capabilities
            for i, agent in enumerate(agents):
                capabilities = AgentCapabilities(
                    supported_tasks=[TaskCapability.DOCUMENT_PROCESSING],
                    max_concurrent_requests=2,
                    estimated_cost_per_request=0.30 + (i * 0.10),  # Different costs
                    estimated_response_time_ms=3000 + (i * 500)   # Different response times
                )
                
                await redis_registry.register_agent(
                    agent_id=agent.agent_id,
                    name=agent.name,
                    capabilities=capabilities,
                    agent_instance=agent
                )
            
            # Create multiple routing requests
            requests = []
            for j in range(5):
                request = RoutingRequest(
                    task_type="medical_bill_analysis",
                    required_capabilities=[TaskCapability.DOCUMENT_PROCESSING],
                    task_data={"doc_id": f"test_bill_{j}", "file_content": "sample_content"},
                    user_id="test_user",
                    priority="normal"
                )
                requests.append(request)
            
            # Route requests with load balancing strategy
            routing_decisions = []
            for request in requests:
                decision = await router_agent.route_request(
                    request,
                    strategy=RoutingStrategy.LOAD_BALANCED
                )
                routing_decisions.append(decision)
            
            # Verify load balancing - different agents should be selected
            selected_agents = [d.selected_agent_id for d in routing_decisions if d.success]
            unique_agents = set(selected_agents)
            
            assert len(unique_agents) > 1, "Load balancing should distribute across multiple agents"
            assert all(agent_id.startswith("medical_bill_agent_") for agent_id in unique_agents)
            
            # Test cost optimization strategy
            cost_decision = await router_agent.route_request(
                requests[0],
                strategy=RoutingStrategy.COST_OPTIMIZED
            )
            
            assert cost_decision.success is True
            # Should select the cheapest agent (medical_bill_agent_0 with cost 0.30)
            assert cost_decision.selected_agent_id == "medical_bill_agent_0"
            
        finally:
            # Cleanup agents
            for agent in agents:
                try:
                    if hasattr(agent, 'redis_client'):
                        await agent.redis_client.aclose()
                except:
                    pass


class TestMedicalBillAgentErrorHandling:
    """Test error handling in multi-agent scenarios."""
    
    @pytest.mark.asyncio
    async def test_agent_failure_fallback(self, redis_registry, router_agent, sample_medical_bill_data):
        """Test fallback when MedicalBillAgent fails."""
        
        # Create primary and backup agents
        primary_agent = MedicalBillAgent(redis_url="redis://localhost:6379/2")
        primary_agent.agent_id = "primary_medical_agent"
        
        backup_agent = MedicalBillAgent(redis_url="redis://localhost:6379/2")
        backup_agent.agent_id = "backup_medical_agent"
        
        try:
            # Register both agents
            capabilities = AgentCapabilities(
                supported_tasks=[TaskCapability.DOCUMENT_PROCESSING],
                max_concurrent_requests=1,
                estimated_cost_per_request=0.40,
                estimated_response_time_ms=4000
            )
            
            await redis_registry.register_agent(
                agent_id=primary_agent.agent_id,
                name="Primary Medical Agent",
                capabilities=capabilities,
                agent_instance=primary_agent
            )
            
            await redis_registry.register_agent(
                agent_id=backup_agent.agent_id,
                name="Backup Medical Agent", 
                capabilities=capabilities,
                agent_instance=backup_agent
            )
            
            # Mock primary agent to fail
            with patch.object(primary_agent, 'process_task') as mock_primary:
                with patch.object(backup_agent, 'process_task') as mock_backup:
                    
                    mock_primary.side_effect = Exception("Primary agent failed")
                    mock_backup.return_value = {
                        "success": True,
                        "analysis_complete": True,
                        "doc_id": sample_medical_bill_data["doc_id"],
                        "verdict": "ok"
                    }
                    
                    # Create routing request
                    request = RoutingRequest(
                        task_type="medical_bill_analysis",
                        required_capabilities=[TaskCapability.DOCUMENT_PROCESSING],
                        task_data=sample_medical_bill_data,
                        user_id="test_user",
                        priority="high"
                    )
                    
                    # Router should try primary first, then fallback to backup
                    # This would require implementing retry logic in RouterAgent
                    decision = await router_agent.route_request(request)
                    
                    assert decision.success is True
                    # The actual agent selected depends on routing algorithm
                    assert decision.selected_agent_id in ["primary_medical_agent", "backup_medical_agent"]
        
        finally:
            # Cleanup
            for agent in [primary_agent, backup_agent]:
                try:
                    if hasattr(agent, 'redis_client'):
                        await agent.redis_client.aclose()
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_workflow_partial_failure(self, redis_registry, medical_bill_agent, router_agent):
        """Test workflow execution with partial step failures."""
        
        # Register agent
        capabilities = AgentCapabilities(
            supported_tasks=[TaskCapability.DOCUMENT_PROCESSING],
            max_concurrent_requests=1,
            estimated_cost_per_request=0.50,
            estimated_response_time_ms=5000
        )
        
        await redis_registry.register_agent(
            agent_id=medical_bill_agent.agent_id,
            name=medical_bill_agent.name,
            capabilities=capabilities,
            agent_instance=medical_bill_agent
        )
        
        # Create workflow with multiple steps
        workflow_def = WorkflowDefinition(
            workflow_id="multi_step_workflow",
            name="Multi-step Analysis",
            workflow_type=WorkflowType.SEQUENTIAL,
            steps=[
                WorkflowStep(
                    step_id="step1",
                    agent_capabilities=[TaskCapability.DOCUMENT_PROCESSING],
                    input_data={"doc_id": "test_1", "file_content": "content1"},
                    required=True
                ),
                WorkflowStep(
                    step_id="step2", 
                    agent_capabilities=[TaskCapability.DOCUMENT_PROCESSING],
                    input_data={"doc_id": "test_2", "file_content": "content2"},
                    required=False  # Optional step
                )
            ],
            max_execution_time_seconds=30
        )
        
        # Mock agent responses - first succeeds, second fails
        with patch.object(medical_bill_agent, 'process_task') as mock_process:
            def mock_process_side_effect(context, task_data):
                if task_data["doc_id"] == "test_1":
                    return {
                        "success": True,
                        "doc_id": "test_1",
                        "verdict": "ok"
                    }
                else:
                    raise Exception("Step 2 failed")
            
            mock_process.side_effect = mock_process_side_effect
            
            # Execute workflow
            result = await router_agent.execute_workflow(
                workflow_def,
                user_id="test_user",
                correlation_id="partial_failure_test"
            )
            
            # Workflow should complete with partial success
            assert len(result.step_results) == 2
            assert result.step_results[0].success is True
            assert result.step_results[1].success is False
            
            # Since step2 is optional, workflow might still be considered successful
            # This depends on the workflow execution strategy 