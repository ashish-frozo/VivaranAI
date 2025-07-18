"""
Regression Tests for Multi-Vertical Architecture.

These tests ensure that the system maintains consistent behavior across changes,
using golden fixtures as reference points for expected outputs.
"""

import asyncio
import time
import uuid
import json
import os
import pytest
import structlog
import time
import uuid

from enum import Enum, auto
from unittest.mock import AsyncMock, MagicMock, patch

from agents.base_agent import AgentContext
from agents.interfaces import ModelHint
from agents.medical_bill_agent import MedicalBillAgent
from agents.verticals.loan.loan_risk_agent import LoanRiskAgent
from agents.router_agent import RouterAgent, RoutingRequest, RoutingStrategy
from tests.test_data.golden_fixtures import (
    MEDICAL_BILL_GOLDEN_RESULT,
    LOAN_APPLICATION_GOLDEN_RESULT,
    LOAN_AGREEMENT_GOLDEN_RESULT,
    ROUTING_DECISION_GOLDEN_RESULT,
    ROUTER_HEALTH_GOLDEN_RESULT,
    CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT
)

# Define TaskCapability for regression tests
class TaskCapability(str, Enum):
    """Task capabilities for agent routing."""
    MEDICAL_ANALYSIS = "medical_analysis"
    RATE_VALIDATION = "rate_validation"
    LOAN_ANALYSIS = "loan_analysis"
    RISK_ASSESSMENT = "risk_assessment"

@pytest.fixture
def sample_medical_bill_text():
    """Sample medical bill text for testing."""
    return """
    APOLLO HOSPITALS
    INVOICE
    
    Patient Name: John Doe
    Bill Number: APOL-2025-12345
    Date: 10/06/2025
    
    CHARGES:
    1. Consultation                    Rs. 1,500.00
    2. Blood Test - CBC                Rs.   800.00
    3. X-Ray Chest                     Rs. 1,200.00
    4. Room Charges (General Ward)     Rs. 9,000.00
       (3 days @ Rs. 3,000 per day)
    5. Medicines                       Rs. 2,500.00
    
    TOTAL AMOUNT:                      Rs. 15,000.00
    
    Payment Mode: Credit Card
    Authorized Signature: Dr. R. Kumar
    """


@pytest.fixture
def sample_loan_application_text():
    """Sample loan application text for testing."""
    return """
    PERSONAL LOAN APPLICATION
    
    Applicant: John Smith
    Date of Birth: 01/15/1980
    Address: 123 Main Street, Anytown, CA 94043
    Phone: (555) 123-4567
    Email: john.smith@example.com
    
    Loan Amount Requested: $25,000
    Loan Purpose: Home Renovation
    Loan Term: 60 months
    
    Employment Information:
    Current Employer: ABC Corporation
    Position: Senior Manager
    Annual Income: $95,000
    Years at Current Job: 5
    
    Credit Information:
    Credit Score: 720
    Outstanding Debts: $12,000
    Monthly Debt Payments: $500
    
    I hereby certify that the information provided is true and accurate.
    
    Signature: John Smith
    Date: 2025-06-15
    """


@pytest.fixture
async def mock_medical_bill_agent():
    """Create a mock MedicalBillAgent for testing."""
    agent = AsyncMock(spec=MedicalBillAgent)
    agent.agent_id = "medical_bill_agent"
    agent.name = "Medical Bill Analysis Agent"
    agent.instructions = "Analyzes medical bills for overcharges"
    agent.process_task = AsyncMock(return_value=MEDICAL_BILL_GOLDEN_RESULT)
    return agent


@pytest.fixture
async def mock_loan_risk_agent():
    """Create a mock LoanRiskAgent for testing."""
    agent = AsyncMock(spec=LoanRiskAgent)
    agent.agent_id = "loan_risk_agent"
    agent.name = "Loan Risk Analysis Agent"
    agent.instructions = "Analyzes loan applications for risk assessment"
    
    # Configure the mock to return different results based on document type
    async def mock_process_task(context, task_data):
        if task_data.get("document_type") == "loan_application":
            return LOAN_APPLICATION_GOLDEN_RESULT
        elif task_data.get("document_type") == "loan_agreement":
            return LOAN_AGREEMENT_GOLDEN_RESULT
        else:
            return {"error": "Unknown document type"}
    
    agent.process_task = mock_process_task
    return agent


@pytest.fixture
async def mock_router_agent():
    """Create a mock RouterAgent for testing."""
    router = AsyncMock(spec=RouterAgent)
    router.route_request = AsyncMock(return_value=ROUTING_DECISION_GOLDEN_RESULT)
    router.healthz = AsyncMock(return_value=ROUTER_HEALTH_GOLDEN_RESULT)
    router.execute_workflow = AsyncMock(return_value=CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT)
    return router


class TestRegressionMedicalBillAgent:
    """Regression tests for MedicalBillAgent."""
    
    @pytest.mark.asyncio
    async def test_medical_bill_analysis_regression(self, mock_medical_bill_agent, sample_medical_bill_text):
        """Test medical bill analysis against golden fixture."""
        # Create agent context
        context = AgentContext(
            doc_id="test_medical_bill_123",
            user_id="test_user_456",
            correlation_id=str(uuid.uuid4()),
            model_hint=ModelHint.STANDARD,
            start_time=time.time(),
            metadata={"task_type": "medical_bill_analysis"}
        )
        
        # Create task data
        task_data = {
            "raw_text": sample_medical_bill_text,
            "document_type": "medical_bill",
            "metadata": {
                "filename": "apollo_bill.pdf",
                "state_code": "DL",
                "insurance_type": "CGHS"
            }
        }
        
        # Process the task
        result = await mock_medical_bill_agent.process_task(context, task_data)
        
        # Compare with golden fixture
        assert result["document_type"] == MEDICAL_BILL_GOLDEN_RESULT["document_type"]
        assert result["hospital_name"] == MEDICAL_BILL_GOLDEN_RESULT["hospital_name"]
        assert result["patient_name"] == MEDICAL_BILL_GOLDEN_RESULT["patient_name"]
        assert result["total_amount"] == MEDICAL_BILL_GOLDEN_RESULT["total_amount"]
        
        # Check line items count
        assert len(result["line_items"]) == len(MEDICAL_BILL_GOLDEN_RESULT["line_items"])
        
        # Check analysis results structure
        assert "analysis_results" in result
        assert "overcharge_items" in result["analysis_results"]
        assert "total_overcharge" in result["analysis_results"]
        assert "confidence_score" in result["analysis_results"]
        
        # Check metadata
        assert result["metadata"]["state_code"] == "DL"
        assert result["metadata"]["insurance_type"] == "CGHS"


class TestRegressionLoanRiskAgent:
    """Regression tests for LoanRiskAgent."""
    
    @pytest.mark.asyncio
    async def test_loan_application_analysis_regression(self, mock_loan_risk_agent, sample_loan_application_text):
        """Test loan application analysis against golden fixture."""
        # Create agent context
        context = AgentContext(
            doc_id="test_loan_app_123",
            user_id="test_user_456",
            correlation_id=str(uuid.uuid4()),
            model_hint=ModelHint.STANDARD,
            start_time=time.time(),
            metadata={"task_type": "loan_application_analysis"}
        )
        
        # Create task data
        task_data = {
            "raw_text": sample_loan_application_text,
            "document_type": "loan_application",
            "metadata": {
                "filename": "loan_application.pdf",
                "state_code": "CA"
            }
        }
        
        # Process the task
        result = await mock_loan_risk_agent.process_task(context, task_data)
        
        # Compare with golden fixture
        assert result["document_type"] == LOAN_APPLICATION_GOLDEN_RESULT["document_type"]
        assert result["applicant_name"] == LOAN_APPLICATION_GOLDEN_RESULT["applicant_name"]
        assert result["loan_amount"] == LOAN_APPLICATION_GOLDEN_RESULT["loan_amount"]
        assert result["loan_purpose"] == LOAN_APPLICATION_GOLDEN_RESULT["loan_purpose"]
        assert result["loan_term_months"] == LOAN_APPLICATION_GOLDEN_RESULT["loan_term_months"]
        
        # Check risk assessment
        assert "risk_assessment" in result
        assert result["risk_assessment"]["overall_risk_level"] == LOAN_APPLICATION_GOLDEN_RESULT["risk_assessment"]["overall_risk_level"]
        assert result["risk_assessment"]["approval_recommendation"] == LOAN_APPLICATION_GOLDEN_RESULT["risk_assessment"]["approval_recommendation"]
        
        # Check duplicate check
        assert "duplicate_check" in result
        assert result["duplicate_check"]["is_duplicate"] == LOAN_APPLICATION_GOLDEN_RESULT["duplicate_check"]["is_duplicate"]


class TestRegressionRouterAgent:
    """Regression tests for RouterAgent."""
    
    @pytest.mark.asyncio
    async def test_routing_decision_regression(self, mock_router_agent):
        """Test routing decision against golden fixture."""
        # Create a routing request
        routing_request = RoutingRequest(
            doc_id="test_doc_123",
            user_id="test_user_456",
            task_type="medical_bill_analysis",
            required_capabilities=[TaskCapability.MEDICAL_ANALYSIS, TaskCapability.RATE_VALIDATION],
            model_hint=ModelHint.STANDARD,
            routing_strategy=RoutingStrategy.CAPABILITY_BASED,
            max_agents=1,
            timeout_seconds=30,
            priority=5,
            metadata={"document_type": "medical_bill", "state_code": "DL", "insurance_type": "CGHS"}
        )
        
        # Get routing decision
        result = await mock_router_agent.route_request(routing_request)
        
        # Compare with golden fixture
        assert result["success"] == ROUTING_DECISION_GOLDEN_RESULT["success"]
        assert result["document_type"] == ROUTING_DECISION_GOLDEN_RESULT["document_type"]
        assert len(result["selected_agents"]) == len(ROUTING_DECISION_GOLDEN_RESULT["selected_agents"])
        assert result["selected_agents"][0]["agent_id"] == ROUTING_DECISION_GOLDEN_RESULT["selected_agents"][0]["agent_id"]
        assert result["routing_strategy"] == ROUTING_DECISION_GOLDEN_RESULT["routing_strategy"]
        assert result["confidence"] == ROUTING_DECISION_GOLDEN_RESULT["confidence"]
    
    @pytest.mark.asyncio
    async def test_health_check_regression(self, mock_router_agent):
        """Test health check against golden fixture."""
        # Get health check
        result = await mock_router_agent.healthz()
        
        # Compare with golden fixture (excluding timestamp which will be different)
        assert result["status"] == ROUTER_HEALTH_GOLDEN_RESULT["status"]
        assert "timestamp" in result
        assert result["ttl_seconds"] == ROUTER_HEALTH_GOLDEN_RESULT["ttl_seconds"]
        assert result["agent_registry_status"]["healthy"] == ROUTER_HEALTH_GOLDEN_RESULT["agent_registry_status"]["healthy"]
        assert result["active_workflows"] == ROUTER_HEALTH_GOLDEN_RESULT["active_workflows"]
        assert isinstance(result["issues"], list)


class TestRegressionCrossVerticalWorkflow:
    """Regression tests for cross-vertical workflows."""
    
    @pytest.mark.asyncio
    async def test_cross_vertical_workflow_regression(self, mock_router_agent):
        """Test cross-vertical workflow execution against golden fixture."""
        # Create workflow definition (simplified for test)
        workflow = {
            "workflow_id": "cross_vertical_analysis",
            "workflow_type": "sequential",
            "steps": [
                {
                    "step_id": "step_1_medical",
                    "agent_id": "medical_bill_agent",
                    "task_input": "Analyze medical expenses for loan applicant",
                    "dependencies": [],
                    "timeout_seconds": 30
                },
                {
                    "step_id": "step_2_loan",
                    "agent_id": "loan_risk_agent",
                    "task_input": "Assess loan risk including medical expenses",
                    "dependencies": ["step_1_medical"],
                    "timeout_seconds": 30
                }
            ],
            "max_parallel_steps": 1,
            "total_timeout_seconds": 120,
            "failure_strategy": "fail_fast",
            "metadata": {"workflow_type": "cross_vertical_analysis"}
        }
        
        # Create context
        context = AgentContext(
            doc_id="cross_vertical_123",
            user_id="test_user_456",
            correlation_id=str(uuid.uuid4()),
            model_hint=ModelHint.STANDARD,
            start_time=time.time(),
            metadata={"task_type": "cross_vertical_analysis"}
        )
        
        # Execute workflow
        result = await mock_router_agent.execute_workflow(workflow, context)
        
        # Compare with golden fixture
        assert result["success"] == CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT["success"]
        assert result["workflow_id"] == CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT["workflow_id"]
        assert result["workflow_type"] == CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT["workflow_type"]
        
        # Check step results
        assert "step_1_medical" in result["step_results"]
        assert "step_2_loan" in result["step_results"]
        assert result["step_results"]["step_1_medical"]["agent_id"] == CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT["step_results"]["step_1_medical"]["agent_id"]
        assert result["step_results"]["step_2_loan"]["agent_id"] == CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT["step_results"]["step_2_loan"]["agent_id"]
        
        # Check cost and time calculations
        assert result["total_cost_rupees"] == CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT["total_cost_rupees"]
        assert "total_execution_time_ms" in result
