"""
Loan Risk Agent Tests.

Tests the LoanRiskAgent implementation including:
- Document processing for loan applications
- Rule pack loading and application
- Duplicate detection for loan documents
- Multi-vertical pack-driven architecture support
"""

import asyncio
import os
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog

from agents.base_agent import BaseAgent, AgentContext
from agents.interfaces import ModelHint
from agents.loan_risk_agent import LoanRiskAgent
from agents.tools.duplicate_detector import DuplicateDetector
from packs.loan.rule_loader import LoanRuleLoader
from packs.loan.rule_executor import LoanRuleExecutor


@pytest.fixture
def sample_loan_application():
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
    
    References:
    1. Jane Doe, (555) 987-6543
    2. Robert Johnson, (555) 456-7890
    
    I hereby certify that the information provided is true and accurate.
    
    Signature: John Smith
    Date: 2025-06-15
    """


@pytest.fixture
def sample_loan_agreement():
    """Sample loan agreement text for testing."""
    return """
    LOAN AGREEMENT
    
    Agreement Date: June 20, 2025
    
    LENDER: First National Bank
    BORROWER: John Smith
    
    LOAN DETAILS:
    Principal Amount: $25,000.00
    Interest Rate: 5.75% per annum
    Term: 60 months
    Monthly Payment: $480.25
    
    PAYMENT SCHEDULE:
    First Payment Due: August 1, 2025
    Payment Frequency: Monthly
    Final Payment Date: July 1, 2030
    
    SECURITY:
    This loan is unsecured.
    
    DEFAULT:
    If the Borrower fails to make any payment when due and such failure continues for 15 days, the entire unpaid balance shall become immediately due and payable.
    
    PREPAYMENT:
    The Borrower may prepay this loan in whole or in part without penalty.
    
    GOVERNING LAW:
    This Agreement shall be governed by the laws of the State of California.
    
    SIGNATURES:
    
    ________________________
    First National Bank
    
    ________________________
    John Smith (Borrower)
    """


@pytest.fixture
def golden_loan_analysis_result():
    """Golden fixture for loan analysis result."""
    return {
        "document_type": "loan_application",
        "applicant_name": "John Smith",
        "loan_amount": 25000.0,
        "loan_purpose": "Home Renovation",
        "loan_term_months": 60,
        "annual_income": 95000.0,
        "credit_score": 720,
        "debt_to_income_ratio": 0.063,  # ($500 * 12) / $95,000
        "risk_assessment": {
            "credit_risk_score": 82,
            "income_stability_score": 90,
            "debt_burden_score": 85,
            "overall_risk_level": "low",
            "approval_recommendation": "approve"
        },
        "duplicate_check": {
            "is_duplicate": False,
            "similarity_score": 0.0,
            "similar_documents": []
        },
        "rule_violations": [],
        "flags": []
    }


@pytest.fixture
async def loan_risk_agent():
    """Create a LoanRiskAgent instance for testing."""
    # Create mock tools
    mock_duplicate_detector = AsyncMock(spec=DuplicateDetector)
    mock_duplicate_detector.check_duplicate.return_value = {
        "is_duplicate": False,
        "similarity_score": 0.0,
        "similar_documents": []
    }
    
    # Create mock rule loader and executor
    mock_rule_loader = MagicMock(spec=LoanRuleLoader)
    mock_rule_loader.load_rules.return_value = {
        "credit_score_rules": [
            {"min_score": 700, "risk_level": "low"},
            {"min_score": 650, "risk_level": "medium"},
            {"min_score": 0, "risk_level": "high"}
        ],
        "debt_to_income_rules": [
            {"max_ratio": 0.36, "risk_level": "low"},
            {"max_ratio": 0.43, "risk_level": "medium"},
            {"max_ratio": 1.0, "risk_level": "high"}
        ]
    }
    
    mock_rule_executor = AsyncMock(spec=LoanRuleExecutor)
    mock_rule_executor.execute_rules.return_value = {
        "credit_risk_score": 82,
        "income_stability_score": 90,
        "debt_burden_score": 85,
        "overall_risk_level": "low",
        "approval_recommendation": "approve",
        "rule_violations": [],
        "flags": []
    }
    
    # Create agent with mock tools
    with patch('agents.loan_risk_agent.DuplicateDetector', return_value=mock_duplicate_detector), \
         patch('agents.loan_risk_agent.LoanRuleLoader', return_value=mock_rule_loader), \
         patch('agents.loan_risk_agent.LoanRuleExecutor', return_value=mock_rule_executor):
        
        agent = LoanRiskAgent()
        await agent.start()
        yield agent
        await agent.stop()


@pytest.mark.asyncio
async def test_loan_application_analysis(loan_risk_agent, sample_loan_application, golden_loan_analysis_result):
    """Test loan application analysis with the LoanRiskAgent."""
    # Create agent context
    context = AgentContext(
        doc_id="loan_app_123",
        user_id="customer_456",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "loan_application_analysis"}
    )
    
    # Create task data
    task_data = {
        "raw_text": sample_loan_application,
        "document_type": "loan_application",
        "metadata": {
            "filename": "loan_application.pdf",
            "state_code": "CA"
        }
    }
    
    # Process the task
    result = await loan_risk_agent.process_task(context, task_data)
    
    # Verify the result matches golden fixture (with some tolerance for floating point values)
    assert result["document_type"] == golden_loan_analysis_result["document_type"]
    assert result["applicant_name"] == golden_loan_analysis_result["applicant_name"]
    assert result["loan_amount"] == golden_loan_analysis_result["loan_amount"]
    assert result["loan_purpose"] == golden_loan_analysis_result["loan_purpose"]
    assert result["loan_term_months"] == golden_loan_analysis_result["loan_term_months"]
    assert result["annual_income"] == golden_loan_analysis_result["annual_income"]
    assert result["credit_score"] == golden_loan_analysis_result["credit_score"]
    
    # Check risk assessment
    risk = result["risk_assessment"]
    golden_risk = golden_loan_analysis_result["risk_assessment"]
    assert risk["overall_risk_level"] == golden_risk["overall_risk_level"]
    assert risk["approval_recommendation"] == golden_risk["approval_recommendation"]
    
    # Check duplicate detection
    assert result["duplicate_check"]["is_duplicate"] == golden_loan_analysis_result["duplicate_check"]["is_duplicate"]


@pytest.mark.asyncio
async def test_loan_agreement_analysis(loan_risk_agent, sample_loan_agreement):
    """Test loan agreement analysis with the LoanRiskAgent."""
    # Create agent context
    context = AgentContext(
        doc_id="loan_agreement_789",
        user_id="customer_456",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "loan_agreement_analysis"}
    )
    
    # Create task data
    task_data = {
        "raw_text": sample_loan_agreement,
        "document_type": "loan_agreement",
        "metadata": {
            "filename": "loan_agreement.pdf",
            "state_code": "CA"
        }
    }
    
    # Process the task
    result = await loan_risk_agent.process_task(context, task_data)
    
    # Verify the result structure for loan agreement
    assert result["document_type"] == "loan_agreement"
    assert "principal_amount" in result
    assert "interest_rate" in result
    assert "term_months" in result
    assert "monthly_payment" in result
    assert "lender_name" in result
    assert "borrower_name" in result
    
    # Check compliance analysis is present
    assert "compliance_check" in result
    assert "governing_law" in result
    assert "rule_violations" in result


@pytest.mark.asyncio
async def test_duplicate_detection_integration(loan_risk_agent, sample_loan_application):
    """Test duplicate detection integration in LoanRiskAgent."""
    # Mock the duplicate detector to return a duplicate
    loan_risk_agent.duplicate_detector.check_duplicate.return_value = {
        "is_duplicate": True,
        "similarity_score": 0.92,
        "similar_documents": [
            {
                "doc_id": "previous_loan_app_456",
                "similarity": 0.92,
                "timestamp": "2025-06-10T14:30:00Z"
            }
        ]
    }
    
    # Create agent context
    context = AgentContext(
        doc_id="duplicate_loan_app_123",
        user_id="customer_456",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "loan_application_analysis"}
    )
    
    # Create task data
    task_data = {
        "raw_text": sample_loan_application,
        "document_type": "loan_application",
        "metadata": {
            "filename": "loan_application.pdf",
            "state_code": "CA"
        }
    }
    
    # Process the task
    result = await loan_risk_agent.process_task(context, task_data)
    
    # Verify duplicate detection results
    assert result["duplicate_check"]["is_duplicate"] == True
    assert result["duplicate_check"]["similarity_score"] >= 0.9
    assert len(result["duplicate_check"]["similar_documents"]) > 0
    assert "flags" in result
    assert any("duplicate" in flag.lower() for flag in result["flags"])


@pytest.mark.asyncio
async def test_rule_execution_integration(loan_risk_agent, sample_loan_application):
    """Test rule execution integration in LoanRiskAgent."""
    # Mock the rule executor to return rule violations
    loan_risk_agent.rule_executor.execute_rules.return_value = {
        "credit_risk_score": 60,
        "income_stability_score": 50,
        "debt_burden_score": 45,
        "overall_risk_level": "high",
        "approval_recommendation": "reject",
        "rule_violations": [
            "Credit score below minimum threshold",
            "Debt-to-income ratio exceeds maximum allowed"
        ],
        "flags": [
            "HIGH_RISK_APPLICANT",
            "INSUFFICIENT_INCOME"
        ]
    }
    
    # Create agent context
    context = AgentContext(
        doc_id="high_risk_loan_app_123",
        user_id="customer_789",
        correlation_id=str(uuid.uuid4()),
        model_hint=ModelHint.STANDARD,
        start_time=time.time(),
        metadata={"task_type": "loan_application_analysis"}
    )
    
    # Create task data
    task_data = {
        "raw_text": sample_loan_application,
        "document_type": "loan_application",
        "metadata": {
            "filename": "loan_application.pdf",
            "state_code": "CA"
        }
    }
    
    # Process the task
    result = await loan_risk_agent.process_task(context, task_data)
    
    # Verify rule execution results
    assert result["risk_assessment"]["overall_risk_level"] == "high"
    assert result["risk_assessment"]["approval_recommendation"] == "reject"
    assert len(result["rule_violations"]) > 0
    assert len(result["flags"]) > 0
    assert "HIGH_RISK_APPLICANT" in result["flags"]
