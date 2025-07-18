"""
Pack-Driven Architecture Tests.

Tests the multi-vertical, pack-driven architecture including:
- Rule pack loading and validation
- YAML schema validation
- Rule execution across verticals
- Integration with agent workflows
"""

import os
import yaml
import pytest
from unittest.mock import patch, mock_open

from packs import load_rule_pack, validate_rule_pack_schema
from packs.loan.rule_loader import LoanRuleLoader
from packs.loan.rule_executor import LoanRuleExecutor
from packs.medical.rule_loader import MedicalRuleLoader
from packs.medical.rule_executor import MedicalRuleExecutor


@pytest.fixture
def sample_loan_rule_pack():
    """Sample loan rule pack YAML content."""
    return """
    pack_id: loan_risk_assessment
    version: 1.0.0
    description: Loan risk assessment rules
    vertical: loan
    rules:
      credit_score_rules:
        - id: high_credit_score
          description: High credit score indicates low risk
          condition: credit_score >= 720
          risk_level: low
          confidence: 0.9
          
        - id: medium_credit_score
          description: Medium credit score indicates moderate risk
          condition: credit_score >= 650 and credit_score < 720
          risk_level: medium
          confidence: 0.8
          
        - id: low_credit_score
          description: Low credit score indicates high risk
          condition: credit_score < 650
          risk_level: high
          confidence: 0.85
          
      debt_to_income_rules:
        - id: low_dti_ratio
          description: Low debt-to-income ratio indicates low risk
          condition: debt_to_income_ratio <= 0.36
          risk_level: low
          confidence: 0.9
          
        - id: medium_dti_ratio
          description: Medium debt-to-income ratio indicates moderate risk
          condition: debt_to_income_ratio > 0.36 and debt_to_income_ratio <= 0.43
          risk_level: medium
          confidence: 0.8
          
        - id: high_dti_ratio
          description: High debt-to-income ratio indicates high risk
          condition: debt_to_income_ratio > 0.43
          risk_level: high
          confidence: 0.85
          
      loan_term_rules:
        - id: short_term_loan
          description: Short-term loans have lower risk
          condition: loan_term_months <= 36
          risk_level: low
          confidence: 0.75
          
        - id: medium_term_loan
          description: Medium-term loans have moderate risk
          condition: loan_term_months > 36 and loan_term_months <= 60
          risk_level: medium
          confidence: 0.7
          
        - id: long_term_loan
          description: Long-term loans have higher risk
          condition: loan_term_months > 60
          risk_level: high
          confidence: 0.8
    """


@pytest.fixture
def sample_medical_rule_pack():
    """Sample medical rule pack YAML content."""
    return """
    pack_id: medical_bill_analysis
    version: 1.0.0
    description: Medical bill analysis rules
    vertical: medical
    rules:
      rate_validation_rules:
        - id: cghs_rate_comparison
          description: Compare billed amount with CGHS rates
          condition: billed_amount > cghs_rate * 1.1
          action: flag_overcharge
          confidence: 0.9
          
        - id: esi_rate_comparison
          description: Compare billed amount with ESI rates
          condition: billed_amount > esi_rate * 1.15
          action: flag_overcharge
          confidence: 0.85
          
      prohibited_item_rules:
        - id: non_medical_items
          description: Detect non-medical items in bill
          items:
            - "room service"
            - "television"
            - "telephone"
            - "guest meals"
          action: flag_prohibited
          confidence: 0.95
          
        - id: luxury_items
          description: Detect luxury items in bill
          items:
            - "deluxe room"
            - "vip suite"
            - "premium amenities"
          action: flag_prohibited
          confidence: 0.9
          
      duplicate_detection_rules:
        - id: same_day_procedures
          description: Detect duplicate procedures on same day
          condition: procedure_count > 1 and same_day
          action: flag_duplicate
          confidence: 0.8
    """


@pytest.fixture
def sample_loan_application_data():
    """Sample loan application data for rule execution."""
    return {
        "applicant_name": "John Smith",
        "loan_amount": 25000,
        "loan_purpose": "Home Renovation",
        "loan_term_months": 60,
        "annual_income": 95000,
        "monthly_debt_payments": 500,
        "credit_score": 720,
        "debt_to_income_ratio": 0.063,  # ($500 * 12) / $95,000
    }


@pytest.fixture
def sample_medical_bill_data():
    """Sample medical bill data for rule execution."""
    return {
        "patient_name": "Jane Doe",
        "hospital_name": "City Hospital",
        "bill_date": "2025-06-15",
        "line_items": [
            {
                "item_name": "Consultation",
                "billed_amount": 1500,
                "cghs_rate": 1000,
                "esi_rate": 1200,
                "quantity": 1
            },
            {
                "item_name": "Blood Test",
                "billed_amount": 800,
                "cghs_rate": 600,
                "esi_rate": 700,
                "quantity": 1
            },
            {
                "item_name": "X-Ray",
                "billed_amount": 1200,
                "cghs_rate": 1000,
                "esi_rate": 1100,
                "quantity": 1
            },
            {
                "item_name": "Room Charges",
                "billed_amount": 3000,
                "cghs_rate": 2500,
                "esi_rate": 2800,
                "quantity": 2
            }
        ],
        "total_amount": 6500
    }


def test_rule_pack_schema_validation(sample_loan_rule_pack):
    """Test rule pack schema validation."""
    # Parse YAML content
    rule_pack = yaml.safe_load(sample_loan_rule_pack)
    
    # Validate schema
    with patch('packs.validate_rule_pack_schema') as mock_validate:
        mock_validate.return_value = (True, None)
        
        # Call the function
        is_valid, error = validate_rule_pack_schema(rule_pack)
        
        # Verify validation was called
        mock_validate.assert_called_once()
        
        # Check result
        assert is_valid
        assert error is None


def test_load_rule_pack():
    """Test loading rule packs from files."""
    # Mock the open function to return our sample rule pack
    sample_content = """
    pack_id: test_pack
    version: 1.0.0
    description: Test rule pack
    vertical: test
    rules:
      test_rules:
        - id: test_rule_1
          description: Test rule 1
          condition: value > 10
          action: flag
    """
    
    # Mock file operations
    with patch('builtins.open', mock_open(read_data=sample_content)), \
         patch('os.path.exists', return_value=True), \
         patch('packs.validate_rule_pack_schema', return_value=(True, None)):
        
        # Load the rule pack
        rule_pack = load_rule_pack("test_vertical", "test_pack")
        
        # Verify the loaded rule pack
        assert rule_pack["pack_id"] == "test_pack"
        assert rule_pack["version"] == "1.0.0"
        assert rule_pack["vertical"] == "test"
        assert "rules" in rule_pack
        assert "test_rules" in rule_pack["rules"]
        assert len(rule_pack["rules"]["test_rules"]) == 1
        assert rule_pack["rules"]["test_rules"][0]["id"] == "test_rule_1"


def test_loan_rule_loader(sample_loan_rule_pack):
    """Test loan rule loader functionality."""
    # Mock file operations
    with patch('builtins.open', mock_open(read_data=sample_loan_rule_pack)), \
         patch('os.path.exists', return_value=True), \
         patch('packs.validate_rule_pack_schema', return_value=(True, None)), \
         patch('packs.load_rule_pack', return_value=yaml.safe_load(sample_loan_rule_pack)):
        
        # Create rule loader
        rule_loader = LoanRuleLoader()
        
        # Load rules
        rules = rule_loader.load_rules()
        
        # Verify rules were loaded
        assert "credit_score_rules" in rules
        assert "debt_to_income_rules" in rules
        assert "loan_term_rules" in rules
        
        # Check rule counts
        assert len(rules["credit_score_rules"]) == 3
        assert len(rules["debt_to_income_rules"]) == 3
        assert len(rules["loan_term_rules"]) == 3


def test_medical_rule_loader(sample_medical_rule_pack):
    """Test medical rule loader functionality."""
    # Mock file operations
    with patch('builtins.open', mock_open(read_data=sample_medical_rule_pack)), \
         patch('os.path.exists', return_value=True), \
         patch('packs.validate_rule_pack_schema', return_value=(True, None)), \
         patch('packs.load_rule_pack', return_value=yaml.safe_load(sample_medical_rule_pack)):
        
        # Create rule loader
        rule_loader = MedicalRuleLoader()
        
        # Load rules
        rules = rule_loader.load_rules()
        
        # Verify rules were loaded
        assert "rate_validation_rules" in rules
        assert "prohibited_item_rules" in rules
        assert "duplicate_detection_rules" in rules
        
        # Check rule counts
        assert len(rules["rate_validation_rules"]) == 2
        assert len(rules["prohibited_item_rules"]) == 2
        assert len(rules["duplicate_detection_rules"]) == 1


@pytest.mark.asyncio
async def test_loan_rule_executor(sample_loan_rule_pack, sample_loan_application_data):
    """Test loan rule executor functionality."""
    # Mock rule loader
    mock_rule_loader = LoanRuleLoader()
    mock_rule_loader.load_rules = lambda: {
        "credit_score_rules": [
            {"id": "high_credit_score", "condition": "credit_score >= 720", "risk_level": "low", "confidence": 0.9},
            {"id": "medium_credit_score", "condition": "credit_score >= 650 and credit_score < 720", "risk_level": "medium", "confidence": 0.8},
            {"id": "low_credit_score", "condition": "credit_score < 650", "risk_level": "high", "confidence": 0.85}
        ],
        "debt_to_income_rules": [
            {"id": "low_dti_ratio", "condition": "debt_to_income_ratio <= 0.36", "risk_level": "low", "confidence": 0.9},
            {"id": "medium_dti_ratio", "condition": "debt_to_income_ratio > 0.36 and debt_to_income_ratio <= 0.43", "risk_level": "medium", "confidence": 0.8},
            {"id": "high_dti_ratio", "condition": "debt_to_income_ratio > 0.43", "risk_level": "high", "confidence": 0.85}
        ],
        "loan_term_rules": [
            {"id": "medium_term_loan", "condition": "loan_term_months > 36 and loan_term_months <= 60", "risk_level": "medium", "confidence": 0.7}
        ]
    }
    
    # Create rule executor with mock loader
    rule_executor = LoanRuleExecutor(rule_loader=mock_rule_loader)
    
    # Execute rules
    result = await rule_executor.execute_rules(sample_loan_application_data)
    
    # Verify rule execution results
    assert "credit_risk_score" in result
    assert "debt_burden_score" in result
    assert "overall_risk_level" in result
    assert "approval_recommendation" in result
    
    # Check specific rule outcomes
    assert result["overall_risk_level"] in ["low", "medium", "high"]
    
    # For this specific test data:
    # - Credit score is 720 (high) -> low risk
    # - DTI is 0.063 (low) -> low risk
    # - Loan term is 60 months (medium) -> medium risk
    # Overall should be low or medium risk
    assert result["overall_risk_level"] in ["low", "medium"]
    assert result["approval_recommendation"] in ["approve", "review"]


@pytest.mark.asyncio
async def test_medical_rule_executor(sample_medical_rule_pack, sample_medical_bill_data):
    """Test medical rule executor functionality."""
    # Mock rule loader
    mock_rule_loader = MedicalRuleLoader()
    mock_rule_loader.load_rules = lambda: {
        "rate_validation_rules": [
            {"id": "cghs_rate_comparison", "condition": "billed_amount > cghs_rate * 1.1", "action": "flag_overcharge", "confidence": 0.9}
        ],
        "prohibited_item_rules": [
            {"id": "luxury_items", "items": ["deluxe room", "vip suite", "premium amenities"], "action": "flag_prohibited", "confidence": 0.9}
        ]
    }
    
    # Create rule executor with mock loader
    rule_executor = MedicalRuleExecutor(rule_loader=mock_rule_loader)
    
    # Execute rules
    result = await rule_executor.execute_rules(sample_medical_bill_data)
    
    # Verify rule execution results
    assert "overcharge_items" in result
    assert "prohibited_items" in result
    assert "total_overcharge" in result
    assert "confidence_score" in result
    
    # Check overcharge detection
    # Consultation: billed 1500, CGHS 1000 -> overcharge
    # Blood Test: billed 800, CGHS 600 -> overcharge
    # X-Ray: billed 1200, CGHS 1000 -> overcharge
    assert len(result["overcharge_items"]) > 0
    
    # No prohibited items in our test data
    assert len(result["prohibited_items"]) == 0
    
    # Total overcharge should be positive
    assert result["total_overcharge"] > 0
    
    # Confidence score should be between 0 and 1
    assert 0 <= result["confidence_score"] <= 1


def test_cross_vertical_rule_integration():
    """Test integration of rule packs across verticals."""
    # Test that we can load rule packs from different verticals
    with patch('packs.load_rule_pack') as mock_load:
        # Mock different returns based on vertical
        def side_effect(vertical, pack_id):
            if vertical == "loan":
                return {"pack_id": pack_id, "vertical": "loan", "rules": {}}
            elif vertical == "medical":
                return {"pack_id": pack_id, "vertical": "medical", "rules": {}}
            else:
                return None
        
        mock_load.side_effect = side_effect
        
        # Load rule packs from different verticals
        loan_pack = load_rule_pack("loan", "loan_risk_assessment")
        medical_pack = load_rule_pack("medical", "medical_bill_analysis")
        
        # Verify the correct packs were loaded
        assert loan_pack["vertical"] == "loan"
        assert medical_pack["vertical"] == "medical"
        
        # Verify the function was called with correct parameters
        mock_load.assert_any_call("loan", "loan_risk_assessment")
        mock_load.assert_any_call("medical", "medical_bill_analysis")
