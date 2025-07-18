"""
Golden fixtures for regression testing.

This file contains expected outputs for various test scenarios to ensure
that changes to the codebase don't break existing functionality.
"""

# Medical bill analysis golden fixture
MEDICAL_BILL_GOLDEN_RESULT = {
    "document_type": "medical_bill",
    "hospital_name": "Apollo Hospital",
    "patient_name": "John Doe",
    "bill_date": "2025-06-10",
    "bill_number": "APOL-2025-12345",
    "total_amount": 15000.0,
    "line_items": [
        {
            "item": "Consultation",
            "quantity": 1,
            "unit_price": 1500.0,
            "amount": 1500.0,
            "category": "professional_fee"
        },
        {
            "item": "Blood Test - CBC",
            "quantity": 1,
            "unit_price": 800.0,
            "amount": 800.0,
            "category": "laboratory"
        },
        {
            "item": "X-Ray Chest",
            "quantity": 1,
            "unit_price": 1200.0,
            "amount": 1200.0,
            "category": "radiology"
        },
        {
            "item": "Room Charges (General Ward)",
            "quantity": 3,
            "unit_price": 3000.0,
            "amount": 9000.0,
            "category": "room_charges"
        },
        {
            "item": "Medicines",
            "quantity": 1,
            "unit_price": 2500.0,
            "amount": 2500.0,
            "category": "pharmacy"
        }
    ],
    "analysis_results": {
        "overcharge_items": [
            {
                "item": "Consultation",
                "billed_amount": 1500.0,
                "reference_rate": 1000.0,
                "overcharge_amount": 500.0,
                "overcharge_percentage": 50.0,
                "reference_source": "CGHS Delhi 2023"
            },
            {
                "item": "X-Ray Chest",
                "billed_amount": 1200.0,
                "reference_rate": 800.0,
                "overcharge_amount": 400.0,
                "overcharge_percentage": 50.0,
                "reference_source": "CGHS Delhi 2023"
            }
        ],
        "total_overcharge": 900.0,
        "overcharge_percentage": 6.0,
        "prohibited_items": [],
        "duplicate_items": [],
        "confidence_score": 0.92,
        "recommendation": "Dispute the overcharged items"
    },
    "metadata": {
        "state_code": "DL",
        "insurance_type": "CGHS",
        "processing_time_ms": 3500,
        "model_used": "gpt-4o",
        "version": "1.0.0"
    }
}

# Loan application analysis golden fixture
LOAN_APPLICATION_GOLDEN_RESULT = {
    "document_type": "loan_application",
    "applicant_name": "John Smith",
    "loan_amount": 25000.0,
    "loan_purpose": "Home Renovation",
    "loan_term_months": 60,
    "annual_income": 95000.0,
    "credit_score": 720,
    "debt_to_income_ratio": 0.063,
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
    "flags": [],
    "metadata": {
        "state_code": "CA",
        "processing_time_ms": 2800,
        "model_used": "gpt-4o",
        "version": "1.0.0"
    }
}

# Loan agreement analysis golden fixture
LOAN_AGREEMENT_GOLDEN_RESULT = {
    "document_type": "loan_agreement",
    "lender_name": "First National Bank",
    "borrower_name": "John Smith",
    "principal_amount": 25000.0,
    "interest_rate": 5.75,
    "term_months": 60,
    "monthly_payment": 480.25,
    "start_date": "2025-08-01",
    "end_date": "2030-07-01",
    "compliance_check": {
        "governing_law": "California",
        "has_prepayment_penalty": False,
        "has_default_clause": True,
        "default_grace_period_days": 15,
        "compliant": True
    },
    "rule_violations": [],
    "flags": [],
    "metadata": {
        "state_code": "CA",
        "processing_time_ms": 2500,
        "model_used": "gpt-4o",
        "version": "1.0.0"
    }
}

# Router agent routing decision golden fixture
ROUTING_DECISION_GOLDEN_RESULT = {
    "success": True,
    "document_type": "medical_bill",
    "selected_agents": [
        {
            "agent_id": "medical_bill_agent",
            "name": "Medical Bill Analysis Agent",
            "confidence": 0.95
        }
    ],
    "routing_strategy": "capability_based",
    "confidence": 0.95,
    "estimated_cost_rupees": 3.5,
    "estimated_time_ms": 3000,
    "metadata": {
        "state_code": "DL",
        "insurance_type": "CGHS"
    }
}

# Router agent health check golden fixture
ROUTER_HEALTH_GOLDEN_RESULT = {
    "status": "healthy",
    "timestamp": 1721350000.0,  # Example timestamp
    "ttl_seconds": 300,
    "agent_registry_status": {
        "healthy": True,
        "registered_agents": 2,
        "active_agents": 2
    },
    "active_workflows": 0,
    "issues": []
}

# Cross-vertical workflow golden fixture
CROSS_VERTICAL_WORKFLOW_GOLDEN_RESULT = {
    "success": True,
    "workflow_id": "cross_vertical_analysis",
    "workflow_type": "sequential",
    "step_results": {
        "step_1_medical": {
            "success": True,
            "agent_id": "medical_bill_agent",
            "data": {
                "document_type": "medical_bill",
                "total_amount": 5000.0,
                "overcharge_amount": 200.0
            },
            "execution_time_ms": 2000,
            "cost_rupees": 3.5,
            "confidence": 0.95
        },
        "step_2_loan": {
            "success": True,
            "agent_id": "loan_risk_agent",
            "data": {
                "document_type": "loan_application",
                "loan_amount": 25000.0,
                "risk_assessment": {
                    "overall_risk_level": "low",
                    "approval_recommendation": "approve"
                }
            },
            "execution_time_ms": 1500,
            "cost_rupees": 2.8,
            "confidence": 0.92
        }
    },
    "total_cost_rupees": 6.3,
    "total_execution_time_ms": 3500,
    "metadata": {
        "workflow_type": "cross_vertical_analysis"
    }
}
