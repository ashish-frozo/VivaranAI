"""
Loan Risk Validator Tool - Pack-driven loan document validation.

This tool provides loan-specific validation capabilities using external
rule packs instead of hard-coded logic.
"""

from __future__ import annotations

import asyncio
import structlog
from typing import Dict, Any, List, Optional

from agents.base.tools import BaseTool
from agents.base.validators import BaseRateValidator, ValidationDelta
from packs import get_pack_loader

logger = structlog.get_logger(__name__)


class LoanRiskValidatorTool(BaseTool):
    """
    Loan Risk Validator Tool using pack-driven architecture.
    
    Provides loan document validation capabilities using external rule packs.
    """
    
    def __init__(self, tool_id: str = "loan_risk_validator_tool"):
        """Initialize the loan risk validator tool."""
        super().__init__(
            tool_id=tool_id,
            tool_name="Loan Risk Validator Tool",
            tool_version="1.0.0",
            description="Validates loan documents against pack-driven rules for risk assessment"
        )
        self.validator: Optional[LoanRiskValidator] = None
        
    async def initialize(self) -> bool:
        """Initialize the loan risk validator tool."""
        try:
            # Import here to avoid circular imports
            from agents.verticals.loan.loan_risk_agent import LoanRiskValidator
            
            self.validator = LoanRiskValidator()
            success = await self.validator.initialize()
            
            if success:
                logger.info("Loan risk validator tool initialized successfully")
                return True
            else:
                logger.error("Failed to initialize loan risk validator tool")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize loan risk validator tool: {str(e)}")
            return False
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute loan risk validation.
        
        Args:
            **kwargs: Validation parameters including:
                - items: List of loan items to validate
                - validation_type: Type of validation to perform
                
        Returns:
            Validation results with deltas and summary
        """
        try:
            if not self.validator or not self.validator.is_initialized():
                return {
                    "success": False,
                    "error": "Loan risk validator not initialized",
                    "deltas": [],
                    "summary": {}
                }
            
            items = kwargs.get('items', [])
            validation_type = kwargs.get('validation_type', 'full')
            
            if not items:
                return {
                    "success": False,
                    "error": "No items provided for validation",
                    "deltas": [],
                    "summary": {}
                }
            
            # Perform validation
            deltas = await self.validator.validate(items, **kwargs)
            
            # Generate summary
            summary = self._generate_validation_summary(items, deltas)
            
            result = {
                "success": True,
                "validation_type": validation_type,
                "deltas": [delta.to_dict() for delta in deltas],
                "summary": summary,
                "pack_id": "loan"
            }
            
            logger.info(
                "Loan risk validation completed",
                items_validated=len(items),
                deltas_found=len(deltas),
                validation_type=validation_type
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Loan risk validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "deltas": [],
                "summary": {}
            }
    
    def _generate_validation_summary(self, items: List[Dict[str, Any]], deltas: List[ValidationDelta]) -> Dict[str, Any]:
        """Generate validation summary."""
        total_items = len(items)
        total_deltas = len(deltas)
        
        # Group deltas by violation type
        violation_counts = {}
        for delta in deltas:
            violation_type = delta.violation_type
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        # Calculate risk metrics
        total_amount = sum(float(item.get('amount', item.get('total_amount', 0))) for item in items)
        risk_amount = sum(delta.item_amount for delta in deltas if delta.violation_type in ['overcharge', 'prohibited'])
        
        return {
            "total_items": total_items,
            "items_flagged": total_deltas,
            "flag_percentage": (total_deltas / total_items * 100) if total_items > 0 else 0,
            "violation_counts": violation_counts,
            "total_amount": total_amount,
            "risk_amount": risk_amount,
            "risk_percentage": (risk_amount / total_amount * 100) if total_amount > 0 else 0,
            "high_risk_items": len([d for d in deltas if d.severity == "high"]),
            "medium_risk_items": len([d for d in deltas if d.severity == "medium"]),
            "low_risk_items": len([d for d in deltas if d.severity == "low"])
        }
    
    async def validate_loan_terms(self, loan_items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Validate loan terms and conditions.
        
        Args:
            loan_items: List of loan items to validate
            **kwargs: Additional validation parameters
            
        Returns:
            Validation results
        """
        return await self.execute(items=loan_items, validation_type="loan_terms", **kwargs)
    
    async def assess_risk_factors(self, loan_items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Assess risk factors in loan document.
        
        Args:
            loan_items: List of loan items to assess
            **kwargs: Additional assessment parameters
            
        Returns:
            Risk assessment results
        """
        return await self.execute(items=loan_items, validation_type="risk_assessment", **kwargs)
    
    async def detect_prohibited_charges(self, loan_items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Detect prohibited or undisclosed charges.
        
        Args:
            loan_items: List of loan items to check
            **kwargs: Additional detection parameters
            
        Returns:
            Detection results
        """
        return await self.execute(items=loan_items, validation_type="prohibited_charges", **kwargs)
    
    async def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return [
            "validate_loan_terms",
            "assess_risk_factors",
            "detect_prohibited_charges",
            "validate_interest_rates",
            "check_processing_fees"
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            is_healthy = (
                self.validator is not None and 
                self.validator.is_initialized()
            )
            
            return {
                "healthy": is_healthy,
                "tool_id": self.tool_id,
                "validator_initialized": self.validator.is_initialized() if self.validator else False,
                "pack_id": "loan"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
