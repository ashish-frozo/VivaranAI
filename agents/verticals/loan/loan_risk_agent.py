"""
Loan Risk Agent - MVP stub using pack-driven architecture.

This agent provides loan document analysis and risk assessment using
external rule packs instead of hard-coded logic.
"""

from __future__ import annotations

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agents.base_agent import BaseAgent
from agents.interfaces import AgentContext
from agents.base.validators import BaseRateValidator, ValidationDelta
from packs import get_pack_loader
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TaskData:
    """Task data for agent processing."""
    task_type: str
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

logger = structlog.get_logger(__name__)


class LoanRiskValidator(BaseRateValidator):
    """
    Loan-specific risk validator using pack-driven architecture.
    
    Validates loan documents against loan pack rules for risk assessment.
    """
    
    def __init__(self, pack_loader=None):
        """Initialize loan risk validator."""
        super().__init__(pack_id="loan", pack_loader=pack_loader)
        
    async def initialize(self) -> bool:
        """Initialize the loan validator with pack data."""
        return await self.load_pack("loan")
    
    async def validate(self, items: List[Dict[str, Any]], **kwargs) -> List[ValidationDelta]:
        """
        Validate loan document items against pack rules.
        
        Args:
            items: List of loan document items to validate
            **kwargs: Additional validation parameters
            
        Returns:
            List of validation deltas/findings
        """
        if not self.is_initialized():
            logger.error("Loan validator not initialized")
            return []
        
        all_deltas = []
        
        try:
            # 1. Detect duplicates
            duplicate_deltas = await self.detect_duplicates(items)
            all_deltas.extend(duplicate_deltas)
            
            # 2. Apply regex rules from pack
            regex_deltas = await self.apply_regex_rules(items)
            all_deltas.extend(regex_deltas)
            
            # 3. Check for prohibited items
            prohibited_deltas = await self._check_prohibited_items(items)
            all_deltas.extend(prohibited_deltas)
            
            logger.info(f"Loan validation completed: {len(all_deltas)} findings")
            return all_deltas
            
        except Exception as e:
            logger.error(f"Loan validation failed: {str(e)}")
            return []
    
    async def _check_prohibited_items(self, items: List[Dict[str, Any]]) -> List[ValidationDelta]:
        """Check for prohibited items from pack configuration."""
        deltas = []
        prohibited_items = self.pack_config.get('prohibited_items', [])
        
        for item in items:
            description = item.get('description', '').lower().strip()
            amount = float(item.get('amount', item.get('total_amount', 0)))
            
            for prohibited in prohibited_items:
                if prohibited.lower() in description:
                    delta = ValidationDelta(
                        item_description=description,
                        item_amount=amount,
                        violation_type="prohibited",
                        severity="high",
                        rule_source="pack_loan",
                        confidence=0.95,
                        metadata={
                            "prohibited_pattern": prohibited
                        }
                    )
                    deltas.append(delta)
                    break
        
        return deltas
    
    def _create_duplicate_key(self, item: Dict[str, Any]) -> str:
        """Create a key for duplicate detection specific to loan items."""
        description = item.get('description', '').lower().strip()
        amount = item.get('amount', item.get('total_amount', 0))
        
        # Map entity for better duplicate detection
        mapped_entity = self.map_entity(description)
        
        return f"{mapped_entity}_{amount}"


class LoanRiskAgent(BaseAgent):
    """
    Loan Risk Agent - MVP stub for loan document analysis.
    
    Accepts PDF documents, runs duplicate detection and loan pack rules,
    returns structured JSON identical to medical agent schema.
    """
    
    def __init__(self, agent_id: str = "loan_risk_agent"):
        """Initialize the loan risk agent."""
        super().__init__(
            agent_id=agent_id,
            agent_name="Loan Risk Agent",
            agent_version="1.0.0",
            supported_document_types=["loan_document", "loan_agreement", "loan_application"]
        )
        self.validator: Optional[LoanRiskValidator] = None
        
    async def initialize(self) -> bool:
        """Initialize the loan risk agent."""
        try:
            self.validator = LoanRiskValidator()
            success = await self.validator.initialize()
            
            if success:
                logger.info("Loan risk agent initialized successfully")
                return True
            else:
                logger.error("Failed to initialize loan risk agent")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize loan risk agent: {str(e)}")
            return False
    
    async def process_task(self, task_data: TaskData, context: AgentContext) -> Dict[str, Any]:
        """
        Process loan document analysis task.
        
        Args:
            task_data: Task data containing document information
            context: Agent execution context
            
        Returns:
            Structured analysis result identical to medical agent schema
        """
        try:
            if not self.validator or not self.validator.is_initialized():
                return {
                    "success": False,
                    "error": "Loan risk validator not initialized",
                    "analysis_type": "loan_risk",
                    "confidence": 0.0,
                    "red_flags": [],
                    "recommendations": []
                }
            
            # Extract line items from task data
            line_items = []
            if hasattr(task_data, 'line_items') and task_data.line_items:
                line_items = task_data.line_items
            elif hasattr(task_data, 'raw_text') and task_data.raw_text:
                # Simple extraction from raw text for MVP
                line_items = await self._extract_loan_items_from_text(task_data.raw_text)
            
            if not line_items:
                return {
                    "success": False,
                    "error": "No loan items found for analysis",
                    "analysis_type": "loan_risk",
                    "confidence": 0.0,
                    "red_flags": [],
                    "recommendations": []
                }
            
            # Validate using pack-driven validator
            deltas = await self.validator.validate(line_items)
            
            # Convert deltas to red flags
            red_flags = []
            for delta in deltas:
                red_flag = {
                    "type": delta.violation_type,
                    "severity": delta.severity,
                    "item": delta.item_description,
                    "amount": delta.item_amount,
                    "reference_amount": delta.reference_amount,
                    "excess_amount": delta.delta_amount,
                    "confidence": delta.confidence,
                    "source": delta.rule_source,
                    "message": self._generate_red_flag_message(delta)
                }
                red_flags.append(red_flag)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(line_items, deltas)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(deltas)
            
            # Calculate totals
            total_amount = sum(float(item.get('amount', item.get('total_amount', 0))) for item in line_items)
            total_risk_amount = sum(delta.item_amount for delta in deltas if delta.violation_type in ['overcharge', 'prohibited'])
            
            result = {
                "success": True,
                "analysis_type": "loan_risk",
                "confidence": confidence,
                "total_amount": total_amount,
                "total_risk_amount": total_risk_amount,
                "line_items": line_items,
                "red_flags": red_flags,
                "recommendations": recommendations,
                "validation_summary": {
                    "total_items": len(line_items),
                    "items_flagged": len(red_flags),
                    "risk_percentage": (total_risk_amount / total_amount * 100) if total_amount > 0 else 0,
                    "violation_types": list(set(delta.violation_type for delta in deltas))
                },
                "pack_driven": True,
                "pack_id": "loan"
            }
            
            logger.info(
                "Loan risk analysis completed",
                total_items=len(line_items),
                red_flags_found=len(red_flags),
                confidence=confidence,
                total_risk_amount=total_risk_amount
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Loan risk analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "analysis_type": "loan_risk",
                "confidence": 0.0,
                "red_flags": [],
                "recommendations": []
            }
    
    async def _extract_loan_items_from_text(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Extract loan items from raw text (simple MVP implementation).
        
        Args:
            raw_text: Raw text from document
            
        Returns:
            List of extracted loan items
        """
        items = []
        
        # Simple regex-based extraction for MVP
        import re
        
        # Look for interest rate patterns
        interest_patterns = [
            r'interest\s+rate[:\s]+(\d+(?:\.\d+)?)\s*%',
            r'rate\s+of\s+interest[:\s]+(\d+(?:\.\d+)?)\s*%',
            r'interest[:\s]+(\d+(?:\.\d+)?)\s*%\s*per\s+annum'
        ]
        
        for pattern in interest_patterns:
            matches = re.finditer(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                rate = float(match.group(1))
                items.append({
                    "description": "interest rate",
                    "amount": rate,
                    "item_type": "rate",
                    "unit": "percentage"
                })
        
        # Look for processing fee patterns
        fee_patterns = [
            r'processing\s+fee[:\s]+₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'processing\s+charge[:\s]+₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        for pattern in fee_patterns:
            matches = re.finditer(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                amount = float(match.group(1).replace(',', ''))
                items.append({
                    "description": "processing fee",
                    "amount": amount,
                    "item_type": "fee",
                    "unit": "currency"
                })
        
        # Look for loan amount patterns
        loan_patterns = [
            r'loan\s+amount[:\s]+₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)',
            r'principal[:\s]+₹?\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        ]
        
        for pattern in loan_patterns:
            matches = re.finditer(pattern, raw_text, re.IGNORECASE)
            for match in matches:
                amount = float(match.group(1).replace(',', ''))
                items.append({
                    "description": "loan amount",
                    "amount": amount,
                    "item_type": "principal",
                    "unit": "currency"
                })
        
        logger.info(f"Extracted {len(items)} loan items from text")
        return items
    
    def _generate_red_flag_message(self, delta: ValidationDelta) -> str:
        """Generate human-readable red flag message."""
        item = delta.item_description
        amount = delta.item_amount
        violation_type = delta.violation_type
        
        if violation_type == "overcharge":
            if "interest" in item.lower():
                return f"High interest rate detected: {amount}% (exceeds recommended maximum)"
            else:
                reference = delta.reference_amount
                excess = delta.delta_amount
                return f"Excessive charge for {item}: ₹{amount} vs ₹{reference} maximum (excess: ₹{excess})"
        elif violation_type == "duplicate":
            return f"Duplicate entry detected: {item} ({amount})"
        elif violation_type == "prohibited":
            return f"Prohibited/undisclosed charge found: {item} ({amount})"
        else:
            return f"Risk factor identified for {item}: {violation_type} ({amount})"
    
    def _calculate_confidence(self, line_items: List[Dict[str, Any]], deltas: List[ValidationDelta]) -> float:
        """Calculate confidence score for the analysis."""
        if not line_items:
            return 0.0
        
        # Base confidence
        base_confidence = 0.7
        
        # Reduce confidence based on number of items analyzed
        item_factor = min(len(line_items) / 10, 1.0) * 0.2
        
        # Increase confidence based on validation findings
        finding_factor = min(len(deltas) / len(line_items), 0.5) * 0.1
        
        return min(base_confidence + item_factor + finding_factor, 1.0)
    
    def _generate_recommendations(self, deltas: List[ValidationDelta]) -> List[str]:
        """Generate recommendations based on validation findings."""
        recommendations = []
        
        # Check for high interest rates
        high_interest = [d for d in deltas if d.violation_type == "overcharge" and "interest" in d.item_description.lower()]
        if high_interest:
            recommendations.append("Consider negotiating a lower interest rate or exploring alternative lenders")
        
        # Check for excessive processing fees
        high_fees = [d for d in deltas if d.violation_type == "overcharge" and "fee" in d.item_description.lower()]
        if high_fees:
            recommendations.append("Review processing fees and compare with industry standards")
        
        # Check for prohibited items
        prohibited = [d for d in deltas if d.violation_type == "prohibited"]
        if prohibited:
            recommendations.append("Request clarification on undisclosed charges and fees")
        
        # Check for duplicates
        duplicates = [d for d in deltas if d.violation_type == "duplicate"]
        if duplicates:
            recommendations.append("Verify and remove duplicate charges from the loan agreement")
        
        # General recommendations
        if deltas:
            recommendations.append("Review all loan terms carefully before signing")
            recommendations.append("Consider seeking financial advice for loan optimization")
        
        return recommendations
    
    async def get_supported_operations(self) -> List[str]:
        """Get list of supported operations."""
        return [
            "analyze_loan_document",
            "validate_loan_terms",
            "assess_risk_factors",
            "generate_recommendations"
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
                "agent_id": self.agent_id,
                "validator_initialized": self.validator.is_initialized() if self.validator else False,
                "pack_id": "loan"
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
