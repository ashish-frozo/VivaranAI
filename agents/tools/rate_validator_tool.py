"""
Rate Validator Tool - Async wrapper for RateValidator.

This tool wraps the existing RateValidator component to make it compatible
with the OpenAI Agent SDK framework for medical bill rate validation.
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from medbillguardagent.rate_validator import RateValidator

logger = structlog.get_logger(__name__)


class RateValidatorInput(BaseModel):
    """Input schema for rate validator tool."""
    line_items: List[Dict[str, Any]] = Field(..., description="List of extracted line items")
    state_code: Optional[str] = Field(default=None, description="State code for regional rates")
    validation_sources: List[str] = Field(default=["cghs", "esi"], description="Validation sources to use")


class RateValidatorTool:
    """
    Async tool wrapper for RateValidator.
    
    Provides rate validation capabilities including:
    - CGHS rate validation
    - ESI rate validation  
    - NPPA price validation
    - State-specific rate validation
    - Overcharge detection and red flag generation
    """
    
    def __init__(self, reference_data_loader=None):
        """Initialize the rate validator tool."""
        self.validator = RateValidator(reference_loader=reference_data_loader)
        logger.info("Initialized RateValidatorTool")
    
    async def __call__(
        self,
        line_items: List[Dict[str, Any]],
        state_code: Optional[str] = None,
        validation_sources: List[str] = None,
        dynamic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate medical charges against reference rates.
        
        Args:
            line_items: List of extracted line items with descriptions and amounts
            state_code: Optional state code for regional rate validation
            validation_sources: List of validation sources to use
            
        Returns:
            Dict containing validation results and red flags
        """
        try:
            logger.info(
                "Starting rate validation",
                item_count=len(line_items),
                state_code=state_code,
                sources=validation_sources or ["cghs", "esi"]
            )
            
            if not line_items:
                return {
                    "success": True,
                    "rate_matches": [],
                    "red_flags": [],
                    "validation_summary": {
                        "total_items": 0,
                        "items_validated": 0,
                        "total_billed": 0.0,
                        "total_allowed": 0.0,
                        "total_overcharge": 0.0,
                        "overcharge_percentage": 0.0,
                        "violation_count": 0
                    }
                }
            
            # Prepare items and costs for validation
            items = [item.get("description", "") for item in line_items]
            item_costs = {
                item.get("description", ""): float(item.get("total_amount", 0)) 
                for item in line_items
                if item.get("description") and item.get("total_amount")
            }
            
            # Validate rates
            rate_matches = await self.validator.validate_item_rates(
                items, item_costs, state_code=state_code, dynamic_data=dynamic_data
            )
            
            # Generate red flags from rate matches
            red_flags = self.validator.generate_red_flags(rate_matches)
            
            # Calculate summary statistics
            total_billed = sum(float(item.get("total_amount", 0)) for item in line_items)
            total_overcharge = sum(
                flag.get("overcharge_amount", 0) 
                for flag in red_flags 
                if flag.get("type") == "overcharge"
            )
            total_allowed = total_billed - total_overcharge
            overcharge_percentage = (total_overcharge / total_billed * 100) if total_billed > 0 else 0
            
            # Convert rate matches to serializable format
            serialized_matches = []
            for match in rate_matches:
                serialized_matches.append({
                    "bill_item": match.bill_item,
                    "reference_item": match.reference_item,
                    "billed_amount": float(match.billed_amount),
                    "reference_rate": float(match.reference_rate),
                    "overcharge_amount": float(match.overcharge_amount),
                    "overcharge_percentage": float(match.overcharge_percentage),
                    "source": match.source.value,
                    "confidence": float(match.confidence),
                    "item_type": match.item_type.value,
                    "match_method": match.match_method,
                    "state_code": match.state_code
                })
            
            result = {
                "success": True,
                "rate_matches": serialized_matches,
                "red_flags": red_flags,
                "validation_summary": {
                    "total_items": len(line_items),
                    "items_validated": len(rate_matches),
                    "total_billed": float(total_billed),
                    "total_allowed": float(total_allowed),
                    "total_overcharge": float(total_overcharge),
                    "overcharge_percentage": float(overcharge_percentage),
                    "violation_count": len([f for f in red_flags if f.get("type") == "overcharge"])
                },
                "validation_sources_used": validation_sources or ["cghs", "esi"],
                "state_validation": state_code is not None
            }
            
            logger.info(
                "Rate validation completed",
                items_validated=len(rate_matches),
                red_flags_found=len(red_flags),
                total_overcharge=total_overcharge,
                overcharge_percentage=overcharge_percentage
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Rate validation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "rate_matches": [],
                "red_flags": [],
                "validation_summary": {
                    "total_items": len(line_items) if line_items else 0,
                    "items_validated": 0,
                    "total_billed": 0.0,
                    "total_allowed": 0.0,
                    "total_overcharge": 0.0,
                    "overcharge_percentage": 0.0,
                    "violation_count": 0
                }
            }


# Tool schema for OpenAI Agent SDK
RateValidatorTool._tool_schema = {
    "type": "function",
    "function": {
        "name": "validate_rates",
        "description": "Validate medical charges against CGHS, ESI, NPPA and state reference rates to detect overcharges",
        "parameters": {
            "type": "object",
            "properties": {
                "line_items": {
                    "type": "array",
                    "description": "List of medical bill line items to validate",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Item description or service name"
                            },
                            "total_amount": {
                                "type": "number",
                                "description": "Total charged amount for this item"
                            },
                            "quantity": {
                                "type": "number",
                                "description": "Quantity of the item",
                                "default": 1
                            },
                            "item_type": {
                                "type": "string",
                                "description": "Type of medical item (consultation, diagnostic, medication, etc.)"
                            }
                        },
                        "required": ["description", "total_amount"]
                    }
                },
                "state_code": {
                    "type": "string",
                    "description": "Two-letter state code for regional rate validation (e.g., 'DL', 'MH')",
                    "pattern": "^[A-Z]{2}$"
                },
                "validation_sources": {
                    "type": "array",
                    "description": "List of validation sources to use",
                    "items": {
                        "type": "string",
                        "enum": ["cghs", "esi", "nppa", "state"]
                    },
                    "default": ["cghs", "esi"]
                }
            },
            "required": ["line_items"]
        }
    }
} 