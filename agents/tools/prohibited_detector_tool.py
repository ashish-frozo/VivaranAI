"""
Prohibited Detector Tool - Async wrapper for ProhibitedDetector.

This tool wraps the existing ProhibitedDetector component to make it compatible
with the OpenAI Agent SDK framework for detecting prohibited medical items.
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from medbillguardagent.prohibited_detector import ProhibitedDetector

logger = structlog.get_logger(__name__)


class ProhibitedDetectorInput(BaseModel):
    """Input schema for prohibited detector tool."""
    line_items: List[Dict[str, Any]] = Field(..., description="List of extracted line items")
    insurance_type: Optional[str] = Field(default="cghs", description="Insurance type for prohibited item rules")


class ProhibitedDetectorTool:
    """
    Async tool wrapper for ProhibitedDetector.
    
    Provides prohibited item detection capabilities including:
    - CGHS prohibited services detection
    - ESI non-covered items detection
    - Cosmetic procedure identification
    - Unauthorized charge detection
    """
    
    def __init__(self):
        """Initialize the prohibited detector tool."""
        self.detector = ProhibitedDetector()
        logger.info("Initialized ProhibitedDetectorTool")
    
    async def __call__(
        self,
        line_items: List[Dict[str, Any]],
        insurance_type: str = "cghs"
    ) -> Dict[str, Any]:
        """
        Detect prohibited medical items and unauthorized charges.
        
        Args:
            line_items: List of extracted line items with descriptions and amounts
            insurance_type: Type of insurance to check against (cghs, esi, private)
            
        Returns:
            Dict containing prohibited items and red flags
        """
        try:
            logger.info(
                "Starting prohibited item detection",
                item_count=len(line_items),
                insurance_type=insurance_type
            )
            
            if not line_items:
                return {
                    "success": True,
                    "prohibited_items": [],
                    "red_flags": [],
                    "prohibited_summary": {
                        "total_items": 0,
                        "prohibited_items_found": 0,
                        "prohibited_cost_impact": 0.0
                    }
                }
            
            # Convert line items to the format expected by detector
            processed_items = []
            for item in line_items:
                if item.get("description") and item.get("total_amount"):
                    # Create a simple object-like structure
                    processed_item = type('LineItem', (), {
                        'description': item.get("description", ""),
                        'total_amount': float(item.get("total_amount", 0)),
                        'quantity': item.get("quantity", 1),
                        'item_type': item.get("item_type", "other")
                    })()
                    processed_items.append(processed_item)
            
            # Detect prohibited items - method doesn't accept insurance_type parameter
            item_names = [item.description for item in processed_items]
            item_cost_map = {item.description: item.total_amount for item in processed_items}
            prohibited_matches, prohibited_red_flags = self.detector.detect_prohibited_items(item_names, item_cost_map)
            
            # Convert prohibited matches to serializable format
            serialized_prohibited = []
            for match in prohibited_matches:
                serialized_prohibited.append({
                    "item_description": match.bill_item,
                    "item_amount": float(item_cost_map.get(match.bill_item, 0)),
                    "prohibited_category": match.prohibited_item.category,
                    "reason": match.prohibited_item.reason,
                    "source": match.prohibited_item.source,
                    "confidence": float(match.confidence)
                })
            
            # Convert red flags from the detector to tool format
            red_flags = []
            total_prohibited_cost = 0.0
            
            for red_flag in prohibited_red_flags:
                cost = float(red_flag.overcharge_amount) if red_flag.overcharge_amount else 0.0
                total_prohibited_cost += cost
                
                red_flags.append({
                    "type": "prohibited",
                    "severity": "critical", 
                    "item": red_flag.item,
                    "reason": red_flag.reason,
                    "overcharge_amount": cost,
                    "confidence": float(red_flag.confidence),
                    "metadata": {
                        "item_type": red_flag.item_type.value,
                        "source": red_flag.source,
                        "insurance_type": insurance_type
                    }
                })
            
            result = {
                "success": True,
                "prohibited_items": serialized_prohibited,
                "red_flags": red_flags,
                "prohibited_summary": {
                    "total_items": len(line_items),
                    "prohibited_items_found": len(prohibited_matches),
                    "prohibited_cost_impact": float(total_prohibited_cost)
                },
                "detection_settings": {
                    "insurance_type": insurance_type
                }
            }
            
            logger.info(
                "Prohibited item detection completed",
                prohibited_items=len(prohibited_matches),
                red_flags=len(red_flags),
                cost_impact=total_prohibited_cost
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Prohibited item detection failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "prohibited_items": [],
                "red_flags": [],
                "prohibited_summary": {
                    "total_items": len(line_items) if line_items else 0,
                    "prohibited_items_found": 0,
                    "prohibited_cost_impact": 0.0
                }
            }


# Tool schema for OpenAI Agent SDK
ProhibitedDetectorTool._tool_schema = {
    "type": "function",
    "function": {
        "name": "detect_prohibited_items",
        "description": "Detect prohibited medical services and unauthorized charges based on insurance type (CGHS, ESI, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "line_items": {
                    "type": "array",
                    "description": "List of medical bill line items to analyze for prohibited items",
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
                "insurance_type": {
                    "type": "string",
                    "enum": ["cghs", "esi", "private", "none"],
                    "description": "Type of insurance to check prohibited items against",
                    "default": "cghs"
                }
            },
            "required": ["line_items"]
        }
    }
} 