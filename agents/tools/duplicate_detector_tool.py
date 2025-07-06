"""
Duplicate Detector Tool - Async wrapper for DuplicateDetector.

This tool wraps the existing DuplicateDetector component to make it compatible
with the OpenAI Agent SDK framework for detecting duplicate medical services.
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from medbillguardagent.duplicate_detector import DuplicateDetector

logger = structlog.get_logger(__name__)


class DuplicateDetectorInput(BaseModel):
    """Input schema for duplicate detector tool."""
    line_items: List[Dict[str, Any]] = Field(..., description="List of extracted line items")
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold for duplicate detection")


class DuplicateDetectorTool:
    """
    Async tool wrapper for DuplicateDetector.
    
    Provides duplicate detection capabilities including:
    - Exact duplicate detection
    - Similar item detection using fuzzy matching
    - Standardized medical term matching
    - Cost impact analysis for duplicates
    """
    
    def __init__(self):
        """Initialize the duplicate detector tool."""
        self.detector = DuplicateDetector()
        logger.info("Initialized DuplicateDetectorTool")
    
    async def __call__(
        self,
        line_items: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Detect duplicate medical items and services in a bill.
        
        Args:
            line_items: List of extracted line items with descriptions and amounts
            similarity_threshold: Threshold for considering items as duplicates (0.0-1.0)
            
        Returns:
            Dict containing duplicate groups and red flags
        """
        try:
            logger.info(
                "Starting duplicate detection",
                item_count=len(line_items),
                similarity_threshold=similarity_threshold
            )
            
            if not line_items:
                return {
                    "success": True,
                    "duplicate_groups": [],
                    "red_flags": [],
                    "duplicate_summary": {
                        "total_items": 0,
                        "duplicate_groups_found": 0,
                        "total_duplicate_items": 0,
                        "duplicate_cost_impact": 0.0
                    }
                }
            
            # Extract item descriptions and costs
            items = [item.get("description", "") for item in line_items if item.get("description")]
            item_costs = {
                item.get("description", ""): float(item.get("total_amount", 0)) 
                for item in line_items
                if item.get("description") and item.get("total_amount")
            }
            
            # Update detector threshold if provided
            if similarity_threshold != 0.8:
                self.detector.similarity_threshold = similarity_threshold
            
            # Detect duplicates
            duplicate_groups, red_flags = self.detector.detect_duplicates(items, item_costs)
            
            # Convert duplicate groups to serializable format
            serialized_groups = []
            for group in duplicate_groups:
                serialized_groups.append({
                    "canonical_name": group.canonical_name,
                    "items": group.items,
                    "item_type": group.item_type.value,
                    "total_occurrences": group.total_occurrences,
                    "confidence": float(group.confidence)
                })
            
            # Convert red flags to serializable format
            serialized_red_flags = []
            for flag in red_flags:
                # Round decimal values to prevent precision errors
                overcharge_amount = float(flag.overcharge_amount) if flag.overcharge_amount else 0.0
                overcharge_amount = round(overcharge_amount, 2)
                
                serialized_red_flags.append({
                    "type": "duplicate",  # RedFlag doesn't have a 'type' attribute, using fixed value
                    "severity": "high",   # Assuming duplicates are high severity
                    "item": flag.item,
                    "reason": flag.reason,
                    "overcharge_amount": overcharge_amount,
                    "confidence": float(flag.confidence),
                    "metadata": {
                        "item_type": flag.item_type.value if hasattr(flag, 'item_type') else "unknown",
                        "is_duplicate": flag.is_duplicate,
                        "source": flag.source
                    }
                })
            
            # Calculate summary statistics
            total_duplicate_items = sum(
                group.total_occurrences - 1  # Subtract 1 to exclude the original item
                for group in duplicate_groups
                if group.total_occurrences > 1
            )
            
            duplicate_cost_impact = sum(
                float(flag.overcharge_amount) if flag.overcharge_amount else 0.0
                for flag in red_flags
            )
            duplicate_cost_impact = round(duplicate_cost_impact, 2)
            
            result = {
                "success": True,
                "duplicate_groups": serialized_groups,
                "red_flags": serialized_red_flags,
                "duplicate_summary": {
                    "total_items": len(line_items),
                    "duplicate_groups_found": len(duplicate_groups),
                    "total_duplicate_items": total_duplicate_items,
                    "duplicate_cost_impact": float(duplicate_cost_impact)
                },
                "detection_settings": {
                    "similarity_threshold": similarity_threshold,
                    "exact_match_threshold": self.detector.exact_match_threshold
                }
            }
            
            logger.info(
                "Duplicate detection completed",
                duplicate_groups=len(duplicate_groups),
                red_flags=len(red_flags),
                duplicate_items=total_duplicate_items,
                cost_impact=duplicate_cost_impact
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Duplicate detection failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "success": False,
                "error": error_msg,
                "duplicate_groups": [],
                "red_flags": [],
                "duplicate_summary": {
                    "total_items": len(line_items) if line_items else 0,
                    "duplicate_groups_found": 0,
                    "total_duplicate_items": 0,
                    "duplicate_cost_impact": 0.0
                }
            }


# Tool schema for OpenAI Agent SDK
DuplicateDetectorTool._tool_schema = {
    "type": "function",
    "function": {
        "name": "detect_duplicates",
        "description": "Detect duplicate medical services, procedures, and tests in a medical bill using fuzzy matching",
        "parameters": {
            "type": "object",
            "properties": {
                "line_items": {
                    "type": "array",
                    "description": "List of medical bill line items to analyze for duplicates",
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
                "similarity_threshold": {
                    "type": "number",
                    "description": "Similarity threshold for duplicate detection (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8
                }
            },
            "required": ["line_items"]
        }
    }
} 