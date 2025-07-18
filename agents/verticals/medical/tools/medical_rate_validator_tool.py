"""
Medical Rate Validator Tool - Pack-driven version.

This tool provides medical rate validation using the pack-driven architecture
instead of hard-coded CGHS/ESI logic.
"""

from __future__ import annotations

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from agents.interfaces import ToolCapabilityDeclaration, AgentContext
from agents.tools.base_tool import BaseTool
from agents.verticals.medical.medical_rate_validator import MedicalRateValidator

logger = structlog.get_logger(__name__)


class MedicalRateValidatorInput(BaseModel):
    """Input schema for medical rate validator tool."""
    line_items: List[Dict[str, Any]] = Field(..., description="List of extracted medical line items")
    state_code: Optional[str] = Field(default=None, description="State code for regional rates")
    validation_sources: List[str] = Field(default=["cghs", "esi"], description="Validation sources to use")


class MedicalRateValidatorTool(BaseTool):
    """
    Medical rate validator tool using pack-driven architecture.
    
    Provides medical bill validation capabilities using external rule packs
    instead of hard-coded logic.
    """
    
    def __init__(self, pack_id: str = "medical"):
        """Initialize the medical rate validator tool."""
        super().__init__(
            tool_name="medical_rate_validator",
            tool_version="3.0.0",
            initialization_timeout=30,
            health_check_interval=300
        )
        self.pack_id = pack_id
        self.validator: Optional[MedicalRateValidator] = None
        
        logger.info(f"Initialized MedicalRateValidatorTool with pack: {pack_id}")
    
    async def _build_capabilities(self) -> ToolCapabilityDeclaration:
        """Build tool capabilities declaration."""
        return ToolCapabilityDeclaration(
            tool_name=self.tool_name,
            version=self.tool_version,
            description="Validates medical charges using pack-driven architecture with CGHS, ESI, NPPA rates",
            supported_operations=[
                "validate_rates",
                "check_overcharges", 
                "generate_red_flags",
                "get_rate_info"
            ],
            required_dependencies=[
                "agents.verticals.medical.medical_rate_validator",
                "packs"
            ],
            optional_dependencies=[],
            resource_requirements={
                "memory_mb": 256,
                "cpu_cores": 1,
                "disk_mb": 100
            },
            performance_characteristics={
                "avg_execution_time_ms": 400,
                "max_execution_time_ms": 4000,
                "throughput_per_second": 12
            },
            metadata={
                "pack_id": self.pack_id,
                "supported_validation_sources": ["cghs", "esi", "nppa"],
                "supported_document_types": ["medical_bill", "pharmacy_invoice"],
                "pack_driven": True,
                "rate_validation_accuracy": 0.95
            }
        )
    
    async def _initialize_tool(self) -> bool:
        """Initialize the medical rate validator."""
        try:
            self.validator = MedicalRateValidator()
            success = await self.validator.initialize()
            
            if success:
                logger.info("Medical rate validator initialized successfully")
                return True
            else:
                logger.error("Failed to initialize medical rate validator")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize medical rate validator: {str(e)}")
            return False
    
    async def _execute_operation(
        self, 
        operation: str, 
        context: AgentContext, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute medical rate validation operation."""
        if operation == "validate_rates":
            return await self._validate_rates(**kwargs)
        elif operation == "check_overcharges":
            return await self._check_overcharges(**kwargs)
        elif operation == "generate_red_flags":
            return await self._generate_red_flags(**kwargs)
        elif operation == "get_rate_info":
            return await self._get_rate_info(**kwargs)
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }
    
    async def _check_tool_health(self) -> Dict[str, Any]:
        """Check medical rate validator health."""
        try:
            if not self.validator or not self.validator.is_initialized():
                return {
                    "healthy": False,
                    "error": "Validator not initialized"
                }
            
            # Test validation with sample data
            test_items = [{"description": "consultation", "amount": 100}]
            deltas = await self.validator.validate(test_items)
            
            return {
                "healthy": True,
                "validator_initialized": True,
                "pack_id": self.pack_id,
                "test_validation_count": len(deltas)
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _shutdown_tool(self) -> bool:
        """Shutdown medical rate validator."""
        try:
            self.validator = None
            logger.info("Medical rate validator shutdown completed")
            return True
        except Exception as e:
            logger.error(f"Error during medical rate validator shutdown: {str(e)}")
            return False
    
    async def _validate_rates(self, **kwargs) -> Dict[str, Any]:
        """Validate rates operation."""
        line_items = kwargs.get('line_items', [])
        state_code = kwargs.get('state_code')
        validation_sources = kwargs.get('validation_sources', ['cghs', 'esi'])
        
        if not self.validator:
            return {
                "success": False,
                "error": "Validator not initialized"
            }
        
        try:
            deltas = await self.validator.validate(
                line_items,
                state_code=state_code,
                validation_sources=validation_sources
            )
            
            return {
                "success": True,
                "validation_deltas": [
                    {
                        "item_description": delta.item_description,
                        "item_amount": delta.item_amount,
                        "reference_amount": delta.reference_amount,
                        "delta_amount": delta.delta_amount,
                        "delta_percentage": delta.delta_percentage,
                        "violation_type": delta.violation_type,
                        "severity": delta.severity,
                        "rule_source": delta.rule_source,
                        "confidence": delta.confidence,
                        "metadata": delta.metadata
                    }
                    for delta in deltas
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Rate validation failed: {str(e)}"
            }
    
    async def _check_overcharges(self, **kwargs) -> Dict[str, Any]:
        """Check for overcharges operation."""
        result = await self._validate_rates(**kwargs)
        
        if not result.get("success"):
            return result
        
        # Filter for overcharge violations
        overcharges = [
            delta for delta in result["validation_deltas"]
            if delta["violation_type"] == "overcharge"
        ]
        
        total_overcharge = sum(delta["delta_amount"] or 0 for delta in overcharges)
        
        return {
            "success": True,
            "overcharges": overcharges,
            "total_overcharge_amount": total_overcharge,
            "overcharge_count": len(overcharges)
        }
    
    async def _generate_red_flags(self, **kwargs) -> Dict[str, Any]:
        """Generate red flags operation."""
        result = await self._validate_rates(**kwargs)
        
        if not result.get("success"):
            return result
        
        # Convert deltas to red flags format
        red_flags = []
        for delta in result["validation_deltas"]:
            red_flag = {
                "type": delta["violation_type"],
                "severity": delta["severity"],
                "item": delta["item_description"],
                "amount": delta["item_amount"],
                "reference_amount": delta["reference_amount"],
                "excess_amount": delta["delta_amount"],
                "confidence": delta["confidence"],
                "source": delta["rule_source"],
                "message": self._generate_red_flag_message(delta)
            }
            red_flags.append(red_flag)
        
        return {
            "success": True,
            "red_flags": red_flags,
            "total_red_flags": len(red_flags)
        }
    
    async def _get_rate_info(self, **kwargs) -> Dict[str, Any]:
        """Get rate information for an item."""
        item_description = kwargs.get('item_description', '')
        validation_sources = kwargs.get('validation_sources', ['cghs', 'esi', 'nppa'])
        
        if not self.validator:
            return {
                "success": False,
                "error": "Validator not initialized"
            }
        
        try:
            rate_info = await self.validator.get_rate_for_item(
                item_description,
                validation_sources
            )
            
            return {
                "success": True,
                "rate_info": rate_info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get rate info: {str(e)}"
            }
    
    def _generate_red_flag_message(self, delta: Dict[str, Any]) -> str:
        """Generate human-readable red flag message."""
        item = delta["item_description"]
        amount = delta["item_amount"]
        violation_type = delta["violation_type"]
        
        if violation_type == "overcharge":
            reference = delta["reference_amount"]
            excess = delta["delta_amount"]
            return f"Overcharge detected for {item}: ₹{amount} charged vs ₹{reference} allowed (excess: ₹{excess})"
        elif violation_type == "duplicate":
            return f"Duplicate item detected: {item} (₹{amount})"
        elif violation_type == "prohibited":
            return f"Prohibited item found: {item} (₹{amount})"
        else:
            return f"Validation issue for {item}: {violation_type} (₹{amount})"
    
    async def __call__(
        self,
        line_items: List[Dict[str, Any]],
        state_code: Optional[str] = None,
        validation_sources: List[str] = None,
        dynamic_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate medical charges using pack-driven architecture.
        
        Args:
            line_items: List of extracted line items with descriptions and amounts
            state_code: Optional state code for regional rate validation
            validation_sources: List of validation sources to use
            dynamic_data: Optional dynamic data (not used in pack-driven version)
            
        Returns:
            Validation results with rate matches, red flags, and summary
        """
        if not self.validator or not self.validator.is_initialized():
            return {
                "success": False,
                "error": "Medical rate validator not initialized",
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
        
        try:
            # Validate using pack-driven validator
            deltas = await self.validator.validate(
                line_items,
                state_code=state_code,
                validation_sources=validation_sources or ["cghs", "esi"]
            )
            
            # Convert deltas to legacy format for compatibility
            rate_matches = []
            red_flags = []
            total_billed = 0.0
            total_allowed = 0.0
            total_overcharge = 0.0
            
            for item in line_items:
                amount = float(item.get('amount', item.get('total_amount', 0)))
                total_billed += amount
            
            for delta in deltas:
                if delta.violation_type == "overcharge" and delta.reference_amount:
                    # Rate match entry
                    rate_match = {
                        "item": delta.item_description,
                        "billed_amount": delta.item_amount,
                        "allowed_amount": delta.reference_amount,
                        "overcharge": delta.delta_amount,
                        "source": delta.rule_source
                    }
                    rate_matches.append(rate_match)
                    total_allowed += delta.reference_amount
                    total_overcharge += delta.delta_amount or 0
                
                # Red flag entry
                red_flag = {
                    "type": delta.violation_type,
                    "severity": delta.severity,
                    "item": delta.item_description,
                    "amount": delta.item_amount,
                    "reference_amount": delta.reference_amount,
                    "excess_amount": delta.delta_amount,
                    "confidence": delta.confidence,
                    "source": delta.rule_source,
                    "message": self._generate_red_flag_message({
                        "item_description": delta.item_description,
                        "item_amount": delta.item_amount,
                        "violation_type": delta.violation_type,
                        "reference_amount": delta.reference_amount,
                        "delta_amount": delta.delta_amount
                    })
                }
                red_flags.append(red_flag)
            
            # Calculate overcharge percentage
            overcharge_percentage = 0.0
            if total_allowed > 0:
                overcharge_percentage = (total_overcharge / total_allowed) * 100
            
            result = {
                "success": True,
                "rate_matches": rate_matches,
                "red_flags": red_flags,
                "validation_summary": {
                    "total_items": len(line_items),
                    "items_validated": len(rate_matches),
                    "total_billed": float(total_billed),
                    "total_allowed": float(total_allowed),
                    "total_overcharge": float(total_overcharge),
                    "overcharge_percentage": float(overcharge_percentage),
                    "violation_count": len(red_flags)
                },
                "validation_sources_used": validation_sources or ["cghs", "esi"],
                "state_validation": state_code is not None,
                "pack_driven": True,
                "pack_id": self.pack_id
            }
            
            logger.info(
                "Pack-driven medical validation completed",
                items_validated=len(rate_matches),
                red_flags_found=len(red_flags),
                total_overcharge=total_overcharge,
                overcharge_percentage=overcharge_percentage,
                pack_id=self.pack_id
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Pack-driven medical validation failed: {str(e)}"
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
MedicalRateValidatorTool._tool_schema = {
    "type": "function",
    "function": {
        "name": "validate_medical_rates",
        "description": "Validate medical charges using pack-driven architecture with CGHS, ESI, NPPA rates",
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
                        "enum": ["cghs", "esi", "nppa"]
                    },
                    "default": ["cghs", "esi"]
                }
            },
            "required": ["line_items"]
        }
    }
}
