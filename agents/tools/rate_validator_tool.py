"""
Rate Validator Tool - Enhanced tool with lifecycle management.

This tool provides comprehensive rate validation capabilities with proper
lifecycle management, error handling, and performance monitoring.
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from medbillguardagent.rate_validator import RateValidator
from agents.interfaces import ToolCapabilityDeclaration, AgentContext
from .base_tool import BaseTool

logger = structlog.get_logger(__name__)


class RateValidatorInput(BaseModel):
    """Input schema for rate validator tool."""
    line_items: List[Dict[str, Any]] = Field(..., description="List of extracted line items")
    state_code: Optional[str] = Field(default=None, description="State code for regional rates")
    validation_sources: List[str] = Field(default=["cghs", "esi"], description="Validation sources to use")


class RateValidatorTool(BaseTool):
    """
    Enhanced rate validator tool with lifecycle management.
    
    Provides rate validation capabilities including:
    - CGHS rate validation
    - ESI rate validation  
    - NPPA price validation
    - State-specific rate validation
    - Overcharge detection and red flag generation
    """
    
    def __init__(self, reference_data_loader=None):
        """Initialize the rate validator tool."""
        super().__init__(
            tool_name="rate_validator",
            tool_version="2.0.0",
            initialization_timeout=30,
            health_check_interval=300
        )
        self.reference_data_loader = reference_data_loader
        self.validator: Optional[RateValidator] = None
        
        logger.info("Initialized RateValidatorTool with lifecycle management")
    
    # BaseTool abstract method implementations
    
    async def _build_capabilities(self) -> ToolCapabilityDeclaration:
        """Build tool capabilities declaration."""
        return ToolCapabilityDeclaration(
            tool_name=self.tool_name,
            version=self.tool_version,
            description="Validates medical charges against CGHS, ESI, NPPA and state reference rates",
            supported_operations=[
                "validate_rates",
                "check_overcharges",
                "generate_red_flags"
            ],
            required_dependencies=[
                "medbillguardagent.rate_validator"
            ],
            optional_dependencies=[
                "reference_data_loader"
            ],
            resource_requirements={
                "memory_mb": 256,
                "cpu_cores": 1,
                "disk_mb": 100
            },
            performance_characteristics={
                "avg_execution_time_ms": 500,
                "max_execution_time_ms": 5000,
                "throughput_per_second": 10
            },
            metadata={
                "supported_validation_sources": ["cghs", "esi", "nppa", "state"],
                "supported_document_types": ["medical_bill", "pharmacy_invoice"],
                "rate_validation_accuracy": 0.95
            }
        )
    
    async def _initialize_tool(self) -> bool:
        """Initialize the rate validator."""
        try:
            self.validator = RateValidator(reference_loader=self.reference_data_loader)
            
            # Test validator functionality
            test_items = ["Consultation"]
            test_costs = {"Consultation": 100.0}
            
            # Quick validation test
            await self.validator.validate_item_rates(test_items, test_costs)
            
            logger.info("Rate validator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize rate validator: {str(e)}")
            return False
    
    async def _execute_operation(
        self, 
        operation: str, 
        context: AgentContext, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute rate validation operation."""
        if operation == "validate_rates":
            return await self._validate_rates(**kwargs)
        elif operation == "check_overcharges":
            return await self._check_overcharges(**kwargs)
        elif operation == "generate_red_flags":
            return await self._generate_red_flags(**kwargs)
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }
    
    async def _check_tool_health(self) -> Dict[str, Any]:
        """Check rate validator health."""
        try:
            if not self.validator:
                return {
                    "validator_status": "not_initialized",
                    "healthy": False
                }
            
            # Test basic functionality
            test_items = ["Test Item"]
            test_costs = {"Test Item": 50.0}
            
            await self.validator.validate_item_rates(test_items, test_costs)
            
            return {
                "validator_status": "healthy",
                "reference_data_loaded": bool(self.reference_data_loader),
                "healthy": True
            }
            
        except Exception as e:
            return {
                "validator_status": "error",
                "error": str(e),
                "healthy": False
            }
    
    async def _shutdown_tool(self) -> bool:
        """Shutdown rate validator."""
        try:
            if self.validator:
                # Cleanup validator resources if needed
                self.validator = None
            
            logger.info("Rate validator shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Rate validator shutdown error: {str(e)}")
            return False
    
    # Helper methods for operations
    
    async def _validate_rates(self, **kwargs) -> Dict[str, Any]:
        """Validate rates operation."""
        line_items = kwargs.get('line_items', [])
        state_code = kwargs.get('state_code')
        validation_sources = kwargs.get('validation_sources', ['cghs', 'esi'])
        dynamic_data = kwargs.get('dynamic_data')
        
        return await self.__call__(
            line_items=line_items,
            state_code=state_code,
            validation_sources=validation_sources,
            dynamic_data=dynamic_data
        )
    
    async def _check_overcharges(self, **kwargs) -> Dict[str, Any]:
        """Check for overcharges operation."""
        result = await self._validate_rates(**kwargs)
        if result.get('success'):
            # Filter for overcharge red flags only
            overcharge_flags = [
                flag for flag in result.get('red_flags', [])
                if flag.get('type') == 'overcharge'
            ]
            result['red_flags'] = overcharge_flags
            result['overcharge_count'] = len(overcharge_flags)
        return result
    
    async def _generate_red_flags(self, **kwargs) -> Dict[str, Any]:
        """Generate red flags operation."""
        result = await self._validate_rates(**kwargs)
        if result.get('success'):
            # Return only red flags
            return {
                'success': True,
                'red_flags': result.get('red_flags', []),
                'red_flag_count': len(result.get('red_flags', [])),
                'tool_name': self.tool_name
            }
        return result
    
    # Legacy method for backward compatibility
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