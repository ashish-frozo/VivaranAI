"""
Base Tool - Foundation class for all MedBillGuard tools.

Provides standardized lifecycle management, error handling, and interface compliance
for all tools in the MedBillGuard multi-agent system.
"""

import asyncio
import time
import structlog
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
from enum import Enum

from agents.interfaces import ITool, ToolState, ToolCapabilityDeclaration, AgentContext
from .resilience import (
    resilience_manager, 
    CircuitBreakerConfig, 
    RateLimitConfig, 
    RetryConfig
)
from .monitoring import metrics_collector, PerformanceTracker

logger = structlog.get_logger(__name__)


class BaseTool(ABC):
    """
    Base class for all MedBillGuard tools implementing ITool interface.
    
    Provides:
    - Standardized lifecycle management
    - Error handling and recovery
    - Health monitoring
    - Performance metrics
    - Interface compliance validation
    """
    
    def __init__(
        self,
        tool_name: str,
        tool_version: str = "1.0.0",
        initialization_timeout: int = 30,
        health_check_interval: int = 300,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """Initialize the base tool with lifecycle management."""
        self._tool_name = tool_name
        self._tool_version = tool_version
        self._initialization_timeout = initialization_timeout
        self._health_check_interval = health_check_interval
        
        # Resilience configuration
        self._circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        self._rate_limit_config = rate_limit_config or RateLimitConfig()
        self._retry_config = retry_config or RetryConfig()
        
        # Performance metrics
        self._total_executions = 0
        self._total_errors = 0
        self._total_execution_time_ms = 0
        self._last_health_check = 0
        
        # Initialization tracking
        self._initialized_at: Optional[float] = None
        self._last_error: Optional[str] = None
        
        logger.info(
            "Initialized BaseTool",
            tool_name=tool_name,
            version=tool_version
        )
    
    # ITool interface implementation
    
    @property
    def tool_name(self) -> str:
        """Get tool name."""
        return self._tool_name
    
    @property
    def tool_version(self) -> str:
        """Get tool version."""
        return self._tool_version
    
    @property
    def state(self) -> ToolState:
        """Get current tool state."""
        return self._state
    
    async def get_capabilities(self) -> ToolCapabilityDeclaration:
        """Get tool capabilities and metadata."""
        return await self._build_capabilities()
    
    async def initialize(self) -> bool:
        """Initialize the tool."""
        if self._state != ToolState.UNINITIALIZED:
            logger.warning(
                "Tool already initialized",
                tool_name=self.tool_name,
                current_state=self._state.value
            )
            return self._state == ToolState.READY
        
        try:
            self._state = ToolState.INITIALIZING
            logger.info("Initializing tool", tool_name=self.tool_name)
            
            # Run tool-specific initialization with timeout
            success = await asyncio.wait_for(
                self._initialize_tool(),
                timeout=self._initialization_timeout
            )
            
            if success:
                self._state = ToolState.READY
                self._initialized_at = time.time()
                logger.info(
                    "Tool initialized successfully",
                    tool_name=self.tool_name,
                    initialization_time_ms=int((time.time() - (self._initialized_at or 0)) * 1000)
                )
                return True
            else:
                self._state = ToolState.ERROR
                logger.error("Tool initialization failed", tool_name=self.tool_name)
                return False
                
        except asyncio.TimeoutError:
            self._state = ToolState.ERROR
            self._last_error = "Initialization timeout"
            logger.error(
                "Tool initialization timeout",
                tool_name=self.tool_name,
                timeout_seconds=self._initialization_timeout
            )
            return False
            
        except Exception as e:
            self._state = ToolState.ERROR
            self._last_error = str(e)
            logger.error(
                "Tool initialization error",
                tool_name=self.tool_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def execute(
        self, 
        operation: str, 
        context: AgentContext, 
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a tool operation with resilience patterns."""
        if self._state != ToolState.READY:
            return {
                "success": False,
                "error": f"Tool not ready (state: {self._state.value})",
                "tool_name": self.tool_name
            }
        
        # Validate input
        validation_result = await self.validate_input(operation, **kwargs)
        if not validation_result.get("valid", True):
            return {
                "success": False,
                "error": validation_result.get("error", "Invalid input"),
                "tool_name": self.tool_name
            }
        
        # Execute with resilience patterns (circuit breaker, rate limiting, retry)
        try:
            async with PerformanceTracker(
                metrics_collector,
                self.tool_name,
                operation,
                labels={"tool_version": self.tool_version}
            ):
                result = await resilience_manager.execute_with_resilience(
                    f"{self.tool_name}_{operation}",
                    self._execute_operation,
                    operation,
                    context,
                    circuit_breaker_config=self._circuit_breaker_config,
                    rate_limit_config=self._rate_limit_config,
                    retry_config=self._retry_config,
                    **kwargs
                )
                
                # Update metrics
                self._total_executions += 1
                
                # Ensure result has required fields
                if not isinstance(result, dict):
                    result = {"success": False, "error": "Invalid result format"}
                
                result["tool_name"] = self.tool_name
                return result
                
        except Exception as e:
            # Update error metrics
            self._total_executions += 1
            self._total_errors += 1
            self._last_error = str(e)
            
            logger.error(
                "Tool execution failed",
                tool_name=self.tool_name,
                operation=operation,
                error=str(e),
                exc_info=True
            )
            
            return {
                "success": False,
                "error": str(e),
                "tool_name": self.tool_name,
                "error_type": type(e).__name__
            }
    
    async def validate_input(self, operation: str, **kwargs) -> bool:
        """Validate input parameters for an operation."""
        # Default implementation - override in subclasses
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Check tool health status."""
        current_time = time.time()
        
        # Basic health metrics
        health_data = {
            "tool_name": self.tool_name,
            "version": self.tool_version,
            "state": self._state.value,
            "healthy": self._state == ToolState.READY,
            "initialized_at": self._initialized_at,
            "last_error": self._last_error,
            "metrics": {
                "total_executions": self._total_executions,
                "total_errors": self._total_errors,
                "error_rate": self._total_errors / max(self._total_executions, 1),
                "avg_execution_time_ms": (
                    self._total_execution_time_ms / max(self._total_executions, 1)
                ),
                "uptime_seconds": (
                    int(current_time - self._initialized_at) if self._initialized_at else 0
                )
            }
        }
        
        # Run tool-specific health checks
        try:
            tool_health = await self._check_tool_health()
            health_data.update(tool_health)
        except Exception as e:
            health_data["healthy"] = False
            health_data["health_check_error"] = str(e)
            logger.error(
                "Tool health check failed",
                tool_name=self.tool_name,
                error=str(e)
            )
        
        self._last_health_check = current_time
        return health_data
    
    async def shutdown(self) -> bool:
        """Shutdown the tool."""
        if self._state == ToolState.SHUTDOWN:
            return True
        
        try:
            logger.info("Shutting down tool", tool_name=self.tool_name)
            self._state = ToolState.SHUTTING_DOWN
            
            # Run tool-specific shutdown
            success = await self._shutdown_tool()
            
            if success:
                self._state = ToolState.SHUTDOWN
                logger.info("Tool shutdown completed", tool_name=self.tool_name)
                return True
            else:
                self._state = ToolState.ERROR
                logger.error("Tool shutdown failed", tool_name=self.tool_name)
                return False
                
        except Exception as e:
            self._state = ToolState.ERROR
            self._last_error = str(e)
            logger.error(
                "Tool shutdown error",
                tool_name=self.tool_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    # Abstract methods for subclasses to implement
    
    @abstractmethod
    async def _build_capabilities(self) -> ToolCapabilityDeclaration:
        """Build tool capabilities declaration."""
        pass
    
    @abstractmethod
    async def _initialize_tool(self) -> bool:
        """Tool-specific initialization logic."""
        pass
    
    @abstractmethod
    async def _execute_operation(
        self, 
        operation: str, 
        context: AgentContext, 
        **kwargs
    ) -> Dict[str, Any]:
        """Tool-specific operation execution."""
        pass
    
    async def _check_tool_health(self) -> Dict[str, Any]:
        """Tool-specific health checks (override in subclasses)."""
        return {"tool_specific_health": "ok"}
    
    async def _shutdown_tool(self) -> bool:
        """Tool-specific shutdown logic (override in subclasses)."""
        return True
    
    # Utility methods
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get tool performance metrics."""
        return {
            "total_executions": self._total_executions,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(self._total_executions, 1),
            "avg_execution_time_ms": (
                self._total_execution_time_ms / max(self._total_executions, 1)
            ),
            "uptime_seconds": (
                int(time.time() - self._initialized_at) if self._initialized_at else 0
            )
        }
