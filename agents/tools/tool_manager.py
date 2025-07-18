"""
Tool Manager - Centralized tool lifecycle management and coordination.

Provides system-wide tool management including:
- Tool registration and discovery
- Lifecycle coordination (initialization, shutdown)
- Health monitoring and recovery
- Performance tracking
- Dependency management
"""

import asyncio
import structlog
import time
from typing import Dict, List, Optional, Any, Type, Set, Callable
from dataclasses import dataclass
from enum import Enum

from agents.interfaces import ITool, ToolState, ToolCapabilityDeclaration, AgentContext
from .scaling import scaling_manager, ScalingConfig, LoadBalancingStrategy
from .load_balancer import load_balancer, LoadBalancerConfig
from .monitoring import metrics_collector
from .base_tool import BaseTool

logger = structlog.get_logger(__name__)


class ToolManagerState(Enum):
    """Tool manager states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"
    SHUTDOWN = "shutdown"
    ERROR = "error"


@dataclass
class ToolRegistration:
    """Tool registration information."""
    tool_name: str
    tool_class: Type[ITool]
    tool_instance: Optional[ITool] = None
    tool_factory: Optional[Callable[[], ITool]] = None
    capabilities: Optional[ToolCapabilityDeclaration] = None
    registration_time: float = 0.0
    initialization_time: Optional[float] = None
    last_health_check: Optional[float] = None
    health_status: Dict[str, Any] = {}
    error_count: int = 0
    last_error: Optional[str] = None
    scaling_enabled: bool = False
    scaling_config: Optional[ScalingConfig] = None
    load_balancer_config: Optional[LoadBalancerConfig] = None


class ToolManager:
    """
    Centralized tool lifecycle manager.
    
    Manages all tools in the MedBillGuard system with:
    - Automatic tool discovery and registration
    - Coordinated initialization and shutdown
    - Health monitoring and auto-recovery
    - Performance tracking and optimization
    - Dependency resolution
    """
    
    def __init__(
        self,
        health_check_interval: int = 300,
        auto_recovery_enabled: bool = True,
        max_initialization_retries: int = 3
    ):
        self.state = ToolManagerState.UNINITIALIZED
        self.health_check_interval = health_check_interval
        self.auto_recovery_enabled = auto_recovery_enabled
        self.max_initialization_retries = max_initialization_retries
        
        # Tool registry
        self._tools: Dict[str, ToolRegistration] = {}
        self._tool_classes: Dict[str, Type[ITool]] = {}
        
        # Manager state
        self._initialized_at: Optional[float] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("ToolManager created")
    
    async def initialize(self) -> bool:
        """Initialize the tool manager and all registered tools."""
        if self.state != ToolManagerState.UNINITIALIZED:
            logger.warning(f"Tool manager already initialized (state: {self.state.value})")
            return self.state == ToolManagerState.READY
        
        try:
            self.state = ToolManagerState.INITIALIZING
            logger.info("Initializing tool manager")
            
            # Auto-discover and register tools
            await self._discover_tools()
            
            # Initialize all registered tools
            initialization_results = await self._initialize_all_tools()
            
            # Check if any critical tools failed
            failed_tools = [
                name for name, success in initialization_results.items() 
                if not success
            ]
            
            if failed_tools:
                logger.warning(f"Some tools failed to initialize: {failed_tools}")
                self.state = ToolManagerState.DEGRADED
            else:
                self.state = ToolManagerState.READY
            
            self._initialized_at = time.time()
            
            # Start health monitoring
            if self.auto_recovery_enabled:
                self._health_check_task = asyncio.create_task(self._health_monitor_loop())
            
            logger.info(
                f"Tool manager initialized successfully",
                state=self.state.value,
                total_tools=len(self._tools),
                failed_tools=len(failed_tools)
            )
            
            return self.state in [ToolManagerState.READY, ToolManagerState.DEGRADED]
            
        except Exception as e:
            self.state = ToolManagerState.ERROR
            logger.error(f"Tool manager initialization failed: {str(e)}", exc_info=True)
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the tool manager and all tools."""
        if self.state == ToolManagerState.SHUTDOWN:
            return True
        
        try:
            logger.info("Shutting down tool manager")
            self.state = ToolManagerState.SHUTTING_DOWN
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all tools
            shutdown_results = await self._shutdown_all_tools()
            
            failed_shutdowns = [
                name for name, success in shutdown_results.items() 
                if not success
            ]
            
            if failed_shutdowns:
                logger.warning(f"Some tools failed to shutdown cleanly: {failed_shutdowns}")
            
            self.state = ToolManagerState.SHUTDOWN
            logger.info("Tool manager shutdown completed")
            
            return len(failed_shutdowns) == 0
            
        except Exception as e:
            self.state = ToolManagerState.ERROR
            logger.error(f"Tool manager shutdown failed: {str(e)}", exc_info=True)
            return False
    
    async def register_tool(self, tool_class: Type[ITool], tool_name: Optional[str] = None) -> bool:
        try:
            # Create tool factory for scaling
            def tool_factory() -> ITool:
                return tool_class()
            
            # Create tool registration
            registration = ToolRegistration(
                tool_name=tool_name,
                tool_class=tool_class,
                tool_instance=None,
                tool_factory=tool_factory,
                capabilities=None,
                registration_time=time.time(),
                initialization_time=None,
                last_health_check=None,
                health_status={},
                error_count=0,
                last_error=None,
                scaling_enabled=enable_scaling,
                scaling_config=scaling_config,
                load_balancer_config=load_balancer_config
            )
            
            self._tools[tool_name] = registration
            
            # Register with scaling manager if enabled
            if enable_scaling and registration.tool_factory:
                await scaling_manager.register_tool(
                    tool_name,
                    registration.tool_factory,
                    scaling_config
                )
                logger.info(
                    "Tool registered for horizontal scaling",
                    tool_name=tool_name,
                    scaling_config=scaling_config
                )
            
            # Auto-initialize if requested and not using scaling
            if auto_initialize and not enable_scaling:
                await self._initialize_tool(tool_name)
            
            logger.info(
                "Tool registered successfully",
                tool_name=tool_name,
                auto_initialize=auto_initialize,
                scaling_enabled=enable_scaling
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to register tool",
                tool_name=tool_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_tool(self, tool_name: str, context: Optional[AgentContext] = None) -> Optional[ITool]:
        """Get a tool instance by name with load balancing support."""
        if tool_name not in self._tools:
            logger.error(f"Tool not found: {tool_name}")
            return None
        
        registration = self._tools[tool_name]
        
        # Use scaling manager if scaling is enabled
        if registration.scaling_enabled:
            instance = await scaling_manager.get_tool_instance(tool_name, context)
            if instance:
                return instance
            logger.warning(
                "No scaled instances available, falling back to single instance",
                tool_name=tool_name
            )
        
        # Initialize tool if not already done
        if registration.tool_instance is None:
            success = await self._initialize_tool(tool_name)
            if not success:
                return None
        
        return registration.tool_instance
    
    async def get_tool_capabilities(self, tool_name: str) -> Optional[ToolCapabilityDeclaration]:
        """Get tool capabilities."""
        registration = self._tools.get(tool_name)
        if not registration:
            return None
        
        if not registration.capabilities and registration.tool_instance:
            try:
                registration.capabilities = await registration.tool_instance.get_capabilities()
            except Exception as e:
                logger.error(f"Failed to get capabilities for {tool_name}: {str(e)}")
        
        return registration.capabilities
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            total_tools = len(self._tools)
            healthy_tools = 0
            degraded_tools = 0
            error_tools = 0
            
            tool_health = {}
            
            for tool_name, registration in self._tools.items():
                if registration.tool_instance:
                    try:
                        health = await registration.tool_instance.health_check()
                        tool_health[tool_name] = health
                        
                        if health.get('healthy', False):
                            healthy_tools += 1
                        elif registration.tool_instance.state == ToolState.ERROR:
                            error_tools += 1
                        else:
                            degraded_tools += 1
                    except Exception as e:
                        tool_health[tool_name] = {
                            'healthy': False,
                            'error': str(e)
                        }
                        error_tools += 1
                else:
                    tool_health[tool_name] = {
                        'healthy': False,
                        'error': 'Tool not initialized'
                    }
                    error_tools += 1
            
            overall_health = {
                'manager_state': self.state.value,
                'total_tools': total_tools,
                'healthy_tools': healthy_tools,
                'degraded_tools': degraded_tools,
                'error_tools': error_tools,
                'health_percentage': (healthy_tools / max(total_tools, 1)) * 100,
                'uptime_seconds': (
                    int(time.time() - self._initialized_at) if self._initialized_at else 0
                ),
                'tool_health': tool_health
            }
            
            return overall_health
            
        except Exception as e:
            logger.error(f"Failed to get system health: {str(e)}")
            return {
                'manager_state': self.state.value,
                'healthy': False,
                'error': str(e)
            }
    
    async def _discover_tools(self) -> None:
        """Auto-discover available tools."""
        try:
            # Import and register known tools
            from .rate_validator_tool import RateValidatorTool
            from .generic_ocr_tool import GenericOCRTool
            
            await self.register_tool(RateValidatorTool, "rate_validator")
            await self.register_tool(GenericOCRTool, "generic_ocr")
            
            logger.info(f"Auto-discovered {len(self._tools)} tools")
            
        except Exception as e:
            logger.error(f"Tool discovery failed: {str(e)}")
    
    async def _initialize_all_tools(self) -> Dict[str, bool]:
        """Initialize all registered tools."""
        results = {}
        
        for tool_name in self._tools.keys():
            success = await self._initialize_tool(tool_name)
            results[tool_name] = success
        
        return results
    
    async def _initialize_tool(self, tool_name: str) -> bool:
        """Initialize a specific tool."""
        registration = self._tools.get(tool_name)
        if not registration:
            return False
        
        try:
            # Create tool instance
            if not registration.tool_instance:
                registration.tool_instance = registration.tool_class()
            
            # Initialize tool
            success = await registration.tool_instance.initialize()
            
            if success:
                registration.initialization_time = time.time()
                registration.capabilities = await registration.tool_instance.get_capabilities()
                logger.info(f"Tool {tool_name} initialized successfully")
            else:
                registration.error_count += 1
                registration.last_error = "Initialization failed"
                logger.error(f"Tool {tool_name} initialization failed")
            
            return success
            
        except Exception as e:
            registration.error_count += 1
            registration.last_error = str(e)
            logger.error(f"Tool {tool_name} initialization error: {str(e)}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown all tools and cleanup resources."""
        logger.info("Shutting down ToolManager")
        
        # Shutdown scaling manager
        await scaling_manager.shutdown()
        
        # Shutdown load balancer
        await load_balancer.shutdown()
        
        # Shutdown all tools
        for tool_name, registration in self._tools.items():
            if registration.tool_instance:
                try:
                    await registration.tool_instance.shutdown()
                    logger.info(f"Tool {tool_name} shutdown completed")
                except Exception as e:
                    logger.error(f"Error shutting down tool {tool_name}: {str(e)}")
        
        # Cancel health monitoring
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
        
        # Clear registrations
        self._tools.clear()
        
        logger.info("ToolManager shutdown completed")
    
    async def _shutdown_all_tools(self) -> Dict[str, bool]:
        """Shutdown all tools."""
        results = {}
        
        for tool_name, registration in self._tools.items():
            if registration.tool_instance:
                try:
                    success = await registration.tool_instance.shutdown()
                    results[tool_name] = success
                except Exception as e:
                    logger.error(f"Tool {tool_name} shutdown error: {str(e)}")
                    results[tool_name] = False
            else:
                results[tool_name] = True
        
        return results
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._perform_health_checks()
                    await asyncio.sleep(self.health_check_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error: {str(e)}")
                    await asyncio.sleep(60)  # Retry after 1 minute
        
        except asyncio.CancelledError:
            logger.info("Health monitoring stopped")
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all tools."""
        for tool_name, registration in self._tools.items():
            if registration.tool_instance:
                try:
                    health = await registration.tool_instance.health_check()
                    registration.health_status = health
                    registration.last_health_check = time.time()
                    
                    # Auto-recovery for unhealthy tools
                    if not health.get('healthy', False) and self.auto_recovery_enabled:
                        logger.warning(f"Tool {tool_name} unhealthy, attempting recovery")
                        await self._attempt_tool_recovery(tool_name)
                        
                except Exception as e:
                    registration.error_count += 1
                    registration.last_error = str(e)
                    logger.error(f"Health check failed for {tool_name}: {str(e)}")
    
    async def _attempt_tool_recovery(self, tool_name: str) -> bool:
        """Attempt to recover an unhealthy tool."""
        registration = self._tools.get(tool_name)
        if not registration or not registration.tool_instance:
            return False
        
        try:
            # Try to shutdown and reinitialize
            await registration.tool_instance.shutdown()
            registration.tool_instance = registration.tool_class()
            success = await registration.tool_instance.initialize()
            
            if success:
                logger.info(f"Tool {tool_name} recovered successfully")
                return True
            else:
                logger.error(f"Tool {tool_name} recovery failed")
                return False
                
        except Exception as e:
            logger.error(f"Tool {tool_name} recovery error: {str(e)}")
            return False


    async def configure_tool_scaling(
        self,
        tool_name: str,
        scaling_config: ScalingConfig,
        load_balancer_config: Optional[LoadBalancerConfig] = None
    ) -> bool:
        """Configure scaling for an existing tool."""
        if tool_name not in self._tools:
            logger.error(f"Tool {tool_name} not found for scaling configuration")
            return False
        
        registration = self._tools[tool_name]
        
        try:
            # Update registration with scaling config
            registration.scaling_enabled = True
            registration.scaling_config = scaling_config
            registration.load_balancer_config = load_balancer_config
            
            # Register with scaling manager if tool factory exists
            if registration.tool_factory:
                await scaling_manager.register_tool(
                    tool_name,
                    registration.tool_factory,
                    scaling_config
                )
                
                logger.info(
                    "Tool scaling configured successfully",
                    tool_name=tool_name,
                    scaling_config=scaling_config
                )
                return True
            else:
                logger.error(
                    "Cannot configure scaling: tool factory not available",
                    tool_name=tool_name
                )
                return False
                
        except Exception as e:
            logger.error(
                "Failed to configure tool scaling",
                tool_name=tool_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_tool_metrics(self, tool_name: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a tool."""
        if tool_name not in self._tools:
            return {"error": f"Tool {tool_name} not found"}
        
        registration = self._tools[tool_name]
        
        # Get basic tool metrics
        metrics = {
            "tool_name": tool_name,
            "registration_time": registration.registration_time,
            "error_count": registration.error_count,
            "last_error": registration.last_error,
            "scaling_enabled": registration.scaling_enabled
        }
        
        # Add tool instance metrics if available
        if registration.tool_instance:
            tool_metrics = await registration.tool_instance.get_metrics()
            metrics.update(tool_metrics)
        
        # Add scaling metrics if enabled
        if registration.scaling_enabled:
            scaling_metrics = await scaling_manager.get_tool_metrics(tool_name)
            metrics["scaling_metrics"] = scaling_metrics
            
            # Add load balancer metrics
            lb_metrics = await load_balancer.get_tool_metrics(tool_name)
            metrics["load_balancer_metrics"] = lb_metrics
        
        return metrics
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all tools and infrastructure."""
        tool_metrics = {}
        
        for tool_name in self._tools.keys():
            tool_metrics[tool_name] = await self.get_tool_metrics(tool_name)
        
        return {
            "manager_status": self.get_manager_status(),
            "tool_metrics": tool_metrics,
            "system_metrics": await metrics_collector.get_system_metrics(),
            "timestamp": time.time()
        }


# Global tool manager instance
tool_manager = ToolManager()
