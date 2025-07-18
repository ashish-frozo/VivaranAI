"""
Production Integration for ToolManager with Scaling and Load Balancing

This module provides integration utilities for connecting the enhanced ToolManager
with the existing agent system and production infrastructure.
"""

import asyncio
import structlog
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .tool_manager import tool_manager
from .scaling import ScalingConfig, LoadBalancingStrategy
from .load_balancer import LoadBalancerConfig
from agents.verticals.medical.tools.medical_rate_validator_tool import MedicalRateValidatorTool
from .generic_ocr_tool import GenericOCRTool
from agents.interfaces import AgentContext

logger = structlog.get_logger(__name__)


@dataclass
class ProductionConfig:
    """Configuration for production deployment."""
    enable_scaling: bool = True
    enable_monitoring: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30
    metrics_collection_interval: int = 60


class ProductionIntegration:
    """Integration layer for production-ready tool management."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def initialize_production_tools(self) -> bool:
        """Initialize all tools with production-ready configurations."""
        try:
            logger.info("Initializing production tools")
            
            # Configure RateValidatorTool with scaling
            rate_validator_scaling = ScalingConfig(
                min_instances=2,
                max_instances=10,
                target_cpu_utilization=70.0,
                target_memory_utilization=80.0,
                scale_up_threshold=75.0,
                scale_down_threshold=25.0,
                cooldown_period=300,
                load_balancing_strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
            )
            
            rate_validator_lb_config = LoadBalancerConfig(
                strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
                health_check_enabled=True,
                health_check_interval=30,
                max_retries=3,
                retry_delay=1.0,
                enable_sticky_sessions=True,
                session_timeout=3600
            )
            
            # Register MedicalRateValidatorTool with scaling
            success = await tool_manager.register_tool(
                tool_name="rate_validator",
                tool_class=MedicalRateValidatorTool,
                auto_initialize=False,
                enable_scaling=self.config.enable_scaling,
                scaling_config=rate_validator_scaling,
                load_balancer_config=rate_validator_lb_config
            )
            
            if not success:
                logger.error("Failed to register RateValidatorTool")
                return False
            
            # Configure GenericOCRTool with scaling
            ocr_scaling = ScalingConfig(
                min_instances=1,
                max_instances=5,
                target_cpu_utilization=80.0,
                target_memory_utilization=85.0,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                cooldown_period=180,
                load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS
            )
            
            ocr_lb_config = LoadBalancerConfig(
                strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
                health_check_enabled=True,
                health_check_interval=45,
                max_retries=2,
                retry_delay=2.0,
                enable_sticky_sessions=False
            )
            
            # Register GenericOCRTool with scaling
            success = await tool_manager.register_tool(
                tool_name="generic_ocr",
                tool_class=GenericOCRTool,
                auto_initialize=False,
                enable_scaling=self.config.enable_scaling,
                scaling_config=ocr_scaling,
                load_balancer_config=ocr_lb_config
            )
            
            if not success:
                logger.error("Failed to register GenericOCRTool")
                return False
            
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            # Start health checks if enabled
            if self.config.enable_health_checks:
                await self._start_health_checks()
            
            logger.info("Production tools initialized successfully")
            return True
            
        except Exception as e:
            logger.error(
                "Failed to initialize production tools",
                error=str(e),
                exc_info=True
            )
            return False
    
    async def execute_tool_with_context(
        self,
        tool_name: str,
        operation: str,
        context: AgentContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a tool operation with full production context."""
        try:
            # Add production metadata to context
            context.metadata = context.metadata or {}
            context.metadata.update({
                "production_mode": True,
                "scaling_enabled": self.config.enable_scaling,
                "monitoring_enabled": self.config.enable_monitoring
            })
            
            # Execute with load balancing
            result = await tool_manager.execute_tool(
                tool_name=tool_name,
                operation=operation,
                context=context,
                **kwargs
            )
            
            # Add production metadata to result
            result["production_metadata"] = {
                "tool_name": tool_name,
                "operation": operation,
                "context_id": context.request_id,
                "scaling_used": self.config.enable_scaling
            }
            
            return result
            
        except Exception as e:
            logger.error(
                "Production tool execution failed",
                tool_name=tool_name,
                operation=operation,
                error=str(e),
                exc_info=True
            )
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "tool_name": tool_name,
                "operation": operation
            }
    
    async def get_production_status(self) -> Dict[str, Any]:
        """Get comprehensive production status."""
        try:
            # Get tool manager status
            manager_status = tool_manager.get_manager_status()
            
            # Get all metrics
            all_metrics = await tool_manager.get_all_metrics()
            
            # Get infrastructure health
            infrastructure_health = await self._check_infrastructure_health()
            
            return {
                "production_config": {
                    "scaling_enabled": self.config.enable_scaling,
                    "monitoring_enabled": self.config.enable_monitoring,
                    "health_checks_enabled": self.config.enable_health_checks
                },
                "manager_status": manager_status,
                "metrics": all_metrics,
                "infrastructure_health": infrastructure_health,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(
                "Failed to get production status",
                error=str(e),
                exc_info=True
            )
            return {
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def _start_monitoring(self) -> None:
        """Start monitoring tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Production monitoring started")
    
    async def _start_health_checks(self) -> None:
        """Start health check tasks."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Production health checks started")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_collection_interval)
                
                # Collect metrics
                metrics = await tool_manager.get_all_metrics()
                
                # Log key metrics
                logger.info(
                    "Production metrics collected",
                    total_tools=metrics.get("manager_status", {}).get("manager_info", {}).get("total_tools", 0),
                    healthy_tools=metrics.get("manager_status", {}).get("manager_info", {}).get("healthy_tools", 0),
                    scaled_tools=metrics.get("manager_status", {}).get("manager_info", {}).get("scaled_tools", 0)
                )
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(
                    "Error in monitoring loop",
                    error=str(e),
                    exc_info=True
                )
    
    async def _health_check_loop(self) -> None:
        """Main health check loop."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Perform health checks
                health_status = await self._check_infrastructure_health()
                
                # Log health status
                logger.info(
                    "Health check completed",
                    overall_healthy=health_status.get("overall_healthy", False),
                    tool_manager_healthy=health_status.get("tool_manager_healthy", False),
                    scaling_healthy=health_status.get("scaling_healthy", False),
                    load_balancer_healthy=health_status.get("load_balancer_healthy", False)
                )
                
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error(
                    "Error in health check loop",
                    error=str(e),
                    exc_info=True
                )
    
    async def _check_infrastructure_health(self) -> Dict[str, Any]:
        """Check the health of all infrastructure components."""
        try:
            # Check tool manager health
            manager_status = tool_manager.get_manager_status()
            tool_manager_healthy = (
                manager_status.get("manager_info", {}).get("healthy_tools", 0) > 0
            )
            
            # Check scaling infrastructure health
            scaling_status = manager_status.get("scaling_infrastructure", {})
            scaling_healthy = scaling_status.get("healthy", True)
            
            # Check load balancer health
            lb_status = manager_status.get("load_balancer", {})
            load_balancer_healthy = lb_status.get("healthy", True)
            
            overall_healthy = all([
                tool_manager_healthy,
                scaling_healthy,
                load_balancer_healthy
            ])
            
            return {
                "overall_healthy": overall_healthy,
                "tool_manager_healthy": tool_manager_healthy,
                "scaling_healthy": scaling_healthy,
                "load_balancer_healthy": load_balancer_healthy,
                "details": {
                    "tool_manager": manager_status,
                    "scaling": scaling_status,
                    "load_balancer": lb_status
                }
            }
            
        except Exception as e:
            logger.error(
                "Error checking infrastructure health",
                error=str(e),
                exc_info=True
            )
            return {
                "overall_healthy": False,
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Shutdown production integration."""
        logger.info("Shutting down production integration")
        
        # Cancel monitoring tasks
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown tool manager
        await tool_manager.shutdown()
        
        logger.info("Production integration shutdown completed")


# Global production integration instance
production_integration = ProductionIntegration(
    ProductionConfig(
        enable_scaling=True,
        enable_monitoring=True,
        enable_health_checks=True,
        health_check_interval=30,
        metrics_collection_interval=60
    )
)
