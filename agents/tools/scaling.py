"""
Scaling Infrastructure for MedBillGuard Tools.

Provides horizontal scaling, load balancing, and distributed tool management
for production-ready tool operations.
"""

import asyncio
import hashlib
import time
import structlog
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import random

from agents.interfaces import ITool, ToolState, AgentContext

logger = structlog.get_logger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for tool instances."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"


@dataclass
class ToolInstanceMetrics:
    """Metrics for a tool instance."""
    instance_id: str
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time_ms: float = 0.0
    last_health_check: float = 0.0
    health_status: bool = True
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    weight: float = 1.0


@dataclass
class ScalingConfig:
    """Configuration for tool scaling."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    health_check_interval: int = 30  # seconds
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS


class ToolInstancePool:
    """Pool of tool instances with load balancing and health monitoring."""
    
    def __init__(
        self,
        tool_name: str,
        tool_factory: Callable[[], ITool],
        scaling_config: ScalingConfig
    ):
        self.tool_name = tool_name
        self.tool_factory = tool_factory
        self.scaling_config = scaling_config
        
        # Instance management
        self.instances: Dict[str, ITool] = {}
        self.metrics: Dict[str, ToolInstanceMetrics] = {}
        self.round_robin_index = 0
        self.consistent_hash_ring: List[str] = []
        
        # Scaling state
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0
        self.scaling_in_progress = False
        
        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._auto_scaler_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            "Created tool instance pool",
            tool_name=tool_name,
            min_instances=scaling_config.min_instances,
            max_instances=scaling_config.max_instances
        )
    
    async def initialize(self) -> bool:
        """Initialize the tool pool with minimum instances."""
        try:
            # Create minimum instances
            for i in range(self.scaling_config.min_instances):
                await self._create_instance()
            
            # Start background tasks
            self._health_monitor_task = asyncio.create_task(self._health_monitor())
            self._auto_scaler_task = asyncio.create_task(self._auto_scaler())
            
            logger.info(
                "Tool pool initialized",
                tool_name=self.tool_name,
                instance_count=len(self.instances)
            )
            return True
            
        except Exception as e:
            logger.error(
                "Failed to initialize tool pool",
                tool_name=self.tool_name,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def get_instance(self, context: Optional[AgentContext] = None) -> Optional[ITool]:
        """Get a tool instance using the configured load balancing strategy."""
        healthy_instances = [
            instance_id for instance_id, metrics in self.metrics.items()
            if metrics.health_status and instance_id in self.instances
        ]
        
        if not healthy_instances:
            logger.warning(
                "No healthy instances available",
                tool_name=self.tool_name,
                total_instances=len(self.instances)
            )
            return None
        
        # Select instance based on load balancing strategy
        selected_id = await self._select_instance(healthy_instances, context)
        
        if selected_id and selected_id in self.instances:
            # Update connection count
            self.metrics[selected_id].active_connections += 1
            return self.instances[selected_id]
        
        return None
    
    async def release_instance(self, instance: ITool) -> None:
        """Release a tool instance back to the pool."""
        instance_id = self._get_instance_id(instance)
        if instance_id in self.metrics:
            self.metrics[instance_id].active_connections = max(
                0, self.metrics[instance_id].active_connections - 1
            )
    
    async def _select_instance(
        self, 
        healthy_instances: List[str], 
        context: Optional[AgentContext] = None
    ) -> Optional[str]:
        """Select an instance based on the load balancing strategy."""
        strategy = self.scaling_config.load_balancing_strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected = healthy_instances[self.round_robin_index % len(healthy_instances)]
            self.round_robin_index += 1
            return selected
        
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(
                healthy_instances,
                key=lambda x: self.metrics[x].active_connections
            )
        
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Select based on weights
            total_weight = sum(self.metrics[x].weight for x in healthy_instances)
            if total_weight <= 0:
                return random.choice(healthy_instances)
            
            target = random.uniform(0, total_weight)
            current_weight = 0
            for instance_id in healthy_instances:
                current_weight += self.metrics[instance_id].weight
                if current_weight >= target:
                    return instance_id
            return healthy_instances[-1]
        
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            if context and hasattr(context, 'user_id'):
                # Hash based on user_id for session affinity
                hash_key = hashlib.md5(str(context.user_id).encode()).hexdigest()
                hash_value = int(hash_key, 16)
                return healthy_instances[hash_value % len(healthy_instances)]
            else:
                return random.choice(healthy_instances)
        
        elif strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_instances)
        
        return healthy_instances[0]
    
    async def _create_instance(self) -> Optional[str]:
        """Create a new tool instance."""
        try:
            instance = self.tool_factory()
            instance_id = self._generate_instance_id()
            
            # Initialize the instance
            if await instance.initialize():
                self.instances[instance_id] = instance
                self.metrics[instance_id] = ToolInstanceMetrics(instance_id=instance_id)
                
                logger.info(
                    "Created tool instance",
                    tool_name=self.tool_name,
                    instance_id=instance_id,
                    total_instances=len(self.instances)
                )
                return instance_id
            else:
                logger.error(
                    "Failed to initialize tool instance",
                    tool_name=self.tool_name,
                    instance_id=instance_id
                )
                return None
                
        except Exception as e:
            logger.error(
                "Error creating tool instance",
                tool_name=self.tool_name,
                error=str(e),
                exc_info=True
            )
            return None
    
    async def _remove_instance(self, instance_id: str) -> bool:
        """Remove a tool instance."""
        try:
            if instance_id in self.instances:
                instance = self.instances[instance_id]
                await instance.shutdown()
                del self.instances[instance_id]
                del self.metrics[instance_id]
                
                logger.info(
                    "Removed tool instance",
                    tool_name=self.tool_name,
                    instance_id=instance_id,
                    remaining_instances=len(self.instances)
                )
                return True
            return False
            
        except Exception as e:
            logger.error(
                "Error removing tool instance",
                tool_name=self.tool_name,
                instance_id=instance_id,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def _health_monitor(self) -> None:
        """Monitor health of all tool instances."""
        while not self._shutdown_event.is_set():
            try:
                for instance_id, instance in list(self.instances.items()):
                    try:
                        health_data = await instance.health_check()
                        metrics = self.metrics[instance_id]
                        
                        # Update health status
                        metrics.health_status = health_data.get("healthy", False)
                        metrics.last_health_check = time.time()
                        
                        # Update performance metrics
                        if "metrics" in health_data:
                            perf_metrics = health_data["metrics"]
                            metrics.avg_response_time_ms = perf_metrics.get(
                                "avg_execution_time_ms", metrics.avg_response_time_ms
                            )
                            metrics.total_requests = perf_metrics.get(
                                "total_executions", metrics.total_requests
                            )
                            metrics.total_errors = perf_metrics.get(
                                "total_errors", metrics.total_errors
                            )
                        
                        # Update resource usage (mock values for now)
                        metrics.cpu_usage = random.uniform(20, 90)
                        metrics.memory_usage = random.uniform(30, 85)
                        
                    except Exception as e:
                        logger.error(
                            "Health check failed for instance",
                            tool_name=self.tool_name,
                            instance_id=instance_id,
                            error=str(e)
                        )
                        self.metrics[instance_id].health_status = False
                
                await asyncio.sleep(self.scaling_config.health_check_interval)
                
            except Exception as e:
                logger.error(
                    "Health monitor error",
                    tool_name=self.tool_name,
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(5)
    
    async def _auto_scaler(self) -> None:
        """Automatically scale instances based on metrics."""
        while not self._shutdown_event.is_set():
            try:
                if self.scaling_in_progress:
                    await asyncio.sleep(10)
                    continue
                
                # Calculate average metrics
                healthy_instances = [
                    metrics for metrics in self.metrics.values()
                    if metrics.health_status
                ]
                
                if not healthy_instances:
                    await asyncio.sleep(30)
                    continue
                
                avg_cpu = sum(m.cpu_usage for m in healthy_instances) / len(healthy_instances)
                avg_memory = sum(m.memory_usage for m in healthy_instances) / len(healthy_instances)
                avg_connections = sum(m.active_connections for m in healthy_instances) / len(healthy_instances)
                
                current_time = time.time()
                
                # Scale up decision
                should_scale_up = (
                    avg_cpu > self.scaling_config.scale_up_threshold or
                    avg_memory > self.scaling_config.target_memory_utilization or
                    avg_connections > 5  # Arbitrary threshold
                )
                
                if (should_scale_up and 
                    len(self.instances) < self.scaling_config.max_instances and
                    current_time - self.last_scale_up > self.scaling_config.scale_up_cooldown):
                    
                    await self._scale_up()
                
                # Scale down decision
                should_scale_down = (
                    avg_cpu < self.scaling_config.scale_down_threshold and
                    avg_memory < self.scaling_config.scale_down_threshold and
                    avg_connections < 2
                )
                
                if (should_scale_down and 
                    len(self.instances) > self.scaling_config.min_instances and
                    current_time - self.last_scale_down > self.scaling_config.scale_down_cooldown):
                    
                    await self._scale_down()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(
                    "Auto scaler error",
                    tool_name=self.tool_name,
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(30)
    
    async def _scale_up(self) -> None:
        """Scale up by adding a new instance."""
        self.scaling_in_progress = True
        try:
            logger.info(
                "Scaling up tool pool",
                tool_name=self.tool_name,
                current_instances=len(self.instances)
            )
            
            instance_id = await self._create_instance()
            if instance_id:
                self.last_scale_up = time.time()
                logger.info(
                    "Scaled up successfully",
                    tool_name=self.tool_name,
                    new_instance_id=instance_id,
                    total_instances=len(self.instances)
                )
        finally:
            self.scaling_in_progress = False
    
    async def _scale_down(self) -> None:
        """Scale down by removing an instance."""
        self.scaling_in_progress = True
        try:
            # Find instance with least connections
            candidate_instances = [
                (instance_id, metrics) for instance_id, metrics in self.metrics.items()
                if metrics.active_connections == 0
            ]
            
            if candidate_instances:
                # Remove instance with least total requests
                instance_id, _ = min(candidate_instances, key=lambda x: x[1].total_requests)
                
                logger.info(
                    "Scaling down tool pool",
                    tool_name=self.tool_name,
                    removing_instance=instance_id,
                    current_instances=len(self.instances)
                )
                
                if await self._remove_instance(instance_id):
                    self.last_scale_down = time.time()
                    logger.info(
                        "Scaled down successfully",
                        tool_name=self.tool_name,
                        removed_instance=instance_id,
                        remaining_instances=len(self.instances)
                    )
        finally:
            self.scaling_in_progress = False
    
    def _generate_instance_id(self) -> str:
        """Generate a unique instance ID."""
        return f"{self.tool_name}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    
    def _get_instance_id(self, instance: ITool) -> Optional[str]:
        """Get instance ID for a tool instance."""
        for instance_id, stored_instance in self.instances.items():
            if stored_instance is instance:
                return instance_id
        return None
    
    async def shutdown(self) -> None:
        """Shutdown the tool pool."""
        logger.info("Shutting down tool pool", tool_name=self.tool_name)
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
        if self._auto_scaler_task:
            self._auto_scaler_task.cancel()
        
        # Shutdown all instances
        for instance_id, instance in list(self.instances.items()):
            await self._remove_instance(instance_id)
        
        logger.info("Tool pool shutdown completed", tool_name=self.tool_name)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status and metrics."""
        healthy_count = sum(1 for m in self.metrics.values() if m.health_status)
        total_connections = sum(m.active_connections for m in self.metrics.values())
        
        return {
            "tool_name": self.tool_name,
            "total_instances": len(self.instances),
            "healthy_instances": healthy_count,
            "total_active_connections": total_connections,
            "scaling_config": {
                "min_instances": self.scaling_config.min_instances,
                "max_instances": self.scaling_config.max_instances,
                "load_balancing_strategy": self.scaling_config.load_balancing_strategy.value
            },
            "instances": [
                {
                    "instance_id": instance_id,
                    "health_status": metrics.health_status,
                    "active_connections": metrics.active_connections,
                    "total_requests": metrics.total_requests,
                    "total_errors": metrics.total_errors,
                    "avg_response_time_ms": metrics.avg_response_time_ms,
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage
                }
                for instance_id, metrics in self.metrics.items()
            ]
        }


class HorizontalScalingManager:
    """Manager for horizontal scaling of tool instances."""
    
    def __init__(self):
        self.tool_pools: Dict[str, ToolInstancePool] = {}
        self.global_metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "avg_response_time_ms": 0.0
        }
        
        logger.info("Horizontal scaling manager initialized")
    
    async def register_tool(
        self,
        tool_name: str,
        tool_factory: Callable[[], ITool],
        scaling_config: Optional[ScalingConfig] = None
    ) -> bool:
        """Register a tool for horizontal scaling."""
        if tool_name in self.tool_pools:
            logger.warning(
                "Tool already registered for scaling",
                tool_name=tool_name
            )
            return False
        
        config = scaling_config or ScalingConfig()
        pool = ToolInstancePool(tool_name, tool_factory, config)
        
        if await pool.initialize():
            self.tool_pools[tool_name] = pool
            logger.info(
                "Tool registered for horizontal scaling",
                tool_name=tool_name,
                min_instances=config.min_instances,
                max_instances=config.max_instances
            )
            return True
        else:
            logger.error(
                "Failed to register tool for scaling",
                tool_name=tool_name
            )
            return False
    
    async def get_tool_instance(
        self, 
        tool_name: str, 
        context: Optional[AgentContext] = None
    ) -> Optional[ITool]:
        """Get a tool instance with load balancing."""
        if tool_name not in self.tool_pools:
            logger.error(
                "Tool not registered for scaling",
                tool_name=tool_name
            )
            return None
        
        return await self.tool_pools[tool_name].get_instance(context)
    
    async def release_tool_instance(self, tool_name: str, instance: ITool) -> None:
        """Release a tool instance back to the pool."""
        if tool_name in self.tool_pools:
            await self.tool_pools[tool_name].release_instance(instance)
    
    async def shutdown(self) -> None:
        """Shutdown all tool pools."""
        logger.info("Shutting down horizontal scaling manager")
        
        for pool in self.tool_pools.values():
            await pool.shutdown()
        
        self.tool_pools.clear()
        logger.info("Horizontal scaling manager shutdown completed")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get overall scaling status."""
        return {
            "registered_tools": list(self.tool_pools.keys()),
            "total_pools": len(self.tool_pools),
            "global_metrics": self.global_metrics,
            "pools": {
                tool_name: pool.get_pool_status()
                for tool_name, pool in self.tool_pools.items()
            }
        }


# Global scaling manager instance
scaling_manager = HorizontalScalingManager()
