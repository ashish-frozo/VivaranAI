"""
Tool Registry - Dynamic tool discovery and lifecycle management.

Provides centralized registration, discovery, and health monitoring of tools
in the MedBillGuard multi-agent system.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Gauge, Counter

from .interfaces import ITool, ToolState, ToolCapabilityDeclaration
from .redis_state import RedisStateManager

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# Prometheus metrics
REGISTERED_TOOLS = Gauge(
    'registered_tools_total',
    'Total number of registered tools',
    ['status']
)

TOOL_REGISTRATIONS = Counter(
    'tool_registrations_total',
    'Total tool registration events',
    ['tool_name', 'status']
)

TOOL_EXECUTIONS = Counter(
    'tool_executions_total',
    'Total tool executions',
    ['tool_name', 'operation', 'status']
)


@dataclass
class ToolRegistration:
    """Tool registration record."""
    tool_name: str
    tool_version: str
    description: str
    status: ToolState
    capabilities: ToolCapabilityDeclaration
    registration_time: datetime
    last_heartbeat: datetime
    total_executions: int
    total_errors: int
    avg_execution_time_ms: int
    current_operations: int
    tool_instance: Optional[ITool] = None
    metadata: Dict[str, Any] = None
    
    def to_redis_value(self) -> str:
        """Convert to JSON string for Redis storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data['registration_time'] = self.registration_time.isoformat()
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        # Remove tool_instance (not serializable)
        data.pop('tool_instance', None)
        return json.dumps(data)
    
    @classmethod
    def from_redis_value(cls, value: str) -> "ToolRegistration":
        """Create from Redis JSON string."""
        data = json.loads(value)
        # Convert ISO strings back to datetime
        data['registration_time'] = datetime.fromisoformat(data['registration_time'])
        data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        # Convert enums
        data['status'] = ToolState(data['status'])
        data['capabilities'] = ToolCapabilityDeclaration(**data['capabilities'])
        # tool_instance will be None (not serialized)
        return cls(**data)


class ToolRegistry:
    """Centralized registry for tool discovery and lifecycle management."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/2"):
        self.state_manager = RedisStateManager(redis_url)
        self._tool_cache: Dict[str, ToolRegistration] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_refresh = 0
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.REGISTRATION_TTL = 3600  # 1 hour
        self.HEARTBEAT_INTERVAL = 30  # 30 seconds
        self.HEARTBEAT_TIMEOUT = 90   # 90 seconds
        self.CLEANUP_INTERVAL = 300   # 5 minutes
        
        logger.info("Initialized ToolRegistry")
    
    async def start(self):
        """Start the tool registry service."""
        try:
            await self.state_manager.start()
            self._running = True
            
            # Start background monitoring
            self._monitor_task = asyncio.create_task(self._heartbeat_monitor())
            
            logger.info("ToolRegistry started successfully")
            
        except Exception as e:
            logger.error("Failed to start ToolRegistry", error=str(e), exc_info=True)
            raise
    
    async def stop(self):
        """Stop the tool registry service."""
        try:
            self._running = False
            
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
            
            await self.state_manager.stop()
            
            logger.info("ToolRegistry stopped successfully")
            
        except Exception as e:
            logger.error("Failed to stop ToolRegistry", error=str(e), exc_info=True)
    
    async def register_tool(
        self,
        tool: ITool,
        capabilities: Optional[ToolCapabilityDeclaration] = None
    ) -> bool:
        """
        Register a tool with the registry.
        
        Args:
            tool: ITool instance to register
            capabilities: Tool capability metadata (will be fetched if not provided)
            
        Returns:
            True if registration successful
        """
        try:
            with tracer.start_as_current_span("tool_registry.register_tool") as span:
                tool_name = tool.tool_name
                tool_version = tool.tool_version
                
                span.set_attributes({
                    "tool.name": tool_name,
                    "tool.version": tool_version
                })
                
                # Get capabilities if not provided
                if capabilities is None:
                    capabilities = await tool.get_capabilities()
                
                # Create registration record
                registration = ToolRegistration(
                    tool_name=tool_name,
                    tool_version=tool_version,
                    description=capabilities.description,
                    status=tool.state,
                    capabilities=capabilities,
                    registration_time=datetime.utcnow(),
                    last_heartbeat=datetime.utcnow(),
                    total_executions=0,
                    total_errors=0,
                    avg_execution_time_ms=capabilities.avg_execution_time_ms,
                    current_operations=0,
                    tool_instance=tool,
                    metadata=capabilities.metadata or {}
                )
                
                # Store in Redis
                key = f"tool_registry:{tool_name}:{tool_version}"
                if self.state_manager.redis_client:
                    await self.state_manager.redis_client.setex(
                        key,
                        self.REGISTRATION_TTL,
                        registration.to_redis_value()
                    )
                
                # Update cache
                cache_key = f"{tool_name}:{tool_version}"
                self._tool_cache[cache_key] = registration
                
                # Update metrics
                TOOL_REGISTRATIONS.labels(
                    tool_name=tool_name,
                    status="registered"
                ).inc()
                
                REGISTERED_TOOLS.labels(status="online").inc()
                
                logger.info(
                    "Tool registered successfully",
                    tool_name=tool_name,
                    tool_version=tool_version,
                    capabilities=len(capabilities.supported_operations)
                )
                
                return True
                
        except Exception as e:
            logger.error(
                "Failed to register tool",
                tool_name=getattr(tool, 'tool_name', 'unknown'),
                error=str(e),
                exc_info=True
            )
            
            TOOL_REGISTRATIONS.labels(
                tool_name=getattr(tool, 'tool_name', 'unknown'),
                status="failed"
            ).inc()
            
            return False
    
    async def deregister_tool(self, tool_name: str, tool_version: str) -> bool:
        """
        Deregister a tool from the registry.
        
        Args:
            tool_name: Tool name to deregister
            tool_version: Tool version to deregister
            
        Returns:
            True if deregistration successful
        """
        try:
            with tracer.start_as_current_span("tool_registry.deregister_tool") as span:
                span.set_attributes({
                    "tool.name": tool_name,
                    "tool.version": tool_version
                })
                
                # Remove from Redis
                key = f"tool_registry:{tool_name}:{tool_version}"
                if self.state_manager.redis_client:
                    await self.state_manager.redis_client.delete(key)
                
                # Remove from cache
                cache_key = f"{tool_name}:{tool_version}"
                if cache_key in self._tool_cache:
                    del self._tool_cache[cache_key]
                
                # Update metrics
                TOOL_REGISTRATIONS.labels(
                    tool_name=tool_name,
                    status="deregistered"
                ).inc()
                
                REGISTERED_TOOLS.labels(status="online").dec()
                
                logger.info(
                    "Tool deregistered successfully",
                    tool_name=tool_name,
                    tool_version=tool_version
                )
                
                return True
                
        except Exception as e:
            logger.error(
                "Failed to deregister tool",
                tool_name=tool_name,
                tool_version=tool_version,
                error=str(e),
                exc_info=True
            )
            return False
    
    async def discover_tools(
        self,
        operation: str,
        tool_name: Optional[str] = None,
        max_tools: int = 5
    ) -> List[ToolRegistration]:
        """Discover tools that can handle the required operation."""
        try:
            with tracer.start_as_current_span("tool_registry.discover_tools") as span:
                span.set_attributes({
                    "operation": operation,
                    "tool_name": tool_name or "any",
                    "max_tools": max_tools
                })
                
                await self._refresh_cache_if_needed()
                
                matching_tools = []
                
                for registration in self._tool_cache.values():
                    # Skip if not online
                    if registration.status != ToolState.READY:
                        continue
                    
                    # Filter by tool name if specified
                    if tool_name and registration.tool_name != tool_name:
                        continue
                    
                    # Check if tool supports the operation
                    if operation in registration.capabilities.supported_operations:
                        matching_tools.append(registration)
                
                # Sort by suitability (lower execution time and error rate)
                matching_tools.sort(key=lambda t: (
                    t.avg_execution_time_ms,
                    t.total_errors / max(t.total_executions, 1)
                ))
                
                result = matching_tools[:max_tools]
                
                logger.info(
                    "Tool discovery completed",
                    operation=operation,
                    found_tools=len(result),
                    tool_names=[t.tool_name for t in result]
                )
                
                return result
                
        except Exception as e:
            logger.error(
                "Tool discovery failed",
                operation=operation,
                error=str(e),
                exc_info=True
            )
            return []
    
    async def get_tool_status(self, tool_name: str, tool_version: str) -> Optional[ToolRegistration]:
        """Get current status of a specific tool."""
        cache_key = f"{tool_name}:{tool_version}"
        return self._tool_cache.get(cache_key)
    
    async def list_all_tools(self) -> List[ToolRegistration]:
        """List all registered tools."""
        await self._refresh_cache_if_needed()
        return list(self._tool_cache.values())
    
    async def list_online_tools(self) -> List[ToolRegistration]:
        """List only online tools."""
        await self._refresh_cache_if_needed()
        return [t for t in self._tool_cache.values() if t.status == ToolState.READY]
    
    async def update_tool_metrics(
        self,
        tool_name: str,
        tool_version: str,
        execution_time_ms: int,
        success: bool
    ):
        """Update tool performance metrics after execution."""
        try:
            cache_key = f"{tool_name}:{tool_version}"
            registration = self._tool_cache.get(cache_key)
            
            if registration:
                # Update metrics
                registration.total_executions += 1
                if not success:
                    registration.total_errors += 1
                
                # Update average execution time (exponential moving average)
                alpha = 0.1  # Smoothing factor
                registration.avg_execution_time_ms = int(
                    alpha * execution_time_ms + 
                    (1 - alpha) * registration.avg_execution_time_ms
                )
                
                registration.last_heartbeat = datetime.utcnow()
                
                # Update in Redis
                key = f"tool_registry:{tool_name}:{tool_version}"
                if self.state_manager.redis_client:
                    await self.state_manager.redis_client.setex(
                        key,
                        self.REGISTRATION_TTL,
                        registration.to_redis_value()
                    )
                
                # Update Prometheus metrics
                TOOL_EXECUTIONS.labels(
                    tool_name=tool_name,
                    operation="unknown",  # Could be passed as parameter
                    status="success" if success else "error"
                ).inc()
                
        except Exception as e:
            logger.error(
                "Failed to update tool metrics",
                tool_name=tool_name,
                tool_version=tool_version,
                error=str(e)
            )
    
    async def _refresh_cache_if_needed(self):
        """Refresh tool cache if TTL expired."""
        current_time = time.time()
        if current_time - self._last_cache_refresh > self._cache_ttl:
            await self._refresh_cache()
    
    async def _refresh_cache(self):
        """Refresh tool cache from Redis."""
        try:
            if not self.state_manager.redis_client:
                return
            
            # Get all tool registrations
            pattern = "tool_registry:*"
            keys = await self.state_manager.redis_client.keys(pattern)
            
            new_cache = {}
            
            for key in keys:
                try:
                    value = await self.state_manager.redis_client.get(key)
                    if value:
                        registration = ToolRegistration.from_redis_value(value)
                        cache_key = f"{registration.tool_name}:{registration.tool_version}"
                        new_cache[cache_key] = registration
                except Exception as e:
                    logger.warning(f"Failed to parse tool registration: {key}", error=str(e))
            
            self._tool_cache = new_cache
            self._last_cache_refresh = time.time()
            
            # Update metrics
            REGISTERED_TOOLS.labels(status="online").set(
                len([t for t in new_cache.values() if t.status == ToolState.READY])
            )
            
        except Exception as e:
            logger.error("Failed to refresh tool cache", error=str(e), exc_info=True)
    
    async def _heartbeat_monitor(self):
        """Monitor tool heartbeats and update status."""
        while self._running:
            try:
                await self._check_tool_heartbeats()
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Tool heartbeat monitor error", error=str(e), exc_info=True)
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
    
    async def _check_tool_heartbeats(self):
        """Check tool heartbeats and mark stale tools as offline."""
        current_time = datetime.utcnow()
        timeout_threshold = current_time - timedelta(seconds=self.HEARTBEAT_TIMEOUT)
        
        await self._refresh_cache()
        
        for cache_key, registration in self._tool_cache.items():
            if (registration.status == ToolState.READY and 
                registration.last_heartbeat < timeout_threshold):
                
                logger.warning(
                    "Tool heartbeat timeout - marking as degraded",
                    tool_name=registration.tool_name,
                    tool_version=registration.tool_version,
                    last_heartbeat=registration.last_heartbeat.isoformat()
                )
                
                # Update status
                registration.status = ToolState.DEGRADED
                
                # Update in Redis
                key = f"tool_registry:{registration.tool_name}:{registration.tool_version}"
                if self.state_manager.redis_client:
                    await self.state_manager.redis_client.setex(
                        key,
                        self.REGISTRATION_TTL,
                        registration.to_redis_value()
                    )


# Global tool registry instance
tool_registry = ToolRegistry()
