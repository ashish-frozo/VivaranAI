"""
Production Load Balancer for MedBillGuard Tools.

Provides intelligent load balancing, health-aware routing, and performance optimization
for distributed tool operations.
"""

import asyncio
import time
import structlog
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

from agents.interfaces import ITool, AgentContext
from .scaling import scaling_manager, LoadBalancingStrategy
from .monitoring import metrics_collector

logger = structlog.get_logger(__name__)


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    health_check_interval: int = 30
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    sticky_session_timeout: int = 3600  # 1 hour
    performance_window_size: int = 100
    latency_threshold_ms: float = 5000.0
    error_rate_threshold: float = 0.1


@dataclass
class InstancePerformance:
    """Performance metrics for a tool instance."""
    instance_id: str
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_count: int = 0
    last_error_time: float = 0.0
    circuit_breaker_open: bool = False
    circuit_breaker_open_time: float = 0.0
    
    @property
    def avg_response_time(self) -> float:
        """Get average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def error_rate(self) -> float:
        """Get error rate."""
        total_requests = self.error_count + self.success_count
        return self.error_count / total_requests if total_requests > 0 else 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if instance is healthy."""
        return not self.circuit_breaker_open and self.error_rate < 0.5


class StickySessionManager:
    """Manages sticky sessions for consistent routing."""
    
    def __init__(self, timeout: int = 3600):
        self.timeout = timeout
        self.sessions: Dict[str, Tuple[str, float]] = {}  # session_id -> (instance_id, timestamp)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup()
    
    def _start_cleanup(self):
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = [
                    session_id for session_id, (_, timestamp) in self.sessions.items()
                    if current_time - timestamp > self.timeout
                ]
                
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                
                if expired_sessions:
                    logger.debug(
                        "Cleaned up expired sticky sessions",
                        expired_count=len(expired_sessions)
                    )
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(
                    "Error in sticky session cleanup",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(60)
    
    def get_instance_for_session(self, session_id: str) -> Optional[str]:
        """Get instance ID for a session."""
        if session_id in self.sessions:
            instance_id, timestamp = self.sessions[session_id]
            # Check if session is still valid
            if time.time() - timestamp < self.timeout:
                return instance_id
            else:
                del self.sessions[session_id]
        return None
    
    def bind_session_to_instance(self, session_id: str, instance_id: str):
        """Bind a session to an instance."""
        self.sessions[session_id] = (instance_id, time.time())
    
    def remove_session(self, session_id: str):
        """Remove a session."""
        self.sessions.pop(session_id, None)
    
    async def shutdown(self):
        """Shutdown the session manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        self.sessions.clear()


class IntelligentLoadBalancer:
    """Intelligent load balancer with health-aware routing and performance optimization."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.performance_metrics: Dict[str, InstancePerformance] = {}
        self.sticky_session_manager = StickySessionManager(self.config.sticky_session_timeout)
        
        # Background tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._circuit_breaker_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info("Intelligent load balancer initialized")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        self._health_monitor_task = asyncio.create_task(self._health_monitor())
        self._circuit_breaker_task = asyncio.create_task(self._circuit_breaker_monitor())
    
    async def route_request(
        self,
        tool_name: str,
        operation: str,
        context: AgentContext,
        **kwargs
    ) -> Tuple[Optional[ITool], Optional[str]]:
        """Route a request to the best available tool instance."""
        
        # Check for sticky session
        session_id = self._get_session_id(context)
        if session_id:
            sticky_instance_id = self.sticky_session_manager.get_instance_for_session(session_id)
            if sticky_instance_id:
                instance = await scaling_manager.get_tool_instance(tool_name, context)
                if instance and self._is_instance_healthy(sticky_instance_id):
                    logger.debug(
                        "Using sticky session routing",
                        tool_name=tool_name,
                        session_id=session_id,
                        instance_id=sticky_instance_id
                    )
                    return instance, sticky_instance_id
        
        # Get available instances from scaling manager
        instance = await scaling_manager.get_tool_instance(tool_name, context)
        if not instance:
            logger.warning(
                "No available instances for tool",
                tool_name=tool_name,
                operation=operation
            )
            return None, None
        
        # Get instance ID (simplified - in real implementation would track this better)
        instance_id = f"{tool_name}_{id(instance)}"
        
        # Initialize performance tracking if needed
        if instance_id not in self.performance_metrics:
            self.performance_metrics[instance_id] = InstancePerformance(instance_id=instance_id)
        
        # Bind session if needed
        if session_id:
            self.sticky_session_manager.bind_session_to_instance(session_id, instance_id)
        
        logger.debug(
            "Routed request to instance",
            tool_name=tool_name,
            operation=operation,
            instance_id=instance_id
        )
        
        return instance, instance_id
    
    async def execute_with_load_balancing(
        self,
        tool_name: str,
        operation: str,
        context: AgentContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute a tool operation with intelligent load balancing."""
        
        start_time = time.time()
        instance, instance_id = await self.route_request(tool_name, operation, context, **kwargs)
        
        if not instance or not instance_id:
            return {
                "success": False,
                "error": "No available instances",
                "tool_name": tool_name,
                "load_balancer_error": True
            }
        
        try:
            # Execute the operation
            result = await instance.execute(operation, context, **kwargs)
            
            # Record performance metrics
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            self._record_success(instance_id, execution_time)
            
            # Add load balancer metadata
            result["load_balancer_metadata"] = {
                "instance_id": instance_id,
                "execution_time_ms": execution_time,
                "routing_strategy": "intelligent"
            }
            
            return result
            
        except Exception as e:
            # Record error metrics
            execution_time = (time.time() - start_time) * 1000
            self._record_error(instance_id, execution_time)
            
            logger.error(
                "Load balanced execution failed",
                tool_name=tool_name,
                operation=operation,
                instance_id=instance_id,
                error=str(e),
                execution_time_ms=execution_time,
                exc_info=True
            )
            
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "load_balancer_metadata": {
                    "instance_id": instance_id,
                    "execution_time_ms": execution_time,
                    "routing_strategy": "intelligent",
                    "error_type": type(e).__name__
                }
            }
        
        finally:
            # Release instance back to pool
            await scaling_manager.release_tool_instance(tool_name, instance)
    
    def _get_session_id(self, context: AgentContext) -> Optional[str]:
        """Extract session ID from context."""
        if hasattr(context, 'user_id') and context.user_id:
            return str(context.user_id)
        elif hasattr(context, 'session_id') and context.session_id:
            return str(context.session_id)
        return None
    
    def _is_instance_healthy(self, instance_id: str) -> bool:
        """Check if an instance is healthy."""
        if instance_id not in self.performance_metrics:
            return True  # Assume healthy if no metrics yet
        
        metrics = self.performance_metrics[instance_id]
        return metrics.is_healthy
    
    def _record_success(self, instance_id: str, response_time_ms: float):
        """Record a successful operation."""
        if instance_id not in self.performance_metrics:
            self.performance_metrics[instance_id] = InstancePerformance(instance_id=instance_id)
        
        metrics = self.performance_metrics[instance_id]
        metrics.response_times.append(response_time_ms)
        metrics.success_count += 1
        
        # Close circuit breaker if error rate is low
        if metrics.error_rate < self.config.error_rate_threshold:
            metrics.circuit_breaker_open = False
    
    def _record_error(self, instance_id: str, response_time_ms: float):
        """Record a failed operation."""
        if instance_id not in self.performance_metrics:
            self.performance_metrics[instance_id] = InstancePerformance(instance_id=instance_id)
        
        metrics = self.performance_metrics[instance_id]
        metrics.response_times.append(response_time_ms)
        metrics.error_count += 1
        metrics.last_error_time = time.time()
        
        # Open circuit breaker if error rate is high
        if metrics.error_rate > self.config.error_rate_threshold:
            metrics.circuit_breaker_open = True
            metrics.circuit_breaker_open_time = time.time()
            
            logger.warning(
                "Circuit breaker opened for instance",
                instance_id=instance_id,
                error_rate=metrics.error_rate,
                error_count=metrics.error_count
            )
    
    async def _health_monitor(self):
        """Monitor health of all instances."""
        while not self._shutdown_event.is_set():
            try:
                # Get current scaling status
                scaling_status = scaling_manager.get_scaling_status()
                
                # Update performance metrics based on scaling manager data
                for tool_name, pool_status in scaling_status.get("pools", {}).items():
                    for instance_data in pool_status.get("instances", []):
                        instance_id = instance_data["instance_id"]
                        
                        if instance_id not in self.performance_metrics:
                            self.performance_metrics[instance_id] = InstancePerformance(
                                instance_id=instance_id
                            )
                        
                        # Update metrics from scaling manager
                        metrics = self.performance_metrics[instance_id]
                        if not instance_data["health_status"]:
                            metrics.circuit_breaker_open = True
                            metrics.circuit_breaker_open_time = time.time()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(
                    "Health monitor error",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(30)
    
    async def _circuit_breaker_monitor(self):
        """Monitor and manage circuit breakers."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                for instance_id, metrics in self.performance_metrics.items():
                    if (metrics.circuit_breaker_open and 
                        current_time - metrics.circuit_breaker_open_time > self.config.circuit_breaker_timeout):
                        
                        # Try to close circuit breaker
                        metrics.circuit_breaker_open = False
                        logger.info(
                            "Circuit breaker closed for instance",
                            instance_id=instance_id,
                            timeout_seconds=self.config.circuit_breaker_timeout
                        )
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(
                    "Circuit breaker monitor error",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """Shutdown the load balancer."""
        logger.info("Shutting down intelligent load balancer")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
        if self._circuit_breaker_task:
            self._circuit_breaker_task.cancel()
        
        # Shutdown sticky session manager
        await self.sticky_session_manager.shutdown()
        
        logger.info("Intelligent load balancer shutdown completed")
    
    def get_load_balancer_status(self) -> Dict[str, Any]:
        """Get current load balancer status."""
        healthy_instances = sum(1 for m in self.performance_metrics.values() if m.is_healthy)
        total_instances = len(self.performance_metrics)
        
        return {
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "circuit_breakers_open": sum(1 for m in self.performance_metrics.values() if m.circuit_breaker_open),
            "sticky_sessions": len(self.sticky_session_manager.sessions),
            "config": {
                "health_check_interval": self.config.health_check_interval,
                "circuit_breaker_threshold": self.config.circuit_breaker_threshold,
                "circuit_breaker_timeout": self.config.circuit_breaker_timeout,
                "sticky_session_timeout": self.config.sticky_session_timeout
            },
            "instance_performance": [
                {
                    "instance_id": instance_id,
                    "avg_response_time_ms": metrics.avg_response_time,
                    "error_rate": metrics.error_rate,
                    "success_count": metrics.success_count,
                    "error_count": metrics.error_count,
                    "circuit_breaker_open": metrics.circuit_breaker_open,
                    "is_healthy": metrics.is_healthy
                }
                for instance_id, metrics in self.performance_metrics.items()
            ]
        }


# Global load balancer instance
load_balancer = IntelligentLoadBalancer()
