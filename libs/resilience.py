#!/usr/bin/env python3
"""
Advanced Resilience Library for VivaranAI Production

This module provides circuit breakers, retry mechanisms, timeout handling,
and fault tolerance patterns for production-grade error handling.
"""

import asyncio
import functools
import logging
import random
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, Awaitable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from collections import defaultdict, deque

import structlog

logger = structlog.get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, requests fail fast
    HALF_OPEN = "half_open" # Testing if service is back


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Failures before opening circuit
    recovery_timeout: float = 60.0      # Seconds before trying half-open
    expected_exception: Type[Exception] = Exception
    success_threshold: int = 3          # Successes to close from half-open
    request_timeout: float = 30.0       # Individual request timeout
    monitor_requests: int = 100         # Number of requests to monitor
    failure_rate_threshold: float = 0.5  # Failure rate to open circuit


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class RetryConfig:
    """Configuration for retry mechanism"""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True,
                 exponential_base: float = 2.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.exponential_base = exponential_base


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_count = 0
        self.recent_requests = deque(maxlen=config.monitor_requests)
        self._lock = threading.Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = 0
        
        logger.info(f"🔧 Circuit breaker '{name}' initialized", 
                   threshold=config.failure_threshold,
                   timeout=config.recovery_timeout)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if asyncio.iscoroutinefunction(func):
            return asyncio.run(self._async_call(func, *args, **kwargs))
        else:
            return self._sync_call(func, *args, **kwargs)
    
    def _sync_call(self, func: Callable, *args, **kwargs) -> Any:
        """Synchronous call with circuit breaker"""
        with self._lock:
            self.total_requests += 1
            
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.total_failures += 1
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
            
            # Execute function
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                self._record_success(execution_time)
                return result
                
            except self.config.expected_exception as e:
                self._record_failure(str(e))
                raise
            except Exception as e:
                # Unexpected exceptions don't count as failures
                logger.warning(f"Unexpected exception in circuit breaker '{self.name}': {e}")
                raise
    
    async def _async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Asynchronous call with circuit breaker"""
        with self._lock:
            self.total_requests += 1
            
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.total_failures += 1
                    raise CircuitBreakerError(f"Circuit breaker '{self.name}' is OPEN")
        
        # Execute function
        try:
            start_time = time.time()
            
            # Apply timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=self.config.request_timeout
            )
            
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            return result
            
        except asyncio.TimeoutError:
            self._record_failure("Request timeout")
            raise
        except self.config.expected_exception as e:
            self._record_failure(str(e))
            raise
        except Exception as e:
            # Unexpected exceptions don't count as failures
            logger.warning(f"Unexpected exception in circuit breaker '{self.name}': {e}")
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.now() - self.last_failure_time).total_seconds() > self.config.recovery_timeout
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.state_changes += 1
        logger.info(f"🔄 Circuit breaker '{self.name}' transitioned to HALF_OPEN")
    
    def _record_success(self, execution_time: float):
        """Record successful request"""
        with self._lock:
            self.total_successes += 1
            self.recent_requests.append(('success', time.time(), execution_time))
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    def _record_failure(self, error: str):
        """Record failed request"""
        with self._lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            self.recent_requests.append(('failure', time.time(), error))
            
            if self.state == CircuitBreakerState.CLOSED:
                if self._should_open_circuit():
                    self._transition_to_open()
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened"""
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check failure rate
        if len(self.recent_requests) >= self.config.monitor_requests:
            failures = sum(1 for req in self.recent_requests if req[0] == 'failure')
            failure_rate = failures / len(self.recent_requests)
            return failure_rate >= self.config.failure_rate_threshold
        
        return False
    
    def _transition_to_open(self):
        """Transition circuit breaker to open state"""
        self.state = CircuitBreakerState.OPEN
        self.state_changes += 1
        logger.warning(f"⚠️  Circuit breaker '{self.name}' transitioned to OPEN", 
                      failures=self.failure_count)
    
    def _transition_to_closed(self):
        """Transition circuit breaker to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_changes += 1
        logger.info(f"✅ Circuit breaker '{self.name}' transitioned to CLOSED")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'total_requests': self.total_requests,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'state_changes': self.state_changes,
            'failure_rate': self.total_failures / max(self.total_requests, 1),
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class RetryMechanism:
    """Advanced retry mechanism with exponential backoff"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        if asyncio.iscoroutinefunction(func):
            return asyncio.run(self._async_retry(func, *args, **kwargs))
        else:
            return self._sync_retry(func, *args, **kwargs)
    
    def _sync_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Synchronous retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"❌ All {self.config.max_attempts} retry attempts failed", 
                               function=func.__name__, error=str(e))
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"🔄 Retry attempt {attempt + 1}/{self.config.max_attempts} failed, "
                             f"retrying in {delay:.2f}s", 
                             function=func.__name__, error=str(e))
                
                time.sleep(delay)
        
        raise last_exception
    
    async def _async_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Asynchronous retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.config.max_attempts - 1:
                    logger.error(f"❌ All {self.config.max_attempts} retry attempts failed", 
                               function=func.__name__, error=str(e))
                    raise
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"🔄 Retry attempt {attempt + 1}/{self.config.max_attempts} failed, "
                             f"retrying in {delay:.2f}s", 
                             function=func.__name__, error=str(e))
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** attempt),
            self.config.max_delay
        )
        
        if self.config.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random())
        
        return delay


class TimeoutManager:
    """Advanced timeout handling"""
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.active_timeouts = {}
    
    @contextmanager
    def timeout(self, seconds: float = None):
        """Context manager for timeout handling"""
        timeout_value = seconds or self.default_timeout
        
        if asyncio.iscoroutinefunction:
            # For async contexts, we'd need to handle this differently
            yield timeout_value
        else:
            # Synchronous timeout handling
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {timeout_value} seconds")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout_value))
            
            try:
                yield timeout_value
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)


class FaultTolerantService:
    """Service wrapper with comprehensive fault tolerance"""
    
    def __init__(self, 
                 name: str,
                 circuit_breaker_config: CircuitBreakerConfig = None,
                 retry_config: RetryConfig = None,
                 timeout: float = 30.0):
        self.name = name
        self.circuit_breaker = CircuitBreaker(name, circuit_breaker_config or CircuitBreakerConfig())
        self.retry_mechanism = RetryMechanism(retry_config or RetryConfig())
        self.timeout_manager = TimeoutManager(timeout)
        
        # Metrics
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
        logger.info(f"🛡️  Fault tolerant service '{name}' initialized")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with full fault tolerance"""
        self.request_count += 1
        
        try:
            # Wrap function with retry mechanism
            def retriable_func(*args, **kwargs):
                return self.circuit_breaker.call(func, *args, **kwargs)
            
            return self.retry_mechanism.retry(retriable_func, *args, **kwargs)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Fault tolerant service '{self.name}' failed", 
                        function=func.__name__, error=str(e))
            raise
    
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with full fault tolerance"""
        self.request_count += 1
        
        try:
            # Wrap function with retry mechanism
            async def retriable_func(*args, **kwargs):
                return await self.circuit_breaker._async_call(func, *args, **kwargs)
            
            return await self.retry_mechanism._async_retry(retriable_func, *args, **kwargs)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Fault tolerant service '{self.name}' failed", 
                        function=func.__name__, error=str(e))
            raise
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'service_name': self.name,
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'circuit_breaker': self.circuit_breaker.get_metrics(),
            'status': 'healthy' if self.circuit_breaker.state == CircuitBreakerState.CLOSED else 'degraded'
        }


class ResilienceManager:
    """Central manager for all resilience patterns"""
    
    def __init__(self):
        self.services = {}
        self.global_metrics = {
            'total_requests': 0,
            'total_failures': 0,
            'start_time': datetime.now()
        }
    
    def register_service(self, 
                        name: str,
                        circuit_breaker_config: CircuitBreakerConfig = None,
                        retry_config: RetryConfig = None,
                        timeout: float = 30.0) -> FaultTolerantService:
        """Register a new fault tolerant service"""
        service = FaultTolerantService(name, circuit_breaker_config, retry_config, timeout)
        self.services[name] = service
        
        logger.info(f"📋 Registered fault tolerant service: {name}")
        return service
    
    def get_service(self, name: str) -> Optional[FaultTolerantService]:
        """Get registered service"""
        return self.services.get(name)
    
    def get_global_health(self) -> Dict[str, Any]:
        """Get global system health"""
        total_requests = sum(service.request_count for service in self.services.values())
        total_errors = sum(service.error_count for service in self.services.values())
        
        healthy_services = sum(1 for service in self.services.values() 
                             if service.circuit_breaker.state == CircuitBreakerState.CLOSED)
        
        uptime = (datetime.now() - self.global_metrics['start_time']).total_seconds()
        
        return {
            'total_services': len(self.services),
            'healthy_services': healthy_services,
            'degraded_services': len(self.services) - healthy_services,
            'total_requests': total_requests,
            'total_errors': total_errors,
            'global_error_rate': total_errors / max(total_requests, 1),
            'uptime_seconds': uptime,
            'services': {name: service.get_health_metrics() 
                        for name, service in self.services.items()},
            'overall_health': 'healthy' if healthy_services == len(self.services) else 'degraded'
        }
    
    def shutdown(self):
        """Graceful shutdown of all services"""
        logger.info("🛑 Shutting down resilience manager")
        
        for name, service in self.services.items():
            logger.info(f"📊 Final metrics for {name}: {service.get_health_metrics()}")
        
        self.services.clear()


# Global resilience manager instance
resilience_manager = ResilienceManager()


# Decorators for easy usage
def circuit_breaker(name: str, **kwargs):
    """Decorator for circuit breaker protection"""
    def decorator(func):
        config = CircuitBreakerConfig(**kwargs)
        breaker = CircuitBreaker(name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


def retry(max_attempts: int = 3, **kwargs):
    """Decorator for retry mechanism"""
    def decorator(func):
        config = RetryConfig(max_attempts=max_attempts, **kwargs)
        retry_mechanism = RetryMechanism(config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_mechanism.retry(func, *args, **kwargs)
        
        return wrapper
    return decorator


def fault_tolerant(service_name: str, **kwargs):
    """Decorator for full fault tolerance"""
    def decorator(func):
        service = resilience_manager.register_service(service_name, **kwargs)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return service.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


# Example usage functions
@circuit_breaker("external_api", failure_threshold=3, recovery_timeout=60)
def call_external_api(url: str) -> str:
    """Example function protected by circuit breaker"""
    import requests
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


@retry(max_attempts=3, base_delay=1.0, backoff_factor=2.0)
def unreliable_operation():
    """Example function with retry mechanism"""
    if random.random() < 0.7:  # 70% chance of failure
        raise ValueError("Simulated failure")
    return "Success"


@fault_tolerant("database_service", 
               circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
               retry_config=RetryConfig(max_attempts=3))
def database_operation(query: str):
    """Example database operation with full fault tolerance"""
    # Simulate database operation
    if random.random() < 0.1:  # 10% chance of failure
        raise Exception("Database connection failed")
    return f"Result for: {query}"


if __name__ == "__main__":
    # Example usage
    logger.info("🧪 Testing resilience mechanisms...")
    
    # Test circuit breaker
    try:
        result = call_external_api("https://httpbin.org/get")
        logger.info(f"✅ API call successful: {len(result)} bytes")
    except Exception as e:
        logger.error(f"❌ API call failed: {e}")
    
    # Test retry mechanism
    try:
        result = unreliable_operation()
        logger.info(f"✅ Unreliable operation succeeded: {result}")
    except Exception as e:
        logger.error(f"❌ Unreliable operation failed: {e}")
    
    # Test fault tolerant service
    try:
        result = database_operation("SELECT * FROM users")
        logger.info(f"✅ Database operation succeeded: {result}")
    except Exception as e:
        logger.error(f"❌ Database operation failed: {e}")
    
    # Print global health
    health = resilience_manager.get_global_health()
    logger.info(f"📊 System health: {health}") 