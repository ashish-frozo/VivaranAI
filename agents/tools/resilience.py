"""
Resilience patterns for tool operations.

Implements circuit breakers, rate limiting, retry logic, and other
production-ready resilience patterns for tool operations.
"""

import asyncio
import time
import structlog
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading

logger = structlog.get_logger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout_seconds: float = 30.0  # Operation timeout


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests: int = 100  # Max requests per window
    window_seconds: int = 60  # Time window in seconds
    burst_allowance: int = 10  # Extra requests allowed in burst


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay between retries
    max_delay: float = 60.0  # Maximum delay between retries
    exponential_base: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to delays


class CircuitBreaker:
    """
    Circuit breaker implementation for tool operations.
    
    Prevents cascading failures by temporarily blocking requests
    to failing services and allowing them time to recover.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self._lock = threading.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized", config=config)
    
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
                else:
                    # Try half-open
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            # Record success
            await self._record_success()
            return result
            
        except asyncio.TimeoutError:
            await self._record_failure()
            raise CircuitBreakerTimeoutError(
                f"Operation timed out after {self.config.timeout_seconds}s"
            )
        except Exception as e:
            await self._record_failure()
            raise
    
    async def _record_success(self):
        """Record successful operation."""
        with self._lock:
            self.last_success_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    logger.info(f"Circuit breaker '{self.name}' CLOSED (recovered)")
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    async def _record_failure(self):
        """Record failed operation."""
        with self._lock:
            self.last_failure_time = time.time()
            self.failure_count += 1
            
            if (self.state == CircuitBreakerState.CLOSED and 
                self.failure_count >= self.config.failure_threshold):
                self.state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker '{self.name}' OPENED",
                    failure_count=self.failure_count
                )
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' failed in HALF_OPEN, back to OPEN")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "last_success_time": self.last_success_time,
                "config": self.config
            }


class RateLimiter:
    """
    Token bucket rate limiter for tool operations.
    
    Prevents overwhelming services with too many requests
    and provides burst capacity for occasional spikes.
    """
    
    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        self.tokens = config.max_requests
        self.last_refill = time.time()
        self._lock = threading.Lock()
        
        # Track request timestamps for window-based limiting
        self.request_times: deque = deque()
        
        logger.info(f"Rate limiter '{name}' initialized", config=config)
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the rate limiter."""
        with self._lock:
            now = time.time()
            
            # Refill tokens based on time elapsed
            self._refill_tokens(now)
            
            # Clean old requests from sliding window
            self._clean_old_requests(now)
            
            # Check if we can allow this request
            if (len(self.request_times) + tokens <= self.config.max_requests and
                self.tokens >= tokens):
                
                self.tokens -= tokens
                self.request_times.extend([now] * tokens)
                return True
            
            return False
    
    def _refill_tokens(self, now: float):
        """Refill tokens based on elapsed time."""
        elapsed = now - self.last_refill
        if elapsed > 0:
            # Calculate tokens to add (rate per second)
            tokens_to_add = elapsed * (self.config.max_requests / self.config.window_seconds)
            self.tokens = min(
                self.config.max_requests + self.config.burst_allowance,
                self.tokens + tokens_to_add
            )
            self.last_refill = now
    
    def _clean_old_requests(self, now: float):
        """Remove requests outside the current window."""
        cutoff = now - self.config.window_seconds
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current rate limiter state."""
        with self._lock:
            return {
                "name": self.name,
                "tokens_available": self.tokens,
                "requests_in_window": len(self.request_times),
                "config": self.config
            }


class RetryHandler:
    """
    Exponential backoff retry handler with jitter.
    
    Implements intelligent retry logic with exponential backoff
    and jitter to prevent thundering herd problems.
    """
    
    def __init__(self, name: str, config: RetryConfig):
        self.name = name
        self.config = config
        logger.info(f"Retry handler '{name}' initialized", config=config)
    
    async def execute(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.info(
                        f"Retrying {self.name} (attempt {attempt + 1}/{self.config.max_attempts})",
                        delay=delay
                    )
                    await asyncio.sleep(delay)
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(f"Retry successful for {self.name} on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1} failed for {self.name}",
                    error=str(e),
                    remaining_attempts=self.config.max_attempts - attempt - 1
                )
                
                # Don't retry on certain types of errors
                if self._is_non_retryable_error(e):
                    logger.info(f"Non-retryable error for {self.name}, not retrying")
                    break
        
        # All retries exhausted
        logger.error(f"All retry attempts exhausted for {self.name}")
        raise RetryExhaustedError(
            f"Failed after {self.config.max_attempts} attempts"
        ) from last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)
        
        if self.config.jitter:
            # Add random jitter (±25%)
            import random
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter
        
        return max(0, delay)
    
    def _is_non_retryable_error(self, error: Exception) -> bool:
        """Check if error should not be retried."""
        # Don't retry validation errors, authentication errors, etc.
        non_retryable_types = (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
        )
        
        return isinstance(error, non_retryable_types)


class ResilienceManager:
    """
    Centralized resilience management for tools.
    
    Coordinates circuit breakers, rate limiters, and retry handlers
    to provide comprehensive resilience for tool operations.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self._lock = threading.Lock()
        
        logger.info("Resilience manager initialized")
    
    def get_circuit_breaker(
        self, 
        name: str, 
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker."""
        with self._lock:
            if name not in self.circuit_breakers:
                config = config or CircuitBreakerConfig()
                self.circuit_breakers[name] = CircuitBreaker(name, config)
            return self.circuit_breakers[name]
    
    def get_rate_limiter(
        self, 
        name: str, 
        config: Optional[RateLimitConfig] = None
    ) -> RateLimiter:
        """Get or create rate limiter."""
        with self._lock:
            if name not in self.rate_limiters:
                config = config or RateLimitConfig()
                self.rate_limiters[name] = RateLimiter(name, config)
            return self.rate_limiters[name]
    
    def get_retry_handler(
        self, 
        name: str, 
        config: Optional[RetryConfig] = None
    ) -> RetryHandler:
        """Get or create retry handler."""
        with self._lock:
            if name not in self.retry_handlers:
                config = config or RetryConfig()
                self.retry_handlers[name] = RetryHandler(name, config)
            return self.retry_handlers[name]
    
    async def execute_with_resilience(
        self,
        name: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        **kwargs
    ) -> Any:
        """Execute function with full resilience protection."""
        
        # Get resilience components
        rate_limiter = self.get_rate_limiter(name, rate_limit_config)
        circuit_breaker = self.get_circuit_breaker(name, circuit_breaker_config)
        retry_handler = self.get_retry_handler(name, retry_config)
        
        # Check rate limit
        if not await rate_limiter.acquire():
            raise RateLimitExceededError(f"Rate limit exceeded for {name}")
        
        # Execute with circuit breaker and retry
        async def protected_execution():
            return await circuit_breaker.call(func, *args, **kwargs)
        
        return await retry_handler.execute(protected_execution)
    
    def get_system_state(self) -> Dict[str, Any]:
        """Get state of all resilience components."""
        with self._lock:
            return {
                "circuit_breakers": {
                    name: cb.get_state() 
                    for name, cb in self.circuit_breakers.items()
                },
                "rate_limiters": {
                    name: rl.get_state() 
                    for name, rl in self.rate_limiters.items()
                },
                "retry_handlers": list(self.retry_handlers.keys())
            }


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open."""
    pass


class CircuitBreakerTimeoutError(Exception):
    """Operation timed out."""
    pass


class RateLimitExceededError(Exception):
    """Rate limit exceeded."""
    pass


class RetryExhaustedError(Exception):
    """All retry attempts exhausted."""
    pass


# Global resilience manager
resilience_manager = ResilienceManager()
