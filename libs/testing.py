"""
Testing utilities for MedBillGuardAgent.

Provides decorators and helpers for performance testing and snapshot testing.
"""

import time
import functools
import json
from pathlib import Path
from typing import Any, Callable, Optional
import pytest


def timed_test(max_ms: int):
    """
    Decorator to enforce maximum execution time for tests.
    
    Args:
        max_ms: Maximum allowed execution time in milliseconds
        
    Raises:
        AssertionError: If test execution exceeds max_ms
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                if elapsed_ms > max_ms:
                    pytest.fail(
                        f"Test {func.__name__} took {elapsed_ms:.1f}ms, "
                        f"exceeding limit of {max_ms}ms"
                    )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                elapsed_ms = (end_time - start_time) * 1000
                if elapsed_ms > max_ms:
                    pytest.fail(
                        f"Test {func.__name__} took {elapsed_ms:.1f}ms, "
                        f"exceeding limit of {max_ms}ms"
                    )
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def snapshot_test(snapshot_file: str, update_snapshots: bool = False):
    """
    Decorator for snapshot testing - compare output with golden files.
    
    Args:
        snapshot_file: Path to snapshot file relative to fixtures/
        update_snapshots: If True, update snapshot files instead of comparing
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Convert result to JSON for comparison
            if hasattr(result, 'dict'):
                # Pydantic model
                result_data = result.dict()
            elif hasattr(result, '__dict__'):
                # Regular object
                result_data = result.__dict__
            else:
                # Primitive or dict
                result_data = result
            
            snapshot_path = Path("fixtures") / snapshot_file
            
            if update_snapshots:
                # Update snapshot file
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                with open(snapshot_path, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)
                print(f"Updated snapshot: {snapshot_path}")
            else:
                # Compare with existing snapshot
                if not snapshot_path.exists():
                    pytest.fail(f"Snapshot file not found: {snapshot_path}")
                
                with open(snapshot_path, 'r') as f:
                    expected_data = json.load(f)
                
                assert result_data == expected_data, (
                    f"Snapshot mismatch for {snapshot_file}. "
                    f"Run with update_snapshots=True to update."
                )
            
            return result
        return wrapper
    return decorator


class PerformanceTracker:
    """Track performance metrics across test runs."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, test_name: str, duration_ms: float, metadata: Optional[dict] = None):
        """Record a performance metric."""
        self.metrics[test_name] = {
            "duration_ms": duration_ms,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
    
    def get_stats(self) -> dict:
        """Get performance statistics."""
        if not self.metrics:
            return {}
        
        durations = [m["duration_ms"] for m in self.metrics.values()]
        return {
            "total_tests": len(self.metrics),
            "avg_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "tests": self.metrics
        }
    
    def save_report(self, filepath: str):
        """Save performance report to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)


# Global performance tracker instance
perf_tracker = PerformanceTracker()


def benchmark_test(func: Callable) -> Callable:
    """
    Decorator to benchmark test execution and record metrics.
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            perf_tracker.record(func.__name__, duration_ms)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            perf_tracker.record(func.__name__, duration_ms)
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper 