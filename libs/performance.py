#!/usr/bin/env python3
"""
Performance Optimization Library for VivaranAI Production

This module provides comprehensive performance optimization including
advanced caching, query optimization, and performance monitoring.
"""

import asyncio
import functools
import hashlib
import json
import time
import threading
import weakref
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import redis
import psutil
import numpy as np
from contextlib import contextmanager
import concurrent.futures

import structlog

logger = structlog.get_logger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"          # Least Recently Used
    LFU = "lfu"          # Least Frequently Used
    FIFO = "fifo"        # First In, First Out
    TTL = "ttl"          # Time To Live
    ADAPTIVE = "adaptive" # Adaptive based on access patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def update_access(self):
        """Update access metadata"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class PerformanceMetrics:
    """Performance metrics collector"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = datetime.now()
        self._lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a performance metric"""
        with self._lock:
            metric_entry = {
                'timestamp': datetime.now().isoformat(),
                'value': value,
                'tags': tags or {}
            }
            self.metrics[name].append(metric_entry)
            
            # Keep only last 1000 entries per metric
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metric_stats(self, name: str, window_minutes: int = 5) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        if name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [
            entry['value'] for entry in self.metrics[name]
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'avg': sum(recent_values) / len(recent_values),
            'p50': np.percentile(recent_values, 50),
            'p95': np.percentile(recent_values, 95),
            'p99': np.percentile(recent_values, 99)
        }
    
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        for metric_name in self.metrics:
            summary[metric_name] = self.get_metric_stats(metric_name)
        return summary


class AdvancedCache:
    """Advanced in-memory cache with multiple eviction strategies"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU,
                 default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.insertion_order = deque()  # For FIFO
        self._lock = threading.RLock()
        self.metrics = PerformanceMetrics()
        
        logger.info(f"🚀 Advanced cache initialized",
                   max_size=max_size,
                   strategy=strategy.value,
                   default_ttl=default_ttl)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.time()
        
        with self._lock:
            if key not in self.cache:
                self.metrics.record_metric('cache_miss', 1, {'key': key})
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self.metrics.record_metric('cache_miss', 1, {'key': key, 'reason': 'expired'})
                return None
            
            # Update access patterns
            entry.update_access()
            self._update_access_patterns(key)
            
            self.metrics.record_metric('cache_hit', 1, {'key': key})
            self.metrics.record_metric('cache_get_time', time.time() - start_time)
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Set value in cache"""
        start_time = time.time()
        
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                size_bytes=size_bytes,
                ttl=ttl or self.default_ttl
            )
            
            # Check if key already exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.cache[key] = entry
            else:
                # Check if cache is full
                if len(self.cache) >= self.max_size:
                    self._evict_entry()
                
                self.cache[key] = entry
                self.insertion_order.append(key)
            
            # Update access patterns
            self._update_access_patterns(key)
            
            self.metrics.record_metric('cache_set', 1, {'key': key})
            self.metrics.record_metric('cache_set_time', time.time() - start_time)
            self.metrics.record_metric('cache_size', len(self.cache))
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        with self._lock:
            if key in self.cache:
                self._remove_entry(key)
                self.metrics.record_metric('cache_delete', 1, {'key': key})
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_frequency.clear()
            self.insertion_order.clear()
            self.metrics.record_metric('cache_clear', 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size,
                'total_size_bytes': total_size,
                'avg_entry_size': total_size / max(len(self.cache), 1),
                'strategy': self.strategy.value,
                'metrics': self.metrics.get_all_metrics_summary()
            }
    
    def _evict_entry(self):
        """Evict entry based on strategy"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            key_to_evict = next(iter(self.access_order))
        elif self.strategy == CacheStrategy.LFU:
            key_to_evict = min(self.access_frequency.keys(), 
                              key=lambda k: self.access_frequency[k])
        elif self.strategy == CacheStrategy.FIFO:
            key_to_evict = self.insertion_order[0]
        elif self.strategy == CacheStrategy.TTL:
            # Find earliest expiring entry
            key_to_evict = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].created_at + timedelta(seconds=self.cache[k].ttl or float('inf'))
            )
        elif self.strategy == CacheStrategy.ADAPTIVE:
            key_to_evict = self._adaptive_eviction()
        else:
            key_to_evict = next(iter(self.cache))
        
        self._remove_entry(key_to_evict)
        self.metrics.record_metric('cache_eviction', 1, 
                                 {'key': key_to_evict, 'strategy': self.strategy.value})
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on access patterns"""
        # Score entries based on frequency, recency, and size
        scores = {}
        now = datetime.now()
        
        for key, entry in self.cache.items():
            recency_score = (now - entry.last_accessed).total_seconds()
            frequency_score = 1.0 / (entry.access_count + 1)
            size_score = entry.size_bytes / 1024  # Penalize large entries
            
            # Combined score (higher = more likely to evict)
            scores[key] = recency_score + frequency_score + size_score
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _update_access_patterns(self, key: str):
        """Update access patterns for different strategies"""
        if self.strategy == CacheStrategy.LRU:
            self.access_order.move_to_end(key)
            if key not in self.access_order:
                self.access_order[key] = True
        
        if self.strategy in [CacheStrategy.LFU, CacheStrategy.ADAPTIVE]:
            self.access_frequency[key] += 1
    
    def _remove_entry(self, key: str):
        """Remove entry and cleanup access patterns"""
        if key in self.cache:
            del self.cache[key]
        
        self.access_order.pop(key, None)
        self.access_frequency.pop(key, None)
        
        if key in self.insertion_order:
            self.insertion_order.remove(key)
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            return len(json.dumps(value, default=str).encode('utf-8'))
        except:
            return 1024  # Default size estimate


class RedisCache:
    """Redis-backed distributed cache"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 key_prefix: str = "vivaranai:", default_ttl: int = 3600):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.key_prefix = key_prefix
            self.default_ttl = default_ttl
            self.metrics = PerformanceMetrics()
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"🔗 Redis cache connected: {redis_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        start_time = time.time()
        full_key = f"{self.key_prefix}{key}"
        
        try:
            value = self.redis_client.get(full_key)
            
            if value is None:
                self.metrics.record_metric('redis_cache_miss', 1, {'key': key})
                return None
            
            self.metrics.record_metric('redis_cache_hit', 1, {'key': key})
            self.metrics.record_metric('redis_get_time', time.time() - start_time)
            
            return json.loads(value.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"❌ Redis get failed for key {key}: {e}")
            self.metrics.record_metric('redis_error', 1, {'operation': 'get', 'key': key})
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
        
        start_time = time.time()
        full_key = f"{self.key_prefix}{key}"
        ttl = ttl or self.default_ttl
        
        try:
            serialized_value = json.dumps(value, default=str).encode('utf-8')
            result = self.redis_client.setex(full_key, ttl, serialized_value)
            
            self.metrics.record_metric('redis_set', 1, {'key': key})
            self.metrics.record_metric('redis_set_time', time.time() - start_time)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"❌ Redis set failed for key {key}: {e}")
            self.metrics.record_metric('redis_error', 1, {'operation': 'set', 'key': key})
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache"""
        if not self.redis_client:
            return False
        
        full_key = f"{self.key_prefix}{key}"
        
        try:
            result = self.redis_client.delete(full_key)
            self.metrics.record_metric('redis_delete', 1, {'key': key})
            return bool(result)
            
        except Exception as e:
            logger.error(f"❌ Redis delete failed for key {key}: {e}")
            self.metrics.record_metric('redis_error', 1, {'operation': 'delete', 'key': key})
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            full_pattern = f"{self.key_prefix}{pattern}"
            keys = self.redis_client.keys(full_pattern)
            
            if keys:
                result = self.redis_client.delete(*keys)
                self.metrics.record_metric('redis_clear_pattern', len(keys), {'pattern': pattern})
                return result
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Redis clear pattern failed for {pattern}: {e}")
            return 0


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) tiers"""
    
    def __init__(self, l1_cache: AdvancedCache, l2_cache: Optional[RedisCache] = None):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.metrics = PerformanceMetrics()
        
        logger.info("🏗️  Multi-level cache initialized",
                   l1_enabled=True,
                   l2_enabled=l2_cache is not None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache"""
        start_time = time.time()
        
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            self.metrics.record_metric('l1_hit', 1, {'key': key})
            self.metrics.record_metric('cache_get_total_time', time.time() - start_time)
            return value
        
        # Try L2 cache if available
        if self.l2_cache:
            value = self.l2_cache.get(key)
            if value is not None:
                # Promote to L1 cache
                self.l1_cache.set(key, value)
                self.metrics.record_metric('l2_hit', 1, {'key': key})
                self.metrics.record_metric('cache_get_total_time', time.time() - start_time)
                return value
        
        self.metrics.record_metric('cache_miss_all_levels', 1, {'key': key})
        self.metrics.record_metric('cache_get_total_time', time.time() - start_time)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in multi-level cache"""
        start_time = time.time()
        
        # Set in L1 cache
        l1_success = self.l1_cache.set(key, value, ttl)
        
        # Set in L2 cache if available
        l2_success = True
        if self.l2_cache:
            l2_success = self.l2_cache.set(key, value, ttl)
        
        self.metrics.record_metric('cache_set_total_time', time.time() - start_time)
        return l1_success and l2_success
    
    def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        l1_result = self.l1_cache.delete(key)
        l2_result = True
        
        if self.l2_cache:
            l2_result = self.l2_cache.delete(key)
        
        return l1_result or l2_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'l1_cache': self.l1_cache.get_stats(),
            'multilevel_metrics': self.metrics.get_all_metrics_summary()
        }
        
        if self.l2_cache:
            stats['l2_cache'] = {
                'metrics': self.l2_cache.metrics.get_all_metrics_summary()
            }
        
        return stats


class QueryOptimizer:
    """SQL query optimization and analysis"""
    
    def __init__(self):
        self.query_cache = AdvancedCache(max_size=500, strategy=CacheStrategy.LFU)
        self.query_stats = defaultdict(list)
        self.slow_query_threshold = 1.0  # seconds
        self.metrics = PerformanceMetrics()
    
    def optimize_query(self, query: str, params: Tuple = None) -> str:
        """Optimize SQL query"""
        # Generate cache key
        cache_key = self._generate_query_key(query, params)
        
        # Check cache for optimized query
        cached_result = self.query_cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Apply optimization rules
        optimized_query = self._apply_optimization_rules(query)
        
        # Cache optimized query
        self.query_cache.set(cache_key, optimized_query)
        
        return optimized_query
    
    def analyze_query_performance(self, query: str, execution_time: float, 
                                rows_affected: int = 0) -> Dict[str, Any]:
        """Analyze query performance"""
        query_signature = self._get_query_signature(query)
        
        # Record performance metrics
        self.query_stats[query_signature].append({
            'execution_time': execution_time,
            'rows_affected': rows_affected,
            'timestamp': datetime.now().isoformat()
        })
        
        # Check for slow queries
        if execution_time > self.slow_query_threshold:
            self.metrics.record_metric('slow_query', 1, {
                'signature': query_signature,
                'execution_time': execution_time
            })
            
            logger.warning(f"🐌 Slow query detected",
                         execution_time=execution_time,
                         signature=query_signature[:100])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(query, execution_time, rows_affected)
        
        return {
            'query_signature': query_signature,
            'execution_time': execution_time,
            'rows_affected': rows_affected,
            'is_slow': execution_time > self.slow_query_threshold,
            'recommendations': recommendations
        }
    
    def get_query_performance_report(self, limit: int = 10) -> Dict[str, Any]:
        """Generate query performance report"""
        report = {
            'total_queries': sum(len(stats) for stats in self.query_stats.values()),
            'unique_queries': len(self.query_stats),
            'slow_queries': [],
            'top_queries_by_frequency': [],
            'top_queries_by_avg_time': []
        }
        
        # Analyze each query
        query_analysis = []
        for signature, stats in self.query_stats.items():
            if not stats:
                continue
            
            execution_times = [s['execution_time'] for s in stats]
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            
            analysis = {
                'signature': signature,
                'frequency': len(stats),
                'avg_execution_time': avg_time,
                'max_execution_time': max_time,
                'total_execution_time': sum(execution_times)
            }
            
            query_analysis.append(analysis)
            
            # Track slow queries
            if max_time > self.slow_query_threshold:
                report['slow_queries'].append(analysis)
        
        # Sort and limit results
        report['top_queries_by_frequency'] = sorted(
            query_analysis, key=lambda x: x['frequency'], reverse=True
        )[:limit]
        
        report['top_queries_by_avg_time'] = sorted(
            query_analysis, key=lambda x: x['avg_execution_time'], reverse=True
        )[:limit]
        
        report['slow_queries'] = sorted(
            report['slow_queries'], key=lambda x: x['max_execution_time'], reverse=True
        )[:limit]
        
        return report
    
    def _generate_query_key(self, query: str, params: Tuple = None) -> str:
        """Generate cache key for query"""
        key_data = query + str(params or "")
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_query_signature(self, query: str) -> str:
        """Get normalized query signature"""
        # Remove extra whitespace and normalize
        normalized = ' '.join(query.strip().split())
        
        # Replace parameter placeholders with generic markers
        import re
        normalized = re.sub(r'\$\d+', '$?', normalized)  # PostgreSQL parameters
        normalized = re.sub(r'\?', '$?', normalized)     # Generic parameters
        normalized = re.sub(r"'[^']*'", "'?'", normalized)  # String literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)    # Numeric literals
        
        return normalized
    
    def _apply_optimization_rules(self, query: str) -> str:
        """Apply query optimization rules"""
        optimized = query
        
        # Add basic optimizations
        optimization_rules = [
            # Add LIMIT if missing in SELECT without WHERE
            (r'SELECT\s+[^;]*(?<!LIMIT\s+\d+)\s*;?$', 
             lambda m: m.group(0).rstrip(';') + ' LIMIT 1000;'),
            
            # Suggest index hints for common patterns
            (r'WHERE\s+(\w+)\s*=\s*', 
             lambda m: m.group(0)),  # Could add index hints here
        ]
        
        for pattern, replacement in optimization_rules:
            try:
                import re
                optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
            except Exception:
                continue
        
        return optimized
    
    def _generate_recommendations(self, query: str, execution_time: float, 
                                rows_affected: int) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if execution_time > self.slow_query_threshold:
            recommendations.append("Consider adding appropriate indexes")
            recommendations.append("Review WHERE clause for optimization opportunities")
        
        if 'SELECT *' in query.upper():
            recommendations.append("Avoid SELECT *, specify only needed columns")
        
        if 'ORDER BY' in query.upper() and 'LIMIT' not in query.upper():
            recommendations.append("Consider adding LIMIT with ORDER BY")
        
        if rows_affected > 10000:
            recommendations.append("Consider pagination for large result sets")
        
        return recommendations


class PerformanceMonitor:
    """Comprehensive performance monitoring"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = datetime.now()
        self.system_metrics_interval = 60  # seconds
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start background performance monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
        self._monitor_thread.start()
        
        logger.info("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logger.info("📊 Performance monitoring stopped")
    
    @contextmanager
    def measure_time(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager to measure operation time"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.metrics.record_metric(f"{operation_name}_time", execution_time, tags)
    
    def record_database_operation(self, operation_type: str, table: str, 
                                 execution_time: float, rows_affected: int = 0):
        """Record database operation metrics"""
        tags = {'operation': operation_type, 'table': table}
        
        self.metrics.record_metric('db_operation_time', execution_time, tags)
        self.metrics.record_metric('db_rows_affected', rows_affected, tags)
        
        if execution_time > 1.0:  # Slow query threshold
            self.metrics.record_metric('db_slow_query', 1, tags)
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, 
                          response_time: float):
        """Record API request metrics"""
        tags = {
            'endpoint': endpoint,
            'method': method,
            'status_code': str(status_code)
        }
        
        self.metrics.record_metric('api_request_time', response_time, tags)
        self.metrics.record_metric('api_request_count', 1, tags)
        
        if status_code >= 400:
            self.metrics.record_metric('api_error', 1, tags)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        summary = {
            'uptime_seconds': uptime,
            'monitoring_active': self._monitoring,
            'metrics_summary': self.metrics.get_all_metrics_summary(),
            'system_info': self._get_system_info(),
            'recommendations': self._generate_performance_recommendations()
        }
        
        return summary
    
    def _monitor_system_metrics(self):
        """Background thread for system metrics collection"""
        while self._monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics.record_metric('system_cpu_percent', cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.metrics.record_metric('system_memory_percent', memory.percent)
                self.metrics.record_metric('system_memory_available_bytes', memory.available)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.metrics.record_metric('system_disk_percent', disk.percent)
                self.metrics.record_metric('system_disk_free_bytes', disk.free)
                
                # Network metrics (if available)
                try:
                    network = psutil.net_io_counters()
                    self.metrics.record_metric('system_network_bytes_sent', network.bytes_sent)
                    self.metrics.record_metric('system_network_bytes_recv', network.bytes_recv)
                except:
                    pass
                
                time.sleep(self.system_metrics_interval)
                
            except Exception as e:
                logger.error(f"❌ System metrics collection failed: {e}")
                time.sleep(self.system_metrics_interval)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3),
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system info: {e}")
            return {}
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze recent metrics for recommendations
        cpu_stats = self.metrics.get_metric_stats('system_cpu_percent')
        memory_stats = self.metrics.get_metric_stats('system_memory_percent')
        
        if cpu_stats and cpu_stats.get('avg', 0) > 80:
            recommendations.append("High CPU usage detected - consider scaling or optimization")
        
        if memory_stats and memory_stats.get('avg', 0) > 85:
            recommendations.append("High memory usage detected - check for memory leaks")
        
        # Check for slow queries
        slow_query_stats = self.metrics.get_metric_stats('db_slow_query')
        if slow_query_stats and slow_query_stats.get('count', 0) > 10:
            recommendations.append("Multiple slow database queries detected - review and optimize")
        
        # Check API error rates
        api_error_stats = self.metrics.get_metric_stats('api_error')
        if api_error_stats and api_error_stats.get('count', 0) > 5:
            recommendations.append("High API error rate detected - investigate error causes")
        
        return recommendations


# Decorators for easy performance monitoring
def cache_result(cache: Union[AdvancedCache, MultiLevelCache], ttl: Optional[int] = None):
    """Decorator to cache function results"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


def measure_performance(monitor: PerformanceMonitor, operation_name: str = None):
    """Decorator to measure function performance"""
    def decorator(func):
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.measure_time(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global instances
global_cache = MultiLevelCache(
    l1_cache=AdvancedCache(max_size=1000, strategy=CacheStrategy.LRU),
    l2_cache=None  # Will be initialized if Redis is available
)

global_monitor = PerformanceMonitor()
global_query_optimizer = QueryOptimizer()


def initialize_performance_system(redis_url: str = None):
    """Initialize global performance system"""
    global global_cache, global_monitor
    
    # Initialize Redis cache if URL provided
    if redis_url:
        try:
            redis_cache = RedisCache(redis_url)
            global_cache.l2_cache = redis_cache
            logger.info("✅ Redis cache initialized for global cache")
        except Exception as e:
            logger.warning(f"⚠️  Redis cache initialization failed: {e}")
    
    # Start performance monitoring
    global_monitor.start_monitoring()
    
    logger.info("🚀 Performance optimization system initialized")


def main():
    """CLI interface for performance tools"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VivaranAI Performance Tools")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Cache commands
    cache_parser = subparsers.add_parser("cache", help="Cache operations")
    cache_parser.add_argument("action", choices=["stats", "clear", "test"])
    
    # Performance monitoring
    monitor_parser = subparsers.add_parser("monitor", help="Performance monitoring")
    monitor_parser.add_argument("action", choices=["start", "stop", "status", "report"])
    
    # Query optimization
    query_parser = subparsers.add_parser("query", help="Query optimization")
    query_parser.add_argument("action", choices=["analyze", "report"])
    query_parser.add_argument("--query", help="SQL query to analyze")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == "cache":
            if args.action == "stats":
                stats = global_cache.get_stats()
                print(json.dumps(stats, indent=2))
            elif args.action == "clear":
                global_cache.l1_cache.clear()
                print("✅ Cache cleared")
            elif args.action == "test":
                # Test cache performance
                import random
                start_time = time.time()
                
                for i in range(1000):
                    key = f"test_key_{i}"
                    value = f"test_value_{random.randint(1, 1000)}"
                    global_cache.set(key, value)
                
                for i in range(1000):
                    key = f"test_key_{i}"
                    global_cache.get(key)
                
                elapsed = time.time() - start_time
                print(f"✅ Cache test completed in {elapsed:.2f}s")
        
        elif args.command == "monitor":
            if args.action == "start":
                global_monitor.start_monitoring()
                print("✅ Performance monitoring started")
            elif args.action == "stop":
                global_monitor.stop_monitoring()
                print("✅ Performance monitoring stopped")
            elif args.action == "status":
                summary = global_monitor.get_performance_summary()
                print(json.dumps(summary, indent=2))
            elif args.action == "report":
                summary = global_monitor.get_performance_summary()
                print(json.dumps(summary, indent=2))
        
        elif args.command == "query":
            if args.action == "report":
                report = global_query_optimizer.get_query_performance_report()
                print(json.dumps(report, indent=2))
            elif args.action == "analyze" and args.query:
                optimized = global_query_optimizer.optimize_query(args.query)
                print(f"Original: {args.query}")
                print(f"Optimized: {optimized}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 