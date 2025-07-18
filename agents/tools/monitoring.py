"""
Advanced monitoring and metrics collection for tool operations.

Provides comprehensive observability including:
- Performance metrics and timing
- Error tracking and analysis
- Resource utilization monitoring
- Custom business metrics
- Health scoring and alerting
"""

import asyncio
import time
import psutil
import structlog
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import json
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    message: str
    timestamp: float
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for a tool operation."""
    tool_name: str
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collection system.
    
    Collects, aggregates, and provides access to various system
    and application metrics for monitoring and alerting.
    """
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.retention_seconds = retention_hours * 3600
        
        # Metric storage
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.timers: Dict[str, List[MetricPoint]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics: deque = deque(maxlen=10000)
        self.error_metrics: deque = deque(maxlen=1000)
        
        # System metrics
        self.system_metrics: Dict[str, Any] = {}
        self.last_system_update = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._system_metrics_task: Optional[asyncio.Task] = None
        
        logger.info("Metrics collector initialized")
    
    async def start(self):
        """Start background metric collection tasks."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._system_metrics_task = asyncio.create_task(self._system_metrics_loop())
        logger.info("Metrics collection started")
    
    async def stop(self):
        """Stop background tasks."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._system_metrics_task:
            self._system_metrics_task.cancel()
        logger.info("Metrics collection stopped")
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value."""
        with self._lock:
            key = self._make_key(name, labels)
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            self.histograms[key].append(point)
    
    def record_timer(self, name: str, duration_ms: float, labels: Dict[str, str] = None):
        """Record a timer value."""
        with self._lock:
            key = self._make_key(name, labels)
            point = MetricPoint(
                timestamp=time.time(),
                value=duration_ms,
                labels=labels or {}
            )
            self.timers[key].append(point)
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics for a tool operation."""
        with self._lock:
            self.performance_metrics.append(metrics)
            
            # Update derived metrics
            self.increment_counter(
                "tool_operations_total",
                labels={
                    "tool_name": metrics.tool_name,
                    "operation": metrics.operation,
                    "success": str(metrics.success)
                }
            )
            
            self.record_timer(
                "tool_operation_duration_ms",
                metrics.duration_ms,
                labels={
                    "tool_name": metrics.tool_name,
                    "operation": metrics.operation
                }
            )
            
            if not metrics.success:
                self.increment_counter(
                    "tool_errors_total",
                    labels={
                        "tool_name": metrics.tool_name,
                        "operation": metrics.operation,
                        "error_type": metrics.error_type or "unknown"
                    }
                )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self._lock:
            now = time.time()
            cutoff = now - 3600  # Last hour
            
            # Recent performance metrics
            recent_perf = [
                m for m in self.performance_metrics 
                if m.start_time > cutoff
            ]
            
            # Calculate aggregations
            total_operations = len(recent_perf)
            successful_operations = sum(1 for m in recent_perf if m.success)
            failed_operations = total_operations - successful_operations
            
            avg_duration = (
                sum(m.duration_ms for m in recent_perf) / max(total_operations, 1)
            )
            
            # Tool-specific metrics
            tool_stats = defaultdict(lambda: {
                'operations': 0,
                'successes': 0,
                'failures': 0,
                'avg_duration_ms': 0,
                'total_duration_ms': 0
            })
            
            for m in recent_perf:
                stats = tool_stats[m.tool_name]
                stats['operations'] += 1
                stats['total_duration_ms'] += m.duration_ms
                if m.success:
                    stats['successes'] += 1
                else:
                    stats['failures'] += 1
            
            # Calculate averages
            for tool_name, stats in tool_stats.items():
                if stats['operations'] > 0:
                    stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['operations']
                    stats['success_rate'] = stats['successes'] / stats['operations']
            
            return {
                "timestamp": now,
                "period_hours": 1,
                "summary": {
                    "total_operations": total_operations,
                    "successful_operations": successful_operations,
                    "failed_operations": failed_operations,
                    "success_rate": successful_operations / max(total_operations, 1),
                    "avg_duration_ms": avg_duration
                },
                "tool_stats": dict(tool_stats),
                "system_metrics": self.system_metrics,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges)
            }
    
    def get_tool_metrics(self, tool_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get metrics for a specific tool."""
        with self._lock:
            cutoff = time.time() - (hours * 3600)
            
            tool_perf = [
                m for m in self.performance_metrics 
                if m.tool_name == tool_name and m.start_time > cutoff
            ]
            
            if not tool_perf:
                return {
                    "tool_name": tool_name,
                    "period_hours": hours,
                    "no_data": True
                }
            
            total_ops = len(tool_perf)
            successful_ops = sum(1 for m in tool_perf if m.success)
            
            durations = [m.duration_ms for m in tool_perf]
            durations.sort()
            
            return {
                "tool_name": tool_name,
                "period_hours": hours,
                "total_operations": total_ops,
                "successful_operations": successful_ops,
                "failed_operations": total_ops - successful_ops,
                "success_rate": successful_ops / total_ops,
                "duration_stats": {
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "avg_ms": sum(durations) / len(durations),
                    "p50_ms": durations[len(durations) // 2],
                    "p95_ms": durations[int(len(durations) * 0.95)],
                    "p99_ms": durations[int(len(durations) * 0.99)]
                }
            }
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for a metric."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    async def _cleanup_loop(self):
        """Background task to clean up old metrics."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics cleanup error: {str(e)}")
    
    async def _cleanup_old_metrics(self):
        """Remove old metric data points."""
        cutoff = time.time() - self.retention_seconds
        
        with self._lock:
            # Clean histograms
            for key in self.histograms:
                self.histograms[key] = [
                    p for p in self.histograms[key] 
                    if p.timestamp > cutoff
                ]
            
            # Clean timers
            for key in self.timers:
                self.timers[key] = [
                    p for p in self.timers[key] 
                    if p.timestamp > cutoff
                ]
            
            # Clean performance metrics (handled by deque maxlen)
            # Clean error metrics (handled by deque maxlen)
    
    async def _system_metrics_loop(self):
        """Background task to collect system metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                await self._update_system_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System metrics error: {str(e)}")
    
    async def _update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                network_stats = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
            except:
                network_stats = {}
            
            with self._lock:
                self.system_metrics = {
                    "timestamp": time.time(),
                    "cpu": {
                        "percent": cpu_percent,
                        "count": cpu_count
                    },
                    "memory": {
                        "total_mb": memory.total / (1024 * 1024),
                        "available_mb": memory.available / (1024 * 1024),
                        "used_mb": memory.used / (1024 * 1024),
                        "percent": memory.percent
                    },
                    "disk": {
                        "total_gb": disk.total / (1024 * 1024 * 1024),
                        "used_gb": disk.used / (1024 * 1024 * 1024),
                        "free_gb": disk.free / (1024 * 1024 * 1024),
                        "percent": (disk.used / disk.total) * 100
                    },
                    "network": network_stats
                }
                
                # Update gauge metrics
                self.set_gauge("system_cpu_percent", cpu_percent)
                self.set_gauge("system_memory_percent", memory.percent)
                self.set_gauge("system_disk_percent", (disk.used / disk.total) * 100)
                
        except Exception as e:
            logger.error(f"System metrics update failed: {str(e)}")


class AlertManager:
    """
    Alert management system for tool monitoring.
    
    Manages alerts based on metric thresholds and system conditions.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Callable] = []
        self._lock = threading.Lock()
        
        # Default alert rules
        self._setup_default_rules()
        
        logger.info("Alert manager initialized")
    
    def add_alert_rule(self, rule: Callable[[Dict[str, Any]], Optional[Alert]]):
        """Add a custom alert rule."""
        self.alert_rules.append(rule)
    
    async def check_alerts(self) -> List[Alert]:
        """Check all alert rules and return active alerts."""
        metrics = self.metrics_collector.get_metrics_summary()
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                alert = rule(metrics)
                if alert:
                    with self._lock:
                        if alert.id not in self.alerts:
                            self.alerts[alert.id] = alert
                            new_alerts.append(alert)
                            logger.warning(
                                f"Alert triggered: {alert.message}",
                                severity=alert.severity.value,
                                source=alert.source
                            )
            except Exception as e:
                logger.error(f"Alert rule error: {str(e)}")
        
        return new_alerts
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolved_at = time.time()
                logger.info(f"Alert resolved: {alert_id}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        
        def high_error_rate_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
            """Alert on high error rate."""
            summary = metrics.get("summary", {})
            success_rate = summary.get("success_rate", 1.0)
            total_ops = summary.get("total_operations", 0)
            
            if total_ops > 10 and success_rate < 0.8:  # Less than 80% success
                return Alert(
                    id="high_error_rate",
                    severity=AlertSeverity.WARNING,
                    message=f"High error rate detected: {(1-success_rate)*100:.1f}% failure rate",
                    timestamp=time.time(),
                    source="tool_monitoring",
                    labels={"success_rate": str(success_rate)}
                )
            return None
        
        def high_cpu_usage_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
            """Alert on high CPU usage."""
            system_metrics = metrics.get("system_metrics", {})
            cpu_percent = system_metrics.get("cpu", {}).get("percent", 0)
            
            if cpu_percent > 90:
                return Alert(
                    id="high_cpu_usage",
                    severity=AlertSeverity.ERROR,
                    message=f"High CPU usage: {cpu_percent:.1f}%",
                    timestamp=time.time(),
                    source="system_monitoring",
                    labels={"cpu_percent": str(cpu_percent)}
                )
            return None
        
        def high_memory_usage_rule(metrics: Dict[str, Any]) -> Optional[Alert]:
            """Alert on high memory usage."""
            system_metrics = metrics.get("system_metrics", {})
            memory_percent = system_metrics.get("memory", {}).get("percent", 0)
            
            if memory_percent > 85:
                return Alert(
                    id="high_memory_usage",
                    severity=AlertSeverity.WARNING,
                    message=f"High memory usage: {memory_percent:.1f}%",
                    timestamp=time.time(),
                    source="system_monitoring",
                    labels={"memory_percent": str(memory_percent)}
                )
            return None
        
        self.alert_rules.extend([
            high_error_rate_rule,
            high_cpu_usage_rule,
            high_memory_usage_rule
        ])


class PerformanceTracker:
    """
    Context manager for tracking tool operation performance.
    
    Automatically collects timing, resource usage, and success metrics.
    """
    
    def __init__(
        self, 
        metrics_collector: MetricsCollector,
        tool_name: str,
        operation: str,
        labels: Dict[str, str] = None
    ):
        self.metrics_collector = metrics_collector
        self.tool_name = tool_name
        self.operation = operation
        self.labels = labels or {}
        
        self.start_time = 0
        self.start_memory = 0
        self.success = False
        self.error_type = None
        self.error_message = None
    
    async def __aenter__(self):
        """Start performance tracking."""
        self.start_time = time.time()
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except:
            self.start_memory = 0
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End performance tracking and record metrics."""
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000
        
        # Determine success
        self.success = exc_type is None
        if exc_type:
            self.error_type = exc_type.__name__
            self.error_message = str(exc_val) if exc_val else None
        
        # Get memory usage
        end_memory = 0
        try:
            process = psutil.Process()
            end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        except:
            pass
        
        # Record performance metrics
        perf_metrics = PerformanceMetrics(
            tool_name=self.tool_name,
            operation=self.operation,
            start_time=self.start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            success=self.success,
            error_type=self.error_type,
            error_message=self.error_message,
            memory_usage_mb=end_memory - self.start_memory if self.start_memory > 0 else None,
            labels=self.labels
        )
        
        self.metrics_collector.record_performance(perf_metrics)


# Global instances
metrics_collector = MetricsCollector()
alert_manager = AlertManager(metrics_collector)
