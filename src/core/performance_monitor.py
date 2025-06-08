"""
Performance Monitoring for Open-Sourcefy
Phase 1 Implementation: Performance Optimization & Resource Usage
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

# Third-party imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    active_threads: int = 0
    open_files: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
            'active_threads': self.active_threads,
            'open_files': self.open_files
        }


@dataclass
class ProcessMetrics:
    """Container for process-specific metrics"""
    process_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    total_disk_read_mb: float = 0.0
    total_disk_write_mb: float = 0.0
    exit_code: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'process_name': self.process_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_cpu_percent': self.avg_cpu_percent,
            'total_disk_read_mb': self.total_disk_read_mb,
            'total_disk_write_mb': self.total_disk_write_mb,
            'exit_code': self.exit_code,
            'metadata': self.metadata
        }


class PerformanceMonitor:
    """Real-time performance monitoring for Open-Sourcefy operations"""
    
    def __init__(self, sample_interval: float = 1.0, max_samples: int = 3600):
        """
        Initialize performance monitor
        
        Args:
            sample_interval: Seconds between samples
            max_samples: Maximum number of samples to keep in memory
        """
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.logger = logging.getLogger("PerformanceMonitor")
        
        # Data storage
        self.system_metrics: List[PerformanceMetrics] = []
        self.process_metrics: Dict[str, ProcessMetrics] = {}
        self.alerts: List[Dict[str, Any]] = []
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0
        }
        
        # Baseline metrics for comparison
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - performance monitoring will be limited")
    
    def start_monitoring(self):
        """Start real-time performance monitoring"""
        if self.monitoring:
            self.logger.warning("Performance monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                if metrics:
                    with self.lock:
                        self.system_metrics.append(metrics)
                        
                        # Trim old samples
                        if len(self.system_metrics) > self.max_samples:
                            self.system_metrics.pop(0)
                        
                        # Check for alerts
                        self._check_alerts(metrics)
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.sample_interval)
    
    def _collect_system_metrics(self) -> Optional[PerformanceMetrics]:
        """Collect current system metrics"""
        if not PSUTIL_AVAILABLE:
            return PerformanceMetrics(timestamp=time.time())
        
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            # Process info
            try:
                current_process = psutil.Process()
                active_threads = current_process.num_threads()
                open_files = len(current_process.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                active_threads = 0
                open_files = 0
            
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                disk_io_read_mb=(disk_io.read_bytes / (1024 * 1024)) if disk_io else 0,
                disk_io_write_mb=(disk_io.write_bytes / (1024 * 1024)) if disk_io else 0,
                network_sent_mb=(network_io.bytes_sent / (1024 * 1024)) if network_io else 0,
                network_recv_mb=(network_io.bytes_recv / (1024 * 1024)) if network_io else 0,
                active_threads=active_threads,
                open_files=open_files
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return None
    
    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'timestamp': metrics.timestamp,
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent'],
                'message': f"High CPU usage: {metrics.cpu_percent:.1f}%"
            })
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'timestamp': metrics.timestamp,
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_percent'],
                'message': f"High memory usage: {metrics.memory_percent:.1f}%"
            })
        
        # Disk space alert (if available)
        if PSUTIL_AVAILABLE:
            try:
                disk_usage = psutil.disk_usage('/')
                disk_percent = (disk_usage.used / disk_usage.total) * 100
                if disk_percent > self.alert_thresholds['disk_usage_percent']:
                    alerts.append({
                        'type': 'disk_high',
                        'timestamp': metrics.timestamp,
                        'value': disk_percent,
                        'threshold': self.alert_thresholds['disk_usage_percent'],
                        'message': f"High disk usage: {disk_percent:.1f}%"
                    })
            except Exception:
                pass
        
        # Add alerts
        for alert in alerts:
            self.alerts.append(alert)
            self.logger.warning(alert['message'])
    
    def start_process_monitoring(self, process_name: str, **metadata) -> str:
        """Start monitoring a specific process/operation"""
        process_id = f"{process_name}_{int(time.time())}"
        
        process_metrics = ProcessMetrics(
            process_name=process_name,
            start_time=time.time(),
            metadata=metadata
        )
        
        with self.lock:
            self.process_metrics[process_id] = process_metrics
        
        self.logger.info(f"Started monitoring process: {process_name}")
        return process_id
    
    def stop_process_monitoring(self, process_id: str, exit_code: int = 0):
        """Stop monitoring a specific process/operation"""
        with self.lock:
            if process_id in self.process_metrics:
                process_metrics = self.process_metrics[process_id]
                process_metrics.end_time = time.time()
                process_metrics.duration = process_metrics.end_time - process_metrics.start_time
                process_metrics.exit_code = exit_code
                
                self.logger.info(f"Stopped monitoring process: {process_metrics.process_name}, "
                               f"duration: {process_metrics.duration:.2f}s")
    
    def update_process_metrics(self, process_id: str, **updates):
        """Update metrics for a specific process"""
        with self.lock:
            if process_id in self.process_metrics:
                process_metrics = self.process_metrics[process_id]
                for key, value in updates.items():
                    if hasattr(process_metrics, key):
                        setattr(process_metrics, key, value)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent system metrics"""
        with self.lock:
            return self.system_metrics[-1] if self.system_metrics else None
    
    def get_metrics_summary(self, duration_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Get summary of metrics over specified duration"""
        with self.lock:
            if not self.system_metrics:
                return {"error": "No metrics available"}
            
            # Filter metrics by duration if specified
            if duration_seconds:
                cutoff_time = time.time() - duration_seconds
                filtered_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
            else:
                filtered_metrics = self.system_metrics
            
            if not filtered_metrics:
                return {"error": "No metrics in specified time range"}
            
            # Calculate statistics
            cpu_values = [m.cpu_percent for m in filtered_metrics]
            memory_values = [m.memory_percent for m in filtered_metrics]
            memory_mb_values = [m.memory_used_mb for m in filtered_metrics]
            
            summary = {
                "time_range": {
                    "start": datetime.fromtimestamp(filtered_metrics[0].timestamp).isoformat(),
                    "end": datetime.fromtimestamp(filtered_metrics[-1].timestamp).isoformat(),
                    "duration_seconds": filtered_metrics[-1].timestamp - filtered_metrics[0].timestamp,
                    "sample_count": len(filtered_metrics)
                },
                "cpu": {
                    "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    "max": max(cpu_values) if cpu_values else 0,
                    "min": min(cpu_values) if cpu_values else 0
                },
                "memory": {
                    "avg_percent": sum(memory_values) / len(memory_values) if memory_values else 0,
                    "max_percent": max(memory_values) if memory_values else 0,
                    "avg_mb": sum(memory_mb_values) / len(memory_mb_values) if memory_mb_values else 0,
                    "max_mb": max(memory_mb_values) if memory_mb_values else 0
                },
                "alerts": len([a for a in self.alerts 
                             if a['timestamp'] >= filtered_metrics[0].timestamp])
            }
            
            return summary
    
    def get_process_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored processes"""
        with self.lock:
            summary = {
                "total_processes": len(self.process_metrics),
                "active_processes": len([p for p in self.process_metrics.values() if p.end_time is None]),
                "completed_processes": len([p for p in self.process_metrics.values() if p.end_time is not None]),
                "processes": {}
            }
            
            for process_id, metrics in self.process_metrics.items():
                summary["processes"][process_id] = metrics.to_dict()
            
            return summary
    
    def export_metrics(self, output_path: Path = None, include_raw_data: bool = False, output_dir: str = None):
        """Export metrics to JSON file
        
        Args:
            output_path: Specific path to write to (legacy support)
            include_raw_data: Whether to include raw system metrics
            output_dir: Base output directory to use logs subdirectory
        """
        with self.lock:
            export_data = {
                "export_timestamp": time.time(),
                "export_date": datetime.now().isoformat(),
                "monitoring_summary": self.get_metrics_summary(),
                "process_summary": self.get_process_summary(),
                "alerts": self.alerts[-100:],  # Last 100 alerts
                "alert_thresholds": self.alert_thresholds
            }
            
            if include_raw_data:
                export_data["raw_system_metrics"] = [m.to_dict() for m in self.system_metrics[-1000:]]  # Last 1000 samples
        
        # Determine output path - prefer logs subdirectory if output_dir provided
        if output_path is None and output_dir:
            logs_dir = Path(output_dir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            output_path = logs_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        elif output_path is None:
            raise ValueError("Either output_path or output_dir must be provided")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Performance metrics exported to {output_path}")
    
    def set_baseline(self):
        """Set current metrics as baseline for comparison"""
        current = self.get_current_metrics()
        if current:
            self.baseline_metrics = current
            self.logger.info("Performance baseline set")
    
    def compare_to_baseline(self) -> Optional[Dict[str, Any]]:
        """Compare current metrics to baseline"""
        if not self.baseline_metrics:
            return None
        
        current = self.get_current_metrics()
        if not current:
            return None
        
        comparison = {
            "baseline_timestamp": datetime.fromtimestamp(self.baseline_metrics.timestamp).isoformat(),
            "current_timestamp": datetime.fromtimestamp(current.timestamp).isoformat(),
            "cpu_change": current.cpu_percent - self.baseline_metrics.cpu_percent,
            "memory_change_percent": current.memory_percent - self.baseline_metrics.memory_percent,
            "memory_change_mb": current.memory_used_mb - self.baseline_metrics.memory_used_mb,
            "threads_change": current.active_threads - self.baseline_metrics.active_threads
        }
        
        return comparison
    
    def optimize_performance(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        current = self.get_current_metrics()
        if not current:
            return ["Unable to get current metrics for optimization analysis"]
        
        # CPU recommendations
        if current.cpu_percent > 80:
            recommendations.append("High CPU usage detected. Consider reducing parallel processes or batch size.")
        
        # Memory recommendations
        if current.memory_percent > 75:
            recommendations.append("High memory usage detected. Consider reducing memory limits or processing smaller chunks.")
        
        # Thread recommendations
        if current.active_threads > 50:
            recommendations.append("High thread count detected. Consider using process-based parallelism instead of threading.")
        
        # Get summary for trend analysis
        summary = self.get_metrics_summary(duration_seconds=300)  # Last 5 minutes
        if "cpu" in summary:
            if summary["cpu"]["avg"] > 70:
                recommendations.append("Sustained high CPU usage. Consider implementing CPU throttling or load balancing.")
            
            if summary["memory"]["max_percent"] > 90:
                recommendations.append("Memory usage spikes detected. Implement memory cleanup between operations.")
        
        # Alert-based recommendations
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 300]  # Last 5 minutes
        if len(recent_alerts) > 5:
            recommendations.append("Frequent performance alerts. Review system configuration and resource allocation.")
        
        if not recommendations:
            recommendations.append("Performance appears optimal. No specific recommendations at this time.")
        
        return recommendations


class PerformanceContext:
    """Context manager for performance monitoring"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str, **metadata):
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata
        self.process_id = None
        
    def __enter__(self):
        self.process_id = self.monitor.start_process_monitoring(self.operation_name, **self.metadata)
        return self.process_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process_id:
            exit_code = 0 if exc_type is None else 1
            self.monitor.stop_process_monitoring(self.process_id, exit_code)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def monitor_operation(operation_name: str, **metadata):
    """Decorator for monitoring function performance"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with PerformanceContext(monitor, operation_name, **metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Convenience functions
def start_monitoring():
    """Start global performance monitoring"""
    get_performance_monitor().start_monitoring()


def stop_monitoring():
    """Stop global performance monitoring"""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


def get_current_performance() -> Optional[PerformanceMetrics]:
    """Get current performance metrics"""
    return get_performance_monitor().get_current_metrics()


def export_performance_report(output_path: Path):
    """Export performance report"""
    get_performance_monitor().export_metrics(output_path, include_raw_data=True)


if __name__ == "__main__":
    # Test performance monitoring
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitoring Test")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    parser.add_argument("--export", type=str, help="Export metrics to file")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(sample_interval=0.5)
    monitor.start_monitoring()
    monitor.set_baseline()
    
    print(f"Monitoring performance for {args.duration} seconds...")
    
    # Simulate some work
    with PerformanceContext(monitor, "test_operation", test_param="value"):
        time.sleep(args.duration)
    
    monitor.stop_monitoring()
    
    # Print summary
    summary = monitor.get_metrics_summary()
    print("\nPerformance Summary:")
    print(json.dumps(summary, indent=2))
    
    # Print recommendations
    recommendations = monitor.optimize_performance()
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  - {rec}")
    
    # Export if requested
    if args.export:
        monitor.export_metrics(Path(args.export))
        print(f"\nMetrics exported to {args.export}")