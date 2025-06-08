"""
Shared Utilities for Open-Sourcefy Matrix Pipeline
Common utilities for logging, validation, performance monitoring, and data processing
"""

import os
import sys
import time
import threading
import logging
import traceback
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import json


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_usage_start: Optional[float] = None
    cpu_usage_end: Optional[float] = None
    memory_usage_start: Optional[int] = None
    memory_usage_end: Optional[int] = None
    peak_memory_usage: Optional[int] = None
    operation_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self):
        """Mark the end of the operation and calculate metrics"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        try:
            process = psutil.Process()
            self.cpu_usage_end = process.cpu_percent()
            self.memory_usage_end = process.memory_info().rss
        except Exception:
            pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'cpu_usage_start': self.cpu_usage_start,
            'cpu_usage_end': self.cpu_usage_end,
            'memory_usage_start': self.memory_usage_start,
            'memory_usage_end': self.memory_usage_end,
            'peak_memory_usage': self.peak_memory_usage,
            'metadata': self.metadata
        }


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.logger = logging.getLogger("PerformanceMonitor")
        self.metrics: List[PerformanceMetrics] = []
        self._current_operation: Optional[PerformanceMetrics] = None
        self._lock = threading.Lock()
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """Start monitoring an operation"""
        with self._lock:
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                metadata=metadata or {}
            )
            
            try:
                process = psutil.Process()
                metrics.cpu_usage_start = process.cpu_percent()
                metrics.memory_usage_start = process.memory_info().rss
            except Exception as e:
                self.logger.warning(f"Failed to get initial system metrics: {e}")
            
            self.metrics.append(metrics)
            self._current_operation = metrics
            
            self.logger.debug(f"Started monitoring operation: {operation_name}")
            return metrics
    
    def end_operation(self, metrics: Optional[PerformanceMetrics] = None) -> PerformanceMetrics:
        """End monitoring an operation"""
        with self._lock:
            if metrics is None:
                metrics = self._current_operation
            
            if metrics is None:
                raise ValueError("No operation to end")
            
            metrics.finish()
            
            if metrics == self._current_operation:
                self._current_operation = None
            
            self.logger.debug(f"Ended monitoring operation: {metrics.operation_name} "
                            f"(duration: {metrics.duration:.3f}s)")
            
            return metrics
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        if not self.metrics:
            return {}
        
        operations = {}
        total_duration = 0
        
        for metric in self.metrics:
            name = metric.operation_name
            if name not in operations:
                operations[name] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_duration': 0,
                    'min_duration': float('inf'),
                    'max_duration': 0,
                    'total_memory_delta': 0
                }
            
            op_stats = operations[name]
            op_stats['count'] += 1
            
            if metric.duration:
                op_stats['total_duration'] += metric.duration
                op_stats['min_duration'] = min(op_stats['min_duration'], metric.duration)
                op_stats['max_duration'] = max(op_stats['max_duration'], metric.duration)
                total_duration += metric.duration
            
            if metric.memory_usage_start and metric.memory_usage_end:
                memory_delta = metric.memory_usage_end - metric.memory_usage_start
                op_stats['total_memory_delta'] += memory_delta
        
        # Calculate averages
        for op_stats in operations.values():
            if op_stats['count'] > 0:
                op_stats['avg_duration'] = op_stats['total_duration'] / op_stats['count']
                if op_stats['min_duration'] == float('inf'):
                    op_stats['min_duration'] = 0
        
        return {
            'total_operations': len(self.metrics),
            'total_duration': total_duration,
            'operations': operations,
            'start_time': min(m.start_time for m in self.metrics) if self.metrics else None,
            'end_time': max(m.end_time for m in self.metrics if m.end_time) if self.metrics else None
        }
    
    def export_metrics(self, file_path: Union[str, Path]):
        """Export metrics to JSON file"""
        data = {
            'summary': self.get_metrics_summary(),
            'detailed_metrics': [m.to_dict() for m in self.metrics]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)


def performance_monitor(operation_name: str = None):
    """Decorator for automatic performance monitoring"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Get or create monitor
            if hasattr(wrapper, '_monitor'):
                monitor = wrapper._monitor
            else:
                monitor = PerformanceMonitor()
                wrapper._monitor = monitor
            
            metrics = monitor.start_operation(name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end_operation(metrics)
        
        return wrapper
    return decorator


class MatrixLogger:
    """Enhanced logging with Matrix-themed formatting"""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '🔮 %(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def add_file_handler(self, log_file: Union[str, Path], level: int = logging.DEBUG):
        """Add file handler for detailed logging"""
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        self.logger.addHandler(file_handler)
    
    def matrix_info(self, agent_name: str, message: str):
        """Log with Matrix agent theming"""
        self.logger.info(f"🤖 {agent_name}: {message}")
    
    def matrix_success(self, agent_name: str, message: str):
        """Log success with Matrix theming"""
        self.logger.info(f"✅ {agent_name}: {message}")
    
    def matrix_error(self, agent_name: str, message: str):
        """Log error with Matrix theming"""
        self.logger.error(f"❌ {agent_name}: {message}")
    
    def matrix_warning(self, agent_name: str, message: str):
        """Log warning with Matrix theming"""
        self.logger.warning(f"⚠️  {agent_name}: {message}")


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], must_exist: bool = True) -> Path:
        """Validate file path"""
        path = Path(path)
        
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        return path
    
    @staticmethod
    def validate_directory(path: Union[str, Path], create_if_missing: bool = False) -> Path:
        """Validate directory path"""
        path = Path(path)
        
        if not path.exists():
            if create_if_missing:
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(f"Directory not found: {path}")
        
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        return path
    
    @staticmethod
    def validate_binary_file(path: Union[str, Path]) -> Path:
        """Validate binary file for analysis"""
        path = DataValidator.validate_file_path(path)
        
        # Check file is not empty
        if path.stat().st_size == 0:
            raise ValueError(f"Binary file is empty: {path}")
        
        # Check file has reasonable size (not too large)
        max_size = 500 * 1024 * 1024  # 500MB
        if path.stat().st_size > max_size:
            raise ValueError(f"Binary file too large (>{max_size} bytes): {path}")
        
        return path
    
    @staticmethod
    def validate_agent_id(agent_id: Union[int, str]) -> int:
        """Validate agent ID"""
        if isinstance(agent_id, str):
            try:
                agent_id = int(agent_id)
            except ValueError:
                raise ValueError(f"Invalid agent ID: {agent_id}")
        
        if not isinstance(agent_id, int):
            raise TypeError(f"Agent ID must be int, got {type(agent_id)}")
        
        if agent_id < 0 or agent_id > 20:
            raise ValueError(f"Agent ID must be 0-20, got {agent_id}")
        
        return agent_id
    
    @staticmethod
    def validate_timeout(timeout: Union[int, float]) -> float:
        """Validate timeout value"""
        if not isinstance(timeout, (int, float)):
            raise TypeError(f"Timeout must be numeric, got {type(timeout)}")
        
        if timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {timeout}")
        
        if timeout > 3600:  # 1 hour max
            raise ValueError(f"Timeout too large (max 3600s), got {timeout}")
        
        return float(timeout)


class RetryHelper:
    """Retry logic for unreliable operations"""
    
    @staticmethod
    def retry_operation(func: Callable, max_retries: int = 3, delay: float = 1.0,
                       backoff_factor: float = 2.0, exceptions: Tuple = (Exception,)) -> Any:
        """Retry operation with exponential backoff"""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except exceptions as e:
                last_exception = e
                
                if attempt < max_retries:
                    sleep_time = delay * (backoff_factor ** attempt)
                    time.sleep(sleep_time)
                else:
                    break
        
        raise last_exception


class SystemUtils:
    """System utilities and resource management"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            return {
                'cpu_count': psutil.cpu_count(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': {
                    path: {
                        'total': psutil.disk_usage(path).total,
                        'used': psutil.disk_usage(path).used,
                        'free': psutil.disk_usage(path).free
                    }
                    for path in ['/'] if os.path.exists(path)
                },
                'platform': sys.platform,
                'python_version': sys.version
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def check_memory_usage(threshold_percent: float = 90.0) -> bool:
        """Check if memory usage is above threshold"""
        try:
            memory_percent = psutil.virtual_memory().percent
            return memory_percent > threshold_percent
        except Exception:
            return False
    
    @staticmethod
    def check_disk_space(path: Union[str, Path], min_free_gb: float = 1.0) -> bool:
        """Check if disk has minimum free space"""
        try:
            usage = psutil.disk_usage(str(path))
            free_gb = usage.free / (1024 ** 3)
            return free_gb >= min_free_gb
        except Exception:
            return False
    
    @staticmethod
    def get_process_info(pid: Optional[int] = None) -> Dict[str, Any]:
        """Get information about current or specified process"""
        try:
            process = psutil.Process(pid)
            return {
                'pid': process.pid,
                'name': process.name(),
                'cpu_percent': process.cpu_percent(),
                'memory_info': process.memory_info()._asdict(),
                'memory_percent': process.memory_percent(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time(),
                'cwd': process.cwd(),
                'exe': process.exe()
            }
        except Exception as e:
            return {'error': str(e)}


class ProgressTracker:
    """Progress tracking for long-running operations"""
    
    def __init__(self, total_steps: int, operation_name: str = "Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.logger = MatrixLogger(f"Progress_{operation_name}")
    
    def step(self, description: str = "", increment: int = 1):
        """Advance progress by one or more steps"""
        self.current_step = min(self.current_step + increment, self.total_steps)
        step_time = time.time()
        self.step_times.append(step_time)
        
        progress_percent = (self.current_step / self.total_steps) * 100
        elapsed_time = step_time - self.start_time
        
        if self.current_step > 0:
            avg_time_per_step = elapsed_time / self.current_step
            eta = avg_time_per_step * (self.total_steps - self.current_step)
            eta_str = f"ETA: {timedelta(seconds=int(eta))}"
        else:
            eta_str = "ETA: Unknown"
        
        status = f"{self.operation_name}: {self.current_step}/{self.total_steps} " \
                f"({progress_percent:.1f}%) | {eta_str}"
        
        if description:
            status += f" | {description}"
        
        self.logger.logger.info(status)
    
    def finish(self):
        """Mark operation as complete"""
        total_time = time.time() - self.start_time
        self.logger.matrix_success(
            "ProgressTracker",
            f"{self.operation_name} completed in {timedelta(seconds=int(total_time))}"
        )


class ErrorHandler:
    """Centralized error handling and reporting"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("ErrorHandler")
        self.error_count = 0
        self.errors: List[Dict[str, Any]] = []
    
    def handle_error(self, error: Exception, context: str = "", 
                    agent_name: str = "", reraise: bool = False) -> Dict[str, Any]:
        """Handle and log error with context"""
        self.error_count += 1
        
        error_info = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'agent_name': agent_name,
            'traceback': traceback.format_exc()
        }
        
        self.errors.append(error_info)
        
        # Log the error
        log_message = f"Error in {context}" if context else "Error occurred"
        if agent_name:
            log_message = f"[{agent_name}] {log_message}"
        
        log_message += f": {error_info['error_type']}: {error_info['error_message']}"
        self.logger.error(log_message)
        
        if reraise:
            raise error
        
        return error_info
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all handled errors"""
        error_types = {}
        agent_errors = {}
        
        for error in self.errors:
            error_type = error['error_type']
            agent_name = error['agent_name']
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if agent_name:
                agent_errors[agent_name] = agent_errors.get(agent_name, 0) + 1
        
        return {
            'total_errors': self.error_count,
            'error_types': error_types,
            'agent_errors': agent_errors,
            'recent_errors': self.errors[-5:] if self.errors else []
        }


# Utility functions
def format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def safe_dict_get(data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Safely get nested dictionary value using dot notation"""
    keys = key_path.split('.')
    value = data
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries recursively"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def ensure_list(value: Any) -> List[Any]:
    """Ensure value is a list"""
    if value is None:
        return []
    elif isinstance(value, list):
        return value
    else:
        return [value]


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate string to maximum length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def create_timestamp(include_microseconds: bool = False) -> str:
    """Create timestamp string"""
    now = datetime.now()
    
    if include_microseconds:
        return now.strftime("%Y%m%d_%H%M%S_%f")
    else:
        return now.strftime("%Y%m%d_%H%M%S")


# Global instances
_performance_monitor = PerformanceMonitor()
_error_handler = ErrorHandler()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return _performance_monitor


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    return _error_handler


class LoggingUtils:
    """Utility class for logging operations"""
    
    @staticmethod
    def setup_agent_logging(agent_id: int, agent_name: str, 
                           log_level: int = logging.INFO) -> logging.Logger:
        """Set up logging for an agent"""
        logger_name = f"Agent{agent_id:02d}_{agent_name}"
        logger = logging.getLogger(logger_name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {logger_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
        
        return logger
    
    @staticmethod
    def setup_file_logging(log_file: str, log_level: int = logging.INFO) -> logging.Logger:
        """Set up file logging"""
        logger = logging.getLogger("FileLogger")
        
        if not logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
        
        return logger


class FileOperations:
    """Utility class for file operations"""
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """Ensure directory exists, create if it doesn't"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def safe_read_file(file_path: Union[str, Path], 
                      encoding: str = 'utf-8') -> Optional[str]:
        """Safely read file contents"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            return None
    
    @staticmethod
    def safe_write_file(file_path: Union[str, Path], content: str, 
                       encoding: str = 'utf-8') -> bool:
        """Safely write file contents"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            logging.error(f"Failed to write file {file_path}: {e}")
            return False
    
    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> Optional[int]:
        """Get file size in bytes"""
        try:
            return Path(file_path).stat().st_size
        except Exception:
            return None
    
    @staticmethod
    def file_exists(file_path: Union[str, Path]) -> bool:
        """Check if file exists"""
        return Path(file_path).exists()


class ValidationUtils:
    """Utility class for validation operations"""
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> bool:
        """Validate if file path exists and is accessible"""
        try:
            path = Path(file_path)
            return path.exists() and path.is_file()
        except Exception:
            return False
    
    @staticmethod
    def validate_directory_path(dir_path: Union[str, Path]) -> bool:
        """Validate if directory path exists and is accessible"""
        try:
            path = Path(dir_path)
            return path.exists() and path.is_dir()
        except Exception:
            return False
    
    @staticmethod
    def validate_agent_result(result: Any) -> bool:
        """Validate agent result structure"""
        if not hasattr(result, 'success'):
            return False
        if not isinstance(result.success, bool):
            return False
        return True
    
    @staticmethod
    def validate_binary_format(file_path: Union[str, Path]) -> bool:
        """Basic validation of binary file format"""
        try:
            path = Path(file_path)
            if not path.exists():
                return False
            
            # Check file size (should be reasonable for a binary)
            size = path.stat().st_size
            if size < 100 or size > 100 * 1024 * 1024:  # 100 bytes to 100MB
                return False
            
            # Check if file is executable
            with open(path, 'rb') as f:
                header = f.read(4)
                
            # Check for common executable headers
            pe_header = b'MZ'  # PE (Windows)
            elf_header = b'\x7fELF'  # ELF (Linux)
            macho_header = b'\xfe\xed\xfa'  # Mach-O (macOS)
            
            return (header.startswith(pe_header) or 
                   header.startswith(elf_header) or 
                   header.startswith(macho_header))
        except Exception:
            return False
    
    @staticmethod
    def validate_config_dict(config: Dict[str, Any], 
                           required_keys: List[str]) -> bool:
        """Validate configuration dictionary"""
        if not isinstance(config, dict):
            return False
        
        for key in required_keys:
            if key not in config:
                return False
        
        return True
    
    @staticmethod
    def validate_agent_id(agent_id: Any) -> bool:
        """Validate agent ID"""
        try:
            return isinstance(agent_id, int) and 0 <= agent_id <= 50
        except Exception:
            return False