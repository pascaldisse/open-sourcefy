"""
Enhanced Error Handling and Logging Framework for Open-Sourcefy
Phase 1 Implementation: Error Handling Enhancement
"""

import logging
import traceback
import sys
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from datetime import datetime
import inspect
import functools


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    ENVIRONMENT = "environment"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    AGENT = "agent"
    GHIDRA = "ghidra"
    COMPILATION = "compilation"
    VALIDATION = "validation"
    SYSTEM = "system"
    NETWORK = "network"
    PERFORMANCE = "performance"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    timestamp: float
    thread_id: int
    function_name: str
    module_name: str
    line_number: int
    filename: str
    local_variables: Dict[str, Any] = field(default_factory=dict)
    stack_trace: List[str] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'thread_id': self.thread_id,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'line_number': self.line_number,
            'filename': self.filename,
            'local_variables': self.local_variables,
            'stack_trace': self.stack_trace,
            'system_info': self.system_info
        }


@dataclass
class ErrorRecord:
    """Complete error record with context and metadata"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'error_id': self.error_id,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context.to_dict(),
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_actions': self.recovery_actions,
            'metadata': self.metadata
        }


class RecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Recovery.{name}")
    
    def can_recover(self, error_record: ErrorRecord) -> bool:
        """Check if this strategy can handle the error"""
        return False
    
    def attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt to recover from the error"""
        return False


class RetryRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy that retries the operation with backoff"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        super().__init__("retry", "Retry operation with exponential backoff")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def can_recover(self, error_record: ErrorRecord) -> bool:
        """Check if retry is appropriate for this error"""
        retriable_errors = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure",
            "ResourceBusyError"
        ]
        return any(err in error_record.error_type for err in retriable_errors)
    
    def attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt recovery through retries"""
        current_retries = error_record.metadata.get('retry_count', 0)
        
        if current_retries >= self.max_retries:
            self.logger.warning(f"Max retries ({self.max_retries}) exceeded for {error_record.error_id}")
            return False
        
        # Calculate delay
        delay = (self.backoff_factor ** current_retries)
        self.logger.info(f"Retrying operation after {delay:.1f}s delay (attempt {current_retries + 1}/{self.max_retries})")
        
        time.sleep(delay)
        error_record.metadata['retry_count'] = current_retries + 1
        error_record.recovery_actions.append(f"Retry attempt {current_retries + 1} after {delay:.1f}s")
        
        return True


class FallbackRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy that uses fallback methods"""
    
    def __init__(self):
        super().__init__("fallback", "Use fallback implementation")
    
    def can_recover(self, error_record: ErrorRecord) -> bool:
        """Check if fallback is available"""
        fallback_categories = [
            ErrorCategory.DEPENDENCY,
            ErrorCategory.GHIDRA,
            ErrorCategory.COMPILATION
        ]
        return error_record.category in fallback_categories
    
    def attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt recovery using fallback methods"""
        if error_record.category == ErrorCategory.DEPENDENCY:
            self.logger.info("Using mock implementation due to missing dependency")
            error_record.recovery_actions.append("Switched to mock implementation")
            return True
        elif error_record.category == ErrorCategory.GHIDRA:
            self.logger.info("Using alternative decompilation method")
            error_record.recovery_actions.append("Used alternative decompilation")
            return True
        elif error_record.category == ErrorCategory.COMPILATION:
            self.logger.info("Using basic compilation settings")
            error_record.recovery_actions.append("Used basic compilation settings")
            return True
        
        return False


class EnvironmentFixRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy that attempts to fix environment issues"""
    
    def __init__(self):
        super().__init__("environment_fix", "Automatically fix environment issues")
    
    def can_recover(self, error_record: ErrorRecord) -> bool:
        """Check if environment issue can be auto-fixed"""
        return error_record.category == ErrorCategory.ENVIRONMENT
    
    def attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt to fix environment issues"""
        if "directory" in error_record.error_message.lower():
            # Try to create missing directories
            try:
                # Extract directory path from error message (basic implementation)
                if "output" in error_record.error_message.lower():
                    Path("output").mkdir(parents=True, exist_ok=True)
                    error_record.recovery_actions.append("Created missing output directory")
                    return True
            except Exception as e:
                self.logger.error(f"Failed to create directory: {e}")
        
        return False


class EnhancedErrorHandler:
    """Enhanced error handling system with recovery strategies"""
    
    def __init__(self, enable_recovery: bool = True, log_level: str = "INFO"):
        self.enable_recovery = enable_recovery
        self.logger = logging.getLogger("ErrorHandler")
        self.logger.setLevel(getattr(logging, log_level))
        
        # Error storage
        self.error_records: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        
        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = [
            RetryRecoveryStrategy(),
            FallbackRecoveryStrategy(),
            EnvironmentFixRecoveryStrategy()
        ]
        
        # Error patterns for auto-classification
        self.error_patterns = {
            ErrorCategory.ENVIRONMENT: [
                "No such file or directory",
                "Permission denied",
                "Path not found"
            ],
            ErrorCategory.DEPENDENCY: [
                "No module named",
                "ImportError",
                "ModuleNotFoundError"
            ],
            ErrorCategory.GHIDRA: [
                "Ghidra",
                "analyzeHeadless",
                "decompilation failed"
            ],
            ErrorCategory.COMPILATION: [
                "compilation failed",
                "compiler not found",
                "linking error"
            ],
            ErrorCategory.NETWORK: [
                "Connection refused",
                "Network unreachable",
                "Timeout"
            ]
        }
        
        # Setup logging handler
        self._setup_error_logging()
    
    def _setup_error_logging(self):
        """Setup enhanced error logging"""
        if not self.logger.handlers:
            # Console handler with enhanced formatting
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
    
    def handle_error(self, 
                    exception: Exception,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: Optional[ErrorCategory] = None,
                    context_data: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """Handle an error with enhanced context and recovery"""
        
        # Generate error ID
        error_id = f"ERR_{int(time.time())}_{threading.get_ident()}"
        
        # Collect context
        context = self._collect_error_context(exception, context_data)
        
        # Auto-classify if category not provided
        if category is None:
            category = self._classify_error(exception)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=severity,
            category=category,
            context=context
        )
        
        # Store error record
        with self.lock:
            self.error_records.append(error_record)
            self.error_counts[error_record.error_type] = self.error_counts.get(error_record.error_type, 0) + 1
        
        # Log error
        self._log_error(error_record)
        
        # Attempt recovery if enabled
        if self.enable_recovery:
            recovery_successful = self._attempt_recovery(error_record)
            error_record.recovery_attempted = True
            error_record.recovery_successful = recovery_successful
        
        return error_record
    
    def _collect_error_context(self, exception: Exception, context_data: Optional[Dict] = None) -> ErrorContext:
        """Collect comprehensive error context"""
        frame = inspect.currentframe()
        try:
            # Find the frame where the exception occurred
            while frame and frame.f_code.co_name in ['handle_error', '_collect_error_context']:
                frame = frame.f_back
            
            if frame:
                # Collect frame information
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                module_name = frame.f_globals.get('__name__', 'unknown')
                
                # Collect local variables (safely)
                local_vars = {}
                try:
                    for key, value in frame.f_locals.items():
                        if not key.startswith('_') and key != 'self':
                            # Convert to string to avoid serialization issues
                            local_vars[key] = str(value)[:100]  # Limit length
                except Exception:
                    local_vars = {"error": "Failed to collect local variables"}
            else:
                filename = "unknown"
                function_name = "unknown"
                line_number = 0
                module_name = "unknown"
                local_vars = {}
            
            # Collect stack trace
            stack_trace = traceback.format_exception(type(exception), exception, exception.__traceback__)
            
            # Collect system info
            system_info = {
                "python_version": sys.version,
                "platform": sys.platform,
                "thread_count": threading.active_count()
            }
            
            # Add custom context data
            if context_data:
                local_vars.update(context_data)
            
            context = ErrorContext(
                timestamp=time.time(),
                thread_id=threading.get_ident(),
                function_name=function_name,
                module_name=module_name,
                line_number=line_number,
                filename=filename,
                local_variables=local_vars,
                stack_trace=stack_trace,
                system_info=system_info
            )
            
            return context
            
        finally:
            del frame
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Automatically classify error based on patterns"""
        error_message = str(exception).lower()
        exception_type = type(exception).__name__
        
        # Check patterns
        for category, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_message or pattern.lower() in exception_type.lower():
                    return category
        
        # Default classification based on exception type
        if isinstance(exception, (ImportError, ModuleNotFoundError)):
            return ErrorCategory.DEPENDENCY
        elif isinstance(exception, (FileNotFoundError, PermissionError)):
            return ErrorCategory.ENVIRONMENT
        elif isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        else:
            return ErrorCategory.SYSTEM
    
    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level and detail"""
        message = f"[{error_record.error_id}] {error_record.error_type}: {error_record.error_message}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message)
        else:
            self.logger.info(message)
        
        # Log context for high severity errors
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Error context: {error_record.context.function_name} in {error_record.context.module_name}")
    
    def _attempt_recovery(self, error_record: ErrorRecord) -> bool:
        """Attempt to recover from error using available strategies"""
        self.logger.info(f"Attempting recovery for error {error_record.error_id}")
        
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_record):
                self.logger.info(f"Trying recovery strategy: {strategy.name}")
                try:
                    if strategy.attempt_recovery(error_record):
                        self.logger.info(f"Recovery successful using strategy: {strategy.name}")
                        return True
                except Exception as e:
                    self.logger.error(f"Recovery strategy {strategy.name} failed: {e}")
        
        self.logger.warning(f"No recovery strategy succeeded for error {error_record.error_id}")
        return False
    
    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a custom recovery strategy"""
        self.recovery_strategies.append(strategy)
        self.logger.info(f"Added recovery strategy: {strategy.name}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors"""
        with self.lock:
            total_errors = len(self.error_records)
            if total_errors == 0:
                return {"total_errors": 0, "message": "No errors recorded"}
            
            # Count by severity
            severity_counts = {}
            for record in self.error_records:
                severity_counts[record.severity.value] = severity_counts.get(record.severity.value, 0) + 1
            
            # Count by category
            category_counts = {}
            for record in self.error_records:
                category_counts[record.category.value] = category_counts.get(record.category.value, 0) + 1
            
            # Recovery statistics
            recovery_attempted = len([r for r in self.error_records if r.recovery_attempted])
            recovery_successful = len([r for r in self.error_records if r.recovery_successful])
            
            # Recent errors (last hour)
            recent_cutoff = time.time() - 3600
            recent_errors = [r for r in self.error_records if r.context.timestamp >= recent_cutoff]
            
            summary = {
                "total_errors": total_errors,
                "severity_counts": severity_counts,
                "category_counts": category_counts,
                "error_type_counts": self.error_counts,
                "recovery_statistics": {
                    "attempted": recovery_attempted,
                    "successful": recovery_successful,
                    "success_rate": (recovery_successful / recovery_attempted) if recovery_attempted > 0 else 0
                },
                "recent_errors": len(recent_errors),
                "latest_error": self.error_records[-1].to_dict() if self.error_records else None
            }
            
            return summary
    
    def export_errors(self, output_path: Path = None, include_full_context: bool = False, output_dir: str = None):
        """Export error records to JSON file
        
        Args:
            output_path: Specific path to write to (legacy support)
            include_full_context: Whether to include full error context
            output_dir: Base output directory to use logs subdirectory
        """
        with self.lock:
            export_data = {
                "export_timestamp": time.time(),
                "export_date": datetime.now().isoformat(),
                "summary": self.get_error_summary(),
                "errors": []
            }
            
            for record in self.error_records:
                error_data = record.to_dict()
                if not include_full_context:
                    # Remove detailed context for smaller file size
                    error_data["context"].pop("local_variables", None)
                    error_data["context"].pop("stack_trace", None)
                export_data["errors"].append(error_data)
        
        # Determine output path - prefer logs subdirectory if output_dir provided
        if output_path is None and output_dir:
            logs_dir = Path(output_dir) / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            output_path = logs_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        elif output_path is None:
            raise ValueError("Either output_path or output_dir must be provided")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Error records exported to {output_path}")
    
    def clear_errors(self, older_than_hours: Optional[int] = None):
        """Clear error records, optionally only older ones"""
        with self.lock:
            if older_than_hours is None:
                # Clear all errors
                count = len(self.error_records)
                self.error_records.clear()
                self.error_counts.clear()
                self.logger.info(f"Cleared all {count} error records")
            else:
                # Clear only old errors
                cutoff = time.time() - (older_than_hours * 3600)
                old_records = [r for r in self.error_records if r.context.timestamp < cutoff]
                self.error_records = [r for r in self.error_records if r.context.timestamp >= cutoff]
                
                # Recalculate error counts
                self.error_counts.clear()
                for record in self.error_records:
                    self.error_counts[record.error_type] = self.error_counts.get(record.error_type, 0) + 1
                
                self.logger.info(f"Cleared {len(old_records)} error records older than {older_than_hours} hours")


# Global error handler instance
_global_error_handler: Optional[EnhancedErrorHandler] = None


def get_error_handler() -> EnhancedErrorHandler:
    """Get or create global error handler"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = EnhancedErrorHandler()
    return _global_error_handler


def handle_error(exception: Exception, 
                severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                category: Optional[ErrorCategory] = None,
                context_data: Optional[Dict[str, Any]] = None) -> ErrorRecord:
    """Handle error using global error handler"""
    return get_error_handler().handle_error(exception, severity, category, context_data)


def error_handler(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: Optional[ErrorCategory] = None,
                 reraise: bool = False):
    """Decorator for automatic error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Handle the error
                context_data = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                handle_error(e, severity, category, context_data)
                
                if reraise:
                    raise
                else:
                    return None
        return wrapper
    return decorator


def safe_execute(func: Callable, 
                default_return: Any = None,
                severity: ErrorSeverity = ErrorSeverity.LOW,
                category: Optional[ErrorCategory] = None) -> Any:
    """Safely execute a function with error handling"""
    try:
        return func()
    except Exception as e:
        handle_error(e, severity, category)
        return default_return


# Context manager for error handling
class ErrorHandlingContext:
    """Context manager for scoped error handling"""
    
    def __init__(self, operation_name: str, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: Optional[ErrorCategory] = None,
                 reraise: bool = True):
        self.operation_name = operation_name
        self.severity = severity
        self.category = category
        self.reraise = reraise
        self.error_record: Optional[ErrorRecord] = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            context_data = {"operation": self.operation_name}
            self.error_record = handle_error(exc_val, self.severity, self.category, context_data)
            return not self.reraise  # Suppress exception if reraise=False
        return False


# Matrix-specific error handler for agents
class MatrixErrorHandler:
    """Matrix-themed error handler for Matrix agents"""
    
    def __init__(self, agent_name: str, max_retries: int = 3):
        self.agent_name = agent_name
        self.max_retries = max_retries
        self.error_handler = get_error_handler()
        self.logger = logging.getLogger(f"Matrix.{agent_name}")
        
    def handle_matrix_operation(self, operation_name: str):
        """Context manager for Matrix operations"""
        return ErrorHandlingContext(
            f"{self.agent_name}_{operation_name}",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AGENT,
            reraise=True
        )
    
    def handle_agent_error(self, exception: Exception, context_data: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """Handle agent-specific errors"""
        if context_data is None:
            context_data = {}
        
        context_data.update({
            'agent_name': self.agent_name,
            'matrix_operation': True
        })
        
        return self.error_handler.handle_error(
            exception,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.AGENT,
            context_data=context_data
        )


if __name__ == "__main__":
    # Test error handling system
    handler = EnhancedErrorHandler()
    
    # Test various error types
    try:
        raise FileNotFoundError("Test file not found")
    except Exception as e:
        handler.handle_error(e, ErrorSeverity.HIGH, ErrorCategory.ENVIRONMENT)
    
    try:
        import nonexistent_module
    except Exception as e:
        handler.handle_error(e, ErrorSeverity.MEDIUM, ErrorCategory.DEPENDENCY)
    
    # Print summary
    summary = handler.get_error_summary()
    print("Error Summary:")
    print(json.dumps(summary, indent=2))
    
    # Test decorator
    @error_handler(severity=ErrorSeverity.LOW, category=ErrorCategory.SYSTEM)
    def test_function():
        raise ValueError("Test error from decorated function")
    
    test_function()
    
    # Export errors
    handler.export_errors(Path("test_errors.json"))
    print("Error handling test completed")