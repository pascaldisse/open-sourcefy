"""
Centralized Error Handling for Open-Sourcefy Matrix System
Provides comprehensive error handling, retry logic, and recovery mechanisms.

This module implements production-ready error handling for the Matrix system:
- Centralized error handling with context management
- Configurable retry mechanisms with exponential backoff
- Error classification and recovery strategies
- Comprehensive logging and error reporting
- Circuit breaker patterns for stability

Production-ready implementation following SOLID principles and clean code standards.
"""

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Dict, Any, List, Optional, Callable, Type, Union, Generator
from datetime import datetime, timedelta

from .exceptions import (
    MatrixAgentError, ValidationError, ConfigurationError, 
    BinaryAnalysisError, DecompilationError, ReconstructionError,
    SecurityViolationError
)


class ErrorSeverity(Enum):
    """Error severity levels for classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for handling strategies"""
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    BINARY_ANALYSIS = "binary_analysis"
    DECOMPILATION = "decompilation"
    RECONSTRUCTION = "reconstruction"
    SYSTEM = "system"
    NETWORK = "network"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    MANUAL = "manual"


@dataclass
class ErrorInfo:
    """Comprehensive error information structure"""
    error_id: str
    error_type: str
    error_message: str
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    agent_id: Optional[int] = None
    operation: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    is_recoverable: bool = True


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_multiplier: float = 1.5


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascading failures"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_attempt: Optional[datetime] = None


class MatrixErrorHandler:
    """
    Centralized error handler for Matrix agents with comprehensive error management
    
    Features:
    - Configurable retry mechanisms with exponential backoff
    - Error classification and recovery strategies
    - Circuit breaker patterns for stability
    - Comprehensive error logging and reporting
    - Context-aware error handling
    """
    
    def __init__(
        self, 
        agent_name: str, 
        max_retries: int = 3,
        retry_config: Optional[RetryConfig] = None
    ):
        self.agent_name = agent_name
        self.max_retries = max_retries
        self.retry_config = retry_config or RetryConfig(max_attempts=max_retries)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Error classification mappings
        self.error_mappings = self._setup_error_mappings()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup error handler logging"""
        logger_name = f"ErrorHandler_{self.agent_name}"
        logger = logging.getLogger(logger_name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {logger_name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.WARNING)
        
        return logger
    
    def _setup_error_mappings(self) -> Dict[Type[Exception], Dict[str, Any]]:
        """Setup error type to category/severity mappings"""
        return {
            ValidationError: {
                'category': ErrorCategory.VALIDATION,
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.RETRY,
                'recoverable': True
            },
            ConfigurationError: {
                'category': ErrorCategory.CONFIGURATION,
                'severity': ErrorSeverity.HIGH,
                'recovery': RecoveryStrategy.ABORT,
                'recoverable': False
            },
            BinaryAnalysisError: {
                'category': ErrorCategory.BINARY_ANALYSIS,
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.FALLBACK,
                'recoverable': True
            },
            DecompilationError: {
                'category': ErrorCategory.DECOMPILATION,
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.RETRY,
                'recoverable': True
            },
            ReconstructionError: {
                'category': ErrorCategory.RECONSTRUCTION,
                'severity': ErrorSeverity.MEDIUM,
                'recovery': RecoveryStrategy.FALLBACK,
                'recoverable': True
            },
            TimeoutError: {
                'category': ErrorCategory.TIMEOUT,
                'severity': ErrorSeverity.HIGH,
                'recovery': RecoveryStrategy.RETRY,
                'recoverable': True
            },
            MemoryError: {
                'category': ErrorCategory.RESOURCE,
                'severity': ErrorSeverity.CRITICAL,
                'recovery': RecoveryStrategy.ABORT,
                'recoverable': False
            },
            FileNotFoundError: {
                'category': ErrorCategory.SYSTEM,
                'severity': ErrorSeverity.HIGH,
                'recovery': RecoveryStrategy.ABORT,
                'recoverable': False
            },
            PermissionError: {
                'category': ErrorCategory.SYSTEM,
                'severity': ErrorSeverity.HIGH,
                'recovery': RecoveryStrategy.ABORT,
                'recoverable': False
            },
            SecurityViolationError: {
                'category': ErrorCategory.SYSTEM,
                'severity': ErrorSeverity.CRITICAL,
                'recovery': RecoveryStrategy.ABORT,
                'recoverable': False
            }
        }
    
    def classify_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """
        Classify an error and create ErrorInfo structure
        
        Args:
            error: Exception to classify
            context: Additional context information
            
        Returns:
            ErrorInfo object with classification details
        """
        error_type = type(error)
        error_mapping = self.error_mappings.get(error_type, {
            'category': ErrorCategory.UNKNOWN,
            'severity': ErrorSeverity.MEDIUM,
            'recovery': RecoveryStrategy.RETRY,
            'recoverable': True
        })
        
        error_id = f"{self.agent_name}_{int(time.time())}"
        
        return ErrorInfo(
            error_id=error_id,
            error_type=error_type.__name__,
            error_message=str(error),
            category=error_mapping['category'],
            severity=error_mapping['severity'],
            recovery_strategy=error_mapping['recovery'],
            context=context or {},
            stack_trace=traceback.format_exc(),
            is_recoverable=error_mapping['recoverable'],
            max_retries=self.max_retries
        )
    
    def handle_error(
        self, 
        error: Exception, 
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[int] = None
    ) -> ErrorInfo:
        """
        Handle an error with comprehensive processing
        
        Args:
            error: Exception to handle
            operation: Name of the operation that failed
            context: Additional context information
            agent_id: ID of the agent that encountered the error
            
        Returns:
            ErrorInfo object with handling details
        """
        # Classify the error
        error_info = self.classify_error(error, context)
        error_info.operation = operation
        error_info.agent_id = agent_id
        
        # Log the error
        self._log_error(error_info)
        
        # Track error statistics
        self._track_error(error_info)
        
        # Add to error history
        self.error_history.append(error_info)
        
        return error_info
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error with appropriate level based on severity"""
        log_message = (
            f"Error in {error_info.operation or 'unknown operation'}: "
            f"{error_info.error_message}"
        )
        
        extra_info = {
            'error_id': error_info.error_id,
            'error_type': error_info.error_type,
            'category': error_info.category.value,
            'severity': error_info.severity.value,
            'agent_id': error_info.agent_id,
            'recoverable': error_info.is_recoverable
        }
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra=extra_info)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra=extra_info)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra=extra_info)
        else:
            self.logger.info(log_message, extra=extra_info)
    
    def _track_error(self, error_info: ErrorInfo) -> None:
        """Track error statistics for monitoring"""
        error_key = f"{error_info.category.value}_{error_info.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """
        Determine if an operation should be retried based on error info
        
        Args:
            error_info: Error information
            
        Returns:
            True if operation should be retried, False otherwise
        """
        # Check if error is recoverable
        if not error_info.is_recoverable:
            return False
        
        # Check retry count
        if error_info.retry_count >= error_info.max_retries:
            return False
        
        # Check recovery strategy
        if error_info.recovery_strategy not in [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]:
            return False
        
        # Check circuit breaker
        operation_key = error_info.operation or "default"
        if self._is_circuit_open(operation_key):
            return False
        
        return True
    
    def execute_with_retry(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[int] = None,
        custom_retry_config: Optional[RetryConfig] = None
    ) -> Any:
        """
        Execute an operation with retry logic
        
        Args:
            operation: Function to execute
            operation_name: Name of the operation for logging
            context: Additional context information
            agent_id: ID of the agent executing the operation
            custom_retry_config: Custom retry configuration
            
        Returns:
            Result of the operation
            
        Raises:
            Last exception if all retries failed
        """
        retry_config = custom_retry_config or self.retry_config
        last_error = None
        
        for attempt in range(retry_config.max_attempts):
            try:
                # Check circuit breaker
                if self._is_circuit_open(operation_name):
                    raise Exception(f"Circuit breaker open for operation: {operation_name}")
                
                # Execute operation
                result = operation()
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker(operation_name)
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Handle the error
                error_info = self.handle_error(e, operation_name, context, agent_id)
                error_info.retry_count = attempt
                
                # Update circuit breaker
                self._record_failure(operation_name)
                
                # Check if we should retry
                if attempt < retry_config.max_attempts - 1 and self.should_retry(error_info):
                    delay = self._calculate_retry_delay(attempt, retry_config)
                    self.logger.info(
                        f"Retrying {operation_name} in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_attempts})"
                    )
                    time.sleep(delay)
                else:
                    # No more retries or not retryable
                    break
        
        # All retries failed
        if last_error:
            raise last_error
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt with exponential backoff"""
        delay = config.base_delay * (config.exponential_base ** attempt)
        delay = min(delay * config.backoff_multiplier, config.max_delay)
        
        # Add jitter if enabled
        if config.jitter:
            import random
            jitter = delay * 0.1 * random.random()
            delay += jitter
        
        return delay
    
    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for an operation"""
        if operation not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[operation]
        
        if breaker.state == CircuitBreakerState.CLOSED:
            return False
        elif breaker.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (breaker.next_attempt and 
                datetime.now() >= breaker.next_attempt):
                breaker.state = CircuitBreakerState.HALF_OPEN
                return False
            return True
        else:  # HALF_OPEN
            return False
    
    def _record_failure(self, operation: str) -> None:
        """Record a failure for circuit breaker tracking"""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker()
        
        breaker = self.circuit_breakers[operation]
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.now()
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = CircuitBreakerState.OPEN
            breaker.next_attempt = datetime.now() + timedelta(seconds=breaker.recovery_timeout)
    
    def _reset_circuit_breaker(self, operation: str) -> None:
        """Reset circuit breaker on successful operation"""
        if operation in self.circuit_breakers:
            breaker = self.circuit_breakers[operation]
            breaker.failure_count = 0
            breaker.state = CircuitBreakerState.CLOSED
            breaker.next_attempt = None
    
    @contextmanager
    def handle_matrix_operation(
        self, 
        operation_name: str,
        context: Optional[Dict[str, Any]] = None,
        agent_id: Optional[int] = None
    ) -> Generator[None, None, None]:
        """
        Context manager for handling Matrix operations with comprehensive error handling
        
        Args:
            operation_name: Name of the operation
            context: Additional context information
            agent_id: ID of the agent executing the operation
            
        Yields:
            None
            
        Raises:
            MatrixAgentError: Wrapped exception with Matrix context
        """
        try:
            yield
        except Exception as e:
            error_info = self.handle_error(e, operation_name, context, agent_id)
            
            # Wrap in MatrixAgentError with additional context
            raise MatrixAgentError(
                f"Operation '{operation_name}' failed in {self.agent_name}: {str(e)}",
                agent_id=agent_id or 0,
                error_id=error_info.error_id,
                category=error_info.category.value,
                severity=error_info.severity.value,
                recoverable=error_info.is_recoverable
            ) from e
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error statistics"""
        return {
            'total_errors': len(self.error_history),
            'error_counts_by_type': self.error_counts.copy(),
            'recent_errors': [
                {
                    'error_id': err.error_id,
                    'operation': err.operation,
                    'error_type': err.error_type,
                    'severity': err.severity.value,
                    'timestamp': err.timestamp.isoformat()
                }
                for err in self.error_history[-10:]  # Last 10 errors
            ],
            'circuit_breaker_status': {
                op: {
                    'state': breaker.state.value,
                    'failure_count': breaker.failure_count,
                    'last_failure': breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
                }
                for op, breaker in self.circuit_breakers.items()
            }
        }
    
    def clear_error_history(self) -> None:
        """Clear error history and reset statistics"""
        self.error_history.clear()
        self.error_counts.clear()
        self.circuit_breakers.clear()


# Decorator for automatic error handling
def handle_matrix_errors(
    operation_name: Optional[str] = None,
    max_retries: int = 3,
    agent_name: Optional[str] = None
):
    """
    Decorator for automatic error handling in Matrix operations
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        max_retries: Maximum retry attempts
        agent_name: Name of the agent (defaults to function's module)
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create error handler
            handler_name = agent_name or func.__module__
            error_handler = MatrixErrorHandler(handler_name, max_retries)
            
            # Execute with error handling
            op_name = operation_name or func.__name__
            return error_handler.execute_with_retry(
                lambda: func(*args, **kwargs),
                op_name
            )
        
        return wrapper
    return decorator


# Global error handler instance for module-level operations
_global_error_handler: Optional[MatrixErrorHandler] = None


def get_global_error_handler(agent_name: str = "Global") -> MatrixErrorHandler:
    """Get or create global error handler instance"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = MatrixErrorHandler(agent_name)
    return _global_error_handler


def log_matrix_error(
    error: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    agent_id: Optional[int] = None
) -> ErrorInfo:
    """
    Convenience function to log errors using global error handler
    
    Args:
        error: Exception to log
        operation: Operation name
        context: Additional context
        agent_id: Agent ID
        
    Returns:
        ErrorInfo object
    """
    handler = get_global_error_handler()
    return handler.handle_error(error, operation, context, agent_id)