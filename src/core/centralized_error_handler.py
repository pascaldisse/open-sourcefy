"""
Centralized Error Handler for Matrix Pipeline
Provides robust error handling and recovery mechanisms.
"""

import logging
import traceback
import sys
from typing import Any, Dict, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Base exception for pipeline errors"""
    pass

class RecoverableError(PipelineError):
    """Error that can be recovered from"""
    pass

class CriticalError(PipelineError):
    """Error that requires pipeline termination"""
    pass

def safe_execute(error_type: type = Exception, 
                default_return: Any = None,
                log_error: bool = True) -> Callable:
    """Decorator for safe function execution with error handling"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except error_type as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
            except Exception as e:
                if log_error:
                    logger.critical(f"Unexpected error in {func.__name__}: {e}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        return wrapper
    return decorator

def handle_agent_error(agent_id: int, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle agent execution errors gracefully"""
    
    error_result = {
        'agent_id': agent_id,
        'status': 'failed',
        'error': str(error),
        'error_type': type(error).__name__,
        'recoverable': isinstance(error, RecoverableError),
        'context_preserved': True
    }
    
    # Log appropriate level based on error type
    if isinstance(error, CriticalError):
        logger.critical(f"Critical error in Agent {agent_id}: {error}")
    elif isinstance(error, RecoverableError):
        logger.warning(f"Recoverable error in Agent {agent_id}: {error}")
    else:
        logger.error(f"Error in Agent {agent_id}: {error}")
    
    return error_result
