"""
Matrix Framework Exception Classes
Production-ready exception hierarchy for Matrix agents and pipeline
"""

from typing import Optional, Dict, Any, List


class MatrixError(Exception):
    """Base exception for all Matrix framework errors"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class MatrixAgentError(MatrixError):
    """Exception raised by Matrix agents during execution"""
    
    def __init__(
        self, 
        message: str, 
        agent_id: Optional[int] = None, 
        matrix_character: Optional[str] = None, 
        context: Optional[Dict[str, Any]] = None,
        error_id: Optional[str] = None,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        recoverable: bool = True
    ):
        context = context or {}
        if agent_id is not None:
            context['agent_id'] = agent_id
        if matrix_character is not None:
            context['matrix_character'] = matrix_character
        if error_id is not None:
            context['error_id'] = error_id
        if category is not None:
            context['category'] = category
        if severity is not None:
            context['severity'] = severity
        context['recoverable'] = recoverable
        
        super().__init__(message, context)
        self.agent_id = agent_id
        self.matrix_character = matrix_character
        self.error_id = error_id
        self.category = category
        self.severity = severity
        self.recoverable = recoverable


class ValidationError(MatrixError):
    """Exception raised when validation fails"""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, 
                 expected: Optional[Any] = None, actual: Optional[Any] = None):
        context = {}
        if validation_type:
            context['validation_type'] = validation_type
        if expected is not None:
            context['expected'] = expected
        if actual is not None:
            context['actual'] = actual
        
        super().__init__(message, context)
        self.validation_type = validation_type
        self.expected = expected
        self.actual = actual


class ConfigurationError(MatrixError):
    """Exception raised when configuration is invalid or missing"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 config_file: Optional[str] = None):
        context = {}
        if config_key:
            context['config_key'] = config_key
        if config_file:
            context['config_file'] = config_file
        
        super().__init__(message, context)
        self.config_key = config_key
        self.config_file = config_file


class PipelineFailureError(MatrixError):
    """Exception raised when pipeline execution fails"""
    
    def __init__(self, message: str, failed_agent_id: Optional[int] = None, 
                 pipeline_stage: Optional[str] = None):
        context = {}
        if failed_agent_id is not None:
            context['failed_agent_id'] = failed_agent_id
        if pipeline_stage:
            context['pipeline_stage'] = pipeline_stage
        
        super().__init__(message, context)
        self.failed_agent_id = failed_agent_id
        self.pipeline_stage = pipeline_stage


class DependencyError(MatrixError):
    """Exception raised when agent dependencies are not satisfied"""
    
    def __init__(self, message: str, missing_dependencies: Optional[list] = None,
                 agent_id: Optional[int] = None):
        context = {}
        if missing_dependencies:
            context['missing_dependencies'] = missing_dependencies
        if agent_id is not None:
            context['agent_id'] = agent_id
        
        super().__init__(message, context)
        self.missing_dependencies = missing_dependencies or []
        self.agent_id = agent_id


class QualityThresholdError(ValidationError):
    """Exception raised when quality metrics don't meet minimum thresholds"""
    
    def __init__(self, message: str, quality_score: float, threshold: float, 
                 metric_name: Optional[str] = None):
        super().__init__(
            message, 
            validation_type='quality_threshold',
            expected=threshold,
            actual=quality_score
        )
        self.quality_score = quality_score
        self.threshold = threshold
        self.metric_name = metric_name


class BinaryAnalysisError(MatrixAgentError):
    """Exception raised during binary analysis operations"""
    
    def __init__(self, message: str, binary_path: Optional[str] = None,
                 analysis_stage: Optional[str] = None):
        context = {}
        if binary_path:
            context['binary_path'] = binary_path
        if analysis_stage:
            context['analysis_stage'] = analysis_stage
        
        super().__init__(message, context=context)
        self.binary_path = binary_path
        self.analysis_stage = analysis_stage


class GhidraIntegrationError(MatrixAgentError):
    """Exception raised during Ghidra integration operations"""
    
    def __init__(self, message: str, ghidra_operation: Optional[str] = None,
                 ghidra_exit_code: Optional[int] = None):
        context = {}
        if ghidra_operation:
            context['ghidra_operation'] = ghidra_operation
        if ghidra_exit_code is not None:
            context['ghidra_exit_code'] = ghidra_exit_code
        
        super().__init__(message, context=context)
        self.ghidra_operation = ghidra_operation
        self.ghidra_exit_code = ghidra_exit_code


class LangChainAgentError(MatrixAgentError):
    """Exception raised during LangChain agent operations"""
    
    def __init__(self, message: str, langchain_operation: Optional[str] = None,
                 ai_model_error: Optional[str] = None):
        context = {}
        if langchain_operation:
            context['langchain_operation'] = langchain_operation
        if ai_model_error:
            context['ai_model_error'] = ai_model_error
        
        super().__init__(message, context=context)
        self.langchain_operation = langchain_operation
        self.ai_model_error = ai_model_error


class ResourceError(MatrixError):
    """Exception raised when system resources are insufficient"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 required: Optional[Any] = None, available: Optional[Any] = None):
        context = {}
        if resource_type:
            context['resource_type'] = resource_type
        if required is not None:
            context['required'] = required
        if available is not None:
            context['available'] = available
        
        super().__init__(message, context)
        self.resource_type = resource_type
        self.required = required
        self.available = available


class TimeoutError(MatrixError):
    """Exception raised when operations exceed timeout limits"""
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None):
        context = {}
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        if operation:
            context['operation'] = operation
        
        super().__init__(message, context)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class DecompilationError(MatrixAgentError):
    """Exception raised during decompilation operations"""
    
    def __init__(self, message: str, binary_path: Optional[str] = None,
                 decompilation_stage: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if binary_path:
            context['binary_path'] = binary_path
        if decompilation_stage:
            context['decompilation_stage'] = decompilation_stage
        
        super().__init__(message, context=context, **kwargs)
        self.binary_path = binary_path
        self.decompilation_stage = decompilation_stage


class ReconstructionError(MatrixAgentError):
    """Exception raised during reconstruction operations"""
    
    def __init__(self, message: str, reconstruction_stage: Optional[str] = None,
                 source_files: Optional[List[str]] = None, **kwargs):
        context = kwargs.get('context', {})
        if reconstruction_stage:
            context['reconstruction_stage'] = reconstruction_stage
        if source_files:
            context['source_files'] = source_files
        
        super().__init__(message, context=context, **kwargs)
        self.reconstruction_stage = reconstruction_stage
        self.source_files = source_files or []


class SecurityViolationError(MatrixError):
    """Exception raised when security policies are violated"""
    
    def __init__(self, message: str, violation_type: str, 
                 security_context: Optional[Dict[str, Any]] = None):
        context = security_context or {}
        context['violation_type'] = violation_type
        context['security_level'] = 'CRITICAL'
        
        super().__init__(message, context)
        self.violation_type = violation_type


class SelfCorrectionError(MatrixError):
    """Exception raised during self-correction operations"""
    
    def __init__(self, message: str, correction_cycle: Optional[int] = None,
                 correction_strategy: Optional[str] = None):
        context = {}
        if correction_cycle is not None:
            context['correction_cycle'] = correction_cycle
        if correction_strategy:
            context['correction_strategy'] = correction_strategy
        
        super().__init__(message, context)
        self.correction_cycle = correction_cycle
        self.correction_strategy = correction_strategy


class FunctionalIdentityError(MatrixError):
    """Exception raised when 100% functional identity requirement fails"""
    
    def __init__(self, message: str, identity_score: Optional[float] = None,
                 required_score: float = 1.0, difference_details: Optional[Dict] = None):
        context = {}
        if identity_score is not None:
            context['identity_score'] = identity_score
        context['required_score'] = required_score
        if difference_details:
            context['difference_details'] = difference_details
        
        super().__init__(message, context)
        self.identity_score = identity_score
        self.required_score = required_score
        self.difference_details = difference_details or {}


class BinaryDiffError(MatrixError):
    """Exception raised during binary diff analysis operations"""
    
    def __init__(self, message: str, diff_level: Optional[str] = None,
                 binary_paths: Optional[Dict[str, str]] = None):
        context = {}
        if diff_level:
            context['diff_level'] = diff_level
        if binary_paths:
            context['binary_paths'] = binary_paths
        
        super().__init__(message, context)
        self.diff_level = diff_level
        self.binary_paths = binary_paths or {}


# Exception handling utilities
def handle_matrix_exception(e: Exception, logger, agent_id: Optional[int] = None) -> MatrixError:
    """
    Convert any exception to appropriate Matrix exception with logging
    
    Args:
        e: Original exception
        logger: Logger instance for error logging
        agent_id: Optional agent ID for context
    
    Returns:
        MatrixError subclass appropriate for the exception type
    """
    # Log the original exception
    logger.error(f"Exception occurred: {type(e).__name__}: {e}", exc_info=True)
    
    # Convert to appropriate Matrix exception
    if isinstance(e, MatrixError):
        return e
    elif isinstance(e, FileNotFoundError):
        return ConfigurationError(f"Required file not found: {e}", context={'original_error': str(e)})
    elif isinstance(e, PermissionError):
        return ResourceError(f"Permission denied: {e}", resource_type='file_access', context={'original_error': str(e)})
    elif isinstance(e, MemoryError):
        return ResourceError(f"Insufficient memory: {e}", resource_type='memory', context={'original_error': str(e)})
    elif isinstance(e, TimeoutError):
        return TimeoutError(f"Operation timed out: {e}", context={'original_error': str(e)})
    else:
        # Generic Matrix agent error for unhandled exceptions
        return MatrixAgentError(
            f"Unexpected error: {e}",
            agent_id=agent_id,
            context={'original_error': str(e), 'original_type': type(e).__name__}
        )