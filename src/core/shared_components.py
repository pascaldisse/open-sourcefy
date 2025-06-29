"""
Shared Components for Matrix Agents
Common functionality to reduce boilerplate across all agents
"""

import logging
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from contextlib import contextmanager

from .config_manager import get_config_manager


@dataclass
class SharedMemory:
    """Shared memory structure for inter-agent communication"""
    binary_metadata: Dict[str, Any]
    analysis_results: Dict[str, Any]
    decompilation_data: Dict[str, Any] 
    reconstruction_info: Dict[str, Any]
    validation_status: Dict[str, Any]
    
    def __post_init__(self):
        # Initialize sub-dictionaries if not provided
        for field_name in ['binary_metadata', 'analysis_results', 'decompilation_data', 
                          'reconstruction_info', 'validation_status']:
            if getattr(self, field_name) is None:
                setattr(self, field_name, {})


class MatrixLogger:
    """Enhanced logging functionality for Matrix agents"""
    
    def __init__(self, agent_name: str, matrix_character: str):
        self.agent_name = agent_name
        self.matrix_character = matrix_character
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup standardized logger"""
        logger = logging.getLogger(f"Matrix.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[{self.matrix_character.upper()}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def matrix_entry(self, message: str):
        """Log agent entry into the Matrix"""
        self.logger.info(f"🎬 {message}")
    
    def matrix_progress(self, message: str, progress: float = None):
        """Log progress update"""
        if progress is not None:
            self.logger.info(f"⚡ {message} ({progress:.1%})")
        else:
            self.logger.info(f"⚡ {message}")
    
    def matrix_success(self, message: str):
        """Log successful completion"""
        self.logger.info(f"✅ {message}")
    
    def matrix_warning(self, message: str):
        """Log warning"""
        self.logger.warning(f"⚠️ {message}")
    
    def matrix_error(self, message: str, exc_info: bool = False):
        """Log error"""
        self.logger.error(f"❌ {message}", exc_info=exc_info)


class MatrixFileManager:
    """Standardized file operations for Matrix agents"""
    
    def __init__(self, output_paths: Dict[str, Path]):
        self.output_paths = output_paths
        self.config = get_config_manager()
    
    def save_agent_data(self, agent_id: int, matrix_character: str, data: Dict[str, Any]) -> Path:
        """Save agent data to standardized location"""
        agent_dir = self.output_paths['agents'] / f"agent_{agent_id:02d}_{matrix_character}"
        agent_dir.mkdir(exist_ok=True)
        
        output_file = agent_dir / f"agent_{agent_id:02d}_results.json"
        self.save_json(data, output_file)
        return output_file
    
    def save_json(self, data: Dict[str, Any], filepath: Path) -> None:
        """Save JSON data with error handling"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            raise IOError(f"Failed to save JSON to {filepath}: {e}")
    
    def load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load JSON data with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise IOError(f"Failed to load JSON from {filepath}: {e}")
    
    def save_binary_data(self, data: bytes, filepath: Path) -> None:
        """Save binary data"""
        try:
            with open(filepath, 'wb') as f:
                f.write(data)
        except Exception as e:
            raise IOError(f"Failed to save binary data to {filepath}: {e}")
    
    def create_agent_workspace(self, agent_id: int, matrix_character: str) -> Path:
        """Create workspace directory for agent"""
        workspace = self.output_paths['temp'] / f"agent_{agent_id:02d}_{matrix_character}"
        workspace.mkdir(exist_ok=True)
        return workspace


class MatrixValidator:
    """Common validation functions for Matrix agents"""
    
    @staticmethod
    def validate_binary_path(binary_path: Union[str, Path]) -> bool:
        """Validate binary file exists and is readable"""
        path = Path(binary_path)
        return path.exists() and path.is_file() and path.stat().st_size > 0
    
    @staticmethod
    def validate_context_keys(context: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate required context keys exist, return missing keys"""
        missing = []
        for key in required_keys:
            if key not in context:
                missing.append(key)
        return missing
    
    @staticmethod
    def validate_dependency_results(context: Dict[str, Any], dependencies: List[int]) -> List[int]:
        """Validate dependency results exist and succeeded, return failed dependencies"""
        agent_results = context.get('agent_results', {})
        failed = []
        
        for dep_id in dependencies:
            dep_result = agent_results.get(dep_id)
            if not dep_result or dep_result.status != 'success':
                failed.append(dep_id)
                
        return failed
    
    @staticmethod
    def validate_quality_threshold(score: float, threshold: float) -> bool:
        """Validate quality score meets threshold"""
        return score >= threshold
    
    @staticmethod
    def calculate_file_hash(filepath: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash for validation"""
        hash_func = getattr(hashlib, algorithm)()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()


class MatrixProgressTracker:
    """Progress tracking for Matrix agents"""
    
    def __init__(self, total_steps: int, agent_name: str):
        self.total_steps = total_steps
        self.current_step = 0
        self.agent_name = agent_name
        self.start_time = time.time()
        self.step_times = []
    
    def step(self, description: str) -> None:
        """Advance to next step"""
        self.current_step += 1
        step_time = time.time()
        self.step_times.append(step_time)
        
        progress = self.current_step / self.total_steps
        elapsed = step_time - self.start_time
        
        if self.current_step > 1:
            avg_step_time = elapsed / self.current_step
            eta = avg_step_time * (self.total_steps - self.current_step)
            eta_str = f" (ETA: {eta:.1f}s)"
        else:
            eta_str = ""
            
        logging.getLogger(f"Matrix.{self.agent_name}").info(
            f"⚡ Step {self.current_step}/{self.total_steps}: {description} "
            f"({progress:.1%}){eta_str}"
        )
    
    def complete(self) -> float:
        """Mark completion and return total time"""
        total_time = time.time() - self.start_time
        logging.getLogger(f"Matrix.{self.agent_name}").info(
            f"✅ All {self.total_steps} steps completed in {total_time:.2f}s"
        )
        return total_time


class MatrixErrorHandler:
    """Standardized error handling for Matrix agents"""
    
    def __init__(self, agent_name: str, max_retries: int = 2):
        self.agent_name = agent_name
        self.max_retries = max_retries
        self.retry_count = 0
        self.logger = logging.getLogger(f"Matrix.{agent_name}")
    
    @contextmanager
    def handle_matrix_operation(self, operation_name: str):
        """Context manager for handling Matrix operations with retry logic"""
        last_exception = None
        local_retry_count = 0
        
        while local_retry_count <= self.max_retries:
            try:
                yield
                return  # Success, exit completely
                
            except Exception as e:
                last_exception = e
                local_retry_count += 1
                if local_retry_count <= self.max_retries:
                    self.logger.warning(
                        f"⚠️ {operation_name} failed (attempt {local_retry_count}/{self.max_retries + 1}): {e}"
                    )
                    time.sleep(min(2 ** local_retry_count, 10))  # Exponential backoff
                else:
                    self.logger.error(f"❌ {operation_name} failed after {self.max_retries + 1} attempts: {e}")
                    raise last_exception
    
    def log_and_raise(self, message: str, exception_class: Exception = Exception) -> None:
        """Log error and raise exception"""
        self.logger.error(f"❌ {message}")
        raise exception_class(message)


class MatrixMetrics:
    """Performance metrics collection for Matrix agents"""
    
    def __init__(self, agent_id: int, matrix_character: str):
        self.agent_id = agent_id
        self.matrix_character = matrix_character
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'execution_time': 0.0,
            'memory_usage': {},
            'operations_count': 0,
            'errors_count': 0,
            'quality_scores': {}
        }
    
    def start_tracking(self):
        """Start metrics tracking"""
        self.metrics['start_time'] = time.time()
    
    def end_tracking(self):
        """End metrics tracking"""
        self.metrics['end_time'] = time.time()
        if self.metrics['start_time']:
            self.metrics['execution_time'] = self.metrics['end_time'] - self.metrics['start_time']
    
    def increment_operations(self, count: int = 1):
        """Increment operations counter"""
        self.metrics['operations_count'] += count
    
    def increment_errors(self, count: int = 1):
        """Increment errors counter"""
        self.metrics['errors_count'] += count
    
    def set_quality_score(self, metric_name: str, score: float):
        """Set quality score metric"""
        self.metrics['quality_scores'][metric_name] = score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = self.metrics.copy()
        summary.update({
            'agent_id': self.agent_id,
            'matrix_character': self.matrix_character,
            'success_rate': 1.0 - (self.metrics['errors_count'] / max(self.metrics['operations_count'], 1))
        })
        return summary
    
    @property
    def execution_time(self) -> float:
        """Get execution time"""
        return self.metrics['execution_time']
    
    @property
    def start_time(self) -> float:
        """Get start time"""
        return self.metrics['start_time'] or 0.0


def create_shared_memory() -> SharedMemory:
    """Create initialized shared memory structure"""
    return SharedMemory(
        binary_metadata={},
        analysis_results={},
        decompilation_data={},
        reconstruction_info={},
        validation_status={}
    )


def setup_output_structure(base_output_path: Path) -> Dict[str, Path]:
    """Setup standardized output directory structure"""
    output_paths = {
        'base': base_output_path,
        'agents': base_output_path / 'agents',
        'ghidra': base_output_path / 'ghidra', 
        'compilation': base_output_path / 'compilation',
        'reports': base_output_path / 'reports',
        'logs': base_output_path / 'logs',
        'temp': base_output_path / 'temp',
        'tests': base_output_path / 'tests',
        'docs': base_output_path / 'docs'
    }
    
    # Create all directories
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return output_paths


def get_matrix_config() -> Dict[str, Any]:
    """Get Matrix-specific configuration"""
    config = get_config_manager()
    
    return {
        'parallel_execution': config.get_value('matrix.parallel_execution', True),
        'batch_size': config.get_value('matrix.batch_size', 8),
        'agent_timeout': config.get_value('matrix.agent_timeout', 300),
        'max_retries': config.get_value('matrix.max_retries', 2),
        'quality_threshold': config.get_value('matrix.quality_threshold', 0.75),
        'enable_ai_enhancement': config.get_value('matrix.enable_ai_enhancement', True),
        'ghidra_timeout': config.get_value('ghidra.timeout', 600),
        'cleanup_temp': config.get_value('matrix.cleanup_temp', True)
    }


class SharedAnalysisTools:
    """Shared analysis tools for Matrix agents"""
    
    @staticmethod
    def calculate_entropy(data: bytes) -> float:
        """Calculate Shannon entropy of binary data"""
        import math
        
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        for count in byte_counts:
            if count > 0:
                p = count / data_len
                entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def detect_patterns(data: bytes, min_length: int = 4) -> List[Dict[str, Any]]:
        """Detect repeating patterns in binary data"""
        patterns = []
        data_len = len(data)
        
        for length in range(min_length, min(data_len // 4, 32)):
            seen_patterns = {}
            
            for i in range(data_len - length + 1):
                pattern = data[i:i + length]
                if pattern in seen_patterns:
                    seen_patterns[pattern].append(i)
                else:
                    seen_patterns[pattern] = [i]
            
            # Report patterns that occur multiple times
            for pattern, positions in seen_patterns.items():
                if len(positions) > 2:
                    patterns.append({
                        'pattern': pattern.hex(),
                        'length': length,
                        'occurrences': len(positions),
                        'positions': positions[:10]  # Limit to first 10
                    })
        
        return sorted(patterns, key=lambda x: x['occurrences'], reverse=True)
    
    @staticmethod
    def extract_strings(binary_path: Path, min_length: int = 4, max_strings: int = 1000) -> List[str]:
        """Extract printable strings from binary file"""
        import string
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            strings = []
            current_string = ""
            printable_chars = set(string.printable) - set(string.whitespace) | {' ', '\t'}
            
            for byte in data:
                char = chr(byte) if byte < 128 else None
                
                if char and char in printable_chars:
                    current_string += char
                else:
                    if len(current_string) >= min_length:
                        strings.append(current_string)
                        if len(strings) >= max_strings:
                            break
                    current_string = ""
            
            # Don't forget the last string
            if len(current_string) >= min_length:
                strings.append(current_string)
            
            return strings
            
        except Exception as e:
            return []


class SharedValidationTools:
    """Shared validation tools for Matrix agents"""
    
    @staticmethod
    def validate_context_keys(context: Dict[str, Any], required_keys: List[str]) -> List[str]:
        """Validate that all required context keys are present"""
        missing_keys = []
        for key in required_keys:
            if key not in context:
                missing_keys.append(key)
        return missing_keys
    
    @staticmethod
    def validate_dependency_results(context: Dict[str, Any], dependencies: List[int]) -> List[int]:
        """Validate dependency results exist and succeeded, return failed dependencies"""
        agent_results = context.get('agent_results', {})
        failed = []
        
        for dep_id in dependencies:
            dep_result = agent_results.get(dep_id)
            if not dep_result or not hasattr(dep_result, 'status'):
                failed.append(dep_id)
            else:
                # Handle both enum values and string values
                status_value = dep_result.status.value if hasattr(dep_result.status, 'value') else str(dep_result.status)
                if status_value != 'success':
                    failed.append(dep_id)
                
        return failed
    
    @staticmethod
    def validate_binary_path(binary_path: Path) -> bool:
        """Validate that binary path exists and is accessible"""
        try:
            return binary_path.exists() and binary_path.is_file() and binary_path.stat().st_size > 0
        except Exception:
            return False
    
    @staticmethod
    def validate_pe_structure(binary_path: Path) -> Dict[str, Any]:
        """Validate PE file structure"""
        try:
            with open(binary_path, 'rb') as f:
                # Check DOS header
                dos_header = f.read(64)
                if len(dos_header) < 64 or dos_header[:2] != b'MZ':
                    return {'valid': False, 'error': 'Invalid DOS header'}
                
                # Get PE offset
                pe_offset = int.from_bytes(dos_header[60:64], 'little')
                f.seek(pe_offset)
                
                # Check PE signature
                pe_sig = f.read(4)
                if pe_sig != b'PE\x00\x00':
                    return {'valid': False, 'error': 'Invalid PE signature'}
                
                return {'valid': True, 'pe_offset': pe_offset}
                
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    @staticmethod
    def validate_code_quality(code: str) -> Dict[str, Any]:
        """Validate generated code quality"""
        if not code or not code.strip():
            return {'quality_score': 0.0, 'issues': ['Empty code']}
        
        issues = []
        quality_score = 1.0
        
        # Check for placeholder patterns
        placeholder_patterns = [
            'TODO', 'FIXME', 'placeholder', 'dummy', 
            'undefined', 'unknown', 'temp'
        ]
        
        for pattern in placeholder_patterns:
            if pattern.lower() in code.lower():
                issues.append(f'Contains placeholder: {pattern}')
                quality_score -= 0.1
        
        # Check for basic C structure
        if 'int main(' not in code and 'void main(' not in code:
            if '#include' not in code:
                quality_score -= 0.2
                issues.append('Missing includes')
        
        # Check for function definitions
        if 'function' in code.lower() or 'def ' in code:
            quality_score -= 0.3
            issues.append('Contains non-C syntax')
        
        return {
            'quality_score': max(0.0, quality_score),
            'issues': issues
        }