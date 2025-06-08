"""
Agent 00: Deus Ex Machina - Master Orchestrator
The Machine God that prepares the global context and coordinates the Matrix

This agent serves as the master orchestrator that:
1. Validates the entire system environment
2. Prepares global execution context
3. Initializes shared resources
4. Coordinates agent execution order
5. Manages system-wide configuration
"""

import os
import sys
import time
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from ..matrix_agent_base import MatrixAgentBase, AgentResult, AgentStatus, AgentType
from ..config_manager import get_config_manager
from ..binary_utils import BinaryAnalyzer, BinaryInfo
from ..file_utils import FileManager
from ..ai_engine_interface import get_ai_engine


class DeusExMachinaAgent(MatrixAgentBase):
    """The Machine God - Master orchestrator for the Matrix pipeline"""
    
    def __init__(self):
        super().__init__(
            agent_id=0,
            agent_name="DeusExMachina",
            matrix_character="Deus Ex Machina",
            agent_type=AgentType.COLLECTIVE_AI
        )
        
        # System validation results
        self.system_status: Dict[str, Any] = {}
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        # Global context preparation
        self.global_context: Dict[str, Any] = {}

    def get_description(self) -> str:
        """Get agent description"""
        return ("The Machine God that orchestrates the entire Matrix pipeline. "
                "Validates system environment, prepares global context, and "
                "coordinates the execution of all Matrix agents.")
    
    def get_dependencies(self) -> List[int]:
        """Deus Ex Machina has no dependencies - it is the beginning"""
        return []
    
    def get_prerequisites(self) -> List[str]:
        """Prerequisites for the master orchestrator"""
        return ["binary_file"]

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the master orchestration process"""
        self.logger.info("🔮 Deus Ex Machina awakens to orchestrate the Matrix...")
        
        start_time = time.time()
        
        try:
            # Phase 1: System Environment Validation
            self.logger.info("Phase 1: Validating system environment...")
            system_valid = self._validate_system_environment(context)
            
            if not system_valid:
                return self._create_validation_failure_result()
            
            # Phase 2: Binary Analysis and Validation
            self.logger.info("Phase 2: Analyzing target binary...")
            binary_info = self._analyze_target_binary(context)
            
            if not binary_info:
                return self._create_binary_failure_result()
            
            # Phase 3: Global Context Preparation
            self.logger.info("Phase 3: Preparing global execution context...")
            self._prepare_global_context(context, binary_info)
            
            # Phase 4: Resource Initialization
            self.logger.info("Phase 4: Initializing shared resources...")
            self._initialize_shared_resources(context)
            
            # Phase 5: Agent Coordination Setup
            self.logger.info("Phase 5: Setting up agent coordination...")
            execution_plan = self._create_execution_plan(context)
            
            # Phase 6: AI Engine Validation
            self.logger.info("Phase 6: Validating AI engine connectivity...")
            ai_status = self._validate_ai_engine()
            
            execution_time = time.time() - start_time
            
            # Create comprehensive result
            result_data = {
                'system_status': self.system_status,
                'binary_info': self._binary_info_to_dict(binary_info),
                'global_context': self.global_context,
                'execution_plan': execution_plan,
                'ai_engine_status': ai_status,
                'validation_summary': {
                    'errors': self.validation_errors,
                    'warnings': self.validation_warnings,
                    'total_errors': len(self.validation_errors),
                    'total_warnings': len(self.validation_warnings)
                }
            }
            
            # Store global context in the main context
            context.update(self.global_context)
            
            self.logger.info(f"🎯 Deus Ex Machina orchestration completed in {execution_time:.2f}s")
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                status=AgentStatus.SUCCESS,
                data=result_data,
                execution_time=execution_time,
                metadata={
                    'matrix_character': self.matrix_character,
                    'system_validated': True,
                    'binary_analyzed': True,
                    'context_prepared': True,
                    'total_warnings': len(self.validation_warnings),
                    'orchestration_version': '2.0.0'
                }
            )
            
        except Exception as e:
            self.logger.error(f"💥 Deus Ex Machina orchestration failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                status=AgentStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - start_time,
                metadata={'matrix_character': self.matrix_character}
            )

    def _validate_system_environment(self, context: Dict[str, Any]) -> bool:
        """Validate the entire system environment"""
        self.logger.info("Validating system environment...")
        
        # System information
        self.system_status['platform'] = sys.platform
        self.system_status['python_version'] = sys.version
        self.system_status['cpu_count'] = psutil.cpu_count()
        self.system_status['memory_total'] = psutil.virtual_memory().total
        self.system_status['memory_available'] = psutil.virtual_memory().available
        self.system_status['disk_free'] = psutil.disk_usage('.').free
        
        # Python version check
        if sys.version_info < (3, 8):
            self.validation_errors.append("Python 3.8+ required")
        
        # Memory check
        memory_gb = psutil.virtual_memory().total / (1024**3)
        min_memory = self.config.get_value('pipeline.min_memory_gb', 4)
        if memory_gb < min_memory:
            self.validation_warnings.append(f"Low memory: {memory_gb:.1f}GB < {min_memory}GB recommended")
        
        # Disk space check
        disk_gb = psutil.disk_usage('.').free / (1024**3)
        min_disk = self.config.get_value('pipeline.min_disk_gb', 2)
        if disk_gb < min_disk:
            self.validation_warnings.append(f"Low disk space: {disk_gb:.1f}GB < {min_disk}GB recommended")
        
        # Validate configuration
        config_issues = self._validate_configuration()
        self.validation_errors.extend(config_issues)
        
        # Validate tools
        tool_issues = self._validate_external_tools()
        self.validation_warnings.extend(tool_issues)
        
        # Validate permissions
        perm_issues = self._validate_permissions(context)
        self.validation_errors.extend(perm_issues)
        
        self.system_status['validation_complete'] = True
        self.system_status['has_errors'] = len(self.validation_errors) > 0
        self.system_status['has_warnings'] = len(self.validation_warnings) > 0
        
        if self.validation_errors:
            self.logger.error(f"System validation failed with {len(self.validation_errors)} errors")
            for error in self.validation_errors:
                self.logger.error(f"  ❌ {error}")
        
        if self.validation_warnings:
            self.logger.warning(f"System validation completed with {len(self.validation_warnings)} warnings")
            for warning in self.validation_warnings:
                self.logger.warning(f"  ⚠️ {warning}")
        
        return len(self.validation_errors) == 0
    
    def _validate_configuration(self) -> List[str]:
        """Validate system configuration"""
        issues = []
        
        # Check critical configuration values
        critical_configs = [
            ('pipeline.batch_size', int, 1, 32),
            ('pipeline.timeout_agent', int, 30, 3600),
            ('ai_engine.provider', str, None, None)
        ]
        
        for config_key, expected_type, min_val, max_val in critical_configs:
            value = self.config.get_value(config_key)
            
            if value is None:
                issues.append(f"Missing configuration: {config_key}")
                continue
            
            if not isinstance(value, expected_type):
                issues.append(f"Invalid type for {config_key}: expected {expected_type.__name__}")
                continue
            
            if expected_type in (int, float) and min_val is not None and value < min_val:
                issues.append(f"Configuration {config_key} too low: {value} < {min_val}")
            
            if expected_type in (int, float) and max_val is not None and value > max_val:
                issues.append(f"Configuration {config_key} too high: {value} > {max_val}")
        
        return issues
    
    def _validate_external_tools(self) -> List[str]:
        """Validate external tools availability"""
        warnings = []
        
        # Ghidra validation
        ghidra_path = self.config.get_path('ghidra_home')
        if not ghidra_path:
            warnings.append("Ghidra path not configured (auto-detection will be attempted)")
        elif not Path(ghidra_path).exists():
            warnings.append(f"Ghidra not found at configured path: {ghidra_path}")
        else:
            # Check for headless script
            headless_script = Path(ghidra_path) / "support" / "analyzeHeadless"
            if not headless_script.exists():
                warnings.append("Ghidra headless script not found")
            else:
                self.system_status['ghidra_available'] = True
        
        # Java validation (for Ghidra)
        java_path = self.config.get_value('ghidra.java_path')
        if java_path:
            if not Path(java_path).exists():
                warnings.append(f"Java not found at configured path: {java_path}")
        
        # Check for optional tools
        optional_tools = {
            'git': 'Git version control',
            'cmake': 'CMake build system',
            'msbuild': 'MSBuild compilation'
        }
        
        for tool, description in optional_tools.items():
            if not self._check_tool_available(tool):
                warnings.append(f"{description} not available in PATH")
        
        return warnings
    
    def _check_tool_available(self, tool_name: str) -> bool:
        """Check if a command-line tool is available"""
        try:
            import subprocess
            result = subprocess.run([tool_name, '--version'], 
                                   capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _validate_permissions(self, context: Dict[str, Any]) -> List[str]:
        """Validate file system permissions"""
        issues = []
        
        # Check binary file access
        binary_path = context.get('binary_path')
        if binary_path:
            binary_path = Path(binary_path)
            if not os.access(binary_path, os.R_OK):
                issues.append(f"Cannot read binary file: {binary_path}")
        
        # Check output directory permissions
        output_dir = self.config.get_path('default_output_dir', 'output')
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = output_path / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(f"Cannot write to output directory {output_dir}: {e}")
        
        return issues
    
    def _analyze_target_binary(self, context: Dict[str, Any]) -> Optional[BinaryInfo]:
        """Analyze the target binary file"""
        binary_path = context.get('binary_path')
        if not binary_path:
            self.validation_errors.append("No binary file specified")
            return None
        
        binary_path = Path(binary_path)
        if not binary_path.exists():
            self.validation_errors.append(f"Binary file not found: {binary_path}")
            return None
        
        try:
            analyzer = BinaryAnalyzer()
            binary_info = analyzer.analyze_file(binary_path)
            
            # Store binary info in system status
            self.system_status['binary_analysis'] = {
                'format': binary_info.format.value,
                'architecture': binary_info.architecture.value,
                'size': binary_info.file_size,
                'is_64bit': binary_info.is_64bit,
                'is_packed': binary_info.is_packed,
                'section_count': len(binary_info.sections),
                'import_count': len(binary_info.imports),
                'export_count': len(binary_info.exports)
            }
            
            # Validate binary compatibility
            if binary_info.format.value == 'unknown':
                self.validation_warnings.append("Unknown binary format detected")
            
            if binary_info.is_packed:
                self.validation_warnings.append("Binary appears to be packed/compressed")
            
            if binary_info.file_size > 100 * 1024 * 1024:  # 100MB
                self.validation_warnings.append(f"Large binary file: {binary_info.file_size // (1024*1024)}MB")
            
            self.logger.info(f"Binary analysis complete: {binary_info.format.value} {binary_info.architecture.value}")
            return binary_info
            
        except Exception as e:
            self.validation_errors.append(f"Binary analysis failed: {e}")
            return None
    
    def _prepare_global_context(self, context: Dict[str, Any], binary_info: BinaryInfo) -> None:
        """Prepare global execution context for all agents"""
        self.logger.info("Preparing global execution context...")
        
        # Create timestamped output directory structure
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_manager = FileManager()
        output_base = Path(self.config.get_path('default_output_dir', 'output'))
        session_dir = output_base / session_id
        
        output_paths = file_manager.ensure_output_structure(session_dir)
        
        # Global context dictionary
        self.global_context = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'binary_path': str(binary_info.file_path),
            'binary_info': binary_info,
            'output_paths': output_paths,
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version_info[:3],
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3)
            },
            'pipeline_config': {
                'parallel_mode': self.config.get_value('pipeline.parallel_mode', 'thread'),
                'batch_size': self.config.get_value('pipeline.batch_size', 4),
                'timeout_agent': self.config.get_value('pipeline.timeout_agent', 300),
                'max_retries': self.config.get_value('pipeline.max_retries', 3)
            },
            'agent_results': {},  # Will be populated as agents execute
            'shared_memory': {},  # Shared data between agents
            'execution_metrics': {
                'start_time': time.time(),
                'agents_completed': 0,
                'agents_failed': 0,
                'total_execution_time': 0.0
            }
        }
        
        self.logger.info(f"Global context prepared for session: {session_id}")
    
    def _initialize_shared_resources(self, context: Dict[str, Any]) -> None:
        """Initialize shared resources for agent communication"""
        self.logger.info("Initializing shared resources...")
        
        # Create shared memory structure
        shared_memory = {
            'binary_metadata': {},
            'analysis_cache': {},
            'decompilation_results': {},
            'compilation_artifacts': {},
            'validation_results': {},
            'performance_metrics': {}
        }
        
        self.global_context['shared_memory'] = shared_memory
        
        # Initialize output directory logging
        log_dir = self.global_context['output_paths']['logs']
        self._setup_session_logging(log_dir)
        
        self.logger.info("Shared resources initialized")
    
    def _setup_session_logging(self, log_dir: Path) -> None:
        """Setup session-wide logging configuration"""
        import logging
        
        # Create session log file
        session_log = log_dir / f"session_{self.global_context['session_id']}.log"
        
        # Configure session logger
        session_logger = logging.getLogger("Matrix.Session")
        if not session_logger.handlers:
            file_handler = logging.FileHandler(session_log)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            session_logger.addHandler(file_handler)
            session_logger.setLevel(logging.INFO)
        
        session_logger.info(f"Matrix session started: {self.global_context['session_id']}")
    
    def _create_execution_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution plan for Matrix agents"""
        self.logger.info("Creating agent execution plan...")
        
        # Define Matrix agent execution order and dependencies
        agent_plan = {
            'execution_order': [
                {'batch': 1, 'agents': [1], 'description': 'Sentinel - Binary Discovery'},
                {'batch': 2, 'agents': [2], 'description': 'The Architect - Architecture Analysis'},
                {'batch': 3, 'agents': [3, 4], 'description': 'Parallel Analysis Phase'},
                {'batch': 4, 'agents': [5], 'description': 'Neo - Advanced Decompilation'},
                {'batch': 5, 'agents': [6, 7, 8], 'description': 'Processing Phase'},
                {'batch': 6, 'agents': [9], 'description': 'Commander Locke - Global Reconstruction'},
                {'batch': 7, 'agents': [10], 'description': 'The Machine - Compilation'},
                {'batch': 8, 'agents': [11], 'description': 'The Oracle - Final Validation'}
            ],
            'agent_matrix_mapping': {
                1: {'name': 'Sentinel', 'type': 'machine'},
                2: {'name': 'The Architect', 'type': 'program'},
                3: {'name': 'The Merovingian', 'type': 'program'},
                4: {'name': 'Agent Smith', 'type': 'program'},
                5: {'name': 'Neo (Glitch)', 'type': 'fragment'},
                6: {'name': 'The Twins', 'type': 'program'},
                7: {'name': 'The Trainman', 'type': 'program'},
                8: {'name': 'The Keymaker', 'type': 'program'},
                9: {'name': 'Commander Locke', 'type': 'ai_commander'},
                10: {'name': 'The Machine', 'type': 'collective'},
                11: {'name': 'The Oracle', 'type': 'program'}
            },
            'total_batches': 8,
            'estimated_duration': self._estimate_execution_duration(),
            'parallelization_factor': self.config.get_value('pipeline.batch_size', 4)
        }
        
        return agent_plan
    
    def _estimate_execution_duration(self) -> Dict[str, float]:
        """Estimate execution duration for the pipeline"""
        # Base estimates per agent type (in seconds)
        base_estimates = {
            1: 30,   # Binary discovery
            2: 60,   # Architecture analysis
            3: 120,  # Pattern matching
            4: 90,   # Structure analysis
            5: 300,  # Advanced decompilation
            6: 150,  # Binary comparison
            7: 200,  # Assembly analysis
            8: 180,  # Resource reconstruction
            9: 240,  # Global reconstruction
            10: 600, # Compilation
            11: 120  # Validation
        }
        
        # Adjust based on binary size and complexity
        binary_info = self.global_context.get('binary_info')
        complexity_factor = 1.0
        
        if binary_info:
            # Size factor
            size_mb = binary_info.file_size / (1024 * 1024)
            if size_mb > 10:
                complexity_factor *= 1.5
            if size_mb > 50:
                complexity_factor *= 2.0
            
            # Packing factor
            if binary_info.is_packed:
                complexity_factor *= 1.8
            
            # Architecture factor
            if binary_info.architecture.value == 'x64':
                complexity_factor *= 1.2
        
        # Apply complexity factor
        adjusted_estimates = {k: v * complexity_factor for k, v in base_estimates.items()}
        
        return {
            'per_agent': adjusted_estimates,
            'total_sequential': sum(adjusted_estimates.values()),
            'total_parallel': max(adjusted_estimates.values()) * 4,  # Rough parallel estimate
            'complexity_factor': complexity_factor
        }
    
    def _validate_ai_engine(self) -> Dict[str, Any]:
        """Validate AI engine connectivity and functionality"""
        self.logger.info("Validating AI engine...")
        
        try:
            # Test AI engine response
            test_response = self.generate_ai_response(
                prompt="Respond with 'MATRIX_AI_TEST_SUCCESS' to confirm functionality.",
                system_message="You are testing the AI engine connectivity for the Matrix pipeline."
            )
            
            ai_status = {
                'available': test_response.success,
                'provider': self.ai_engine.get_engine_status() if hasattr(self.ai_engine, 'get_engine_status') else {},
                'test_successful': 'MATRIX_AI_TEST_SUCCESS' in test_response.content if test_response.success else False,
                'response_time': getattr(test_response, 'response_time', 0.0),
                'error': test_response.error_message if not test_response.success else None
            }
            
            if ai_status['available'] and ai_status['test_successful']:
                self.logger.info("✅ AI engine validation successful")
            else:
                self.validation_warnings.append("AI engine validation failed - pipeline will use fallback mode")
                self.logger.warning("⚠️ AI engine validation failed")
            
            return ai_status
            
        except Exception as e:
            self.logger.error(f"AI engine validation error: {e}")
            return {
                'available': False,
                'error': str(e),
                'test_successful': False
            }
    
    def _binary_info_to_dict(self, binary_info: BinaryInfo) -> Dict[str, Any]:
        """Convert BinaryInfo to dictionary"""
        return {
            'file_path': binary_info.file_path,
            'file_size': binary_info.file_size,
            'format': binary_info.format.value,
            'architecture': binary_info.architecture.value,
            'is_64bit': binary_info.is_64bit,
            'is_packed': binary_info.is_packed,
            'entry_point': binary_info.entry_point,
            'base_address': binary_info.base_address,
            'section_count': len(binary_info.sections),
            'import_count': len(binary_info.imports),
            'export_count': len(binary_info.exports),
            'string_count': len(binary_info.strings),
            'compiler_info': binary_info.compiler_info,
            'hashes': binary_info.hashes
        }
    
    def _create_validation_failure_result(self) -> AgentResult:
        """Create result for validation failure"""
        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            status=AgentStatus.FAILED,
            error_message=f"System validation failed: {'; '.join(self.validation_errors)}",
            data={
                'validation_errors': self.validation_errors,
                'validation_warnings': self.validation_warnings,
                'system_status': self.system_status
            },
            metadata={'matrix_character': self.matrix_character}
        )
    
    def _create_binary_failure_result(self) -> AgentResult:
        """Create result for binary analysis failure"""
        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            status=AgentStatus.FAILED,
            error_message="Binary analysis failed",
            data={
                'validation_errors': self.validation_errors,
                'system_status': self.system_status
            },
            metadata={'matrix_character': self.matrix_character}
        )


# Register the Deus Ex Machina agent
from ..matrix_agent_base import register_matrix_agent
register_matrix_agent(0, DeusExMachinaAgent)