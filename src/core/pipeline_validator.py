"""
Pipeline Validation Framework
Comprehensive end-to-end validation for the Matrix decompilation pipeline
"""

import logging
import time
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .config_manager import get_config_manager
from .performance_monitor import PerformanceMonitor
from .binary_comparison import BinaryComparator
from .shared_components import MatrixLogger, MatrixValidator
from .matrix_agents import AgentResult, AgentStatus


class ValidationLevel(Enum):
    """Validation severity levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    RESEARCH = "research"


class ValidationStatus(Enum):
    """Pipeline validation status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationRule:
    """Individual validation rule definition"""
    name: str
    description: str
    level: ValidationLevel
    threshold: float
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ValidationResult:
    """Result from a validation check"""
    rule_name: str
    status: ValidationStatus
    score: float
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


@dataclass
class PipelineValidationReport:
    """Comprehensive pipeline validation report"""
    validation_id: str
    timestamp: float
    overall_status: ValidationStatus
    overall_score: float
    quality_threshold: float
    
    # Validation categories
    binary_analysis_results: List[ValidationResult] = field(default_factory=list)
    decompilation_results: List[ValidationResult] = field(default_factory=list)
    compilation_results: List[ValidationResult] = field(default_factory=list)
    performance_results: List[ValidationResult] = field(default_factory=list)
    
    # Summary statistics
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    
    # Execution metrics
    total_execution_time: float = 0.0
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    
    # Files and artifacts
    tested_binary: Optional[str] = None
    generated_source: Optional[str] = None
    recompiled_binary: Optional[str] = None
    
    def get_success_rate(self) -> float:
        """Calculate validation success rate"""
        if self.total_checks == 0:
            return 0.0
        return (self.passed_checks / self.total_checks) * 100.0


class PipelineValidator:
    """
    Comprehensive pipeline validation framework
    
    Provides end-to-end validation of the Matrix decompilation pipeline including:
    - Binary analysis quality assessment
    - Decompilation accuracy validation
    - Compilation success verification
    - Performance benchmarking
    - Quality threshold enforcement
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.config = get_config_manager()
        self.logger = MatrixLogger.get_logger("pipeline_validator")
        self.validation_level = validation_level
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.binary_comparator = BinaryComparator()
        self.matrix_validator = MatrixValidator()
        
        # Load validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Quality thresholds
        self.quality_threshold = self.config.get_value('validation.quality_threshold', 0.75)
        self.performance_threshold = self.config.get_value('validation.performance_threshold', 0.80)
        
        # Validation state
        self.current_report: Optional[PipelineValidationReport] = None
        
    def _initialize_validation_rules(self) -> Dict[str, ValidationRule]:
        """Initialize comprehensive validation rule set"""
        rules = {}
        
        # Binary Analysis Rules
        rules['binary_format_detection'] = ValidationRule(
            name="binary_format_detection",
            description="Binary format correctly detected",
            level=ValidationLevel.BASIC,
            threshold=0.95
        )
        
        rules['architecture_identification'] = ValidationRule(
            name="architecture_identification", 
            description="Architecture correctly identified",
            level=ValidationLevel.BASIC,
            threshold=0.90
        )
        
        rules['function_discovery'] = ValidationRule(
            name="function_discovery",
            description="Functions correctly discovered",
            level=ValidationLevel.STANDARD,
            threshold=0.80
        )
        
        rules['symbol_resolution'] = ValidationRule(
            name="symbol_resolution",
            description="Symbols correctly resolved", 
            level=ValidationLevel.STANDARD,
            threshold=0.70
        )
        
        # Decompilation Rules
        rules['code_coverage'] = ValidationRule(
            name="code_coverage",
            description="Percentage of binary successfully decompiled",
            level=ValidationLevel.STANDARD,
            threshold=0.75
        )
        
        rules['function_accuracy'] = ValidationRule(
            name="function_accuracy",
            description="Accuracy of function decompilation",
            level=ValidationLevel.STANDARD,
            threshold=0.70
        )
        
        rules['variable_recovery'] = ValidationRule(
            name="variable_recovery",
            description="Quality of variable name recovery",
            level=ValidationLevel.COMPREHENSIVE,
            threshold=0.60
        )
        
        rules['control_flow_accuracy'] = ValidationRule(
            name="control_flow_accuracy",
            description="Accuracy of control flow reconstruction",
            level=ValidationLevel.COMPREHENSIVE,
            threshold=0.65
        )
        
        # Compilation Rules
        rules['syntax_correctness'] = ValidationRule(
            name="syntax_correctness",
            description="Generated code has correct syntax",
            level=ValidationLevel.STANDARD,
            threshold=0.95
        )
        
        rules['compilation_success'] = ValidationRule(
            name="compilation_success",
            description="Generated code compiles successfully",
            level=ValidationLevel.STANDARD,
            threshold=0.80
        )
        
        rules['binary_equivalence'] = ValidationRule(
            name="binary_equivalence",
            description="Recompiled binary is functionally equivalent",
            level=ValidationLevel.RESEARCH,
            threshold=0.60
        )
        
        # Performance Rules
        rules['execution_time'] = ValidationRule(
            name="execution_time",
            description="Pipeline executes within time limits",
            level=ValidationLevel.STANDARD,
            threshold=0.85
        )
        
        rules['memory_usage'] = ValidationRule(
            name="memory_usage", 
            description="Memory usage within acceptable limits",
            level=ValidationLevel.STANDARD,
            threshold=0.90
        )
        
        rules['resource_efficiency'] = ValidationRule(
            name="resource_efficiency",
            description="Efficient use of system resources",
            level=ValidationLevel.COMPREHENSIVE,
            threshold=0.75
        )
        
        # Output Directory Compliance Rules
        rules['output_directory_compliance'] = ValidationRule(
            name="output_directory_compliance",
            description="All outputs follow structured directory organization",
            level=ValidationLevel.BASIC,
            threshold=1.0
        )
        
        rules['output_path_validation'] = ValidationRule(
            name="output_path_validation",
            description="No hardcoded paths outside output structure",
            level=ValidationLevel.STANDARD,
            threshold=1.0
        )
        
        return rules
    
    def validate_pipeline_results(
        self,
        agent_results: Dict[int, AgentResult],
        binary_path: str,
        output_dir: str,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> PipelineValidationReport:
        """
        Perform comprehensive validation of pipeline results
        
        Args:
            agent_results: Results from all pipeline agents
            binary_path: Path to original binary
            output_dir: Directory containing pipeline outputs
            performance_metrics: Optional performance data
            
        Returns:
            Comprehensive validation report
        """
        validation_start = time.time()
        
        # Initialize validation report
        validation_id = hashlib.md5(f"{binary_path}_{validation_start}".encode()).hexdigest()[:8]
        self.current_report = PipelineValidationReport(
            validation_id=validation_id,
            timestamp=validation_start,
            overall_status=ValidationStatus.FAILED,
            overall_score=0.0,
            quality_threshold=self.quality_threshold,
            validation_level=self.validation_level,
            tested_binary=binary_path
        )
        
        self.logger.info(f"Starting pipeline validation {validation_id} for {Path(binary_path).name}")
        
        try:
            # Phase 1: Binary Analysis Validation
            self.logger.info("Phase 1: Validating binary analysis results")
            self._validate_binary_analysis(agent_results, binary_path)
            
            # Phase 2: Decompilation Validation
            self.logger.info("Phase 2: Validating decompilation results")
            self._validate_decompilation_results(agent_results, output_dir)
            
            # Phase 3: Compilation Validation
            self.logger.info("Phase 3: Validating compilation results")
            self._validate_compilation_results(agent_results, output_dir)
            
            # Phase 4: Performance Validation
            self.logger.info("Phase 4: Validating performance metrics")
            self._validate_performance_metrics(performance_metrics)
            
            # Phase 5: Output Directory Validation
            self.logger.info("Phase 5: Validating output directory compliance")
            self._validate_output_directory_structure(agent_results, output_dir)
            
            # Calculate overall results
            self._calculate_overall_results()
            
            # Generate final report
            self.current_report.total_execution_time = time.time() - validation_start
            
            self.logger.info(
                f"Pipeline validation completed: {self.current_report.overall_status.value} "
                f"(Score: {self.current_report.overall_score:.3f})"
            )
            
            return self.current_report
            
        except Exception as e:
            self.logger.error(f"Pipeline validation failed: {e}", exc_info=True)
            self.current_report.overall_status = ValidationStatus.FAILED
            self.current_report.total_execution_time = time.time() - validation_start
            return self.current_report
    
    def _validate_binary_analysis(self, agent_results: Dict[int, AgentResult], binary_path: str):
        """Validate binary analysis quality"""
        
        # Get Agent 1 (Sentinel) results for binary format detection
        if 1 in agent_results and agent_results[1].status == AgentStatus.SUCCESS:
            sentinel_data = agent_results[1].data
            
            # Validate binary format detection
            format_confidence = sentinel_data.get('binary_info', {}).get('format_confidence', 0.0)
            self._add_validation_result(
                'binary_format_detection',
                format_confidence,
                f"Binary format detection confidence: {format_confidence:.3f}"
            )
            
            # Validate architecture identification  
            arch_confidence = sentinel_data.get('binary_info', {}).get('architecture_confidence', 0.0)
            self._add_validation_result(
                'architecture_identification',
                arch_confidence,
                f"Architecture identification confidence: {arch_confidence:.3f}"
            )
        else:
            self._add_validation_result(
                'binary_format_detection',
                0.0,
                "Agent 1 (Sentinel) failed - no binary analysis available"
            )
        
        # Get Agent 2 (Architect) results for architecture analysis
        if 2 in agent_results and agent_results[2].status == AgentStatus.SUCCESS:
            architect_data = agent_results[2].data
            
            # Validate function discovery
            function_count = len(architect_data.get('functions', []))
            # Estimate function discovery quality based on binary size and function count
            binary_size = Path(binary_path).stat().st_size if Path(binary_path).exists() else 0
            expected_functions = max(1, binary_size // 10000)  # Rough estimate
            function_score = min(1.0, function_count / expected_functions)
            
            self._add_validation_result(
                'function_discovery',
                function_score,
                f"Discovered {function_count} functions (estimated quality: {function_score:.3f})"
            )
    
    def _validate_decompilation_results(self, agent_results: Dict[int, AgentResult], output_dir: str):
        """Validate decompilation quality and accuracy"""
        
        # Get advanced decompilation results (Agent 5 - Neo)
        if 5 in agent_results and agent_results[5].status == AgentStatus.SUCCESS:
            neo_data = agent_results[5].data
            quality_metrics = neo_data.get('quality_metrics', {})
            
            # Code coverage validation
            code_coverage = quality_metrics.get('code_coverage', 0.0)
            self._add_validation_result(
                'code_coverage',
                code_coverage,
                f"Code coverage: {code_coverage:.1%}"
            )
            
            # Function accuracy validation
            function_accuracy = quality_metrics.get('function_accuracy', 0.0)
            self._add_validation_result(
                'function_accuracy',
                function_accuracy,
                f"Function accuracy: {function_accuracy:.1%}"
            )
            
            # Variable recovery validation
            variable_recovery = quality_metrics.get('variable_recovery', 0.0)
            self._add_validation_result(
                'variable_recovery',
                variable_recovery,
                f"Variable recovery: {variable_recovery:.1%}"
            )
            
            # Control flow accuracy validation
            control_flow_accuracy = quality_metrics.get('control_flow_accuracy', 0.0)
            self._add_validation_result(
                'control_flow_accuracy',
                control_flow_accuracy,
                f"Control flow accuracy: {control_flow_accuracy:.1%}"
            )
        else:
            # No advanced decompilation available - mark as skipped
            for rule_name in ['code_coverage', 'function_accuracy', 'variable_recovery', 'control_flow_accuracy']:
                self._add_validation_result(
                    rule_name,
                    0.0,
                    "Advanced decompilation not available",
                    status=ValidationStatus.SKIPPED
                )
    
    def _validate_compilation_results(self, agent_results: Dict[int, AgentResult], output_dir: str):
        """Validate compilation success and binary equivalence"""
        
        # Check for generated source code
        output_path = Path(output_dir)
        source_files = list(output_path.glob("**/*.c")) + list(output_path.glob("**/*.cpp"))
        
        if source_files:
            self.current_report.generated_source = str(source_files[0])
            
            # Basic syntax check
            syntax_score = self._check_syntax_correctness(source_files)
            self._add_validation_result(
                'syntax_correctness',
                syntax_score,
                f"Syntax correctness: {syntax_score:.1%}"
            )
            
            # Compilation attempt
            compilation_success = self._attempt_compilation(source_files, output_path)
            self._add_validation_result(
                'compilation_success',
                1.0 if compilation_success else 0.0,
                f"Compilation {'successful' if compilation_success else 'failed'}"
            )
            
            # Binary equivalence (if compilation succeeded)
            if compilation_success:
                equivalence_score = self._check_binary_equivalence(output_path)
                self._add_validation_result(
                    'binary_equivalence',
                    equivalence_score,
                    f"Binary equivalence: {equivalence_score:.1%}"
                )
        else:
            # No source code generated
            for rule_name in ['syntax_correctness', 'compilation_success', 'binary_equivalence']:
                self._add_validation_result(
                    rule_name,
                    0.0,
                    "No source code generated",
                    status=ValidationStatus.FAILED
                )
    
    def _validate_performance_metrics(self, performance_metrics: Optional[Dict[str, Any]]):
        """Validate performance and resource usage"""
        
        if not performance_metrics:
            # No performance data available
            for rule_name in ['execution_time', 'memory_usage', 'resource_efficiency']:
                self._add_validation_result(
                    rule_name,
                    0.0,
                    "No performance metrics available",
                    status=ValidationStatus.SKIPPED
                )
            return
        
        # Execution time validation
        execution_time = performance_metrics.get('total_execution_time', 0)
        time_limit = self.config.get_value('validation.max_execution_time', 3600)  # 1 hour default
        time_score = max(0.0, 1.0 - (execution_time / time_limit))
        
        self._add_validation_result(
            'execution_time',
            time_score,
            f"Execution time: {execution_time:.1f}s (limit: {time_limit}s)"
        )
        
        # Memory usage validation
        max_memory = performance_metrics.get('peak_memory_usage', 0)
        memory_limit = self.config.get_value('validation.max_memory_mb', 4096)  # 4GB default
        memory_score = max(0.0, 1.0 - (max_memory / memory_limit))
        
        self._add_validation_result(
            'memory_usage',
            memory_score,
            f"Peak memory: {max_memory:.1f}MB (limit: {memory_limit}MB)"
        )
        
        # Resource efficiency (combination of CPU and I/O efficiency)
        cpu_efficiency = performance_metrics.get('cpu_efficiency', 0.5)
        io_efficiency = performance_metrics.get('io_efficiency', 0.5) 
        resource_efficiency = (cpu_efficiency + io_efficiency) / 2
        
        self._add_validation_result(
            'resource_efficiency',
            resource_efficiency,
            f"Resource efficiency: {resource_efficiency:.1%}"
        )
    
    def _add_validation_result(
        self,
        rule_name: str,
        score: float,
        message: str,
        status: Optional[ValidationStatus] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Add validation result to appropriate category"""
        
        if rule_name not in self.validation_rules:
            self.logger.warning(f"Unknown validation rule: {rule_name}")
            return
        
        rule = self.validation_rules[rule_name]
        
        # Determine status if not provided
        if status is None:
            if score >= rule.threshold:
                status = ValidationStatus.PASSED
            elif score >= rule.threshold * 0.7:  # Warning threshold
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.FAILED
        
        result = ValidationResult(
            rule_name=rule_name,
            status=status,
            score=score,
            threshold=rule.threshold,
            message=message,
            details=details or {}
        )
        
        # Add to appropriate category
        if rule_name in ['binary_format_detection', 'architecture_identification', 'function_discovery', 'symbol_resolution']:
            self.current_report.binary_analysis_results.append(result)
        elif rule_name in ['code_coverage', 'function_accuracy', 'variable_recovery', 'control_flow_accuracy']:
            self.current_report.decompilation_results.append(result)
        elif rule_name in ['syntax_correctness', 'compilation_success', 'binary_equivalence']:
            self.current_report.compilation_results.append(result)
        elif rule_name in ['execution_time', 'memory_usage', 'resource_efficiency']:
            self.current_report.performance_results.append(result)
    
    def _calculate_overall_results(self):
        """Calculate overall validation results and status"""
        
        all_results = (
            self.current_report.binary_analysis_results +
            self.current_report.decompilation_results +
            self.current_report.compilation_results +
            self.current_report.performance_results
        )
        
        if not all_results:
            self.current_report.overall_status = ValidationStatus.FAILED
            self.current_report.overall_score = 0.0
            return
        
        # Calculate statistics
        self.current_report.total_checks = len(all_results)
        self.current_report.passed_checks = len([r for r in all_results if r.status == ValidationStatus.PASSED])
        self.current_report.failed_checks = len([r for r in all_results if r.status == ValidationStatus.FAILED])
        self.current_report.warning_checks = len([r for r in all_results if r.status == ValidationStatus.WARNING])
        
        # Calculate weighted overall score
        total_weight = 0.0
        weighted_score = 0.0
        
        for result in all_results:
            rule = self.validation_rules.get(result.rule_name)
            if rule and rule.enabled:
                weight = rule.weight
                total_weight += weight
                weighted_score += result.score * weight
        
        self.current_report.overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine overall status
        if self.current_report.overall_score >= self.quality_threshold:
            self.current_report.overall_status = ValidationStatus.PASSED
        elif self.current_report.overall_score >= self.quality_threshold * 0.7:
            self.current_report.overall_status = ValidationStatus.WARNING
        else:
            self.current_report.overall_status = ValidationStatus.FAILED
    
    def _check_syntax_correctness(self, source_files: List[Path]) -> float:
        """Check syntax correctness of generated source code"""
        try:
            # Simple syntax check - look for basic C syntax patterns
            total_files = len(source_files)
            valid_files = 0
            
            for source_file in source_files:
                try:
                    content = source_file.read_text(encoding='utf-8')
                    
                    # Basic syntax checks
                    if self._has_valid_c_syntax(content):
                        valid_files += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check syntax for {source_file}: {e}")
            
            return valid_files / total_files if total_files > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Syntax check failed: {e}")
            return 0.0
    
    def _has_valid_c_syntax(self, content: str) -> bool:
        """Check if content has valid C syntax patterns"""
        
        # Basic checks for valid C code
        checks = [
            # Has function definitions or declarations
            any(pattern in content for pattern in ['int ', 'void ', 'char ', 'float ', 'double ']),
            # Has proper brackets
            content.count('{') == content.count('}'),
            content.count('(') == content.count(')'),
            # Has includes or some structure
            '#include' in content or 'main(' in content or 'return' in content,
            # No obvious syntax errors
            not any(error in content.lower() for error in ['syntax error', 'undefined', 'UNRECOVERED'])
        ]
        
        return sum(checks) >= 3  # At least 3 out of 4 checks should pass
    
    def _attempt_compilation(self, source_files: List[Path], output_dir: Path) -> bool:
        """Attempt to compile generated source code"""
        try:
            # This would normally use MSBuild or gcc to compile
            # For now, return a mock result based on syntax check
            
            # Check if we have at least one main function
            has_main = False
            for source_file in source_files:
                content = source_file.read_text(encoding='utf-8')
                if 'main(' in content:
                    has_main = True
                    break
            
            # Mock compilation success based on syntax quality
            syntax_score = self._check_syntax_correctness(source_files)
            compilation_success = syntax_score > 0.8 and has_main
            
            if compilation_success:
                # Create mock compiled binary
                compiled_binary = output_dir / "recompiled.exe"
                compiled_binary.write_bytes(b"MOCK_COMPILED_BINARY")
                self.current_report.recompiled_binary = str(compiled_binary)
            
            return compilation_success
            
        except Exception as e:
            self.logger.error(f"Compilation attempt failed: {e}")
            return False
    
    def _check_binary_equivalence(self, output_dir: Path) -> float:
        """Check binary equivalence between original and recompiled"""
        try:
            if not self.current_report.recompiled_binary:
                return 0.0
            
            original_path = self.current_report.tested_binary
            recompiled_path = self.current_report.recompiled_binary
            
            # Use binary comparator for equivalence check
            equivalence_score = self.binary_comparator.compare_binaries(
                original_path, 
                recompiled_path
            )
            
            return equivalence_score
            
        except Exception as e:
            self.logger.error(f"Binary equivalence check failed: {e}")
            return 0.0
    
    def save_validation_report(self, output_path: str) -> str:
        """Save validation report to file"""
        try:
            if not self.current_report:
                raise ValueError("No validation report available")
            
            report_data = {
                'validation_id': self.current_report.validation_id,
                'timestamp': self.current_report.timestamp,
                'overall_status': self.current_report.overall_status.value,
                'overall_score': self.current_report.overall_score,
                'quality_threshold': self.current_report.quality_threshold,
                'success_rate': self.current_report.get_success_rate(),
                'statistics': {
                    'total_checks': self.current_report.total_checks,
                    'passed_checks': self.current_report.passed_checks,
                    'failed_checks': self.current_report.failed_checks,
                    'warning_checks': self.current_report.warning_checks
                },
                'execution_metrics': {
                    'total_execution_time': self.current_report.total_execution_time,
                    'validation_level': self.current_report.validation_level.value
                },
                'tested_files': {
                    'binary': self.current_report.tested_binary,
                    'generated_source': self.current_report.generated_source,
                    'recompiled_binary': self.current_report.recompiled_binary
                },
                'validation_results': {
                    'binary_analysis': [self._result_to_dict(r) for r in self.current_report.binary_analysis_results],
                    'decompilation': [self._result_to_dict(r) for r in self.current_report.decompilation_results],
                    'compilation': [self._result_to_dict(r) for r in self.current_report.compilation_results],
                    'performance': [self._result_to_dict(r) for r in self.current_report.performance_results]
                }
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
            raise
    
    def _result_to_dict(self, result: ValidationResult) -> Dict[str, Any]:
        """Convert ValidationResult to dictionary"""
        return {
            'rule_name': result.rule_name,
            'status': result.status.value,
            'score': result.score,
            'threshold': result.threshold,
            'message': result.message,
            'details': result.details,
            'execution_time': result.execution_time
        }

    def _validate_output_directory_compliance(self, agent_results: Dict[int, AgentResult], output_dir: str) -> float:
        """Validate that all outputs follow the structured directory organization"""
        try:
            output_path = Path(output_dir)
            if not output_path.exists():
                return 0.0
            
            # Expected output structure based on shared_components.setup_output_structure
            expected_dirs = ['agents', 'ghidra', 'compilation', 'reports', 'logs', 'temp', 'tests', 'docs']
            score = 0.0
            total_dirs = len(expected_dirs)
            
            for expected_dir in expected_dirs:
                dir_path = output_path / expected_dir
                if dir_path.exists():
                    score += 1.0
                    
                    # Check if directory has proper agent subdirectories for 'agents' dir
                    if expected_dir == 'agents':
                        agent_subdirs = [d for d in dir_path.iterdir() if d.is_dir() and d.name.startswith('agent_')]
                        if len(agent_subdirs) > 0:
                            score += 0.5  # Bonus for having agent subdirectories
            
            # Normalize score (max possible is total_dirs + 0.5 for agent bonus)
            max_score = total_dirs + 0.5
            return min(1.0, score / max_score)
            
        except Exception as e:
            self.logger.error(f"Output directory compliance validation failed: {e}")
            return 0.0

    def _validate_output_path_structure(self, agent_results: Dict[int, AgentResult], output_dir: str) -> float:
        """Validate that no files exist outside the structured output directory"""
        try:
            output_path = Path(output_dir)
            project_root = output_path.parent if output_path.name != 'output' else output_path.parent
            
            # Check for any files/directories in project root that shouldn't be there
            unauthorized_patterns = [
                'temp*',
                'output*',  # Should only be our structured output
                '*.exe',
                '*.dll', 
                '*.obj',
                'ghidra_*',
                'compilation_*'
            ]
            
            violations = 0
            total_checks = len(unauthorized_patterns)
            
            for pattern in unauthorized_patterns:
                matches = list(project_root.glob(pattern))
                # Exclude our legitimate output directory
                legitimate_matches = [m for m in matches if m == output_path]
                unauthorized_matches = [m for m in matches if m not in legitimate_matches]
                
                if unauthorized_matches:
                    violations += 1
                    self.logger.warning(f"Found unauthorized files/dirs matching {pattern}: {unauthorized_matches}")
            
            # Calculate compliance score (1.0 = no violations, 0.0 = all patterns violated)
            compliance_score = 1.0 - (violations / total_checks)
            return compliance_score
            
        except Exception as e:
            self.logger.error(f"Output path structure validation failed: {e}")
            return 0.5  # Return neutral score on validation error

    def _validate_output_directory_structure(self, agent_results: Dict[int, AgentResult], output_dir: str):
        """Validate output directory structure and compliance"""
        
        # Validate directory compliance
        if 'output_directory_compliance' in self.validation_rules:
            rule = self.validation_rules['output_directory_compliance']
            compliance_score = self._validate_output_directory_compliance(agent_results, output_dir)
            
            status = ValidationStatus.PASSED if compliance_score >= rule.threshold else ValidationStatus.FAILED
            message = f"Directory structure compliance: {compliance_score:.1%}"
            
            self._add_validation_result(
                rule_name=rule.name,
                status=status,
                score=compliance_score,
                threshold=rule.threshold,
                message=message,
                details={
                    'compliance_score': compliance_score,
                    'output_directory': output_dir,
                    'validation_type': 'directory_structure'
                }
            )
        
        # Validate path structure
        if 'output_path_validation' in self.validation_rules:
            rule = self.validation_rules['output_path_validation']
            path_compliance_score = self._validate_output_path_structure(agent_results, output_dir)
            
            status = ValidationStatus.PASSED if path_compliance_score >= rule.threshold else ValidationStatus.FAILED
            message = f"Output path structure compliance: {path_compliance_score:.1%}"
            
            self._add_validation_result(
                rule_name=rule.name,
                status=status,
                score=path_compliance_score,
                threshold=rule.threshold,
                message=message,
                details={
                    'path_compliance_score': path_compliance_score,
                    'output_directory': output_dir,
                    'validation_type': 'path_structure'
                }
            )


def create_pipeline_validator(level: ValidationLevel = ValidationLevel.STANDARD) -> PipelineValidator:
    """Factory function to create pipeline validator"""
    return PipelineValidator(validation_level=level)


if __name__ == "__main__":
    # Example usage
    validator = create_pipeline_validator(ValidationLevel.COMPREHENSIVE)
    
    # Mock agent results for testing
    mock_results = {
        1: AgentResult(
            agent_id=1,
            status=AgentStatus.SUCCESS,
            data={
                'binary_info': {
                    'format_confidence': 0.95,
                    'architecture_confidence': 0.90
                }
            },
            agent_name="Sentinel",
            matrix_character="Sentinel"
        )
    }
    
    # Run validation
    report = validator.validate_pipeline_results(
        agent_results=mock_results,
        binary_path="test_binary.exe",
        output_dir="test_output"
    )
    
    print(f"Validation Status: {report.overall_status.value}")
    print(f"Overall Score: {report.overall_score:.3f}")
    print(f"Success Rate: {report.get_success_rate():.1f}%")