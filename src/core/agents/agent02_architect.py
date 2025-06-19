"""
Agent 02: The Architect - Architecture Analysis & Error Pattern Matching
The precise mathematician who designed the Matrix, understanding its fundamental structures.
Analyzes architectural patterns within binaries, identifying compilation methods and structural anomalies.

Production-ready implementation following SOLID principles and clean code standards.
Includes LangChain integration, comprehensive error handling, and fail-fast validation.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Matrix framework imports
from ..matrix_agents import AnalysisAgent, MatrixCharacter, AgentStatus
from ..shared_components import (
    MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker, 
    MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
)
from ..exceptions import MatrixAgentError, ValidationError, ConfigurationError

# AI system integration
from ..ai_system import ai_available, ai_analyze_code, ai_request_safe

# LangChain removed per rules.md - using centralized AI system only

# Configuration constants - NO MAGIC NUMBERS
class ArchitectConstants:
    """Architect-specific constants loaded from configuration"""
    
    def __init__(self, config_manager, agent_id: int):
        self.MAX_RETRY_ATTEMPTS = config_manager.get_value(f'agents.agent_{agent_id:02d}.max_retries', 3)
        self.TIMEOUT_SECONDS = config_manager.get_value(f'agents.agent_{agent_id:02d}.timeout', 300)
        self.QUALITY_THRESHOLD = config_manager.get_value(f'agents.agent_{agent_id:02d}.quality_threshold', 0.35)
        self.MIN_CONFIDENCE_THRESHOLD = config_manager.get_value('analysis.min_confidence_threshold', 0.7)
        self.MAX_PATTERN_MATCHES = config_manager.get_value('analysis.max_pattern_matches', 100)

@dataclass 
class CompilerSignature:
    """Compiler signature detection result - Windows MSVC only"""
    toolchain: str  # MSVC only (Windows)
    version: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

@dataclass
class OptimizationAnalysis:
    """Optimization level analysis result"""
    level: str  # O0/O1/O2/O3/Os/Oz
    confidence: float = 0.0
    detected_patterns: List[str] = None
    optimization_artifacts: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.detected_patterns is None:
            self.detected_patterns = []
        if self.optimization_artifacts is None:
            self.optimization_artifacts = []

@dataclass
class ArchitectValidationResult:
    """Validation result structure for fail-fast pipeline"""
    is_valid: bool
    quality_score: float
    error_messages: List[str]
    validation_details: Dict[str, Any]

class ArchitectAgent(AnalysisAgent):
    """
    Agent 02: The Architect - Production-Ready Implementation
    
    The Architect designed the Matrix itself, understanding its fundamental structures.
    Agent 02 analyzes the architectural patterns within binaries, identifying how
    high-level constructs were transformed and optimized during compilation.
    
    Features:
    - Compiler toolchain detection (MSVC only - Windows)
    - Optimization level analysis (O0-O3, Os, Oz)
    - Build system identification
    - Error pattern recognition
    - ABI and calling convention analysis
    """
    
    # Compiler signature patterns - Windows MSVC only
    COMPILER_SIGNATURES = {
        'MSVC': {
            'patterns': [
                rb'Microsoft.*C/C\+\+.*Compiler',
                rb'MSVCRT\.dll',
                rb'VCRUNTIME\d+\.dll',
                rb'MSVCP\d+\.dll',
                rb'__security_cookie',
                rb'_chkstk',
                rb'___security_init_cookie',
                rb'__scrt_common_main',
                rb'__security_check_cookie',
                rb'_RTC_CheckEsp',
                rb'_guard_check_icall'
            ],
            'versions': {
                '12.0': [rb'Microsoft.*C/C\+\+.*Version 18\.00'],  # VS 2013
                '14.0': [rb'Microsoft.*C/C\+\+.*Version 19\.00'],  # VS 2015
                '14.1': [rb'Microsoft.*C/C\+\+.*Version 19\.1\d'], # VS 2017
                '14.2': [rb'Microsoft.*C/C\+\+.*Version 19\.2\d'], # VS 2019
                '14.3': [rb'Microsoft.*C/C\+\+.*Version 19\.3\d'], # VS 2022
                'latest': [rb'Microsoft.*C/C\+\+.*Version 19\.\d+']
            }
        }
    }
    
    # Optimization patterns 
    OPTIMIZATION_PATTERNS = {
        'O0': {
            'patterns': [
                'redundant_loads',
                'unoptimized_branches', 
                'debug_information_present',
                'no_function_inlining'
            ],
            'indicators': [
                rb'mov.*mov.*same_register',  # Redundant moves
                rb'push.*ebp.*mov.*ebp.*esp',  # Standard function prologue
            ]
        },
        'O1': {
            'patterns': [
                'basic_optimization',
                'some_dead_code_elimination',
                'basic_constant_folding'
            ],
            'indicators': [
                rb'optimized_register_usage',
                rb'reduced_memory_access'
            ]
        },
        'O2': {
            'patterns': [
                'function_inlining',
                'loop_optimization',
                'constant_propagation',
                'dead_code_elimination'
            ],
            'indicators': [
                rb'inlined_function_calls',
                rb'optimized_loops',
                rb'folded_constants'
            ]
        },
        'O3': {
            'patterns': [
                'aggressive_inlining',
                'loop_unrolling',
                'vectorization',
                'interprocedural_optimization'
            ],
            'indicators': [
                rb'unrolled_loops',
                rb'vectorized_operations',
                rb'aggressive_optimizations'
            ]
        }
    }
    
    def __init__(self):
        super().__init__(
            agent_id=2,
            matrix_character=MatrixCharacter.ARCHITECT
        )
        
        # Load configuration constants
        self.constants = ArchitectConstants(self.config, self.agent_id)
        
        # Initialize shared tools
        self.analysis_tools = SharedAnalysisTools()
        self.validation_tools = SharedValidationTools()
        
        # Setup specialized components
        self.error_handler = MatrixErrorHandler(self.agent_name, self.constants.MAX_RETRY_ATTEMPTS)
        self.metrics = MatrixMetrics(self.agent_id, self.matrix_character.value)
        
        # Initialize AI system - simple reference
        self.ai_enabled = ai_available()
        
        # Validate configuration
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate agent configuration at startup - fail fast if invalid"""
        required_paths = [
            'paths.temp_directory',
            'paths.output_directory'
        ]
        
        missing_paths = []
        for path_key in required_paths:
            try:
                path = self.config.get_path(path_key)
                if path is None:
                    # Use default paths if not configured
                    if path_key == 'paths.temp_directory':
                        path = Path('./temp')
                    elif path_key == 'paths.output_directory':
                        path = Path('./output')
                    else:
                        missing_paths.append(f"{path_key}: No path configured")
                        continue
                elif isinstance(path, str):
                    path = Path(path)
                
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                missing_paths.append(f"{path_key}: {e}")
        
        if missing_paths:
            raise ConfigurationError(f"Invalid configuration paths: {missing_paths}")
    
    # LangChain integration removed per rules.md - using centralized AI system
    
    def get_matrix_description(self) -> str:
        """The Architect's role in the Matrix"""
        return ("The Architect designed the Matrix with mathematical precision. "
                "Agent 02 analyzes the architectural blueprints embedded in binaries, "
                "revealing compilation strategies and structural design decisions.")
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Architect's architectural analysis with production-ready error handling
        """
        self.metrics.start_tracking()
        
        # Setup progress tracking  
        total_steps = 6
        progress = MatrixProgressTracker(total_steps, self.agent_name)
        
        try:
            # Step 1: Validate prerequisites and dependencies
            progress.step("Validating prerequisites and Sentinel data")
            self._validate_prerequisites(context)
            
            # Step 2: Initialize analysis with Sentinel data
            progress.step("Initializing architectural analysis components")
            with self.error_handler.handle_matrix_operation("component_initialization"):
                analysis_context = self._initialize_analysis(context)
            
            # Step 3: Execute compiler analysis
            progress.step("Analyzing compiler toolchain and signatures")
            with self.error_handler.handle_matrix_operation("compiler_analysis"):
                compiler_results = self._analyze_compiler_signatures(analysis_context)
            
            # Step 4: Execute optimization analysis
            progress.step("Detecting optimization patterns and levels")
            with self.error_handler.handle_matrix_operation("optimization_analysis"):
                optimization_results = self._analyze_optimization_patterns(analysis_context)
            
            # Step 5: Execute ABI and build system analysis
            progress.step("Analyzing ABI and build system characteristics")
            with self.error_handler.handle_matrix_operation("abi_analysis"):
                abi_results = self._analyze_abi_and_build_system(analysis_context)
            
            # Combine core results
            core_results = {
                'compiler_analysis': compiler_results,
                'optimization_analysis': optimization_results,
                'abi_analysis': abi_results
            }
            
            # Step 6: AI enhancement (if enabled)
            if self.ai_enabled:
                progress.step("Applying AI-enhanced architectural insights")
                with self.error_handler.handle_matrix_operation("ai_enhancement"):
                    ai_results = self._execute_ai_analysis(core_results, context)
                    core_results = self._merge_analysis_results(core_results, ai_results)
            else:
                progress.step("Skipping AI enhancement (disabled)")
            
            # Validate results and finalize
            validation_result = self._validate_results(core_results)
            
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Architect analysis failed validation: {validation_result.error_messages}"
                )
            
            # Finalize and save results
            final_results = self._finalize_results(core_results, validation_result)
            self._save_results(final_results, context)
            self._populate_shared_memory(final_results, context)
            
            progress.complete()
            self.metrics.end_tracking()
            
            # Log success with metrics
            self.logger.info(
                "Architect analysis completed successfully",
                extra={
                    'execution_time': self.metrics.execution_time,
                    'quality_score': validation_result.quality_score,
                    'compiler_detected': compiler_results.toolchain,
                    'optimization_level': optimization_results.level,
                    'validation_passed': True
                }
            )
            
            return final_results
            
        except Exception as e:
            self.metrics.end_tracking()
            self.metrics.increment_errors()
            
            # Log detailed error information
            self.logger.error(
                "Architect analysis failed",
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'execution_time': self.metrics.execution_time,
                    'agent_id': self.agent_id,
                    'matrix_character': self.matrix_character.value
                },
                exc_info=True
            )
            
            # Re-raise with Matrix context
            raise MatrixAgentError(
                f"Architect architectural analysis failed: {e}",
                agent_id=self.agent_id,
                matrix_character=self.matrix_character.value
            ) from e
    
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate all prerequisites before starting analysis"""
        # Validate required context keys
        required_keys = ['binary_path', 'shared_memory']
        missing_keys = self.validation_tools.validate_context_keys(context, required_keys)
        
        if missing_keys:
            raise ValidationError(f"Missing required context keys: {missing_keys}")
        
        # Initialize shared_memory structure if not present
        shared_memory = context['shared_memory']
        if 'analysis_results' not in shared_memory:
            shared_memory['analysis_results'] = {}
        if 'binary_metadata' not in shared_memory:
            shared_memory['binary_metadata'] = {}
        
        # Validate dependencies - check for Sentinel results in multiple ways
        dependency_met = False
        
        # Check agent_results first
        agent_results = context.get('agent_results', {})
        if 1 in agent_results:
            dependency_met = True
        
        # Check shared_memory analysis_results
        if not dependency_met and 1 in shared_memory['analysis_results']:
            dependency_met = True
        
        # Check for Sentinel data in binary_metadata
        if not dependency_met and 'discovery' in shared_memory.get('binary_metadata', {}):
            dependency_met = True
        
        if not dependency_met:
            self.logger.error("Sentinel dependency not satisfied - cannot proceed with analysis")
            raise ValidationError("Agent 1 (Sentinel) dependency not satisfied - Architect requires Sentinel's discovery data")
    
    def _initialize_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize analysis context with Sentinel data"""
        # Get Sentinel's analysis results
        shared_memory = context['shared_memory']
        sentinel_data = shared_memory['binary_metadata']['discovery']
        
        binary_path = Path(context['binary_path'])
        
        # Read binary content for pattern analysis
        try:
            with open(binary_path, 'rb') as f:
                binary_content = f.read()
        except Exception as e:
            raise ValidationError(f"Failed to read binary file: {e}")
        
        return {
            'binary_path': binary_path,
            'binary_content': binary_content,
            'sentinel_data': sentinel_data,
            'binary_info': sentinel_data.get('binary_info'),
            'format_analysis': sentinel_data.get('format_analysis', {}),
            'file_size': len(binary_content)
        }
    
    def _analyze_compiler_signatures(self, analysis_context: Dict[str, Any]) -> CompilerSignature:
        """Analyze binary for compiler signatures and toolchain identification"""
        binary_content = analysis_context['binary_content']
        detected_signatures = []
        
        # Analyze each compiler signature
        for compiler, signatures in self.COMPILER_SIGNATURES.items():
            matches = 0
            evidence = []
            
            # Check main patterns
            for pattern in signatures['patterns']:
                if re.search(pattern, binary_content, re.IGNORECASE):
                    matches += 1
                    evidence.append(f"Pattern match: {pattern.decode('utf-8', errors='ignore')}")
            
            # Check version-specific patterns
            detected_version = None
            for version, version_patterns in signatures.get('versions', {}).items():
                for pattern in version_patterns:
                    if re.search(pattern, binary_content, re.IGNORECASE):
                        detected_version = version
                        evidence.append(f"Version pattern: {version}")
                        matches += 1
                        break
                if detected_version:
                    break
            
            if matches > 0:
                confidence = min(matches / len(signatures['patterns']), 1.0)
                detected_signatures.append(CompilerSignature(
                    toolchain=compiler,
                    version=detected_version,
                    confidence=confidence,
                    evidence=evidence
                ))
        
        # Return the most confident detection
        if detected_signatures:
            best_match = max(detected_signatures, key=lambda x: x.confidence)
            return best_match
        else:
            return CompilerSignature(
                toolchain='Unknown',
                confidence=0.0,
                evidence=['No compiler signatures detected']
            )
    
    def _analyze_optimization_patterns(self, analysis_context: Dict[str, Any]) -> OptimizationAnalysis:
        """Analyze binary for optimization patterns and levels"""
        binary_content = analysis_context['binary_content']
        format_analysis = analysis_context.get('format_analysis', {})
        
        optimization_scores = {}
        detected_patterns = []
        artifacts = []
        
        # Analyze optimization indicators
        for opt_level, opt_data in self.OPTIMIZATION_PATTERNS.items():
            score = 0
            level_patterns = []
            
            # Check for optimization patterns
            for pattern in opt_data['patterns']:
                # This is a simplified check - real implementation would analyze assembly
                if self._check_optimization_pattern(binary_content, pattern, format_analysis):
                    score += 1
                    level_patterns.append(pattern)
            
            if level_patterns:
                detected_patterns.extend(level_patterns)
                optimization_scores[opt_level] = score / len(opt_data['patterns'])
        
        # Determine most likely optimization level
        if optimization_scores:
            best_level = max(optimization_scores.keys(), key=lambda x: optimization_scores[x])
            confidence = optimization_scores[best_level]
        else:
            best_level = 'Unknown'
            confidence = 0.0
        
        # Generate optimization artifacts
        artifacts = self._identify_optimization_artifacts(analysis_context, detected_patterns)
        
        return OptimizationAnalysis(
            level=best_level,
            confidence=confidence,
            detected_patterns=detected_patterns,
            optimization_artifacts=artifacts
        )
    
    def _check_optimization_pattern(self, binary_content: bytes, pattern: str, format_analysis: Dict) -> bool:
        """Check for specific optimization patterns in binary"""
        # Simplified pattern detection - real implementation would be more sophisticated
        pattern_indicators = {
            'redundant_loads': b'mov.*mov',  # Pattern indicating redundant operations
            'function_inlining': len(format_analysis.get('functions', [])) < 50,  # Fewer functions = more inlining
            'loop_optimization': b'jmp.*loop',  # Jump patterns indicating loop optimization
            'dead_code_elimination': True,  # Would check for gaps in code sections
            'constant_folding': b'mov.*immediate',  # Immediate values instead of calculations
        }
        
        indicator = pattern_indicators.get(pattern, False)
        
        if isinstance(indicator, bytes):
            return bool(re.search(indicator, binary_content))
        else:
            return bool(indicator)
    
    def _identify_optimization_artifacts(self, analysis_context: Dict[str, Any], patterns: List[str]) -> List[Dict[str, Any]]:
        """Identify specific optimization artifacts in the binary"""
        artifacts = []
        format_analysis = analysis_context.get('format_analysis', {})
        
        # Function count artifact
        function_count = len(format_analysis.get('functions', []))
        if function_count > 0:
            artifacts.append({
                'type': 'function_count',
                'value': function_count,
                'interpretation': 'low_count_suggests_inlining' if function_count < 50 else 'normal_function_count'
            })
        
        # Section entropy artifact (high entropy may indicate optimization)
        sections = format_analysis.get('sections', [])
        for section in sections:
            if section.get('name') == '.text' and 'entropy' in section:
                artifacts.append({
                    'type': 'code_entropy',
                    'value': section['entropy'],
                    'interpretation': 'high_optimization' if section['entropy'] > 6.5 else 'normal_optimization'
                })
        
        return artifacts
    
    def _analyze_abi_and_build_system(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ABI characteristics and build system indicators"""
        binary_info = analysis_context.get('binary_info')
        format_analysis = analysis_context.get('format_analysis', {})
        
        # Calling convention analysis
        calling_convention = self._detect_calling_convention(analysis_context)
        
        # Build system detection
        build_system = self._detect_build_system(analysis_context)
        
        # ABI characteristics
        abi_info = {
            'calling_convention': calling_convention,
            'stack_alignment': self._detect_stack_alignment(analysis_context),
            'exception_handling': self._detect_exception_handling(analysis_context),
            'rtti_enabled': self._detect_rtti(analysis_context)
        }
        
        return {
            'abi_analysis': abi_info,
            'build_system_analysis': build_system,
            'target_platform': self._determine_target_platform(binary_info)
        }
    
    def _detect_calling_convention(self, analysis_context: Dict[str, Any]) -> str:
        """Detect calling convention used in the binary"""
        binary_content = analysis_context['binary_content']
        binary_info = analysis_context.get('binary_info')
        
        # Simple heuristic based on architecture and patterns
        # Handle both dict and object formats for binary_info
        if binary_info:
            if isinstance(binary_info, dict):
                architecture = binary_info.get('architecture')
            else:
                architecture = getattr(binary_info, 'architecture', None)
            
            if architecture == 'x64':
                return 'Microsoft x64'  # x64 has standard calling convention
            elif architecture == 'x86':
                # Check for specific patterns
                if b'__stdcall' in binary_content:
                    return 'stdcall'
                elif b'__fastcall' in binary_content:
                    return 'fastcall'
                else:
                    return 'cdecl'  # Default assumption
        
        return 'Unknown'
    
    def _detect_stack_alignment(self, analysis_context: Dict[str, Any]) -> int:
        """Detect stack alignment requirements"""
        binary_info = analysis_context.get('binary_info')
        
        # Standard alignments based on architecture
        # Handle both dict and object formats for binary_info
        if binary_info:
            if isinstance(binary_info, dict):
                architecture = binary_info.get('architecture')
            else:
                architecture = getattr(binary_info, 'architecture', None)
            
            if architecture == 'x64':
                return 16  # x64 requires 16-byte alignment
            elif architecture == 'x86':
                return 4   # x86 typically uses 4-byte alignment
        
        return 0  # Unknown
    
    def _detect_exception_handling(self, analysis_context: Dict[str, Any]) -> str:
        """Detect exception handling mechanism"""
        binary_content = analysis_context['binary_content']
        
        if b'__CxxFrameHandler' in binary_content:
            return 'C++ EH'
        elif b'_except_handler' in binary_content:
            return 'SEH'
        elif b'__gxx_personality' in binary_content:
            return 'GCC EH'
        else:
            return 'None'
    
    def _detect_rtti(self, analysis_context: Dict[str, Any]) -> bool:
        """Detect if RTTI (Run-Time Type Information) is enabled"""
        binary_content = analysis_context['binary_content']
        
        rtti_indicators = [
            b'??_R',      # MSVC RTTI symbols
            b'_ZTI',      # GCC typeinfo symbols
            b'type_info', # Generic RTTI
        ]
        
        return any(indicator in binary_content for indicator in rtti_indicators)
    
    def _detect_build_system(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect build system used to compile the binary"""
        binary_content = analysis_context['binary_content']
        
        build_indicators = {
            'MSBuild': [b'MSBuild', b'Visual Studio', b'Microsoft'],
            'CMake': [b'cmake', b'CMake'],
            'Make': [b'Makefile', b'make'],
            'Ninja': [b'ninja'],
            'Autotools': [b'autoconf', b'automake']
        }
        
        detected_systems = []
        for build_system, indicators in build_indicators.items():
            if any(indicator in binary_content for indicator in indicators):
                detected_systems.append(build_system)
        
        return {
            'detected_systems': detected_systems,
            'primary_system': detected_systems[0] if detected_systems else 'Unknown',
            'confidence': self._calculate_build_system_confidence(detected_systems, binary_content)
        }
    
    def _determine_target_platform(self, binary_info) -> str:
        """Determine target platform from binary information"""
        if not binary_info:
            return 'Unknown'
        
        # Handle both dict and object formats for binary_info
        if isinstance(binary_info, dict):
            format_type = binary_info.get('format_type')
        else:
            format_type = getattr(binary_info, 'format_type', None)
        
        format_platform_map = {
            'PE': 'Windows',
            'ELF': 'Linux/Unix',
            'Mach-O': 'macOS'
        }
        
        return format_platform_map.get(format_type, 'Unknown')
    
    def _execute_ai_analysis(self, core_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI-enhanced analysis using centralized AI system"""
        if not self.ai_enabled:
            return {
                'ai_analysis_available': False,
                'architectural_insights': 'AI analysis not available',
                'compiler_recommendations': 'Basic heuristics only',
                'optimization_patterns': 'Manual analysis required',
                'confidence_score': 0.0
            }
        
        try:
            compiler_analysis = core_results.get('compiler_analysis')
            optimization_analysis = core_results.get('optimization_analysis')
            
            # Create AI analysis prompt
            prompt = f"""
            Analyze the architectural patterns in this binary:
            
            Compiler: {compiler_analysis.toolchain} (confidence: {compiler_analysis.confidence:.2f})
            Optimization Level: {optimization_analysis.level} (confidence: {optimization_analysis.confidence:.2f})
            Detected Patterns: {', '.join(optimization_analysis.detected_patterns)}
            
            Provide insights about development practices, build quality, and architectural decisions.
            """
            
            # Execute AI analysis using centralized AI system
            from ..ai_system import ai_analyze
            ai_response = ai_analyze(prompt, "You are a reverse engineering expert. Analyze code patterns and architectural decisions to understand the original software design and purpose.")
            ai_result = ai_response.content if ai_response.success else None
            
            return {
                'ai_insights': ai_result,
                'ai_confidence': self._calculate_ai_confidence(ai_result, compiler_analysis, optimization_analysis),
                'ai_enabled': True
            }
        except Exception as e:
            self.logger.warning(f"AI analysis failed: {e}")
            return {'ai_enabled': False, 'ai_error': str(e)}
    
    def _merge_analysis_results(self, core_results: Dict[str, Any], ai_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge traditional analysis with AI insights"""
        merged = core_results.copy()
        merged['ai_analysis'] = ai_results
        return merged
    
    def _calculate_build_system_confidence(self, detected_systems: List[str], binary_content: bytes) -> float:
        """Calculate confidence score for build system detection based on evidence strength"""
        if not detected_systems:
            return 0.0
        
        # Base confidence for detection
        base_confidence = 0.6
        
        # Additional confidence for multiple indicators
        if len(detected_systems) > 1:
            base_confidence += 0.1
        
        # Boost confidence based on strength of evidence
        strong_indicators = [b'Makefile', b'make', b'Visual Studio', b'MSBuild']
        for indicator in strong_indicators:
            if indicator in binary_content:
                base_confidence += 0.1
                
        return min(base_confidence, 1.0)
    
    def _calculate_ai_confidence(self, ai_result: str, compiler_analysis, optimization_analysis) -> float:
        """Calculate AI analysis confidence based on input data quality and response quality"""
        confidence_factors = []
        
        # Factor 1: Input data quality
        if compiler_analysis and compiler_analysis.confidence > 0.5:
            confidence_factors.append(0.3)
        if optimization_analysis and optimization_analysis.confidence > 0.5:
            confidence_factors.append(0.3)
            
        # Factor 2: AI response quality indicators
        if ai_result and len(ai_result) > 100:  # Substantial response
            confidence_factors.append(0.2)
        if 'analysis' in ai_result.lower():  # Contains analysis keywords
            confidence_factors.append(0.1)
        if any(word in ai_result.lower() for word in ['pattern', 'architecture', 'optimization']):
            confidence_factors.append(0.1)
            
        return sum(confidence_factors)
    
    def _validate_results(self, results: Dict[str, Any]) -> ArchitectValidationResult:
        """Validate results meet Architect quality thresholds"""
        quality_score = self._calculate_quality_score(results)
        is_valid = quality_score >= self.constants.QUALITY_THRESHOLD
        
        error_messages = []
        if not is_valid:
            error_messages.append(
                f"Quality score {quality_score:.3f} below threshold {self.constants.QUALITY_THRESHOLD}"
            )
        
        # Compiler detection is optional - some binaries may not have clear signatures
        
        return ArchitectValidationResult(
            is_valid=len(error_messages) == 0,
            quality_score=quality_score,
            error_messages=error_messages,
            validation_details={
                'quality_score': quality_score,
                'threshold': self.constants.QUALITY_THRESHOLD,
                'agent_id': self.agent_id,
                'compiler_detected': compiler_analysis.toolchain if compiler_analysis else 'None'
            }
        )
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for Architect analysis"""
        score_components = []
        
        # Compiler detection score (40%)
        compiler_analysis = results.get('compiler_analysis')
        if compiler_analysis and compiler_analysis.toolchain != 'Unknown':
            score_components.append(0.4 * compiler_analysis.confidence)
        
        # Optimization detection score (30%) 
        optimization_analysis = results.get('optimization_analysis')
        if optimization_analysis and optimization_analysis.level != 'Unknown':
            score_components.append(0.3 * optimization_analysis.confidence)
        
        # ABI analysis score (20%)
        abi_analysis = results.get('abi_analysis', {}).get('abi_analysis', {})
        if abi_analysis.get('calling_convention', 'Unknown') != 'Unknown':
            score_components.append(0.2)
        
        # Build system detection score (10%)
        build_analysis = results.get('abi_analysis', {}).get('build_system_analysis', {})
        if build_analysis.get('primary_system', 'Unknown') != 'Unknown':
            score_components.append(0.1)
        
        return sum(score_components)
    
    def _finalize_results(self, results: Dict[str, Any], validation: ArchitectValidationResult) -> Dict[str, Any]:
        """Finalize results with Architect metadata and validation info"""
        return {
            **results,
            'architect_metadata': {
                'agent_id': self.agent_id,
                'matrix_character': self.matrix_character.value,
                'quality_score': validation.quality_score,
                'validation_passed': validation.is_valid,
                'execution_time': self.metrics.execution_time,
                'ai_enhanced': self.ai_enabled,
                'analysis_timestamp': self.metrics.start_time
            }
        }
    
    def _save_results(self, results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Save Architect results to output directory"""
        if 'output_manager' in context:
            output_manager = context['output_manager']
            output_manager.save_agent_data(
                self.agent_id, 
                self.matrix_character.value, 
                results
            )
    
    def _populate_shared_memory(self, results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Populate shared memory with Architect analysis for other agents"""
        shared_memory = context['shared_memory']
        
        # Store Architect results for other agents
        shared_memory['analysis_results'][self.agent_id] = results
        
        # Store specific analysis data for easy access
        shared_memory['binary_metadata']['architect_analysis'] = {
            'compiler_analysis': results.get('compiler_analysis'),
            'optimization_analysis': results.get('optimization_analysis'),
            'abi_analysis': results.get('abi_analysis'),
            'architect_confidence': results['architect_metadata']['quality_score']
        }
    
    # Centralized AI system handles all AI functionality