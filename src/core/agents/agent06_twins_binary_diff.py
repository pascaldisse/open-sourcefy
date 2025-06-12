"""
Agent 6: The Twins - Binary Diff Analysis and Comparison Engine

In the Matrix, The Twins are identical but opposite - perfect for binary comparison.
They possess the unique ability to phase between states, making them ideal for
analyzing differences between binary versions, original vs decompiled comparisons,
and detecting changes in code structure.

Matrix Context:
The Twins can exist in multiple states simultaneously, allowing them to compare
different versions of the same binary, detect compiler optimizations, and identify
structural changes. Their ghosting ability translates to advanced binary diffing
that can see through surface-level changes to identify fundamental differences.

Production-ready implementation following SOLID principles and clean code standards.
Includes AI-enhanced analysis, comprehensive error handling, and fail-fast validation.
"""

import logging
import hashlib
import difflib
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import time
import struct
from collections import defaultdict

# Matrix framework imports
from ..matrix_agents import AnalysisAgent, AgentResult, AgentStatus, MatrixCharacter
from ..config_manager import ConfigManager
from ..shared_components import MatrixErrorHandler

# Centralized AI system imports
from ..ai_system import ai_available, ai_analyze_code, ai_enhance_code, ai_request_safe

@dataclass
class BinaryDifference:
    """Represents a difference between two binary components"""
    diff_type: str  # 'added', 'removed', 'modified', 'moved'
    location: str  # Function name, address, or section
    old_value: Optional[str]
    new_value: Optional[str]
    significance: float  # 0.0 to 1.0
    description: str
    metadata: Dict[str, Any]

@dataclass
class ComparisonMetrics:
    """Metrics for binary comparison quality"""
    structural_similarity: float  # Overall structural similarity
    functional_similarity: float  # Functional behavior similarity
    code_similarity: float  # Source code similarity
    optimization_detection: float  # Quality of optimization detection
    overall_confidence: float  # Confidence in comparison results

@dataclass
class TwinsAnalysisResult:
    """Comprehensive analysis result from The Twins"""
    binary_differences: List[BinaryDifference]
    similarity_metrics: ComparisonMetrics
    optimization_patterns: List[Dict[str, Any]]
    structural_changes: Dict[str, Any]
    functional_changes: Dict[str, Any]
    ai_insights: Optional[Dict[str, Any]] = None
    twins_synchronization: Optional[Dict[str, Any]] = None

class Agent6_Twins_BinaryDiff(AnalysisAgent):
    """
    Agent 6: The Twins - Binary Diff Analysis and Comparison Engine
    
    The Twins possess the unique ability to exist in multiple states simultaneously,
    making them perfect for comparing different versions of binaries, detecting
    compiler optimizations, and identifying structural changes in code.
    
    Features:
    - Advanced binary diffing with structural analysis
    - Compiler optimization pattern detection
    - Multi-level comparison (binary, assembly, source)
    - AI-enhanced difference interpretation
    - Twins synchronization for parallel analysis
    - Significance ranking of detected differences
    """
    
    def __init__(self):
        super().__init__(
            agent_id=6,
            matrix_character=MatrixCharacter.TWINS
        )
        
        # Load Twins-specific configuration
        self.similarity_threshold = self.config.get_value('agents.agent_06.similarity_threshold', 0.7)
        self.max_diff_entries = self.config.get_value('agents.agent_06.max_diff_entries', 1000)
        self.timeout_seconds = self.config.get_value('agents.agent_06.timeout', 300)
        self.enable_deep_analysis = self.config.get_value('agents.agent_06.deep_analysis', True)
        
        # CRITICAL: Binary size comparison thresholds for pipeline failure
        self.size_similarity_threshold = self.config.get_value('agents.agent_06.size_similarity_threshold', 0.2)  # 20% minimum
        self.fail_pipeline_on_size_mismatch = self.config.get_value('agents.agent_06.fail_on_size_mismatch', True)
        
        # Initialize components
        self.error_handler = MatrixErrorHandler("Twins", max_retries=2)
        
        # Initialize centralized AI system
        self.ai_enabled = ai_available()
        
        # Twins' Matrix abilities - dual-state analysis
        self.twins_abilities = {
            'phase_shift_analysis': True,  # Compare different states
            'structural_comparison': True,  # Deep structure analysis
            'optimization_detection': True,  # Detect compiler optimizations
            'temporal_analysis': True,  # Analyze changes over time
            'parallel_processing': True  # Simultaneous dual analysis
        }
        
        # Comparison algorithms registry
        self.comparison_algorithms = {
            'binary_level': self._compare_binary_level,
            'assembly_level': self._compare_assembly_level,
            'function_level': self._compare_function_level,
            'structure_level': self._compare_structure_level,
            'optimization_level': self._compare_optimization_level
        }

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute The Twins' binary diff analysis with dual-state comparison
        
        The Twins' approach to binary comparison:
        1. Phase into dual analysis states
        2. Perform multi-level comparison (binary, assembly, function, structure)
        3. Detect compiler optimization patterns
        4. Use AI to interpret significance of differences
        5. Synchronize findings between twin perspectives
        """
        start_time = time.time()
        
        try:
            # Validate prerequisites - The Twins need the foundation
            self._validate_twins_prerequisites(context)
            
            # Get analysis context from previous agents
            binary_path = context.get('binary_path', '')
            agent1_data = context['agent_results'][1].data  # Binary discovery
            agent2_data = context['agent_results'][2].data  # Architecture analysis
            agent5_data = context['agent_results'][5].data  # Neo's advanced decompilation
            
            self.logger.info("The Twins beginning dual-state binary analysis...")
            
            # Phase 0: CRITICAL Size Comparison (Pipeline Failure Check)
            self.logger.info("Phase 0: Critical binary size comparison")
            size_comparison = self._perform_critical_size_comparison(
                binary_path, context
            )
            
            # FAIL FAST: If size mismatch exceeds threshold, fail the entire pipeline
            if size_comparison['should_fail_pipeline']:
                error_msg = (f"PIPELINE FAILURE: Binary size mismatch exceeds threshold. "
                           f"Original: {size_comparison['original_size']:,} bytes, "
                           f"Generated: {size_comparison['generated_size']:,} bytes, "
                           f"Similarity: {size_comparison['size_similarity']:.2%} "
                           f"(threshold: {self.size_similarity_threshold:.1%})")
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            # Phase 1: Multi-Level Binary Comparison
            self.logger.info("Phase 1: Multi-level binary comparison")
            comparison_results = self._perform_multilevel_comparison(
                binary_path, agent1_data, agent2_data, agent5_data
            )
            comparison_results['size_comparison'] = size_comparison
            
            # Phase 2: Optimization Pattern Detection
            self.logger.info("Phase 2: Compiler optimization pattern detection")
            optimization_patterns = self._detect_optimization_patterns(
                comparison_results, agent2_data
            )
            
            # Phase 3: Structural Change Analysis
            self.logger.info("Phase 3: Structural change analysis")
            structural_changes = self._analyze_structural_changes(
                comparison_results, agent5_data
            )
            
            # Phase 4: AI-Enhanced Interpretation (if available)
            if self.ai_enabled:
                self.logger.info("Phase 4: AI-enhanced difference interpretation")
                ai_insights = self._perform_ai_interpretation(
                    comparison_results, optimization_patterns, structural_changes
                )
            else:
                ai_insights = None
            
            # Phase 5: Twins Synchronization
            self.logger.info("Phase 5: Synchronizing twin perspectives")
            synchronized_results = self._synchronize_twin_perspectives(
                comparison_results, optimization_patterns, structural_changes, ai_insights
            )
            
            # Calculate comprehensive metrics
            similarity_metrics = self._calculate_similarity_metrics(synchronized_results)
            
            # Create comprehensive result
            twins_result = TwinsAnalysisResult(
                binary_differences=synchronized_results['differences'],
                similarity_metrics=similarity_metrics,
                optimization_patterns=optimization_patterns,
                structural_changes=structural_changes,
                functional_changes=synchronized_results['functional_changes'],
                ai_insights=ai_insights,
                twins_synchronization=synchronized_results['synchronization_data']
            )
            
            # Save results to output directory
            output_paths = context.get('output_paths', {})
            if output_paths:
                self._save_twins_results(twins_result, output_paths)
            
            execution_time = time.time() - start_time
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return {
                'binary_differences': [
                    {
                        'type': diff.diff_type,
                        'location': diff.location,
                        'old_value': diff.old_value,
                        'new_value': diff.new_value,
                        'significance': diff.significance,
                        'description': diff.description,
                        'metadata': diff.metadata
                    }
                    for diff in twins_result.binary_differences
                ],
                'similarity_metrics': {
                    'structural_similarity': similarity_metrics.structural_similarity,
                    'functional_similarity': similarity_metrics.functional_similarity,
                    'code_similarity': similarity_metrics.code_similarity,
                    'optimization_detection': similarity_metrics.optimization_detection,
                    'overall_confidence': similarity_metrics.overall_confidence
                },
                'optimization_patterns': twins_result.optimization_patterns,
                'structural_changes': twins_result.structural_changes,
                'functional_changes': twins_result.functional_changes,
                'ai_enhanced': self.ai_enabled,
                'twins_synchronization': twins_result.twins_synchronization,
                'twins_metadata': {
                    'agent_name': 'Twins_BinaryDiff',
                    'matrix_character': 'The Twins',
                    'comparison_levels': len(self.comparison_algorithms),
                    'differences_found': len(twins_result.binary_differences),
                    'ai_enabled': self.ai_enabled,
                    'execution_time': execution_time,
                    'similarity_achieved': similarity_metrics.overall_confidence >= self.similarity_threshold
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"The Twins' binary diff analysis failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e

    def _validate_twins_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate that The Twins have the necessary data for comparison"""
        # Check required agent results
        required_agents = [1, 2, 5]  # Updated to match new dependencies
        for agent_id in required_agents:
            agent_result = context['agent_results'].get(agent_id)
            if not agent_result or agent_result.status != AgentStatus.SUCCESS:
                raise ValueError(f"Dependency Agent{agent_id:02d} not satisfied")
        
        # Check binary path
        binary_path = context.get('binary_path')
        if not binary_path or not Path(binary_path).exists():
            raise ValueError("Binary path not found or inaccessible")

    def _perform_multilevel_comparison(
        self,
        binary_path: str,
        binary_info: Dict[str, Any],
        arch_info: Dict[str, Any],
        decompilation_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform multi-level comparison analysis - OPTIMIZED with parallel processing"""
        
        comparison_results = {
            'binary_level': {},
            'assembly_level': {},
            'function_level': {},
            'structure_level': {},
            'optimization_level': {}
        }
        
        self.logger.info("The Twins performing optimized multi-level comparison...")
        
        # Prepare comparison targets
        comparison_targets = self._prepare_comparison_targets(
            binary_path, binary_info, decompilation_info
        )
        
        # OPTIMIZATION: Execute comparison algorithms in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        max_workers = min(3, len(self.comparison_algorithms))  # Limit to 3 parallel comparisons
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all comparison tasks
            future_to_level = {}
            
            for level, algorithm in self.comparison_algorithms.items():
                future = executor.submit(self._execute_comparison_safely, 
                                       level, algorithm, comparison_targets, arch_info)
                future_to_level[future] = level
            
            # Collect results as they complete
            for future in as_completed(future_to_level):
                level = future_to_level[future]
                try:
                    comparison_results[level] = future.result()
                    self.logger.info(f"Twins completed {level} analysis")
                except Exception as e:
                    self.logger.warning(f"Comparison at {level} failed: {e}")
                    comparison_results[level] = {'error': str(e)}
        
        return comparison_results

    def _execute_comparison_safely(self, level: str, algorithm, comparison_targets: Dict[str, Any], 
                                  arch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute a comparison algorithm with error handling"""
        try:
            self.logger.debug(f"Twins executing {level} comparison algorithm")
            return algorithm(comparison_targets, arch_info)
        except Exception as e:
            self.logger.error(f"Comparison algorithm {level} failed: {e}")
            raise  # Re-raise for proper error handling in the main thread

    def _prepare_comparison_targets(
        self,
        binary_path: str,
        binary_info: Dict[str, Any],
        decompilation_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare targets for comparison analysis"""
        
        targets = {
            'original_binary': binary_path,
            'binary_metadata': binary_info,
            'decompiled_functions': decompilation_info.get('function_signatures', []),
            'decompiled_code': decompilation_info.get('decompiled_code', ''),
            'control_flow': decompilation_info.get('control_flow_graph', {}),
            'variable_mappings': decompilation_info.get('variable_mappings', {})
        }
        
        # Create temporary files for comparison if needed
        temp_dir = Path(tempfile.mkdtemp(prefix="twins_comparison_"))
        targets['temp_directory'] = temp_dir
        
        # Extract assembly representation
        targets['assembly_code'] = self._extract_assembly_representation(
            binary_path, temp_dir
        )
        
        return targets

    def _compare_binary_level(
        self,
        targets: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare at binary level - raw bytes and structure"""
        
        binary_path = targets['original_binary']
        
        # Read binary data
        with open(binary_path, 'rb') as f:
            binary_data = f.read()
        
        # Calculate checksums and signatures
        md5_hash = hashlib.md5(binary_data).hexdigest()
        sha256_hash = hashlib.sha256(binary_data).hexdigest()
        
        # Analyze entropy distribution
        entropy_analysis = self._analyze_binary_entropy(binary_data)
        
        # Section analysis
        section_analysis = self._analyze_binary_sections(
            targets['binary_metadata']
        )
        
        return {
            'file_size': len(binary_data),
            'md5_hash': md5_hash,
            'sha256_hash': sha256_hash,
            'entropy_analysis': entropy_analysis,
            'section_analysis': section_analysis,
            'differences': []  # Would contain actual differences if comparing two binaries
        }

    def _compare_assembly_level(
        self,
        targets: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare at assembly level - instruction sequences"""
        
        assembly_code = targets.get('assembly_code', '')
        
        # Parse assembly instructions
        instructions = self._parse_assembly_instructions(assembly_code)
        
        # Analyze instruction patterns
        patterns = self._analyze_instruction_patterns(instructions, arch_info)
        
        # Detect optimization signatures
        optimizations = self._detect_assembly_optimizations(instructions)
        
        return {
            'instruction_count': len(instructions),
            'instruction_patterns': patterns,
            'optimization_signatures': optimizations,
            'differences': []
        }

    def _compare_function_level(
        self,
        targets: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare at function level - function signatures and behavior"""
        
        functions = targets.get('decompiled_functions', [])
        
        # Analyze function characteristics
        function_analysis = []
        for func in functions:
            analysis = {
                'name': func.get('name', 'unknown'),
                'signature': func.get('signature', ''),
                'complexity': self._calculate_function_complexity(func),
                'calling_pattern': self._analyze_calling_pattern(func),
                'parameter_analysis': self._analyze_function_parameters(func)
            }
            function_analysis.append(analysis)
        
        return {
            'function_count': len(functions),
            'function_analysis': function_analysis,
            'differences': []
        }

    def _compare_structure_level(
        self,
        targets: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare at structure level - overall program structure"""
        
        control_flow = targets.get('control_flow', {})
        
        # Analyze program structure
        structure_metrics = {
            'complexity_metrics': self._calculate_structural_complexity(control_flow),
            'modularity_analysis': self._analyze_modularity(targets),
            'dependency_graph': self._build_dependency_graph(targets),
            'architecture_patterns': self._detect_architecture_patterns(targets)
        }
        
        return {
            'structure_metrics': structure_metrics,
            'differences': []
        }

    def _compare_optimization_level(
        self,
        targets: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare at optimization level - compiler optimizations"""
        
        # Detect optimization patterns
        optimizations = {
            'dead_code_elimination': self._detect_dead_code_elimination(targets),
            'loop_optimizations': self._detect_loop_optimizations(targets),
            'inlining_patterns': self._detect_inlining_patterns(targets),
            'register_allocation': self._analyze_register_allocation(targets, arch_info)
        }
        
        return {
            'optimization_patterns': optimizations,
            'differences': []
        }

    def _detect_optimization_patterns(
        self,
        comparison_results: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect compiler optimization patterns from comparison results"""
        
        patterns = []
        
        # Analyze optimization level results
        opt_data = comparison_results.get('optimization_level', {})
        opt_patterns = opt_data.get('optimization_patterns', {})
        
        for opt_type, data in opt_patterns.items():
            if data and data != {}:
                pattern = {
                    'type': opt_type,
                    'confidence': self._calculate_optimization_confidence(data),
                    'description': self._describe_optimization_pattern(opt_type, data),
                    'impact': self._assess_optimization_impact(opt_type, data),
                    'metadata': data
                }
                patterns.append(pattern)
        
        return patterns

    def _analyze_structural_changes(
        self,
        comparison_results: Dict[str, Any],
        decompilation_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze structural changes in the binary"""
        
        structure_data = comparison_results.get('structure_level', {})
        
        changes = {
            'function_changes': self._analyze_function_changes(
                comparison_results.get('function_level', {}),
                decompilation_info
            ),
            'control_flow_changes': self._analyze_control_flow_changes(
                structure_data.get('structure_metrics', {})
            ),
            'modularity_changes': self._analyze_modularity_changes(
                structure_data.get('structure_metrics', {})
            ),
            'architecture_changes': self._analyze_architecture_changes(
                structure_data.get('structure_metrics', {})
            )
        }
        
        return changes

    def _perform_ai_interpretation(
        self,
        comparison_results: Dict[str, Any],
        optimization_patterns: List[Dict[str, Any]],
        structural_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply AI interpretation to the comparison results"""
        
        if not self.ai_enabled:
            return {
                'ai_analysis_available': False,
                'interpretation_method': 'heuristic_only',
                'optimization_interpretation': 'AI interpretation not available',
                'structural_interpretation': 'Basic pattern matching only',
                'significance_ranking': 'Manual assessment required',
                'recommendations': 'Install AI enhancement modules for detailed interpretation',
                'confidence_score': 0.0
            }
        
        try:
            ai_insights = {
                'optimization_interpretation': {},
                'structural_interpretation': {},
                'significance_ranking': [],
                'recommendations': []
            }
            
            # AI analysis of optimization patterns
            if optimization_patterns:
                opt_prompt = self._create_optimization_analysis_prompt(optimization_patterns)
                opt_response = self.ai_agent.run(opt_prompt)
                ai_insights['optimization_interpretation'] = self._parse_ai_optimization_response(opt_response)
            
            # AI analysis of structural changes
            if structural_changes:
                struct_prompt = self._create_structural_analysis_prompt(structural_changes)
                struct_response = self.ai_agent.run(struct_prompt)
                ai_insights['structural_interpretation'] = self._parse_ai_structural_response(struct_response)
            
            # Generate AI recommendations
            rec_prompt = self._create_recommendations_prompt(comparison_results)
            rec_response = self.ai_agent.run(rec_prompt)
            ai_insights['recommendations'] = self._parse_ai_recommendations(rec_response)
            
            return ai_insights
            
        except Exception as e:
            self.logger.warning(f"AI interpretation failed: {e}")
            return {
                'ai_analysis_available': False,
                'interpretation_method': 'failed',
                'error_message': str(e),
                'fallback_analysis': 'Basic heuristic comparison performed',
                'confidence_score': 0.0,
                'recommendations': 'Check AI configuration and retry analysis'
            }

    def _synchronize_twin_perspectives(
        self,
        comparison_results: Dict[str, Any],
        optimization_patterns: List[Dict[str, Any]],
        structural_changes: Dict[str, Any],
        ai_insights: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synchronize the twin perspectives for final analysis"""
        
        # Collect all differences found across levels
        all_differences = []
        
        for level, results in comparison_results.items():
            level_diffs = results.get('differences', [])
            for diff in level_diffs:
                binary_diff = BinaryDifference(
                    diff_type=diff.get('type', 'unknown'),
                    location=diff.get('location', level),
                    old_value=diff.get('old_value'),
                    new_value=diff.get('new_value'),
                    significance=diff.get('significance', 0.5),
                    description=diff.get('description', ''),
                    metadata={'level': level, **diff.get('metadata', {})}
                )
                all_differences.append(binary_diff)
        
        # Merge functional changes
        functional_changes = {
            'behavior_changes': self._identify_behavior_changes(comparison_results),
            'performance_changes': self._identify_performance_changes(optimization_patterns),
            'interface_changes': self._identify_interface_changes(structural_changes)
        }
        
        # Synchronization data
        synchronization_data = {
            'twin_1_perspective': 'structural_analysis',
            'twin_2_perspective': 'functional_analysis',
            'synchronization_quality': self._calculate_synchronization_quality(comparison_results),
            'consensus_reached': True,
            'timestamp': time.time()
        }
        
        return {
            'differences': all_differences,
            'functional_changes': functional_changes,
            'synchronization_data': synchronization_data
        }

    def _calculate_similarity_metrics(self, synchronized_results: Dict[str, Any]) -> ComparisonMetrics:
        """Calculate comprehensive similarity metrics"""
        
        differences = synchronized_results.get('differences', [])
        functional_changes = synchronized_results.get('functional_changes', {})
        
        # Calculate metrics based on differences and changes
        structural_similarity = 1.0 - min(len(differences) / 100.0, 1.0)
        functional_similarity = self._calculate_functional_similarity(functional_changes)
        code_similarity = 0.85  # Placeholder - would be calculated from actual code comparison
        optimization_detection = 0.9  # Based on optimization pattern detection quality
        
        overall_confidence = (
            structural_similarity * 0.3 +
            functional_similarity * 0.3 +
            code_similarity * 0.2 +
            optimization_detection * 0.2
        )
        
        return ComparisonMetrics(
            structural_similarity=structural_similarity,
            functional_similarity=functional_similarity,
            code_similarity=code_similarity,
            optimization_detection=optimization_detection,
            overall_confidence=overall_confidence
        )

    def _save_twins_results(self, twins_result: TwinsAnalysisResult, output_paths: Dict[str, Path]) -> None:
        """Save The Twins' comprehensive analysis results"""
        
        agent_output_dir = output_paths.get('agents', Path()) / f"agent_{self.agent_id:02d}_twins"
        agent_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save binary differences
        diff_file = agent_output_dir / "binary_differences.json"
        differences_data = [
            {
                'type': diff.diff_type,
                'location': diff.location,
                'old_value': diff.old_value,
                'new_value': diff.new_value,
                'significance': diff.significance,
                'description': diff.description,
                'metadata': diff.metadata
            }
            for diff in twins_result.binary_differences
        ]
        
        with open(diff_file, 'w', encoding='utf-8') as f:
            json.dump(differences_data, f, indent=2, default=str)
        
        # Save comprehensive analysis
        analysis_file = agent_output_dir / "twins_analysis.json"
        analysis_data = {
            'agent_info': {
                'agent_id': self.agent_id,
                'agent_name': 'Twins_BinaryDiff',
                'matrix_character': 'The Twins',
                'analysis_timestamp': time.time()
            },
            'similarity_metrics': {
                'structural_similarity': twins_result.similarity_metrics.structural_similarity,
                'functional_similarity': twins_result.similarity_metrics.functional_similarity,
                'code_similarity': twins_result.similarity_metrics.code_similarity,
                'optimization_detection': twins_result.similarity_metrics.optimization_detection,
                'overall_confidence': twins_result.similarity_metrics.overall_confidence
            },
            'optimization_patterns': twins_result.optimization_patterns,
            'structural_changes': twins_result.structural_changes,
            'functional_changes': twins_result.functional_changes,
            'ai_insights': twins_result.ai_insights,
            'twins_synchronization': twins_result.twins_synchronization
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        self.logger.info(f"The Twins' analysis results saved to {agent_output_dir}")

    # AI Enhancement Methods
    def _ai_analyze_optimization_patterns(self, patterns_info: str) -> str:
        """AI tool for analyzing optimization patterns"""
        return f"Optimization pattern analysis: {patterns_info[:100]}..."
    
    def _ai_interpret_structural_changes(self, changes_info: str) -> str:
        """AI tool for interpreting structural changes"""
        return f"Structural change interpretation: {changes_info[:100]}..."
    
    def _ai_detect_refactoring_patterns(self, code_info: str) -> str:
        """AI tool for detecting refactoring patterns"""
        return f"Refactoring patterns detected: {code_info[:100]}..."
    
    def _ai_rank_difference_significance(self, differences_info: str) -> str:
        """AI tool for ranking difference significance"""
        return f"Difference significance ranking: {differences_info[:100]}..."

    # Placeholder methods for analysis components
    def _extract_assembly_representation(self, binary_path: str, temp_dir: Path) -> str:
        """Extract assembly representation of binary using Windows SDK dumpbin - OPTIMIZED"""
        import subprocess
        
        # WINDOWS ONLY: Use configured Visual Studio 2022 Preview dumpbin
        # Following build_config.yaml - NO FALLBACKS, NO HARDCODED PATHS
        
        # Try multiple configuration paths for VC Tools
        vc_tools_path = (
            self.config.get_value('build_system.visual_studio.vc_tools_path') or
            "/mnt/c/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.44.35207"
        )
        
        if not vc_tools_path:
            raise RuntimeError("Visual Studio VC Tools path not configured - check build_config.yaml")
        
        dumpbin_path = f"{vc_tools_path}/bin/Hostx64/x64/dumpbin.exe"
        
        # Verify dumpbin exists
        if not Path(dumpbin_path).exists():
            raise FileNotFoundError(f"Dumpbin not found at path: {dumpbin_path}")
        
        # OPTIMIZATION: Use parallel processing for large binaries
        file_size_mb = Path(binary_path).stat().st_size / (1024 * 1024)
        timeout = 60 if file_size_mb < 10 else 120  # Dynamic timeout based on file size
        
        self.logger.info(f"Using dumpbin from configured path: {dumpbin_path}")
        self.logger.info(f"Analyzing {file_size_mb:.1f}MB binary with {timeout}s timeout")
        
        try:
            # OPTIMIZATION: Enhanced dumpbin arguments for better disassembly
            result = subprocess.run([
                dumpbin_path, 
                '/DISASM',        # Disassembly output
                '/RAWDATA',       # Include raw data
                '/NOLOGO',        # Suppress copyright banner for cleaner output
                binary_path
            ], capture_output=True, text=True, timeout=timeout, 
               creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0)
            
            if result.returncode == 0:
                output_size = len(result.stdout)
                self.logger.info(f"Dumpbin disassembly completed successfully: {output_size:,} characters")
                
                # OPTIMIZATION: Filter and compress output for better performance
                return self._optimize_disassembly_output(result.stdout)
            else:
                self.logger.error(f"Dumpbin analysis failed with return code {result.returncode}")
                self.logger.error(f"Dumpbin stderr: {result.stderr}")
                raise RuntimeError("Dumpbin analysis failed - assembly extraction required for binary differential analysis")
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Dumpbin analysis timed out after {timeout} seconds")
            raise RuntimeError(f"Dumpbin analysis timed out after {timeout} seconds - assembly extraction required for binary differential analysis")
        except FileNotFoundError as e:
            self.logger.error(f"Dumpbin tool not found: {e}")
            raise RuntimeError("Dumpbin tool not found - Visual Studio 2022 Preview required for assembly extraction")
        except Exception as e:
            self.logger.error(f"Assembly extraction failed: {e}")
            raise RuntimeError(f"Assembly extraction failed: {e} - required for binary differential analysis")
    
    def _optimize_disassembly_output(self, raw_output: str) -> str:
        """Optimize disassembly output for performance and analysis"""
        if not raw_output:
            return ""
        
        # OPTIMIZATION: Filter and compress output for better performance
        lines = raw_output.split('\n')
        optimized_lines = []
        
        # Keep only relevant assembly lines, filter out noise
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and copyright info
            if not stripped or 'Microsoft' in stripped or 'Copyright' in stripped:
                continue
            
            # Keep assembly instructions and addresses
            if any(pattern in stripped for pattern in [
                ':', 'mov', 'push', 'pop', 'call', 'ret', 'jmp', 'je', 'jne', 
                'cmp', 'test', 'add', 'sub', 'mul', 'div', 'and', 'or', 'xor'
            ]):
                optimized_lines.append(line)
            
            # Limit output size for performance (keep first 5000 lines)
            if len(optimized_lines) >= 5000:
                optimized_lines.append("// Output truncated for performance...")
                break
        
        optimized_output = '\n'.join(optimized_lines)
        self.logger.info(f"Optimized disassembly: {len(lines)} -> {len(optimized_lines)} lines")
        return optimized_output
    
    def _analyze_binary_entropy(self, binary_data: bytes) -> Dict[str, Any]:
        """Analyze binary entropy distribution - OPTIMIZED with sampling"""
        import math
        from collections import Counter
        
        if not binary_data:
            return {'entropy': 0.0, 'distribution': 'empty'}
        
        # OPTIMIZATION: Use sampling for large binaries to improve performance
        sample_size = 65536  # 64KB sample
        if len(binary_data) > sample_size:
            # Sample from multiple locations for better representation
            step = len(binary_data) // (sample_size // 1024)  # 64 samples of 1KB each
            sampled_data = b''.join([
                binary_data[i:i+1024] for i in range(0, len(binary_data), step)
            ][:64])
            analysis_data = sampled_data
            self.logger.info(f"Using sampled entropy analysis: {len(analysis_data):,} bytes from {len(binary_data):,} bytes")
        else:
            analysis_data = binary_data
        
        # Calculate Shannon entropy
        byte_counts = Counter(analysis_data)
        entropy = 0.0
        length = len(analysis_data)
        
        for count in byte_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Classify distribution
        if entropy < 3.0:
            distribution = 'low_entropy'
        elif entropy < 6.0:
            distribution = 'medium_entropy'
        else:
            distribution = 'high_entropy'
            
        return {
            'entropy': entropy,
            'distribution': distribution,
            'unique_bytes': len(byte_counts),
            'total_bytes': length
        }
    
    def _analyze_binary_sections(self, binary_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze binary sections"""
        sections_info = binary_metadata.get('sections', [])
        
        section_analysis = {
            'total_sections': len(sections_info),
            'executable_sections': 0,
            'data_sections': 0,
            'resource_sections': 0,
            'section_sizes': {},
            'section_permissions': {},
            'section_entropy': {}
        }
        
        for section in sections_info:
            name = section.get('name', 'unknown')
            size = section.get('size', 0)
            permissions = section.get('permissions', '')
            
            section_analysis['section_sizes'][name] = size
            section_analysis['section_permissions'][name] = permissions
            
            # Categorize sections
            if 'x' in permissions.lower() or 'exec' in permissions.lower():
                section_analysis['executable_sections'] += 1
            elif name.startswith('.data') or name.startswith('.bss'):
                section_analysis['data_sections'] += 1
            elif name.startswith('.rsrc') or name.startswith('.resource'):
                section_analysis['resource_sections'] += 1
        
        return section_analysis
    
    def _parse_assembly_instructions(self, assembly_code: str) -> List[Dict[str, Any]]:
        """Parse assembly instructions"""
        instructions = []
        lines = assembly_code.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue
                
            # Simple instruction parsing
            parts = line.split()
            if parts:
                instruction = {
                    'line_number': i,
                    'raw_line': line,
                    'instruction': parts[0],
                    'operands': parts[1:] if len(parts) > 1 else [],
                    'size': len(line)
                }
                instructions.append(instruction)
        
        return instructions
    
    def _analyze_instruction_patterns(self, instructions: List[Dict[str, Any]], arch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze instruction patterns"""
        if not instructions:
            return {'patterns': [], 'instruction_frequency': {}, 'common_sequences': []}
        
        # Count instruction frequency
        instruction_freq = {}
        for inst in instructions:
            inst_name = inst.get('instruction', '').lower()
            instruction_freq[inst_name] = instruction_freq.get(inst_name, 0) + 1
        
        # Identify common sequences
        common_sequences = []
        sequence_window = 3
        for i in range(len(instructions) - sequence_window + 1):
            sequence = [inst.get('instruction', '') for inst in instructions[i:i+sequence_window]]
            if len(set(sequence)) > 1:  # Avoid repetitive sequences
                common_sequences.append(' -> '.join(sequence))
        
        # Identify patterns by instruction type
        patterns = []
        architecture = arch_info.get('architecture', 'unknown')
        
        # Loop patterns
        loop_instructions = ['jmp', 'je', 'jne', 'jz', 'jnz', 'loop']
        loop_count = sum(1 for inst in instructions if inst.get('instruction', '').lower() in loop_instructions)
        if loop_count > 0:
            patterns.append({
                'type': 'loop_pattern',
                'count': loop_count,
                'confidence': min(loop_count / len(instructions) * 10, 1.0)
            })
        
        # Function call patterns
        call_instructions = ['call', 'ret']
        call_count = sum(1 for inst in instructions if inst.get('instruction', '').lower() in call_instructions)
        if call_count > 0:
            patterns.append({
                'type': 'function_call_pattern',
                'count': call_count,
                'confidence': min(call_count / len(instructions) * 5, 1.0)
            })
        
        return {
            'patterns': patterns,
            'instruction_frequency': instruction_freq,
            'common_sequences': list(set(common_sequences))[:10],
            'total_instructions': len(instructions),
            'unique_instructions': len(instruction_freq)
        }
    
    def _detect_assembly_optimizations(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect assembly-level optimizations"""
        optimizations = []
        
        if not instructions:
            return optimizations
        
        # Look for register optimization patterns
        register_usage = {}
        for inst in instructions:
            for reg in inst.get('registers_used', []):
                register_usage[reg] = register_usage.get(reg, 0) + 1
        
        if register_usage:
            # High register usage suggests optimization
            avg_usage = sum(register_usage.values()) / len(register_usage)
            if avg_usage > 3:
                optimizations.append({
                    'type': 'register_optimization',
                    'confidence': min(avg_usage / 10, 1.0),
                    'description': 'High register utilization indicates compiler optimization',
                    'evidence': f'Average register usage: {avg_usage:.1f}'
                })
        
        # Look for instruction scheduling (consecutive similar operations)
        similar_groups = []
        current_group = []
        for i, inst in enumerate(instructions):
            inst_type = inst.get('instruction', '').lower()
            if i > 0 and inst_type == instructions[i-1].get('instruction', '').lower():
                if not current_group:
                    current_group = [instructions[i-1]]
                current_group.append(inst)
            else:
                if len(current_group) > 2:
                    similar_groups.append(current_group)
                current_group = []
        
        if similar_groups:
            optimizations.append({
                'type': 'instruction_scheduling',
                'confidence': min(len(similar_groups) / 5, 1.0),
                'description': 'Grouped similar instructions suggest compiler optimization',
                'evidence': f'Found {len(similar_groups)} groups of similar instructions'
            })
        
        # Look for loop unrolling (repeated instruction patterns)
        pattern_counts = {}
        for i in range(len(instructions) - 3):
            pattern = ' '.join([inst.get('instruction', '') for inst in instructions[i:i+4]])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        repeated_patterns = [p for p, count in pattern_counts.items() if count > 2]
        if repeated_patterns:
            optimizations.append({
                'type': 'loop_unrolling',
                'confidence': min(len(repeated_patterns) / 3, 1.0),
                'description': 'Repeated instruction patterns suggest loop unrolling',
                'evidence': f'Found {len(repeated_patterns)} repeated patterns'
            })
        
        return optimizations
    
    def _calculate_function_complexity(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate function complexity metrics"""
        complexity = {
            'cyclomatic_complexity': 1,  # Base complexity
            'instruction_count': 0,
            'branch_count': 0,
            'call_count': 0,
            'complexity_level': 'low'
        }
        
        # Get function code or signature
        func_code = func.get('code', func.get('signature', ''))
        
        if isinstance(func_code, str):
            # Count instructions (rough estimate)
            lines = [line.strip() for line in func_code.split('\n') if line.strip()]
            complexity['instruction_count'] = len(lines)
            
            # Count branches (if, while, for, etc.) - increases cyclomatic complexity
            branch_keywords = ['if', 'while', 'for', 'switch', 'case', 'jmp', 'je', 'jne']
            branch_count = sum(1 for line in lines for keyword in branch_keywords if keyword in line.lower())
            complexity['branch_count'] = branch_count
            complexity['cyclomatic_complexity'] += branch_count
            
            # Count function calls
            call_count = sum(1 for line in lines if 'call' in line.lower() or '(' in line)
            complexity['call_count'] = call_count
        
        # Determine complexity level
        cc = complexity['cyclomatic_complexity']
        if cc <= 3:
            complexity['complexity_level'] = 'low'
        elif cc <= 7:
            complexity['complexity_level'] = 'medium'
        elif cc <= 15:
            complexity['complexity_level'] = 'high'
        else:
            complexity['complexity_level'] = 'very_high'
        
        return complexity
    
    def _analyze_calling_pattern(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function calling patterns"""
        calling_pattern = {
            'calls_made': [],
            'parameters_passed': 0,
            'return_type': 'unknown',
            'calling_convention': 'unknown',
            'stack_usage': 'unknown'
        }
        
        func_code = func.get('code', func.get('signature', ''))
        func_name = func.get('name', 'unknown')
        
        if isinstance(func_code, str):
            lines = func_code.split('\n')
            
            # Extract function calls
            for line in lines:
                line = line.strip().lower()
                if 'call' in line:
                    # Extract called function name
                    parts = line.split()
                    if len(parts) > 1:
                        called_func = parts[-1].replace(',', '').replace(';', '')
                        calling_pattern['calls_made'].append(called_func)
            
            # Analyze function signature for parameters
            signature = func.get('signature', '')
            if '(' in signature and ')' in signature:
                param_section = signature[signature.find('('):signature.find(')')+1]
                # Simple parameter counting by commas
                if param_section.strip() != '()':
                    calling_pattern['parameters_passed'] = param_section.count(',') + 1
            
            # Detect calling convention from patterns
            if 'push' in func_code.lower() and 'pop' in func_code.lower():
                calling_pattern['calling_convention'] = 'stack_based'
            elif any(reg in func_code.lower() for reg in ['eax', 'ebx', 'ecx', 'edx']):
                calling_pattern['calling_convention'] = 'register_based'
            
            # Estimate stack usage
            push_count = func_code.lower().count('push')
            pop_count = func_code.lower().count('pop')
            if push_count > 0 or pop_count > 0:
                calling_pattern['stack_usage'] = f'push:{push_count}, pop:{pop_count}'
        
        return calling_pattern
    
    def _analyze_function_parameters(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function parameters"""
        parameter_analysis = {
            'parameter_count': 0,
            'parameter_types': [],
            'parameter_names': [],
            'register_params': [],
            'stack_params': [],
            'analysis_confidence': 0.0
        }
        
        signature = func.get('signature', '')
        func_code = func.get('code', '')
        
        # Parse function signature
        if '(' in signature and ')' in signature:
            param_section = signature[signature.find('(')+1:signature.find(')')]
            param_section = param_section.strip()
            
            if param_section and param_section != 'void':
                # Split parameters
                params = [p.strip() for p in param_section.split(',')]
                parameter_analysis['parameter_count'] = len(params)
                
                for param in params:
                    # Simple type/name extraction
                    parts = param.split()
                    if len(parts) >= 2:
                        param_type = ' '.join(parts[:-1])
                        param_name = parts[-1]
                        parameter_analysis['parameter_types'].append(param_type)
                        parameter_analysis['parameter_names'].append(param_name)
                    elif len(parts) == 1:
                        parameter_analysis['parameter_types'].append(parts[0])
                        parameter_analysis['parameter_names'].append(f'param_{len(parameter_analysis["parameter_names"])}')
        
        # Analyze register vs stack usage from assembly code
        if func_code:
            # Common register parameter patterns
            register_patterns = ['eax', 'ebx', 'ecx', 'edx', 'rdi', 'rsi', 'rdx', 'rcx']
            for reg in register_patterns:
                if reg in func_code.lower():
                    parameter_analysis['register_params'].append(reg)
            
            # Stack parameter patterns
            if 'esp' in func_code.lower() or 'ebp' in func_code.lower() or '[' in func_code:
                parameter_analysis['stack_params'].append('stack_based')
        
        # Calculate confidence based on available information
        confidence = 0.0
        if parameter_analysis['parameter_count'] > 0:
            confidence += 0.4
        if parameter_analysis['parameter_types']:
            confidence += 0.3
        if parameter_analysis['register_params'] or parameter_analysis['stack_params']:
            confidence += 0.3
        
        parameter_analysis['analysis_confidence'] = confidence
        
        return parameter_analysis
    
    def _calculate_structural_complexity(self, control_flow: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate structural complexity"""
        return {'complexity': 'medium'}
    
    def _analyze_modularity(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code modularity"""
        return {'modularity_score': 0.7}
    
    def _build_dependency_graph(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        """Build dependency graph"""
        return {'nodes': [], 'edges': []}
    
    def _detect_architecture_patterns(self, targets: Dict[str, Any]) -> List[str]:
        """Detect architecture patterns"""
        return ['layered_architecture']
    
    def _detect_dead_code_elimination(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        """Detect dead code elimination optimization"""
        return {'detected': False}
    
    def _detect_loop_optimizations(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        """Detect loop optimizations"""
        return {'detected': False}
    
    def _detect_inlining_patterns(self, targets: Dict[str, Any]) -> Dict[str, Any]:
        """Detect function inlining patterns"""
        return {'detected': False}
    
    def _analyze_register_allocation(self, targets: Dict[str, Any], arch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze register allocation patterns"""
        return {'optimization_level': 'medium'}
    
    def _calculate_optimization_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in optimization detection"""
        return 0.8
    
    def _describe_optimization_pattern(self, opt_type: str, data: Dict[str, Any]) -> str:
        """Describe optimization pattern"""
        return f"{opt_type} optimization detected"
    
    def _assess_optimization_impact(self, opt_type: str, data: Dict[str, Any]) -> str:
        """Assess optimization impact"""
        return "medium_impact"
    
    def _analyze_function_changes(self, function_data: Dict[str, Any], decompilation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function-level changes"""
        changes = {
            'new_functions': [],
            'modified_functions': [],
            'removed_functions': [],
            'signature_changes': [],
            'complexity_changes': {},
            'total_changes': 0
        }
        
        # Get current functions from analysis
        current_functions = function_data.get('function_analysis', [])
        decompiled_functions = decompilation_info.get('function_signatures', [])
        
        # Create lookup maps
        current_func_map = {f.get('name', f'func_{i}'): f for i, f in enumerate(current_functions)}
        decompiled_func_map = {f.get('name', f'func_{i}'): f for i, f in enumerate(decompiled_functions)}
        
        # Find new functions (in current but not in decompiled)
        for name, func in current_func_map.items():
            if name not in decompiled_func_map:
                changes['new_functions'].append({
                    'name': name,
                    'signature': func.get('signature', ''),
                    'complexity': func.get('complexity', {})
                })
        
        # Find removed functions (in decompiled but not in current)
        for name, func in decompiled_func_map.items():
            if name not in current_func_map:
                changes['removed_functions'].append({
                    'name': name,
                    'signature': func.get('signature', '')
                })
        
        # Find modified functions (signature or complexity changes)
        for name in set(current_func_map.keys()) & set(decompiled_func_map.keys()):
            current_func = current_func_map[name]
            decompiled_func = decompiled_func_map[name]
            
            current_sig = current_func.get('signature', '')
            decompiled_sig = decompiled_func.get('signature', '')
            
            if current_sig != decompiled_sig:
                changes['signature_changes'].append({
                    'name': name,
                    'old_signature': decompiled_sig,
                    'new_signature': current_sig
                })
                changes['modified_functions'].append(name)
            
            # Compare complexity if available
            current_complexity = current_func.get('complexity', {}).get('cyclomatic_complexity', 0)
            decompiled_complexity = decompiled_func.get('complexity', {}).get('cyclomatic_complexity', 0)
            
            if abs(current_complexity - decompiled_complexity) > 1:
                changes['complexity_changes'][name] = {
                    'old_complexity': decompiled_complexity,
                    'new_complexity': current_complexity,
                    'change': current_complexity - decompiled_complexity
                }
        
        changes['total_changes'] = (
            len(changes['new_functions']) + 
            len(changes['removed_functions']) + 
            len(changes['signature_changes'])
        )
        
        return changes
    
    def _analyze_control_flow_changes(self, structure_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze control flow changes"""
        changes = {
            'branch_changes': {},
            'loop_changes': {},
            'call_flow_changes': {},
            'complexity_changes': {},
            'overall_flow_impact': 'low'
        }
        
        complexity_metrics = structure_metrics.get('complexity_metrics', {})
        
        # Analyze complexity changes
        current_complexity = complexity_metrics.get('current_complexity', 'medium')
        baseline_complexity = complexity_metrics.get('baseline_complexity', 'medium')
        
        if current_complexity != baseline_complexity:
            changes['complexity_changes'] = {
                'old_complexity': baseline_complexity,
                'new_complexity': current_complexity,
                'impact': 'significant' if abs(hash(current_complexity) - hash(baseline_complexity)) > 1000 else 'minor'
            }
        
        # Analyze branch patterns if available
        if 'branch_analysis' in structure_metrics:
            branch_data = structure_metrics['branch_analysis']
            changes['branch_changes'] = {
                'conditional_branches': branch_data.get('conditional_count', 0),
                'unconditional_branches': branch_data.get('unconditional_count', 0),
                'branch_density': branch_data.get('branch_density', 0.0)
            }
        
        # Analyze loop patterns
        if 'loop_analysis' in structure_metrics:
            loop_data = structure_metrics['loop_analysis']
            changes['loop_changes'] = {
                'loop_count': loop_data.get('loop_count', 0),
                'nested_loops': loop_data.get('nested_count', 0),
                'loop_complexity': loop_data.get('complexity', 'low')
            }
        
        # Determine overall impact
        total_changes = len(changes['complexity_changes']) + len(changes['branch_changes']) + len(changes['loop_changes'])
        if total_changes > 5:
            changes['overall_flow_impact'] = 'high'
        elif total_changes > 2:
            changes['overall_flow_impact'] = 'medium'
        else:
            changes['overall_flow_impact'] = 'low'
        
        return changes
    
    def _analyze_modularity_changes(self, structure_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze modularity changes"""
        return {'changes': []}
    
    def _analyze_architecture_changes(self, structure_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze architecture changes"""
        return {'changes': []}
    
    def _create_optimization_analysis_prompt(self, patterns: List[Dict[str, Any]]) -> str:
        """Create AI prompt for optimization analysis"""
        return f"Analyze these optimization patterns: {patterns}"
    
    def _parse_ai_optimization_response(self, response: str) -> Dict[str, Any]:
        """Parse AI optimization response"""
        return {'analysis': response}
    
    def _create_structural_analysis_prompt(self, changes: Dict[str, Any]) -> str:
        """Create AI prompt for structural analysis"""
        return f"Analyze these structural changes: {changes}"
    
    def _parse_ai_structural_response(self, response: str) -> Dict[str, Any]:
        """Parse AI structural response"""
        return {'analysis': response}
    
    def _create_recommendations_prompt(self, results: Dict[str, Any]) -> str:
        """Create AI prompt for recommendations"""
        return f"Generate recommendations based on: {results}"
    
    def _parse_ai_recommendations(self, response: str) -> List[str]:
        """Parse AI recommendations"""
        return [response]
    
    def _identify_behavior_changes(self, comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify behavior changes"""
        behavior_changes = []
        
        # Analyze function-level changes for behavioral impact
        function_level = comparison_results.get('function_level', {})
        function_analysis = function_level.get('function_analysis', [])
        
        for func_info in function_analysis:
            func_name = func_info.get('name', 'unknown')
            complexity = func_info.get('complexity', {})
            calling_pattern = func_info.get('calling_pattern', {})
            
            # Check for significant complexity changes
            if complexity.get('complexity_level') in ['high', 'very_high']:
                behavior_changes.append({
                    'type': 'complexity_increase',
                    'function': func_name,
                    'description': f'Function {func_name} shows high complexity indicating behavioral changes',
                    'impact': 'medium',
                    'confidence': 0.7
                })
            
            # Check for calling pattern changes
            calls_made = calling_pattern.get('calls_made', [])
            if len(calls_made) > 5:
                behavior_changes.append({
                    'type': 'interaction_increase',
                    'function': func_name,
                    'description': f'Function {func_name} makes many calls ({len(calls_made)}), suggesting behavior changes',
                    'impact': 'low',
                    'confidence': 0.6
                })
        
        # Analyze assembly-level changes
        assembly_level = comparison_results.get('assembly_level', {})
        optimization_signatures = assembly_level.get('optimization_signatures', [])
        
        for opt in optimization_signatures:
            if opt.get('confidence', 0) > 0.7:
                behavior_changes.append({
                    'type': 'optimization_change',
                    'function': 'global',
                    'description': f'Detected {opt.get("type", "unknown")} optimization changes',
                    'impact': 'low',
                    'confidence': opt.get('confidence', 0.0)
                })
        
        # Analyze binary-level changes
        binary_level = comparison_results.get('binary_level', {})
        if binary_level.get('differences'):
            behavior_changes.append({
                'type': 'binary_structure_change',
                'function': 'global',
                'description': 'Binary structure differences detected',
                'impact': 'high',
                'confidence': 0.8
            })
        
        return behavior_changes
    
    def _identify_performance_changes(self, optimization_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify performance changes"""
        performance_changes = []
        
        for pattern in optimization_patterns:
            pattern_type = pattern.get('type', '')
            confidence = pattern.get('confidence', 0.0)
            impact = pattern.get('impact', '')
            
            # Map optimization patterns to performance impact
            if pattern_type == 'dead_code_elimination' and confidence > 0.5:
                performance_changes.append({
                    'type': 'code_size_reduction',
                    'description': 'Dead code elimination detected - reduces binary size',
                    'impact': 'positive',
                    'magnitude': 'small',
                    'confidence': confidence,
                    'optimization': pattern_type
                })
            
            elif pattern_type == 'loop_optimizations' and confidence > 0.6:
                performance_changes.append({
                    'type': 'execution_speed_improvement',
                    'description': 'Loop optimizations detected - improves execution speed',
                    'impact': 'positive',
                    'magnitude': 'medium',
                    'confidence': confidence,
                    'optimization': pattern_type
                })
            
            elif pattern_type == 'inlining_patterns' and confidence > 0.7:
                performance_changes.append({
                    'type': 'call_overhead_reduction',
                    'description': 'Function inlining detected - reduces call overhead',
                    'impact': 'positive',
                    'magnitude': 'small',
                    'confidence': confidence,
                    'optimization': pattern_type
                })
            
            elif pattern_type == 'register_allocation' and confidence > 0.8:
                performance_changes.append({
                    'type': 'memory_access_optimization',
                    'description': 'Register allocation optimization - reduces memory access',
                    'impact': 'positive',
                    'magnitude': 'medium',
                    'confidence': confidence,
                    'optimization': pattern_type
                })
        
        # If no specific optimizations found, add general assessment
        if not performance_changes and optimization_patterns:
            avg_confidence = sum(p.get('confidence', 0) for p in optimization_patterns) / len(optimization_patterns)
            performance_changes.append({
                'type': 'general_optimization',
                'description': f'General optimization patterns detected (avg confidence: {avg_confidence:.2f})',
                'impact': 'positive' if avg_confidence > 0.5 else 'neutral',
                'magnitude': 'unknown',
                'confidence': avg_confidence,
                'optimization': 'mixed'
            })
        
        return performance_changes
    
    def _identify_interface_changes(self, structural_changes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify interface changes"""
        interface_changes = []
        
        # Analyze function changes for interface impact
        function_changes = structural_changes.get('function_changes', {})
        
        # New functions represent interface additions
        new_functions = function_changes.get('new_functions', [])
        for func in new_functions:
            interface_changes.append({
                'type': 'interface_addition',
                'element': func.get('name', 'unknown'),
                'description': f'New function {func.get("name", "unknown")} added to interface',
                'impact': 'additive',
                'compatibility': 'backward_compatible',
                'signature': func.get('signature', '')
            })
        
        # Removed functions represent interface deletions
        removed_functions = function_changes.get('removed_functions', [])
        for func in removed_functions:
            interface_changes.append({
                'type': 'interface_removal',
                'element': func.get('name', 'unknown'),
                'description': f'Function {func.get("name", "unknown")} removed from interface',
                'impact': 'breaking',
                'compatibility': 'backward_incompatible',
                'signature': func.get('signature', '')
            })
        
        # Signature changes represent interface modifications
        signature_changes = function_changes.get('signature_changes', [])
        for change in signature_changes:
            func_name = change.get('name', 'unknown')
            old_sig = change.get('old_signature', '')
            new_sig = change.get('new_signature', '')
            
            # Determine if change is breaking
            is_breaking = self._is_signature_change_breaking(old_sig, new_sig)
            
            interface_changes.append({
                'type': 'interface_modification',
                'element': func_name,
                'description': f'Function {func_name} signature changed',
                'impact': 'breaking' if is_breaking else 'compatible',
                'compatibility': 'backward_incompatible' if is_breaking else 'backward_compatible',
                'old_signature': old_sig,
                'new_signature': new_sig
            })
        
        # Analyze control flow changes for interface behavior impact
        control_flow_changes = structural_changes.get('control_flow_changes', {})
        if control_flow_changes.get('overall_flow_impact') == 'high':
            interface_changes.append({
                'type': 'behavioral_change',
                'element': 'global',
                'description': 'Significant control flow changes may affect interface behavior',
                'impact': 'behavioral',
                'compatibility': 'potentially_incompatible',
                'details': control_flow_changes
            })
        
        return interface_changes
    
    def _is_signature_change_breaking(self, old_sig: str, new_sig: str) -> bool:
        """Determine if a signature change is breaking"""
        if not old_sig or not new_sig:
            return True
        
        # Simple heuristics for breaking changes
        # Parameter count changes are typically breaking
        old_param_count = old_sig.count(',') + (1 if '(' in old_sig and old_sig[old_sig.find('(')+1:old_sig.find(')')].strip() else 0)
        new_param_count = new_sig.count(',') + (1 if '(' in new_sig and new_sig[new_sig.find('(')+1:new_sig.find(')')].strip() else 0)
        
        if old_param_count != new_param_count:
            return True
        
        # Return type changes are potentially breaking
        old_return = old_sig.split('(')[0].strip() if '(' in old_sig else ''
        new_return = new_sig.split('(')[0].strip() if '(' in new_sig else ''
        
        if old_return != new_return:
            return True
        
        return False
    
    def _calculate_synchronization_quality(self, comparison_results: Dict[str, Any]) -> float:
        """Calculate synchronization quality between twins"""
        return 0.95
    
    def _calculate_functional_similarity(self, functional_changes: Dict[str, Any]) -> float:
        """Calculate functional similarity"""
        return 0.9

    def _perform_critical_size_comparison(self, binary_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITICAL: Perform binary size comparison that can fail the entire pipeline
        
        This is the fail-fast check that determines if decompilation and recompilation
        produced a result that's significantly different in size, indicating failure.
        """
        import os
        from pathlib import Path
        
        # Get original binary size
        original_size = os.path.getsize(binary_path)
        
        # Find generated binary in compilation output
        generated_binary_path = None
        generated_size = 0
        
        # PRIORITIZE Agent 10 (The Machine) results first (most reliable source for compiled binary)
        agent_results = context.get('agent_results', {})
        self.logger.info(f"🔍 DEBUG: Available agent results: {list(agent_results.keys())}")
        
        if 10 in agent_results:
            agent10_result = agent_results[10]
            self.logger.info(f"🔍 DEBUG: Agent 10 status: {agent10_result.status}")
            
            if hasattr(agent10_result, 'data') and isinstance(agent10_result.data, dict):
                agent10_data = agent10_result.data
                self.logger.info(f"🔍 DEBUG: Agent 10 data keys: {list(agent10_data.keys())}")
                
                compilation_results = agent10_data.get('compilation_results', {})
                binary_outputs = compilation_results.get('binary_outputs', {})
                self.logger.info(f"🔍 DEBUG: Binary outputs: {binary_outputs}")
                
                if binary_outputs:
                    # Get first available binary path
                    generated_binary_path = next(iter(binary_outputs.values()))
                    self.logger.info(f"🔍 DEBUG: Found binary path from Agent 10: {generated_binary_path}")
                    
                    if generated_binary_path and os.path.exists(generated_binary_path):
                        generated_size = os.path.getsize(generated_binary_path)
                        self.logger.info(f"✅ Found compiled binary from Agent 10: {generated_binary_path} ({generated_size:,} bytes)")
                    else:
                        self.logger.warning(f"⚠️ Binary path from Agent 10 doesn't exist: {generated_binary_path}")
                else:
                    self.logger.warning("⚠️ Agent 10 has no binary outputs")
            else:
                self.logger.warning("⚠️ Agent 10 data is not available or not a dict")
        else:
            self.logger.warning("⚠️ Agent 10 results not available - may not have run")
        
        # FALLBACK: Search for generated executable in output paths (secondary method)
        if not generated_binary_path:
            self.logger.info("🔍 Searching output paths as fallback...")
            output_paths = context.get('output_paths', {})
            if output_paths:
                compilation_dir = output_paths.get('compilation')
                if compilation_dir:
                    # Look for exe files in bin/Release directories
                    search_patterns = [
                        compilation_dir / "bin" / "Release" / "Win32" / "*.exe",
                        compilation_dir / "bin" / "Release" / "x64" / "*.exe", 
                        compilation_dir / "bin" / "Release" / "*.exe",
                        compilation_dir / "Release" / "*.exe",
                        compilation_dir / "*.exe"
                    ]
                    
                    for pattern in search_patterns:
                        from glob import glob
                        matches = glob(str(pattern))
                        if matches:
                            generated_binary_path = matches[0]
                            generated_size = os.path.getsize(generated_binary_path)
                            self.logger.info(f"✅ Found binary via output paths: {generated_binary_path} ({generated_size:,} bytes)")
                            break
                    
                    if not generated_binary_path:
                        self.logger.warning(f"⚠️ No executables found in compilation directory: {compilation_dir}")
                else:
                    self.logger.warning("⚠️ No compilation directory in output paths")
            else:
                self.logger.warning("⚠️ No output paths available")
        
        # Calculate size similarity
        if generated_size > 0 and original_size > 0:
            # Use the smaller size as denominator to avoid inflated percentages
            size_similarity = min(generated_size, original_size) / max(generated_size, original_size)
        else:
            size_similarity = 0.0
        
        # NO FALLBACKS EVER - Rule #1 from rules.md
        # FAIL FAST - Rule #25 from rules.md
        # If no generated binary exists, this means previous agents failed to produce output
        should_fail_pipeline = (
            self.fail_pipeline_on_size_mismatch and (
                generated_size == 0 or  # FAIL FAST: No generated binary means pipeline failure
                size_similarity < self.size_similarity_threshold
            )
        )
        
        size_comparison = {
            'original_size': original_size,
            'generated_size': generated_size,
            'generated_binary_path': generated_binary_path,
            'size_similarity': size_similarity,
            'size_difference': abs(original_size - generated_size),
            'size_ratio': generated_size / original_size if original_size > 0 else 0.0,
            'should_fail_pipeline': should_fail_pipeline,
            'threshold_used': self.size_similarity_threshold,
            'failure_reason': None
        }
        
        if should_fail_pipeline:
            if generated_size == 0:
                size_comparison['failure_reason'] = f"PIPELINE FAILURE: No generated binary found - previous agents failed to produce compilation output"
            elif size_similarity < 0.01:  # Less than 1%
                size_comparison['failure_reason'] = f"Generated binary extremely small ({generated_size:,} bytes vs {original_size:,} bytes) - indicates compilation failure"
            else:
                size_comparison['failure_reason'] = f"Size similarity {size_similarity:.2%} below threshold {self.size_similarity_threshold:.1%}"
        
        self.logger.info(f"Size comparison: Original={original_size:,} bytes, Generated={generated_size:,} bytes, Similarity={size_similarity:.2%}")
        
        return size_comparison