"""
Agent 8: Binary Diff Analyzer
Compares decompiled output with reference patterns and analyzes differences.
"""

from typing import Dict, Any, List
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent8_BinaryDiffAnalyzer(BaseAgent):
    """Agent 8: Binary difference analysis and pattern matching"""
    
    def __init__(self):
        super().__init__(
            agent_id=8,
            name="BinaryDiffAnalyzer",
            dependencies=[6]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute binary diff analysis"""
        agent6_result = context['agent_results'].get(6)
        if not agent6_result or agent6_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 6 (OptimizationMatcher) did not complete successfully"
            )

        try:
            optimization_data = agent6_result.data
            diff_analysis = self._perform_diff_analysis(optimization_data, context)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=diff_analysis,
                metadata={
                    'depends_on': [6],
                    'analysis_type': 'binary_diff_analysis'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Binary diff analysis failed: {str(e)}"
            )

    def _perform_diff_analysis(self, optimization_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform binary difference analysis"""
        # Get binary path for analysis
        binary_path = context.get('global_data', {}).get('binary_path') or context.get('binary_path')
        
        return {
            'optimization_impact': self._analyze_optimization_impact(optimization_data),
            'pattern_matches': self._perform_pattern_matching(optimization_data),
            'difference_analysis': self._analyze_differences(optimization_data),
            'reconstruction_guidance': self._generate_reconstruction_guidance(optimization_data),
            'binary_path': binary_path,
            'analysis_confidence': self._calculate_analysis_confidence(optimization_data),
            'total_differences_found': self._count_total_differences(optimization_data),
            'critical_differences': self._count_critical_differences(optimization_data),
            'optimization_reversibility': self._calculate_reversibility(optimization_data)
        }

    def _analyze_optimization_impact(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of optimizations on binary structure"""
        # Extract optimization patterns from input data
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        optimization_patterns = optimization_data.get('optimization_patterns', {})
        
        impact_analysis = {
            'code_size_reduction': self._estimate_code_size_reduction(detected_optimizations),
            'execution_speed_improvement': self._estimate_speed_improvement(detected_optimizations),
            'register_usage_optimization': self._analyze_register_optimization(detected_optimizations),
            'memory_access_patterns': {
                'cache_friendly_accesses': self._analyze_cache_friendliness(detected_optimizations),
                'reduced_memory_footprint': self._estimate_memory_reduction(detected_optimizations),
                'eliminated_redundant_loads': self._count_eliminated_loads(detected_optimizations)
            },
            'control_flow_changes': {
                'loop_unrolling_detected': len([opt for opt in detected_optimizations if 'loop' in str(opt).lower()]),
                'branch_elimination': self._count_branch_eliminations(detected_optimizations),
                'function_inlining': self._count_function_inlining(detected_optimizations)
            },
            'data_structure_optimizations': {
                'struct_packing': self._detect_struct_packing(detected_optimizations),
                'array_optimizations': self._count_array_optimizations(detected_optimizations),
                'constant_folding': self._count_constant_folding(detected_optimizations)
            }
        }
        
        return impact_analysis

    def _perform_pattern_matching(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Match patterns against known optimization signatures"""
        optimization_patterns = optimization_data.get('optimization_patterns', {})
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        
        # Common optimization pattern signatures
        known_patterns = {
            'gcc_o2_optimizations': {
                'loop_unrolling': True,
                'function_inlining': True,
                'dead_code_elimination': True,
                'constant_propagation': True
            },
            'msvc_optimizations': {
                'register_allocation': True,
                'instruction_scheduling': True,
                'branch_prediction': True
            },
            'clang_optimizations': {
                'vectorization': True,
                'tail_call_optimization': True,
                'profile_guided_optimization': False
            }
        }
        
        pattern_matches = {
            'matched_patterns': [],
            'confidence_scores': {},
            'optimization_compiler': 'unknown',
            'optimization_level': 'unknown'
        }
        
        # Analyze detected optimizations against known patterns
        for pattern_name, pattern_features in known_patterns.items():
            match_score = 0
            total_features = len(pattern_features)
            
            for feature, expected in pattern_features.items():
                # Check if this feature is present in detected optimizations
                feature_present = any(feature.lower() in str(opt).lower() for opt in detected_optimizations)
                if feature_present == expected:
                    match_score += 1
            
            confidence = match_score / total_features if total_features > 0 else 0
            
            if confidence > 0.6:  # 60% match threshold
                pattern_matches['matched_patterns'].append(pattern_name)
                pattern_matches['confidence_scores'][pattern_name] = confidence
                
                # Determine most likely compiler
                if confidence > pattern_matches['confidence_scores'].get('best_match', 0):
                    if 'gcc' in pattern_name:
                        pattern_matches['optimization_compiler'] = 'GCC'
                        pattern_matches['optimization_level'] = 'O2'
                    elif 'msvc' in pattern_name:
                        pattern_matches['optimization_compiler'] = 'MSVC'
                        pattern_matches['optimization_level'] = 'O2'
                    elif 'clang' in pattern_name:
                        pattern_matches['optimization_compiler'] = 'Clang'
                        pattern_matches['optimization_level'] = 'O2'
        
        return pattern_matches

    def _analyze_differences(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze differences between optimized and expected patterns"""
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        confidence_score = optimization_data.get('confidence_score', 0.5)
        
        differences = {
            'structural_differences': {
                'unexpected_optimizations': [],
                'missing_expected_patterns': [],
                'optimization_level_mismatch': False
            },
            'semantic_differences': {
                'behavior_preserving': True,
                'side_effect_changes': [],
                'performance_impact': 'positive'
            },
            'reconstruction_challenges': {
                'irreversible_optimizations': [],
                'ambiguous_patterns': [],
                'information_loss': 'minimal'
            }
        }
        
        # Analyze for unexpected optimizations
        expected_optimizations = ['function_inlining', 'loop_unrolling', 'dead_code_elimination']
        for opt in detected_optimizations:
            opt_str = str(opt).lower()
            if not any(expected in opt_str for expected in expected_optimizations):
                differences['structural_differences']['unexpected_optimizations'].append(opt)
        
        # Check for missing expected patterns
        for expected in expected_optimizations:
            if not any(expected in str(opt).lower() for opt in detected_optimizations):
                differences['structural_differences']['missing_expected_patterns'].append(expected)
        
        # Assess reconstruction challenges based on optimization complexity
        if len(detected_optimizations) > 20:
            differences['reconstruction_challenges']['information_loss'] = 'significant'
            differences['reconstruction_challenges']['irreversible_optimizations'] = [
                'aggressive_inlining', 'loop_fusion', 'constant_folding'
            ]
        
        return differences

    def _generate_reconstruction_guidance(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate guidance for code reconstruction"""
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        confidence_score = optimization_data.get('confidence_score', 0.5)
        
        guidance = {
            'reconstruction_strategy': 'incremental',
            'priority_areas': [],
            'recommended_approaches': {},
            'complexity_assessment': 'medium',
            'estimated_accuracy': 0.75,
            'reconstruction_steps': []
        }
        
        # Determine reconstruction strategy based on optimization complexity
        if len(detected_optimizations) < 10:
            guidance['reconstruction_strategy'] = 'direct'
            guidance['complexity_assessment'] = 'low'
            guidance['estimated_accuracy'] = 0.90
        elif len(detected_optimizations) > 30:
            guidance['reconstruction_strategy'] = 'heuristic'
            guidance['complexity_assessment'] = 'high'
            guidance['estimated_accuracy'] = 0.60
        
        # Priority areas based on optimization types
        optimization_types = set()
        for opt in detected_optimizations:
            opt_str = str(opt).lower()
            if 'loop' in opt_str:
                optimization_types.add('loop_optimizations')
            elif 'inline' in opt_str:
                optimization_types.add('function_inlining')
            elif 'constant' in opt_str:
                optimization_types.add('constant_optimizations')
            elif 'register' in opt_str:
                optimization_types.add('register_optimizations')
        
        guidance['priority_areas'] = list(optimization_types)
        
        # Recommended approaches for each optimization type
        approach_map = {
            'loop_optimizations': 'pattern_recognition_with_control_flow_analysis',
            'function_inlining': 'call_graph_reconstruction',
            'constant_optimizations': 'static_analysis_with_value_tracking',
            'register_optimizations': 'data_flow_analysis'
        }
        
        for opt_type in optimization_types:
            guidance['recommended_approaches'][opt_type] = approach_map.get(opt_type, 'generic_analysis')
        
        # Generate reconstruction steps
        guidance['reconstruction_steps'] = [
            'identify_function_boundaries',
            'reconstruct_control_flow_graphs',
            'analyze_data_dependencies',
            'reverse_optimization_transformations',
            'validate_semantic_equivalence'
        ]
        
        return guidance
    
    def _calculate_analysis_confidence(self, optimization_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on optimization data quality"""
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        confidence_score = optimization_data.get('confidence_score', 0.5)
        
        # Basic confidence calculation based on available data
        base_confidence = confidence_score
        data_completeness = min(len(detected_optimizations) / 10.0, 1.0)  # Normalize to 0-1
        
        return min(base_confidence * (0.5 + data_completeness * 0.5), 1.0)
    
    def _count_total_differences(self, optimization_data: Dict[str, Any]) -> int:
        """Count total differences found in optimization analysis"""
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        # Simple approximation: assume differences correlate with optimization count
        return len(detected_optimizations)
    
    def _count_critical_differences(self, optimization_data: Dict[str, Any]) -> int:
        """Count critical differences that affect reconstruction accuracy"""
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        # Estimate critical differences as ~30% of total optimizations
        return max(1, len(detected_optimizations) // 3)
    
    def _calculate_reversibility(self, optimization_data: Dict[str, Any]) -> float:
        """Calculate optimization reversibility score"""
        detected_optimizations = optimization_data.get('detected_optimizations', [])
        # Basic reversibility: fewer optimizations = higher reversibility
        if len(detected_optimizations) == 0:
            return 1.0
        elif len(detected_optimizations) < 5:
            return 0.9
        elif len(detected_optimizations) < 15:
            return 0.7
        else:
            return 0.5
    
    def _estimate_code_size_reduction(self, detected_optimizations: List[Any]) -> float:
        """Estimate code size reduction from optimizations"""
        # Basic estimation: each optimization reduces code size by ~2-5%
        size_reducing_opts = ['dead_code_elimination', 'constant_folding', 'loop_unrolling']
        reduction_count = sum(1 for opt in detected_optimizations 
                            if any(pattern in str(opt).lower() for pattern in size_reducing_opts))
        return min(reduction_count * 0.03, 0.4)  # Max 40% reduction
    
    def _estimate_speed_improvement(self, detected_optimizations: List[Any]) -> float:
        """Estimate execution speed improvement from optimizations"""
        # Basic estimation: performance optimizations provide ~5-15% improvement each
        speed_opts = ['loop_unrolling', 'function_inlining', 'register_allocation', 'vectorization']
        improvement_count = sum(1 for opt in detected_optimizations 
                              if any(pattern in str(opt).lower() for pattern in speed_opts))
        return min(improvement_count * 0.1, 0.8)  # Max 80% improvement
    
    def _analyze_register_optimization(self, detected_optimizations: List[Any]) -> float:
        """Analyze register usage optimization level"""
        # Check for register-related optimizations
        register_opts = ['register_allocation', 'register_spilling', 'register_coalescing']
        register_count = sum(1 for opt in detected_optimizations 
                           if any(pattern in str(opt).lower() for pattern in register_opts))
        return min(register_count * 0.2, 1.0)  # Scale 0-1
    
    def _analyze_cache_friendliness(self, detected_optimizations: List[Any]) -> float:
        """Analyze cache-friendly access patterns"""
        # Look for cache-related optimizations
        cache_opts = ['loop_blocking', 'data_locality', 'prefetch', 'vectorization']
        cache_count = sum(1 for opt in detected_optimizations 
                        if any(pattern in str(opt).lower() for pattern in cache_opts))
        return min(cache_count * 0.25, 1.0)  # Scale 0-1
    
    def _estimate_memory_reduction(self, detected_optimizations: List[Any]) -> float:
        """Estimate memory footprint reduction"""
        # Look for memory-related optimizations
        memory_opts = ['dead_code_elimination', 'constant_folding', 'struct_packing']
        memory_count = sum(1 for opt in detected_optimizations 
                         if any(pattern in str(opt).lower() for pattern in memory_opts))
        return min(memory_count * 0.05, 0.3)  # Max 30% memory reduction
    
    def _count_eliminated_loads(self, detected_optimizations: List[Any]) -> int:
        """Count eliminated redundant memory loads"""
        # Look for load-related optimizations
        load_opts = ['load_elimination', 'redundancy_elimination', 'common_subexpression']
        return sum(1 for opt in detected_optimizations 
                  if any(pattern in str(opt).lower() for pattern in load_opts))
    
    def _count_branch_eliminations(self, detected_optimizations: List[Any]) -> int:
        """Count eliminated branch instructions"""
        # Look for branch-related optimizations
        branch_opts = ['branch_elimination', 'branch_prediction', 'predication']
        return sum(1 for opt in detected_optimizations 
                  if any(pattern in str(opt).lower() for pattern in branch_opts))
    
    def _count_function_inlining(self, detected_optimizations: List[Any]) -> int:
        """Count function inlining instances"""
        # Look for inlining-related optimizations
        inline_opts = ['function_inlining', 'inline', 'call_elimination']
        return sum(1 for opt in detected_optimizations 
                  if any(pattern in str(opt).lower() for pattern in inline_opts))
    
    def _detect_struct_packing(self, detected_optimizations: List[Any]) -> bool:
        """Detect structure packing optimizations"""
        # Look for struct/data layout optimizations
        packing_opts = ['struct_packing', 'data_layout', 'padding_elimination']
        return any(any(pattern in str(opt).lower() for pattern in packing_opts) 
                  for opt in detected_optimizations)
    
    def _count_array_optimizations(self, detected_optimizations: List[Any]) -> int:
        """Count array-related optimizations"""
        # Look for array-related optimizations
        array_opts = ['vectorization', 'loop_unrolling', 'array_bounds', 'simd']
        return sum(1 for opt in detected_optimizations 
                  if any(pattern in str(opt).lower() for pattern in array_opts))
    
    def _count_constant_folding(self, detected_optimizations: List[Any]) -> int:
        """Count constant folding optimizations"""
        # Look for constant-related optimizations
        constant_opts = ['constant_folding', 'constant_propagation', 'compile_time_evaluation']
        return sum(1 for opt in detected_optimizations 
                  if any(pattern in str(opt).lower() for pattern in constant_opts))