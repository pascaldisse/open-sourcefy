"""
Agent 6: Optimization Matcher
Detects compiler optimizations and matches optimization patterns.
Enhanced with AI-powered pattern recognition for Phase 3.
"""

import re
from typing import Dict, Any, List, Tuple
from ..agent_base import BaseAgent, AgentResult, AgentStatus
from ..ai_enhancement import AIEnhancementCoordinator, MLModelType

# Import pattern engine
try:
    from src.ml.pattern_engine import PatternEngine
    PATTERN_ENGINE_AVAILABLE = True
except ImportError:
    PATTERN_ENGINE_AVAILABLE = False


class Agent6_OptimizationMatcher(BaseAgent):
    """Agent 6: Optimization pattern detection and matching"""
    
    def __init__(self):
        super().__init__(
            agent_id=6,
            name="OptimizationMatcher",
            dependencies=[4]
        )
        
        # Initialize optimization pattern database
        self.optimization_patterns = self._initialize_optimization_patterns()
        
        # Initialize AI enhancement coordinator
        self.ai_coordinator = AIEnhancementCoordinator()
        
        # Initialize pattern engine if available
        if PATTERN_ENGINE_AVAILABLE:
            self.pattern_engine = PatternEngine()
        else:
            self.pattern_engine = None

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute optimization pattern matching"""
        # Get data from Agent 4
        agent4_result = context['agent_results'].get(4)
        if not agent4_result or agent4_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 4 (BasicDecompiler) did not complete successfully"
            )

        try:
            decompilation_data = agent4_result.data
            # Also get architecture analysis if available
            agent2_result = context['agent_results'].get(2)
            arch_analysis = agent2_result.data if agent2_result else {}
            
            optimization_analysis = self._analyze_optimizations(decompilation_data, arch_analysis, context)
            
            # Apply AI enhancement for improved pattern recognition
            ai_enhancements = self.ai_coordinator.enhance_analysis(
                {**decompilation_data, 'optimization_analysis': optimization_analysis}, 
                context
            )
            
            # Integrate AI insights into optimization analysis
            optimization_analysis['ai_enhancements'] = ai_enhancements
            optimization_analysis['enhanced_patterns'] = self._integrate_ai_patterns(
                optimization_analysis, ai_enhancements
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=optimization_analysis,
                metadata={
                    'depends_on': [4],
                    'analysis_type': 'optimization_matching',
                    'ai_enhanced': True,
                    'enhancement_score': ai_enhancements.get('integration_score', 0.0)
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Optimization matching failed: {str(e)}"
            )

    def _initialize_optimization_patterns(self) -> Dict[str, Any]:
        """Initialize optimization pattern database"""
        return {
            'function_optimizations': {
                'inlining': {
                    'patterns': [
                        r'inline.*function',
                        r'__forceinline',
                        r'static.*inline'
                    ],
                    'indicators': [
                        'small_function_size',
                        'single_call_site',
                        'simple_operations'
                    ],
                    'reversibility': 'high'
                },
                'tail_call_optimization': {
                    'patterns': [
                        r'jmp.*\[.*\]',  # Jump table
                        r'call.*followed_by_ret',
                        r'tail.*call'
                    ],
                    'indicators': [
                        'ending_with_call',
                        'no_local_variables_after_call',
                        'parameter_forwarding'
                    ],
                    'reversibility': 'medium'
                },
                'dead_code_elimination': {
                    'patterns': [
                        r'unreachable.*code',
                        r'unused.*variable',
                        r'eliminated.*branch'
                    ],
                    'indicators': [
                        'missing_code_blocks',
                        'simplified_control_flow',
                        'reduced_binary_size'
                    ],
                    'reversibility': 'low'
                }
            },
            'loop_optimizations': {
                'loop_unrolling': {
                    'patterns': [
                        r'repeated.*instructions',
                        r'unrolled.*loop',
                        r'duplicated.*loop_body'
                    ],
                    'indicators': [
                        'repeated_instruction_sequences',
                        'no_loop_counter',
                        'sequential_array_access'
                    ],
                    'reversibility': 'high'
                },
                'loop_invariant_code_motion': {
                    'patterns': [
                        r'hoisted.*calculation',
                        r'moved.*outside_loop',
                        r'invariant.*expression'
                    ],
                    'indicators': [
                        'calculations_before_loop',
                        'reduced_loop_complexity',
                        'pre_computed_values'
                    ],
                    'reversibility': 'medium'
                }
            },
            'data_optimizations': {
                'constant_propagation': {
                    'patterns': [
                        r'immediate.*value',
                        r'constant.*folding',
                        r'literal.*substitution'
                    ],
                    'indicators': [
                        'hardcoded_values',
                        'no_variable_assignments',
                        'simplified_expressions'
                    ],
                    'reversibility': 'low'
                },
                'register_allocation': {
                    'patterns': [
                        r'optimized.*register_usage',
                        r'minimal.*memory_access',
                        r'register.*spilling'
                    ],
                    'indicators': [
                        'efficient_register_usage',
                        'few_memory_loads',
                        'register_reuse'
                    ],
                    'reversibility': 'medium'
                }
            },
            'control_flow_optimizations': {
                'branch_prediction': {
                    'patterns': [
                        r'likely.*branch',
                        r'unlikely.*branch',
                        r'predicted.*jump'
                    ],
                    'indicators': [
                        'reordered_basic_blocks',
                        'fallthrough_optimization',
                        'branch_target_alignment'
                    ],
                    'reversibility': 'high'
                },
                'jump_threading': {
                    'patterns': [
                        r'threading.*jump',
                        r'duplicated.*basic_block',
                        r'specialized.*path'
                    ],
                    'indicators': [
                        'duplicated_code_blocks',
                        'specialized_execution_paths',
                        'reduced_indirect_jumps'
                    ],
                    'reversibility': 'medium'
                }
            }
        }

    def _analyze_optimizations(self, decompilation_data: Dict[str, Any], arch_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization patterns in decompiled code"""
        analysis = {
            'detected_optimizations': [],
            'optimization_level': 'unknown',
            'compiler_hints': {},
            'reversibility_analysis': {},
            'reconstruction_strategies': [],
            'confidence_scores': {}
        }
        
        # Analyze function-level optimizations
        analysis['detected_optimizations'].extend(self._detect_function_optimizations(decompilation_data))
        
        # Analyze loop optimizations
        analysis['detected_optimizations'].extend(self._detect_loop_optimizations(decompilation_data))
        
        # Analyze data optimizations
        analysis['detected_optimizations'].extend(self._detect_data_optimizations(decompilation_data))
        
        # Analyze control flow optimizations
        analysis['detected_optimizations'].extend(self._detect_control_flow_optimizations(decompilation_data))
        
        # Phase 3 enhancement: Detect algorithm patterns
        algorithm_patterns = self._detect_algorithm_patterns(decompilation_data)
        analysis['algorithm_patterns'] = algorithm_patterns
        analysis['algorithm_reconstruction_strategies'] = self._generate_algorithm_reconstruction_strategies(algorithm_patterns)
        
        # Apply pattern engine analysis if available
        if self.pattern_engine:
            pattern_analysis = self._apply_pattern_engine_analysis(decompilation_data)
            analysis['pattern_engine_results'] = pattern_analysis
            analysis['detected_optimizations'].extend(pattern_analysis.get('optimization_patterns', []))
        
        # Estimate optimization level
        analysis['optimization_level'] = self._estimate_optimization_level(analysis['detected_optimizations'])
        
        # Generate compiler hints
        analysis['compiler_hints'] = self._generate_compiler_hints(analysis['detected_optimizations'], arch_analysis)
        
        # Analyze reversibility
        analysis['reversibility_analysis'] = self._analyze_reversibility(analysis['detected_optimizations'])
        
        # Generate reconstruction strategies
        analysis['reconstruction_strategies'] = self._generate_reconstruction_strategies(analysis['detected_optimizations'])
        
        # Calculate confidence scores
        analysis['confidence_scores'] = self._calculate_confidence_scores(analysis['detected_optimizations'])
        
        return analysis

    def _detect_function_optimizations(self, decompilation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect function-level optimizations"""
        optimizations = []
        
        # Get function data
        functions = decompilation_data.get('decompiled_functions', {})
        analysis_summary = decompilation_data.get('analysis_summary', {})
        
        # Check for inlining indicators
        total_functions = analysis_summary.get('total_functions', 0)
        if total_functions < 5:  # Very few functions might indicate heavy inlining
            optimizations.append({
                'type': 'function_inlining',
                'category': 'function_optimizations',
                'confidence': 0.7,
                'indicators': ['few_functions_detected'],
                'evidence': f'Only {total_functions} functions detected',
                'reversibility': 'medium'
            })
        
        # Tail call optimization detection would be implemented here
        # Currently requires detailed assembly analysis
        
        return optimizations

    def _detect_loop_optimizations(self, decompilation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect loop-level optimizations"""
        optimizations = []
        
        # Analyze objdump output for loop patterns
        objdump_output = decompilation_data.get('objdump_output', {})
        if objdump_output:
            functions = objdump_output.get('functions', [])
            
            for func in functions:
                instructions = func.get('instructions', [])
                
                # Look for unrolled loop patterns
                if self._detect_unrolled_loops(instructions):
                    optimizations.append({
                        'type': 'loop_unrolling',
                        'category': 'loop_optimizations',
                        'confidence': 0.75,
                        'function': func.get('name', 'unknown'),
                        'indicators': ['repeated_instruction_patterns'],
                        'reversibility': 'high'
                    })
        
        return optimizations

    def _detect_data_optimizations(self, decompilation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect data-level optimizations"""
        optimizations = []
        
        # Check for constant propagation
        strings = decompilation_data.get('strings', [])
        if len(strings) < 10:  # Few strings might indicate constant folding
            optimizations.append({
                'type': 'constant_propagation',
                'category': 'data_optimizations',
                'confidence': 0.6,
                'indicators': ['few_string_literals'],
                'evidence': f'Only {len(strings)} strings found',
                'reversibility': 'low'
            })
        
        return optimizations

    def _detect_control_flow_optimizations(self, decompilation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect control flow optimizations"""
        optimizations = []
        
        # Analyze function complexity
        analysis_summary = decompilation_data.get('analysis_summary', {})
        total_functions = analysis_summary.get('total_functions', 0)
        
        if total_functions > 0:
            # If we have objdump data, analyze instruction patterns
            objdump_output = decompilation_data.get('objdump_output', {})
            if objdump_output:
                instruction_count = objdump_output.get('analysis', {}).get('instruction_count', 0)
                if instruction_count > 0:
                    avg_instructions_per_function = instruction_count / total_functions
                    
                    if avg_instructions_per_function < 20:  # Small functions might indicate optimization
                        optimizations.append({
                            'type': 'control_flow_simplification',
                            'category': 'control_flow_optimizations',
                            'confidence': 0.65,
                            'indicators': ['small_function_size'],
                            'evidence': f'Average {avg_instructions_per_function:.1f} instructions per function',
                            'reversibility': 'medium'
                        })
        
        return optimizations

    def _has_tail_call_pattern(self, func_data: Any) -> bool:
        """Check if function has tail call optimization pattern"""
        # Basic pattern detection for tail calls
        if not func_data:
            return False
        
        # Look for simple indicators of tail call optimization
        func_str = str(func_data).lower()
        tail_indicators = ['jmp', 'tail', 'call_followed_by_ret']
        return any(indicator in func_str for indicator in tail_indicators)

    def _detect_unrolled_loops(self, instructions: List[str]) -> bool:
        """Detect unrolled loop patterns in instructions"""
        if len(instructions) < 4:
            return False
        
        # Look for repeated instruction patterns
        pattern_length = 3
        for i in range(len(instructions) - pattern_length * 2):
            pattern = instructions[i:i + pattern_length]
            next_pattern = instructions[i + pattern_length:i + pattern_length * 2]
            
            # Simple pattern matching (this could be more sophisticated)
            if self._instructions_similar(pattern, next_pattern):
                return True
        
        return False

    def _instructions_similar(self, pattern1: List[str], pattern2: List[str]) -> bool:
        """Check if two instruction patterns are similar (indicating unrolling)"""
        if len(pattern1) != len(pattern2):
            return False
        
        similar_count = 0
        for i1, i2 in zip(pattern1, pattern2):
            # Remove addresses and focus on instruction types
            inst1 = re.sub(r'0x[0-9a-fA-F]+', 'ADDR', i1)
            inst2 = re.sub(r'0x[0-9a-fA-F]+', 'ADDR', i2)
            
            if inst1.split()[0] == inst2.split()[0]:  # Same instruction type
                similar_count += 1
        
        return similar_count >= len(pattern1) * 0.7  # 70% similarity threshold

    def _estimate_optimization_level(self, optimizations: List[Dict[str, Any]]) -> str:
        """Estimate compiler optimization level based on detected patterns"""
        if not optimizations:
            return 'O0'  # No optimization
        
        optimization_count = len(optimizations)
        avg_confidence = sum(opt.get('confidence', 0) for opt in optimizations) / optimization_count
        
        if optimization_count >= 5 and avg_confidence > 0.8:
            return 'O3'  # Aggressive optimization
        elif optimization_count >= 3 and avg_confidence > 0.7:
            return 'O2'  # Standard optimization
        elif optimization_count >= 1 and avg_confidence > 0.6:
            return 'O1'  # Basic optimization
        else:
            return 'O0'  # Minimal optimization

    def _generate_compiler_hints(self, optimizations: List[Dict[str, Any]], arch_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate hints about the compiler used"""
        hints = {
            'likely_compiler': 'unknown',
            'optimization_flags': [],
            'target_architecture': arch_analysis.get('architecture', 'unknown'),
            'abi_hints': []
        }
        
        # Analyze optimization patterns to guess compiler
        optimization_types = [opt.get('type', '') for opt in optimizations]
        
        if 'tail_call_optimization' in optimization_types:
            hints['optimization_flags'].append('-foptimize-sibling-calls')
        
        if 'loop_unrolling' in optimization_types:
            hints['optimization_flags'].append('-funroll-loops')
        
        if 'constant_propagation' in optimization_types:
            hints['optimization_flags'].append('-fconstant-propagation')
        
        # Guess compiler based on architecture and patterns
        architecture = arch_analysis.get('architecture', 'unknown')
        if architecture in ['x86', 'x64']:
            hints['likely_compiler'] = 'GCC or MSVC'
        elif architecture in ['ARM', 'ARM64']:
            hints['likely_compiler'] = 'GCC or Clang'
        
        return hints

    def _analyze_reversibility(self, optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how reversible the detected optimizations are"""
        reversibility = {
            'overall_reversibility': 'unknown',
            'high_reversibility': [],
            'medium_reversibility': [],
            'low_reversibility': [],
            'reversibility_score': 0.0
        }
        
        if not optimizations:
            return reversibility
        
        scores = []
        for opt in optimizations:
            rev_level = opt.get('reversibility', 'unknown')
            opt_info = {
                'type': opt.get('type', 'unknown'),
                'confidence': opt.get('confidence', 0)
            }
            
            if rev_level == 'high':
                reversibility['high_reversibility'].append(opt_info)
                scores.append(0.9)
            elif rev_level == 'medium':
                reversibility['medium_reversibility'].append(opt_info)
                scores.append(0.6)
            elif rev_level == 'low':
                reversibility['low_reversibility'].append(opt_info)
                scores.append(0.3)
        
        if scores:
            reversibility['reversibility_score'] = sum(scores) / len(scores)
            
            if reversibility['reversibility_score'] > 0.8:
                reversibility['overall_reversibility'] = 'high'
            elif reversibility['reversibility_score'] > 0.5:
                reversibility['overall_reversibility'] = 'medium'
            else:
                reversibility['overall_reversibility'] = 'low'
        
        return reversibility

    def _generate_reconstruction_strategies(self, optimizations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate strategies for reconstructing original code"""
        strategies = []
        
        optimization_types = [opt.get('type', '') for opt in optimizations]
        
        if 'function_inlining' in optimization_types:
            strategies.append({
                'strategy': 'function_extraction',
                'description': 'Extract inlined functions back to separate functions',
                'difficulty': 'medium',
                'tools': ['static_analysis', 'pattern_matching'],
                'success_probability': 0.7
            })
        
        if 'loop_unrolling' in optimization_types:
            strategies.append({
                'strategy': 'loop_reconstruction',
                'description': 'Reconstruct original loop structure from unrolled code',
                'difficulty': 'low',
                'tools': ['pattern_recognition', 'control_flow_analysis'],
                'success_probability': 0.9
            })
        
        if 'constant_propagation' in optimization_types:
            strategies.append({
                'strategy': 'variable_reconstruction',
                'description': 'Recreate variables from propagated constants',
                'difficulty': 'high',
                'tools': ['data_flow_analysis', 'symbolic_execution'],
                'success_probability': 0.4
            })
        
        if 'tail_call_optimization' in optimization_types:
            strategies.append({
                'strategy': 'call_reconstruction',
                'description': 'Reconstruct function calls from tail call optimizations',
                'difficulty': 'medium',
                'tools': ['control_flow_analysis', 'function_analysis'],
                'success_probability': 0.8
            })
        
        return strategies

    def _calculate_confidence_scores(self, optimizations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects"""
        scores = {
            'optimization_detection': 0.0,
            'level_estimation': 0.0,
            'reversibility_analysis': 0.0,
            'overall_confidence': 0.0
        }
        
        if not optimizations:
            return scores
        
        # Optimization detection confidence
        confidences = [opt.get('confidence', 0) for opt in optimizations]
        scores['optimization_detection'] = sum(confidences) / len(confidences)
        
        # Level estimation confidence (based on number and quality of detections)
        level_confidence = min(len(optimizations) / 5.0, 1.0) * scores['optimization_detection']
        scores['level_estimation'] = level_confidence
        
        # Reversibility analysis confidence
        reversible_opts = [opt for opt in optimizations if opt.get('reversibility') in ['high', 'medium']]
        if reversible_opts:
            scores['reversibility_analysis'] = len(reversible_opts) / len(optimizations)
        
        # Overall confidence
        scores['overall_confidence'] = (
            scores['optimization_detection'] * 0.4 +
            scores['level_estimation'] * 0.3 +
            scores['reversibility_analysis'] * 0.3
        )
        
        return scores
    
    def _integrate_ai_patterns(self, optimization_analysis: Dict[str, Any], ai_enhancements: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate AI-detected patterns with traditional optimization analysis"""
        enhanced_patterns = {
            'traditional_optimizations': optimization_analysis.get('detected_optimizations', []),
            'ai_detected_patterns': [],
            'confidence_improvements': {},
            'new_optimization_types': [],
            'pattern_correlations': []
        }
        
        # Extract AI pattern analysis
        pattern_analysis = ai_enhancements.get('pattern_analysis')
        if pattern_analysis and hasattr(pattern_analysis, 'prediction'):
            ai_patterns = pattern_analysis.prediction
            
            # Check for optimization patterns in AI analysis
            opt_analysis = ai_patterns.get('optimization_analysis', {})
            if opt_analysis:
                enhanced_patterns['ai_detected_patterns'] = opt_analysis.get('detected_optimizations', [])
                
                # Improve confidence scores based on AI analysis
                ai_opt_level = opt_analysis.get('optimization_level_estimate', 'O0')
                traditional_opt_level = optimization_analysis.get('optimization_level', 'unknown')
                
                if ai_opt_level != 'O0' and traditional_opt_level in ['unknown', 'O0']:
                    enhanced_patterns['confidence_improvements']['optimization_level'] = {
                        'traditional': traditional_opt_level,
                        'ai_enhanced': ai_opt_level,
                        'confidence_boost': 0.3
                    }
                
                # Identify new optimization types detected by AI
                ai_opt_types = [opt.get('type', '') for opt in enhanced_patterns['ai_detected_patterns']]
                traditional_opt_types = [opt.get('type', '') for opt in enhanced_patterns['traditional_optimizations']]
                
                new_types = set(ai_opt_types) - set(traditional_opt_types)
                enhanced_patterns['new_optimization_types'] = list(new_types)
                
                # Find pattern correlations
                for ai_opt in enhanced_patterns['ai_detected_patterns']:
                    for trad_opt in enhanced_patterns['traditional_optimizations']:
                        correlation = self._calculate_pattern_correlation(ai_opt, trad_opt)
                        if correlation > 0.7:
                            enhanced_patterns['pattern_correlations'].append({
                                'ai_pattern': ai_opt.get('type', 'unknown'),
                                'traditional_pattern': trad_opt.get('type', 'unknown'),
                                'correlation_score': correlation,
                                'evidence': 'High pattern similarity detected'
                            })
        
        # Integrate quality assessment insights
        quality_assessment = ai_enhancements.get('quality_assessment')
        if quality_assessment and hasattr(quality_assessment, 'prediction'):
            quality_data = quality_assessment.prediction
            performance_analysis = quality_data.get('performance_analysis', {})
            
            if performance_analysis:
                opt_score = performance_analysis.get('optimization_score', 0)
                enhanced_patterns['quality_insights'] = {
                    'optimization_effectiveness': opt_score,
                    'performance_recommendations': quality_data.get('improvement_suggestions', [])
                }
        
        return enhanced_patterns
    
    def _calculate_pattern_correlation(self, ai_pattern: Dict[str, Any], traditional_pattern: Dict[str, Any]) -> float:
        """Calculate correlation between AI-detected and traditionally detected patterns"""
        ai_type = ai_pattern.get('type', '').lower()
        trad_type = traditional_pattern.get('type', '').lower()
        
        # Simple correlation based on pattern type similarity
        if ai_type == trad_type:
            return 1.0
        
        # Check for related pattern types
        related_patterns = {
            'loop_unrolling': ['unrolling', 'loop_optimization'],
            'function_inlining': ['inlining', 'function_optimization'],
            'constant_propagation': ['constant_folding', 'data_optimization'],
            'vectorization': ['simd', 'parallel_optimization']
        }
        
        for base_pattern, related in related_patterns.items():
            if (base_pattern in ai_type or ai_type in related) and \
               (base_pattern in trad_type or trad_type in related):
                return 0.8
        
        # Check for partial matches
        common_words = set(ai_type.split('_')) & set(trad_type.split('_'))
        if common_words:
            return len(common_words) / max(len(ai_type.split('_')), len(trad_type.split('_')))
        
        return 0.0
    
    def _apply_pattern_engine_analysis(self, decompilation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply pattern engine analysis to decompilation data"""
        pattern_results = {
            'patterns_analyzed': 0,
            'optimization_patterns': [],
            'code_suggestions': [],
            'quality_metrics': {},
            'confidence': 0.0
        }
        
        try:
            # Get decompiled functions
            functions = decompilation_data.get('decompiled_functions', {})
            if not functions:
                return pattern_results
            
            total_confidence = 0.0
            analyzed_functions = 0
            
            for func_name, func_data in functions.items():
                if isinstance(func_data, dict):
                    # Get assembly or decompiled code
                    assembly_code = func_data.get('assembly_code', '')
                    decompiled_code = func_data.get('decompiled_code', '')
                    
                    # Use assembly code if available, otherwise use decompiled code
                    code_to_analyze = assembly_code if assembly_code else decompiled_code
                    
                    if code_to_analyze:
                        # Create context for pattern analysis
                        context = {
                            'function_name': func_name,
                            'binary_type': 'pe',  # Assume PE for now
                            'architecture': 'x86'  # Default architecture
                        }
                        
                        # Analyze with pattern engine
                        result = self.pattern_engine.analyze_code_block(code_to_analyze, context)
                        
                        # Extract optimization-related patterns
                        for pattern in result.get('patterns', []):
                            if self._is_optimization_pattern(pattern):
                                optimization_pattern = {
                                    'type': f"pattern_engine_{pattern.get('type', 'unknown')}",
                                    'function': func_name,
                                    'confidence': pattern.get('confidence', 0.5),
                                    'details': pattern.get('details', ''),
                                    'suggestions': result.get('suggestions', [])
                                }
                                pattern_results['optimization_patterns'].append(optimization_pattern)
                        
                        # Accumulate quality metrics
                        total_confidence += result.get('confidence', 0.0)
                        analyzed_functions += 1
                        
                        # Add code suggestions
                        pattern_results['code_suggestions'].extend(result.get('suggestions', []))
            
            # Calculate overall metrics
            if analyzed_functions > 0:
                pattern_results['confidence'] = total_confidence / analyzed_functions
                pattern_results['patterns_analyzed'] = analyzed_functions
                pattern_results['quality_metrics'] = {
                    'average_confidence': pattern_results['confidence'],
                    'functions_analyzed': analyzed_functions,
                    'patterns_found': len(pattern_results['optimization_patterns']),
                    'suggestions_generated': len(pattern_results['code_suggestions'])
                }
        
        except Exception as e:
            pattern_results['error'] = str(e)
            pattern_results['confidence'] = 0.0
        
        return pattern_results
    
    def _is_optimization_pattern(self, pattern: Dict[str, Any]) -> bool:
        """Check if a pattern indicates compiler optimization"""
        pattern_type = pattern.get('type', '').lower()
        subtype = pattern.get('subtype', '').lower()
        
        # Patterns that typically indicate optimizations
        optimization_indicators = [
            'arithmetic',  # Arithmetic optimizations
            'control_flow',  # Control flow optimizations like loop unrolling
            'function_prologue',  # Function inlining indicators
        ]
        
        # Subtypes that indicate optimizations
        optimization_subtypes = [
            'loop',  # Loop optimizations
            'shift',  # Bit shift optimizations
            'multiplication',  # Arithmetic optimizations
        ]
        
        return (pattern_type in optimization_indicators or 
                subtype in optimization_subtypes or
                pattern.get('confidence', 0.0) > 0.8)  # High confidence patterns
    
    # Algorithm pattern recognition methods (Phase 3 enhancement)
    def _detect_algorithm_patterns(self, decompilation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect common algorithm patterns in decompiled code"""
        algorithm_patterns = []
        
        # Get function data for analysis
        functions = decompilation_data.get('decompiled_functions', {})
        
        for func_name, func_data in functions.items():
            if isinstance(func_data, dict):
                code = func_data.get('code', '')
                if code:
                    # Detect sorting algorithms
                    sorting_patterns = self._detect_sorting_algorithms(code, func_name)
                    algorithm_patterns.extend(sorting_patterns)
                    
                    # Detect search algorithms
                    search_patterns = self._detect_search_algorithms(code, func_name)
                    algorithm_patterns.extend(search_patterns)
                    
                    # Detect data structure operations
                    ds_patterns = self._detect_data_structure_patterns(code, func_name)
                    algorithm_patterns.extend(ds_patterns)
                    
                    # Detect mathematical algorithms
                    math_patterns = self._detect_mathematical_patterns(code, func_name)
                    algorithm_patterns.extend(math_patterns)
                    
                    # Detect string algorithms
                    string_patterns = self._detect_string_algorithms(code, func_name)
                    algorithm_patterns.extend(string_patterns)
        
        return algorithm_patterns
    
    def _detect_sorting_algorithms(self, code: str, func_name: str) -> List[Dict[str, Any]]:
        """Detect sorting algorithm patterns"""
        patterns = []
        code_lower = code.lower()
        
        # Bubble sort pattern
        if all(pattern in code_lower for pattern in ['for', 'swap', 'compare']):
            # Look for nested loops with swapping
            if code_lower.count('for') >= 2 or code_lower.count('while') >= 2:
                patterns.append({
                    'algorithm': 'bubble_sort',
                    'type': 'sorting',
                    'function': func_name,
                    'confidence': 0.7,
                    'indicators': ['nested_loops', 'swap_operations', 'comparison'],
                    'complexity': 'O(n²)',
                    'reconstruction_hint': 'Implement bubble sort with nested loops'
                })
        
        # Quick sort pattern
        if all(pattern in code_lower for pattern in ['partition', 'pivot', 'recursive']):
            patterns.append({
                'algorithm': 'quick_sort',
                'type': 'sorting',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['partitioning', 'pivot_selection', 'recursion'],
                'complexity': 'O(n log n)',
                'reconstruction_hint': 'Implement quicksort with partitioning'
            })
        
        # Merge sort pattern
        if all(pattern in code_lower for pattern in ['merge', 'divide', 'conquer']):
            patterns.append({
                'algorithm': 'merge_sort',
                'type': 'sorting',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['divide_and_conquer', 'merging', 'recursion'],
                'complexity': 'O(n log n)',
                'reconstruction_hint': 'Implement merge sort with divide and conquer'
            })
        
        # Heap sort pattern
        if all(pattern in code_lower for pattern in ['heap', 'heapify', 'extract']):
            patterns.append({
                'algorithm': 'heap_sort',
                'type': 'sorting',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['heap_structure', 'heapify_operations', 'extraction'],
                'complexity': 'O(n log n)',
                'reconstruction_hint': 'Implement heapsort with heap data structure'
            })
        
        return patterns
    
    def _detect_search_algorithms(self, code: str, func_name: str) -> List[Dict[str, Any]]:
        """Detect search algorithm patterns"""
        patterns = []
        code_lower = code.lower()
        
        # Binary search pattern
        if all(pattern in code_lower for pattern in ['mid', 'left', 'right']) and 'while' in code_lower:
            # Look for typical binary search structure
            if 'mid' in code_lower and ('left + right' in code_lower or 'low + high' in code_lower):
                patterns.append({
                    'algorithm': 'binary_search',
                    'type': 'search',
                    'function': func_name,
                    'confidence': 0.9,
                    'indicators': ['midpoint_calculation', 'range_narrowing', 'comparison'],
                    'complexity': 'O(log n)',
                    'reconstruction_hint': 'Implement binary search with left/right pointers'
                })
        
        # Linear search pattern
        if 'for' in code_lower and any(search_hint in code_lower for search_hint in ['find', 'search', 'locate']):
            if code_lower.count('for') == 1:  # Single loop
                patterns.append({
                    'algorithm': 'linear_search',
                    'type': 'search',
                    'function': func_name,
                    'confidence': 0.7,
                    'indicators': ['single_loop', 'sequential_access', 'comparison'],
                    'complexity': 'O(n)',
                    'reconstruction_hint': 'Implement linear search with sequential iteration'
                })
        
        # Hash table lookup
        if any(hash_hint in code_lower for hash_hint in ['hash', 'bucket', 'key']):
            patterns.append({
                'algorithm': 'hash_lookup',
                'type': 'search',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['hash_function', 'key_mapping', 'bucket_access'],
                'complexity': 'O(1) average',
                'reconstruction_hint': 'Implement hash table with key-value mapping'
            })
        
        # Depth-first search
        if all(pattern in code_lower for pattern in ['stack', 'visit', 'node']) or \
           ('recursive' in code_lower and 'explore' in code_lower):
            patterns.append({
                'algorithm': 'depth_first_search',
                'type': 'graph_search',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['stack_usage', 'node_visiting', 'recursive_exploration'],
                'complexity': 'O(V + E)',
                'reconstruction_hint': 'Implement DFS with stack or recursion'
            })
        
        # Breadth-first search
        if all(pattern in code_lower for pattern in ['queue', 'level', 'visit']):
            patterns.append({
                'algorithm': 'breadth_first_search',
                'type': 'graph_search',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['queue_usage', 'level_traversal', 'node_visiting'],
                'complexity': 'O(V + E)',
                'reconstruction_hint': 'Implement BFS with queue data structure'
            })
        
        return patterns
    
    def _detect_data_structure_patterns(self, code: str, func_name: str) -> List[Dict[str, Any]]:
        """Detect data structure algorithm patterns"""
        patterns = []
        code_lower = code.lower()
        
        # Stack operations
        if any(stack_op in code_lower for stack_op in ['push', 'pop', 'top', 'stack']):
            patterns.append({
                'algorithm': 'stack_operations',
                'type': 'data_structure',
                'function': func_name,
                'confidence': 0.9,
                'indicators': ['push_operation', 'pop_operation', 'top_access'],
                'complexity': 'O(1)',
                'reconstruction_hint': 'Implement stack with push/pop operations'
            })
        
        # Queue operations
        if any(queue_op in code_lower for queue_op in ['enqueue', 'dequeue', 'front', 'rear', 'queue']):
            patterns.append({
                'algorithm': 'queue_operations',
                'type': 'data_structure',
                'function': func_name,
                'confidence': 0.9,
                'indicators': ['enqueue_operation', 'dequeue_operation', 'front_access'],
                'complexity': 'O(1)',
                'reconstruction_hint': 'Implement queue with enqueue/dequeue operations'
            })
        
        # Linked list operations
        if any(list_op in code_lower for list_op in ['next', 'node', 'link', 'insert', 'delete']):
            patterns.append({
                'algorithm': 'linked_list_operations',
                'type': 'data_structure',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['node_traversal', 'pointer_manipulation', 'insertion_deletion'],
                'complexity': 'O(n)',
                'reconstruction_hint': 'Implement linked list with node structure'
            })
        
        # Tree operations
        if any(tree_op in code_lower for tree_op in ['left', 'right', 'parent', 'child', 'tree', 'node']):
            if 'left' in code_lower and 'right' in code_lower:
                patterns.append({
                    'algorithm': 'binary_tree_operations',
                    'type': 'data_structure',
                    'function': func_name,
                    'confidence': 0.8,
                    'indicators': ['left_child', 'right_child', 'tree_traversal'],
                    'complexity': 'O(h)',  # h = height
                    'reconstruction_hint': 'Implement binary tree with left/right pointers'
                })
        
        # Graph operations
        if any(graph_op in code_lower for graph_op in ['vertex', 'edge', 'graph', 'adjacency', 'neighbor']):
            patterns.append({
                'algorithm': 'graph_operations',
                'type': 'data_structure',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['vertex_access', 'edge_traversal', 'adjacency_structure'],
                'complexity': 'varies',
                'reconstruction_hint': 'Implement graph with adjacency list/matrix'
            })
        
        return patterns
    
    def _detect_mathematical_patterns(self, code: str, func_name: str) -> List[Dict[str, Any]]:
        """Detect mathematical algorithm patterns"""
        patterns = []
        code_lower = code.lower()
        
        # Greatest Common Divisor (Euclidean algorithm)
        if all(pattern in code_lower for pattern in ['gcd', 'mod', 'remainder']) or \
           ('while' in code_lower and 'mod' in code_lower):
            patterns.append({
                'algorithm': 'euclidean_gcd',
                'type': 'mathematical',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['modulo_operation', 'while_loop', 'remainder_calculation'],
                'complexity': 'O(log min(a,b))',
                'reconstruction_hint': 'Implement Euclidean algorithm for GCD'
            })
        
        # Prime number checking
        if any(prime_hint in code_lower for prime_hint in ['prime', 'factor', 'divisible']):
            if 'for' in code_lower and ('mod' in code_lower or '%' in code):
                patterns.append({
                    'algorithm': 'prime_check',
                    'type': 'mathematical',
                    'function': func_name,
                    'confidence': 0.7,
                    'indicators': ['divisibility_test', 'factor_checking', 'loop_iteration'],
                    'complexity': 'O(√n)',
                    'reconstruction_hint': 'Implement prime checking with trial division'
                })
        
        # Fibonacci sequence
        if 'fibonacci' in code_lower or \
           (('fib' in code_lower) and ('recursive' in code_lower or 'previous' in code_lower)):
            patterns.append({
                'algorithm': 'fibonacci',
                'type': 'mathematical',
                'function': func_name,
                'confidence': 0.9,
                'indicators': ['recursive_relation', 'sequence_generation', 'previous_values'],
                'complexity': 'O(n) or O(2^n)',
                'reconstruction_hint': 'Implement Fibonacci with recursion or iteration'
            })
        
        # Matrix operations
        if any(matrix_hint in code_lower for matrix_hint in ['matrix', 'multiply', 'transpose', 'determinant']):
            patterns.append({
                'algorithm': 'matrix_operations',
                'type': 'mathematical',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['matrix_access', 'nested_loops', 'mathematical_operations'],
                'complexity': 'O(n³) for multiplication',
                'reconstruction_hint': 'Implement matrix operations with nested loops'
            })
        
        # Fast Fourier Transform
        if any(fft_hint in code_lower for fft_hint in ['fft', 'fourier', 'transform', 'frequency']):
            patterns.append({
                'algorithm': 'fast_fourier_transform',
                'type': 'mathematical',
                'function': func_name,
                'confidence': 0.9,
                'indicators': ['frequency_analysis', 'complex_numbers', 'divide_conquer'],
                'complexity': 'O(n log n)',
                'reconstruction_hint': 'Implement FFT with divide and conquer'
            })
        
        return patterns
    
    def _detect_string_algorithms(self, code: str, func_name: str) -> List[Dict[str, Any]]:
        """Detect string algorithm patterns"""
        patterns = []
        code_lower = code.lower()
        
        # String matching (KMP algorithm)
        if any(kmp_hint in code_lower for kmp_hint in ['kmp', 'pattern', 'match', 'needle', 'haystack']):
            if 'prefix' in code_lower or 'partial' in code_lower:
                patterns.append({
                    'algorithm': 'kmp_string_matching',
                    'type': 'string',
                    'function': func_name,
                    'confidence': 0.8,
                    'indicators': ['pattern_matching', 'prefix_function', 'partial_match'],
                    'complexity': 'O(n + m)',
                    'reconstruction_hint': 'Implement KMP with prefix function'
                })
        
        # String searching (Boyer-Moore)
        if any(bm_hint in code_lower for bm_hint in ['boyer', 'moore', 'bad_character', 'good_suffix']):
            patterns.append({
                'algorithm': 'boyer_moore_search',
                'type': 'string',
                'function': func_name,
                'confidence': 0.9,
                'indicators': ['bad_character_rule', 'good_suffix_rule', 'skip_table'],
                'complexity': 'O(nm) worst, O(n/m) best',
                'reconstruction_hint': 'Implement Boyer-Moore with skip tables'
            })
        
        # Edit distance (Levenshtein)
        if any(edit_hint in code_lower for edit_hint in ['edit', 'distance', 'levenshtein', 'dp']):
            if 'dynamic' in code_lower or '2d' in code_lower or 'table' in code_lower:
                patterns.append({
                    'algorithm': 'edit_distance',
                    'type': 'string',
                    'function': func_name,
                    'confidence': 0.8,
                    'indicators': ['dynamic_programming', '2d_table', 'edit_operations'],
                    'complexity': 'O(nm)',
                    'reconstruction_hint': 'Implement edit distance with DP table'
                })
        
        # Longest Common Subsequence
        if any(lcs_hint in code_lower for lcs_hint in ['lcs', 'subsequence', 'common', 'longest']):
            patterns.append({
                'algorithm': 'longest_common_subsequence',
                'type': 'string',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['subsequence_matching', 'dynamic_programming', 'optimal_substructure'],
                'complexity': 'O(nm)',
                'reconstruction_hint': 'Implement LCS with dynamic programming'
            })
        
        # String hashing (Rolling hash)
        if any(hash_hint in code_lower for hash_hint in ['rolling', 'hash', 'polynomial', 'rabin']):
            patterns.append({
                'algorithm': 'rolling_hash',
                'type': 'string',
                'function': func_name,
                'confidence': 0.8,
                'indicators': ['polynomial_hashing', 'rolling_window', 'hash_update'],
                'complexity': 'O(n)',
                'reconstruction_hint': 'Implement rolling hash with polynomial function'
            })
        
        return patterns
    
    def _generate_algorithm_reconstruction_strategies(self, algorithm_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate strategies for reconstructing detected algorithms"""
        strategies = {
            'reconstruction_plan': [],
            'implementation_order': [],
            'difficulty_assessment': {},
            'recommended_tools': [],
            'expected_success_rate': 0.0
        }
        
        if not algorithm_patterns:
            return strategies
        
        # Group patterns by type
        pattern_types = {}
        for pattern in algorithm_patterns:
            pattern_type = pattern.get('type', 'unknown')
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = []
            pattern_types[pattern_type].append(pattern)
        
        # Generate reconstruction plan for each type
        difficulty_scores = []
        for pattern_type, patterns in pattern_types.items():
            type_strategy = self._create_type_reconstruction_strategy(pattern_type, patterns)
            strategies['reconstruction_plan'].append(type_strategy)
            difficulty_scores.append(type_strategy['difficulty_score'])
        
        # Order by implementation difficulty (easiest first)
        strategies['reconstruction_plan'].sort(key=lambda x: x['difficulty_score'])
        strategies['implementation_order'] = [plan['type'] for plan in strategies['reconstruction_plan']]
        
        # Overall difficulty assessment
        if difficulty_scores:
            avg_difficulty = sum(difficulty_scores) / len(difficulty_scores)
            strategies['difficulty_assessment'] = {
                'average_difficulty': avg_difficulty,
                'easiest_algorithms': [p for p in algorithm_patterns if p.get('confidence', 0) > 0.8],
                'most_challenging': [p for p in algorithm_patterns if p.get('confidence', 0) < 0.6]
            }
            
            # Expected success rate based on confidence and difficulty
            confidence_scores = [p.get('confidence', 0) for p in algorithm_patterns]
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
            strategies['expected_success_rate'] = avg_confidence * (1.0 - avg_difficulty * 0.3)
        
        # Recommended tools
        strategies['recommended_tools'] = [
            'static_analysis_framework',
            'pattern_matching_library',
            'algorithm_template_database',
            'code_generation_tools',
            'complexity_analyzer'
        ]
        
        return strategies
    
    def _create_type_reconstruction_strategy(self, pattern_type: str, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create reconstruction strategy for a specific algorithm type"""
        strategy = {
            'type': pattern_type,
            'patterns_count': len(patterns),
            'difficulty_score': 0.0,
            'approach': 'unknown',
            'steps': [],
            'tools_needed': [],
            'time_estimate': 'unknown'
        }
        
        # Calculate difficulty based on pattern type and complexity
        complexity_map = {
            'sorting': 0.3,      # Generally well-understood
            'search': 0.2,       # Straightforward patterns
            'data_structure': 0.4, # More complex state management
            'mathematical': 0.6,   # Often algorithm-specific
            'string': 0.5,        # Moderate complexity
            'graph_search': 0.7   # Complex state and traversal
        }
        
        base_difficulty = complexity_map.get(pattern_type, 0.5)
        
        # Adjust based on pattern confidence
        avg_confidence = sum(p.get('confidence', 0) for p in patterns) / len(patterns)
        difficulty_adjustment = (1.0 - avg_confidence) * 0.3
        strategy['difficulty_score'] = min(base_difficulty + difficulty_adjustment, 1.0)
        
        # Define approach based on type
        if pattern_type == 'sorting':
            strategy['approach'] = 'template_matching'
            strategy['steps'] = [
                'Identify sorting pattern characteristics',
                'Match against known sorting algorithm templates',
                'Reconstruct comparison and swap operations',
                'Generate optimized sorting implementation'
            ]
            strategy['tools_needed'] = ['algorithm_templates', 'pattern_matcher']
            strategy['time_estimate'] = '2-4 hours'
            
        elif pattern_type == 'search':
            strategy['approach'] = 'control_flow_analysis'
            strategy['steps'] = [
                'Analyze loop structure and termination conditions',
                'Identify search space partitioning logic',
                'Reconstruct comparison and indexing operations',
                'Generate search algorithm implementation'
            ]
            strategy['tools_needed'] = ['control_flow_analyzer', 'loop_reconstructor']
            strategy['time_estimate'] = '1-3 hours'
            
        elif pattern_type == 'data_structure':
            strategy['approach'] = 'state_machine_reconstruction'
            strategy['steps'] = [
                'Identify data structure operations',
                'Reconstruct state transitions',
                'Generate data structure interface',
                'Implement underlying storage mechanism'
            ]
            strategy['tools_needed'] = ['state_analyzer', 'interface_generator']
            strategy['time_estimate'] = '4-8 hours'
            
        elif pattern_type == 'mathematical':
            strategy['approach'] = 'mathematical_analysis'
            strategy['steps'] = [
                'Identify mathematical operations and formulas',
                'Analyze numerical computation patterns',
                'Reconstruct algorithm mathematical basis',
                'Generate optimized mathematical implementation'
            ]
            strategy['tools_needed'] = ['mathematical_analyzer', 'formula_recognizer']
            strategy['time_estimate'] = '6-12 hours'
            
        elif pattern_type == 'string':
            strategy['approach'] = 'pattern_based_reconstruction'
            strategy['steps'] = [
                'Identify string processing patterns',
                'Analyze character-level operations',
                'Reconstruct string algorithm logic',
                'Generate string processing implementation'
            ]
            strategy['tools_needed'] = ['string_analyzer', 'pattern_library']
            strategy['time_estimate'] = '3-6 hours'
            
        else:
            strategy['approach'] = 'general_analysis'
            strategy['steps'] = [
                'Perform detailed code analysis',
                'Identify algorithm characteristics',
                'Match against algorithm database',
                'Generate implementation'
            ]
            strategy['tools_needed'] = ['general_analyzer']
            strategy['time_estimate'] = '4-8 hours'
        
        return strategy