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
    from ...ml.pattern_engine import PatternEngine
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
        raise NotImplementedError(
            "Tail call pattern detection not implemented - requires detailed "
            "assembly analysis to detect call-followed-by-return patterns"
        )

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