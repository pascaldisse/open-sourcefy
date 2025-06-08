"""
AI Enhancement Module for Open-Sourcefy
Provides machine learning and AI-powered capabilities for binary analysis and code reconstruction.
"""

import re
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib


class MLModelType(Enum):
    PATTERN_RECOGNITION = "pattern_recognition"
    FUNCTION_NAMING = "function_naming"
    OPTIMIZATION_DETECTION = "optimization_detection"
    CODE_QUALITY_ASSESSMENT = "code_quality_assessment"
    CONTEXT_ANALYSIS = "context_analysis"


@dataclass
class AIAnalysisResult:
    """Result from AI analysis"""
    model_type: MLModelType
    confidence: float
    prediction: Any
    evidence: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'model_type': self.model_type.value,
            'confidence': self.confidence,
            'prediction': self.prediction,
            'evidence': self.evidence,
            'metadata': self.metadata
        }


class AIPatternRecognizer:
    """Enhanced pattern recognition using AI techniques"""
    
    def __init__(self):
        self.logger = logging.getLogger("AIPatternRecognizer")
        self.pattern_database = self._initialize_pattern_database()
        self.feature_extractors = self._initialize_feature_extractors()
    
    def _initialize_pattern_database(self) -> Dict[str, Any]:
        """Initialize comprehensive pattern database"""
        return {
            'function_patterns': {
                'constructor_patterns': [
                    r'mov.*esp.*ebp',  # Function prologue
                    r'push.*ebp',      # Stack frame setup
                    r'call.*\$.*init'   # Constructor calls
                ],
                'destructor_patterns': [
                    r'call.*\$.*destroy',
                    r'free.*\[.*\]',
                    r'delete.*operator'
                ],
                'getter_setter_patterns': [
                    r'mov.*\[.*\].*eax',  # Simple field access
                    r'return.*field',
                    r'this\+\d+'
                ]
            },
            'optimization_signatures': {
                'vectorization': [
                    r'xmm\d+',         # SSE registers
                    r'ymm\d+',         # AVX registers
                    r'packed.*operation',
                    r'simd.*instruction'
                ],
                'loop_optimizations': [
                    r'unroll.*\d+',
                    r'vectorize.*loop',
                    r'parallel.*execution'
                ],
                'inlining_signatures': [
                    r'inline.*expansion',
                    r'duplicate.*basic_block',
                    r'eliminated.*call'
                ]
            },
            'compiler_signatures': {
                'gcc_patterns': [
                    r'__gxx_personality',
                    r'_GLOBAL_OFFSET_TABLE_',
                    r'\.eh_frame'
                ],
                'msvc_patterns': [
                    r'__security_cookie',
                    r'__chkesp',
                    r'_RTC_CheckEsp'
                ],
                'clang_patterns': [
                    r'__cxx_global_var_init',
                    r'llvm\..*\.init',
                    r'__clang_call_terminate'
                ]
            }
        }
    
    def _initialize_feature_extractors(self) -> Dict[str, callable]:
        """Initialize feature extraction functions"""
        return {
            'instruction_frequency': self._extract_instruction_frequency,
            'register_usage': self._extract_register_usage,
            'memory_patterns': self._extract_memory_patterns,
            'control_flow': self._extract_control_flow_features,
            'string_features': self._extract_string_features
        }
    
    def analyze_patterns(self, code_data: Dict[str, Any], context: Dict[str, Any]) -> AIAnalysisResult:
        """Perform AI-enhanced pattern analysis"""
        features = self._extract_all_features(code_data)
        patterns = self._recognize_patterns(features)
        confidence = self._calculate_pattern_confidence(patterns, features)
        
        return AIAnalysisResult(
            model_type=MLModelType.PATTERN_RECOGNITION,
            confidence=confidence,
            prediction=patterns,
            evidence=self._generate_pattern_evidence(patterns, features),
            metadata={
                'feature_count': len(features),
                'pattern_types': list(patterns.keys()),
                'analysis_depth': 'enhanced'
            }
        )
    
    def _extract_all_features(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from code data"""
        features = {}
        
        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                features[extractor_name] = extractor_func(code_data)
            except Exception as e:
                self.logger.warning(f"Feature extraction failed for {extractor_name}: {e}")
                features[extractor_name] = {}
        
        return features
    
    def _extract_instruction_frequency(self, code_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract instruction frequency features"""
        freq = {}
        
        # Analyze objdump output if available
        objdump_data = code_data.get('objdump_output', {})
        if objdump_data and 'functions' in objdump_data:
            for func in objdump_data['functions']:
                instructions = func.get('instructions', [])
                for inst in instructions:
                    # Extract instruction mnemonic
                    parts = inst.strip().split()
                    if len(parts) > 1:
                        mnemonic = parts[1]  # Skip address
                        freq[mnemonic] = freq.get(mnemonic, 0) + 1
        
        return freq
    
    def _extract_register_usage(self, code_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract register usage patterns"""
        registers = {}
        
        objdump_data = code_data.get('objdump_output', {})
        if objdump_data and 'functions' in objdump_data:
            for func in objdump_data['functions']:
                instructions = func.get('instructions', [])
                for inst in instructions:
                    # Find register references
                    reg_matches = re.findall(r'%[a-z]+\d*', inst)
                    for reg in reg_matches:
                        registers[reg] = registers.get(reg, 0) + 1
        
        return registers
    
    def _extract_memory_patterns(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory access patterns"""
        patterns = {
            'stack_access': 0,
            'heap_access': 0,
            'global_access': 0,
            'indirect_access': 0
        }
        
        objdump_data = code_data.get('objdump_output', {})
        if objdump_data and 'functions' in objdump_data:
            for func in objdump_data['functions']:
                instructions = func.get('instructions', [])
                for inst in instructions:
                    if '%esp' in inst or '%ebp' in inst:
                        patterns['stack_access'] += 1
                    elif '[' in inst and ']' in inst:
                        patterns['indirect_access'] += 1
                    elif 'malloc' in inst or 'free' in inst:
                        patterns['heap_access'] += 1
        
        return patterns
    
    def _extract_control_flow_features(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract control flow features"""
        features = {
            'conditional_jumps': 0,
            'unconditional_jumps': 0,
            'function_calls': 0,
            'returns': 0,
            'loops_detected': 0
        }
        
        objdump_data = code_data.get('objdump_output', {})
        if objdump_data and 'functions' in objdump_data:
            for func in objdump_data['functions']:
                instructions = func.get('instructions', [])
                for inst in instructions:
                    if re.search(r'j[a-z]+', inst):  # Jump instructions
                        if 'jmp' in inst:
                            features['unconditional_jumps'] += 1
                        else:
                            features['conditional_jumps'] += 1
                    elif 'call' in inst:
                        features['function_calls'] += 1
                    elif 'ret' in inst:
                        features['returns'] += 1
        
        return features
    
    def _extract_string_features(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract string-based features"""
        features = {
            'total_strings': 0,
            'format_strings': 0,
            'error_messages': 0,
            'debug_strings': 0,
            'avg_string_length': 0
        }
        
        strings = code_data.get('strings', [])
        if strings:
            features['total_strings'] = len(strings)
            
            total_length = 0
            for s in strings:
                total_length += len(s)
                if '%' in s:
                    features['format_strings'] += 1
                if any(keyword in s.lower() for keyword in ['error', 'fail', 'exception']):
                    features['error_messages'] += 1
                if any(keyword in s.lower() for keyword in ['debug', 'trace', 'log']):
                    features['debug_strings'] += 1
            
            features['avg_string_length'] = total_length / len(strings) if strings else 0
        
        return features
    
    def _recognize_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize high-level patterns from features"""
        patterns = {}
        
        # Analyze function patterns
        patterns['function_analysis'] = self._analyze_function_patterns(features)
        
        # Analyze optimization patterns
        patterns['optimization_analysis'] = self._analyze_optimization_patterns(features)
        
        # Analyze compiler patterns
        patterns['compiler_analysis'] = self._analyze_compiler_patterns(features)
        
        # Analyze architectural patterns
        patterns['architecture_analysis'] = self._analyze_architecture_patterns(features)
        
        return patterns
    
    def _analyze_function_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function-level patterns"""
        control_flow = features.get('control_flow', {})
        
        analysis = {
            'complexity_estimate': 'low',
            'function_type_hints': [],
            'programming_style': 'unknown'
        }
        
        # Estimate complexity based on control flow
        total_jumps = control_flow.get('conditional_jumps', 0) + control_flow.get('unconditional_jumps', 0)
        if total_jumps > 20:
            analysis['complexity_estimate'] = 'high'
        elif total_jumps > 10:
            analysis['complexity_estimate'] = 'medium'
        
        # Analyze function type hints
        calls = control_flow.get('function_calls', 0)
        returns = control_flow.get('returns', 0)
        
        if calls > returns * 2:
            analysis['function_type_hints'].append('utility_function')
        elif returns > calls:
            analysis['function_type_hints'].append('simple_function')
        
        return analysis
    
    def _analyze_optimization_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization patterns"""
        inst_freq = features.get('instruction_frequency', {})
        reg_usage = features.get('register_usage', {})
        
        analysis = {
            'optimization_level_estimate': 'O0',
            'detected_optimizations': [],
            'vectorization_hints': []
        }
        
        # Check for optimization indicators
        total_instructions = sum(inst_freq.values()) if inst_freq else 0
        unique_instructions = len(inst_freq) if inst_freq else 0
        
        if total_instructions > 0:
            instruction_diversity = unique_instructions / total_instructions
            
            if instruction_diversity < 0.1:  # Low diversity might indicate optimization
                analysis['optimization_level_estimate'] = 'O2'
                analysis['detected_optimizations'].append('instruction_specialization')
        
        # Check for vectorization
        vector_regs = [reg for reg in reg_usage.keys() if 'xmm' in reg or 'ymm' in reg]
        if vector_regs:
            analysis['vectorization_hints'].append('vector_registers_detected')
            analysis['detected_optimizations'].append('vectorization')
        
        return analysis
    
    def _analyze_compiler_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compiler-specific patterns"""
        strings = features.get('string_features', {})
        
        analysis = {
            'likely_compiler': 'unknown',
            'compiler_version_hints': [],
            'compilation_flags': []
        }
        
        # This would normally use more sophisticated analysis
        # For now, provide basic heuristics
        if strings.get('debug_strings', 0) > 0:
            analysis['compilation_flags'].append('-g')
        
        return analysis
    
    def _analyze_architecture_patterns(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze architectural patterns"""
        reg_usage = features.get('register_usage', {})
        memory_patterns = features.get('memory_patterns', {})
        
        analysis = {
            'architecture_hints': [],
            'calling_convention': 'unknown',
            'pointer_size': 'unknown'
        }
        
        # Analyze register usage for architecture hints
        if any('rax' in reg or 'rbx' in reg for reg in reg_usage.keys()):
            analysis['architecture_hints'].append('x64')
            analysis['pointer_size'] = '64-bit'
        elif any('eax' in reg or 'ebx' in reg for reg in reg_usage.keys()):
            analysis['architecture_hints'].append('x86')
            analysis['pointer_size'] = '32-bit'
        
        # Analyze calling convention hints
        stack_access = memory_patterns.get('stack_access', 0)
        if stack_access > 0:
            analysis['calling_convention'] = 'likely_cdecl_or_stdcall'
        
        return analysis
    
    def _calculate_pattern_confidence(self, patterns: Dict[str, Any], features: Dict[str, Any]) -> float:
        """Calculate confidence score for pattern recognition"""
        # Simple confidence calculation based on feature completeness
        feature_completeness = len([v for v in features.values() if v]) / len(features)
        pattern_strength = len([p for p in patterns.values() if p]) / len(patterns)
        
        return (feature_completeness + pattern_strength) / 2
    
    def _generate_pattern_evidence(self, patterns: Dict[str, Any], features: Dict[str, Any]) -> List[str]:
        """Generate evidence for detected patterns"""
        evidence = []
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_data:
                evidence.append(f"Detected {pattern_type} with {len(pattern_data)} indicators")
        
        return evidence


class AIFunctionNamer:
    """AI-powered function naming system"""
    
    def __init__(self):
        self.logger = logging.getLogger("AIFunctionNamer")
        self.naming_patterns = self._initialize_naming_patterns()
    
    def _initialize_naming_patterns(self) -> Dict[str, Any]:
        """Initialize function naming patterns"""
        return {
            'naming_conventions': {
                'camelCase': r'^[a-z][a-zA-Z0-9]*$',
                'snake_case': r'^[a-z][a-z0-9_]*$',
                'PascalCase': r'^[A-Z][a-zA-Z0-9]*$',
                'hungarian': r'^[a-z][A-Z][a-zA-Z0-9]*$'
            },
            'function_type_patterns': {
                'getter': ['get', 'retrieve', 'fetch', 'obtain'],
                'setter': ['set', 'update', 'assign', 'configure'],
                'validator': ['is', 'check', 'validate', 'verify'],
                'constructor': ['init', 'create', 'new', 'construct'],
                'destructor': ['destroy', 'cleanup', 'delete', 'free'],
                'utility': ['util', 'helper', 'process', 'handle']
            },
            'semantic_indicators': {
                'math_operations': ['add', 'sub', 'mul', 'div', 'calc', 'compute'],
                'string_operations': ['concat', 'split', 'trim', 'format', 'parse'],
                'file_operations': ['read', 'write', 'open', 'close', 'save', 'load'],
                'network_operations': ['connect', 'send', 'receive', 'download', 'upload'],
                'memory_operations': ['alloc', 'malloc', 'free', 'copy', 'move']
            }
        }
    
    def suggest_function_names(self, function_data: Dict[str, Any], context: Dict[str, Any]) -> AIAnalysisResult:
        """Suggest intelligent function names based on analysis"""
        suggestions = self._generate_name_suggestions(function_data, context)
        confidence = self._calculate_naming_confidence(suggestions, function_data)
        
        return AIAnalysisResult(
            model_type=MLModelType.FUNCTION_NAMING,
            confidence=confidence,
            prediction=suggestions,
            evidence=self._generate_naming_evidence(suggestions, function_data),
            metadata={
                'original_name': function_data.get('name', 'unknown'),
                'suggestion_count': len(suggestions),
                'naming_strategy': 'semantic_analysis'
            }
        )
    
    def _generate_name_suggestions(self, function_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate intelligent name suggestions"""
        suggestions = []
        
        # Analyze function characteristics
        characteristics = self._analyze_function_characteristics(function_data)
        
        # Generate suggestions based on characteristics
        for char_type, indicators in characteristics.items():
            if indicators:
                suggestions.extend(self._generate_names_for_type(char_type, indicators))
        
        # Rank suggestions by relevance
        suggestions = sorted(suggestions, key=lambda x: x['confidence'], reverse=True)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _analyze_function_characteristics(self, function_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze function characteristics for naming"""
        characteristics = {
            'operations': [],
            'data_types': [],
            'patterns': [],
            'complexity': []
        }
        
        # Analyze function instructions or decompiled code
        instructions = function_data.get('instructions', [])
        
        for inst in instructions:
            # Look for operation indicators
            if any(op in inst.lower() for op in ['add', 'sub', 'mul', 'div']):
                characteristics['operations'].append('mathematical')
            elif any(op in inst.lower() for op in ['mov', 'copy', 'store']):
                characteristics['operations'].append('data_movement')
            elif any(op in inst.lower() for op in ['cmp', 'test', 'je', 'jne']):
                characteristics['operations'].append('comparison')
            elif 'call' in inst.lower():
                characteristics['operations'].append('function_call')
        
        return characteristics
    
    def _generate_names_for_type(self, char_type: str, indicators: List[str]) -> List[Dict[str, Any]]:
        """Generate names for specific characteristic type"""
        suggestions = []
        
        if char_type == 'operations':
            if 'mathematical' in indicators:
                suggestions.append({
                    'name': 'calculateValue',
                    'confidence': 0.8,
                    'reasoning': 'Function performs mathematical operations'
                })
            elif 'data_movement' in indicators:
                suggestions.append({
                    'name': 'processData',
                    'confidence': 0.7,
                    'reasoning': 'Function moves or processes data'
                })
            elif 'comparison' in indicators:
                suggestions.append({
                    'name': 'compareValues',
                    'confidence': 0.75,
                    'reasoning': 'Function performs comparisons'
                })
        
        return suggestions
    
    def _calculate_naming_confidence(self, suggestions: List[Dict[str, Any]], function_data: Dict[str, Any]) -> float:
        """Calculate confidence in naming suggestions"""
        if not suggestions:
            return 0.0
        
        # Average confidence of top suggestions
        top_confidences = [s['confidence'] for s in suggestions[:3]]
        return sum(top_confidences) / len(top_confidences)
    
    def _generate_naming_evidence(self, suggestions: List[Dict[str, Any]], function_data: Dict[str, Any]) -> List[str]:
        """Generate evidence for naming suggestions"""
        evidence = []
        
        for suggestion in suggestions[:3]:  # Top 3 suggestions
            evidence.append(f"Suggested '{suggestion['name']}': {suggestion['reasoning']}")
        
        return evidence


class AICodeQualityAssessor:
    """AI-powered code quality assessment"""
    
    def __init__(self):
        self.logger = logging.getLogger("AICodeQualityAssessor")
        self.quality_metrics = self._initialize_quality_metrics()
    
    def _initialize_quality_metrics(self) -> Dict[str, Any]:
        """Initialize code quality metrics"""
        return {
            'complexity_metrics': {
                'cyclomatic_complexity': 'function_paths',
                'cognitive_complexity': 'mental_load',
                'nesting_depth': 'indentation_levels'
            },
            'maintainability_metrics': {
                'function_length': 'lines_of_code',
                'parameter_count': 'function_parameters',
                'variable_naming': 'identifier_quality'
            },
            'performance_metrics': {
                'memory_efficiency': 'memory_usage_patterns',
                'algorithmic_efficiency': 'time_complexity',
                'optimization_opportunities': 'improvement_potential'
            }
        }
    
    def assess_code_quality(self, code_data: Dict[str, Any], context: Dict[str, Any]) -> AIAnalysisResult:
        """Assess code quality using AI techniques"""
        assessment = self._perform_quality_assessment(code_data)
        confidence = self._calculate_assessment_confidence(assessment)
        
        return AIAnalysisResult(
            model_type=MLModelType.CODE_QUALITY_ASSESSMENT,
            confidence=confidence,
            prediction=assessment,
            evidence=self._generate_quality_evidence(assessment),
            metadata={
                'assessment_categories': list(assessment.keys()),
                'overall_score': assessment.get('overall_score', 0),
                'improvement_potential': assessment.get('improvement_potential', 'unknown')
            }
        )
    
    def _perform_quality_assessment(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive code quality assessment"""
        assessment = {
            'complexity_analysis': self._assess_complexity(code_data),
            'maintainability_analysis': self._assess_maintainability(code_data),
            'performance_analysis': self._assess_performance(code_data),
            'overall_score': 0.0,
            'improvement_suggestions': []
        }
        
        # Calculate overall score
        scores = [
            assessment['complexity_analysis'].get('score', 0),
            assessment['maintainability_analysis'].get('score', 0),
            assessment['performance_analysis'].get('score', 0)
        ]
        assessment['overall_score'] = sum(scores) / len(scores) if scores else 0
        
        # Generate improvement suggestions
        assessment['improvement_suggestions'] = self._generate_improvement_suggestions(assessment)
        
        return assessment
    
    def _assess_complexity(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess code complexity"""
        analysis = {
            'cyclomatic_complexity': 1,  # Base complexity
            'cognitive_complexity': 0,
            'nesting_depth': 0,
            'score': 0.8  # Default good score
        }
        
        # Analyze objdump data for complexity indicators
        objdump_data = code_data.get('objdump_output', {})
        if objdump_data and 'functions' in objdump_data:
            total_complexity = 0
            function_count = 0
            
            for func in objdump_data['functions']:
                instructions = func.get('instructions', [])
                
                # Count conditional branches for cyclomatic complexity
                branches = sum(1 for inst in instructions if re.search(r'j[^m]', inst))
                complexity = branches + 1
                total_complexity += complexity
                function_count += 1
            
            if function_count > 0:
                avg_complexity = total_complexity / function_count
                analysis['cyclomatic_complexity'] = avg_complexity
                
                # Score based on complexity (lower is better)
                if avg_complexity <= 5:
                    analysis['score'] = 0.9
                elif avg_complexity <= 10:
                    analysis['score'] = 0.7
                elif avg_complexity <= 20:
                    analysis['score'] = 0.5
                else:
                    analysis['score'] = 0.3
        
        return analysis
    
    def _assess_maintainability(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess code maintainability"""
        analysis = {
            'function_length_score': 0.8,
            'naming_quality_score': 0.7,
            'documentation_score': 0.5,
            'score': 0.0
        }
        
        # Analyze function characteristics
        functions = code_data.get('decompiled_functions', {})
        if functions:
            total_score = 0
            for func_name, func_data in functions.items():
                # Assess function name quality
                name_score = self._assess_function_name_quality(func_name)
                total_score += name_score
            
            analysis['naming_quality_score'] = total_score / len(functions) if functions else 0.5
        
        # Calculate overall maintainability score
        scores = [
            analysis['function_length_score'],
            analysis['naming_quality_score'],
            analysis['documentation_score']
        ]
        analysis['score'] = sum(scores) / len(scores)
        
        return analysis
    
    def _assess_performance(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess performance characteristics"""
        analysis = {
            'memory_efficiency_score': 0.7,
            'algorithmic_efficiency_score': 0.8,
            'optimization_score': 0.6,
            'score': 0.0
        }
        
        # Analyze optimization data if available
        optimization_data = code_data.get('optimization_analysis')
        if optimization_data:
            opt_level = optimization_data.get('optimization_level', 'O0')
            if opt_level == 'O3':
                analysis['optimization_score'] = 0.9
            elif opt_level == 'O2':
                analysis['optimization_score'] = 0.8
            elif opt_level == 'O1':
                analysis['optimization_score'] = 0.6
            else:
                analysis['optimization_score'] = 0.4
        
        # Calculate overall performance score
        scores = [
            analysis['memory_efficiency_score'],
            analysis['algorithmic_efficiency_score'],
            analysis['optimization_score']
        ]
        analysis['score'] = sum(scores) / len(scores)
        
        return analysis
    
    def _assess_function_name_quality(self, func_name: str) -> float:
        """Assess quality of function name"""
        if not func_name or func_name.startswith('sub_') or func_name.startswith('func_'):
            return 0.2  # Poor auto-generated name
        
        # Check for meaningful naming conventions
        if re.match(r'^[a-z][a-zA-Z0-9_]*[a-zA-Z0-9]$', func_name):
            return 0.8  # Good naming convention
        elif len(func_name) > 3:
            return 0.6  # Reasonable name
        else:
            return 0.4  # Short/unclear name
    
    def _generate_improvement_suggestions(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on assessment"""
        suggestions = []
        
        complexity = assessment.get('complexity_analysis', {})
        if complexity.get('score', 1) < 0.6:
            suggestions.append("Reduce function complexity by breaking down large functions")
        
        maintainability = assessment.get('maintainability_analysis', {})
        if maintainability.get('naming_quality_score', 1) < 0.6:
            suggestions.append("Improve function and variable naming conventions")
        
        performance = assessment.get('performance_analysis', {})
        if performance.get('optimization_score', 1) < 0.7:
            suggestions.append("Consider applying compiler optimizations")
        
        return suggestions
    
    def _calculate_assessment_confidence(self, assessment: Dict[str, Any]) -> float:
        """Calculate confidence in quality assessment"""
        # Base confidence on data availability and score consistency
        data_completeness = len([v for v in assessment.values() if v]) / len(assessment)
        
        scores = [
            assessment.get('complexity_analysis', {}).get('score', 0),
            assessment.get('maintainability_analysis', {}).get('score', 0),
            assessment.get('performance_analysis', {}).get('score', 0)
        ]
        
        score_variance = max(scores) - min(scores) if scores else 0
        consistency = 1 - score_variance  # Lower variance = higher consistency
        
        return (data_completeness + consistency) / 2
    
    def _generate_quality_evidence(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate evidence for quality assessment"""
        evidence = []
        
        overall_score = assessment.get('overall_score', 0)
        evidence.append(f"Overall quality score: {overall_score:.2f}")
        
        complexity_score = assessment.get('complexity_analysis', {}).get('score', 0)
        evidence.append(f"Complexity assessment: {complexity_score:.2f}")
        
        maintainability_score = assessment.get('maintainability_analysis', {}).get('score', 0)
        evidence.append(f"Maintainability assessment: {maintainability_score:.2f}")
        
        suggestions = assessment.get('improvement_suggestions', [])
        if suggestions:
            evidence.append(f"Found {len(suggestions)} improvement opportunities")
        
        return evidence


class AIEnhancementCoordinator:
    """Coordinates all AI enhancement capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger("AIEnhancementCoordinator")
        self.pattern_recognizer = AIPatternRecognizer()
        self.function_namer = AIFunctionNamer()
        self.quality_assessor = AICodeQualityAssessor()
        
        # Import new ML models
        try:
            from ..ml.neural_mapper import NeuralCodeMapper
            from ..ml.function_classifier import FunctionClassifier
            from ..ml.variable_namer import SmartVariableNamer
            from ..ml.quality_scorer import QualityScorer
            from ..ml.pattern_engine import PatternEngine
            
            self.neural_mapper = NeuralCodeMapper()
            self.function_classifier = FunctionClassifier()
            self.variable_namer = SmartVariableNamer()
            self.quality_scorer = QualityScorer()
            self.pattern_engine = PatternEngine()
            
            self.advanced_ml_available = True
            self.logger.info("Advanced ML models loaded successfully")
        except ImportError as e:
            self.logger.warning(f"Advanced ML models not available: {e}")
            self.advanced_ml_available = False
    
    def enhance_analysis(self, agent_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate AI enhancement across all analysis types"""
        enhancements = {
            'pattern_analysis': None,
            'naming_suggestions': None,
            'quality_assessment': None,
            'neural_mapping': None,
            'function_classification': None,
            'variable_naming': None,
            'advanced_quality_scoring': None,
            'integration_score': 0.0,
            'enhancement_summary': {},
            'ml_insights': []
        }
        
        try:
            # Legacy pattern recognition
            pattern_result = self.pattern_recognizer.analyze_patterns(agent_data, context)
            enhancements['pattern_analysis'] = pattern_result.to_dict() if pattern_result else None
            
            # Legacy function naming
            if 'decompiled_functions' in agent_data:
                naming_results = []
                for func_name, func_data in agent_data['decompiled_functions'].items():
                    result = self.function_namer.suggest_function_names(func_data, context)
                    naming_results.append(result.to_dict() if result else {})
                enhancements['naming_suggestions'] = naming_results
            
            # Legacy quality assessment
            quality_result = self.quality_assessor.assess_code_quality(agent_data, context)
            enhancements['quality_assessment'] = quality_result.to_dict() if quality_result else None
            
            # Advanced ML enhancements (if available)
            if self.advanced_ml_available:
                enhancements.update(self._apply_advanced_ml(agent_data, context))
            
            # Calculate integration score
            temp_enhancements = {
                'pattern_analysis': pattern_result,
                'naming_suggestions': [self.function_namer.suggest_function_names(func_data, context) 
                                     for func_name, func_data in agent_data.get('decompiled_functions', {}).items()],
                'quality_assessment': quality_result
            }
            enhancements['integration_score'] = self._calculate_integration_score(temp_enhancements)
            
            # Generate summary
            enhancements['enhancement_summary'] = self._generate_enhancement_summary(temp_enhancements)
            
        except Exception as e:
            self.logger.error(f"AI enhancement failed: {e}")
            enhancements['error'] = str(e)
        
        return enhancements
    
    def _calculate_integration_score(self, enhancements: Dict[str, Any]) -> float:
        """Calculate overall AI enhancement integration score"""
        scores = []
        
        if enhancements['pattern_analysis'] and hasattr(enhancements['pattern_analysis'], 'confidence'):
            scores.append(enhancements['pattern_analysis'].confidence)
        
        if enhancements['naming_suggestions']:
            confidences = []
            for r in enhancements['naming_suggestions']:
                if hasattr(r, 'confidence'):
                    confidences.append(r.confidence)
            if confidences:
                avg_naming_confidence = sum(confidences) / len(confidences)
                scores.append(avg_naming_confidence)
        
        if enhancements['quality_assessment'] and hasattr(enhancements['quality_assessment'], 'confidence'):
            scores.append(enhancements['quality_assessment'].confidence)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_enhancement_summary(self, enhancements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of AI enhancements"""
        summary = {
            'total_enhancements': 0,
            'high_confidence_results': 0,
            'key_insights': [],
            'recommended_actions': []
        }
        
        # Count enhancements
        if enhancements['pattern_analysis'] and hasattr(enhancements['pattern_analysis'], 'confidence'):
            summary['total_enhancements'] += 1
            if enhancements['pattern_analysis'].confidence > 0.8:
                summary['high_confidence_results'] += 1
                if hasattr(enhancements['pattern_analysis'], 'evidence'):
                    summary['key_insights'].extend(enhancements['pattern_analysis'].evidence)
        
        if enhancements['naming_suggestions']:
            summary['total_enhancements'] += len(enhancements['naming_suggestions'])
            for result in enhancements['naming_suggestions']:
                if hasattr(result, 'confidence') and result.confidence > 0.8:
                    summary['high_confidence_results'] += 1
        
        if enhancements['quality_assessment'] and hasattr(enhancements['quality_assessment'], 'confidence'):
            summary['total_enhancements'] += 1
            if enhancements['quality_assessment'].confidence > 0.8:
                summary['high_confidence_results'] += 1
            
            # Add quality-based recommendations
            if hasattr(enhancements['quality_assessment'], 'prediction'):
                quality_data = enhancements['quality_assessment'].prediction
                suggestions = quality_data.get('improvement_suggestions', [])
                summary['recommended_actions'].extend(suggestions)
        
        return summary
    
    def _apply_advanced_ml(self, agent_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply advanced ML models for enhanced analysis"""
        advanced_results = {}
        
        try:
            # Neural assembly-to-C mapping
            if 'objdump_output' in agent_data:
                assembly_blocks = self._extract_assembly_blocks(agent_data['objdump_output'])
                neural_mappings = []
                for block in assembly_blocks[:5]:  # Limit to 5 blocks to avoid overload
                    mapping = self.neural_mapper.map_assembly_to_c(block, context)
                    neural_mappings.append({
                        'assembly': block[:100] + '...' if len(block) > 100 else block,
                        'c_equivalent': mapping.c_equivalent,
                        'confidence': mapping.confidence,
                        'code_type': mapping.code_type.value,
                        'reasoning': mapping.reasoning
                    })
                advanced_results['neural_mapping'] = neural_mappings
            
            # Function classification
            if 'decompiled_functions' in agent_data:
                classifications = []
                for func_name, func_data in agent_data['decompiled_functions'].items():
                    classification = self.function_classifier.classify_function(func_data, context)
                    classifications.append({
                        'function_name': func_name,
                        'purpose': classification.purpose.value,
                        'confidence': classification.confidence,
                        'complexity': classification.complexity.value,
                        'suggested_name': classification.suggested_name,
                        'evidence': classification.evidence
                    })
                advanced_results['function_classification'] = classifications
            
            # Variable naming
            if 'variables' in agent_data or self._can_extract_variables(agent_data):
                variables = self._extract_variables_from_data(agent_data)
                variable_suggestions = []
                for var_data in variables[:10]:  # Limit to 10 variables
                    suggestion = self.variable_namer.suggest_variable_name(var_data, context)
                    variable_suggestions.append({
                        'original_context': var_data.get('register', 'unknown'),
                        'suggested_name': suggestion.suggested_name,
                        'confidence': suggestion.confidence,
                        'data_type': suggestion.data_type.value,
                        'usage_pattern': suggestion.usage_pattern.value,
                        'reasoning': suggestion.reasoning
                    })
                advanced_results['variable_naming'] = variable_suggestions
            
            # Advanced quality scoring
            quality_assessment = self.quality_scorer.assess_code_quality(agent_data, context)
            advanced_results['advanced_quality_scoring'] = {
                'overall_score': quality_assessment.overall_score,
                'quality_level': quality_assessment.quality_level.value,
                'metric_scores': {metric.value: score for metric, score in quality_assessment.metric_scores.items()},
                'issues_count': len(quality_assessment.issues),
                'high_severity_issues': len([issue for issue in quality_assessment.issues if issue.severity == 'high']),
                'recommendations': quality_assessment.recommendations,
                'confidence': quality_assessment.confidence
            }
            
            # Enhanced pattern analysis
            if 'objdump_output' in agent_data:
                pattern_analysis = self.pattern_engine.analyze_code_block(
                    self._convert_objdump_to_assembly(agent_data['objdump_output']), 
                    context
                )
                advanced_results['enhanced_patterns'] = {
                    'patterns_found': len(pattern_analysis['patterns']),
                    'confidence': pattern_analysis['confidence'],
                    'suggestions_count': len(pattern_analysis['suggestions']),
                    'code_quality_score': pattern_analysis['code_quality_score'],
                    'complexity_metrics': pattern_analysis['complexity_metrics']
                }
            
            # Generate ML insights
            advanced_results['ml_insights'] = self._generate_ml_insights(advanced_results)
            
        except Exception as e:
            self.logger.error(f"Advanced ML processing failed: {e}")
            advanced_results['ml_error'] = str(e)
        
        return advanced_results
    
    def _extract_assembly_blocks(self, objdump_data: Dict[str, Any]) -> List[str]:
        """Extract assembly code blocks from objdump data"""
        blocks = []
        
        if 'functions' in objdump_data:
            for func in objdump_data['functions'][:5]:  # Limit to first 5 functions
                instructions = func.get('instructions', [])
                if instructions:
                    block = '\n'.join(instructions[:20])  # First 20 instructions
                    blocks.append(block)
        
        return blocks
    
    def _can_extract_variables(self, agent_data: Dict[str, Any]) -> bool:
        """Check if we can extract variable information"""
        return ('objdump_output' in agent_data or 
                'decompiled_functions' in agent_data or
                'ghidra_output' in agent_data)
    
    def _extract_variables_from_data(self, agent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract variable information from available data"""
        variables = []
        
        # Extract from objdump data
        if 'objdump_output' in agent_data:
            objdump_data = agent_data['objdump_output']
            if 'functions' in objdump_data:
                for func in objdump_data['functions'][:3]:  # First 3 functions
                    instructions = func.get('instructions', [])
                    
                    # Look for register usage patterns
                    registers_used = set()
                    for instruction in instructions:
                        # Extract register references
                        for reg in ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp']:
                            if reg in instruction.lower():
                                registers_used.add(reg)
                    
                    # Create variable entries for unique registers
                    for reg in list(registers_used)[:5]:  # Limit to 5 per function
                        variables.append({
                            'register': reg,
                            'instructions': instructions,
                            'function_context': func.get('name', 'unknown')
                        })
        
        return variables
    
    def _convert_objdump_to_assembly(self, objdump_data: Dict[str, Any]) -> str:
        """Convert objdump data to assembly string"""
        assembly_lines = []
        
        if 'functions' in objdump_data:
            for func in objdump_data['functions'][:3]:  # First 3 functions
                instructions = func.get('instructions', [])
                assembly_lines.extend(instructions[:15])  # First 15 instructions per function
        
        return '\n'.join(assembly_lines)
    
    def _generate_ml_insights(self, advanced_results: Dict[str, Any]) -> List[str]:
        """Generate insights from ML analysis results"""
        insights = []
        
        # Neural mapping insights
        if 'neural_mapping' in advanced_results:
            mappings = advanced_results['neural_mapping']
            high_confidence = [m for m in mappings if m['confidence'] > 0.8]
            if high_confidence:
                insights.append(f"Neural mapper identified {len(high_confidence)} high-confidence C code equivalents")
        
        # Function classification insights
        if 'function_classification' in advanced_results:
            classifications = advanced_results['function_classification']
            purposes = [c['purpose'] for c in classifications]
            most_common = max(set(purposes), key=purposes.count) if purposes else None
            if most_common:
                insights.append(f"Most common function type: {most_common}")
        
        # Quality insights
        if 'advanced_quality_scoring' in advanced_results:
            quality = advanced_results['advanced_quality_scoring']
            level = quality['quality_level']
            score = quality['overall_score']
            insights.append(f"Code quality: {level} ({score:.1%})")
            
            if quality['high_severity_issues'] > 0:
                insights.append(f"Found {quality['high_severity_issues']} critical quality issues")
        
        # Variable naming insights
        if 'variable_naming' in advanced_results:
            suggestions = advanced_results['variable_naming']
            avg_confidence = sum(s['confidence'] for s in suggestions) / len(suggestions) if suggestions else 0
            insights.append(f"Variable naming confidence: {avg_confidence:.1%}")
        
        return insights