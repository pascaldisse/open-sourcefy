"""
Function Purpose Classification Model
Advanced ML model for identifying function purposes and behaviors.
"""

import numpy as np
import re
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import defaultdict, Counter
import math


class FunctionPurpose(Enum):
    """Types of function purposes"""
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor" 
    GETTER = "getter"
    SETTER = "setter"
    VALIDATOR = "validator"
    UTILITY = "utility"
    ALGORITHM = "algorithm"
    IO_OPERATION = "io_operation"
    MEMORY_MANAGEMENT = "memory_management"
    STRING_PROCESSING = "string_processing"
    MATH_OPERATION = "math_operation"
    CONTROL_FLOW = "control_flow"
    DATA_CONVERSION = "data_conversion"
    ERROR_HANDLER = "error_handler"
    CALLBACK = "callback"
    WRAPPER = "wrapper"
    MAIN_LOGIC = "main_logic"
    UNKNOWN = "unknown"


class FunctionComplexity(Enum):
    """Function complexity levels"""
    TRIVIAL = "trivial"         # 1-5 instructions
    SIMPLE = "simple"           # 6-20 instructions  
    MODERATE = "moderate"       # 21-50 instructions
    COMPLEX = "complex"         # 51-100 instructions
    VERY_COMPLEX = "very_complex"  # 100+ instructions


@dataclass
class ClassificationResult:
    """Result from function classification"""
    purpose: FunctionPurpose
    confidence: float
    complexity: FunctionComplexity
    characteristics: Dict[str, Any]
    evidence: List[str]
    suggested_name: str
    alternative_purposes: List[Tuple[FunctionPurpose, float]]


class FunctionAnalyzer:
    """Analyzes function characteristics for classification"""
    
    def __init__(self):
        self.api_patterns = self._build_api_patterns()
        self.naming_patterns = self._build_naming_patterns()
        self.instruction_signatures = self._build_instruction_signatures()
    
    def _build_api_patterns(self) -> Dict[FunctionPurpose, List[str]]:
        """Build patterns for API function calls"""
        return {
            FunctionPurpose.IO_OPERATION: [
                'fopen', 'fclose', 'fread', 'fwrite', 'printf', 'scanf',
                'read', 'write', 'open', 'close', 'lseek',
                'createfile', 'readfile', 'writefile', 'closehandle'
            ],
            FunctionPurpose.MEMORY_MANAGEMENT: [
                'malloc', 'free', 'calloc', 'realloc', 'new', 'delete',
                'alloca', 'mmap', 'munmap', 'virtualalloc', 'virtualfree',
                'heapalloc', 'heapfree', 'globalalloc'
            ],
            FunctionPurpose.STRING_PROCESSING: [
                'strcpy', 'strcat', 'strcmp', 'strlen', 'strstr', 'strchr',
                'strtok', 'sprintf', 'sscanf', 'strncpy', 'strncmp',
                'wcscpy', 'wcscat', 'wcslen', 'memcpy', 'memset', 'memcmp'
            ],
            FunctionPurpose.MATH_OPERATION: [
                'sin', 'cos', 'tan', 'sqrt', 'pow', 'exp', 'log', 'floor',
                'ceil', 'abs', 'fabs', 'atan', 'atan2', 'asin', 'acos'
            ],
            FunctionPurpose.ERROR_HANDLER: [
                'perror', 'strerror', 'abort', 'exit', 'terminate',
                'getlasterror', 'seterror', 'throw', 'catch'
            ]
        }
    
    def _build_naming_patterns(self) -> Dict[FunctionPurpose, List[str]]:
        """Build naming patterns for function purposes"""
        return {
            FunctionPurpose.CONSTRUCTOR: [
                r'.*init.*', r'.*create.*', r'.*new.*', r'.*ctor.*',
                r'.*construct.*', r'.*setup.*', r'.*build.*'
            ],
            FunctionPurpose.DESTRUCTOR: [
                r'.*destroy.*', r'.*delete.*', r'.*cleanup.*', r'.*dtor.*',
                r'.*free.*', r'.*release.*', r'.*close.*', r'.*shutdown.*'
            ],
            FunctionPurpose.GETTER: [
                r'get.*', r'.*get.*', r'fetch.*', r'retrieve.*',
                r'obtain.*', r'access.*', r'read.*'
            ],
            FunctionPurpose.SETTER: [
                r'set.*', r'.*set.*', r'update.*', r'modify.*',
                r'change.*', r'write.*', r'assign.*', r'configure.*'
            ],
            FunctionPurpose.VALIDATOR: [
                r'is.*', r'.*is.*', r'check.*', r'validate.*',
                r'verify.*', r'test.*', r'.*valid.*', r'.*check.*'
            ],
            FunctionPurpose.UTILITY: [
                r'.*util.*', r'.*helper.*', r'.*tool.*', r'.*assist.*',
                r'process.*', r'handle.*', r'manage.*'
            ],
            FunctionPurpose.DATA_CONVERSION: [
                r'.*convert.*', r'.*transform.*', r'.*parse.*', r'.*format.*',
                r'.*encode.*', r'.*decode.*', r'.*serialize.*', r'.*deserialize.*'
            ]
        }
    
    def _build_instruction_signatures(self) -> Dict[FunctionPurpose, Dict[str, float]]:
        """Build instruction signatures for different function types"""
        return {
            FunctionPurpose.MATH_OPERATION: {
                'fadd': 0.8, 'fsub': 0.8, 'fmul': 0.8, 'fdiv': 0.8,
                'add': 0.6, 'sub': 0.6, 'mul': 0.6, 'div': 0.6,
                'shl': 0.4, 'shr': 0.4, 'and': 0.3, 'or': 0.3, 'xor': 0.3
            },
            FunctionPurpose.STRING_PROCESSING: {
                'rep': 0.9, 'movsb': 0.8, 'movsw': 0.8, 'movsd': 0.8,
                'cmpsb': 0.8, 'scasb': 0.7, 'stosb': 0.7, 'lodsb': 0.7
            },
            FunctionPurpose.MEMORY_MANAGEMENT: {
                'malloc': 0.9, 'free': 0.9, 'calloc': 0.9, 'realloc': 0.9,
                'mov': 0.3, 'lea': 0.4, 'push': 0.2, 'pop': 0.2
            },
            FunctionPurpose.IO_OPERATION: {
                'call': 0.5, 'int': 0.7, 'syscall': 0.8,
                'mov': 0.2, 'push': 0.3, 'pop': 0.3
            },
            FunctionPurpose.CONTROL_FLOW: {
                'jmp': 0.8, 'je': 0.7, 'jne': 0.7, 'jz': 0.6, 'jnz': 0.6,
                'call': 0.5, 'ret': 0.4, 'cmp': 0.6, 'test': 0.6
            }
        }
    
    def analyze_function(self, function_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function characteristics"""
        characteristics = {
            'instruction_count': 0,
            'complexity_score': 0.0,
            'api_calls': [],
            'control_flow_complexity': 0,
            'memory_operations': 0,
            'arithmetic_operations': 0,
            'string_operations': 0,
            'branching_factor': 0.0,
            'cyclomatic_complexity': 1,
            'parameter_count': 0,
            'return_paths': 0,
            'function_calls': 0,
            'loop_indicators': 0,
            'stack_frame_size': 0
        }
        
        instructions = function_data.get('instructions', [])
        if isinstance(instructions, str):
            instructions = instructions.split('\n')
        
        characteristics['instruction_count'] = len([i for i in instructions if i.strip()])
        
        # Analyze instructions
        for instruction in instructions:
            if not instruction.strip():
                continue
                
            instruction_lower = instruction.lower().strip()
            
            # Count different operation types
            if any(op in instruction_lower for op in ['add', 'sub', 'mul', 'div', 'shl', 'shr']):
                characteristics['arithmetic_operations'] += 1
            
            if any(op in instruction_lower for op in ['mov', 'lea', 'push', 'pop']):
                characteristics['memory_operations'] += 1
            
            if any(op in instruction_lower for op in ['rep', 'movs', 'cmps', 'scas', 'stos']):
                characteristics['string_operations'] += 1
            
            if any(op in instruction_lower for op in ['call']):
                characteristics['function_calls'] += 1
                # Extract function name for API analysis
                parts = instruction_lower.split()
                if len(parts) >= 2:
                    func_name = parts[-1]
                    characteristics['api_calls'].append(func_name)
            
            if any(op in instruction_lower for op in ['jmp', 'je', 'jne', 'jz', 'jnz', 'jl', 'jg']):
                characteristics['control_flow_complexity'] += 1
            
            if 'ret' in instruction_lower:
                characteristics['return_paths'] += 1
        
        # Calculate derived metrics
        total_instructions = characteristics['instruction_count']
        if total_instructions > 0:
            characteristics['branching_factor'] = characteristics['control_flow_complexity'] / total_instructions
            characteristics['cyclomatic_complexity'] = max(1, characteristics['control_flow_complexity'] + 1)
        
        # Estimate complexity score
        characteristics['complexity_score'] = self._calculate_complexity_score(characteristics)
        
        # Detect loops (simplified)
        characteristics['loop_indicators'] = self._detect_loops(instructions)
        
        # Estimate parameter count from stack frame analysis
        characteristics['parameter_count'] = self._estimate_parameters(instructions)
        
        return characteristics
    
    def _calculate_complexity_score(self, characteristics: Dict[str, Any]) -> float:
        """Calculate overall complexity score"""
        instruction_factor = min(1.0, characteristics['instruction_count'] / 50.0)
        branching_factor = characteristics['branching_factor'] * 2
        call_factor = min(1.0, characteristics['function_calls'] / 10.0)
        
        return (instruction_factor + branching_factor + call_factor) / 3
    
    def _detect_loops(self, instructions: List[str]) -> int:
        """Detect loop patterns in instructions"""
        loop_count = 0
        labels = {}
        
        # Find labels
        for i, instruction in enumerate(instructions):
            if ':' in instruction:
                label = instruction.split(':')[0].strip()
                labels[label] = i
        
        # Find backward jumps
        for i, instruction in enumerate(instructions):
            if any(jump in instruction.lower() for jump in ['jmp', 'je', 'jne', 'jl', 'jg']):
                parts = instruction.split()
                if len(parts) >= 2:
                    target = parts[-1]
                    if target in labels and labels[target] < i:
                        loop_count += 1
        
        return loop_count
    
    def _estimate_parameters(self, instructions: List[str]) -> int:
        """Estimate parameter count from function prologue"""
        param_count = 0
        
        # Look for stack frame setup and parameter access
        for instruction in instructions[:20]:  # Check first 20 instructions
            instruction_lower = instruction.lower()
            
            # Look for parameter access patterns
            if 'ebp+' in instruction_lower or 'rbp+' in instruction_lower:
                # Extract offset to estimate parameter
                offset_match = re.search(r'[er]bp\+(\d+)', instruction_lower)
                if offset_match:
                    offset = int(offset_match.group(1))
                    if offset >= 8:  # Parameters typically start at ebp+8
                        param_index = (offset - 8) // 4  # Assuming 32-bit parameters
                        param_count = max(param_count, param_index + 1)
        
        return min(param_count, 8)  # Cap at reasonable number


class FunctionClassifier:
    """ML-based function purpose classifier"""
    
    def __init__(self):
        self.logger = logging.getLogger("FunctionClassifier")
        self.analyzer = FunctionAnalyzer()
        self.purpose_weights = self._initialize_purpose_weights()
        self.classification_cache = {}
    
    def _initialize_purpose_weights(self) -> Dict[FunctionPurpose, Dict[str, float]]:
        """Initialize weights for different function purposes"""
        return {
            FunctionPurpose.CONSTRUCTOR: {
                'memory_operations': 0.8,
                'function_calls': 0.6,
                'complexity_score': 0.3,
                'parameter_count': 0.4
            },
            FunctionPurpose.DESTRUCTOR: {
                'memory_operations': 0.9,
                'function_calls': 0.5,
                'complexity_score': 0.2,
                'api_calls_free': 0.9
            },
            FunctionPurpose.GETTER: {
                'instruction_count': -0.7,  # Usually simple
                'return_paths': 0.8,
                'memory_operations': 0.6,
                'complexity_score': -0.5  # Should be simple
            },
            FunctionPurpose.SETTER: {
                'instruction_count': -0.6,  # Usually simple
                'memory_operations': 0.8,
                'parameter_count': 0.7,
                'complexity_score': -0.4
            },
            FunctionPurpose.VALIDATOR: {
                'control_flow_complexity': 0.8,
                'return_paths': 0.7,
                'complexity_score': 0.4,
                'arithmetic_operations': 0.3
            },
            FunctionPurpose.ALGORITHM: {
                'complexity_score': 0.9,
                'arithmetic_operations': 0.8,
                'loop_indicators': 0.7,
                'instruction_count': 0.6
            },
            FunctionPurpose.IO_OPERATION: {
                'function_calls': 0.8,
                'api_calls_io': 0.9,
                'complexity_score': 0.4
            },
            FunctionPurpose.STRING_PROCESSING: {
                'string_operations': 0.9,
                'loop_indicators': 0.6,
                'memory_operations': 0.5
            },
            FunctionPurpose.MATH_OPERATION: {
                'arithmetic_operations': 0.9,
                'complexity_score': 0.5,
                'api_calls_math': 0.8
            },
            FunctionPurpose.MEMORY_MANAGEMENT: {
                'memory_operations': 0.9,
                'api_calls_memory': 0.9,
                'function_calls': 0.6
            }
        }
    
    def classify_function(self, function_data: Dict[str, Any], context: Dict[str, Any] = None) -> ClassificationResult:
        """Classify function purpose using ML techniques"""
        # Create cache key
        cache_key = self._create_cache_key(function_data)
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        try:
            # Analyze function characteristics
            characteristics = self.analyzer.analyze_function(function_data)
            
            # Perform name-based classification
            name_scores = self._classify_by_name(function_data.get('name', ''))
            
            # Perform instruction-based classification
            instruction_scores = self._classify_by_instructions(characteristics)
            
            # Perform API-based classification
            api_scores = self._classify_by_api_calls(characteristics.get('api_calls', []))
            
            # Combine scores
            combined_scores = self._combine_scores(name_scores, instruction_scores, api_scores)
            
            # Get best classification
            best_purpose = max(combined_scores.keys(), key=lambda k: combined_scores[k])
            confidence = combined_scores[best_purpose]
            
            # Determine complexity
            complexity = self._determine_complexity(characteristics)
            
            # Generate evidence
            evidence = self._generate_evidence(best_purpose, characteristics, name_scores, instruction_scores, api_scores)
            
            # Suggest function name
            suggested_name = self._suggest_function_name(best_purpose, characteristics, function_data.get('name', ''))
            
            # Get alternatives
            alternatives = self._get_alternatives(combined_scores, best_purpose)
            
            result = ClassificationResult(
                purpose=best_purpose,
                confidence=confidence,
                complexity=complexity,
                characteristics=characteristics,
                evidence=evidence,
                suggested_name=suggested_name,
                alternative_purposes=alternatives
            )
            
            # Cache result
            self.classification_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Function classification failed: {e}")
            return ClassificationResult(
                purpose=FunctionPurpose.UNKNOWN,
                confidence=0.0,
                complexity=FunctionComplexity.SIMPLE,
                characteristics={},
                evidence=[f"Classification error: {str(e)}"],
                suggested_name="unknown_function",
                alternative_purposes=[]
            )
    
    def _create_cache_key(self, function_data: Dict[str, Any]) -> str:
        """Create cache key for function data"""
        key_data = {
            'name': function_data.get('name', ''),
            'instructions_hash': hashlib.md5(str(function_data.get('instructions', '')).encode()).hexdigest()
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def _classify_by_name(self, function_name: str) -> Dict[FunctionPurpose, float]:
        """Classify function based on naming patterns"""
        scores = defaultdict(float)
        
        if not function_name:
            return dict(scores)
        
        name_lower = function_name.lower()
        
        for purpose, patterns in self.analyzer.naming_patterns.items():
            for pattern in patterns:
                if re.search(pattern, name_lower):
                    scores[purpose] += 0.8
                    break
        
        return dict(scores)
    
    def _classify_by_instructions(self, characteristics: Dict[str, Any]) -> Dict[FunctionPurpose, float]:
        """Classify function based on instruction patterns"""
        scores = defaultdict(float)
        
        for purpose, weights in self.purpose_weights.items():
            score = 0.0
            
            for feature, weight in weights.items():
                if feature in characteristics:
                    value = characteristics[feature]
                    if isinstance(value, (int, float)):
                        # Normalize value
                        if feature == 'instruction_count':
                            normalized_value = min(1.0, value / 50.0)
                        elif feature == 'complexity_score':
                            normalized_value = value
                        elif feature.endswith('_operations'):
                            normalized_value = min(1.0, value / 10.0)
                        elif feature == 'parameter_count':
                            normalized_value = min(1.0, value / 5.0)
                        elif feature == 'return_paths':
                            normalized_value = min(1.0, value / 3.0)
                        else:
                            normalized_value = min(1.0, value / 10.0)
                        
                        score += weight * normalized_value
            
            scores[purpose] = max(0.0, score)
        
        return dict(scores)
    
    def _classify_by_api_calls(self, api_calls: List[str]) -> Dict[FunctionPurpose, float]:
        """Classify function based on API calls"""
        scores = defaultdict(float)
        
        for purpose, api_list in self.analyzer.api_patterns.items():
            matches = 0
            for api_call in api_calls:
                if any(api in api_call.lower() for api in api_list):
                    matches += 1
            
            if matches > 0:
                scores[purpose] = min(1.0, matches / 3.0)  # Normalize to max 3 API calls
        
        return dict(scores)
    
    def _combine_scores(self, name_scores: Dict[FunctionPurpose, float], 
                       instruction_scores: Dict[FunctionPurpose, float],
                       api_scores: Dict[FunctionPurpose, float]) -> Dict[FunctionPurpose, float]:
        """Combine different classification scores"""
        combined = defaultdict(float)
        all_purposes = set(name_scores.keys()) | set(instruction_scores.keys()) | set(api_scores.keys())
        
        if not all_purposes:
            all_purposes = [FunctionPurpose.UNKNOWN]
        
        for purpose in all_purposes:
            # Weighted combination
            name_weight = 0.4
            instruction_weight = 0.4
            api_weight = 0.2
            
            combined[purpose] = (
                name_weight * name_scores.get(purpose, 0.0) +
                instruction_weight * instruction_scores.get(purpose, 0.0) +
                api_weight * api_scores.get(purpose, 0.0)
            )
        
        # Ensure we have at least UNKNOWN with some score
        if not combined:
            combined[FunctionPurpose.UNKNOWN] = 0.1
        
        return dict(combined)
    
    def _determine_complexity(self, characteristics: Dict[str, Any]) -> FunctionComplexity:
        """Determine function complexity level"""
        instruction_count = characteristics.get('instruction_count', 0)
        
        if instruction_count <= 5:
            return FunctionComplexity.TRIVIAL
        elif instruction_count <= 20:
            return FunctionComplexity.SIMPLE
        elif instruction_count <= 50:
            return FunctionComplexity.MODERATE
        elif instruction_count <= 100:
            return FunctionComplexity.COMPLEX
        else:
            return FunctionComplexity.VERY_COMPLEX
    
    def _generate_evidence(self, purpose: FunctionPurpose, characteristics: Dict[str, Any],
                          name_scores: Dict[FunctionPurpose, float],
                          instruction_scores: Dict[FunctionPurpose, float],
                          api_scores: Dict[FunctionPurpose, float]) -> List[str]:
        """Generate evidence for classification decision"""
        evidence = []
        
        # Evidence from naming
        if name_scores.get(purpose, 0) > 0.5:
            evidence.append(f"Function name suggests {purpose.value} pattern")
        
        # Evidence from instructions
        if instruction_scores.get(purpose, 0) > 0.3:
            evidence.append(f"Instruction patterns match {purpose.value} behavior")
        
        # Evidence from API calls
        if api_scores.get(purpose, 0) > 0.3:
            evidence.append(f"API calls indicate {purpose.value} functionality")
        
        # Specific characteristics evidence
        if characteristics.get('complexity_score', 0) > 0.7:
            evidence.append("High complexity suggests algorithmic function")
        elif characteristics.get('complexity_score', 0) < 0.3:
            evidence.append("Low complexity suggests simple getter/setter")
        
        if characteristics.get('string_operations', 0) > 3:
            evidence.append("Multiple string operations detected")
        
        if characteristics.get('memory_operations', 0) > 5:
            evidence.append("Significant memory manipulation detected")
        
        if not evidence:
            evidence.append("Classification based on general pattern analysis")
        
        return evidence
    
    def _suggest_function_name(self, purpose: FunctionPurpose, characteristics: Dict[str, Any], original_name: str) -> str:
        """Suggest appropriate function name based on purpose"""
        if original_name and not original_name.startswith(('sub_', 'func_', 'FUN_')):
            return original_name
        
        name_templates = {
            FunctionPurpose.CONSTRUCTOR: ['initializeObject', 'createInstance', 'setupData'],
            FunctionPurpose.DESTRUCTOR: ['cleanupObject', 'destroyInstance', 'releaseResources'],
            FunctionPurpose.GETTER: ['getValue', 'getData', 'getProperty'],
            FunctionPurpose.SETTER: ['setValue', 'setData', 'updateProperty'],
            FunctionPurpose.VALIDATOR: ['isValid', 'checkValue', 'validateInput'],
            FunctionPurpose.UTILITY: ['processData', 'helperFunction', 'utilityOperation'],
            FunctionPurpose.ALGORITHM: ['calculateResult', 'processAlgorithm', 'computeValue'],
            FunctionPurpose.IO_OPERATION: ['readData', 'writeOutput', 'ioOperation'],
            FunctionPurpose.MEMORY_MANAGEMENT: ['allocateMemory', 'manageBuffer', 'handleMemory'],
            FunctionPurpose.STRING_PROCESSING: ['processString', 'manipulateText', 'stringOperation'],
            FunctionPurpose.MATH_OPERATION: ['calculateMath', 'computeResult', 'mathFunction'],
            FunctionPurpose.ERROR_HANDLER: ['handleError', 'processException', 'errorHandler']
        }
        
        templates = name_templates.get(purpose, ['unknownFunction'])
        
        # Select template based on characteristics
        if characteristics.get('complexity_score', 0) > 0.7:
            # Complex function - use more specific name
            return templates[0] if len(templates) > 0 else 'complexFunction'
        else:
            # Simple function - use simpler name
            return templates[-1] if len(templates) > 1 else templates[0]
    
    def _get_alternatives(self, combined_scores: Dict[FunctionPurpose, float], best_purpose: FunctionPurpose) -> List[Tuple[FunctionPurpose, float]]:
        """Get alternative classification possibilities"""
        alternatives = []
        
        sorted_purposes = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        for purpose, score in sorted_purposes[1:4]:  # Top 3 alternatives
            if score > 0.1 and purpose != best_purpose:
                alternatives.append((purpose, score))
        
        return alternatives
    
    def get_classifier_stats(self) -> Dict[str, Any]:
        """Get classifier statistics"""
        return {
            'supported_purposes': [p.value for p in FunctionPurpose],
            'complexity_levels': [c.value for c in FunctionComplexity],
            'cached_classifications': len(self.classification_cache),
            'api_patterns': {purpose.value: len(patterns) for purpose, patterns in self.analyzer.api_patterns.items()},
            'naming_patterns': {purpose.value: len(patterns) for purpose, patterns in self.analyzer.naming_patterns.items()}
        }