"""
Neural Network for Assembly-to-C Code Mapping
Advanced machine learning model for intelligent code reconstruction.
"""

import json
import logging
from typing import Dict, List, Any, Tuple, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    NDArray = np.ndarray
except ImportError:
    NUMPY_AVAILABLE = False
    NDArray = Any  # Fallback type when numpy is not available
from dataclasses import dataclass
from enum import Enum
import re
import hashlib
from collections import defaultdict, Counter
import math


class CodeType(Enum):
    """Types of code constructs"""
    CONTROL_FLOW = "control_flow"
    DATA_OPERATION = "data_operation"
    FUNCTION_CALL = "function_call"
    ARITHMETIC = "arithmetic"
    MEMORY_ACCESS = "memory_access"
    STRING_OPERATION = "string_operation"
    SYSTEM_CALL = "system_call"
    UNKNOWN = "unknown"


@dataclass
class MappingResult:
    """Result from neural mapping"""
    code_type: CodeType
    c_equivalent: str
    confidence: float
    reasoning: str
    features: Dict[str, float]
    alternatives: List[Tuple[str, float]]


class FeatureExtractor:
    """Extract features from assembly code for neural network"""
    
    def __init__(self):
        self.instruction_vocab = self._build_instruction_vocabulary()
        self.register_map = self._build_register_mapping()
        self.pattern_cache = {}
    
    def _build_instruction_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary of common x86 instructions"""
        instructions = [
            'mov', 'add', 'sub', 'mul', 'div', 'imul', 'idiv',
            'and', 'or', 'xor', 'not', 'shl', 'shr', 'sal', 'sar',
            'cmp', 'test', 'je', 'jne', 'jz', 'jnz', 'jl', 'jg', 'jle', 'jge',
            'jmp', 'call', 'ret', 'push', 'pop', 'lea', 'nop',
            'inc', 'dec', 'neg', 'rep', 'movsb', 'movsw', 'movsd',
            'cmpsb', 'scasb', 'stosb', 'lodsb', 'int', 'syscall'
        ]
        return {inst: i for i, inst in enumerate(instructions)}
    
    def _build_register_mapping(self) -> Dict[str, int]:
        """Build mapping of registers to indices"""
        registers = [
            'eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp',
            'ax', 'bx', 'cx', 'dx', 'al', 'bl', 'cl', 'dl',
            'ah', 'bh', 'ch', 'dh', 'rax', 'rbx', 'rcx', 'rdx',
            'rsi', 'rdi', 'rsp', 'rbp', 'r8', 'r9', 'r10', 'r11',
            'r12', 'r13', 'r14', 'r15', 'xmm0', 'xmm1', 'xmm2', 'xmm3'
        ]
        return {reg: i for i, reg in enumerate(registers)}
    
    def extract_features(self, assembly_block: str) -> NDArray:
        """Extract feature vector from assembly code block"""
        cache_key = hashlib.md5(assembly_block.encode()).hexdigest()
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        features = []
        lines = [line.strip().lower() for line in assembly_block.split('\n') if line.strip()]
        
        # Basic statistics
        features.extend(self._extract_basic_stats(lines))
        
        # Instruction frequency features
        features.extend(self._extract_instruction_features(lines))
        
        # Register usage features
        features.extend(self._extract_register_features(lines))
        
        # Pattern features
        features.extend(self._extract_pattern_features(lines))
        
        # Control flow features
        features.extend(self._extract_control_flow_features(lines))
        
        # Memory access features
        features.extend(self._extract_memory_features(lines))
        
        if NUMPY_AVAILABLE:
            feature_vector = np.array(features, dtype=np.float32)
        else:
            feature_vector = features  # Return list when numpy not available
        self.pattern_cache[cache_key] = feature_vector
        
        return feature_vector
    
    def _extract_basic_stats(self, lines: List[str]) -> List[float]:
        """Extract basic statistical features"""
        if NUMPY_AVAILABLE:
            avg_length = np.mean([len(line) for line in lines]) if lines else 0
        else:
            avg_length = sum(len(line) for line in lines) / len(lines) if lines else 0
            
        return [
            len(lines),  # Number of instructions
            avg_length,  # Average line length
            len(set(lines)),  # Unique instructions
            len([line for line in lines if line.startswith('call')]),  # Function calls
            len([line for line in lines if any(j in line for j in ['je', 'jne', 'jmp'])]),  # Jumps
        ]
    
    def _extract_instruction_features(self, lines: List[str]) -> List[float]:
        """Extract instruction frequency features"""
        features = [0.0] * len(self.instruction_vocab)
        
        for line in lines:
            parts = line.split()
            if parts:
                instruction = parts[0]
                if instruction in self.instruction_vocab:
                    features[self.instruction_vocab[instruction]] += 1
        
        # Normalize by total instructions
        total = sum(features)
        if total > 0:
            features = [f / total for f in features]
        
        return features
    
    def _extract_register_features(self, lines: List[str]) -> List[float]:
        """Extract register usage features"""
        features = [0.0] * len(self.register_map)
        
        for line in lines:
            for register in self.register_map:
                if register in line:
                    features[self.register_map[register]] += 1
        
        # Normalize
        total = sum(features)
        if total > 0:
            features = [f / total for f in features]
        
        return features
    
    def _extract_pattern_features(self, lines: List[str]) -> List[float]:
        """Extract high-level pattern features"""
        features = []
        
        # Array access patterns
        array_access = sum(1 for line in lines if '[' in line and ']' in line)
        features.append(array_access / len(lines) if lines else 0)
        
        # Arithmetic patterns
        arithmetic = sum(1 for line in lines if any(op in line for op in ['add', 'sub', 'mul', 'div']))
        features.append(arithmetic / len(lines) if lines else 0)
        
        # String operation patterns
        string_ops = sum(1 for line in lines if any(op in line for op in ['rep', 'movs', 'cmps', 'scas']))
        features.append(string_ops / len(lines) if lines else 0)
        
        # Function prologue/epilogue patterns
        prologue = any('push ebp' in line or 'push rbp' in line for line in lines[:3])
        epilogue = any('pop ebp' in line or 'pop rbp' in line for line in lines[-3:])
        features.extend([float(prologue), float(epilogue)])
        
        # Loop patterns (backward jumps)
        loop_patterns = self._detect_loop_patterns(lines)
        features.append(loop_patterns)
        
        return features
    
    def _extract_control_flow_features(self, lines: List[str]) -> List[float]:
        """Extract control flow specific features"""
        features = []
        
        # Conditional jumps
        conditional_jumps = sum(1 for line in lines if any(j in line for j in ['je', 'jne', 'jz', 'jnz', 'jl', 'jg']))
        features.append(conditional_jumps / len(lines) if lines else 0)
        
        # Unconditional jumps
        unconditional_jumps = sum(1 for line in lines if 'jmp' in line and not any(c in line for c in ['je', 'jne']))
        features.append(unconditional_jumps / len(lines) if lines else 0)
        
        # Return statements
        returns = sum(1 for line in lines if 'ret' in line)
        features.append(returns / len(lines) if lines else 0)
        
        # Function calls
        calls = sum(1 for line in lines if 'call' in line)
        features.append(calls / len(lines) if lines else 0)
        
        return features
    
    def _extract_memory_features(self, lines: List[str]) -> List[float]:
        """Extract memory access pattern features"""
        features = []
        
        # Stack access patterns
        stack_access = sum(1 for line in lines if any(sp in line for sp in ['esp', 'rsp', 'ebp', 'rbp']))
        features.append(stack_access / len(lines) if lines else 0)
        
        # Heap access patterns (indirect addressing)
        heap_access = sum(1 for line in lines if '[' in line and not any(sp in line for sp in ['esp', 'rsp', 'ebp', 'rbp']))
        features.append(heap_access / len(lines) if lines else 0)
        
        # Pointer arithmetic
        pointer_arith = sum(1 for line in lines if 'lea' in line)
        features.append(pointer_arith / len(lines) if lines else 0)
        
        return features
    
    def _detect_loop_patterns(self, lines: List[str]) -> float:
        """Detect loop patterns in assembly"""
        labels = {}
        for i, line in enumerate(lines):
            if ':' in line:
                label = line.split(':')[0].strip()
                labels[label] = i
        
        backward_jumps = 0
        for i, line in enumerate(lines):
            if any(j in line for j in ['jmp', 'je', 'jne', 'jl', 'jg']):
                parts = line.split()
                if len(parts) >= 2:
                    target = parts[-1]
                    if target in labels and labels[target] < i:
                        backward_jumps += 1
        
        return backward_jumps / len(lines) if lines else 0


class NeuralCodeMapper:
    """Neural network for mapping assembly to C code"""
    
    def __init__(self):
        self.logger = logging.getLogger("NeuralCodeMapper")
        self.feature_extractor = FeatureExtractor()
        self.weights = self._initialize_weights()
        self.training_data = []
        self.c_templates = self._initialize_c_templates()
    
    def _initialize_weights(self) -> Dict[str, NDArray]:
        """Initialize neural network weights"""
        if not NUMPY_AVAILABLE:
            return {}  # Return empty dict when numpy is not available
            
        input_size = self._get_feature_size()
        hidden_size = 128
        output_size = len(CodeType)
        
        return {
            'W1': np.random.randn(input_size, hidden_size) * 0.1,
            'b1': np.zeros(hidden_size),
            'W2': np.random.randn(hidden_size, hidden_size) * 0.1,
            'b2': np.zeros(hidden_size),
            'W3': np.random.randn(hidden_size, output_size) * 0.1,
            'b3': np.zeros(output_size)
        }
    
    def _get_feature_size(self) -> int:
        """Calculate total feature vector size"""
        # This should match the total features extracted
        return (
            5 +  # Basic stats
            len(self.feature_extractor.instruction_vocab) +  # Instruction features
            len(self.feature_extractor.register_map) +  # Register features
            6 +  # Pattern features
            4 +  # Control flow features
            3    # Memory features
        )
    
    def _initialize_c_templates(self) -> Dict[CodeType, List[str]]:
        """Initialize C code templates for each code type"""
        return {
            CodeType.CONTROL_FLOW: [
                "if ({condition}) {\n    {body}\n}",
                "while ({condition}) {\n    {body}\n}",
                "for (int i = 0; i < {limit}; i++) {\n    {body}\n}",
                "switch ({variable}) {\n    case {value}:\n        {body}\n        break;\n}"
            ],
            CodeType.DATA_OPERATION: [
                "{variable} = {value};",
                "{array}[{index}] = {value};",
                "*{pointer} = {value};",
                "{struct_var}.{field} = {value};"
            ],
            CodeType.FUNCTION_CALL: [
                "{function}({parameters});",
                "{result} = {function}({parameters});",
                "return {function}({parameters});"
            ],
            CodeType.ARITHMETIC: [
                "{result} = {operand1} + {operand2};",
                "{result} = {operand1} * {operand2};",
                "{result} = {operand1} << {shift_amount};",
                "{result} = {operand1} & {mask};"
            ],
            CodeType.MEMORY_ACCESS: [
                "{variable} = *{pointer};",
                "*{pointer} = {value};",
                "{pointer} = &{variable};",
                "memcpy({dest}, {src}, {size});"
            ],
            CodeType.STRING_OPERATION: [
                "strcpy({dest}, {src});",
                "strcmp({str1}, {str2});",
                "strlen({string});",
                "strcat({dest}, {src});"
            ],
            CodeType.SYSTEM_CALL: [
                "malloc({size});",
                "free({pointer});",
                "printf(\"{format}\", {args});",
                "exit({code});"
            ]
        }
    
    def map_assembly_to_c(self, assembly_block: str, context: Dict[str, Any] = None) -> MappingResult:
        """Map assembly code block to C equivalent"""
        try:
            if not NUMPY_AVAILABLE:
                # Fallback to rule-based mapping when numpy is not available
                return self._rule_based_mapping(assembly_block, context or {})
                
            # Extract features
            features = self.feature_extractor.extract_features(assembly_block)
            
            # Forward pass through neural network
            code_type_probs = self._forward_pass(features)
            
            # Get predicted code type
            predicted_type_idx = np.argmax(code_type_probs)
            code_types = list(CodeType)
            predicted_type = code_types[predicted_type_idx]
            confidence = float(code_type_probs[predicted_type_idx])
            
            # Generate C code based on type and features
            c_code, reasoning = self._generate_c_code(predicted_type, features, assembly_block, context)
            
            # Generate alternatives
            alternatives = self._generate_alternatives(code_type_probs, features, assembly_block, context)
            
            return MappingResult(
                code_type=predicted_type,
                c_equivalent=c_code,
                confidence=confidence,
                reasoning=reasoning,
                features=self._features_to_dict(features),
                alternatives=alternatives
            )
            
        except Exception as e:
            self.logger.error(f"Neural mapping failed: {e}")
            return MappingResult(
                code_type=CodeType.UNKNOWN,
                c_equivalent="// Could not map assembly code",
                confidence=0.0,
                reasoning=f"Error in neural mapping: {str(e)}",
                features={},
                alternatives=[]
            )
    
    def _rule_based_mapping(self, assembly_block: str, context: Dict[str, Any]) -> MappingResult:
        """Simple rule-based mapping when numpy is not available"""
        lines = assembly_block.strip().split('\n')
        
        # Simple pattern matching
        if any('call' in line for line in lines):
            code_type = CodeType.FUNCTION_CALL
            c_code = "// Function call detected\nfunction_call();"
        elif any(any(op in line for op in ['add', 'sub', 'mul', 'div']) for line in lines):
            code_type = CodeType.ARITHMETIC
            c_code = "// Arithmetic operation detected\nresult = a + b;"
        elif any(any(op in line for op in ['mov', 'ld', 'st']) for line in lines):
            code_type = CodeType.MEMORY_ACCESS
            c_code = "// Memory access detected\nvalue = *ptr;"
        elif any(any(op in line for op in ['je', 'jne', 'jmp', 'cmp']) for line in lines):
            code_type = CodeType.CONTROL_FLOW
            c_code = "// Control flow detected\nif (condition) { /* code */ }"
        else:
            code_type = CodeType.UNKNOWN
            c_code = "// Unknown assembly pattern"
            
        return MappingResult(
            code_type=code_type,
            c_equivalent=c_code,
            confidence=0.5,  # Medium confidence for rule-based
            reasoning="Rule-based mapping (numpy not available)",
            features={},
            alternatives=[]
        )
    
    def _forward_pass(self, features: NDArray) -> NDArray:
        """Forward pass through neural network"""
        if not NUMPY_AVAILABLE:
            # Return dummy probabilities if numpy not available
            return [0.8, 0.1, 0.05, 0.025, 0.015, 0.01, 0.005, 0.0]
            
        # Ensure features is the right size
        expected_size = self._get_feature_size()
        if len(features) < expected_size:
            # Pad with zeros
            features = np.pad(features, (0, expected_size - len(features)))
        elif len(features) > expected_size:
            # Truncate
            features = features[:expected_size]
        
        # Layer 1
        z1 = np.dot(features, self.weights['W1']) + self.weights['b1']
        a1 = self._relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self._relu(z2)
        
        # Output layer
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        a3 = self._softmax(z3)
        
        return a3
    
    def _relu(self, x: NDArray) -> NDArray:
        """ReLU activation function"""
        if not NUMPY_AVAILABLE:
            return [max(0, val) for val in x]
        return np.maximum(0, x)
    
    def _softmax(self, x: NDArray) -> NDArray:
        """Softmax activation function"""
        if not NUMPY_AVAILABLE:
            import math
            max_val = max(x)
            exp_x = [math.exp(val - max_val) for val in x]
            sum_exp = sum(exp_x)
            return [val / sum_exp for val in exp_x]
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def _generate_c_code(self, code_type: CodeType, features: NDArray, assembly: str, context: Dict[str, Any]) -> Tuple[str, str]:
        """Generate C code based on predicted type and features"""
        templates = self.c_templates.get(code_type, ["// Unknown code pattern"])
        
        # Analyze assembly for specific patterns
        analysis = self._analyze_assembly_details(assembly, code_type)
        
        # Select best template
        template = self._select_best_template(templates, analysis)
        
        # Fill template with context-specific values
        c_code = self._fill_template(template, analysis, context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(code_type, analysis, features)
        
        return c_code, reasoning
    
    def _analyze_assembly_details(self, assembly: str, code_type: CodeType) -> Dict[str, Any]:
        """Analyze assembly for specific details"""
        lines = [line.strip().lower() for line in assembly.split('\n') if line.strip()]
        analysis = {
            'registers_used': [],
            'memory_operations': [],
            'constants': [],
            'function_calls': [],
            'jump_targets': []
        }
        
        for line in lines:
            # Extract registers
            for reg in self.feature_extractor.register_map:
                if reg in line:
                    analysis['registers_used'].append(reg)
            
            # Extract constants
            constants = re.findall(r'\b\d+\b', line)
            analysis['constants'].extend(constants)
            
            # Extract function calls
            if 'call' in line:
                parts = line.split()
                if len(parts) >= 2:
                    analysis['function_calls'].append(parts[-1])
            
            # Extract memory operations
            if '[' in line and ']' in line:
                mem_op = re.search(r'\[([^\]]+)\]', line)
                if mem_op:
                    analysis['memory_operations'].append(mem_op.group(1))
        
        # Remove duplicates
        for key in analysis:
            if isinstance(analysis[key], list):
                analysis[key] = list(set(analysis[key]))
        
        return analysis
    
    def _select_best_template(self, templates: List[str], analysis: Dict[str, Any]) -> str:
        """Select the best template based on analysis"""
        # Simple heuristic - could be improved with ML
        if analysis['function_calls']:
            # Prefer function call templates
            func_templates = [t for t in templates if 'function' in t]
            if func_templates:
                return func_templates[0]
        
        if analysis['memory_operations']:
            # Prefer memory access templates
            mem_templates = [t for t in templates if any(kw in t for kw in ['*', '[', 'memcpy'])]
            if mem_templates:
                return mem_templates[0]
        
        # Default to first template
        return templates[0]
    
    def _fill_template(self, template: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Fill template with context-specific values"""
        filled = template
        
        # Replace placeholders with intelligent values
        replacements = {
            '{condition}': self._generate_condition(analysis),
            '{body}': '// TODO: implement',
            '{variable}': self._generate_variable_name(analysis, context),
            '{value}': self._generate_value(analysis),
            '{function}': self._generate_function_name(analysis),
            '{parameters}': self._generate_parameters(analysis),
            '{operand1}': 'a',
            '{operand2}': 'b',
            '{result}': 'result',
            '{pointer}': 'ptr',
            '{size}': 'size',
            '{limit}': 'n',
            '{index}': 'i'
        }
        
        for placeholder, value in replacements.items():
            filled = filled.replace(placeholder, value)
        
        return filled
    
    def _generate_condition(self, analysis: Dict[str, Any]) -> str:
        """Generate appropriate condition based on analysis"""
        if analysis['constants']:
            const = analysis['constants'][0]
            return f"value == {const}"
        if analysis['registers_used']:
            reg = analysis['registers_used'][0].replace('e', '').replace('r', '')
            return f"{reg} != 0"
        return "condition"
    
    def _generate_variable_name(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate appropriate variable name"""
        if context and 'variable_hints' in context:
            hints = context['variable_hints']
            if hints:
                return hints[0]
        
        if analysis['registers_used']:
            reg = analysis['registers_used'][0]
            if reg in ['eax', 'rax']:
                return 'result'
            elif reg in ['ebx', 'rbx']:
                return 'data'
            elif reg in ['ecx', 'rcx']:
                return 'counter'
        
        return 'var'
    
    def _generate_value(self, analysis: Dict[str, Any]) -> str:
        """Generate appropriate value"""
        if analysis['constants']:
            return analysis['constants'][0]
        return '0'
    
    def _generate_function_name(self, analysis: Dict[str, Any]) -> str:
        """Generate appropriate function name"""
        if analysis['function_calls']:
            func = analysis['function_calls'][0]
            # Clean up function name
            if func.startswith('_'):
                func = func[1:]
            return func
        return 'function'
    
    def _generate_parameters(self, analysis: Dict[str, Any]) -> str:
        """Generate appropriate parameters"""
        if analysis['registers_used']:
            params = []
            for reg in analysis['registers_used'][:3]:  # Max 3 parameters
                if reg in ['eax', 'rax']:
                    params.append('arg1')
                elif reg in ['ebx', 'rbx']:
                    params.append('arg2')
                elif reg in ['ecx', 'rcx']:
                    params.append('arg3')
            if params:
                return ', '.join(params)
        return 'void'
    
    def _generate_reasoning(self, code_type: CodeType, analysis: Dict[str, Any], features: NDArray) -> str:
        """Generate reasoning for the mapping decision"""
        reasons = [f"Classified as {code_type.value}"]
        
        if analysis['function_calls']:
            reasons.append(f"Detected function calls: {', '.join(analysis['function_calls'][:2])}")
        
        if analysis['memory_operations']:
            reasons.append(f"Found memory operations: {len(analysis['memory_operations'])} patterns")
        
        if analysis['registers_used']:
            common_regs = [r for r in analysis['registers_used'] if r in ['eax', 'ebx', 'ecx', 'edx']]
            if common_regs:
                reasons.append(f"Uses general-purpose registers: {', '.join(common_regs[:3])}")
        
        return '; '.join(reasons)
    
    def _generate_alternatives(self, code_type_probs: NDArray, features: NDArray, assembly: str, context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Generate alternative C code suggestions"""
        alternatives = []
        code_types = list(CodeType)
        
        if not NUMPY_AVAILABLE:
            # Simple fallback alternatives
            return [
                ("// Alternative: arithmetic operation\nresult = a * b;", 0.3),
                ("// Alternative: function call\ncall_function();", 0.2)
            ]
        
        # Get top 3 alternative types
        top_indices = np.argsort(code_type_probs)[-3:][::-1]
        
        for idx in top_indices[1:]:  # Skip the top prediction
            alt_type = code_types[idx]
            confidence = float(code_type_probs[idx])
            
            if confidence > 0.1:  # Only include reasonable alternatives
                alt_code, _ = self._generate_c_code(alt_type, features, assembly, context)
                alternatives.append((alt_code, confidence))
        
        return alternatives
    
    def _features_to_dict(self, features: NDArray) -> Dict[str, float]:
        """Convert feature vector to dictionary for inspection"""
        feature_names = self._get_feature_names()
        if NUMPY_AVAILABLE and hasattr(features, 'tolist'):
            features_list = features.tolist()[:len(feature_names)]
        else:
            features_list = list(features)[:len(feature_names)]
        return dict(zip(feature_names, features_list))
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        names = ['line_count', 'avg_line_length', 'unique_instructions', 'function_calls', 'jumps']
        names.extend([f'inst_{inst}' for inst in self.feature_extractor.instruction_vocab.keys()])
        names.extend([f'reg_{reg}' for reg in self.feature_extractor.register_map.keys()])
        names.extend(['array_access', 'arithmetic', 'string_ops', 'prologue', 'epilogue', 'loops'])
        names.extend(['conditional_jumps', 'unconditional_jumps', 'returns', 'calls'])
        names.extend(['stack_access', 'heap_access', 'pointer_arith'])
        return names
    
    def train_on_example(self, assembly: str, expected_c: str, code_type: CodeType):
        """Add training example (for future learning implementation)"""
        self.training_data.append({
            'assembly': assembly,
            'expected_c': expected_c,
            'code_type': code_type,
            'features': self.feature_extractor.extract_features(assembly)
        })
        
        self.logger.info(f"Added training example for {code_type.value}")
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about the neural model"""
        return {
            'feature_size': self._get_feature_size(),
            'training_examples': len(self.training_data),
            'supported_types': [t.value for t in CodeType],
            'template_count': sum(len(templates) for templates in self.c_templates.values())
        }