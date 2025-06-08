"""
AI-Powered Variable Naming System
Intelligent variable name generation based on context and usage patterns.
"""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import defaultdict, Counter
import math


class VariableType(Enum):
    """Types of variables"""
    POINTER = "pointer"
    ARRAY = "array"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    STRUCT = "struct"
    FUNCTION_PTR = "function_ptr"
    BOOLEAN = "boolean"
    ENUM = "enum"
    UNION = "union"
    VOID_PTR = "void_ptr"
    UNKNOWN = "unknown"


class VariableScope(Enum):
    """Variable scope types"""
    LOCAL = "local"
    PARAMETER = "parameter"
    GLOBAL = "global"
    STATIC = "static"
    MEMBER = "member"
    UNKNOWN = "unknown"


class VariableUsage(Enum):
    """Variable usage patterns"""
    COUNTER = "counter"
    INDEX = "index"
    BUFFER = "buffer"
    SIZE = "size"
    FLAG = "flag"
    RESULT = "result"
    TEMPORARY = "temporary"
    ITERATOR = "iterator"
    ACCUMULATOR = "accumulator"
    HANDLE = "handle"
    POINTER_TO_DATA = "pointer_to_data"
    CONFIG = "config"
    STATUS = "status"
    UNKNOWN = "unknown"


@dataclass
class VariableContext:
    """Context information for variable"""
    register: str
    memory_location: Optional[str]
    data_type: VariableType
    scope: VariableScope
    usage_pattern: VariableUsage
    operations: List[str]
    related_functions: List[str]
    access_patterns: List[str]
    size_hint: Optional[int]
    initialization_value: Optional[Any]


@dataclass
class NamingResult:
    """Result from variable naming"""
    suggested_name: str
    confidence: float
    reasoning: str
    data_type: VariableType
    scope: VariableScope
    usage_pattern: VariableUsage
    alternatives: List[Tuple[str, float]]
    naming_convention: str


class VariableAnalyzer:
    """Analyzes variable usage patterns and context"""
    
    def __init__(self):
        self.register_map = self._build_register_semantics()
        self.operation_patterns = self._build_operation_patterns()
        self.naming_conventions = self._build_naming_conventions()
        self.context_cache = {}
    
    def _build_register_semantics(self) -> Dict[str, Dict[str, Any]]:
        """Build semantic meaning of registers"""
        return {
            'eax': {'typical_use': 'return_value', 'data_type': VariableType.INTEGER, 'scope': VariableScope.LOCAL},
            'rax': {'typical_use': 'return_value', 'data_type': VariableType.INTEGER, 'scope': VariableScope.LOCAL},
            'ebx': {'typical_use': 'data_pointer', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'rbx': {'typical_use': 'data_pointer', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'ecx': {'typical_use': 'counter', 'data_type': VariableType.INTEGER, 'scope': VariableScope.LOCAL},
            'rcx': {'typical_use': 'counter', 'data_type': VariableType.INTEGER, 'scope': VariableScope.LOCAL},
            'edx': {'typical_use': 'data', 'data_type': VariableType.INTEGER, 'scope': VariableScope.LOCAL},
            'rdx': {'typical_use': 'data', 'data_type': VariableType.INTEGER, 'scope': VariableScope.LOCAL},
            'esi': {'typical_use': 'source_index', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'rsi': {'typical_use': 'source_index', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'edi': {'typical_use': 'dest_index', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'rdi': {'typical_use': 'dest_index', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'esp': {'typical_use': 'stack_pointer', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'rsp': {'typical_use': 'stack_pointer', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'ebp': {'typical_use': 'base_pointer', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'rbp': {'typical_use': 'base_pointer', 'data_type': VariableType.POINTER, 'scope': VariableScope.LOCAL},
            'xmm0': {'typical_use': 'float_data', 'data_type': VariableType.FLOAT, 'scope': VariableScope.LOCAL},
            'xmm1': {'typical_use': 'float_data', 'data_type': VariableType.FLOAT, 'scope': VariableScope.LOCAL}
        }
    
    def _build_operation_patterns(self) -> Dict[VariableUsage, List[str]]:
        """Build patterns that indicate variable usage"""
        return {
            VariableUsage.COUNTER: [
                'inc', 'dec', 'add.*1', 'sub.*1', 'cmp.*limit', 'loop'
            ],
            VariableUsage.INDEX: [
                'lea.*\\[.*\\+.*\\]', 'mov.*\\[.*\\+.*\\]', 'shl.*2', 'shl.*3'
            ],
            VariableUsage.BUFFER: [
                'rep.*movs', 'rep.*stos', 'call.*memcpy', 'call.*strcpy'
            ],
            VariableUsage.SIZE: [
                'cmp.*0', 'test.*test', 'call.*strlen', 'call.*sizeof'
            ],
            VariableUsage.FLAG: [
                'test.*test', 'and.*1', 'or.*1', 'xor.*xor', 'cmp.*0'
            ],
            VariableUsage.RESULT: [
                'call.*', 'mov.*eax', 'mov.*rax', 'ret'
            ],
            VariableUsage.TEMPORARY: [
                'mov.*mov', 'push.*pop', 'xchg'
            ],
            VariableUsage.ITERATOR: [
                'mov.*\\[.*\\]', 'inc.*4', 'add.*4', 'add.*8'
            ],
            VariableUsage.ACCUMULATOR: [
                'add.*add', 'mul.*mul', 'xor.*xor', 'or.*or'
            ],
            VariableUsage.HANDLE: [
                'call.*open', 'call.*create', 'cmp.*-1', 'cmp.*null'
            ],
            VariableUsage.POINTER_TO_DATA: [
                'lea.*', 'mov.*\\[.*\\]', 'call.*malloc', 'call.*new'
            ]
        }
    
    def _build_naming_conventions(self) -> Dict[str, Dict[str, str]]:
        """Build naming convention templates"""
        return {
            'hungarian': {
                VariableType.POINTER.value: 'p{name}',
                VariableType.ARRAY.value: 'arr{name}',
                VariableType.INTEGER.value: 'n{name}',
                VariableType.STRING.value: 'sz{name}',
                VariableType.BOOLEAN.value: 'b{name}',
                VariableType.FLOAT.value: 'f{name}',
                'default': '{name}'
            },
            'camelCase': {
                'default': '{name}',
                'compound': '{prefix}{Name}'
            },
            'snake_case': {
                'default': '{name}',
                'compound': '{prefix}_{name}'
            },
            'PascalCase': {
                'default': '{Name}',
                'compound': '{Prefix}{Name}'
            }
        }
    
    def analyze_variable_context(self, variable_data: Dict[str, Any], function_context: Dict[str, Any]) -> VariableContext:
        """Analyze variable context from assembly code"""
        cache_key = self._create_context_cache_key(variable_data)
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        context = VariableContext(
            register=variable_data.get('register', ''),
            memory_location=variable_data.get('memory_location'),
            data_type=VariableType.UNKNOWN,
            scope=VariableScope.UNKNOWN,
            usage_pattern=VariableUsage.UNKNOWN,
            operations=[],
            related_functions=[],
            access_patterns=[],
            size_hint=None,
            initialization_value=None
        )
        
        # Analyze register semantics
        if context.register:
            reg_info = self.register_map.get(context.register.lower(), {})
            context.data_type = reg_info.get('data_type', VariableType.UNKNOWN)
            context.scope = reg_info.get('scope', VariableScope.LOCAL)
        
        # Analyze usage patterns from instructions
        instructions = variable_data.get('instructions', [])
        context.operations = self._extract_operations(instructions)
        context.usage_pattern = self._detect_usage_pattern(context.operations)
        
        # Analyze access patterns
        context.access_patterns = self._analyze_access_patterns(instructions, context.register)
        
        # Detect scope from memory location
        if context.memory_location:
            context.scope = self._detect_scope_from_memory(context.memory_location)
        
        # Analyze data type from operations
        context.data_type = self._refine_data_type(context.data_type, context.operations, context.access_patterns)
        
        # Extract related functions
        context.related_functions = self._extract_related_functions(instructions)
        
        # Detect size hints
        context.size_hint = self._detect_size_hints(context.operations, context.access_patterns)
        
        # Cache result
        self.context_cache[cache_key] = context
        
        return context
    
    def _create_context_cache_key(self, variable_data: Dict[str, Any]) -> str:
        """Create cache key for variable context"""
        key_data = {
            'register': variable_data.get('register', ''),
            'instructions_hash': hashlib.md5(str(variable_data.get('instructions', '')).encode()).hexdigest()
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def _extract_operations(self, instructions: List[str]) -> List[str]:
        """Extract operations performed on variable"""
        operations = []
        for instruction in instructions:
            if not instruction.strip():
                continue
            
            # Extract instruction mnemonic
            parts = instruction.strip().split()
            if parts:
                mnemonic = parts[0].lower()
                operations.append(mnemonic)
        
        return operations
    
    def _detect_usage_pattern(self, operations: List[str]) -> VariableUsage:
        """Detect usage pattern from operations"""
        operation_str = ' '.join(operations)
        
        scores = defaultdict(float)
        
        for usage, patterns in self.operation_patterns.items():
            for pattern in patterns:
                if re.search(pattern, operation_str, re.IGNORECASE):
                    scores[usage] += 1.0
        
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        
        return VariableUsage.UNKNOWN
    
    def _analyze_access_patterns(self, instructions: List[str], register: str) -> List[str]:
        """Analyze how variable is accessed"""
        patterns = []
        
        for instruction in instructions:
            if not register or register.lower() not in instruction.lower():
                continue
            
            instruction_lower = instruction.lower()
            
            # Direct access
            if f'mov {register.lower()}' in instruction_lower:
                patterns.append('direct_load')
            elif f'mov.*{register.lower()}' in instruction_lower:
                patterns.append('direct_store')
            
            # Array access
            if f'[{register.lower()}+' in instruction_lower or f'[{register.lower()} +' in instruction_lower:
                patterns.append('array_access')
            
            # Pointer dereference
            if f'[{register.lower()}]' in instruction_lower:
                patterns.append('pointer_dereference')
            
            # Arithmetic operations
            if any(op in instruction_lower for op in ['add', 'sub', 'inc', 'dec']):
                if register.lower() in instruction_lower:
                    patterns.append('arithmetic_operation')
            
            # Comparison
            if 'cmp' in instruction_lower and register.lower() in instruction_lower:
                patterns.append('comparison')
            
            # Function parameter
            if 'push' in instruction_lower and register.lower() in instruction_lower:
                patterns.append('function_parameter')
        
        return list(set(patterns))
    
    def _detect_scope_from_memory(self, memory_location: str) -> VariableScope:
        """Detect variable scope from memory location"""
        mem_lower = memory_location.lower()
        
        if 'ebp+' in mem_lower or 'rbp+' in mem_lower:
            offset_match = re.search(r'[er]bp\+(\d+)', mem_lower)
            if offset_match:
                offset = int(offset_match.group(1))
                if offset >= 8:
                    return VariableScope.PARAMETER
                else:
                    return VariableScope.LOCAL
        
        if 'ebp-' in mem_lower or 'rbp-' in mem_lower:
            return VariableScope.LOCAL
        
        if 'esp' in mem_lower or 'rsp' in mem_lower:
            return VariableScope.LOCAL
        
        # Global address patterns
        if re.match(r'^0x[0-9a-f]+$', mem_lower):
            return VariableScope.GLOBAL
        
        return VariableScope.UNKNOWN
    
    def _refine_data_type(self, initial_type: VariableType, operations: List[str], access_patterns: List[str]) -> VariableType:
        """Refine data type based on operations and access patterns"""
        # Float operations
        if any(op.startswith('f') for op in operations if op not in ['free']):
            return VariableType.FLOAT
        
        # String operations
        if any(op in operations for op in ['rep', 'movs', 'cmps', 'scas', 'stos']):
            return VariableType.STRING
        
        # Array access patterns
        if 'array_access' in access_patterns:
            return VariableType.ARRAY
        
        # Pointer patterns
        if 'pointer_dereference' in access_patterns:
            return VariableType.POINTER
        
        # Boolean patterns
        if any(op in operations for op in ['test', 'and', 'or']) and 'comparison' in access_patterns:
            return VariableType.BOOLEAN
        
        return initial_type
    
    def _extract_related_functions(self, instructions: List[str]) -> List[str]:
        """Extract function calls related to variable"""
        functions = []
        
        for instruction in instructions:
            if 'call' in instruction.lower():
                parts = instruction.split()
                if len(parts) >= 2:
                    func_name = parts[-1]
                    functions.append(func_name)
        
        return list(set(functions))
    
    def _detect_size_hints(self, operations: List[str], access_patterns: List[str]) -> Optional[int]:
        """Detect size hints from operations"""
        # Look for common size patterns
        if 'arithmetic_operation' in access_patterns:
            # Check for pointer arithmetic with specific increments
            for op in operations:
                if 'add' in op:
                    # Extract immediate values that might indicate size
                    if '4' in op:
                        return 4  # 32-bit
                    elif '8' in op:
                        return 8  # 64-bit
                    elif '1' in op:
                        return 1  # byte
        
        return None


class SmartVariableNamer:
    """AI-powered variable naming system"""
    
    def __init__(self):
        self.logger = logging.getLogger("SmartVariableNamer")
        self.analyzer = VariableAnalyzer()
        self.name_templates = self._build_name_templates()
        self.semantic_database = self._build_semantic_database()
        self.naming_cache = {}
    
    def _build_name_templates(self) -> Dict[VariableUsage, List[str]]:
        """Build templates for different usage patterns"""
        return {
            VariableUsage.COUNTER: ['count', 'counter', 'i', 'j', 'k', 'n'],
            VariableUsage.INDEX: ['index', 'idx', 'pos', 'position', 'i', 'j'],
            VariableUsage.BUFFER: ['buffer', 'buf', 'data', 'temp_buffer', 'work_area'],
            VariableUsage.SIZE: ['size', 'length', 'len', 'count', 'num_elements'],
            VariableUsage.FLAG: ['flag', 'is_valid', 'enabled', 'active', 'success'],
            VariableUsage.RESULT: ['result', 'return_value', 'output', 'ret_val'],
            VariableUsage.TEMPORARY: ['temp', 'tmp', 'temp_var', 'scratch'],
            VariableUsage.ITERATOR: ['iter', 'iterator', 'current', 'ptr', 'walker'],
            VariableUsage.ACCUMULATOR: ['sum', 'total', 'accumulator', 'acc', 'aggregate'],
            VariableUsage.HANDLE: ['handle', 'hndl', 'resource', 'descriptor', 'fd'],
            VariableUsage.POINTER_TO_DATA: ['data_ptr', 'ptr', 'address', 'reference'],
            VariableUsage.CONFIG: ['config', 'settings', 'params', 'options'],
            VariableUsage.STATUS: ['status', 'state', 'condition', 'error_code']
        }
    
    def _build_semantic_database(self) -> Dict[str, List[str]]:
        """Build semantic associations for better naming"""
        return {
            'file_operations': ['file', 'fp', 'fd', 'handle', 'stream'],
            'string_operations': ['str', 'text', 'message', 'buffer', 'line'],
            'math_operations': ['value', 'number', 'operand', 'result', 'sum'],
            'memory_operations': ['ptr', 'address', 'buffer', 'block', 'chunk'],
            'network_operations': ['socket', 'connection', 'packet', 'data'],
            'ui_operations': ['window', 'control', 'widget', 'element'],
            'database_operations': ['record', 'row', 'field', 'query', 'connection']
        }
    
    def suggest_variable_name(self, variable_data: Dict[str, Any], function_context: Dict[str, Any] = None) -> NamingResult:
        """Suggest intelligent variable name based on context"""
        # Create cache key
        cache_key = self._create_naming_cache_key(variable_data, function_context)
        if cache_key in self.naming_cache:
            return self.naming_cache[cache_key]
        
        try:
            # Analyze variable context
            context = self.analyzer.analyze_variable_context(variable_data, function_context or {})
            
            # Generate base name suggestions
            base_suggestions = self._generate_base_suggestions(context, function_context)
            
            # Apply naming conventions
            final_suggestions = self._apply_naming_conventions(base_suggestions, context)
            
            # Score and rank suggestions
            scored_suggestions = self._score_suggestions(final_suggestions, context, function_context)
            
            # Select best suggestion
            best_name, best_score = scored_suggestions[0] if scored_suggestions else ('var', 0.3)
            
            # Generate reasoning
            reasoning = self._generate_naming_reasoning(best_name, context)
            
            # Determine naming convention used
            naming_convention = self._determine_naming_convention(best_name)
            
            # Get alternatives
            alternatives = scored_suggestions[1:4] if len(scored_suggestions) > 1 else []
            
            result = NamingResult(
                suggested_name=best_name,
                confidence=best_score,
                reasoning=reasoning,
                data_type=context.data_type,
                scope=context.scope,
                usage_pattern=context.usage_pattern,
                alternatives=alternatives,
                naming_convention=naming_convention
            )
            
            # Cache result
            self.naming_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Variable naming failed: {e}")
            return NamingResult(
                suggested_name='var',
                confidence=0.0,
                reasoning=f"Naming error: {str(e)}",
                data_type=VariableType.UNKNOWN,
                scope=VariableScope.UNKNOWN,
                usage_pattern=VariableUsage.UNKNOWN,
                alternatives=[],
                naming_convention='camelCase'
            )
    
    def _create_naming_cache_key(self, variable_data: Dict[str, Any], function_context: Dict[str, Any]) -> str:
        """Create cache key for naming"""
        key_data = {
            'register': variable_data.get('register', ''),
            'memory': variable_data.get('memory_location', ''),
            'function_name': function_context.get('name', '') if function_context else ''
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def _generate_base_suggestions(self, context: VariableContext, function_context: Dict[str, Any]) -> List[str]:
        """Generate base name suggestions"""
        suggestions = []
        
        # Suggestions based on usage pattern
        usage_templates = self.name_templates.get(context.usage_pattern, ['var'])
        suggestions.extend(usage_templates)
        
        # Suggestions based on register semantics
        if context.register:
            reg_info = self.analyzer.register_map.get(context.register.lower(), {})
            typical_use = reg_info.get('typical_use')
            if typical_use:
                suggestions.append(typical_use.replace('_', ''))
        
        # Suggestions based on data type
        type_suggestions = {
            VariableType.POINTER: ['ptr', 'pointer', 'address'],
            VariableType.ARRAY: ['array', 'list', 'items'],
            VariableType.STRING: ['str', 'text', 'string'],
            VariableType.INTEGER: ['num', 'value', 'int_val'],
            VariableType.FLOAT: ['float_val', 'decimal', 'real'],
            VariableType.BOOLEAN: ['flag', 'is_set', 'enabled']
        }
        
        if context.data_type in type_suggestions:
            suggestions.extend(type_suggestions[context.data_type])
        
        # Suggestions based on function context
        if function_context:
            func_name = function_context.get('name', '')
            if func_name:
                semantic_suggestions = self._get_semantic_suggestions(func_name)
                suggestions.extend(semantic_suggestions)
        
        # Suggestions based on related functions
        for func in context.related_functions:
            semantic_suggestions = self._get_semantic_suggestions(func)
            suggestions.extend(semantic_suggestions[:2])  # Limit to avoid too many
        
        # Remove duplicates while preserving order
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:10]  # Limit to top 10 base suggestions
    
    def _get_semantic_suggestions(self, function_name: str) -> List[str]:
        """Get semantic suggestions based on function name"""
        suggestions = []
        func_lower = function_name.lower()
        
        for category, names in self.semantic_database.items():
            if any(keyword in func_lower for keyword in category.split('_')):
                suggestions.extend(names[:3])  # Limit per category
        
        return suggestions
    
    def _apply_naming_conventions(self, base_suggestions: List[str], context: VariableContext) -> List[str]:
        """Apply naming conventions to base suggestions"""
        conventions = self.analyzer.naming_conventions
        final_suggestions = []
        
        for convention_name, templates in conventions.items():
            for suggestion in base_suggestions:
                # Apply different convention styles
                if context.scope == VariableScope.PARAMETER:
                    # Parameters often use camelCase or snake_case
                    if convention_name in ['camelCase', 'snake_case']:
                        formatted = self._format_name(suggestion, convention_name, templates)
                        final_suggestions.append(formatted)
                elif context.scope == VariableScope.GLOBAL:
                    # Globals often use snake_case or PascalCase
                    if convention_name in ['snake_case', 'PascalCase']:
                        formatted = self._format_name(suggestion, convention_name, templates)
                        final_suggestions.append(formatted)
                else:
                    # Local variables - use all conventions
                    formatted = self._format_name(suggestion, convention_name, templates)
                    final_suggestions.append(formatted)
        
        return final_suggestions
    
    def _format_name(self, base_name: str, convention: str, templates: Dict[str, str]) -> str:
        """Format name according to convention"""
        if convention == 'camelCase':
            return base_name
        elif convention == 'snake_case':
            return base_name.lower().replace(' ', '_')
        elif convention == 'PascalCase':
            return base_name.capitalize()
        elif convention == 'hungarian':
            # Apply Hungarian notation based on data type
            template = templates.get('default', '{name}')
            return template.format(name=base_name)
        else:
            return base_name
    
    def _score_suggestions(self, suggestions: List[str], context: VariableContext, function_context: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Score and rank name suggestions"""
        scored = []
        
        for suggestion in suggestions:
            score = self._calculate_name_score(suggestion, context, function_context)
            scored.append((suggestion, score))
        
        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def _calculate_name_score(self, name: str, context: VariableContext, function_context: Dict[str, Any]) -> float:
        """Calculate quality score for a variable name"""
        score = 0.5  # Base score
        
        # Length penalty/bonus
        if 3 <= len(name) <= 12:
            score += 0.2
        elif len(name) < 3:
            score -= 0.2
        elif len(name) > 15:
            score -= 0.1
        
        # Descriptiveness bonus
        if len(name) > 4 and not name.startswith(('temp', 'tmp', 'var')):
            score += 0.2
        
        # Pattern matching bonus
        usage_templates = self.name_templates.get(context.usage_pattern, [])
        if name in usage_templates:
            score += 0.3
        
        # Data type consistency
        type_keywords = {
            VariableType.POINTER: ['ptr', 'pointer', 'addr'],
            VariableType.ARRAY: ['array', 'list', 'arr'],
            VariableType.STRING: ['str', 'string', 'text'],
            VariableType.INTEGER: ['num', 'int', 'count'],
            VariableType.FLOAT: ['float', 'real', 'decimal'],
            VariableType.BOOLEAN: ['flag', 'is_', 'enabled']
        }
        
        if context.data_type in type_keywords:
            if any(keyword in name.lower() for keyword in type_keywords[context.data_type]):
                score += 0.2
        
        # Scope appropriateness
        if context.scope == VariableScope.PARAMETER and len(name) <= 8:
            score += 0.1
        elif context.scope == VariableScope.GLOBAL and len(name) >= 5:
            score += 0.1
        
        # Convention consistency
        if self._is_consistent_convention(name):
            score += 0.1
        
        # Avoid generic names for complex contexts
        if name in ['var', 'temp', 'tmp'] and context.usage_pattern != VariableUsage.TEMPORARY:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _is_consistent_convention(self, name: str) -> bool:
        """Check if name follows consistent naming convention"""
        # camelCase
        if re.match(r'^[a-z][a-zA-Z0-9]*$', name):
            return True
        
        # snake_case
        if re.match(r'^[a-z][a-z0-9_]*$', name):
            return True
        
        # PascalCase
        if re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            return True
        
        return False
    
    def _generate_naming_reasoning(self, name: str, context: VariableContext) -> str:
        """Generate reasoning for the chosen name"""
        reasons = []
        
        # Usage pattern reasoning
        if context.usage_pattern != VariableUsage.UNKNOWN:
            reasons.append(f"Named as {context.usage_pattern.value} based on usage pattern")
        
        # Data type reasoning
        if context.data_type != VariableType.UNKNOWN:
            reasons.append(f"Typed as {context.data_type.value} from analysis")
        
        # Scope reasoning
        if context.scope != VariableScope.UNKNOWN:
            reasons.append(f"Scoped as {context.scope.value} variable")
        
        # Register reasoning
        if context.register:
            reasons.append(f"Based on {context.register} register usage")
        
        # Operations reasoning
        if context.operations:
            op_summary = ', '.join(set(context.operations[:3]))
            reasons.append(f"Operations: {op_summary}")
        
        if not reasons:
            reasons.append("Generated from general naming patterns")
        
        return '; '.join(reasons)
    
    def _determine_naming_convention(self, name: str) -> str:
        """Determine which naming convention was used"""
        if re.match(r'^[a-z][a-zA-Z0-9]*$', name):
            return 'camelCase'
        elif re.match(r'^[a-z][a-z0-9_]*$', name):
            return 'snake_case'
        elif re.match(r'^[A-Z][a-zA-Z0-9]*$', name):
            return 'PascalCase'
        elif re.match(r'^[a-z][A-Z]', name):
            return 'hungarian'
        else:
            return 'custom'
    
    def get_namer_stats(self) -> Dict[str, Any]:
        """Get statistics about the variable namer"""
        return {
            'supported_types': [t.value for t in VariableType],
            'supported_scopes': [s.value for s in VariableScope],
            'supported_usages': [u.value for u in VariableUsage],
            'cached_names': len(self.naming_cache),
            'naming_conventions': list(self.analyzer.naming_conventions.keys()),
            'semantic_categories': list(self.semantic_database.keys())
        }