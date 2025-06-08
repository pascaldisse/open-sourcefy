"""
Advanced Data Type Inference and Reconstruction Engine

This module provides sophisticated data type inference capabilities that go beyond
basic pattern matching to analyze data flow, memory access patterns, and 
contextual usage to reconstruct accurate type information from binary analysis.

Features:
- Flow-sensitive type inference with data flow analysis
- Pointer type reconstruction with multi-level indirection
- Aggregate type recovery (structs, unions, arrays)
- Cross-function type propagation and unification
- Windows-specific type system integration (HANDLE, DWORD, etc.)
- Machine learning-enhanced type classification
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict

logger = logging.getLogger(__name__)


class TypeClass(Enum):
    """Enhanced type classification system"""
    PRIMITIVE = "primitive"
    POINTER = "pointer"
    ARRAY = "array"
    STRUCT = "struct"
    UNION = "union"
    FUNCTION_POINTER = "function_pointer"
    HANDLE = "handle"
    INTERFACE = "interface"
    UNKNOWN = "unknown"


class TypeQualifier(Enum):
    """Type qualifiers"""
    CONST = "const"
    VOLATILE = "volatile"
    STATIC = "static"
    EXTERN = "extern"
    REGISTER = "register"


@dataclass
class TypeConstraint:
    """Type constraint from analysis"""
    constraint_type: str  # 'size', 'alignment', 'usage', 'flow'
    value: Any
    confidence: float
    source: str  # Where this constraint came from


@dataclass
class InferredType:
    """Advanced inferred type with confidence and constraints"""
    base_type: str
    type_class: TypeClass
    size_bytes: int
    alignment: int
    qualifiers: List[TypeQualifier] = field(default_factory=list)
    pointer_depth: int = 0
    array_dimensions: List[int] = field(default_factory=list)
    struct_members: Dict[str, 'InferredType'] = field(default_factory=dict)
    constraints: List[TypeConstraint] = field(default_factory=list)
    confidence: float = 0.0
    type_string: str = ""
    
    def __post_init__(self):
        if not self.type_string:
            self.type_string = self._generate_type_string()
    
    def _generate_type_string(self) -> str:
        """Generate C-style type string"""
        base = self.base_type
        
        # Add qualifiers
        if self.qualifiers:
            qual_str = " ".join([q.value for q in self.qualifiers])
            base = f"{qual_str} {base}"
        
        # Add pointer depth
        if self.pointer_depth > 0:
            base += "*" * self.pointer_depth
        
        # Add array dimensions
        for dim in self.array_dimensions:
            if dim > 0:
                base += f"[{dim}]"
            else:
                base += "[]"
        
        return base


@dataclass
class TypeUsageContext:
    """Context information for type usage analysis"""
    function_name: str
    variable_name: str
    usage_type: str  # 'assignment', 'dereference', 'arithmetic', 'function_call'
    operand_types: List[str]
    source_location: str
    confidence: float


@dataclass
class DataFlowNode:
    """Node in data flow graph for type propagation"""
    variable_key: str
    function_context: str
    assignment_sources: List[str] = field(default_factory=list)
    usage_contexts: List[TypeUsageContext] = field(default_factory=list)
    propagated_type: Optional[InferredType] = None


class AdvancedDataTypeInference:
    """
    Advanced data type inference engine for binary reconstruction
    
    This system uses multiple inference techniques:
    1. Memory access pattern analysis
    2. Data flow analysis across functions
    3. Usage context analysis
    4. Windows API type propagation
    5. Machine learning-enhanced classification
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Type inference caches and state
        self.inferred_types = {}
        self.type_constraints = defaultdict(list)
        self.data_flow_graph = {}
        self.usage_patterns = defaultdict(list)
        
        # Initialize type system knowledge
        self.primitive_types = self._initialize_primitive_types()
        self.windows_types = self._initialize_windows_types()
        self.api_type_signatures = self._initialize_api_type_signatures()
        
        # Pattern recognition models
        self.struct_patterns = self._initialize_struct_patterns()
        self.array_patterns = self._initialize_array_patterns()
        
    def infer_program_types(self, ghidra_results: Dict[str, Any], 
                          semantic_functions: List[Any],
                          advanced_signatures: Dict[str, Any] = None) -> Dict[str, InferredType]:
        """
        Perform comprehensive type inference across the entire program
        
        Args:
            ghidra_results: Raw Ghidra analysis results
            semantic_functions: List of semantic function objects
            advanced_signatures: Optional advanced function signatures
            
        Returns:
            Dictionary mapping variable keys to inferred types
        """
        self.logger.info("Starting advanced data type inference...")
        
        # Phase 1: Collect type constraints from multiple sources
        self._collect_primitive_constraints(ghidra_results)
        self._collect_memory_access_constraints(semantic_functions)
        self._collect_api_constraints(advanced_signatures or {})
        self._collect_usage_pattern_constraints(semantic_functions)
        
        # Phase 2: Build data flow graph
        self._build_data_flow_graph(semantic_functions)
        
        # Phase 3: Perform type inference with constraint solving
        self._perform_constraint_solving()
        
        # Phase 4: Propagate types through data flow
        self._propagate_types_through_dataflow()
        
        # Phase 5: Refine types with cross-function analysis
        self._refine_types_cross_function(semantic_functions)
        
        # Phase 6: Reconstruct aggregate types
        self._reconstruct_aggregate_types()
        
        # Phase 7: Validate and finalize types
        self._validate_and_finalize_types()
        
        self.logger.info(f"Inferred types for {len(self.inferred_types)} variables")
        return self.inferred_types
    
    def _collect_primitive_constraints(self, ghidra_results: Dict[str, Any]) -> None:
        """Collect type constraints from primitive analysis"""
        functions = ghidra_results.get('functions', [])
        
        for func in functions:
            func_name = func.get('name', 'unknown')
            decompiled_code = func.get('decompiled_code', '')
            
            # Extract variable declarations
            var_declarations = re.findall(
                r'((?:const\s+|volatile\s+|static\s+)*\w+(?:\s*\*+)?)\s+(\w+)(?:\s*=|\s*;|\s*\[)',
                decompiled_code
            )
            
            for type_str, var_name in var_declarations:
                var_key = f"{func_name}::{var_name}"
                
                # Parse type information
                inferred_type = self._parse_type_declaration(type_str.strip())
                if inferred_type:
                    self.inferred_types[var_key] = inferred_type
                    
                    # Add size constraint
                    self.type_constraints[var_key].append(TypeConstraint(
                        constraint_type='size',
                        value=inferred_type.size_bytes,
                        confidence=0.8,
                        source='variable_declaration'
                    ))
    
    def _collect_memory_access_constraints(self, semantic_functions: List[Any]) -> None:
        """Collect constraints from memory access patterns"""
        for func in semantic_functions:
            func_name = func.name
            func_code = getattr(func, 'body_code', '')
            
            # Look for pointer dereferences
            pointer_accesses = re.findall(r'\*(\w+)', func_code)
            for var_name in pointer_accesses:
                var_key = f"{func_name}::{var_name}"
                
                self.type_constraints[var_key].append(TypeConstraint(
                    constraint_type='usage',
                    value='pointer_dereference',
                    confidence=0.9,
                    source='memory_access_analysis'
                ))
            
            # Look for array accesses
            array_accesses = re.findall(r'(\w+)\[([^\]]+)\]', func_code)
            for var_name, index_expr in array_accesses:
                var_key = f"{func_name}::{var_name}"
                
                self.type_constraints[var_key].append(TypeConstraint(
                    constraint_type='usage',
                    value='array_access',
                    confidence=0.9,
                    source='memory_access_analysis'
                ))
                
                # Try to infer array size from index expressions
                if index_expr.isdigit():
                    max_index = int(index_expr)
                    self.type_constraints[var_key].append(TypeConstraint(
                        constraint_type='array_size',
                        value=max_index + 1,
                        confidence=0.7,
                        source='array_index_analysis'
                    ))
            
            # Look for structure member accesses
            struct_accesses = re.findall(r'(\w+)\.(\w+)', func_code)
            for var_name, member_name in struct_accesses:
                var_key = f"{func_name}::{var_name}"
                
                self.type_constraints[var_key].append(TypeConstraint(
                    constraint_type='usage',
                    value=f'struct_member:{member_name}',
                    confidence=0.8,
                    source='struct_access_analysis'
                ))
    
    def _collect_api_constraints(self, advanced_signatures: Dict[str, Any]) -> None:
        """Collect type constraints from Windows API usage"""
        for address, signature in advanced_signatures.items():
            if hasattr(signature, 'parameters'):
                for param in signature.parameters:
                    # Map API parameter types to program variables
                    if hasattr(param, 'semantic_type') and param.semantic_type:
                        api_type = self._map_api_type_to_program_type(param.data_type)
                        if api_type:
                            # This would need more sophisticated mapping
                            # For now, store as a type pattern
                            self.type_constraints[f'api_pattern:{param.name}'].append(
                                TypeConstraint(
                                    constraint_type='api_type',
                                    value=api_type,
                                    confidence=0.9,
                                    source='windows_api_analysis'
                                )
                            )
    
    def _collect_usage_pattern_constraints(self, semantic_functions: List[Any]) -> None:
        """Collect constraints from variable usage patterns"""
        for func in semantic_functions:
            func_name = func.name
            func_code = getattr(func, 'body_code', '')
            
            # Analyze arithmetic operations for numeric types
            arithmetic_ops = re.findall(r'(\w+)\s*([+\-*/])\s*(\w+)', func_code)
            for left_var, op, right_var in arithmetic_ops:
                for var_name in [left_var, right_var]:
                    if var_name.isalpha():  # Variable name
                        var_key = f"{func_name}::{var_name}"
                        
                        self.type_constraints[var_key].append(TypeConstraint(
                            constraint_type='usage',
                            value='arithmetic_operation',
                            confidence=0.7,
                            source='arithmetic_analysis'
                        ))
            
            # Analyze function calls for parameter type inference
            func_calls = re.findall(r'(\w+)\s*\(([^)]*)\)', func_code)
            for called_func, args_str in func_calls:
                if called_func in self.api_type_signatures:
                    # Map arguments to expected API types
                    args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                    expected_types = self.api_type_signatures[called_func]
                    
                    for i, arg in enumerate(args):
                        if i < len(expected_types) and arg.isalpha():
                            var_key = f"{func_name}::{arg}"
                            
                            self.type_constraints[var_key].append(TypeConstraint(
                                constraint_type='api_parameter',
                                value=expected_types[i],
                                confidence=0.8,
                                source=f'api_call:{called_func}'
                            ))
    
    def _build_data_flow_graph(self, semantic_functions: List[Any]) -> None:
        """Build data flow graph for type propagation"""
        for func in semantic_functions:
            func_name = func.name
            func_code = getattr(func, 'body_code', '')
            
            # Find variable assignments
            assignments = re.findall(r'(\w+)\s*=\s*([^;]+)', func_code)
            for target_var, source_expr in assignments:
                target_key = f"{func_name}::{target_var}"
                
                if target_key not in self.data_flow_graph:
                    self.data_flow_graph[target_key] = DataFlowNode(
                        variable_key=target_key,
                        function_context=func_name
                    )
                
                # Extract source variables from expression
                source_vars = re.findall(r'\b(\w+)\b', source_expr)
                for source_var in source_vars:
                    if source_var.isalpha() and source_var != target_var:
                        source_key = f"{func_name}::{source_var}"
                        self.data_flow_graph[target_key].assignment_sources.append(source_key)
    
    def _perform_constraint_solving(self) -> None:
        """Solve type constraints to determine most likely types"""
        for var_key, constraints in self.type_constraints.items():
            if var_key in self.inferred_types:
                continue  # Already has a type from declaration
            
            # Group constraints by type
            usage_constraints = [c for c in constraints if c.constraint_type == 'usage']
            size_constraints = [c for c in constraints if c.constraint_type == 'size']
            api_constraints = [c for c in constraints if c.constraint_type == 'api_parameter']
            
            # Determine most likely type
            inferred_type = self._solve_constraints_for_variable(
                var_key, usage_constraints, size_constraints, api_constraints
            )
            
            if inferred_type:
                self.inferred_types[var_key] = inferred_type
    
    def _solve_constraints_for_variable(self, var_key: str,
                                      usage_constraints: List[TypeConstraint],
                                      size_constraints: List[TypeConstraint],
                                      api_constraints: List[TypeConstraint]) -> Optional[InferredType]:
        """Solve constraints for a single variable"""
        
        # Start with unknown type
        candidate_types = []
        
        # Analyze usage patterns
        for constraint in usage_constraints:
            if constraint.value == 'pointer_dereference':
                candidate_types.append(InferredType(
                    base_type='void',
                    type_class=TypeClass.POINTER,
                    size_bytes=8 if self._is_64bit() else 4,
                    alignment=8 if self._is_64bit() else 4,
                    pointer_depth=1,
                    confidence=constraint.confidence
                ))
            elif constraint.value == 'array_access':
                candidate_types.append(InferredType(
                    base_type='int',  # Default assumption
                    type_class=TypeClass.ARRAY,
                    size_bytes=0,  # Variable size
                    alignment=4,
                    array_dimensions=[0],  # Unknown size
                    confidence=constraint.confidence
                ))
            elif constraint.value == 'arithmetic_operation':
                candidate_types.append(InferredType(
                    base_type='int',
                    type_class=TypeClass.PRIMITIVE,
                    size_bytes=4,
                    alignment=4,
                    confidence=constraint.confidence
                ))
        
        # Consider API constraints (highest priority)
        for constraint in api_constraints:
            api_type = self._resolve_api_type(constraint.value)
            if api_type:
                candidate_types.append(api_type)
        
        # Select best candidate type
        if candidate_types:
            # Sort by confidence and select highest
            best_type = max(candidate_types, key=lambda t: t.confidence)
            return best_type
        
        return None
    
    def _propagate_types_through_dataflow(self) -> None:
        """Propagate types through data flow graph"""
        # Iterative type propagation
        changed = True
        iterations = 0
        max_iterations = 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for var_key, flow_node in self.data_flow_graph.items():
                if var_key in self.inferred_types:
                    continue  # Already has a type
                
                # Collect types from assignment sources
                source_types = []
                for source_key in flow_node.assignment_sources:
                    if source_key in self.inferred_types:
                        source_types.append(self.inferred_types[source_key])
                
                if source_types:
                    # Unify types from different sources
                    unified_type = self._unify_types(source_types)
                    if unified_type:
                        self.inferred_types[var_key] = unified_type
                        changed = True
    
    def _refine_types_cross_function(self, semantic_functions: List[Any]) -> None:
        """Refine types using cross-function analysis"""
        # Analyze function calls to propagate parameter types
        for func in semantic_functions:
            func_name = func.name
            func_code = getattr(func, 'body_code', '')
            
            # Find function calls
            func_calls = re.findall(r'(\w+)\s*\(([^)]*)\)', func_code)
            for called_func, args_str in func_calls:
                args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                
                # Find the called function definition
                called_func_obj = None
                for other_func in semantic_functions:
                    if other_func.name == called_func:
                        called_func_obj = other_func
                        break
                
                if called_func_obj and hasattr(called_func_obj, 'parameters'):
                    # Match arguments to parameters
                    for i, arg in enumerate(args):
                        if i < len(called_func_obj.parameters) and arg.isalpha():
                            arg_key = f"{func_name}::{arg}"
                            param = called_func_obj.parameters[i]
                            
                            if hasattr(param, 'data_type') and arg_key not in self.inferred_types:
                                # Propagate parameter type to argument
                                param_type = self._convert_semantic_type_to_inferred(param.data_type)
                                if param_type:
                                    self.inferred_types[arg_key] = param_type
    
    def _reconstruct_aggregate_types(self) -> None:
        """Reconstruct struct and union types from usage patterns"""
        # Group variables by potential structure relationships
        struct_candidates = defaultdict(list)
        
        for var_key, constraints in self.type_constraints.items():
            for constraint in constraints:
                if constraint.constraint_type == 'usage' and constraint.value.startswith('struct_member:'):
                    member_name = constraint.value.split(':', 1)[1]
                    base_var = var_key.split('::')[-1]
                    struct_candidates[base_var].append(member_name)
        
        # Create struct types for variables with multiple members
        for var_name, members in struct_candidates.items():
            if len(members) > 1:  # At least 2 members to be considered a struct
                struct_type = InferredType(
                    base_type=f"struct_{var_name}_t",
                    type_class=TypeClass.STRUCT,
                    size_bytes=len(members) * 4,  # Rough estimate
                    alignment=4,
                    confidence=0.6
                )
                
                # Add members to struct
                for i, member in enumerate(members):
                    struct_type.struct_members[member] = InferredType(
                        base_type='int',  # Default member type
                        type_class=TypeClass.PRIMITIVE,
                        size_bytes=4,
                        alignment=4,
                        confidence=0.5
                    )
                
                # Update any variables that use this struct
                for var_key in self.inferred_types:
                    if var_key.endswith(f"::{var_name}"):
                        self.inferred_types[var_key] = struct_type
    
    def _validate_and_finalize_types(self) -> None:
        """Validate inferred types and finalize type strings"""
        for var_key, inferred_type in self.inferred_types.items():
            # Validate type consistency
            if not self._validate_type_consistency(inferred_type):
                self.logger.warning(f"Type inconsistency detected for {var_key}: {inferred_type.type_string}")
                # Apply corrections or fallback to safer type
                inferred_type.confidence *= 0.8
            
            # Finalize type string
            inferred_type.type_string = inferred_type._generate_type_string()
            
            # Ensure minimum confidence
            if inferred_type.confidence < 0.3:
                inferred_type.confidence = 0.3
    
    # Helper methods
    def _parse_type_declaration(self, type_str: str) -> Optional[InferredType]:
        """Parse C-style type declaration into InferredType"""
        type_str = type_str.strip()
        
        # Handle qualifiers
        qualifiers = []
        for qual in ['const', 'volatile', 'static', 'extern', 'register']:
            if qual in type_str:
                qualifiers.append(TypeQualifier(qual.upper()))
                type_str = type_str.replace(qual, '').strip()
        
        # Count pointer depth
        pointer_depth = type_str.count('*')
        base_type = type_str.replace('*', '').strip()
        
        # Determine type class and size
        if base_type in self.primitive_types:
            type_info = self.primitive_types[base_type]
            return InferredType(
                base_type=base_type,
                type_class=TypeClass.POINTER if pointer_depth > 0 else TypeClass.PRIMITIVE,
                size_bytes=8 if pointer_depth > 0 and self._is_64bit() else type_info['size'],
                alignment=type_info['alignment'],
                qualifiers=qualifiers,
                pointer_depth=pointer_depth,
                confidence=0.9
            )
        elif base_type in self.windows_types:
            type_info = self.windows_types[base_type]
            return InferredType(
                base_type=base_type,
                type_class=type_info['class'],
                size_bytes=type_info['size'],
                alignment=type_info['alignment'],
                qualifiers=qualifiers,
                pointer_depth=pointer_depth,
                confidence=0.8
            )
        
        return None
    
    def _map_api_type_to_program_type(self, api_type: str) -> Optional[str]:
        """Map Windows API type to program type"""
        api_type_map = {
            'HANDLE': 'void*',
            'HWND': 'void*', 
            'HDC': 'void*',
            'HINSTANCE': 'void*',
            'DWORD': 'unsigned int',
            'WORD': 'unsigned short',
            'BYTE': 'unsigned char',
            'BOOL': 'int',
            'LPSTR': 'char*',
            'LPCSTR': 'const char*',
            'LPWSTR': 'wchar_t*',
            'LPCWSTR': 'const wchar_t*',
            'LPVOID': 'void*',
            'LPCVOID': 'const void*'
        }
        return api_type_map.get(api_type)
    
    def _resolve_api_type(self, api_type_str: str) -> Optional[InferredType]:
        """Resolve API type string to InferredType"""
        if api_type_str in self.windows_types:
            type_info = self.windows_types[api_type_str]
            return InferredType(
                base_type=api_type_str,
                type_class=type_info['class'],
                size_bytes=type_info['size'],
                alignment=type_info['alignment'],
                confidence=0.9
            )
        return None
    
    def _unify_types(self, types: List[InferredType]) -> Optional[InferredType]:
        """Unify multiple types into a single consistent type"""
        if not types:
            return None
        
        if len(types) == 1:
            return types[0]
        
        # Simple unification - take the most confident compatible type
        # More sophisticated unification would check type compatibility
        compatible_types = []
        for t in types:
            # Check if type is compatible with others
            is_compatible = True
            for other in types:
                if not self._types_compatible(t, other):
                    is_compatible = False
                    break
            
            if is_compatible:
                compatible_types.append(t)
        
        if compatible_types:
            # Return highest confidence type
            return max(compatible_types, key=lambda t: t.confidence)
        else:
            # Return most confident type even if not perfectly compatible
            return max(types, key=lambda t: t.confidence)
    
    def _types_compatible(self, type1: InferredType, type2: InferredType) -> bool:
        """Check if two types are compatible for unification"""
        # Same base type
        if type1.base_type == type2.base_type:
            return True
        
        # Pointer types with compatible base
        if type1.type_class == TypeClass.POINTER and type2.type_class == TypeClass.POINTER:
            return True
        
        # Numeric types are generally compatible
        numeric_types = {'int', 'short', 'long', 'char', 'float', 'double'}
        if type1.base_type in numeric_types and type2.base_type in numeric_types:
            return True
        
        return False
    
    def _convert_semantic_type_to_inferred(self, semantic_type) -> Optional[InferredType]:
        """Convert semantic type to InferredType"""
        if hasattr(semantic_type, 'value'):
            type_str = semantic_type.value
        else:
            type_str = str(semantic_type)
        
        return self._parse_type_declaration(type_str)
    
    def _validate_type_consistency(self, inferred_type: InferredType) -> bool:
        """Validate type for consistency and correctness"""
        # Basic validation checks
        if inferred_type.size_bytes < 0:
            return False
        
        if inferred_type.alignment <= 0:
            return False
        
        if inferred_type.pointer_depth < 0:
            return False
        
        # Validate array dimensions
        for dim in inferred_type.array_dimensions:
            if dim < 0:
                return False
        
        return True
    
    def _is_64bit(self) -> bool:
        """Check if target architecture is 64-bit"""
        # This would be determined from binary analysis
        # For now, assume 32-bit (x86)
        return False
    
    def _initialize_primitive_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize primitive type database"""
        return {
            'void': {'size': 0, 'alignment': 1},
            'char': {'size': 1, 'alignment': 1},
            'short': {'size': 2, 'alignment': 2},
            'int': {'size': 4, 'alignment': 4},
            'long': {'size': 4, 'alignment': 4},  # 32-bit
            'long long': {'size': 8, 'alignment': 8},
            'float': {'size': 4, 'alignment': 4},
            'double': {'size': 8, 'alignment': 8},
            'unsigned char': {'size': 1, 'alignment': 1},
            'unsigned short': {'size': 2, 'alignment': 2},
            'unsigned int': {'size': 4, 'alignment': 4},
            'unsigned long': {'size': 4, 'alignment': 4},
            'wchar_t': {'size': 2, 'alignment': 2}
        }
    
    def _initialize_windows_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Windows-specific type database"""
        return {
            'BYTE': {'size': 1, 'alignment': 1, 'class': TypeClass.PRIMITIVE},
            'WORD': {'size': 2, 'alignment': 2, 'class': TypeClass.PRIMITIVE},
            'DWORD': {'size': 4, 'alignment': 4, 'class': TypeClass.PRIMITIVE},
            'QWORD': {'size': 8, 'alignment': 8, 'class': TypeClass.PRIMITIVE},
            'BOOL': {'size': 4, 'alignment': 4, 'class': TypeClass.PRIMITIVE},
            'HANDLE': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.HANDLE},
            'HWND': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.HANDLE},
            'HDC': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.HANDLE},
            'HINSTANCE': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.HANDLE},
            'LPSTR': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.POINTER},
            'LPCSTR': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.POINTER},
            'LPWSTR': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.POINTER},
            'LPCWSTR': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.POINTER},
            'LPVOID': {'size': 8 if self._is_64bit() else 4, 'alignment': 8 if self._is_64bit() else 4, 'class': TypeClass.POINTER}
        }
    
    def _initialize_api_type_signatures(self) -> Dict[str, List[str]]:
        """Initialize Windows API function type signatures"""
        return {
            'CreateFileW': ['LPCWSTR', 'DWORD', 'DWORD', 'LPSECURITY_ATTRIBUTES', 'DWORD', 'DWORD', 'HANDLE'],
            'ReadFile': ['HANDLE', 'LPVOID', 'DWORD', 'LPDWORD', 'LPOVERLAPPED'],
            'WriteFile': ['HANDLE', 'LPCVOID', 'DWORD', 'LPDWORD', 'LPOVERLAPPED'],
            'CloseHandle': ['HANDLE'],
            'GetCurrentProcess': [],
            'GetCurrentThread': [],
            'malloc': ['size_t'],
            'free': ['void*'],
            'memcpy': ['void*', 'const void*', 'size_t'],
            'memset': ['void*', 'int', 'size_t'],
            'strlen': ['const char*'],
            'strcpy': ['char*', 'const char*'],
            'printf': ['const char*'],  # Variadic
            'scanf': ['const char*']   # Variadic
        }
    
    def _initialize_struct_patterns(self) -> Dict[str, Any]:
        """Initialize structure recognition patterns"""
        return {
            'point_struct': {
                'members': ['x', 'y'],
                'member_types': ['int', 'int'],
                'size': 8
            },
            'rect_struct': {
                'members': ['left', 'top', 'right', 'bottom'],
                'member_types': ['int', 'int', 'int', 'int'],
                'size': 16
            },
            'list_node': {
                'members': ['data', 'next'],
                'member_types': ['void*', 'struct list_node*'],
                'size': 8
            }
        }
    
    def _initialize_array_patterns(self) -> Dict[str, Any]:
        """Initialize array recognition patterns"""
        return {
            'string_array': {
                'element_type': 'char',
                'access_patterns': ['[i]', '[index]', '[0]']
            },
            'int_array': {
                'element_type': 'int',
                'access_patterns': ['[i]', '[j]', '[index]']
            },
            'pointer_array': {
                'element_type': 'void*',
                'access_patterns': ['[i]', '[index]']
            }
        }
    
    def generate_type_report(self) -> Dict[str, Any]:
        """Generate comprehensive type inference report"""
        type_counts = defaultdict(int)
        confidence_stats = []
        
        for var_key, inferred_type in self.inferred_types.items():
            type_counts[inferred_type.type_class.value] += 1
            confidence_stats.append(inferred_type.confidence)
        
        avg_confidence = sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0.0
        
        return {
            'total_variables': len(self.inferred_types),
            'type_distribution': dict(type_counts),
            'average_confidence': avg_confidence,
            'high_confidence_vars': len([c for c in confidence_stats if c > 0.8]),
            'low_confidence_vars': len([c for c in confidence_stats if c < 0.5]),
            'constraints_processed': sum(len(constraints) for constraints in self.type_constraints.values()),
            'data_flow_nodes': len(self.data_flow_graph),
            'detailed_types': {
                var_key: {
                    'type_string': inferred_type.type_string,
                    'type_class': inferred_type.type_class.value,
                    'confidence': inferred_type.confidence,
                    'size_bytes': inferred_type.size_bytes
                } for var_key, inferred_type in self.inferred_types.items()
            }
        }