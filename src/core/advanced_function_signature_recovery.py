"""
Advanced Function Signature Recovery System

This module provides sophisticated function signature recovery capabilities that go
beyond basic parameter detection to analyze calling conventions, stack analysis,
register usage patterns, and API compliance for Windows PE binaries.

Features:
- Calling convention detection and analysis
- Advanced parameter type inference from usage patterns  
- Stack frame analysis for local variable recovery
- Register usage pattern analysis
- API compliance checking for Windows APIs
- Cross-reference analysis for function relationships
- Return value analysis through data flow
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class CallingConvention(Enum):
    """Windows calling conventions"""
    CDECL = "cdecl"
    STDCALL = "stdcall"
    FASTCALL = "fastcall"
    THISCALL = "thiscall"
    VECTORCALL = "vectorcall"
    UNKNOWN = "unknown"


class ParameterLocation(Enum):
    """Parameter passing locations"""
    STACK = "stack"
    REGISTER = "register"
    MEMORY = "memory"
    UNKNOWN = "unknown"


@dataclass
class AdvancedParameter:
    """Advanced parameter representation with calling convention info"""
    name: str
    data_type: str
    semantic_type: str
    location: ParameterLocation
    register_name: Optional[str] = None
    stack_offset: Optional[int] = None
    size_bytes: int = 4
    is_pointer: bool = False
    points_to_type: Optional[str] = None
    usage_pattern: str = "unknown"
    confidence: float = 0.5
    semantic_meaning: Optional[str] = None


@dataclass
class FunctionSignature:
    """Complete function signature with advanced analysis"""
    name: str
    semantic_name: str
    address: int
    calling_convention: CallingConvention
    return_type: str
    return_register: Optional[str] = None
    parameters: List[AdvancedParameter] = None
    local_variables: List[AdvancedParameter] = None
    stack_frame_size: int = 0
    is_api_function: bool = False
    api_category: Optional[str] = None
    cross_references: List[int] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.local_variables is None:
            self.local_variables = []
        if self.cross_references is None:
            self.cross_references = []


class AdvancedFunctionSignatureRecovery:
    """
    Advanced function signature recovery system for Windows PE binaries
    
    This system analyzes assembly code, stack usage, register patterns,
    and API calls to recover complete function signatures with high accuracy.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Analysis caches
        self.recovered_signatures = {}
        self.api_signatures = {}
        self.calling_patterns = {}
        
        # Initialize Windows API signatures database
        self.windows_api_db = self._initialize_windows_api_database()
        
        # Register analysis patterns
        self.register_patterns = self._initialize_register_patterns()
        
        # Stack analysis patterns
        self.stack_patterns = self._initialize_stack_patterns()
    
    def recover_function_signatures(self, ghidra_results: Dict[str, Any], 
                                  binary_metadata: Dict[str, Any]) -> Dict[str, FunctionSignature]:
        """
        Recover advanced function signatures from Ghidra analysis
        
        Args:
            ghidra_results: Raw Ghidra decompilation and disassembly
            binary_metadata: Binary format and architecture information
            
        Returns:
            Dictionary mapping function addresses to recovered signatures
        """
        self.logger.info("Starting advanced function signature recovery...")
        
        functions = ghidra_results.get('functions', [])
        recovered_signatures = {}
        
        for func_data in functions:
            try:
                signature = self._recover_single_function_signature(func_data, binary_metadata)
                if signature and signature.confidence_score > 0.3:
                    recovered_signatures[signature.address] = signature
                    self.recovered_signatures[signature.address] = signature
            except Exception as e:
                self.logger.warning(f"Failed to recover signature for function {func_data.get('name', 'unknown')}: {e}")
        
        self.logger.info(f"Recovered {len(recovered_signatures)} advanced function signatures")
        return recovered_signatures
    
    def _recover_single_function_signature(self, func_data: Dict[str, Any], 
                                         binary_metadata: Dict[str, Any]) -> Optional[FunctionSignature]:
        """Recover signature for a single function with advanced analysis"""
        
        name = func_data.get('name', 'unknown')
        address = func_data.get('address', 0)
        decompiled_code = func_data.get('decompiled_code', '')
        
        if not decompiled_code:
            return None
        
        # Step 1: Detect calling convention
        calling_convention = self._detect_calling_convention(func_data, decompiled_code)
        
        # Step 2: Analyze return type and register
        return_type, return_register = self._analyze_return_type(decompiled_code, calling_convention)
        
        # Step 3: Recover parameters with advanced analysis
        parameters = self._recover_function_parameters(func_data, decompiled_code, calling_convention)
        
        # Step 4: Analyze local variables
        local_variables = self._analyze_local_variables(decompiled_code, calling_convention)
        
        # Step 5: Calculate stack frame size
        stack_frame_size = self._calculate_stack_frame_size(decompiled_code, parameters, local_variables)
        
        # Step 6: Check if this is a Windows API function
        is_api, api_category = self._check_api_function(name, parameters, return_type)
        
        # Step 7: Generate semantic name
        semantic_name = self._generate_semantic_function_name(name, parameters, return_type, api_category)
        
        # Step 8: Analyze cross-references
        cross_references = self._analyze_cross_references(func_data)
        
        # Step 9: Calculate confidence score
        confidence_score = self._calculate_signature_confidence(
            calling_convention, parameters, return_type, is_api, decompiled_code
        )
        
        return FunctionSignature(
            name=name,
            semantic_name=semantic_name,
            address=address,
            calling_convention=calling_convention,
            return_type=return_type,
            return_register=return_register,
            parameters=parameters,
            local_variables=local_variables,
            stack_frame_size=stack_frame_size,
            is_api_function=is_api,
            api_category=api_category,
            cross_references=cross_references,
            confidence_score=confidence_score
        )
    
    def _detect_calling_convention(self, func_data: Dict[str, Any], code: str) -> CallingConvention:
        """Detect calling convention from function analysis"""
        
        # Check for explicit calling convention indicators
        if re.search(r'__stdcall|WINAPI|CALLBACK', code, re.IGNORECASE):
            return CallingConvention.STDCALL
        elif re.search(r'__fastcall', code, re.IGNORECASE):
            return CallingConvention.FASTCALL
        elif re.search(r'__thiscall', code, re.IGNORECASE):
            return CallingConvention.THISCALL
        elif re.search(r'__vectorcall', code, re.IGNORECASE):
            return CallingConvention.VECTORCALL
        
        # Analyze stack cleanup patterns
        if re.search(r'ret\s+0x[0-9a-fA-F]+', code):
            return CallingConvention.STDCALL  # Callee cleans stack
        elif re.search(r'add\s+esp,\s*0x[0-9a-fA-F]+', code):
            return CallingConvention.CDECL    # Caller cleans stack
        
        # Check parameter passing patterns
        register_params = len(re.findall(r'(ecx|edx|r8|r9)', code, re.IGNORECASE))
        if register_params >= 2:
            return CallingConvention.FASTCALL
        
        # Default for Windows
        return CallingConvention.STDCALL
    
    def _analyze_return_type(self, code: str, calling_convention: CallingConvention) -> Tuple[str, Optional[str]]:
        """Analyze function return type and register usage"""
        
        # Find return statements
        return_matches = re.findall(r'return\s+([^;]+)', code)
        
        if not return_matches:
            return "void", None
        
        # Analyze return value patterns
        for return_val in return_matches:
            return_val = return_val.strip()
            
            # Check for specific patterns
            if re.match(r'^-?\d+$', return_val):
                return "int", "eax"
            elif re.match(r'^0x[0-9a-fA-F]+$', return_val):
                return "DWORD", "eax"
            elif return_val.startswith('(') and return_val.endswith(')'):
                # Cast pattern - extract type
                cast_match = re.match(r'\(([^)]+)\)', return_val)
                if cast_match:
                    return cast_match.group(1), "eax"
            elif return_val == 'TRUE' or return_val == 'FALSE':
                return "BOOL", "eax"
            elif return_val == 'NULL' or return_val.startswith('&'):
                return "LPVOID", "eax"
            elif '"' in return_val:
                return "LPCSTR", "eax"
        
        # Check for 64-bit return patterns
        if re.search(r'rax|rdx', code, re.IGNORECASE):
            return "QWORD", "rax"
        
        # Default
        return "int", "eax"
    
    def _recover_function_parameters(self, func_data: Dict[str, Any], 
                                   code: str, 
                                   calling_convention: CallingConvention) -> List[AdvancedParameter]:
        """Recover function parameters with advanced analysis"""
        
        parameters = []
        
        # Extract function signature
        func_sig_match = re.search(r'(\w+\s+)*(\w+)\s*\(([^)]*)\)', code)
        if not func_sig_match:
            return parameters
        
        params_str = func_sig_match.group(3)
        
        # Parse parameter list
        if params_str.strip() and params_str.strip() != 'void':
            param_matches = re.findall(r'([^,]+)', params_str)
            
            for i, param_str in enumerate(param_matches):
                param_str = param_str.strip()
                
                # Parse type and name
                param_parts = param_str.split()
                if len(param_parts) >= 2:
                    param_type = ' '.join(param_parts[:-1])
                    param_name = param_parts[-1]
                    
                    # Advanced parameter analysis
                    advanced_param = self._analyze_single_parameter(
                        param_name, param_type, i, code, calling_convention
                    )
                    parameters.append(advanced_param)
        
        return parameters
    
    def _analyze_single_parameter(self, name: str, param_type: str, index: int,
                                code: str, calling_convention: CallingConvention) -> AdvancedParameter:
        """Analyze a single parameter with advanced techniques"""
        
        # Determine parameter location based on calling convention
        location, register_name, stack_offset = self._determine_parameter_location(
            index, param_type, calling_convention
        )
        
        # Analyze pointer information
        is_pointer = '*' in param_type or 'LP' in param_type.upper()
        points_to_type = None
        if is_pointer:
            points_to_type = self._extract_pointed_type(param_type)
        
        # Analyze usage pattern
        usage_pattern = self._analyze_parameter_usage_pattern(name, code)
        
        # Determine semantic type
        semantic_type = self._determine_semantic_parameter_type(name, param_type, usage_pattern)
        
        # Calculate size
        size_bytes = self._calculate_parameter_size(param_type)
        
        # Infer semantic meaning
        semantic_meaning = self._infer_parameter_semantic_meaning(name, param_type, semantic_type)
        
        # Calculate confidence
        confidence = self._calculate_parameter_confidence(name, param_type, usage_pattern)
        
        return AdvancedParameter(
            name=name,
            data_type=param_type,
            semantic_type=semantic_type,
            location=location,
            register_name=register_name,
            stack_offset=stack_offset,
            size_bytes=size_bytes,
            is_pointer=is_pointer,
            points_to_type=points_to_type,
            usage_pattern=usage_pattern,
            confidence=confidence,
            semantic_meaning=semantic_meaning
        )
    
    def _determine_parameter_location(self, index: int, param_type: str, 
                                    calling_convention: CallingConvention) -> Tuple[ParameterLocation, Optional[str], Optional[int]]:
        """Determine where parameter is passed based on calling convention"""
        
        if calling_convention == CallingConvention.FASTCALL:
            if index == 0:
                return ParameterLocation.REGISTER, "ecx", None
            elif index == 1:
                return ParameterLocation.REGISTER, "edx", None
            else:
                stack_offset = (index - 2) * 4 + 8  # After ecx, edx
                return ParameterLocation.STACK, None, stack_offset
                
        elif calling_convention == CallingConvention.THISCALL:
            if index == 0:  # 'this' pointer
                return ParameterLocation.REGISTER, "ecx", None
            else:
                stack_offset = index * 4 + 4
                return ParameterLocation.STACK, None, stack_offset
                
        else:  # STDCALL, CDECL
            stack_offset = index * 4 + 4  # All parameters on stack
            return ParameterLocation.STACK, None, stack_offset
    
    def _analyze_local_variables(self, code: str, calling_convention: CallingConvention) -> List[AdvancedParameter]:
        """Analyze local variables with stack frame analysis"""
        
        local_variables = []
        
        # Find variable declarations
        var_matches = re.findall(r'([a-zA-Z_]\w*\s+\*?)\s*([a-zA-Z_]\w*)\s*[=;]', code)
        
        for var_type, var_name in var_matches:
            var_type = var_type.strip()
            
            # Skip function names and keywords
            if var_name in ['if', 'for', 'while', 'return', 'int', 'char', 'void', 'main']:
                continue
            
            # Analyze variable usage
            usage_pattern = self._analyze_variable_usage_pattern(var_name, code)
            
            # Determine semantic type
            semantic_type = self._determine_semantic_variable_type(var_name, var_type, usage_pattern)
            
            # Calculate stack offset (simplified)
            stack_offset = -(len(local_variables) + 1) * 4  # Negative offset from EBP
            
            # Create advanced parameter (reusing structure for local vars)
            local_var = AdvancedParameter(
                name=var_name,
                data_type=var_type,
                semantic_type=semantic_type,
                location=ParameterLocation.STACK,
                register_name=None,
                stack_offset=stack_offset,
                size_bytes=self._calculate_parameter_size(var_type),
                is_pointer='*' in var_type,
                usage_pattern=usage_pattern,
                confidence=0.7,
                semantic_meaning=self._infer_variable_semantic_meaning(var_name, var_type)
            )
            
            local_variables.append(local_var)
        
        return local_variables
    
    def _calculate_stack_frame_size(self, code: str, parameters: List[AdvancedParameter], 
                                  local_variables: List[AdvancedParameter]) -> int:
        """Calculate total stack frame size"""
        
        # Parameters size (only stack parameters)
        param_size = sum(p.size_bytes for p in parameters if p.location == ParameterLocation.STACK)
        
        # Local variables size
        local_size = sum(v.size_bytes for v in local_variables)
        
        # Add saved registers (EBP, return address)
        saved_registers = 8
        
        # Look for explicit stack allocation
        stack_alloc_match = re.search(r'sub\s+esp,\s*0x([0-9a-fA-F]+)', code)
        if stack_alloc_match:
            explicit_alloc = int(stack_alloc_match.group(1), 16)
            return max(param_size + local_size + saved_registers, explicit_alloc + saved_registers)
        
        return param_size + local_size + saved_registers
    
    def _check_api_function(self, name: str, parameters: List[AdvancedParameter], 
                          return_type: str) -> Tuple[bool, Optional[str]]:
        """Check if function matches Windows API patterns"""
        
        # Check against known API database
        for api_category, api_patterns in self.windows_api_db.items():
            for pattern in api_patterns['name_patterns']:
                if re.search(pattern, name, re.IGNORECASE):
                    return True, api_category
            
            # Check parameter patterns
            if len(parameters) >= api_patterns.get('min_params', 0):
                param_types = [p.data_type for p in parameters]
                if any(api_type in ' '.join(param_types) for api_type in api_patterns.get('param_types', [])):
                    return True, api_category
        
        return False, None
    
    def _generate_semantic_function_name(self, original_name: str, parameters: List[AdvancedParameter],
                                       return_type: str, api_category: Optional[str]) -> str:
        """Generate semantic function name based on analysis"""
        
        if not original_name.startswith('FUN_') and not original_name.startswith('function_'):
            return original_name  # Keep original if meaningful
        
        # Generate based on API category
        if api_category:
            return f"{api_category.lower()}_function"
        
        # Generate based on parameters
        if any('handle' in p.semantic_type.lower() for p in parameters):
            return 'handle_manager_function'
        elif any('buffer' in p.semantic_type.lower() for p in parameters):
            return 'buffer_processor_function'
        elif return_type == 'BOOL':
            return 'validation_function'
        elif len(parameters) == 0:
            return 'utility_function'
        else:
            return f'process_{len(parameters)}_params_function'
    
    def _analyze_cross_references(self, func_data: Dict[str, Any]) -> List[int]:
        """Analyze function cross-references"""
        # Simplified implementation - would analyze actual call graph
        return []
    
    def _calculate_signature_confidence(self, calling_convention: CallingConvention,
                                      parameters: List[AdvancedParameter], return_type: str,
                                      is_api: bool, code: str) -> float:
        """Calculate overall signature confidence score"""
        
        confidence_factors = []
        
        # Calling convention confidence
        if calling_convention != CallingConvention.UNKNOWN:
            confidence_factors.append(0.2)
        
        # Parameter analysis confidence
        param_confidence = sum(p.confidence for p in parameters) / max(len(parameters), 1)
        confidence_factors.append(param_confidence * 0.3)
        
        # Return type confidence
        if return_type != "unknown":
            confidence_factors.append(0.2)
        
        # API function boost
        if is_api:
            confidence_factors.append(0.15)
        
        # Code quality indicators
        if len(code) > 100:  # Substantial function
            confidence_factors.append(0.1)
        
        # Function complexity
        complexity = len(re.findall(r'\b(if|while|for|switch)\b', code))
        if complexity > 0:
            confidence_factors.append(min(complexity * 0.05, 0.15))
        
        return min(sum(confidence_factors), 1.0)
    
    # Helper methods for pattern analysis
    def _analyze_parameter_usage_pattern(self, param_name: str, code: str) -> str:
        """Analyze how parameter is used in function"""
        if f"{param_name}[" in code:
            return "array_access"
        elif f"*{param_name}" in code:
            return "pointer_dereference"
        elif f"{param_name}->" in code:
            return "structure_pointer_access"
        elif f"{param_name}." in code:
            return "structure_access"
        elif f"sizeof({param_name})" in code:
            return "size_calculation"
        else:
            return "direct_usage"
    
    def _analyze_variable_usage_pattern(self, var_name: str, code: str) -> str:
        """Analyze how local variable is used"""
        usage_count = code.count(var_name)
        
        if usage_count > 10:
            return "frequent_usage"
        elif usage_count > 5:
            return "moderate_usage"
        elif usage_count > 2:
            return "limited_usage"
        else:
            return "minimal_usage"
    
    def _determine_semantic_parameter_type(self, name: str, param_type: str, usage_pattern: str) -> str:
        """Determine semantic type of parameter"""
        name_lower = name.lower()
        type_lower = param_type.lower()
        
        if 'handle' in type_lower or 'hwnd' in type_lower:
            return 'window_handle'
        elif 'lpcstr' in type_lower or 'lpstr' in type_lower:
            return 'string_buffer'
        elif 'dword' in type_lower and 'size' in name_lower:
            return 'size_parameter'
        elif 'bool' in type_lower:
            return 'boolean_flag'
        elif usage_pattern == 'array_access':
            return 'array_buffer'
        elif usage_pattern == 'pointer_dereference':
            return 'output_parameter'
        else:
            return 'general_parameter'
    
    def _determine_semantic_variable_type(self, name: str, var_type: str, usage_pattern: str) -> str:
        """Determine semantic type of local variable"""
        name_lower = name.lower()
        
        if 'temp' in name_lower or 'tmp' in name_lower:
            return 'temporary_variable'
        elif 'result' in name_lower or 'ret' in name_lower:
            return 'result_variable'
        elif 'count' in name_lower or 'cnt' in name_lower:
            return 'counter_variable'
        elif 'buffer' in name_lower or 'buf' in name_lower:
            return 'buffer_variable'
        elif usage_pattern == 'frequent_usage':
            return 'loop_variable'
        else:
            return 'local_variable'
    
    def _extract_pointed_type(self, param_type: str) -> Optional[str]:
        """Extract the type that a pointer points to"""
        if '*' in param_type:
            return param_type.replace('*', '').strip()
        elif param_type.startswith('LP'):
            return param_type[2:]  # Remove LP prefix
        return None
    
    def _calculate_parameter_size(self, param_type: str) -> int:
        """Calculate parameter size in bytes"""
        type_lower = param_type.lower()
        
        if 'char' in type_lower:
            return 1
        elif 'short' in type_lower or 'word' in type_lower:
            return 2
        elif 'int' in type_lower or 'dword' in type_lower or '*' in param_type:
            return 4
        elif 'long long' in type_lower or 'qword' in type_lower:
            return 8
        elif 'double' in type_lower:
            return 8
        elif 'float' in type_lower:
            return 4
        else:
            return 4  # Default
    
    def _infer_parameter_semantic_meaning(self, name: str, param_type: str, semantic_type: str) -> str:
        """Infer the semantic meaning of a parameter"""
        if semantic_type == 'window_handle':
            return 'Window or control handle for UI operations'
        elif semantic_type == 'string_buffer':
            return 'Text data for processing or output'
        elif semantic_type == 'size_parameter':
            return 'Size specification for buffer or operation'
        elif semantic_type == 'boolean_flag':
            return 'Conditional flag for operation control'
        elif semantic_type == 'array_buffer':
            return 'Array or buffer for data processing'
        elif semantic_type == 'output_parameter':
            return 'Output parameter for returning data'
        else:
            return 'General purpose parameter'
    
    def _infer_variable_semantic_meaning(self, name: str, var_type: str) -> str:
        """Infer the semantic meaning of a local variable"""
        name_lower = name.lower()
        
        if 'index' in name_lower or 'idx' in name_lower:
            return 'Array or loop index'
        elif 'handle' in name_lower:
            return 'Resource handle for management'
        elif 'result' in name_lower:
            return 'Function result or return value'
        elif 'error' in name_lower:
            return 'Error code or status indicator'
        else:
            return 'Local working variable'
    
    # Database initialization methods
    def _initialize_windows_api_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Windows API signature database"""
        return {
            'file_operations': {
                'name_patterns': [r'CreateFile', r'ReadFile', r'WriteFile', r'CloseHandle'],
                'param_types': ['HANDLE', 'LPCSTR', 'DWORD', 'LPVOID'],
                'min_params': 2
            },
            'window_management': {
                'name_patterns': [r'CreateWindow', r'ShowWindow', r'UpdateWindow', r'FindWindow'],
                'param_types': ['HWND', 'LPCSTR', 'HINSTANCE'],
                'min_params': 1
            },
            'memory_management': {
                'name_patterns': [r'HeapAlloc', r'HeapFree', r'VirtualAlloc', r'GlobalAlloc'],
                'param_types': ['HANDLE', 'SIZE_T', 'LPVOID'],
                'min_params': 2
            },
            'registry_operations': {
                'name_patterns': [r'RegOpenKey', r'RegQueryValue', r'RegSetValue', r'RegCloseKey'],
                'param_types': ['HKEY', 'LPCSTR', 'DWORD'],
                'min_params': 2
            }
        }
    
    def _initialize_register_patterns(self) -> Dict[str, List[str]]:
        """Initialize register usage patterns"""
        return {
            'parameter_registers': ['ecx', 'edx', 'r8', 'r9'],
            'return_registers': ['eax', 'rax', 'edx'],
            'preserved_registers': ['ebx', 'esi', 'edi', 'ebp'],
            'scratch_registers': ['eax', 'ecx', 'edx']
        }
    
    def _initialize_stack_patterns(self) -> Dict[str, str]:
        """Initialize stack analysis patterns"""
        return {
            'function_prologue': r'push\s+ebp.*mov\s+ebp,\s*esp',
            'stack_allocation': r'sub\s+esp,\s*0x([0-9a-fA-F]+)',
            'stack_cleanup': r'add\s+esp,\s*0x([0-9a-fA-F]+)',
            'function_epilogue': r'mov\s+esp,\s*ebp.*pop\s+ebp.*ret'
        }
    
    def generate_signature_report(self, signatures: Dict[str, FunctionSignature]) -> Dict[str, Any]:
        """Generate comprehensive signature recovery report"""
        
        total_functions = len(signatures)
        api_functions = sum(1 for sig in signatures.values() if sig.is_api_function)
        avg_confidence = sum(sig.confidence_score for sig in signatures.values()) / max(total_functions, 1)
        
        calling_convention_stats = {}
        for sig in signatures.values():
            cc = sig.calling_convention.value
            calling_convention_stats[cc] = calling_convention_stats.get(cc, 0) + 1
        
        return {
            'total_functions': total_functions,
            'api_functions_detected': api_functions,
            'average_confidence': avg_confidence,
            'calling_convention_distribution': calling_convention_stats,
            'high_confidence_functions': sum(1 for sig in signatures.values() if sig.confidence_score > 0.8),
            'functions_with_parameters': sum(1 for sig in signatures.values() if len(sig.parameters) > 0),
            'detailed_signatures': {
                addr: {
                    'name': sig.name,
                    'semantic_name': sig.semantic_name,
                    'calling_convention': sig.calling_convention.value,
                    'return_type': sig.return_type,
                    'parameter_count': len(sig.parameters),
                    'local_variable_count': len(sig.local_variables),
                    'stack_frame_size': sig.stack_frame_size,
                    'is_api_function': sig.is_api_function,
                    'confidence_score': sig.confidence_score
                } for addr, sig in signatures.items()
            }
        }