"""
Semantic Analysis Engine for AI-Enhanced Binary Decompilation

This module provides advanced semantic analysis capabilities including:
- Semantic function naming using ML models and pattern recognition
- Variable purpose inference based on usage patterns
- Algorithm pattern recognition for common algorithms and data structures
- Code style analysis and architectural pattern detection
- Integration with AI engines for enhanced analysis

Based on research from CodeBERT, GraphCodeBERT, and neural decompilation papers.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

from .config_manager import ConfigManager
from .ai_setup import get_ai_setup, is_ai_enabled
from .shared_components import MatrixLogger, MatrixFileManager, MatrixValidator


class FunctionType(Enum):
    """Function type classifications"""
    MEMORY_MANAGEMENT = "memory_management"
    STRING_MANIPULATION = "string_manipulation"
    FILE_IO = "file_io"
    NETWORK_IO = "network_io"
    CRYPTOGRAPHIC = "cryptographic"
    MATHEMATICAL = "mathematical"
    UTILITY = "utility"
    MAIN_LOGIC = "main_logic"
    ERROR_HANDLING = "error_handling"
    UNKNOWN = "unknown"


class VariableType(Enum):
    """Variable type classifications"""
    COUNTER = "counter"
    INDEX = "index"
    BUFFER = "buffer"
    POINTER = "pointer"
    FLAG = "flag"
    SIZE = "size"
    STATUS = "status"
    HANDLE = "handle"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


class AlgorithmPattern(Enum):
    """Algorithm pattern classifications"""
    SORTING = "sorting"
    SEARCHING = "searching"
    HASHING = "hashing"
    ENCRYPTION = "encryption"
    COMPRESSION = "compression"
    GRAPH_ALGORITHM = "graph_algorithm"
    TREE_TRAVERSAL = "tree_traversal"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GREEDY = "greedy"
    RECURSIVE = "recursive"
    ITERATIVE = "iterative"
    UNKNOWN = "unknown"


@dataclass
class SemanticFunction:
    """Semantic information about a function"""
    name: str
    suggested_name: str = ""
    function_type: FunctionType = FunctionType.UNKNOWN
    purpose: str = ""
    confidence: float = 0.0
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: str = "unknown"
    complexity: str = "unknown"
    patterns: List[AlgorithmPattern] = field(default_factory=list)
    api_calls: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'suggested_name': self.suggested_name,
            'function_type': self.function_type.value,
            'purpose': self.purpose,
            'confidence': self.confidence,
            'parameters': self.parameters,
            'return_type': self.return_type,
            'complexity': self.complexity,
            'patterns': [p.value for p in self.patterns],
            'api_calls': self.api_calls
        }


@dataclass
class SemanticVariable:
    """Semantic information about a variable"""
    name: str
    suggested_name: str = ""
    variable_type: VariableType = VariableType.UNKNOWN
    purpose: str = ""
    confidence: float = 0.0
    data_type: str = "unknown"
    scope: str = "local"
    usage_pattern: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'suggested_name': self.suggested_name,
            'variable_type': self.variable_type.value,
            'purpose': self.purpose,
            'confidence': self.confidence,
            'data_type': self.data_type,
            'scope': self.scope,
            'usage_pattern': self.usage_pattern
        }


@dataclass
class CodeStyleAnalysis:
    """Code style and architectural pattern analysis"""
    architectural_patterns: List[str] = field(default_factory=list)
    coding_style: str = "unknown"
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    maintainability_index: float = 0.0
    technical_debt: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'architectural_patterns': self.architectural_patterns,
            'coding_style': self.coding_style,
            'complexity_metrics': self.complexity_metrics,
            'quality_score': self.quality_score,
            'maintainability_index': self.maintainability_index,
            'technical_debt': self.technical_debt
        }


class SemanticAnalyzer:
    """
    Advanced semantic analyzer for decompiled code
    
    This class provides AI-enhanced semantic analysis to improve the quality
    and readability of decompiled code through intelligent naming, pattern
    recognition, and architectural analysis.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """
        Initialize the semantic analyzer
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager or ConfigManager()
        self.logger = MatrixLogger(self.__class__.__name__)
        self.file_manager = MatrixFileManager()
        self.validator = MatrixValidator()
        
        # AI integration
        self.ai_setup = get_ai_setup(self.config)
        self.ai_enabled = is_ai_enabled()
        self.ai_interface = self.ai_setup.get_ai_interface() if self.ai_enabled else None
        
        # Pattern databases
        self.function_patterns = self._load_function_patterns()
        self.variable_patterns = self._load_variable_patterns()
        self.algorithm_patterns = self._load_algorithm_patterns()
        
        # Analysis cache
        self.analysis_cache = {}
        
    def analyze_functions(
        self, 
        functions: Dict[str, str], 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, SemanticFunction]:
        """
        Analyze functions for semantic information
        
        Args:
            functions: Dictionary mapping function names to their code
            context: Optional context information
            
        Returns:
            Dictionary mapping function names to semantic analysis results
        """
        self.logger.info(f"Analyzing {len(functions)} functions for semantic information")
        
        results = {}
        context = context or {}
        
        for func_name, func_code in functions.items():
            try:
                semantic_func = self._analyze_single_function(func_name, func_code, context)
                results[func_name] = semantic_func
                
                self.logger.debug(f"Analyzed function {func_name}: {semantic_func.function_type.value}")
                
            except Exception as e:
                self.logger.error(f"Failed to analyze function {func_name}: {e}")
                results[func_name] = SemanticFunction(name=func_name)
        
        # Apply cross-function analysis
        self._apply_cross_function_analysis(results, context)
        
        return results
    
    def _analyze_single_function(
        self, 
        func_name: str, 
        func_code: str, 
        context: Dict[str, Any]
    ) -> SemanticFunction:
        """Analyze a single function for semantic information"""
        semantic_func = SemanticFunction(name=func_name)
        
        # Pattern-based analysis
        semantic_func.function_type = self._classify_function_type(func_code)
        semantic_func.patterns = self._detect_algorithm_patterns(func_code)
        semantic_func.api_calls = self._extract_api_calls(func_code)
        
        # Generate semantic name
        semantic_func.suggested_name = self._generate_function_name(
            func_name, func_code, semantic_func.function_type, semantic_func.patterns
        )
        
        # Analyze parameters
        semantic_func.parameters = self._analyze_function_parameters(func_code)
        
        # Determine return type and complexity
        semantic_func.return_type = self._infer_return_type(func_code)
        semantic_func.complexity = self._assess_function_complexity(func_code)
        
        # Generate purpose description
        semantic_func.purpose = self._generate_function_purpose(
            semantic_func.function_type, semantic_func.patterns, semantic_func.api_calls
        )
        
        # Calculate confidence score
        semantic_func.confidence = self._calculate_function_confidence(semantic_func, func_code)
        
        # AI enhancement if available
        if self.ai_enabled and self.ai_interface:
            semantic_func = self._enhance_function_with_ai(semantic_func, func_code)
        
        return semantic_func
    
    def _classify_function_type(self, code: str) -> FunctionType:
        """Classify function type based on code patterns"""
        code_lower = code.lower()
        
        # Memory management patterns
        if any(pattern in code_lower for pattern in ['malloc', 'free', 'calloc', 'realloc']):
            return FunctionType.MEMORY_MANAGEMENT
        
        # String manipulation patterns
        if any(pattern in code_lower for pattern in ['strcpy', 'strcat', 'strlen', 'strcmp', 'sprintf']):
            return FunctionType.STRING_MANIPULATION
        
        # File I/O patterns
        if any(pattern in code_lower for pattern in ['fopen', 'fclose', 'fread', 'fwrite', 'fprintf']):
            return FunctionType.FILE_IO
        
        # Network I/O patterns
        if any(pattern in code_lower for pattern in ['socket', 'bind', 'listen', 'accept', 'connect']):
            return FunctionType.NETWORK_IO
        
        # Cryptographic patterns
        if any(pattern in code_lower for pattern in ['encrypt', 'decrypt', 'hash', 'cipher', 'crypto']):
            return FunctionType.CRYPTOGRAPHIC
        
        # Mathematical patterns
        if any(pattern in code_lower for pattern in ['sqrt', 'pow', 'sin', 'cos', 'log', 'exp']):
            return FunctionType.MATHEMATICAL
        
        # Error handling patterns
        if any(pattern in code_lower for pattern in ['error', 'exception', 'fail', 'abort', 'exit']):
            return FunctionType.ERROR_HANDLING
        
        # Main logic patterns (contains main control flow)
        if any(pattern in code_lower for pattern in ['switch', 'while', 'for']) and len(code) > 500:
            return FunctionType.MAIN_LOGIC
        
        return FunctionType.UTILITY
    
    def _detect_algorithm_patterns(self, code: str) -> List[AlgorithmPattern]:
        """Detect algorithm patterns in function code"""
        patterns = []
        code_lower = code.lower()
        
        # Sorting patterns
        if any(pattern in code_lower for pattern in ['sort', 'qsort', 'bubble', 'merge', 'heap']):
            patterns.append(AlgorithmPattern.SORTING)
        
        # Searching patterns
        if any(pattern in code_lower for pattern in ['search', 'find', 'binary_search', 'bsearch']):
            patterns.append(AlgorithmPattern.SEARCHING)
        
        # Hashing patterns
        if any(pattern in code_lower for pattern in ['hash', 'crc', 'checksum', 'md5', 'sha']):
            patterns.append(AlgorithmPattern.HASHING)
        
        # Encryption patterns
        if any(pattern in code_lower for pattern in ['encrypt', 'decrypt', 'cipher', 'aes', 'rsa']):
            patterns.append(AlgorithmPattern.ENCRYPTION)
        
        # Compression patterns
        if any(pattern in code_lower for pattern in ['compress', 'decompress', 'zip', 'gzip', 'lz']):
            patterns.append(AlgorithmPattern.COMPRESSION)
        
        # Recursion detection
        func_name_match = re.search(r'(\w+)\s*\(', code)
        if func_name_match:
            func_name = func_name_match.group(1)
            if func_name in code and code.count(func_name) > 1:
                patterns.append(AlgorithmPattern.RECURSIVE)
        
        # Iteration detection
        if any(pattern in code_lower for pattern in ['for', 'while', 'do']):
            patterns.append(AlgorithmPattern.ITERATIVE)
        
        return patterns
    
    def _extract_api_calls(self, code: str) -> List[str]:
        """Extract API calls from function code"""
        api_calls = []
        
        # Common Windows API patterns
        windows_apis = re.findall(r'\b([A-Z][a-zA-Z]*(?:A|W)?)\s*\(', code)
        api_calls.extend(windows_apis)
        
        # Common C library functions
        c_functions = re.findall(r'\b(malloc|free|printf|scanf|strlen|strcpy|strcmp)\s*\(', code)
        api_calls.extend(c_functions)
        
        # Custom function calls
        custom_calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        
        # Filter out common keywords
        keywords = {'if', 'while', 'for', 'switch', 'sizeof', 'return'}
        custom_calls = [call for call in custom_calls if call not in keywords]
        
        api_calls.extend(custom_calls)
        
        return list(set(api_calls))  # Remove duplicates
    
    def _generate_function_name(
        self, 
        original_name: str, 
        code: str, 
        func_type: FunctionType, 
        patterns: List[AlgorithmPattern]
    ) -> str:
        """Generate semantic function name"""
        # If already has a meaningful name, keep it
        if not original_name.startswith(('FUN_', 'sub_', 'loc_', 'func_')):
            return original_name
        
        # Generate based on function type and patterns
        type_prefixes = {
            FunctionType.MEMORY_MANAGEMENT: 'mem_',
            FunctionType.STRING_MANIPULATION: 'str_',
            FunctionType.FILE_IO: 'file_',
            FunctionType.NETWORK_IO: 'net_',
            FunctionType.CRYPTOGRAPHIC: 'crypto_',
            FunctionType.MATHEMATICAL: 'math_',
            FunctionType.ERROR_HANDLING: 'error_',
            FunctionType.MAIN_LOGIC: 'main_',
            FunctionType.UTILITY: 'util_'
        }
        
        prefix = type_prefixes.get(func_type, 'func_')
        
        # Add pattern-specific suffixes
        if AlgorithmPattern.SORTING in patterns:
            return prefix + 'sort'
        elif AlgorithmPattern.SEARCHING in patterns:
            return prefix + 'search'
        elif AlgorithmPattern.HASHING in patterns:
            return prefix + 'hash'
        elif AlgorithmPattern.ENCRYPTION in patterns:
            return prefix + 'encrypt'
        elif AlgorithmPattern.COMPRESSION in patterns:
            return prefix + 'compress'
        
        # Look for specific operations in code
        code_lower = code.lower()
        if 'init' in code_lower:
            return prefix + 'init'
        elif 'cleanup' in code_lower or 'destroy' in code_lower:
            return prefix + 'cleanup'
        elif 'process' in code_lower:
            return prefix + 'process'
        elif 'handle' in code_lower:
            return prefix + 'handler'
        elif 'validate' in code_lower or 'check' in code_lower:
            return prefix + 'validate'
        
        return prefix + 'function'
    
    def _analyze_function_parameters(self, code: str) -> List[Dict[str, Any]]:
        """Analyze function parameters for semantic information"""
        parameters = []
        
        # Extract function signature
        signature_match = re.search(r'(\w+)\s*\([^)]*\)', code)
        if not signature_match:
            return parameters
        
        # Parse parameters from signature
        param_matches = re.findall(r'(\w+\s*\*?\s*\w+)(?:\s*,|\s*\))', signature_match.group(0))
        
        for param_match in param_matches:
            param_parts = param_match.strip().split()
            if len(param_parts) >= 2:
                param_type = ' '.join(param_parts[:-1])
                param_name = param_parts[-1]
                
                parameters.append({
                    'name': param_name,
                    'type': param_type,
                    'purpose': self._infer_parameter_purpose(param_name, param_type, code)
                })
        
        return parameters
    
    def _infer_parameter_purpose(self, name: str, param_type: str, code: str) -> str:
        """Infer parameter purpose from name, type, and usage"""
        name_lower = name.lower()
        
        # Common parameter patterns
        if 'size' in name_lower or 'len' in name_lower:
            return 'size or length value'
        elif 'buf' in name_lower or 'buffer' in name_lower:
            return 'data buffer'
        elif 'count' in name_lower or 'num' in name_lower:
            return 'count or number'
        elif 'index' in name_lower or 'idx' in name_lower:
            return 'array index'
        elif 'flag' in name_lower:
            return 'boolean flag'
        elif 'handle' in name_lower or 'hnd' in name_lower:
            return 'resource handle'
        elif '*' in param_type:
            return 'pointer parameter'
        
        return 'parameter'
    
    def _infer_return_type(self, code: str) -> str:
        """Infer function return type from code analysis"""
        # Look for return statements
        return_matches = re.findall(r'return\s+([^;]+);', code)
        
        if not return_matches:
            return 'void'
        
        # Analyze return values
        for return_val in return_matches:
            return_val = return_val.strip()
            
            if return_val in ['0', '1', 'TRUE', 'FALSE']:
                return 'bool'
            elif return_val.isdigit() or return_val.startswith('-'):
                return 'int'
            elif return_val == 'NULL':
                return 'pointer'
            elif '"' in return_val:
                return 'string'
        
        return 'unknown'
    
    def _assess_function_complexity(self, code: str) -> str:
        """Assess function complexity"""
        line_count = len(code.split('\n'))
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(code)
        
        if line_count < 20 and cyclomatic_complexity < 5:
            return 'low'
        elif line_count < 50 and cyclomatic_complexity < 10:
            return 'medium'
        elif line_count < 100 and cyclomatic_complexity < 20:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_cyclomatic_complexity(self, code: str) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        # Count decision points
        decision_keywords = ['if', 'while', 'for', 'switch', 'case', '&&', '||', '?']
        
        for keyword in decision_keywords:
            complexity += len(re.findall(r'\b' + re.escape(keyword) + r'\b', code))
        
        return complexity
    
    def _generate_function_purpose(
        self, 
        func_type: FunctionType, 
        patterns: List[AlgorithmPattern], 
        api_calls: List[str]
    ) -> str:
        """Generate function purpose description"""
        type_purposes = {
            FunctionType.MEMORY_MANAGEMENT: "Manages memory allocation and deallocation",
            FunctionType.STRING_MANIPULATION: "Performs string operations and manipulations",
            FunctionType.FILE_IO: "Handles file input/output operations",
            FunctionType.NETWORK_IO: "Manages network communication",
            FunctionType.CRYPTOGRAPHIC: "Performs cryptographic operations",
            FunctionType.MATHEMATICAL: "Executes mathematical calculations",
            FunctionType.ERROR_HANDLING: "Handles errors and exceptions",
            FunctionType.MAIN_LOGIC: "Contains main program logic",
            FunctionType.UTILITY: "Provides utility functionality"
        }
        
        base_purpose = type_purposes.get(func_type, "Performs unknown functionality")
        
        # Add pattern-specific details
        if patterns:
            pattern_descriptions = {
                AlgorithmPattern.SORTING: "with sorting algorithms",
                AlgorithmPattern.SEARCHING: "with search algorithms",
                AlgorithmPattern.HASHING: "with hashing functions",
                AlgorithmPattern.ENCRYPTION: "with encryption methods",
                AlgorithmPattern.COMPRESSION: "with compression techniques",
                AlgorithmPattern.RECURSIVE: "using recursive approach",
                AlgorithmPattern.ITERATIVE: "using iterative approach"
            }
            
            for pattern in patterns:
                if pattern in pattern_descriptions:
                    base_purpose += f" {pattern_descriptions[pattern]}"
        
        return base_purpose
    
    def _calculate_function_confidence(self, semantic_func: SemanticFunction, code: str) -> float:
        """Calculate confidence score for function analysis"""
        confidence = 0.3  # Base confidence
        
        # Boost confidence based on recognized patterns
        if semantic_func.function_type != FunctionType.UNKNOWN:
            confidence += 0.2
        
        if semantic_func.patterns:
            confidence += 0.1 * len(semantic_func.patterns)
        
        if semantic_func.api_calls:
            confidence += 0.05 * min(len(semantic_func.api_calls), 5)
        
        # Penalty for generic names
        if semantic_func.suggested_name.startswith(('func_', 'util_')):
            confidence -= 0.1
        
        # Boost for specific naming
        if not semantic_func.name.startswith(('FUN_', 'sub_', 'loc_')):
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _enhance_function_with_ai(
        self, 
        semantic_func: SemanticFunction, 
        code: str
    ) -> SemanticFunction:
        """Enhance function analysis using AI"""
        if not self.ai_interface:
            return semantic_func
        
        try:
            # Create AI prompt for function analysis
            prompt = f"""
            Analyze this decompiled function for semantic meaning:
            
            Function Name: {semantic_func.name}
            Current Classification: {semantic_func.function_type.value}
            
            Code:
            {code}
            
            Provide:
            1. A more descriptive function name
            2. The main purpose of this function
            3. Any algorithm patterns you recognize
            4. Confidence level (0.0-1.0) in your analysis
            
            Format as JSON with keys: suggested_name, purpose, patterns, confidence
            """
            
            response = self.ai_interface.generate_response(prompt)
            
            if response.success:
                try:
                    ai_analysis = json.loads(response.content)
                    
                    # Update semantic function with AI insights
                    if 'suggested_name' in ai_analysis:
                        semantic_func.suggested_name = ai_analysis['suggested_name']
                    
                    if 'purpose' in ai_analysis:
                        semantic_func.purpose = ai_analysis['purpose']
                    
                    if 'confidence' in ai_analysis:
                        # Combine AI confidence with existing confidence
                        ai_confidence = float(ai_analysis['confidence'])
                        semantic_func.confidence = (semantic_func.confidence + ai_confidence) / 2
                    
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse AI response: {e}")
            
        except Exception as e:
            self.logger.warning(f"AI enhancement failed: {e}")
        
        return semantic_func
    
    def _apply_cross_function_analysis(
        self, 
        functions: Dict[str, SemanticFunction], 
        context: Dict[str, Any]
    ):
        """Apply cross-function analysis to improve semantic understanding"""
        # Identify function call relationships
        call_graph = self._build_call_graph(functions)
        
        # Propagate semantic information through call graph
        self._propagate_semantic_info(functions, call_graph)
        
        # Detect architectural patterns
        self._detect_architectural_patterns(functions, context)
    
    def _build_call_graph(self, functions: Dict[str, SemanticFunction]) -> Dict[str, List[str]]:
        """Build function call graph"""
        call_graph = defaultdict(list)
        
        for func_name, semantic_func in functions.items():
            for api_call in semantic_func.api_calls:
                if api_call in functions:
                    call_graph[func_name].append(api_call)
        
        return dict(call_graph)
    
    def _propagate_semantic_info(
        self, 
        functions: Dict[str, SemanticFunction], 
        call_graph: Dict[str, List[str]]
    ):
        """Propagate semantic information through function call relationships"""
        # Simple propagation: if a function calls only memory management functions,
        # it might also be memory management related
        for func_name, called_functions in call_graph.items():
            if func_name in functions and called_functions:
                called_types = [functions[called].function_type for called in called_functions if called in functions]
                
                # If all called functions are of the same type, consider updating caller's type
                if len(set(called_types)) == 1 and called_types[0] != FunctionType.UNKNOWN:
                    if functions[func_name].function_type == FunctionType.UTILITY:
                        functions[func_name].function_type = called_types[0]
                        functions[func_name].confidence += 0.1
    
    def _detect_architectural_patterns(
        self, 
        functions: Dict[str, SemanticFunction], 
        context: Dict[str, Any]
    ):
        """Detect architectural patterns in the codebase"""
        # Count function types to identify architectural patterns
        type_counts = Counter(func.function_type for func in functions.values())
        
        # Store architectural insights in context
        if 'architectural_analysis' not in context:
            context['architectural_analysis'] = {}
        
        context['architectural_analysis']['function_distribution'] = {
            ftype.value: count for ftype, count in type_counts.items()
        }
        
        # Identify potential design patterns
        if type_counts[FunctionType.MEMORY_MANAGEMENT] > len(functions) * 0.3:
            context['architectural_analysis']['patterns'] = ['memory_intensive']
        
        if type_counts[FunctionType.FILE_IO] > len(functions) * 0.2:
            context['architectural_analysis']['patterns'] = context['architectural_analysis'].get('patterns', []) + ['file_processing']
    
    def analyze_variables(
        self, 
        code_segments: Dict[str, str]
    ) -> Dict[str, List[SemanticVariable]]:
        """
        Analyze variables for semantic information
        
        Args:
            code_segments: Dictionary mapping segment names to code
            
        Returns:
            Dictionary mapping segment names to lists of semantic variables
        """
        self.logger.info(f"Analyzing variables in {len(code_segments)} code segments")
        
        results = {}
        
        for segment_name, code in code_segments.items():
            try:
                variables = self._extract_variables(code)
                semantic_variables = []
                
                for var_name, var_info in variables.items():
                    semantic_var = self._analyze_single_variable(var_name, var_info, code)
                    semantic_variables.append(semantic_var)
                
                results[segment_name] = semantic_variables
                
            except Exception as e:
                self.logger.error(f"Failed to analyze variables in {segment_name}: {e}")
                results[segment_name] = []
        
        return results
    
    def _extract_variables(self, code: str) -> Dict[str, Dict[str, Any]]:
        """Extract variable declarations from code"""
        variables = {}
        
        # Match variable declarations
        var_patterns = [
            r'(\w+\s*\*?\s*)(\w+)\s*[=;,)]',  # Type varname
            r'(\w+)\s+(\w+)\s*\[',            # Array declarations
        ]
        
        for pattern in var_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if len(match) == 2:
                    var_type, var_name = match
                    if var_name not in {'if', 'while', 'for', 'return', 'int', 'char', 'void'}:
                        variables[var_name] = {
                            'type': var_type.strip(),
                            'declaration_line': self._find_declaration_line(code, var_name)
                        }
        
        return variables
    
    def _find_declaration_line(self, code: str, var_name: str) -> int:
        """Find the line number where variable is declared"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if re.search(rf'\b{re.escape(var_name)}\b', line):
                return i + 1
        return 0
    
    def _analyze_single_variable(
        self, 
        var_name: str, 
        var_info: Dict[str, Any], 
        code: str
    ) -> SemanticVariable:
        """Analyze a single variable for semantic information"""
        semantic_var = SemanticVariable(name=var_name)
        
        # Classify variable type
        semantic_var.variable_type = self._classify_variable_type(var_name, var_info, code)
        
        # Generate semantic name
        semantic_var.suggested_name = self._generate_variable_name(var_name, semantic_var.variable_type)
        
        # Analyze usage pattern
        semantic_var.usage_pattern = self._analyze_variable_usage(var_name, code)
        
        # Set data type and scope
        semantic_var.data_type = var_info.get('type', 'unknown')
        semantic_var.scope = self._determine_variable_scope(var_name, code)
        
        # Generate purpose
        semantic_var.purpose = self._generate_variable_purpose(semantic_var.variable_type, semantic_var.usage_pattern)
        
        # Calculate confidence
        semantic_var.confidence = self._calculate_variable_confidence(semantic_var, var_name)
        
        return semantic_var
    
    def _classify_variable_type(self, var_name: str, var_info: Dict[str, Any], code: str) -> VariableType:
        """Classify variable type based on name and usage patterns"""
        name_lower = var_name.lower()
        var_type = var_info.get('type', '').lower()
        
        # Counter patterns
        if any(pattern in name_lower for pattern in ['count', 'cnt', 'num', 'n']):
            return VariableType.COUNTER
        
        # Index patterns
        if any(pattern in name_lower for pattern in ['index', 'idx', 'i', 'j', 'k']):
            return VariableType.INDEX
        
        # Buffer patterns
        if any(pattern in name_lower for pattern in ['buf', 'buffer', 'data', 'arr']):
            return VariableType.BUFFER
        
        # Pointer patterns
        if '*' in var_type or 'ptr' in name_lower:
            return VariableType.POINTER
        
        # Flag patterns
        if any(pattern in name_lower for pattern in ['flag', 'bool', 'is', 'has', 'can']):
            return VariableType.FLAG
        
        # Size patterns
        if any(pattern in name_lower for pattern in ['size', 'len', 'length']):
            return VariableType.SIZE
        
        # Status patterns
        if any(pattern in name_lower for pattern in ['status', 'state', 'result', 'ret']):
            return VariableType.STATUS
        
        # Handle patterns
        if any(pattern in name_lower for pattern in ['handle', 'hnd', 'fd', 'file']):
            return VariableType.HANDLE
        
        # Temporary patterns
        if any(pattern in name_lower for pattern in ['temp', 'tmp', 'var']):
            return VariableType.TEMPORARY
        
        return VariableType.UNKNOWN
    
    def _generate_variable_name(self, original_name: str, var_type: VariableType) -> str:
        """Generate semantic variable name"""
        # If already has a meaningful name, keep it
        if not original_name.startswith(('var_', 'local_', 'param_', 'uVar', 'iVar')):
            return original_name
        
        # Generate based on variable type
        type_prefixes = {
            VariableType.COUNTER: 'count',
            VariableType.INDEX: 'index',
            VariableType.BUFFER: 'buffer',
            VariableType.POINTER: 'ptr',
            VariableType.FLAG: 'flag',
            VariableType.SIZE: 'size',
            VariableType.STATUS: 'status',
            VariableType.HANDLE: 'handle',
            VariableType.TEMPORARY: 'temp'
        }
        
        prefix = type_prefixes.get(var_type, 'var')
        
        # Extract numeric suffix if present
        suffix_match = re.search(r'(\d+)$', original_name)
        suffix = suffix_match.group(1) if suffix_match else '1'
        
        return f"{prefix}_{suffix}"
    
    def _analyze_variable_usage(self, var_name: str, code: str) -> str:
        """Analyze how a variable is used in the code"""
        usage_patterns = []
        
        # Check if used in loops
        if re.search(rf'for\s*\([^)]*{re.escape(var_name)}[^)]*\)', code):
            usage_patterns.append('loop_control')
        
        # Check if used in conditions
        if re.search(rf'if\s*\([^)]*{re.escape(var_name)}[^)]*\)', code):
            usage_patterns.append('conditional')
        
        # Check if used in function calls
        if re.search(rf'\w+\s*\([^)]*{re.escape(var_name)}[^)]*\)', code):
            usage_patterns.append('function_parameter')
        
        # Check if used in assignments
        if re.search(rf'{re.escape(var_name)}\s*=', code):
            usage_patterns.append('assignment_target')
        
        # Check if used in arithmetic
        if re.search(rf'{re.escape(var_name)}\s*[+\-*/]', code):
            usage_patterns.append('arithmetic')
        
        return ', '.join(usage_patterns) if usage_patterns else 'unknown'
    
    def _determine_variable_scope(self, var_name: str, code: str) -> str:
        """Determine variable scope"""
        declaration_line = self._find_declaration_line(code, var_name)
        
        if declaration_line == 0:
            return 'unknown'
        
        lines = code.split('\n')
        if declaration_line < len(lines):
            line = lines[declaration_line - 1]
            if line.strip().startswith('static'):
                return 'static'
            elif 'global' in line:
                return 'global'
        
        return 'local'
    
    def _generate_variable_purpose(self, var_type: VariableType, usage_pattern: str) -> str:
        """Generate variable purpose description"""
        type_purposes = {
            VariableType.COUNTER: "Used for counting operations",
            VariableType.INDEX: "Used as array or loop index",
            VariableType.BUFFER: "Stores data temporarily",
            VariableType.POINTER: "Points to memory location",
            VariableType.FLAG: "Boolean flag for state tracking",
            VariableType.SIZE: "Represents size or length",
            VariableType.STATUS: "Tracks status or result codes",
            VariableType.HANDLE: "Handle to system resource",
            VariableType.TEMPORARY: "Temporary storage variable"
        }
        
        base_purpose = type_purposes.get(var_type, "General purpose variable")
        
        if usage_pattern and usage_pattern != 'unknown':
            base_purpose += f" ({usage_pattern})"
        
        return base_purpose
    
    def _calculate_variable_confidence(self, semantic_var: SemanticVariable, original_name: str) -> float:
        """Calculate confidence score for variable analysis"""
        confidence = 0.4  # Base confidence
        
        # Boost confidence for recognized patterns
        if semantic_var.variable_type != VariableType.UNKNOWN:
            confidence += 0.2
        
        # Boost for meaningful usage patterns
        if semantic_var.usage_pattern and semantic_var.usage_pattern != 'unknown':
            confidence += 0.2
        
        # Penalty for generic names
        if original_name.startswith(('var_', 'local_', 'param_')):
            confidence -= 0.1
        
        # Boost for specific naming
        if not original_name.startswith(('var_', 'local_', 'param_', 'uVar', 'iVar')):
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def analyze_code_style(self, code_segments: Dict[str, str]) -> CodeStyleAnalysis:
        """
        Analyze code style and architectural patterns
        
        Args:
            code_segments: Dictionary mapping segment names to code
            
        Returns:
            Code style analysis results
        """
        self.logger.info(f"Analyzing code style in {len(code_segments)} segments")
        
        analysis = CodeStyleAnalysis()
        
        # Aggregate all code for analysis
        all_code = '\n'.join(code_segments.values())
        
        # Analyze architectural patterns
        analysis.architectural_patterns = self._detect_code_patterns(all_code)
        
        # Analyze coding style
        analysis.coding_style = self._analyze_coding_style(all_code)
        
        # Calculate complexity metrics
        analysis.complexity_metrics = self._calculate_complexity_metrics(all_code)
        
        # Calculate quality score
        analysis.quality_score = self._calculate_quality_score(analysis)
        
        # Calculate maintainability index
        analysis.maintainability_index = self._calculate_maintainability_index(analysis)
        
        # Identify technical debt
        analysis.technical_debt = self._identify_technical_debt(all_code)
        
        return analysis
    
    def _detect_code_patterns(self, code: str) -> List[str]:
        """Detect architectural and design patterns in code"""
        patterns = []
        
        # Common design patterns
        if re.search(r'struct.*factory', code, re.IGNORECASE):
            patterns.append('Factory Pattern')
        
        if re.search(r'singleton', code, re.IGNORECASE):
            patterns.append('Singleton Pattern')
        
        if re.search(r'observer|callback', code, re.IGNORECASE):
            patterns.append('Observer Pattern')
        
        # Architectural patterns
        if code.count('layer') > 3 or code.count('tier') > 3:
            patterns.append('Layered Architecture')
        
        if code.count('component') > 5:
            patterns.append('Component-Based Architecture')
        
        if re.search(r'mvc|model.*view.*controller', code, re.IGNORECASE):
            patterns.append('MVC Pattern')
        
        return patterns
    
    def _analyze_coding_style(self, code: str) -> str:
        """Analyze coding style characteristics"""
        # Simple heuristics for style analysis
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 'unknown'
        
        # Check indentation style
        tab_count = sum(1 for line in non_empty_lines if line.startswith('\t'))
        space_count = sum(1 for line in non_empty_lines if line.startswith(' '))
        
        indentation_style = 'tabs' if tab_count > space_count else 'spaces'
        
        # Check brace style
        opening_braces_same_line = len(re.findall(r'{\s*$', code))
        opening_braces_new_line = len(re.findall(r'^\s*{', code))
        
        brace_style = 'K&R' if opening_braces_same_line > opening_braces_new_line else 'Allman'
        
        # Check naming convention
        camel_case_count = len(re.findall(r'\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b', code))
        snake_case_count = len(re.findall(r'\b[a-z][a-z0-9_]*[a-z0-9]\b', code))
        
        naming_style = 'camelCase' if camel_case_count > snake_case_count else 'snake_case'
        
        return f"{indentation_style}, {brace_style}, {naming_style}"
    
    def _calculate_complexity_metrics(self, code: str) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        metrics = {
            'lines_of_code': len(non_empty_lines),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(code),
            'nesting_depth': self._calculate_max_nesting_depth(code),
            'function_count': len(re.findall(r'\w+\s*\([^)]*\)\s*{', code))
        }
        
        # Halstead metrics (simplified)
        operators = re.findall(r'[+\-*/=<>!&|]+', code)
        operands = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
        
        unique_operators = len(set(operators))
        unique_operands = len(set(operands))
        total_operators = len(operators)
        total_operands = len(operands)
        
        if unique_operators > 0 and unique_operands > 0:
            vocabulary = unique_operators + unique_operands
            length = total_operators + total_operands
            metrics['halstead_difficulty'] = (unique_operators / 2) * (total_operands / unique_operands) if unique_operands > 0 else 0
            metrics['halstead_volume'] = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
        
        return metrics
    
    def _calculate_max_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _calculate_quality_score(self, analysis: CodeStyleAnalysis) -> float:
        """Calculate overall code quality score"""
        score = 1.0  # Start with perfect score
        
        # Penalties for high complexity
        if analysis.complexity_metrics.get('cyclomatic_complexity', 0) > 10:
            score -= 0.2
        
        if analysis.complexity_metrics.get('nesting_depth', 0) > 5:
            score -= 0.1
        
        if analysis.complexity_metrics.get('lines_of_code', 0) > 1000:
            score -= 0.1
        
        # Bonus for architectural patterns
        if analysis.architectural_patterns:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_maintainability_index(self, analysis: CodeStyleAnalysis) -> float:
        """Calculate maintainability index"""
        # Simplified maintainability index calculation
        loc = analysis.complexity_metrics.get('lines_of_code', 1)
        complexity = analysis.complexity_metrics.get('cyclomatic_complexity', 1)
        
        # Formula based on standard maintainability index
        if loc > 0 and complexity > 0:
            maintainability = 171 - 5.2 * (complexity ** 0.23) - 0.23 * complexity - 16.2 * (loc ** 0.5)
            return max(0.0, min(100.0, maintainability)) / 100.0
        
        return 0.5
    
    def _identify_technical_debt(self, code: str) -> List[str]:
        """Identify technical debt indicators"""
        debt_indicators = []
        
        # TODO/FIXME comments
        if re.search(r'//.*(?:TODO|FIXME|HACK)', code, re.IGNORECASE):
            debt_indicators.append('TODO/FIXME comments present')
        
        # Magic numbers
        magic_numbers = re.findall(r'\b(?!0|1)\d{2,}\b', code)
        if len(magic_numbers) > 5:
            debt_indicators.append('Multiple magic numbers detected')
        
        # Long functions
        functions = re.findall(r'\w+\s*\([^)]*\)\s*{[^}]{200,}}', code)
        if functions:
            debt_indicators.append('Functions with high line count')
        
        # Duplicated code (simple check)
        lines = code.split('\n')
        line_counts = Counter(line.strip() for line in lines if line.strip())
        duplicated_lines = [line for line, count in line_counts.items() if count > 3 and len(line) > 20]
        if duplicated_lines:
            debt_indicators.append('Potential code duplication')
        
        # Deep nesting
        max_nesting = self._calculate_max_nesting_depth(code)
        if max_nesting > 4:
            debt_indicators.append('Deep nesting levels')
        
        return debt_indicators
    
    def _load_function_patterns(self) -> Dict[str, Any]:
        """Load function pattern database"""
        # This would load from configuration or external files
        return {}
    
    def _load_variable_patterns(self) -> Dict[str, Any]:
        """Load variable pattern database"""
        # This would load from configuration or external files
        return {}
    
    def _load_algorithm_patterns(self) -> Dict[str, Any]:
        """Load algorithm pattern database"""
        # This would load from configuration or external files
        return {}
    
    def generate_semantic_report(
        self, 
        function_analysis: Dict[str, SemanticFunction],
        variable_analysis: Dict[str, List[SemanticVariable]],
        style_analysis: CodeStyleAnalysis,
        output_path: str
    ) -> bool:
        """
        Generate comprehensive semantic analysis report
        
        Args:
            function_analysis: Function analysis results
            variable_analysis: Variable analysis results
            style_analysis: Code style analysis results
            output_path: Path to save the report
            
        Returns:
            True if report generated successfully
        """
        try:
            report = {
                'semantic_analysis_report': {
                    'timestamp': self.file_manager.get_timestamp(),
                    'summary': {
                        'functions_analyzed': len(function_analysis),
                        'variables_analyzed': sum(len(vars_list) for vars_list in variable_analysis.values()),
                        'average_function_confidence': sum(func.confidence for func in function_analysis.values()) / len(function_analysis) if function_analysis else 0.0
                    },
                    'function_analysis': {name: func.to_dict() for name, func in function_analysis.items()},
                    'variable_analysis': {name: [var.to_dict() for var in vars_list] for name, vars_list in variable_analysis.items()},
                    'style_analysis': style_analysis.to_dict()
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Semantic analysis report generated: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate semantic report: {e}")
            return False


def run_semantic_analysis(
    functions: Dict[str, str],
    code_segments: Optional[Dict[str, str]] = None,
    config_manager: Optional[ConfigManager] = None,
    output_path: Optional[str] = None
) -> Tuple[Dict[str, SemanticFunction], Dict[str, List[SemanticVariable]], CodeStyleAnalysis]:
    """
    Convenience function to run complete semantic analysis
    
    Args:
        functions: Dictionary mapping function names to their code
        code_segments: Optional code segments for variable analysis
        config_manager: Optional configuration manager
        output_path: Optional path to save analysis report
        
    Returns:
        Tuple of (function_analysis, variable_analysis, style_analysis)
    """
    analyzer = SemanticAnalyzer(config_manager)
    
    # Analyze functions
    function_analysis = analyzer.analyze_functions(functions)
    
    # Analyze variables
    code_segments = code_segments or functions
    variable_analysis = analyzer.analyze_variables(code_segments)
    
    # Analyze code style
    style_analysis = analyzer.analyze_code_style(code_segments)
    
    # Generate report if output path provided
    if output_path:
        analyzer.generate_semantic_report(
            function_analysis, variable_analysis, style_analysis, output_path
        )
    
    return function_analysis, variable_analysis, style_analysis


if __name__ == "__main__":
    # Test the semantic analyzer
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python semantic_analyzer.py <code_file> [output_file]")
        sys.exit(1)
    
    code_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "semantic_analysis_report.json"
    
    logging.basicConfig(level=logging.INFO)
    
    # Read code file
    try:
        with open(code_file, 'r') as f:
            code = f.read()
        
        # Mock function structure for testing
        functions = {"test_function": code}
        
        # Run semantic analysis
        func_analysis, var_analysis, style_analysis = run_semantic_analysis(
            functions, output_path=output_file
        )
        
        print(f"Semantic analysis completed. Report saved to: {output_file}")
        print(f"Functions analyzed: {len(func_analysis)}")
        print(f"Quality score: {style_analysis.quality_score:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)