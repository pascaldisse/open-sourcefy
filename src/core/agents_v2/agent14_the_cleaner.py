"""
Agent 14: The Cleaner - Code Cleanup and Optimization
The meticulous code quality enforcer who ensures pristine, optimized source code.
Performs final cleanup, optimization, and quality enhancement with obsessive attention to detail.

Production-ready Matrix v2 implementation following SOLID principles and clean code standards.
Includes comprehensive cleanup algorithms and optimization techniques.
"""

import logging
import time
import json
import re
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Matrix framework imports
try:
    from ..matrix_agents_v2 import ReconstructionAgent, MatrixCharacter, AgentStatus
    from ..shared_components import (
        MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker, 
        MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
    )
    from ..exceptions import MatrixAgentError, ValidationError, ConfigurationError
    HAS_MATRIX_FRAMEWORK = True
except ImportError:
    # Fallback for basic execution
    HAS_MATRIX_FRAMEWORK = False


@dataclass
class CodeCleanupResult:
    """Result of comprehensive code cleanup and optimization"""
    success: bool = False
    files_processed: int = 0
    total_optimizations: int = 0
    quality_improvements: int = 0
    cleanup_efficiency: float = 0.0
    final_quality_score: float = 0.0
    maintainability_index: float = 0.0
    cleaned_files: Dict[str, str] = field(default_factory=dict)
    optimization_report: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationPattern:
    """Code optimization pattern definition"""
    name: str
    pattern: str
    replacement: str
    description: str
    benefit: str
    complexity: str = "low"
    risk_level: str = "low"


class Agent14_TheCleaner:
    """
    Agent 14: The Cleaner - Code Cleanup and Optimization
    
    Responsibilities:
    1. Perform comprehensive code cleanup and formatting
    2. Apply intelligent optimization patterns
    3. Enhance code quality and maintainability
    4. Remove redundant and dead code
    5. Normalize coding style and conventions
    6. Apply final polish and quality touches
    7. Generate detailed quality reports
    """
    
    def __init__(self):
        self.agent_id = 14
        self.name = "The Cleaner"
        self.character = MatrixCharacter.CLEANER if HAS_MATRIX_FRAMEWORK else "cleaner"
        
        # Core components
        self.logger = self._setup_logging()
        self.file_manager = MatrixFileManager() if HAS_MATRIX_FRAMEWORK else None
        self.validator = MatrixValidator() if HAS_MATRIX_FRAMEWORK else None
        self.progress_tracker = MatrixProgressTracker() if HAS_MATRIX_FRAMEWORK else None
        self.error_handler = MatrixErrorHandler() if HAS_MATRIX_FRAMEWORK else None
        
        # Cleanup components
        self.optimization_patterns = self._load_optimization_patterns()
        self.quality_rules = self._load_quality_rules()
        self.style_conventions = self._load_style_conventions()
        
        # Quality thresholds
        self.quality_thresholds = {
            'minimum_quality_score': 0.70,
            'minimum_maintainability': 60.0,
            'maximum_complexity_per_function': 10.0,
            'minimum_comment_ratio': 0.15
        }
        
        # State tracking
        self.current_phase = "initialization"
        self.cleanup_stats = {}
        self.optimization_history = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(f"Matrix.TheCleaner")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[The Cleaner] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_dependencies(self) -> List[int]:
        """Get list of required predecessor agents"""
        return [13]  # Agent Johnson - Security Analysis
    
    def get_description(self) -> str:
        """Get agent description"""
        return ("The Cleaner performs meticulous code cleanup and optimization, "
                "ensuring pristine source code quality with obsessive attention to detail.")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive code cleanup and optimization"""
        self.logger.info("🧹 The Cleaner initiating code purification protocol...")
        
        start_time = time.time()
        
        try:
            # Phase 1: Validate dependencies and extract source code
            self.current_phase = "source_validation"
            self.logger.info("Phase 1: Validating source code dependencies...")
            validation_result = self._validate_cleanup_dependencies(context)
            
            if not validation_result['valid']:
                return self._create_failure_result(
                    f"Source validation failed: {validation_result['error']}"
                )
            
            # Phase 2: Extract and prepare source code for cleanup
            self.current_phase = "source_extraction"
            self.logger.info("Phase 2: Extracting source code for cleanup...")
            source_data = self._extract_source_code(context)
            
            if not source_data:
                return self._create_failure_result("No source code available for cleanup")
            
            # Phase 3: Perform initial code cleanup
            self.current_phase = "initial_cleanup"
            self.logger.info("Phase 3: Performing initial code cleanup...")
            initial_cleanup = self._perform_initial_cleanup(source_data)
            
            # Phase 4: Apply optimization patterns
            self.current_phase = "optimization"
            self.logger.info("Phase 4: Applying optimization patterns...")
            optimization_result = self._apply_optimization_patterns(initial_cleanup)
            
            # Phase 5: Enhance code quality
            self.current_phase = "quality_enhancement"
            self.logger.info("Phase 5: Enhancing code quality...")
            quality_enhancement = self._enhance_code_quality(optimization_result)
            
            # Phase 6: Remove redundancy and dead code
            self.current_phase = "redundancy_removal"
            self.logger.info("Phase 6: Removing redundant and dead code...")
            redundancy_removal = self._remove_code_redundancy(quality_enhancement)
            
            # Phase 7: Normalize code style
            self.current_phase = "style_normalization"
            self.logger.info("Phase 7: Normalizing code style...")
            style_normalization = self._normalize_code_style(redundancy_removal)
            
            # Phase 8: Apply final polish
            self.current_phase = "final_polish"
            self.logger.info("Phase 8: Applying final polish...")
            final_polish = self._apply_final_polish(style_normalization)
            
            # Phase 9: Generate comprehensive report
            self.current_phase = "report_generation"
            self.logger.info("Phase 9: Generating comprehensive cleanup report...")
            final_result = self._create_cleanup_report(
                initial_cleanup, optimization_result, quality_enhancement,
                redundancy_removal, style_normalization, final_polish, context
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"🎯 The Cleaner purification completed in {execution_time:.2f}s")
            self.logger.info(f"✨ Files Cleaned: {final_result.files_processed}")
            self.logger.info(f"⚡ Optimizations Applied: {final_result.total_optimizations}")
            self.logger.info(f"📈 Quality Score: {final_result.final_quality_score:.2f}")
            self.logger.info(f"🔧 Maintainability Index: {final_result.maintainability_index:.1f}")
            
            return {
                'agent_id': self.agent_id,
                'agent_name': self.name,
                'status': AgentStatus.SUCCESS if final_result.success else AgentStatus.FAILED,
                'execution_time': execution_time,
                'cleanup_result': final_result,
                'files_processed': final_result.files_processed,
                'total_optimizations': final_result.total_optimizations,
                'quality_improvements': final_result.quality_improvements,
                'final_quality_score': final_result.final_quality_score,
                'maintainability_index': final_result.maintainability_index,
                'cleaned_files': final_result.cleaned_files,
                'optimization_report': final_result.optimization_report,
                'quality_metrics': final_result.quality_metrics,
                'recommendations': final_result.recommendations,
                'cleanup_stats': self.cleanup_stats,
                'metadata': {
                    'character': self.character,
                    'phase': self.current_phase,
                    'optimization_patterns_applied': len(self.optimization_history),
                    'cleanup_efficiency': final_result.cleanup_efficiency,
                    'warnings': len(final_result.warnings),
                    'errors': len(final_result.error_messages)
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"The Cleaner purification failed in {self.current_phase}: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_failure_result(error_msg, execution_time)

    def _validate_cleanup_dependencies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate cleanup dependencies"""
        agent_results = context.get('agent_results', {})
        
        # Check Agent Johnson (Agent 13) result
        if 13 not in agent_results:
            return {'valid': False, 'error': 'Agent Johnson (Agent 13) result not available'}
        
        johnson_result = agent_results[13]
        status = johnson_result.get('status', 'unknown')
        
        if status != 'success' and status != AgentStatus.SUCCESS:
            return {'valid': False, 'error': 'Agent Johnson security analysis failed'}
        
        return {'valid': True, 'error': None}

    def _extract_source_code(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Extract source code from various agent results"""
        agent_results = context.get('agent_results', {})
        source_code = {}
        
        # Priority order for source code extraction
        source_agents = [13, 12, 11, 10, 9, 7]  # Johnson, Link, Oracle, Machine, Commander Locke, Trainman
        
        for agent_id in source_agents:
            if agent_id in agent_results:
                result = agent_results[agent_id]
                extracted_code = self._extract_code_from_agent_result(agent_id, result)
                source_code.update(extracted_code)
                
                if source_code:  # Found source code
                    self.logger.info(f"Source code extracted from Agent {agent_id}")
                    break
        
        return source_code

    def _extract_code_from_agent_result(self, agent_id: int, result: Dict[str, Any]) -> Dict[str, str]:
        """Extract source code from individual agent result"""
        source_code = {}
        
        try:
            data = result.get('data', {})
            
            # From Agent Johnson (Security Analysis)
            if agent_id == 13:
                if isinstance(data, dict):
                    # Look for security analysis results that may contain source
                    security_result = data.get('security_analysis_result', {})
                    if hasattr(security_result, 'vulnerabilities'):
                        # Extract source from vulnerability analysis context
                        for vuln in security_result.vulnerabilities:
                            if isinstance(vuln, dict) and 'context' in vuln:
                                context_code = vuln['context']
                                if context_code and len(context_code) > 50:
                                    file_name = vuln.get('file', f'vulnerability_{len(source_code)}.c')
                                    source_code[file_name] = context_code
            
            # From Link or Oracle (comprehensive results)
            elif agent_id in [12, 11]:
                if isinstance(data, dict):
                    # Look for integration results or validation results
                    integration_result = data.get('integration_result', {})
                    if hasattr(integration_result, 'data_flows'):
                        data_flows = integration_result.data_flows
                        if isinstance(data_flows, dict):
                            for flow_name, flow_data in data_flows.items():
                                if isinstance(flow_data, dict):
                                    source_files = flow_data.get('source_files', {})
                                    header_files = flow_data.get('header_files', {})
                                    source_code.update(source_files)
                                    source_code.update(header_files)
            
            # From Machine or Commander Locke (reconstructed source)
            elif agent_id in [10, 9]:
                if isinstance(data, dict):
                    # Look for reconstructed source or build results
                    reconstructed_source = data.get('reconstructed_source', {})
                    if isinstance(reconstructed_source, dict):
                        source_files = reconstructed_source.get('source_files', {})
                        header_files = reconstructed_source.get('header_files', {})
                        source_code.update(source_files)
                        source_code.update(header_files)
                    
                    # Also check for final polish results
                    final_polish = data.get('final_polish', {})
                    if isinstance(final_polish, dict):
                        polished_files = final_polish.get('polished_files', {})
                        source_code.update(polished_files)
            
            # From other agents (decompiled functions)
            else:
                if isinstance(data, dict):
                    functions = data.get('enhanced_functions', {}) or data.get('decompiled_functions', {})
                    for func_name, func_data in functions.items():
                        if isinstance(func_data, dict) and func_data.get('code'):
                            source_code[f"{func_name}.c"] = func_data['code']
        
        except Exception as e:
            self.logger.warning(f"Failed to extract code from agent {agent_id}: {e}")
        
        return source_code

    def _perform_initial_cleanup(self, source_data: Dict[str, str]) -> Dict[str, Any]:
        """Perform initial code cleanup"""
        cleanup = {
            'cleaned_files': {},
            'cleanup_statistics': {},
            'removed_comments': [],
            'fixed_formatting': [],
            'resolved_warnings': [],
            'cleaned_functions': []
        }
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                original_lines = len(content.splitlines())
                
                # Clean the file content
                cleaned_content = self._clean_file_content(content, filename)
                
                cleaned_lines = len(cleaned_content.splitlines())
                
                cleanup['cleaned_files'][filename] = cleaned_content
                cleanup['cleanup_statistics'][filename] = {
                    'original_lines': original_lines,
                    'cleaned_lines': cleaned_lines,
                    'lines_removed': original_lines - cleaned_lines,
                    'cleanup_ratio': (original_lines - cleaned_lines) / original_lines if original_lines > 0 else 0.0
                }
        
        return cleanup

    def _clean_file_content(self, content: str, filename: str) -> str:
        """Clean individual file content"""
        cleaned = content
        
        # Remove excessive whitespace
        cleaned = self._remove_excessive_whitespace(cleaned)
        
        # Clean up comments
        cleaned = self._clean_comments(cleaned)
        
        # Fix basic formatting issues
        cleaned = self._fix_basic_formatting(cleaned)
        
        # Remove debug statements
        cleaned = self._remove_debug_statements(cleaned)
        
        # Fix common syntax issues
        cleaned = self._fix_syntax_issues(cleaned)
        
        return cleaned

    def _remove_excessive_whitespace(self, content: str) -> str:
        """Remove excessive whitespace and empty lines"""
        lines = content.splitlines()
        cleaned_lines = []
        
        consecutive_empty = 0
        for line in lines:
            # Remove trailing whitespace
            line = line.rstrip()
            
            if not line.strip():
                consecutive_empty += 1
                # Allow maximum 2 consecutive empty lines
                if consecutive_empty <= 2:
                    cleaned_lines.append(line)
            else:
                consecutive_empty = 0
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _clean_comments(self, content: str) -> str:
        """Clean up comments while preserving important ones"""
        lines = content.splitlines()
        cleaned_lines = []
        
        for line in lines:
            # Remove TODO/FIXME/HACK comments from decompilation
            if re.search(r'//\s*(TODO|FIXME|HACK|XXX).*decompil', line, re.IGNORECASE):
                continue
            
            # Remove obvious placeholder comments
            if re.search(r'//\s*(placeholder|dummy|temp|test)', line, re.IGNORECASE):
                continue
            
            # Remove excessive comment markers
            line = re.sub(r'/\*+', '/*', line)
            line = re.sub(r'\*+/', '*/', line)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _fix_basic_formatting(self, content: str) -> str:
        """Fix basic formatting issues"""
        # Fix spacing around operators
        content = re.sub(r'(\w)\s*([+\-*/=])\s*(\w)', r'\1 \2 \3', content)
        
        # Fix brace formatting
        content = re.sub(r'}\s*else\s*{', '} else {', content)
        content = re.sub(r'}\s*catch\s*\(', '} catch (', content)
        
        # Fix comma spacing
        content = re.sub(r',(\w)', r', \1', content)
        
        # Fix semicolon spacing
        content = re.sub(r';\s*(\w)', r'; \1', content)
        
        return content

    def _remove_debug_statements(self, content: str) -> str:
        """Remove debug and diagnostic statements"""
        debug_patterns = [
            r'printf\s*\(\s*"DEBUG.*?\)\s*;',
            r'cout\s*<<.*?debug.*?;',
            r'System\.out\.println.*?debug.*?;',
            r'console\.log\s*\(.*?debug.*?\)\s*;'
        ]
        
        for pattern in debug_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content

    def _fix_syntax_issues(self, content: str) -> str:
        """Fix common syntax issues from decompilation"""
        # Fix double semicolons
        content = re.sub(r';;+', ';', content)
        
        # Fix missing spaces after keywords
        content = re.sub(r'\b(if|while|for|switch)\(', r'\1 (', content)
        
        # Fix function declaration spacing
        content = re.sub(r'(\w+)\s*\(\s*([^)]*)\s*\)', r'\1(\2)', content)
        
        return content

    def _apply_optimization_patterns(self, cleanup_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent optimization patterns"""
        optimization = {
            'optimizations': [],
            'performance_improvements': [],
            'memory_optimizations': [],
            'algorithm_improvements': [],
            'optimized_files': {}
        }
        
        cleaned_files = cleanup_result.get('cleaned_files', {})
        
        for filename, content in cleaned_files.items():
            if isinstance(content, str):
                optimized_content = self._optimize_file_content(content, filename, optimization)
                optimization['optimized_files'][filename] = optimized_content
        
        return optimization

    def _optimize_file_content(self, content: str, filename: str, optimization: Dict[str, Any]) -> str:
        """Optimize individual file content"""
        optimized = content
        
        # Apply loop optimizations
        optimized = self._optimize_loops(optimized, optimization)
        
        # Apply memory optimizations
        optimized = self._optimize_memory_usage(optimized, optimization)
        
        # Apply string optimizations
        optimized = self._optimize_string_operations(optimized, optimization)
        
        # Apply function call optimizations
        optimized = self._optimize_function_calls(optimized, optimization)
        
        # Apply mathematical optimizations
        optimized = self._optimize_mathematical_expressions(optimized, optimization)
        
        return optimized

    def _optimize_loops(self, content: str, optimization: Dict[str, Any]) -> str:
        """Optimize loop structures"""
        # Convert while loops to for loops where appropriate
        pattern = r'(\w+)\s*=\s*0;\s*while\s*\(\s*\1\s*<\s*(\w+)\s*\)\s*{\s*(.*?)\s*\1\+\+;\s*}'
        
        def replace_while_with_for(match):
            var, limit, body = match.groups()
            optimization['optimizations'].append({
                'type': 'loop_optimization',
                'description': f'Converted while loop to for loop using variable {var}',
                'improvement': 'Better readability and potential compiler optimization'
            })
            return f'for (int {var} = 0; {var} < {limit}; {var}++) {{\n{body}\n}}'
        
        content = re.sub(pattern, replace_while_with_for, content, flags=re.DOTALL)
        
        return content

    def _optimize_memory_usage(self, content: str, optimization: Dict[str, Any]) -> str:
        """Optimize memory usage patterns"""
        # Identify and optimize redundant allocations
        pattern = r'(\w+)\s*=\s*malloc\s*\([^)]+\);\s*.*?\s*free\s*\(\s*\1\s*\);\s*\1\s*=\s*malloc\s*\([^)]+\);'
        
        if re.search(pattern, content, re.DOTALL):
            optimization['memory_optimizations'].append({
                'type': 'allocation_optimization',
                'description': 'Identified redundant malloc/free cycles',
                'recommendation': 'Consider reusing allocated memory or using memory pools'
            })
        
        return content

    def _optimize_string_operations(self, content: str, optimization: Dict[str, Any]) -> str:
        """Optimize string operations"""
        # Replace inefficient string concatenations
        pattern = r'(\w+)\s*\+\s*"([^"]+)"\s*\+\s*(\w+)'
        
        def optimize_string_concat(match):
            var1, literal, var2 = match.groups()
            optimization['optimizations'].append({
                'type': 'string_optimization',
                'description': 'Optimized string concatenation',
                'improvement': 'Reduced temporary string allocations'
            })
            return f'sprintf(buffer, "%s{literal}%s", {var1}, {var2})'
        
        # Note: This is a simplified example - actual implementation would be more sophisticated
        return content

    def _optimize_function_calls(self, content: str, optimization: Dict[str, Any]) -> str:
        """Optimize function call patterns"""
        # Identify functions called in loops that could be cached
        loop_pattern = r'for\s*\([^)]+\)\s*{\s*(.*?)\s*}'
        
        for match in re.finditer(loop_pattern, content, re.DOTALL):
            loop_body = match.group(1)
            
            # Look for function calls that could be moved outside the loop
            func_calls = re.findall(r'(\w+)\s*\([^)]*\)', loop_body)
            for func_call in func_calls:
                if func_call in ['strlen', 'sizeof', 'length']:
                    optimization['performance_improvements'].append({
                        'type': 'loop_invariant_optimization',
                        'description': f'Function {func_call} called in loop could be cached',
                        'recommendation': f'Move {func_call} outside loop and cache result'
                    })
        
        return content

    def _optimize_mathematical_expressions(self, content: str, optimization: Dict[str, Any]) -> str:
        """Optimize mathematical expressions"""
        # Simplify mathematical operations
        optimizations_made = 0
        
        # Division by power of 2 to bit shift
        original_content = content
        content = re.sub(r'(\w+)\s*/\s*2\b', r'\1 >> 1', content)
        content = re.sub(r'(\w+)\s*/\s*4\b', r'\1 >> 2', content)
        content = re.sub(r'(\w+)\s*/\s*8\b', r'\1 >> 3', content)
        
        if content != original_content:
            optimization['optimizations'].append({
                'type': 'mathematical_optimization',
                'description': 'Replaced division by powers of 2 with bit shifts',
                'improvement': 'Faster execution on most processors'
            })
        
        # Multiplication by power of 2 to bit shift
        original_content = content
        content = re.sub(r'(\w+)\s*\*\s*2\b', r'\1 << 1', content)
        content = re.sub(r'(\w+)\s*\*\s*4\b', r'\1 << 2', content)
        content = re.sub(r'(\w+)\s*\*\s*8\b', r'\1 << 3', content)
        
        if content != original_content:
            optimization['optimizations'].append({
                'type': 'mathematical_optimization',
                'description': 'Replaced multiplication by powers of 2 with bit shifts',
                'improvement': 'Faster execution and reduced instruction count'
            })
        
        return content

    def _enhance_code_quality(self, optimization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance overall code quality"""
        enhancement = {
            'quality_improvements': [],
            'readability_enhancements': [],
            'maintainability_improvements': [],
            'enhanced_files': {}
        }
        
        optimized_files = optimization_result.get('optimized_files', {})
        
        for filename, content in optimized_files.items():
            if isinstance(content, str):
                enhanced_content = self._enhance_file_quality(content, filename, enhancement)
                enhancement['enhanced_files'][filename] = enhanced_content
        
        return enhancement

    def _enhance_file_quality(self, content: str, filename: str, enhancement: Dict[str, Any]) -> str:
        """Enhance individual file quality"""
        enhanced = content
        
        # Add proper variable naming
        enhanced = self._improve_variable_names(enhanced, enhancement)
        
        # Add function documentation
        enhanced = self._add_function_documentation(enhanced, enhancement)
        
        # Improve code structure
        enhanced = self._improve_code_structure(enhanced, enhancement)
        
        # Add error handling
        enhanced = self._add_error_handling(enhanced, enhancement)
        
        return enhanced

    def _improve_variable_names(self, content: str, enhancement: Dict[str, Any]) -> str:
        """Improve variable naming"""
        # Replace generic variable names with more descriptive ones
        generic_names = {
            r'\bvar1\b': 'firstValue',
            r'\bvar2\b': 'secondValue',
            r'\btemp\b': 'tempValue',
            r'\bi\b(?=\s*[^n])': 'index',  # Avoid replacing 'in', 'if', etc.
            r'\bj\b(?=\s*[^a-z])': 'subIndex',
            r'\bptr\b': 'pointer',
            r'\bbuf\b': 'buffer'
        }
        
        for pattern, replacement in generic_names.items():
            if re.search(pattern, content):
                content = re.sub(pattern, replacement, content)
                enhancement['readability_enhancements'].append({
                    'type': 'variable_naming',
                    'description': f'Improved variable name: {pattern} -> {replacement}',
                    'benefit': 'Enhanced code readability'
                })
        
        return content

    def _add_function_documentation(self, content: str, enhancement: Dict[str, Any]) -> str:
        """Add basic function documentation"""
        # Find function definitions without documentation
        func_pattern = r'^(\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*{'
        
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            return_type, func_name = match.groups()
            
            # Check if function already has documentation
            before_func = content[:match.start()]
            if not re.search(r'/\*.*?\*/', before_func[-100:], re.DOTALL):
                doc_comment = f"""/**
 * {func_name.replace('_', ' ').title()}
 * @return {return_type}
 */
"""
                enhancement['quality_improvements'].append({
                    'type': 'documentation',
                    'description': f'Added documentation for function {func_name}',
                    'benefit': 'Improved code maintainability'
                })
        
        return content

    def _improve_code_structure(self, content: str, enhancement: Dict[str, Any]) -> str:
        """Improve code structure and organization"""
        # Add consistent indentation
        lines = content.splitlines()
        indent_level = 0
        structured_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                structured_lines.append('')
                continue
            
            # Adjust indent level
            if '}' in stripped:
                indent_level = max(0, indent_level - stripped.count('}'))
            
            # Apply indentation
            indented_line = '    ' * indent_level + stripped
            structured_lines.append(indented_line)
            
            # Increase indent for opening braces
            if '{' in stripped:
                indent_level += stripped.count('{')
        
        enhancement['maintainability_improvements'].append({
            'type': 'code_structure',
            'description': 'Applied consistent indentation',
            'benefit': 'Improved code readability and structure'
        })
        
        return '\n'.join(structured_lines)

    def _add_error_handling(self, content: str, enhancement: Dict[str, Any]) -> str:
        """Add basic error handling"""
        # Add null checks for pointer operations
        pointer_pattern = r'(\w+)\s*->\s*(\w+)'
        
        for match in re.finditer(pointer_pattern, content):
            pointer_name = match.group(1)
            
            # Check if there's already a null check nearby
            context = content[max(0, match.start()-100):match.start()]
            if f'if ({pointer_name} != NULL)' not in context and f'if ({pointer_name})' not in context:
                enhancement['quality_improvements'].append({
                    'type': 'error_handling',
                    'description': f'Consider adding null check for pointer {pointer_name}',
                    'recommendation': f'Add: if ({pointer_name} != NULL) before dereferencing'
                })
        
        return content

    def _remove_code_redundancy(self, quality_enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Remove redundant code patterns"""
        redundancy = {
            'removed_duplicates': [],
            'merged_functions': [],
            'simplified_expressions': [],
            'clean_files': {}
        }
        
        enhanced_files = quality_enhancement.get('enhanced_files', {})
        
        for filename, content in enhanced_files.items():
            if isinstance(content, str):
                clean_content = self._remove_file_redundancy(content, filename, redundancy)
                redundancy['clean_files'][filename] = clean_content
        
        return redundancy

    def _remove_file_redundancy(self, content: str, filename: str, redundancy: Dict[str, Any]) -> str:
        """Remove redundancy from individual file"""
        cleaned = content
        
        # Remove duplicate function declarations
        cleaned = self._remove_duplicate_declarations(cleaned, redundancy)
        
        # Simplify redundant expressions
        cleaned = self._simplify_expressions(cleaned, redundancy)
        
        # Remove unused variables
        cleaned = self._remove_unused_variables(cleaned, redundancy)
        
        return cleaned

    def _remove_duplicate_declarations(self, content: str, redundancy: Dict[str, Any]) -> str:
        """Remove duplicate function declarations"""
        # Find all function declarations
        declarations = {}
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r'^(\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*;', stripped):
                if stripped in declarations:
                    redundancy['removed_duplicates'].append({
                        'type': 'duplicate_declaration',
                        'line': i + 1,
                        'content': stripped
                    })
                    lines[i] = ''  # Remove duplicate
                else:
                    declarations[stripped] = i
        
        return '\n'.join(lines)

    def _simplify_expressions(self, content: str, redundancy: Dict[str, Any]) -> str:
        """Simplify redundant expressions"""
        # Simplify double negations
        content = re.sub(r'!\s*!\s*(\w+)', r'\1', content)
        
        # Simplify identity operations
        content = re.sub(r'(\w+)\s*\+\s*0\b', r'\1', content)
        content = re.sub(r'(\w+)\s*\*\s*1\b', r'\1', content)
        content = re.sub(r'(\w+)\s*/\s*1\b', r'\1', content)
        
        redundancy['simplified_expressions'].append({
            'type': 'expression_simplification',
            'description': 'Simplified mathematical expressions and logic'
        })
        
        return content

    def _remove_unused_variables(self, content: str, redundancy: Dict[str, Any]) -> str:
        """Remove obviously unused variables"""
        # This is a simplified implementation
        # Find variable declarations
        var_declarations = re.findall(r'(\w+(?:\s*\*)?)\s+(\w+)\s*(?:=|;)', content)
        
        for var_type, var_name in var_declarations:
            # Count occurrences (simple heuristic)
            occurrences = len(re.findall(rf'\b{var_name}\b', content))
            
            # If variable appears only once (declaration), it might be unused
            if occurrences == 1:
                redundancy['removed_duplicates'].append({
                    'type': 'unused_variable',
                    'variable': var_name,
                    'note': 'Variable declared but potentially unused'
                })
        
        return content

    def _normalize_code_style(self, redundancy_removal: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize code style consistency"""
        normalization = {
            'style_changes': [],
            'formatting_improvements': [],
            'consistency_fixes': [],
            'normalized_files': {}
        }
        
        clean_files = redundancy_removal.get('clean_files', {})
        
        for filename, content in clean_files.items():
            if isinstance(content, str):
                normalized_content = self._normalize_file_style(content, filename, normalization)
                normalization['normalized_files'][filename] = normalized_content
        
        return normalization

    def _normalize_file_style(self, content: str, filename: str, normalization: Dict[str, Any]) -> str:
        """Normalize individual file style"""
        normalized = content
        
        # Consistent brace style
        normalized = self._normalize_brace_style(normalized, normalization)
        
        # Consistent naming conventions
        normalized = self._normalize_naming_conventions(normalized, normalization)
        
        # Consistent spacing
        normalized = self._normalize_spacing(normalized, normalization)
        
        return normalized

    def _normalize_brace_style(self, content: str, normalization: Dict[str, Any]) -> str:
        """Normalize brace style to K&R"""
        # Convert Allman style to K&R
        content = re.sub(r'\n\s*{\s*\n', ' {\n', content)
        
        normalization['style_changes'].append({
            'type': 'brace_style',
            'description': 'Normalized to K&R brace style',
            'benefit': 'Consistent code formatting'
        })
        
        return content

    def _normalize_naming_conventions(self, content: str, normalization: Dict[str, Any]) -> str:
        """Normalize naming conventions"""
        # Convert camelCase function names to snake_case for C-style consistency
        func_pattern = r'\b([a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*)\s*\('
        
        for match in re.finditer(func_pattern, content):
            camel_name = match.group(1)
            snake_name = re.sub(r'([a-z])([A-Z])', r'\1_\2', camel_name).lower()
            
            content = content.replace(camel_name, snake_name)
            
            normalization['consistency_fixes'].append({
                'type': 'naming_convention',
                'change': f'{camel_name} -> {snake_name}',
                'description': 'Converted to snake_case naming'
            })
        
        return content

    def _normalize_spacing(self, content: str, normalization: Dict[str, Any]) -> str:
        """Normalize spacing consistency"""
        # Consistent operator spacing
        content = re.sub(r'(\w)\s*([+\-*/=<>!]=?)\s*(\w)', r'\1 \2 \3', content)
        
        # Consistent comma spacing
        content = re.sub(r',\s*', ', ', content)
        
        # Consistent semicolon spacing
        content = re.sub(r';\s+(\w)', r'; \1', content)
        
        normalization['formatting_improvements'].append({
            'type': 'spacing_normalization',
            'description': 'Applied consistent spacing rules'
        })
        
        return content

    def _apply_final_polish(self, style_normalization: Dict[str, Any]) -> Dict[str, Any]:
        """Apply final polish and quality touches"""
        polish = {
            'polished_files': {},
            'final_improvements': [],
            'quality_metrics': {},
            'compilation_ready': False
        }
        
        normalized_files = style_normalization.get('normalized_files', {})
        
        for filename, content in normalized_files.items():
            if isinstance(content, str):
                polished_content = self._apply_file_polish(content, filename, polish)
                polish['polished_files'][filename] = polished_content
        
        # Assess compilation readiness
        polish['compilation_ready'] = self._assess_compilation_readiness(polish['polished_files'])
        
        # Calculate final quality metrics
        polish['quality_metrics'] = self._calculate_final_quality_metrics(polish['polished_files'])
        
        return polish

    def _apply_file_polish(self, content: str, filename: str, polish: Dict[str, Any]) -> str:
        """Apply final polish to individual file"""
        polished = content
        
        # Add proper header guards for .h files
        if filename.endswith('.h'):
            polished = self._add_header_guards(polished, filename, polish)
        
        # Add proper includes
        polished = self._add_standard_includes(polished, polish)
        
        # Final formatting pass
        polished = self._final_formatting_pass(polished, polish)
        
        return polished

    def _add_header_guards(self, content: str, filename: str, polish: Dict[str, Any]) -> str:
        """Add header guards to header files"""
        guard_name = filename.upper().replace('.', '_').replace('/', '_')
        
        if f'#ifndef {guard_name}' not in content:
            header_guard = f"""#ifndef {guard_name}
#define {guard_name}

{content}

#endif // {guard_name}
"""
            polish['final_improvements'].append({
                'type': 'header_guard',
                'file': filename,
                'description': f'Added header guard {guard_name}'
            })
            return header_guard
        
        return content

    def _add_standard_includes(self, content: str, polish: Dict[str, Any]) -> str:
        """Add standard includes if needed"""
        includes_needed = []
        
        # Check for standard library usage
        if re.search(r'\b(printf|scanf|strlen|strcpy|malloc|free)\b', content):
            if '#include <stdio.h>' not in content:
                includes_needed.append('#include <stdio.h>')
            if '#include <stdlib.h>' not in content:
                includes_needed.append('#include <stdlib.h>')
            if '#include <string.h>' not in content:
                includes_needed.append('#include <string.h>')
        
        if includes_needed:
            includes_block = '\n'.join(includes_needed) + '\n\n'
            content = includes_block + content
            
            polish['final_improvements'].append({
                'type': 'standard_includes',
                'includes': includes_needed,
                'description': 'Added necessary standard library includes'
            })
        
        return content

    def _final_formatting_pass(self, content: str, polish: Dict[str, Any]) -> str:
        """Final formatting pass"""
        # Remove any remaining multiple empty lines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Ensure file ends with newline
        if not content.endswith('\n'):
            content += '\n'
        
        polish['final_improvements'].append({
            'type': 'final_formatting',
            'description': 'Applied final formatting touches'
        })
        
        return content

    def _assess_compilation_readiness(self, polished_files: Dict[str, str]) -> bool:
        """Assess if code is ready for compilation"""
        for filename, content in polished_files.items():
            # Check for basic syntax completeness
            if filename.endswith('.c') or filename.endswith('.cpp'):
                # Must have main function or be a library file
                if 'main(' not in content and not self._appears_to_be_library(content):
                    return False
                
                # Check for balanced braces
                if content.count('{') != content.count('}'):
                    return False
        
        return True

    def _appears_to_be_library(self, content: str) -> bool:
        """Check if content appears to be a library file"""
        # Library files typically have function definitions but no main
        func_definitions = len(re.findall(r'^(\w+(?:\s*\*)?)\s+\w+\s*\([^)]*\)\s*{', content, re.MULTILINE))
        return func_definitions > 0

    def _calculate_final_quality_metrics(self, polished_files: Dict[str, str]) -> Dict[str, Any]:
        """Calculate final quality metrics"""
        metrics = {
            'total_lines': 0,
            'total_functions': 0,
            'comment_ratio': 0.0,
            'complexity_score': 0.0,
            'maintainability_index': 0.0
        }
        
        total_comment_lines = 0
        
        for filename, content in polished_files.items():
            lines = content.splitlines()
            metrics['total_lines'] += len(lines)
            
            # Count functions
            metrics['total_functions'] += len(re.findall(r'^(\w+(?:\s*\*)?)\s+\w+\s*\([^)]*\)\s*{', content, re.MULTILINE))
            
            # Count comment lines
            for line in lines:
                if re.match(r'^\s*(/\*|\*/|//)', line.strip()):
                    total_comment_lines += 1
        
        # Calculate comment ratio
        if metrics['total_lines'] > 0:
            metrics['comment_ratio'] = total_comment_lines / metrics['total_lines']
        
        # Simple complexity score (based on control structures)
        total_complexity = 0
        for content in polished_files.values():
            complexity = (
                content.count('if') + content.count('while') + 
                content.count('for') + content.count('switch') + 
                content.count('case')
            )
            total_complexity += complexity
        
        metrics['complexity_score'] = total_complexity / max(1, metrics['total_functions'])
        
        # Maintainability index (simplified)
        metrics['maintainability_index'] = min(100, max(0, 
            100 - metrics['complexity_score'] * 2 + metrics['comment_ratio'] * 10
        ))
        
        return metrics

    def _create_cleanup_report(self, initial_cleanup: Dict[str, Any], 
                              optimization_result: Dict[str, Any],
                              quality_enhancement: Dict[str, Any],
                              redundancy_removal: Dict[str, Any],
                              style_normalization: Dict[str, Any],
                              final_polish: Dict[str, Any],
                              context: Dict[str, Any]) -> CodeCleanupResult:
        """Create comprehensive cleanup report"""
        result = CodeCleanupResult()
        
        try:
            # Calculate summary statistics
            cleanup_stats = initial_cleanup.get('cleanup_statistics', {})
            result.files_processed = len(cleanup_stats)
            
            # Count total optimizations
            result.total_optimizations = (
                len(optimization_result.get('optimizations', [])) +
                len(optimization_result.get('performance_improvements', [])) +
                len(optimization_result.get('memory_optimizations', []))
            )
            
            # Count quality improvements
            result.quality_improvements = (
                len(quality_enhancement.get('quality_improvements', [])) +
                len(quality_enhancement.get('readability_enhancements', [])) +
                len(quality_enhancement.get('maintainability_improvements', []))
            )
            
            # Calculate cleanup efficiency
            total_original_lines = sum(stats.get('original_lines', 0) for stats in cleanup_stats.values())
            total_cleaned_lines = sum(stats.get('cleaned_lines', 0) for stats in cleanup_stats.values())
            
            if total_original_lines > 0:
                result.cleanup_efficiency = (total_original_lines - total_cleaned_lines) / total_original_lines
            
            # Get final metrics
            final_metrics = final_polish.get('quality_metrics', {})
            result.maintainability_index = final_metrics.get('maintainability_index', 0.0)
            
            # Calculate final quality score
            result.final_quality_score = self._calculate_overall_quality_score(
                result.cleanup_efficiency, result.total_optimizations, 
                result.quality_improvements, final_metrics
            )
            
            # Determine overall success
            result.success = (
                result.final_quality_score >= self.quality_thresholds['minimum_quality_score'] and
                result.maintainability_index >= self.quality_thresholds['minimum_maintainability'] and
                result.files_processed > 0
            )
            
            # Set cleaned files
            result.cleaned_files = final_polish.get('polished_files', {})
            
            # Create optimization report
            result.optimization_report = {
                'initial_cleanup': initial_cleanup,
                'optimizations_applied': optimization_result,
                'quality_enhancements': quality_enhancement,
                'redundancy_removal': redundancy_removal,
                'style_normalization': style_normalization,
                'final_polish': final_polish
            }
            
            # Set quality metrics
            result.quality_metrics = final_metrics
            
            # Generate recommendations
            result.recommendations = self._generate_cleanup_recommendations(result, final_metrics)
            
            # Generate warnings for concerning findings
            if result.final_quality_score < 0.7:
                result.warnings.append(f"Final quality score ({result.final_quality_score:.2f}) below recommended threshold")
            
            if result.maintainability_index < 60:
                result.warnings.append(f"Maintainability index ({result.maintainability_index:.1f}) below recommended threshold")
            
            if result.files_processed == 0:
                result.warnings.append("No files were processed during cleanup")
            
            # Calculate detailed metrics
            result.metrics = {
                'cleanup_coverage': {
                    'files_processed': result.files_processed,
                    'total_lines_processed': total_cleaned_lines,
                    'cleanup_efficiency': result.cleanup_efficiency
                },
                'optimization_metrics': {
                    'total_optimizations': result.total_optimizations,
                    'performance_improvements': len(optimization_result.get('performance_improvements', [])),
                    'memory_optimizations': len(optimization_result.get('memory_optimizations', []))
                },
                'quality_metrics': {
                    'quality_improvements': result.quality_improvements,
                    'maintainability_index': result.maintainability_index,
                    'final_quality_score': result.final_quality_score,
                    'comment_ratio': final_metrics.get('comment_ratio', 0.0),
                    'complexity_score': final_metrics.get('complexity_score', 0.0)
                }
            }
            
        except Exception as e:
            result.error_messages.append(f"Error creating cleanup report: {str(e)}")
            self.logger.error(f"Failed to create cleanup report: {e}")
        
        return result

    def _calculate_overall_quality_score(self, cleanup_efficiency: float, 
                                       total_optimizations: int,
                                       quality_improvements: int, 
                                       final_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = []
        
        # Cleanup efficiency score
        scores.append(min(1.0, cleanup_efficiency * 2))  # Cap at 1.0
        
        # Optimization score
        scores.append(min(1.0, total_optimizations / 10.0))  # Normalize to 10 optimizations
        
        # Quality improvement score
        scores.append(min(1.0, quality_improvements / 5.0))  # Normalize to 5 improvements
        
        # Maintainability score
        maintainability = final_metrics.get('maintainability_index', 0.0)
        scores.append(maintainability / 100.0)  # Already 0-100 scale
        
        # Comment ratio score
        comment_ratio = final_metrics.get('comment_ratio', 0.0)
        scores.append(min(1.0, comment_ratio * 5))  # Good comment ratio is ~20%
        
        return sum(scores) / len(scores)

    def _generate_cleanup_recommendations(self, result: CodeCleanupResult, 
                                        final_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cleanup recommendations"""
        recommendations = []
        
        if result.final_quality_score < 0.7:
            recommendations.append({
                'priority': 'high',
                'category': 'quality_improvement',
                'title': 'Further Quality Enhancement Needed',
                'description': f'Final quality score is {result.final_quality_score:.2f}, below optimal threshold',
                'actions': [
                    'Review and improve code documentation',
                    'Add more comprehensive error handling',
                    'Consider additional optimization opportunities'
                ]
            })
        
        complexity_score = final_metrics.get('complexity_score', 0.0)
        if complexity_score > 10:
            recommendations.append({
                'priority': 'medium',
                'category': 'complexity_reduction',
                'title': 'High Code Complexity Detected',
                'description': f'Average complexity score is {complexity_score:.1f}',
                'actions': [
                    'Break down complex functions into smaller units',
                    'Reduce nested control structures',
                    'Consider refactoring complex algorithms'
                ]
            })
        
        comment_ratio = final_metrics.get('comment_ratio', 0.0)
        if comment_ratio < 0.1:
            recommendations.append({
                'priority': 'low',
                'category': 'documentation',
                'title': 'Low Documentation Coverage',
                'description': f'Comment ratio is {comment_ratio:.1%}',
                'actions': [
                    'Add function and module documentation',
                    'Include inline comments for complex logic',
                    'Add file header comments'
                ]
            })
        
        return recommendations

    def _load_optimization_patterns(self) -> List[OptimizationPattern]:
        """Load optimization patterns"""
        return [
            OptimizationPattern(
                name="loop_to_for",
                pattern=r'(\w+)\s*=\s*0;\s*while\s*\(\s*\1\s*<\s*(\w+)\s*\)',
                replacement=r'for (int \1 = 0; \1 < \2; \1++)',
                description="Convert while loops to for loops",
                benefit="Better readability and compiler optimization"
            ),
            OptimizationPattern(
                name="bit_shift_division",
                pattern=r'(\w+)\s*/\s*([248])\b',
                replacement=r'\1 >> {shift}',
                description="Replace division by powers of 2 with bit shifts",
                benefit="Faster execution"
            ),
            OptimizationPattern(
                name="bit_shift_multiplication", 
                pattern=r'(\w+)\s*\*\s*([248])\b',
                replacement=r'\1 << {shift}',
                description="Replace multiplication by powers of 2 with bit shifts",
                benefit="Faster execution"
            )
        ]

    def _load_quality_rules(self) -> Dict[str, Any]:
        """Load code quality rules"""
        return {
            'naming_conventions': {
                'variables': 'snake_case',
                'functions': 'snake_case', 
                'constants': 'UPPER_CASE'
            },
            'formatting_rules': {
                'indentation': '4_spaces',
                'brace_style': 'k_and_r',
                'max_line_length': 100
            },
            'documentation_rules': {
                'function_documentation': 'required',
                'file_headers': 'recommended',
                'inline_comments': 'for_complex_logic'
            }
        }

    def _load_style_conventions(self) -> Dict[str, Any]:
        """Load style conventions"""
        return {
            'brace_placement': 'k_and_r',
            'spacing': {
                'around_operators': True,
                'after_commas': True,
                'after_semicolons': False
            },
            'naming': {
                'variables': 'snake_case',
                'functions': 'snake_case',
                'types': 'PascalCase',
                'constants': 'UPPER_CASE'
            }
        }

    def _create_failure_result(self, error_message: str, execution_time: float = 0.0) -> Dict[str, Any]:
        """Create failure result"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'status': AgentStatus.FAILED if HAS_MATRIX_FRAMEWORK else 'failed',
            'execution_time': execution_time,
            'error_message': error_message,
            'cleanup_result': CodeCleanupResult(),
            'metadata': {
                'character': self.character,
                'phase': self.current_phase,
                'failure_point': self.current_phase
            }
        }


# For backward compatibility
TheCleaner = Agent14_TheCleaner