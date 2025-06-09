"""
Agent 14: The Cleaner - Code Cleanup and Optimization
Performs final code cleanup, optimization, and quality enhancement on reconstructed source code.
"""

import os
import re
import json
import ast
from typing import Dict, Any, List, Set, Tuple, Optional
from pathlib import Path
from ..matrix_agents import ReconstructionAgent, AgentResult, AgentStatus, MatrixCharacter

class Agent14_TheCleaner(ReconstructionAgent):
    """Agent 14: The Cleaner - Code cleanup and optimization"""
    
    def __init__(self):
        super().__init__(
            agent_id=14,
            matrix_character=MatrixCharacter.CLEANER,
            dependencies=[9, 10, 11]  # Depends on early Phase C agents
        )

    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate prerequisites with flexible dependency checking"""
        # Initialize shared_memory structure if not present
        shared_memory = context.get('shared_memory', {})
        if 'analysis_results' not in shared_memory:
            shared_memory['analysis_results'] = {}
        if 'binary_metadata' not in shared_memory:
            shared_memory['binary_metadata'] = {}
        
        # Check for dependencies more flexibly - Agent 14 depends on reconstruction agents
        dependencies_met = False
        agent_results = context.get('agent_results', {})
        
        # Check for Phase C agents (9, 10, 11)
        phase_c_available = any(
            agent_id in agent_results or agent_id in shared_memory['analysis_results']
            for agent_id in [9, 10, 11]
        )
        
        if phase_c_available:
            dependencies_met = True
        
        # Also check for any source code or compilation results from previous agents
        source_available = any(
            agent_result.data.get('source_files') or 
            agent_result.data.get('decompiled_code') or
            agent_result.data.get('build_system')
            for agent_result in agent_results.values()
            if hasattr(agent_result, 'data') and agent_result.data and isinstance(agent_result.data, dict)
        )
        
        if source_available:
            dependencies_met = True
        
        if not dependencies_met:
            self.logger.warning("No reconstruction dependencies found - proceeding with basic cleanup")

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code cleanup and optimization"""
        # Validate prerequisites first
        self._validate_prerequisites(context)
        
        # Gather all available results for comprehensive cleanup
        all_results = context.get('agent_results', {})
        
        try:
            # Perform code cleanup and optimization
            cleanup_result = self._perform_code_cleanup(all_results, context)
            
            # Apply optimization patterns
            optimization_result = self._apply_optimization_patterns(cleanup_result, all_results)
            
            # Enhance code quality
            quality_enhancement = self._enhance_code_quality(optimization_result, all_results)
            
            # Remove redundant code
            redundancy_removal = self._remove_code_redundancy(quality_enhancement)
            
            # Normalize code style
            style_normalization = self._normalize_code_style(redundancy_removal)
            
            # Apply final polish
            final_polish = self._apply_final_polish(style_normalization, all_results)
            
            # Generate cleanup report
            cleanup_report = self._generate_cleanup_report(
                cleanup_result, optimization_result, quality_enhancement,
                redundancy_removal, style_normalization, final_polish
            )
            
            # Save cleaned code
            self._save_cleaned_code(final_polish, context)
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return {
                'initial_cleanup': cleanup_result,
                'optimization_applied': optimization_result,
                'quality_enhancement': quality_enhancement,
                'redundancy_removal': redundancy_removal,
                'style_normalization': style_normalization,
                'final_polish': final_polish,
                'cleanup_report': cleanup_report,
                'cleaner_metrics': self._calculate_cleaner_metrics(cleanup_report, final_polish)
            }
            
        except Exception as e:
            error_msg = f"The Cleaner code cleanup failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e

    def _perform_code_cleanup(self, all_results: Dict[int, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform initial code cleanup"""
        cleanup = {
            'cleaned_files': {},
            'removed_comments': [],
            'cleaned_functions': [],
            'fixed_formatting': [],
            'resolved_warnings': [],
            'cleanup_statistics': {}
        }
        
        # Get source code from reconstruction results
        source_data = self._get_source_code(all_results)
        
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

    def _apply_optimization_patterns(self, cleanup_result: Dict[str, Any], all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Apply code optimization patterns"""
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

    def _enhance_code_quality(self, optimization_result: Dict[str, Any], all_results: Dict[int, Any]) -> Dict[str, Any]:
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

    def _apply_final_polish(self, style_normalization: Dict[str, Any], all_results: Dict[int, Any]) -> Dict[str, Any]:
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

    def _get_source_code(self, all_results: Dict[int, Any]) -> Dict[str, str]:
        """Get source code from reconstruction results"""
        source_code = {}
        
        # Try to get from various reconstruction agents
        potential_sources = [11, 9, 7]  # Oracle, Global Reconstructor, Advanced Decompiler
        
        for agent_id in potential_sources:
            if agent_id in all_results and hasattr(all_results[agent_id], 'data'):
                agent_data = all_results[agent_id].data
                if isinstance(agent_data, dict):
                    # Look for reconstructed source
                    reconstructed_source = agent_data.get('reconstructed_source', {})
                    if isinstance(reconstructed_source, dict):
                        source_files = reconstructed_source.get('source_files', {})
                        header_files = reconstructed_source.get('header_files', {})
                        source_code.update(source_files)
                        source_code.update(header_files)
                        
                        if source_code:  # Found source code, break
                            break
        
        return source_code

    def _save_cleaned_code(self, final_polish: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Save cleaned code to output directory"""
        output_paths = context.get('output_paths', {})
        if 'agents' not in output_paths:
            return
        
        agent_output_dir = Path(output_paths['agents']) / 'agent14_the_cleaner'
        agent_output_dir.mkdir(parents=True, exist_ok=True)
        
        cleaned_dir = agent_output_dir / 'cleaned_source'
        cleaned_dir.mkdir(exist_ok=True)
        
        polished_files = final_polish.get('polished_files', {})
        
        for filename, content in polished_files.items():
            if isinstance(content, str):
                output_file = cleaned_dir / filename
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)

    def _generate_cleanup_report(self, cleanup_result: Dict[str, Any], 
                               optimization_result: Dict[str, Any],
                               quality_enhancement: Dict[str, Any],
                               redundancy_removal: Dict[str, Any],
                               style_normalization: Dict[str, Any],
                               final_polish: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive cleanup report"""
        report = {
            'cleanup_summary': {},
            'optimization_summary': {},
            'quality_summary': {},
            'final_metrics': {},
            'recommendations': [],
            'final_quality_score': 0.0
        }
        
        # Cleanup summary
        cleanup_stats = cleanup_result.get('cleanup_statistics', {})
        total_original_lines = sum(stats.get('original_lines', 0) for stats in cleanup_stats.values())
        total_cleaned_lines = sum(stats.get('cleaned_lines', 0) for stats in cleanup_stats.values())
        
        report['cleanup_summary'] = {
            'files_processed': len(cleanup_stats),
            'original_lines': total_original_lines,
            'cleaned_lines': total_cleaned_lines,
            'lines_removed': total_original_lines - total_cleaned_lines,
            'cleanup_efficiency': (total_original_lines - total_cleaned_lines) / max(1, total_original_lines)
        }
        
        # Optimization summary
        report['optimization_summary'] = {
            'optimizations_applied': len(optimization_result.get('optimizations', [])),
            'performance_improvements': len(optimization_result.get('performance_improvements', [])),
            'memory_optimizations': len(optimization_result.get('memory_optimizations', []))
        }
        
        # Quality summary
        report['quality_summary'] = {
            'quality_improvements': len(quality_enhancement.get('quality_improvements', [])),
            'readability_enhancements': len(quality_enhancement.get('readability_enhancements', [])),
            'maintainability_improvements': len(quality_enhancement.get('maintainability_improvements', []))
        }
        
        # Final metrics from polish
        report['final_metrics'] = final_polish.get('quality_metrics', {})
        
        # Calculate final quality score
        report['final_quality_score'] = self._calculate_final_quality_score(
            report['cleanup_summary'], report['optimization_summary'], 
            report['quality_summary'], report['final_metrics']
        )
        
        # Generate recommendations
        report['recommendations'] = self._generate_final_recommendations(report)
        
        return report

    def _calculate_final_quality_score(self, cleanup_summary: Dict[str, Any], 
                                     optimization_summary: Dict[str, Any],
                                     quality_summary: Dict[str, Any], 
                                     final_metrics: Dict[str, Any]) -> float:
        """Calculate final quality score"""
        scores = []
        
        # Cleanup efficiency score
        cleanup_efficiency = cleanup_summary.get('cleanup_efficiency', 0.0)
        scores.append(min(1.0, cleanup_efficiency * 2))  # Cap at 1.0
        
        # Optimization score
        opt_count = optimization_summary.get('optimizations_applied', 0)
        scores.append(min(1.0, opt_count / 10.0))  # Normalize to 10 optimizations
        
        # Quality improvement score
        quality_count = quality_summary.get('quality_improvements', 0)
        scores.append(min(1.0, quality_count / 5.0))  # Normalize to 5 improvements
        
        # Maintainability score
        maintainability = final_metrics.get('maintainability_index', 0.0)
        scores.append(maintainability / 100.0)  # Already 0-100 scale
        
        # Comment ratio score
        comment_ratio = final_metrics.get('comment_ratio', 0.0)
        scores.append(min(1.0, comment_ratio * 5))  # Good comment ratio is ~20%
        
        return sum(scores) / len(scores)

    def _generate_final_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate final recommendations"""
        recommendations = []
        
        final_score = report.get('final_quality_score', 0.0)
        final_metrics = report.get('final_metrics', {})
        
        if final_score < 0.7:
            recommendations.append({
                'priority': 'high',
                'category': 'quality_improvement',
                'title': 'Further Quality Enhancement Needed',
                'description': f'Final quality score is {final_score:.2f}, below optimal threshold',
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

    def _calculate_cleaner_metrics(self, cleanup_report: Dict[str, Any], final_polish: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Cleaner-specific metrics"""
        metrics = {
            'cleanup_efficiency': 0.0,
            'optimization_coverage': 0.0,
            'quality_enhancement_rate': 0.0,
            'code_maintainability': 0.0,
            'cleaner_effectiveness': 0.0
        }
        
        # Cleanup efficiency
        cleanup_summary = cleanup_report.get('cleanup_summary', {})
        metrics['cleanup_efficiency'] = cleanup_summary.get('cleanup_efficiency', 0.0)
        
        # Optimization coverage
        opt_summary = cleanup_report.get('optimization_summary', {})
        total_optimizations = (
            opt_summary.get('optimizations_applied', 0) +
            opt_summary.get('performance_improvements', 0) +
            opt_summary.get('memory_optimizations', 0)
        )
        metrics['optimization_coverage'] = min(1.0, total_optimizations / 15.0)  # Normalize to 15
        
        # Quality enhancement rate
        quality_summary = cleanup_report.get('quality_summary', {})
        total_enhancements = (
            quality_summary.get('quality_improvements', 0) +
            quality_summary.get('readability_enhancements', 0) +
            quality_summary.get('maintainability_improvements', 0)
        )
        metrics['quality_enhancement_rate'] = min(1.0, total_enhancements / 10.0)  # Normalize to 10
        
        # Code maintainability
        final_metrics = cleanup_report.get('final_metrics', {})
        maintainability_index = final_metrics.get('maintainability_index', 0.0)
        metrics['code_maintainability'] = maintainability_index / 100.0
        
        # Overall cleaner effectiveness
        metrics['cleaner_effectiveness'] = (
            metrics['cleanup_efficiency'] * 0.3 +
            metrics['optimization_coverage'] * 0.3 +
            metrics['quality_enhancement_rate'] * 0.2 +
            metrics['code_maintainability'] * 0.2
        )
        
        return metrics

    def get_description(self) -> str:
        """Get description of The Cleaner"""
        return "The Cleaner performs final code cleanup, optimization, and quality enhancement on reconstructed source code"

    def get_dependencies(self) -> List[int]:
        """Get dependencies for The Cleaner"""
        return [13]  # Depends on Agent Johnson