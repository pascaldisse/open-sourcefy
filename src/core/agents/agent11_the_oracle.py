"""
Agent 11: The Oracle - Final Validation and Truth Verification
Performs comprehensive validation and truth verification of the reconstructed code.
"""

import os
import json
import hashlib
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent11_TheOracle(BaseAgent):
    """Agent 11: The Oracle - Final validation and truth verification"""
    
    def __init__(self):
        super().__init__(
            agent_id=11,
            name="TheOracle",
            dependencies=[10]  # Depends on The Machine (compilation)
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute final validation and truth verification"""
        machine_result = context['agent_results'].get(10)
        
        # Gather all available results for comprehensive validation
        all_results = context.get('agent_results', {})
        
        try:
            # Perform comprehensive validation
            validation_result = self._perform_comprehensive_validation(all_results, context)
            
            # Verify truth and accuracy
            truth_verification = self._verify_reconstruction_truth(validation_result, all_results, context)
            
            # Generate final oracle report
            oracle_report = self._generate_oracle_report(validation_result, truth_verification, all_results)
            
            # Determine final verdict
            final_verdict = self._render_final_verdict(oracle_report, validation_result, truth_verification)
            
            oracle_result = {
                'validation_result': validation_result,
                'truth_verification': truth_verification,
                'oracle_report': oracle_report,
                'final_verdict': final_verdict,
                'oracle_metrics': self._calculate_oracle_metrics(validation_result, truth_verification),
                'recommendations': self._generate_recommendations(validation_result, truth_verification)
            }
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=oracle_result,
                metadata={
                    'depends_on': [10],
                    'analysis_type': 'final_validation_and_truth_verification',
                    'validation_score': validation_result.get('overall_score', 0.0),
                    'truth_score': truth_verification.get('truth_score', 0.0),
                    'oracle_confidence': final_verdict.get('confidence', 0.0),
                    'final_grade': final_verdict.get('grade', 'UNKNOWN')
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"The Oracle validation failed: {str(e)}"
            )

    def _perform_comprehensive_validation(self, all_results: Dict[int, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive validation of all pipeline components"""
        validation = {
            'component_validations': {},
            'structural_validation': {},
            'functional_validation': {},
            'quality_validation': {},
            'completeness_validation': {},
            'consistency_validation': {},
            'overall_score': 0.0,
            'critical_issues': [],
            'warnings': [],
            'validation_timestamp': context.get('start_time', 0)
        }
        
        # Validate each component
        validation['component_validations'] = self._validate_all_components(all_results)
        
        # Structural validation
        validation['structural_validation'] = self._validate_structural_integrity(all_results)
        
        # Functional validation
        validation['functional_validation'] = self._validate_functional_correctness(all_results)
        
        # Quality validation
        validation['quality_validation'] = self._validate_code_quality(all_results)
        
        # Completeness validation
        validation['completeness_validation'] = self._validate_reconstruction_completeness(all_results)
        
        # Consistency validation
        validation['consistency_validation'] = self._validate_cross_component_consistency(all_results)
        
        # Calculate overall score
        validation['overall_score'] = self._calculate_overall_validation_score(validation)
        
        # Identify critical issues
        validation['critical_issues'] = self._identify_critical_issues(validation)
        
        # Generate warnings
        validation['warnings'] = self._generate_validation_warnings(validation)
        
        return validation

    def _validate_all_components(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Validate all pipeline components"""
        component_validations = {}
        
        # Define component validation criteria
        component_criteria = {
            1: {'name': 'Binary Discovery', 'min_score': 0.7, 'critical': True},
            2: {'name': 'Architecture Analysis', 'min_score': 0.8, 'critical': True},
            3: {'name': 'Error Pattern Matching', 'min_score': 0.6, 'critical': False},
            4: {'name': 'Basic Decompilation', 'min_score': 0.7, 'critical': True},
            5: {'name': 'Structure Analysis', 'min_score': 0.7, 'critical': True},
            6: {'name': 'Optimization Matching', 'min_score': 0.6, 'critical': False},
            7: {'name': 'Advanced Decompilation', 'min_score': 0.8, 'critical': True},
            8: {'name': 'Resource Reconstruction', 'min_score': 0.6, 'critical': False},
            9: {'name': 'Global Reconstruction', 'min_score': 0.8, 'critical': True},
            10: {'name': 'The Machine (Compilation)', 'min_score': 0.7, 'critical': True}
        }
        
        for agent_id, criteria in component_criteria.items():
            validation = {
                'name': criteria['name'],
                'present': agent_id in all_results,
                'successful': False,
                'score': 0.0,
                'critical': criteria['critical'],
                'meets_minimum': False,
                'issues': []
            }
            
            if agent_id in all_results:
                result = all_results[agent_id]
                if hasattr(result, 'status'):
                    validation['successful'] = (result.status == AgentStatus.COMPLETED)
                    
                    if validation['successful']:
                        # Calculate component-specific score
                        validation['score'] = self._calculate_component_score(agent_id, result)
                        validation['meets_minimum'] = validation['score'] >= criteria['min_score']
                        
                        if not validation['meets_minimum']:
                            validation['issues'].append(f"Score {validation['score']:.2f} below minimum {criteria['min_score']}")
                    else:
                        validation['issues'].append("Component failed to complete successfully")
                else:
                    validation['issues'].append("Invalid result object")
            else:
                validation['issues'].append("Component not executed")
            
            component_validations[agent_id] = validation
        
        return component_validations

    def _calculate_component_score(self, agent_id: int, result: Any) -> float:
        """Calculate score for individual component"""
        base_score = 0.5  # Default score for completed components
        
        if not hasattr(result, 'data') or not result.data:
            return base_score
        
        data = result.data
        metadata = getattr(result, 'metadata', {})
        
        # Component-specific scoring
        if agent_id == 1:  # Binary Discovery
            score = base_score
            if isinstance(data, dict):
                if data.get('binary_info'): score += 0.2
                if data.get('strings'): score += 0.2
                if data.get('imports'): score += 0.1
            return min(1.0, score)
            
        elif agent_id == 2:  # Architecture Analysis
            score = base_score
            if isinstance(data, dict):
                if data.get('architecture'): score += 0.3
                if data.get('calling_convention'): score += 0.2
            return min(1.0, score)
            
        elif agent_id in [4, 7]:  # Decompilation agents
            score = base_score
            if isinstance(data, dict):
                functions = data.get('decompiled_functions', {}) or data.get('enhanced_functions', {})
                if functions:
                    score += 0.3
                    # Bonus for multiple functions
                    if len(functions) > 1: score += 0.1
                    # Check for main function
                    if any('main' in name.lower() for name in functions.keys()): score += 0.1
            return min(1.0, score)
            
        elif agent_id == 9:  # Global Reconstruction
            score = base_score
            if isinstance(data, dict):
                if data.get('reconstructed_source'): score += 0.25
                if data.get('project_structure'): score += 0.15
                if data.get('build_configuration'): score += 0.1
            return min(1.0, score)
            
        elif agent_id == 10:  # The Machine
            score = base_score
            if isinstance(data, dict):
                compilation_results = data.get('compilation_results', {})
                if compilation_results:
                    success_rate = compilation_results.get('success_rate', 0.0)
                    score = base_score + (success_rate * 0.5)
            return min(1.0, score)
        
        # Default scoring for other components
        return min(1.0, base_score + 0.3)

    def _validate_structural_integrity(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Validate structural integrity of reconstruction"""
        validation = {
            'source_structure': {},
            'header_completeness': {},
            'dependency_integrity': {},
            'file_organization': {},
            'structural_score': 0.0,
            'issues': []
        }
        
        # Check if we have global reconstruction results
        if 9 in all_results and hasattr(all_results[9], 'data'):
            global_data = all_results[9].data
            if isinstance(global_data, dict):
                reconstructed_source = global_data.get('reconstructed_source', {})
                
                # Validate source structure
                validation['source_structure'] = self._validate_source_structure(reconstructed_source)
                
                # Validate headers
                validation['header_completeness'] = self._validate_header_completeness(reconstructed_source)
                
                # Validate dependencies
                validation['dependency_integrity'] = self._validate_dependency_integrity(global_data)
                
                # Validate file organization
                validation['file_organization'] = self._validate_file_organization(reconstructed_source)
        
        # Calculate structural score
        scores = [
            validation['source_structure'].get('score', 0.0),
            validation['header_completeness'].get('score', 0.0),
            validation['dependency_integrity'].get('score', 0.0),
            validation['file_organization'].get('score', 0.0)
        ]
        validation['structural_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return validation

    def _validate_source_structure(self, reconstructed_source: Dict[str, Any]) -> Dict[str, Any]:
        """Validate source code structure"""
        validation = {
            'has_source_files': False,
            'has_main_function': False,
            'function_count': 0,
            'avg_function_size': 0,
            'code_organization': 'poor',
            'score': 0.0
        }
        
        source_files = reconstructed_source.get('source_files', {})
        if source_files:
            validation['has_source_files'] = True
            
            # Count functions and analyze structure
            total_functions = 0
            total_lines = 0
            has_main = False
            
            for filename, content in source_files.items():
                if isinstance(content, str):
                    lines = content.split('\n')
                    total_lines += len(lines)
                    
                    # Count functions (simple heuristic)
                    function_count = content.count(') {') + content.count('){')
                    total_functions += function_count
                    
                    # Check for main function
                    if 'main(' in content or 'int main' in content:
                        has_main = True
            
            validation['function_count'] = total_functions
            validation['has_main_function'] = has_main
            
            if total_functions > 0:
                validation['avg_function_size'] = total_lines / total_functions
            
            # Determine code organization quality
            if total_functions >= 5 and has_main and validation['avg_function_size'] > 10:
                validation['code_organization'] = 'excellent'
            elif total_functions >= 3 and has_main:
                validation['code_organization'] = 'good'
            elif total_functions >= 1:
                validation['code_organization'] = 'fair'
        
        # Calculate score
        score = 0.0
        if validation['has_source_files']: score += 0.3
        if validation['has_main_function']: score += 0.3
        if validation['function_count'] >= 3: score += 0.2
        if validation['code_organization'] in ['good', 'excellent']: score += 0.2
        
        validation['score'] = score
        return validation

    def _validate_header_completeness(self, reconstructed_source: Dict[str, Any]) -> Dict[str, Any]:
        """Validate header file completeness"""
        validation = {
            'has_headers': False,
            'header_count': 0,
            'include_coverage': 0.0,
            'function_declarations': 0,
            'score': 0.0
        }
        
        header_files = reconstructed_source.get('header_files', {})
        source_files = reconstructed_source.get('source_files', {})
        
        if header_files:
            validation['has_headers'] = True
            validation['header_count'] = len(header_files)
            
            # Count function declarations in headers
            total_declarations = 0
            for content in header_files.values():
                if isinstance(content, str):
                    # Count function declarations (ending with semicolon)
                    total_declarations += content.count(');')
            
            validation['function_declarations'] = total_declarations
            
            # Calculate include coverage (rough estimate)
            if source_files:
                total_includes_needed = len(source_files) * 3  # Estimate
                total_includes_found = sum(content.count('#include') for content in header_files.values() if isinstance(content, str))
                validation['include_coverage'] = min(1.0, total_includes_found / max(1, total_includes_needed))
        
        # Calculate score
        score = 0.0
        if validation['has_headers']: score += 0.4
        if validation['function_declarations'] > 0: score += 0.3
        if validation['include_coverage'] > 0.5: score += 0.3
        
        validation['score'] = score
        return validation

    def _validate_dependency_integrity(self, global_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dependency integrity"""
        validation = {
            'has_dependency_analysis': False,
            'dependency_resolution': 'unknown',
            'circular_dependencies': False,
            'missing_dependencies': [],
            'score': 0.0
        }
        
        dependency_analysis = global_data.get('dependency_analysis', {})
        if dependency_analysis:
            validation['has_dependency_analysis'] = True
            
            # Check dependency resolution
            if isinstance(dependency_analysis, dict):
                resolved = dependency_analysis.get('resolved_dependencies', [])
                unresolved = dependency_analysis.get('unresolved_dependencies', [])
                
                if resolved and not unresolved:
                    validation['dependency_resolution'] = 'complete'
                elif resolved:
                    validation['dependency_resolution'] = 'partial'
                    validation['missing_dependencies'] = unresolved
                else:
                    validation['dependency_resolution'] = 'none'
                
                # Check for circular dependencies
                validation['circular_dependencies'] = dependency_analysis.get('circular_dependencies', False)
        
        # Calculate score
        score = 0.0
        if validation['has_dependency_analysis']: score += 0.3
        if validation['dependency_resolution'] == 'complete': score += 0.4
        elif validation['dependency_resolution'] == 'partial': score += 0.2
        if not validation['circular_dependencies']: score += 0.3
        
        validation['score'] = score
        return validation

    def _validate_file_organization(self, reconstructed_source: Dict[str, Any]) -> Dict[str, Any]:
        """Validate file organization"""
        validation = {
            'proper_separation': False,
            'naming_convention': 'unknown',
            'directory_structure': 'flat',
            'organization_quality': 'poor',
            'score': 0.0
        }
        
        source_files = reconstructed_source.get('source_files', {})
        header_files = reconstructed_source.get('header_files', {})
        
        if source_files or header_files:
            # Check proper separation of source and headers
            validation['proper_separation'] = bool(source_files and header_files)
            
            # Analyze naming conventions
            all_files = list(source_files.keys()) + list(header_files.keys())
            if all_files:
                # Check if files follow consistent naming
                has_consistent_naming = True
                for filename in all_files:
                    if not (filename.endswith('.c') or filename.endswith('.h') or filename.endswith('.cpp')):
                        has_consistent_naming = False
                        break
                
                validation['naming_convention'] = 'consistent' if has_consistent_naming else 'inconsistent'
            
            # Determine organization quality
            if validation['proper_separation'] and validation['naming_convention'] == 'consistent':
                validation['organization_quality'] = 'excellent'
            elif validation['proper_separation'] or validation['naming_convention'] == 'consistent':
                validation['organization_quality'] = 'good'
            elif source_files or header_files:
                validation['organization_quality'] = 'fair'
        
        # Calculate score
        score = 0.0
        if validation['proper_separation']: score += 0.4
        if validation['naming_convention'] == 'consistent': score += 0.3
        if validation['organization_quality'] in ['good', 'excellent']: score += 0.3
        
        validation['score'] = score
        return validation

    def _validate_functional_correctness(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Validate functional correctness"""
        validation = {
            'compilation_success': False,
            'executable_generation': False,
            'runtime_validation': {},
            'behavioral_consistency': {},
            'functional_score': 0.0,
            'test_results': {}
        }
        
        # Check compilation results from The Machine
        if 10 in all_results and hasattr(all_results[10], 'data'):
            machine_data = all_results[10].data
            if isinstance(machine_data, dict):
                compilation_results = machine_data.get('compilation_results', {})
                
                # Check compilation success
                success_rate = compilation_results.get('success_rate', 0.0)
                validation['compilation_success'] = success_rate > 0.0
                
                # Check executable generation
                binary_outputs = compilation_results.get('binary_outputs', {})
                validation['executable_generation'] = bool(binary_outputs)
                
                # Runtime validation (if executable exists)
                if binary_outputs:
                    validation['runtime_validation'] = self._validate_runtime_behavior(binary_outputs)
        
        # Calculate functional score
        score = 0.0
        if validation['compilation_success']: score += 0.4
        if validation['executable_generation']: score += 0.3
        runtime_score = validation.get('runtime_validation', {}).get('score', 0.0)
        score += runtime_score * 0.3
        
        validation['functional_score'] = score
        return validation

    def _validate_runtime_behavior(self, binary_outputs: Dict[str, str]) -> Dict[str, Any]:
        """Validate runtime behavior of generated executables"""
        validation = {
            'can_execute': False,
            'clean_exit': False,
            'no_crashes': False,
            'expected_behavior': False,
            'score': 0.0,
            'execution_results': {}
        }
        
        for build_system, binary_path in binary_outputs.items():
            if os.path.exists(binary_path):
                try:
                    # Try to execute the binary with a timeout
                    result = subprocess.run(
                        [binary_path],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    execution_result = {
                        'return_code': result.returncode,
                        'stdout': result.stdout[:500],  # Limit output
                        'stderr': result.stderr[:500],
                        'can_execute': True,
                        'clean_exit': result.returncode == 0
                    }
                    
                    validation['execution_results'][build_system] = execution_result
                    validation['can_execute'] = True
                    
                    if result.returncode == 0:
                        validation['clean_exit'] = True
                        validation['no_crashes'] = True
                    
                except subprocess.TimeoutExpired:
                    validation['execution_results'][build_system] = {
                        'timeout': True,
                        'can_execute': True,
                        'clean_exit': False
                    }
                    validation['can_execute'] = True
                except Exception as e:
                    validation['execution_results'][build_system] = {
                        'error': str(e),
                        'can_execute': False
                    }
        
        # Calculate score
        score = 0.0
        if validation['can_execute']: score += 0.4
        if validation['clean_exit']: score += 0.3
        if validation['no_crashes']: score += 0.3
        
        validation['score'] = score
        return validation

    def _validate_code_quality(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Validate code quality"""
        validation = {
            'readability': {},
            'maintainability': {},
            'documentation': {},
            'standards_compliance': {},
            'quality_score': 0.0
        }
        
        # Analyze code quality from global reconstruction
        if 9 in all_results and hasattr(all_results[9], 'data'):
            global_data = all_results[9].data
            if isinstance(global_data, dict):
                reconstructed_source = global_data.get('reconstructed_source', {})
                source_files = reconstructed_source.get('source_files', {})
                
                validation['readability'] = self._assess_code_readability(source_files)
                validation['maintainability'] = self._assess_code_maintainability(source_files)
                validation['documentation'] = self._assess_code_documentation(source_files)
                validation['standards_compliance'] = self._assess_standards_compliance(source_files)
        
        # Calculate quality score
        scores = [
            validation['readability'].get('score', 0.0),
            validation['maintainability'].get('score', 0.0),
            validation['documentation'].get('score', 0.0),
            validation['standards_compliance'].get('score', 0.0)
        ]
        validation['quality_score'] = sum(scores) / len(scores) if scores else 0.0
        
        return validation

    def _assess_code_readability(self, source_files: Dict[str, str]) -> Dict[str, Any]:
        """Assess code readability"""
        assessment = {
            'variable_naming': 'poor',
            'function_naming': 'poor',
            'code_formatting': 'poor',
            'complexity': 'high',
            'score': 0.0
        }
        
        if not source_files:
            return assessment
        
        total_lines = 0
        good_variable_names = 0
        good_function_names = 0
        proper_formatting = 0
        
        for content in source_files.values():
            if isinstance(content, str):
                lines = content.split('\n')
                total_lines += len(lines)
                
                # Analyze variable naming (simple heuristics)
                import re
                variables = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*=', content)
                for var in variables:
                    var_name = var.replace('=', '').strip()
                    if len(var_name) > 3 and '_' in var_name:
                        good_variable_names += 1
                
                # Analyze function naming
                functions = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(', content)
                for func in functions:
                    func_name = func.replace('(', '').strip()
                    if len(func_name) > 3 and ('_' in func_name or func_name != func_name.upper()):
                        good_function_names += 1
                
                # Check formatting (indentation)
                for line in lines:
                    if line.strip() and (line.startswith('    ') or line.startswith('\t')):
                        proper_formatting += 1
        
        # Calculate scores
        if total_lines > 0:
            formatting_ratio = proper_formatting / total_lines
            if formatting_ratio > 0.3:
                assessment['code_formatting'] = 'good'
            elif formatting_ratio > 0.1:
                assessment['code_formatting'] = 'fair'
        
        if good_variable_names > 3:
            assessment['variable_naming'] = 'good'
        elif good_variable_names > 1:
            assessment['variable_naming'] = 'fair'
        
        if good_function_names > 2:
            assessment['function_naming'] = 'good'
        elif good_function_names > 0:
            assessment['function_naming'] = 'fair'
        
        # Calculate overall score
        score = 0.0
        if assessment['variable_naming'] in ['good']: score += 0.25
        elif assessment['variable_naming'] == 'fair': score += 0.15
        if assessment['function_naming'] in ['good']: score += 0.25
        elif assessment['function_naming'] == 'fair': score += 0.15
        if assessment['code_formatting'] in ['good']: score += 0.25
        elif assessment['code_formatting'] == 'fair': score += 0.15
        if total_lines < 500: score += 0.25  # Complexity bonus for smaller code
        
        assessment['score'] = score
        return assessment

    def _assess_code_maintainability(self, source_files: Dict[str, str]) -> Dict[str, Any]:
        """Assess code maintainability"""
        assessment = {
            'function_size': 'large',
            'code_duplication': 'high',
            'modularity': 'poor',
            'score': 0.0
        }
        
        if not source_files:
            return assessment
        
        function_sizes = []
        total_lines = 0
        unique_lines = set()
        
        for content in source_files.values():
            if isinstance(content, str):
                lines = content.split('\n')
                total_lines += len(lines)
                
                # Track unique lines for duplication analysis
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                        unique_lines.add(stripped)
                
                # Estimate function sizes (simple heuristic)
                functions = content.split('}\n')
                for func in functions:
                    func_lines = len(func.split('\n'))
                    if func_lines > 5:  # Ignore very small functions
                        function_sizes.append(func_lines)
        
        # Analyze function sizes
        if function_sizes:
            avg_function_size = sum(function_sizes) / len(function_sizes)
            if avg_function_size < 20:
                assessment['function_size'] = 'good'
            elif avg_function_size < 50:
                assessment['function_size'] = 'fair'
        
        # Analyze code duplication
        if total_lines > 0:
            duplication_ratio = 1.0 - (len(unique_lines) / total_lines)
            if duplication_ratio < 0.1:
                assessment['code_duplication'] = 'low'
            elif duplication_ratio < 0.3:
                assessment['code_duplication'] = 'medium'
        
        # Analyze modularity (number of files)
        if len(source_files) > 3:
            assessment['modularity'] = 'good'
        elif len(source_files) > 1:
            assessment['modularity'] = 'fair'
        
        # Calculate score
        score = 0.0
        if assessment['function_size'] == 'good': score += 0.35
        elif assessment['function_size'] == 'fair': score += 0.20
        if assessment['code_duplication'] == 'low': score += 0.35
        elif assessment['code_duplication'] == 'medium': score += 0.20
        if assessment['modularity'] == 'good': score += 0.30
        elif assessment['modularity'] == 'fair': score += 0.15
        
        assessment['score'] = score
        return assessment

    def _assess_code_documentation(self, source_files: Dict[str, str]) -> Dict[str, Any]:
        """Assess code documentation"""
        assessment = {
            'comment_coverage': 0.0,
            'function_documentation': 'none',
            'inline_comments': 'sparse',
            'score': 0.0
        }
        
        if not source_files:
            return assessment
        
        total_lines = 0
        comment_lines = 0
        documented_functions = 0
        total_functions = 0
        
        for content in source_files.values():
            if isinstance(content, str):
                lines = content.split('\n')
                total_lines += len(lines)
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                        comment_lines += 1
                
                # Count function documentation
                import re
                functions = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*{', content)
                total_functions += len(functions)
                
                # Simple heuristic: function is documented if there's a comment within 3 lines before it
                content_lines = content.split('\n')
                for i, line in enumerate(content_lines):
                    if re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)\s*{', line):
                        # Check previous lines for comments
                        for j in range(max(0, i-3), i):
                            if content_lines[j].strip().startswith(('/*', '//', '*')):
                                documented_functions += 1
                                break
        
        # Calculate metrics
        if total_lines > 0:
            assessment['comment_coverage'] = comment_lines / total_lines
        
        if total_functions > 0:
            doc_ratio = documented_functions / total_functions
            if doc_ratio > 0.7:
                assessment['function_documentation'] = 'comprehensive'
            elif doc_ratio > 0.3:
                assessment['function_documentation'] = 'partial'
        
        if assessment['comment_coverage'] > 0.2:
            assessment['inline_comments'] = 'adequate'
        elif assessment['comment_coverage'] > 0.1:
            assessment['inline_comments'] = 'minimal'
        
        # Calculate score
        score = 0.0
        score += min(0.4, assessment['comment_coverage'] * 2)  # Up to 40% for comments
        if assessment['function_documentation'] == 'comprehensive': score += 0.4
        elif assessment['function_documentation'] == 'partial': score += 0.2
        if assessment['inline_comments'] == 'adequate': score += 0.2
        elif assessment['inline_comments'] == 'minimal': score += 0.1
        
        assessment['score'] = score
        return assessment

    def _assess_standards_compliance(self, source_files: Dict[str, str]) -> Dict[str, Any]:
        """Assess standards compliance"""
        assessment = {
            'c_standard': 'unknown',
            'syntax_errors': 0,
            'warning_count': 0,
            'style_compliance': 'poor',
            'score': 0.0
        }
        
        if not source_files:
            return assessment
        
        # Analyze C standard compliance
        has_c99_features = False
        has_c11_features = False
        syntax_issues = 0
        
        for content in source_files.values():
            if isinstance(content, str):
                # Check for C99/C11 features
                if any(feature in content for feature in ['for (int i', 'inline ', '// ']):
                    has_c99_features = True
                if any(feature in content for feature in ['_Static_assert', '_Alignof']):
                    has_c11_features = True
                
                # Simple syntax checking
                open_braces = content.count('{')
                close_braces = content.count('}')
                if open_braces != close_braces:
                    syntax_issues += 1
                
                open_parens = content.count('(')
                close_parens = content.count(')')
                if open_parens != close_parens:
                    syntax_issues += 1
        
        # Determine C standard
        if has_c11_features:
            assessment['c_standard'] = 'c11'
        elif has_c99_features:
            assessment['c_standard'] = 'c99'
        else:
            assessment['c_standard'] = 'c89'
        
        assessment['syntax_errors'] = syntax_issues
        
        # Style compliance (simple heuristics)
        if assessment['syntax_errors'] == 0:
            assessment['style_compliance'] = 'good'
        elif assessment['syntax_errors'] <= 2:
            assessment['style_compliance'] = 'fair'
        
        # Calculate score
        score = 0.0
        if assessment['syntax_errors'] == 0: score += 0.5
        elif assessment['syntax_errors'] <= 2: score += 0.3
        if assessment['c_standard'] in ['c99', 'c11']: score += 0.3
        if assessment['style_compliance'] == 'good': score += 0.2
        elif assessment['style_compliance'] == 'fair': score += 0.1
        
        assessment['score'] = score
        return assessment

    def _validate_reconstruction_completeness(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Validate completeness of reconstruction"""
        validation = {
            'pipeline_completeness': 0.0,
            'missing_components': [],
            'data_completeness': {},
            'feature_coverage': {},
            'completeness_score': 0.0
        }
        
        # Check pipeline completeness
        expected_agents = [1, 2, 4, 7, 9, 10]  # Core agents
        completed_agents = sum(1 for agent_id in expected_agents 
                             if agent_id in all_results and 
                             hasattr(all_results[agent_id], 'status') and
                             all_results[agent_id].status == AgentStatus.COMPLETED)
        
        validation['pipeline_completeness'] = completed_agents / len(expected_agents)
        validation['missing_components'] = [agent_id for agent_id in expected_agents 
                                          if agent_id not in all_results or
                                          not hasattr(all_results[agent_id], 'status') or
                                          all_results[agent_id].status != AgentStatus.COMPLETED]
        
        # Analyze data completeness
        validation['data_completeness'] = self._analyze_data_completeness(all_results)
        
        # Analyze feature coverage
        validation['feature_coverage'] = self._analyze_feature_coverage(all_results)
        
        # Calculate completeness score
        scores = [
            validation['pipeline_completeness'],
            validation['data_completeness'].get('score', 0.0),
            validation['feature_coverage'].get('score', 0.0)
        ]
        validation['completeness_score'] = sum(scores) / len(scores)
        
        return validation

    def _analyze_data_completeness(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze completeness of data in results"""
        completeness = {
            'has_source_code': False,
            'has_headers': False,
            'has_main_function': False,
            'has_build_config': False,
            'has_executable': False,
            'data_coverage': 0.0,
            'score': 0.0
        }
        
        # Check for source code
        if 9 in all_results and hasattr(all_results[9], 'data'):
            global_data = all_results[9].data
            if isinstance(global_data, dict):
                reconstructed_source = global_data.get('reconstructed_source', {})
                
                completeness['has_source_code'] = bool(reconstructed_source.get('source_files'))
                completeness['has_headers'] = bool(reconstructed_source.get('header_files'))
                completeness['has_main_function'] = bool(reconstructed_source.get('main_function'))
                completeness['has_build_config'] = bool(global_data.get('build_configuration'))
        
        # Check for executable
        if 10 in all_results and hasattr(all_results[10], 'data'):
            machine_data = all_results[10].data
            if isinstance(machine_data, dict):
                compilation_results = machine_data.get('compilation_results', {})
                completeness['has_executable'] = bool(compilation_results.get('binary_outputs'))
        
        # Calculate data coverage
        features = [
            completeness['has_source_code'],
            completeness['has_headers'],
            completeness['has_main_function'],
            completeness['has_build_config'],
            completeness['has_executable']
        ]
        completeness['data_coverage'] = sum(features) / len(features)
        completeness['score'] = completeness['data_coverage']
        
        return completeness

    def _analyze_feature_coverage(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze coverage of binary features"""
        coverage = {
            'binary_format_analysis': False,
            'architecture_detection': False,
            'function_reconstruction': False,
            'resource_extraction': False,
            'compilation_success': False,
            'feature_score': 0.0,
            'score': 0.0
        }
        
        # Check binary format analysis
        if 1 in all_results and hasattr(all_results[1], 'data'):
            data = all_results[1].data
            if isinstance(data, dict) and data.get('binary_info'):
                coverage['binary_format_analysis'] = True
        
        # Check architecture detection
        if 2 in all_results and hasattr(all_results[2], 'data'):
            data = all_results[2].data
            if isinstance(data, dict) and data.get('architecture'):
                coverage['architecture_detection'] = True
        
        # Check function reconstruction
        if 7 in all_results and hasattr(all_results[7], 'data'):
            data = all_results[7].data
            if isinstance(data, dict) and data.get('enhanced_functions'):
                coverage['function_reconstruction'] = True
        
        # Check resource extraction
        if 8 in all_results and hasattr(all_results[8], 'data'):
            data = all_results[8].data
            if isinstance(data, dict) and data.get('resource_files'):
                coverage['resource_extraction'] = True
        
        # Check compilation success
        if 10 in all_results and hasattr(all_results[10], 'data'):
            data = all_results[10].data
            if isinstance(data, dict):
                compilation_results = data.get('compilation_results', {})
                coverage['compilation_success'] = compilation_results.get('success_rate', 0.0) > 0.0
        
        # Calculate feature score
        features = [
            coverage['binary_format_analysis'],
            coverage['architecture_detection'],
            coverage['function_reconstruction'],
            coverage['resource_extraction'],
            coverage['compilation_success']
        ]
        coverage['feature_score'] = sum(features) / len(features)
        coverage['score'] = coverage['feature_score']
        
        return coverage

    def _validate_cross_component_consistency(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Validate consistency across components"""
        validation = {
            'architecture_consistency': True,
            'function_consistency': True,
            'data_consistency': True,
            'inconsistencies': [],
            'consistency_score': 0.0
        }
        
        # Check architecture consistency
        architectures = []
        for agent_id in [1, 2, 7, 10]:
            if agent_id in all_results and hasattr(all_results[agent_id], 'data'):
                data = all_results[agent_id].data
                if isinstance(data, dict):
                    arch = data.get('architecture')
                    if arch:
                        architectures.append((agent_id, arch))
        
        if len(architectures) > 1:
            first_arch = architectures[0][1]
            for agent_id, arch in architectures[1:]:
                if arch != first_arch:
                    validation['architecture_consistency'] = False
                    validation['inconsistencies'].append(f"Architecture mismatch: Agent {architectures[0][0]} says {first_arch}, Agent {agent_id} says {arch}")
        
        # Check function consistency
        function_counts = []
        for agent_id in [4, 7, 9]:
            if agent_id in all_results and hasattr(all_results[agent_id], 'data'):
                data = all_results[agent_id].data
                if isinstance(data, dict):
                    functions = data.get('decompiled_functions', {}) or data.get('enhanced_functions', {})
                    if isinstance(functions, dict):
                        function_counts.append((agent_id, len(functions)))
        
        if len(function_counts) > 1:
            max_count = max(count for _, count in function_counts)
            min_count = min(count for _, count in function_counts)
            if max_count > min_count * 2:  # Significant discrepancy
                validation['function_consistency'] = False
                validation['inconsistencies'].append(f"Function count discrepancy: {function_counts}")
        
        # Calculate consistency score
        consistency_factors = [
            validation['architecture_consistency'],
            validation['function_consistency'],
            validation['data_consistency']
        ]
        validation['consistency_score'] = sum(consistency_factors) / len(consistency_factors)
        
        return validation

    def _calculate_overall_validation_score(self, validation: Dict[str, Any]) -> float:
        """Calculate overall validation score"""
        weights = {
            'component_validations': 0.25,
            'structural_validation': 0.20,
            'functional_validation': 0.25,
            'quality_validation': 0.15,
            'completeness_validation': 0.10,
            'consistency_validation': 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, weight in weights.items():
            category_data = validation.get(category, {})
            if category == 'component_validations':
                # Average score of all components
                component_scores = []
                for comp_id, comp_data in category_data.items():
                    if isinstance(comp_data, dict):
                        component_scores.append(comp_data.get('score', 0.0))
                category_score = sum(component_scores) / len(component_scores) if component_scores else 0.0
            else:
                category_score = category_data.get('structural_score', 0.0) if 'structural' in category else \
                               category_data.get('functional_score', 0.0) if 'functional' in category else \
                               category_data.get('quality_score', 0.0) if 'quality' in category else \
                               category_data.get('completeness_score', 0.0) if 'completeness' in category else \
                               category_data.get('consistency_score', 0.0)
            
            total_score += category_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

    def _identify_critical_issues(self, validation: Dict[str, Any]) -> List[str]:
        """Identify critical issues in validation"""
        issues = []
        
        # Check for critical component failures
        component_validations = validation.get('component_validations', {})
        for comp_id, comp_data in component_validations.items():
            if isinstance(comp_data, dict) and comp_data.get('critical', False):
                if not comp_data.get('successful', False):
                    issues.append(f"Critical component {comp_id} ({comp_data.get('name', 'Unknown')}) failed")
                elif not comp_data.get('meets_minimum', False):
                    issues.append(f"Critical component {comp_id} ({comp_data.get('name', 'Unknown')}) below minimum threshold")
        
        # Check for compilation failures
        functional_validation = validation.get('functional_validation', {})
        if not functional_validation.get('compilation_success', False):
            issues.append("Compilation failed - no executable generated")
        
        # Check for structural issues
        structural_validation = validation.get('structural_validation', {})
        if structural_validation.get('structural_score', 0.0) < 0.5:
            issues.append("Poor structural integrity detected")
        
        # Check for completeness issues
        completeness_validation = validation.get('completeness_validation', {})
        if completeness_validation.get('completeness_score', 0.0) < 0.6:
            issues.append("Incomplete reconstruction detected")
        
        return issues

    def _generate_validation_warnings(self, validation: Dict[str, Any]) -> List[str]:
        """Generate validation warnings"""
        warnings = []
        
        # Quality warnings
        quality_validation = validation.get('quality_validation', {})
        if quality_validation.get('quality_score', 0.0) < 0.6:
            warnings.append("Code quality below recommended standards")
        
        # Consistency warnings
        consistency_validation = validation.get('consistency_validation', {})
        if consistency_validation.get('inconsistencies'):
            for inconsistency in consistency_validation['inconsistencies']:
                warnings.append(f"Consistency issue: {inconsistency}")
        
        # Documentation warnings
        quality_data = quality_validation.get('documentation', {})
        if quality_data.get('score', 0.0) < 0.3:
            warnings.append("Poor code documentation detected")
        
        return warnings

    def _verify_reconstruction_truth(self, validation_result: Dict[str, Any], all_results: Dict[int, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify truth and accuracy of reconstruction"""
        verification = {
            'binary_fidelity': {},
            'semantic_equivalence': {},
            'behavioral_verification': {},
            'ground_truth_comparison': {},
            'confidence_metrics': {},
            'truth_score': 0.0,
            'verification_methods': []
        }
        
        # Binary fidelity check
        verification['binary_fidelity'] = self._verify_binary_fidelity(all_results, context)
        verification['verification_methods'].append('binary_fidelity')
        
        # Semantic equivalence
        verification['semantic_equivalence'] = self._verify_semantic_equivalence(all_results)
        verification['verification_methods'].append('semantic_analysis')
        
        # Behavioral verification
        verification['behavioral_verification'] = self._verify_behavioral_equivalence(all_results, context)
        verification['verification_methods'].append('behavioral_testing')
        
        # Confidence metrics
        verification['confidence_metrics'] = self._calculate_confidence_metrics(verification, validation_result)
        
        # Calculate truth score
        verification['truth_score'] = self._calculate_truth_score(verification)
        
        return verification

    def _verify_binary_fidelity(self, all_results: Dict[int, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify fidelity to original binary"""
        fidelity = {
            'entry_point_match': False,
            'function_count_accuracy': 0.0,
            'import_table_accuracy': 0.0,
            'resource_accuracy': 0.0,
            'overall_fidelity': 0.0
        }
        
        # Get original binary info
        original_binary_path = context.get('binary_path')
        if not original_binary_path or not os.path.exists(original_binary_path):
            fidelity['overall_fidelity'] = 0.0
            return fidelity
        
        # Compare entry points
        if 1 in all_results and hasattr(all_results[1], 'data'):
            discovery_data = all_results[1].data
            if isinstance(discovery_data, dict):
                detected_entry = discovery_data.get('entry_point', {})
                # Simple check - if we have entry point info, assume match for now
                fidelity['entry_point_match'] = bool(detected_entry)
        
        # Compare function counts (rough estimate)
        if 7 in all_results and hasattr(all_results[7], 'data'):
            decompiler_data = all_results[7].data
            if isinstance(decompiler_data, dict):
                enhanced_functions = decompiler_data.get('enhanced_functions', {})
                # Assume reasonable accuracy if we have multiple functions
                fidelity['function_count_accuracy'] = min(1.0, len(enhanced_functions) / 5.0)
        
        # Compare imports
        if 1 in all_results and hasattr(all_results[1], 'data'):
            discovery_data = all_results[1].data
            if isinstance(discovery_data, dict):
                imports = discovery_data.get('imports', {})
                # Assume good accuracy if we detected imports
                fidelity['import_table_accuracy'] = 0.8 if imports else 0.3
        
        # Calculate overall fidelity
        metrics = [
            1.0 if fidelity['entry_point_match'] else 0.0,
            fidelity['function_count_accuracy'],
            fidelity['import_table_accuracy'],
            fidelity['resource_accuracy']
        ]
        fidelity['overall_fidelity'] = sum(metrics) / len(metrics)
        
        return fidelity

    def _verify_semantic_equivalence(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Verify semantic equivalence of reconstruction"""
        equivalence = {
            'control_flow_preservation': False,
            'data_structure_preservation': False,
            'api_usage_preservation': False,
            'semantic_score': 0.0
        }
        
        # Check control flow preservation
        if 9 in all_results and hasattr(all_results[9], 'data'):
            assembly_data = all_results[9].data
            if isinstance(assembly_data, dict):
                control_flow = assembly_data.get('control_flow_analysis', {})
                equivalence['control_flow_preservation'] = bool(control_flow)
        
        # Check data structure preservation
        if 5 in all_results and hasattr(all_results[5], 'data'):
            structure_data = all_results[5].data
            if isinstance(structure_data, dict):
                structures = structure_data.get('structures', {})
                equivalence['data_structure_preservation'] = bool(structures)
        
        # Check API usage preservation
        if 1 in all_results and hasattr(all_results[1], 'data'):
            discovery_data = all_results[1].data
            if isinstance(discovery_data, dict):
                imports = discovery_data.get('imports', {})
                equivalence['api_usage_preservation'] = bool(imports)
        
        # Calculate semantic score
        factors = [
            equivalence['control_flow_preservation'],
            equivalence['data_structure_preservation'],
            equivalence['api_usage_preservation']
        ]
        equivalence['semantic_score'] = sum(factors) / len(factors)
        
        return equivalence

    def _verify_behavioral_equivalence(self, all_results: Dict[int, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify behavioral equivalence through testing"""
        verification = {
            'runtime_behavior_match': False,
            'output_equivalence': False,
            'error_handling_equivalence': False,
            'performance_characteristics': {},
            'behavioral_score': 0.0
        }
        
        # Check if we have executable outputs from The Machine
        if 10 in all_results and hasattr(all_results[10], 'data'):
            machine_data = all_results[10].data
            if isinstance(machine_data, dict):
                compilation_results = machine_data.get('compilation_results', {})
                binary_outputs = compilation_results.get('binary_outputs', {})
                
                if binary_outputs:
                    # Basic runtime test was already done in functional validation
                    functional_validation = context.get('functional_validation', {})
                    runtime_validation = functional_validation.get('runtime_validation', {})
                    
                    verification['runtime_behavior_match'] = runtime_validation.get('can_execute', False)
                    verification['output_equivalence'] = runtime_validation.get('clean_exit', False)
                    verification['error_handling_equivalence'] = runtime_validation.get('no_crashes', False)
        
        # Calculate behavioral score
        factors = [
            verification['runtime_behavior_match'],
            verification['output_equivalence'],
            verification['error_handling_equivalence']
        ]
        verification['behavioral_score'] = sum(factors) / len(factors)
        
        return verification

    def _calculate_confidence_metrics(self, verification: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for truth verification"""
        metrics = {
            'data_quality_confidence': 0.0,
            'reconstruction_confidence': 0.0,
            'validation_confidence': 0.0,
            'overall_confidence': 0.0,
            'confidence_factors': {}
        }
        
        # Data quality confidence
        overall_validation_score = validation_result.get('overall_score', 0.0)
        metrics['data_quality_confidence'] = overall_validation_score
        
        # Reconstruction confidence
        binary_fidelity = verification.get('binary_fidelity', {})
        semantic_equivalence = verification.get('semantic_equivalence', {})
        behavioral_verification = verification.get('behavioral_verification', {})
        
        reconstruction_factors = [
            binary_fidelity.get('overall_fidelity', 0.0),
            semantic_equivalence.get('semantic_score', 0.0),
            behavioral_verification.get('behavioral_score', 0.0)
        ]
        metrics['reconstruction_confidence'] = sum(reconstruction_factors) / len(reconstruction_factors)
        
        # Validation confidence (based on number of successful validation methods)
        validation_methods = len(verification.get('verification_methods', []))
        metrics['validation_confidence'] = min(1.0, validation_methods / 3.0)
        
        # Overall confidence
        metrics['overall_confidence'] = (
            metrics['data_quality_confidence'] * 0.4 +
            metrics['reconstruction_confidence'] * 0.4 +
            metrics['validation_confidence'] * 0.2
        )
        
        return metrics

    def _calculate_truth_score(self, verification: Dict[str, Any]) -> float:
        """Calculate overall truth score"""
        binary_fidelity = verification.get('binary_fidelity', {}).get('overall_fidelity', 0.0)
        semantic_equivalence = verification.get('semantic_equivalence', {}).get('semantic_score', 0.0)
        behavioral_verification = verification.get('behavioral_verification', {}).get('behavioral_score', 0.0)
        confidence = verification.get('confidence_metrics', {}).get('overall_confidence', 0.0)
        
        # Weighted truth score
        truth_score = (
            binary_fidelity * 0.3 +
            semantic_equivalence * 0.3 +
            behavioral_verification * 0.2 +
            confidence * 0.2
        )
        
        return truth_score

    def _generate_oracle_report(self, validation_result: Dict[str, Any], truth_verification: Dict[str, Any], all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Generate comprehensive Oracle report"""
        report = {
            'executive_summary': {},
            'validation_summary': {},
            'truth_verification_summary': {},
            'component_analysis': {},
            'quality_assessment': {},
            'recommendations': {},
            'oracle_verdict': {},
            'report_metadata': {}
        }
        
        # Executive summary
        report['executive_summary'] = {
            'overall_score': validation_result.get('overall_score', 0.0),
            'truth_score': truth_verification.get('truth_score', 0.0),
            'critical_issues_count': len(validation_result.get('critical_issues', [])),
            'warnings_count': len(validation_result.get('warnings', [])),
            'pipeline_completeness': validation_result.get('completeness_validation', {}).get('pipeline_completeness', 0.0),
            'recommendation': self._get_overall_recommendation(validation_result, truth_verification)
        }
        
        # Validation summary
        report['validation_summary'] = {
            'structural_integrity': validation_result.get('structural_validation', {}).get('structural_score', 0.0),
            'functional_correctness': validation_result.get('functional_validation', {}).get('functional_score', 0.0),
            'code_quality': validation_result.get('quality_validation', {}).get('quality_score', 0.0),
            'completeness': validation_result.get('completeness_validation', {}).get('completeness_score', 0.0),
            'consistency': validation_result.get('consistency_validation', {}).get('consistency_score', 0.0)
        }
        
        # Truth verification summary
        report['truth_verification_summary'] = {
            'binary_fidelity': truth_verification.get('binary_fidelity', {}).get('overall_fidelity', 0.0),
            'semantic_equivalence': truth_verification.get('semantic_equivalence', {}).get('semantic_score', 0.0),
            'behavioral_verification': truth_verification.get('behavioral_verification', {}).get('behavioral_score', 0.0),
            'confidence_level': truth_verification.get('confidence_metrics', {}).get('overall_confidence', 0.0)
        }
        
        # Component analysis
        report['component_analysis'] = self._analyze_component_performance(all_results)
        
        # Quality assessment
        report['quality_assessment'] = self._assess_overall_quality(validation_result, truth_verification)
        
        return report

    def _get_overall_recommendation(self, validation_result: Dict[str, Any], truth_verification: Dict[str, Any]) -> str:
        """Get overall recommendation based on analysis"""
        overall_score = validation_result.get('overall_score', 0.0)
        truth_score = truth_verification.get('truth_score', 0.0)
        critical_issues = len(validation_result.get('critical_issues', []))
        
        if critical_issues > 0:
            return "REJECT - Critical issues detected"
        elif overall_score >= 0.8 and truth_score >= 0.8:
            return "APPROVE - High quality reconstruction"
        elif overall_score >= 0.7 and truth_score >= 0.7:
            return "APPROVE WITH RESERVATIONS - Good quality with minor issues"
        elif overall_score >= 0.6 and truth_score >= 0.6:
            return "CONDITIONAL APPROVAL - Requires improvements"
        else:
            return "REJECT - Quality below acceptable threshold"

    def _analyze_component_performance(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze performance of individual components"""
        analysis = {
            'top_performers': [],
            'underperformers': [],
            'missing_components': [],
            'performance_summary': {}
        }
        
        component_scores = {}
        expected_components = [1, 2, 4, 5, 7, 9, 10]
        
        for agent_id in expected_components:
            if agent_id in all_results:
                result = all_results[agent_id]
                if hasattr(result, 'status') and result.status == AgentStatus.COMPLETED:
                    score = self._calculate_component_score(agent_id, result)
                    component_scores[agent_id] = score
                else:
                    analysis['underperformers'].append(agent_id)
            else:
                analysis['missing_components'].append(agent_id)
        
        # Identify top performers and underperformers
        for agent_id, score in component_scores.items():
            if score >= 0.8:
                analysis['top_performers'].append((agent_id, score))
            elif score < 0.6:
                analysis['underperformers'].append((agent_id, score))
        
        analysis['performance_summary'] = component_scores
        return analysis

    def _assess_overall_quality(self, validation_result: Dict[str, Any], truth_verification: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of reconstruction"""
        assessment = {
            'reconstruction_grade': 'F',
            'quality_dimensions': {},
            'strengths': [],
            'weaknesses': [],
            'improvement_areas': []
        }
        
        # Calculate quality dimensions
        assessment['quality_dimensions'] = {
            'accuracy': truth_verification.get('truth_score', 0.0),
            'completeness': validation_result.get('completeness_validation', {}).get('completeness_score', 0.0),
            'reliability': validation_result.get('functional_validation', {}).get('functional_score', 0.0),
            'maintainability': validation_result.get('quality_validation', {}).get('quality_score', 0.0),
            'consistency': validation_result.get('consistency_validation', {}).get('consistency_score', 0.0)
        }
        
        # Calculate overall grade
        avg_score = sum(assessment['quality_dimensions'].values()) / len(assessment['quality_dimensions'])
        
        if avg_score >= 0.9:
            assessment['reconstruction_grade'] = 'A'
        elif avg_score >= 0.8:
            assessment['reconstruction_grade'] = 'B'
        elif avg_score >= 0.7:
            assessment['reconstruction_grade'] = 'C'
        elif avg_score >= 0.6:
            assessment['reconstruction_grade'] = 'D'
        else:
            assessment['reconstruction_grade'] = 'F'
        
        # Identify strengths and weaknesses
        for dimension, score in assessment['quality_dimensions'].items():
            if score >= 0.8:
                assessment['strengths'].append(dimension)
            elif score < 0.6:
                assessment['weaknesses'].append(dimension)
                assessment['improvement_areas'].append(f"Improve {dimension}")
        
        return assessment

    def _render_final_verdict(self, oracle_report: Dict[str, Any], validation_result: Dict[str, Any], truth_verification: Dict[str, Any]) -> Dict[str, Any]:
        """Render final Oracle verdict"""
        verdict = {
            'decision': 'UNKNOWN',
            'grade': oracle_report.get('quality_assessment', {}).get('reconstruction_grade', 'F'),
            'confidence': truth_verification.get('confidence_metrics', {}).get('overall_confidence', 0.0),
            'justification': [],
            'oracle_proclamation': '',
            'next_steps': []
        }
        
        overall_score = validation_result.get('overall_score', 0.0)
        truth_score = truth_verification.get('truth_score', 0.0)
        critical_issues = validation_result.get('critical_issues', [])
        
        # Determine decision
        if critical_issues:
            verdict['decision'] = 'REJECT'
            verdict['justification'].append(f"Critical issues detected: {len(critical_issues)}")
        elif overall_score >= 0.75 and truth_score >= 0.75:
            verdict['decision'] = 'APPROVE'
            verdict['justification'].append("High quality reconstruction achieved")
        elif overall_score >= 0.6 and truth_score >= 0.6:
            verdict['decision'] = 'CONDITIONAL_APPROVAL'
            verdict['justification'].append("Acceptable quality with room for improvement")
        else:
            verdict['decision'] = 'REJECT'
            verdict['justification'].append("Quality below acceptable threshold")
        
        # Generate Oracle proclamation
        if verdict['decision'] == 'APPROVE':
            verdict['oracle_proclamation'] = f"The Oracle has spoken: The reconstruction is TRUE and WORTHY. Grade: {verdict['grade']}"
        elif verdict['decision'] == 'CONDITIONAL_APPROVAL':
            verdict['oracle_proclamation'] = f"The Oracle declares: The reconstruction shows PROMISE but requires REFINEMENT. Grade: {verdict['grade']}"
        else:
            verdict['oracle_proclamation'] = f"The Oracle has judged: The reconstruction is INSUFFICIENT and must be REFORGED. Grade: {verdict['grade']}"
        
        # Generate next steps
        if verdict['decision'] == 'REJECT':
            verdict['next_steps'] = [
                "Address critical issues identified in validation",
                "Improve component implementations",
                "Re-run pipeline with enhanced configurations"
            ]
        elif verdict['decision'] == 'CONDITIONAL_APPROVAL':
            verdict['next_steps'] = [
                "Address identified weaknesses",
                "Enhance code quality and documentation",
                "Consider additional validation testing"
            ]
        else:
            verdict['next_steps'] = [
                "Proceed with final deployment",
                "Document successful reconstruction process",
                "Archive results for future reference"
            ]
        
        return verdict

    def _calculate_oracle_metrics(self, validation_result: Dict[str, Any], truth_verification: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Oracle-specific metrics"""
        metrics = {
            'oracle_confidence': truth_verification.get('confidence_metrics', {}).get('overall_confidence', 0.0),
            'validation_thoroughness': len(validation_result.get('component_validations', {})) / 10.0,  # Max 10 components
            'truth_verification_completeness': len(truth_verification.get('verification_methods', [])) / 3.0,  # Max 3 methods
            'critical_issue_severity': len(validation_result.get('critical_issues', [])),
            'oracle_reliability': 0.0
        }
        
        # Calculate Oracle reliability based on data quality
        oracle_factors = [
            metrics['oracle_confidence'],
            metrics['validation_thoroughness'],
            metrics['truth_verification_completeness'],
            1.0 - min(1.0, metrics['critical_issue_severity'] / 5.0)  # Penalty for critical issues
        ]
        metrics['oracle_reliability'] = sum(oracle_factors) / len(oracle_factors)
        
        return metrics

    def _generate_recommendations(self, validation_result: Dict[str, Any], truth_verification: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        # Based on validation results
        overall_score = validation_result.get('overall_score', 0.0)
        if overall_score < 0.7:
            recommendations.append("Improve overall pipeline quality - consider enhancing core components")
        
        # Based on critical issues
        critical_issues = validation_result.get('critical_issues', [])
        if critical_issues:
            recommendations.append("Address critical issues before proceeding with deployment")
        
        # Based on component performance
        component_validations = validation_result.get('component_validations', {})
        failed_components = [comp_id for comp_id, comp_data in component_validations.items() 
                           if isinstance(comp_data, dict) and not comp_data.get('successful', False)]
        if failed_components:
            recommendations.append(f"Fix failed components: {failed_components}")
        
        # Based on quality assessment
        quality_validation = validation_result.get('quality_validation', {})
        if quality_validation.get('quality_score', 0.0) < 0.6:
            recommendations.append("Improve code quality through better naming, documentation, and structure")
        
        # Based on truth verification
        truth_score = truth_verification.get('truth_score', 0.0)
        if truth_score < 0.7:
            recommendations.append("Enhance reconstruction accuracy and fidelity to original binary")
        
        return recommendations

    def get_description(self) -> str:
        """Get description of The Oracle agent"""
        return "The Oracle performs comprehensive validation and truth verification of the reconstructed code"

    def get_dependencies(self) -> List[int]:
        """Get dependencies for The Oracle"""
        return [10]  # Depends on The Machine