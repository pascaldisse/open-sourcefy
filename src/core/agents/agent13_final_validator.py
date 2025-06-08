"""
Agent 13: Source Code Validation Agent
Comprehensive validation of all generated source code to determine if it represents
a real implementation of the target binary's functionality.
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent13_FinalValidator(BaseAgent):
    """Agent 13: Comprehensive source code validation and pipeline termination"""
    
    def __init__(self):
        super().__init__(
            agent_id=13,
            name="SourceCodeValidator",
            dependencies=[11, 12]  # Depends on source reconstruction and compilation
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute comprehensive source code validation"""
        try:
            # Get outputs from dependent agents
            agent11_result = context['agent_results'].get(11)
            agent12_result = context['agent_results'].get(12)
            
            if not agent11_result or agent11_result.status != AgentStatus.COMPLETED:
                return self._fail_pipeline(
                    "Agent 11 (GlobalReconstructor) did not complete successfully",
                    context
                )
            
            if not agent12_result or agent12_result.status != AgentStatus.COMPLETED:
                return self._fail_pipeline(
                    "Agent 12 (CompilationOrchestrator) did not complete successfully", 
                    context
                )
            
            # Perform comprehensive source code analysis
            validation_report = self._perform_comprehensive_validation(
                agent11_result.data,
                agent12_result.data,
                context
            )
            
            # Generate detailed validation report
            self._save_validation_report(validation_report, context)
            
            # Make final determination
            if not validation_report['is_real_implementation']:
                return self._fail_pipeline(
                    validation_report['failure_reason'],
                    context,
                    validation_report
                )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=validation_report,
                metadata={
                    'analysis_type': 'comprehensive_source_validation',
                    'validation_mode': 'strict'
                }
            )
            
        except Exception as e:
            return self._fail_pipeline(
                f"Validation agent failed with exception: {str(e)}",
                context
            )

    def _perform_comprehensive_validation(self, 
                                        reconstruction_data: Dict[str, Any],
                                        compilation_data: Dict[str, Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive validation of all source code"""
        
        validation_report = {
            'validation_timestamp': context.get('timestamp', ''),
            'target_binary': context.get('global_data', {}).get('binary_path', ''),
            'is_real_implementation': False,
            'failure_reason': '',
            'confidence_score': 0.0,
            'source_analysis': {},
            'method_analysis': {},
            'class_analysis': {},
            'implementation_analysis': {},
            'quality_metrics': {},
            'comparison_analysis': {},
            'detailed_findings': [],
            'recommendation': ''
        }
        
        # Step 1: Analyze all discovered source files
        source_analysis = self._analyze_all_source_files(reconstruction_data, compilation_data)
        validation_report['source_analysis'] = source_analysis
        
        # Step 2: Analyze methods and functions
        method_analysis = self._analyze_all_methods(source_analysis['all_source_content'])
        validation_report['method_analysis'] = method_analysis
        
        # Step 3: Analyze classes and structures  
        class_analysis = self._analyze_all_classes(source_analysis['all_source_content'])
        validation_report['class_analysis'] = class_analysis
        
        # Step 4: Analyze implementation completeness
        implementation_analysis = self._analyze_implementation_completeness(
            source_analysis, method_analysis, class_analysis, context
        )
        validation_report['implementation_analysis'] = implementation_analysis
        
        # Step 5: Calculate quality metrics
        quality_metrics = self._calculate_comprehensive_quality_metrics(
            source_analysis, method_analysis, class_analysis, implementation_analysis
        )
        validation_report['quality_metrics'] = quality_metrics
        
        # Step 6: Compare with original binary capabilities
        comparison_analysis = self._compare_with_original_binary(
            validation_report, context
        )
        validation_report['comparison_analysis'] = comparison_analysis
        
        # Step 7: Make final determination
        final_determination = self._make_final_determination(validation_report)
        validation_report.update(final_determination)
        
        return validation_report

    def _analyze_all_source_files(self, reconstruction_data: Dict[str, Any], 
                                compilation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all discovered source files"""
        
        analysis = {
            'total_files': 0,
            'source_files': {},
            'header_files': {},
            'all_source_content': {},
            'file_types': {},
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'empty_lines': 0,
            'file_summary': []
        }
        
        # Collect source files from reconstruction
        reconstructed_source = reconstruction_data.get('reconstructed_source', {})
        source_files = reconstructed_source.get('source_files', {})
        header_files = reconstructed_source.get('header_files', {})
        
        # Collect source files from compilation output
        generated_source = compilation_data.get('generated_source', {})
        
        # Combine all source content
        all_sources = {}
        all_sources.update(source_files)
        all_sources.update(header_files)
        all_sources.update(generated_source)
        
        for filename, content in all_sources.items():
            if not isinstance(content, str) or not content.strip():
                continue
                
            analysis['total_files'] += 1
            analysis['all_source_content'][filename] = content
            
            # Categorize file type
            if filename.endswith(('.c', '.cpp', '.cc', '.cxx')):
                analysis['source_files'][filename] = content
                analysis['file_types'][filename] = 'source'
            elif filename.endswith(('.h', '.hpp', '.hxx')):
                analysis['header_files'][filename] = content
                analysis['file_types'][filename] = 'header'
            else:
                analysis['file_types'][filename] = 'other'
            
            # Analyze file content
            file_analysis = self._analyze_single_file(filename, content)
            analysis['total_lines'] += file_analysis['total_lines']
            analysis['code_lines'] += file_analysis['code_lines']
            analysis['comment_lines'] += file_analysis['comment_lines']
            analysis['empty_lines'] += file_analysis['empty_lines']
            
            analysis['file_summary'].append({
                'filename': filename,
                'type': analysis['file_types'][filename],
                'size': len(content),
                'lines': file_analysis['total_lines'],
                'code_lines': file_analysis['code_lines'],
                'functions': file_analysis['function_count'],
                'complexity': file_analysis['complexity_score'],
                'includes': file_analysis['includes']
            })
        
        return analysis

    def _analyze_single_file(self, filename: str, content: str) -> Dict[str, Any]:
        """Analyze a single source file in detail"""
        
        analysis = {
            'filename': filename,
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'empty_lines': 0,
            'function_count': 0,
            'complexity_score': 0,
            'includes': [],
            'functions': [],
            'variables': [],
            'structures': [],
            'syntax_errors': [],
            'quality_issues': []
        }
        
        lines = content.split('\n')
        analysis['total_lines'] = len(lines)
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if not stripped:
                analysis['empty_lines'] += 1
            elif stripped.startswith('//') or stripped.startswith('/*') or '*/' in line:
                analysis['comment_lines'] += 1
            elif stripped.startswith('#'):
                # Preprocessor directive
                if 'include' in stripped:
                    include_match = re.search(r'#include\s*[<"]([^>"]+)[>"]', stripped)
                    if include_match:
                        analysis['includes'].append(include_match.group(1))
            else:
                analysis['code_lines'] += 1
        
        # Find function definitions
        function_pattern = r'(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*\{'
        functions = list(re.finditer(function_pattern, content, re.MULTILINE))
        analysis['function_count'] = len(functions)
        
        for func_match in functions:
            return_type = func_match.group(1).strip()
            func_name = func_match.group(2)
            parameters = func_match.group(3).strip()
            
            # Extract function body
            func_body = self._extract_function_body(content, func_match.start())
            func_analysis = self._analyze_function_implementation(func_name, func_body)
            
            analysis['functions'].append({
                'name': func_name,
                'return_type': return_type,
                'parameters': parameters,
                'body_lines': func_analysis['body_lines'],
                'complexity': func_analysis['complexity'],
                'has_real_logic': func_analysis['has_real_logic'],
                'is_placeholder': func_analysis['is_placeholder']
            })
            
            analysis['complexity_score'] += func_analysis['complexity']
        
        # Find variable declarations
        var_pattern = r'\b(int|char|float|double|long|short|unsigned|signed|void\*|struct\s+\w+)\s+(\w+)'
        variables = re.findall(var_pattern, content)
        analysis['variables'] = [{'type': var[0], 'name': var[1]} for var in variables]
        
        # Find structure definitions
        struct_pattern = r'struct\s+(\w+)\s*\{'
        structures = re.findall(struct_pattern, content)
        analysis['structures'] = structures
        
        return analysis

    def _analyze_all_methods(self, all_source_content: Dict[str, str]) -> Dict[str, Any]:
        """Analyze all methods and functions across all source files"""
        
        analysis = {
            'total_methods': 0,
            'real_implementations': 0,
            'placeholder_methods': 0,
            'empty_methods': 0,
            'complex_methods': 0,
            'method_details': [],
            'implementation_quality': 0.0,
            'critical_methods': [],
            'missing_implementations': []
        }
        
        for filename, content in all_source_content.items():
            file_methods = self._extract_all_methods_from_file(filename, content)
            
            for method in file_methods:
                analysis['total_methods'] += 1
                analysis['method_details'].append(method)
                
                if method['is_placeholder']:
                    analysis['placeholder_methods'] += 1
                elif method['is_empty']:
                    analysis['empty_methods'] += 1
                elif method['has_real_implementation']:
                    analysis['real_implementations'] += 1
                    
                if method['complexity'] > 5:
                    analysis['complex_methods'] += 1
                
                # Identify critical methods
                if method['name'] in ['main', 'WinMain', 'DllMain'] or 'init' in method['name'].lower():
                    analysis['critical_methods'].append(method)
        
        # Calculate implementation quality
        if analysis['total_methods'] > 0:
            analysis['implementation_quality'] = analysis['real_implementations'] / analysis['total_methods']
        
        return analysis

    def _extract_all_methods_from_file(self, filename: str, content: str) -> List[Dict[str, Any]]:
        """Extract all methods from a single file"""
        
        methods = []
        function_pattern = r'(\w+(?:\s*\*)*)\s+(\w+)\s*\(([^)]*)\)\s*\{'
        
        for func_match in re.finditer(function_pattern, content, re.MULTILINE):
            return_type = func_match.group(1).strip()
            func_name = func_match.group(2)
            parameters = func_match.group(3).strip()
            
            # Extract and analyze function body
            func_body = self._extract_function_body(content, func_match.start())
            body_analysis = self._analyze_function_implementation(func_name, func_body)
            
            method_info = {
                'filename': filename,
                'name': func_name,
                'return_type': return_type,
                'parameters': parameters,
                'body': func_body,
                'body_lines': body_analysis['body_lines'],
                'complexity': body_analysis['complexity'],
                'has_real_implementation': body_analysis['has_real_logic'],
                'is_placeholder': body_analysis['is_placeholder'],
                'is_empty': body_analysis['is_empty'],
                'control_structures': body_analysis['control_structures'],
                'variable_assignments': body_analysis['variable_assignments'],
                'function_calls': body_analysis['function_calls']
            }
            
            methods.append(method_info)
        
        return methods

    def _analyze_all_classes(self, all_source_content: Dict[str, str]) -> Dict[str, Any]:
        """Analyze all classes and structures"""
        
        analysis = {
            'total_classes': 0,
            'total_structures': 0,
            'class_details': [],
            'structure_details': [],
            'has_object_oriented_design': False,
            'data_structure_complexity': 0.0
        }
        
        for filename, content in all_source_content.items():
            # Find C structures
            struct_pattern = r'struct\s+(\w+)\s*\{([^}]*)\}'
            for struct_match in re.finditer(struct_pattern, content, re.DOTALL):
                struct_name = struct_match.group(1)
                struct_body = struct_match.group(2)
                
                members = self._parse_struct_members(struct_body)
                
                analysis['structure_details'].append({
                    'filename': filename,
                    'name': struct_name,
                    'member_count': len(members),
                    'members': members,
                    'complexity': len(members)
                })
                
                analysis['total_structures'] += 1
                analysis['data_structure_complexity'] += len(members)
            
            # Find C++ classes (if any)
            class_pattern = r'class\s+(\w+)(?:\s*:\s*[^{]*)?s*\{([^}]*)\}'
            for class_match in re.finditer(class_pattern, content, re.DOTALL):
                class_name = class_match.group(1)
                class_body = class_match.group(2)
                
                methods = self._parse_class_methods(class_body)
                members = self._parse_class_members(class_body)
                
                analysis['class_details'].append({
                    'filename': filename,
                    'name': class_name,
                    'method_count': len(methods),
                    'member_count': len(members),
                    'methods': methods,
                    'members': members
                })
                
                analysis['total_classes'] += 1
                analysis['has_object_oriented_design'] = True
        
        return analysis

    def _analyze_implementation_completeness(self,
                                           source_analysis: Dict[str, Any],
                                           method_analysis: Dict[str, Any], 
                                           class_analysis: Dict[str, Any],
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze completeness of the implementation"""
        
        analysis = {
            'completeness_score': 0.0,
            'missing_critical_components': [],
            'placeholder_ratio': 0.0,
            'implementation_depth': 0.0,
            'architectural_completeness': 0.0,
            'functional_completeness': 0.0,
            'completeness_assessment': '',
            'critical_gaps': []
        }
        
        # Calculate placeholder ratio
        total_methods = method_analysis['total_methods']
        placeholder_methods = method_analysis['placeholder_methods']
        
        if total_methods > 0:
            analysis['placeholder_ratio'] = placeholder_methods / total_methods
        
        # Assess implementation depth
        real_implementations = method_analysis['real_implementations']
        complex_methods = method_analysis['complex_methods']
        
        if total_methods > 0:
            analysis['implementation_depth'] = (real_implementations + complex_methods * 0.5) / total_methods
        
        # Check for critical components
        critical_components = ['main', 'WinMain', 'DllMain', 'initialization', 'cleanup']
        found_critical = []
        
        for method in method_analysis['method_details']:
            method_name_lower = method['name'].lower()
            for critical in critical_components:
                if critical in method_name_lower:
                    found_critical.append(critical)
        
        analysis['missing_critical_components'] = [c for c in critical_components if c not in found_critical]
        
        # Assess architectural completeness
        has_headers = len(source_analysis['header_files']) > 0
        has_sources = len(source_analysis['source_files']) > 0
        has_structures = class_analysis['total_structures'] > 0
        has_includes = any(details.get('includes', []) for details in source_analysis['file_summary'])
        
        architectural_factors = [has_headers, has_sources, has_structures, has_includes]
        analysis['architectural_completeness'] = sum(architectural_factors) / len(architectural_factors)
        
        # Assess functional completeness
        functional_factors = [
            analysis['implementation_depth'] > 0.6,
            analysis['placeholder_ratio'] < 0.3,
            len(analysis['missing_critical_components']) == 0,
            method_analysis['implementation_quality'] > 0.5
        ]
        analysis['functional_completeness'] = sum(functional_factors) / len(functional_factors)
        
        # Overall completeness score
        analysis['completeness_score'] = (
            analysis['architectural_completeness'] * 0.3 +
            analysis['functional_completeness'] * 0.5 +
            analysis['implementation_depth'] * 0.2
        )
        
        # Generate assessment
        if analysis['completeness_score'] >= 0.8:
            analysis['completeness_assessment'] = 'Excellent - Comprehensive implementation'
        elif analysis['completeness_score'] >= 0.6:
            analysis['completeness_assessment'] = 'Good - Substantial implementation with some gaps'
        elif analysis['completeness_score'] >= 0.4:
            analysis['completeness_assessment'] = 'Partial - Basic structure with significant gaps'
        elif analysis['completeness_score'] >= 0.2:
            analysis['completeness_assessment'] = 'Minimal - Mostly placeholder code'
        else:
            analysis['completeness_assessment'] = 'Inadequate - No meaningful implementation'
        
        return analysis

    def _calculate_comprehensive_quality_metrics(self,
                                               source_analysis: Dict[str, Any],
                                               method_analysis: Dict[str, Any],
                                               class_analysis: Dict[str, Any],
                                               implementation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        
        metrics = {
            'overall_quality': 0.0,
            'code_structure_quality': 0.0,
            'implementation_quality': 0.0,
            'design_quality': 0.0,
            'documentation_quality': 0.0,
            'maintainability_score': 0.0,
            'quality_grade': '',
            'quality_breakdown': {}
        }
        
        # Code structure quality
        structure_factors = []
        if source_analysis['total_files'] > 0:
            structure_factors.append(min(1.0, source_analysis['code_lines'] / 100))  # At least 100 lines
            structure_factors.append(source_analysis['code_lines'] / source_analysis['total_lines'] if source_analysis['total_lines'] > 0 else 0)
            structure_factors.append(min(1.0, len(source_analysis['source_files']) / 3))  # At least 3 source files
        
        metrics['code_structure_quality'] = sum(structure_factors) / len(structure_factors) if structure_factors else 0
        
        # Implementation quality (from method analysis)
        metrics['implementation_quality'] = method_analysis['implementation_quality']
        
        # Design quality
        design_factors = [
            class_analysis['total_structures'] > 0,  # Has data structures
            len(source_analysis['header_files']) > 0,  # Has header files
            implementation_analysis['architectural_completeness'],
            method_analysis['complex_methods'] / max(method_analysis['total_methods'], 1) > 0.3
        ]
        metrics['design_quality'] = sum(design_factors) / len(design_factors)
        
        # Documentation quality (basic check for comments)
        if source_analysis['total_lines'] > 0:
            comment_ratio = source_analysis['comment_lines'] / source_analysis['total_lines']
            metrics['documentation_quality'] = min(1.0, comment_ratio * 5)  # 20% comments = perfect score
        
        # Maintainability score
        maintainability_factors = [
            metrics['code_structure_quality'],
            metrics['design_quality'],
            1.0 - implementation_analysis['placeholder_ratio'],
            metrics['documentation_quality']
        ]
        metrics['maintainability_score'] = sum(maintainability_factors) / len(maintainability_factors)
        
        # Overall quality (weighted average)
        metrics['overall_quality'] = (
            metrics['code_structure_quality'] * 0.25 +
            metrics['implementation_quality'] * 0.35 +
            metrics['design_quality'] * 0.25 +
            metrics['maintainability_score'] * 0.15
        )
        
        # Quality grade
        if metrics['overall_quality'] >= 0.9:
            metrics['quality_grade'] = 'A+ (Excellent)'
        elif metrics['overall_quality'] >= 0.8:
            metrics['quality_grade'] = 'A (Very Good)'
        elif metrics['overall_quality'] >= 0.7:
            metrics['quality_grade'] = 'B (Good)'
        elif metrics['overall_quality'] >= 0.6:
            metrics['quality_grade'] = 'C (Acceptable)'
        elif metrics['overall_quality'] >= 0.4:
            metrics['quality_grade'] = 'D (Poor)'
        else:
            metrics['quality_grade'] = 'F (Inadequate)'
        
        # Store breakdown
        metrics['quality_breakdown'] = {
            'Structure': f"{metrics['code_structure_quality']:.1%}",
            'Implementation': f"{metrics['implementation_quality']:.1%}",
            'Design': f"{metrics['design_quality']:.1%}",
            'Documentation': f"{metrics['documentation_quality']:.1%}",
            'Maintainability': f"{metrics['maintainability_score']:.1%}"
        }
        
        return metrics

    def _compare_with_original_binary(self, 
                                    validation_report: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare generated source with original binary capabilities"""
        
        comparison = {
            'size_analysis': {},
            'complexity_comparison': {},
            'functionality_assessment': {},
            'equivalence_score': 0.0,
            'likelihood_of_equivalence': '',
            'comparison_confidence': 0.0
        }
        
        binary_path = context.get('global_data', {}).get('binary_path', '')
        
        if binary_path and os.path.exists(binary_path):
            binary_size = os.path.getsize(binary_path)
            
            # Estimate generated code size
            total_code_lines = validation_report['source_analysis']['code_lines']
            estimated_source_size = total_code_lines * 50  # Rough estimate: 50 chars per line
            
            comparison['size_analysis'] = {
                'original_binary_size': binary_size,
                'estimated_source_size': estimated_source_size,
                'size_ratio': estimated_source_size / binary_size if binary_size > 0 else 0,
                'size_reasonableness': 'reasonable' if 0.1 <= (estimated_source_size / binary_size) <= 10 else 'unreasonable'
            }
            
            # Compare complexity
            method_count = validation_report['method_analysis']['total_methods']
            complex_methods = validation_report['method_analysis']['complex_methods']
            
            comparison['complexity_comparison'] = {
                'total_methods': method_count,
                'complex_methods': complex_methods,
                'complexity_density': complex_methods / max(method_count, 1),
                'complexity_assessment': self._assess_complexity_adequacy(method_count, complex_methods, binary_size)
            }
            
            # Assess functionality
            implementation_quality = validation_report['quality_metrics']['implementation_quality']
            placeholder_ratio = validation_report['implementation_analysis']['placeholder_ratio']
            
            comparison['functionality_assessment'] = {
                'implementation_coverage': implementation_quality,
                'placeholder_burden': placeholder_ratio,
                'functional_likelihood': max(0, implementation_quality - placeholder_ratio),
                'assessment': self._assess_functional_equivalence(implementation_quality, placeholder_ratio)
            }
            
            # Calculate equivalence score
            equivalence_factors = [
                comparison['size_analysis']['size_reasonableness'] == 'reasonable',
                comparison['complexity_comparison']['complexity_density'] > 0.2,
                comparison['functionality_assessment']['functional_likelihood'] > 0.5,
                validation_report['quality_metrics']['overall_quality'] > 0.6
            ]
            
            comparison['equivalence_score'] = sum(equivalence_factors) / len(equivalence_factors)
            
            # Determine likelihood
            if comparison['equivalence_score'] >= 0.8:
                comparison['likelihood_of_equivalence'] = 'Very High - Likely equivalent functionality'
            elif comparison['equivalence_score'] >= 0.6:
                comparison['likelihood_of_equivalence'] = 'High - Probably equivalent with some differences'
            elif comparison['equivalence_score'] >= 0.4:
                comparison['likelihood_of_equivalence'] = 'Medium - Partial equivalence likely'
            elif comparison['equivalence_score'] >= 0.2:
                comparison['likelihood_of_equivalence'] = 'Low - Minimal equivalence'
            else:
                comparison['likelihood_of_equivalence'] = 'Very Low - Unlikely to be equivalent'
            
            comparison['comparison_confidence'] = min(1.0, total_code_lines / 50)  # Confidence based on code amount
        
        return comparison

    def _make_final_determination(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Make final determination about implementation validity"""
        
        determination = {
            'is_real_implementation': False,
            'confidence_score': 0.0,
            'failure_reason': '',
            'recommendation': '',
            'detailed_findings': []
        }
        
        # Extract key metrics
        overall_quality = validation_report['quality_metrics']['overall_quality']
        implementation_quality = validation_report['method_analysis']['implementation_quality']
        placeholder_ratio = validation_report['implementation_analysis']['placeholder_ratio']
        completeness_score = validation_report['implementation_analysis']['completeness_score']
        equivalence_score = validation_report['comparison_analysis']['equivalence_score']
        
        # Critical validation criteria (STRICT thresholds to prevent dummy code)
        criteria = {
            'sufficient_quality': overall_quality >= 0.75,  # STRICT: 75% minimum quality
            'real_implementations': implementation_quality >= 0.75,  # STRICT: 75% real implementations  
            'low_placeholders': placeholder_ratio <= 0.15,  # STRICT: max 15% placeholders
            'adequate_completeness': completeness_score >= 0.7,  # STRICT: 70% completeness
            'reasonable_equivalence': equivalence_score >= 0.4,  # STRICT: 40% equivalence minimum
            'size_reasonableness': self._check_size_reasonableness(validation_report),  # NEW: Size check
            'complexity_adequacy': self._check_complexity_adequacy(validation_report)   # NEW: Complexity check
        }
        
        # Count passing criteria
        passing_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        determination['confidence_score'] = passing_criteria / total_criteria
        
        # Record detailed findings
        for criterion, passed in criteria.items():
            status = "PASS" if passed else "FAIL"
            determination['detailed_findings'].append(f"{criterion}: {status}")
        
        # Additional findings
        determination['detailed_findings'].extend([
            f"Overall Quality: {overall_quality:.1%}",
            f"Implementation Quality: {implementation_quality:.1%}",
            f"Placeholder Ratio: {placeholder_ratio:.1%}",
            f"Completeness Score: {completeness_score:.1%}",
            f"Equivalence Score: {equivalence_score:.1%}"
        ])
        
        # STRICT determination - ALL criteria must pass for acceptance
        if passing_criteria == total_criteria:  # ALL criteria must pass (100%)
            determination['is_real_implementation'] = True
            determination['recommendation'] = "ACCEPT - This appears to be a legitimate implementation of the target binary"
        elif passing_criteria >= total_criteria * 0.8:  # At least 80% criteria pass
            determination['is_real_implementation'] = False
            determination['failure_reason'] = f"REJECT - High standards not met. Only {passing_criteria}/{total_criteria} criteria passed. Need ALL criteria for acceptance."
            determination['recommendation'] = "REJECT - Implementation does not meet strict quality thresholds"
        else:
            determination['is_real_implementation'] = False
            determination['failure_reason'] = f"REJECT - Poor implementation quality. Only {passing_criteria}/{total_criteria} criteria passed"
            determination['recommendation'] = "REJECT - This appears to be mostly placeholder code, not a real implementation"
        
        return determination

    def _save_validation_report(self, validation_report: Dict[str, Any], context: Dict[str, Any]):
        """Save comprehensive validation report"""
        
        output_paths = context.get('output_paths', {})
        reports_dir = output_paths.get('reports', 'output/reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = os.path.join(reports_dir, 'source_code_validation_report.json')
        
        try:
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save validation report: {e}")

    def _fail_pipeline(self, reason: str, context: Dict[str, Any], 
                      validation_data: Dict[str, Any] = None) -> AgentResult:
        """Fail the entire pipeline with detailed reason"""
        
        failure_data = {
            'pipeline_terminated': True,
            'termination_reason': reason,
            'termination_agent': self.agent_id,
            'validation_data': validation_data or {}
        }
        
        # Save failure report
        try:
            output_paths = context.get('output_paths', {})
            reports_dir = output_paths.get('reports', 'output/reports')
            os.makedirs(reports_dir, exist_ok=True)
            
            failure_path = os.path.join(reports_dir, 'pipeline_failure_report.json')
            with open(failure_path, 'w') as f:
                json.dump(failure_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save failure report: {e}")
        
        return AgentResult(
            agent_id=self.agent_id,
            status=AgentStatus.FAILED,
            data=failure_data,
            error_message=reason,
            metadata={
                'termination_agent': True,
                'pipeline_terminated': True
            }
        )

    # Helper methods
    def _extract_function_body(self, content: str, start_pos: int) -> str:
        """Extract function body from starting position"""
        try:
            brace_pos = content.find('{', start_pos)
            if brace_pos == -1:
                return ""
            
            brace_count = 1
            pos = brace_pos + 1
            
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            return content[brace_pos:pos] if brace_count == 0 else content[brace_pos:]
        except Exception:
            return ""

    def _analyze_function_implementation(self, func_name: str, func_body: str) -> Dict[str, Any]:
        """Analyze function implementation quality"""
        
        analysis = {
            'body_lines': len(func_body.split('\n')),
            'complexity': 1,
            'has_real_logic': False,
            'is_placeholder': False,
            'is_empty': False,
            'control_structures': 0,
            'variable_assignments': 0,
            'function_calls': 0
        }
        
        if not func_body.strip() or func_body.strip() == '{}':
            analysis['is_empty'] = True
            return analysis
        
        # Check for placeholder patterns
        placeholder_patterns = [
            r'TODO|FIXME|placeholder',
            r'printf.*".*placeholder.*"',
            r'printf.*".*Hello.*World.*"',
            r'return\s+0;\s*}\s*$',
            r'Generated by.*analysis',
            r'Reconstructed.*binary'
        ]
        
        for pattern in placeholder_patterns:
            if re.search(pattern, func_body, re.IGNORECASE):
                analysis['is_placeholder'] = True
                break
        
        # Calculate complexity
        complexity_patterns = [
            r'if\s*\(', r'else', r'while\s*\(', r'for\s*\(',
            r'switch\s*\(', r'case\s+', r'\?\s*.*\s*:', r'&&', r'\|\|'
        ]
        
        for pattern in complexity_patterns:
            matches = len(re.findall(pattern, func_body))
            analysis['complexity'] += matches
            if pattern in [r'if\s*\(', r'while\s*\(', r'for\s*\(', r'switch\s*\(']:
                analysis['control_structures'] += matches
        
        # Count assignments and function calls
        analysis['variable_assignments'] = len(re.findall(r'\w+\s*=\s*[^=]', func_body))
        analysis['function_calls'] = len(re.findall(r'\w+\s*\(', func_body))
        
        # Determine if it has real logic
        analysis['has_real_logic'] = (
            analysis['complexity'] > 2 or
            analysis['control_structures'] > 0 or
            analysis['variable_assignments'] > 2 or
            (analysis['function_calls'] > 1 and not analysis['is_placeholder'])
        )
        
        return analysis

    def _parse_struct_members(self, struct_body: str) -> List[Dict[str, str]]:
        """Parse structure members"""
        members = []
        for line in struct_body.split('\n'):
            line = line.strip()
            if line and not line.startswith('//') and ';' in line:
                # Simple member parsing
                parts = line.replace(';', '').split()
                if len(parts) >= 2:
                    members.append({'type': parts[0], 'name': parts[1]})
        return members

    def _parse_class_methods(self, class_body: str) -> List[str]:
        """Parse class methods (basic implementation)"""
        methods = []
        function_pattern = r'(\w+)\s*\([^)]*\)\s*[;{]'
        for match in re.finditer(function_pattern, class_body):
            methods.append(match.group(1))
        return methods

    def _parse_class_members(self, class_body: str) -> List[Dict[str, str]]:
        """Parse class members (basic implementation)"""
        return self._parse_struct_members(class_body)

    def _assess_complexity_adequacy(self, method_count: int, complex_methods: int, binary_size: int) -> str:
        """Assess if complexity is adequate for binary size"""
        if binary_size < 10000:  # Small binary
            return 'adequate' if method_count >= 3 else 'too_simple'
        elif binary_size < 100000:  # Medium binary
            return 'adequate' if method_count >= 10 and complex_methods >= 3 else 'too_simple'
        else:  # Large binary
            return 'adequate' if method_count >= 20 and complex_methods >= 8 else 'too_simple'

    def _assess_functional_equivalence(self, implementation_quality: float, placeholder_ratio: float) -> str:
        """Assess functional equivalence likelihood"""
        net_quality = implementation_quality - placeholder_ratio
        
        if net_quality >= 0.7:
            return 'Very likely equivalent'
        elif net_quality >= 0.5:
            return 'Probably equivalent'
        elif net_quality >= 0.3:
            return 'Possibly equivalent'
        elif net_quality >= 0.1:
            return 'Unlikely equivalent'
        else:
            return 'Very unlikely equivalent'

    def _check_size_reasonableness(self, validation_report: Dict[str, Any]) -> bool:
        """Check if the generated source code size is reasonable compared to binary size"""
        try:
            size_analysis = validation_report.get('comparison_analysis', {}).get('size_analysis', {})
            size_ratio = size_analysis.get('size_ratio', 0)
            size_reasonableness = size_analysis.get('size_reasonableness', 'unknown')
            
            # A source-to-binary ratio below 0.001 (0.1%) is extremely suspicious for a 5MB binary
            # Real implementations should have at least 0.5% ratio for large binaries
            return size_ratio >= 0.005 and size_reasonableness != 'unreasonable'
        except:
            return False

    def _check_complexity_adequacy(self, validation_report: Dict[str, Any]) -> bool:
        """Check if the code complexity is adequate for the binary size"""
        try:
            complexity_comparison = validation_report.get('comparison_analysis', {}).get('complexity_comparison', {})
            complexity_assessment = complexity_comparison.get('complexity_assessment', 'unknown')
            complex_methods = complexity_comparison.get('complex_methods', 0)
            total_methods = complexity_comparison.get('total_methods', 0)
            
            # For a 5MB binary, we expect significant complexity
            # At least 20% of methods should be complex, and assessment shouldn't be "too_simple"
            if total_methods == 0:
                return False
            
            complexity_ratio = complex_methods / total_methods
            return complexity_ratio >= 0.2 and complexity_assessment != 'too_simple'
        except:
            return False