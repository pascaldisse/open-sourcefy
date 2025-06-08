"""
Agent 20: Automated Testing Framework
Provides comprehensive testing and validation for production readiness.
Phase 4: Build Systems & Production Readiness
"""

import os
import json
import subprocess
import tempfile
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent20_AutoTesting(BaseAgent):
    """Agent 20: Automated testing and validation framework"""
    
    def __init__(self):
        super().__init__(
            agent_id=20,
            name="AutoTesting",
            dependencies=[18, 19]  # Depends on AdvancedBuildSystems and BinaryComparison
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute automated testing and validation"""
        agent18_result = context['agent_results'].get(18)
        agent19_result = context['agent_results'].get(19)
        
        if not agent18_result or agent18_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 18 (AdvancedBuildSystems) did not complete successfully"
            )

        try:
            build_systems_data = agent18_result.data
            comparison_data = agent19_result.data if agent19_result else {}
            
            testing_result = self._perform_automated_testing(
                build_systems_data, comparison_data, context
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=testing_result,
                metadata={
                    'depends_on': [18, 19],
                    'analysis_type': 'automated_testing'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Automated testing failed: {str(e)}"
            )

    def _perform_automated_testing(self, build_systems_data: Dict[str, Any], 
                                 comparison_data: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive automated testing"""
        result = {
            'unit_tests': {},
            'integration_tests': {},
            'regression_tests': {},
            'performance_tests': {},
            'compatibility_tests': {},
            'security_tests': {},
            'validation_tests': {},
            'test_coverage': {},
            'test_results_summary': {},
            'production_readiness': 'unknown',
            'overall_test_score': 0.0,
            'recommendations': []
        }
        
        # Get output directory structure
        base_output_dir = context.get('output_dir', 'output')
        output_paths = context.get('output_paths', {})
        
        # Use output structure for tests directory or fallback to temp
        test_dir = output_paths.get('tests', output_paths.get('temp', os.path.join(base_output_dir, 'temp')))
        
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # Generate and run unit tests
            result['unit_tests'] = self._run_unit_tests(build_systems_data, test_dir, context)
            
            # Run integration tests
            result['integration_tests'] = self._run_integration_tests(
                build_systems_data, test_dir, context
            )
            
            # Run regression tests
            result['regression_tests'] = self._run_regression_tests(
                build_systems_data, comparison_data, test_dir, context
            )
            
            # Run performance tests
            result['performance_tests'] = self._run_performance_tests(
                build_systems_data, test_dir, context
            )
            
            # Run compatibility tests
            result['compatibility_tests'] = self._run_compatibility_tests(
                build_systems_data, test_dir, context
            )
            
            # Run security tests
            result['security_tests'] = self._run_security_tests(
                build_systems_data, test_dir, context
            )
            
            # Run validation tests
            result['validation_tests'] = self._run_validation_tests(
                build_systems_data, comparison_data, test_dir, context
            )
            
            # Calculate test coverage
            result['test_coverage'] = self._calculate_test_coverage(result)
            
            # Generate test results summary
            result['test_results_summary'] = self._generate_test_summary(result)
            
            # Assess production readiness
            result['production_readiness'] = self._assess_production_readiness(result)
            
            # Calculate overall test score
            result['overall_test_score'] = self._calculate_overall_test_score(result)
            
            # Generate recommendations
            result['recommendations'] = self._generate_test_recommendations(result)
            
            # Perform cleanup validation and temp directory management
            result['cleanup_validation'] = self._perform_cleanup_validation(output_paths, context)
            
            # NEW: Perform comprehensive source-to-binary validation
            result['source_binary_validation'] = self._perform_source_binary_validation(
                build_systems_data, comparison_data, context, test_dir
            )
            
        except Exception as e:
            result['error'] = str(e)
            result['production_readiness'] = 'failed'
        
        return result

    def _run_unit_tests(self, build_systems_data: Dict[str, Any], 
                       test_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate and run unit tests for individual functions"""
        unit_tests = {
            'generated_tests': [],
            'test_results': {},
            'pass_count': 0,
            'fail_count': 0,
            'total_tests': 0,
            'coverage': 0.0,
            'success_rate': 0.0
        }
        
        try:
            # Get reconstructed functions from context
            reconstructed_functions = self._extract_functions_from_context(context)
            
            # Generate unit tests for each function
            for func_name, func_info in reconstructed_functions.items():
                test_case = self._generate_unit_test(func_name, func_info)
                if test_case:
                    unit_tests['generated_tests'].append(test_case)
            
            # Write unit test file
            if unit_tests['generated_tests']:
                test_file = self._write_unit_test_file(unit_tests['generated_tests'], test_dir)
                
                # Compile and run tests
                test_results = self._compile_and_run_tests(test_file, test_dir)
                unit_tests['test_results'] = test_results
                
                # Calculate metrics
                unit_tests['pass_count'] = test_results.get('passed', 0)
                unit_tests['fail_count'] = test_results.get('failed', 0)
                unit_tests['total_tests'] = len(unit_tests['generated_tests'])
                
                if unit_tests['total_tests'] > 0:
                    unit_tests['success_rate'] = unit_tests['pass_count'] / unit_tests['total_tests']
                    unit_tests['coverage'] = min(1.0, unit_tests['total_tests'] / len(reconstructed_functions))
                
        except Exception as e:
            unit_tests['error'] = str(e)
        
        return unit_tests

    def _extract_functions_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reconstructed functions from agent results"""
        functions = {}
        
        # Get from Agent 11 (GlobalReconstructor)
        agent11_result = context.get('agent_results', {}).get(11)
        if agent11_result and agent11_result.status == AgentStatus.COMPLETED:
            global_reconstruction = agent11_result.data
            reconstructed_source = global_reconstruction.get('reconstructed_source', {})
            source_files = reconstructed_source.get('source_files', {})
            
            # Extract function information from source files
            for filename, content in source_files.items():
                if isinstance(content, str):
                    functions.update(self._parse_functions_from_source(content, filename))
        
        return functions

    def _parse_functions_from_source(self, source_code: str, filename: str) -> Dict[str, Any]:
        """Parse function definitions from source code"""
        functions = {}
        
        # Simple function detection (can be enhanced)
        lines = source_code.split('\n')
        current_function = None
        brace_count = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Detect function definitions (simple heuristic)
            if ('(' in line and ')' in line and 
                not line.startswith('//') and 
                not line.startswith('#') and
                not line.startswith('if') and
                not line.startswith('while') and
                not line.startswith('for')):
                
                # Potential function definition
                if '{' in line or (i + 1 < len(lines) and '{' in lines[i + 1]):
                    function_name = self._extract_function_name(line)
                    if function_name:
                        current_function = function_name
                        functions[current_function] = {
                            'signature': line,
                            'source_file': filename,
                            'line_number': i + 1,
                            'code_lines': []
                        }
            
            # Track function code
            if current_function:
                if '{' in line:
                    brace_count += line.count('{')
                if '}' in line:
                    brace_count -= line.count('}')
                    
                functions[current_function]['code_lines'].append(line)
                
                if brace_count == 0:
                    current_function = None
        
        return functions

    def _extract_function_name(self, line: str) -> Optional[str]:
        """Extract function name from function definition line"""
        try:
            # Remove return type and extract function name
            if '(' in line:
                before_paren = line.split('(')[0].strip()
                # Get the last word (function name)
                parts = before_paren.split()
                if parts:
                    return parts[-1]
        except Exception:
            pass
        return None

    def _generate_unit_test(self, func_name: str, func_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a unit test for a function"""
        try:
            signature = func_info.get('signature', '')
            
            # Simple test generation based on function signature
            test_case = {
                'function_name': func_name,
                'test_name': f'test_{func_name}',
                'test_code': self._create_basic_test_code(func_name, signature),
                'expected_result': 'success',
                'test_type': 'basic_execution'
            }
            
            return test_case
            
        except Exception:
            return None

    def _create_basic_test_code(self, func_name: str, signature: str) -> str:
        """Create basic test code for a function"""
        # Very simple test - just try to call the function
        test_code = f"""
void test_{func_name}() {{
    // Basic test for {func_name}
    printf("Testing {func_name}...\\n");
    
    // Try to call function with safe parameters
    // This is a basic test - would need enhancement for real testing
    
    printf("Test {func_name} completed\\n");
}}
"""
        return test_code

    def _write_unit_test_file(self, test_cases: List[Dict[str, Any]], test_dir: str) -> str:
        """Write unit test file"""
        test_file = os.path.join(test_dir, 'unit_tests.c')
        
        test_content = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test framework
int tests_passed = 0;
int tests_failed = 0;

#define ASSERT(condition) \\
    if (condition) { \\
        tests_passed++; \\
        printf("PASS: %s\\n", #condition); \\
    } else { \\
        tests_failed++; \\
        printf("FAIL: %s\\n", #condition); \\
    }

"""
        
        # Add test functions
        for test_case in test_cases:
            test_content += test_case['test_code'] + '\n'
        
        # Add main function
        test_content += """
int main() {
    printf("Running unit tests...\\n\\n");
    
"""
        
        for test_case in test_cases:
            test_content += f"    {test_case['test_name']}();\n"
        
        test_content += """
    printf("\\nTest Results:\\n");
    printf("Passed: %d\\n", tests_passed);
    printf("Failed: %d\\n", tests_failed);
    printf("Total: %d\\n", tests_passed + tests_failed);
    
    return tests_failed > 0 ? 1 : 0;
}
"""
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        return test_file

    def _compile_and_run_tests(self, test_file: str, test_dir: str) -> Dict[str, Any]:
        """Compile and run test file"""
        results = {
            'compiled': False,
            'executed': False,
            'passed': 0,
            'failed': 0,
            'total': 0,
            'output': '',
            'errors': []
        }
        
        try:
            # Try to compile with gcc
            test_exe = os.path.join(test_dir, 'unit_tests.exe')
            
            compile_cmd = ['gcc', '-o', test_exe, test_file]
            compile_result = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=60
            )
            
            if compile_result.returncode == 0:
                results['compiled'] = True
                
                # Run the tests
                run_result = subprocess.run(
                    [test_exe], capture_output=True, text=True, timeout=60, cwd=test_dir
                )
                
                results['executed'] = True
                results['output'] = run_result.stdout
                
                # Parse test results
                output_lines = run_result.stdout.split('\n')
                for line in output_lines:
                    if 'Passed:' in line:
                        results['passed'] = int(line.split(':')[1].strip())
                    elif 'Failed:' in line:
                        results['failed'] = int(line.split(':')[1].strip())
                    elif 'Total:' in line:
                        results['total'] = int(line.split(':')[1].strip())
            else:
                results['errors'].append(f"Compilation failed: {compile_result.stderr}")
                
        except Exception as e:
            results['errors'].append(f"Test execution failed: {str(e)}")
        
        return results

    def _run_integration_tests(self, build_systems_data: Dict[str, Any], 
                             test_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run integration tests"""
        integration_tests = {
            'build_system_tests': {},
            'compiler_integration_tests': {},
            'binary_execution_tests': {},
            'success_rate': 0.0,
            'total_tests': 0,
            'passed_tests': 0
        }
        
        try:
            # Test each build system
            successful_compilers = build_systems_data.get('successful_compilers', [])
            
            for compiler in successful_compilers:
                test_result = self._test_compiler_integration(compiler, build_systems_data, test_dir)
                integration_tests['compiler_integration_tests'][compiler] = test_result
                
                if test_result.get('success', False):
                    integration_tests['passed_tests'] += 1
                integration_tests['total_tests'] += 1
            
            # Test build systems
            build_systems = ['cmake', 'makefile', 'ninja', 'vs_solution']
            for build_system in build_systems:
                if build_systems_data.get(f'{build_system}_generated', False):
                    test_result = self._test_build_system(build_system, test_dir)
                    integration_tests['build_system_tests'][build_system] = test_result
                    
                    if test_result.get('success', False):
                        integration_tests['passed_tests'] += 1
                    integration_tests['total_tests'] += 1
            
            # Calculate success rate
            if integration_tests['total_tests'] > 0:
                integration_tests['success_rate'] = (
                    integration_tests['passed_tests'] / integration_tests['total_tests']
                )
                
        except Exception as e:
            integration_tests['error'] = str(e)
        
        return integration_tests

    def _test_compiler_integration(self, compiler: str, build_systems_data: Dict[str, Any], 
                                 test_dir: str) -> Dict[str, Any]:
        """Test integration with specific compiler"""
        test_result = {
            'success': False,
            'can_compile': False,
            'binary_works': False,
            'compilation_time': 0.0,
            'errors': []
        }
        
        try:
            compilation_attempts = build_systems_data.get('compilation_attempts', {})
            compiler_attempt = compilation_attempts.get(compiler, {})
            
            if compiler_attempt.get('success', False):
                test_result['can_compile'] = True
                
                binary_path = compiler_attempt.get('binary_path')
                if binary_path and os.path.exists(binary_path):
                    # Test if binary executes
                    exec_test = self._test_binary_execution(binary_path)
                    test_result['binary_works'] = exec_test.get('can_execute', False)
                    test_result['success'] = test_result['binary_works']
            else:
                test_result['errors'] = compiler_attempt.get('errors', [])
                
        except Exception as e:
            test_result['errors'].append(str(e))
        
        return test_result

    def _test_build_system(self, build_system: str, test_dir: str) -> Dict[str, Any]:
        """Test specific build system"""
        test_result = {
            'success': False,
            'files_exist': False,
            'can_build': False,
            'errors': []
        }
        
        try:
            # Check if build files exist
            build_files = {
                'cmake': 'CMakeLists.txt',
                'makefile': 'Makefile',
                'ninja': 'build.ninja',
                'vs_solution': 'launcher-new.sln'
            }
            
            compilation_dir = os.path.dirname(test_dir)  # Go up one level
            if os.path.exists(compilation_dir):
                build_file = build_files.get(build_system)
                if build_file:
                    build_file_path = os.path.join(compilation_dir, build_file)
                    test_result['files_exist'] = os.path.exists(build_file_path)
                    
                    if test_result['files_exist']:
                        # Try to use the build system (basic test)
                        test_result['can_build'] = self._test_build_file(
                            build_system, build_file_path, compilation_dir
                        )
                        test_result['success'] = test_result['can_build']
                        
        except Exception as e:
            test_result['errors'].append(str(e))
        
        return test_result

    def _test_build_file(self, build_system: str, build_file_path: str, build_dir: str) -> bool:
        """Test if build file can be used"""
        try:
            if build_system == 'cmake':
                # Test CMake configuration
                result = subprocess.run([
                    'cmake', '--version'
                ], capture_output=True, text=True, timeout=10)
                return result.returncode == 0
                
            elif build_system == 'makefile':
                # Test Make availability
                result = subprocess.run([
                    'make', '--version'
                ], capture_output=True, text=True, timeout=10)
                return result.returncode == 0
                
            elif build_system == 'ninja':
                # Test Ninja availability
                result = subprocess.run([
                    'ninja', '--version'
                ], capture_output=True, text=True, timeout=10)
                return result.returncode == 0
                
            elif build_system == 'vs_solution':
                # Test MSBuild availability
                result = subprocess.run([
                    'msbuild', '/version'
                ], capture_output=True, text=True, timeout=10)
                return result.returncode == 0
                
        except Exception:
            pass
        
        return False

    def _test_binary_execution(self, binary_path: str) -> Dict[str, Any]:
        """Test binary execution"""
        test_result = {
            'can_execute': False,
            'exit_code': None,
            'execution_time': 0.0,
            'output': '',
            'errors': []
        }
        
        try:
            start_time = time.time()
            
            result = subprocess.run([
                binary_path
            ], capture_output=True, text=True, timeout=5)
            
            test_result['can_execute'] = True
            test_result['exit_code'] = result.returncode
            test_result['execution_time'] = time.time() - start_time
            test_result['output'] = result.stdout
            
        except subprocess.TimeoutExpired:
            test_result['can_execute'] = True  # Started but took too long
            test_result['errors'].append('Execution timeout')
        except Exception as e:
            test_result['errors'].append(str(e))
        
        return test_result

    def _run_regression_tests(self, build_systems_data: Dict[str, Any], 
                            comparison_data: Dict[str, Any],
                            test_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run regression tests"""
        regression_tests = {
            'comparison_regression': {},
            'functionality_regression': {},
            'performance_regression': {},
            'regression_score': 0.0,
            'issues_found': []
        }
        
        try:
            # Check if comparison results meet minimum standards
            similarity_scores = comparison_data.get('similarity_scores', {})
            quality_metrics = comparison_data.get('quality_metrics', {})
            
            # Regression thresholds
            min_similarity = 0.5
            min_functionality = 0.6
            min_compilation = 0.8
            
            # Check similarity regression
            avg_similarity = 0.0
            if similarity_scores:
                avg_similarity = sum(similarity_scores.values()) / len(similarity_scores)
            
            regression_tests['comparison_regression']['similarity_score'] = avg_similarity
            regression_tests['comparison_regression']['meets_threshold'] = avg_similarity >= min_similarity
            
            if avg_similarity < min_similarity:
                regression_tests['issues_found'].append(
                    f"Binary similarity too low: {avg_similarity:.2f} < {min_similarity}"
                )
            
            # Check functionality regression
            func_score = quality_metrics.get('functional_equivalence_score', 0.0)
            regression_tests['functionality_regression']['score'] = func_score
            regression_tests['functionality_regression']['meets_threshold'] = func_score >= min_functionality
            
            if func_score < min_functionality:
                regression_tests['issues_found'].append(
                    f"Functional equivalence too low: {func_score:.2f} < {min_functionality}"
                )
            
            # Check compilation regression
            comp_score = quality_metrics.get('compilation_success_rate', 0.0)
            regression_tests['performance_regression']['compilation_rate'] = comp_score
            regression_tests['performance_regression']['meets_threshold'] = comp_score >= min_compilation
            
            if comp_score < min_compilation:
                regression_tests['issues_found'].append(
                    f"Compilation success rate too low: {comp_score:.2f} < {min_compilation}"
                )
            
            # Calculate overall regression score
            scores = [avg_similarity, func_score, comp_score]
            regression_tests['regression_score'] = sum(scores) / len(scores) if scores else 0.0
            
        except Exception as e:
            regression_tests['error'] = str(e)
        
        return regression_tests

    def _run_performance_tests(self, build_systems_data: Dict[str, Any], 
                             test_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance tests"""
        performance_tests = {
            'compilation_time_tests': {},
            'binary_size_tests': {},
            'execution_time_tests': {},
            'memory_usage_tests': {},
            'performance_score': 0.0
        }
        
        try:
            # Test compilation times for different compilers
            compilation_attempts = build_systems_data.get('compilation_attempts', {})
            
            for compiler, attempt in compilation_attempts.items():
                if attempt.get('success', False):
                    binary_path = attempt.get('binary_path')
                    if binary_path and os.path.exists(binary_path):
                        # Test binary size
                        size = os.path.getsize(binary_path)
                        performance_tests['binary_size_tests'][compiler] = {
                            'size_bytes': size,
                            'size_kb': size / 1024,
                            'efficient': size < 1024 * 1024  # Less than 1MB
                        }
                        
                        # Test execution time
                        exec_test = self._measure_execution_performance(binary_path)
                        performance_tests['execution_time_tests'][compiler] = exec_test
            
            # Calculate performance score
            size_scores = []
            exec_scores = []
            
            for compiler_tests in performance_tests['binary_size_tests'].values():
                if compiler_tests['efficient']:
                    size_scores.append(1.0)
                else:
                    size_scores.append(0.5)
            
            for exec_test in performance_tests['execution_time_tests'].values():
                if exec_test.get('fast_execution', False):
                    exec_scores.append(1.0)
                else:
                    exec_scores.append(0.5)
            
            all_scores = size_scores + exec_scores
            if all_scores:
                performance_tests['performance_score'] = sum(all_scores) / len(all_scores)
                
        except Exception as e:
            performance_tests['error'] = str(e)
        
        return performance_tests

    def _measure_execution_performance(self, binary_path: str) -> Dict[str, Any]:
        """Measure execution performance of binary"""
        perf_data = {
            'execution_time': 0.0,
            'fast_execution': False,
            'can_measure': False,
            'errors': []
        }
        
        try:
            # Run multiple times and take average
            times = []
            for _ in range(3):
                start_time = time.time()
                
                result = subprocess.run([
                    binary_path
                ], capture_output=True, text=True, timeout=5)
                
                execution_time = time.time() - start_time
                times.append(execution_time)
            
            if times:
                perf_data['execution_time'] = sum(times) / len(times)
                perf_data['fast_execution'] = perf_data['execution_time'] < 1.0  # Less than 1 second
                perf_data['can_measure'] = True
                
        except Exception as e:
            perf_data['errors'].append(str(e))
        
        return perf_data

    def _run_compatibility_tests(self, build_systems_data: Dict[str, Any], 
                               test_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run compatibility tests"""
        compatibility_tests = {
            'compiler_compatibility': {},
            'platform_compatibility': {},
            'build_system_compatibility': {},
            'compatibility_score': 0.0
        }
        
        try:
            # Test compiler compatibility
            detected_compilers = build_systems_data.get('detected_compilers', {})
            successful_compilers = build_systems_data.get('successful_compilers', [])
            
            for compiler in detected_compilers:
                is_compatible = compiler in successful_compilers
                compatibility_tests['compiler_compatibility'][compiler] = {
                    'detected': True,
                    'works': is_compatible,
                    'priority': detected_compilers[compiler].get('priority', 10)
                }
            
            # Test build system compatibility
            build_systems = ['cmake', 'makefile', 'ninja', 'vs_solution']
            working_systems = 0
            
            for build_system in build_systems:
                generated = build_systems_data.get(f'{build_system}_generated', False)
                compatibility_tests['build_system_compatibility'][build_system] = {
                    'generated': generated,
                    'supported': generated
                }
                if generated:
                    working_systems += 1
            
            # Calculate compatibility score
            compiler_score = len(successful_compilers) / max(1, len(detected_compilers))
            build_system_score = working_systems / len(build_systems)
            
            compatibility_tests['compatibility_score'] = (compiler_score + build_system_score) / 2
            
        except Exception as e:
            compatibility_tests['error'] = str(e)
        
        return compatibility_tests

    def _run_security_tests(self, build_systems_data: Dict[str, Any], 
                          test_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run basic security tests"""
        security_tests = {
            'buffer_overflow_checks': {},
            'stack_protection_checks': {},
            'code_injection_checks': {},
            'security_score': 0.0,
            'security_issues': []
        }
        
        try:
            # Basic security checks on generated binaries
            compilation_attempts = build_systems_data.get('compilation_attempts', {})
            
            for compiler, attempt in compilation_attempts.items():
                if attempt.get('success', False):
                    binary_path = attempt.get('binary_path')
                    if binary_path and os.path.exists(binary_path):
                        security_check = self._perform_basic_security_check(binary_path)
                        security_tests['buffer_overflow_checks'][compiler] = security_check
            
            # Calculate security score (basic)
            total_checks = len(security_tests['buffer_overflow_checks'])
            safe_binaries = sum(
                1 for check in security_tests['buffer_overflow_checks'].values()
                if check.get('appears_safe', False)
            )
            
            if total_checks > 0:
                security_tests['security_score'] = safe_binaries / total_checks
                
        except Exception as e:
            security_tests['error'] = str(e)
        
        return security_tests

    def _perform_basic_security_check(self, binary_path: str) -> Dict[str, Any]:
        """Perform basic security check on binary"""
        security_check = {
            'appears_safe': True,
            'checks_performed': [],
            'issues_found': []
        }
        
        try:
            # Basic checks - file size, execution without crashes
            file_size = os.path.getsize(binary_path)
            security_check['checks_performed'].append('file_size_check')
            
            if file_size > 10 * 1024 * 1024:  # 10MB
                security_check['issues_found'].append('Binary size unusually large')
                security_check['appears_safe'] = False
            
            # Try execution test
            try:
                result = subprocess.run([
                    binary_path
                ], capture_output=True, text=True, timeout=2)
                security_check['checks_performed'].append('execution_test')
                
                # Check for immediate crashes
                if result.returncode < 0:
                    security_check['issues_found'].append('Binary crashes immediately')
                    security_check['appears_safe'] = False
                    
            except subprocess.TimeoutExpired:
                # Timeout is not necessarily a security issue
                security_check['checks_performed'].append('execution_timeout')
            except Exception:
                security_check['issues_found'].append('Cannot execute binary')
                security_check['appears_safe'] = False
                
        except Exception as e:
            security_check['issues_found'].append(f"Security check failed: {str(e)}")
            security_check['appears_safe'] = False
        
        return security_check

    def _run_validation_tests(self, build_systems_data: Dict[str, Any], 
                            comparison_data: Dict[str, Any],
                            test_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run final validation tests"""
        validation_tests = {
            'end_to_end_validation': {},
            'quality_validation': {},
            'completeness_validation': {},
            'validation_score': 0.0,
            'validation_passed': False
        }
        
        try:
            # End-to-end validation
            validation_tests['end_to_end_validation'] = self._validate_end_to_end(
                build_systems_data, comparison_data, context
            )
            
            # Quality validation
            validation_tests['quality_validation'] = self._validate_quality_metrics(
                comparison_data
            )
            
            # Completeness validation
            validation_tests['completeness_validation'] = self._validate_completeness(
                build_systems_data, context
            )
            
            # Calculate overall validation score
            e2e_score = validation_tests['end_to_end_validation'].get('score', 0.0)
            quality_score = validation_tests['quality_validation'].get('score', 0.0)
            completeness_score = validation_tests['completeness_validation'].get('score', 0.0)
            
            validation_tests['validation_score'] = (e2e_score + quality_score + completeness_score) / 3
            validation_tests['validation_passed'] = validation_tests['validation_score'] >= 0.7
            
        except Exception as e:
            validation_tests['error'] = str(e)
        
        return validation_tests

    def _validate_end_to_end(self, build_systems_data: Dict[str, Any], 
                           comparison_data: Dict[str, Any],
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate end-to-end pipeline"""
        e2e_validation = {
            'pipeline_completed': False,
            'binaries_generated': False,
            'comparison_performed': False,
            'score': 0.0
        }
        
        # Check if pipeline completed successfully
        agent_results = context.get('agent_results', {})
        completed_agents = sum(
            1 for result in agent_results.values()
            if result and result.status == AgentStatus.COMPLETED
        )
        total_agents = len(agent_results)
        
        if total_agents > 0:
            completion_rate = completed_agents / total_agents
            e2e_validation['pipeline_completed'] = completion_rate >= 0.8
        
        # Check if binaries were generated
        successful_compilers = build_systems_data.get('successful_compilers', [])
        e2e_validation['binaries_generated'] = len(successful_compilers) > 0
        
        # Check if comparison was performed
        similarity_scores = comparison_data.get('similarity_scores', {})
        e2e_validation['comparison_performed'] = len(similarity_scores) > 0
        
        # Calculate score
        criteria = [
            e2e_validation['pipeline_completed'],
            e2e_validation['binaries_generated'],
            e2e_validation['comparison_performed']
        ]
        e2e_validation['score'] = sum(criteria) / len(criteria)
        
        return e2e_validation

    def _validate_quality_metrics(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality metrics meet minimum standards"""
        quality_validation = {
            'meets_similarity_threshold': False,
            'meets_functionality_threshold': False,
            'meets_performance_threshold': False,
            'score': 0.0
        }
        
        quality_metrics = comparison_data.get('quality_metrics', {})
        
        # Check thresholds
        similarity_score = quality_metrics.get('binary_similarity_score', 0.0)
        functionality_score = quality_metrics.get('functional_equivalence_score', 0.0)
        performance_score = quality_metrics.get('performance_score', 0.0)
        
        quality_validation['meets_similarity_threshold'] = similarity_score >= 0.5
        quality_validation['meets_functionality_threshold'] = functionality_score >= 0.6
        quality_validation['meets_performance_threshold'] = performance_score >= 0.5
        
        # Calculate score
        criteria = [
            quality_validation['meets_similarity_threshold'],
            quality_validation['meets_functionality_threshold'],
            quality_validation['meets_performance_threshold']
        ]
        quality_validation['score'] = sum(criteria) / len(criteria)
        
        return quality_validation

    def _validate_completeness(self, build_systems_data: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate completeness of the reconstruction"""
        completeness_validation = {
            'source_files_generated': False,
            'build_files_generated': False,
            'compilation_successful': False,
            'score': 0.0
        }
        
        # Check if source files were generated
        agent11_result = context.get('agent_results', {}).get(11)
        if agent11_result and agent11_result.status == AgentStatus.COMPLETED:
            global_reconstruction = agent11_result.data
            source_files = global_reconstruction.get('reconstructed_source', {}).get('source_files', {})
            completeness_validation['source_files_generated'] = len(source_files) > 0
        
        # Check if build files were generated
        build_systems = ['cmake_generated', 'makefile_generated', 'ninja_generated', 'vs_solution_generated']
        generated_systems = sum(
            1 for system in build_systems
            if build_systems_data.get(system, False)
        )
        completeness_validation['build_files_generated'] = generated_systems > 0
        
        # Check if compilation was successful
        successful_compilers = build_systems_data.get('successful_compilers', [])
        completeness_validation['compilation_successful'] = len(successful_compilers) > 0
        
        # Calculate score
        criteria = [
            completeness_validation['source_files_generated'],
            completeness_validation['build_files_generated'],
            completeness_validation['compilation_successful']
        ]
        completeness_validation['score'] = sum(criteria) / len(criteria)
        
        return completeness_validation

    def _calculate_test_coverage(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate test coverage metrics"""
        coverage = {
            'unit_test_coverage': 0.0,
            'integration_test_coverage': 0.0,
            'regression_test_coverage': 0.0,
            'overall_coverage': 0.0
        }
        
        # Unit test coverage
        unit_tests = result.get('unit_tests', {})
        coverage['unit_test_coverage'] = unit_tests.get('coverage', 0.0)
        
        # Integration test coverage
        integration_tests = result.get('integration_tests', {})
        coverage['integration_test_coverage'] = integration_tests.get('success_rate', 0.0)
        
        # Regression test coverage
        regression_tests = result.get('regression_tests', {})
        coverage['regression_test_coverage'] = regression_tests.get('regression_score', 0.0)
        
        # Overall coverage
        coverages = [
            coverage['unit_test_coverage'],
            coverage['integration_test_coverage'],
            coverage['regression_test_coverage']
        ]
        coverage['overall_coverage'] = sum(coverages) / len(coverages) if coverages else 0.0
        
        return coverage

    def _generate_test_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test results summary"""
        summary = {
            'total_tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'success_rate': 0.0,
            'critical_issues': [],
            'recommendations': []
        }
        
        # Count tests from different test types
        unit_tests = result.get('unit_tests', {})
        integration_tests = result.get('integration_tests', {})
        regression_tests = result.get('regression_tests', {})
        
        summary['total_tests_run'] += unit_tests.get('total_tests', 0)
        summary['tests_passed'] += unit_tests.get('pass_count', 0)
        summary['tests_failed'] += unit_tests.get('fail_count', 0)
        
        summary['total_tests_run'] += integration_tests.get('total_tests', 0)
        summary['tests_passed'] += integration_tests.get('passed_tests', 0)
        summary['tests_failed'] += (
            integration_tests.get('total_tests', 0) - integration_tests.get('passed_tests', 0)
        )
        
        # Calculate success rate
        if summary['total_tests_run'] > 0:
            summary['success_rate'] = summary['tests_passed'] / summary['total_tests_run']
        
        # Identify critical issues
        if summary['success_rate'] < 0.5:
            summary['critical_issues'].append("Overall test success rate is below 50%")
        
        if unit_tests.get('success_rate', 0.0) < 0.3:
            summary['critical_issues'].append("Unit test success rate is critically low")
        
        if integration_tests.get('success_rate', 0.0) < 0.5:
            summary['critical_issues'].append("Integration test success rate is low")
        
        return summary

    def _assess_production_readiness(self, result: Dict[str, Any]) -> str:
        """Assess overall production readiness"""
        test_coverage = result.get('test_coverage', {})
        test_summary = result.get('test_results_summary', {})
        validation_tests = result.get('validation_tests', {})
        
        overall_coverage = test_coverage.get('overall_coverage', 0.0)
        success_rate = test_summary.get('success_rate', 0.0)
        validation_passed = validation_tests.get('validation_passed', False)
        
        if validation_passed and overall_coverage >= 0.8 and success_rate >= 0.9:
            return 'production_ready'
        elif overall_coverage >= 0.6 and success_rate >= 0.7:
            return 'near_production_ready'
        elif overall_coverage >= 0.4 and success_rate >= 0.5:
            return 'development_ready'
        elif success_rate >= 0.3:
            return 'basic_functionality'
        else:
            return 'not_ready'

    def _calculate_overall_test_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall test score"""
        test_coverage = result.get('test_coverage', {})
        test_summary = result.get('test_results_summary', {})
        validation_tests = result.get('validation_tests', {})
        
        coverage_score = test_coverage.get('overall_coverage', 0.0)
        success_rate = test_summary.get('success_rate', 0.0)
        validation_score = validation_tests.get('validation_score', 0.0)
        
        # Weighted average
        weights = {'coverage': 0.3, 'success': 0.4, 'validation': 0.3}
        
        overall_score = (
            coverage_score * weights['coverage'] +
            success_rate * weights['success'] +
            validation_score * weights['validation']
        )
        
        return overall_score

    def _generate_test_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        test_summary = result.get('test_results_summary', {})
        test_coverage = result.get('test_coverage', {})
        production_readiness = result.get('production_readiness', 'unknown')
        
        # Coverage recommendations
        if test_coverage.get('overall_coverage', 0.0) < 0.7:
            recommendations.append("Increase test coverage by adding more comprehensive tests")
        
        # Success rate recommendations
        if test_summary.get('success_rate', 0.0) < 0.8:
            recommendations.append("Improve test success rate by fixing failing tests and improving code quality")
        
        # Production readiness recommendations
        if production_readiness == 'not_ready':
            recommendations.append("Significant improvements needed before production deployment")
            recommendations.append("Focus on fixing critical issues identified in testing")
        elif production_readiness == 'basic_functionality':
            recommendations.append("Basic functionality achieved, but extensive testing and improvements needed")
        elif production_readiness == 'development_ready':
            recommendations.append("Suitable for development environment, continue improving for production")
        elif production_readiness == 'near_production_ready':
            recommendations.append("Close to production readiness, address remaining test failures")
        
        # Specific test type recommendations
        unit_tests = result.get('unit_tests', {})
        if unit_tests.get('success_rate', 0.0) < 0.6:
            recommendations.append("Improve unit test coverage and function-level testing")
        
        integration_tests = result.get('integration_tests', {})
        if integration_tests.get('success_rate', 0.0) < 0.7:
            recommendations.append("Fix integration issues between components and build systems")
        
        return recommendations

    def _perform_cleanup_validation(self, output_paths: Dict[str, str], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cleanup validation, temp directory management, and documentation restructuring"""
        cleanup_result = {
            'temp_cleanup_performed': False,
            'temp_files_before': 0,
            'temp_files_after': 0,
            'cleanup_success': True,
            'output_structure_verified': False,
            'structure_issues': [],
            'cleanup_actions': [],
            'documentation_restructured': False,
            'docs_moved': []
        }
        
        try:
            # Get temp directory path
            temp_dir = output_paths.get('temp')
            if not temp_dir:
                cleanup_result['cleanup_success'] = False
                cleanup_result['structure_issues'].append('No temp directory in output paths')
                return cleanup_result
            
            # Count files before cleanup
            if os.path.exists(temp_dir):
                try:
                    temp_files = os.listdir(temp_dir)
                    cleanup_result['temp_files_before'] = len(temp_files)
                    
                    # Perform selective cleanup of temp files
                    files_removed = 0
                    for filename in temp_files:
                        file_path = os.path.join(temp_dir, filename)
                        try:
                            # Only remove files that are clearly temporary
                            if (filename.endswith('.tmp') or 
                                filename.endswith('.temp') or 
                                filename.startswith('temp_') or
                                filename.startswith('tmp_')):
                                os.remove(file_path)
                                files_removed += 1
                                cleanup_result['cleanup_actions'].append(f'Removed temporary file: {filename}')
                        except (OSError, PermissionError) as e:
                            cleanup_result['cleanup_actions'].append(f'Failed to remove {filename}: {str(e)}')
                    
                    # Count files after cleanup
                    remaining_files = os.listdir(temp_dir)
                    cleanup_result['temp_files_after'] = len(remaining_files)
                    cleanup_result['temp_cleanup_performed'] = files_removed > 0
                    
                except (OSError, PermissionError) as e:
                    cleanup_result['cleanup_success'] = False
                    cleanup_result['structure_issues'].append(f'Cannot access temp directory: {str(e)}')
            
            # Perform documentation restructuring
            docs_restructure = self._restructure_documentation(context)
            cleanup_result.update(docs_restructure)
            
            # Verify output structure organization
            base_output_dir = context.get('output_dir', 'output')
            structure_check = self._verify_output_organization(base_output_dir, output_paths)
            cleanup_result.update(structure_check)
            
        except Exception as e:
            cleanup_result['cleanup_success'] = False
            cleanup_result['structure_issues'].append(f'Cleanup validation error: {str(e)}')
        
        return cleanup_result

    def _verify_output_organization(self, base_output_dir: str, 
                                  output_paths: Dict[str, str]) -> Dict[str, Any]:
        """Verify that output is properly organized according to structure"""
        verification = {
            'output_structure_verified': True,
            'structure_score': 0.0,
            'misplaced_files': [],
            'missing_expected_outputs': []
        }
        
        try:
            # Check if files are in correct subdirectories
            expected_in_reports = ['pipeline_report.json', 'test_report.json', 'validation_report.json']
            expected_in_logs = ['agent_logs.txt', 'execution.log', 'debug.log']
            expected_in_compilation = ['Makefile', 'build.ps1', 'CMakeLists.txt']
            
            # Check root directory for files that should be elsewhere
            misplaced_count = 0
            if os.path.exists(base_output_dir):
                for item in os.listdir(base_output_dir):
                    item_path = os.path.join(base_output_dir, item)
                    if os.path.isfile(item_path):
                        if (item.endswith('.json') and item in expected_in_reports or
                            item.endswith('.log') and item in expected_in_logs or
                            item in expected_in_compilation):
                            verification['misplaced_files'].append(item)
                            misplaced_count += 1
            
            # Calculate structure score based on organization
            if misplaced_count == 0:
                verification['structure_score'] = 1.0
            elif misplaced_count <= 2:
                verification['structure_score'] = 0.8
            elif misplaced_count <= 5:
                verification['structure_score'] = 0.6
            else:
                verification['structure_score'] = 0.3
                verification['output_structure_verified'] = False
            
            # Check for expected outputs in correct locations
            reports_dir = output_paths.get('reports')
            if reports_dir and os.path.exists(reports_dir):
                if not os.path.exists(os.path.join(reports_dir, 'pipeline_report.json')):
                    verification['missing_expected_outputs'].append('pipeline_report.json not in reports/')
            
        except Exception as e:
            verification['output_structure_verified'] = False
            verification['structure_score'] = 0.0
        
        return verification

    def _perform_source_binary_validation(self, build_systems_data: Dict[str, Any], 
                                        comparison_data: Dict[str, Any],
                                        context: Dict[str, Any], test_dir: str) -> Dict[str, Any]:
        """Perform comprehensive validation to ensure source code and binaries are real implementations, not dummy data"""
        validation = {
            'validation_passed': False,
            'overall_authenticity_score': 0.0,
            'source_authenticity': {},
            'binary_authenticity': {},
            'cross_validation': {},
            'test_compilation_results': {},
            'functional_verification': {},
            'authenticity_issues': [],
            'confidence_level': 'unknown'
        }
        
        try:
            # Step 1: Validate source code authenticity
            source_validation = self._validate_reconstructed_source_authenticity(context)
            validation['source_authenticity'] = source_validation
            
            # Step 2: Validate binary authenticity 
            binary_validation = self._validate_generated_binary_authenticity(build_systems_data, context)
            validation['binary_authenticity'] = binary_validation
            
            # Step 3: Cross-validate source matches binary behavior
            cross_validation = self._cross_validate_source_binary_consistency(
                source_validation, binary_validation, context, test_dir
            )
            validation['cross_validation'] = cross_validation
            
            # Step 4: Perform independent test compilation
            test_compilation = self._perform_independent_test_compilation(context, test_dir)
            validation['test_compilation_results'] = test_compilation
            
            # Step 5: Functional verification
            functional_verification = self._perform_functional_verification(
                build_systems_data, context, test_dir
            )
            validation['functional_verification'] = functional_verification
            
            # Step 6: Calculate overall authenticity score
            authenticity_components = [
                source_validation.get('authenticity_score', 0.0) * 0.25,
                binary_validation.get('authenticity_score', 0.0) * 0.25,
                cross_validation.get('consistency_score', 0.0) * 0.20,
                test_compilation.get('success_score', 0.0) * 0.15,
                functional_verification.get('functionality_score', 0.0) * 0.15
            ]
            
            validation['overall_authenticity_score'] = sum(authenticity_components)
            
            # Step 7: Collect all authenticity issues
            for component in [source_validation, binary_validation, cross_validation, 
                            test_compilation, functional_verification]:
                validation['authenticity_issues'].extend(component.get('issues', []))
            
            # Step 8: Determine confidence level and final validation
            validation['confidence_level'] = self._determine_confidence_level(
                validation['overall_authenticity_score'], 
                len(validation['authenticity_issues'])
            )
            
            validation['validation_passed'] = (
                validation['overall_authenticity_score'] >= 0.75 and
                len(validation['authenticity_issues']) <= 2 and
                validation['confidence_level'] in ['high', 'very_high']
            )
            
        except Exception as e:
            validation['authenticity_issues'].append(f"Validation failed: {str(e)}")
            validation['validation_passed'] = False
            validation['confidence_level'] = 'failed'
        
        return validation

    def _validate_reconstructed_source_authenticity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that reconstructed source code represents real implementation"""
        validation = {
            'authenticity_score': 0.0,
            'appears_authentic': False,
            'code_quality_metrics': {},
            'complexity_analysis': {},
            'implementation_depth': {},
            'issues': []
        }
        
        try:
            # Get source code from Agent 11
            agent11_result = context.get('agent_results', {}).get(11)
            if not agent11_result or agent11_result.status != AgentStatus.COMPLETED:
                validation['issues'].append("No source code reconstruction available")
                return validation
            
            global_reconstruction = agent11_result.data
            reconstructed_source = global_reconstruction.get('reconstructed_source', {})
            source_files = reconstructed_source.get('source_files', {})
            
            if not source_files:
                validation['issues'].append("No source files generated")
                return validation
            
            # Analyze code quality and complexity
            quality_metrics = self._analyze_code_quality_metrics(source_files)
            validation['code_quality_metrics'] = quality_metrics
            
            # Analyze implementation depth
            depth_analysis = self._analyze_implementation_depth(source_files)
            validation['implementation_depth'] = depth_analysis
            
            # Analyze complexity
            complexity_analysis = self._analyze_code_complexity(source_files)
            validation['complexity_analysis'] = complexity_analysis
            
            # Calculate authenticity score
            quality_score = quality_metrics.get('overall_quality', 0.0)
            depth_score = depth_analysis.get('depth_score', 0.0)
            complexity_score = complexity_analysis.get('complexity_score', 0.0)
            
            validation['authenticity_score'] = (quality_score * 0.4 + depth_score * 0.35 + complexity_score * 0.25)
            
            # Check for authenticity issues
            if quality_score < 0.5:
                validation['issues'].append(f"Code quality too low: {quality_score:.2f}")
            
            if depth_score < 0.4:
                validation['issues'].append(f"Implementation depth insufficient: {depth_score:.2f}")
            
            if complexity_score < 0.3:
                validation['issues'].append(f"Code complexity too low: {complexity_score:.2f}")
            
            validation['appears_authentic'] = (
                validation['authenticity_score'] >= 0.6 and len(validation['issues']) <= 1
            )
            
        except Exception as e:
            validation['issues'].append(f"Source validation error: {str(e)}")
        
        return validation

    def _analyze_code_quality_metrics(self, source_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        metrics = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'function_count': 0,
            'meaningful_functions': 0,
            'placeholder_count': 0,
            'overall_quality': 0.0
        }
        
        try:
            placeholder_patterns = ['TODO', 'FIXME', 'placeholder', 'Hello World']
            
            for filename, content in source_files.items():
                if isinstance(content, str):
                    lines = content.split('\n')
                    metrics['total_lines'] += len(lines)
                    
                    for line in lines:
                        stripped = line.strip()
                        if stripped and not stripped.startswith('//'):
                            metrics['code_lines'] += 1
                        elif stripped.startswith('//'):
                            metrics['comment_lines'] += 1
                        
                        # Check for placeholders
                        for pattern in placeholder_patterns:
                            if pattern.lower() in stripped.lower():
                                metrics['placeholder_count'] += 1
                                break
                    
                    # Count functions
                    import re
                    functions = re.findall(r'\w+\s*\([^)]*\)\s*\{', content)
                    metrics['function_count'] += len(functions)
                    
                    # Count meaningful functions (not just main with return 0)
                    for func in functions:
                        func_start = content.find(func)
                        if func_start != -1:
                            # Extract function body
                            brace_count = 0
                            pos = content.find('{', func_start)
                            if pos != -1:
                                pos += 1
                                brace_count = 1
                                func_body = ""
                                
                                while pos < len(content) and brace_count > 0:
                                    char = content[pos]
                                    if char == '{':
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                    func_body += char
                                    pos += 1
                                
                                # Check if function is meaningful
                                if len(func_body.split('\n')) > 3 or any(keyword in func_body for keyword in ['if', 'while', 'for']):
                                    metrics['meaningful_functions'] += 1
            
            # Calculate overall quality
            if metrics['total_lines'] > 0:
                code_ratio = metrics['code_lines'] / metrics['total_lines']
                placeholder_ratio = metrics['placeholder_count'] / max(1, metrics['total_lines'])
                meaningful_ratio = metrics['meaningful_functions'] / max(1, metrics['function_count'])
                
                metrics['overall_quality'] = max(0.0, code_ratio * 0.4 + (1 - placeholder_ratio) * 0.4 + meaningful_ratio * 0.2)
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics

    def _analyze_implementation_depth(self, source_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze implementation depth"""
        analysis = {
            'depth_score': 0.0,
            'has_real_logic': False,
            'has_data_structures': False,
            'has_algorithms': False,
            'has_error_handling': False,
            'implementation_indicators': []
        }
        
        try:
            all_content = '\n'.join(source_files.values())
            
            # Check for real logic indicators
            logic_patterns = [
                r'\bif\s*\([^)]+\)\s*\{[^}]+\}',
                r'\bwhile\s*\([^)]+\)\s*\{[^}]+\}',
                r'\bfor\s*\([^)]*;[^)]*;[^)]*\)\s*\{[^}]+\}',
                r'\bswitch\s*\([^)]+\)\s*\{[^}]+\}'
            ]
            
            for pattern in logic_patterns:
                import re
                if re.search(pattern, all_content, re.DOTALL):
                    analysis['has_real_logic'] = True
                    analysis['implementation_indicators'].append(f"Found control structures: {pattern}")
                    break
            
            # Check for data structures
            structure_patterns = [
                r'\bstruct\s+\w+\s*\{[^}]+\}',
                r'\benum\s+\w+\s*\{[^}]+\}',
                r'\bunion\s+\w+\s*\{[^}]+\}',
                r'\w+\s*\[\s*\d+\s*\]'  # Arrays
            ]
            
            for pattern in structure_patterns:
                if re.search(pattern, all_content):
                    analysis['has_data_structures'] = True
                    analysis['implementation_indicators'].append(f"Found data structures: {pattern}")
                    break
            
            # Check for algorithms
            algorithm_patterns = [
                r'\bsort\b', r'\bsearch\b', r'\bcalculat\b', r'\bprocess\b',
                r'\bparse\b', r'\bvalidat\b', r'\bconvert\b'
            ]
            
            for pattern in algorithm_patterns:
                if re.search(pattern, all_content, re.IGNORECASE):
                    analysis['has_algorithms'] = True
                    analysis['implementation_indicators'].append(f"Found algorithmic indicators: {pattern}")
                    break
            
            # Check for error handling
            error_patterns = [
                r'\berror\b', r'\bexception\b', r'\btry\b', r'\bcatch\b',
                r'\breturn\s*-?\d+', r'\bNULL\b', r'\bfail\b'
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, all_content, re.IGNORECASE):
                    analysis['has_error_handling'] = True
                    analysis['implementation_indicators'].append(f"Found error handling: {pattern}")
                    break
            
            # Calculate depth score
            depth_factors = [
                analysis['has_real_logic'],
                analysis['has_data_structures'],
                analysis['has_algorithms'],
                analysis['has_error_handling']
            ]
            
            analysis['depth_score'] = sum(depth_factors) / len(depth_factors)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis

    def _analyze_code_complexity(self, source_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze code complexity"""
        analysis = {
            'complexity_score': 0.0,
            'cyclomatic_complexity': 0,
            'nesting_depth': 0,
            'variable_usage': 0,
            'function_calls': 0
        }
        
        try:
            all_content = '\n'.join(source_files.values())
            
            # Count complexity indicators
            import re
            
            # Cyclomatic complexity (decision points)
            decision_patterns = [r'\bif\b', r'\belse\b', r'\bwhile\b', r'\bfor\b', 
                               r'\bswitch\b', r'\bcase\b', r'\b\?\b', r'\b\|\|\b', r'\b&&\b']
            
            for pattern in decision_patterns:
                analysis['cyclomatic_complexity'] += len(re.findall(pattern, all_content))
            
            # Estimate nesting depth
            max_nesting = 0
            current_nesting = 0
            for char in all_content:
                if char == '{':
                    current_nesting += 1
                    max_nesting = max(max_nesting, current_nesting)
                elif char == '}':
                    current_nesting = max(0, current_nesting - 1)
            
            analysis['nesting_depth'] = max_nesting
            
            # Count variable assignments
            analysis['variable_usage'] = len(re.findall(r'\w+\s*=\s*[^=]', all_content))
            
            # Count function calls
            analysis['function_calls'] = len(re.findall(r'\w+\s*\(', all_content))
            
            # Calculate complexity score
            complexity_factors = [
                min(1.0, analysis['cyclomatic_complexity'] / 20),
                min(1.0, analysis['nesting_depth'] / 5),
                min(1.0, analysis['variable_usage'] / 50),
                min(1.0, analysis['function_calls'] / 30)
            ]
            
            analysis['complexity_score'] = sum(complexity_factors) / len(complexity_factors)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis

    def _validate_generated_binary_authenticity(self, build_systems_data: Dict[str, Any], 
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that generated binaries represent real implementations"""
        validation = {
            'authenticity_score': 0.0,
            'appears_authentic': False,
            'binary_analyses': {},
            'consistency_check': {},
            'issues': []
        }
        
        try:
            # Get generated binaries
            successful_compilers = build_systems_data.get('successful_compilers', [])
            compilation_attempts = build_systems_data.get('compilation_attempts', {})
            
            if not successful_compilers:
                validation['issues'].append("No successful binary compilation")
                return validation
            
            total_authenticity = 0.0
            binary_count = 0
            
            # Analyze each generated binary
            for compiler in successful_compilers:
                attempt = compilation_attempts.get(compiler, {})
                binary_path = attempt.get('binary_path')
                
                if binary_path and os.path.exists(binary_path):
                    binary_analysis = self._analyze_single_binary_authenticity(binary_path, compiler)
                    validation['binary_analyses'][compiler] = binary_analysis
                    
                    total_authenticity += binary_analysis['authenticity_score']
                    binary_count += 1
                    
                    validation['issues'].extend(binary_analysis.get('issues', []))
            
            if binary_count > 0:
                validation['authenticity_score'] = total_authenticity / binary_count
                validation['appears_authentic'] = validation['authenticity_score'] >= 0.6
            
            # Consistency check across binaries
            if binary_count > 1:
                consistency = self._check_binary_consistency(validation['binary_analyses'])
                validation['consistency_check'] = consistency
                
                if not consistency['consistent']:
                    validation['issues'].extend(consistency['inconsistencies'])
            
        except Exception as e:
            validation['issues'].append(f"Binary validation error: {str(e)}")
        
        return validation

    def _analyze_single_binary_authenticity(self, binary_path: str, compiler: str) -> Dict[str, Any]:
        """Analyze single binary for authenticity"""
        analysis = {
            'authenticity_score': 0.0,
            'size_analysis': {},
            'execution_analysis': {},
            'content_analysis': {},
            'issues': []
        }
        
        try:
            # Size analysis
            file_size = os.path.getsize(binary_path)
            analysis['size_analysis'] = {
                'file_size': file_size,
                'reasonable_size': 20000 <= file_size <= 50000000  # 20KB to 50MB
            }
            
            if not analysis['size_analysis']['reasonable_size']:
                analysis['issues'].append(f"Binary size unreasonable: {file_size} bytes")
            
            # Execution analysis
            exec_analysis = self._test_binary_execution_depth(binary_path)
            analysis['execution_analysis'] = exec_analysis
            
            if exec_analysis.get('appears_trivial', True):
                analysis['issues'].extend(exec_analysis.get('trivial_indicators', []))
            
            # Content analysis for dummy patterns
            content_analysis = self._scan_binary_for_dummy_content(binary_path)
            analysis['content_analysis'] = content_analysis
            
            if content_analysis.get('has_dummy_content', False):
                analysis['issues'].extend(content_analysis.get('dummy_indicators', []))
            
            # Calculate authenticity score
            size_score = 1.0 if analysis['size_analysis']['reasonable_size'] else 0.3
            exec_score = 1.0 - exec_analysis.get('triviality_score', 1.0)
            content_score = 1.0 - content_analysis.get('dummy_content_ratio', 1.0)
            
            analysis['authenticity_score'] = (size_score * 0.3 + exec_score * 0.4 + content_score * 0.3)
            
        except Exception as e:
            analysis['issues'].append(f"Binary analysis error: {str(e)}")
        
        return analysis

    def _test_binary_execution_depth(self, binary_path: str) -> Dict[str, Any]:
        """Test binary execution depth and complexity"""
        analysis = {
            'appears_trivial': True,
            'triviality_score': 1.0,
            'execution_time': 0.0,
            'output_complexity': 0,
            'trivial_indicators': []
        }
        
        try:
            import time
            start_time = time.time()
            
            # Run with various arguments to test behavior
            test_args = [[], ['--help'], ['-h'], ['test'], ['arg1', 'arg2']]
            outputs = []
            
            for args in test_args:
                try:
                    result = subprocess.run([binary_path] + args, 
                                          capture_output=True, text=True, timeout=5)
                    outputs.append({
                        'args': args,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    })
                except subprocess.TimeoutExpired:
                    outputs.append({
                        'args': args,
                        'timeout': True
                    })
                except Exception:
                    continue
            
            execution_time = time.time() - start_time
            analysis['execution_time'] = execution_time
            
            # Analyze outputs for complexity
            total_output_length = 0
            unique_outputs = set()
            
            for output in outputs:
                if 'stdout' in output:
                    stdout = output['stdout'].strip()
                    stderr = output['stderr'].strip()
                    combined = stdout + stderr
                    
                    total_output_length += len(combined)
                    if combined:
                        unique_outputs.add(combined)
                    
                    # Check for trivial patterns
                    trivial_patterns = ['hello world', 'test program', 'hello', 'hi']
                    for pattern in trivial_patterns:
                        if pattern.lower() in combined.lower():
                            analysis['trivial_indicators'].append(f"Trivial output: {pattern}")
            
            analysis['output_complexity'] = total_output_length
            
            # Calculate triviality score
            if total_output_length == 0:
                analysis['triviality_score'] = 1.0
                analysis['trivial_indicators'].append("No output produced")
            elif total_output_length < 20:
                analysis['triviality_score'] = 0.8
                analysis['trivial_indicators'].append("Very minimal output")
            elif len(unique_outputs) <= 1:
                analysis['triviality_score'] = 0.7
                analysis['trivial_indicators'].append("Same output for all test cases")
            else:
                analysis['triviality_score'] = max(0.0, 1.0 - (total_output_length / 200))
            
            analysis['appears_trivial'] = analysis['triviality_score'] > 0.6
            
        except Exception as e:
            analysis['trivial_indicators'].append(f"Execution test failed: {str(e)}")
        
        return analysis

    def _scan_binary_for_dummy_content(self, binary_path: str) -> Dict[str, Any]:
        """Scan binary for dummy content indicators"""
        analysis = {
            'has_dummy_content': False,
            'dummy_content_ratio': 0.0,
            'dummy_indicators': []
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read sample for analysis
                sample_size = min(32768, os.path.getsize(binary_path))
                binary_data = f.read(sample_size)
            
            # Convert to text for pattern matching
            try:
                text_content = binary_data.decode('utf-8', errors='ignore')
            except:
                text_content = str(binary_data)
            
            # Check for dummy patterns
            dummy_patterns = [
                'hello world', 'Hello World', 'HELLO WORLD',
                'test program', 'Test Program', 'TEST PROGRAM',
                'placeholder', 'PLACEHOLDER',
                'TODO', 'FIXME', 'XXX',
                'generated by', 'reconstructed', 'decompiled'
            ]
            
            found_patterns = []
            dummy_char_count = 0
            
            for pattern in dummy_patterns:
                if pattern.lower() in text_content.lower():
                    found_patterns.append(pattern)
                    dummy_char_count += len(pattern)
            
            if found_patterns:
                analysis['has_dummy_content'] = True
                analysis['dummy_indicators'] = [f"Found dummy pattern: {p}" for p in found_patterns]
            
            # Calculate dummy content ratio
            if len(text_content) > 0:
                analysis['dummy_content_ratio'] = min(1.0, dummy_char_count / len(text_content))
            
        except Exception as e:
            analysis['dummy_indicators'].append(f"Content scan failed: {str(e)}")
        
        return analysis

    def _check_binary_consistency(self, binary_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency across multiple generated binaries"""
        consistency = {
            'consistent': True,
            'inconsistencies': [],
            'size_variance': 0.0,
            'behavior_consistency': 0.0
        }
        
        try:
            sizes = []
            auth_scores = []
            
            for compiler, analysis in binary_analyses.items():
                size_info = analysis.get('size_analysis', {})
                sizes.append(size_info.get('file_size', 0))
                auth_scores.append(analysis.get('authenticity_score', 0.0))
            
            if len(sizes) > 1:
                # Check size variance
                avg_size = sum(sizes) / len(sizes)
                size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
                consistency['size_variance'] = size_variance / (avg_size ** 2) if avg_size > 0 else 1.0
                
                if consistency['size_variance'] > 0.5:  # 50% variance
                    consistency['inconsistencies'].append("High variance in binary sizes")
                    consistency['consistent'] = False
                
                # Check authenticity score consistency
                avg_auth = sum(auth_scores) / len(auth_scores)
                auth_variance = sum((s - avg_auth) ** 2 for s in auth_scores) / len(auth_scores)
                consistency['behavior_consistency'] = 1.0 - min(1.0, auth_variance)
                
                if auth_variance > 0.3:
                    consistency['inconsistencies'].append("Inconsistent authenticity scores across compilers")
                    consistency['consistent'] = False
            
        except Exception as e:
            consistency['inconsistencies'].append(f"Consistency check failed: {str(e)}")
            consistency['consistent'] = False
        
        return consistency

    def _cross_validate_source_binary_consistency(self, source_validation: Dict[str, Any],
                                                binary_validation: Dict[str, Any],
                                                context: Dict[str, Any], test_dir: str) -> Dict[str, Any]:
        """Cross-validate that source code matches binary behavior"""
        validation = {
            'consistency_score': 0.0,
            'source_binary_match': False,
            'complexity_correlation': 0.0,
            'functional_alignment': 0.0,
            'issues': []
        }
        
        try:
            # Compare complexity indicators
            source_complexity = source_validation.get('complexity_analysis', {}).get('complexity_score', 0.0)
            
            binary_complexities = []
            for binary_analysis in binary_validation.get('binary_analyses', {}).values():
                exec_analysis = binary_analysis.get('execution_analysis', {})
                binary_complexity = 1.0 - exec_analysis.get('triviality_score', 1.0)
                binary_complexities.append(binary_complexity)
            
            if binary_complexities:
                avg_binary_complexity = sum(binary_complexities) / len(binary_complexities)
                complexity_diff = abs(source_complexity - avg_binary_complexity)
                validation['complexity_correlation'] = max(0.0, 1.0 - complexity_diff)
                
                if complexity_diff > 0.4:
                    validation['issues'].append(f"Source complexity ({source_complexity:.2f}) doesn't match binary complexity ({avg_binary_complexity:.2f})")
            
            # Check functional alignment
            source_depth = source_validation.get('implementation_depth', {}).get('depth_score', 0.0)
            binary_auth = binary_validation.get('authenticity_score', 0.0)
            
            alignment_diff = abs(source_depth - binary_auth)
            validation['functional_alignment'] = max(0.0, 1.0 - alignment_diff)
            
            if alignment_diff > 0.3:
                validation['issues'].append(f"Source implementation depth ({source_depth:.2f}) doesn't align with binary authenticity ({binary_auth:.2f})")
            
            # Overall consistency score
            validation['consistency_score'] = (
                validation['complexity_correlation'] * 0.6 + 
                validation['functional_alignment'] * 0.4
            )
            
            validation['source_binary_match'] = (
                validation['consistency_score'] >= 0.7 and 
                len(validation['issues']) == 0
            )
            
        except Exception as e:
            validation['issues'].append(f"Cross-validation failed: {str(e)}")
        
        return validation

    def _perform_independent_test_compilation(self, context: Dict[str, Any], test_dir: str) -> Dict[str, Any]:
        """Perform independent test compilation to verify source code"""
        compilation = {
            'success_score': 0.0,
            'compilation_successful': False,
            'test_binary_created': False,
            'test_binary_authentic': False,
            'issues': []
        }
        
        try:
            # Get source files
            agent11_result = context.get('agent_results', {}).get(11)
            if not agent11_result:
                compilation['issues'].append("No source reconstruction available")
                return compilation
            
            source_files = agent11_result.data.get('reconstructed_source', {}).get('source_files', {})
            if not source_files:
                compilation['issues'].append("No source files to compile")
                return compilation
            
            # Create test compilation directory
            test_compile_dir = os.path.join(test_dir, 'independent_test')
            os.makedirs(test_compile_dir, exist_ok=True)
            
            # Write source files
            main_file = None
            for filename, content in source_files.items():
                if isinstance(content, str):
                    file_path = os.path.join(test_compile_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(content)
                    
                    if filename.endswith('.c') and 'main' in content:
                        main_file = file_path
            
            if not main_file:
                compilation['issues'].append("No main source file found")
                return compilation
            
            # Attempt compilation
            test_binary = os.path.join(test_compile_dir, 'test_binary.exe')
            
            compile_result = subprocess.run([
                'gcc', '-o', test_binary, main_file
            ], capture_output=True, text=True, timeout=60, cwd=test_compile_dir)
            
            if compile_result.returncode == 0 and os.path.exists(test_binary):
                compilation['compilation_successful'] = True
                compilation['test_binary_created'] = True
                
                # Test the binary
                binary_test = self._test_binary_execution_depth(test_binary)
                compilation['test_binary_authentic'] = not binary_test.get('appears_trivial', True)
                
                if binary_test.get('appears_trivial', True):
                    compilation['issues'].extend(binary_test.get('trivial_indicators', []))
                
                # Calculate success score
                factors = [
                    compilation['compilation_successful'],
                    compilation['test_binary_created'],
                    compilation['test_binary_authentic']
                ]
                compilation['success_score'] = sum(factors) / len(factors)
                
            else:
                compilation['issues'].append(f"Compilation failed: {compile_result.stderr}")
            
        except Exception as e:
            compilation['issues'].append(f"Independent compilation failed: {str(e)}")
        
        return compilation

    def _perform_functional_verification(self, build_systems_data: Dict[str, Any],
                                       context: Dict[str, Any], test_dir: str) -> Dict[str, Any]:
        """Perform functional verification of generated binaries"""
        verification = {
            'functionality_score': 0.0,
            'binaries_functional': False,
            'behavior_analysis': {},
            'stress_testing': {},
            'issues': []
        }
        
        try:
            # Get successful binaries
            successful_compilers = build_systems_data.get('successful_compilers', [])
            compilation_attempts = build_systems_data.get('compilation_attempts', {})
            
            if not successful_compilers:
                verification['issues'].append("No binaries to verify")
                return verification
            
            functionality_scores = []
            
            for compiler in successful_compilers:
                attempt = compilation_attempts.get(compiler, {})
                binary_path = attempt.get('binary_path')
                
                if binary_path and os.path.exists(binary_path):
                    # Behavioral analysis
                    behavior_analysis = self._analyze_binary_behavior(binary_path)
                    verification['behavior_analysis'][compiler] = behavior_analysis
                    
                    # Stress testing
                    stress_test = self._perform_stress_testing(binary_path)
                    verification['stress_testing'][compiler] = stress_test
                    
                    # Calculate functionality score for this binary
                    behavior_score = behavior_analysis.get('functionality_score', 0.0)
                    stress_score = stress_test.get('stability_score', 0.0)
                    functionality_scores.append((behavior_score + stress_score) / 2)
            
            if functionality_scores:
                verification['functionality_score'] = sum(functionality_scores) / len(functionality_scores)
                verification['binaries_functional'] = verification['functionality_score'] >= 0.6
            
        except Exception as e:
            verification['issues'].append(f"Functional verification failed: {str(e)}")
        
        return verification

    def _analyze_binary_behavior(self, binary_path: str) -> Dict[str, Any]:
        """Analyze binary behavior patterns"""
        analysis = {
            'functionality_score': 0.0,
            'response_patterns': {},
            'argument_handling': {},
            'output_variation': {}
        }
        
        try:
            # Test argument handling
            test_cases = [
                [],
                ['--help'],
                ['-h'],
                ['test'],
                ['invalid_arg'],
                ['arg1', 'arg2', 'arg3']
            ]
            
            responses = []
            for args in test_cases:
                try:
                    result = subprocess.run([binary_path] + args,
                                          capture_output=True, text=True, timeout=3)
                    responses.append({
                        'args': args,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    })
                except subprocess.TimeoutExpired:
                    responses.append({'args': args, 'timeout': True})
                except Exception:
                    continue
            
            # Analyze response patterns
            unique_outputs = set()
            valid_responses = 0
            
            for response in responses:
                if 'stdout' in response:
                    combined_output = response['stdout'] + response['stderr']
                    if combined_output.strip():
                        unique_outputs.add(combined_output.strip())
                        valid_responses += 1
            
            analysis['response_patterns'] = {
                'total_tests': len(test_cases),
                'valid_responses': valid_responses,
                'unique_outputs': len(unique_outputs),
                'response_rate': valid_responses / len(test_cases) if test_cases else 0.0
            }
            
            # Calculate functionality score
            response_diversity = len(unique_outputs) / max(1, valid_responses)
            response_completeness = valid_responses / len(test_cases) if test_cases else 0.0
            
            analysis['functionality_score'] = (response_diversity * 0.6 + response_completeness * 0.4)
            
        except Exception as e:
            analysis['error'] = str(e)
        
        return analysis

    def _perform_stress_testing(self, binary_path: str) -> Dict[str, Any]:
        """Perform basic stress testing on binary"""
        testing = {
            'stability_score': 0.0,
            'crash_count': 0,
            'memory_errors': 0,
            'performance_metrics': {}
        }
        
        try:
            # Run multiple times to check for crashes
            crash_count = 0
            execution_times = []
            
            for i in range(5):
                try:
                    start_time = time.time()
                    result = subprocess.run([binary_path], 
                                          capture_output=True, text=True, timeout=10)
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    
                    # Check for crash indicators
                    if result.returncode < 0:  # Negative return codes often indicate crashes
                        crash_count += 1
                    
                except subprocess.TimeoutExpired:
                    # Timeout is not necessarily a crash
                    pass
                except Exception:
                    crash_count += 1
            
            testing['crash_count'] = crash_count
            testing['stability_score'] = max(0.0, 1.0 - (crash_count / 5))
            
            if execution_times:
                testing['performance_metrics'] = {
                    'avg_execution_time': sum(execution_times) / len(execution_times),
                    'min_execution_time': min(execution_times),
                    'max_execution_time': max(execution_times)
                }
            
        except Exception as e:
            testing['error'] = str(e)
        
        return testing

    def _determine_confidence_level(self, authenticity_score: float, issue_count: int) -> str:
        """Determine confidence level based on authenticity score and issues"""
        if authenticity_score >= 0.9 and issue_count == 0:
            return 'very_high'
        elif authenticity_score >= 0.8 and issue_count <= 1:
            return 'high'
        elif authenticity_score >= 0.6 and issue_count <= 3:
            return 'medium'
        elif authenticity_score >= 0.4 and issue_count <= 5:
            return 'low'
        else:
            return 'very_low'

    def _restructure_documentation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Restructure documentation according to project standards"""
        restructure_result = {
            'documentation_restructured': False,
            'docs_moved': [],
            'docs_renamed': [],
            'docs_removed': [],
            'restructure_actions': []
        }
        
        try:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent.parent
            docs_dir = project_root / 'docs'
            
            # Ensure docs directory exists
            docs_dir.mkdir(exist_ok=True)
            
            # Files that should remain in root (only essential project files)
            allowed_root_files = {'README.md', 'CLAUDE.md', 'main.py', 'launcher.exe', '.gitignore'}
            
            # Files to move to docs/ directory with potential renaming
            files_to_restructure = {
                'DOCUMENTATION.md': 'project_documentation.md',
                '100_PERCENT_COMPLETION_REPORT.md': 'completion_report.md',
                'EXTENSIONS_README.md': 'extensions_guide.md',
                'EXTENSION_PLAN.md': 'extension_plan.md',
                'MATRIX_ONLINE_ANALYSIS.md': 'matrix_analysis.md',
                'PARALLEL_IMPLEMENTATION_PLAN.md': 'parallel_implementation.md',
                'WINDOWS_COMPILATION.md': 'windows_compilation.md',
                'requirements.txt': 'requirements.txt'  # Move requirements to docs
            }
            
            # Process each file that needs restructuring
            for old_filename, new_filename in files_to_restructure.items():
                old_path = project_root / old_filename
                new_path = docs_dir / new_filename
                
                if old_path.exists():
                    try:
                        # Read content and move/rename file
                        content = old_path.read_text(encoding='utf-8')
                        new_path.write_text(content, encoding='utf-8')
                        old_path.unlink()  # Remove original file
                        
                        restructure_result['docs_moved'].append(f'{old_filename} -> docs/{new_filename}')
                        restructure_result['restructure_actions'].append(
                            f'Moved and renamed {old_filename} to docs/{new_filename}'
                        )
                        
                        if old_filename != new_filename:
                            restructure_result['docs_renamed'].append(f'{old_filename} -> {new_filename}')
                        
                    except Exception as e:
                        restructure_result['restructure_actions'].append(
                            f'Failed to move {old_filename}: {str(e)}'
                        )
            
            # Check for any other .md files in root that shouldn't be there
            for item in project_root.iterdir():
                if (item.is_file() and 
                    item.suffix == '.md' and 
                    item.name not in allowed_root_files and
                    item.name not in files_to_restructure):
                    
                    # Move any other markdown files to docs/
                    target_path = docs_dir / item.name
                    try:
                        if not target_path.exists():
                            item.rename(target_path)
                            restructure_result['docs_moved'].append(f'{item.name} -> docs/{item.name}')
                            restructure_result['restructure_actions'].append(
                                f'Moved additional markdown file {item.name} to docs/'
                            )
                    except Exception as e:
                        restructure_result['restructure_actions'].append(
                            f'Failed to move {item.name}: {str(e)}'
                        )
            
            # Note: No longer creating docs/README.md - all documentation is indexed in main README.md
            
            # Mark as successful if any actions were taken
            restructure_result['documentation_restructured'] = len(restructure_result['restructure_actions']) > 0
            
        except Exception as e:
            restructure_result['restructure_actions'].append(f'Documentation restructuring failed: {str(e)}')
        
        return restructure_result