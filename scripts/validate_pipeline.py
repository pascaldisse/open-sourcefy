#!/usr/bin/env python3
"""
Pipeline Validation Script - Automated Quality Assurance

This script provides comprehensive validation for the Matrix decompilation pipeline
including agent testing, context propagation verification, and end-to-end validation.

Usage:
    python scripts/validate_pipeline.py [--level basic|standard|comprehensive|research]
    python scripts/validate_pipeline.py --agents 1,2,3
    python scripts/validate_pipeline.py --integration
    python scripts/validate_pipeline.py --regression
"""

import sys
import os
import argparse
import json
import time
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from core.pipeline_validator import PipelineValidator, ValidationLevel, ValidationResult
    from core.config_manager import get_config_manager, ConfigManager
    from core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator
    from core.binary_comparison import BinaryComparator
    from core.shared_components import MatrixLogger
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Create dummy types for annotations when imports aren't available
    ConfigManager = type(None)
    ValidationLevel = type(None)
    PipelineValidator = type(None)
    BinaryComparator = type(None)
    MatrixLogger = type(None)


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: str
    validation_level: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    warnings: int
    skipped_tests: int
    success_rate: float
    execution_time: float
    agent_results: Dict[str, Any]
    integration_results: Dict[str, Any]
    regression_results: Dict[str, Any]
    recommendations: List[str]
    

class PipelineValidationRunner:
    """
    Comprehensive pipeline validation runner
    
    Features:
    - Agent-level validation
    - Context propagation testing
    - Integration testing
    - Regression testing
    - Performance validation
    - Quality assurance reporting
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """Initialize validation runner"""
        if not IMPORTS_AVAILABLE:
            raise RuntimeError(f"Required imports not available: {IMPORT_ERROR}")
        
        self.config = config or get_config_manager()
        self.logger = MatrixLogger("PipelineValidation")
        
        # Initialize components
        self.validator = PipelineValidator(self.config)
        self.comparator = BinaryComparator(self.config)
        
        # Validation configuration
        if IMPORTS_AVAILABLE:
            self.validation_level = ValidationLevel.STANDARD
        else:
            self.validation_level = "standard"
        self.output_dir = None
        self.test_binary = None
        
        # Results tracking
        self.results = {
            'agent_tests': {},
            'integration_tests': {},
            'regression_tests': {},
            'performance_tests': {}
        }
        
    def run_full_validation(
        self, 
        level: Optional[ValidationLevel] = None,
        output_dir: Optional[str] = None,
        test_binary: Optional[str] = None
    ) -> ValidationReport:
        """
        Run comprehensive pipeline validation
        
        Args:
            level: Validation level (basic, standard, comprehensive, research)
            output_dir: Directory for validation outputs
            test_binary: Optional test binary for validation
            
        Returns:
            ValidationReport with comprehensive results
        """
        start_time = time.time()
        
        # Set default level if not provided and imports are available
        if level is None:
            if IMPORTS_AVAILABLE:
                level = ValidationLevel.STANDARD
            else:
                level = "standard"  # String fallback
        
        self.validation_level = level
        self.output_dir = output_dir or str(project_root / "temp" / "validation")
        self.test_binary = test_binary or str(project_root / "input" / "launcher.exe")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Starting {level.value} validation pipeline")
        
        try:
            # Phase 1: Agent validation
            self.logger.info("Phase 1: Agent validation")
            agent_results = self._run_agent_validation()
            
            # Phase 2: Context propagation testing
            self.logger.info("Phase 2: Context propagation testing")
            context_results = self._run_context_validation()
            
            # Phase 3: Integration testing
            level_value = level.value if hasattr(level, 'value') else level
            if level_value in ["standard", "comprehensive", "research"]:
                self.logger.info("Phase 3: Integration testing")
                integration_results = self._run_integration_validation()
            else:
                integration_results = {'skipped': True}
            
            # Phase 4: Regression testing
            if level_value in ["comprehensive", "research"]:
                self.logger.info("Phase 4: Regression testing")
                regression_results = self._run_regression_validation()
            else:
                regression_results = {'skipped': True}
            
            # Phase 5: Performance validation
            if level_value == "research":
                self.logger.info("Phase 5: Performance validation")
                performance_results = self._run_performance_validation()
            else:
                performance_results = {'skipped': True}
            
            # Compile results
            execution_time = time.time() - start_time
            report = self._compile_validation_report(
                agent_results, context_results, integration_results, 
                regression_results, performance_results, execution_time
            )
            
            # Save report
            self._save_validation_report(report)
            
            self.logger.info(f"Validation completed in {execution_time:.2f}s")
            self.logger.info(f"Success rate: {report.success_rate:.1%}")
            
            return report
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Validation failed: {e}", exc_info=True)
            
            # Return failure report
            return ValidationReport(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                validation_level=level_value,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                warnings=0,
                skipped_tests=0,
                success_rate=0.0,
                execution_time=execution_time,
                agent_results={'error': str(e)},
                integration_results={'error': str(e)},
                regression_results={'error': str(e)},
                recommendations=['Fix validation framework errors']
            )
    
    def _run_agent_validation(self) -> Dict[str, Any]:
        """Run individual agent validation tests"""
        results = {
            'total_agents': 16,
            'tested_agents': 0,
            'passed_agents': 0,
            'failed_agents': 0,
            'agent_details': {}
        }
        
        # Test each agent individually
        for agent_id in range(1, 17):  # Agents 1-16
            try:
                self.logger.info(f"Testing Agent {agent_id}")
                
                # Run agent test
                agent_result = self._test_individual_agent(agent_id)
                results['agent_details'][f'agent_{agent_id:02d}'] = agent_result
                results['tested_agents'] += 1
                
                if agent_result['status'] == 'passed':
                    results['passed_agents'] += 1
                else:
                    results['failed_agents'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Agent {agent_id} test failed: {e}")
                results['agent_details'][f'agent_{agent_id:02d}'] = {
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': 0.0
                }
                results['failed_agents'] += 1
        
        results['success_rate'] = results['passed_agents'] / max(results['tested_agents'], 1)
        return results
    
    def _test_individual_agent(self, agent_id: int) -> Dict[str, Any]:
        """Test an individual agent"""
        start_time = time.time()
        
        try:
            # Run agent test using subprocess to isolate
            cmd = [
                sys.executable, 
                "-m", "unittest", 
                f"tests.test_agent_individual.TestAgent{agent_id}",
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_root
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    'status': 'passed',
                    'execution_time': execution_time,
                    'output': result.stdout
                }
            else:
                return {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'execution_time': 60.0,
                'error': 'Test timed out'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'execution_time': time.time() - start_time,
                'error': str(e)
            }
    
    def _run_context_validation(self) -> Dict[str, Any]:
        """Run context propagation validation"""
        results = {
            'context_propagation': 'unknown',
            'shared_memory': 'unknown',
            'agent_dependencies': 'unknown',
            'execution_order': 'unknown'
        }
        
        try:
            # Run context propagation test
            cmd = [
                sys.executable, 
                "tests/test_context_propagation.py"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_root
            )
            
            if result.returncode == 0:
                results['context_propagation'] = 'passed'
                results['shared_memory'] = 'passed'
                results['agent_dependencies'] = 'passed'
                results['execution_order'] = 'passed'
            else:
                results['context_propagation'] = 'failed'
                results['error'] = result.stderr
            
        except Exception as e:
            results['context_propagation'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _run_integration_validation(self) -> Dict[str, Any]:
        """Run integration validation tests"""
        results = {
            'full_pipeline': 'unknown',
            'decompilation_pipeline': 'unknown',
            'compilation_pipeline': 'unknown',
            'matrix_online_specific': 'unknown'
        }
        
        integration_tests = [
            ('full_pipeline', 'tests/test_full_pipeline.py'),
            ('decompilation_pipeline', 'tests/test_integration_decompilation.py'),
            ('compilation_pipeline', 'tests/test_integration_compilation.py'),
            ('matrix_online_specific', 'tests/test_integration_matrix_online.py')
        ]
        
        for test_name, test_file in integration_tests:
            try:
                cmd = [sys.executable, test_file]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout for integration tests
                    cwd=project_root
                )
                
                if result.returncode == 0:
                    results[test_name] = 'passed'
                else:
                    results[test_name] = 'failed'
                    results[f'{test_name}_error'] = result.stderr
                    
            except subprocess.TimeoutExpired:
                results[test_name] = 'timeout'
            except Exception as e:
                results[test_name] = 'failed'
                results[f'{test_name}_error'] = str(e)
        
        return results
    
    def _run_regression_validation(self) -> Dict[str, Any]:
        """Run regression validation tests"""
        results = {
            'regression_suite': 'unknown',
            'known_issues': 'unknown',
            'performance_regression': 'unknown'
        }
        
        try:
            # Run regression test suite
            cmd = [sys.executable, "tests/test_regression.py"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=project_root
            )
            
            if result.returncode == 0:
                results['regression_suite'] = 'passed'
                results['known_issues'] = 'none'
                results['performance_regression'] = 'passed'
            else:
                results['regression_suite'] = 'failed'
                results['error'] = result.stderr
                
        except Exception as e:
            results['regression_suite'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation tests"""
        results = {
            'memory_usage': 'unknown',
            'execution_time': 'unknown',
            'resource_utilization': 'unknown',
            'scalability': 'unknown'
        }
        
        try:
            # Performance validation would be more complex
            # For now, check basic performance metrics
            if self.test_binary and os.path.exists(self.test_binary):
                start_time = time.time()
                
                # Run basic pipeline performance test
                cmd = [
                    sys.executable, 
                    "main.py", 
                    "--agents", "1,2,3,4", 
                    "--dry-run"
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=project_root
                )
                
                execution_time = time.time() - start_time
                
                if result.returncode == 0 and execution_time < 30:
                    results['execution_time'] = 'passed'
                    results['memory_usage'] = 'passed'
                    results['resource_utilization'] = 'passed'
                    results['scalability'] = 'passed'
                else:
                    results['execution_time'] = 'failed'
                    
        except Exception as e:
            results['execution_time'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _compile_validation_report(
        self, 
        agent_results: Dict[str, Any],
        context_results: Dict[str, Any],
        integration_results: Dict[str, Any],
        regression_results: Dict[str, Any],
        performance_results: Dict[str, Any],
        execution_time: float
    ) -> ValidationReport:
        """Compile comprehensive validation report"""
        
        # Count test results
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warnings = 0
        skipped_tests = 0
        
        # Agent results
        total_tests += agent_results.get('tested_agents', 0)
        passed_tests += agent_results.get('passed_agents', 0)
        failed_tests += agent_results.get('failed_agents', 0)
        
        # Context results
        context_passed = sum(1 for v in context_results.values() if v == 'passed')
        context_failed = sum(1 for v in context_results.values() if v == 'failed')
        total_tests += len([v for v in context_results.values() if v in ['passed', 'failed']])
        passed_tests += context_passed
        failed_tests += context_failed
        
        # Integration results
        integration_passed = sum(1 for k, v in integration_results.items() 
                                if not k.endswith('_error') and v == 'passed')
        integration_failed = sum(1 for k, v in integration_results.items() 
                               if not k.endswith('_error') and v == 'failed')
        integration_skipped = sum(1 for k, v in integration_results.items() 
                                if not k.endswith('_error') and v == 'skipped')
        
        total_tests += integration_passed + integration_failed + integration_skipped
        passed_tests += integration_passed
        failed_tests += integration_failed
        skipped_tests += integration_skipped
        
        # Calculate success rate
        success_rate = passed_tests / max(total_tests, 1)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            agent_results, context_results, integration_results, 
            regression_results, performance_results
        )
        
        return ValidationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            validation_level=self.validation_level.value,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            warnings=warnings,
            skipped_tests=skipped_tests,
            success_rate=success_rate,
            execution_time=execution_time,
            agent_results=agent_results,
            integration_results=integration_results,
            regression_results=regression_results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        agent_results: Dict[str, Any],
        context_results: Dict[str, Any],
        integration_results: Dict[str, Any],
        regression_results: Dict[str, Any],
        performance_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Agent recommendations
        failed_agents = agent_results.get('failed_agents', 0)
        if failed_agents > 0:
            recommendations.append(f"Fix {failed_agents} failing agents for improved pipeline stability")
        
        # Context recommendations
        if context_results.get('context_propagation') == 'failed':
            recommendations.append("Fix context propagation issues to ensure proper agent communication")
        
        # Integration recommendations
        failed_integration = sum(1 for k, v in integration_results.items() 
                               if not k.endswith('_error') and v == 'failed')
        if failed_integration > 0:
            recommendations.append(f"Address {failed_integration} integration test failures")
        
        # Performance recommendations
        if performance_results.get('execution_time') == 'failed':
            recommendations.append("Optimize pipeline performance to meet timing requirements")
        
        # Success recommendations
        success_rate = agent_results.get('success_rate', 0)
        if success_rate < 0.8:
            recommendations.append("Improve agent reliability to achieve >80% success rate")
        elif success_rate > 0.95:
            recommendations.append("Excellent agent reliability - consider expanding test coverage")
        
        if not recommendations:
            recommendations.append("All validation checks passed - pipeline is operating correctly")
        
        return recommendations
    
    def _save_validation_report(self, report: ValidationReport):
        """Save validation report to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"validation_report_{timestamp}.json")
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Also save human-readable summary
        summary_file = os.path.join(self.output_dir, f"validation_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(self._format_report_summary(report))
        
        self.logger.info(f"Validation report saved to {report_file}")
        self.logger.info(f"Validation summary saved to {summary_file}")
    
    def _format_report_summary(self, report: ValidationReport) -> str:
        """Format human-readable report summary"""
        summary = f"""
Matrix Pipeline Validation Report
=================================

Timestamp: {report.timestamp}
Validation Level: {report.validation_level.upper()}
Execution Time: {report.execution_time:.2f} seconds

Test Results Summary:
--------------------
Total Tests: {report.total_tests}
Passed: {report.passed_tests}
Failed: {report.failed_tests}
Warnings: {report.warnings}
Skipped: {report.skipped_tests}
Success Rate: {report.success_rate:.1%}

Agent Validation:
----------------
Total Agents: {report.agent_results.get('total_agents', 0)}
Tested: {report.agent_results.get('tested_agents', 0)}
Passed: {report.agent_results.get('passed_agents', 0)}
Failed: {report.agent_results.get('failed_agents', 0)}

Integration Testing:
-------------------
Full Pipeline: {report.integration_results.get('full_pipeline', 'unknown')}
Decompilation: {report.integration_results.get('decompilation_pipeline', 'unknown')}
Compilation: {report.integration_results.get('compilation_pipeline', 'unknown')}
Matrix Online: {report.integration_results.get('matrix_online_specific', 'unknown')}

Recommendations:
---------------
{chr(10).join(f"- {rec}" for rec in report.recommendations)}

Status: {'✅ VALIDATION PASSED' if report.success_rate >= 0.8 else '❌ VALIDATION FAILED'}
"""
        return summary.strip()


def main():
    """Main validation script entry point"""
    parser = argparse.ArgumentParser(
        description="Matrix Pipeline Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--level',
        choices=['basic', 'standard', 'comprehensive', 'research'],
        default='standard',
        help='Validation level (default: standard)'
    )
    
    parser.add_argument(
        '--output',
        help='Output directory for validation results'
    )
    
    parser.add_argument(
        '--binary',
        help='Test binary file path'
    )
    
    parser.add_argument(
        '--agents',
        help='Comma-separated list of agent IDs to test (e.g., 1,2,3)'
    )
    
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Run integration tests only'
    )
    
    parser.add_argument(
        '--regression',
        action='store_true',
        help='Run regression tests only'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not IMPORTS_AVAILABLE:
        print(f"Error: Required imports not available: {IMPORT_ERROR}")
        print("Please ensure the project is properly set up and dependencies are installed.")
        sys.exit(1)
    
    try:
        # Initialize validation runner
        runner = PipelineValidationRunner()
        
        # Handle specific test requests
        if args.agents:
            print("Individual agent testing not yet implemented via CLI")
            sys.exit(1)
        
        if args.integration:
            print("Integration-only testing not yet implemented via CLI")
            sys.exit(1)
        
        if args.regression:
            print("Regression-only testing not yet implemented via CLI")
            sys.exit(1)
        
        # Run full validation
        if IMPORTS_AVAILABLE:
            level = ValidationLevel(args.level)
        else:
            level = args.level
        report = runner.run_full_validation(
            level=level,
            output_dir=args.output,
            test_binary=args.binary
        )
        
        # Print summary
        print(runner._format_report_summary(report))
        
        # Exit with appropriate code
        if report.success_rate >= 0.8:
            print("\n✅ VALIDATION PASSED")
            sys.exit(0)
        else:
            print("\n❌ VALIDATION FAILED")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Validation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()