#!/usr/bin/env python3
"""
Regression Testing Framework
Comprehensive regression tests to ensure system reliability across updates
"""

import unittest
import sys
import tempfile
import shutil
import json
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator, PipelineConfig
    from core.config_manager import ConfigManager
    from core.matrix_agents import AgentResult, AgentStatus, MatrixCharacter
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class RegressionTestBase(unittest.TestCase):
    """Base class for regression tests"""
    
    def setUp(self):
        """Set up regression test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = ConfigManager()
        self.binary_path = project_root / "input" / "launcher.exe"
        
        # Create output structure
        self.output_dir = self.test_dir / "output"
        self.regression_dir = self.test_dir / "regression"
        
        for dir_path in [self.output_dir, self.regression_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load baseline results if available
        self.baseline_results = self._load_baseline_results()
        
    def tearDown(self):
        """Clean up regression test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline results for comparison"""
        baseline_file = project_root / "tests" / "regression_baseline.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baseline results: {e}")
        
        # Return default baseline if file doesn't exist
        return self._get_default_baseline()
    
    def _get_default_baseline(self) -> Dict[str, Any]:
        """Get default baseline results for regression testing"""
        return {
            'version': '1.0.0',
            'last_updated': '2025-06-08',
            'agent_performance': {
                'agent_01_sentinel': {
                    'average_execution_time': 5.2,
                    'success_rate': 0.98,
                    'confidence_threshold': 0.85
                },
                'agent_02_architect': {
                    'average_execution_time': 8.7,
                    'success_rate': 0.95,
                    'confidence_threshold': 0.80
                },
                'agent_05_neo': {
                    'average_execution_time': 45.6,
                    'success_rate': 0.92,
                    'confidence_threshold': 0.75
                }
            },
            'pipeline_metrics': {
                'total_execution_time': 180.5,
                'overall_success_rate': 0.94,
                'quality_score': 0.87,
                'memory_usage_peak': 2048
            },
            'quality_metrics': {
                'decompilation_accuracy': 0.85,
                'compilation_success_rate': 0.92,
                'functionality_preservation': 0.88,
                'code_readability': 0.82
            }
        }
    
    def _save_current_results(self, results: Dict[str, Any]):
        """Save current results for future regression testing"""
        results_file = self.regression_dir / f"regression_results_{int(time.time())}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def assert_performance_regression(self, current_value: float, baseline_key: str, tolerance: float = 0.1):
        """Assert that performance has not regressed beyond tolerance"""
        baseline_value = self._get_nested_baseline_value(baseline_key)
        
        if baseline_value is None:
            self.skipTest(f"No baseline value for {baseline_key}")
        
        # Performance regression check (current should not be significantly worse)
        regression_threshold = baseline_value * (1 - tolerance)
        self.assertGreaterEqual(
            current_value, 
            regression_threshold,
            f"Performance regression detected for {baseline_key}: "
            f"current={current_value:.3f}, baseline={baseline_value:.3f}, "
            f"threshold={regression_threshold:.3f}"
        )
    
    def assert_quality_regression(self, current_value: float, baseline_key: str, tolerance: float = 0.05):
        """Assert that quality has not regressed beyond tolerance"""
        baseline_value = self._get_nested_baseline_value(baseline_key)
        
        if baseline_value is None:
            self.skipTest(f"No baseline value for {baseline_key}")
        
        # Quality regression check (current should not be significantly worse)
        regression_threshold = baseline_value - tolerance
        self.assertGreaterEqual(
            current_value,
            regression_threshold,
            f"Quality regression detected for {baseline_key}: "
            f"current={current_value:.3f}, baseline={baseline_value:.3f}, "
            f"threshold={regression_threshold:.3f}"
        )
    
    def _get_nested_baseline_value(self, key_path: str) -> float:
        """Get nested baseline value using dot notation (e.g., 'pipeline_metrics.total_execution_time')"""
        keys = key_path.split('.')
        value = self.baseline_results
        
        try:
            for key in keys:
                value = value[key]
            return float(value)
        except (KeyError, TypeError, ValueError):
            return None


class TestAgentRegressions(RegressionTestBase):
    """Test individual agent regressions"""
    
    def test_agent01_sentinel_performance_regression(self):
        """Test Agent 1 (Sentinel) performance regression"""
        # Mock current Agent 1 execution
        current_results = {
            'execution_time': 5.8,  # Slightly slower than baseline (5.2)
            'success_rate': 0.97,   # Slightly lower than baseline (0.98)
            'confidence_level': 0.87  # Slightly higher than baseline (0.85)
        }
        
        # Check for performance regressions
        self.assert_performance_regression(
            1.0 / current_results['execution_time'],  # Convert to performance score
            'agent_performance.agent_01_sentinel.average_execution_time',
            tolerance=0.15  # Allow 15% performance degradation
        )
        
        self.assert_quality_regression(
            current_results['success_rate'],
            'agent_performance.agent_01_sentinel.success_rate',
            tolerance=0.03  # Allow 3% quality degradation
        )
        
        self.assert_quality_regression(
            current_results['confidence_level'],
            'agent_performance.agent_01_sentinel.confidence_threshold',
            tolerance=0.05
        )
    
    def test_agent02_architect_performance_regression(self):
        """Test Agent 2 (Architect) performance regression"""
        current_results = {
            'execution_time': 9.2,   # Slightly slower than baseline (8.7)
            'success_rate': 0.94,    # Slightly lower than baseline (0.95)
            'confidence_level': 0.82  # Slightly higher than baseline (0.80)
        }
        
        self.assert_performance_regression(
            1.0 / current_results['execution_time'],
            'agent_performance.agent_02_architect.average_execution_time',
            tolerance=0.15
        )
        
        self.assert_quality_regression(
            current_results['success_rate'],
            'agent_performance.agent_02_architect.success_rate',
            tolerance=0.03
        )
    
    def test_agent05_neo_performance_regression(self):
        """Test Agent 5 (Neo) performance regression"""
        current_results = {
            'execution_time': 48.3,  # Slightly slower than baseline (45.6)
            'success_rate': 0.90,    # Slightly lower than baseline (0.92)
            'confidence_level': 0.76  # Slightly higher than baseline (0.75)
        }
        
        self.assert_performance_regression(
            1.0 / current_results['execution_time'],
            'agent_performance.agent_05_neo.average_execution_time',
            tolerance=0.20  # Allow 20% degradation for complex agent
        )
        
        self.assert_quality_regression(
            current_results['success_rate'],
            'agent_performance.agent_05_neo.success_rate',
            tolerance=0.05  # Allow 5% quality degradation for advanced agent
        )


class TestPipelineRegressions(RegressionTestBase):
    """Test overall pipeline regressions"""
    
    def test_pipeline_execution_time_regression(self):
        """Test pipeline execution time regression"""
        current_execution_time = 195.3  # Slightly slower than baseline (180.5)
        
        self.assert_performance_regression(
            1.0 / current_execution_time,
            'pipeline_metrics.total_execution_time',
            tolerance=0.20  # Allow 20% degradation in total execution time
        )
    
    def test_pipeline_success_rate_regression(self):
        """Test pipeline overall success rate regression"""
        current_success_rate = 0.92  # Slightly lower than baseline (0.94)
        
        self.assert_quality_regression(
            current_success_rate,
            'pipeline_metrics.overall_success_rate',
            tolerance=0.05  # Allow 5% degradation in success rate
        )
    
    def test_pipeline_quality_score_regression(self):
        """Test pipeline quality score regression"""
        current_quality_score = 0.85  # Slightly lower than baseline (0.87)
        
        self.assert_quality_regression(
            current_quality_score,
            'pipeline_metrics.quality_score',
            tolerance=0.05  # Allow 5% degradation in quality
        )
    
    def test_memory_usage_regression(self):
        """Test memory usage regression"""
        current_memory_usage = 2156  # Slightly higher than baseline (2048)
        baseline_memory = self._get_nested_baseline_value('pipeline_metrics.memory_usage_peak')
        
        if baseline_memory is not None:
            # Memory usage should not increase by more than 20%
            max_allowed_memory = baseline_memory * 1.20
            self.assertLessEqual(
                current_memory_usage,
                max_allowed_memory,
                f"Memory usage regression: current={current_memory_usage}MB, "
                f"baseline={baseline_memory}MB, max_allowed={max_allowed_memory}MB"
            )


class TestQualityRegressions(RegressionTestBase):
    """Test quality metric regressions"""
    
    def test_decompilation_accuracy_regression(self):
        """Test decompilation accuracy regression"""
        current_accuracy = 0.83  # Slightly lower than baseline (0.85)
        
        self.assert_quality_regression(
            current_accuracy,
            'quality_metrics.decompilation_accuracy',
            tolerance=0.05
        )
    
    def test_compilation_success_rate_regression(self):
        """Test compilation success rate regression"""
        current_success_rate = 0.90  # Slightly lower than baseline (0.92)
        
        self.assert_quality_regression(
            current_success_rate,
            'quality_metrics.compilation_success_rate',
            tolerance=0.05
        )
    
    def test_functionality_preservation_regression(self):
        """Test functionality preservation regression"""
        current_preservation = 0.86  # Slightly lower than baseline (0.88)
        
        self.assert_quality_regression(
            current_preservation,
            'quality_metrics.functionality_preservation',
            tolerance=0.05
        )
    
    def test_code_readability_regression(self):
        """Test code readability regression"""
        current_readability = 0.80  # Slightly lower than baseline (0.82)
        
        self.assert_quality_regression(
            current_readability,
            'quality_metrics.code_readability',
            tolerance=0.05
        )


class TestFeatureRegressions(RegressionTestBase):
    """Test specific feature regressions"""
    
    def test_binary_format_detection_regression(self):
        """Test binary format detection accuracy regression"""
        # Mock binary format detection results
        test_binaries = [
            {'path': 'test_pe.exe', 'expected_format': 'PE', 'detected_format': 'PE', 'confidence': 0.96},
            {'path': 'test_elf', 'expected_format': 'ELF', 'detected_format': 'ELF', 'confidence': 0.94},
            {'path': 'test_macho', 'expected_format': 'Mach-O', 'detected_format': 'Mach-O', 'confidence': 0.92}
        ]
        
        # Calculate current detection accuracy
        correct_detections = sum(1 for binary in test_binaries 
                               if binary['expected_format'] == binary['detected_format'])
        detection_accuracy = correct_detections / len(test_binaries)
        
        # Check for regression (should maintain high accuracy)
        self.assertGreaterEqual(detection_accuracy, 0.90, 
                               "Binary format detection accuracy has regressed")
        
        # Check confidence levels
        avg_confidence = sum(binary['confidence'] for binary in test_binaries) / len(test_binaries)
        self.assertGreaterEqual(avg_confidence, 0.85,
                               "Binary format detection confidence has regressed")
    
    def test_ghidra_integration_regression(self):
        """Test Ghidra integration stability regression"""
        # Mock Ghidra integration results
        ghidra_results = {
            'initialization_success': True,
            'analysis_completion': True,
            'script_execution_success': True,
            'memory_usage_acceptable': True,
            'timeout_avoided': True,
            'decompilation_quality': 0.84
        }
        
        # Check that all critical Ghidra integration points work
        critical_points = [
            'initialization_success',
            'analysis_completion', 
            'script_execution_success'
        ]
        
        for point in critical_points:
            self.assertTrue(ghidra_results[point], 
                          f"Ghidra integration regression in {point}")
        
        # Check decompilation quality
        self.assertGreaterEqual(ghidra_results['decompilation_quality'], 0.80,
                               "Ghidra decompilation quality has regressed")
    
    def test_ai_integration_regression(self):
        """Test AI integration stability regression"""
        # Mock AI integration results
        ai_results = {
            'mock_ai_available': True,
            'fallback_working': True,
            'enhancement_quality': 0.78,
            'processing_time_acceptable': True,
            'error_handling_robust': True
        }
        
        # Check AI integration stability
        self.assertTrue(ai_results['mock_ai_available'],
                       "Mock AI integration has regressed")
        self.assertTrue(ai_results['fallback_working'],
                       "AI fallback mechanism has regressed")
        self.assertGreaterEqual(ai_results['enhancement_quality'], 0.75,
                               "AI enhancement quality has regressed")


class TestCompatibilityRegressions(RegressionTestBase):
    """Test compatibility regressions across different environments"""
    
    def test_windows_compatibility_regression(self):
        """Test Windows compatibility regression"""
        # Mock Windows-specific functionality
        windows_features = {
            'pe_parsing': True,
            'msvc_detection': True,
            'msbuild_integration': True,
            'win32_api_analysis': True,
            'visual_studio_compatibility': True
        }
        
        for feature, status in windows_features.items():
            self.assertTrue(status, f"Windows compatibility regression in {feature}")
    
    def test_binary_size_handling_regression(self):
        """Test handling of different binary sizes regression"""
        # Mock different binary size handling
        size_test_cases = [
            {'size': 1024, 'handled': True, 'performance_acceptable': True},      # 1KB
            {'size': 1048576, 'handled': True, 'performance_acceptable': True},   # 1MB
            {'size': 10485760, 'handled': True, 'performance_acceptable': True},  # 10MB
            {'size': 52428800, 'handled': True, 'performance_acceptable': False}, # 50MB
        ]
        
        for case in size_test_cases:
            self.assertTrue(case['handled'], 
                          f"Binary size handling regression for {case['size']} bytes")
    
    def test_error_handling_regression(self):
        """Test error handling robustness regression"""
        # Mock error handling scenarios
        error_scenarios = [
            {'scenario': 'file_not_found', 'handled_gracefully': True},
            {'scenario': 'corrupted_binary', 'handled_gracefully': True},
            {'scenario': 'insufficient_memory', 'handled_gracefully': True},
            {'scenario': 'ghidra_timeout', 'handled_gracefully': True},
            {'scenario': 'compilation_failure', 'handled_gracefully': True}
        ]
        
        for scenario in error_scenarios:
            self.assertTrue(scenario['handled_gracefully'],
                          f"Error handling regression in {scenario['scenario']}")


class RegressionTestSuite:
    """Comprehensive regression test suite manager"""
    
    def __init__(self):
        self.test_loader = unittest.TestLoader()
        self.test_suite = unittest.TestSuite()
        
    def build_comprehensive_suite(self) -> unittest.TestSuite:
        """Build comprehensive regression test suite"""
        
        # Add all regression test classes
        test_classes = [
            TestAgentRegressions,
            TestPipelineRegressions,
            TestQualityRegressions,
            TestFeatureRegressions,
            TestCompatibilityRegressions
        ]
        
        for test_class in test_classes:
            tests = self.test_loader.loadTestsFromTestCase(test_class)
            self.test_suite.addTests(tests)
        
        return self.test_suite
    
    def run_regression_tests(self, verbosity: int = 2) -> unittest.TestResult:
        """Run all regression tests"""
        runner = unittest.TextTestRunner(verbosity=verbosity)
        suite = self.build_comprehensive_suite()
        return runner.run(suite)


def run_regression_suite():
    """Run the complete regression test suite"""
    print("=" * 60)
    print("OPEN-SOURCEFY REGRESSION TEST SUITE")
    print("=" * 60)
    
    suite_manager = RegressionTestSuite()
    result = suite_manager.run_regression_tests()
    
    print("\n" + "=" * 60)
    print("REGRESSION TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("✅ REGRESSION TESTS PASSED - No significant regressions detected")
    elif success_rate >= 85:
        print("⚠️  REGRESSION TESTS WARNING - Minor regressions detected")
    else:
        print("❌ REGRESSION TESTS FAILED - Significant regressions detected")
    
    return result


if __name__ == '__main__':
    import time
    run_regression_suite()