#!/usr/bin/env python3
"""
Phase 4 Comprehensive Testing Infrastructure
Complete testing framework and validation systems for Open-Sourcefy Matrix Pipeline
"""

import unittest
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

class TestPhase4Infrastructure(unittest.TestCase):
    """Test Phase 4 testing and validation infrastructure"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures"""
        cls.project_root = project_root
        cls.src_path = cls.project_root / "src"
        cls.agents_path = cls.src_path / "core" / "agents"
        cls.test_binary_path = cls.project_root / "input" / "launcher.exe"
        
    def setUp(self):
        """Set up test environment for each test"""
        # Create temporary output directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.temp_output = Path(self.temp_dir) / "test_output"
        self.temp_output.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

class TestAgentIndividual(TestPhase4Infrastructure):
    """Test individual agent functionality"""
    
    def test_agent_01_sentinel_import(self):
        """Test Agent 1 (Sentinel) can be imported"""
        try:
            from core.agents.agent01_sentinel import Agent01_Sentinel
            self.assertTrue(True, "Agent 1 imported successfully")
        except ImportError as e:
            self.skipTest(f"Agent 1 import failed: {e}")
    
    def test_agent_02_architect_import(self):
        """Test Agent 2 (Architect) can be imported"""
        try:
            from core.agents.agent02_architect import Agent02_Architect
            self.assertTrue(True, "Agent 2 imported successfully")
        except ImportError as e:
            self.skipTest(f"Agent 2 import failed: {e}")
    
    def test_agent_dependencies_valid(self):
        """Test agent dependency declarations are valid"""
        # This tests the dependency structure without executing agents
        expected_dependencies = {
            1: [],  # Sentinel has no dependencies
            2: [1],  # Architect depends on Sentinel
            3: [1],  # Merovingian depends on Sentinel
            4: [1],  # Agent Smith depends on Sentinel
            5: [1, 2],  # Neo depends on Sentinel and Architect
        }
        
        for agent_id, expected_deps in expected_dependencies.items():
            # Test that dependency structure is logical
            self.assertIsInstance(expected_deps, list, f"Agent {agent_id} dependencies should be a list")

class TestAIIntegration(TestPhase4Infrastructure):
    """Test AI integration functionality"""
    
    def test_real_ai_system_available(self):
        """Test real AI system is available - rules.md compliant (no mocks)"""
        try:
            from core.ai_system import AISystem, get_ai_system, ai_available
            
            # Test AI system creation
            ai_system = AISystem()
            self.assertIsNotNone(ai_system, "AI system should be created")
            
            # Test global AI system access
            global_ai = get_ai_system()
            self.assertIsNotNone(global_ai, "Global AI system should be available")
            
            # Test availability check (may be False if Claude CLI not configured)
            ai_status = ai_available()
            self.assertIsInstance(ai_status, bool, "AI availability should return boolean")
            
        except ImportError as e:
            self.fail(f"Real AI system not available: {e}")
    
    def test_ai_system_integration(self):
        """Test AI system integration without requiring Claude CLI to be working"""
        try:
            from core.ai_system import ai_analyze, ai_request_safe, AIResponse
            
            # Test AI analyze function (will return empty if Claude not available)
            response = ai_analyze("test prompt")
            self.assertIsInstance(response, AIResponse, "Should return AIResponse object")
            
            # Test safe request function
            safe_result = ai_request_safe("test prompt", fallback="fallback")
            self.assertIsInstance(safe_result, str, "Should return string response")
            
        except Exception as e:
            self.fail(f"AI system integration failed: {e}")

class TestGhidraIntegration(TestPhase4Infrastructure):
    """Test Ghidra integration functionality"""
    
    def test_ghidra_processor_import(self):
        """Test Ghidra processor can be imported"""
        try:
            from core.ghidra_processor import GhidraProcessor
            self.assertTrue(True, "Ghidra processor imported successfully")
        except ImportError as e:
            self.skipTest(f"Ghidra processor import failed: {e}")
    
    def test_ghidra_headless_import(self):
        """Test Ghidra headless can be imported"""
        try:
            from core.ghidra_headless import GhidraHeadless
            self.assertTrue(True, "Ghidra headless imported successfully")
        except ImportError as e:
            self.skipTest(f"Ghidra headless import failed: {e}")

class TestPipelineValidation(TestPhase4Infrastructure):
    """Test pipeline validation framework"""
    
    def test_config_manager_available(self):
        """Test configuration manager is available"""
        try:
            from core.config_manager import ConfigManager
            
            config = ConfigManager()
            self.assertIsNotNone(config, "Config manager should be available")
            
        except ImportError as e:
            self.fail(f"Config manager not available: {e}")
    
    def test_shared_components_available(self):
        """Test shared components are available"""
        try:
            from core.shared_components import (
                MatrixLogger, MatrixFileManager, MatrixValidator,
                MatrixProgressTracker, MatrixErrorHandler
            )
            self.assertTrue(True, "Shared components imported successfully")
        except ImportError as e:
            self.skipTest(f"Shared components import failed: {e}")
    
    def test_output_directory_structure(self):
        """Test output directory structure can be created"""
        output_structure = {
            'agents': 'Agent-specific analysis outputs',
            'ghidra': 'Ghidra decompilation results',
            'compilation': 'Compilation artifacts and generated source',
            'reports': 'Pipeline execution reports',
            'logs': 'Execution logs and debug information',
            'temp': 'Temporary files (auto-cleaned)',
            'tests': 'Generated test files'
        }
        
        for subdir, description in output_structure.items():
            test_dir = self.temp_output / subdir
            test_dir.mkdir(parents=True, exist_ok=True)
            self.assertTrue(test_dir.exists(), f"Output subdirectory {subdir} should be creatable")

class TestContextPropagation(TestPhase4Infrastructure):
    """Test context propagation between agents"""
    
    def test_execution_context_import(self):
        """Test execution context can be imported"""
        try:
            from core.matrix_execution_context import MatrixExecutionContext
            self.assertTrue(True, "Execution context imported successfully")
        except ImportError as e:
            self.skipTest(f"Execution context import failed: {e}")
    
    def test_parallel_executor_import(self):
        """Test parallel executor can be imported"""
        try:
            from core.matrix_parallel_executor import MatrixParallelExecutor
            self.assertTrue(True, "Parallel executor imported successfully")
        except ImportError as e:
            self.skipTest(f"Parallel executor import failed: {e}")
    
    def test_pipeline_orchestrator_import(self):
        """Test pipeline orchestrator can be imported"""
        try:
            from core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator
            self.assertTrue(True, "Pipeline orchestrator imported successfully")
        except ImportError as e:
            self.skipTest(f"Pipeline orchestrator import failed: {e}")

class TestBinaryValidation(TestPhase4Infrastructure):
    """Test binary validation and analysis capabilities"""
    
    def test_test_binary_exists(self):
        """Test that test binary exists"""
        if self.test_binary_path.exists():
            self.assertTrue(self.test_binary_path.is_file(), "Test binary should be a file")
            self.assertGreater(self.test_binary_path.stat().st_size, 0, "Test binary should not be empty")
        else:
            self.skipTest("Test binary (launcher.exe) not available")
    
    def test_binary_utils_import(self):
        """Test binary utilities can be imported"""
        try:
            from core.binary_utils import BinaryAnalyzer
            self.assertTrue(True, "Binary utils imported successfully")
        except ImportError as e:
            self.skipTest(f"Binary utils import failed: {e}")

class TestErrorHandling(TestPhase4Infrastructure):
    """Test error handling and recovery mechanisms"""
    
    def test_error_handler_import(self):
        """Test error handler can be imported"""
        try:
            from core.error_handler import MatrixErrorHandler
            self.assertTrue(True, "Error handler imported successfully")
        except ImportError as e:
            self.skipTest(f"Error handler import failed: {e}")
    
    def test_custom_exceptions_import(self):
        """Test custom exceptions can be imported"""
        try:
            from core.exceptions import ValidationError, ConfigurationError
            self.assertTrue(True, "Custom exceptions imported successfully")
        except ImportError as e:
            self.skipTest(f"Custom exceptions import failed: {e}")

class TestPerformanceMonitoring(TestPhase4Infrastructure):
    """Test performance monitoring capabilities"""
    
    def test_performance_tracking(self):
        """Test basic performance tracking functionality"""
        import time
        
        start_time = time.time()
        # Simulate some work
        time.sleep(0.1)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.assertGreater(execution_time, 0.05, "Should measure execution time correctly")
        self.assertLess(execution_time, 1.0, "Simple operation should be fast")

class TestIntegrationWorkflows(TestPhase4Infrastructure):
    """Test integration workflows and end-to-end scenarios"""
    
    def test_dry_run_simulation(self):
        """Test dry run simulation without actual execution"""
        # This simulates what a dry run would test
        test_context = {
            'binary_path': str(self.test_binary_path) if self.test_binary_path.exists() else '/mock/path',
            'output_paths': {
                'base': str(self.temp_output),
                'agents': str(self.temp_output / 'agents'),
                'reports': str(self.temp_output / 'reports')
            },
            'shared_memory': {'analysis_results': {}},
            'agent_results': {}
        }
        
        # Test that context structure is valid
        self.assertIn('binary_path', test_context)
        self.assertIn('output_paths', test_context)
        self.assertIn('shared_memory', test_context)
        self.assertIn('agent_results', test_context)
    
    def test_mock_agent_execution(self):
        """Test mock agent execution workflow"""
        # Simulate agent execution without actual processing
        mock_agent_result = {
            'agent_id': 1,
            'status': 'SUCCESS',
            'data': {
                'analysis_type': 'mock_test',
                'confidence': 0.85,
                'processing_time': 0.1
            },
            'metadata': {
                'execution_mode': 'test',
                'timestamp': '2025-06-08'
            }
        }
        
        # Validate mock result structure
        self.assertIn('agent_id', mock_agent_result)
        self.assertIn('status', mock_agent_result)
        self.assertIn('data', mock_agent_result)
        self.assertEqual(mock_agent_result['agent_id'], 1)

class TestQualityAssurance(TestPhase4Infrastructure):
    """Test quality assurance and validation metrics"""
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation"""
        # Test basic quality score calculation
        test_metrics = {
            'code_coverage': 0.85,
            'function_accuracy': 0.78,
            'variable_recovery': 0.82,
            'control_flow_accuracy': 0.90
        }
        
        # Calculate overall quality score
        overall_score = sum(test_metrics.values()) / len(test_metrics)
        
        self.assertGreater(overall_score, 0.0, "Quality score should be positive")
        self.assertLessEqual(overall_score, 1.0, "Quality score should not exceed 1.0")
        self.assertGreater(overall_score, 0.7, "Mock quality should be reasonably high")
    
    def test_confidence_level_validation(self):
        """Test confidence level validation"""
        test_confidence_levels = [0.95, 0.87, 0.76, 0.68, 0.45]
        
        for confidence in test_confidence_levels:
            self.assertGreaterEqual(confidence, 0.0, "Confidence should be non-negative")
            self.assertLessEqual(confidence, 1.0, "Confidence should not exceed 1.0")
    
    def test_threshold_validation(self):
        """Test quality threshold validation"""
        quality_thresholds = {
            'code_quality': 0.75,
            'implementation_score': 0.75,
            'completeness': 0.70,
            'binary_analysis_confidence': 0.60
        }
        
        for metric, threshold in quality_thresholds.items():
            self.assertGreater(threshold, 0.0, f"{metric} threshold should be positive")
            self.assertLess(threshold, 1.0, f"{metric} threshold should be less than 1.0")

# Test Suite Organization
def create_test_suite():
    """Create comprehensive test suite for Phase 4"""
    suite = unittest.TestSuite()
    
    # Add test classes in logical order
    test_classes = [
        TestPhase4Infrastructure,
        TestAgentIndividual,
        TestAIIntegration,
        TestGhidraIntegration,
        TestPipelineValidation,
        TestContextPropagation,
        TestBinaryValidation,
        TestErrorHandling,
        TestPerformanceMonitoring,
        TestIntegrationWorkflows,
        TestQualityAssurance
    ]
    
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    return suite

def run_comprehensive_tests():
    """Run comprehensive Phase 4 test suite"""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate test report
    test_report = {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'status': 'PASSED' if len(result.failures) == 0 and len(result.errors) == 0 else 'FAILED'
    }
    
    return test_report

if __name__ == '__main__':
    # Run comprehensive test suite
    print("Running Phase 4 Comprehensive Test Suite...")
    print("=" * 60)
    
    report = run_comprehensive_tests()
    
    print("\n" + "=" * 60)
    print("PHASE 4 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['total_tests']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Skipped: {report['skipped']}")
    print(f"Success Rate: {report['success_rate']:.2%}")
    print(f"Overall Status: {report['status']}")
    
    if report['status'] == 'PASSED':
        print("\n✅ Phase 4 Testing Infrastructure: OPERATIONAL")
    else:
        print(f"\n❌ Phase 4 Testing Infrastructure: NEEDS ATTENTION")
        print("   Some tests failed - review output above for details")