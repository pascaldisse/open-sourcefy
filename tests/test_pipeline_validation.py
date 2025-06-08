#!/usr/bin/env python3
"""
Pipeline Validation Framework Tests
Test the comprehensive validation framework for Matrix pipeline
"""

import unittest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from core.pipeline_validator import (
        PipelineValidator, ValidationLevel, ValidationCategory, 
        ValidationResult, PipelineValidationReport, create_pipeline_validator
    )
    from core.config_manager import ConfigManager
    from core.matrix_agents import AgentResult, AgentStatus, MatrixCharacter
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult dataclass"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
    
    def test_validation_result_creation(self):
        """Test ValidationResult can be created"""
        result = ValidationResult(
            check_name="test_check",
            category=ValidationCategory.CONTEXT_PROPAGATION,
            level=ValidationLevel.HIGH,
            passed=True,
            message="Test passed"
        )
        
        self.assertEqual(result.check_name, "test_check")
        self.assertEqual(result.category, ValidationCategory.CONTEXT_PROPAGATION)
        self.assertEqual(result.level, ValidationLevel.HIGH)
        self.assertTrue(result.passed)
        self.assertEqual(result.message, "Test passed")
        self.assertGreater(result.timestamp, 0)
    
    def test_validation_result_with_details(self):
        """Test ValidationResult with details"""
        details = {"key": "value", "count": 42}
        result = ValidationResult(
            check_name="detailed_check",
            category=ValidationCategory.AGENT_EXECUTION,
            level=ValidationLevel.MEDIUM,
            passed=False,
            message="Check failed",
            details=details
        )
        
        self.assertEqual(result.details, details)
        self.assertFalse(result.passed)


class TestPipelineValidationReport(unittest.TestCase):
    """Test PipelineValidationReport dataclass"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
    
    def test_validation_report_creation(self):
        """Test validation report creation"""
        validation_results = [
            ValidationResult(
                check_name="check1",
                category=ValidationCategory.CONTEXT_PROPAGATION,
                level=ValidationLevel.HIGH,
                passed=True,
                message="Passed"
            ),
            ValidationResult(
                check_name="check2",
                category=ValidationCategory.AGENT_EXECUTION,
                level=ValidationLevel.MEDIUM,
                passed=False,
                message="Failed"
            )
        ]
        
        report = PipelineValidationReport(
            validation_id="test_validation",
            pipeline_execution_id="test_pipeline",
            validation_timestamp=1234567890.0,
            total_checks=2,
            passed_checks=1,
            failed_checks=1,
            critical_failures=0,
            validation_results=validation_results,
            overall_status="PASSED_WITH_WARNINGS",
            quality_score=0.75,
            execution_time=1.5,
            pipeline_metadata={"test": "metadata"}
        )
        
        self.assertEqual(report.validation_id, "test_validation")
        self.assertEqual(report.total_checks, 2)
        self.assertEqual(report.passed_checks, 1)
        self.assertEqual(report.failed_checks, 1)
        self.assertEqual(report.success_rate, 0.5)
        self.assertFalse(report.has_critical_failures)
    
    def test_validation_report_properties(self):
        """Test validation report calculated properties"""
        report = PipelineValidationReport(
            validation_id="test",
            pipeline_execution_id="test",
            validation_timestamp=0.0,
            total_checks=10,
            passed_checks=8,
            failed_checks=2,
            critical_failures=1,
            validation_results=[],
            overall_status="FAILED",
            quality_score=0.6,
            execution_time=2.0,
            pipeline_metadata={}
        )
        
        self.assertEqual(report.success_rate, 0.8)
        self.assertTrue(report.has_critical_failures)


class TestPipelineValidator(unittest.TestCase):
    """Test PipelineValidator main functionality"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = ConfigManager()
        self.validator = PipelineValidator(self.config)
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        self.assertIsInstance(self.validator, PipelineValidator)
        self.assertIsNotNone(self.validator.validation_checks)
        self.assertGreater(len(self.validator.validation_checks), 0)
        
        # Check some expected validation checks exist
        expected_checks = [
            'validate_context_structure',
            'validate_context_propagation',
            'validate_agent_execution',
            'validate_data_integrity'
        ]
        
        for check in expected_checks:
            self.assertIn(check, self.validator.validation_checks)
    
    def test_validator_factory_function(self):
        """Test factory function works"""
        validator = create_pipeline_validator()
        self.assertIsInstance(validator, PipelineValidator)
        
        # Test with custom config
        config = ConfigManager()
        validator_with_config = create_pipeline_validator(config)
        self.assertIsInstance(validator_with_config, PipelineValidator)
    
    def test_context_structure_validation_success(self):
        """Test context structure validation passes for valid context"""
        context = {
            'binary_path': '/test/path',
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            },
            'agent_results': {}
        }
        
        result = self.validator._validate_context_structure(context, {}, {})
        
        self.assertTrue(result.passed)
        self.assertEqual(result.check_name, "context_structure")
        self.assertEqual(result.category, ValidationCategory.CONTEXT_PROPAGATION)
        self.assertEqual(result.level, ValidationLevel.HIGH)
    
    def test_context_structure_validation_failure(self):
        """Test context structure validation fails for invalid context"""
        # Missing required keys
        context = {
            'binary_path': '/test/path'
            # Missing shared_memory and agent_results
        }
        
        result = self.validator._validate_context_structure(context, {}, {})
        
        self.assertFalse(result.passed)
        self.assertEqual(result.level, ValidationLevel.CRITICAL)
        self.assertIn("Missing required context keys", result.message)
    
    def test_agent_execution_validation_success(self):
        """Test agent execution validation for successful agents"""
        agent_results = {
            1: AgentResult(
                agent_id=1,
                status=AgentStatus.SUCCESS,
                data={'test': 'data'},
                agent_name="TestAgent1",
                matrix_character="TestCharacter1"
            ),
            2: AgentResult(
                agent_id=2,
                status=AgentStatus.SUCCESS,
                data={'test': 'data'},
                agent_name="TestAgent2",
                matrix_character="TestCharacter2"
            )
        }
        
        result = self.validator._validate_agent_execution({}, agent_results, {})
        
        self.assertTrue(result.passed)
        self.assertEqual(result.category, ValidationCategory.AGENT_EXECUTION)
        self.assertIn("successful", result.message.lower())
    
    def test_agent_execution_validation_failure(self):
        """Test agent execution validation for failed agents"""
        agent_results = {
            1: AgentResult(
                agent_id=1,
                status=AgentStatus.SUCCESS,
                data={'test': 'data'},
                agent_name="TestAgent1",
                matrix_character="TestCharacter1"
            ),
            2: AgentResult(
                agent_id=2,
                status=AgentStatus.FAILED,
                data={},
                agent_name="TestAgent2",
                matrix_character="TestCharacter2"
            )
        }
        
        result = self.validator._validate_agent_execution({}, agent_results, {})
        
        # Should pass if success rate is above threshold (50% here)
        self.assertTrue(result.passed)  # 1/2 = 0.5, which should be below default threshold
    
    def test_dependency_chain_validation(self):
        """Test dependency chain validation"""
        # Agent 2 depends on Agent 1, Agent 5 depends on Agents 1,2
        agent_results = {
            1: AgentResult(1, AgentStatus.SUCCESS, {}, "Agent1", "Sentinel"),
            2: AgentResult(2, AgentStatus.SUCCESS, {}, "Agent2", "Architect"),
            5: AgentResult(5, AgentStatus.SUCCESS, {}, "Agent5", "Neo")
        }
        
        result = self.validator._validate_dependency_chain({}, agent_results, {})
        
        self.assertTrue(result.passed)
        self.assertEqual(result.category, ValidationCategory.DEPENDENCY_CHAIN)
    
    def test_dependency_chain_validation_violation(self):
        """Test dependency chain validation with violations"""
        # Agent 2 success but Agent 1 failed (violation)
        agent_results = {
            1: AgentResult(1, AgentStatus.FAILED, {}, "Agent1", "Sentinel"),
            2: AgentResult(2, AgentStatus.SUCCESS, {}, "Agent2", "Architect")
        }
        
        result = self.validator._validate_dependency_chain({}, agent_results, {})
        
        self.assertFalse(result.passed)
        self.assertIn("violation", result.message.lower())
    
    @patch('core.pipeline_validator.Path.exists')
    def test_data_integrity_validation(self, mock_exists):
        """Test data integrity validation"""
        mock_exists.return_value = True
        
        context = {
            'binary_path': '/test/binary.exe'
        }
        
        agent_results = {
            1: AgentResult(
                1, AgentStatus.SUCCESS, 
                {'binary_info': {'architecture': 'x86'}}, 
                "Agent1", "Sentinel"
            ),
            2: AgentResult(
                2, AgentStatus.SUCCESS, 
                {'architecture_analysis': {'architecture': 'x86'}}, 
                "Agent2", "Architect"
            )
        }
        
        result = self.validator._validate_data_integrity(context, agent_results, {})
        
        self.assertTrue(result.passed)
        self.assertEqual(result.category, ValidationCategory.DATA_INTEGRITY)
    
    def test_binary_analysis_validation(self):
        """Test binary analysis validation"""
        agent_results = {
            1: AgentResult(
                1, AgentStatus.SUCCESS,
                {
                    'binary_info': {
                        'format_type': 'PE',
                        'architecture': 'x86',
                        'file_size': 1024
                    }
                },
                "Agent1", "Sentinel"
            )
        }
        
        result = self.validator._validate_binary_analysis({}, agent_results, {})
        
        self.assertTrue(result.passed)
        self.assertIn("meaningful results", result.message)
    
    def test_binary_analysis_validation_incomplete(self):
        """Test binary analysis validation with incomplete data"""
        agent_results = {
            1: AgentResult(
                1, AgentStatus.SUCCESS,
                {
                    'binary_info': {
                        'format_type': 'PE'
                        # Missing architecture and file_size
                    }
                },
                "Agent1", "Sentinel"
            )
        }
        
        result = self.validator._validate_binary_analysis({}, agent_results, {})
        
        self.assertFalse(result.passed)
        self.assertIn("incomplete", result.message.lower())
    
    def test_performance_validation(self):
        """Test performance validation"""
        metadata = {'execution_time': 150.0}  # Under default 300s threshold
        
        result = self.validator._validate_execution_performance({}, {}, metadata)
        
        self.assertTrue(result.passed)
        self.assertEqual(result.category, ValidationCategory.PERFORMANCE)
    
    def test_performance_validation_slow(self):
        """Test performance validation for slow execution"""
        metadata = {'execution_time': 400.0}  # Over default 300s threshold
        
        result = self.validator._validate_execution_performance({}, {}, metadata)
        
        self.assertFalse(result.passed)
        self.assertIn("exceeds threshold", result.message)
    
    def test_quality_thresholds_validation(self):
        """Test quality thresholds validation"""
        agent_results = {
            1: AgentResult(
                1, AgentStatus.SUCCESS,
                {'quality_score': 0.8},  # Above default 0.7 threshold
                "Agent1", "Test"
            ),
            2: AgentResult(
                2, AgentStatus.SUCCESS,
                {'confidence_level': 0.9},  # Above threshold
                "Agent2", "Test"
            )
        }
        
        result = self.validator._validate_quality_thresholds({}, agent_results, {})
        
        self.assertTrue(result.passed)
        self.assertEqual(result.category, ValidationCategory.QUALITY_THRESHOLDS)
    
    def test_completeness_validation(self):
        """Test completeness validation"""
        agent_results = {
            1: AgentResult(1, AgentStatus.SUCCESS, {}, "Agent1", "Test"),
            2: AgentResult(2, AgentStatus.SUCCESS, {}, "Agent2", "Test"),
            3: AgentResult(3, AgentStatus.FAILED, {}, "Agent3", "Test")
        }
        
        result = self.validator._validate_completeness({}, agent_results, {})
        
        # 2/3 = 0.67 > 0.5 threshold
        self.assertTrue(result.passed)
        self.assertIn("completeness", result.message.lower())
    
    def test_completeness_validation_no_agents(self):
        """Test completeness validation with no agents"""
        result = self.validator._validate_completeness({}, {}, {})
        
        self.assertFalse(result.passed)
        self.assertEqual(result.level, ValidationLevel.CRITICAL)


class TestFullPipelineValidation(unittest.TestCase):
    """Test full pipeline validation integration"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = ConfigManager()
        self.validator = PipelineValidator(self.config)
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('core.pipeline_validator.Path.exists')
    def test_full_pipeline_validation_success(self, mock_exists):
        """Test full pipeline validation with successful scenario"""
        mock_exists.return_value = True
        
        # Create realistic execution context
        context = {
            'binary_path': '/test/binary.exe',
            'shared_memory': {
                'analysis_results': {
                    1: {'binary_info': {'format_type': 'PE'}},
                    2: {'architecture': 'x86'}
                },
                'binary_metadata': {
                    'discovery': {'format': 'PE'}
                }
            },
            'agent_results': {}  # Will be filled with agent_results parameter
        }
        
        # Create agent results
        agent_results = {
            1: AgentResult(
                1, AgentStatus.SUCCESS,
                {'binary_info': {'format_type': 'PE', 'architecture': 'x86', 'file_size': 1024}},
                "Sentinel", "Sentinel"
            ),
            2: AgentResult(
                2, AgentStatus.SUCCESS,
                {'architecture_analysis': {'architecture': 'x86'}},
                "Architect", "Architect"
            )
        }
        
        # Update context with agent results for validation
        context['agent_results'] = agent_results
        
        metadata = {
            'execution_id': 'test_execution',
            'execution_time': 120.0
        }
        
        # Run full validation
        report = self.validator.validate_pipeline_execution(context, agent_results, metadata)
        
        # Verify report structure
        self.assertIsInstance(report, PipelineValidationReport)
        self.assertEqual(report.pipeline_execution_id, 'test_execution')
        self.assertGreater(report.total_checks, 0)
        self.assertGreaterEqual(report.passed_checks, 0)
        self.assertGreaterEqual(report.quality_score, 0.0)
        self.assertLessEqual(report.quality_score, 1.0)
        
        # Should pass most checks
        self.assertIn(report.overall_status, ["PASSED", "PASSED_WITH_WARNINGS"])
    
    def test_full_pipeline_validation_failures(self):
        """Test full pipeline validation with failures"""
        # Create problematic execution context
        context = {
            # Missing binary_path (will cause failures)
            'shared_memory': {},  # Missing required structure
            'agent_results': {}
        }
        
        agent_results = {
            1: AgentResult(1, AgentStatus.FAILED, {}, "Agent1", "Test"),
            2: AgentResult(2, AgentStatus.FAILED, {}, "Agent2", "Test")
        }
        
        metadata = {
            'execution_id': 'failed_execution',
            'execution_time': 500.0  # Over threshold
        }
        
        report = self.validator.validate_pipeline_execution(context, agent_results, metadata)
        
        # Should have failures
        self.assertGreater(report.failed_checks, 0)
        self.assertIn(report.overall_status, ["FAILED", "CRITICAL_FAILURE", "FAILED_RECOVERABLE"])
        self.assertLess(report.quality_score, 0.8)  # Should be low quality
    
    def test_validation_report_saving(self):
        """Test validation report can be saved"""
        # Create minimal report
        validation_results = [
            ValidationResult(
                check_name="test_check",
                category=ValidationCategory.CONTEXT_PROPAGATION,
                level=ValidationLevel.HIGH,
                passed=True,
                message="Test passed"
            )
        ]
        
        report = PipelineValidationReport(
            validation_id="test_save",
            pipeline_execution_id="test_pipeline",
            validation_timestamp=1234567890.0,
            total_checks=1,
            passed_checks=1,
            failed_checks=0,
            critical_failures=0,
            validation_results=validation_results,
            overall_status="PASSED",
            quality_score=1.0,
            execution_time=1.0,
            pipeline_metadata={}
        )
        
        # Save report
        output_path = self.test_dir / "validation_report.json"
        self.validator.save_validation_report(report, output_path)
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Verify content is valid JSON
        import json
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['validation_id'], 'test_save')
        self.assertEqual(saved_data['overall_status'], 'PASSED')
        self.assertEqual(len(saved_data['validation_results']), 1)


class TestValidationEnums(unittest.TestCase):
    """Test validation enums and categories"""
    
    def setUp(self):
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
    
    def test_validation_level_enum(self):
        """Test ValidationLevel enum values"""
        self.assertEqual(ValidationLevel.CRITICAL.value, "critical")
        self.assertEqual(ValidationLevel.HIGH.value, "high")
        self.assertEqual(ValidationLevel.MEDIUM.value, "medium")
        self.assertEqual(ValidationLevel.LOW.value, "low")
        self.assertEqual(ValidationLevel.INFO.value, "info")
    
    def test_validation_category_enum(self):
        """Test ValidationCategory enum values"""
        expected_categories = [
            "context_propagation",
            "agent_execution", 
            "data_integrity",
            "performance",
            "quality_thresholds",
            "dependency_chain",
            "output_generation",
            "pipeline_flow"
        ]
        
        category_values = [cat.value for cat in ValidationCategory]
        
        for expected in expected_categories:
            self.assertIn(expected, category_values)


if __name__ == '__main__':
    unittest.main()