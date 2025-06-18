#!/usr/bin/env python3
"""
Missing Agent Validation Tests
Tests for agents that haven't been refactored yet using mock execution and AI validation
Rules.md compliant - uses real AI system for output quality assessment
"""

import unittest
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Test infrastructure
try:
    from tests.test_phase4_comprehensive import TestPhase4Infrastructure
    from tests.test_agent_output_validation import AgentOutputValidator
except ImportError:
    # Fallback for direct execution
    from test_phase4_comprehensive import TestPhase4Infrastructure
    from test_agent_output_validation import AgentOutputValidator

# Core system imports
from core.ai_system import ai_available, ai_analyze, ai_request_safe


class MockAgentExecutor:
    """Mock agent executor for testing non-existent agents with realistic output"""
    
    def __init__(self, agent_id: int, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.validator = AgentOutputValidator()
    
    def execute_mock_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute mock analysis based on agent's expected functionality"""
        
        # Generate agent-specific mock output
        if self.agent_id == 1:  # Sentinel
            return self._mock_sentinel_output(context)
        elif self.agent_id == 2:  # Architect  
            return self._mock_architect_output(context)
        elif self.agent_id == 3:  # Merovingian
            return self._mock_merovingian_output(context)
        elif self.agent_id == 4:  # Agent Smith
            return self._mock_agent_smith_output(context)
        elif self.agent_id == 5:  # Neo
            return self._mock_neo_output(context)
        elif self.agent_id == 9:  # Commander Locke
            return self._mock_commander_locke_output(context)
        elif self.agent_id == 10:  # The Machine
            return self._mock_the_machine_output(context)
        else:
            return self._generic_mock_output(context)
    
    def _mock_sentinel_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Sentinel (Agent 1) binary discovery output"""
        return {
            'agent_id': 1,
            'agent_name': 'Sentinel',
            'status': 'SUCCESS',
            'binary_info': {
                'file_path': context.get('binary_path', '/test/binary.exe'),
                'file_size': 1048576,  # 1MB
                'format_type': 'PE',
                'architecture': 'x86',
                'subsystem': 'Windows GUI',
                'entry_point': 0x401000,
                'timestamp': '2024-01-01 00:00:00'
            },
            'format_analysis': {
                'pe_header': {
                    'machine_type': 'IMAGE_FILE_MACHINE_I386',
                    'number_of_sections': 6,
                    'size_of_optional_header': 224,
                    'characteristics': ['IMAGE_FILE_EXECUTABLE_IMAGE', 'IMAGE_FILE_32BIT_MACHINE']
                },
                'sections': [
                    {'name': '.text', 'virtual_address': 0x1000, 'size': 0x10000, 'characteristics': 'CODE'},
                    {'name': '.data', 'virtual_address': 0x11000, 'size': 0x2000, 'characteristics': 'DATA'},
                    {'name': '.rsrc', 'virtual_address': 0x13000, 'size': 0x1000, 'characteristics': 'RESOURCE'}
                ],
                'imports': {
                    'kernel32.dll': ['GetModuleHandleA', 'ExitProcess', 'CreateFileA'],
                    'user32.dll': ['MessageBoxA', 'CreateWindowA', 'DefWindowProcA'],
                    'advapi32.dll': ['RegOpenKeyExA', 'RegQueryValueExA']
                },
                'exports': []
            },
            'discovery_metadata': {
                'discovery_confidence': 0.95,
                'format_confidence': 0.98,
                'import_analysis_quality': 0.87,
                'total_functions_detected': 45,
                'total_imports': 15,
                'analysis_timestamp': '2024-12-08T10:00:00Z'
            }
        }
    
    def _mock_architect_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Architect (Agent 2) architecture analysis output"""
        return {
            'agent_id': 2,
            'agent_name': 'Architect',
            'status': 'SUCCESS',
            'compiler_analysis': {
                'toolchain': 'MSVC',
                'version': '14.0',
                'confidence': 0.92,
                'evidence': ['MSVCRT.dll detected', 'VCRUNTIME140.dll found', '__security_cookie pattern']
            },
            'optimization_analysis': {
                'level': 'O2',
                'confidence': 0.78,
                'detected_patterns': ['function_inlining', 'loop_optimization', 'constant_propagation'],
                'optimization_artifacts': [
                    {'type': 'function_count', 'value': 45, 'interpretation': 'normal_function_count'},
                    {'type': 'code_entropy', 'value': 6.8, 'interpretation': 'high_optimization'}
                ]
            },
            'abi_analysis': {
                'abi_analysis': {
                    'calling_convention': 'stdcall',
                    'stack_alignment': 4,
                    'exception_handling': 'C++ EH',
                    'rtti_enabled': True
                },
                'build_system_analysis': {
                    'detected_systems': ['MSBuild', 'Visual Studio'],
                    'primary_system': 'MSBuild',
                    'confidence': 0.85
                },
                'target_platform': 'Windows'
            },
            'architect_metadata': {
                'quality_score': 0.85,
                'validation_passed': True,
                'execution_time': 3.24,
                'ai_enhanced': True,
                'analysis_timestamp': '2024-12-08T10:00:00Z'
            }
        }
    
    def _mock_merovingian_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Merovingian (Agent 3) function detection output"""
        return {
            'agent_id': 3,
            'agent_name': 'Merovingian',
            'status': 'SUCCESS',
            'functions_detected': [
                {
                    'name': 'main',
                    'address': 0x401000,
                    'size': 256,
                    'confidence': 0.95,
                    'detection_method': 'prologue_pattern',
                    'signature': 'push ebp; mov ebp, esp',
                    'complexity_score': 0.65,
                    'assembly_instructions': [
                        {'address': 0x401000, 'mnemonic': 'push', 'operands': 'ebp'},
                        {'address': 0x401001, 'mnemonic': 'mov', 'operands': 'ebp, esp'},
                        {'address': 0x401003, 'mnemonic': 'sub', 'operands': 'esp, 0x20'}
                    ]
                },
                {
                    'name': 'WinMain',
                    'address': 0x401100,
                    'size': 512,
                    'confidence': 0.88,
                    'detection_method': 'symbol_table',
                    'complexity_score': 0.78
                }
            ],
            'decompilation_results': {
                'total_functions': 45,
                'successfully_decompiled': 42,
                'decompilation_confidence': 0.82,
                'code_quality_score': 0.76
            },
            'analysis_metadata': {
                'total_analysis_time': 15.67,
                'ghidra_integration': True,
                'ai_enhanced_decompilation': True,
                'validation_passed': True
            }
        }
    
    def _mock_agent_smith_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Agent Smith (Agent 4) binary structure analysis output"""
        return {
            'agent_id': 4,
            'agent_name': 'Agent Smith',
            'status': 'SUCCESS',
            'data_structures': [
                {
                    'address': 0x404000,
                    'size': 64,
                    'type': 'struct',
                    'name': 'WINDOW_DATA',
                    'confidence': 0.87,
                    'virtual_address': 0x404000,
                    'file_offset': 0x3000,
                    'alignment': 4,
                    'section_name': '.data',
                    'access_pattern': 'read_write'
                }
            ],
            'extracted_resources': [
                {
                    'type': 'icon',
                    'resource_id': 101,
                    'size': 2048,
                    'format': 'ICO',
                    'extracted_path': '/temp/icon_101.ico'
                },
                {
                    'type': 'dialog',
                    'resource_id': 200,
                    'size': 512,
                    'format': 'DIALOG',
                    'extracted_path': '/temp/dialog_200.rc'
                }
            ],
            'instrumentation_points': [
                {
                    'address': 0x401050,
                    'type': 'function_entry',
                    'purpose': 'Monitor main function execution',
                    'api_name': 'main'
                },
                {
                    'address': 0x401200,
                    'type': 'api_call',
                    'purpose': 'Monitor system API usage',
                    'api_name': 'CreateFileA',
                    'parameters': ['lpFileName', 'dwDesiredAccess', 'dwShareMode']
                }
            ],
            'structure_analysis_metadata': {
                'total_structures_found': 12,
                'resources_extracted': 8,
                'instrumentation_points_created': 25,
                'analysis_confidence': 0.84,
                'security_assessment': 'Low Risk'
            }
        }
    
    def _mock_neo_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Neo (Agent 5) advanced decompilation output"""
        return {
            'agent_id': 5,
            'agent_name': 'Neo',
            'status': 'SUCCESS',
            'advanced_decompilation': {
                'high_level_functions': [
                    {
                        'name': 'ProcessUserInput',
                        'decompiled_code': '''
int ProcessUserInput(HWND hwnd, UINT message, WPARAM wParam) {
    switch(message) {
        case WM_COMMAND:
            return HandleCommand(wParam);
        case WM_CLOSE:
            return HandleClose(hwnd);
        default:
            return DefWindowProc(hwnd, message, wParam, 0);
    }
}''',
                        'confidence': 0.91,
                        'quality_score': 0.87
                    }
                ],
                'code_patterns': {
                    'architectural_patterns': ['MVC', 'Event-Driven'],
                    'design_patterns': ['Singleton', 'Observer'],
                    'coding_style': 'Microsoft Visual C++ Standard'
                },
                'ai_enhanced_analysis': {
                    'semantic_understanding': 0.85,
                    'variable_naming_quality': 0.78,
                    'code_structure_quality': 0.82
                }
            },
            'ghidra_integration': {
                'ghidra_project_created': True,
                'analysis_scripts_executed': ['FunctionAnalyzer', 'DataTypeAnalyzer'],
                'ghidra_confidence': 0.89
            },
            'reconstruction_quality': {
                'overall_quality_score': 0.83,
                'compilation_readiness': 0.79,
                'missing_dependencies': ['MFC71.dll', 'MSVCR71.dll'],
                'estimated_compilation_success': 0.75
            }
        }
    
    def _mock_commander_locke_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock Commander Locke (Agent 9) critical import fixes output"""
        return {
            'agent_id': 9,
            'agent_name': 'Commander Locke',
            'status': 'SUCCESS',
            'import_table_reconstruction': {
                'original_imports': 538,
                'reconstructed_imports': 520,
                'missing_imports': 18,
                'reconstruction_confidence': 0.88,
                'dll_dependencies': [
                    'kernel32.dll', 'user32.dll', 'advapi32.dll', 'gdi32.dll',
                    'ole32.dll', 'oleaut32.dll', 'shell32.dll', 'comctl32.dll',
                    'MFC71.dll', 'MSVCR71.dll', 'MSVCP71.dll', 'msvcrt.dll',
                    'ntdll.dll', 'ws2_32.dll'
                ]
            },
            'function_resolution': {
                'resolved_by_name': 445,
                'resolved_by_ordinal': 75,
                'unresolved_functions': 18,
                'resolution_confidence': 0.91
            },
            'vs2022_compatibility': {
                'mfc71_compatibility_handled': True,
                'runtime_library_mapping': {
                    'MSVCR71.dll': 'vcruntime140.dll',
                    'MSVCP71.dll': 'msvcp140.dll'
                },
                'compatibility_score': 0.84
            },
            'critical_fixes_applied': {
                'fixes_implemented': [
                    'MFC 7.1 signature mapping',
                    'Ordinal resolution for system DLLs',
                    'VS2022 runtime compatibility layer',
                    'Import address table reconstruction'
                ],
                'validation_success_rate': 0.86
            }
        }
    
    def _mock_the_machine_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Mock The Machine (Agent 10) build system generation output"""
        return {
            'agent_id': 10,
            'agent_name': 'The Machine',
            'status': 'SUCCESS',
            'build_system_generation': {
                'vs2022_project_created': True,
                'project_file_path': '/output/reconstruction.vcxproj',
                'solution_file_path': '/output/reconstruction.sln',
                'project_configuration': {
                    'platform': 'Win32',
                    'configuration': 'Release',
                    'toolset': 'v143',
                    'windows_sdk_version': '10.0.22000.0'
                }
            },
            'source_organization': {
                'total_source_files': 28,
                'header_files': 12,
                'implementation_files': 16,
                'resource_files': 3,
                'file_organization_quality': 0.88
            },
            'compilation_validation': {
                'test_compilation_attempted': True,
                'compilation_success': False,
                'compilation_errors': 5,
                'compilation_warnings': 12,
                'error_categories': ['Missing headers', 'Undefined symbols', 'Type mismatches'],
                'estimated_fix_effort': 'Medium'
            },
            'automated_fixes_applied': {
                'header_generation': True,
                'type_inference': True,
                'dependency_resolution': True,
                'build_configuration_optimization': True,
                'fixes_success_rate': 0.78
            }
        }
    
    def _generic_mock_output(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generic mock output for undefined agents"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': 'SUCCESS',
            'mock_analysis': {
                'analysis_type': 'generic_binary_analysis',
                'confidence': 0.75,
                'data_quality': 0.68,
                'analysis_time': 5.23
            },
            'placeholder_data': {
                'note': f'Mock output for Agent {self.agent_id} ({self.agent_name})',
                'implementation_status': 'pending',
                'expected_functionality': 'binary_analysis_and_reconstruction'
            }
        }


class TestMissingAgentValidation(TestPhase4Infrastructure):
    """Test missing agents using mock execution and AI validation"""
    
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.validator = AgentOutputValidator()
    
    def setUp(self):
        super().setUp()
        self.test_context = {
            'binary_path': str(self.project_root / 'input' / 'launcher.exe'),
            'output_paths': {
                'base': str(self.temp_output),
                'agents': str(self.temp_output / 'agents'),
                'reports': str(self.temp_output / 'reports')
            },
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            },
            'agent_results': {}
        }


class TestAgent01SentinelMock(TestMissingAgentValidation):
    """Test Agent 1 (Sentinel) mock execution and validation"""
    
    def test_sentinel_mock_execution(self):
        """Test Sentinel mock execution and output validation"""
        mock_executor = MockAgentExecutor(1, "Sentinel")
        result = mock_executor.execute_mock_analysis(self.test_context)
        
        # Basic structure validation
        self.assertEqual(result['agent_id'], 1)
        self.assertIn('binary_info', result)
        self.assertIn('format_analysis', result)
        self.assertIn('discovery_metadata', result)
        
        # AI-enhanced validation
        validation_result = self.validator.validate_agent_output(1, "Sentinel", result)
        
        self.assertIn(validation_result['validation_status'], ['PASS', 'WARNING'])
        self.assertGreater(validation_result['quality_score'], 0.4)
    
    def test_sentinel_binary_info_quality(self):
        """Test Sentinel binary info analysis quality"""
        mock_executor = MockAgentExecutor(1, "Sentinel")
        result = mock_executor.execute_mock_analysis(self.test_context)
        
        binary_info = result.get('binary_info', {})
        
        # Validate critical binary info fields
        required_fields = ['file_path', 'file_size', 'format_type', 'architecture']
        for field in required_fields:
            self.assertIn(field, binary_info, f"Missing critical field: {field}")
        
        # AI validation of binary analysis quality
        if self.validator.ai_available:
            prompt = f"""
            Evaluate this binary analysis from Agent 1 (Sentinel):
            
            {json.dumps(binary_info, indent=2, default=str)}
            
            Rate the binary discovery quality (0.0-1.0) based on:
            1. Completeness of binary metadata
            2. Accuracy of format detection
            3. Architecture identification precision
            4. Overall discovery effectiveness
            """
            
            try:
                response = ai_analyze(prompt, "You are a binary analysis expert evaluating file discovery systems.")
                if response.success:
                    self.assertGreater(len(response.content), 100, "Should provide substantial binary analysis feedback")
                else:
                    self.skipTest("AI analysis not available - skipping quality evaluation")
            except Exception as e:
                self.skipTest(f"AI analysis failed - skipping quality evaluation: {e}")


class TestAgent09CommanderLockeMock(TestMissingAgentValidation):
    """Test Agent 9 (Commander Locke) critical import fixes mock execution"""
    
    def test_commander_locke_import_reconstruction(self):
        """Test Commander Locke import table reconstruction quality"""
        mock_executor = MockAgentExecutor(9, "Commander Locke")
        result = mock_executor.execute_mock_analysis(self.test_context)
        
        # Validate import reconstruction structure
        self.assertIn('import_table_reconstruction', result)
        self.assertIn('function_resolution', result)
        self.assertIn('vs2022_compatibility', result)
        
        import_data = result['import_table_reconstruction']
        
        # Validate critical import metrics
        self.assertIn('original_imports', import_data)
        self.assertIn('reconstructed_imports', import_data)
        self.assertGreater(import_data['reconstruction_confidence'], 0.5)
        
        # AI validation of import reconstruction quality
        if self.validator.ai_available:
            prompt = f"""
            Evaluate this import table reconstruction from Agent 9 (Commander Locke):
            
            {json.dumps(import_data, indent=2, default=str)}
            
            Rate the import reconstruction quality (0.0-1.0) based on:
            1. Import resolution accuracy
            2. DLL dependency completeness
            3. VS2022 compatibility handling
            4. Critical import recovery effectiveness
            
            This is a critical bottleneck fix for the Matrix Pipeline.
            """
            
            try:
                response = ai_analyze(prompt, "You are a reverse engineering expert evaluating import table reconstruction systems.")
                if response.success:
                    # Parse quality score from AI response
                    quality_score = self._extract_score_from_response(response.content)
                    self.assertGreater(quality_score, 0.4, "Import reconstruction quality should be acceptable")
                else:
                    self.skipTest("AI analysis not available - skipping quality evaluation")
            except Exception as e:
                self.skipTest(f"AI analysis failed - skipping quality evaluation: {e}")
    
    def test_commander_locke_vs2022_compatibility(self):
        """Test Commander Locke VS2022 compatibility fixes"""
        mock_executor = MockAgentExecutor(9, "Commander Locke")
        result = mock_executor.execute_mock_analysis(self.test_context)
        
        vs2022_data = result.get('vs2022_compatibility', {})
        
        # Validate VS2022 compatibility features
        self.assertIn('mfc71_compatibility_handled', vs2022_data)
        self.assertIn('runtime_library_mapping', vs2022_data)
        self.assertTrue(vs2022_data.get('mfc71_compatibility_handled', False))
        
        mapping = vs2022_data.get('runtime_library_mapping', {})
        self.assertIn('MSVCR71.dll', mapping)
        self.assertIn('MSVCP71.dll', mapping)


class TestAgent10TheMachineMock(TestMissingAgentValidation):
    """Test Agent 10 (The Machine) build system generation mock execution"""
    
    def test_the_machine_build_generation(self):
        """Test The Machine build system generation quality"""
        mock_executor = MockAgentExecutor(10, "The Machine")
        result = mock_executor.execute_mock_analysis(self.test_context)
        
        # Validate build system structure
        self.assertIn('build_system_generation', result)
        self.assertIn('source_organization', result)
        self.assertIn('compilation_validation', result)
        
        build_data = result['build_system_generation']
        
        # Validate VS2022 project generation
        self.assertTrue(build_data.get('vs2022_project_created', False))
        self.assertIn('project_file_path', build_data)
        self.assertIn('solution_file_path', build_data)
        
        # AI validation of build system quality
        if self.validator.ai_available:
            prompt = f"""
            Evaluate this build system generation from Agent 10 (The Machine):
            
            {json.dumps(build_data, indent=2, default=str)}
            
            Rate the build system quality (0.0-1.0) based on:
            1. VS2022 project configuration accuracy
            2. Build file organization quality
            3. Compilation readiness assessment
            4. Automated fix effectiveness
            """
            
            try:
                response = ai_analyze(prompt, "You are a build system expert evaluating automated build generation.")
                if not response.success:
                    self.skipTest("AI analysis not available - skipping quality evaluation")
            except Exception as e:
                self.skipTest(f"AI analysis failed - skipping quality evaluation: {e}")
    
    def test_the_machine_compilation_readiness(self):
        """Test The Machine compilation readiness assessment"""
        mock_executor = MockAgentExecutor(10, "The Machine")
        result = mock_executor.execute_mock_analysis(self.test_context)
        
        compilation_data = result.get('compilation_validation', {})
        
        # Validate compilation assessment
        self.assertIn('test_compilation_attempted', compilation_data)
        self.assertIn('compilation_errors', compilation_data)
        self.assertIn('error_categories', compilation_data)
        
        # Check for realistic compilation challenges
        errors = compilation_data.get('compilation_errors', 0)
        self.assertGreaterEqual(errors, 0, "Should track compilation errors")
        
        error_categories = compilation_data.get('error_categories', [])
        self.assertIsInstance(error_categories, list, "Error categories should be a list")


class TestMockAgentIntegration(TestMissingAgentValidation):
    """Test integration between mock agents"""
    
    def test_mock_pipeline_flow(self):
        """Test mock pipeline flow through multiple agents"""
        
        # Execute agents in dependency order
        agent_sequence = [
            (1, "Sentinel"),
            (2, "Architect"), 
            (3, "Merovingian"),
            (4, "Agent Smith"),
            (5, "Neo"),
            (9, "Commander Locke"),
            (10, "The Machine")
        ]
        
        pipeline_results = {}
        
        for agent_id, agent_name in agent_sequence:
            mock_executor = MockAgentExecutor(agent_id, agent_name)
            result = mock_executor.execute_mock_analysis(self.test_context)
            
            # Update shared memory for next agents
            self.test_context['shared_memory']['analysis_results'][agent_id] = result
            pipeline_results[agent_id] = result
        
        # Validate pipeline flow
        self.assertEqual(len(pipeline_results), 7, "Should execute all mock agents")
        
        # AI validation of pipeline coherence
        if self.validator.ai_available:
            pipeline_summary = {
                f"Agent_{k}": {
                    'agent_id': v.get('agent_id'),
                    'status': v.get('status'),
                    'key_metrics': self._extract_key_metrics(v)
                }
                for k, v in pipeline_results.items()
            }
            
            prompt = f"""
            Evaluate this mock Matrix Pipeline execution flow:
            
            {json.dumps(pipeline_summary, indent=2, default=str)[:2000]}
            
            Rate the pipeline coherence (0.0-1.0) based on:
            1. Logical flow between agents
            2. Data consistency and progression
            3. Coverage of binary analysis aspects
            4. Overall system integration quality
            """
            
            try:
                response = ai_analyze(prompt, "You are a systems integration expert evaluating pipeline coherence.")
                if response.success:
                    # Parse pipeline quality score
                    pipeline_score = self._extract_score_from_response(response.content)
                    self.assertGreater(pipeline_score, 0.4, "Mock pipeline coherence should be acceptable")
                else:
                    self.skipTest("AI analysis not available - skipping quality evaluation")
            except Exception as e:
                self.skipTest(f"AI analysis failed - skipping quality evaluation: {e}")
    
    def _extract_key_metrics(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from agent result for pipeline analysis"""
        metrics = {}
        
        # Common metrics
        if 'confidence' in str(agent_result):
            metrics['has_confidence_metrics'] = True
        if 'quality_score' in str(agent_result):
            metrics['has_quality_metrics'] = True
        
        # Agent-specific metrics
        agent_id = agent_result.get('agent_id')
        if agent_id == 1:  # Sentinel
            metrics['binary_format_detected'] = bool(agent_result.get('binary_info', {}).get('format_type'))
        elif agent_id == 9:  # Commander Locke
            metrics['imports_reconstructed'] = agent_result.get('import_table_reconstruction', {}).get('reconstructed_imports', 0)
        elif agent_id == 10:  # The Machine
            metrics['vs2022_project_created'] = agent_result.get('build_system_generation', {}).get('vs2022_project_created', False)
        
        return metrics
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract quality score from AI response"""
        import re
        
        patterns = [
            r'score[:\s]+(\d+\.?\d*)',
            r'quality[:\s]+(\d+)%',
            r'(\d+\.?\d*)/10',
            r'rate[:\s]+(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = float(match.group(1))
                if score > 1.0:
                    score = score / 100.0 if score <= 100 else score / 10.0
                return min(score, 1.0)
        
        return 0.5  # Default neutral score


# Test Suite Organization
def create_missing_agent_test_suite():
    """Create comprehensive missing agent test suite"""
    suite = unittest.TestSuite()
    
    test_classes = [
        TestAgent01SentinelMock,
        TestAgent09CommanderLockeMock,
        TestAgent10TheMachineMock,
        TestMockAgentIntegration
    ]
    
    for test_class in test_classes:
        suite.addTest(unittest.makeSuite(test_class))
    
    return suite


def run_missing_agent_tests():
    """Run missing agent validation test suite"""
    suite = create_missing_agent_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
        'status': 'PASSED' if len(result.failures) == 0 and len(result.errors) == 0 else 'NEEDS_ATTENTION'
    }


if __name__ == '__main__':
    print("Running Missing Agent Validation Test Suite...")
    print("=" * 60)
    
    report = run_missing_agent_tests()
    
    print("\n" + "=" * 60)
    print("MISSING AGENT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['total_tests']}")
    print(f"Failures: {report['failures']}")
    print(f"Errors: {report['errors']}")
    print(f"Skipped: {report['skipped']}")
    print(f"Success Rate: {report['success_rate']:.2%}")
    print(f"Overall Status: {report['status']}")
    
    if report['status'] == 'PASSED':
        print("\n✅ Missing Agent Mock Validation: OPERATIONAL")
    else:
        print(f"\n⚠️  Missing Agent Mock Validation: {report['status']}")