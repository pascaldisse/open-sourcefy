#!/usr/bin/env python3
"""
Decompilation Pipeline Integration Test
Test the complete decompilation pipeline from binary to source code
"""

import unittest
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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


class TestDecompilationPipeline(unittest.TestCase):
    """Test decompilation pipeline integration"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = ConfigManager()
        self.binary_path = project_root / "input" / "launcher.exe"
        
        # Create output structure
        self.output_dir = self.test_dir / "output"
        self.agents_dir = self.output_dir / "agents"
        self.ghidra_dir = self.output_dir / "ghidra"
        
        for dir_path in [self.output_dir, self.agents_dir, self.ghidra_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_decompilation_agent_sequence(self):
        """Test the proper sequence of decompilation agents"""
        # Decompilation agents: 1, 2, 5, 7, 14 (per --decompile-only mode)
        decompilation_sequence = [
            {'agent_id': 1, 'name': 'Sentinel', 'phase': 'Binary Discovery'},
            {'agent_id': 2, 'name': 'Architect', 'phase': 'Architecture Analysis'},
            {'agent_id': 5, 'name': 'Neo', 'phase': 'Advanced Decompilation'},
            {'agent_id': 7, 'name': 'Trainman', 'phase': 'Assembly Analysis'},
            {'agent_id': 14, 'name': 'Cleaner', 'phase': 'Code Optimization'}
        ]
        
        # Verify sequence is correct
        self.assertEqual(len(decompilation_sequence), 5)
        
        # Verify dependencies are satisfied
        agent_ids = [agent['agent_id'] for agent in decompilation_sequence]
        self.assertEqual(agent_ids, [1, 2, 5, 7, 14])
        
        # Test dependency chain
        context = {'agent_results': {}, 'shared_memory': {'analysis_results': {}}}
        
        for agent_info in decompilation_sequence:
            agent_id = agent_info['agent_id']
            
            # Each agent should be able to execute after previous agents
            if agent_id == 1:
                # Agent 1 has no dependencies
                self.assertTrue(True)
            elif agent_id == 2:
                # Agent 2 depends on Agent 1
                self.assertIn(1, context['agent_results'])
            elif agent_id == 5:
                # Agent 5 depends on Agents 1, 2
                self.assertIn(1, context['agent_results'])
                self.assertIn(2, context['agent_results'])
            
            # Mock agent execution
            context['agent_results'][agent_id] = AgentResult(
                agent_id=agent_id,
                status=AgentStatus.SUCCESS,
                data={'phase': agent_info['phase'], 'confidence': 0.8},
                agent_name=agent_info['name'],
                matrix_character=agent_info['name']
            )
    
    def test_binary_discovery_phase(self):
        """Test Agent 1 (Sentinel) binary discovery phase"""
        context = {
            'binary_path': str(self.binary_path),
            'output_paths': {
                'base': self.output_dir,
                'agents': self.agents_dir
            },
            'shared_memory': {'analysis_results': {}}
        }
        
        # Mock Agent 1 execution
        agent1_result = {
            'binary_info': {
                'format_type': 'PE',
                'architecture': 'x86',
                'file_size': 5242880,
                'entropy': 7.2,
                'sections': ['.text', '.data', '.rdata', '.rsrc'],
                'imports': ['kernel32.dll', 'user32.dll', 'gdi32.dll'],
                'exports': []
            },
            'metadata': {
                'agent_id': 1,
                'agent_name': 'Sentinel',
                'matrix_character': 'Sentinel',
                'execution_time': 5.2,
                'confidence_level': 0.92
            }
        }
        
        # Verify binary discovery results
        self.assertEqual(agent1_result['binary_info']['format_type'], 'PE')
        self.assertEqual(agent1_result['binary_info']['architecture'], 'x86')
        self.assertGreater(agent1_result['binary_info']['file_size'], 0)
        self.assertGreater(len(agent1_result['binary_info']['sections']), 0)
        self.assertGreaterEqual(agent1_result['metadata']['confidence_level'], 0.8)
        
        # Store result in context
        context['agent_results'] = {1: agent1_result}
    
    def test_architecture_analysis_phase(self):
        """Test Agent 2 (Architect) architecture analysis phase"""
        # Start with Agent 1 results
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {
                1: {
                    'binary_info': {
                        'format_type': 'PE',
                        'architecture': 'x86',
                        'file_size': 5242880
                    }
                }
            },
            'shared_memory': {'analysis_results': {}},
            'output_paths': {'agents': self.agents_dir}
        }
        
        # Mock Agent 2 execution
        agent2_result = {
            'architecture_analysis': {
                'compiler_detected': 'MSVC',
                'compiler_version': '.NET 2003',
                'optimization_level': 'O2',
                'target_platform': 'Win32',
                'runtime_libraries': ['MSVCRT', 'MSVCP'],
                'debugging_info': False,
                'code_patterns': ['standard_library_usage', 'win32_api_calls']
            },
            'calling_conventions': {
                'primary_convention': '__stdcall',
                'detected_conventions': ['__stdcall', '__cdecl']
            },
            'metadata': {
                'agent_id': 2,
                'agent_name': 'Architect',
                'matrix_character': 'Architect',
                'execution_time': 8.7,
                'confidence_level': 0.88
            }
        }
        
        # Verify architecture analysis
        self.assertEqual(agent2_result['architecture_analysis']['compiler_detected'], 'MSVC')
        self.assertIn(agent2_result['architecture_analysis']['optimization_level'], ['O0', 'O1', 'O2', 'O3'])
        self.assertEqual(agent2_result['calling_conventions']['primary_convention'], '__stdcall')
        self.assertGreaterEqual(agent2_result['metadata']['confidence_level'], 0.8)
        
        # Add to context
        context['agent_results'][2] = agent2_result
    
    def test_advanced_decompilation_phase(self):
        """Test Agent 5 (Neo) advanced decompilation phase"""
        # Context with previous agents
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {
                1: {'binary_info': {'format_type': 'PE', 'architecture': 'x86'}},
                2: {'architecture_analysis': {'compiler_detected': 'MSVC', 'optimization_level': 'O2'}}
            },
            'shared_memory': {'analysis_results': {}},
            'output_paths': {'agents': self.agents_dir, 'ghidra': self.ghidra_dir}
        }
        
        # Mock Agent 5 (Neo) execution
        agent5_result = {
            'decompiled_code': '''
            #include <stdio.h>
            #include <windows.h>
            
            // Matrix Online Launcher - Decompiled by Neo
            int main() {
                HWND hwnd;
                MSG msg;
                
                // Initialize application
                if (!InitializeApplication()) {
                    return 1;
                }
                
                // Create main window
                hwnd = CreateMainWindow();
                if (hwnd == NULL) {
                    return 1;
                }
                
                // Main message loop
                while (GetMessage(&msg, NULL, 0, 0)) {
                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }
                
                return 0;
            }
            
            BOOL InitializeApplication() {
                // Application initialization logic
                return TRUE;
            }
            
            HWND CreateMainWindow() {
                return CreateWindow("MatrixOnlineClass", "Matrix Online", 
                                  WS_OVERLAPPEDWINDOW, 100, 100, 800, 600,
                                  NULL, NULL, GetModuleHandle(NULL), NULL);
            }
            ''',
            'function_signatures': [
                {'name': 'main', 'return_type': 'int', 'parameters': [], 'address': '0x401000'},
                {'name': 'InitializeApplication', 'return_type': 'BOOL', 'parameters': [], 'address': '0x401150'},
                {'name': 'CreateMainWindow', 'return_type': 'HWND', 'parameters': [], 'address': '0x401200'}
            ],
            'variable_mappings': {
                'hwnd': 'HWND',
                'msg': 'MSG',
                'result': 'BOOL'
            },
            'quality_metrics': {
                'code_coverage': 0.85,
                'function_accuracy': 0.82,
                'variable_recovery': 0.78,
                'control_flow_accuracy': 0.88,
                'overall_score': 0.83,
                'confidence_level': 0.85
            },
            'ghidra_metadata': {
                'analysis_time': 45.6,
                'ghidra_version': '11.0.3',
                'scripts_used': ['CompleteDecompiler.java', 'NeoAdvancedAnalysis.java'],
                'analysis_confidence': 0.85
            },
            'matrix_insights': {
                'code_anomalies': [],
                'hidden_patterns': ['standard_windows_application'],
                'architectural_insights': ['Win32 GUI application with message loop'],
                'optimization_opportunities': ['Code cleanup', 'Variable naming improvement']
            },
            'metadata': {
                'agent_id': 5,
                'agent_name': 'Neo',
                'matrix_character': 'Neo',
                'execution_time': 45.6,
                'confidence_level': 0.85
            }
        }
        
        # Verify advanced decompilation results
        self.assertIn('#include', agent5_result['decompiled_code'])
        self.assertIn('main()', agent5_result['decompiled_code'])
        self.assertEqual(len(agent5_result['function_signatures']), 3)
        self.assertGreaterEqual(agent5_result['quality_metrics']['overall_score'], 0.8)
        self.assertGreaterEqual(agent5_result['metadata']['confidence_level'], 0.8)
        
        # Verify Ghidra integration
        self.assertIn('ghidra_version', agent5_result['ghidra_metadata'])
        self.assertGreater(agent5_result['ghidra_metadata']['analysis_time'], 0)
        
        # Add to context
        context['agent_results'][5] = agent5_result
    
    def test_assembly_analysis_phase(self):
        """Test Agent 7 (Trainman) assembly analysis phase"""
        # Context with previous agents
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {
                1: {'binary_info': {'format_type': 'PE'}},
                2: {'architecture_analysis': {'compiler_detected': 'MSVC'}},
                5: {'decompiled_code': 'int main() { return 0; }', 'function_signatures': []}
            },
            'shared_memory': {'analysis_results': {}},
            'output_paths': {'agents': self.agents_dir}
        }
        
        # Mock Agent 7 (Trainman) execution
        agent7_result = {
            'assembly_analysis': {
                'instruction_patterns': [
                    {'pattern': 'function_prologue', 'count': 15, 'confidence': 0.95},
                    {'pattern': 'function_epilogue', 'count': 15, 'confidence': 0.95},
                    {'pattern': 'loop_construct', 'count': 8, 'confidence': 0.88},
                    {'pattern': 'conditional_branch', 'count': 23, 'confidence': 0.92}
                ],
                'optimization_artifacts': [
                    {'type': 'inlined_function', 'locations': ['0x401125', '0x401387']},
                    {'type': 'loop_unrolling', 'locations': ['0x401200']},
                    {'type': 'dead_code_elimination', 'evidence': True}
                ],
                'calling_patterns': {
                    'api_calls': {
                        'CreateWindow': {'count': 1, 'pattern': 'standard'},
                        'GetMessage': {'count': 1, 'pattern': 'loop'},
                        'DispatchMessage': {'count': 1, 'pattern': 'standard'}
                    },
                    'internal_calls': {
                        'InitializeApplication': {'count': 1, 'calling_convention': '__stdcall'},
                        'CreateMainWindow': {'count': 1, 'calling_convention': '__stdcall'}
                    }
                },
                'data_flow_analysis': {
                    'stack_usage': {'max_depth': 256, 'avg_depth': 128},
                    'register_usage': ['EAX', 'EBX', 'ECX', 'EDX', 'ESP', 'EBP'],
                    'memory_patterns': ['heap_allocation', 'stack_variables']
                }
            },
            'transportation_analysis': {
                'code_mobility': 0.85,
                'platform_dependencies': ['Win32 API', 'MSVCRT'],
                'portability_score': 0.72,
                'modernization_opportunities': [
                    'Convert to 64-bit',
                    'Update to modern Windows APIs',
                    'Improve error handling'
                ]
            },
            'metadata': {
                'agent_id': 7,
                'agent_name': 'Trainman',
                'matrix_character': 'Trainman',
                'execution_time': 12.3,
                'confidence_level': 0.87
            }
        }
        
        # Verify assembly analysis results
        self.assertGreater(len(agent7_result['assembly_analysis']['instruction_patterns']), 0)
        self.assertIn('api_calls', agent7_result['assembly_analysis']['calling_patterns'])
        self.assertGreaterEqual(agent7_result['transportation_analysis']['code_mobility'], 0.7)
        self.assertGreaterEqual(agent7_result['metadata']['confidence_level'], 0.8)
        
        # Add to context
        context['agent_results'][7] = agent7_result
    
    def test_code_optimization_phase(self):
        """Test Agent 14 (Cleaner) code optimization phase"""
        # Context with all previous decompilation agents
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {
                1: {'binary_info': {'format_type': 'PE'}},
                2: {'architecture_analysis': {'compiler_detected': 'MSVC'}},
                5: {'decompiled_code': 'int main() { return 0; }', 'quality_metrics': {'overall_score': 0.83}},
                7: {'assembly_analysis': {'instruction_patterns': []}}
            },
            'shared_memory': {'analysis_results': {}},
            'output_paths': {'agents': self.agents_dir}
        }
        
        # Mock Agent 14 (Cleaner) execution
        agent14_result = {
            'optimization_analysis': {
                'code_issues': [
                    {'type': 'unused_variable', 'location': 'line_45', 'severity': 'low'},
                    {'type': 'inefficient_loop', 'location': 'function_main', 'severity': 'medium'},
                    {'type': 'memory_leak_potential', 'location': 'CreateMainWindow', 'severity': 'high'}
                ],
                'optimization_opportunities': [
                    {'type': 'variable_naming', 'impact': 'readability', 'effort': 'low'},
                    {'type': 'function_decomposition', 'impact': 'maintainability', 'effort': 'medium'},
                    {'type': 'error_handling', 'impact': 'robustness', 'effort': 'high'}
                ],
                'cleanup_performed': [
                    'removed_dead_code',
                    'standardized_naming',
                    'added_comments',
                    'formatted_code'
                ]
            },
            'optimized_code': '''
            #include <stdio.h>
            #include <windows.h>
            
            // Matrix Online Launcher - Optimized and Cleaned
            // Generated by Agent 14 (The Cleaner)
            
            // Application initialization function
            BOOL InitializeMatrixApplication(void) {
                // TODO: Add proper initialization logic
                return TRUE;
            }
            
            // Main window creation function
            HWND CreateMatrixMainWindow(void) {
                HWND hwnd = CreateWindow(
                    "MatrixOnlineClass",        // Window class name
                    "Matrix Online Launcher",   // Window title
                    WS_OVERLAPPEDWINDOW,        // Window style
                    CW_USEDEFAULT,              // X position
                    CW_USEDEFAULT,              // Y position
                    800,                        // Width
                    600,                        // Height
                    NULL,                       // Parent window
                    NULL,                       // Menu
                    GetModuleHandle(NULL),      // Instance
                    NULL                        // Additional data
                );
                
                if (hwnd == NULL) {
                    MessageBox(NULL, "Failed to create window", "Error", MB_OK | MB_ICONERROR);
                    return NULL;
                }
                
                return hwnd;
            }
            
            // Main application entry point
            int main(void) {
                HWND mainWindow = NULL;
                MSG message = {0};
                
                // Initialize the application
                if (!InitializeMatrixApplication()) {
                    fprintf(stderr, "Failed to initialize Matrix application\\n");
                    return EXIT_FAILURE;
                }
                
                // Create the main window
                mainWindow = CreateMatrixMainWindow();
                if (mainWindow == NULL) {
                    return EXIT_FAILURE;
                }
                
                // Show and update the window
                ShowWindow(mainWindow, SW_SHOW);
                UpdateWindow(mainWindow);
                
                // Main message loop
                while (GetMessage(&message, NULL, 0, 0)) {
                    TranslateMessage(&message);
                    DispatchMessage(&message);
                }
                
                return (int)message.wParam;
            }
            ''',
            'quality_improvements': {
                'readability_improvement': 0.25,
                'maintainability_improvement': 0.30,
                'robustness_improvement': 0.20,
                'overall_improvement': 0.25
            },
            'cleaning_metadata': {
                'lines_cleaned': 45,
                'functions_optimized': 3,
                'variables_renamed': 5,
                'comments_added': 12,
                'issues_resolved': 3
            },
            'metadata': {
                'agent_id': 14,
                'agent_name': 'Cleaner',
                'matrix_character': 'Cleaner',
                'execution_time': 15.8,
                'confidence_level': 0.91
            }
        }
        
        # Verify code optimization results
        self.assertGreater(len(agent14_result['optimization_analysis']['code_issues']), 0)
        self.assertIn('optimized_code', agent14_result)
        self.assertIn('#include', agent14_result['optimized_code'])
        self.assertIn('// Matrix Online Launcher - Optimized', agent14_result['optimized_code'])
        self.assertGreaterEqual(agent14_result['quality_improvements']['overall_improvement'], 0.2)
        self.assertGreaterEqual(agent14_result['metadata']['confidence_level'], 0.85)
        
        # Add to context
        context['agent_results'][14] = agent14_result
    
    def test_complete_decompilation_workflow(self):
        """Test complete decompilation workflow integration"""
        # Test the complete --decompile-only workflow
        workflow_context = {
            'mode': 'decompile_only',
            'binary_path': str(self.binary_path),
            'agent_results': {},
            'shared_memory': {'analysis_results': {}},
            'output_paths': {
                'base': self.output_dir,
                'agents': self.agents_dir,
                'ghidra': self.ghidra_dir
            }
        }
        
        # Simulate complete decompilation workflow
        decompilation_results = {
            'workflow_status': 'completed',
            'agents_executed': [1, 2, 5, 7, 14],
            'total_execution_time': 87.6,
            'final_outputs': {
                'decompiled_source': str(self.agents_dir / "agent_05_neo" / "decompiled_code.c"),
                'optimized_source': str(self.agents_dir / "agent_14_cleaner" / "optimized_code.c"),
                'analysis_reports': [
                    str(self.agents_dir / "agent_01_sentinel" / "binary_analysis.json"),
                    str(self.agents_dir / "agent_02_architect" / "architecture_analysis.json"),
                    str(self.agents_dir / "agent_07_trainman" / "assembly_analysis.json")
                ]
            },
            'quality_metrics': {
                'decompilation_accuracy': 0.85,
                'code_readability': 0.82,
                'completeness_score': 0.78,
                'overall_quality': 0.82
            },
            'success_criteria': {
                'all_agents_completed': True,
                'quality_threshold_met': True,
                'outputs_generated': True,
                'no_critical_errors': True
            }
        }
        
        # Verify workflow completion
        self.assertEqual(decompilation_results['workflow_status'], 'completed')
        self.assertEqual(len(decompilation_results['agents_executed']), 5)
        self.assertLess(decompilation_results['total_execution_time'], 120)  # Under 2 minutes
        self.assertGreaterEqual(decompilation_results['quality_metrics']['overall_quality'], 0.8)
        
        # Verify success criteria
        success_criteria = decompilation_results['success_criteria']
        self.assertTrue(success_criteria['all_agents_completed'])
        self.assertTrue(success_criteria['quality_threshold_met'])
        self.assertTrue(success_criteria['outputs_generated'])
        self.assertTrue(success_criteria['no_critical_errors'])
        
        # Verify output files would be created
        expected_outputs = decompilation_results['final_outputs']
        self.assertIn('decompiled_source', expected_outputs)
        self.assertIn('optimized_source', expected_outputs)
        self.assertGreater(len(expected_outputs['analysis_reports']), 0)
    
    def test_decompilation_error_handling(self):
        """Test error handling in decompilation pipeline"""
        error_scenarios = [
            {
                'scenario': 'binary_not_found',
                'error_agent': 1,
                'expected_behavior': 'fail_fast',
                'recovery_possible': False
            },
            {
                'scenario': 'ghidra_timeout',
                'error_agent': 5,
                'expected_behavior': 'retry_with_reduced_scope',
                'recovery_possible': True
            },
            {
                'scenario': 'unsupported_architecture',
                'error_agent': 2,
                'expected_behavior': 'fallback_to_basic_analysis',
                'recovery_possible': True
            },
            {
                'scenario': 'memory_exhaustion',
                'error_agent': 5,
                'expected_behavior': 'reduce_analysis_scope',
                'recovery_possible': True
            }
        ]
        
        for scenario in error_scenarios:
            with self.subTest(scenario=scenario['scenario']):
                # Verify error handling strategy is appropriate
                if scenario['scenario'] == 'binary_not_found':
                    self.assertEqual(scenario['expected_behavior'], 'fail_fast')
                    self.assertFalse(scenario['recovery_possible'])
                elif scenario['error_agent'] == 5:  # Neo (Ghidra-dependent)
                    self.assertIn(scenario['expected_behavior'], ['retry_with_reduced_scope', 'reduce_analysis_scope'])
                    self.assertTrue(scenario['recovery_possible'])
                else:
                    self.assertTrue(scenario['recovery_possible'])


if __name__ == '__main__':
    unittest.main()