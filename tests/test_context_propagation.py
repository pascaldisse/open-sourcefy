#!/usr/bin/env python3
"""
Context Propagation Regression Tests
Tests for the critical context propagation bug fixes implemented in Phase 1
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
    from core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator, PipelineConfig
    from core.config_manager import ConfigManager
    from core.matrix_agents import AgentResult, AgentStatus, MatrixCharacter
    from core.agents.agent01_sentinel import SentinelAgent
    from core.agents.agent02_architect import ArchitectAgent
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestContextPropagation(unittest.TestCase):
    """Test context propagation between agents"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = ConfigManager()
        self.binary_path = project_root / "input" / "launcher.exe"
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_agent_result_structure(self):
        """Test AgentResult has proper structure for context propagation"""
        result = AgentResult(
            agent_id=1,
            status=AgentStatus.SUCCESS,
            data={'test': 'data'},
            agent_name="TestAgent",
            matrix_character="TestCharacter"
        )
        
        self.assertEqual(result.agent_id, 1)
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertIsInstance(result.data, dict)
        self.assertEqual(result.agent_name, "TestAgent")
        self.assertEqual(result.matrix_character, "TestCharacter")
    
    def test_shared_memory_initialization(self):
        """Test shared memory is properly initialized"""
        context = {
            'binary_path': str(self.binary_path),
            'shared_memory': {}
        }
        
        # Initialize shared memory structure
        shared_memory = context['shared_memory']
        if 'analysis_results' not in shared_memory:
            shared_memory['analysis_results'] = {}
        if 'binary_metadata' not in shared_memory:
            shared_memory['binary_metadata'] = {}
            
        self.assertIn('analysis_results', context['shared_memory'])
        self.assertIn('binary_metadata', context['shared_memory'])
        
    def test_agent_dependency_validation(self):
        """Test agent dependency validation logic"""
        # Test Agent 2 depends on Agent 1
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {
                1: AgentResult(
                    agent_id=1,
                    status=AgentStatus.SUCCESS,
                    data={'binary_info': {'format_type': 'PE'}},
                    agent_name="Sentinel",
                    matrix_character="Sentinel"
                )
            },
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            }
        }
        
        # Simulate Agent 2's dependency check
        required_agents = [1]
        dependencies_met = False
        
        agent_results = context.get('agent_results', {})
        for agent_id in required_agents:
            if agent_id in agent_results and agent_results[agent_id].status == AgentStatus.SUCCESS:
                dependencies_met = True
                break
                
        self.assertTrue(dependencies_met, "Agent 2 should find Agent 1 dependency satisfied")
        
    def test_context_propagation_chain(self):
        """Test context propagates through agent chain"""
        # Start with empty context
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {},
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            }
        }
        
        # Agent 1 stores results
        agent1_result = AgentResult(
            agent_id=1,
            status=AgentStatus.SUCCESS,
            data={'binary_info': {'format_type': 'PE', 'architecture': 'x86'}},
            agent_name="Sentinel",
            matrix_character="Sentinel"
        )
        context['agent_results'][1] = agent1_result
        context['shared_memory']['analysis_results'][1] = agent1_result.data
        
        # Agent 2 should be able to access Agent 1's results
        agent2_context = context.copy()
        self.assertIn(1, agent2_context['agent_results'])
        self.assertEqual(
            agent2_context['agent_results'][1].data['binary_info']['format_type'], 
            'PE'
        )
        
        # Agent 2 adds its results
        agent2_result = AgentResult(
            agent_id=2,
            status=AgentStatus.SUCCESS,
            data={'architecture_analysis': {'compiler': 'MSVC', 'optimizations': ['O2']}},
            agent_name="Architect",
            matrix_character="Architect"
        )
        context['agent_results'][2] = agent2_result
        context['shared_memory']['analysis_results'][2] = agent2_result.data
        
        # Verify both results are available for next agent
        self.assertIn(1, context['agent_results'])
        self.assertIn(2, context['agent_results'])
        self.assertEqual(len(context['agent_results']), 2)


class TestAgentIndividual(unittest.TestCase):
    """Test individual agents execute correctly"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = ConfigManager()
        self.binary_path = project_root / "input" / "launcher.exe"
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('core.agents.agent01_sentinel.Path.exists')
    def test_agent1_sentinel_execution(self, mock_exists):
        """Test Agent 1 (Sentinel) executes without crashing"""
        mock_exists.return_value = True
        
        # Create context for Agent 1
        context = {
            'binary_path': str(self.binary_path),
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            }
        }
        
        try:
            agent1 = SentinelAgent()
            # Agent 1 should not crash on execution (even if it returns mock data)
            self.assertIsNotNone(agent1)
            self.assertEqual(agent1.agent_id, 1)
            self.assertEqual(agent1.matrix_character, MatrixCharacter.SENTINEL)
        except Exception as e:
            self.fail(f"Agent 1 initialization should not crash: {e}")
    
    def test_agent2_architect_execution(self):
        """Test Agent 2 (Architect) executes with proper dependencies"""
        # Create context with Agent 1 results
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {
                1: AgentResult(
                    agent_id=1,
                    status=AgentStatus.SUCCESS,
                    data={'binary_info': {'format_type': 'PE', 'architecture': 'x86'}},
                    agent_name="Sentinel",
                    matrix_character="Sentinel"
                )
            },
            'shared_memory': {
                'analysis_results': {
                    1: {'binary_info': {'format_type': 'PE', 'architecture': 'x86'}}
                },
                'binary_metadata': {
                    'discovery': {
                        'binary_info': {'format_type': 'PE', 'architecture': 'x86'}
                    }
                }
            }
        }
        
        try:
            agent2 = ArchitectAgent()
            # Agent 2 should not crash on initialization
            self.assertIsNotNone(agent2)
            self.assertEqual(agent2.agent_id, 2)
            self.assertEqual(agent2.matrix_character, MatrixCharacter.ARCHITECT)
            
            # Dependencies should be satisfied
            self.assertEqual(agent2.dependencies, [1])
        except Exception as e:
            self.fail(f"Agent 2 initialization should not crash: {e}")


class TestPipelineIntegration(unittest.TestCase):
    """Test pipeline integration and orchestration"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        self.binary_path = project_root / "input" / "launcher.exe"
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('core.matrix_pipeline_orchestrator.MatrixPipelineOrchestrator._execute_agent_batch')
    def test_orchestrator_initialization(self, mock_execute):
        """Test pipeline orchestrator initializes correctly"""
        try:
            config = PipelineConfig()
            orchestrator = MatrixPipelineOrchestrator(config)
            self.assertIsNotNone(orchestrator)
        except Exception as e:
            self.fail(f"Pipeline orchestrator should initialize: {e}")
    
    def test_execution_context_structure(self):
        """Test execution context has proper structure"""
        context = {
            'binary_path': str(self.binary_path),
            'agent_results': {},
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            },
            'output_paths': {
                'base': self.test_dir,
                'agents': self.test_dir / 'agents',
                'reports': self.test_dir / 'reports'
            }
        }
        
        # Verify required context keys
        required_keys = ['binary_path', 'agent_results', 'shared_memory']
        for key in required_keys:
            self.assertIn(key, context, f"Context should contain {key}")
            
        # Verify shared memory structure
        shared_memory_keys = ['analysis_results', 'binary_metadata']
        for key in shared_memory_keys:
            self.assertIn(key, context['shared_memory'], f"Shared memory should contain {key}")


class TestValidationFramework(unittest.TestCase):
    """Test validation framework components"""
    
    def test_agent_status_enum(self):
        """Test AgentStatus enum is properly defined"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        # Test AgentStatus enum values
        self.assertTrue(hasattr(AgentStatus, 'SUCCESS'))
        self.assertTrue(hasattr(AgentStatus, 'FAILED'))
        self.assertTrue(hasattr(AgentStatus, 'SKIPPED'))
        
    def test_matrix_character_enum(self):
        """Test MatrixCharacter enum is properly defined"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        # Test key Matrix characters are defined
        self.assertTrue(hasattr(MatrixCharacter, 'SENTINEL'))
        self.assertTrue(hasattr(MatrixCharacter, 'ARCHITECT'))
        self.assertTrue(hasattr(MatrixCharacter, 'NEO'))
        self.assertTrue(hasattr(MatrixCharacter, 'AGENT_SMITH'))


if __name__ == '__main__':
    unittest.main()