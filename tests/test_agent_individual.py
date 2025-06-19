#!/usr/bin/env python3
"""
Individual Agent Tests
Test each Matrix agent individually for proper functionality
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
    from core.config_manager import ConfigManager
    from core.matrix_agents import AgentResult, AgentStatus, MatrixCharacter
    # Import available agents using correct class names
    from core.agents.agent01_sentinel import SentinelAgent
    from core.agents.agent02_architect import ArchitectAgent
    from core.agents.agent03_merovingian import MerovingianAgent
    from core.agents.agent04_agent_smith import AgentSmithAgent
    from core.agents.agent05_neo_advanced_decompiler import NeoAgent
    from core.agents.agent10_the_machine import Agent10_TheMachine
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestAgent1Sentinel(unittest.TestCase):
    """Test Agent 1 - Sentinel (Binary Discovery)"""
    
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
    
    def test_agent1_initialization(self):
        """Test Agent 1 initializes correctly"""
        agent = SentinelAgent()
        
        self.assertEqual(agent.agent_id, 1)
        self.assertEqual(agent.matrix_character, MatrixCharacter.SENTINEL)
        self.assertEqual(agent.dependencies, [])  # No dependencies
        
    def test_agent1_matrix_description(self):
        """Test Agent 1 has proper Matrix description"""
        agent = SentinelAgent()
        description = agent.get_matrix_description()
        
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 10)
        self.assertIn("Sentinel", description)
        
    @patch('core.agents.agent01_sentinel.Path.exists')
    def test_agent1_validate_prerequisites(self, mock_exists):
        """Test Agent 1 prerequisite validation"""
        mock_exists.return_value = True
        
        agent = SentinelAgent()
        context = {
            'binary_path': str(self.binary_path),
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            }
        }
        
        # Should not raise exception for valid context
        try:
            agent._validate_prerequisites(context)
        except Exception as e:
            self.fail(f"Agent 1 prerequisite validation should pass: {e}")


class TestAgent2Architect(unittest.TestCase):
    """Test Agent 2 - Architect (Architecture Analysis)"""
    
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
    
    def test_agent2_initialization(self):
        """Test Agent 2 initializes correctly"""
        agent = ArchitectAgent()
        
        self.assertEqual(agent.agent_id, 2)
        self.assertEqual(agent.matrix_character, MatrixCharacter.ARCHITECT)
        self.assertEqual(agent.dependencies, [1])  # Depends on Sentinel
        
    def test_agent2_dependency_validation(self):
        """Test Agent 2 validates dependencies correctly"""
        agent = ArchitectAgent()
        
        # Context with Agent 1 results
        context_with_deps = {
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
                'analysis_results': {
                    1: {'binary_info': {'format_type': 'PE'}}
                },
                'binary_metadata': {
                    'discovery': {'binary_info': {'format_type': 'PE'}}
                }
            }
        }
        
        # Should not raise exception with dependencies met
        try:
            agent._validate_prerequisites(context_with_deps)
        except Exception as e:
            self.fail(f"Agent 2 should validate successfully with dependencies: {e}")


class TestAgent3Merovingian(unittest.TestCase):
    """Test Agent 3 - Merovingian (Basic Decompiler)"""
    
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
    
    def test_agent3_initialization(self):
        """Test Agent 3 initializes correctly"""
        agent = MerovingianAgent()
        
        self.assertEqual(agent.agent_id, 3)
        self.assertEqual(agent.matrix_character, MatrixCharacter.MEROVINGIAN)
        self.assertEqual(agent.dependencies, [1])  # Depends on Sentinel
        
    def test_agent3_get_required_context_keys(self):
        """Test Agent 3 defines required context keys"""
        agent = MerovingianAgent()
        required_keys = agent._get_required_context_keys()
        
        self.assertIsInstance(required_keys, list)
        # Agent 3 requires binary_path context, not necessarily agent dependency 1
        self.assertIn('binary_path', required_keys)


class TestAgent4AgentSmith(unittest.TestCase):
    """Test Agent 4 - Agent Smith (Binary Structure Analysis)"""
    
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
    
    def test_agent4_initialization(self):
        """Test Agent 4 initializes correctly"""
        agent = AgentSmithAgent()
        
        self.assertEqual(agent.agent_id, 4)
        self.assertEqual(agent.matrix_character, MatrixCharacter.AGENT_SMITH)
        self.assertEqual(agent.dependencies, [1])  # Depends on Sentinel
        
    def test_agent4_matrix_description(self):
        """Test Agent 4 has proper Matrix description"""
        agent = AgentSmithAgent()
        description = agent.get_matrix_description()
        
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 10)
        self.assertIn("Agent Smith", description)


class TestAgent5Neo(unittest.TestCase):
    """Test Agent 5 - Neo (Advanced Decompiler)"""
    
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
    
    def test_agent5_initialization(self):
        """Test Agent 5 initializes correctly"""
        agent = NeoAgent()
        
        self.assertEqual(agent.agent_id, 5)
        self.assertEqual(agent.matrix_character, MatrixCharacter.NEO)
        self.assertEqual(agent.dependencies, [1, 2, 3])  # Depends on Sentinel, Architect, and Merovingian
        
    def test_agent5_matrix_description(self):
        """Test Agent 5 has proper Matrix description"""
        agent = NeoAgent()
        description = agent.get_matrix_description()
        
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 10)
        self.assertIn("Neo", description)


class TestAgent10Twins(unittest.TestCase):
    """Test Agent 10 - The Twins (Binary Diff Analysis)"""
    
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
    
    def test_agent10_initialization(self):
        """Test Agent 10 initializes correctly"""
        agent = Agent10_Twins_BinaryDiff()
        
        self.assertEqual(agent.agent_id, 10)
        self.assertEqual(agent.matrix_character, MatrixCharacter.TWINS)
        self.assertEqual(agent.dependencies, [9])  # Depends on Agent 9
        
    def test_agent10_binary_diff_analysis(self):
        """Test Agent 10 can perform binary diff analysis"""
        agent = Agent10_Twins_BinaryDiff()
        
        # Test binary diff analysis methods exist
        self.assertTrue(hasattr(agent, '_perform_multilevel_comparison'))
        self.assertTrue(hasattr(agent, '_compare_binary_level'))
        self.assertTrue(hasattr(agent, '_compare_assembly_level'))


class TestAgentErrorHandling(unittest.TestCase):
    """Test agent error handling and validation"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
        self.test_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_dir') and self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_invalid_context_handling(self):
        """Test agents handle invalid context gracefully"""
        agent = SentinelAgent()
        
        # Empty context should be handled gracefully
        empty_context = {}
        
        with self.assertRaises(Exception):
            agent._validate_prerequisites(empty_context)
            
    def test_missing_binary_path_handling(self):
        """Test agents handle missing binary path"""
        agent = SentinelAgent()
        
        context = {
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            }
        }
        
        with self.assertRaises(Exception):
            agent._validate_prerequisites(context)
    
    def test_agent_result_validation(self):
        """Test AgentResult validation"""
        # Valid AgentResult
        valid_result = AgentResult(
            agent_id=1,
            status=AgentStatus.SUCCESS,
            data={'test': 'data'},
            agent_name="TestAgent",
            matrix_character="TestCharacter"
        )
        
        self.assertEqual(valid_result.agent_id, 1)
        self.assertEqual(valid_result.status, AgentStatus.SUCCESS)
        
        # Test that required fields are enforced
        with self.assertRaises(TypeError):
            AgentResult()  # Missing required arguments


class TestAgentDependencyChain(unittest.TestCase):
    """Test agent dependency chain validation"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Required imports not available: {IMPORT_ERROR}")
            
    def test_dependency_chain_definition(self):
        """Test agent dependency chain is properly defined"""
        # Agent 1 - no dependencies
        agent1 = SentinelAgent()
        self.assertEqual(agent1.dependencies, [])
        
        # Agent 2 - depends on Agent 1
        agent2 = ArchitectAgent()
        self.assertEqual(agent2.dependencies, [1])
        
        # Agent 3 - depends on Agent 1
        agent3 = MerovingianAgent()
        self.assertEqual(agent3.dependencies, [1])
        
        # Agent 4 - depends on Agent 1
        agent4 = AgentSmithAgent()
        self.assertEqual(agent4.dependencies, [1])
        
        # Agent 5 - depends on Agents 1,2,3
        agent5 = NeoAgent()
        self.assertEqual(agent5.dependencies, [1, 2, 3])
        
        # Agent 10 - depends on Agent 9
        agent10 = Agent10_Twins_BinaryDiff()
        self.assertEqual(agent10.dependencies, [9])
    
    def test_circular_dependency_check(self):
        """Test no circular dependencies exist"""
        agents = [
            SentinelAgent(),
            ArchitectAgent(),
            MerovingianAgent(),
            AgentSmithAgent(),
            NeoAgent(),
            Agent10_Twins_BinaryDiff()
        ]
        
        # Build dependency graph
        dependencies = {}
        for agent in agents:
            dependencies[agent.agent_id] = agent.dependencies
            
        # Check for circular dependencies (simplified check)
        for agent_id, deps in dependencies.items():
            for dep_id in deps:
                # An agent should not depend on itself
                self.assertNotEqual(agent_id, dep_id, f"Agent {agent_id} cannot depend on itself")
                
                # Dependency should not have the current agent as dependency (simple cycle check)
                if dep_id in dependencies:
                    self.assertNotIn(agent_id, dependencies[dep_id], 
                                   f"Circular dependency between Agent {agent_id} and Agent {dep_id}")


if __name__ == '__main__':
    unittest.main()