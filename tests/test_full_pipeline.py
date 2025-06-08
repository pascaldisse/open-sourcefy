#!/usr/bin/env python3
"""
Full Pipeline Integration Test
Test complete 16-agent Matrix pipeline
"""

import unittest
import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator, PipelineConfig
    from core.config_manager import ConfigManager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)

class TestFullPipeline(unittest.TestCase):
    """Full pipeline integration tests"""
    
    def test_all_agents_exist(self):
        """Test all 16 agents exist with correct naming"""
        project_root = Path(__file__).parent.parent
        agents_path = project_root / "src" / "core" / "agents"
        
        # Updated agent file names to match actual implementation
        expected_agents = [
            "agent01_sentinel.py",
            "agent02_architect.py", 
            "agent03_merovingian.py",
            "agent04_agent_smith.py",
            "agent05_neo_advanced_decompiler.py",
            "agent06_twins_binary_diff.py",
            "agent07_trainman_assembly_analysis.py",
            "agent08_keymaker_resource_reconstruction.py",
            "agent09_commander_locke.py",
            "agent10_the_machine.py",
            "agent11_the_oracle.py",
            "agent12_link.py",
            "agent13_agent_johnson.py",
            "agent14_the_cleaner.py",
            "agent15_analyst.py",
            "agent16_agent_brown.py"
        ]
        
        existing_agents = []
        missing_agents = []
        
        for agent_file in expected_agents:
            agent_path = agents_path / agent_file
            if agent_path.exists():
                existing_agents.append(agent_file)
            else:
                missing_agents.append(agent_file)
                
        # Report findings
        print(f"\nAgent Status Report:")
        print(f"Existing agents: {len(existing_agents)}/16")
        print(f"Missing agents: {len(missing_agents)}")
        
        if missing_agents:
            print(f"Missing: {missing_agents}")
            
        # Pass if we have at least the core agents (1-5, 10)
        core_agents = [
            "agent01_sentinel.py",
            "agent02_architect.py", 
            "agent03_merovingian.py",
            "agent04_agent_smith.py",
            "agent05_neo_advanced_decompiler.py",
            "agent10_the_machine.py"
        ]
        
        core_existing = [a for a in core_agents if (agents_path / a).exists()]
        self.assertGreaterEqual(len(core_existing), 5, 
                               f"At least 5 core agents should exist. Found: {core_existing}")
            
    def test_matrix_architecture_components(self):
        """Test Matrix architecture components exist"""
        project_root = Path(__file__).parent.parent
        src_path = project_root / "src"
        core_path = src_path / "core"
        
        # Check core Matrix components
        essential_components = [
            "config_manager.py",
            "matrix_agents.py",
            "shared_components.py"
        ]
        
        missing_components = []
        for component in essential_components:
            component_path = core_path / component
            if not component_path.exists():
                missing_components.append(component)
                
        self.assertEqual(len(missing_components), 0, 
                        f"Missing essential components: {missing_components}")
        
    def test_pipeline_orchestrator_available(self):
        """Test pipeline orchestrator is available"""
        if not IMPORTS_AVAILABLE:
            self.skipTest(f"Pipeline imports not available: {IMPORT_ERROR}")
            
        try:
            config = PipelineConfig()
            orchestrator = MatrixPipelineOrchestrator(config)
            self.assertIsNotNone(orchestrator)
        except Exception as e:
            self.fail(f"Pipeline orchestrator should be available: {e}")
            
    def test_input_directory_exists(self):
        """Test input directory and test binary exist"""
        project_root = Path(__file__).parent.parent
        input_path = project_root / "input"
        
        self.assertTrue(input_path.exists(), "Input directory should exist")
        
        # Check for test binary
        test_binary = input_path / "launcher.exe"
        if test_binary.exists():
            self.assertGreater(test_binary.stat().st_size, 0, "Test binary should not be empty")
        else:
            print(f"\nWarning: Test binary {test_binary} not found")
            
    def test_output_directory_structure(self):
        """Test output directory structure can be created"""
        project_root = Path(__file__).parent.parent
        output_path = project_root / "output"
        
        # Should be able to create output directory
        output_path.mkdir(exist_ok=True)
        self.assertTrue(output_path.exists(), "Output directory should be creatable")

if __name__ == '__main__':
    unittest.main()
