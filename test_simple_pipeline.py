#!/usr/bin/env python3
"""
Simple pipeline test for Matrix Agents 5-9
Tests basic execution flow with minimal dependencies
"""

import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def create_test_context(binary_path: str) -> Dict[str, Any]:
    """Create a minimal test context for agent execution"""
    test_dir = Path(tempfile.mkdtemp(prefix="simple_test_"))
    
    # Create output directory structure
    output_paths = {
        'root': test_dir / 'output',
        'agents': test_dir / 'output' / 'agents',
        'ghidra': test_dir / 'output' / 'ghidra',
        'compilation': test_dir / 'output' / 'compilation',
        'reports': test_dir / 'output' / 'reports',
        'logs': test_dir / 'output' / 'logs',
        'temp': test_dir / 'output' / 'temp',
        'tests': test_dir / 'output' / 'tests'
    }
    
    # Create directories
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Create mock results for dependency agents
    from core.agent_base import AgentResult, AgentStatus
    
    mock_results = {}
    
    # Mock Agent 1 (Binary Discovery)
    mock_results[1] = AgentResult(
        agent_id=1,
        status=AgentStatus.COMPLETED,
        data={
            'file_type': 'PE32 executable',
            'architecture': 'x86',
            'file_size': 5000000,
            'entropy': 7.5,
            'sections': ['.text', '.data', '.rdata'],
            'imports': ['kernel32.dll', 'user32.dll'],
            'strings': ['hello world', 'error message']
        },
        metadata={'agent_name': 'Sentinel_BinaryDiscovery'}
    )
    
    # Mock Agent 2 (Architecture Analysis)
    mock_results[2] = AgentResult(
        agent_id=2,
        status=AgentStatus.COMPLETED,
        data={
            'architecture': 'x86',
            'instruction_set': 'x86-32',
            'calling_convention': 'stdcall',
            'compiler_hints': ['MSVC', '.NET Framework'],
            'optimization_level': 'O2'
        },
        metadata={'agent_name': 'Architect_ArchAnalysis'}
    )
    
    # Mock Agent 4 (if needed as dependency)
    mock_results[4] = AgentResult(
        agent_id=4,
        status=AgentStatus.COMPLETED,
        data={
            'binary_structure': {'functions': 150, 'loops': 45},
            'complexity_metrics': {'average': 5.2},
            'analysis_quality': 0.85
        },
        metadata={'agent_name': 'AgentSmith_BinaryStructure'}
    )
    
    context = {
        'global_data': {
            'binary_path': binary_path,
            'timestamp': '20250608_test',
            'session_id': 'simple_test'
        },
        'agent_results': mock_results,
        'output_paths': {str(k): str(v) for k, v in output_paths.items()}
    }
    
    return context, test_dir

def test_agent_execution(agent_class, agent_id: int, context: Dict[str, Any]):
    """Test execution of a single agent"""
    print(f"\n--- Testing Agent {agent_id} ---")
    
    try:
        # Create agent instance
        agent = agent_class()
        print(f"✓ Agent {agent_id} created: {getattr(agent, 'name', 'Unknown')}")
        print(f"  Dependencies: {getattr(agent, 'dependencies', [])}")
        
        # Execute agent
        result = agent.execute(context)
        print(f"✓ Agent {agent_id} executed with status: {result.status.value}")
        
        if result.status.value == 'completed':
            print(f"  Data keys: {list(result.data.keys()) if result.data else 'None'}")
            if result.metadata:
                exec_time = result.metadata.get('execution_time', 'N/A')
                print(f"  Execution time: {exec_time}")
        else:
            print(f"  Error: {result.error_message}")
        
        return result
        
    except Exception as e:
        print(f"✗ Agent {agent_id} failed: {e}")
        return None

def run_simple_pipeline_test():
    """Run a simple test of Matrix agents 5-9"""
    print("Simple Matrix Pipeline Test")
    print("=" * 50)
    
    # Use the test binary
    binary_path = str(Path(__file__).parent / "input" / "launcher.exe")
    if not Path(binary_path).exists():
        print(f"✗ Test binary not found: {binary_path}")
        return False
    
    print(f"✓ Test binary found: {binary_path}")
    
    # Create test context
    context, test_dir = create_test_context(binary_path)
    print(f"✓ Test environment: {test_dir}")
    
    # Import our working agents
    try:
        from core.agents.agent05_neo_advanced_decompiler import Agent5_Neo_AdvancedDecompiler
        from core.agents.agent06_twins_binary_diff import Agent6_Twins_BinaryDiff
        from core.agents.agent07_trainman_assembly_analysis import Agent7_Trainman_AssemblyAnalysis
        from core.agents.agent08_keymaker_resource_reconstruction import Agent8_Keymaker_ResourceReconstruction
        print("✓ All Matrix agents imported successfully")
    except Exception as e:
        print(f"✗ Failed to import agents: {e}")
        return False
    
    # Test agents in dependency order
    agents_to_test = [
        (5, Agent5_Neo_AdvancedDecompiler),
        (6, Agent6_Twins_BinaryDiff),  # Depends on 5
        (7, Agent7_Trainman_AssemblyAnalysis),  # Depends on 5
        (8, Agent8_Keymaker_ResourceReconstruction),  # Depends on 5,7
    ]
    
    results = {}
    success_count = 0
    
    for agent_id, agent_class in agents_to_test:
        result = test_agent_execution(agent_class, agent_id, context)
        results[agent_id] = result
        
        if result and result.status.value == 'completed':
            success_count += 1
            # Add result to context for dependent agents
            context['agent_results'][agent_id] = result
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Simple Pipeline Test Results:")
    print(f"✓ Agents tested: {len(agents_to_test)}")
    print(f"✓ Agents successful: {success_count}")
    print(f"✓ Success rate: {success_count/len(agents_to_test)*100:.1f}%")
    
    if success_count == len(agents_to_test):
        print("\n🎉 ALL MATRIX AGENTS 5-8 ARE WORKING! 🎉")
        return True
    else:
        print(f"\n⚠️  {len(agents_to_test) - success_count} agents need attention")
        return False

if __name__ == "__main__":
    success = run_simple_pipeline_test()
    sys.exit(0 if success else 1)