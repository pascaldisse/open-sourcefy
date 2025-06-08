#!/usr/bin/env python3
"""
Comprehensive functional test for Matrix Agents 5-9
Tests actual agent instantiation and execution with proper context
"""

import sys
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def create_test_context(binary_path: str) -> Dict[str, Any]:
    """Create a comprehensive test context for Matrix agent execution"""
    test_dir = Path(tempfile.mkdtemp(prefix="matrix_agents_test_"))
    
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
    
    # Create comprehensive mock results from dependency agents
    from core.agent_base import AgentResult, AgentStatus
    
    mock_results = {}
    
    # Mock Agent 1 (Binary Discovery)
    mock_results[1] = AgentResult(
        agent_id=1,
        status=AgentStatus.COMPLETED,
        data={
            'file_type': 'PE32 executable (console) Intel 80386, for MS Windows',
            'architecture': 'x86',
            'file_size': 5342208,
            'entropy': 7.2,
            'sections': [
                {'name': '.text', 'virtual_address': 0x1000, 'size': 0x4A0000},
                {'name': '.data', 'virtual_address': 0x4A1000, 'size': 0x12000},
                {'name': '.rdata', 'virtual_address': 0x4B3000, 'size': 0x8000}
            ],
            'imports': ['kernel32.dll', 'user32.dll', 'advapi32.dll', 'msvcrt.dll'],
            'exports': [],
            'strings': [
                'Matrix Online Launcher',
                'Connecting to server...',
                'Failed to connect',
                'User authentication required',
                'Loading game data...'
            ],
            'format_info': {
                'format': 'PE',
                'subsystem': 'Console',
                'linker_version': '14.0',
                'timestamp': 1234567890
            }
        },
        metadata={
            'agent_name': 'Sentinel_BinaryDiscovery',
            'execution_time': 2.5,
            'confidence': 0.95
        }
    )
    
    # Mock Agent 2 (Architecture Analysis)
    mock_results[2] = AgentResult(
        agent_id=2,
        status=AgentStatus.COMPLETED,
        data={
            'architecture': 'x86',
            'instruction_set': 'x86-32',
            'calling_convention': 'stdcall',
            'compiler_hints': ['Microsoft Visual C++', '.NET Framework'],
            'optimization_level': 'O2',
            'endianness': 'little',
            'pointer_size': 4,
            'stack_analysis': {
                'stack_frame_setup': 'standard',
                'stack_alignment': 4
            }
        },
        metadata={
            'agent_name': 'Architect_ArchAnalysis',
            'execution_time': 1.8,
            'confidence': 0.90
        }
    )
    
    # Mock Agent 4 (Binary Structure Analysis) - dependency for some agents
    mock_results[4] = AgentResult(
        agent_id=4,
        status=AgentStatus.COMPLETED,
        data={
            'binary_structure': {
                'functions': 150,
                'loops': 45,
                'conditionals': 89,
                'data_structures': 12
            },
            'complexity_metrics': {
                'cyclomatic_complexity': 5.2,
                'average_function_size': 85,
                'max_nesting_depth': 6
            },
            'analysis_quality': 0.85,
            'patterns_detected': ['matrix_authentication', 'network_protocol', 'game_engine']
        },
        metadata={
            'agent_name': 'AgentSmith_BinaryStructure',
            'execution_time': 4.2,
            'confidence': 0.82
        }
    )
    
    context = {
        'global_data': {
            'binary_path': binary_path,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
            'session_id': 'matrix_test_session'
        },
        'agent_results': mock_results,
        'output_paths': {str(k): str(v) for k, v in output_paths.items()}
    }
    
    return context, test_dir

def test_agent_execution(agent_class, agent_id: int, agent_name: str, context: Dict[str, Any]):
    """Test execution of a single Matrix agent"""
    print(f"\n--- Testing Agent {agent_id}: {agent_name} ---")
    
    try:
        # Create agent instance
        start_time = time.time()
        agent = agent_class()
        init_time = time.time() - start_time
        
        print(f"✓ Agent {agent_id} created: {getattr(agent, 'name', 'Unknown')}")
        print(f"  Dependencies: {getattr(agent, 'dependencies', [])}")
        print(f"  Initialization time: {init_time:.2f}s")
        
        # Execute agent
        print(f"  Executing agent...")
        exec_start = time.time()
        result = agent.execute(context)
        exec_time = time.time() - exec_start
        
        print(f"✓ Agent {agent_id} executed in {exec_time:.2f}s")
        print(f"  Status: {result.status.value}")
        
        if result.status.value == 'completed':
            print(f"  Data keys: {list(result.data.keys()) if result.data else 'None'}")
            
            if result.metadata:
                for key, value in result.metadata.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, str) and len(value) < 50:
                        print(f"  {key}: {value}")
            
            # Check for key outputs
            if hasattr(result, 'data') and result.data:
                data_summary = {}
                for key, value in result.data.items():
                    if isinstance(value, list):
                        data_summary[key] = f"list({len(value)} items)"
                    elif isinstance(value, dict):
                        data_summary[key] = f"dict({len(value)} keys)"
                    elif isinstance(value, (int, float)):
                        data_summary[key] = value
                    else:
                        data_summary[key] = f"{type(value).__name__}"
                
                print(f"  Data summary: {data_summary}")
            
        else:
            print(f"  ✗ Error: {result.error_message}")
        
        return result
        
    except Exception as e:
        print(f"✗ Agent {agent_id} failed: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return None

def run_matrix_agents_test():
    """Run comprehensive test of Matrix agents 5-9"""
    print("Matrix Agents 5-9 Comprehensive Functional Test")
    print("=" * 60)
    
    # Use a real binary path for testing
    binary_path = str(Path(__file__).parent / "launcher.exe")
    if not Path(binary_path).exists():
        # Try alternative paths
        alt_paths = [
            "input/launcher.exe",
            "launcher-new-final",
            "test.exe"
        ]
        for alt_path in alt_paths:
            if Path(alt_path).exists():
                binary_path = str(Path(alt_path).absolute())
                break
        else:
            print(f"✗ No test binary found. Tried: {[binary_path] + alt_paths}")
            return False
    
    print(f"✓ Test binary: {binary_path}")
    
    # Create test context
    context, test_dir = create_test_context(binary_path)
    print(f"✓ Test environment: {test_dir}")
    
    # Import Matrix agents
    try:
        from core.agents.agent05_neo_advanced_decompiler import Agent5_Neo_AdvancedDecompiler
        from core.agents.agent06_twins_binary_diff import Agent6_Twins_BinaryDiff
        from core.agents.agent07_trainman_assembly_analysis import Agent7_Trainman_AssemblyAnalysis
        from core.agents.agent08_keymaker_resource_reconstruction import Agent8_Keymaker_ResourceReconstruction
        from core.agents.agent09_commander_locke import CommanderLockeAgent
        print("✓ All Matrix agents imported successfully")
    except Exception as e:
        print(f"✗ Failed to import agents: {e}")
        return False
    
    # Test agents in dependency order
    agents_to_test = [
        (5, "Neo (Advanced Decompiler)", Agent5_Neo_AdvancedDecompiler),
        (6, "Twins (Binary Diff)", Agent6_Twins_BinaryDiff),
        (7, "Trainman (Assembly Analysis)", Agent7_Trainman_AssemblyAnalysis),
        (8, "Keymaker (Resource Reconstruction)", Agent8_Keymaker_ResourceReconstruction),
        (9, "Commander Locke (Global Reconstruction)", CommanderLockeAgent),
    ]
    
    results = {}
    success_count = 0
    total_execution_time = 0
    
    for agent_id, agent_name, agent_class in agents_to_test:
        result = test_agent_execution(agent_class, agent_id, agent_name, context)
        results[agent_id] = result
        
        if result and result.status.value == 'completed':
            success_count += 1
            # Add result to context for dependent agents
            context['agent_results'][agent_id] = result
            
            # Track execution time
            if hasattr(result, 'metadata') and result.metadata:
                exec_time = result.metadata.get('execution_time', 0)
                if isinstance(exec_time, (int, float)):
                    total_execution_time += exec_time
    
    # Generate comprehensive summary
    print(f"\n{'=' * 60}")
    print(f"MATRIX AGENTS 5-9 COMPREHENSIVE TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"✓ Agents tested: {len(agents_to_test)}")
    print(f"✓ Agents successful: {success_count}")
    print(f"✓ Success rate: {success_count/len(agents_to_test)*100:.1f}%")
    print(f"✓ Total execution time: {total_execution_time:.2f}s")
    print(f"✓ Average time per agent: {total_execution_time/max(success_count, 1):.2f}s")
    
    # Detailed results per agent
    print(f"\nDetailed Results:")
    for agent_id, agent_name, _ in agents_to_test:
        result = results[agent_id]
        if result:
            status_icon = "✓" if result.status.value == 'completed' else "✗"
            exec_time = result.metadata.get('execution_time', 0) if result.metadata else 0
            print(f"{status_icon} Agent {agent_id} ({agent_name.split('(')[0].strip()}): {result.status.value} ({exec_time:.2f}s)")
            
            if result.status.value == 'failed' and result.error_message:
                print(f"    Error: {result.error_message}")
        else:
            print(f"✗ Agent {agent_id} ({agent_name.split('(')[0].strip()}): failed to execute")
    
    # Test environment info
    print(f"\n✓ Test artifacts saved to: {test_dir}")
    print(f"✓ Agent outputs available in: {test_dir}/output/agents/")
    
    if success_count == len(agents_to_test):
        print(f"\n🎉 ALL MATRIX AGENTS 5-9 ARE FULLY FUNCTIONAL! 🎉")
        print(f"The Matrix reconstruction pipeline is ready for production.")
        return True
    else:
        failed_count = len(agents_to_test) - success_count
        print(f"\n⚠️  {failed_count} agents need attention")
        print(f"Pipeline completion: {success_count}/{len(agents_to_test)} agents functional")
        return False

if __name__ == "__main__":
    success = run_matrix_agents_test()
    sys.exit(0 if success else 1)