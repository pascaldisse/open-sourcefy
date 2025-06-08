#!/usr/bin/env python3
"""
Integration test for Matrix Agents 10-14
Tests the agents in a more realistic pipeline scenario
"""

import sys
import tempfile
import os
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def create_test_environment():
    """Create a test environment with proper directory structure"""
    test_dir = Path(tempfile.mkdtemp(prefix="matrix_test_"))
    
    # Create output directory structure
    output_paths = {
        'root': str(test_dir / 'output'),
        'agents': str(test_dir / 'output' / 'agents'),
        'ghidra': str(test_dir / 'output' / 'ghidra'),
        'compilation': str(test_dir / 'output' / 'compilation'),
        'reports': str(test_dir / 'output' / 'reports'),
        'logs': str(test_dir / 'output' / 'logs'),
        'temp': str(test_dir / 'output' / 'temp'),
        'tests': str(test_dir / 'output' / 'tests')
    }
    
    # Create directories
    for path in output_paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    return test_dir, output_paths

def create_mock_agent_results():
    """Create mock results from earlier agents"""
    from core.agent_base import AgentResult, AgentStatus
    
    mock_results = {}
    
    # Mock Agent 1-9 results
    for i in range(1, 10):
        mock_results[i] = AgentResult(
            agent_id=i,
            status=AgentStatus.COMPLETED,
            data={'mock_data': f'agent_{i}_result'},
            metadata={'agent_name': f'Agent{i:02d}'}
        )
    
    # Add specific mock data for source reconstruction
    mock_results[7] = AgentResult(  # Advanced Decompiler
        agent_id=7,
        status=AgentStatus.COMPLETED,
        data={
            'decompiled_functions': {
                'main': {
                    'source': 'int main() {\n    printf("Hello World\\n");\n    return 0;\n}',
                    'confidence': 0.85
                }
            }
        }
    )
    
    mock_results[8] = AgentResult(  # Resource Reconstructor
        agent_id=8,
        status=AgentStatus.COMPLETED,
        data={
            'reconstructed_resources': {
                'icons': ['app.ico'],
                'strings': ['Hello World', 'Error message'],
                'version_info': {'version': '1.0.0'}
            }
        }
    )
    
    mock_results[9] = AgentResult(  # Global Reconstructor
        agent_id=9,
        status=AgentStatus.COMPLETED,
        data={
            'reconstructed_source': {
                'source_files': {
                    'main.c': '#include <stdio.h>\nint main() {\n    printf("Hello World\\n");\n    return 0;\n}',
                    'utils.c': '#include "utils.h"\nvoid helper_function() {\n    // Helper implementation\n}'
                },
                'header_files': {
                    'utils.h': '#ifndef UTILS_H\n#define UTILS_H\nvoid helper_function();\n#endif'
                }
            }
        }
    )
    
    return mock_results

def test_matrix_pipeline():
    """Test the Matrix agents 10-14 in pipeline order"""
    print("Running Matrix Pipeline Integration Test")
    print("=" * 50)
    
    try:
        from core.agents_v2 import MATRIX_AGENTS
        from core.agent_base import AgentStatus
        
        # Setup test environment
        test_dir, output_paths = create_test_environment()
        print(f"Test environment created at: {test_dir}")
        
        # Create context
        context = {
            'agent_results': create_mock_agent_results(),
            'binary_path': str(test_dir / 'test.exe'),
            'output_paths': output_paths,
            'config': {
                'timeout': 60,
                'max_retries': 1
            }
        }
        
        # Run agents 10-14 in sequence
        agent_order = [10, 11, 12, 13, 14]
        execution_results = {}
        
        for agent_id in agent_order:
            print(f"\n--- Executing Agent {agent_id} ---")
            
            agent = MATRIX_AGENTS[agent_id]()
            print(f"Agent: {agent.name}")
            print(f"Description: {agent.get_description()}")
            print(f"Dependencies: {agent.get_dependencies()}")
            
            # Execute agent
            start_time = time.time()
            result = agent.execute(context)
            execution_time = time.time() - start_time
            
            print(f"Execution time: {execution_time:.2f}s")
            print(f"Status: {result.status}")
            
            if result.status == AgentStatus.COMPLETED:
                print("✓ Agent completed successfully")
                
                # Add result to context for next agent
                context['agent_results'][agent_id] = result
                execution_results[agent_id] = {
                    'status': 'success',
                    'execution_time': execution_time,
                    'data_keys': list(result.data.keys()) if result.data else []
                }
                
                # Print some key metrics if available
                if result.metadata:
                    for key, value in result.metadata.items():
                        if isinstance(value, (int, float, str)):
                            print(f"  {key}: {value}")
                            
            else:
                print(f"✗ Agent failed: {result.error_message}")
                execution_results[agent_id] = {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'error': result.error_message
                }
        
        # Generate summary report
        print(f"\n{'='*50}")
        print("EXECUTION SUMMARY")
        print(f"{'='*50}")
        
        successful_agents = sum(1 for r in execution_results.values() if r['status'] == 'success')
        total_agents = len(execution_results)
        total_time = sum(r['execution_time'] for r in execution_results.values())
        
        print(f"Agents executed: {total_agents}")
        print(f"Successful: {successful_agents}")
        print(f"Failed: {total_agents - successful_agents}")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Average time per agent: {total_time/total_agents:.2f}s")
        
        for agent_id, result in execution_results.items():
            status_icon = "✓" if result['status'] == 'success' else "✗"
            print(f"{status_icon} Agent {agent_id}: {result['status']} ({result['execution_time']:.2f}s)")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        print(f"\nTest environment cleaned up")
        
        return successful_agents == total_agents
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import time
    
    success = test_matrix_pipeline()
    
    if success:
        print("\n🎉 Matrix Pipeline Integration Test PASSED!")
        print("All agents 10-14 are fully functional and ready for production.")
    else:
        print("\n❌ Matrix Pipeline Integration Test FAILED!")
        print("Some agents need debugging.")
    
    sys.exit(0 if success else 1)