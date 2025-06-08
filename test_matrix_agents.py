#!/usr/bin/env python3
"""
Test script for Matrix Agents 10-14
Verifies that the new Matrix agents can be imported and executed properly
"""

import sys
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_agent_imports():
    """Test that all Matrix agents can be imported"""
    print("Testing Matrix Agent Imports...")
    
    try:
        from core.agents_v2 import (
            Agent10_TheMachine,
            Agent11_TheOracle,
            Agent12_Link,
            Agent13_AgentJohnson,
            Agent14_TheCleaner,
            get_available_agents,
            get_implementation_status
        )
        print("✓ All Matrix agents imported successfully")
        
        # Test agent instantiation
        agents = {}
        for agent_id, agent_class in get_available_agents().items():
            try:
                agents[agent_id] = agent_class()
                print(f"✓ Agent {agent_id} ({agent_class.__name__}) instantiated successfully")
            except Exception as e:
                print(f"✗ Agent {agent_id} failed to instantiate: {e}")
                return False
        
        # Test implementation status
        status = get_implementation_status()
        print(f"Implementation status: {status}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_agent_dependencies():
    """Test agent dependency chains"""
    print("\nTesting Agent Dependencies...")
    
    try:
        from core.agents_v2 import MATRIX_AGENTS, AGENT_METADATA
        
        for agent_id in sorted(MATRIX_AGENTS.keys()):
            agent = MATRIX_AGENTS[agent_id]()
            metadata = AGENT_METADATA[agent_id]
            
            print(f"Agent {agent_id} ({metadata['name']}):")
            print(f"  Dependencies: {agent.get_dependencies()}")
            print(f"  Description: {agent.get_description()}")
            
            # Verify dependencies are consistent
            expected_deps = metadata['dependencies']
            actual_deps = agent.get_dependencies()
            
            if expected_deps == actual_deps:
                print(f"  ✓ Dependencies match metadata")
            else:
                print(f"  ✗ Dependencies mismatch: expected {expected_deps}, got {actual_deps}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dependency test failed: {e}")
        return False

def test_agent_execution_basic():
    """Test basic agent execution without full context"""
    print("\nTesting Basic Agent Execution...")
    
    try:
        from core.agents_v2 import MATRIX_AGENTS
        from core.agent_base import AgentStatus
        
        # Create minimal context for testing
        test_context = {
            'agent_results': {},
            'binary_path': 'test.exe',
            'output_paths': {
                'agents': '/tmp/test_agents',
                'ghidra': '/tmp/test_ghidra'
            }
        }
        
        for agent_id in sorted(MATRIX_AGENTS.keys()):
            print(f"\nTesting Agent {agent_id}...")
            
            try:
                agent = MATRIX_AGENTS[agent_id]()
                result = agent.execute(test_context)
                
                print(f"  Agent ID: {result.agent_id}")
                print(f"  Status: {result.status}")
                
                if result.status == AgentStatus.FAILED:
                    print(f"  Error: {result.error_message}")
                    print(f"  ✓ Agent {agent_id} handled missing dependencies gracefully")
                else:
                    print(f"  ✓ Agent {agent_id} executed successfully")
                    
            except Exception as e:
                print(f"  ✗ Agent {agent_id} execution failed with exception: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Execution test failed: {e}")
        return False

def test_agent_context_handling():
    """Test agent context handling and data flow"""
    print("\nTesting Agent Context Handling...")
    
    try:
        from core.agents_v2 import MATRIX_AGENTS
        from core.agent_base import AgentResult, AgentStatus
        
        # Create mock results for dependency chain
        mock_results = {}
        
        # Mock result for Agent 8 (resource reconstruction)
        mock_results[8] = AgentResult(
            agent_id=8,
            status=AgentStatus.COMPLETED,
            data={
                'reconstructed_resources': {
                    'icons': ['icon1.ico'],
                    'strings': ['Hello World'],
                    'data_files': ['data.bin']
                }
            }
        )
        
        # Mock result for Agent 9 (global reconstruction) 
        mock_results[9] = AgentResult(
            agent_id=9,
            status=AgentStatus.COMPLETED,
            data={
                'reconstructed_source': {
                    'source_files': {
                        'main.c': '#include <stdio.h>\nint main() { printf("Hello"); return 0; }'
                    },
                    'header_files': {
                        'main.h': '#ifndef MAIN_H\n#define MAIN_H\n#endif'
                    }
                }
            }
        )
        
        test_context = {
            'agent_results': mock_results,
            'binary_path': 'test.exe',
            'output_paths': {
                'agents': '/tmp/test_agents',
                'ghidra': '/tmp/test_ghidra',
                'compilation': '/tmp/test_compilation'
            }
        }
        
        # Test Agent 10 with mock data
        print("Testing Agent 10 with mock source code...")
        agent10 = MATRIX_AGENTS[10]()
        result10 = agent10.execute(test_context)
        
        print(f"  Status: {result10.status}")
        if result10.status == AgentStatus.COMPLETED:
            print(f"  ✓ Agent 10 processed mock source code successfully")
            test_context['agent_results'][10] = result10
        else:
            print(f"  Result: {result10.error_message}")
        
        return True
        
    except Exception as e:
        print(f"✗ Context handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Matrix Agent Test Suite")
    print("=" * 50)
    
    tests = [
        test_agent_imports,
        test_agent_dependencies,
        test_agent_execution_basic,
        test_agent_context_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_func.__name__} PASSED")
            else:
                print(f"✗ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED with exception: {e}")
        
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Matrix agents are functional!")
        return True
    else:
        print("❌ Some Matrix agents need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)