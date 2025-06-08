#!/usr/bin/env python3
"""
Test script to validate all 16 Matrix agents can be instantiated successfully
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_agent_instantiation():
    """Test instantiation of all 16 agents"""
    results = {}
    
    # Test each agent
    agents_to_test = [
        (1, "agent01_sentinel", "SentinelAgent"),
        (2, "agent02_architect", "ArchitectAgent"),
        (3, "agent03_merovingian", "MerovingianAgent"),
        (4, "agent04_agent_smith", "AgentSmithAgent"),
        (5, "agent05_neo_advanced_decompiler", "Agent5_Neo_AdvancedDecompiler"),
        (6, "agent06_twins_binary_diff", "Agent6_Twins_BinaryDiff"),
        (7, "agent07_trainman_assembly_analysis", "Agent7_Trainman_AssemblyAnalysis"),
        (8, "agent08_keymaker_resource_reconstruction", "Agent8_Keymaker_ResourceReconstruction"),
        (9, "agent09_commander_locke", "CommanderLockeAgent"),
        (10, "agent10_the_machine", "Agent10_TheMachine"),
        (11, "agent11_the_oracle", "Agent11_TheOracle"),
        (12, "agent12_link", "Agent12_Link"),
        (13, "agent13_agent_johnson", "Agent13_AgentJohnson"),
        (14, "agent14_the_cleaner", "Agent14_TheCleaner"),
        (15, "agent15_analyst", "Agent15_Analyst"),
        (16, "agent16_agent_brown", "Agent16_AgentBrown")
    ]
    
    print("Testing Matrix Agent Instantiation...")
    print("=" * 50)
    
    for agent_id, module_name, class_name in agents_to_test:
        try:
            print(f"Testing Agent {agent_id:02d} ({class_name})...", end="")
            
            # Import the module
            module = __import__(f'core.agents.{module_name}', fromlist=[class_name])
            agent_class = getattr(module, class_name)
            
            # Try to instantiate
            agent_instance = agent_class()
            
            # Check if it has required methods
            required_methods = ['execute', '__init__']
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(agent_instance, method):
                    missing_methods.append(method)
            
            if missing_methods:
                results[agent_id] = {
                    'status': 'partial',
                    'error': f'Missing methods: {missing_methods}',
                    'class_name': class_name
                }
                print(f" ⚠️  PARTIAL (missing: {missing_methods})")
            else:
                results[agent_id] = {
                    'status': 'success',
                    'error': None,
                    'class_name': class_name
                }
                print(" ✅ SUCCESS")
                
        except Exception as e:
            results[agent_id] = {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc(),
                'class_name': class_name
            }
            print(f" ❌ FAILED: {str(e)}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    
    successful = [aid for aid, result in results.items() if result['status'] == 'success']
    partial = [aid for aid, result in results.items() if result['status'] == 'partial']
    failed = [aid for aid, result in results.items() if result['status'] == 'failed']
    
    print(f"✅ Successful: {len(successful)}/16 agents")
    print(f"⚠️  Partial: {len(partial)}/16 agents")
    print(f"❌ Failed: {len(failed)}/16 agents")
    
    if successful:
        print(f"\nSuccessful agents: {successful}")
    
    if partial:
        print(f"\nPartial agents (missing methods): {partial}")
        for agent_id in partial:
            print(f"  Agent {agent_id}: {results[agent_id]['error']}")
    
    if failed:
        print(f"\nFailed agents: {failed}")
        for agent_id in failed:
            print(f"  Agent {agent_id}: {results[agent_id]['error']}")
    
    # Detailed error analysis
    if failed or partial:
        print("\n" + "=" * 50)
        print("DETAILED ERROR ANALYSIS:")
        print("=" * 50)
        
        for agent_id, result in results.items():
            if result['status'] in ['failed', 'partial']:
                print(f"\nAgent {agent_id:02d} ({result['class_name']}):")
                print(f"Status: {result['status'].upper()}")
                print(f"Error: {result['error']}")
                
                if 'traceback' in result:
                    print("Traceback:")
                    print(result['traceback'])
    
    return results

if __name__ == "__main__":
    results = test_agent_instantiation()
    
    # Exit with appropriate code
    failed_count = len([r for r in results.values() if r['status'] == 'failed'])
    if failed_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)