#!/usr/bin/env python3
"""
Test Matrix agents with realistic source code input
"""

import sys
import tempfile
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_with_realistic_code():
    """Test agents with realistic C source code"""
    print("Testing Matrix Agents with Realistic Source Code")
    print("=" * 50)
    
    try:
        from core.agents_v2 import MATRIX_AGENTS
        from core.agent_base import AgentResult, AgentStatus
        
        # Create realistic source code examples
        realistic_source = {
            'main.c': '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <filename>\\n", argv[0]);
        return 1;
    }
    
    FILE* file = fopen(argv[1], "r");
    if (!file) {
        printf("Error: Could not open file %s\\n", argv[1]);
        return 1;
    }
    
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) {
        printf("%s", buffer);
    }
    
    fclose(file);
    return 0;
}''',
            'utils.c': '''#include "utils.h"
#include <string.h>
#include <stdlib.h>

char* string_duplicate(const char* src) {
    if (!src) return NULL;
    
    size_t len = strlen(src);
    char* dest = malloc(len + 1);
    if (!dest) return NULL;
    
    strcpy(dest, src);
    return dest;
}

int calculate_checksum(const char* data, size_t length) {
    int checksum = 0;
    for (size_t i = 0; i < length; i++) {
        checksum += data[i];
    }
    return checksum;
}''',
            'utils.h': '''#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

char* string_duplicate(const char* src);
int calculate_checksum(const char* data, size_t length);

#endif // UTILS_H'''
        }
        
        # Create test environment
        test_dir = Path(tempfile.mkdtemp(prefix="realistic_test_"))
        output_paths = {
            'agents': str(test_dir / 'agents'),
            'compilation': str(test_dir / 'compilation')
        }
        
        for path in output_paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Create mock previous agent results with realistic source
        mock_results = {}
        mock_results[9] = AgentResult(
            agent_id=9,
            status=AgentStatus.COMPLETED,
            data={
                'reconstructed_source': {
                    'source_files': {
                        'main.c': realistic_source['main.c'],
                        'utils.c': realistic_source['utils.c']
                    },
                    'header_files': {
                        'utils.h': realistic_source['utils.h']
                    }
                }
            }
        )
        
        # Fill in other mock results
        for i in [8, 10, 11, 12]:
            mock_results[i] = AgentResult(
                agent_id=i,
                status=AgentStatus.COMPLETED,
                data={'mock_data': f'agent_{i}'}
            )
        
        context = {
            'agent_results': mock_results,
            'binary_path': 'test.exe',
            'output_paths': output_paths
        }
        
        # Test each agent with realistic input
        test_results = {}
        
        for agent_id in [10, 11, 12, 13, 14]:
            print(f"\n--- Testing Agent {agent_id} with realistic code ---")
            
            agent = MATRIX_AGENTS[agent_id]()
            result = agent.execute(context)
            
            print(f"Status: {result.status}")
            
            if result.status == AgentStatus.COMPLETED:
                print("✓ Successfully processed realistic source code")
                
                # Show key metrics
                if result.metadata:
                    for key, value in result.metadata.items():
                        if key in ['vulnerabilities_found', 'files_cleaned', 'quality_score', 'security_score']:
                            print(f"  {key}: {value}")
                
                # Add to context for next agent
                context['agent_results'][agent_id] = result
                test_results[agent_id] = True
                
            else:
                print(f"✗ Failed: {result.error_message}")
                test_results[agent_id] = False
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        # Summary
        successful = sum(test_results.values())
        total = len(test_results)
        
        print(f"\n{'='*50}")
        print(f"REALISTIC INPUT TEST RESULTS: {successful}/{total} agents passed")
        
        return successful == total
        
    except Exception as e:
        print(f"Realistic input test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_with_realistic_code()
    
    if success:
        print("\n🎉 All Matrix agents handle realistic source code successfully!")
    else:
        print("\n❌ Some agents had issues with realistic input.")
    
    sys.exit(0 if success else 1)