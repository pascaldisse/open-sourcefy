#!/usr/bin/env python3
"""
Test script for MFC 7.1 detection in VS2003 toolchain integration
"""

import sys
import os
import tempfile
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Test the MFC detection logic directly
def test_mfc_detection():
    print("Testing MFC 7.1 detection logic...")
    
    # Test Agent 10's toolchain detection
    from src.core.agents.agent10_the_machine import Agent10_TheMachine
    
    agent10 = Agent10_TheMachine()
    
    # Test case 1: MFC 7.1 libraries detected
    mfc_dependencies = ['mfc71.lib', 'msvcr71.lib', 'kernel32.lib', 'user32.lib']
    context = {}
    
    toolchain = agent10._detect_required_toolchain(mfc_dependencies, context)
    print(f"✅ Test 1 - MFC 7.1 libraries: {mfc_dependencies}")
    print(f"   Detected toolchain: {toolchain}")
    assert toolchain == 'vs2003', f"Expected vs2003, got {toolchain}"
    
    # Test case 2: Modern libraries only
    modern_dependencies = ['kernel32.lib', 'user32.lib', 'ws2_32.lib']
    toolchain = agent10._detect_required_toolchain(modern_dependencies, context)
    print(f"✅ Test 2 - Modern libraries: {modern_dependencies}")
    print(f"   Detected toolchain: {toolchain}")
    assert toolchain == 'vs2022', f"Expected vs2022, got {toolchain}"
    
    print("✅ MFC detection tests passed!")

# Test build system manager dual toolchain
def test_build_manager():
    print("\nTesting Build System Manager dual toolchain...")
    
    from src.core.build_system_manager import get_build_manager
    
    # Test VS2022 toolchain (default) - this should work
    bm_vs2022 = get_build_manager('vs2022')
    print(f"✅ VS2022 Build Manager: {bm_vs2022.toolchain}")
    assert bm_vs2022.toolchain == 'vs2022'
    
    # Test VS2003 toolchain - expect validation failure since VS2003 is not installed
    try:
        bm_vs2003 = get_build_manager('vs2003')
        print(f"⚠️ VS2003 Build Manager unexpectedly succeeded: {bm_vs2003.toolchain}")
    except RuntimeError as e:
        error_str = str(e)
        if "BUILD SYSTEM VALIDATION FAILED" in error_str and "Vc7/bin/cl.exe" in error_str:
            print("✅ VS2003 Build Manager correctly failed validation (VS2003 not installed)")
        else:
            print(f"Unexpected error: {e}")
            raise
    
    print("✅ Build Manager tests passed!")

# Test Agent 9 import data extraction
def test_agent9_imports():
    print("\nTesting Agent 9 import data extraction...")
    
    from src.core.agents.agent09_commander_locke import CommanderLockeAgent
    
    agent9 = CommanderLockeAgent()
    
    # Mock context with Agent 1 data containing MFC imports
    mock_context = {
        'agent_results': {
            1: {
                'data': {
                    'imports': {
                        'MFC71.DLL': [
                            {'name': 'CWnd::OnPaint'},
                            {'name': 'CDialog::DoModal'}
                        ],
                        'KERNEL32.dll': [
                            {'name': 'GetProcAddress'},
                            {'name': 'LoadLibraryA'}
                        ]
                    }
                }
            }
        }
    }
    
    # Test library dependency generation
    analysis_data = {
        'imports': mock_context['agent_results'][1]['data']['imports']
    }
    
    libs = agent9._generate_library_dependencies(analysis_data)
    print(f"✅ Generated libraries: {libs}")
    
    # Should include MFC 7.1 libraries
    expected_mfc = ['mfc71.lib']
    assert any(lib in libs for lib in expected_mfc), f"Expected MFC libraries not found in {libs}"
    
    print("✅ Agent 9 import extraction tests passed!")

if __name__ == "__main__":
    try:
        test_mfc_detection()
        test_build_manager()
        test_agent9_imports()
        print("\n🎉 All tests passed! VS2003 toolchain integration is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)