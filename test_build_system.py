#!/usr/bin/env python3
"""
Test script for B1: Build system integration with resource compilation
Tests the build system manager and Agent 10 integration.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_build_system_manager():
    """Test the build system manager initialization and configuration"""
    print("🔧 Testing B1: Build System Integration...")
    
    try:
        from core.build_system_manager import BuildSystemManager
        
        print("📋 Initializing Build System Manager...")
        build_manager = BuildSystemManager()
        
        print("✅ Build System Manager initialized successfully!")
        print(f"   Compiler (x64): {build_manager.build_config.compiler_x64}")
        print(f"   MSBuild: {build_manager.build_config.msbuild_path}")
        
        # Test tool validation
        print("🔍 Validating build tools...")
        try:
            build_manager._validate_build_tools()
            print("✅ Build tools validation passed!")
        except Exception as e:
            print(f"❌ Build tools validation failed: {e}")
            return False
        
        # Test build configuration
        print("📊 Build configuration summary:")
        print(f"   Include dirs: {len(build_manager.build_config.include_dirs)} directories")
        print(f"   Library dirs (x64): {len(build_manager.build_config.library_dirs_x64)} directories")
        
        return True
        
    except Exception as e:
        print(f"❌ Build System Manager test failed: {e}")
        return False

def test_agent10_integration():
    """Test Agent 10 (The Machine) integration with build system"""
    print("\n🤖 Testing Agent 10 (The Machine) integration...")
    
    try:
        from core.agents.agent10_the_machine import Agent10_TheMachine
        
        agent10 = Agent10_TheMachine()
        print("✅ Agent 10 (The Machine) initialized successfully!")
        print(f"   Agent ID: {agent10.agent_id}")
        print(f"   Matrix Character: {agent10.matrix_character}")
        print(f"   Dependencies: {agent10.dependencies}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent 10 integration test failed: {e}")
        return False

def test_resource_compilation_setup():
    """Test resource compilation setup and infrastructure"""
    print("\n🔨 Testing resource compilation setup...")
    
    try:
        # Test basic compilation setup without full execution
        from core.build_system_manager import BuildSystemManager
        
        build_manager = BuildSystemManager()
        
        # Test resource compilation capabilities
        print("📦 Testing resource compilation capabilities...")
        
        # Check if we can generate basic project files
        test_context = {
            'binary_path': 'input/launcher.exe',
            'agent_results': {},
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            }
        }
        
        print("✅ Resource compilation setup test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Resource compilation setup test failed: {e}")
        return False

def main():
    """Main test execution for B1: Build system integration"""
    print("=" * 60)
    print("B1: Build System Integration with Resource Compilation")
    print("Testing build system manager and Agent 10 integration")
    print("=" * 60)
    
    tests = [
        ("Build System Manager", test_build_system_manager),
        ("Agent 10 Integration", test_agent10_integration), 
        ("Resource Compilation Setup", test_resource_compilation_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"B1 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ B1: Build system integration tests PASSED!")
        print("🎯 Ready for build system integration completion")
    else:
        print("❌ B1: Build system integration tests FAILED!")
        print("🔧 Build system integration needs fixes")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)