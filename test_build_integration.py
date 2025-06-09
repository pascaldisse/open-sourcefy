#!/usr/bin/env python3
"""
Comprehensive Build System Integration Test for B1 Completion
Tests actual compilation functionality with the build system.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_sample_compilation():
    """Test actual source code compilation using the build system"""
    print("🔨 Testing actual source code compilation...")
    
    try:
        from core.build_system_manager import BuildSystemManager
        
        # Create build system manager
        build_manager = BuildSystemManager()
        print("✅ Build System Manager initialized")
        
        # Create test source file
        test_dir = Path("output/test_compilation")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        source_file = test_dir / "test_hello.c"
        output_file = test_dir / "test_hello.exe"
        
        # Write simple test program
        test_code = '''#include <stdio.h>

int main() {
    printf("Hello from Open-Sourcefy Build System!\\n");
    printf("Matrix Pipeline compilation test successful.\\n");
    return 0;
}
'''
        
        with open(source_file, 'w') as f:
            f.write(test_code)
        
        print(f"📝 Created test source: {source_file}")
        
        # Test compilation
        print("🔧 Compiling test source...")
        success, output = build_manager.compile_source(
            source_file, output_file, architecture="x64", configuration="Release"
        )
        
        if success:
            print("✅ Compilation successful!")
            print(f"   Output file: {output_file}")
            
            # Verify output file exists
            if output_file.exists():
                print(f"✅ Output executable created: {output_file.stat().st_size} bytes")
                return True
            else:
                print("❌ Output executable not found")
                return False
        else:
            print(f"❌ Compilation failed: {output}")
            return False
            
    except Exception as e:
        print(f"❌ Compilation test failed: {e}")
        return False

def test_msbuild_project_generation():
    """Test MSBuild project generation functionality"""
    print("\n📋 Testing MSBuild project generation...")
    
    try:
        from core.build_system_manager import BuildSystemManager
        
        build_manager = BuildSystemManager()
        
        # Create test project directory
        project_dir = Path("output/test_msbuild_project")
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multiple test source files
        source_files = []
        
        # Main source file
        main_source = project_dir / "main.c"
        main_code = '''#include <stdio.h>
#include "utils.h"

int main() {
    printf("Matrix Build System Test Project\\n");
    test_function();
    return 0;
}
'''
        with open(main_source, 'w') as f:
            f.write(main_code)
        source_files.append(main_source)
        
        # Utility source file
        utils_source = project_dir / "utils.c"
        utils_code = '''#include <stdio.h>
#include "utils.h"

void test_function() {
    printf("Utility function called successfully\\n");
}
'''
        with open(utils_source, 'w') as f:
            f.write(utils_code)
        source_files.append(utils_source)
        
        # Header file
        utils_header = project_dir / "utils.h"
        header_code = '''#ifndef UTILS_H
#define UTILS_H

void test_function(void);

#endif
'''
        with open(utils_header, 'w') as f:
            f.write(header_code)
        
        print(f"📝 Created test project with {len(source_files)} source files")
        
        # Generate Visual Studio project
        project_file = build_manager.create_vcxproj(
            "MatrixTestProject", source_files, project_dir, "x64"
        )
        
        if project_file.exists():
            print(f"✅ VS Project created: {project_file}")
            
            # Verify project file content
            with open(project_file, 'r') as f:
                content = f.read()
                if "MatrixTestProject" in content and "ClCompile" in content:
                    print("✅ Project file content validated")
                    return True
                else:
                    print("❌ Project file content invalid")
                    return False
        else:
            print("❌ Project file not created")
            return False
            
    except Exception as e:
        print(f"❌ Project generation test failed: {e}")
        return False

def test_build_paths_verification():
    """Test build tool paths verification"""
    print("\n🔍 Testing build tool paths verification...")
    
    try:
        from core.build_system_manager import get_build_manager
        
        build_manager = get_build_manager()
        
        # Test compiler paths
        x64_compiler = build_manager.get_compiler_path("x64")
        x86_compiler = build_manager.get_compiler_path("x86")
        
        print(f"✅ x64 Compiler: {x64_compiler}")
        print(f"✅ x86 Compiler: {x86_compiler}")
        
        # Test MSBuild path
        msbuild_path = build_manager.get_msbuild_path()
        print(f"✅ MSBuild: {msbuild_path}")
        
        # Test include directories
        include_dirs = build_manager.get_include_dirs()
        print(f"✅ Include directories: {len(include_dirs)} found")
        
        # Test library directories
        lib_dirs_x64 = build_manager.get_library_dirs("x64")
        lib_dirs_x86 = build_manager.get_library_dirs("x86")
        
        print(f"✅ x64 Library directories: {len(lib_dirs_x64)} found")
        print(f"✅ x86 Library directories: {len(lib_dirs_x86)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Build paths verification failed: {e}")
        return False

def main():
    """Main test execution for comprehensive B1 completion"""
    print("=" * 70)
    print("B1: COMPREHENSIVE BUILD SYSTEM INTEGRATION TEST")
    print("Testing complete build system functionality for Matrix Pipeline")
    print("=" * 70)
    
    tests = [
        ("Build Tool Paths Verification", test_build_paths_verification),
        ("Source Code Compilation", test_sample_compilation),
        ("MSBuild Project Generation", test_msbuild_project_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 70)
    print(f"B1 COMPREHENSIVE TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ B1: BUILD SYSTEM INTEGRATION COMPLETED SUCCESSFULLY!")
        print("🎯 Matrix Pipeline build system fully operational")
        print("🔧 Ready for production binary reconstruction")
    else:
        print("❌ B1: Build system integration incomplete")
        print("🔧 Some build functionality needs fixes")
    
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)