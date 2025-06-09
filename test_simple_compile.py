#!/usr/bin/env python3
"""
Simple compilation test to verify build system basics
"""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_build_system_compilation():
    """Test compilation using build system manager"""
    print("🔧 Testing compilation with build system manager...")
    
    try:
        from core.build_system_manager import get_build_manager
        
        build_manager = get_build_manager()
        
        # Create simple test file
        test_dir = Path("output/build_system_test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        source_file = test_dir / "simple.c"
        output_file = test_dir / "simple.exe"
        
        with open(source_file, 'w') as f:
            f.write('''#include <stdio.h>
int main() {
    printf("Build system compilation test\\n");
    return 0;
}
''')
        
        print(f"Source: {source_file}")
        print(f"Output: {output_file}")
        
        # Use build system manager
        success, output = build_manager.compile_source(
            source_file, output_file, architecture="x64", configuration="Release"
        )
        
        print(f"Success: {success}")
        print(f"Output: {output}")
        
        if success and output_file.exists():
            print(f"✅ Build system compilation successful! ({output_file.stat().st_size} bytes)")
            return True
        else:
            print("❌ Build system compilation failed")
            return False
            
    except Exception as e:
        print(f"❌ Build system compilation test error: {e}")
        return False

def test_environment_setup():
    """Test if we can access VS environment"""
    print("\n🌍 Testing Visual Studio environment...")
    
    try:
        from core.build_system_manager import get_build_manager
        
        build_manager = get_build_manager()
        
        # Test paths exist
        compiler = Path(build_manager.get_compiler_path("x64"))
        if not compiler.exists():
            print(f"❌ Compiler not found: {compiler}")
            return False
        
        print(f"✅ Compiler found: {compiler}")
        
        # Test include directories
        includes = build_manager.get_include_dirs()
        valid_includes = 0
        for inc in includes:
            if Path(inc).exists():
                valid_includes += 1
                print(f"✅ Include dir: {inc}")
            else:
                print(f"❌ Missing include dir: {inc}")
        
        if valid_includes == len(includes):
            print("✅ All include directories found")
            return True
        else:
            print(f"❌ Only {valid_includes}/{len(includes)} include directories found")
            return False
            
    except Exception as e:
        print(f"❌ Environment test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE BUILD SYSTEM TEST")
    print("=" * 50)
    
    tests = [
        test_environment_setup,
        test_build_system_compilation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResult: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ Basic build system functionality verified")
    else:
        print("❌ Build system needs fixes")