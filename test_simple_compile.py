#!/usr/bin/env python3
"""
Simple compilation test to debug the build system issue
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.core.build_system_manager import BuildSystemManager


def test_debug_compilation():
    """Debug compilation with detailed logging"""
    print("🔍 Debugging compilation issue...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create simple hello world program
        source_file = temp_path / "hello.c"
        output_file = temp_path / "hello.exe"
        
        source_content = """#include <stdio.h>

int main() {
    printf("Hello World!\\n");
    return 0;
}
"""
        
        with open(source_file, 'w') as f:
            f.write(source_content)
        
        # Initialize build manager with debug logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        
        manager = BuildSystemManager()
        
        print("\n📋 Build Configuration:")
        print(f"Compiler: {manager.get_compiler_path('x64')}")
        print(f"Include dirs: {manager.get_include_dirs()}")
        print(f"Library dirs: {manager.get_library_dirs('x64')}")
        
        print(f"\n🔨 Compiling: {source_file} -> {output_file}")
        
        success, output = manager.compile_source(source_file, output_file)
        
        print(f"\n📊 Result: {'Success' if success else 'Failed'}")
        print(f"Output:\n{output}")
        
        if not success:
            # Check if files exist
            print(f"\n🔍 Diagnostic Information:")
            print(f"Source exists: {source_file.exists()}")
            print(f"Source content: {source_file.read_text()[:100]}...")
            
            # Check if compiler exists
            compiler_path = Path(manager.get_compiler_path('x64'))
            print(f"Compiler exists: {compiler_path.exists()}")
            
            # Check first include directory
            first_include = Path(manager.get_include_dirs()[0])
            print(f"First include dir exists: {first_include.exists()}")
            if first_include.exists():
                stdio_path = first_include / "stdio.h"
                print(f"stdio.h exists: {stdio_path.exists()}")


if __name__ == "__main__":
    test_debug_compilation()