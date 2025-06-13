#!/usr/bin/env python3
"""
Import Table Analysis Script
Analyzes the original launcher.exe import table structure to understand 
the import table mismatch issue in the Matrix decompilation pipeline.
"""

import os
import sys
from pathlib import Path

# Add the src path to import core modules
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False
    print("ERROR: pefile library not available. Run: pip install pefile")
    sys.exit(1)

def analyze_pe_imports(binary_path):
    """Analyze PE import table structure"""
    print(f"Analyzing import table for: {binary_path}")
    print("=" * 60)
    
    try:
        pe = pefile.PE(binary_path)
        
        print(f"Binary Format: PE32 (Windows)")
        print(f"Architecture: {pe.FILE_HEADER.Machine}")
        print(f"Entry Point: 0x{pe.OPTIONAL_HEADER.AddressOfEntryPoint:08X}")
        
        # Check if binary has imports
        if not hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            print("\n❌ NO IMPORT TABLE FOUND!")
            print("This confirms the binary is PACKED - import table is hidden")
            return
            
        # Analyze import structure
        print(f"\n✅ IMPORT TABLE FOUND")
        print("-" * 40)
        
        total_imports = 0
        dll_count = 0
        
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode('utf-8', errors='ignore')
            dll_count += 1
            function_count = len(entry.imports)
            total_imports += function_count
            
            print(f"\nDLL #{dll_count}: {dll_name}")
            print(f"  Functions: {function_count}")
            
            # Show first 10 functions for each DLL
            for i, imp in enumerate(entry.imports[:10]):
                if imp.name:
                    func_name = imp.name.decode('utf-8', errors='ignore')
                    print(f"    {i+1:2d}. {func_name} (Address: 0x{imp.address:08X})")
                elif imp.ordinal:
                    print(f"    {i+1:2d}. Ordinal_{imp.ordinal} (Address: 0x{imp.address:08X})")
                    
            if function_count > 10:
                print(f"    ... and {function_count - 10} more functions")
        
        print("\n" + "=" * 60)
        print("IMPORT TABLE SUMMARY")
        print("=" * 60)
        print(f"Total DLLs: {dll_count}")
        print(f"Total Functions: {total_imports}")
        
        # Compare with our current reconstruction
        print(f"\nCURRENT RECONSTRUCTION vs ORIGINAL:")
        print("-" * 40)
        our_dlls = ["kernel32.dll", "user32.dll", "ws2_32.dll", "wininet.dll", "shlwapi.lib"]
        print(f"Our DLLs: {len(our_dlls)} ({', '.join(our_dlls)})")
        print(f"Original DLLs: {dll_count}")
        print(f"Missing DLLs: {dll_count - len(our_dlls)}")
        print(f"Missing Functions: {total_imports - len(our_dlls)}")
        
        # Calculate import table mismatch impact
        mismatch_percentage = ((dll_count - len(our_dlls)) / dll_count) * 100 if dll_count > 0 else 0
        print(f"\nIMPORT TABLE MISMATCH: {mismatch_percentage:.1f}%")
        
        print(f"\n🔧 RECOMMENDATIONS:")
        print("1. Extract complete import list from original binary")
        print("2. Generate comprehensive .lib dependencies in VS project")
        print("3. Include all imported functions as extern declarations")
        print("4. Reconstruct Import Address Table (IAT) structure")
        
    except pefile.PEFormatError as e:
        print(f"❌ Invalid PE format: {e}")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
    finally:
        try:
            pe.close()
        except:
            pass

def analyze_reconstructed_binary(reconstructed_path):
    """Analyze our reconstructed binary's imports"""
    if not Path(reconstructed_path).exists():
        print(f"Reconstructed binary not found: {reconstructed_path}")
        return
        
    print(f"\n" + "=" * 60)
    print("RECONSTRUCTED BINARY ANALYSIS")
    print("=" * 60)
    
    try:
        pe = pefile.PE(reconstructed_path)
        
        if not hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            print("❌ NO IMPORT TABLE FOUND in reconstructed binary!")
            return
            
        dll_count = 0
        total_imports = 0
        
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode('utf-8', errors='ignore')
            dll_count += 1
            function_count = len(entry.imports)
            total_imports += function_count
            print(f"DLL: {dll_name} ({function_count} functions)")
            
        print(f"\nReconstructed Binary Import Summary:")
        print(f"  DLLs: {dll_count}")
        print(f"  Functions: {total_imports}")
        
    except Exception as e:
        print(f"Failed to analyze reconstructed binary: {e}")
    finally:
        try:
            pe.close()
        except:
            pass

if __name__ == "__main__":
    # Analyze original binary
    original_binary = Path("input/launcher.exe")
    if original_binary.exists():
        analyze_pe_imports(str(original_binary))
    else:
        print(f"Original binary not found: {original_binary}")
        
    # Try to find and analyze reconstructed binary
    output_dirs = list(Path("output/launcher").glob("*/compilation/bin/*/*"))
    if output_dirs:
        for exe_file in output_dirs:
            if exe_file.suffix == '.exe':
                analyze_reconstructed_binary(str(exe_file))
                break