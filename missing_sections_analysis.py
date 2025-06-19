#!/usr/bin/env python3
"""
PE Missing Sections Analysis
Extracts and analyzes the components missing from our 4.61MB reconstruction
"""

import struct
import os
from pathlib import Path

def extract_missing_sections():
    """Extract the missing sections from original launcher.exe"""
    
    input_file = "/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/input/launcher.exe"
    output_dir = "/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/missing_components"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the PE file
    with open(input_file, 'rb') as f:
        data = f.read()
    
    print(f"Original file size: {len(data):,} bytes ({len(data)/1024/1024:.2f} MB)")
    
    # Extract missing sections
    sections = {
        '.text': {'offset': 0x1000, 'size': 0xa8000, 'description': 'Executable code section'},
        'STLPORT_': {'offset': 0xd3000, 'size': 0x1000, 'description': 'STLPort library data'},
        '.reloc': {'offset': 0x4ed000, 'size': 0x19000, 'description': 'Relocation table'},
        'headers': {'offset': 0x0, 'size': 0x1000, 'description': 'PE headers (DOS + PE + sections)'}
    }
    
    total_missing = 0
    
    for name, info in sections.items():
        section_data = data[info['offset']:info['offset'] + info['size']]
        output_file = os.path.join(output_dir, f"{name}.bin")
        
        with open(output_file, 'wb') as f:
            f.write(section_data)
        
        # Analyze content
        zeros = section_data.count(0)
        zero_percent = zeros / len(section_data) * 100
        
        print(f"\n{name} section:")
        print(f"  Size: {len(section_data):,} bytes ({len(section_data)/1024:.1f} KB)")
        print(f"  Description: {info['description']}")
        print(f"  Zero bytes: {zeros:,}/{len(section_data):,} ({zero_percent:.1f}%)")
        print(f"  First 16 bytes: {section_data[:16].hex()}")
        print(f"  Saved to: {output_file}")
        
        total_missing += len(section_data)
    
    print(f"\nSummary:")
    print(f"Total missing components: {total_missing:,} bytes ({total_missing/1024/1024:.2f} MB)")
    print(f"Extracted components: 4,468,736 bytes (4.26 MB)")
    print(f"Original file: {len(data):,} bytes ({len(data)/1024/1024:.2f} MB)")
    print(f"Verification: {total_missing + 4468736} == {len(data)} ? {total_missing + 4468736 == len(data)}")

def analyze_pe_structure():
    """Detailed analysis of PE structure"""
    
    input_file = "/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/input/launcher.exe"
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    # Get PE header offset
    pe_offset = struct.unpack('<I', data[0x3c:0x40])[0]
    
    # Parse COFF header
    coff_header = data[pe_offset+4:pe_offset+24]
    machine, num_sections, timestamp, ptr_to_symbol_table, num_symbols, size_optional_header, characteristics = struct.unpack('<HHIIIHH', coff_header)
    
    # Optional header
    optional_header_start = pe_offset + 24
    opt_header = data[optional_header_start:optional_header_start+size_optional_header]
    
    # Parse key fields from optional header
    magic, major_linker, minor_linker, size_of_code, size_of_init_data, size_of_uninit_data, entry_point, base_of_code = struct.unpack('<HBBIIIII', opt_header[0:24])
    
    print("\nPE Structure Analysis:")
    print(f"PE signature offset: 0x{pe_offset:x}")
    print(f"Machine type: 0x{machine:x} (Intel 386)")
    print(f"Number of sections: {num_sections}")
    print(f"Optional header size: {size_optional_header}")
    print(f"Entry point: 0x{entry_point:x}")
    print(f"Base of code: 0x{base_of_code:x}")
    print(f"Size of code: {size_of_code:,} bytes")
    print(f"Size of initialized data: {size_of_init_data:,} bytes")
    print(f"Size of uninitialized data: {size_of_uninit_data:,} bytes")

def create_reconstruction_strategy():
    """Create a strategy for complete reconstruction"""
    
    strategy = """
PE Reconstruction Strategy
=========================

To achieve exact size match (5,267,456 bytes), we need to reconstruct:

1. PE Headers (4,096 bytes):
   - DOS header with stub
   - PE signature and COFF header
   - Optional header with all fields
   - Section table

2. .text section (688,128 bytes):
   - Executable machine code
   - Contains the main program logic
   - Critical for functionality
   
3. STLPORT_ section (4,096 bytes):
   - STLPort library data
   - Mostly zeros (99.4%)
   - Template instantiation data

4. .reloc section (102,400 bytes):
   - Base relocation table
   - Needed for ASLR support
   - 61% data, 39% padding

Components already extracted:
- .rdata: 118,784 bytes (read-only data)
- .data: 53,248 bytes (initialized data)  
- .rsrc: 4,296,704 bytes (resources)

Next steps:
1. Extract .text section and disassemble
2. Create stub implementations for all functions
3. Reconstruct relocation table structure
4. Build complete PE with proper headers
5. Verify byte-for-byte match
"""
    
    with open("/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/reconstruction_strategy.txt", 'w') as f:
        f.write(strategy)
    
    print("Reconstruction strategy saved to reconstruction_strategy.txt")

if __name__ == "__main__":
    print("PE Missing Sections Analysis")
    print("=" * 50)
    
    extract_missing_sections()
    analyze_pe_structure()
    create_reconstruction_strategy()