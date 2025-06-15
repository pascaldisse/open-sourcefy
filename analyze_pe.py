#!/usr/bin/env python3

import struct
import os

def analyze_pe_sections(filepath):
    try:
        with open(filepath, 'rb') as f:
            # Read DOS header
            dos_header = f.read(64)
            pe_offset = struct.unpack('<I', dos_header[60:64])[0]
            
            # Read PE header
            f.seek(pe_offset)
            pe_sig = f.read(4)
            if pe_sig != b'PE\x00\x00':
                return None
                
            # Read COFF header
            coff_header = f.read(20)
            num_sections = struct.unpack('<H', coff_header[2:4])[0]
            optional_header_size = struct.unpack('<H', coff_header[16:18])[0]
            
            # Skip optional header
            f.seek(pe_offset + 24 + optional_header_size)
            
            # Read section headers
            sections = []
            for i in range(num_sections):
                section_header = f.read(40)
                name = section_header[:8].decode('ascii', errors='ignore').rstrip('\x00')
                virtual_size = struct.unpack('<I', section_header[8:12])[0]
                raw_size = struct.unpack('<I', section_header[16:20])[0]
                sections.append((name, virtual_size, raw_size))
            
            return sections
            
    except Exception as e:
        print(f'Error: {e}')
        return None

# Analyze both binaries
original_path = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/input/launcher.exe'
compiled_path = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/temp/manual_compile/bin/Release/launcher.exe'

print("=== ORIGINAL BINARY ANALYSIS ===")
sections = analyze_pe_sections(original_path)
if sections:
    print('Original launcher.exe sections:')
    total_size = 0
    for name, vsize, rsize in sections:
        print(f'  {name:<8} Virtual: {vsize:>8} Raw: {rsize:>8}')
        total_size += rsize
    file_size = os.path.getsize(original_path)
    print(f'Total raw size: {total_size}')
    print(f'File size: {file_size}')
    print(f'Difference: {file_size - total_size} (likely padding/overlay)')

print("\n=== COMPILED BINARY ANALYSIS ===")
if os.path.exists(compiled_path):
    sections = analyze_pe_sections(compiled_path)
    if sections:
        print('Compiled launcher.exe sections:')
        total_size = 0
        for name, vsize, rsize in sections:
            print(f'  {name:<8} Virtual: {vsize:>8} Raw: {rsize:>8}')
            total_size += rsize
        file_size = os.path.getsize(compiled_path)
        print(f'Total raw size: {total_size}')
        print(f'File size: {file_size}')
        print(f'Difference: {file_size - total_size} (likely padding)')
else:
    print('Compiled binary not found')