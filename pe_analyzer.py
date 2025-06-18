#!/usr/bin/env python3
import struct
import sys

def analyze_pe(filename):
    with open(filename, 'rb') as f:
        # Read DOS header
        dos_header = f.read(64)
        if dos_header[:2] != b'MZ':
            print('Not a valid PE file')
            return
        
        # Get PE header offset
        pe_offset = struct.unpack('<L', dos_header[60:64])[0]
        f.seek(pe_offset)
        
        # Read PE signature
        pe_sig = f.read(4)
        if pe_sig != b'PE\x00\x00':
            print('Invalid PE signature')
            return
        
        # Read COFF header
        coff_header = f.read(20)
        machine, num_sections, timestamp, ptr_to_symtab, num_symbols, size_of_opt_header, characteristics = struct.unpack('<HHLLLHH', coff_header)
        
        print(f'Machine: 0x{machine:04x}')
        print(f'Number of sections: {num_sections}')
        print(f'Size of optional header: {size_of_opt_header}')
        print(f'Characteristics: 0x{characteristics:04x} (', end='')
        
        # Decode characteristics
        char_flags = []
        if characteristics & 0x0001: char_flags.append('RELOCS_STRIPPED')
        if characteristics & 0x0002: char_flags.append('EXECUTABLE_IMAGE')
        if characteristics & 0x0004: char_flags.append('LINE_NUMBERS_STRIPPED')
        if characteristics & 0x0008: char_flags.append('LOCAL_SYMS_STRIPPED')
        if characteristics & 0x0020: char_flags.append('LARGE_ADDRESS_AWARE')
        if characteristics & 0x0100: char_flags.append('32BIT_MACHINE')
        if characteristics & 0x0200: char_flags.append('DEBUG_STRIPPED')
        if characteristics & 0x1000: char_flags.append('SYSTEM')
        if characteristics & 0x2000: char_flags.append('DLL')
        if characteristics & 0x4000: char_flags.append('UP_SYSTEM_ONLY')
        print(', '.join(char_flags) + ')')
        
        # Read optional header
        opt_header = f.read(size_of_opt_header)
        if len(opt_header) >= 28:
            # PE32 optional header format: Magic(2) + MajorLinkerVersion(1) + MinorLinkerVersion(1) + SizeOfCode(4) + SizeOfInitializedData(4) + SizeOfUninitializedData(4) + AddressOfEntryPoint(4) + BaseOfCode(4) + BaseOfData(4)
            magic = struct.unpack('<H', opt_header[0:2])[0]
            major_linker = struct.unpack('<B', opt_header[2:3])[0]
            minor_linker = struct.unpack('<B', opt_header[3:4])[0]
            size_of_code = struct.unpack('<L', opt_header[4:8])[0]
            size_of_init_data = struct.unpack('<L', opt_header[8:12])[0]
            size_of_uninit_data = struct.unpack('<L', opt_header[12:16])[0]
            entry_point = struct.unpack('<L', opt_header[16:20])[0]
            base_of_code = struct.unpack('<L', opt_header[20:24])[0]
            
            print(f'Magic: 0x{magic:04x} ({"PE32" if magic == 0x10b else "PE32+" if magic == 0x20b else "UNKNOWN"})')
            print(f'Linker version: {major_linker}.{minor_linker}')
            print(f'Size of code: {size_of_code:,} bytes ({size_of_code/1024:.1f} KB)')
            print(f'Size of initialized data: {size_of_init_data:,} bytes ({size_of_init_data/1024:.1f} KB)')
            print(f'Size of uninitialized data: {size_of_uninit_data:,} bytes')
            print(f'Entry point: 0x{entry_point:08x}')
            print(f'Base of code: 0x{base_of_code:08x}')
        
        if len(opt_header) >= 32:
            # Read more optional header fields for PE32
            # BaseOfData(4) + ImageBase(4) + SectionAlignment(4) + FileAlignment(4)
            base_of_data = struct.unpack('<L', opt_header[24:28])[0]
            image_base = struct.unpack('<L', opt_header[28:32])[0]
            print(f'Base of data: 0x{base_of_data:08x}')
            print(f'Image base: 0x{image_base:08x}')
            
        if len(opt_header) >= 40:
            section_alignment = struct.unpack('<L', opt_header[32:36])[0]
            file_alignment = struct.unpack('<L', opt_header[36:40])[0]
            print(f'Section alignment: 0x{section_alignment:x}')
            print(f'File alignment: 0x{file_alignment:x}')
            
            # Read subsystem info - it's at offset 68 in PE32
            if len(opt_header) >= 70:
                subsystem = struct.unpack('<H', opt_header[68:70])[0]
                subsystem_names = {
                    0: 'UNKNOWN', 1: 'NATIVE', 2: 'WINDOWS_GUI', 3: 'WINDOWS_CUI',
                    5: 'OS2_CUI', 7: 'POSIX_CUI', 8: 'NATIVE_WINDOWS', 9: 'WINDOWS_CE_GUI',
                    10: 'EFI_APPLICATION', 11: 'EFI_BOOT_SERVICE_DRIVER', 12: 'EFI_RUNTIME_DRIVER',
                    13: 'EFI_ROM', 14: 'XBOX'
                }
                print(f'Subsystem: {subsystem} ({subsystem_names.get(subsystem, "UNKNOWN")})')
        
        # Read section headers
        print(f'\nSections ({num_sections} total):')
        print('Name      VirtSize  VirtAddr  RawSize   RawPtr    Characteristics')
        print('-' * 70)
        total_raw_size = 0
        for i in range(num_sections):
            section_header = f.read(40)
            if len(section_header) < 40:
                break
            name = section_header[:8].rstrip(b'\x00').decode('ascii', errors='ignore')
            # Section header: VirtualSize(4) + VirtualAddress(4) + SizeOfRawData(4) + PointerToRawData(4) + PointerToRelocations(4) + PointerToLinenumbers(4) + NumberOfRelocations(2) + NumberOfLinenumbers(2) + Characteristics(4)
            virtual_size, virtual_address, size_of_raw_data, ptr_to_raw_data, ptr_to_relocs, ptr_to_line_nums, num_relocs, num_line_nums, characteristics = struct.unpack('<LLLLLLHHL', section_header[8:])
            
            # Decode section characteristics
            char_flags = []
            if characteristics & 0x00000020: char_flags.append('CODE')
            if characteristics & 0x00000040: char_flags.append('INIT_DATA')
            if characteristics & 0x00000080: char_flags.append('UNINIT_DATA')
            if characteristics & 0x02000000: char_flags.append('DISCARDABLE')
            if characteristics & 0x04000000: char_flags.append('NOT_CACHED')
            if characteristics & 0x08000000: char_flags.append('NOT_PAGED')
            if characteristics & 0x10000000: char_flags.append('SHARED')
            if characteristics & 0x20000000: char_flags.append('EXECUTE')
            if characteristics & 0x40000000: char_flags.append('READ')
            if characteristics & 0x80000000: char_flags.append('WRITE')
            
            print(f'{name:<8} {virtual_size:8} 0x{virtual_address:08x} {size_of_raw_data:8} 0x{ptr_to_raw_data:08x} {"|".join(char_flags)}')
            total_raw_size += size_of_raw_data
        
        print(f'\nTotal raw section size: {total_raw_size:,} bytes ({total_raw_size/1024:.1f} KB)')
        
        # Check for resources by looking for .rsrc section
        f.seek(pe_offset + 24 + size_of_opt_header)  # Go back to section headers
        has_resources = False
        rsrc_size = 0
        for i in range(num_sections):
            section_header = f.read(40)
            if len(section_header) < 40:
                break
            name = section_header[:8].rstrip(b'\x00').decode('ascii', errors='ignore')
            if name == '.rsrc':
                has_resources = True
                virtual_size, virtual_address, size_of_raw_data = struct.unpack('<LLL', section_header[8:20])
                rsrc_size = size_of_raw_data
                print(f'\nResource section found: {rsrc_size:,} bytes ({rsrc_size/1024:.1f} KB)')
                break
        
        if not has_resources:
            print('\nNo resource section (.rsrc) found')
        
        # Get total file size
        import os
        file_size = os.path.getsize(filename)
        print(f'\nTotal file size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)')
        
        if has_resources:
            code_and_data_size = file_size - rsrc_size
            print(f'Code + Data size (excluding resources): {code_and_data_size:,} bytes ({code_and_data_size/1024/1024:.2f} MB)')
            print(f'Resources make up {rsrc_size/file_size*100:.1f}% of the file')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 pe_analyzer.py <pe_file>')
        sys.exit(1)
    
    analyze_pe(sys.argv[1])