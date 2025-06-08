"""
Agent 15: Binary Metadata Analysis
Extracts and analyzes binary metadata including PE headers, imports, exports, and resources.
"""

import os
import struct
import json
from typing import Dict, Any, List, Optional, Tuple
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent15_MetadataAnalysis(BaseAgent):
    """Agent 15: Binary metadata extraction and analysis"""
    
    def __init__(self):
        super().__init__(
            agent_id=15,
            name="MetadataAnalysis",
            dependencies=[1, 2]  # Depends on binary discovery and architecture analysis
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute binary metadata analysis"""
        # Check dependencies
        for dep_id in self.dependencies:
            dep_result = context['agent_results'].get(dep_id)
            if not dep_result or dep_result.status != AgentStatus.COMPLETED:
                return AgentResult(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    data={},
                    error_message=f"Dependency Agent {dep_id} did not complete successfully"
                )

        try:
            binary_path = context.get('binary_path', '')
            output_paths = context.get('output_paths', {})
            analysis_dir = output_paths.get('agents', context.get('output_dir', 'output'))
            
            metadata_result = self._analyze_binary_metadata(binary_path, analysis_dir, context)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=metadata_result,
                metadata={
                    'depends_on': self.dependencies,
                    'analysis_type': 'binary_metadata_analysis'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Binary metadata analysis failed: {str(e)}"
            )

    def _analyze_binary_metadata(self, binary_path: str, output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze binary metadata comprehensively"""
        result = {
            'file_info': {},
            'pe_header': {},
            'sections': {},
            'imports': {},
            'exports': {},
            'resources': {},
            'version_info': {},
            'debug_info': {},
            'certificates': {},
            'strings': {},
            'entropy_analysis': {},
            'analysis_quality': 'unknown'
        }
        
        if not os.path.exists(binary_path):
            result['error'] = f"Binary file not found: {binary_path}"
            result['analysis_quality'] = 'failed'
            return result
        
        try:
            # Basic file information
            result['file_info'] = self._get_file_info(binary_path)
            
            # Check file format and parse accordingly
            if self._is_pe_file(binary_path):
                result['file_format'] = 'PE'
                result['pe_header'] = self._parse_pe_header(binary_path)
                result['sections'] = self._parse_pe_sections(binary_path)
                result['imports'] = self._parse_imports(binary_path)
                result['exports'] = self._parse_exports(binary_path)
                result['resources'] = self._parse_resources(binary_path)
                result['version_info'] = self._parse_version_info(binary_path)
                result['debug_info'] = self._parse_debug_info(binary_path)
                result['certificates'] = self._parse_certificates(binary_path)
            elif self._is_elf_file(binary_path):
                result['file_format'] = 'ELF'
                result['elf_header'] = self._parse_elf_header(binary_path)
                result['sections'] = self._parse_elf_sections(binary_path)
                result['symbols'] = self._parse_elf_symbols(binary_path)
                result['dynamic_section'] = self._parse_elf_dynamic(binary_path)
                result['program_headers'] = self._parse_elf_program_headers(binary_path)
            elif self._is_macho_file(binary_path):
                result['file_format'] = 'Mach-O'
                result['macho_header'] = self._parse_macho_header(binary_path)
                result['load_commands'] = self._parse_macho_load_commands(binary_path)
                result['sections'] = self._parse_macho_sections(binary_path)
            else:
                result['file_format'] = 'Unknown'
                result['format_analysis'] = 'Unsupported binary format'
            
            # Universal analysis
            result['strings'] = self._extract_strings(binary_path)
            result['entropy_analysis'] = self._analyze_entropy(binary_path)
            
            # Assess overall quality
            result['analysis_quality'] = self._assess_metadata_quality(result)
            
        except Exception as e:
            result['error'] = str(e)
            result['analysis_quality'] = 'failed'
        
        return result

    def _get_file_info(self, binary_path: str) -> Dict[str, Any]:
        """Get basic file information"""
        try:
            stat_info = os.stat(binary_path)
            return {
                'file_size': stat_info.st_size,
                'modification_time': stat_info.st_mtime,
                'creation_time': stat_info.st_ctime,
                'file_path': binary_path,
                'file_name': os.path.basename(binary_path)
            }
        except Exception as e:
            return {'error': str(e)}

    def _is_pe_file(self, binary_path: str) -> bool:
        """Check if file is a PE executable"""
        try:
            with open(binary_path, 'rb') as f:
                # Check DOS header
                dos_header = f.read(64)
                if len(dos_header) < 64 or dos_header[:2] != b'MZ':
                    return False
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                
                # Check PE signature
                f.seek(pe_offset)
                pe_sig = f.read(4)
                return pe_sig == b'PE\x00\x00'
        except:
            return False

    def _parse_pe_header(self, binary_path: str) -> Dict[str, Any]:
        """Parse PE header information"""
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                
                # Read PE header
                f.seek(pe_offset)
                pe_signature = f.read(4)
                
                # Read COFF header
                coff_header = f.read(20)
                machine, num_sections, timestamp, ptr_to_symbols, num_symbols, size_of_optional, characteristics = struct.unpack('<HHIIIHH', coff_header)
                
                # Read optional header
                optional_header = f.read(size_of_optional)
                
                return {
                    'pe_signature': pe_signature.hex(),
                    'machine_type': self._get_machine_type(machine),
                    'number_of_sections': num_sections,
                    'timestamp': timestamp,
                    'characteristics': self._parse_characteristics(characteristics),
                    'subsystem': self._get_subsystem(optional_header) if optional_header else 'unknown',
                    'entry_point': self._get_entry_point(optional_header) if optional_header else 0,
                    'image_base': self._get_image_base(optional_header) if optional_header else 0
                }
        except Exception as e:
            return {'error': str(e)}

    def _get_machine_type(self, machine: int) -> str:
        """Get machine type from PE header"""
        machine_types = {
            0x014c: 'IMAGE_FILE_MACHINE_I386',
            0x0200: 'IMAGE_FILE_MACHINE_IA64',
            0x8664: 'IMAGE_FILE_MACHINE_AMD64',
            0x01c0: 'IMAGE_FILE_MACHINE_ARM',
            0xaa64: 'IMAGE_FILE_MACHINE_ARM64'
        }
        return machine_types.get(machine, f'Unknown (0x{machine:04x})')

    def _parse_characteristics(self, characteristics: int) -> List[str]:
        """Parse PE characteristics flags"""
        flags = []
        flag_names = {
            0x0001: 'RELOCS_STRIPPED',
            0x0002: 'EXECUTABLE_IMAGE',
            0x0004: 'LINE_NUMBERS_STRIPPED',
            0x0008: 'LOCAL_SYMS_STRIPPED',
            0x0010: 'AGGR_WS_TRIM',
            0x0020: 'LARGE_ADDRESS_AWARE',
            0x0080: 'BYTES_REVERSED_LO',
            0x0100: '32BIT_MACHINE',
            0x0200: 'DEBUG_STRIPPED',
            0x0400: 'REMOVABLE_RUN_FROM_SWAP',
            0x0800: 'NET_RUN_FROM_SWAP',
            0x1000: 'SYSTEM',
            0x2000: 'DLL',
            0x4000: 'UP_SYSTEM_ONLY',
            0x8000: 'BYTES_REVERSED_HI'
        }
        
        for flag, name in flag_names.items():
            if characteristics & flag:
                flags.append(name)
        
        return flags

    def _get_subsystem(self, optional_header: bytes) -> str:
        """Get subsystem from optional header"""
        if len(optional_header) < 68:
            return 'unknown'
        
        subsystem = struct.unpack('<H', optional_header[68:70])[0]
        subsystems = {
            0: 'UNKNOWN',
            1: 'NATIVE',
            2: 'WINDOWS_GUI',
            3: 'WINDOWS_CUI',
            5: 'OS2_CUI',
            7: 'POSIX_CUI',
            9: 'WINDOWS_CE_GUI',
            10: 'EFI_APPLICATION',
            11: 'EFI_BOOT_SERVICE_DRIVER',
            12: 'EFI_RUNTIME_DRIVER',
            13: 'EFI_ROM',
            14: 'XBOX',
            16: 'WINDOWS_BOOT_APPLICATION'
        }
        return subsystems.get(subsystem, f'Unknown ({subsystem})')

    def _get_entry_point(self, optional_header: bytes) -> int:
        """Get entry point from optional header"""
        if len(optional_header) < 20:
            return 0
        return struct.unpack('<I', optional_header[16:20])[0]

    def _get_image_base(self, optional_header: bytes) -> int:
        """Get image base from optional header"""
        if len(optional_header) < 28:
            return 0
        return struct.unpack('<I', optional_header[28:32])[0]

    def _parse_pe_sections(self, binary_path: str) -> Dict[str, Any]:
        """Parse PE sections"""
        try:
            sections = {}
            with open(binary_path, 'rb') as f:
                # Navigate to section headers
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset + 4 + 20)  # Skip PE sig + COFF header
                
                # Read optional header size and skip it
                size_of_optional = struct.unpack('<H', f.read(2))[0]
                f.seek(f.tell() + size_of_optional - 2)
                
                # Read number of sections from COFF header
                f.seek(pe_offset + 6)
                num_sections = struct.unpack('<H', f.read(2))[0]
                
                # Go to section headers
                f.seek(pe_offset + 4 + 20 + size_of_optional)
                
                for i in range(num_sections):
                    section_data = f.read(40)
                    if len(section_data) < 40:
                        break
                    
                    name = section_data[:8].rstrip(b'\x00').decode('ascii', errors='ignore')
                    virtual_size, virtual_address, raw_size, raw_address = struct.unpack('<IIII', section_data[8:24])
                    characteristics = struct.unpack('<I', section_data[36:40])[0]
                    
                    sections[name] = {
                        'virtual_size': virtual_size,
                        'virtual_address': virtual_address,
                        'raw_size': raw_size,
                        'raw_address': raw_address,
                        'characteristics': self._parse_section_characteristics(characteristics)
                    }
            
            return sections
        except Exception as e:
            return {'error': str(e)}

    def _parse_section_characteristics(self, characteristics: int) -> List[str]:
        """Parse section characteristics"""
        flags = []
        flag_names = {
            0x20: 'CNT_CODE',
            0x40: 'CNT_INITIALIZED_DATA',
            0x80: 'CNT_UNINITIALIZED_DATA',
            0x20000000: 'MEM_EXECUTE',
            0x40000000: 'MEM_READ',
            0x80000000: 'MEM_WRITE'
        }
        
        for flag, name in flag_names.items():
            if characteristics & flag:
                flags.append(name)
        
        return flags

    def _parse_imports(self, binary_path: str) -> Dict[str, Any]:
        """Parse import table with full PE parser implementation"""
        try:
            imported_dlls = []
            imported_functions = {}
            
            with open(binary_path, 'rb') as f:
                # Navigate to data directories
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                
                # Skip to optional header
                f.seek(pe_offset + 4 + 20)  # PE sig + COFF header
                size_of_optional = struct.unpack('<H', f.read(2))[0]
                f.seek(f.tell() - 2)  # Go back
                
                if size_of_optional >= 96:  # Minimum for data directories
                    optional_header = f.read(size_of_optional)
                    
                    # Import table is the 2nd data directory (index 1)
                    if len(optional_header) >= 96:
                        import_rva = struct.unpack('<I', optional_header[92:96])[0]
                        import_size = struct.unpack('<I', optional_header[96:100])[0] if len(optional_header) >= 100 else 0
                        
                        if import_rva > 0:
                            # Convert RVA to file offset (simplified)
                            import_offset = self._rva_to_offset(binary_path, import_rva)
                            if import_offset > 0:
                                f.seek(import_offset)
                                
                                # Parse import descriptors
                                while True:
                                    descriptor = f.read(20)  # Import descriptor size
                                    if len(descriptor) < 20:
                                        break
                                    
                                    name_rva, _, _, _, thunk_rva = struct.unpack('<IIIII', descriptor)
                                    if name_rva == 0:  # End of descriptors
                                        break
                                    
                                    # Get DLL name
                                    dll_name = self._read_string_at_rva(f, binary_path, name_rva)
                                    if dll_name:
                                        imported_dlls.append(dll_name)
                                        imported_functions[dll_name] = []
                                        
                                        # Parse import thunks (function names)
                                        if thunk_rva > 0:
                                            functions = self._parse_import_thunks(f, binary_path, thunk_rva)
                                            imported_functions[dll_name] = functions
            
            return {
                'imported_dlls': imported_dlls,
                'imported_functions': imported_functions,
                'total_imports': sum(len(funcs) for funcs in imported_functions.values()),
                'import_analysis': f'Successfully parsed {len(imported_dlls)} DLLs',
                'confidence': 0.9 if imported_dlls else 0.3
            }
            
        except Exception as e:
            return {
                'imported_dlls': [],
                'imported_functions': {},
                'import_analysis': f'Import parsing failed: {str(e)}',
                'confidence': 0.1
            }

    def _parse_exports(self, binary_path: str) -> Dict[str, Any]:
        """Parse export table with full implementation"""
        try:
            exported_functions = []
            
            with open(binary_path, 'rb') as f:
                # Navigate to data directories
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                
                # Skip to optional header
                f.seek(pe_offset + 4 + 20)
                size_of_optional = struct.unpack('<H', f.read(2))[0]
                f.seek(f.tell() - 2)
                
                if size_of_optional >= 96:
                    optional_header = f.read(size_of_optional)
                    
                    # Export table is the 1st data directory (index 0)
                    if len(optional_header) >= 92:
                        export_rva = struct.unpack('<I', optional_header[88:92])[0]
                        export_size = struct.unpack('<I', optional_header[92:96])[0] if len(optional_header) >= 96 else 0
                        
                        if export_rva > 0:
                            export_offset = self._rva_to_offset(binary_path, export_rva)
                            if export_offset > 0:
                                f.seek(export_offset)
                                
                                # Read export directory
                                export_dir = f.read(40)
                                if len(export_dir) >= 40:
                                    # Parse export directory structure
                                    (characteristics, timestamp, major_version, minor_version,
                                     name_rva, ordinal_base, num_functions, num_names,
                                     functions_rva, names_rva, ordinals_rva) = struct.unpack('<IIHHHIIIII', export_dir)
                                    
                                    # Read function names
                                    if names_rva > 0 and num_names > 0:
                                        names_offset = self._rva_to_offset(binary_path, names_rva)
                                        if names_offset > 0:
                                            f.seek(names_offset)
                                            for i in range(min(num_names, 1000)):  # Safety limit
                                                name_rva_data = f.read(4)
                                                if len(name_rva_data) < 4:
                                                    break
                                                name_rva = struct.unpack('<I', name_rva_data)[0]
                                                func_name = self._read_string_at_rva(f, binary_path, name_rva)
                                                if func_name:
                                                    exported_functions.append(func_name)
            
            return {
                'exported_functions': exported_functions,
                'total_exports': len(exported_functions),
                'export_analysis': f'Successfully parsed {len(exported_functions)} exported functions',
                'confidence': 0.9 if exported_functions else 0.3
            }
            
        except Exception as e:
            return {
                'exported_functions': [],
                'export_analysis': f'Export parsing failed: {str(e)}',
                'confidence': 0.1
            }

    def _parse_resources(self, binary_path: str) -> Dict[str, Any]:
        """Parse resource table with basic implementation"""
        try:
            resources = []
            resource_types = {}
            
            with open(binary_path, 'rb') as f:
                # Navigate to data directories
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                
                # Skip to optional header
                f.seek(pe_offset + 4 + 20)
                size_of_optional = struct.unpack('<H', f.read(2))[0]
                f.seek(f.tell() - 2)
                
                if size_of_optional >= 104:  # Need at least 3rd data directory
                    optional_header = f.read(size_of_optional)
                    
                    # Resource table is the 3rd data directory (index 2)
                    if len(optional_header) >= 104:
                        resource_rva = struct.unpack('<I', optional_header[100:104])[0]
                        resource_size = struct.unpack('<I', optional_header[104:108])[0] if len(optional_header) >= 108 else 0
                        
                        if resource_rva > 0:
                            resource_offset = self._rva_to_offset(binary_path, resource_rva)
                            if resource_offset > 0:
                                # Basic resource directory parsing
                                f.seek(resource_offset)
                                
                                # Read resource directory header
                                resource_dir = f.read(16)
                                if len(resource_dir) >= 16:
                                    characteristics, timestamp, major_version, minor_version, num_name_entries, num_id_entries = struct.unpack('<IIHHHH', resource_dir)
                                    
                                    total_entries = num_name_entries + num_id_entries
                                    
                                    # Parse resource entries
                                    for i in range(min(total_entries, 100)):  # Safety limit
                                        entry = f.read(8)
                                        if len(entry) < 8:
                                            break
                                        
                                        name_or_id, data_rva = struct.unpack('<II', entry)
                                        
                                        # Determine resource type
                                        resource_type = self._get_resource_type_name(name_or_id)
                                        resources.append({
                                            'type': resource_type,
                                            'id': name_or_id,
                                            'data_rva': data_rva
                                        })
                                        
                                        if resource_type in resource_types:
                                            resource_types[resource_type] += 1
                                        else:
                                            resource_types[resource_type] = 1
            
            return {
                'resources': resources[:50],  # Limit output size
                'resource_types': resource_types,
                'total_resources': len(resources),
                'resource_analysis': f'Successfully identified {len(resources)} resources of {len(resource_types)} types',
                'confidence': 0.8 if resources else 0.3
            }
            
        except Exception as e:
            return {
                'resources': [],
                'resource_analysis': f'Resource parsing failed: {str(e)}',
                'confidence': 0.1
            }

    def _parse_version_info(self, binary_path: str) -> Dict[str, Any]:
        """Parse version information"""
        raise NotImplementedError("Version information parsing not implemented - requires resource extraction and version info parsing")

    def _parse_debug_info(self, binary_path: str) -> Dict[str, Any]:
        """Parse debug information"""
        raise NotImplementedError("Debug information parsing not implemented - requires PDB file analysis and debug directory parsing")

    def _parse_certificates(self, binary_path: str) -> Dict[str, Any]:
        """Parse code signing certificates"""
        raise NotImplementedError("Certificate parsing not implemented - requires X.509 certificate parsing and signature verification")

    def _extract_strings(self, binary_path: str) -> Dict[str, Any]:
        """Extract strings from binary"""
        try:
            strings_ascii = []
            strings_unicode = []
            
            with open(binary_path, 'rb') as f:
                data = f.read()
                
                # Extract ASCII strings (minimum length 4)
                current_string = b''
                for byte in data:
                    if 32 <= byte <= 126:  # Printable ASCII
                        current_string += bytes([byte])
                    else:
                        if len(current_string) >= 4:
                            strings_ascii.append(current_string.decode('ascii'))
                        current_string = b''
                
                # Add final string if exists
                if len(current_string) >= 4:
                    strings_ascii.append(current_string.decode('ascii'))
            
            return {
                'ascii_strings': strings_ascii[:100],  # Limit to first 100
                'unicode_strings': strings_unicode[:100],
                'total_ascii_count': len(strings_ascii),
                'total_unicode_count': len(strings_unicode),
                'confidence': 0.8
            }
        except Exception as e:
            return {'error': str(e), 'confidence': 0.0}

    def _analyze_entropy(self, binary_path: str) -> Dict[str, Any]:
        """Analyze binary entropy (simplified version)"""
        try:
            with open(binary_path, 'rb') as f:
                data = f.read(min(65536, os.path.getsize(binary_path)))  # Sample first 64KB
                
                # Calculate byte frequency
                byte_counts = [0] * 256
                for byte in data:
                    byte_counts[byte] += 1
                
                # Calculate entropy
                entropy = 0.0
                data_len = len(data)
                if data_len > 0:
                    for count in byte_counts:
                        if count > 0:
                            probability = count / data_len
                            entropy -= probability * (probability.bit_length() - 1)
                
                return {
                    'entropy': entropy,
                    'entropy_rating': 'high' if entropy > 7.0 else 'medium' if entropy > 5.0 else 'low',
                    'analysis': 'High entropy may indicate packed/encrypted code',
                    'confidence': 0.7
                }
        except Exception as e:
            return {'error': str(e), 'confidence': 0.0}

    def _assess_metadata_quality(self, result: Dict[str, Any]) -> str:
        """Assess overall metadata analysis quality"""
        success_count = 0
        total_analyses = 0
        
        for key, value in result.items():
            if key == 'analysis_quality':
                continue
            
            total_analyses += 1
            if isinstance(value, dict):
                if 'error' not in value:
                    success_count += 1
                    # Bonus for high confidence analyses
                    confidence = value.get('confidence', 0.5)
                    if confidence > 0.7:
                        success_count += 0.5
        
        if total_analyses == 0:
            return 'unknown'
        
        success_rate = success_count / total_analyses
        
        if success_rate >= 0.9:
            return 'excellent'
        elif success_rate >= 0.7:
            return 'good'
        elif success_rate >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def _rva_to_offset(self, binary_path: str, rva: int) -> int:
        """Convert RVA (Relative Virtual Address) to file offset"""
        try:
            with open(binary_path, 'rb') as f:
                # Get section headers
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                
                # Skip to after COFF header
                f.seek(pe_offset + 4 + 20)
                size_of_optional = struct.unpack('<H', f.read(2))[0]
                f.seek(pe_offset + 6)
                num_sections = struct.unpack('<H', f.read(2))[0]
                
                # Go to section headers
                f.seek(pe_offset + 4 + 20 + size_of_optional)
                
                # Find the section containing this RVA
                for i in range(num_sections):
                    section_data = f.read(40)
                    if len(section_data) < 40:
                        break
                    
                    virtual_size, virtual_address, raw_size, raw_address = struct.unpack('<IIII', section_data[8:24])
                    
                    # Check if RVA is in this section
                    if virtual_address <= rva < virtual_address + virtual_size:
                        return raw_address + (rva - virtual_address)
                
                return 0  # RVA not found in any section
        except:
            return 0
    
    def _read_string_at_rva(self, f, binary_path: str, rva: int) -> str:
        """Read null-terminated string at RVA"""
        try:
            offset = self._rva_to_offset(binary_path, rva)
            if offset > 0:
                current_pos = f.tell()
                f.seek(offset)
                
                # Read string until null terminator
                string_bytes = b''
                while True:
                    byte = f.read(1)
                    if not byte or byte == b'\x00':
                        break
                    string_bytes += byte
                    if len(string_bytes) > 256:  # Safety limit
                        break
                
                f.seek(current_pos)  # Restore position
                return string_bytes.decode('ascii', errors='ignore')
        except:
            pass
        return ''
    
    def _parse_import_thunks(self, f, binary_path: str, thunk_rva: int) -> List[str]:
        """Parse import thunks to get function names"""
        functions = []
        try:
            thunk_offset = self._rva_to_offset(binary_path, thunk_rva)
            if thunk_offset > 0:
                current_pos = f.tell()
                f.seek(thunk_offset)
                
                # Read thunks (4 bytes each for 32-bit)
                for _ in range(100):  # Limit to prevent infinite loop
                    thunk_data = f.read(4)
                    if len(thunk_data) < 4:
                        break
                    
                    thunk_value = struct.unpack('<I', thunk_data)[0]
                    if thunk_value == 0:  # End of thunks
                        break
                    
                    # Check if import by name (not by ordinal)
                    if thunk_value & 0x80000000 == 0:
                        # It's an RVA to import by name structure
                        func_name = self._read_import_name(f, binary_path, thunk_value)
                        if func_name:
                            functions.append(func_name)
                    else:
                        # Import by ordinal
                        ordinal = thunk_value & 0xFFFF
                        functions.append(f"Ordinal_{ordinal}")
                
                f.seek(current_pos)  # Restore position
        except:
            pass
        return functions
    
    def _read_import_name(self, f, binary_path: str, name_rva: int) -> str:
        """Read import name structure"""
        try:
            name_offset = self._rva_to_offset(binary_path, name_rva)
            if name_offset > 0:
                current_pos = f.tell()
                f.seek(name_offset)
                
                # Skip hint (2 bytes) and read name
                f.read(2)
                name = self._read_string_at_rva(f, binary_path, name_rva + 2)
                
                f.seek(current_pos)
                return name
        except:
            pass
        return ''
    
    def _get_resource_type_name(self, type_id: int) -> str:
        """Get resource type name from ID"""
        resource_types = {
            1: 'RT_CURSOR',
            2: 'RT_BITMAP', 
            3: 'RT_ICON',
            4: 'RT_MENU',
            5: 'RT_DIALOG',
            6: 'RT_STRING',
            7: 'RT_FONTDIR',
            8: 'RT_FONT',
            9: 'RT_ACCELERATOR',
            10: 'RT_RCDATA',
            11: 'RT_MESSAGETABLE',
            12: 'RT_GROUP_CURSOR',
            14: 'RT_GROUP_ICON',
            16: 'RT_VERSION',
            17: 'RT_DLGINCLUDE',
            19: 'RT_PLUGPLAY',
            20: 'RT_VXD',
            21: 'RT_ANICURSOR',
            22: 'RT_ANIICON',
            23: 'RT_HTML',
            24: 'RT_MANIFEST'
        }
        return resource_types.get(type_id, f'UNKNOWN_{type_id}')
    
    # ELF Binary Format Support
    def _is_elf_file(self, binary_path: str) -> bool:
        """Check if file is an ELF executable"""
        try:
            with open(binary_path, 'rb') as f:
                elf_header = f.read(16)
                # ELF magic number: 0x7F followed by 'ELF'
                return len(elf_header) >= 4 and elf_header[:4] == b'\x7fELF'
        except:
            return False
    
    def _parse_elf_header(self, binary_path: str) -> Dict[str, Any]:
        """Parse ELF header information"""
        try:
            with open(binary_path, 'rb') as f:
                elf_header = f.read(64)  # ELF header is 64 bytes for 64-bit, 52 for 32-bit
                
                if len(elf_header) < 52:
                    return {'error': 'Invalid ELF header size'}
                
                # Parse ELF identification
                e_ident = elf_header[:16]
                ei_class = e_ident[4]  # 32-bit (1) or 64-bit (2)
                ei_data = e_ident[5]   # Little-endian (1) or big-endian (2)
                ei_version = e_ident[6]
                ei_osabi = e_ident[7]
                
                # Parse main header (assuming little-endian for now)
                e_type = struct.unpack('<H', elf_header[16:18])[0]
                e_machine = struct.unpack('<H', elf_header[18:20])[0]
                e_version = struct.unpack('<I', elf_header[20:24])[0]
                
                if ei_class == 2:  # 64-bit
                    e_entry = struct.unpack('<Q', elf_header[24:32])[0]
                    e_phoff = struct.unpack('<Q', elf_header[32:40])[0]
                    e_shoff = struct.unpack('<Q', elf_header[40:48])[0]
                else:  # 32-bit
                    e_entry = struct.unpack('<I', elf_header[24:28])[0]
                    e_phoff = struct.unpack('<I', elf_header[28:32])[0]
                    e_shoff = struct.unpack('<I', elf_header[32:36])[0]
                
                return {
                    'class': '64-bit' if ei_class == 2 else '32-bit',
                    'data_encoding': 'little-endian' if ei_data == 1 else 'big-endian',
                    'version': ei_version,
                    'os_abi': self._get_elf_osabi_name(ei_osabi),
                    'file_type': self._get_elf_type_name(e_type),
                    'machine': self._get_elf_machine_name(e_machine),
                    'entry_point': e_entry,
                    'program_header_offset': e_phoff,
                    'section_header_offset': e_shoff
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def _get_elf_osabi_name(self, osabi: int) -> str:
        """Get ELF OS/ABI name"""
        osabi_names = {
            0: 'SYSV',
            1: 'HPUX',
            2: 'NETBSD',
            3: 'LINUX',
            4: 'HURD',
            6: 'SOLARIS',
            7: 'AIX',
            8: 'IRIX',
            9: 'FREEBSD',
            10: 'TRU64',
            11: 'MODESTO',
            12: 'OPENBSD'
        }
        return osabi_names.get(osabi, f'Unknown ({osabi})')
    
    def _get_elf_type_name(self, e_type: int) -> str:
        """Get ELF file type name"""
        type_names = {
            0: 'ET_NONE',
            1: 'ET_REL',
            2: 'ET_EXEC',
            3: 'ET_DYN',
            4: 'ET_CORE'
        }
        return type_names.get(e_type, f'Unknown ({e_type})')
    
    def _get_elf_machine_name(self, machine: int) -> str:
        """Get ELF machine type name"""
        machine_names = {
            0: 'EM_NONE',
            3: 'EM_386',
            8: 'EM_MIPS',
            20: 'EM_PPC',
            40: 'EM_ARM',
            62: 'EM_X86_64',
            183: 'EM_AARCH64'
        }
        return machine_names.get(machine, f'Unknown ({machine})')
    
    def _parse_elf_sections(self, binary_path: str) -> Dict[str, Any]:
        """Parse ELF sections"""
        try:
            sections = {}
            with open(binary_path, 'rb') as f:
                # Get basic header info first
                elf_header = f.read(64)
                ei_class = elf_header[4]  # 32-bit or 64-bit
                
                if ei_class == 2:  # 64-bit
                    e_shoff = struct.unpack('<Q', elf_header[40:48])[0]
                    e_shentsize = struct.unpack('<H', elf_header[58:60])[0]
                    e_shnum = struct.unpack('<H', elf_header[60:62])[0]
                    e_shstrndx = struct.unpack('<H', elf_header[62:64])[0]
                else:  # 32-bit
                    e_shoff = struct.unpack('<I', elf_header[32:36])[0]
                    e_shentsize = struct.unpack('<H', elf_header[46:48])[0]
                    e_shnum = struct.unpack('<H', elf_header[48:50])[0]
                    e_shstrndx = struct.unpack('<H', elf_header[50:52])[0]
                
                # Read section headers (simplified)
                if e_shoff > 0 and e_shnum > 0:
                    f.seek(e_shoff)
                    for i in range(min(e_shnum, 50)):  # Limit for safety
                        if ei_class == 2:  # 64-bit section header
                            section_header = f.read(64)
                            if len(section_header) < 64:
                                break
                            sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack('<IIQQQQ', section_header[:40])
                        else:  # 32-bit section header
                            section_header = f.read(40)
                            if len(section_header) < 40:
                                break
                            sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size = struct.unpack('<IIIIII', section_header[:24])
                        
                        sections[f'section_{i}'] = {
                            'name_offset': sh_name,
                            'type': self._get_elf_section_type_name(sh_type),
                            'flags': sh_flags,
                            'virtual_address': sh_addr,
                            'file_offset': sh_offset,
                            'size': sh_size
                        }
            
            return sections
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_elf_section_type_name(self, sh_type: int) -> str:
        """Get ELF section type name"""
        type_names = {
            0: 'SHT_NULL',
            1: 'SHT_PROGBITS',
            2: 'SHT_SYMTAB',
            3: 'SHT_STRTAB',
            4: 'SHT_RELA',
            5: 'SHT_HASH',
            6: 'SHT_DYNAMIC',
            7: 'SHT_NOTE',
            8: 'SHT_NOBITS',
            9: 'SHT_REL',
            11: 'SHT_DYNSYM'
        }
        return type_names.get(sh_type, f'Unknown ({sh_type})')
    
    def _parse_elf_symbols(self, binary_path: str) -> Dict[str, Any]:
        """Parse ELF symbol table (basic implementation)"""
        raise NotImplementedError("ELF symbol table parsing not implemented - requires ELF symbol table interpretation")
    
    def _parse_elf_dynamic(self, binary_path: str) -> Dict[str, Any]:
        """Parse ELF dynamic section"""
        raise NotImplementedError("ELF dynamic section parsing not implemented - requires dynamic linking table interpretation")
    
    def _parse_elf_program_headers(self, binary_path: str) -> Dict[str, Any]:
        """Parse ELF program headers"""
        raise NotImplementedError("ELF program header parsing not implemented - requires program header table interpretation")
    
    # Mach-O Binary Format Support (macOS)
    def _is_macho_file(self, binary_path: str) -> bool:
        """Check if file is a Mach-O executable"""
        try:
            with open(binary_path, 'rb') as f:
                magic = f.read(4)
                # Mach-O magic numbers
                macho_magics = [
                    b'\xfe\xed\xfa\xce',  # MH_MAGIC (32-bit big-endian)
                    b'\xce\xfa\xed\xfe',  # MH_MAGIC (32-bit little-endian)
                    b'\xfe\xed\xfa\xcf',  # MH_MAGIC_64 (64-bit big-endian)
                    b'\xcf\xfa\xed\xfe',  # MH_MAGIC_64 (64-bit little-endian)
                    b'\xca\xfe\xba\xbe',  # FAT_MAGIC (universal binary)
                    b'\xbe\xba\xfe\xca'   # FAT_MAGIC (universal binary, swapped)
                ]
                return magic in macho_magics
        except:
            return False
    
    def _parse_macho_header(self, binary_path: str) -> Dict[str, Any]:
        """Parse Mach-O header (basic implementation)"""
        try:
            with open(binary_path, 'rb') as f:
                magic = f.read(4)
                
                # Determine endianness and architecture
                if magic == b'\xfe\xed\xfa\xce':
                    arch = '32-bit big-endian'
                elif magic == b'\xce\xfa\xed\xfe':
                    arch = '32-bit little-endian'
                elif magic == b'\xfe\xed\xfa\xcf':
                    arch = '64-bit big-endian'
                elif magic == b'\xcf\xfa\xed\xfe':
                    arch = '64-bit little-endian'
                else:
                    arch = 'Universal binary or unknown'
                
                return {
                    'magic': magic.hex(),
                    'architecture': arch,
                    'macho_analysis': 'Basic Mach-O detection implemented',
                    'confidence': 0.5
                }
                
        except Exception as e:
            return {'error': str(e)}
    
    def _parse_macho_load_commands(self, binary_path: str) -> Dict[str, Any]:
        """Parse Mach-O load commands"""
        raise NotImplementedError("Mach-O load command parsing not implemented - requires Mach-O format specification implementation")
    
    def _parse_macho_sections(self, binary_path: str) -> Dict[str, Any]:
        """Parse Mach-O sections"""
        raise NotImplementedError("Mach-O section parsing not implemented - requires Mach-O section header interpretation")