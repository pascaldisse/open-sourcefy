"""
Agent 5: Binary Structure Analyzer
Analyzes binary structure, sections, and memory layout.
"""

import struct
import os
from typing import Dict, Any, List, Tuple
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent5_BinaryStructureAnalyzer(BaseAgent):
    """Agent 5: Binary structure and memory layout analysis"""
    
    def __init__(self):
        super().__init__(
            agent_id=5,
            name="BinaryStructureAnalyzer", 
            dependencies=[2]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute binary structure analysis"""
        # Get data from Agent 2
        agent2_result = context['agent_results'].get(2)
        if not agent2_result or agent2_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 2 (ArchAnalysis) did not complete successfully"
            )

        # Get binary path from context
        binary_path = context['global_data'].get('binary_path')
        if not binary_path or not os.path.exists(binary_path):
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Binary path not found in context"
            )

        try:
            arch_analysis = agent2_result.data
            agent1_result = context['agent_results'].get(1)
            binary_info = agent1_result.data if agent1_result else {}
            
            structure_analysis = self._analyze_binary_structure(binary_path, arch_analysis, binary_info)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=structure_analysis,
                metadata={
                    'depends_on': [2],
                    'analysis_type': 'binary_structure_analysis'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Binary structure analysis failed: {str(e)}"
            )

    def _analyze_binary_structure(self, binary_path: str, arch_analysis: Dict[str, Any], binary_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive binary structure analysis"""
        analysis = {
            'memory_layout': {},
            'sections': [],
            'segments': [],
            'entry_points': [],
            'data_structures': {},
            'address_space': {},
            'relocations': [],
            'debug_info': {}
        }
        
        # Get format from binary info
        binary_format = binary_info.get('format_info', {}).get('format', 'Unknown')
        
        if binary_format == 'PE':
            analysis.update(self._analyze_pe_structure(binary_path))
        elif binary_format == 'ELF':
            analysis.update(self._analyze_elf_structure(binary_path))
        elif binary_format == 'Mach-O':
            analysis.update(self._analyze_macho_structure(binary_path))
        else:
            analysis.update(self._analyze_generic_structure(binary_path))
        
        # Add architecture-specific analysis
        analysis['memory_layout'] = self._analyze_memory_layout(analysis, arch_analysis)
        analysis['address_space'] = self._analyze_address_space(analysis, arch_analysis)
        
        return analysis

    def _analyze_pe_structure(self, binary_path: str) -> Dict[str, Any]:
        """Analyze PE file structure"""
        pe_analysis = {
            'pe_header': {},
            'sections': [],
            'imports': [],
            'exports': [],
            'resources': [],
            'relocations': []
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                if not dos_header.startswith(b'MZ'):
                    return pe_analysis
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset)
                
                # Read PE signature
                pe_signature = f.read(4)
                if pe_signature != b'PE\\x00\\x00':
                    return pe_analysis
                
                # Read COFF header
                coff_header = f.read(20)
                machine, num_sections, timestamp, ptr_to_symbols, num_symbols, opt_header_size, characteristics = struct.unpack('<HHIIIHH', coff_header)
                
                pe_analysis['pe_header'] = {
                    'machine': machine,
                    'num_sections': num_sections,
                    'timestamp': timestamp,
                    'characteristics': characteristics,
                    'optional_header_size': opt_header_size
                }
                
                # Read optional header
                if opt_header_size > 0:
                    opt_header = f.read(opt_header_size)
                    pe_analysis['pe_header']['optional_header'] = self._parse_pe_optional_header(opt_header)
                
                # Read section headers
                pe_analysis['sections'] = self._parse_pe_sections(f, num_sections)
                
        except Exception as e:
            self.logger.error(f"PE structure analysis failed: {e}")
            pe_analysis['analysis_error'] = str(e)
        
        return pe_analysis

    def _parse_pe_optional_header(self, opt_header: bytes) -> Dict[str, Any]:
        """Parse PE optional header"""
        if len(opt_header) < 28:
            return {}
        
        magic = struct.unpack('<H', opt_header[0:2])[0]
        is_64bit = magic == 0x20b
        
        if is_64bit:
            # PE32+
            entry_point, image_base = struct.unpack('<IQ', opt_header[16:28])
        else:
            # PE32
            entry_point, image_base = struct.unpack('<II', opt_header[16:24])
        
        return {
            'magic': magic,
            'is_64bit': is_64bit,
            'entry_point': entry_point,
            'image_base': image_base
        }

    def _parse_pe_sections(self, file_handle, num_sections: int) -> List[Dict[str, Any]]:
        """Parse PE section headers"""
        sections = []
        
        for i in range(num_sections):
            section_header = file_handle.read(40)
            if len(section_header) < 40:
                break
            
            name = section_header[0:8].rstrip(b'\\x00').decode('ascii', errors='ignore')
            virtual_size, virtual_address, raw_size, raw_address, relocs, line_nums, num_relocs, num_line_nums, characteristics = struct.unpack('<IIIIIIIHHI', section_header[8:40])
            
            sections.append({
                'name': name,
                'virtual_size': virtual_size,
                'virtual_address': virtual_address,
                'raw_size': raw_size,
                'raw_address': raw_address,
                'characteristics': characteristics,
                'type': self._get_section_type(name, characteristics)
            })
        
        return sections

    def _analyze_elf_structure(self, binary_path: str) -> Dict[str, Any]:
        """Analyze ELF file structure"""
        elf_analysis = {
            'elf_header': {},
            'program_headers': [],
            'section_headers': [],
            'symbols': [],
            'dynamic_entries': []
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read ELF header
                elf_header = f.read(64)
                if not elf_header.startswith(b'\\x7fELF'):
                    return elf_analysis
                
                # Parse ELF header
                ei_class = elf_header[4]  # 32-bit or 64-bit
                ei_data = elf_header[5]   # Endianness
                is_64bit = ei_class == 2
                is_little_endian = ei_data == 1
                
                endian_char = '<' if is_little_endian else '>'
                
                if is_64bit:
                    # 64-bit ELF
                    e_type, e_machine, e_version, e_entry, e_phoff, e_shoff = struct.unpack(f'{endian_char}HHIQqq', elf_header[16:48])
                    e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack(f'{endian_char}IHHHHH', elf_header[48:64])
                else:
                    # 32-bit ELF
                    e_type, e_machine, e_version, e_entry, e_phoff, e_shoff = struct.unpack(f'{endian_char}HHIIII', elf_header[16:36])
                    e_flags, e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx = struct.unpack(f'{endian_char}IHHHHH', elf_header[36:48])
                
                elf_analysis['elf_header'] = {
                    'class': ei_class,
                    'data': ei_data,
                    'is_64bit': is_64bit,
                    'type': e_type,
                    'machine': e_machine,
                    'entry_point': e_entry,
                    'program_header_offset': e_phoff,
                    'section_header_offset': e_shoff,
                    'program_header_count': e_phnum,
                    'section_header_count': e_shnum
                }
                
                # Parse program headers
                if e_phoff > 0 and e_phnum > 0:
                    elf_analysis['program_headers'] = self._parse_elf_program_headers(f, e_phoff, e_phnum, e_phentsize, is_64bit, endian_char)
                
                # Parse section headers
                if e_shoff > 0 and e_shnum > 0:
                    elf_analysis['section_headers'] = self._parse_elf_section_headers(f, e_shoff, e_shnum, e_shentsize, is_64bit, endian_char)
                
        except Exception as e:
            self.logger.error(f"ELF structure analysis failed: {e}")
            elf_analysis['analysis_error'] = str(e)
        
        return elf_analysis

    def _parse_elf_program_headers(self, file_handle, offset: int, count: int, entry_size: int, is_64bit: bool, endian_char: str) -> List[Dict[str, Any]]:
        """Parse ELF program headers"""
        headers = []
        file_handle.seek(offset)
        
        for i in range(count):
            if is_64bit:
                header_data = file_handle.read(56)
                if len(header_data) < 56:
                    break
                p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack(f'{endian_char}IIQQQQQQ', header_data)
            else:
                header_data = file_handle.read(32)
                if len(header_data) < 32:
                    break
                p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align = struct.unpack(f'{endian_char}IIIIIIII', header_data)
            
            headers.append({
                'type': p_type,
                'flags': p_flags,
                'offset': p_offset,
                'virtual_address': p_vaddr,
                'physical_address': p_paddr,
                'file_size': p_filesz,
                'memory_size': p_memsz,
                'alignment': p_align
            })
        
        return headers

    def _parse_elf_section_headers(self, file_handle, offset: int, count: int, entry_size: int, is_64bit: bool, endian_char: str) -> List[Dict[str, Any]]:
        """Parse ELF section headers"""
        headers = []
        file_handle.seek(offset)
        
        for i in range(count):
            if is_64bit:
                header_data = file_handle.read(64)
                if len(header_data) < 64:
                    break
                sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack(f'{endian_char}IIQQQQIIQQ', header_data)
            else:
                header_data = file_handle.read(40)
                if len(header_data) < 40:
                    break
                sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack(f'{endian_char}IIIIIIIIII', header_data)
            
            headers.append({
                'name_offset': sh_name,
                'type': sh_type,
                'flags': sh_flags,
                'address': sh_addr,
                'offset': sh_offset,
                'size': sh_size,
                'link': sh_link,
                'info': sh_info,
                'alignment': sh_addralign,
                'entry_size': sh_entsize
            })
        
        return headers

    def _analyze_macho_structure(self, binary_path: str) -> Dict[str, Any]:
        """Analyze Mach-O file structure"""
        macho_analysis = {
            'mach_header': {},
            'load_commands': [],
            'segments': [],
            'sections': []
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read Mach-O header
                header = f.read(32)
                magic = struct.unpack('<I', header[0:4])[0]
                
                is_64bit = magic in [0xfeedfacf, 0xcffaedfe]
                is_little_endian = magic in [0xfeedface, 0xfeedfacf]
                
                endian_char = '<' if is_little_endian else '>'
                
                if is_64bit:
                    cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags, reserved = struct.unpack(f'{endian_char}IIIIIIII', header[4:32])
                else:
                    cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags = struct.unpack(f'{endian_char}IIIIII', header[4:28])
                    reserved = 0
                
                macho_analysis['mach_header'] = {
                    'magic': magic,
                    'is_64bit': is_64bit,
                    'cputype': cputype,
                    'cpusubtype': cpusubtype,
                    'filetype': filetype,
                    'ncmds': ncmds,
                    'sizeofcmds': sizeofcmds,
                    'flags': flags
                }
                
                # Parse load commands (simplified)
                offset = 32 if is_64bit else 28
                for i in range(ncmds):
                    if offset >= len(header) + sizeofcmds:
                        break
                    
                    f.seek(offset)
                    cmd_header = f.read(8)
                    if len(cmd_header) < 8:
                        break
                    
                    cmd, cmdsize = struct.unpack(f'{endian_char}II', cmd_header)
                    macho_analysis['load_commands'].append({
                        'cmd': cmd,
                        'cmdsize': cmdsize,
                        'offset': offset
                    })
                    
                    offset += cmdsize
                
        except Exception as e:
            self.logger.error(f"Mach-O structure analysis failed: {e}")
            macho_analysis['analysis_error'] = str(e)
        
        return macho_analysis

    def _analyze_generic_structure(self, binary_path: str) -> Dict[str, Any]:
        """Generic binary structure analysis"""
        analysis = {
            'file_size': 0,
            'entropy_analysis': {},
            'byte_patterns': {},
            'potential_sections': []
        }
        
        try:
            stat = os.stat(binary_path)
            analysis['file_size'] = stat.st_size
            
            # Simple entropy analysis
            with open(binary_path, 'rb') as f:
                data = f.read(min(1024 * 1024, stat.st_size))  # Read first 1MB
                analysis['entropy_analysis'] = self._calculate_entropy(data)
                analysis['byte_patterns'] = self._analyze_byte_patterns(data)
                
        except Exception as e:
            self.logger.error(f"Generic structure analysis failed: {e}")
            analysis['analysis_error'] = str(e)
        
        return analysis

    def _analyze_memory_layout(self, structure_analysis: Dict[str, Any], arch_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory layout based on structure and architecture"""
        layout = {
            'base_address': 0,
            'entry_point': 0,
            'code_sections': [],
            'data_sections': [],
            'virtual_size': 0,
            'address_ranges': {}
        }
        
        # Extract base address and entry point
        if 'pe_header' in structure_analysis:
            pe_header = structure_analysis['pe_header']
            layout['base_address'] = pe_header.get('optional_header', {}).get('image_base', 0)
            layout['entry_point'] = pe_header.get('optional_header', {}).get('entry_point', 0)
            
        elif 'elf_header' in structure_analysis:
            elf_header = structure_analysis['elf_header']
            layout['entry_point'] = elf_header.get('entry_point', 0)
            # ELF base address typically determined at runtime
            
        # Categorize sections
        sections = structure_analysis.get('sections', [])
        for section in sections:
            section_type = self._categorize_section(section)
            if section_type == 'code':
                layout['code_sections'].append(section)
            elif section_type == 'data':
                layout['data_sections'].append(section)
        
        return layout

    def _analyze_address_space(self, structure_analysis: Dict[str, Any], arch_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze address space usage"""
        address_space = {
            'architecture': arch_analysis.get('architecture', 'Unknown'),
            'address_size': 0,
            'used_ranges': [],
            'gaps': [],
            'alignment': 0
        }
        
        # Determine address size from architecture
        arch = arch_analysis.get('architecture', 'Unknown')
        bitness = arch_analysis.get('abi_info', {}).get('pointer_size', 0)
        
        if bitness == 8:
            address_space['address_size'] = 64
        elif bitness == 4:
            address_space['address_size'] = 32
        
        return address_space

    def _get_section_type(self, name: str, characteristics: int) -> str:
        """Determine section type from name and characteristics"""
        name = name.lower()
        
        if name in ['.text', '.code']:
            return 'code'
        elif name in ['.data', '.bss', '.rodata']:
            return 'data'
        elif name in ['.rsrc', '.resource']:
            return 'resource'
        elif name in ['.reloc', '.rel']:
            return 'relocation'
        else:
            return 'unknown'

    def _categorize_section(self, section: Dict[str, Any]) -> str:
        """Categorize section based on its properties"""
        name = section.get('name', '').lower()
        
        if 'text' in name or 'code' in name:
            return 'code'
        elif 'data' in name or 'bss' in name or 'rodata' in name:
            return 'data'
        elif 'rsrc' in name or 'resource' in name:
            return 'resource'
        else:
            return 'unknown'

    def _calculate_entropy(self, data: bytes) -> Dict[str, float]:
        """Calculate entropy of binary data"""
        if not data:
            return {'entropy': 0.0}
        
        # Count byte frequencies
        frequencies = [0] * 256
        for byte in data:
            frequencies[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        length = len(data)
        
        for freq in frequencies:
            if freq > 0:
                probability = freq / length
                entropy -= probability * (probability.bit_length() - 1)
        
        return {
            'entropy': entropy,
            'max_entropy': 8.0,
            'normalized_entropy': entropy / 8.0 if entropy > 0 else 0.0
        }

    def _analyze_byte_patterns(self, data: bytes) -> Dict[str, Any]:
        """Analyze byte patterns in binary data"""
        patterns = {
            'null_bytes': 0,
            'printable_bytes': 0,
            'high_entropy_regions': 0,
            'repeated_patterns': []
        }
        
        if not data:
            return patterns
        
        # Count different byte types
        for byte in data:
            if byte == 0:
                patterns['null_bytes'] += 1
            elif 32 <= byte <= 126:  # Printable ASCII
                patterns['printable_bytes'] += 1
        
        # Calculate percentages
        total_bytes = len(data)
        patterns['null_byte_percentage'] = (patterns['null_bytes'] / total_bytes) * 100
        patterns['printable_percentage'] = (patterns['printable_bytes'] / total_bytes) * 100
        
        return patterns