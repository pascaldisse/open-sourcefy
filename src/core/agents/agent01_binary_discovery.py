"""
Agent 1: Binary Discovery
Analyzes binary file format, detects architecture, and extracts basic metadata.
"""

import os
import struct
import hashlib
from typing import Dict, Any, Optional, List
from ..agent_base import BaseAgent, AgentResult, AgentStatus

# Import binary analysis libraries with error handling
try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False

try:
    from elftools.elf.elffile import ELFFile
    from elftools.common.exceptions import ELFError
    HAS_ELFTOOLS = True
except ImportError:
    HAS_ELFTOOLS = False

try:
    from macholib.MachO import MachO
    from macholib.mach_o import CPU_TYPE_NAMES, MH_MAGIC, MH_MAGIC_64
    HAS_MACHOLIB = True
except ImportError:
    HAS_MACHOLIB = False


class Agent1_BinaryDiscovery(BaseAgent):
    """Agent 1: Binary file discovery and basic analysis"""
    
    def __init__(self):
        super().__init__(
            agent_id=1,
            name="BinaryDiscovery",
            dependencies=[]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute binary discovery analysis"""
        binary_path = context['global_data'].get('binary_path')
        if not binary_path:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="No binary path provided in context"
            )

        if not os.path.exists(binary_path):
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Binary file not found: {binary_path}"
            )

        try:
            discovery_data = self._analyze_binary(binary_path)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=discovery_data,
                metadata={
                    'binary_path': binary_path,
                    'analysis_type': 'binary_discovery'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Binary analysis failed: {str(e)}"
            )

    def _analyze_binary(self, binary_path: str) -> Dict[str, Any]:
        """Perform detailed binary analysis"""
        file_info = self._get_file_info(binary_path)
        format_info = self._detect_format(binary_path)
        
        data = {
            'file_info': file_info,
            'format_info': format_info,
            'architecture': self._detect_architecture(binary_path),
            'checksums': self._calculate_checksums(binary_path)
        }
        
        # Format-specific analysis
        format_type = format_info.get('format', 'Unknown')
        if format_type == 'PE' and HAS_PEFILE:
            data['pe_analysis'] = self._analyze_pe(binary_path)
        elif format_type == 'ELF' and HAS_ELFTOOLS:
            data['elf_analysis'] = self._analyze_elf(binary_path)
        elif format_type == 'Mach-O' and HAS_MACHOLIB:
            data['macho_analysis'] = self._analyze_macho(binary_path)
        else:
            data['format_analysis'] = {
                'status': 'limited',
                'reason': f"No parser available for {format_type} format"
            }
        
        return data

    def _get_file_info(self, binary_path: str) -> Dict[str, Any]:
        """Get basic file information"""
        stat = os.stat(binary_path)
        return {
            'filename': os.path.basename(binary_path),
            'filepath': binary_path,
            'size': stat.st_size,
            'modified_time': stat.st_mtime,
            'created_time': stat.st_ctime
        }

    def _detect_format(self, binary_path: str) -> Dict[str, Any]:
        """Detect binary format (PE, ELF, Mach-O, etc.)"""
        with open(binary_path, 'rb') as f:
            header = f.read(64)
        
        if header.startswith(b'MZ'):
            return {'format': 'PE', 'subtype': 'Windows Executable'}
        elif header.startswith(b'\x7fELF'):
            return {'format': 'ELF', 'subtype': 'Unix/Linux Executable'}
        elif header.startswith(b'\xfe\xed\xfa\xce') or header.startswith(b'\xce\xfa\xed\xfe'):
            return {'format': 'Mach-O', 'subtype': 'macOS Executable'}
        elif header.startswith(b'\xcf\xfa\xed\xfe') or header.startswith(b'\xfe\xed\xfa\xcf'):
            return {'format': 'Mach-O', 'subtype': 'macOS 64-bit Executable'}
        else:
            return {'format': 'Unknown', 'subtype': 'Unknown Binary Format'}

    def _detect_architecture(self, binary_path: str) -> Dict[str, Any]:
        """Detect target architecture"""
        with open(binary_path, 'rb') as f:
            header = f.read(64)
        
        arch_info = {'architecture': 'Unknown', 'bitness': 'Unknown', 'endianness': 'Unknown'}
        
        if header.startswith(b'MZ'):  # PE
            # Read PE header location
            with open(binary_path, 'rb') as f:
                f.seek(0x3c)
                pe_offset = struct.unpack('<I', f.read(4))[0]
                f.seek(pe_offset + 4)  # Skip PE signature
                machine = struct.unpack('<H', f.read(2))[0]
            
                if machine == 0x014c:  # IMAGE_FILE_MACHINE_I386
                    arch_info = {'architecture': 'x86', 'bitness': '32-bit', 'endianness': 'little'}
                elif machine == 0x8664:  # IMAGE_FILE_MACHINE_AMD64
                    arch_info = {'architecture': 'x64', 'bitness': '64-bit', 'endianness': 'little'}
                elif machine == 0x01c0:  # IMAGE_FILE_MACHINE_ARM
                    arch_info = {'architecture': 'ARM', 'bitness': '32-bit', 'endianness': 'little'}
                elif machine == 0xaa64:  # IMAGE_FILE_MACHINE_ARM64
                    arch_info = {'architecture': 'ARM64', 'bitness': '64-bit', 'endianness': 'little'}
                
        elif header.startswith(b'\x7fELF'):  # ELF
            ei_class = header[4]
            ei_data = header[5]
            e_machine = struct.unpack('<H' if ei_data == 1 else '>H', header[18:20])[0]
            
            bitness = '32-bit' if ei_class == 1 else '64-bit'
            endianness = 'little' if ei_data == 1 else 'big'
            
            if e_machine == 0x03:  # EM_386
                arch_info = {'architecture': 'x86', 'bitness': bitness, 'endianness': endianness}
            elif e_machine == 0x3e:  # EM_X86_64
                arch_info = {'architecture': 'x64', 'bitness': bitness, 'endianness': endianness}
            elif e_machine == 0x28:  # EM_ARM
                arch_info = {'architecture': 'ARM', 'bitness': bitness, 'endianness': endianness}
            elif e_machine == 0xb7:  # EM_AARCH64
                arch_info = {'architecture': 'ARM64', 'bitness': bitness, 'endianness': endianness}
        
        return arch_info

    def _calculate_checksums(self, binary_path: str) -> Dict[str, str]:
        """Calculate various checksums for the binary"""
        checksums = {}
        
        with open(binary_path, 'rb') as f:
            data = f.read()
            
        checksums['md5'] = hashlib.md5(data).hexdigest()
        checksums['sha1'] = hashlib.sha1(data).hexdigest()
        checksums['sha256'] = hashlib.sha256(data).hexdigest()
        
        return checksums

    def _analyze_pe(self, binary_path: str) -> Dict[str, Any]:
        """Analyze PE-specific structures"""
        if not HAS_PEFILE:
            return {'error': 'pefile library not available'}
        
        try:
            pe = pefile.PE(binary_path)
            
            analysis = {
                'pe_header': self._extract_pe_header(pe),
                'sections': self._extract_pe_sections(pe),
                'imports': self._extract_pe_imports(pe),
                'exports': self._extract_pe_exports(pe),
                'resources': self._extract_pe_resources(pe),
                'security': self._extract_pe_security(pe),
                'version_info': self._extract_pe_version_info(pe)
            }
            
            pe.close()
            return analysis
            
        except Exception as e:
            return {'error': f'PE analysis failed: {str(e)}'}
    
    def _extract_pe_header(self, pe) -> Dict[str, Any]:
        """Extract PE header information"""
        header_info = {
            'machine': pe.FILE_HEADER.Machine,
            'machine_name': pefile.MACHINE_TYPE.get(pe.FILE_HEADER.Machine, 'Unknown'),
            'number_of_sections': pe.FILE_HEADER.NumberOfSections,
            'time_date_stamp': pe.FILE_HEADER.TimeDateStamp,
            'characteristics': pe.FILE_HEADER.Characteristics,
            'subsystem': pe.OPTIONAL_HEADER.Subsystem if hasattr(pe, 'OPTIONAL_HEADER') else None,
            'dll_characteristics': pe.OPTIONAL_HEADER.DllCharacteristics if hasattr(pe, 'OPTIONAL_HEADER') else None,
            'entry_point': pe.OPTIONAL_HEADER.AddressOfEntryPoint if hasattr(pe, 'OPTIONAL_HEADER') else None,
            'image_base': pe.OPTIONAL_HEADER.ImageBase if hasattr(pe, 'OPTIONAL_HEADER') else None
        }
        
        if hasattr(pe, 'OPTIONAL_HEADER'):
            header_info.update({
                'size_of_image': pe.OPTIONAL_HEADER.SizeOfImage,
                'size_of_headers': pe.OPTIONAL_HEADER.SizeOfHeaders,
                'checksum': pe.OPTIONAL_HEADER.CheckSum,
                'size_of_stack_reserve': pe.OPTIONAL_HEADER.SizeOfStackReserve,
                'size_of_heap_reserve': pe.OPTIONAL_HEADER.SizeOfHeapReserve
            })
        
        return header_info
    
    def _extract_pe_sections(self, pe) -> List[Dict[str, Any]]:
        """Extract PE section information"""
        sections = []
        for section in pe.sections:
            sections.append({
                'name': section.Name.decode('utf-8', errors='ignore').rstrip('\x00'),
                'virtual_address': section.VirtualAddress,
                'virtual_size': section.Misc_VirtualSize,
                'raw_size': section.SizeOfRawData,
                'raw_address': section.PointerToRawData,
                'characteristics': section.Characteristics,
                'entropy': section.get_entropy() if hasattr(section, 'get_entropy') else 0.0
            })
        return sections
    
    def _extract_pe_imports(self, pe) -> List[Dict[str, Any]]:
        """Extract PE import table information"""
        imports = []
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_imports = {
                    'dll': entry.dll.decode('utf-8', errors='ignore'),
                    'functions': []
                }
                for imp in entry.imports:
                    func_info = {
                        'name': imp.name.decode('utf-8', errors='ignore') if imp.name else None,
                        'ordinal': imp.ordinal,
                        'address': imp.address
                    }
                    dll_imports['functions'].append(func_info)
                imports.append(dll_imports)
        return imports
    
    def _extract_pe_exports(self, pe) -> Dict[str, Any]:
        """Extract PE export table information"""
        exports = {'functions': [], 'dll_name': None}
        if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
            exports['dll_name'] = pe.DIRECTORY_ENTRY_EXPORT.name.decode('utf-8', errors='ignore')
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                func_info = {
                    'name': exp.name.decode('utf-8', errors='ignore') if exp.name else None,
                    'ordinal': exp.ordinal,
                    'address': exp.address
                }
                exports['functions'].append(func_info)
        return exports
    
    def _extract_pe_resources(self, pe) -> List[Dict[str, Any]]:
        """Extract PE resource information"""
        resources = []
        if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                for resource_id in resource_type.directory.entries:
                    for resource_lang in resource_id.directory.entries:
                        resource_info = {
                            'type': resource_type.id,
                            'type_name': pefile.RESOURCE_TYPE.get(resource_type.id, 'Unknown'),
                            'id': resource_id.id,
                            'language': resource_lang.id,
                            'size': resource_lang.data.struct.Size,
                            'offset': resource_lang.data.struct.OffsetToData
                        }
                        resources.append(resource_info)
        return resources
    
    def _extract_pe_security(self, pe) -> Dict[str, Any]:
        """Extract PE security/certificate information"""
        security = {'has_signature': False, 'certificates': []}
        if hasattr(pe, 'DIRECTORY_ENTRY_SECURITY'):
            security['has_signature'] = True
            for cert in pe.DIRECTORY_ENTRY_SECURITY:
                cert_info = {
                    'size': cert.struct.dwLength,
                    'revision': cert.struct.wRevision,
                    'certificate_type': cert.struct.wCertificateType
                }
                security['certificates'].append(cert_info)
        return security
    
    def _extract_pe_version_info(self, pe) -> Dict[str, Any]:
        """Extract PE version information"""
        version_info = {}
        if hasattr(pe, 'VS_VERSIONINFO') and pe.VS_VERSIONINFO:
            for entry in pe.VS_VERSIONINFO:
                if hasattr(entry, 'StringFileInfo') and entry.StringFileInfo:
                    for string_table in entry.StringFileInfo:
                        for string_entry in string_table.entries.items():
                            version_info[string_entry[0]] = string_entry[1]
        return version_info

    def _analyze_elf(self, binary_path: str) -> Dict[str, Any]:
        """Analyze ELF-specific structures"""
        if not HAS_ELFTOOLS:
            return {'error': 'pyelftools library not available'}
        
        try:
            with open(binary_path, 'rb') as f:
                elffile = ELFFile(f)
                
                analysis = {
                    'elf_header': self._extract_elf_header(elffile),
                    'sections': self._extract_elf_sections(elffile),
                    'segments': self._extract_elf_segments(elffile),
                    'symbols': self._extract_elf_symbols(elffile),
                    'dynamic': self._extract_elf_dynamic(elffile),
                    'relocations': self._extract_elf_relocations(elffile)
                }
                
                return analysis
                
        except Exception as e:
            return {'error': f'ELF analysis failed: {str(e)}'}
    
    def _extract_elf_header(self, elffile) -> Dict[str, Any]:
        """Extract ELF header information"""
        header = elffile.header
        return {
            'ei_class': header['e_ident']['EI_CLASS'],
            'ei_data': header['e_ident']['EI_DATA'],
            'ei_version': header['e_ident']['EI_VERSION'],
            'ei_osabi': header['e_ident']['EI_OSABI'],
            'e_type': header['e_type'],
            'e_machine': header['e_machine'],
            'e_version': header['e_version'],
            'e_entry': header['e_entry'],
            'e_phoff': header['e_phoff'],
            'e_shoff': header['e_shoff'],
            'e_flags': header['e_flags'],
            'e_ehsize': header['e_ehsize'],
            'e_phentsize': header['e_phentsize'],
            'e_phnum': header['e_phnum'],
            'e_shentsize': header['e_shentsize'],
            'e_shnum': header['e_shnum'],
            'e_shstrndx': header['e_shstrndx']
        }
    
    def _extract_elf_sections(self, elffile) -> List[Dict[str, Any]]:
        """Extract ELF section information"""
        sections = []
        for section in elffile.iter_sections():
            sections.append({
                'name': section.name,
                'type': section['sh_type'],
                'flags': section['sh_flags'],
                'addr': section['sh_addr'],
                'offset': section['sh_offset'],
                'size': section['sh_size'],
                'link': section['sh_link'],
                'info': section['sh_info'],
                'addralign': section['sh_addralign'],
                'entsize': section['sh_entsize']
            })
        return sections
    
    def _extract_elf_segments(self, elffile) -> List[Dict[str, Any]]:
        """Extract ELF program header/segment information"""
        segments = []
        for segment in elffile.iter_segments():
            segments.append({
                'type': segment['p_type'],
                'flags': segment['p_flags'],
                'offset': segment['p_offset'],
                'vaddr': segment['p_vaddr'],
                'paddr': segment['p_paddr'],
                'filesz': segment['p_filesz'],
                'memsz': segment['p_memsz'],
                'align': segment['p_align']
            })
        return segments
    
    def _extract_elf_symbols(self, elffile) -> Dict[str, List[Dict[str, Any]]]:
        """Extract ELF symbol table information"""
        symbols = {'static': [], 'dynamic': []}
        
        # Static symbols (.symtab)
        symtab = elffile.get_section_by_name('.symtab')
        if symtab:
            for symbol in symtab.iter_symbols():
                symbols['static'].append({
                    'name': symbol.name,
                    'value': symbol['st_value'],
                    'size': symbol['st_size'],
                    'type': symbol['st_info']['type'],
                    'bind': symbol['st_info']['bind'],
                    'visibility': symbol['st_other']['visibility'],
                    'shndx': symbol['st_shndx']
                })
        
        # Dynamic symbols (.dynsym)
        dynsym = elffile.get_section_by_name('.dynsym')
        if dynsym:
            for symbol in dynsym.iter_symbols():
                symbols['dynamic'].append({
                    'name': symbol.name,
                    'value': symbol['st_value'],
                    'size': symbol['st_size'],
                    'type': symbol['st_info']['type'],
                    'bind': symbol['st_info']['bind'],
                    'visibility': symbol['st_other']['visibility'],
                    'shndx': symbol['st_shndx']
                })
        
        return symbols
    
    def _extract_elf_dynamic(self, elffile) -> List[Dict[str, Any]]:
        """Extract ELF dynamic section information"""
        dynamic_entries = []
        dynamic_section = elffile.get_section_by_name('.dynamic')
        if dynamic_section:
            for tag in dynamic_section.iter_tags():
                dynamic_entries.append({
                    'tag': tag['d_tag'],
                    'value': tag['d_val'],
                    'ptr': tag.get('d_ptr', 0)
                })
        return dynamic_entries
    
    def _extract_elf_relocations(self, elffile) -> List[Dict[str, Any]]:
        """Extract ELF relocation information"""
        relocations = []
        for section in elffile.iter_sections():
            if hasattr(section, 'iter_relocations'):
                for reloc in section.iter_relocations():
                    relocations.append({
                        'section': section.name,
                        'offset': reloc['r_offset'],
                        'info': reloc['r_info'],
                        'type': reloc['r_info_type'],
                        'sym': reloc['r_info_sym'],
                        'addend': reloc.get('r_addend', 0)
                    })
        return relocations

    def _analyze_macho(self, binary_path: str) -> Dict[str, Any]:
        """Analyze Mach-O specific structures"""
        if not HAS_MACHOLIB:
            return {'error': 'macholib library not available'}
        
        try:
            macho = MachO(binary_path)
            
            analysis = {
                'architectures': [],
                'load_commands': [],
                'segments': [],
                'symbols': []
            }
            
            for header in macho.headers:
                arch_analysis = {
                    'header': self._extract_macho_header(header),
                    'load_commands': self._extract_macho_load_commands(header),
                    'segments': self._extract_macho_segments(header)
                }
                analysis['architectures'].append(arch_analysis)
            
            return analysis
            
        except Exception as e:
            return {'error': f'Mach-O analysis failed: {str(e)}'}
    
    def _extract_macho_header(self, header) -> Dict[str, Any]:
        """Extract Mach-O header information"""
        return {
            'magic': header.header.magic,
            'cputype': header.header.cputype,
            'cpusubtype': header.header.cpusubtype,
            'filetype': header.header.filetype,
            'ncmds': header.header.ncmds,
            'sizeofcmds': header.header.sizeofcmds,
            'flags': header.header.flags
        }
    
    def _extract_macho_load_commands(self, header) -> List[Dict[str, Any]]:
        """Extract Mach-O load command information"""
        load_commands = []
        for load_command, cmd, data in header.commands:
            cmd_info = {
                'cmd': cmd.cmd,
                'cmdsize': cmd.cmdsize,
                'type': type(cmd).__name__
            }
            
            # Add specific command data
            if hasattr(cmd, 'segname'):
                cmd_info['segname'] = cmd.segname.decode('utf-8', errors='ignore').rstrip('\x00')
            if hasattr(cmd, 'vmaddr'):
                cmd_info['vmaddr'] = cmd.vmaddr
            if hasattr(cmd, 'vmsize'):
                cmd_info['vmsize'] = cmd.vmsize
            if hasattr(cmd, 'fileoff'):
                cmd_info['fileoff'] = cmd.fileoff
            if hasattr(cmd, 'filesize'):
                cmd_info['filesize'] = cmd.filesize
                
            load_commands.append(cmd_info)
        
        return load_commands
    
    def _extract_macho_segments(self, header) -> List[Dict[str, Any]]:
        """Extract Mach-O segment information"""
        segments = []
        for load_command, cmd, data in header.commands:
            if hasattr(cmd, 'segname'):
                segment_info = {
                    'name': cmd.segname.decode('utf-8', errors='ignore').rstrip('\x00'),
                    'vmaddr': getattr(cmd, 'vmaddr', 0),
                    'vmsize': getattr(cmd, 'vmsize', 0),
                    'fileoff': getattr(cmd, 'fileoff', 0),
                    'filesize': getattr(cmd, 'filesize', 0),
                    'maxprot': getattr(cmd, 'maxprot', 0),
                    'initprot': getattr(cmd, 'initprot', 0),
                    'nsects': getattr(cmd, 'nsects', 0),
                    'flags': getattr(cmd, 'flags', 0),
                    'sections': []
                }
                
                # Extract sections if available
                if hasattr(cmd, 'sections'):
                    for section in cmd.sections:
                        section_info = {
                            'sectname': section.sectname.decode('utf-8', errors='ignore').rstrip('\x00'),
                            'segname': section.segname.decode('utf-8', errors='ignore').rstrip('\x00'),
                            'addr': section.addr,
                            'size': section.size,
                            'offset': section.offset,
                            'align': section.align,
                            'reloff': section.reloff,
                            'nreloc': section.nreloc,
                            'flags': section.flags
                        }
                        segment_info['sections'].append(section_info)
                
                segments.append(segment_info)
        
        return segments