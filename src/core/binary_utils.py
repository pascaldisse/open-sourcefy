"""
Binary Analysis Utilities for Open-Sourcefy Matrix Pipeline
Shared utilities for binary file analysis, format detection, and metadata extraction
"""

import os
import hashlib
import struct
import mmap
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
import logging

# Optional imports with fallbacks
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False

try:
    from elftools.elf.elffile import ELFFile
    from elftools.common.py3compat import bytes2str
    HAS_PYELFTOOLS = True
except ImportError:
    HAS_PYELFTOOLS = False


class BinaryFormat(Enum):
    """Binary file format enumeration"""
    PE = "pe"
    ELF = "elf"
    MACH_O = "mach-o"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Binary architecture enumeration"""
    X86 = "x86"
    X64 = "x64"
    ARM = "arm"
    ARM64 = "arm64"
    MIPS = "mips"
    RISC_V = "risc-v"
    UNKNOWN = "unknown"


class BinaryInfo:
    """Container for binary file information"""
    
    def __init__(self):
        self.file_path: Optional[str] = None
        self.file_size: int = 0
        self.format: BinaryFormat = BinaryFormat.UNKNOWN
        self.architecture: Architecture = Architecture.UNKNOWN
        self.entry_point: Optional[int] = None
        self.base_address: Optional[int] = None
        self.sections: List[Dict[str, Any]] = []
        self.imports: List[str] = []
        self.exports: List[str] = []
        self.strings: List[str] = []
        self.hashes: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self.is_64bit: bool = False
        self.is_packed: bool = False
        self.compiler_info: Optional[str] = None
        self.confidence_score: float = 0.8  # Default confidence for binary analysis


class BinaryAnalyzer:
    """Binary file analyzer with format detection and metadata extraction"""
    
    def __init__(self):
        self.logger = logging.getLogger("BinaryAnalyzer")
    
    def analyze_file(self, file_path: Union[str, Path]) -> BinaryInfo:
        """Analyze binary file and extract comprehensive information"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Binary file not found: {file_path}")
        
        info = BinaryInfo()
        info.file_path = str(file_path)
        info.file_size = file_path.stat().st_size
        
        # Calculate file hashes
        info.hashes = self.calculate_hashes(file_path)
        
        # Detect binary format
        info.format = self.detect_format(file_path)
        
        # Extract format-specific information
        if info.format == BinaryFormat.PE:
            self._analyze_pe_file(file_path, info)
        elif info.format == BinaryFormat.ELF:
            self._analyze_elf_file(file_path, info)
        elif info.format == BinaryFormat.MACH_O:
            self._analyze_macho_file(file_path, info)
        
        # Extract strings
        info.strings = self.extract_strings(file_path)
        
        # Detect packing
        info.is_packed = self.detect_packing(info)
        
        return info
    
    def detect_format(self, file_path: Path) -> BinaryFormat:
        """Detect binary file format"""
        try:
            # Read file header
            with open(file_path, 'rb') as f:
                header = f.read(64)
            
            # PE format detection
            if header.startswith(b'MZ'):
                # Look for PE signature
                dos_header = header[:64]
                if len(dos_header) >= 60:
                    pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                    if pe_offset < file_path.stat().st_size - 4:
                        with open(file_path, 'rb') as f:
                            f.seek(pe_offset)
                            pe_sig = f.read(4)
                            if pe_sig == b'PE\x00\x00':
                                return BinaryFormat.PE
            
            # ELF format detection
            if header.startswith(b'\x7fELF'):
                return BinaryFormat.ELF
            
            # Mach-O format detection
            if header.startswith((b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf',
                               b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe')):
                return BinaryFormat.MACH_O
            
            # Additional format detection using python-magic if available
            if HAS_MAGIC:
                mime_type = magic.from_file(str(file_path), mime=True)
                if 'executable' in mime_type or 'application/x-' in mime_type:
                    # Try to determine format from magic
                    file_type = magic.from_file(str(file_path))
                    if 'PE32' in file_type or 'MS-DOS executable' in file_type:
                        return BinaryFormat.PE
                    elif 'ELF' in file_type:
                        return BinaryFormat.ELF
                    elif 'Mach-O' in file_type:
                        return BinaryFormat.MACH_O
            
        except Exception as e:
            self.logger.warning(f"Error detecting format for {file_path}: {e}")
        
        return BinaryFormat.UNKNOWN
    
    def _analyze_pe_file(self, file_path: Path, info: BinaryInfo):
        """Analyze PE file format"""
        if not HAS_PEFILE:
            self.logger.warning("pefile not available, PE analysis limited")
            self._analyze_pe_basic(file_path, info)
            return
        
        try:
            pe = pefile.PE(str(file_path))
            
            # Basic PE information
            info.entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            info.base_address = pe.OPTIONAL_HEADER.ImageBase
            info.is_64bit = pe.PE_TYPE == pefile.OPTIONAL_HEADER_MAGIC_PE_PLUS
            
            # Architecture detection
            machine = pe.FILE_HEADER.Machine
            if machine == 0x014c:  # IMAGE_FILE_MACHINE_I386
                info.architecture = Architecture.X86
            elif machine == 0x8664:  # IMAGE_FILE_MACHINE_AMD64
                info.architecture = Architecture.X64
            elif machine == 0x01c0:  # IMAGE_FILE_MACHINE_ARM
                info.architecture = Architecture.ARM
            elif machine == 0xaa64:  # IMAGE_FILE_MACHINE_ARM64
                info.architecture = Architecture.ARM64
            
            # Sections
            for section in pe.sections:
                section_info = {
                    'name': section.Name.decode('utf-8').rstrip('\x00'),
                    'virtual_address': section.VirtualAddress,
                    'virtual_size': section.Misc_VirtualSize,
                    'raw_address': section.PointerToRawData,
                    'raw_size': section.SizeOfRawData,
                    'characteristics': section.Characteristics
                }
                info.sections.append(section_info)
            
            # Imports
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8')
                    for imp in entry.imports:
                        if imp.name:
                            func_name = imp.name.decode('utf-8')
                            info.imports.append(f"{dll_name}:{func_name}")
            
            # Exports
            if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                    if exp.name:
                        info.exports.append(exp.name.decode('utf-8'))
            
            # Compiler detection
            info.compiler_info = self._detect_pe_compiler(pe)
            
            pe.close()
            
        except Exception as e:
            self.logger.error(f"Error analyzing PE file {file_path}: {e}")
            self._analyze_pe_basic(file_path, info)
    
    def _analyze_pe_basic(self, file_path: Path, info: BinaryInfo):
        """Basic PE analysis without pefile library"""
        try:
            with open(file_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                
                # Read PE header
                f.seek(pe_offset)
                pe_signature = f.read(4)
                if pe_signature != b'PE\x00\x00':
                    return
                
                # Read COFF header
                coff_header = f.read(20)
                machine, num_sections, timestamp, ptr_to_symtab, num_symbols, size_opt_header, characteristics = struct.unpack('<HHIIIHH', coff_header)
                
                # Architecture from machine type
                if machine == 0x014c:
                    info.architecture = Architecture.X86
                elif machine == 0x8664:
                    info.architecture = Architecture.X64
                elif machine == 0x01c0:
                    info.architecture = Architecture.ARM
                elif machine == 0xaa64:
                    info.architecture = Architecture.ARM64
                
                # Read optional header
                if size_opt_header > 0:
                    opt_header = f.read(size_opt_header)
                    if len(opt_header) >= 24:
                        magic = struct.unpack('<H', opt_header[:2])[0]
                        info.is_64bit = (magic == 0x20b)  # PE32+
                        
                        if magic in (0x10b, 0x20b):  # PE32 or PE32+
                            if info.is_64bit and len(opt_header) >= 28:
                                info.entry_point = struct.unpack('<I', opt_header[16:20])[0]
                                info.base_address = struct.unpack('<Q', opt_header[24:32])[0]
                            elif not info.is_64bit and len(opt_header) >= 28:
                                info.entry_point = struct.unpack('<I', opt_header[16:20])[0]
                                info.base_address = struct.unpack('<I', opt_header[28:32])[0]
                
        except Exception as e:
            self.logger.error(f"Error in basic PE analysis for {file_path}: {e}")
    
    def _analyze_elf_file(self, file_path: Path, info: BinaryInfo):
        """Analyze ELF file format"""
        if not HAS_PYELFTOOLS:
            self.logger.warning("pyelftools not available, ELF analysis limited")
            self._analyze_elf_basic(file_path, info)
            return
        
        try:
            with open(file_path, 'rb') as f:
                elf = ELFFile(f)
                
                # Basic ELF information
                info.entry_point = elf.header['e_entry']
                info.is_64bit = elf.elfclass == 64
                
                # Architecture detection
                machine = elf.header['e_machine']
                if machine == 'EM_386':
                    info.architecture = Architecture.X86
                elif machine == 'EM_X86_64':
                    info.architecture = Architecture.X64
                elif machine == 'EM_ARM':
                    info.architecture = Architecture.ARM
                elif machine == 'EM_AARCH64':
                    info.architecture = Architecture.ARM64
                elif machine == 'EM_MIPS':
                    info.architecture = Architecture.MIPS
                elif machine == 'EM_RISCV':
                    info.architecture = Architecture.RISC_V
                
                # Sections
                for section in elf.iter_sections():
                    section_info = {
                        'name': section.name,
                        'type': section['sh_type'],
                        'address': section['sh_addr'],
                        'offset': section['sh_offset'],
                        'size': section['sh_size'],
                        'flags': section['sh_flags']
                    }
                    info.sections.append(section_info)
                
                # Dynamic symbols (imports/exports)
                symbol_tables = [s for s in elf.iter_sections() if s.name in ('.dynsym', '.symtab')]
                for symtab in symbol_tables:
                    for symbol in symtab.iter_symbols():
                        if symbol.name:
                            symbol_name = bytes2str(symbol.name)
                            if symbol['st_shndx'] == 'SHN_UNDEF':
                                info.imports.append(symbol_name)
                            else:
                                info.exports.append(symbol_name)
                
        except Exception as e:
            self.logger.error(f"Error analyzing ELF file {file_path}: {e}")
            self._analyze_elf_basic(file_path, info)
    
    def _analyze_elf_basic(self, file_path: Path, info: BinaryInfo):
        """Basic ELF analysis without pyelftools"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(64)
                
                if len(header) < 16:
                    return
                
                # ELF class (32/64-bit)
                info.is_64bit = (header[4] == 2)
                
                # Architecture from e_machine field
                machine_offset = 18
                if len(header) > machine_offset + 1:
                    machine = struct.unpack('<H', header[machine_offset:machine_offset+2])[0]
                    
                    if machine == 3:  # EM_386
                        info.architecture = Architecture.X86
                    elif machine == 62:  # EM_X86_64
                        info.architecture = Architecture.X64
                    elif machine == 40:  # EM_ARM
                        info.architecture = Architecture.ARM
                    elif machine == 183:  # EM_AARCH64
                        info.architecture = Architecture.ARM64
                    elif machine == 8:  # EM_MIPS
                        info.architecture = Architecture.MIPS
                    elif machine == 243:  # EM_RISCV
                        info.architecture = Architecture.RISC_V
                
                # Entry point
                entry_offset = 24 if info.is_64bit else 24
                if len(header) >= entry_offset + (8 if info.is_64bit else 4):
                    if info.is_64bit:
                        info.entry_point = struct.unpack('<Q', header[entry_offset:entry_offset+8])[0]
                    else:
                        info.entry_point = struct.unpack('<I', header[entry_offset:entry_offset+4])[0]
                
        except Exception as e:
            self.logger.error(f"Error in basic ELF analysis for {file_path}: {e}")
    
    def _analyze_macho_file(self, file_path: Path, info: BinaryInfo):
        """Analyze Mach-O file format (basic implementation)"""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(32)
                
                if len(header) < 28:
                    return
                
                # Mach-O magic numbers
                magic = struct.unpack('<I', header[:4])[0]
                
                if magic in (0xfeedface, 0xfeedfacf):  # 32-bit and 64-bit little endian
                    info.is_64bit = (magic == 0xfeedfacf)
                elif magic in (0xcefaedfe, 0xcffaedfe):  # 32-bit and 64-bit big endian
                    info.is_64bit = (magic == 0xcffaedfe)
                
                # CPU type
                cpu_type = struct.unpack('<I', header[4:8])[0]
                if cpu_type == 7:  # CPU_TYPE_X86
                    info.architecture = Architecture.X86
                elif cpu_type == 16777223:  # CPU_TYPE_X86_64
                    info.architecture = Architecture.X64
                elif cpu_type == 12:  # CPU_TYPE_ARM
                    info.architecture = Architecture.ARM
                elif cpu_type == 16777228:  # CPU_TYPE_ARM64
                    info.architecture = Architecture.ARM64
                
        except Exception as e:
            self.logger.error(f"Error analyzing Mach-O file {file_path}: {e}")
    
    def calculate_hashes(self, file_path: Path) -> Dict[str, str]:
        """Calculate various hashes for the binary file"""
        hashes = {}
        
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks for memory efficiency
                md5_hash = hashlib.md5()
                sha1_hash = hashlib.sha1()
                sha256_hash = hashlib.sha256()
                
                chunk_size = 8192
                while chunk := f.read(chunk_size):
                    md5_hash.update(chunk)
                    sha1_hash.update(chunk)
                    sha256_hash.update(chunk)
                
                hashes['md5'] = md5_hash.hexdigest()
                hashes['sha1'] = sha1_hash.hexdigest()
                hashes['sha256'] = sha256_hash.hexdigest()
                
        except Exception as e:
            self.logger.error(f"Error calculating hashes for {file_path}: {e}")
        
        return hashes
    
    def extract_strings(self, file_path: Path, min_length: int = 4) -> List[str]:
        """Extract printable strings from binary file"""
        strings = []
        
        try:
            with open(file_path, 'rb') as f:
                # Use memory mapping for large files
                if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        strings = self._extract_strings_from_data(mm, min_length)
                else:
                    data = f.read()
                    strings = self._extract_strings_from_data(data, min_length)
                    
        except Exception as e:
            self.logger.error(f"Error extracting strings from {file_path}: {e}")
        
        return strings[:1000]  # Limit to first 1000 strings
    
    def _extract_strings_from_data(self, data, min_length: int) -> List[str]:
        """Extract strings from binary data"""
        strings = []
        current_string = bytearray()
        
        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII
                current_string.append(byte)
            else:
                if len(current_string) >= min_length:
                    try:
                        string_val = current_string.decode('ascii')
                        strings.append(string_val)
                    except UnicodeDecodeError:
                        pass
                current_string = bytearray()
        
        # Handle final string
        if len(current_string) >= min_length:
            try:
                string_val = current_string.decode('ascii')
                strings.append(string_val)
            except UnicodeDecodeError:
                pass
        
        return strings
    
    def detect_packing(self, info: BinaryInfo) -> bool:
        """Detect if binary is packed/compressed"""
        packing_indicators = []
        
        # High entropy sections
        if info.sections:
            executable_sections = [s for s in info.sections if 'executable' in str(s.get('characteristics', ''))]
            if len(executable_sections) < 3:
                packing_indicators.append("few_sections")
        
        # Suspicious section names
        suspicious_names = ['.upx', '.aspack', '.themida', '.vmp', '.enigma']
        for section in info.sections:
            section_name = section.get('name', '').lower()
            if any(sus in section_name for sus in suspicious_names):
                packing_indicators.append("suspicious_section_names")
                break
        
        # Few imports (common in packed files)
        if len(info.imports) < 10:
            packing_indicators.append("few_imports")
        
        # Common packer function imports
        packer_imports = ['VirtualAlloc', 'VirtualProtect', 'LoadLibrary', 'GetProcAddress']
        import_names = [imp.split(':')[-1] for imp in info.imports]
        if any(packer_imp in import_names for packer_imp in packer_imports):
            packing_indicators.append("packer_imports")
        
        return len(packing_indicators) >= 2  # Require multiple indicators
    
    def _detect_pe_compiler(self, pe) -> Optional[str]:
        """Detect compiler used to create PE file"""
        try:
            # Check for common compiler signatures in sections
            for section in pe.sections:
                section_name = section.Name.decode('utf-8').rstrip('\x00')
                if section_name == '.rdata':
                    # Check for compiler strings in .rdata section
                    section_data = section.get_data()
                    if b'Microsoft' in section_data:
                        return "Microsoft Visual C++"
                    elif b'GCC' in section_data:
                        return "GCC"
                    elif b'Clang' in section_data:
                        return "Clang"
            
            # Check import table for compiler hints
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8').lower()
                    if 'msvcr' in dll_name or 'vcruntime' in dll_name:
                        return "Microsoft Visual C++"
                    elif 'mingw' in dll_name:
                        return "MinGW GCC"
            
        except Exception:
            pass
        
        return None


# Utility functions
def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get basic file information"""
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    
    stat = file_path.stat()
    return {
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'created': stat.st_ctime,
        'extension': file_path.suffix.lower(),
        'name': file_path.name
    }


def is_executable_file(file_path: Union[str, Path]) -> bool:
    """Check if file appears to be an executable"""
    file_path = Path(file_path)
    
    # Check file extension
    executable_extensions = {'.exe', '.dll', '.sys', '.scr', '.com', '.bat', '.cmd'}
    if file_path.suffix.lower() in executable_extensions:
        return True
    
    # Check file permissions (Unix-like systems)
    if hasattr(os, 'access') and os.access(file_path, os.X_OK):
        return True
    
    # Check magic bytes
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
            # PE signature
            if header.startswith(b'MZ'):
                return True
            # ELF signature
            if header.startswith(b'\x7fELF'):
                return True
            # Mach-O signatures
            if header.startswith((b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf',
                               b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe')):
                return True
    except Exception:
        pass
    
    return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def extract_metadata_summary(info: BinaryInfo) -> Dict[str, Any]:
    """Extract summary metadata from BinaryInfo"""
    return {
        'file_path': info.file_path,
        'file_size': info.file_size,
        'file_size_formatted': format_file_size(info.file_size),
        'format': info.format.value,
        'architecture': info.architecture.value,
        'is_64bit': info.is_64bit,
        'is_packed': info.is_packed,
        'entry_point': hex(info.entry_point) if info.entry_point else None,
        'base_address': hex(info.base_address) if info.base_address else None,
        'section_count': len(info.sections),
        'import_count': len(info.imports),
        'export_count': len(info.exports),
        'string_count': len(info.strings),
        'compiler': info.compiler_info,
        'hashes': info.hashes
    }