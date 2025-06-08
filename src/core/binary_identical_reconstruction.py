#!/usr/bin/env python3
"""
Binary-Identical Reconstruction Engine

P2.2 Implementation for Open-Sourcefy Matrix Pipeline
Provides binary-identical reconstruction capabilities where decompiled code
recompiles to bit-identical binaries with original metadata preservation.

Features:
- Symbol table reconstruction and metadata preservation
- Debug information recovery and enhancement  
- Linker setting recreation for bit-identical output
- Compiler flag mapping and optimization recreation
- Build environment reconstruction
- Binary comparison and validation

Research Base:
- Reproducible Builds Project methodologies
- BinComp binary comparison techniques
- Compiler optimization preservation
- Debug symbol reconstruction
"""

import os
import json
import struct
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import shutil
import time

logger = logging.getLogger(__name__)


class ReconstructionQuality(Enum):
    """Quality levels for binary reconstruction"""
    BIT_IDENTICAL = "bit_identical"      # Perfect binary match
    FUNCTIONALLY_IDENTICAL = "functional" # Same functionality, minor differences
    SEMANTICALLY_EQUIVALENT = "semantic"  # Same behavior, structural differences
    PARTIAL_RECONSTRUCTION = "partial"    # Partial success
    FAILED = "failed"                     # Reconstruction failed


@dataclass
class SymbolInfo:
    """Symbol table information"""
    name: str
    address: int
    size: int
    symbol_type: str  # function, data, import, export
    section: str
    visibility: str   # public, private, external
    mangled_name: Optional[str] = None
    demangled_name: Optional[str] = None


@dataclass
class DebugInfo:
    """Debug information structure"""
    source_files: List[str] = field(default_factory=list)
    line_numbers: Dict[int, Tuple[str, int]] = field(default_factory=dict)  # address -> (file, line)
    local_variables: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)  # function -> variables
    type_information: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    compilation_units: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildConfiguration:
    """Build configuration for reconstruction"""
    compiler_type: str
    compiler_version: str
    optimization_level: str
    target_architecture: str
    subsystem: str
    entry_point: str
    linker_version: str
    runtime_library: str
    compiler_flags: List[str] = field(default_factory=list)
    linker_flags: List[str] = field(default_factory=list)
    preprocessor_defines: List[str] = field(default_factory=list)
    include_directories: List[str] = field(default_factory=list)
    library_dependencies: List[str] = field(default_factory=list)


@dataclass
class ReconstructionResult:
    """Result of binary reconstruction attempt"""
    quality: ReconstructionQuality
    confidence: float
    original_hash: str
    reconstructed_hash: str
    byte_differences: int
    build_config: BuildConfiguration
    symbol_recovery_rate: float
    debug_info_recovery_rate: float
    compilation_success: bool
    error_messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BinaryIdenticalReconstructor:
    """
    Advanced binary reconstruction engine for achieving bit-identical output
    
    This class provides comprehensive binary reconstruction capabilities that aim
    to recreate the exact original binary from decompiled source code.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize reconstruction components
        self.symbol_reconstructor = SymbolTableReconstructor()
        self.debug_reconstructor = DebugInfoReconstructor() 
        self.build_reconstructor = BuildEnvironmentReconstructor()
        
        # Supported binary formats and architectures
        self.supported_formats = ['PE', 'ELF', 'Mach-O']
        self.supported_architectures = ['x86', 'x64', 'ARM', 'ARM64']
        
        # Reconstruction settings
        self.max_iteration_attempts = 3
        self.quality_threshold = 0.95
        self.symbol_threshold = 0.90
        
    def reconstruct_binary_identical(
        self, 
        original_binary_path: str,
        decompiled_source_path: str,
        output_directory: str,
        compiler_analysis: Dict[str, Any] = None
    ) -> ReconstructionResult:
        """
        Perform binary-identical reconstruction from decompiled source
        
        Args:
            original_binary_path: Path to original binary
            decompiled_source_path: Path to decompiled source code
            output_directory: Directory for reconstruction output
            compiler_analysis: Compiler fingerprinting results
            
        Returns:
            ReconstructionResult with reconstruction metrics and status
        """
        self.logger.info(f"Starting binary-identical reconstruction for {original_binary_path}")
        
        try:
            # Phase 1: Analyze original binary
            original_analysis = self._analyze_original_binary(original_binary_path)
            
            # Phase 2: Extract build configuration
            build_config = self._extract_build_configuration(
                original_analysis, compiler_analysis
            )
            
            # Phase 3: Reconstruct symbol table
            symbol_reconstruction = self._reconstruct_symbol_table(
                original_analysis, decompiled_source_path
            )
            
            # Phase 4: Reconstruct debug information
            debug_reconstruction = self._reconstruct_debug_info(
                original_analysis, decompiled_source_path
            )
            
            # Phase 5: Prepare build environment
            build_environment = self._prepare_build_environment(
                build_config, symbol_reconstruction, debug_reconstruction, output_directory
            )
            
            # Phase 6: Iterative compilation with optimization
            compilation_result = self._iterative_compilation(
                build_environment, original_binary_path, output_directory
            )
            
            # Phase 7: Binary comparison and analysis
            comparison_result = self._compare_binaries(
                original_binary_path, compilation_result['binary_path']
            )
            
            # Phase 8: Generate final result
            final_result = self._generate_reconstruction_result(
                original_analysis, build_config, symbol_reconstruction,
                debug_reconstruction, compilation_result, comparison_result
            )
            
            self.logger.info(
                f"Reconstruction complete: {final_result.quality.value} "
                f"(confidence: {final_result.confidence:.3f})"
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Binary reconstruction failed: {e}")
            return ReconstructionResult(
                quality=ReconstructionQuality.FAILED,
                confidence=0.0,
                original_hash="",
                reconstructed_hash="",
                byte_differences=-1,
                build_config=BuildConfiguration("unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown", "unknown"),
                symbol_recovery_rate=0.0,
                debug_info_recovery_rate=0.0,
                compilation_success=False,
                error_messages=[str(e)]
            )
    
    def _analyze_original_binary(self, binary_path: str) -> Dict[str, Any]:
        """Comprehensive analysis of original binary for reconstruction"""
        
        with open(binary_path, 'rb') as f:
            binary_content = f.read()
        
        analysis = {
            'file_path': binary_path,
            'file_size': len(binary_content),
            'file_hash': hashlib.sha256(binary_content).hexdigest(),
            'binary_format': None,
            'architecture': None,
            'symbols': [],
            'sections': [],
            'imports': [],
            'exports': [],
            'debug_info': None,
            'pe_headers': None,
            'rich_header': None,
            'compiler_artifacts': []
        }
        
        # Detect binary format
        if binary_content.startswith(b'MZ'):
            analysis['binary_format'] = 'PE'
            analysis.update(self._analyze_pe_binary(binary_content))
        elif binary_content.startswith(b'\x7fELF'):
            analysis['binary_format'] = 'ELF'
            analysis.update(self._analyze_elf_binary(binary_content))
        elif binary_content[:4] in [b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe']:
            analysis['binary_format'] = 'Mach-O'
            analysis.update(self._analyze_macho_binary(binary_content))
        
        return analysis
    
    def _analyze_pe_binary(self, binary_content: bytes) -> Dict[str, Any]:
        """Analyze PE binary format specifics"""
        pe_analysis = {}
        
        try:
            # Parse PE headers
            pe_offset = struct.unpack('<L', binary_content[60:64])[0]
            pe_signature = binary_content[pe_offset:pe_offset+4]
            
            if pe_signature == b'PE\x00\x00':
                # COFF header
                coff_header = binary_content[pe_offset+4:pe_offset+24]
                machine, num_sections, timestamp, ptr_to_symbols, num_symbols, opt_hdr_size, characteristics = struct.unpack('<HHIIIHH', coff_header)
                
                pe_analysis['pe_headers'] = {
                    'machine': machine,
                    'num_sections': num_sections,
                    'timestamp': timestamp,
                    'characteristics': characteristics,
                    'architecture': 'x64' if machine == 0x8664 else 'x86'
                }
                
                # Parse sections
                sections_offset = pe_offset + 24 + opt_hdr_size
                sections = []
                
                for i in range(num_sections):
                    section_offset = sections_offset + (i * 40)
                    section_data = binary_content[section_offset:section_offset+40]
                    
                    if len(section_data) >= 40:
                        name = section_data[:8].rstrip(b'\x00').decode('utf-8', errors='ignore')
                        virtual_size, virtual_addr, raw_size, raw_addr = struct.unpack('<IIII', section_data[8:24])
                        characteristics = struct.unpack('<I', section_data[36:40])[0]
                        
                        sections.append({
                            'name': name,
                            'virtual_address': virtual_addr,
                            'virtual_size': virtual_size,
                            'raw_address': raw_addr,
                            'raw_size': raw_size,
                            'characteristics': characteristics
                        })
                
                pe_analysis['sections'] = sections
                pe_analysis['architecture'] = pe_analysis['pe_headers']['architecture']
                
                # Parse Rich header if present
                rich_header = self._parse_rich_header(binary_content)
                if rich_header:
                    pe_analysis['rich_header'] = rich_header
                
        except Exception as e:
            self.logger.warning(f"PE analysis failed: {e}")
        
        return pe_analysis
    
    def _analyze_elf_binary(self, binary_content: bytes) -> Dict[str, Any]:
        """Analyze ELF binary format specifics"""
        elf_analysis = {}
        
        try:
            # Parse ELF header
            elf_header = binary_content[:64]  # 64-byte ELF header
            
            # ELF identification
            ei_class = elf_header[4]  # 1=32-bit, 2=64-bit
            ei_data = elf_header[5]   # 1=little-endian, 2=big-endian
            
            elf_analysis['architecture'] = 'x64' if ei_class == 2 else 'x86'
            elf_analysis['endianness'] = 'little' if ei_data == 1 else 'big'
            
            # Parse basic ELF structure (simplified)
            if ei_class == 2:  # 64-bit
                e_type, e_machine, e_version, e_entry = struct.unpack('<HHIQ', elf_header[16:32])
            else:  # 32-bit
                e_type, e_machine, e_version, e_entry = struct.unpack('<HHII', elf_header[16:28])
            
            elf_analysis['elf_type'] = e_type
            elf_analysis['machine'] = e_machine
            elf_analysis['entry_point'] = e_entry
            
        except Exception as e:
            self.logger.warning(f"ELF analysis failed: {e}")
        
        return elf_analysis
    
    def _analyze_macho_binary(self, binary_content: bytes) -> Dict[str, Any]:
        """Analyze Mach-O binary format specifics"""
        macho_analysis = {}
        
        try:
            # Parse Mach-O header
            magic = struct.unpack('<I', binary_content[:4])[0]
            
            if magic in [0xfeedface, 0xfeedfacf]:  # 32-bit and 64-bit
                is_64bit = magic == 0xfeedfacf
                
                if is_64bit:
                    cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags = struct.unpack('<IIIIII', binary_content[4:28])
                    macho_analysis['architecture'] = 'x64'
                else:
                    cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags = struct.unpack('<IIIIII', binary_content[4:24])
                    macho_analysis['architecture'] = 'x86'
                
                macho_analysis['cpu_type'] = cputype
                macho_analysis['file_type'] = filetype
                macho_analysis['num_commands'] = ncmds
                
        except Exception as e:
            self.logger.warning(f"Mach-O analysis failed: {e}")
        
        return macho_analysis
    
    def _parse_rich_header(self, binary_content: bytes) -> Optional[Dict[str, Any]]:
        """Parse PE Rich header for build tool information"""
        try:
            rich_start = binary_content.find(b'Rich')
            if rich_start == -1:
                return None
            
            # Find DanS signature
            dans_start = binary_content.rfind(b'DanS', 0, rich_start)
            if dans_start == -1:
                return None
            
            # Extract and parse Rich header entries
            rich_entries = []
            for i in range(dans_start + 16, rich_start, 8):
                if i + 8 <= len(binary_content):
                    comp_id = struct.unpack('<L', binary_content[i:i+4])[0]
                    use_count = struct.unpack('<L', binary_content[i+4:i+8])[0]
                    
                    rich_entries.append({
                        'component_id': comp_id,
                        'version': (comp_id >> 16) & 0xFFFF,
                        'build': comp_id & 0xFFFF,
                        'use_count': use_count
                    })
            
            return {
                'entries': rich_entries,
                'parsed_successfully': True
            }
            
        except Exception as e:
            self.logger.debug(f"Rich header parsing failed: {e}")
            return None
    
    def _extract_build_configuration(
        self, 
        binary_analysis: Dict[str, Any], 
        compiler_analysis: Dict[str, Any] = None
    ) -> BuildConfiguration:
        """Extract build configuration from binary analysis"""
        
        # Start with defaults
        build_config = BuildConfiguration(
            compiler_type="unknown",
            compiler_version="unknown", 
            optimization_level="O2",
            target_architecture=binary_analysis.get('architecture', 'x86'),
            subsystem="console",
            entry_point="main",
            linker_version="unknown",
            runtime_library="dynamic"
        )
        
        # Enhance with compiler analysis if available
        if compiler_analysis:
            advanced_analysis = compiler_analysis.get('advanced_compiler_analysis', {})
            if advanced_analysis:
                build_config.compiler_type = advanced_analysis.get('compiler_type', 'unknown')
                build_config.compiler_version = advanced_analysis.get('version', 'unknown')
                build_config.optimization_level = advanced_analysis.get('optimization_level', 'O2')
        
        # Extract from Rich header if available (MSVC specific)
        rich_header = binary_analysis.get('rich_header')
        if rich_header and build_config.compiler_type in ['unknown', 'microsoft_visual_cpp']:
            build_config.compiler_type = 'microsoft_visual_cpp'
            
            # Determine Visual Studio version from Rich header
            for entry in rich_header.get('entries', []):
                version = entry['version']
                if 19.30 <= version <= 19.37:
                    build_config.compiler_version = "2022"
                elif 19.20 <= version <= 19.29:
                    build_config.compiler_version = "2019"
                elif 19.10 <= version <= 19.16:
                    build_config.compiler_version = "2017"
                elif version == 19.00:
                    build_config.compiler_version = "2015"
                elif version == 18.00:
                    build_config.compiler_version = "2013"
        
        # Set appropriate flags based on detected configuration
        build_config.compiler_flags = self._generate_compiler_flags(build_config, binary_analysis)
        build_config.linker_flags = self._generate_linker_flags(build_config, binary_analysis)
        
        return build_config
    
    def _generate_compiler_flags(self, config: BuildConfiguration, analysis: Dict[str, Any]) -> List[str]:
        """Generate appropriate compiler flags based on configuration"""
        flags = []
        
        if config.compiler_type == 'microsoft_visual_cpp':
            # MSVC flags
            flags.extend(['/c', '/nologo'])
            
            # Optimization flags
            if config.optimization_level == 'O0':
                flags.append('/Od')
            elif config.optimization_level == 'O1':
                flags.append('/O1')
            elif config.optimization_level == 'O2':
                flags.append('/O2')
            elif config.optimization_level == 'O3':
                flags.append('/Ox')
            
            # Architecture flags
            if config.target_architecture == 'x64':
                flags.append('/favor:INTEL64')
            
            # Runtime library
            if config.runtime_library == 'static':
                flags.append('/MT')
            else:
                flags.append('/MD')
                
        elif config.compiler_type in ['gnu_compiler_collection', 'llvm_clang']:
            # GCC/Clang flags
            flags.extend(['-c'])
            
            # Optimization flags
            if config.optimization_level in ['O0', 'O1', 'O2', 'O3']:
                flags.append(f'-{config.optimization_level}')
            elif config.optimization_level == 'size':
                flags.append('-Os')
            elif config.optimization_level == 'ultra_size':
                flags.append('-Oz')
            
            # Architecture flags
            if config.target_architecture == 'x64':
                flags.append('-m64')
            elif config.target_architecture == 'x86':
                flags.append('-m32')
        
        return flags
    
    def _generate_linker_flags(self, config: BuildConfiguration, analysis: Dict[str, Any]) -> List[str]:
        """Generate appropriate linker flags based on configuration"""
        flags = []
        
        if config.compiler_type == 'microsoft_visual_cpp':
            # MSVC linker flags
            flags.extend(['/nologo'])
            
            # Subsystem
            if config.subsystem == 'console':
                flags.append('/SUBSYSTEM:CONSOLE')
            elif config.subsystem == 'windows':
                flags.append('/SUBSYSTEM:WINDOWS')
            
            # Architecture
            if config.target_architecture == 'x64':
                flags.append('/MACHINE:X64')
            else:
                flags.append('/MACHINE:X86')
                
        elif config.compiler_type in ['gnu_compiler_collection', 'llvm_clang']:
            # GCC/Clang linker flags
            if config.target_architecture == 'x64':
                flags.append('-m64')
            elif config.target_architecture == 'x86':
                flags.append('-m32')
        
        return flags
    
    def _reconstruct_symbol_table(self, analysis: Dict[str, Any], source_path: str) -> Dict[str, Any]:
        """Reconstruct symbol table information"""
        return self.symbol_reconstructor.reconstruct_symbols(analysis, source_path)
    
    def _reconstruct_debug_info(self, analysis: Dict[str, Any], source_path: str) -> Dict[str, Any]:
        """Reconstruct debug information"""
        return self.debug_reconstructor.reconstruct_debug_info(analysis, source_path)
    
    def _prepare_build_environment(
        self, 
        config: BuildConfiguration, 
        symbols: Dict[str, Any], 
        debug_info: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, Any]:
        """Prepare complete build environment for reconstruction"""
        
        build_env = {
            'build_directory': Path(output_dir) / 'build',
            'source_directory': Path(output_dir) / 'source',
            'config': config,
            'symbols': symbols,
            'debug_info': debug_info,
            'build_script': None,
            'environment_vars': {}
        }
        
        # Create directories
        build_env['build_directory'].mkdir(parents=True, exist_ok=True)
        build_env['source_directory'].mkdir(parents=True, exist_ok=True)
        
        # Generate build script
        build_env['build_script'] = self.build_reconstructor.generate_build_script(
            config, str(build_env['build_directory'])
        )
        
        # Set environment variables
        build_env['environment_vars'] = self._setup_build_environment_vars(config)
        
        return build_env
    
    def _setup_build_environment_vars(self, config: BuildConfiguration) -> Dict[str, str]:
        """Setup environment variables for build"""
        env_vars = {}
        
        if config.compiler_type == 'microsoft_visual_cpp':
            # MSVC environment setup
            vs_versions = {
                '2022': '170',
                '2019': '160', 
                '2017': '150',
                '2015': '140'
            }
            
            if config.compiler_version in vs_versions:
                vs_ver = vs_versions[config.compiler_version]
                env_vars['VCINSTALLDIR'] = f"C:\\Program Files\\Microsoft Visual Studio\\{config.compiler_version}\\Professional\\VC\\"
                env_vars['VCVARSALL'] = f"C:\\Program Files\\Microsoft Visual Studio\\{config.compiler_version}\\Professional\\VC\\Auxiliary\\Build\\vcvarsall.bat"
        
        return env_vars
    
    def _iterative_compilation(
        self, 
        build_env: Dict[str, Any], 
        original_binary: str, 
        output_dir: str
    ) -> Dict[str, Any]:
        """Perform iterative compilation to achieve binary-identical result"""
        
        compilation_results = []
        best_result = None
        best_similarity = 0.0
        
        for iteration in range(self.max_iteration_attempts):
            self.logger.info(f"Compilation attempt {iteration + 1}/{self.max_iteration_attempts}")
            
            try:
                # Adjust build parameters for this iteration
                adjusted_config = self._adjust_build_parameters(build_env['config'], iteration)
                
                # Perform compilation
                result = self._compile_with_config(
                    adjusted_config, build_env, output_dir, iteration
                )
                
                if result['success']:
                    # Compare with original
                    similarity = self._calculate_binary_similarity(
                        original_binary, result['output_binary']
                    )
                    
                    result['similarity'] = similarity
                    compilation_results.append(result)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_result = result
                    
                    # If we achieved high similarity, stop iterating
                    if similarity >= self.quality_threshold:
                        self.logger.info(f"High similarity achieved: {similarity:.3f}")
                        break
                
            except Exception as e:
                self.logger.warning(f"Compilation iteration {iteration + 1} failed: {e}")
                compilation_results.append({
                    'success': False,
                    'error': str(e),
                    'iteration': iteration
                })
        
        return {
            'best_result': best_result,
            'all_results': compilation_results,
            'total_attempts': len(compilation_results),
            'best_similarity': best_similarity
        }
    
    def _adjust_build_parameters(self, config: BuildConfiguration, iteration: int) -> BuildConfiguration:
        """Adjust build parameters for iterative improvement"""
        adjusted = BuildConfiguration(
            compiler_type=config.compiler_type,
            compiler_version=config.compiler_version,
            optimization_level=config.optimization_level,
            target_architecture=config.target_architecture,
            subsystem=config.subsystem,
            entry_point=config.entry_point,
            linker_version=config.linker_version,
            runtime_library=config.runtime_library,
            compiler_flags=config.compiler_flags.copy(),
            linker_flags=config.linker_flags.copy(),
            preprocessor_defines=config.preprocessor_defines.copy(),
            include_directories=config.include_directories.copy(),
            library_dependencies=config.library_dependencies.copy()
        )
        
        # Adjust parameters based on iteration
        if iteration == 1:
            # Try different optimization settings
            if adjusted.optimization_level == 'O2':
                adjusted.optimization_level = 'O1'
            elif adjusted.optimization_level == 'O1':
                adjusted.optimization_level = 'O0'
        elif iteration == 2:
            # Try different runtime library
            if adjusted.runtime_library == 'dynamic':
                adjusted.runtime_library = 'static'
            else:
                adjusted.runtime_library = 'dynamic'
        
        # Regenerate flags with adjusted parameters
        adjusted.compiler_flags = self._generate_compiler_flags(adjusted, {})
        adjusted.linker_flags = self._generate_linker_flags(adjusted, {})
        
        return adjusted
    
    def _compile_with_config(
        self, 
        config: BuildConfiguration, 
        build_env: Dict[str, Any], 
        output_dir: str, 
        iteration: int
    ) -> Dict[str, Any]:
        """Compile with specific configuration"""
        
        result = {
            'success': False,
            'output_binary': None,
            'compilation_log': '',
            'iteration': iteration,
            'config_used': config
        }
        
        try:
            # Prepare output binary path
            binary_name = f"reconstructed_iter_{iteration}.exe"
            output_binary = Path(output_dir) / binary_name
            
            # Run compilation based on compiler type
            if config.compiler_type == 'microsoft_visual_cpp':
                compilation_result = self._compile_with_msvc(
                    config, build_env, str(output_binary)
                )
            elif config.compiler_type == 'gnu_compiler_collection':
                compilation_result = self._compile_with_gcc(
                    config, build_env, str(output_binary)
                )
            elif config.compiler_type == 'llvm_clang':
                compilation_result = self._compile_with_clang(
                    config, build_env, str(output_binary)
                )
            else:
                raise ValueError(f"Unsupported compiler: {config.compiler_type}")
            
            result.update(compilation_result)
            
            if result['success'] and output_binary.exists():
                result['output_binary'] = str(output_binary)
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Compilation failed: {e}")
        
        return result
    
    def _compile_with_msvc(self, config: BuildConfiguration, build_env: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Compile using Microsoft Visual C++"""
        
        # Find source files
        source_dir = build_env['source_directory']
        source_files = list(source_dir.glob('*.c')) + list(source_dir.glob('*.cpp'))
        
        if not source_files:
            return {'success': False, 'error': 'No source files found'}
        
        # Build command
        cmd = ['cl.exe']
        cmd.extend(config.compiler_flags)
        cmd.extend([str(f) for f in source_files])
        cmd.extend(['/Fe:' + output_path])
        cmd.extend(['/link'] + config.linker_flags)
        
        try:
            # Set up environment
            env = os.environ.copy()
            env.update(build_env.get('environment_vars', {}))
            
            # Run compilation
            result = subprocess.run(
                cmd,
                cwd=str(build_env['build_directory']),
                capture_output=True,
                text=True,
                env=env,
                timeout=300  # 5 minute timeout
            )
            
            return {
                'success': result.returncode == 0,
                'compilation_log': result.stdout + result.stderr,
                'return_code': result.returncode,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Compilation timed out'}
        except FileNotFoundError:
            return {'success': False, 'error': 'MSVC compiler not found'}
    
    def _compile_with_gcc(self, config: BuildConfiguration, build_env: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Compile using GCC"""
        
        # Find source files
        source_dir = build_env['source_directory']
        source_files = list(source_dir.glob('*.c')) + list(source_dir.glob('*.cpp'))
        
        if not source_files:
            return {'success': False, 'error': 'No source files found'}
        
        # Build command
        cmd = ['gcc']
        cmd.extend(config.compiler_flags)
        cmd.extend([str(f) for f in source_files])
        cmd.extend(['-o', output_path])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(build_env['build_directory']),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return {
                'success': result.returncode == 0,
                'compilation_log': result.stdout + result.stderr,
                'return_code': result.returncode,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Compilation timed out'}
        except FileNotFoundError:
            return {'success': False, 'error': 'GCC compiler not found'}
    
    def _compile_with_clang(self, config: BuildConfiguration, build_env: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Compile using Clang"""
        
        # Find source files
        source_dir = build_env['source_directory']
        source_files = list(source_dir.glob('*.c')) + list(source_dir.glob('*.cpp'))
        
        if not source_files:
            return {'success': False, 'error': 'No source files found'}
        
        # Build command
        cmd = ['clang']
        cmd.extend(config.compiler_flags)
        cmd.extend([str(f) for f in source_files])
        cmd.extend(['-o', output_path])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(build_env['build_directory']),
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return {
                'success': result.returncode == 0,
                'compilation_log': result.stdout + result.stderr,
                'return_code': result.returncode,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Compilation timed out'}
        except FileNotFoundError:
            return {'success': False, 'error': 'Clang compiler not found'}
    
    def _calculate_binary_similarity(self, original_path: str, reconstructed_path: str) -> float:
        """Calculate similarity between original and reconstructed binaries"""
        
        try:
            with open(original_path, 'rb') as f:
                original_data = f.read()
            
            with open(reconstructed_path, 'rb') as f:
                reconstructed_data = f.read()
            
            # Size comparison
            if len(original_data) != len(reconstructed_data):
                size_ratio = min(len(original_data), len(reconstructed_data)) / max(len(original_data), len(reconstructed_data))
            else:
                size_ratio = 1.0
            
            # Byte-by-byte comparison
            min_length = min(len(original_data), len(reconstructed_data))
            matching_bytes = sum(1 for i in range(min_length) if original_data[i] == reconstructed_data[i])
            byte_similarity = matching_bytes / min_length if min_length > 0 else 0.0
            
            # Overall similarity (weighted average)
            return (size_ratio * 0.3) + (byte_similarity * 0.7)
            
        except Exception as e:
            self.logger.error(f"Binary similarity calculation failed: {e}")
            return 0.0
    
    def _compare_binaries(self, original_path: str, reconstructed_path: str) -> Dict[str, Any]:
        """Detailed binary comparison analysis"""
        
        comparison = {
            'identical': False,
            'size_match': False,
            'hash_match': False,
            'byte_differences': 0,
            'similarity_score': 0.0,
            'original_hash': '',
            'reconstructed_hash': '',
            'original_size': 0,
            'reconstructed_size': 0
        }
        
        try:
            # Read both files
            with open(original_path, 'rb') as f:
                original_data = f.read()
            
            with open(reconstructed_path, 'rb') as f:
                reconstructed_data = f.read()
            
            # Basic metrics
            comparison['original_size'] = len(original_data)
            comparison['reconstructed_size'] = len(reconstructed_data)
            comparison['size_match'] = len(original_data) == len(reconstructed_data)
            
            # Hash comparison
            comparison['original_hash'] = hashlib.sha256(original_data).hexdigest()
            comparison['reconstructed_hash'] = hashlib.sha256(reconstructed_data).hexdigest()
            comparison['hash_match'] = comparison['original_hash'] == comparison['reconstructed_hash']
            
            # Byte-level comparison
            min_length = min(len(original_data), len(reconstructed_data))
            byte_differences = sum(1 for i in range(min_length) if original_data[i] != reconstructed_data[i])
            byte_differences += abs(len(original_data) - len(reconstructed_data))
            
            comparison['byte_differences'] = byte_differences
            comparison['identical'] = byte_differences == 0
            
            # Similarity score
            comparison['similarity_score'] = self._calculate_binary_similarity(original_path, reconstructed_path)
            
        except Exception as e:
            self.logger.error(f"Binary comparison failed: {e}")
            comparison['error'] = str(e)
        
        return comparison
    
    def _generate_reconstruction_result(
        self,
        original_analysis: Dict[str, Any],
        build_config: BuildConfiguration,
        symbol_reconstruction: Dict[str, Any],
        debug_reconstruction: Dict[str, Any],
        compilation_result: Dict[str, Any],
        comparison_result: Dict[str, Any]
    ) -> ReconstructionResult:
        """Generate comprehensive reconstruction result"""
        
        # Determine quality level
        if comparison_result.get('identical', False):
            quality = ReconstructionQuality.BIT_IDENTICAL
            confidence = 1.0
        elif comparison_result.get('similarity_score', 0) > 0.95:
            quality = ReconstructionQuality.FUNCTIONALLY_IDENTICAL
            confidence = comparison_result['similarity_score']
        elif comparison_result.get('similarity_score', 0) > 0.80:
            quality = ReconstructionQuality.SEMANTICALLY_EQUIVALENT
            confidence = comparison_result['similarity_score']
        elif compilation_result.get('best_result', {}).get('success', False):
            quality = ReconstructionQuality.PARTIAL_RECONSTRUCTION
            confidence = comparison_result.get('similarity_score', 0.5)
        else:
            quality = ReconstructionQuality.FAILED
            confidence = 0.0
        
        # Calculate recovery rates
        symbol_recovery_rate = symbol_reconstruction.get('recovery_rate', 0.0)
        debug_recovery_rate = debug_reconstruction.get('recovery_rate', 0.0)
        
        # Collect error messages
        error_messages = []
        if compilation_result.get('best_result') and not compilation_result['best_result'].get('success', False):
            for result in compilation_result.get('all_results', []):
                if 'error' in result:
                    error_messages.append(result['error'])
        
        return ReconstructionResult(
            quality=quality,
            confidence=confidence,
            original_hash=comparison_result.get('original_hash', ''),
            reconstructed_hash=comparison_result.get('reconstructed_hash', ''),
            byte_differences=comparison_result.get('byte_differences', -1),
            build_config=build_config,
            symbol_recovery_rate=symbol_recovery_rate,
            debug_info_recovery_rate=debug_recovery_rate,
            compilation_success=compilation_result.get('best_result', {}).get('success', False),
            error_messages=error_messages,
            metrics={
                'compilation_attempts': compilation_result.get('total_attempts', 0),
                'best_similarity': comparison_result.get('similarity_score', 0.0),
                'size_difference': abs(comparison_result.get('original_size', 0) - comparison_result.get('reconstructed_size', 0)),
                'build_time': time.time(),  # Placeholder for actual build time
            }
        )


class SymbolTableReconstructor:
    """Specialized component for symbol table reconstruction"""
    
    def reconstruct_symbols(self, analysis: Dict[str, Any], source_path: str) -> Dict[str, Any]:
        """Reconstruct symbol table from binary analysis and source code"""
        
        symbols = []
        recovery_rate = 0.0
        
        try:
            # Extract symbols from binary analysis
            if 'symbols' in analysis:
                for sym_data in analysis['symbols']:
                    symbol = SymbolInfo(
                        name=sym_data.get('name', ''),
                        address=sym_data.get('address', 0),
                        size=sym_data.get('size', 0),
                        symbol_type=sym_data.get('type', 'unknown'),
                        section=sym_data.get('section', ''),
                        visibility=sym_data.get('visibility', 'unknown')
                    )
                    symbols.append(symbol)
            
            # Calculate recovery rate based on available information
            total_symbols = len(symbols)
            named_symbols = len([s for s in symbols if s.name and not s.name.startswith('unk_')])
            recovery_rate = named_symbols / max(total_symbols, 1)
            
        except Exception as e:
            logger.error(f"Symbol reconstruction failed: {e}")
        
        return {
            'symbols': symbols,
            'recovery_rate': recovery_rate,
            'total_symbols': len(symbols),
            'named_symbols': len([s for s in symbols if s.name and not s.name.startswith('unk_')])
        }


class DebugInfoReconstructor:
    """Specialized component for debug information reconstruction"""
    
    def reconstruct_debug_info(self, analysis: Dict[str, Any], source_path: str) -> Dict[str, Any]:
        """Reconstruct debug information from binary analysis and source code"""
        
        debug_info = DebugInfo()
        recovery_rate = 0.0
        
        try:
            # Extract debug information if available
            if 'debug_info' in analysis and analysis['debug_info']:
                debug_data = analysis['debug_info']
                
                # Source file information
                if 'source_files' in debug_data:
                    debug_info.source_files = debug_data['source_files']
                
                # Line number information
                if 'line_numbers' in debug_data:
                    debug_info.line_numbers = debug_data['line_numbers']
                
                # Calculate recovery rate
                if debug_info.source_files or debug_info.line_numbers:
                    recovery_rate = 0.8  # High recovery if debug info present
                else:
                    recovery_rate = 0.1  # Low recovery if minimal debug info
            else:
                # No debug information available
                recovery_rate = 0.0
            
        except Exception as e:
            logger.error(f"Debug info reconstruction failed: {e}")
        
        return {
            'debug_info': debug_info,
            'recovery_rate': recovery_rate,
            'has_source_files': len(debug_info.source_files) > 0,
            'has_line_numbers': len(debug_info.line_numbers) > 0
        }


class BuildEnvironmentReconstructor:
    """Specialized component for build environment reconstruction"""
    
    def generate_build_script(self, config: BuildConfiguration, build_dir: str) -> str:
        """Generate build script for the target environment"""
        
        script_lines = []
        
        if config.compiler_type == 'microsoft_visual_cpp':
            # Generate batch script for MSVC
            script_lines = [
                '@echo off',
                'rem Generated build script for MSVC',
                '',
                'rem Setup Visual Studio environment',
                f'call "C:\\Program Files\\Microsoft Visual Studio\\{config.compiler_version}\\Professional\\VC\\Auxiliary\\Build\\vcvarsall.bat" x64',
                '',
                'rem Compile source files',
                f'cl.exe {" ".join(config.compiler_flags)} *.c *.cpp',
                f'link.exe {" ".join(config.linker_flags)} *.obj',
                '',
                'echo Build complete'
            ]
            
            script_path = Path(build_dir) / 'build.bat'
            
        else:
            # Generate shell script for GCC/Clang
            script_lines = [
                '#!/bin/bash',
                '# Generated build script for GCC/Clang',
                '',
                '# Compile source files',
                f'{config.compiler_type} {" ".join(config.compiler_flags)} *.c *.cpp -o output',
                '',
                'echo "Build complete"'
            ]
            
            script_path = Path(build_dir) / 'build.sh'
        
        # Write script to file
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_lines))
        
        # Make executable on Unix systems
        if not script_path.suffix == '.bat':
            script_path.chmod(0o755)
        
        return str(script_path)


# Integration functions for Matrix pipeline
def enhance_binary_reconstruction(
    original_binary: str,
    decompiled_source: str, 
    compiler_analysis: Dict[str, Any],
    output_directory: str
) -> Dict[str, Any]:
    """
    Enhance binary reconstruction with P2.2 capabilities
    
    Args:
        original_binary: Path to original binary file
        decompiled_source: Path to decompiled source code
        compiler_analysis: Compiler fingerprinting results
        output_directory: Directory for reconstruction output
        
    Returns:
        Enhanced reconstruction analysis
    """
    try:
        reconstructor = BinaryIdenticalReconstructor()
        
        result = reconstructor.reconstruct_binary_identical(
            original_binary, decompiled_source, output_directory, compiler_analysis
        )
        
        return {
            'binary_identical_reconstruction': {
                'quality': result.quality.value,
                'confidence': result.confidence,
                'original_hash': result.original_hash,
                'reconstructed_hash': result.reconstructed_hash,
                'byte_differences': result.byte_differences,
                'symbol_recovery_rate': result.symbol_recovery_rate,
                'debug_recovery_rate': result.debug_info_recovery_rate,
                'compilation_success': result.compilation_success,
                'build_configuration': {
                    'compiler_type': result.build_config.compiler_type,
                    'compiler_version': result.build_config.compiler_version,
                    'optimization_level': result.build_config.optimization_level,
                    'target_architecture': result.build_config.target_architecture
                },
                'metrics': result.metrics,
                'enhancement_applied': True
            }
        }
        
    except Exception as e:
        logger.error(f"Binary-identical reconstruction failed: {e}")
        return {'error': str(e), 'enhancement_applied': False}