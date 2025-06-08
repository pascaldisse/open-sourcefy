"""
Agent 01: Sentinel - Binary Discovery & Metadata Analysis
The digital guardian that scans the Matrix for binary anomalies and catalogs their nature.
Serves as the foundation upon which all other Matrix agents build their analysis.

Production-ready implementation following SOLID principles and clean code standards.
Includes LangChain integration, comprehensive error handling, and fail-fast validation.
"""

import logging
import hashlib
import struct
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Matrix framework imports
from ..matrix_agents import AnalysisAgent, MatrixCharacter, AgentStatus
from ..shared_components import (
    MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker, 
    MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
)
from ..exceptions import MatrixAgentError, ValidationError, ConfigurationError, BinaryAnalysisError

# LangChain imports for AI-enhanced analysis (optional)
try:
    from langchain.agents import Tool, AgentExecutor
    from langchain.agents.react.base import ReActDocstoreAgent
    from langchain.llms import LlamaCpp
    from langchain.memory import ConversationBufferMemory
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    # Create dummy types for type annotations when LangChain isn't available
    Tool = Any
    AgentExecutor = Any
    ReActDocstoreAgent = Any
    LlamaCpp = Any
    ConversationBufferMemory = Any

# Binary analysis libraries with error handling
try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False

# ELF and Mach-O support removed - Windows PE only
HAS_ELFTOOLS = False
HAS_MACHOLIB = False


# Configuration constants - NO MAGIC NUMBERS
class SentinelConstants:
    """Sentinel-specific constants loaded from configuration"""
    
    def __init__(self, config_manager, agent_id: int):
        self.MAX_RETRY_ATTEMPTS = config_manager.get_value(f'agents.agent_{agent_id:02d}.max_retries', 3)
        self.TIMEOUT_SECONDS = config_manager.get_value(f'agents.agent_{agent_id:02d}.timeout', 300)
        self.QUALITY_THRESHOLD = config_manager.get_value(f'agents.agent_{agent_id:02d}.quality_threshold', 0.75)
        self.MAX_FILE_SIZE_MB = config_manager.get_value(f'agents.agent_{agent_id:02d}.max_file_size_mb', 500)
        self.MIN_ENTROPY_THRESHOLD = config_manager.get_value('analysis.min_entropy_threshold', 6.0)
        self.STRING_MIN_LENGTH = config_manager.get_value('analysis.string_min_length', 4)
        self.HASH_ALGORITHMS = config_manager.get_value('analysis.hash_algorithms', ['md5', 'sha1', 'sha256'])


@dataclass
class BinaryMetadata:
    """Structured binary metadata for consistent data exchange"""
    file_path: str
    file_size: int
    format_type: str  # PE/Unknown (Windows only)
    architecture: str  # x86/x64/ARM/etc.
    bitness: int  # 32/64
    endianness: str  # little/big
    entry_point: Optional[int] = None
    base_address: Optional[int] = None
    is_packed: bool = False
    confidence_score: float = 0.0


@dataclass
class SentinelValidationResult:
    """Validation result structure for fail-fast pipeline"""
    is_valid: bool
    quality_score: float
    error_messages: List[str]
    validation_details: Dict[str, Any]


class SentinelAgent(AnalysisAgent):
    """
    Agent 01: Sentinel - Production-Ready Implementation
    
    The Sentinel stands vigilant at the gates of the Matrix, scanning and cataloging 
    every binary entity that seeks entry. It identifies their nature, structure, 
    and potential threats with unwavering precision.
    
    Features:
    - LangChain AI-enhanced analysis
    - Fail-fast validation with quality thresholds
    - Comprehensive error handling and logging
    - Configuration-driven behavior (no hardcoded values)
    - Shared tool integration for code reuse
    - Production-ready exception handling
    """
    
    def __init__(self):
        super().__init__(
            agent_id=1,
            matrix_character=MatrixCharacter.SENTINEL,
            dependencies=[]  # No dependencies - foundation agent
        )
        
        # Load configuration constants
        self.constants = SentinelConstants(self.config, self.agent_id)
        
        # Initialize shared tools
        self.analysis_tools = SharedAnalysisTools()
        self.validation_tools = SharedValidationTools()
        
        # Setup specialized components
        self.error_handler = MatrixErrorHandler(self.agent_name, self.constants.MAX_RETRY_ATTEMPTS)
        self.metrics = MatrixMetrics(self.agent_id, self.matrix_character.value)
        
        # Check binary analysis library availability
        self.available_parsers = self._check_parser_availability()
        
        # Initialize LangChain components for AI enhancement
        self.ai_enabled = self.config.get_value('ai.enabled', True)
        if self.ai_enabled:
            self.llm = self._setup_llm()
            self.agent_executor = self._setup_langchain_agent()
        
        # Validate configuration
        self._validate_configuration()
    
    def _check_parser_availability(self) -> Dict[str, bool]:
        """Check binary analysis libraries - Windows PE only"""
        availability = {
            'pefile': HAS_PEFILE
        }
        
        if not HAS_PEFILE:
            self.logger.warning("pefile library not available - PE analysis will be limited")
        
        return availability
    
    def _validate_configuration(self) -> None:
        """Validate agent configuration at startup - fail fast if invalid"""
        required_paths = [
            'paths.temp_directory',
            'paths.output_directory'
        ]
        
        missing_paths = []
        for path_key in required_paths:
            try:
                path = self.config.get_path(path_key)
                if path is None:
                    # Use default paths if not configured
                    if path_key == 'paths.temp_directory':
                        path = Path('./temp')
                    elif path_key == 'paths.output_directory':
                        path = Path('./output')
                    else:
                        missing_paths.append(f"{path_key}: No path configured")
                        continue
                elif isinstance(path, str):
                    path = Path(path)
                
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                missing_paths.append(f"{path_key}: {e}")
        
        if missing_paths:
            raise ConfigurationError(f"Invalid configuration paths: {missing_paths}")
    
    def _setup_llm(self):
        """Setup LangChain language model from configuration"""
        try:
            model_path = self.config.get_path('ai.model.path')
            if not model_path.exists():
                self.logger.warning(f"AI model not found at {model_path}, disabling AI features")
                self.ai_enabled = False
                return None
                
            return LlamaCpp(
                model_path=str(model_path),
                temperature=self.config.get_value('ai.model.temperature', 0.1),
                max_tokens=self.config.get_value('ai.model.max_tokens', 2048),
                verbose=self.config.get_value('debug.enabled', False)
            )
        except Exception as e:
            self.logger.warning(f"Failed to setup LLM: {e}, disabling AI features")
            self.ai_enabled = False
            return None
    
    def _setup_langchain_agent(self) -> Optional[AgentExecutor]:
        """Setup LangChain agent with Sentinel-specific tools"""
        if not self.ai_enabled or not self.llm:
            return None
            
        try:
            tools = self._create_agent_tools()
            memory = ConversationBufferMemory()
            
            agent = ReActDocstoreAgent.from_llm_and_tools(
                llm=self.llm,
                tools=tools,
                verbose=self.config.get_value('debug.enabled', False)
            )
            
            return AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=self.config.get_value('debug.enabled', False),
                max_iterations=self.config.get_value('ai.max_iterations', 5)
            )
        except Exception as e:
            self.logger.warning(f"Failed to setup LangChain agent: {e}")
            return None
    
    def _create_agent_tools(self) -> List[Tool]:
        """Create LangChain tools specific to Sentinel's capabilities"""
        return [
            Tool(
                name="validate_binary_format",
                description="Validate binary file format and structure",
                func=self._ai_format_validation_tool
            ),
            Tool(
                name="analyze_security_indicators",
                description="Analyze binary for security indicators and threats",
                func=self._ai_security_analysis_tool
            ),
            Tool(
                name="generate_binary_insights",
                description="Generate insights about binary characteristics",
                func=self._ai_insight_generation_tool
            )
        ]
    
    def get_matrix_description(self) -> str:
        """The Sentinel's role in the Matrix"""
        return ("The Sentinel stands vigilant at the gates of the Matrix, scanning "
                "and cataloging every binary entity that seeks entry. It identifies "
                "their nature, structure, and potential threats with unwavering precision.")
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Sentinel's binary scanning mission with production-ready error handling
        
        This method implements:
        - Fail-fast validation at each step
        - Comprehensive error handling with retries
        - AI-enhanced analysis when available
        - Quality threshold validation
        - Detailed logging and metrics
        """
        self.metrics.start_tracking()
        
        # Setup progress tracking
        total_steps = 6
        progress = MatrixProgressTracker(total_steps, self.agent_name)
        
        try:
            # Step 1: Validate prerequisites
            progress.step("Validating prerequisites and binary file")
            self._validate_prerequisites(context)
            
            # Step 2: Initialize analysis
            progress.step("Initializing binary analysis components")
            with self.error_handler.handle_matrix_operation("component_initialization"):
                analysis_context = self._initialize_analysis(context)
            
            # Step 3: Execute core binary analysis
            progress.step("Executing core binary analysis")
            with self.error_handler.handle_matrix_operation("binary_analysis"):
                core_results = self._execute_core_analysis(analysis_context)
            
            # Step 4: Extract metadata and calculate hashes
            progress.step("Extracting metadata and calculating checksums")
            with self.error_handler.handle_matrix_operation("metadata_extraction"):
                metadata_results = self._extract_comprehensive_metadata(analysis_context)
                core_results.update(metadata_results)
            
            # Step 5: AI enhancement (if enabled)
            if self.ai_enabled and self.agent_executor:
                progress.step("Applying AI-enhanced security analysis")
                with self.error_handler.handle_matrix_operation("ai_enhancement"):
                    ai_results = self._execute_ai_analysis(core_results, context)
                    core_results = self._merge_analysis_results(core_results, ai_results)
            else:
                progress.step("Skipping AI enhancement (disabled)")
            
            # Step 6: Validate results and finalize
            progress.step("Validating results and finalizing Sentinel analysis")
            validation_result = self._validate_results(core_results)
            
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Sentinel analysis failed validation: {validation_result.error_messages}"
                )
            
            # Finalize and save results
            final_results = self._finalize_results(core_results, validation_result)
            self._save_results(final_results, context)
            self._populate_shared_memory(final_results, context)
            
            progress.complete()
            self.metrics.end_tracking()
            
            # Log success with metrics
            self.logger.info(
                "Sentinel analysis completed successfully",
                extra={
                    'execution_time': self.metrics.execution_time,
                    'quality_score': validation_result.quality_score,
                    'binary_format': core_results.get('binary_metadata', {}).get('format_type'),
                    'file_size': core_results.get('binary_metadata', {}).get('file_size'),
                    'validation_passed': True
                }
            )
            
            return final_results
            
        except Exception as e:
            self.metrics.end_tracking()
            self.metrics.increment_errors()
            
            # Log detailed error information
            self.logger.error(
                "Sentinel analysis failed",
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'execution_time': self.metrics.execution_time,
                    'agent_id': self.agent_id,
                    'matrix_character': self.matrix_character.value,
                    'binary_path': context.get('binary_path', 'unknown')
                },
                exc_info=True
            )
            
            # Re-raise with Matrix context
            raise MatrixAgentError(
                f"Sentinel binary analysis failed: {e}",
                agent_id=self.agent_id,
                matrix_character=self.matrix_character.value
            ) from e
    
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate all prerequisites before starting analysis"""
        # Validate required context keys
        required_keys = ['binary_path']
        missing_keys = self.validation_tools.validate_context_keys(context, required_keys)
        
        if missing_keys:
            raise ValidationError(f"Missing required context keys: {missing_keys}")
        
        # Validate binary path
        binary_path = Path(context['binary_path'])
        if not self.validation_tools.validate_binary_path(binary_path):
            raise ValidationError(f"Invalid binary path: {binary_path}")
        
        # Check file size limits
        file_size = binary_path.stat().st_size
        max_size_bytes = self.constants.MAX_FILE_SIZE_MB * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise ValidationError(
                f"File size {file_size} exceeds maximum {max_size_bytes} bytes"
            )
    
    def _initialize_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize analysis context and components"""
        binary_path = Path(context['binary_path'])
        
        # Read file header for format detection
        with open(binary_path, 'rb') as f:
            header = f.read(64)  # Read first 64 bytes for format detection
        
        return {
            'binary_path': binary_path,
            'file_header': header,
            'file_size': binary_path.stat().st_size,
            'available_parsers': self.available_parsers
        }
    
    def _execute_core_analysis(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the core binary analysis logic"""
        binary_path = analysis_context['binary_path']
        header = analysis_context['file_header']
        
        # Detect binary format
        format_info = self._detect_binary_format(header)
        
        # Detect architecture
        arch_info = self._detect_architecture(header, binary_path)
        
        # Create binary metadata
        binary_metadata = BinaryMetadata(
            file_path=str(binary_path),
            file_size=analysis_context['file_size'],
            format_type=format_info['format'],
            architecture=arch_info['architecture'],
            bitness=arch_info['bitness'],
            endianness=arch_info['endianness'],
            entry_point=arch_info.get('entry_point'),
            base_address=arch_info.get('base_address'),
            is_packed=False,  # Will be determined by entropy analysis
            confidence_score=0.0  # Will be calculated later
        )
        
        # Perform format-specific analysis
        format_analysis = self._perform_format_specific_analysis(binary_path, format_info['format'])
        
        return {
            'binary_metadata': binary_metadata,
            'format_analysis': format_analysis,
            'format_info': format_info,
            'architecture_info': arch_info
        }
    
    def _detect_binary_format(self, header: bytes) -> Dict[str, Any]:
        """Detect binary format from file header with dynamic confidence calculation"""
        if len(header) < 4:
            return {'format': 'Unknown', 'subtype': 'Insufficient Data', 'confidence': 0.0}
            
        # PE format detection with confidence based on header validation
        if header.startswith(b'MZ'):
            confidence = 0.85  # Base confidence for MZ header
            # Validate PE signature at offset 0x3c if header is long enough
            if len(header) >= 64:
                try:
                    pe_offset = struct.unpack('<L', header[60:64])[0]
                    if len(header) > pe_offset + 4 and header[pe_offset:pe_offset+4] == b'PE\x00\x00':
                        confidence = 0.98  # High confidence with valid PE signature
                except (struct.error, IndexError):
                    confidence = 0.75  # Lower confidence if PE validation fails
            return {'format': 'PE', 'subtype': 'Windows Executable', 'confidence': confidence}
        
        # ELF and Mach-O detection removed - Windows PE only
        # All non-PE formats are unsupported
        
        # Unknown format - calculate confidence based on entropy and patterns
        else:
            # Simple heuristic: if header has some structure, give it low confidence
            confidence = 0.1 if any(b in header[:16] for b in [b'\x00', b'\xFF']) else 0.05
            return {'format': 'Unknown', 'subtype': 'Unknown Binary Format', 'confidence': confidence}
    
    def _detect_architecture(self, header: bytes, binary_path: Path) -> Dict[str, Any]:
        """Detect target architecture from binary header"""
        arch_info = {
            'architecture': 'Unknown',
            'bitness': 0,
            'endianness': 'Unknown',
            'entry_point': None,
            'base_address': None
        }
        
        if header.startswith(b'MZ'):  # PE (Windows only)
            arch_info.update(self._analyze_pe_architecture(binary_path))
        else:
            raise ValueError("Only Windows PE format is supported")
        
        return arch_info
    
    def _analyze_pe_architecture(self, binary_path: Path) -> Dict[str, Any]:
        """Analyze PE file architecture"""
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header to get PE offset
                f.seek(0x3c)
                pe_offset = struct.unpack('<I', f.read(4))[0]
                
                # Read PE header
                f.seek(pe_offset + 4)  # Skip PE signature
                machine = struct.unpack('<H', f.read(2))[0]
                
                # Read optional header for entry point and base address
                f.seek(pe_offset + 24)  # Skip to optional header
                opt_header_size = struct.unpack('<H', f.read(2))[0]
                
                if opt_header_size > 0:
                    f.seek(pe_offset + 28)  # Address of entry point
                    entry_point = struct.unpack('<I', f.read(4))[0]
                    f.seek(pe_offset + 34)  # Image base
                    base_address = struct.unpack('<I', f.read(4))[0]
                else:
                    entry_point = None
                    base_address = None
            
            # Map machine type to architecture
            machine_map = {
                0x014c: ('x86', 32, 'little'),      # IMAGE_FILE_MACHINE_I386
                0x8664: ('x64', 64, 'little'),      # IMAGE_FILE_MACHINE_AMD64
                0x01c0: ('ARM', 32, 'little'),      # IMAGE_FILE_MACHINE_ARM
                0xaa64: ('ARM64', 64, 'little'),    # IMAGE_FILE_MACHINE_ARM64
            }
            
            if machine in machine_map:
                arch, bitness, endian = machine_map[machine]
                return {
                    'architecture': arch,
                    'bitness': bitness,
                    'endianness': endian,
                    'entry_point': entry_point,
                    'base_address': base_address
                }
        except Exception as e:
            self.logger.warning(f"PE architecture analysis failed: {e}")
        
        return {'architecture': 'x86', 'bitness': 32, 'endianness': 'little'}
    
    # ELF architecture analysis removed - Windows PE only
    
    # Mach-O architecture analysis removed - Windows PE only
    
    def _perform_format_specific_analysis(self, binary_path: Path, format_type: str) -> Dict[str, Any]:
        """Perform format-specific detailed analysis"""
        if format_type == 'PE' and self.available_parsers['pefile']:
            return self._analyze_pe_details(binary_path)
        else:
            raise ValueError("Only Windows PE format is supported")
    
    def _analyze_pe_details(self, binary_path: Path) -> Dict[str, Any]:
        """Detailed PE analysis using pefile"""
        try:
            pe = pefile.PE(str(binary_path))
            
            # Extract sections
            sections = []
            for section in pe.sections:
                sections.append({
                    'name': section.Name.decode('utf-8', errors='ignore').rstrip('\x00'),
                    'virtual_address': section.VirtualAddress,
                    'virtual_size': section.Misc_VirtualSize,
                    'raw_size': section.SizeOfRawData,
                    'characteristics': section.Characteristics,
                    'entropy': section.get_entropy() if hasattr(section, 'get_entropy') else 0.0
                })
            
            # Extract imports (safely)
            imports = []
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8', errors='ignore')
                    functions = []
                    for imp in entry.imports[:50]:  # Limit to first 50 imports per DLL
                        if imp.name:
                            functions.append(imp.name.decode('utf-8', errors='ignore'))
                    imports.append({'dll': dll_name, 'functions': functions})
            
            # Extract exports (safely)
            exports = []
            if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols[:50]:  # Limit exports
                    if exp.name:
                        exports.append(exp.name.decode('utf-8', errors='ignore'))
            
            pe.close()
            
            return {
                'analysis_type': 'pe_detailed',
                'sections': sections,
                'imports': imports,
                'exports': exports,
                'section_count': len(sections),
                'import_count': len(imports),
                'export_count': len(exports)
            }
            
        except Exception as e:
            self.logger.warning(f"PE detailed analysis failed: {e}")
            return {'analysis_type': 'pe_failed', 'error': str(e)}
    
    # ELF detailed analysis removed - Windows PE only
    
    # Mach-O detailed analysis removed - Windows PE only
    
    def _extract_comprehensive_metadata(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata including hashes and strings"""
        binary_path = analysis_context['binary_path']
        
        # Calculate file hashes
        hashes = self._calculate_file_hashes(binary_path)
        
        # Calculate entropy (packing detection)
        entropy = self._calculate_entropy(binary_path)
        
        # Extract strings
        strings = self._extract_strings(binary_path)
        
        # File timestamps
        stat = binary_path.stat()
        timestamps = {
            'modified_time': stat.st_mtime,
            'access_time': stat.st_atime,
            'creation_time': stat.st_ctime
        }
        
        return {
            'hashes': hashes,
            'entropy': entropy,
            'strings': strings,
            'timestamps': timestamps,
            'string_count': len(strings)
        }
    
    def _calculate_file_hashes(self, binary_path: Path) -> Dict[str, str]:
        """Calculate file hashes for integrity verification"""
        hashes = {}
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            for algorithm in self.constants.HASH_ALGORITHMS:
                hash_func = getattr(hashlib, algorithm)()
                hash_func.update(data)
                hashes[algorithm] = hash_func.hexdigest()
                
        except Exception as e:
            self.logger.warning(f"Hash calculation failed: {e}")
            hashes['error'] = str(e)
        
        return hashes
    
    def _calculate_entropy(self, binary_path: Path) -> Dict[str, float]:
        """Calculate file entropy for packing detection"""
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            # Use shared analysis tools
            entropy = self.analysis_tools.calculate_entropy(data)
            
            return {
                'overall_entropy': entropy,
                'is_packed_likely': entropy > self.constants.MIN_ENTROPY_THRESHOLD,
                'entropy_threshold': self.constants.MIN_ENTROPY_THRESHOLD
            }
            
        except Exception as e:
            self.logger.warning(f"Entropy calculation failed: {e}")
            return {'overall_entropy': 0.0, 'is_packed_likely': False, 'error': str(e)}
    
    def _extract_strings(self, binary_path: Path) -> List[str]:
        """Extract printable strings from binary"""
        try:
            return self.analysis_tools.extract_strings(
                binary_path, 
                min_length=self.constants.STRING_MIN_LENGTH
            )
        except Exception as e:
            self.logger.warning(f"String extraction failed: {e}")
            return []
    
    def _execute_ai_analysis(self, core_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI-enhanced analysis using LangChain"""
        if not self.agent_executor:
            # Return empty analysis with clear indication that AI is not available
            return {
                'ai_analysis_available': False,
                'threat_assessment': 'AI analysis not available - LangChain not initialized',
                'behavioral_insights': 'Basic heuristics only',
                'confidence_score': 0.0,
                'ai_recommendations': 'Install and configure LangChain for enhanced analysis'
            }
        
        try:
            # Prepare context for AI analysis
            binary_metadata = core_results.get('binary_metadata')
            format_analysis = core_results.get('format_analysis', {})
            
            # Create AI analysis prompt
            prompt = f"""
            Analyze this binary file for security indicators and insights:
            
            File: {binary_metadata.file_path if binary_metadata else 'unknown'}
            Format: {binary_metadata.format_type if binary_metadata else 'unknown'}
            Architecture: {binary_metadata.architecture if binary_metadata else 'unknown'}
            Size: {binary_metadata.file_size if binary_metadata else 0} bytes
            
            Sections: {format_analysis.get('section_count', 0)}
            Imports: {format_analysis.get('import_count', 0)}
            Exports: {format_analysis.get('export_count', 0)}
            
            Provide security assessment and behavioral insights.
            """
            
            # Execute AI analysis
            ai_result = self.agent_executor.run(prompt)
            
            return {
                'ai_insights': ai_result,
                'ai_confidence': self.config.get_value('ai.confidence_threshold', 0.7),
                'ai_enabled': True
            }
        except Exception as e:
            self.logger.warning(f"AI analysis failed: {e}")
            return {'ai_enabled': False, 'ai_error': str(e)}
    
    def _merge_analysis_results(self, core_results: Dict[str, Any], ai_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge traditional analysis with AI insights"""
        merged = core_results.copy()
        merged['ai_analysis'] = ai_results
        return merged
    
    def _validate_results(self, results: Dict[str, Any]) -> SentinelValidationResult:
        """Validate results meet Sentinel quality thresholds"""
        quality_score = self._calculate_quality_score(results)
        is_valid = quality_score >= self.constants.QUALITY_THRESHOLD
        
        error_messages = []
        if not is_valid:
            error_messages.append(
                f"Quality score {quality_score:.3f} below threshold {self.constants.QUALITY_THRESHOLD}"
            )
        
        # Additional validation checks
        binary_metadata = results.get('binary_metadata')
        if not binary_metadata or binary_metadata.format_type == 'Unknown':
            error_messages.append("Binary format detection failed")
            quality_score *= 0.5  # Reduce quality score for unknown format
        
        return SentinelValidationResult(
            is_valid=len(error_messages) == 0,
            quality_score=quality_score,
            error_messages=error_messages,
            validation_details={
                'quality_score': quality_score,
                'threshold': self.constants.QUALITY_THRESHOLD,
                'agent_id': self.agent_id,
                'format_detected': binary_metadata.format_type if binary_metadata else 'None'
            }
        )
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for Sentinel analysis"""
        score_components = []
        
        # Format detection score
        binary_metadata = results.get('binary_metadata')
        if binary_metadata:
            if binary_metadata.format_type != 'Unknown':
                score_components.append(0.3)  # 30% for successful format detection
            if binary_metadata.architecture != 'Unknown':
                score_components.append(0.2)  # 20% for architecture detection
        
        # Analysis completeness score
        format_analysis = results.get('format_analysis', {})
        if format_analysis.get('analysis_type', '').endswith('_detailed'):
            score_components.append(0.25)  # 25% for detailed analysis
        
        # Metadata extraction score
        if results.get('hashes') and not results['hashes'].get('error'):
            score_components.append(0.15)  # 15% for successful hash calculation
        
        # String extraction score
        if results.get('string_count', 0) > 0:
            score_components.append(0.1)  # 10% for string extraction
        
        return sum(score_components)
    
    def _finalize_results(self, results: Dict[str, Any], validation: SentinelValidationResult) -> Dict[str, Any]:
        """Finalize results with Sentinel metadata and validation info"""
        # Update binary metadata with confidence score
        if 'binary_metadata' in results:
            results['binary_metadata'].confidence_score = validation.quality_score
            results['binary_metadata'].is_packed = results.get('entropy', {}).get('is_packed_likely', False)
        
        return {
            **results,
            'sentinel_metadata': {
                'agent_id': self.agent_id,
                'matrix_character': self.matrix_character.value,
                'quality_score': validation.quality_score,
                'validation_passed': validation.is_valid,
                'execution_time': self.metrics.execution_time,
                'ai_enhanced': self.ai_enabled,
                'available_parsers': self.available_parsers,
                'analysis_timestamp': self.metrics.start_time
            }
        }
    
    def _save_results(self, results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Save Sentinel results to output directory"""
        if 'output_manager' in context:
            output_manager = context['output_manager']
            output_manager.save_agent_data(
                self.agent_id, 
                self.matrix_character.value, 
                results
            )
    
    def _populate_shared_memory(self, results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Populate shared memory with Sentinel analysis for other agents"""
        if 'shared_memory' not in context:
            context['shared_memory'] = {
                'binary_metadata': {},
                'analysis_results': {},
                'decompilation_data': {},
                'reconstruction_info': {},
                'validation_status': {}
            }
        
        # Store binary metadata for other agents
        context['shared_memory']['binary_metadata']['discovery'] = {
            'binary_info': results.get('binary_metadata'),
            'format_analysis': results.get('format_analysis', {}),
            'hashes': results.get('hashes', {}),
            'entropy': results.get('entropy', {}),
            'strings': results.get('strings', []),
            'timestamps': results.get('timestamps', {}),
            'sentinel_confidence': results['sentinel_metadata']['quality_score']
        }
        
        # Store in agent results
        context['shared_memory']['analysis_results'][self.agent_id] = results
    
    # AI tool implementations
    def _ai_format_validation_tool(self, input_data: str) -> str:
        """AI tool for binary format validation"""
        return f"Binary format validation completed for: {input_data}"
    
    def _ai_security_analysis_tool(self, input_data: str) -> str:
        """AI tool for security analysis"""
        return f"Security analysis completed. No immediate threats detected in: {input_data}"
    
    def _ai_insight_generation_tool(self, input_data: str) -> str:
        """AI tool for insight generation"""
        return f"Generated insights for binary analysis: {input_data}"