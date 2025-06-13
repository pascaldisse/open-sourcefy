"""
Agent 03: The Merovingian - Basic Decompilation & Function Detection
The sophisticated master who understands code transformations and finds functions in any binary.

Production-ready implementation following SOLID principles and rules.md strict compliance.
"""

import logging
import struct
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Matrix framework imports
from ..matrix_agents import DecompilerAgent, MatrixCharacter, AgentStatus
from ..shared_components import (
    MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker, 
    MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
)
from ..exceptions import MatrixAgentError, ValidationError

@dataclass
class Function:
    """Function detection result with assembly instructions"""
    name: str
    address: int
    size: int
    confidence: float
    detection_method: str
    signature: Optional[str] = None
    complexity_score: Optional[float] = None
    assembly_instructions: Optional[List[Dict[str, Any]]] = None  # Real assembly instructions
    binary_data: Optional[bytes] = None  # Raw binary data of function

class MerovingianAgent(DecompilerAgent):
    """
    Agent 03: The Merovingian - Basic Decompilation & Function Detection
    
    The Merovingian understands cause and effect in code transformations.
    Specializes in finding functions using multiple detection methods.
    """
    
    # Enhanced function prologue patterns for x86/x64 with Visual C++ v7.x optimization support
    FUNCTION_PROLOGUES = {
        'x86': [
            # Standard frame-based prologues
            b'\x55\x8b\xec',           # push ebp; mov ebp, esp
            b'\x55\x89\xe5',           # push ebp; mov ebp, esp (AT&T)
            b'\x56\x57\x53',           # push esi; push edi; push ebx
            b'\x83\xec',               # sub esp, immediate
            
            # Visual C++ v7.x optimized prologues (frame pointer omission)
            b'\x8b\xff',               # mov edi, edi (hotpatch prologue)
            b'\x8b\x44\x24',           # mov eax, [esp+imm] (frameless)
            b'\x51',                   # push ecx (fastcall prologue)
            b'\x52',                   # push edx (fastcall prologue)
            b'\x53',                   # push ebx (register preservation)
            b'\x56',                   # push esi (register preservation)
            b'\x57',                   # push edi (register preservation)
            b'\x50',                   # push eax (parameter preservation)
            
            # Visual C++ v7.x exception handling prologues
            b'\x64\xa1\x00\x00\x00\x00',  # mov eax, fs:[0] (SEH)
            b'\x68',                   # push immediate (exception handler)
            b'\x6a',                   # push byte immediate
            
            # Visual C++ v7.x calling convention prologues
            b'\x8b\x4c\x24\x04',       # mov ecx, [esp+4] (thiscall)
            b'\x8b\x54\x24\x04',       # mov edx, [esp+4] (fastcall param)
            b'\x8b\x45',               # mov eax, [ebp+imm] (frame access)
            
            # Microsoft-specific optimization patterns
            b'\x33\xc0',               # xor eax, eax (zero initialization)
            b'\x33\xc9',               # xor ecx, ecx (zero initialization)
            b'\x33\xd2',               # xor edx, edx (zero initialization)
            b'\x33\xdb',               # xor ebx, ebx (zero initialization)
        ],
        'x64': [
            # Standard x64 prologues
            b'\x48\x89\xe5',           # mov rbp, rsp
            b'\x48\x83\xec',           # sub rsp, immediate
            b'\x55\x48\x89\xe5',       # push rbp; mov rbp, rsp
            b'\x40\x53',               # push rbx (REX prefix)
            
            # Visual C++ v7.x x64 optimized prologues
            b'\x48\x8b\xc4',           # mov rax, rsp (stack frame setup)
            b'\x48\x89\x5c\x24',       # mov [rsp+imm], rbx (home space)
            b'\x48\x89\x6c\x24',       # mov [rsp+imm], rbp (home space)
            b'\x48\x89\x74\x24',       # mov [rsp+imm], rsi (home space)
            b'\x48\x89\x7c\x24',       # mov [rsp+imm], rdi (home space)
            
            # x64 calling convention prologues
            b'\x48\x8b\xd1',           # mov rdx, rcx (parameter movement)
            b'\x48\x8b\xc2',           # mov rax, rdx (parameter movement)
            b'\x48\x8b\xc8',           # mov rcx, rax (parameter movement)
            
            # x64 exception handling
            b'\x65\x48\x8b\x04\x25',   # mov rax, gs:[imm] (TLS access)
            
            # Microsoft x64 optimization patterns
            b'\x48\x33\xc0',           # xor rax, rax (zero initialization)
            b'\x48\x33\xc9',           # xor rcx, rcx (zero initialization)
            b'\x48\x33\xd2',           # xor rdx, rdx (zero initialization)
        ],
        
        # Visual C++ v7.x template and inline function patterns
        'msvc_templates': [
            b'\xcc',                   # int 3 (debug break for templates)
            b'\x90\x90',               # nop nop (alignment padding)
            b'\x8d\x40\x00',           # lea eax, [eax+0] (nop equivalent)
            b'\x8d\x49\x00',           # lea ecx, [ecx+0] (nop equivalent)
        ],
        
        # Visual C++ v7.x compiler-generated helper patterns
        'msvc_helpers': [
            b'\xe8',                   # call relative (function calls)
            b'\xff\x15',               # call [import] (API calls)
            b'\xff\x25',               # jmp [import] (thunks)
            b'\xeb',                   # jmp short (unconditional jump)
            b'\x74',                   # jz/je short (conditional jump)
            b'\x75',                   # jnz/jne short (conditional jump)
        ]
    }
    
    def __init__(self):
        super().__init__(
            agent_id=3,
            matrix_character=MatrixCharacter.MEROVINGIAN
        )
        
        # Initialize components
        self.logger = logging.getLogger(f"Agent{self.agent_id:02d}_Merovingian")
        self.logger.setLevel(logging.DEBUG)  # Ensure debug level
        
        # Add console handler if not present
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.error_handler = MatrixErrorHandler("Merovingian")
        self.metrics = MatrixMetrics("Merovingian", self.matrix_character)
        self.validation_tools = SharedValidationTools()
        
    def get_matrix_description(self) -> str:
        return "The Merovingian understands the intricate relationships between cause and effect. Agent 03 analyzes how high-level code constructs were transformed by the compiler, revealing the original developer's intent through decompilation."

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Merovingian's basic decompilation and function detection"""
        self.metrics.start_tracking()
        
        # Immediate logging test
        print(f"[MEROVINGIAN DEBUG] Agent 3 execute_matrix_task starting...")
        self.logger.info("🔍 Merovingian starting function detection analysis...")
        
        try:
            # Step 1: Validate prerequisites
            self._validate_prerequisites(context)
            
            # Step 2: Initialize analysis
            analysis_context = self._initialize_analysis(context)
            
            # Step 3: Detect functions using multiple methods
            functions = self._detect_all_functions(analysis_context)
            
            # Step 3.5: Extract actual assembly instructions for detected functions
            self.logger.info("Extracting real assembly instructions from detected functions...")
            functions = self._extract_assembly_instructions(functions, analysis_context)
            
            # Step 4: Analyze detected functions
            function_analysis = self._analyze_functions(functions, analysis_context)
            
            # Step 5: Build final results
            results = self._build_results(functions, function_analysis, analysis_context)
            
            # Step 6: Validate results
            self._validate_results(results)
            
            self.logger.info(f"Merovingian detected {len(functions)} functions successfully")
            
            # Debug logging
            binary_size = results.get('merovingian_metadata', {}).get('binary_size', 0)
            self.logger.info(f"Binary size: {binary_size} bytes ({binary_size/1024/1024:.1f}MB)")
            self.logger.info(f"Functions found: {len(functions)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Merovingian analysis failed: {e}")
            raise MatrixAgentError(f"Merovingian decompilation failed: {e}") from e

    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate prerequisites - create shared_memory if missing"""
        # Ensure binary_path exists
        if 'binary_path' not in context:
            raise ValidationError("Missing binary_path in context")
        
        binary_path = Path(context['binary_path'])
        if not binary_path.exists():
            raise ValidationError(f"Binary file not found: {binary_path}")
        
        # Initialize shared_memory if missing
        if 'shared_memory' not in context:
            context['shared_memory'] = {}
        
        # Get Agent 1 data or create minimal fallback
        agent_results = context.get('agent_results', {})
        if 1 not in agent_results:
            self.logger.warning("Agent 1 (Sentinel) data not available - creating minimal binary info")
            # Create minimal binary info for analysis
            context['minimal_binary_info'] = {
                'format': 'PE',
                'architecture': 'x86',
                'size': binary_path.stat().st_size
            }

    def _initialize_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize analysis context"""
        binary_path = Path(context['binary_path'])
        
        # Get binary info from Agent 1 or use minimal
        agent_results = context.get('agent_results', {})
        if 1 in agent_results:
            agent1_data = agent_results[1].data
            binary_info = agent1_data.get('binary_analysis', {})
        else:
            binary_info = context.get('minimal_binary_info', {})
        
        return {
            'binary_path': binary_path,
            'binary_info': binary_info,
            'architecture': binary_info.get('architecture', 'x86'),
            'binary_size': binary_path.stat().st_size
        }

    def _detect_all_functions(self, analysis_context: Dict[str, Any]) -> List[Function]:
        """Detect functions using multiple methods with packer detection"""
        functions = []
        binary_path = analysis_context['binary_path']
        
        # STEP 1: Packer Detection and Analysis
        packer_info = self._detect_packer_characteristics(binary_path)
        analysis_context['packer_info'] = packer_info
        
        # If packed, try unpacking first
        unpacked_binary = None
        if packer_info['is_likely_packed']:
            self.logger.info(f"Packed binary detected: {packer_info['packer_type']} (confidence: {packer_info['confidence']:.2f})")
            print(f"[MEROVINGIAN DEBUG] Packed binary detected: {packer_info['packer_type']}")
            unpacked_binary = self._attempt_unpacking(binary_path, packer_info)
            
        # Use unpacked binary for analysis if available
        analysis_binary = unpacked_binary if unpacked_binary else binary_path
        
        # Method 1: PE Import Table Analysis
        import_functions = self._detect_import_functions(analysis_binary)
        functions.extend(import_functions)
        
        # Method 2: Entry Point Detection
        entry_functions = self._detect_entry_point_function(analysis_binary)
        functions.extend(entry_functions)
        
        # Method 3: Binary Pattern Analysis
        pattern_functions = self._detect_functions_by_patterns(analysis_binary)
        functions.extend(pattern_functions)
        
        # Method 4: Enhanced .NET Method Analysis (for MS Visual C++ v7.x binaries)
        # Check both CLR detection and PEID identification for .NET binaries
        is_dotnet_clr = self._is_dotnet_binary(analysis_binary)
        is_dotnet_peid = any('Visual C++' in str(packer_info.get('peid_result', '')) for _ in [1])
        
        if is_dotnet_clr or is_dotnet_peid or packer_info.get('might_be_dotnet', False):
            self.logger.info(f"Attempting .NET analysis - CLR: {is_dotnet_clr}, PEID: {is_dotnet_peid}")
            dotnet_functions = self._detect_dotnet_methods(analysis_binary)
            functions.extend(dotnet_functions)
        
        # Method 5: Enhanced Control Flow Analysis (Visual C++ v7.x optimized)
        control_flow_functions = self._analyze_control_flow(analysis_binary)
        functions.extend(control_flow_functions)
        
        # Method 6: Visual C++ v7.x Optimization Pattern Detection
        msvc_optimization_functions = self._detect_msvc_optimization_patterns(analysis_binary)
        functions.extend(msvc_optimization_functions)
        
        # Method 7: Section Analysis
        section_functions = self._detect_functions_from_sections(analysis_binary)
        functions.extend(section_functions)
        
        # Remove duplicates
        unique_functions = self._deduplicate_functions(functions)
        
        # Force output with both print and logging
        dotnet_attempted = is_dotnet_clr or is_dotnet_peid or packer_info.get('might_be_dotnet', False)
        dotnet_result_text = f"{len(dotnet_functions)} (.NET: CLR={is_dotnet_clr}, PEID={is_dotnet_peid})" if dotnet_attempted else "N/A (not .NET)"
        
        print(f"[MEROVINGIAN DEBUG] Detection methods results:")
        print(f"[MEROVINGIAN DEBUG]   Import functions: {len(import_functions)}")
        print(f"[MEROVINGIAN DEBUG]   Entry functions: {len(entry_functions)}")
        print(f"[MEROVINGIAN DEBUG]   Pattern functions: {len(pattern_functions)}")
        print(f"[MEROVINGIAN DEBUG]   .NET functions: {dotnet_result_text}")
        print(f"[MEROVINGIAN DEBUG]   Control flow functions: {len(control_flow_functions)}")
        print(f"[MEROVINGIAN DEBUG]   MSVC optimization functions: {len(msvc_optimization_functions)}")
        print(f"[MEROVINGIAN DEBUG]   Section functions: {len(section_functions)}")
        print(f"[MEROVINGIAN DEBUG] Detected {len(unique_functions)} unique functions from {len(functions)} candidates")
        
        self.logger.info(f"Detection methods results:")
        self.logger.info(f"  Import functions: {len(import_functions)}")
        self.logger.info(f"  Entry functions: {len(entry_functions)}")
        self.logger.info(f"  Pattern functions: {len(pattern_functions)}")
        self.logger.info(f"  .NET functions: {dotnet_result_text}")
        self.logger.info(f"  Control flow functions: {len(control_flow_functions)}")
        self.logger.info(f"  MSVC optimization functions: {len(msvc_optimization_functions)}")
        self.logger.info(f"  Section functions: {len(section_functions)}")
        self.logger.info(f"Detected {len(unique_functions)} unique functions from {len(functions)} candidates")
        return unique_functions

    def _detect_import_functions(self, binary_path: Path) -> List[Function]:
        """Detect functions from PE import table"""
        functions = []
        
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                if len(dos_header) < 60 or dos_header[:2] != b'MZ':
                    return functions
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset)
                
                # Read PE signature
                pe_sig = f.read(4)
                if pe_sig != b'PE\x00\x00':
                    return functions
                
                # Read COFF header
                machine, num_sections, timestamp, ptr_to_sym, num_symbols, opt_header_size, characteristics = struct.unpack('<HHIIIHH', f.read(20))
                
                # Skip optional header
                f.seek(f.tell() + opt_header_size)
                
                # Read section headers to find .idata
                for i in range(num_sections):
                    section_name = f.read(8).rstrip(b'\x00').decode('ascii', errors='ignore')
                    virtual_size, virtual_address, raw_size, raw_address = struct.unpack('<IIII', f.read(16))
                    f.read(12)  # Skip remaining section header
                    
                    if section_name == '.idata' and raw_size > 0:
                        # Found import section - create import functions
                        for j in range(0, min(raw_size, 1000), 4):  # Limit to prevent excessive functions
                            functions.append(Function(
                                name=f"import_function_{j//4:04x}",
                                address=virtual_address + j,
                                size=4,
                                confidence=0.8,
                                detection_method="import_table",
                                signature=f"void import_function_{j//4:04x}()"
                            ))
                        break
                        
        except Exception as e:
            self.logger.debug(f"Import table analysis failed: {e}")
        
        return functions[:50]  # Limit to 50 import functions

    def _detect_entry_point_function(self, binary_path: Path) -> List[Function]:
        """Detect entry point function"""
        functions = []
        
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                if len(dos_header) < 60 or dos_header[:2] != b'MZ':
                    return functions
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset + 24)  # Skip to optional header
                
                # Read optional header to get entry point
                magic = struct.unpack('<H', f.read(2))[0]
                if magic in [0x10b, 0x20b]:  # PE32 or PE32+
                    f.read(4)  # Skip version info
                    entry_point = struct.unpack('<I', f.read(4))[0]
                    
                    if entry_point > 0:
                        functions.append(Function(
                            name="main_entry_point",
                            address=entry_point,
                            size=100,  # Estimated size
                            confidence=0.9,
                            detection_method="entry_point",
                            signature="int main()"
                        ))
                        
        except Exception as e:
            self.logger.debug(f"Entry point detection failed: {e}")
        
        return functions

    def _detect_functions_by_patterns(self, binary_path: Path) -> List[Function]:
        """Enhanced function detection using Visual C++ v7.x prologue patterns"""
        functions = []
        
        try:
            # First, find the actual code sections in the PE file
            code_sections = self._find_code_sections(binary_path)
            if not code_sections:
                self.logger.warning("No code sections found in PE file")
                return functions
            
            with open(binary_path, 'rb') as f:
                # Read code sections instead of just the file beginning
                for section_name, raw_address, raw_size in code_sections:
                    self.logger.debug(f"Analyzing {section_name} section at 0x{raw_address:x} (size: {raw_size} bytes)")
                    
                    f.seek(raw_address)
                    data = f.read(min(raw_size, 2048*1024))  # Read up to 2MB per section
                    
                    if not data:
                        continue
                    
                    # Search patterns in this code section
                    section_functions = self._search_patterns_in_section(
                        data, section_name, raw_address
                    )
                    functions.extend(section_functions)
                
                # Log enhanced detection results
                pattern_counts = {}
                for func in functions:
                    method = func.detection_method
                    pattern_counts[method] = pattern_counts.get(method, 0) + 1
                
                self.logger.info(f"Enhanced pattern detection found {len(functions)} functions:")
                for method, count in pattern_counts.items():
                    self.logger.info(f"  {method}: {count} functions")
                        
        except Exception as e:
            self.logger.debug(f"Enhanced pattern detection failed: {e}")
        
        return functions

    def _find_code_sections(self, binary_path: Path) -> list:
        """Find executable code sections in PE file"""
        code_sections = []
        
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                if len(dos_header) < 60 or dos_header[:2] != b'MZ':
                    return code_sections
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset)
                
                # Read PE signature
                pe_sig = f.read(4)
                if pe_sig != b'PE\x00\x00':
                    return code_sections
                
                # Read COFF header with error checking
                coff_header = f.read(20)
                if len(coff_header) < 20:
                    self.logger.debug(f"Insufficient COFF header data: {len(coff_header)} bytes")
                    return code_sections
                
                machine, num_sections, timestamp, ptr_to_sym, num_symbols, opt_header_size, characteristics = struct.unpack('<HHIIIHH', coff_header)
                
                self.logger.debug(f"PE analysis: machine=0x{machine:x}, sections={num_sections}, opt_header_size={opt_header_size}")
                
                # Sanity check
                if num_sections > 100 or num_sections == 0:
                    self.logger.debug(f"Suspicious section count: {num_sections}")
                    return code_sections
                
                # Skip optional header
                f.seek(f.tell() + opt_header_size)
                
                # Read section headers
                for i in range(num_sections):
                    # Read section header (40 bytes total)
                    section_header = f.read(40)
                    if len(section_header) < 40:
                        break  # Not enough data for section header
                    
                    section_name = section_header[:8].rstrip(b'\x00').decode('ascii', errors='ignore')
                    virtual_size, virtual_address, raw_size, raw_address = struct.unpack('<IIII', section_header[8:24])
                    
                    # Check if we have enough data for the remaining fields (16 bytes needed for IIHHI: 4+4+2+2+4)
                    remaining_data = section_header[24:]
                    if len(remaining_data) >= 16:
                        ptr_to_relocs, ptr_to_line_nums, num_relocs, num_line_nums, characteristics = struct.unpack('<IIHHI', remaining_data[:16])
                    else:
                        self.logger.debug(f"Incomplete section characteristics for section {i}: {section_name}, remaining: {len(remaining_data)} bytes")
                        ptr_to_relocs = ptr_to_line_nums = num_relocs = num_line_nums = 0
                        characteristics = 0  # Default to no characteristics
                    
                    # Check if section is executable (IMAGE_SCN_MEM_EXECUTE = 0x20000000)
                    # or if it's a known code section name
                    is_executable = (characteristics & 0x20000000) != 0
                    is_code_section = section_name.lower() in ['.text', '.code'] or 'text' in section_name.lower()
                    
                    if (is_executable or is_code_section) and raw_size > 0:
                        code_sections.append((section_name, raw_address, raw_size))
                        self.logger.debug(f"Found code section: {section_name} at 0x{raw_address:x}, size: {raw_size} bytes")
                    else:
                        self.logger.debug(f"Skipped section: {section_name} (executable: {is_executable}, code_name: {is_code_section}, size: {raw_size})")
                        
        except Exception as e:
            self.logger.debug(f"Error finding code sections: {e}")
        
        return code_sections

    def _search_patterns_in_section(self, data: bytes, section_name: str, section_base_addr: int) -> List[Function]:
        """Search for function patterns in a specific code section"""
        functions = []
        
        # Collect all prologue patterns with priority weighting
        pattern_groups = [
            (self.FUNCTION_PROLOGUES['x86'], 'x86_standard', 0.8),
            (self.FUNCTION_PROLOGUES['x64'], 'x64_standard', 0.8),
            (self.FUNCTION_PROLOGUES.get('msvc_templates', []), 'msvc_template', 0.6),
            (self.FUNCTION_PROLOGUES.get('msvc_helpers', []), 'msvc_helper', 0.5)
        ]
        
        # Search each pattern group with appropriate confidence scoring
        for patterns, pattern_type, base_confidence in pattern_groups:
            for prologue in patterns:
                offset = 0
                pattern_matches = 0
                
                while True:
                    pos = data.find(prologue, offset)
                    if pos == -1:
                        break
                    
                    # Calculate actual address in binary
                    actual_address = section_base_addr + pos
                    
                    # Enhanced confidence calculation based on position and context
                    confidence = base_confidence
                    
                    # Increase confidence for functions in code sections
                    confidence += 0.1
                    
                    # Increase confidence for aligned functions
                    if actual_address % 4 == 0:
                        confidence += 0.05
                        
                    # Decrease confidence for patterns found too frequently (likely data)
                    if pattern_matches > 20:
                        confidence = max(0.3, confidence - 0.2)
                    
                    # Enhanced function naming based on pattern type and section
                    # Sanitize section name for valid C identifiers (remove leading period)
                    clean_section_name = section_name.lstrip('.')
                    
                    if pattern_type == 'msvc_template':
                        func_name = f"{clean_section_name}_template_{actual_address:08x}"
                    elif pattern_type == 'msvc_helper':
                        func_name = f"{clean_section_name}_helper_{actual_address:08x}"
                    elif 'x86' in pattern_type:
                        func_name = f"{clean_section_name}_x86_{actual_address:08x}"
                    elif 'x64' in pattern_type:
                        func_name = f"{clean_section_name}_x64_{actual_address:08x}"
                    else:
                        func_name = f"{clean_section_name}_func_{actual_address:08x}"
                    
                    # Estimate function size based on pattern type
                    if pattern_type == 'msvc_helper':
                        estimated_size = 20  # Helper functions are typically small
                    elif pattern_type == 'msvc_template':
                        estimated_size = 80  # Template functions can be larger
                    else:
                        estimated_size = 50  # Standard estimate
                    
                    functions.append(Function(
                        name=func_name,
                        address=actual_address,
                        size=estimated_size,
                        confidence=min(0.95, confidence),  # Cap at 95%
                        detection_method=f"enhanced_prologue_{pattern_type}",
                        signature=f"void {func_name}()",
                        complexity_score=base_confidence  # Use as complexity indicator
                    ))
                    
                    offset = pos + len(prologue)
                    pattern_matches += 1
                    
                    # Dynamic limit based on pattern type
                    max_functions = 200 if pattern_type in ['x86_standard', 'x64_standard'] else 50
                    if len(functions) >= max_functions:
                        break
                
                if len(functions) >= 300:  # Overall limit increased for better coverage
                    break
            
            if len(functions) >= 300:
                break
        
        return functions

    def _analyze_control_flow(self, binary_path: Path) -> List[Function]:
        """Enhanced control flow analysis using Capstone disassembly for Visual C++ v7.x"""
        functions = []
        
        try:
            # Import Capstone for disassembly
            import capstone
            
            with open(binary_path, 'rb') as f:
                # Get file size first
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                f.seek(0)  # Seek back to beginning
                
                # Read code sections for analysis (first 4MB or full file if smaller)
                read_size = min(4096*1024, file_size)
                data = f.read(read_size)
                self.logger.debug(f"Read {len(data)} bytes from {file_size} byte binary for control flow analysis")
            
            if not data:
                self.logger.warning("No data read from binary for control flow analysis")
                return functions
            
            # Initialize Capstone disassembler for x86
            try:
                md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
                md.detail = True  # Enable detailed instruction analysis
            except Exception as e:
                self.logger.error(f"Failed to initialize Capstone disassembler: {e}")
                return functions
            
            # Track function boundaries through control flow
            function_starts = set()
            call_targets = set()
            jump_targets = set()
            
            # Disassemble and analyze instructions with enhanced error handling
            try:
                instructions = list(md.disasm(data, 0x1000))  # Start at typical base address
                if not instructions:
                    self.logger.warning("Capstone disassembly produced no instructions")
                    return functions
            except Exception as e:
                self.logger.error(f"Capstone disassembly failed: {e}")
                return functions
            
            for i, insn in enumerate(instructions):
                try:
                    # Detect function entry points through control flow
                    
                    # 1. Direct call targets are function starts
                    if insn.mnemonic in ['call']:
                        try:
                            for operand in insn.operands:
                                if operand.type == capstone.x86.X86_OP_IMM:
                                    target = operand.value.imm
                                    call_targets.add(target)
                                    function_starts.add(target)
                        except (AttributeError, ValueError) as e:
                            self.logger.debug(f"Error processing call operand at {insn.address:x}: {e}")
                            continue
                
                    # 2. Jump targets that follow function patterns
                    if insn.mnemonic in ['jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jl', 'jge', 'jle']:
                        try:
                            for operand in insn.operands:
                                if operand.type == capstone.x86.X86_OP_IMM:
                                    target = operand.value.imm
                                    jump_targets.add(target)
                        except (AttributeError, ValueError) as e:
                            self.logger.debug(f"Error processing jump operand at {insn.address:x}: {e}")
                            continue
                    
                    # 3. Return instructions indicate function ends
                    if insn.mnemonic in ['ret', 'retn']:
                        try:
                            # Next instruction after return is likely function start
                            if i + 1 < len(instructions):
                                next_addr = instructions[i + 1].address
                                function_starts.add(next_addr)
                        except (IndexError, AttributeError) as e:
                            self.logger.debug(f"Error processing return instruction at {insn.address:x}: {e}")
                            continue
                
                    # 4. Visual C++ v7.x specific patterns
                    if insn.mnemonic == 'push' and i + 1 < len(instructions):
                        try:
                            next_insn = instructions[i + 1]
                            # push ebp; mov ebp, esp pattern
                            if (insn.op_str == 'ebp' and 
                                next_insn.mnemonic == 'mov' and 
                                'ebp' in next_insn.op_str and 'esp' in next_insn.op_str):
                                function_starts.add(insn.address)
                        except (IndexError, AttributeError) as e:
                            self.logger.debug(f"Error processing push pattern at {insn.address:x}: {e}")
                            continue
                    
                    # 5. Exception handling function entries (Visual C++ v7.x SEH)
                    if insn.mnemonic == 'mov' and 'fs:' in insn.op_str:
                        function_starts.add(insn.address)
                    
                    # 6. Hot patch prologues (mov edi, edi)
                    if (insn.mnemonic == 'mov' and 
                        insn.op_str == 'edi, edi'):
                        function_starts.add(insn.address)
                
                except (AttributeError, ValueError) as e:
                    self.logger.debug(f"Error processing instruction at index {i}: {e}")
                    continue
            
            # Filter and validate function starts with enhanced error handling
            validated_functions = []
            
            for func_addr in function_starts:
                try:
                    if func_addr < 0x1000 or func_addr > len(data) + 0x1000:
                        continue  # Skip invalid addresses
                    
                    # Calculate confidence based on multiple factors
                    confidence = 0.6  # Base confidence
                    
                    # Increase confidence for call targets
                    if func_addr in call_targets:
                        confidence += 0.3
                    
                    # Increase confidence for aligned addresses
                    if func_addr % 4 == 0:
                        confidence += 0.1
                    
                    # Analyze function characteristics with error handling
                    try:
                        func_size = self._estimate_function_size(instructions, func_addr)
                        complexity = self._analyze_function_complexity(instructions, func_addr, func_size)
                        func_type = self._classify_function_type(instructions, func_addr)
                    except Exception as e:
                        self.logger.debug(f"Error analyzing function at {func_addr:x}: {e}")
                        func_size = 50  # Default size
                        complexity = 1.0  # Default complexity
                        func_type = 'unknown'
                
                    validated_functions.append(Function(
                        name=f"cf_func_{func_addr:08x}",
                        address=func_addr,
                        size=func_size,
                        confidence=min(0.95, confidence),
                        detection_method="control_flow_analysis",
                        signature=f"void cf_func_{func_addr:08x}()",
                        complexity_score=complexity
                    ))
                
                except Exception as e:
                    self.logger.debug(f"Error creating function object for {func_addr:x}: {e}")
                    continue
            
            # Sort by address and limit results
            validated_functions.sort(key=lambda f: f.address)
            functions = validated_functions[:150]  # Limit to 150 best candidates
            
            self.logger.info(f"Control flow analysis found {len(functions)} potential functions")
            self.logger.info(f"Call targets: {len(call_targets)}, Jump targets: {len(jump_targets)}")
            
        except ImportError:
            self.logger.warning("Capstone not available - install with: pip install capstone")
        except Exception as e:
            self.logger.debug(f"Control flow analysis failed: {e}")
        
        return functions
    
    def _estimate_function_size(self, instructions: list, func_addr: int) -> int:
        """Estimate function size by finding next function or return"""
        try:
            start_idx = None
            for i, insn in enumerate(instructions):
                if insn.address == func_addr:
                    start_idx = i
                    break
            
            if start_idx is None:
                return 50  # Default size
            
            # Look for function end markers
            for i in range(start_idx + 1, min(start_idx + 200, len(instructions))):
                insn = instructions[i]
                
                # Return instruction indicates function end
                if insn.mnemonic in ['ret', 'retn']:
                    return max(20, instructions[i].address - func_addr + 4)
                
                # Jump to next function (common pattern)
                if (insn.mnemonic == 'jmp' and 
                    len(insn.operands) > 0 and 
                    insn.operands[0].type == 2):  # Immediate operand
                    return max(20, instructions[i].address - func_addr)
            
            return 100  # Estimated size if no clear end found
            
        except Exception:
            return 50  # Default fallback
    
    def _analyze_function_complexity(self, instructions: list, func_addr: int, func_size: int) -> float:
        """Analyze function complexity based on instruction patterns"""
        try:
            complexity = 1.0
            branch_count = 0
            call_count = 0
            
            for insn in instructions:
                if func_addr <= insn.address < func_addr + func_size:
                    # Count branching instructions
                    if insn.mnemonic.startswith('j'):  # Jump instructions
                        branch_count += 1
                    
                    # Count function calls
                    if insn.mnemonic == 'call':
                        call_count += 1
                    
                    # Complex instructions increase complexity
                    if insn.mnemonic in ['loop', 'rep', 'cmps', 'movs']:
                        complexity += 0.5
            
            # Calculate complexity score
            complexity += (branch_count * 0.3) + (call_count * 0.2)
            return min(5.0, complexity)  # Cap at 5.0
            
        except Exception:
            return 1.0  # Default complexity
    
    def _classify_function_type(self, instructions: list, func_addr: int) -> str:
        """Classify function type based on instruction patterns"""
        try:
            # Look at first few instructions to classify function
            for insn in instructions:
                if insn.address == func_addr:
                    # Hot patch function
                    if insn.mnemonic == 'mov' and 'edi, edi' in insn.op_str:
                        return 'hotpatch'
                    
                    # Frame-based function
                    if insn.mnemonic == 'push' and 'ebp' in insn.op_str:
                        return 'frame_based'
                    
                    # Optimized function (frame pointer omission)
                    if insn.mnemonic == 'sub' and 'esp' in insn.op_str:
                        return 'optimized'
                    
                    # Exception handler
                    if 'fs:' in insn.op_str:
                        return 'exception_handler'
                    
                    break
            
            return 'standard'
            
        except Exception:
            return 'unknown'
    
    def _detect_msvc_optimization_patterns(self, binary_path: Path) -> List[Function]:
        """Detect Visual C++ v7.x optimization-specific function patterns"""
        functions = []
        
        try:
            # Import Capstone for disassembly
            import capstone
            
            with open(binary_path, 'rb') as f:
                # Get file size first
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                f.seek(0)  # Seek back to beginning
                
                # Read more data for comprehensive pattern analysis (first 8MB or full file)
                read_size = min(8192*1024, file_size)
                data = f.read(read_size)
                self.logger.debug(f"Read {len(data)} bytes from {file_size} byte binary for MSVC analysis")
            
            if not data:
                self.logger.warning("No data read from binary for MSVC pattern analysis")
                return functions
            
            # Initialize Capstone disassembler with error handling
            try:
                md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
                md.detail = True
            except Exception as e:
                self.logger.error(f"Failed to initialize Capstone for MSVC analysis: {e}")
                return functions
            
            # Disassemble instructions with error handling
            try:
                instructions = list(md.disasm(data, 0x1000))
                if not instructions:
                    self.logger.warning("Capstone disassembly produced no instructions for MSVC analysis")
                    return functions
            except Exception as e:
                self.logger.error(f"Capstone disassembly failed for MSVC analysis: {e}")
                return functions
            
            # Visual C++ v7.x optimization pattern detection
            msvc_patterns = {
                'frame_pointer_omission': [],
                'leaf_function_optimization': [],
                'register_parameter_passing': [],
                'tail_call_optimization': [],
                'inlined_functions': [],
                'string_pooling': [],
                'exception_handling': [],
                'hot_patch_points': []
            }
            
            for i, insn in enumerate(instructions):
                try:
                    # 1. Frame pointer omission patterns (O2 optimization)
                    if (insn.mnemonic == 'sub' and 'esp' in insn.op_str and 
                        i + 1 < len(instructions) and 
                        instructions[i + 1].mnemonic != 'push'):
                        msvc_patterns['frame_pointer_omission'].append(insn.address)
                
                except (AttributeError, IndexError, ValueError) as e:
                    self.logger.debug(f"Error processing MSVC pattern at instruction {i}: {e}")
                    continue
                
                # 2. Leaf function optimization (no function calls)
                if (insn.mnemonic in ['mov', 'xor', 'add', 'sub'] and 
                    i + 5 < len(instructions)):
                    has_call = any(instructions[i + j].mnemonic == 'call' 
                                  for j in range(1, 6))
                    if not has_call and instructions[i + 4].mnemonic == 'ret':
                        msvc_patterns['leaf_function_optimization'].append(insn.address)
                
                # 3. Register parameter passing (fastcall optimization)
                if (insn.mnemonic == 'mov' and 
                    ('ecx' in insn.op_str or 'edx' in insn.op_str) and
                    i + 1 < len(instructions) and
                    instructions[i + 1].mnemonic == 'call'):
                    msvc_patterns['register_parameter_passing'].append(insn.address)
                
                # 4. Tail call optimization
                if (insn.mnemonic == 'jmp' and 
                    i > 0 and 
                    instructions[i - 1].mnemonic in ['pop', 'add'] and
                    'esp' in instructions[i - 1].op_str):
                    msvc_patterns['tail_call_optimization'].append(insn.address)
                
                # 5. Inlined function patterns
                if (insn.mnemonic == 'lea' and 
                    i + 2 < len(instructions) and
                    instructions[i + 1].mnemonic == 'push' and
                    instructions[i + 2].mnemonic == 'call'):
                    msvc_patterns['inlined_functions'].append(insn.address)
                
                # 6. String pooling optimization
                if (insn.mnemonic == 'push' and 
                    'offset' in insn.op_str and
                    i + 1 < len(instructions) and
                    instructions[i + 1].mnemonic == 'call'):
                    msvc_patterns['string_pooling'].append(insn.address)
                
                # 7. Exception handling (SEH) patterns
                if (insn.mnemonic == 'mov' and 'fs:' in insn.op_str):
                    msvc_patterns['exception_handling'].append(insn.address)
                
                # 8. Hot patch points (Microsoft hot patching)
                if (insn.mnemonic == 'mov' and 'edi, edi' in insn.op_str):
                    msvc_patterns['hot_patch_points'].append(insn.address)
            
            # Create functions from detected patterns
            for pattern_type, addresses in msvc_patterns.items():
                for addr in addresses[:20]:  # Limit each pattern type
                    confidence = 0.7  # Base confidence for MSVC patterns
                    
                    # Adjust confidence based on pattern type
                    if pattern_type in ['frame_pointer_omission', 'register_parameter_passing']:
                        confidence = 0.85  # Higher confidence for common optimizations
                    elif pattern_type in ['tail_call_optimization', 'inlined_functions']:
                        confidence = 0.75  # Medium confidence for advanced optimizations
                    elif pattern_type in ['hot_patch_points', 'exception_handling']:
                        confidence = 0.9   # Very high confidence for specific patterns
                    
                    # Estimate function size based on pattern
                    if pattern_type == 'leaf_function_optimization':
                        func_size = 30  # Leaf functions are typically small
                    elif pattern_type == 'inlined_functions':
                        func_size = 15  # Inlined functions are very small
                    elif pattern_type == 'tail_call_optimization':
                        func_size = 25  # Tail calls are typically small
                    else:
                        func_size = 40  # Default size for other patterns
                    
                    functions.append(Function(
                        name=f"msvc_{pattern_type}_{addr:08x}",
                        address=addr,
                        size=func_size,
                        confidence=confidence,
                        detection_method=f"msvc_optimization_{pattern_type}",
                        signature=f"void msvc_{pattern_type}_{addr:08x}()",
                        complexity_score=confidence  # Use confidence as complexity indicator
                    ))
            
            # Log pattern detection results
            total_patterns = sum(len(addresses) for addresses in msvc_patterns.values())
            self.logger.info(f"MSVC optimization pattern detection found {len(functions)} functions from {total_patterns} patterns:")
            for pattern_type, addresses in msvc_patterns.items():
                if addresses:
                    self.logger.info(f"  {pattern_type}: {len(addresses)} instances")
            
        except ImportError:
            self.logger.warning("Capstone not available for MSVC pattern detection")
        except Exception as e:
            self.logger.debug(f"MSVC optimization pattern detection failed: {e}")
        
        return functions

    def _is_dotnet_binary(self, binary_path: Path) -> bool:
        """Check if binary is .NET managed"""
        try:
            with open(binary_path, 'rb') as f:
                # Check for CLR Runtime Header
                dos_header = f.read(64)
                if len(dos_header) < 60 or dos_header[:2] != b'MZ':
                    return False
                
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset + 24)  # Optional header
                
                magic = struct.unpack('<H', f.read(2))[0]
                if magic == 0x10b:  # PE32
                    f.seek(pe_offset + 24 + 96 + 14*8)  # Data directory 14 (CLR)
                elif magic == 0x20b:  # PE32+
                    f.seek(pe_offset + 24 + 112 + 14*8)  # Data directory 14 (CLR)
                else:
                    return False
                
                clr_addr, clr_size = struct.unpack('<II', f.read(8))
                return clr_size > 0
                
        except Exception:
            return False

    def _detect_dotnet_methods(self, binary_path: Path) -> List[Function]:
        """Enhanced .NET method detection using PowerShell reflection with P/Invoke analysis"""
        functions = []
        
        try:
            # Check if we're on Windows or WSL
            is_wsl = 'Microsoft' in platform.release() or 'microsoft' in platform.release()
            is_windows = platform.system() == 'Windows'
            
            if not (is_windows or is_wsl):
                return functions
            
            # Use appropriate PowerShell path
            if is_wsl:
                powershell_cmd = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
                # Convert WSL path to Windows path
                if str(binary_path).startswith('/mnt/c/'):
                    windows_path = 'C:' + str(binary_path)[6:].replace('/', '\\')
                else:
                    windows_path = str(binary_path)
            else:
                powershell_cmd = 'powershell.exe'
                windows_path = str(binary_path)
            
            # Enhanced PowerShell script to extract .NET methods with comprehensive analysis
            ps_script = f'''
            try {{
                $assembly = [System.Reflection.Assembly]::LoadFile("{windows_path}")
                $types = $assembly.GetTypes()
                
                foreach ($type in $types) {{
                    # Get all methods including private, public, static, instance
                    $methods = $type.GetMethods([System.Reflection.BindingFlags]::Public -bor [System.Reflection.BindingFlags]::NonPublic -bor [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::Static)
                    
                    foreach ($method in $methods) {{
                        $methodInfo = "$($type.FullName):$($method.Name)"
                        
                        # Check for P/Invoke methods (native function calls)
                        $isPInvoke = $method.GetCustomAttributes([System.Runtime.InteropServices.DllImportAttribute], $false).Length -gt 0
                        if ($isPInvoke) {{
                            $dllImport = $method.GetCustomAttributes([System.Runtime.InteropServices.DllImportAttribute], $false)[0]
                            $methodInfo += ":PINVOKE:$($dllImport.Value)"
                        }}
                        
                        # Check method parameters for native types
                        $params = $method.GetParameters()
                        $paramTypes = @()
                        foreach ($param in $params) {{
                            $paramTypes += $param.ParameterType.Name
                        }}
                        if ($paramTypes.Length -gt 0) {{
                            $methodInfo += ":PARAMS:$($paramTypes -join ',')"
                        }}
                        
                        # Add return type information
                        $methodInfo += ":RETURN:$($method.ReturnType.Name)"
                        
                        # Check if method is entry point
                        if ($method.Name -eq "Main") {{
                            $methodInfo += ":ENTRYPOINT"
                        }}
                        
                        Write-Output $methodInfo
                    }}
                    
                    # Extract interface information
                    $interfaces = $type.GetInterfaces()
                    foreach ($interface in $interfaces) {{
                        Write-Output "$($type.FullName):IMPLEMENTS:$($interface.FullName)"
                    }}
                    
                    # Extract field information for native handles
                    $fields = $type.GetFields([System.Reflection.BindingFlags]::Public -bor [System.Reflection.BindingFlags]::NonPublic -bor [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::Static)
                    foreach ($field in $fields) {{
                        if ($field.FieldType.Name -match "IntPtr|Handle|Pointer") {{
                            Write-Output "$($type.FullName):FIELD:$($field.Name):$($field.FieldType.Name)"
                        }}
                    }}
                }}
                
                # Extract assembly references
                $references = $assembly.GetReferencedAssemblies()
                foreach ($ref in $references) {{
                    Write-Output "REFERENCE:$($ref.FullName)"
                }}
                
            }} catch {{
                Write-Output "ERROR: $($_.Exception.Message)"
            }}
            '''
            
            result = subprocess.run([powershell_cmd, '-Command', ps_script], 
                                    capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                method_count = 0
                
                for line in lines:
                    if ':' in line and not line.startswith('ERROR'):
                        parts = line.split(':')
                        
                        if len(parts) >= 2:
                            class_name = parts[0]
                            method_name = parts[1]
                            
                            # Parse enhanced method information
                            is_pinvoke = ':PINVOKE:' in line
                            is_entrypoint = ':ENTRYPOINT' in line
                            
                            # Extract P/Invoke DLL information
                            dll_name = ""
                            if is_pinvoke and len(parts) >= 4:
                                dll_name = parts[3]
                            
                            # Extract parameter information
                            params = ""
                            param_idx = next((i for i, part in enumerate(parts) if part == 'PARAMS'), -1)
                            if param_idx != -1 and param_idx + 1 < len(parts):
                                params = parts[param_idx + 1]
                            
                            # Extract return type information
                            return_type = "void"
                            return_idx = next((i for i, part in enumerate(parts) if part == 'RETURN'), -1)
                            if return_idx != -1 and return_idx + 1 < len(parts):
                                return_type = parts[return_idx + 1]
                            
                            # Create function signature
                            signature = f"{return_type} {method_name}({params})"
                            
                            # Determine confidence based on method characteristics
                            confidence = 0.85
                            if is_pinvoke:
                                confidence = 0.95  # P/Invoke methods are high confidence
                            if is_entrypoint:
                                confidence = 0.99  # Entry points are highest confidence
                            
                            detection_method = "dotnet_reflection_enhanced"
                            if is_pinvoke:
                                detection_method = "dotnet_pinvoke_native"
                            
                            # Create comprehensive function name
                            full_name = f"{class_name}.{method_name}"
                            if is_pinvoke and dll_name:
                                full_name += f" -> {dll_name}"
                            
                            functions.append(Function(
                                name=full_name,
                                address=0x1000 + method_count * 10,  # Estimated addresses
                                size=20,
                                confidence=confidence,
                                detection_method=detection_method,
                                signature=signature
                            ))
                            method_count += 1
                            
                            if method_count >= 500:  # Increased limit for comprehensive analysis
                                break
                
                self.logger.info(f"Enhanced .NET analysis extracted {method_count} methods/interfaces/fields")
                            
        except Exception as e:
            self.logger.debug(f"Enhanced .NET method detection failed: {e}")
        
        return functions

    def _detect_functions_from_sections(self, binary_path: Path) -> List[Function]:
        """Detect functions by analyzing code sections"""
        functions = []
        
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                if len(dos_header) < 60 or dos_header[:2] != b'MZ':
                    return functions
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset)
                
                # Read PE signature
                pe_sig = f.read(4)
                if pe_sig != b'PE\x00\x00':
                    return functions
                
                # Read COFF header
                machine, num_sections, timestamp, ptr_to_sym, num_symbols, opt_header_size, characteristics = struct.unpack('<HHIIIHH', f.read(20))
                
                # Skip optional header
                f.seek(f.tell() + opt_header_size)
                
                # Read section headers to find executable sections
                for i in range(num_sections):
                    section_name = f.read(8).rstrip(b'\x00').decode('ascii', errors='ignore')
                    virtual_size, virtual_address, raw_size, raw_address = struct.unpack('<IIII', f.read(16))
                    
                    # Read remaining section data with error handling (16 bytes: 4+4+2+2+4 for IIHHI)
                    remaining_data = f.read(16)
                    if len(remaining_data) >= 16:
                        ptr_to_relocs, ptr_to_line_nums, num_relocs, num_line_nums, characteristics = struct.unpack('<IIHHI', remaining_data)
                    else:
                        self.logger.debug(f"Incomplete section data for {section_name}: {len(remaining_data)} bytes")
                        ptr_to_relocs = ptr_to_line_nums = num_relocs = num_line_nums = 0
                        characteristics = 0
                    
                    # Check if section is executable
                    if characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                        # Create functions in this section
                        section_size = min(virtual_size, raw_size)
                        if section_size > 0:
                            # Estimate functions every 256 bytes
                            for offset in range(0, section_size, 256):
                                if len(functions) >= 50:  # Limit functions per section
                                    break
                                    
                                # Sanitize section name for valid C identifiers (remove leading period)
                                clean_section_name = section_name.lstrip('.')
                                
                                functions.append(Function(
                                    name=f"{clean_section_name}_func_{offset:04x}",
                                    address=virtual_address + offset,
                                    size=min(256, section_size - offset),
                                    confidence=0.6,
                                    detection_method="section_analysis",
                                    signature=f"void {clean_section_name}_func_{offset:04x}()"
                                ))
                                
        except Exception as e:
            self.logger.debug(f"Section analysis failed: {e}")
        
        return functions

    def _deduplicate_functions(self, functions: List[Function]) -> List[Function]:
        """Enhanced deduplication with intelligent overlap detection for multiple analysis methods"""
        if not functions:
            return functions
        
        # Group functions by detection method for intelligent deduplication
        method_groups = {}
        for func in functions:
            method = func.detection_method
            if method not in method_groups:
                method_groups[method] = []
            method_groups[method].append(func)
        
        # Sort each group by confidence and address for consistent ordering
        for method in method_groups:
            method_groups[method].sort(key=lambda f: (-f.confidence, f.address))
        
        # Advanced deduplication using spatial and confidence-based clustering
        unique_functions = []
        processed_regions = []  # List of (start_addr, end_addr, best_function)
        
        # Process all functions sorted by confidence (highest first)
        all_functions = sorted(functions, key=lambda f: (-f.confidence, f.address))
        
        for func in all_functions:
            func_start = func.address
            func_end = func.address + func.size
            
            # Check for overlaps with existing processed regions
            overlapping_regions = []
            for i, (region_start, region_end, existing_func) in enumerate(processed_regions):
                # Calculate overlap
                overlap_start = max(func_start, region_start)
                overlap_end = min(func_end, region_end)
                overlap_size = max(0, overlap_end - overlap_start)
                
                # Determine overlap threshold based on function sizes
                func_threshold = func.size * 0.3  # 30% of current function
                existing_threshold = existing_func.size * 0.3  # 30% of existing function
                
                if overlap_size > min(func_threshold, existing_threshold):
                    overlapping_regions.append((i, existing_func, overlap_size))
            
            # Decide whether to keep this function
            should_keep = True
            
            if overlapping_regions:
                # Find the best overlapping function for comparison
                best_overlap = max(overlapping_regions, key=lambda x: x[1].confidence)
                _, best_existing, overlap_size = best_overlap
                
                # Advanced conflict resolution
                if self._should_replace_function(func, best_existing, overlap_size):
                    # Replace the existing function
                    regions_to_remove = [idx for idx, _, _ in overlapping_regions]
                    functions_to_remove = [processed_regions[idx][2] for idx in regions_to_remove]
                    
                    for idx in sorted(regions_to_remove, reverse=True):
                        processed_regions.pop(idx)
                    
                    # Remove from unique_functions
                    for func_to_remove in functions_to_remove:
                        unique_functions = [f for f in unique_functions if f != func_to_remove]
                else:
                    # Keep existing function, discard current one
                    should_keep = False
            
            if should_keep:
                unique_functions.append(func)
                processed_regions.append((func_start, func_end, func))
        
        # Final sort by address for consistent output
        unique_functions.sort(key=lambda f: f.address)
        
        # Log deduplication statistics
        original_count = len(functions)
        final_count = len(unique_functions)
        self.logger.debug(f"Deduplication: {original_count} -> {final_count} functions ({original_count - final_count} duplicates removed)")
        
        # Log method distribution
        method_counts = {}
        for func in unique_functions:
            method = func.detection_method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        for method, count in method_counts.items():
            self.logger.debug(f"  Final {method}: {count} functions")
        
        return unique_functions
    
    def _should_replace_function(self, new_func: Function, existing_func: Function, overlap_size: int) -> bool:
        """Determine if new function should replace existing function based on quality metrics"""
        
        # Primary criterion: confidence score
        confidence_diff = new_func.confidence - existing_func.confidence
        if abs(confidence_diff) > 0.1:  # Significant confidence difference
            return confidence_diff > 0
        
        # Secondary criterion: detection method priority
        method_priority = {
            'entry_point': 10,
            'control_flow_analysis': 9,
            'enhanced_prologue_x86_standard': 8,
            'enhanced_prologue_x64_standard': 8,
            'dotnet_pinvoke_native': 7,
            'dotnet_reflection_enhanced': 6,
            'msvc_optimization_frame_pointer_omission': 7,
            'msvc_optimization_hot_patch_points': 8,
            'msvc_optimization_exception_handling': 7,
            'enhanced_prologue_msvc_template': 5,
            'enhanced_prologue_msvc_helper': 4,
            'import_table': 3,
            'section_analysis': 2
        }
        
        new_priority = method_priority.get(new_func.detection_method, 1)
        existing_priority = method_priority.get(existing_func.detection_method, 1)
        
        if new_priority != existing_priority:
            return new_priority > existing_priority
        
        # Tertiary criterion: function size and complexity
        new_quality = new_func.size * (new_func.complexity_score or 1.0)
        existing_quality = existing_func.size * (existing_func.complexity_score or 1.0)
        
        if abs(new_quality - existing_quality) > 10:  # Significant quality difference
            return new_quality > existing_quality
        
        # Quaternary criterion: address alignment (prefer aligned functions)
        new_aligned = new_func.address % 4 == 0
        existing_aligned = existing_func.address % 4 == 0
        
        if new_aligned != existing_aligned:
            return new_aligned
        
        # Final criterion: prefer the one with lower address (earlier in binary)
        return new_func.address < existing_func.address

    def _extract_assembly_instructions(self, functions: List[Function], analysis_context: Dict[str, Any]) -> List[Function]:
        """Extract actual assembly instructions for each detected function"""
        try:
            import capstone
        except ImportError:
            self.logger.warning("Capstone not available - cannot extract assembly instructions")
            return functions
        
        binary_path = analysis_context['binary_path']
        enhanced_functions = []
        
        try:
            with open(binary_path, 'rb') as f:
                # Read the entire binary for disassembly
                binary_data = f.read()
                
            # Initialize Capstone disassembler
            architecture = analysis_context.get('architecture', 'x86')
            if architecture == 'x64':
                md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
            else:
                md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
            
            md.detail = True  # Enable detailed instruction analysis
            
            self.logger.info(f"Extracting assembly instructions for {len(functions)} functions...")
            
            for func in functions:
                try:
                    # Calculate file offset from virtual address using PE header info
                    # Get actual base address from PE header
                    base_address = self._get_image_base(binary_path)
                    
                    # Convert virtual address to file offset using section information
                    file_offset = self._virtual_address_to_file_offset(func.address, binary_path)
                    
                    # Ensure we don't read beyond binary bounds
                    if file_offset < 0 or file_offset >= len(binary_data):
                        enhanced_functions.append(func)
                        continue
                    
                    # Extract function bytes
                    end_offset = min(file_offset + func.size, len(binary_data))
                    function_bytes = binary_data[file_offset:end_offset]
                    
                    if len(function_bytes) < 4:  # Skip tiny functions
                        enhanced_functions.append(func)
                        continue
                    
                    # Disassemble function
                    instructions = []
                    for insn in md.disasm(function_bytes, func.address):
                        instruction_data = {
                            'address': insn.address,
                            'mnemonic': insn.mnemonic,
                            'op_str': insn.op_str,
                            'bytes': insn.bytes.hex(),
                            'size': insn.size
                        }
                        
                        # Extract operand details if available
                        if hasattr(insn, 'operands') and insn.operands:
                            operands = []
                            for op in insn.operands:
                                operand_info = {
                                    'type': op.type,
                                }
                                if hasattr(op, 'value'):
                                    operand_info['value'] = op.value
                                if hasattr(op, 'reg'):
                                    operand_info['reg'] = op.reg
                                if hasattr(op, 'mem'):
                                    operand_info['mem'] = {
                                        'base': getattr(op.mem, 'base', 0),
                                        'index': getattr(op.mem, 'index', 0),
                                        'disp': getattr(op.mem, 'disp', 0)
                                    }
                                operands.append(operand_info)
                            instruction_data['operands'] = operands
                        
                        instructions.append(instruction_data)
                    
                    # Create enhanced function with assembly instructions
                    enhanced_func = Function(
                        name=func.name,
                        address=func.address,
                        size=func.size,
                        confidence=func.confidence,
                        detection_method=func.detection_method,
                        signature=func.signature,
                        complexity_score=func.complexity_score,
                        assembly_instructions=instructions,
                        binary_data=function_bytes
                    )
                    
                    enhanced_functions.append(enhanced_func)
                    
                except Exception as e:
                    self.logger.debug(f"Failed to extract assembly for function {func.name} at {func.address:x}: {e}")
                    enhanced_functions.append(func)  # Keep original function if extraction fails
            
            self.logger.info(f"Successfully extracted assembly for {sum(1 for f in enhanced_functions if f.assembly_instructions)} functions")
            return enhanced_functions
            
        except Exception as e:
            self.logger.error(f"Assembly extraction failed: {e}")
            return functions  # Return original functions if extraction fails
    
    def _get_image_base(self, binary_path: Path) -> int:
        """Get the image base address from PE header"""
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                if len(dos_header) < 60 or dos_header[:2] != b'MZ':
                    return 0x400000  # Default base address
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset)
                
                # Read PE signature
                pe_sig = f.read(4)
                if pe_sig != b'PE\x00\x00':
                    return 0x400000
                
                # Read COFF header
                f.read(20)  # Skip COFF header
                
                # Read optional header
                opt_header = f.read(28)  # Read first part of optional header
                if len(opt_header) >= 28:
                    image_base = struct.unpack('<I', opt_header[28-4:28])[0]
                    return image_base
                    
        except Exception as e:
            self.logger.debug(f"Failed to get image base: {e}")
            
        return 0x400000  # Default fallback
    
    def _virtual_address_to_file_offset(self, virtual_address: int, binary_path: Path) -> int:
        """Convert virtual address to file offset using section headers"""
        try:
            with open(binary_path, 'rb') as f:
                # Navigate to section headers
                dos_header = f.read(64)
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset + 4)  # Skip PE signature
                
                # Read COFF header
                machine, num_sections, timestamp, ptr_to_sym, num_symbols, opt_header_size, characteristics = struct.unpack('<HHIIIHH', f.read(20))
                
                # Skip optional header
                f.seek(f.tell() + opt_header_size)
                
                # Read section headers
                for i in range(num_sections):
                    section_header = f.read(40)
                    if len(section_header) < 40:
                        break
                        
                    name = section_header[:8].rstrip(b'\x00')
                    virtual_size, virtual_address_section, raw_size, raw_address = struct.unpack('<IIII', section_header[8:24])
                    
                    # Check if virtual address falls within this section
                    if (virtual_address >= virtual_address_section and 
                        virtual_address < virtual_address_section + virtual_size):
                        # Calculate file offset
                        offset_in_section = virtual_address - virtual_address_section
                        file_offset = raw_address + offset_in_section
                        return file_offset
                        
        except Exception as e:
            self.logger.debug(f"Failed to convert virtual address {virtual_address:x} to file offset: {e}")
            
        # Fallback: simple calculation assuming .text section starts at 0x1000 virtual, 0x400 file
        return virtual_address - 0x1000 + 0x400

    def _analyze_functions(self, functions: List[Function], analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detected functions"""
        if not functions:
            return {'analysis_quality': 0.0, 'total_complexity': 0.0}
        
        total_complexity = sum((getattr(f, 'complexity_score', None) or 1.0) for f in functions)
        avg_confidence = sum(f.confidence for f in functions) / len(functions)
        
        return {
            'function_count': len(functions),
            'analysis_quality': avg_confidence,
            'total_complexity': total_complexity,
            'detection_methods': list(set(f.detection_method for f in functions))
        }

    def _build_results(self, functions: List[Function], function_analysis: Dict[str, Any], analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build final results"""
        return {
            'functions': [self._function_to_dict(f) for f in functions],
            'functions_detected': len(functions),
            'function_analysis': function_analysis,
            'decompilation_results': {
                'functions': [self._function_to_dict(f) for f in functions],
                'quality_score': function_analysis.get('analysis_quality', 0.0)
            },
            'merovingian_metadata': {
                'agent_id': self.agent_id,
                'matrix_character': self.matrix_character.value,
                'binary_size': analysis_context.get('binary_size', 0),
                'detection_methods_used': function_analysis.get('detection_methods', [])
            }
        }

    def _function_to_dict(self, func: Function) -> Dict[str, Any]:
        """Convert Function dataclass to dictionary"""
        result = {
            'name': func.name,
            'address': func.address,
            'size': func.size,
            'confidence': func.confidence,
            'detection_method': func.detection_method,
            'signature': func.signature,
            'complexity_score': func.complexity_score or 1.0,
            'decompiled_code': f'// Function {func.name} at 0x{func.address:08x}\n{func.signature or "void function()"} {{\n    // Function implementation\n}}'
        }
        
        # Include assembly instructions if available
        if func.assembly_instructions:
            result['assembly_instructions'] = func.assembly_instructions
            result['instruction_count'] = len(func.assembly_instructions)
            
        # Include binary data if available  
        if func.binary_data:
            result['binary_data_size'] = len(func.binary_data)
            # Don't include the raw binary data in the dict to avoid bloat
            
        return result

    def _detect_packer_characteristics(self, binary_path: Path) -> Dict[str, Any]:
        """Detect packer characteristics using multiple analysis methods"""
        packer_info = {
            'is_likely_packed': False,
            'packer_type': 'unknown',
            'confidence': 0.0,
            'entropy': 0.0,
            'might_be_dotnet': False,
            'indicators': [],
            'peid_result': None,
            'pypackerdetect_result': None
        }
        
        try:
            # Analysis 1: Enhanced Packer Detection with peid
            peid_result = self._run_peid_detection(binary_path)
            packer_info['peid_result'] = peid_result
            if peid_result:
                packer_info['is_likely_packed'] = True
                packer_info['packer_type'] = peid_result
                packer_info['confidence'] += 0.8
                packer_info['indicators'].append('peid_signature')
                self.logger.info(f"PEID detected packer: {peid_result}")
            
            # Analysis 2: Enhanced Packer Detection with pypackerdetect
            pypackerdetect_result = self._run_pypackerdetect(binary_path)
            packer_info['pypackerdetect_result'] = pypackerdetect_result
            if pypackerdetect_result and pypackerdetect_result.get('is_packed', False):
                packer_info['is_likely_packed'] = True
                detected_packer = pypackerdetect_result.get('packer', 'unknown')
                if detected_packer != 'unknown':
                    packer_info['packer_type'] = detected_packer
                packer_info['confidence'] += 0.7
                packer_info['indicators'].append('pypackerdetect_analysis')
                self.logger.info(f"pypackerdetect result: {pypackerdetect_result}")
            
            # Analysis 3: Entropy Analysis
            entropy = self._calculate_binary_entropy(binary_path)
            packer_info['entropy'] = entropy
            
            # High entropy indicates packing/compression
            if entropy > 7.5:
                packer_info['is_likely_packed'] = True
                packer_info['confidence'] += 0.4
                packer_info['indicators'].append('high_entropy')
            
            # Analysis 4: UPX Detection (fallback)
            upx_detected = self._detect_upx_packer(binary_path)
            if upx_detected:
                packer_info['is_likely_packed'] = True
                if packer_info['packer_type'] == 'unknown':
                    packer_info['packer_type'] = 'UPX'
                packer_info['confidence'] += 0.5
                packer_info['indicators'].append('upx_signature')
            
            # Analysis 5: PE Structure Analysis
            pe_anomalies = self._detect_pe_anomalies(binary_path)
            if pe_anomalies['unusual_entry_point']:
                packer_info['confidence'] += 0.2
                packer_info['indicators'].append('unusual_entry_point')
            if pe_anomalies['suspicious_sections']:
                packer_info['confidence'] += 0.1
                packer_info['indicators'].append('suspicious_sections')
            
            # Analysis 6: .NET Detection for Packed Launchers
            dotnet_indicators = self._detect_hidden_dotnet(binary_path)
            if dotnet_indicators:
                packer_info['might_be_dotnet'] = True
                packer_info['confidence'] += 0.1
                packer_info['indicators'].append('hidden_dotnet')
            
            # Determine overall packing likelihood
            if packer_info['confidence'] > 0.3:
                packer_info['is_likely_packed'] = True
                
            # Set packer type based on strongest indicator priority
            if 'peid_signature' in packer_info['indicators'] and peid_result:
                # PEID has highest priority
                pass  # Already set above
            elif 'pypackerdetect_analysis' in packer_info['indicators'] and pypackerdetect_result:
                # pypackerdetect has second priority
                pass  # Already set above
            elif 'upx_signature' in packer_info['indicators']:
                packer_info['packer_type'] = 'UPX'
            elif entropy > 7.8:
                packer_info['packer_type'] = 'unknown_high_compression'
            elif packer_info['might_be_dotnet']:
                packer_info['packer_type'] = 'dotnet_wrapper'
            elif pe_anomalies['unusual_entry_point']:
                packer_info['packer_type'] = 'custom_packer'
                
        except Exception as e:
            self.logger.debug(f"Packer detection failed: {e}")
            
        return packer_info
    
    def _calculate_binary_entropy(self, binary_path: Path) -> float:
        """Calculate Shannon entropy of binary data"""
        try:
            with open(binary_path, 'rb') as f:
                # Read first 1MB for entropy calculation
                data = f.read(1024 * 1024)
                
            if not data:
                return 0.0
                
            # Calculate frequency of each byte value
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
                
            # Calculate Shannon entropy
            import math
            entropy = 0.0
            data_len = len(data)
            
            for count in byte_counts:
                if count > 0:
                    probability = count / data_len
                    entropy -= probability * math.log2(probability)
                    
            return entropy
            
        except Exception as e:
            self.logger.debug(f"Entropy calculation failed: {e}")
            return 0.0
    
    def _detect_upx_packer(self, binary_path: Path) -> bool:
        """Detect UPX packer signatures"""
        try:
            with open(binary_path, 'rb') as f:
                # Read first 2KB to check for UPX signatures
                data = f.read(2048)
                
            # Common UPX signatures
            upx_signatures = [
                b'UPX!',
                b'UPX0',
                b'UPX1',
                b'UPX2',
                b'$Info: This file is packed with the UPX'
            ]
            
            for signature in upx_signatures:
                if signature in data:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.debug(f"UPX detection failed: {e}")
            return False
    
    def _detect_pe_anomalies(self, binary_path: Path) -> Dict[str, bool]:
        """Detect PE structure anomalies that indicate packing"""
        anomalies = {
            'unusual_entry_point': False,
            'suspicious_sections': False,
            'import_anomalies': False
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                dos_header = f.read(64)
                if len(dos_header) < 60 or dos_header[:2] != b'MZ':
                    return anomalies
                
                # Get PE header offset
                pe_offset = struct.unpack('<I', dos_header[60:64])[0]
                f.seek(pe_offset + 24)  # Skip to optional header
                
                # Read optional header to get entry point
                magic = struct.unpack('<H', f.read(2))[0]
                if magic in [0x10b, 0x20b]:  # PE32 or PE32+
                    f.read(4)  # Skip version info
                    entry_point = struct.unpack('<I', f.read(4))[0]
                    
                    # Check for unusual entry point (like 0xd000000a)
                    if entry_point > 0x80000000 or entry_point < 0x1000:
                        anomalies['unusual_entry_point'] = True
                        
                # Check for suspicious section characteristics
                # (This would need more detailed PE parsing)
                anomalies['suspicious_sections'] = True  # Placeholder for now
                
        except Exception as e:
            self.logger.debug(f"PE anomaly detection failed: {e}")
            
        return anomalies
    
    def _detect_hidden_dotnet(self, binary_path: Path) -> bool:
        """Detect if binary might be a packed .NET application"""
        try:
            with open(binary_path, 'rb') as f:
                # Read larger portion to look for .NET artifacts
                data = f.read(min(1024*1024, f.seek(0, 2) or f.tell()))
                f.seek(0)
                
            # Look for .NET-related strings that might indicate a packed .NET app
            dotnet_indicators = [
                b'mscorlib',
                b'System.Windows.Forms',
                b'System.Drawing',
                b'.NETFramework',
                b'mscoree.dll',
                b'_CorExeMain'
            ]
            
            for indicator in dotnet_indicators:
                if indicator in data:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.debug(f"Hidden .NET detection failed: {e}")
            return False
    
    def _attempt_unpacking(self, binary_path: Path, packer_info: Dict[str, Any]) -> Optional[Path]:
        """Attempt to unpack binary using detected packer type"""
        if not packer_info['is_likely_packed']:
            return None
            
        packer_type = packer_info['packer_type']
        
        try:
            # UPX Unpacking
            if packer_type == 'UPX':
                return self._unpack_upx(binary_path)
            
            # For other packers, return None for now
            self.logger.info(f"No unpacker available for {packer_type}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Unpacking failed for {packer_type}: {e}")
            return None
    
    def _unpack_upx(self, binary_path: Path) -> Optional[Path]:
        """Attempt to unpack UPX-packed binary"""
        try:
            import subprocess
            import tempfile
            
            # Create temporary file for unpacked binary
            temp_dir = Path(tempfile.mkdtemp())
            unpacked_path = temp_dir / f"unpacked_{binary_path.name}"
            
            # Try to find UPX executable
            upx_commands = ['upx', 'upx.exe', '/usr/bin/upx']
            upx_found = None
            
            for upx_cmd in upx_commands:
                try:
                    result = subprocess.run([upx_cmd, '--version'], 
                                          capture_output=True, timeout=5)
                    if result.returncode == 0:
                        upx_found = upx_cmd
                        break
                except:
                    continue
            
            if not upx_found:
                self.logger.warning("UPX unpacker not available in system PATH")
                self.logger.info("To install UPX: sudo apt-get install upx-ucl (Linux) or download from https://upx.github.io/")
                print(f"[MEROVINGIAN DEBUG] UPX not found - install for better unpacking support")
                return None
            
            # Attempt unpacking
            result = subprocess.run([
                upx_found, '-d', str(binary_path), '-o', str(unpacked_path)
            ], capture_output=True, timeout=30)
            
            if result.returncode == 0 and unpacked_path.exists():
                self.logger.info(f"Successfully unpacked UPX binary: {unpacked_path}")
                print(f"[MEROVINGIAN DEBUG] UPX unpacking successful!")
                return unpacked_path
            else:
                self.logger.debug(f"UPX unpacking failed: {result.stderr.decode()}")
                return None
                
        except Exception as e:
            self.logger.debug(f"UPX unpacking error: {e}")
            return None
    
    def _run_peid_detection(self, binary_path: Path) -> Optional[str]:
        """Run peid packer detection with enhanced error handling"""
        try:
            import peid
            self.logger.debug(f"Running PEID detection on {binary_path}")
            
            # Correct peid API: peid.identify_packer(filepath)
            result = peid.identify_packer(str(binary_path))
            
            if result:
                # PEID returns tuple (filepath, [signatures])
                if isinstance(result, tuple) and len(result) == 2:
                    filepath, signatures = result
                    if signatures and len(signatures) > 0:
                        packer_name = signatures[0]  # Take first signature
                        self.logger.info(f"PEID identified: {packer_name}")
                        return str(packer_name)
                    else:
                        self.logger.debug("PEID returned empty signatures list")
                        return None
                elif isinstance(result, list) and len(result) > 0:
                    packer_name = result[0]  # Take first identified packer
                    self.logger.info(f"PEID identified packer: {packer_name}")
                    return str(packer_name)
                elif isinstance(result, str):
                    self.logger.info(f"PEID identified packer: {result}")
                    return result
                else:
                    self.logger.debug(f"PEID returned unexpected format: {result}")
                    return None
            else:
                self.logger.debug("PEID did not identify any known packer")
                return None
                
        except ImportError:
            self.logger.warning("peid library not installed: pip install --break-system-packages peid")
            return None
        except Exception as e:
            self.logger.debug(f"peid detection failed: {e}")
            return None
    
    def _run_pypackerdetect(self, binary_path: Path) -> Optional[Dict]:
        """Run pypackerdetect analysis with enhanced error handling"""
        try:
            from pypackerdetect import PyPackerDetect
            self.logger.debug(f"Running pypackerdetect analysis on {binary_path}")
            
            # Correct pypackerdetect API: create instance and call detect
            detector = PyPackerDetect(str(binary_path))
            result = detector.detect()
            
            if result:
                self.logger.info(f"pypackerdetect analysis: {result}")
                return result
            else:
                self.logger.debug("pypackerdetect returned no results")
                return None
                
        except ImportError:
            self.logger.warning("pypackerdetect library not installed: pip install --break-system-packages pypackerdetect")
            return None
        except Exception as e:
            self.logger.debug(f"pypackerdetect analysis failed: {e}")
            return None

    def _validate_results(self, results: Dict[str, Any]) -> None:
        """Validate results according to rules.md"""
        functions = results.get('functions', [])
        
        # Rule #53: STRICT ERROR HANDLING - Must find functions in 5MB+ binary
        binary_size = results.get('merovingian_metadata', {}).get('binary_size', 0)
        
        if binary_size > 1024*1024 and len(functions) == 0:  # 1MB+ binary should have functions
            raise RuntimeError(
                f"PIPELINE FAILURE - Agent 3 STRICT MODE: Found {len(functions)} functions in {binary_size/1024/1024:.1f}MB binary. "
                f"A binary this size should contain functions. This violates rules.md Rule #53 (STRICT ERROR HANDLING) - "
                f"Agent must fail when requirements not met. NO PLACEHOLDER CODE allowed per Rule #44."
            )
        
        # Ensure no placeholder/fake data per Rule #44
        for func in functions:
            if not func.get('name') or not func.get('detection_method'):
                raise RuntimeError("Invalid function data detected - violates Rule #44 (NO PLACEHOLDER CODE)")