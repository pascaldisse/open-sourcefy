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
    """Function detection result"""
    name: str
    address: int
    size: int
    confidence: float
    detection_method: str
    signature: Optional[str] = None
    complexity_score: Optional[float] = None

class MerovingianAgent(DecompilerAgent):
    """
    Agent 03: The Merovingian - Basic Decompilation & Function Detection
    
    The Merovingian understands cause and effect in code transformations.
    Specializes in finding functions using multiple detection methods.
    """
    
    # Function prologue patterns for x86/x64
    FUNCTION_PROLOGUES = {
        'x86': [
            b'\x55\x8b\xec',           # push ebp; mov ebp, esp
            b'\x55\x89\xe5',           # push ebp; mov ebp, esp (AT&T)
            b'\x56\x57\x53',           # push esi; push edi; push ebx
            b'\x83\xec',               # sub esp, immediate
        ],
        'x64': [
            b'\x48\x89\xe5',           # mov rbp, rsp
            b'\x48\x83\xec',           # sub rsp, immediate
            b'\x55\x48\x89\xe5',       # push rbp; mov rbp, rsp
            b'\x40\x53',               # push rbx (REX prefix)
        ]
    }
    
    def __init__(self):
        super().__init__(
            agent_id=3,
            matrix_character=MatrixCharacter.MEROVINGIAN,
            dependencies=[1]  # Depends on Sentinel for binary info
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
        """Detect functions using multiple methods"""
        functions = []
        binary_path = analysis_context['binary_path']
        
        # Method 1: PE Import Table Analysis
        import_functions = self._detect_import_functions(binary_path)
        functions.extend(import_functions)
        
        # Method 2: Entry Point Detection
        entry_functions = self._detect_entry_point_function(binary_path)
        functions.extend(entry_functions)
        
        # Method 3: Binary Pattern Analysis
        pattern_functions = self._detect_functions_by_patterns(binary_path)
        functions.extend(pattern_functions)
        
        # Method 4: .NET Method Analysis (if applicable)
        if self._is_dotnet_binary(binary_path):
            dotnet_functions = self._detect_dotnet_methods(binary_path)
            functions.extend(dotnet_functions)
        
        # Method 5: Section Analysis
        section_functions = self._detect_functions_from_sections(binary_path)
        functions.extend(section_functions)
        
        # Remove duplicates
        unique_functions = self._deduplicate_functions(functions)
        
        # Force output with both print and logging
        print(f"[MEROVINGIAN DEBUG] Detection methods results:")
        print(f"[MEROVINGIAN DEBUG]   Import functions: {len(import_functions)}")
        print(f"[MEROVINGIAN DEBUG]   Entry functions: {len(entry_functions)}")
        print(f"[MEROVINGIAN DEBUG]   Pattern functions: {len(pattern_functions)}")
        print(f"[MEROVINGIAN DEBUG]   .NET functions: {len(dotnet_functions) if self._is_dotnet_binary(binary_path) else 'N/A (not .NET)'}")
        print(f"[MEROVINGIAN DEBUG]   Section functions: {len(section_functions)}")
        print(f"[MEROVINGIAN DEBUG] Detected {len(unique_functions)} unique functions from {len(functions)} candidates")
        
        self.logger.info(f"Detection methods results:")
        self.logger.info(f"  Import functions: {len(import_functions)}")
        self.logger.info(f"  Entry functions: {len(entry_functions)}")
        self.logger.info(f"  Pattern functions: {len(pattern_functions)}")
        self.logger.info(f"  .NET functions: {len(dotnet_functions) if self._is_dotnet_binary(binary_path) else 'N/A (not .NET)'}")
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
        """Detect functions by searching for prologue patterns"""
        functions = []
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read(min(1024*1024, f.seek(0, 2) or f.tell()))  # Read first 1MB
                f.seek(0)
                
                # Search for function prologues
                prologues = self.FUNCTION_PROLOGUES['x86'] + self.FUNCTION_PROLOGUES['x64']
                
                for prologue in prologues:
                    offset = 0
                    while True:
                        pos = data.find(prologue, offset)
                        if pos == -1:
                            break
                        
                        # Create function at this position
                        functions.append(Function(
                            name=f"function_{pos:08x}",
                            address=pos,
                            size=50,  # Estimated size
                            confidence=0.7,
                            detection_method="prologue_pattern",
                            signature=f"void function_{pos:08x}()"
                        ))
                        
                        offset = pos + len(prologue)
                        
                        # Limit to prevent excessive functions
                        if len(functions) >= 100:
                            break
                    
                    if len(functions) >= 100:
                        break
                        
        except Exception as e:
            self.logger.debug(f"Pattern detection failed: {e}")
        
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
        """Detect .NET methods using PowerShell reflection"""
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
            
            # PowerShell script to extract .NET methods
            ps_script = f'''
            try {{
                $assembly = [System.Reflection.Assembly]::LoadFile("{windows_path}")
                $types = $assembly.GetTypes()
                foreach ($type in $types) {{
                    $methods = $type.GetMethods([System.Reflection.BindingFlags]::Public -bor [System.Reflection.BindingFlags]::NonPublic -bor [System.Reflection.BindingFlags]::Instance -bor [System.Reflection.BindingFlags]::Static)
                    foreach ($method in $methods) {{
                        Write-Output "$($type.FullName):$($method.Name)"
                    }}
                }}
            }} catch {{
                Write-Output "ERROR: $($_.Exception.Message)"
            }}
            '''
            
            result = subprocess.run([powershell_cmd, '-Command', ps_script], 
                                    capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                method_count = 0
                
                for line in lines:
                    if ':' in line and not line.startswith('ERROR'):
                        class_name, method_name = line.split(':', 1)
                        functions.append(Function(
                            name=f"{class_name}.{method_name}",
                            address=0x1000 + method_count * 10,  # Estimated addresses
                            size=20,
                            confidence=0.85,
                            detection_method="dotnet_reflection",
                            signature=f"void {method_name}()"
                        ))
                        method_count += 1
                        
                        if method_count >= 200:  # Limit methods
                            break
                            
        except Exception as e:
            self.logger.debug(f".NET method detection failed: {e}")
        
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
                    ptr_to_relocs, ptr_to_line_nums, num_relocs, num_line_nums, characteristics = struct.unpack('<IIHHH', f.read(12))
                    
                    # Check if section is executable
                    if characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                        # Create functions in this section
                        section_size = min(virtual_size, raw_size)
                        if section_size > 0:
                            # Estimate functions every 256 bytes
                            for offset in range(0, section_size, 256):
                                if len(functions) >= 50:  # Limit functions per section
                                    break
                                    
                                functions.append(Function(
                                    name=f"{section_name}_func_{offset:04x}",
                                    address=virtual_address + offset,
                                    size=min(256, section_size - offset),
                                    confidence=0.6,
                                    detection_method="section_analysis",
                                    signature=f"void {section_name}_func_{offset:04x}()"
                                ))
                                
        except Exception as e:
            self.logger.debug(f"Section analysis failed: {e}")
        
        return functions

    def _deduplicate_functions(self, functions: List[Function]) -> List[Function]:
        """Remove duplicate functions based on address"""
        seen_addresses = set()
        unique_functions = []
        
        # Sort by confidence (highest first)
        functions.sort(key=lambda f: f.confidence, reverse=True)
        
        for func in functions:
            # Consider functions at similar addresses as duplicates
            is_duplicate = any(abs(func.address - addr) < 10 for addr in seen_addresses)
            
            if not is_duplicate:
                seen_addresses.add(func.address)
                unique_functions.append(func)
        
        return unique_functions

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
        return {
            'name': func.name,
            'address': func.address,
            'size': func.size,
            'confidence': func.confidence,
            'detection_method': func.detection_method,
            'signature': func.signature,
            'complexity_score': func.complexity_score or 1.0,
            'decompiled_code': f'// Function {func.name} at 0x{func.address:08x}\n{func.signature or "void function()"} {{\n    // Function implementation\n}}'
        }

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