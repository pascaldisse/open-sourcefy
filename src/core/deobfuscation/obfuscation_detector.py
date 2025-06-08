"""
Advanced Obfuscation Detection and Analysis Module

Implements sophisticated obfuscation detection for:
- Control flow flattening detection and reversal
- Virtual machine obfuscation (VMProtect, Themida)
- Anti-analysis techniques and evasion detection
- Polymorphic and metamorphic code detection
- Original Entry Point (OEP) identification

Part of Phase 1: Foundational Analysis and Deobfuscation
"""

import logging
import struct
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib

try:
    import capstone
    import pefile
    CAPSTONE_AVAILABLE = True
    PEFILE_AVAILABLE = True
except ImportError:
    CAPSTONE_AVAILABLE = False
    PEFILE_AVAILABLE = False
    logging.warning("capstone or pefile not available, using fallback detection")

from .entropy_analyzer import EntropyAnalyzer
from .cfg_reconstructor import AdvancedControlFlowAnalyzer


class ObfuscationType(Enum):
    """Types of obfuscation techniques."""
    CONTROL_FLOW_FLATTENING = "control_flow_flattening"
    VIRTUAL_MACHINE = "virtual_machine"
    ANTI_DEBUG = "anti_debug"
    ANTI_VM = "anti_vm"
    POLYMORPHIC = "polymorphic"
    METAMORPHIC = "metamorphic"
    API_HASHING = "api_hashing"
    STRING_ENCRYPTION = "string_encryption"
    JUNK_CODE = "junk_code"
    OPAQUE_PREDICATES = "opaque_predicates"


@dataclass
class ObfuscationIndicator:
    """Indicator of obfuscation technique."""
    obfuscation_type: ObfuscationType
    confidence: float
    description: str
    evidence: List[str]
    locations: List[int]
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class OEPAnalysisResult:
    """Original Entry Point analysis result."""
    found_oep: bool
    oep_address: int
    confidence: float
    detection_method: str
    analysis_steps: List[str]
    alternative_oeps: List[int]


@dataclass
class ObfuscationAnalysisResult:
    """Complete obfuscation analysis result."""
    obfuscated: bool
    obfuscation_level: str  # 'none', 'light', 'moderate', 'heavy', 'extreme'
    indicators: List[ObfuscationIndicator]
    oep_analysis: OEPAnalysisResult
    deobfuscation_recommendations: List[str]
    estimated_difficulty: str  # 'trivial', 'easy', 'medium', 'hard', 'expert'
    anti_analysis_techniques: List[str]
    metadata: Dict[str, Any]


class ObfuscationDetector:
    """
    Advanced obfuscation detection using multiple analysis techniques.
    
    Combines static analysis, pattern recognition, and heuristics
    to detect various obfuscation and anti-analysis techniques.
    """
    
    def __init__(self, config_manager=None):
        """Initialize obfuscation detector."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.entropy_analyzer = EntropyAnalyzer(config_manager)
        self.cfg_analyzer = AdvancedControlFlowAnalyzer(config_manager)
        
        # Initialize disassembler
        if CAPSTONE_AVAILABLE:
            self.disasm_x86 = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
            self.disasm_x64 = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
            self.disasm_x86.detail = True
            self.disasm_x64.detail = True
        
        # Anti-analysis API patterns
        self.anti_debug_apis = {
            'IsDebuggerPresent', 'CheckRemoteDebuggerPresent', 'NtQueryInformationProcess',
            'OutputDebugString', 'GetTickCount', 'QueryPerformanceCounter',
            'ZwSetInformationThread', 'NtSetInformationThread'
        }
        
        self.anti_vm_apis = {
            'cpuid', 'rdtsc', 'sidt', 'sgdt', 'sldt', 'smsw',
            'GetSystemInfo', 'GetModuleHandle', 'CreateToolhelp32Snapshot'
        }
    
    def analyze_obfuscation(self, binary_path: Path) -> ObfuscationAnalysisResult:
        """
        Perform comprehensive obfuscation analysis.
        
        Args:
            binary_path: Path to binary file to analyze
            
        Returns:
            ObfuscationAnalysisResult with detailed analysis
        """
        try:
            self.logger.info(f"Starting obfuscation analysis for {binary_path}")
            
            # Read binary data
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            # Initialize result
            indicators = []
            anti_analysis_techniques = []
            
            # Phase 1: Control flow flattening detection
            self.logger.debug("Phase 1: Control flow flattening analysis")
            cff_indicators = self._detect_control_flow_flattening(binary_data)
            indicators.extend(cff_indicators)
            
            # Phase 2: Virtual machine obfuscation detection
            self.logger.debug("Phase 2: Virtual machine obfuscation analysis")
            vm_indicators = self._detect_vm_obfuscation(binary_data)
            indicators.extend(vm_indicators)
            
            # Phase 3: Anti-analysis techniques
            self.logger.debug("Phase 3: Anti-analysis techniques")
            anti_analysis = self._detect_anti_analysis(binary_data)
            indicators.extend(anti_analysis['indicators'])
            anti_analysis_techniques.extend(anti_analysis['techniques'])
            
            # Phase 4: Polymorphic/metamorphic detection
            self.logger.debug("Phase 4: Polymorphic/metamorphic analysis")
            poly_indicators = self._detect_polymorphic_code(binary_data)
            indicators.extend(poly_indicators)
            
            # Phase 5: API obfuscation detection
            self.logger.debug("Phase 5: API obfuscation analysis")
            api_indicators = self._detect_api_obfuscation(binary_data)
            indicators.extend(api_indicators)
            
            # Phase 6: OEP analysis
            self.logger.debug("Phase 6: Original Entry Point analysis")
            oep_analysis = self._analyze_original_entry_point(binary_data)
            
            # Phase 7: Overall assessment
            obfuscation_level = self._assess_obfuscation_level(indicators)
            estimated_difficulty = self._estimate_deobfuscation_difficulty(indicators)
            recommendations = self._generate_deobfuscation_recommendations(indicators)
            
            result = ObfuscationAnalysisResult(
                obfuscated=len(indicators) > 0,
                obfuscation_level=obfuscation_level,
                indicators=indicators,
                oep_analysis=oep_analysis,
                deobfuscation_recommendations=recommendations,
                estimated_difficulty=estimated_difficulty,
                anti_analysis_techniques=anti_analysis_techniques,
                metadata={
                    'total_indicators': len(indicators),
                    'analysis_timestamp': self._get_timestamp(),
                    'binary_size': len(binary_data)
                }
            )
            
            self.logger.info(f"Obfuscation analysis complete: {obfuscation_level} obfuscation "
                           f"({len(indicators)} indicators)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in obfuscation analysis: {e}")
            return self._create_error_result(str(e))
    
    def _detect_control_flow_flattening(self, binary_data: bytes) -> List[ObfuscationIndicator]:
        """Detect control flow flattening obfuscation."""
        indicators = []
        
        try:
            # Analyze control flow structure
            cfg_result = self.cfg_analyzer.analyze_control_flow(binary_data)
            
            # Look for CFF patterns
            cff_patterns = self._identify_cff_patterns(cfg_result)
            
            if cff_patterns['switch_dispatcher_count'] > 2:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.CONTROL_FLOW_FLATTENING,
                    confidence=0.8,
                    description="Multiple switch-based dispatchers detected",
                    evidence=[f"Found {cff_patterns['switch_dispatcher_count']} switch dispatchers"],
                    locations=cff_patterns['dispatcher_locations'],
                    severity='high'
                ))
            
            if cff_patterns['indirect_jump_ratio'] > 0.3:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.CONTROL_FLOW_FLATTENING,
                    confidence=0.7,
                    description="High ratio of indirect jumps indicates CFF",
                    evidence=[f"Indirect jump ratio: {cff_patterns['indirect_jump_ratio']:.2f}"],
                    locations=[],
                    severity='medium'
                ))
            
            if cff_patterns['state_variable_updates'] > 10:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.CONTROL_FLOW_FLATTENING,
                    confidence=0.6,
                    description="Frequent state variable updates",
                    evidence=[f"State updates: {cff_patterns['state_variable_updates']}"],
                    locations=[],
                    severity='medium'
                ))
        
        except Exception as e:
            self.logger.debug(f"Error in CFF detection: {e}")
        
        return indicators
    
    def _identify_cff_patterns(self, cfg_result) -> Dict[str, Any]:
        """Identify control flow flattening patterns in CFG."""
        patterns = {
            'switch_dispatcher_count': 0,
            'dispatcher_locations': [],
            'indirect_jump_ratio': 0.0,
            'state_variable_updates': 0
        }
        
        try:
            total_jumps = 0
            indirect_jumps = 0
            
            # Analyze basic blocks for CFF patterns
            for addr, block in cfg_result.basic_blocks.items():
                # Count jump types
                for insn in block.instructions:
                    if insn['mnemonic'].lower() in ['jmp', 'call']:
                        total_jumps += 1
                        if 'ptr' in insn['op_str'] or '[' in insn['op_str']:
                            indirect_jumps += 1
                
                # Look for switch dispatcher patterns
                if self._is_switch_dispatcher(block):
                    patterns['switch_dispatcher_count'] += 1
                    patterns['dispatcher_locations'].append(addr)
                
                # Count state variable updates
                patterns['state_variable_updates'] += self._count_state_updates(block)
            
            # Calculate indirect jump ratio
            if total_jumps > 0:
                patterns['indirect_jump_ratio'] = indirect_jumps / total_jumps
        
        except Exception as e:
            self.logger.debug(f"Error identifying CFF patterns: {e}")
        
        return patterns
    
    def _is_switch_dispatcher(self, block) -> bool:
        """Check if basic block is a switch dispatcher."""
        # Look for patterns like: mov reg, [state_var]; jmp [table + reg*4]
        has_state_load = False
        has_computed_jump = False
        
        for insn in block.instructions:
            mnemonic = insn['mnemonic'].lower()
            op_str = insn['op_str']
            
            # Look for memory loads (state variable access)
            if mnemonic == 'mov' and '[' in op_str:
                has_state_load = True
            
            # Look for computed indirect jumps
            if mnemonic == 'jmp' and ('[' in op_str and '+' in op_str):
                has_computed_jump = True
        
        return has_state_load and has_computed_jump
    
    def _count_state_updates(self, block) -> int:
        """Count potential state variable updates in block."""
        state_updates = 0
        
        for insn in block.instructions:
            mnemonic = insn['mnemonic'].lower()
            op_str = insn['op_str']
            
            # Look for memory writes that could be state updates
            if mnemonic in ['mov', 'add', 'sub', 'xor'] and ',' in op_str:
                parts = op_str.split(',')
                if len(parts) == 2 and '[' in parts[0]:
                    state_updates += 1
        
        return state_updates
    
    def _detect_vm_obfuscation(self, binary_data: bytes) -> List[ObfuscationIndicator]:
        """Detect virtual machine obfuscation techniques."""
        indicators = []
        
        try:
            # Look for VM patterns
            vm_patterns = self._identify_vm_patterns(binary_data)
            
            if vm_patterns['vm_handler_count'] > 5:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.VIRTUAL_MACHINE,
                    confidence=0.85,
                    description="Multiple VM handlers detected",
                    evidence=[f"VM handlers: {vm_patterns['vm_handler_count']}"],
                    locations=vm_patterns['handler_locations'],
                    severity='critical'
                ))
            
            if vm_patterns['opcode_table_found']:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.VIRTUAL_MACHINE,
                    confidence=0.9,
                    description="VM opcode table detected",
                    evidence=["Bytecode interpretation table found"],
                    locations=[vm_patterns['opcode_table_location']],
                    severity='critical'
                ))
            
            if vm_patterns['context_switching'] > 3:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.VIRTUAL_MACHINE,
                    confidence=0.7,
                    description="VM context switching detected",
                    evidence=[f"Context switches: {vm_patterns['context_switching']}"],
                    locations=[],
                    severity='high'
                ))
        
        except Exception as e:
            self.logger.debug(f"Error in VM detection: {e}")
        
        return indicators
    
    def _identify_vm_patterns(self, binary_data: bytes) -> Dict[str, Any]:
        """Identify virtual machine obfuscation patterns."""
        patterns = {
            'vm_handler_count': 0,
            'handler_locations': [],
            'opcode_table_found': False,
            'opcode_table_location': 0,
            'context_switching': 0
        }
        
        try:
            if not CAPSTONE_AVAILABLE:
                return patterns
            
            # Disassemble and look for VM patterns
            disasm = self.disasm_x86
            
            # Look for handler dispatch patterns
            handler_pattern = re.compile(rb'\x8b[\x00-\xff]{1,10}\xff\x24\x85')  # mov + jmp [table + reg*4]
            handler_matches = list(handler_pattern.finditer(binary_data))
            
            patterns['vm_handler_count'] = len(handler_matches)
            patterns['handler_locations'] = [match.start() for match in handler_matches[:10]]
            
            # Look for opcode tables (sequences of function pointers)
            patterns['opcode_table_found'], patterns['opcode_table_location'] = self._find_opcode_table(binary_data)
            
            # Count context switching patterns
            patterns['context_switching'] = self._count_context_switches(binary_data)
        
        except Exception as e:
            self.logger.debug(f"Error identifying VM patterns: {e}")
        
        return patterns
    
    def _find_opcode_table(self, binary_data: bytes) -> Tuple[bool, int]:
        """Find VM opcode dispatch table."""
        try:
            # Look for arrays of function pointers
            for i in range(0, len(binary_data) - 32, 4):
                # Check for sequence of aligned pointers
                potential_table = []
                for j in range(8):  # Check 8 consecutive entries
                    if i + j * 4 + 4 > len(binary_data):
                        break
                    
                    ptr_value = struct.unpack('<L', binary_data[i + j * 4:i + j * 4 + 4])[0]
                    
                    # Check if this looks like a code pointer
                    if 0x400000 <= ptr_value <= 0x7fffffff:
                        potential_table.append(ptr_value)
                    else:
                        break
                
                # If we found a table of reasonable size
                if len(potential_table) >= 6:
                    return True, i
            
            return False, 0
            
        except:
            return False, 0
    
    def _count_context_switches(self, binary_data: bytes) -> int:
        """Count VM context switching patterns."""
        # Look for patterns that save/restore execution context
        context_patterns = [
            rb'\x60\x9c',  # pusha + pushf
            rb'\x9d\x61',  # popf + popa
            rb'\x50\x53\x51\x52',  # push eax,ebx,ecx,edx
        ]
        
        count = 0
        for pattern in context_patterns:
            count += binary_data.count(pattern)
        
        return count
    
    def _detect_anti_analysis(self, binary_data: bytes) -> Dict[str, Any]:
        """Detect anti-analysis and anti-debugging techniques."""
        indicators = []
        techniques = []
        
        try:
            # Check for anti-debug APIs
            anti_debug_found = self._detect_anti_debug_apis(binary_data)
            if anti_debug_found['count'] > 0:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.ANTI_DEBUG,
                    confidence=0.8,
                    description="Anti-debugging APIs detected",
                    evidence=anti_debug_found['apis'],
                    locations=[],
                    severity='high'
                ))
                techniques.extend(anti_debug_found['apis'])
            
            # Check for anti-VM techniques
            anti_vm_found = self._detect_anti_vm_techniques(binary_data)
            if anti_vm_found['count'] > 0:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.ANTI_VM,
                    confidence=0.7,
                    description="Anti-VM techniques detected",
                    evidence=anti_vm_found['techniques'],
                    locations=[],
                    severity='medium'
                ))
                techniques.extend(anti_vm_found['techniques'])
            
            # Check for timing checks
            timing_checks = self._detect_timing_checks(binary_data)
            if timing_checks > 0:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.ANTI_DEBUG,
                    confidence=0.6,
                    description="Timing-based anti-debugging",
                    evidence=[f"Timing checks: {timing_checks}"],
                    locations=[],
                    severity='medium'
                ))
                techniques.append("Timing checks")
        
        except Exception as e:
            self.logger.debug(f"Error in anti-analysis detection: {e}")
        
        return {
            'indicators': indicators,
            'techniques': techniques
        }
    
    def _detect_anti_debug_apis(self, binary_data: bytes) -> Dict[str, Any]:
        """Detect anti-debugging API usage."""
        found_apis = []
        
        for api in self.anti_debug_apis:
            if api.encode('ascii') in binary_data:
                found_apis.append(api)
        
        return {
            'count': len(found_apis),
            'apis': found_apis
        }
    
    def _detect_anti_vm_techniques(self, binary_data: bytes) -> Dict[str, Any]:
        """Detect anti-VM techniques."""
        found_techniques = []
        
        # Check for VM-detection strings
        vm_strings = [
            b'VMware', b'VirtualBox', b'QEMU', b'Xen', b'Hyper-V',
            b'vbox', b'vmware', b'qemu'
        ]
        
        for vm_string in vm_strings:
            if vm_string in binary_data:
                found_techniques.append(f"VM string: {vm_string.decode('ascii', errors='ignore')}")
        
        # Check for VM-specific instructions
        if b'\x0f\x01\xd0' in binary_data:  # XGETBV
            found_techniques.append("XGETBV instruction")
        
        if b'\x0f\xa2' in binary_data:  # CPUID
            found_techniques.append("CPUID instruction")
        
        return {
            'count': len(found_techniques),
            'techniques': found_techniques
        }
    
    def _detect_timing_checks(self, binary_data: bytes) -> int:
        """Detect timing-based anti-debugging."""
        timing_patterns = [
            b'\x0f\x31',  # RDTSC
            b'\x8b\x44\x24\x04',  # GetTickCount pattern
        ]
        
        count = 0
        for pattern in timing_patterns:
            count += binary_data.count(pattern)
        
        return count
    
    def _detect_polymorphic_code(self, binary_data: bytes) -> List[ObfuscationIndicator]:
        """Detect polymorphic and metamorphic code."""
        indicators = []
        
        try:
            # Look for code generation patterns
            code_gen_patterns = self._identify_code_generation(binary_data)
            
            if code_gen_patterns['self_modification'] > 0:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.POLYMORPHIC,
                    confidence=0.7,
                    description="Self-modifying code detected",
                    evidence=[f"Self-modifications: {code_gen_patterns['self_modification']}"],
                    locations=[],
                    severity='high'
                ))
            
            if code_gen_patterns['dynamic_allocation'] > 3:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.METAMORPHIC,
                    confidence=0.6,
                    description="Dynamic code allocation patterns",
                    evidence=[f"Dynamic allocations: {code_gen_patterns['dynamic_allocation']}"],
                    locations=[],
                    severity='medium'
                ))
        
        except Exception as e:
            self.logger.debug(f"Error in polymorphic detection: {e}")
        
        return indicators
    
    def _identify_code_generation(self, binary_data: bytes) -> Dict[str, int]:
        """Identify code generation patterns."""
        patterns = {
            'self_modification': 0,
            'dynamic_allocation': 0
        }
        
        # Count VirtualAlloc calls (dynamic allocation)
        if b'VirtualAlloc' in binary_data:
            patterns['dynamic_allocation'] += binary_data.count(b'VirtualAlloc')
        
        # Look for self-modification patterns (writing to executable memory)
        self_mod_patterns = [
            rb'\xc6\x05',  # mov byte ptr [addr], imm8
            rb'\xc7\x05',  # mov dword ptr [addr], imm32
        ]
        
        for pattern in self_mod_patterns:
            patterns['self_modification'] += binary_data.count(pattern)
        
        return patterns
    
    def _detect_api_obfuscation(self, binary_data: bytes) -> List[ObfuscationIndicator]:
        """Detect API obfuscation techniques."""
        indicators = []
        
        try:
            # Check for API hashing
            hash_patterns = self._detect_api_hashing(binary_data)
            if hash_patterns['hash_count'] > 0:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.API_HASHING,
                    confidence=0.8,
                    description="API hashing detected",
                    evidence=[f"Hash constants: {hash_patterns['hash_count']}"],
                    locations=hash_patterns['locations'],
                    severity='high'
                ))
            
            # Check for string encryption
            encrypted_strings = self._detect_string_encryption(binary_data)
            if encrypted_strings > 0:
                indicators.append(ObfuscationIndicator(
                    obfuscation_type=ObfuscationType.STRING_ENCRYPTION,
                    confidence=0.7,
                    description="String encryption detected",
                    evidence=[f"Encrypted strings: {encrypted_strings}"],
                    locations=[],
                    severity='medium'
                ))
        
        except Exception as e:
            self.logger.debug(f"Error in API obfuscation detection: {e}")
        
        return indicators
    
    def _detect_api_hashing(self, binary_data: bytes) -> Dict[str, Any]:
        """Detect API hashing patterns."""
        # Look for common hash constants used in API hashing
        common_hash_constants = [
            0x1505,      # ROR13 hash constant
            0x2b992dda,  # Common hash seed
            0x9e3779b9,  # Golden ratio constant
        ]
        
        hash_count = 0
        locations = []
        
        for i in range(len(binary_data) - 4):
            dword = struct.unpack('<L', binary_data[i:i+4])[0]
            if dword in common_hash_constants:
                hash_count += 1
                locations.append(i)
        
        return {
            'hash_count': hash_count,
            'locations': locations[:10]  # Limit to first 10
        }
    
    def _detect_string_encryption(self, binary_data: bytes) -> int:
        """Detect encrypted strings."""
        # Look for patterns indicating string decryption
        decrypt_patterns = [
            rb'\x30[\x00-\xff]\x40',  # xor [reg], byte; inc reg
            rb'\x80[\x30-\x37][\x00-\xff]',  # xor byte ptr [reg], imm8
        ]
        
        count = 0
        for pattern in decrypt_patterns:
            count += len(re.findall(pattern, binary_data))
        
        return count
    
    def _analyze_original_entry_point(self, binary_data: bytes) -> OEPAnalysisResult:
        """Analyze and attempt to find Original Entry Point (OEP)."""
        try:
            if not PEFILE_AVAILABLE:
                return OEPAnalysisResult(
                    found_oep=False,
                    oep_address=0,
                    confidence=0.0,
                    detection_method="PE analysis not available",
                    analysis_steps=[],
                    alternative_oeps=[]
                )
            
            pe = pefile.PE(data=binary_data, fast_load=True)
            current_ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            image_base = pe.OPTIONAL_HEADER.ImageBase
            
            analysis_steps = []
            alternative_oeps = []
            
            # Method 1: Check if current EP is in last section (packer indicator)
            last_section = pe.sections[-1]
            ep_in_last_section = (current_ep >= last_section.VirtualAddress and 
                                current_ep < last_section.VirtualAddress + last_section.Misc_VirtualSize)
            
            analysis_steps.append(f"Entry point in last section: {ep_in_last_section}")
            
            # Method 2: Look for jumps from current EP
            ep_offset = self._rva_to_offset(pe, current_ep)
            if ep_offset and ep_offset < len(binary_data) - 10:
                ep_code = binary_data[ep_offset:ep_offset + 50]
                jump_targets = self._find_jump_targets_from_ep(ep_code, image_base + current_ep)
                alternative_oeps.extend(jump_targets)
                analysis_steps.append(f"Found {len(jump_targets)} jump targets from EP")
            
            # Method 3: Look for function prologue patterns
            prologue_candidates = self._find_function_prologues(binary_data, pe)
            alternative_oeps.extend(prologue_candidates[:5])  # Limit to top 5
            analysis_steps.append(f"Found {len(prologue_candidates)} prologue candidates")
            
            # Method 4: Entropy analysis of sections
            text_section = self._find_text_section(pe)
            if text_section:
                text_data = text_section.get_data()
                text_entropy = self.entropy_analyzer.calculate_shannon_entropy(text_data)
                analysis_steps.append(f"Text section entropy: {text_entropy:.2f}")
                
                # Low entropy in text section might indicate unpacked code
                if text_entropy < 6.0:
                    confidence = 0.8
                    method = "Low entropy text section suggests unpacked code"
                else:
                    confidence = 0.3
                    method = "High entropy suggests packed code"
            else:
                confidence = 0.1
                method = "No text section found"
            
            # Determine best OEP candidate
            if alternative_oeps and ep_in_last_section:
                # If EP is in last section and we have alternatives, use best alternative
                best_oep = alternative_oeps[0]
                confidence = 0.7
                method = "Alternative OEP from jump analysis"
            else:
                # Use current EP
                best_oep = image_base + current_ep
                if not ep_in_last_section:
                    confidence = 0.9
                    method = "Current EP appears to be original"
                else:
                    confidence = 0.4
                    method = "Current EP suspicious but no alternatives found"
            
            return OEPAnalysisResult(
                found_oep=confidence > 0.5,
                oep_address=best_oep,
                confidence=confidence,
                detection_method=method,
                analysis_steps=analysis_steps,
                alternative_oeps=alternative_oeps
            )
            
        except Exception as e:
            self.logger.error(f"Error in OEP analysis: {e}")
            return OEPAnalysisResult(
                found_oep=False,
                oep_address=0,
                confidence=0.0,
                detection_method=f"Error: {e}",
                analysis_steps=[],
                alternative_oeps=[]
            )
    
    def _rva_to_offset(self, pe, rva: int) -> Optional[int]:
        """Convert RVA to file offset."""
        try:
            for section in pe.sections:
                if (rva >= section.VirtualAddress and 
                    rva < section.VirtualAddress + section.Misc_VirtualSize):
                    return rva - section.VirtualAddress + section.PointerToRawData
            return None
        except:
            return None
    
    def _find_jump_targets_from_ep(self, code: bytes, ep_va: int) -> List[int]:
        """Find jump targets from entry point code."""
        targets = []
        
        if not CAPSTONE_AVAILABLE:
            return targets
        
        try:
            disasm = self.disasm_x86
            
            for insn in disasm.disasm(code, ep_va):
                if insn.mnemonic in ['jmp', 'call']:
                    # Look for direct jumps/calls
                    if len(insn.operands) > 0 and insn.operands[0].type == capstone.CS_OP_IMM:
                        target = insn.operands[0].imm
                        if 0x400000 <= target <= 0x7fffffff:  # Valid address range
                            targets.append(target)
                
                # Stop after a few instructions
                if len(targets) > 10 or insn.address - ep_va > 100:
                    break
        
        except:
            pass
        
        return targets
    
    def _find_function_prologues(self, binary_data: bytes, pe) -> List[int]:
        """Find function prologue patterns that might indicate OEP."""
        candidates = []
        
        # Common function prologue patterns
        prologue_patterns = [
            rb'\x55\x8b\xec',           # push ebp; mov ebp, esp
            rb'\x55\x89\xe5',           # push ebp; mov ebp, esp
            rb'\x6a[\x00-\xff]\x68',    # push imm8; push imm32 (common in main)
            rb'\x68[\x00-\xff]{4}\x6a', # push imm32; push imm8
        ]
        
        for pattern in prologue_patterns:
            for match in re.finditer(pattern, binary_data):
                offset = match.start()
                
                # Convert to VA
                va = self._offset_to_va(pe, offset)
                if va:
                    candidates.append(va)
        
        return candidates[:20]  # Return top 20 candidates
    
    def _offset_to_va(self, pe, offset: int) -> Optional[int]:
        """Convert file offset to virtual address."""
        try:
            for section in pe.sections:
                if (offset >= section.PointerToRawData and 
                    offset < section.PointerToRawData + section.SizeOfRawData):
                    rva = offset - section.PointerToRawData + section.VirtualAddress
                    return pe.OPTIONAL_HEADER.ImageBase + rva
            return None
        except:
            return None
    
    def _find_text_section(self, pe):
        """Find the main text/code section."""
        for section in pe.sections:
            section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
            if section_name.lower() in ['.text', 'code', '.code']:
                return section
        
        # Fallback: find executable section
        for section in pe.sections:
            if section.Characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                return section
        
        return None
    
    def _assess_obfuscation_level(self, indicators: List[ObfuscationIndicator]) -> str:
        """Assess overall obfuscation level."""
        if not indicators:
            return 'none'
        
        total_confidence = sum(indicator.confidence for indicator in indicators)
        critical_count = sum(1 for indicator in indicators if indicator.severity == 'critical')
        high_count = sum(1 for indicator in indicators if indicator.severity == 'high')
        
        if critical_count > 0 or total_confidence > 2.5:
            return 'extreme'
        elif high_count > 2 or total_confidence > 1.5:
            return 'heavy'
        elif high_count > 0 or total_confidence > 0.8:
            return 'moderate'
        elif total_confidence > 0.3:
            return 'light'
        else:
            return 'none'
    
    def _estimate_deobfuscation_difficulty(self, indicators: List[ObfuscationIndicator]) -> str:
        """Estimate difficulty of deobfuscation."""
        if not indicators:
            return 'trivial'
        
        vm_indicators = [i for i in indicators if i.obfuscation_type == ObfuscationType.VIRTUAL_MACHINE]
        cff_indicators = [i for i in indicators if i.obfuscation_type == ObfuscationType.CONTROL_FLOW_FLATTENING]
        anti_analysis = [i for i in indicators if i.obfuscation_type in [ObfuscationType.ANTI_DEBUG, ObfuscationType.ANTI_VM]]
        
        if vm_indicators and any(i.severity == 'critical' for i in vm_indicators):
            return 'expert'
        elif cff_indicators and anti_analysis:
            return 'hard'
        elif cff_indicators or (len(indicators) > 3):
            return 'medium'
        elif len(indicators) > 1:
            return 'easy'
        else:
            return 'trivial'
    
    def _generate_deobfuscation_recommendations(self, indicators: List[ObfuscationIndicator]) -> List[str]:
        """Generate recommendations for deobfuscation."""
        recommendations = []
        
        obfuscation_types = {indicator.obfuscation_type for indicator in indicators}
        
        if ObfuscationType.VIRTUAL_MACHINE in obfuscation_types:
            recommendations.append("Use specialized VM analysis tools (VMAttack, VMP analyzer)")
            recommendations.append("Consider dynamic analysis with VM stepping")
        
        if ObfuscationType.CONTROL_FLOW_FLATTENING in obfuscation_types:
            recommendations.append("Apply control flow deflattening techniques")
            recommendations.append("Use symbolic execution for state variable tracking")
        
        if ObfuscationType.ANTI_DEBUG in obfuscation_types:
            recommendations.append("Patch anti-debugging checks")
            recommendations.append("Use stealth debugging techniques")
        
        if ObfuscationType.API_HASHING in obfuscation_types:
            recommendations.append("Reconstruct API names from hash values")
            recommendations.append("Hook API resolution routines")
        
        if ObfuscationType.STRING_ENCRYPTION in obfuscation_types:
            recommendations.append("Identify and reverse string decryption routines")
            recommendations.append("Use dynamic analysis to capture decrypted strings")
        
        if not recommendations:
            recommendations.append("Apply standard deobfuscation techniques")
            recommendations.append("Use both static and dynamic analysis approaches")
        
        return recommendations
    
    def _create_error_result(self, error_msg: str) -> ObfuscationAnalysisResult:
        """Create error result for failed analysis."""
        return ObfuscationAnalysisResult(
            obfuscated=False,
            obfuscation_level='unknown',
            indicators=[],
            oep_analysis=OEPAnalysisResult(
                found_oep=False,
                oep_address=0,
                confidence=0.0,
                detection_method=f"Error: {error_msg}",
                analysis_steps=[],
                alternative_oeps=[]
            ),
            deobfuscation_recommendations=[],
            estimated_difficulty='unknown',
            anti_analysis_techniques=[],
            metadata={'error': error_msg}
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()


class AntiAnalysisDetector:
    """
    Specialized detector for anti-analysis techniques.
    
    Focus on detecting and circumventing anti-debugging,
    anti-VM, and other analysis evasion techniques.
    """
    
    def __init__(self, config_manager=None):
        """Initialize anti-analysis detector."""
        self.obfuscation_detector = ObfuscationDetector(config_manager)
        self.logger = logging.getLogger(__name__)
    
    def detect_anti_analysis(self, binary_path: Path) -> Dict[str, Any]:
        """
        Detect anti-analysis techniques in binary.
        
        Args:
            binary_path: Path to binary file
            
        Returns:
            Dictionary with anti-analysis detection results
        """
        result = self.obfuscation_detector.analyze_obfuscation(binary_path)
        
        # Extract anti-analysis specific information
        anti_analysis_indicators = [
            indicator for indicator in result.indicators
            if indicator.obfuscation_type in [
                ObfuscationType.ANTI_DEBUG,
                ObfuscationType.ANTI_VM
            ]
        ]
        
        return {
            'anti_analysis_detected': len(anti_analysis_indicators) > 0,
            'techniques': result.anti_analysis_techniques,
            'indicators': anti_analysis_indicators,
            'bypass_recommendations': self._generate_bypass_recommendations(anti_analysis_indicators)
        }
    
    def _generate_bypass_recommendations(self, indicators: List[ObfuscationIndicator]) -> List[str]:
        """Generate recommendations for bypassing anti-analysis."""
        recommendations = []
        
        for indicator in indicators:
            if indicator.obfuscation_type == ObfuscationType.ANTI_DEBUG:
                recommendations.extend([
                    "Use ScyllaHide or similar anti-anti-debug plugin",
                    "Patch IsDebuggerPresent and related APIs",
                    "Use kernel-mode debugging",
                    "Modify PEB flags to hide debugger presence"
                ])
            
            elif indicator.obfuscation_type == ObfuscationType.ANTI_VM:
                recommendations.extend([
                    "Use bare metal analysis environment",
                    "Modify VM artifacts (registry, files, processes)",
                    "Use nested virtualization",
                    "Patch CPUID and other detection methods"
                ])
        
        return list(set(recommendations))  # Remove duplicates


# Factory functions
def create_obfuscation_detector(config_manager=None) -> ObfuscationDetector:
    """Create configured obfuscation detector instance."""
    return ObfuscationDetector(config_manager)


def create_anti_analysis_detector(config_manager=None) -> AntiAnalysisDetector:
    """Create configured anti-analysis detector instance."""
    return AntiAnalysisDetector(config_manager)


# Example usage
if __name__ == "__main__":
    # Example usage
    detector = ObfuscationDetector()
    
    print("Obfuscation detection module loaded successfully")
    print("Available detection types:")
    for obf_type in ObfuscationType:
        print(f"  - {obf_type.value}")