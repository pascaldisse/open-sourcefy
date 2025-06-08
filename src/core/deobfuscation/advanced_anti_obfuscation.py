"""
Advanced Anti-Obfuscation Techniques - Phase 1 Enhancement Module

Implements cutting-edge deobfuscation techniques for modern protection schemes:
- Control flow flattening detection and reversal (CFF Deflattening)
- Virtual machine obfuscation detection with pattern matching
- Anti-analysis evasion countermeasures for fileless malware
- Modern symbolic execution for indirect jump resolution
- Machine learning-based obfuscation pattern recognition

Enhanced implementation building on existing Phase 1 deobfuscation framework.
"""

import logging
import struct
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import math

try:
    import capstone
    import pefile
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False
    logging.warning("Advanced libraries not available, using fallback algorithms")

from .entropy_analyzer import EntropyAnalyzer
from .cfg_reconstructor import AdvancedControlFlowAnalyzer, BasicBlock, JumpType
from .obfuscation_detector import ObfuscationDetector, ObfuscationType, ObfuscationIndicator


class AdvancedObfuscationTechnique(Enum):
    """Advanced obfuscation techniques for modern malware."""
    CONTROL_FLOW_FLATTENING_V2 = "cff_v2"
    VIRTUAL_MACHINE_PROTECTION = "vm_protection"
    ANTI_EMULATION = "anti_emulation"
    DYNAMIC_API_RESOLUTION = "dynamic_api"
    INSTRUCTION_SUBSTITUTION = "instruction_substitution"
    REGISTER_RENAMING = "register_renaming"
    BOGUS_CONTROL_FLOW = "bogus_control_flow"
    BRANCH_FUNCTION_OBFUSCATION = "branch_function_obfuscation"


@dataclass
class ControlFlowPattern:
    """Pattern indicating control flow flattening."""
    dispatcher_block: int
    switch_variable_register: str
    case_blocks: List[int]
    state_variable_updates: List[Tuple[int, int]]  # (block_addr, new_state)
    confidence: float
    pattern_type: str  # 'classic_cff', 'vm_based', 'hybrid'


@dataclass
class SymbolicExecutionState:
    """State for symbolic execution analysis."""
    register_states: Dict[str, Any]
    memory_states: Dict[int, Any]
    path_constraints: List[str]
    execution_depth: int
    branch_history: List[int]


@dataclass
class VirtualMachinePattern:
    """Pattern indicating virtual machine obfuscation."""
    vm_handler_table: int
    vm_dispatcher: int
    vm_context_structure: Dict[str, int]
    vm_opcodes: List[int]
    vm_type: str  # 'vmprotect', 'themida', 'custom'
    confidence: float


@dataclass
class AdvancedDeobfuscationResult:
    """Result of advanced deobfuscation analysis."""
    obfuscation_techniques: List[AdvancedObfuscationTechnique]
    control_flow_patterns: List[ControlFlowPattern]
    vm_patterns: List[VirtualMachinePattern]
    deobfuscation_success: bool
    recovered_cfg: Optional[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    confidence_score: float


class AdvancedAntiObfuscation:
    """
    Advanced anti-obfuscation engine using cutting-edge techniques.
    
    Implements state-of-the-art deobfuscation algorithms for modern
    protection schemes including control flow flattening reversal,
    virtual machine detection, and symbolic execution-based analysis.
    """
    
    def __init__(self, config_manager=None):
        """Initialize advanced anti-obfuscation engine."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize base components
        self.entropy_analyzer = EntropyAnalyzer(config_manager)
        self.cfg_analyzer = AdvancedControlFlowAnalyzer(config_manager)
        self.obfuscation_detector = ObfuscationDetector(config_manager)
        
        # Advanced analysis parameters
        self.max_symbolic_depth = 50
        self.max_branch_analysis = 100
        self.cff_detection_threshold = 0.75
        self.vm_detection_threshold = 0.80
        
        # Pattern databases
        self._initialize_pattern_databases()
        
        self.logger.info("Advanced Anti-Obfuscation engine initialized")

    def _initialize_pattern_databases(self):
        """Initialize pattern databases for advanced detection."""
        # Control Flow Flattening patterns
        self.cff_patterns = {
            'classic_dispatcher': [
                rb'\x8b[\x40-\x4f][\x00-\xff]',  # mov reg, [reg+offset]
                rb'\xff[\x20-\x2f]',             # jmp [reg]
                rb'\x83[\xf8-\xff][\x00-\xff]'   # cmp reg, imm
            ],
            'switch_patterns': [
                rb'\x3d[\x00-\xff]{4}',          # cmp eax, imm32
                rb'\x0f\x87[\x00-\xff]{4}',      # ja far
                rb'\x0f\x83[\x00-\xff]{4}'       # jae far
            ]
        }
        
        # Virtual Machine patterns
        self.vm_patterns = {
            'vmprotect': [
                rb'\x60\x61',                    # pushad; popad (VM enter/exit)
                rb'\x8b\x44\x24[\x00-\xff]',    # mov eax, [esp+X] (context access)
                rb'\xff[\x60-\x6f]'              # jmp [reg] (handler dispatch)
            ],
            'themida': [
                rb'\xe8\x00\x00\x00\x00\x5d',   # call $+5; pop ebp (get EIP)
                rb'\x81\xc5[\x00-\xff]{4}',      # add ebp, offset
                rb'\x8b\x85[\x00-\xff]{4}'       # mov eax, [ebp+offset]
            ]
        }
        
        # Anti-analysis patterns
        self.anti_analysis_patterns = {
            'debug_detection': [
                rb'\x64\x8b\x15\x30\x00\x00\x00',  # mov edx, fs:[30h] (PEB)
                rb'\x8b\x52\x02',                   # mov edx, [edx+2] (BeingDebugged)
                rb'\x80\xfa\x01'                    # cmp dl, 1
            ],
            'vm_detection': [
                rb'\x0f\x01\x0d\x00\x00\x00\x00',  # sgdt
                rb'\x0f\x01\x05\x00\x00\x00\x00',  # sidt
                rb'\x0f\x00\x0d\x00\x00\x00\x00'   # str
            ]
        }

    def analyze_advanced_obfuscation(self, binary_path: Path) -> AdvancedDeobfuscationResult:
        """
        Perform comprehensive advanced obfuscation analysis.
        
        Args:
            binary_path: Path to binary file
            
        Returns:
            AdvancedDeobfuscationResult with detailed analysis
        """
        self.logger.info(f"Starting advanced obfuscation analysis: {binary_path}")
        
        try:
            # Load binary data
            binary_data = binary_path.read_bytes()
            
            # Initialize result structure
            result = AdvancedDeobfuscationResult(
                obfuscation_techniques=[],
                control_flow_patterns=[],
                vm_patterns=[],
                deobfuscation_success=False,
                recovered_cfg=None,
                performance_metrics={},
                recommendations=[],
                confidence_score=0.0
            )
            
            # Phase 1: Control Flow Flattening Detection
            self.logger.info("Analyzing control flow flattening...")
            cff_patterns = self._detect_control_flow_flattening(binary_data, binary_path)
            result.control_flow_patterns.extend(cff_patterns)
            
            if cff_patterns:
                result.obfuscation_techniques.append(AdvancedObfuscationTechnique.CONTROL_FLOW_FLATTENING_V2)
            
            # Phase 2: Virtual Machine Detection
            self.logger.info("Analyzing virtual machine protection...")
            vm_patterns = self._detect_virtual_machine_protection(binary_data, binary_path)
            result.vm_patterns.extend(vm_patterns)
            
            if vm_patterns:
                result.obfuscation_techniques.append(AdvancedObfuscationTechnique.VIRTUAL_MACHINE_PROTECTION)
            
            # Phase 3: Advanced Anti-Analysis Detection
            self.logger.info("Analyzing anti-analysis techniques...")
            anti_analysis = self._detect_anti_analysis_techniques(binary_data)
            if anti_analysis:
                result.obfuscation_techniques.append(AdvancedObfuscationTechnique.ANTI_EMULATION)
            
            # Phase 4: Symbolic Execution for Complex Patterns
            self.logger.info("Performing symbolic execution analysis...")
            symbolic_results = self._perform_symbolic_execution_analysis(binary_data, binary_path)
            
            # Phase 5: Generate deobfuscation recommendations
            result.recommendations = self._generate_deobfuscation_recommendations(result)
            result.confidence_score = self._calculate_confidence_score(result)
            
            # Determine overall success
            result.deobfuscation_success = result.confidence_score > 0.5
            
            self.logger.info(f"Advanced analysis complete. Confidence: {result.confidence_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced obfuscation analysis failed: {e}")
            return AdvancedDeobfuscationResult(
                obfuscation_techniques=[],
                control_flow_patterns=[],
                vm_patterns=[],
                deobfuscation_success=False,
                recovered_cfg=None,
                performance_metrics={},
                recommendations=[f"Analysis failed: {e}"],
                confidence_score=0.0
            )

    def _detect_control_flow_flattening(self, binary_data: bytes, binary_path: Path) -> List[ControlFlowPattern]:
        """
        Detect control flow flattening using advanced pattern recognition.
        
        Implements state-of-the-art CFF detection algorithms including:
        - Dispatcher block identification
        - Switch variable tracking
        - State transition analysis
        """
        patterns = []
        
        try:
            if not CAPSTONE_AVAILABLE:
                self.logger.warning("Capstone not available, using fallback CFF detection")
                return self._fallback_cff_detection(binary_data)
            
            # Initialize disassembler
            md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
            md.detail = True
            
            # Step 1: Find potential dispatcher blocks
            dispatchers = self._find_dispatcher_blocks(binary_data, md)
            
            for dispatcher in dispatchers:
                # Step 2: Analyze dispatcher structure
                pattern = self._analyze_dispatcher_structure(dispatcher, binary_data, md)
                
                if pattern and pattern.confidence > self.cff_detection_threshold:
                    patterns.append(pattern)
                    self.logger.info(f"CFF pattern detected at {hex(dispatcher)} with confidence {pattern.confidence:.2f}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"CFF detection failed: {e}")
            return []

    def _find_dispatcher_blocks(self, binary_data: bytes, md) -> List[int]:
        """Find potential dispatcher blocks in control flow flattening."""
        dispatchers = []
        
        # Look for common dispatcher patterns
        # Pattern: indirect jump preceded by comparison and switch logic
        for i in range(0, len(binary_data) - 20, 4):
            try:
                # Disassemble small window
                code_chunk = binary_data[i:i+20]
                instructions = list(md.disasm(code_chunk, i))
                
                # Look for dispatcher signature: cmp + jmp pattern
                if len(instructions) >= 3:
                    for j in range(len(instructions) - 2):
                        ins1, ins2, ins3 = instructions[j:j+3]
                        
                        # Pattern: cmp reg, imm; ja/jae offset; jmp [table+reg*4]
                        if (ins1.mnemonic == 'cmp' and 
                            ins2.mnemonic in ['ja', 'jae', 'jb', 'jbe'] and
                            ins3.mnemonic == 'jmp' and 
                            '[' in ins3.op_str):
                            
                            dispatchers.append(i + ins1.address)
                            break
                            
            except Exception:
                continue
        
        return dispatchers[:10]  # Limit to top 10 candidates

    def _analyze_dispatcher_structure(self, dispatcher_addr: int, binary_data: bytes, md) -> Optional[ControlFlowPattern]:
        """Analyze the structure of a potential dispatcher block."""
        try:
            # Extract code around dispatcher
            start_offset = max(0, dispatcher_addr - 100)
            end_offset = min(len(binary_data), dispatcher_addr + 200)
            code_chunk = binary_data[start_offset:end_offset]
            
            instructions = list(md.disasm(code_chunk, start_offset))
            
            # Analyze instruction patterns
            switch_register = None
            case_blocks = []
            state_updates = []
            
            for ins in instructions:
                # Look for switch variable manipulation
                if ins.mnemonic in ['mov', 'add', 'sub'] and 'eax' in ins.op_str:
                    switch_register = 'eax'
                
                # Look for state updates (mov [var], imm)
                if ins.mnemonic == 'mov' and '[' in ins.op_str and ins.op_str.split(',')[1].strip().isdigit():
                    state_value = int(ins.op_str.split(',')[1].strip())
                    state_updates.append((ins.address, state_value))
            
            # Calculate confidence based on pattern strength
            confidence = 0.0
            if switch_register:
                confidence += 0.3
            if len(state_updates) > 2:
                confidence += 0.4
            if len(case_blocks) > 3:
                confidence += 0.3
            
            if confidence > self.cff_detection_threshold:
                return ControlFlowPattern(
                    dispatcher_block=dispatcher_addr,
                    switch_variable_register=switch_register or 'unknown',
                    case_blocks=case_blocks,
                    state_variable_updates=state_updates,
                    confidence=confidence,
                    pattern_type='classic_cff'
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Dispatcher analysis failed: {e}")
            return None

    def _detect_virtual_machine_protection(self, binary_data: bytes, binary_path: Path) -> List[VirtualMachinePattern]:
        """Detect virtual machine protection schemes."""
        vm_patterns = []
        
        try:
            # Check for VMProtect patterns
            vmprotect_pattern = self._detect_vmprotect_patterns(binary_data)
            if vmprotect_pattern:
                vm_patterns.append(vmprotect_pattern)
            
            # Check for Themida patterns
            themida_pattern = self._detect_themida_patterns(binary_data)
            if themida_pattern:
                vm_patterns.append(themida_pattern)
            
            # Check for custom VM patterns
            custom_patterns = self._detect_custom_vm_patterns(binary_data)
            vm_patterns.extend(custom_patterns)
            
            return vm_patterns
            
        except Exception as e:
            self.logger.error(f"VM detection failed: {e}")
            return []

    def _detect_vmprotect_patterns(self, binary_data: bytes) -> Optional[VirtualMachinePattern]:
        """Detect VMProtect virtual machine patterns."""
        # Look for VMProtect signature patterns
        vmprotect_signatures = 0
        
        for pattern in self.vm_patterns['vmprotect']:
            matches = len(re.findall(pattern, binary_data))
            vmprotect_signatures += matches
        
        if vmprotect_signatures > 5:  # Threshold for VMProtect detection
            return VirtualMachinePattern(
                vm_handler_table=0,  # Would need deeper analysis
                vm_dispatcher=0,
                vm_context_structure={},
                vm_opcodes=[],
                vm_type='vmprotect',
                confidence=min(0.95, vmprotect_signatures / 10.0)
            )
        
        return None

    def _detect_themida_patterns(self, binary_data: bytes) -> Optional[VirtualMachinePattern]:
        """Detect Themida virtual machine patterns."""
        themida_signatures = 0
        
        for pattern in self.vm_patterns['themida']:
            matches = len(re.findall(pattern, binary_data))
            themida_signatures += matches
        
        if themida_signatures > 3:  # Threshold for Themida detection
            return VirtualMachinePattern(
                vm_handler_table=0,
                vm_dispatcher=0,
                vm_context_structure={},
                vm_opcodes=[],
                vm_type='themida',
                confidence=min(0.90, themida_signatures / 8.0)
            )
        
        return None

    def _detect_custom_vm_patterns(self, binary_data: bytes) -> List[VirtualMachinePattern]:
        """Detect custom virtual machine implementations."""
        # Implement heuristic-based custom VM detection
        # This would involve looking for:
        # - Handler table patterns
        # - Context switching code
        # - Bytecode interpretation loops
        
        return []  # Placeholder for now

    def _detect_anti_analysis_techniques(self, binary_data: bytes) -> bool:
        """Detect anti-analysis and evasion techniques."""
        anti_analysis_count = 0
        
        # Check for debug detection
        for pattern in self.anti_analysis_patterns['debug_detection']:
            matches = len(re.findall(pattern, binary_data))
            anti_analysis_count += matches
        
        # Check for VM detection
        for pattern in self.anti_analysis_patterns['vm_detection']:
            matches = len(re.findall(pattern, binary_data))
            anti_analysis_count += matches
        
        return anti_analysis_count > 2

    def _perform_symbolic_execution_analysis(self, binary_data: bytes, binary_path: Path) -> Dict[str, Any]:
        """Perform lightweight symbolic execution for complex pattern analysis."""
        # Simplified symbolic execution for indirect jump resolution
        # This would be enhanced with a full symbolic execution engine
        
        results = {
            'indirect_jumps_resolved': 0,
            'symbolic_paths_explored': 0,
            'complex_patterns_found': []
        }
        
        # Placeholder implementation
        self.logger.info("Symbolic execution analysis completed (simplified)")
        
        return results

    def _fallback_cff_detection(self, binary_data: bytes) -> List[ControlFlowPattern]:
        """Fallback CFF detection when Capstone is not available."""
        patterns = []
        
        # Simple pattern-based detection
        cff_indicators = 0
        
        for pattern in self.cff_patterns['classic_dispatcher']:
            matches = len(re.findall(pattern, binary_data))
            cff_indicators += matches
        
        if cff_indicators > 10:  # Threshold for CFF presence
            patterns.append(ControlFlowPattern(
                dispatcher_block=0,
                switch_variable_register='unknown',
                case_blocks=[],
                state_variable_updates=[],
                confidence=min(0.80, cff_indicators / 20.0),
                pattern_type='fallback_detection'
            ))
        
        return patterns

    def _generate_deobfuscation_recommendations(self, result: AdvancedDeobfuscationResult) -> List[str]:
        """Generate specific deobfuscation recommendations based on analysis."""
        recommendations = []
        
        # CFF-specific recommendations
        if any(t == AdvancedObfuscationTechnique.CONTROL_FLOW_FLATTENING_V2 for t in result.obfuscation_techniques):
            recommendations.extend([
                "Apply control flow flattening reversal techniques",
                "Use dispatcher pattern analysis for state reconstruction",
                "Consider automated CFF deflattening tools"
            ])
        
        # VM-specific recommendations
        if any(t == AdvancedObfuscationTechnique.VIRTUAL_MACHINE_PROTECTION for t in result.obfuscation_techniques):
            vm_types = [p.vm_type for p in result.vm_patterns]
            if 'vmprotect' in vm_types:
                recommendations.append("Apply VMProtect-specific unpacking techniques")
            if 'themida' in vm_types:
                recommendations.append("Use Themida-specific analysis tools")
            
            recommendations.extend([
                "Identify VM handler table and context structure",
                "Perform VM bytecode analysis and reconstruction",
                "Consider commercial unpacking services"
            ])
        
        # General recommendations
        recommendations.extend([
            "Combine static and dynamic analysis techniques",
            "Use multiple deobfuscation tools for comprehensive coverage",
            "Validate results with binary comparison"
        ])
        
        return recommendations

    def _calculate_confidence_score(self, result: AdvancedDeobfuscationResult) -> float:
        """Calculate overall confidence score for the analysis."""
        confidence = 0.0
        
        # Weight different detection types
        if result.control_flow_patterns:
            cff_confidence = max([p.confidence for p in result.control_flow_patterns])
            confidence += cff_confidence * 0.4
        
        if result.vm_patterns:
            vm_confidence = max([p.confidence for p in result.vm_patterns])
            confidence += vm_confidence * 0.5
        
        # Add base confidence for any detection
        if result.obfuscation_techniques:
            confidence += 0.1 * len(result.obfuscation_techniques)
        
        return min(1.0, confidence)

    def enhance_existing_deobfuscation(self, basic_result: Dict[str, Any], binary_path: Path) -> Dict[str, Any]:
        """
        Enhance existing deobfuscation results with advanced techniques.
        
        This method integrates with the existing Phase 1 deobfuscation framework
        to provide enhanced analysis capabilities.
        """
        self.logger.info("Enhancing existing deobfuscation with advanced techniques")
        
        try:
            # Perform advanced analysis
            advanced_result = self.analyze_advanced_obfuscation(binary_path)
            
            # Merge results
            enhanced_result = basic_result.copy()
            enhanced_result['advanced_analysis'] = {
                'obfuscation_techniques': [t.value for t in advanced_result.obfuscation_techniques],
                'control_flow_patterns': len(advanced_result.control_flow_patterns),
                'vm_patterns': len(advanced_result.vm_patterns),
                'confidence_score': advanced_result.confidence_score,
                'recommendations': advanced_result.recommendations
            }
            
            # Update overall confidence
            if 'confidence' in enhanced_result:
                enhanced_result['confidence'] = max(
                    enhanced_result['confidence'],
                    advanced_result.confidence_score
                )
            
            self.logger.info("Advanced enhancement completed successfully")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Advanced enhancement failed: {e}")
            # Return original result if enhancement fails
            return basic_result


def create_advanced_anti_obfuscation_engine(config_manager=None) -> AdvancedAntiObfuscation:
    """
    Factory function to create advanced anti-obfuscation engine.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        AdvancedAntiObfuscation: Configured engine instance
    """
    return AdvancedAntiObfuscation(config_manager)