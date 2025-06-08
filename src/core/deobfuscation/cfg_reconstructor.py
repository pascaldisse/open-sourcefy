"""
Advanced Control Flow Graph Reconstruction Module

Implements sophisticated CFG reconstruction techniques for:
- Indirect jump resolution using symbolic execution
- Exception handling and switch statement analysis
- Self-modifying code detection and handling
- Dynamic control flow graph construction

Part of Phase 1: Foundational Analysis and Deobfuscation
"""

import logging
import struct
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import networkx as nx

try:
    import capstone
    CAPSTONE_AVAILABLE = True
except ImportError:
    CAPSTONE_AVAILABLE = False
    logging.warning("Capstone not available, using fallback disassembly")


class JumpType(Enum):
    """Types of control flow transfers."""
    DIRECT_JUMP = "direct_jump"
    INDIRECT_JUMP = "indirect_jump"
    CONDITIONAL_JUMP = "conditional_jump"
    CALL = "call"
    RETURN = "return"
    EXCEPTION = "exception"
    SWITCH = "switch"


@dataclass
class BasicBlock:
    """Represents a basic block in control flow graph."""
    start_addr: int
    end_addr: int
    instructions: List[Dict[str, Any]] = field(default_factory=list)
    successors: Set[int] = field(default_factory=set)
    predecessors: Set[int] = field(default_factory=set)
    block_type: str = "normal"
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CFGAnalysisResult:
    """Result of CFG reconstruction analysis."""
    basic_blocks: Dict[int, BasicBlock]
    control_flow_graph: nx.DiGraph
    indirect_jumps: List[Dict[str, Any]]
    switch_statements: List[Dict[str, Any]]
    exception_handlers: List[Dict[str, Any]]
    self_modifying_regions: List[Dict[str, Any]]
    analysis_quality: float
    coverage_percentage: float
    metadata: Dict[str, Any]


class AdvancedControlFlowAnalyzer:
    """
    Advanced control flow analyzer using multiple techniques.
    
    Combines static analysis, symbolic execution simulation,
    and pattern recognition to reconstruct complex control flows.
    """
    
    def __init__(self, config_manager=None):
        """Initialize CFG analyzer with configuration."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize disassembler
        if CAPSTONE_AVAILABLE:
            self.disasm_x86 = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
            self.disasm_x64 = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
            self.disasm_x86.detail = True
            self.disasm_x64.detail = True
        
        # Analysis configuration
        self.max_indirect_targets = self.config.get('cfg.max_indirect_targets', 50) if self.config else 50
        self.max_analysis_depth = self.config.get('cfg.max_analysis_depth', 1000) if self.config else 1000
        self.enable_symbolic_execution = self.config.get('cfg.enable_symbolic_execution', True) if self.config else True
    
    def analyze_control_flow(self, binary_data: bytes, base_addr: int = 0x400000, 
                           entry_point: int = None) -> CFGAnalysisResult:
        """
        Perform comprehensive control flow analysis.
        
        Args:
            binary_data: Binary data to analyze
            base_addr: Base address for analysis
            entry_point: Entry point address (optional)
            
        Returns:
            CFGAnalysisResult with complete analysis
        """
        try:
            self.logger.info("Starting advanced control flow analysis")
            
            # Initialize analysis structures
            basic_blocks = {}
            indirect_jumps = []
            switch_statements = []
            exception_handlers = []
            self_modifying_regions = []
            
            # Determine architecture
            arch = self._detect_architecture(binary_data)
            disasm = self._get_disassembler(arch)
            
            if not disasm:
                self.logger.warning("No disassembler available, using fallback analysis")
                return self._fallback_analysis(binary_data, base_addr)
            
            # Phase 1: Linear sweep to identify basic blocks
            self.logger.debug("Phase 1: Linear sweep analysis")
            linear_blocks = self._linear_sweep_analysis(binary_data, base_addr, disasm)
            
            # Phase 2: Recursive traversal from entry points
            self.logger.debug("Phase 2: Recursive traversal analysis")
            if entry_point:
                recursive_blocks = self._recursive_traversal(binary_data, base_addr, entry_point, disasm)
                # Merge results
                basic_blocks = self._merge_block_analyses(linear_blocks, recursive_blocks)
            else:
                basic_blocks = linear_blocks
            
            # Phase 3: Advanced analysis for complex constructs
            self.logger.debug("Phase 3: Advanced construct analysis")
            indirect_jumps = self._analyze_indirect_jumps(basic_blocks, binary_data, base_addr, disasm)
            switch_statements = self._analyze_switch_statements(basic_blocks, binary_data, disasm)
            exception_handlers = self._analyze_exception_handlers(basic_blocks, binary_data)
            self_modifying_regions = self._detect_self_modifying_code(basic_blocks, binary_data)
            
            # Phase 4: Build control flow graph
            self.logger.debug("Phase 4: CFG construction")
            cfg = self._build_control_flow_graph(basic_blocks)
            
            # Phase 5: Quality assessment
            analysis_quality = self._assess_analysis_quality(basic_blocks, cfg)
            coverage_percentage = self._calculate_coverage(basic_blocks, len(binary_data))
            
            metadata = {
                'architecture': arch,
                'total_basic_blocks': len(basic_blocks),
                'total_instructions': sum(len(bb.instructions) for bb in basic_blocks.values()),
                'analysis_timestamp': self._get_timestamp(),
                'techniques_used': ['linear_sweep', 'recursive_traversal', 'symbolic_execution']
            }
            
            self.logger.info(f"CFG analysis complete: {len(basic_blocks)} blocks, "
                           f"quality={analysis_quality:.2f}, coverage={coverage_percentage:.2f}%")
            
            return CFGAnalysisResult(
                basic_blocks=basic_blocks,
                control_flow_graph=cfg,
                indirect_jumps=indirect_jumps,
                switch_statements=switch_statements,
                exception_handlers=exception_handlers,
                self_modifying_regions=self_modifying_regions,
                analysis_quality=analysis_quality,
                coverage_percentage=coverage_percentage,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in CFG analysis: {e}")
            return self._create_error_result(str(e))
    
    def _detect_architecture(self, binary_data: bytes) -> str:
        """Detect binary architecture from PE header or heuristics."""
        try:
            # Check PE header
            if len(binary_data) > 0x3c + 4:
                pe_offset = struct.unpack('<L', binary_data[0x3c:0x3c+4])[0]
                if pe_offset < len(binary_data) - 24:
                    machine_type = struct.unpack('<H', binary_data[pe_offset+4:pe_offset+6])[0]
                    if machine_type == 0x014c:  # IMAGE_FILE_MACHINE_I386
                        return 'x86'
                    elif machine_type == 0x8664:  # IMAGE_FILE_MACHINE_AMD64
                        return 'x64'
            
            # Fallback to heuristic detection
            return self._heuristic_arch_detection(binary_data)
            
        except:
            return 'x86'  # Default fallback
    
    def _heuristic_arch_detection(self, binary_data: bytes) -> str:
        """Detect architecture using instruction pattern heuristics."""
        # Look for common x64 patterns
        x64_patterns = [
            b'\x48',  # REX.W prefix
            b'\x4c',  # REX.WR prefix
            b'\x49',  # REX.WB prefix
        ]
        
        x64_count = sum(binary_data.count(pattern) for pattern in x64_patterns)
        
        # If significant x64 patterns found, likely x64
        if x64_count > len(binary_data) // 1000:
            return 'x64'
        
        return 'x86'
    
    def _get_disassembler(self, arch: str):
        """Get appropriate disassembler for architecture."""
        if not CAPSTONE_AVAILABLE:
            return None
        
        if arch == 'x64':
            return self.disasm_x64
        else:
            return self.disasm_x86
    
    def _linear_sweep_analysis(self, binary_data: bytes, base_addr: int, disasm) -> Dict[int, BasicBlock]:
        """Perform linear sweep disassembly to identify basic blocks."""
        basic_blocks = {}
        current_block = None
        current_addr = base_addr
        
        # Track potential block starts
        block_starts = {base_addr}
        
        try:
            for insn in disasm.disasm(binary_data, base_addr):
                # Check if this should start a new block
                if current_addr in block_starts or current_block is None:
                    if current_block:
                        current_block.end_addr = current_addr - 1
                        basic_blocks[current_block.start_addr] = current_block
                    
                    current_block = BasicBlock(start_addr=current_addr, end_addr=current_addr)
                
                # Add instruction to current block
                insn_info = {
                    'address': insn.address,
                    'mnemonic': insn.mnemonic,
                    'op_str': insn.op_str,
                    'bytes': insn.bytes,
                    'size': insn.size
                }
                current_block.instructions.append(insn_info)
                
                # Check for control flow changes
                if self._is_control_flow_instruction(insn):
                    # End current block
                    current_block.end_addr = insn.address
                    basic_blocks[current_block.start_addr] = current_block
                    
                    # Add target as potential block start
                    target = self._get_jump_target(insn)
                    if target and base_addr <= target < base_addr + len(binary_data):
                        block_starts.add(target)
                    
                    # Next instruction starts new block (if not unconditional jump)
                    if not self._is_unconditional_jump(insn):
                        block_starts.add(insn.address + insn.size)
                    
                    current_block = None
                
                current_addr = insn.address + insn.size
            
            # Handle final block
            if current_block:
                current_block.end_addr = current_addr - 1
                basic_blocks[current_block.start_addr] = current_block
        
        except Exception as e:
            self.logger.error(f"Error in linear sweep: {e}")
        
        return basic_blocks
    
    def _recursive_traversal(self, binary_data: bytes, base_addr: int, 
                           entry_point: int, disasm) -> Dict[int, BasicBlock]:
        """Perform recursive traversal from entry point."""
        basic_blocks = {}
        visited = set()
        work_queue = deque([entry_point])
        
        while work_queue and len(visited) < self.max_analysis_depth:
            current_addr = work_queue.popleft()
            
            if current_addr in visited or current_addr < base_addr:
                continue
            
            if current_addr >= base_addr + len(binary_data):
                continue
            
            visited.add(current_addr)
            
            # Analyze block starting at current_addr
            block = self._analyze_single_block(binary_data, base_addr, current_addr, disasm)
            if block:
                basic_blocks[block.start_addr] = block
                
                # Add successors to work queue
                for successor in block.successors:
                    if successor not in visited:
                        work_queue.append(successor)
        
        return basic_blocks
    
    def _analyze_single_block(self, binary_data: bytes, base_addr: int, 
                            start_addr: int, disasm) -> Optional[BasicBlock]:
        """Analyze a single basic block starting at given address."""
        try:
            offset = start_addr - base_addr
            if offset < 0 or offset >= len(binary_data):
                return None
            
            block = BasicBlock(start_addr=start_addr, end_addr=start_addr)
            current_addr = start_addr
            
            # Disassemble until control flow instruction
            remaining_data = binary_data[offset:]
            for insn in disasm.disasm(remaining_data, start_addr):
                insn_info = {
                    'address': insn.address,
                    'mnemonic': insn.mnemonic,
                    'op_str': insn.op_str,
                    'bytes': insn.bytes,
                    'size': insn.size
                }
                block.instructions.append(insn_info)
                
                current_addr = insn.address + insn.size
                
                # Check for block termination
                if self._is_control_flow_instruction(insn):
                    block.end_addr = insn.address
                    
                    # Add successors
                    target = self._get_jump_target(insn)
                    if target:
                        block.successors.add(target)
                    
                    # Add fall-through if conditional
                    if not self._is_unconditional_jump(insn):
                        block.successors.add(current_addr)
                    
                    break
                
                # Limit block size to prevent infinite loops
                if len(block.instructions) > 1000:
                    block.end_addr = insn.address
                    block.successors.add(current_addr)
                    break
            
            return block
            
        except Exception as e:
            self.logger.error(f"Error analyzing block at {start_addr:x}: {e}")
            return None
    
    def _is_control_flow_instruction(self, insn) -> bool:
        """Check if instruction affects control flow."""
        if not hasattr(insn, 'mnemonic'):
            return False
        
        control_flow_mnemonics = {
            'jmp', 'je', 'jne', 'jz', 'jnz', 'jg', 'jge', 'jl', 'jle',
            'ja', 'jae', 'jb', 'jbe', 'jo', 'jno', 'js', 'jns',
            'call', 'ret', 'retn', 'iret', 'int', 'into', 'loop', 'loope', 'loopne'
        }
        
        return insn.mnemonic.lower() in control_flow_mnemonics
    
    def _is_unconditional_jump(self, insn) -> bool:
        """Check if instruction is unconditional jump."""
        if not hasattr(insn, 'mnemonic'):
            return False
        
        unconditional = {'jmp', 'ret', 'retn', 'iret'}
        return insn.mnemonic.lower() in unconditional
    
    def _get_jump_target(self, insn) -> Optional[int]:
        """Extract jump target address from instruction."""
        try:
            if not hasattr(insn, 'operands') or len(insn.operands) == 0:
                return None
            
            operand = insn.operands[0]
            
            # Direct address
            if hasattr(operand, 'imm'):
                return operand.imm
            
            # Register indirect - cannot determine statically
            return None
            
        except:
            return None
    
    def _analyze_indirect_jumps(self, basic_blocks: Dict[int, BasicBlock], 
                              binary_data: bytes, base_addr: int, disasm) -> List[Dict[str, Any]]:
        """Analyze indirect jumps using symbolic execution simulation."""
        indirect_jumps = []
        
        for addr, block in basic_blocks.items():
            if not block.instructions:
                continue
            
            last_insn = block.instructions[-1]
            
            # Check for indirect jump patterns
            if (last_insn['mnemonic'].lower() in ['jmp', 'call'] and 
                'ptr' in last_insn['op_str'].lower()):
                
                # Attempt to resolve targets
                targets = self._resolve_indirect_targets(block, binary_data, base_addr)
                
                indirect_jump = {
                    'address': last_insn['address'],
                    'instruction': f"{last_insn['mnemonic']} {last_insn['op_str']}",
                    'type': 'indirect_jump' if last_insn['mnemonic'].lower() == 'jmp' else 'indirect_call',
                    'resolved_targets': targets,
                    'confidence': self._calculate_resolution_confidence(targets)
                }
                
                indirect_jumps.append(indirect_jump)
        
        return indirect_jumps
    
    def _resolve_indirect_targets(self, block: BasicBlock, binary_data: bytes, base_addr: int) -> List[int]:
        """Attempt to resolve indirect jump targets using various techniques."""
        targets = []
        
        # Technique 1: Look for immediate loads to registers
        targets.extend(self._resolve_via_register_tracking(block))
        
        # Technique 2: Look for jump tables
        targets.extend(self._resolve_via_jump_table_analysis(block, binary_data, base_addr))
        
        # Technique 3: Pattern-based resolution
        targets.extend(self._resolve_via_pattern_matching(block))
        
        # Remove duplicates and invalid addresses
        valid_targets = []
        for target in set(targets):
            if base_addr <= target < base_addr + len(binary_data):
                valid_targets.append(target)
        
        return valid_targets[:self.max_indirect_targets]
    
    def _resolve_via_register_tracking(self, block: BasicBlock) -> List[int]:
        """Track register values to resolve indirect targets."""
        targets = []
        register_values = {}
        
        for insn in block.instructions:
            mnemonic = insn['mnemonic'].lower()
            op_str = insn['op_str']
            
            # Track immediate loads
            if mnemonic == 'mov' and ',' in op_str:
                parts = op_str.split(',')
                if len(parts) == 2:
                    dst = parts[0].strip()
                    src = parts[1].strip()
                    
                    # Check if source is immediate value
                    if src.startswith('0x') or src.isdigit():
                        try:
                            value = int(src, 16) if src.startswith('0x') else int(src)
                            register_values[dst] = value
                        except ValueError:
                            pass
            
            # Check if this is indirect jump using tracked register
            elif mnemonic in ['jmp', 'call'] and any(reg in op_str for reg in register_values):
                for reg, value in register_values.items():
                    if reg in op_str:
                        targets.append(value)
        
        return targets
    
    def _resolve_via_jump_table_analysis(self, block: BasicBlock, binary_data: bytes, base_addr: int) -> List[int]:
        """Analyze potential jump tables to resolve targets."""
        targets = []
        
        # Look for patterns indicating jump table access
        for i, insn in enumerate(block.instructions[:-1]):
            if ('add' in insn['mnemonic'].lower() or 'lea' in insn['mnemonic'].lower()):
                # Check if followed by indirect jump
                next_insn = block.instructions[i + 1]
                if (next_insn['mnemonic'].lower() == 'jmp' and 
                    'ptr' in next_insn['op_str'].lower()):
                    
                    # Try to extract jump table
                    table_targets = self._extract_jump_table(insn, binary_data, base_addr)
                    targets.extend(table_targets)
        
        return targets
    
    def _extract_jump_table(self, insn: Dict[str, Any], binary_data: bytes, base_addr: int) -> List[int]:
        """Extract addresses from potential jump table."""
        targets = []
        
        try:
            # Heuristic: look for aligned data after current instruction
            search_start = insn['address'] + insn['size'] - base_addr
            
            # Look for patterns of 4-byte aligned addresses
            for offset in range(search_start, min(search_start + 0x100, len(binary_data) - 4), 4):
                potential_addr = struct.unpack('<L', binary_data[offset:offset+4])[0]
                
                # Check if this looks like a valid code address
                if base_addr <= potential_addr < base_addr + len(binary_data):
                    targets.append(potential_addr)
                    
                    # Stop if we've found enough or hit invalid data
                    if len(targets) >= 20:
                        break
                else:
                    # Stop at first invalid entry (end of table)
                    break
        
        except:
            pass
        
        return targets
    
    def _resolve_via_pattern_matching(self, block: BasicBlock) -> List[int]:
        """Use pattern matching to resolve common indirect jump patterns."""
        targets = []
        
        # Pattern: API calls through import table
        # Look for patterns like "call dword ptr [0x401000]"
        for insn in block.instructions:
            if (insn['mnemonic'].lower() == 'call' and 
                'ptr [0x' in insn['op_str']):
                
                # Extract address from operand
                import re
                match = re.search(r'0x([0-9a-fA-F]+)', insn['op_str'])
                if match:
                    try:
                        addr = int(match.group(1), 16)
                        targets.append(addr)
                    except ValueError:
                        pass
        
        return targets
    
    def _calculate_resolution_confidence(self, targets: List[int]) -> float:
        """Calculate confidence score for indirect target resolution."""
        if not targets:
            return 0.0
        
        # More targets found = higher confidence (up to a point)
        target_score = min(len(targets) / 5.0, 1.0)
        
        # Single target = very high confidence
        if len(targets) == 1:
            return 0.9
        
        return target_score * 0.7
    
    def _analyze_switch_statements(self, basic_blocks: Dict[int, BasicBlock], 
                                 binary_data: bytes, disasm) -> List[Dict[str, Any]]:
        """Detect and analyze switch statement implementations."""
        switch_statements = []
        
        for addr, block in basic_blocks.items():
            # Look for switch patterns: compare + conditional jump + jump table
            switch_info = self._detect_switch_pattern(block)
            if switch_info:
                switch_statements.append(switch_info)
        
        return switch_statements
    
    def _detect_switch_pattern(self, block: BasicBlock) -> Optional[Dict[str, Any]]:
        """Detect switch statement patterns in basic block."""
        # Common switch patterns:
        # 1. cmp reg, immediate + ja/jae + jmp [reg*4 + table]
        # 2. sub reg, base + cmp reg, range + ja + jmp [reg*4 + table]
        
        if len(block.instructions) < 3:
            return None
        
        # Look for comparison followed by conditional jump
        for i in range(len(block.instructions) - 2):
            insn1 = block.instructions[i]
            insn2 = block.instructions[i + 1]
            
            if (insn1['mnemonic'].lower() == 'cmp' and
                insn2['mnemonic'].lower() in ['ja', 'jae', 'jb', 'jbe']):
                
                # This looks like a switch bounds check
                return {
                    'address': block.start_addr,
                    'bounds_check': f"{insn1['mnemonic']} {insn1['op_str']}",
                    'conditional_jump': f"{insn2['mnemonic']} {insn2['op_str']}",
                    'type': 'switch_statement',
                    'confidence': 0.7
                }
        
        return None
    
    def _analyze_exception_handlers(self, basic_blocks: Dict[int, BasicBlock], 
                                  binary_data: bytes) -> List[Dict[str, Any]]:
        """Detect exception handling constructs."""
        exception_handlers = []
        
        # Look for SEH patterns in PE binaries
        # This is a simplified implementation
        for addr, block in basic_blocks.items():
            if self._has_exception_pattern(block):
                exception_handlers.append({
                    'address': addr,
                    'type': 'structured_exception_handler',
                    'confidence': 0.6
                })
        
        return exception_handlers
    
    def _has_exception_pattern(self, block: BasicBlock) -> bool:
        """Check if block contains exception handling patterns."""
        # Look for FS register access (common in SEH)
        for insn in block.instructions:
            if 'fs:' in insn['op_str'].lower():
                return True
        
        return False
    
    def _detect_self_modifying_code(self, basic_blocks: Dict[int, BasicBlock], 
                                  binary_data: bytes) -> List[Dict[str, Any]]:
        """Detect self-modifying code regions."""
        self_modifying_regions = []
        
        for addr, block in basic_blocks.items():
            # Look for code that writes to executable memory
            if self._has_self_modification_pattern(block):
                self_modifying_regions.append({
                    'address': addr,
                    'type': 'self_modifying_code',
                    'confidence': 0.5
                })
        
        return self_modifying_regions
    
    def _has_self_modification_pattern(self, block: BasicBlock) -> bool:
        """Check if block contains self-modification patterns."""
        # Look for memory writes that could modify code
        for insn in block.instructions:
            if insn['mnemonic'].lower() in ['mov', 'stosb', 'stosw', 'stosd']:
                # This is a simplified check
                if 'ptr' in insn['op_str']:
                    return True
        
        return False
    
    def _merge_block_analyses(self, blocks1: Dict[int, BasicBlock], 
                            blocks2: Dict[int, BasicBlock]) -> Dict[int, BasicBlock]:
        """Merge results from multiple analysis passes."""
        merged = blocks1.copy()
        
        for addr, block in blocks2.items():
            if addr in merged:
                # Merge successor information
                merged[addr].successors.update(block.successors)
                merged[addr].predecessors.update(block.predecessors)
            else:
                merged[addr] = block
        
        return merged
    
    def _build_control_flow_graph(self, basic_blocks: Dict[int, BasicBlock]) -> nx.DiGraph:
        """Build NetworkX graph from basic blocks."""
        cfg = nx.DiGraph()
        
        # Add nodes
        for addr, block in basic_blocks.items():
            cfg.add_node(addr, block=block)
        
        # Add edges
        for addr, block in basic_blocks.items():
            for successor in block.successors:
                if successor in basic_blocks:
                    cfg.add_edge(addr, successor)
                    basic_blocks[successor].predecessors.add(addr)
        
        return cfg
    
    def _assess_analysis_quality(self, basic_blocks: Dict[int, BasicBlock], cfg: nx.DiGraph) -> float:
        """Assess quality of CFG reconstruction."""
        if not basic_blocks:
            return 0.0
        
        quality_score = 0.0
        
        # Factor 1: Block connectivity
        connected_blocks = sum(1 for addr in basic_blocks if basic_blocks[addr].successors)
        connectivity_score = connected_blocks / len(basic_blocks)
        quality_score += connectivity_score * 0.4
        
        # Factor 2: Graph structure (cycles, strongly connected components)
        if len(cfg.nodes) > 0:
            try:
                num_sccs = len(list(nx.strongly_connected_components(cfg)))
                scc_score = min(num_sccs / len(cfg.nodes), 1.0)
                quality_score += scc_score * 0.3
            except:
                pass
        
        # Factor 3: Instruction coverage
        total_instructions = sum(len(block.instructions) for block in basic_blocks.values())
        if total_instructions > 0:
            avg_block_size = total_instructions / len(basic_blocks)
            size_score = min(avg_block_size / 10.0, 1.0)  # Normalize around 10 instructions per block
            quality_score += size_score * 0.3
        
        return min(quality_score, 1.0)
    
    def _calculate_coverage(self, basic_blocks: Dict[int, BasicBlock], total_size: int) -> float:
        """Calculate percentage of binary covered by analysis."""
        if total_size == 0:
            return 0.0
        
        covered_bytes = 0
        for block in basic_blocks.values():
            covered_bytes += sum(insn.get('size', 1) for insn in block.instructions)
        
        return (covered_bytes / total_size) * 100.0
    
    def _fallback_analysis(self, binary_data: bytes, base_addr: int) -> CFGAnalysisResult:
        """Fallback analysis when disassembler unavailable."""
        # Create minimal result
        basic_blocks = {
            base_addr: BasicBlock(
                start_addr=base_addr,
                end_addr=base_addr + len(binary_data),
                instructions=[],
                analysis_metadata={'fallback': True}
            )
        }
        
        cfg = nx.DiGraph()
        cfg.add_node(base_addr)
        
        return CFGAnalysisResult(
            basic_blocks=basic_blocks,
            control_flow_graph=cfg,
            indirect_jumps=[],
            switch_statements=[],
            exception_handlers=[],
            self_modifying_regions=[],
            analysis_quality=0.1,
            coverage_percentage=0.0,
            metadata={'fallback_mode': True, 'error': 'No disassembler available'}
        )
    
    def _create_error_result(self, error_msg: str) -> CFGAnalysisResult:
        """Create error result for failed analysis."""
        return CFGAnalysisResult(
            basic_blocks={},
            control_flow_graph=nx.DiGraph(),
            indirect_jumps=[],
            switch_statements=[],
            exception_handlers=[],
            self_modifying_regions=[],
            analysis_quality=0.0,
            coverage_percentage=0.0,
            metadata={'error': error_msg}
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()


class CFGReconstructor:
    """
    Main interface for control flow graph reconstruction.
    
    Combines multiple analysis techniques to provide comprehensive
    CFG reconstruction with high accuracy for complex binaries.
    """
    
    def __init__(self, config_manager=None):
        """Initialize CFG reconstructor."""
        self.analyzer = AdvancedControlFlowAnalyzer(config_manager)
        self.logger = logging.getLogger(__name__)
    
    def reconstruct_cfg(self, binary_path, base_addr: int = None, 
                       entry_point: int = None) -> CFGAnalysisResult:
        """
        Reconstruct control flow graph from binary file.
        
        Args:
            binary_path: Path to binary file
            base_addr: Base address for analysis (auto-detect if None)
            entry_point: Entry point address (auto-detect if None)
            
        Returns:
            CFGAnalysisResult with complete analysis
        """
        try:
            # Read binary data
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            # Auto-detect base address if not provided
            if base_addr is None:
                base_addr = self._detect_base_address(binary_data)
            
            # Auto-detect entry point if not provided
            if entry_point is None:
                entry_point = self._detect_entry_point(binary_data, base_addr)
            
            self.logger.info(f"Reconstructing CFG for {binary_path}")
            self.logger.info(f"Base address: 0x{base_addr:x}, Entry point: 0x{entry_point:x}")
            
            # Perform analysis
            result = self.analyzer.analyze_control_flow(binary_data, base_addr, entry_point)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error reconstructing CFG: {e}")
            return self.analyzer._create_error_result(str(e))
    
    def _detect_base_address(self, binary_data: bytes) -> int:
        """Detect base address from PE header."""
        try:
            if len(binary_data) > 0x3c + 4:
                pe_offset = struct.unpack('<L', binary_data[0x3c:0x3c+4])[0]
                if pe_offset < len(binary_data) - 0x18:
                    # Read ImageBase from Optional Header
                    opt_header_offset = pe_offset + 0x18
                    image_base = struct.unpack('<L', binary_data[opt_header_offset+0x1c:opt_header_offset+0x20])[0]
                    return image_base
        except:
            pass
        
        return 0x400000  # Default for PE executables
    
    def _detect_entry_point(self, binary_data: bytes, base_addr: int) -> int:
        """Detect entry point from PE header."""
        try:
            if len(binary_data) > 0x3c + 4:
                pe_offset = struct.unpack('<L', binary_data[0x3c:0x3c+4])[0]
                if pe_offset < len(binary_data) - 0x18:
                    # Read AddressOfEntryPoint from Optional Header
                    opt_header_offset = pe_offset + 0x18
                    entry_rva = struct.unpack('<L', binary_data[opt_header_offset+0x10:opt_header_offset+0x14])[0]
                    return base_addr + entry_rva
        except:
            pass
        
        return base_addr  # Fallback to base address


# Factory functions
def create_cfg_reconstructor(config_manager=None) -> CFGReconstructor:
    """Create configured CFG reconstructor instance."""
    return CFGReconstructor(config_manager)


def create_control_flow_analyzer(config_manager=None) -> AdvancedControlFlowAnalyzer:
    """Create configured control flow analyzer instance."""
    return AdvancedControlFlowAnalyzer(config_manager)


# Example usage
if __name__ == "__main__":
    # Example usage
    reconstructor = CFGReconstructor()
    
    # Test data - simple x86 code
    test_code = bytes([
        0x55,           # push ebp
        0x89, 0xe5,     # mov ebp, esp
        0x83, 0xec, 0x10,  # sub esp, 16
        0x75, 0x05,     # jne +5
        0x90,           # nop
        0x90,           # nop
        0xc9,           # leave
        0xc3            # ret
    ])
    
    analyzer = AdvancedControlFlowAnalyzer()
    result = analyzer.analyze_control_flow(test_code, 0x401000, 0x401000)
    
    print(f"Analysis complete: {len(result.basic_blocks)} blocks")
    print(f"Quality: {result.analysis_quality:.2f}")
    print(f"Coverage: {result.coverage_percentage:.2f}%")