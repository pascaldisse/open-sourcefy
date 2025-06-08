"""
Agent 9: Advanced Assembly Analyzer
Performs detailed assembly analysis and instruction-level reconstruction.
"""

from typing import Dict, Any, List, Set, Tuple
import re
import subprocess
import json
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent9_AdvancedAssemblyAnalyzer(BaseAgent):
    """Agent 9: Advanced assembly analysis and instruction reconstruction"""
    
    def __init__(self):
        super().__init__(
            agent_id=9,
            name="AdvancedAssemblyAnalyzer",
            dependencies=[7]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute advanced assembly analysis"""
        agent7_result = context['agent_results'].get(7)
        if not agent7_result or agent7_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 7 (AdvancedDecompiler) did not complete successfully"
            )

        try:
            advanced_decompilation = agent7_result.data
            assembly_analysis = self._perform_assembly_analysis(advanced_decompilation, context)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=assembly_analysis,
                metadata={
                    'depends_on': [7],
                    'analysis_type': 'advanced_assembly_analysis'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Advanced assembly analysis failed: {str(e)}"
            )

    def _perform_assembly_analysis(self, advanced_decompilation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed assembly analysis"""
        # Get binary path for analysis
        binary_path = context.get('global_data', {}).get('binary_path') or context.get('binary_path')
        
        return {
            'instruction_analysis': self._analyze_instructions(advanced_decompilation),
            'register_usage': self._analyze_register_usage(advanced_decompilation),
            'memory_patterns': self._analyze_memory_patterns(advanced_decompilation),
            'calling_conventions': self._analyze_calling_conventions(advanced_decompilation),
            'asm_to_c_mapping': self._create_asm_to_c_mapping(advanced_decompilation),
            'binary_path': binary_path,
            'analysis_confidence': self._calculate_analysis_confidence(advanced_decompilation),
            'total_instructions_analyzed': self._count_total_instructions(advanced_decompilation),
            'unique_instruction_patterns': self._count_unique_patterns(advanced_decompilation),
            'complexity_score': self._calculate_complexity_score(advanced_decompilation)
        }

    def _analyze_instructions(self, advanced_decompilation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze instruction patterns and sequences"""
        try:
            # Extract assembly code from decompilation results
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            
            # Analyze control flow patterns
            control_flow = self._analyze_control_flow_patterns(assembly_data)
            
            # Classify instruction types
            instruction_types = self._classify_instructions(assembly_data)
            
            # Detect optimization patterns
            optimizations = self._detect_optimization_patterns(assembly_data)
            
            return {
                'control_flow_structures': control_flow,
                'instruction_classification': instruction_types,
                'optimization_patterns': optimizations,
                'total_basic_blocks': len(control_flow.get('basic_blocks', [])),
                'loop_structures': control_flow.get('loops', []),
                'conditional_branches': control_flow.get('conditionals', []),
                'function_calls': control_flow.get('function_calls', [])
            }
        except Exception as e:
            return {
                'error': f"Instruction analysis failed: {str(e)}",
                'control_flow_structures': {},
                'instruction_classification': {},
                'optimization_patterns': {}
            }

    def _analyze_register_usage(self, advanced_decompilation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze register usage patterns"""
        try:
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            
            # Track register usage across functions
            register_usage = self._track_register_usage(assembly_data)
            
            # Detect calling conventions
            calling_convention = self._detect_calling_convention(register_usage)
            
            # Calculate register pressure
            pressure_analysis = self._calculate_register_pressure(register_usage)
            
            return {
                'register_frequency': register_usage,
                'calling_convention': calling_convention,
                'register_pressure': pressure_analysis,
                'function_signatures': self._infer_function_signatures(register_usage),
                'parameter_registers': calling_convention.get('parameter_registers', []),
                'return_registers': calling_convention.get('return_registers', [])
            }
        except Exception as e:
            return {
                'error': f"Register analysis failed: {str(e)}",
                'register_frequency': {},
                'calling_convention': {'detected': 'unknown'},
                'register_pressure': {'average': 0.0}
            }

    def _analyze_memory_patterns(self, advanced_decompilation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        try:
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            
            # Analyze memory access patterns
            memory_accesses = self._extract_memory_accesses(assembly_data)
            
            # Detect data structures
            data_structures = self._detect_data_structures(memory_accesses)
            
            # Analyze stack usage
            stack_analysis = self._analyze_stack_usage(memory_accesses)
            
            # Track pointer operations
            pointer_analysis = self._analyze_pointer_operations(memory_accesses)
            
            return {
                'memory_access_patterns': memory_accesses,
                'detected_structures': data_structures,
                'stack_analysis': stack_analysis,
                'pointer_operations': pointer_analysis,
                'heap_operations': self._detect_heap_operations(memory_accesses),
                'array_accesses': data_structures.get('arrays', []),
                'struct_accesses': data_structures.get('structs', [])
            }
        except Exception as e:
            return {
                'error': f"Memory pattern analysis failed: {str(e)}",
                'memory_access_patterns': {},
                'detected_structures': {},
                'stack_analysis': {}
            }

    def _analyze_calling_conventions(self, advanced_decompilation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze calling convention usage"""
        try:
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            
            # Analyze function prologues and epilogues
            prologue_patterns = self._analyze_function_prologues(assembly_data)
            epilogue_patterns = self._analyze_function_epilogues(assembly_data)
            
            # Detect parameter passing patterns
            parameter_passing = self._analyze_parameter_passing(assembly_data)
            
            # Classify calling convention
            convention = self._classify_calling_convention(prologue_patterns, parameter_passing)
            
            return {
                'detected_convention': convention,
                'prologue_patterns': prologue_patterns,
                'epilogue_patterns': epilogue_patterns,
                'parameter_passing': parameter_passing,
                'stack_cleanup': convention.get('stack_cleanup', 'caller'),
                'function_signatures': self._extract_function_signatures(assembly_data)
            }
        except Exception as e:
            return {
                'error': f"Calling convention analysis failed: {str(e)}",
                'detected_convention': {'name': 'unknown'},
                'prologue_patterns': [],
                'epilogue_patterns': []
            }

    def _create_asm_to_c_mapping(self, advanced_decompilation: Dict[str, Any]) -> Dict[str, Any]:
        """Create mapping between assembly and C constructs"""
        try:
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            
            # Map control flow structures
            control_flow_mapping = self._map_control_flow_to_c(assembly_data)
            
            # Map data types and operations
            data_type_mapping = self._map_data_types_to_c(assembly_data)
            
            # Map function calls and signatures
            function_mapping = self._map_functions_to_c(assembly_data)
            
            # Generate C code patterns
            c_patterns = self._generate_c_patterns(control_flow_mapping, data_type_mapping, function_mapping)
            
            return {
                'control_flow_mapping': control_flow_mapping,
                'data_type_mapping': data_type_mapping,
                'function_mapping': function_mapping,
                'c_code_patterns': c_patterns,
                'confidence_score': self._calculate_mapping_confidence(c_patterns),
                'reconstruction_hints': self._generate_reconstruction_hints(c_patterns)
            }
        except Exception as e:
            return {
                'error': f"ASM to C mapping failed: {str(e)}",
                'control_flow_mapping': {},
                'data_type_mapping': {},
                'function_mapping': {}
            }


    def _calculate_complexity_score(self, advanced_decompilation: Dict[str, Any]) -> float:
        """Calculate code complexity score"""
        try:
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            
            # Calculate cyclomatic complexity
            cyclomatic = self._calculate_cyclomatic_complexity(assembly_data)
            
            # Calculate nesting depth
            nesting_depth = self._calculate_nesting_depth(assembly_data)
            
            # Calculate instruction complexity
            instruction_complexity = self._calculate_instruction_complexity(assembly_data)
            
            # Combine metrics for overall complexity score
            complexity_score = (cyclomatic * 0.4) + (nesting_depth * 0.3) + (instruction_complexity * 0.3)
            
            return min(complexity_score, 10.0)  # Cap at 10.0
        except Exception:
            return 5.0  # Default moderate complexity
    
    def _calculate_analysis_confidence(self, advanced_decompilation: Dict[str, Any]) -> float:
        """Calculate analysis confidence based on decompilation data quality"""
        try:
            confidence_factors = []
            
            # Check decompilation completeness
            if isinstance(advanced_decompilation, dict):
                if advanced_decompilation.get('decompiled_functions'):
                    confidence_factors.append(0.3)
                if advanced_decompilation.get('ghidra_analysis'):
                    confidence_factors.append(0.3)
                if advanced_decompilation.get('control_flow_graph'):
                    confidence_factors.append(0.2)
                if advanced_decompilation.get('type_analysis'):
                    confidence_factors.append(0.2)
            
            return min(sum(confidence_factors), 1.0)
        except Exception:
            return 0.5  # Default moderate confidence
    
    def _count_total_instructions(self, advanced_decompilation: Dict[str, Any]) -> int:
        """Count total instructions analyzed"""
        try:
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            total_instructions = 0
            
            for function_data in assembly_data.values():
                if isinstance(function_data, dict) and 'instructions' in function_data:
                    total_instructions += len(function_data['instructions'])
            
            return total_instructions
        except Exception:
            return 0
    
    def _count_unique_patterns(self, advanced_decompilation: Dict[str, Any]) -> int:
        """Count unique instruction patterns detected"""
        try:
            assembly_data = self._extract_assembly_from_decompilation(advanced_decompilation)
            unique_patterns = set()
            
            for function_data in assembly_data.values():
                if isinstance(function_data, dict) and 'instructions' in function_data:
                    for instruction in function_data['instructions']:
                        # Extract instruction opcode as pattern
                        opcode = instruction.split()[0] if instruction.strip() else ''
                        if opcode:
                            unique_patterns.add(opcode.lower())
            
            return len(unique_patterns)
        except Exception:
            return 0

    # Helper methods for control flow reconstruction
    def _extract_assembly_from_decompilation(self, advanced_decompilation: Dict[str, Any]) -> Dict[str, Any]:
        """Extract assembly data from decompilation results"""
        assembly_data = {}
        
        try:
            if isinstance(advanced_decompilation, dict):
                # Extract from Ghidra analysis if available
                ghidra_data = advanced_decompilation.get('ghidra_analysis', {})
                if isinstance(ghidra_data, dict):
                    functions = ghidra_data.get('functions', {})
                    for func_name, func_data in functions.items():
                        if isinstance(func_data, dict):
                            assembly_data[func_name] = {
                                'instructions': self._parse_assembly_instructions(func_data.get('assembly', '')),
                                'address': func_data.get('address', '0x0'),
                                'size': func_data.get('size', 0),
                                'code': func_data.get('code', '')
                            }
                
                # Also extract from decompiled functions
                decompiled = advanced_decompilation.get('decompiled_functions', {})
                for func_name, func_data in decompiled.items():
                    if func_name not in assembly_data and isinstance(func_data, dict):
                        # Try to extract assembly from code comments or annotations
                        code = func_data.get('code', '')
                        assembly_data[func_name] = {
                            'instructions': self._extract_instructions_from_code(code),
                            'address': func_data.get('address', '0x0'),
                            'size': func_data.get('size', 0),
                            'code': code
                        }
            
            return assembly_data
        except Exception:
            return {}
    
    def _parse_assembly_instructions(self, assembly_text: str) -> List[str]:
        """Parse assembly text into individual instructions"""
        if not assembly_text:
            return []
        
        instructions = []
        lines = assembly_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith(';') and not line.startswith('#'):
                # Remove address prefixes (e.g., "0x401000: mov eax, ebx")
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        instruction = parts[1].strip()
                        if instruction:
                            instructions.append(instruction)
                else:
                    instructions.append(line)
        
        return instructions
    
    def _extract_instructions_from_code(self, code: str) -> List[str]:
        """Extract assembly instructions from C code comments"""
        instructions = []
        lines = code.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for assembly in comments
            if '//' in line and ('mov' in line or 'push' in line or 'call' in line):
                comment_part = line.split('//', 1)[1].strip()
                if comment_part:
                    instructions.append(comment_part)
        
        return instructions
    
    def _analyze_control_flow_patterns(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze control flow patterns from assembly"""
        control_flow = {
            'basic_blocks': [],
            'loops': [],
            'conditionals': [],
            'function_calls': [],
            'jumps': []
        }
        
        try:
            for func_name, func_data in assembly_data.items():
                instructions = func_data.get('instructions', [])
                
                # Detect basic blocks
                basic_blocks = self._identify_basic_blocks(instructions)
                control_flow['basic_blocks'].extend(basic_blocks)
                
                # Detect loops
                loops = self._detect_loops(instructions)
                control_flow['loops'].extend(loops)
                
                # Detect conditionals
                conditionals = self._detect_conditionals(instructions)
                control_flow['conditionals'].extend(conditionals)
                
                # Detect function calls
                calls = self._detect_function_calls(instructions)
                control_flow['function_calls'].extend(calls)
                
                # Detect jumps
                jumps = self._detect_jumps(instructions)
                control_flow['jumps'].extend(jumps)
        
        except Exception:
            pass
        
        return control_flow
    
    def _identify_basic_blocks(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Identify basic blocks in assembly instructions"""
        basic_blocks = []
        current_block = []
        block_start = 0
        
        for i, instruction in enumerate(instructions):
            current_block.append(instruction)
            
            # End block on control flow instructions
            if any(op in instruction.lower() for op in ['jmp', 'je', 'jne', 'jz', 'jnz', 'call', 'ret']):
                if current_block:
                    basic_blocks.append({
                        'start_index': block_start,
                        'end_index': i,
                        'instructions': current_block.copy(),
                        'size': len(current_block)
                    })
                current_block = []
                block_start = i + 1
        
        # Add final block if it exists
        if current_block:
            basic_blocks.append({
                'start_index': block_start,
                'end_index': len(instructions) - 1,
                'instructions': current_block,
                'size': len(current_block)
            })
        
        return basic_blocks
    
    def _detect_loops(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Detect loop structures in assembly"""
        loops = []
        
        # Look for common loop patterns
        for i, instruction in enumerate(instructions):
            instruction_lower = instruction.lower()
            
            # Detect for loops (loop instruction)
            if 'loop' in instruction_lower:
                loops.append({
                    'type': 'for_loop',
                    'instruction_index': i,
                    'pattern': 'loop instruction',
                    'instruction': instruction
                })
            
            # Detect while loops (conditional jump backwards)
            elif any(jmp in instruction_lower for jmp in ['je', 'jne', 'jz', 'jnz', 'jl', 'jg']):
                # Check if this is a backward jump (simplified heuristic)
                if '$-' in instruction or any(reg in instruction for reg in ['loop', 'dec', 'inc']):
                    loops.append({
                        'type': 'while_loop',
                        'instruction_index': i,
                        'pattern': 'conditional backward jump',
                        'instruction': instruction
                    })
        
        return loops
    
    def _detect_conditionals(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Detect conditional structures in assembly"""
        conditionals = []
        
        for i, instruction in enumerate(instructions):
            instruction_lower = instruction.lower()
            
            # Detect conditional jumps
            if any(jmp in instruction_lower for jmp in ['je', 'jne', 'jz', 'jnz', 'jl', 'jg', 'jle', 'jge']):
                conditionals.append({
                    'type': 'conditional_jump',
                    'instruction_index': i,
                    'condition': self._extract_condition_type(instruction),
                    'instruction': instruction
                })
            
            # Detect compare instructions that often precede conditionals
            elif any(cmp in instruction_lower for cmp in ['cmp', 'test']):
                conditionals.append({
                    'type': 'comparison',
                    'instruction_index': i,
                    'comparison_type': 'register_comparison',
                    'instruction': instruction
                })
        
        return conditionals
    
    def _detect_function_calls(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Detect function calls in assembly"""
        function_calls = []
        
        for i, instruction in enumerate(instructions):
            instruction_lower = instruction.lower()
            
            if 'call' in instruction_lower:
                # Extract function name or address
                parts = instruction.split()
                target = parts[1] if len(parts) > 1 else 'unknown'
                
                function_calls.append({
                    'instruction_index': i,
                    'target': target,
                    'call_type': 'direct_call' if not target.startswith('[') else 'indirect_call',
                    'instruction': instruction
                })
        
        return function_calls
    
    def _detect_jumps(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Detect jump instructions in assembly"""
        jumps = []
        
        for i, instruction in enumerate(instructions):
            instruction_lower = instruction.lower()
            
            if 'jmp' in instruction_lower:
                parts = instruction.split()
                target = parts[1] if len(parts) > 1 else 'unknown'
                
                jumps.append({
                    'instruction_index': i,
                    'target': target,
                    'jump_type': 'unconditional',
                    'instruction': instruction
                })
        
        return jumps
    
    def _extract_condition_type(self, instruction: str) -> str:
        """Extract condition type from conditional instruction"""
        instruction_lower = instruction.lower()
        
        if 'je' in instruction_lower or 'jz' in instruction_lower:
            return 'equal_zero'
        elif 'jne' in instruction_lower or 'jnz' in instruction_lower:
            return 'not_equal_zero'
        elif 'jl' in instruction_lower:
            return 'less_than'
        elif 'jg' in instruction_lower:
            return 'greater_than'
        elif 'jle' in instruction_lower:
            return 'less_equal'
        elif 'jge' in instruction_lower:
            return 'greater_equal'
        else:
            return 'unknown'
    
    def _classify_instructions(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify instructions by type"""
        classification = {
            'arithmetic': [],
            'memory': [],
            'control_flow': [],
            'data_movement': [],
            'logical': []
        }
        
        try:
            for func_name, func_data in assembly_data.items():
                instructions = func_data.get('instructions', [])
                
                for instruction in instructions:
                    instruction_lower = instruction.lower()
                    
                    # Classify instruction
                    if any(op in instruction_lower for op in ['add', 'sub', 'mul', 'div', 'inc', 'dec']):
                        classification['arithmetic'].append(instruction)
                    elif any(op in instruction_lower for op in ['mov', 'push', 'pop', 'lea']):
                        classification['data_movement'].append(instruction)
                    elif any(op in instruction_lower for op in ['jmp', 'je', 'jne', 'call', 'ret']):
                        classification['control_flow'].append(instruction)
                    elif any(op in instruction_lower for op in ['and', 'or', 'xor', 'not', 'shl', 'shr']):
                        classification['logical'].append(instruction)
                    elif any(op in instruction_lower for op in ['mov', 'load', 'store', 'ld', 'st']):
                        classification['memory'].append(instruction)
        
        except Exception:
            pass
        
        return classification
    
    def _detect_optimization_patterns(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect compiler optimization patterns"""
        optimizations = {
            'loop_unrolling': [],
            'constant_folding': [],
            'dead_code_elimination': [],
            'register_allocation': [],
            'inlining': []
        }
        
        try:
            for func_name, func_data in assembly_data.items():
                instructions = func_data.get('instructions', [])
                
                # Detect loop unrolling (repeated instruction patterns)
                repeated_patterns = self._detect_repeated_patterns(instructions)
                if repeated_patterns:
                    optimizations['loop_unrolling'].extend(repeated_patterns)
                
                # Detect constant folding (immediate values)
                constants = self._detect_immediate_constants(instructions)
                if constants:
                    optimizations['constant_folding'].extend(constants)
                
                # Detect register reuse patterns
                register_patterns = self._detect_register_reuse(instructions)
                if register_patterns:
                    optimizations['register_allocation'].extend(register_patterns)
        
        except Exception:
            pass
        
        return optimizations
    
    def _detect_repeated_patterns(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Detect repeated instruction patterns indicating loop unrolling"""
        patterns = []
        
        # Simple pattern detection - look for sequences that repeat
        sequence_length = 3
        for i in range(len(instructions) - sequence_length * 2):
            sequence1 = instructions[i:i + sequence_length]
            sequence2 = instructions[i + sequence_length:i + sequence_length * 2]
            
            if sequence1 == sequence2:
                patterns.append({
                    'pattern': 'repeated_sequence',
                    'start_index': i,
                    'sequence_length': sequence_length,
                    'instructions': sequence1
                })
        
        return patterns
    
    def _detect_immediate_constants(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Detect immediate constants indicating constant folding"""
        constants = []
        
        for i, instruction in enumerate(instructions):
            # Look for immediate values (starting with $ or #)
            if '$' in instruction or '#' in instruction:
                constants.append({
                    'instruction_index': i,
                    'type': 'immediate_constant',
                    'instruction': instruction
                })
        
        return constants
    
    def _detect_register_reuse(self, instructions: List[str]) -> List[Dict[str, Any]]:
        """Detect register reuse patterns"""
        patterns = []
        register_usage = {}
        
        for i, instruction in enumerate(instructions):
            # Extract registers from instruction
            registers = self._extract_registers(instruction)
            
            for reg in registers:
                if reg not in register_usage:
                    register_usage[reg] = []
                register_usage[reg].append(i)
        
        # Identify heavily reused registers
        for reg, usage_indices in register_usage.items():
            if len(usage_indices) > 5:  # Threshold for heavy usage
                patterns.append({
                    'register': reg,
                    'usage_count': len(usage_indices),
                    'usage_indices': usage_indices
                })
        
        return patterns
    
    def _extract_registers(self, instruction: str) -> Set[str]:
        """Extract register names from instruction"""
        registers = set()
        
        # Common x86/x64 registers
        reg_patterns = [
            r'\b(eax|ebx|ecx|edx|esi|edi|esp|ebp)\b',
            r'\b(ax|bx|cx|dx|si|di|sp|bp)\b',
            r'\b(al|bl|cl|dl|ah|bh|ch|dh)\b',
            r'\b(rax|rbx|rcx|rdx|rsi|rdi|rsp|rbp)\b',
            r'\b(r[8-9]|r1[0-5])[dwb]?\b'
        ]
        
        for pattern in reg_patterns:
            matches = re.findall(pattern, instruction.lower())
            registers.update(matches)
        
        return registers
    
    def _calculate_cyclomatic_complexity(self, assembly_data: Dict[str, Any]) -> float:
        """Calculate cyclomatic complexity from control flow"""
        total_complexity = 0
        function_count = 0
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            # Count decision points (conditional jumps)
            decision_points = 0
            for instruction in instructions:
                if any(jmp in instruction.lower() for jmp in ['je', 'jne', 'jz', 'jnz', 'jl', 'jg']):
                    decision_points += 1
            
            # Cyclomatic complexity = decision_points + 1
            complexity = decision_points + 1
            total_complexity += complexity
            function_count += 1
        
        return total_complexity / max(function_count, 1)
    
    def _calculate_nesting_depth(self, assembly_data: Dict[str, Any]) -> float:
        """Calculate average nesting depth"""
        total_depth = 0
        function_count = 0
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            # Estimate nesting depth by tracking call depth and loop depth
            max_depth = 0
            current_depth = 0
            
            for instruction in instructions:
                instruction_lower = instruction.lower()
                
                if 'call' in instruction_lower:
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif 'ret' in instruction_lower:
                    current_depth = max(0, current_depth - 1)
                elif any(loop in instruction_lower for loop in ['loop', 'je', 'jne']):
                    # Simplified loop detection
                    current_depth += 0.5
                    max_depth = max(max_depth, current_depth)
            
            total_depth += max_depth
            function_count += 1
        
        return total_depth / max(function_count, 1)
    
    def _calculate_instruction_complexity(self, assembly_data: Dict[str, Any]) -> float:
        """Calculate instruction complexity score"""
        total_complexity = 0
        instruction_count = 0
        
        # Complexity weights for different instruction types
        complexity_weights = {
            'simple': 1.0,    # mov, add, sub
            'medium': 2.0,    # mul, div, shift operations
            'complex': 3.0,   # floating point, SIMD
            'control': 2.5    # jumps, calls
        }
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            for instruction in instructions:
                instruction_lower = instruction.lower()
                
                # Classify instruction complexity
                if any(op in instruction_lower for op in ['mov', 'add', 'sub', 'inc', 'dec']):
                    complexity = complexity_weights['simple']
                elif any(op in instruction_lower for op in ['mul', 'div', 'shl', 'shr']):
                    complexity = complexity_weights['medium']
                elif any(op in instruction_lower for op in ['fld', 'fst', 'fadd', 'fsub', 'sse', 'mmx']):
                    complexity = complexity_weights['complex']
                elif any(op in instruction_lower for op in ['jmp', 'je', 'jne', 'call']):
                    complexity = complexity_weights['control']
                else:
                    complexity = complexity_weights['simple']
                
                total_complexity += complexity
                instruction_count += 1
        
        return total_complexity / max(instruction_count, 1)
    
    # Helper methods for register usage analysis
    def _track_register_usage(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track register usage patterns across functions"""
        register_stats = {}
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            for instruction in instructions:
                registers = self._extract_registers(instruction)
                
                for reg in registers:
                    if reg not in register_stats:
                        register_stats[reg] = {
                            'frequency': 0,
                            'functions': set(),
                            'contexts': []
                        }
                    
                    register_stats[reg]['frequency'] += 1
                    register_stats[reg]['functions'].add(func_name)
                    register_stats[reg]['contexts'].append({
                        'function': func_name,
                        'instruction': instruction,
                        'type': self._classify_register_usage(instruction, reg)
                    })
        
        # Convert sets to lists for JSON serialization
        for reg_data in register_stats.values():
            reg_data['functions'] = list(reg_data['functions'])
        
        return register_stats
    
    def _classify_register_usage(self, instruction: str, register: str) -> str:
        """Classify how a register is used in an instruction"""
        instruction_lower = instruction.lower()
        
        if instruction_lower.startswith('mov') and register in instruction_lower.split(',')[0]:
            return 'destination'
        elif instruction_lower.startswith('mov') and register in instruction_lower.split(',')[1]:
            return 'source'
        elif any(op in instruction_lower for op in ['push', 'call']):
            return 'parameter'
        elif 'ret' in instruction_lower:
            return 'return_value'
        elif any(op in instruction_lower for op in ['add', 'sub', 'mul', 'div']):
            return 'arithmetic'
        else:
            return 'general'
    
    def _detect_calling_convention(self, register_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Detect calling convention from register usage patterns"""
        convention_hints = {
            'parameter_registers': [],
            'return_registers': [],
            'preserved_registers': [],
            'detected': 'unknown'
        }
        
        # Analyze register usage patterns
        for reg, stats in register_usage.items():
            contexts = stats.get('contexts', [])
            
            # Check for parameter passing patterns
            param_usage = sum(1 for ctx in contexts if ctx['type'] == 'parameter')
            if param_usage > 0:
                convention_hints['parameter_registers'].append(reg)
            
            # Check for return value patterns
            return_usage = sum(1 for ctx in contexts if ctx['type'] == 'return_value')
            if return_usage > 0:
                convention_hints['return_registers'].append(reg)
        
        # Detect specific calling conventions
        if 'eax' in convention_hints['return_registers']:
            if any(reg in convention_hints['parameter_registers'] for reg in ['ecx', 'edx']):
                convention_hints['detected'] = 'fastcall'
            else:
                convention_hints['detected'] = 'cdecl'
        elif 'rax' in convention_hints['return_registers']:
            if any(reg in convention_hints['parameter_registers'] for reg in ['rcx', 'rdx', 'r8', 'r9']):
                convention_hints['detected'] = 'microsoft_x64'
            else:
                convention_hints['detected'] = 'system_v_x64'
        
        return convention_hints
    
    def _calculate_register_pressure(self, register_usage: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate register pressure analysis"""
        total_usage = sum(stats['frequency'] for stats in register_usage.values())
        num_registers = len(register_usage)
        
        pressure_analysis = {
            'average_usage': total_usage / max(num_registers, 1),
            'most_used_register': max(register_usage.items(), 
                                     key=lambda x: x[1]['frequency'], 
                                     default=('none', {'frequency': 0}))[0],
            'register_distribution': {},
            'pressure_level': 'low'
        }
        
        # Calculate distribution
        for reg, stats in register_usage.items():
            pressure_analysis['register_distribution'][reg] = stats['frequency']
        
        # Determine pressure level
        if pressure_analysis['average_usage'] > 20:
            pressure_analysis['pressure_level'] = 'high'
        elif pressure_analysis['average_usage'] > 10:
            pressure_analysis['pressure_level'] = 'medium'
        
        return pressure_analysis
    
    def _infer_function_signatures(self, register_usage: Dict[str, Any]) -> Dict[str, str]:
        """Infer function signatures from register usage"""
        signatures = {}
        
        # Simple signature inference based on register patterns
        functions = set()
        for stats in register_usage.values():
            functions.update(stats.get('functions', []))
        
        for func in functions:
            # Analyze parameter and return patterns for this function
            param_regs = []
            return_regs = []
            
            for reg, stats in register_usage.items():
                contexts = [ctx for ctx in stats.get('contexts', []) if ctx['function'] == func]
                
                if any(ctx['type'] == 'parameter' for ctx in contexts):
                    param_regs.append(reg)
                if any(ctx['type'] == 'return_value' for ctx in contexts):
                    return_regs.append(reg)
            
            # Generate signature
            return_type = 'int' if return_regs else 'void'
            param_count = len(param_regs)
            
            if param_count == 0:
                signatures[func] = f"{return_type} {func}(void)"
            else:
                params = ', '.join([f"int param{i}" for i in range(param_count)])
                signatures[func] = f"{return_type} {func}({params})"
        
        return signatures
    
    # Helper methods for memory pattern analysis
    def _extract_memory_accesses(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory access patterns from assembly"""
        memory_accesses = {
            'loads': [],
            'stores': [],
            'stack_accesses': [],
            'heap_accesses': []
        }
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            for i, instruction in enumerate(instructions):
                instruction_lower = instruction.lower()
                
                # Detect memory loads
                if any(op in instruction_lower for op in ['mov', 'ld', 'load']):
                    if '[' in instruction and ']' in instruction:
                        memory_accesses['loads'].append({
                            'function': func_name,
                            'instruction_index': i,
                            'instruction': instruction,
                            'address_pattern': self._extract_address_pattern(instruction)
                        })
                
                # Detect memory stores
                if any(op in instruction_lower for op in ['mov', 'st', 'store']):
                    if '[' in instruction and ']' in instruction:
                        # Check if destination is memory
                        parts = instruction.split(',')
                        if len(parts) > 1 and '[' in parts[1]:
                            memory_accesses['stores'].append({
                                'function': func_name,
                                'instruction_index': i,
                                'instruction': instruction,
                                'address_pattern': self._extract_address_pattern(instruction)
                            })
                
                # Detect stack accesses
                if any(stack_op in instruction_lower for stack_op in ['push', 'pop', 'esp', 'ebp', 'rsp', 'rbp']):
                    memory_accesses['stack_accesses'].append({
                        'function': func_name,
                        'instruction_index': i,
                        'instruction': instruction,
                        'stack_operation': self._classify_stack_operation(instruction)
                    })
        
        return memory_accesses
    
    def _extract_address_pattern(self, instruction: str) -> Dict[str, Any]:
        """Extract addressing pattern from memory instruction"""
        pattern = {
            'base_register': None,
            'index_register': None,
            'displacement': None,
            'scale': None
        }
        
        # Find memory operand in brackets
        start = instruction.find('[')
        end = instruction.find(']')
        
        if start != -1 and end != -1:
            memory_operand = instruction[start+1:end]
            
            # Parse addressing components
            if '+' in memory_operand:
                parts = memory_operand.split('+')
                for part in parts:
                    part = part.strip()
                    if part.isdigit() or part.startswith('0x'):
                        pattern['displacement'] = part
                    elif any(reg in part.lower() for reg in ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi', 'esp', 'ebp']):
                        if pattern['base_register'] is None:
                            pattern['base_register'] = part
                        else:
                            pattern['index_register'] = part
            else:
                # Single component
                if memory_operand.isdigit() or memory_operand.startswith('0x'):
                    pattern['displacement'] = memory_operand
                else:
                    pattern['base_register'] = memory_operand
        
        return pattern
    
    def _classify_stack_operation(self, instruction: str) -> str:
        """Classify type of stack operation"""
        instruction_lower = instruction.lower()
        
        if 'push' in instruction_lower:
            return 'push'
        elif 'pop' in instruction_lower:
            return 'pop'
        elif 'esp' in instruction_lower or 'rsp' in instruction_lower:
            return 'stack_pointer_manipulation'
        elif 'ebp' in instruction_lower or 'rbp' in instruction_lower:
            return 'frame_pointer_access'
        else:
            return 'unknown_stack_operation'
    
    def _detect_data_structures(self, memory_accesses: Dict[str, Any]) -> Dict[str, Any]:
        """Detect data structures from memory access patterns"""
        structures = {
            'arrays': [],
            'structs': [],
            'pointers': []
        }
        
        # Analyze memory access patterns for structure detection
        loads = memory_accesses.get('loads', [])
        stores = memory_accesses.get('stores', [])
        
        # Detect array access patterns (base + index * scale)
        for access in loads + stores:
            pattern = access.get('address_pattern', {})
            
            if pattern.get('index_register') and pattern.get('base_register'):
                structures['arrays'].append({
                    'base_register': pattern['base_register'],
                    'index_register': pattern['index_register'],
                    'access_instruction': access['instruction'],
                    'function': access['function']
                })
            
            # Detect struct access patterns (base + fixed offset)
            elif pattern.get('base_register') and pattern.get('displacement'):
                structures['structs'].append({
                    'base_register': pattern['base_register'],
                    'offset': pattern['displacement'],
                    'access_instruction': access['instruction'],
                    'function': access['function']
                })
            
            # Detect pointer dereferencing
            elif pattern.get('base_register') and not pattern.get('displacement'):
                structures['pointers'].append({
                    'pointer_register': pattern['base_register'],
                    'access_instruction': access['instruction'],
                    'function': access['function']
                })
        
        return structures
    
    def _analyze_stack_usage(self, memory_accesses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stack usage patterns"""
        stack_accesses = memory_accesses.get('stack_accesses', [])
        
        analysis = {
            'total_stack_operations': len(stack_accesses),
            'push_count': 0,
            'pop_count': 0,
            'frame_accesses': 0,
            'stack_depth_estimate': 0
        }
        
        current_depth = 0
        max_depth = 0
        
        for access in stack_accesses:
            operation = access.get('stack_operation', '')
            
            if operation == 'push':
                analysis['push_count'] += 1
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif operation == 'pop':
                analysis['pop_count'] += 1
                current_depth = max(0, current_depth - 1)
            elif operation == 'frame_pointer_access':
                analysis['frame_accesses'] += 1
        
        analysis['stack_depth_estimate'] = max_depth
        analysis['stack_balance'] = analysis['push_count'] - analysis['pop_count']
        
        return analysis
    
    def _analyze_pointer_operations(self, memory_accesses: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pointer operations and dereferencing"""
        loads = memory_accesses.get('loads', [])
        stores = memory_accesses.get('stores', [])
        
        pointer_analysis = {
            'dereference_count': 0,
            'pointer_arithmetic': [],
            'function_pointers': [],
            'data_pointers': []
        }
        
        # Analyze memory accesses for pointer patterns
        for access in loads + stores:
            instruction = access.get('instruction', '')
            pattern = access.get('address_pattern', {})
            
            # Count dereferencing operations
            if '[' in instruction and ']' in instruction:
                pointer_analysis['dereference_count'] += 1
                
                # Classify pointer type
                if 'call' in instruction.lower():
                    pointer_analysis['function_pointers'].append(access)
                else:
                    pointer_analysis['data_pointers'].append(access)
            
            # Detect pointer arithmetic
            if pattern.get('index_register') or pattern.get('displacement'):
                pointer_analysis['pointer_arithmetic'].append({
                    'instruction': instruction,
                    'pattern': pattern,
                    'function': access['function']
                })
        
        return pointer_analysis
    
    def _detect_heap_operations(self, memory_accesses: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect heap allocation and deallocation operations"""
        heap_operations = []
        
        # This is a simplified detection - in reality, would need to analyze
        # call patterns to malloc, free, new, delete, etc.
        loads = memory_accesses.get('loads', [])
        stores = memory_accesses.get('stores', [])
        
        for access in loads + stores:
            instruction = access.get('instruction', '')
            
            # Look for patterns that might indicate heap access
            if any(heap_hint in instruction.lower() for heap_hint in ['malloc', 'free', 'new', 'delete']):
                heap_operations.append({
                    'operation_type': 'allocation' if any(op in instruction.lower() for op in ['malloc', 'new']) else 'deallocation',
                    'instruction': instruction,
                    'function': access['function']
                })
        
        return heap_operations
    
    # Helper methods for calling convention analysis
    def _analyze_function_prologues(self, assembly_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze function prologue patterns"""
        prologues = []
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            if len(instructions) > 3:
                # Check first few instructions for prologue pattern
                prologue_instructions = instructions[:3]
                
                # Common prologue patterns
                if any('push' in inst.lower() and 'ebp' in inst.lower() for inst in prologue_instructions):
                    prologues.append({
                        'function': func_name,
                        'pattern': 'standard_prologue',
                        'instructions': prologue_instructions,
                        'frame_setup': True
                    })
                elif any('sub' in inst.lower() and 'esp' in inst.lower() for inst in prologue_instructions):
                    prologues.append({
                        'function': func_name,
                        'pattern': 'stack_allocation',
                        'instructions': prologue_instructions,
                        'frame_setup': False
                    })
        
        return prologues
    
    def _analyze_function_epilogues(self, assembly_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze function epilogue patterns"""
        epilogues = []
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            if len(instructions) > 3:
                # Check last few instructions for epilogue pattern
                epilogue_instructions = instructions[-3:]
                
                # Common epilogue patterns
                if any('pop' in inst.lower() and 'ebp' in inst.lower() for inst in epilogue_instructions):
                    epilogues.append({
                        'function': func_name,
                        'pattern': 'standard_epilogue',
                        'instructions': epilogue_instructions,
                        'frame_cleanup': True
                    })
                elif any('add' in inst.lower() and 'esp' in inst.lower() for inst in epilogue_instructions):
                    epilogues.append({
                        'function': func_name,
                        'pattern': 'stack_restoration',
                        'instructions': epilogue_instructions,
                        'frame_cleanup': False
                    })
        
        return epilogues
    
    def _analyze_parameter_passing(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze parameter passing patterns"""
        parameter_analysis = {
            'register_parameters': [],
            'stack_parameters': [],
            'mixed_parameters': []
        }
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            # Look for parameter passing patterns in function calls
            for i, instruction in enumerate(instructions):
                if 'call' in instruction.lower():
                    # Analyze instructions before call for parameter setup
                    param_setup = instructions[max(0, i-5):i]
                    
                    register_params = []
                    stack_params = []
                    
                    for setup_inst in param_setup:
                        if 'push' in setup_inst.lower():
                            stack_params.append(setup_inst)
                        elif any(reg in setup_inst.lower() for reg in ['mov', 'lea']) and \
                             any(param_reg in setup_inst.lower() for param_reg in ['ecx', 'edx', 'r8', 'r9']):
                            register_params.append(setup_inst)
                    
                    if register_params and stack_params:
                        parameter_analysis['mixed_parameters'].append({
                            'function': func_name,
                            'call_instruction': instruction,
                            'register_params': register_params,
                            'stack_params': stack_params
                        })
                    elif register_params:
                        parameter_analysis['register_parameters'].append({
                            'function': func_name,
                            'call_instruction': instruction,
                            'register_params': register_params
                        })
                    elif stack_params:
                        parameter_analysis['stack_parameters'].append({
                            'function': func_name,
                            'call_instruction': instruction,
                            'stack_params': stack_params
                        })
        
        return parameter_analysis
    
    def _classify_calling_convention(self, prologue_patterns: List[Dict[str, Any]], 
                                   parameter_passing: Dict[str, Any]) -> Dict[str, Any]:
        """Classify calling convention based on patterns"""
        convention = {
            'name': 'unknown',
            'confidence': 0.0,
            'stack_cleanup': 'unknown',
            'parameter_order': 'unknown',
            'evidence': []
        }
        
        # Analyze evidence for different calling conventions
        evidence_score = 0
        
        # Check for register parameter usage
        register_params = parameter_passing.get('register_parameters', [])
        stack_params = parameter_passing.get('stack_parameters', [])
        
        if register_params:
            convention['evidence'].append('Uses register parameters')
            evidence_score += 0.3
            
            # Check specific register patterns for x64 calling conventions
            for param_group in register_params:
                for param_inst in param_group.get('register_params', []):
                    if any(reg in param_inst.lower() for reg in ['rcx', 'rdx', 'r8', 'r9']):
                        convention['name'] = 'microsoft_x64'
                        convention['evidence'].append('Microsoft x64 register pattern detected')
                        evidence_score += 0.4
                        break
                    elif any(reg in param_inst.lower() for reg in ['rdi', 'rsi', 'rdx', 'rcx']):
                        convention['name'] = 'system_v_x64'
                        convention['evidence'].append('System V x64 register pattern detected')
                        evidence_score += 0.4
                        break
        
        if stack_params:
            convention['evidence'].append('Uses stack parameters')
            evidence_score += 0.2
            
            if not register_params:
                convention['name'] = 'cdecl'
                convention['stack_cleanup'] = 'caller'
                convention['evidence'].append('Stack-only parameters suggest cdecl')
                evidence_score += 0.3
        
        # Check prologue patterns for additional evidence
        standard_prologues = sum(1 for p in prologue_patterns if p.get('pattern') == 'standard_prologue')
        if standard_prologues > 0:
            convention['evidence'].append(f'{standard_prologues} functions use standard prologue')
            evidence_score += 0.1
        
        convention['confidence'] = min(evidence_score, 1.0)
        
        return convention
    
    def _extract_function_signatures(self, assembly_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract function signatures from assembly analysis"""
        signatures = {}
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            # Analyze function for signature clues
            return_type = 'int'  # Default
            parameters = []
            
            # Look for return patterns
            for instruction in instructions:
                if 'ret' in instruction.lower():
                    # Check if return value is set
                    if any(reg in instruction for reg in ['eax', 'rax']):
                        return_type = 'int'
                    elif 'void' in instruction.lower():
                        return_type = 'void'
            
            # Estimate parameter count from stack usage or register usage
            param_count = 0
            for instruction in instructions:
                if 'push' in instruction.lower() and 'ebp' not in instruction.lower():
                    param_count += 1
                elif any(reg in instruction.lower() for reg in ['ecx', 'edx', 'r8', 'r9']):
                    param_count = max(param_count, 1)
            
            # Generate signature
            if param_count == 0:
                signatures[func_name] = f"{return_type} {func_name}(void)"
            else:
                params = ', '.join([f"int param{i}" for i in range(min(param_count, 4))])
                signatures[func_name] = f"{return_type} {func_name}({params})"
        
        return signatures
    
    # Helper methods for ASM-to-C mapping
    def _map_control_flow_to_c(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map assembly control flow to C constructs"""
        control_flow_map = {
            'loops': [],
            'conditionals': [],
            'switches': [],
            'function_calls': []
        }
        
        # Analyze control flow patterns for each function
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            # Map loops
            loops = self._detect_loops(instructions)
            for loop in loops:
                c_construct = self._map_loop_to_c(loop, instructions)
                control_flow_map['loops'].append({
                    'function': func_name,
                    'assembly_pattern': loop,
                    'c_construct': c_construct
                })
            
            # Map conditionals
            conditionals = self._detect_conditionals(instructions)
            for conditional in conditionals:
                c_construct = self._map_conditional_to_c(conditional, instructions)
                control_flow_map['conditionals'].append({
                    'function': func_name,
                    'assembly_pattern': conditional,
                    'c_construct': c_construct
                })
            
            # Map function calls
            calls = self._detect_function_calls(instructions)
            for call in calls:
                c_construct = self._map_function_call_to_c(call)
                control_flow_map['function_calls'].append({
                    'function': func_name,
                    'assembly_pattern': call,
                    'c_construct': c_construct
                })
        
        return control_flow_map
    
    def _map_loop_to_c(self, loop_pattern: Dict[str, Any], instructions: List[str]) -> str:
        """Map assembly loop pattern to C construct"""
        loop_type = loop_pattern.get('type', 'unknown')
        
        if loop_type == 'for_loop':
            return "for (int i = 0; i < count; i++) { /* loop body */ }"
        elif loop_type == 'while_loop':
            condition = self._extract_loop_condition(loop_pattern, instructions)
            return f"while ({condition}) {{ /* loop body */ }}"
        else:
            return "/* Unknown loop pattern */"
    
    def _extract_loop_condition(self, loop_pattern: Dict[str, Any], instructions: List[str]) -> str:
        """Extract loop condition from assembly pattern"""
        instruction = loop_pattern.get('instruction', '')
        
        if 'jne' in instruction.lower():
            return "condition != 0"
        elif 'je' in instruction.lower():
            return "condition == 0"
        elif 'jl' in instruction.lower():
            return "variable < limit"
        elif 'jg' in instruction.lower():
            return "variable > limit"
        else:
            return "condition"
    
    def _map_conditional_to_c(self, conditional_pattern: Dict[str, Any], instructions: List[str]) -> str:
        """Map assembly conditional to C construct"""
        condition_type = conditional_pattern.get('condition', 'unknown')
        
        condition_map = {
            'equal_zero': 'if (variable == 0)',
            'not_equal_zero': 'if (variable != 0)',
            'less_than': 'if (variable < value)',
            'greater_than': 'if (variable > value)',
            'less_equal': 'if (variable <= value)',
            'greater_equal': 'if (variable >= value)'
        }
        
        c_condition = condition_map.get(condition_type, 'if (condition)')
        return f"{c_condition} {{ /* conditional body */ }}"
    
    def _map_function_call_to_c(self, call_pattern: Dict[str, Any]) -> str:
        """Map assembly function call to C construct"""
        target = call_pattern.get('target', 'unknown_function')
        call_type = call_pattern.get('call_type', 'direct_call')
        
        if call_type == 'direct_call':
            # Clean up the target name
            clean_target = target.replace('@', '').replace('$', '').replace('[', '').replace(']', '')
            return f"{clean_target}(/* parameters */);"
        else:
            return "(*function_pointer)(/* parameters */);"
    
    def _map_data_types_to_c(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map assembly data operations to C data types"""
        data_type_map = {
            'integers': [],
            'pointers': [],
            'arrays': [],
            'structures': []
        }
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            for instruction in instructions:
                instruction_lower = instruction.lower()
                
                # Detect integer operations
                if any(op in instruction_lower for op in ['add', 'sub', 'mul', 'div', 'inc', 'dec']):
                    data_type_map['integers'].append({
                        'function': func_name,
                        'instruction': instruction,
                        'inferred_type': self._infer_integer_type(instruction)
                    })
                
                # Detect pointer operations
                if '[' in instruction and ']' in instruction:
                    data_type_map['pointers'].append({
                        'function': func_name,
                        'instruction': instruction,
                        'pointer_type': self._infer_pointer_type(instruction)
                    })
                
                # Detect array-like access patterns
                if '+' in instruction and any(reg in instruction_lower for reg in ['esi', 'edi', 'ecx']):
                    data_type_map['arrays'].append({
                        'function': func_name,
                        'instruction': instruction,
                        'array_type': 'int[]'  # Simplified
                    })
        
        return data_type_map
    
    def _infer_integer_type(self, instruction: str) -> str:
        """Infer integer type from instruction"""
        if any(size_hint in instruction.lower() for size_hint in ['byte', 'db', 'al', 'bl']):
            return 'char'
        elif any(size_hint in instruction.lower() for size_hint in ['word', 'dw', 'ax', 'bx']):
            return 'short'
        elif any(size_hint in instruction.lower() for size_hint in ['dword', 'dd', 'eax', 'ebx']):
            return 'int'
        elif any(size_hint in instruction.lower() for size_hint in ['qword', 'dq', 'rax', 'rbx']):
            return 'long long'
        else:
            return 'int'  # Default
    
    def _infer_pointer_type(self, instruction: str) -> str:
        """Infer pointer type from instruction"""
        if any(hint in instruction.lower() for hint in ['byte ptr', 'db']):
            return 'char*'
        elif any(hint in instruction.lower() for hint in ['word ptr', 'dw']):
            return 'short*'
        elif any(hint in instruction.lower() for hint in ['dword ptr', 'dd']):
            return 'int*'
        elif any(hint in instruction.lower() for hint in ['qword ptr', 'dq']):
            return 'long long*'
        else:
            return 'void*'  # Generic pointer
    
    def _map_functions_to_c(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Map assembly functions to C function constructs"""
        function_map = {}
        
        for func_name, func_data in assembly_data.items():
            instructions = func_data.get('instructions', [])
            
            # Analyze function characteristics
            function_analysis = {
                'name': func_name,
                'estimated_parameters': 0,
                'return_type': 'int',
                'complexity': len(instructions),
                'calls_other_functions': False,
                'uses_stack': False
            }
            
            # Count parameters and analyze function behavior
            for instruction in instructions:
                instruction_lower = instruction.lower()
                
                if 'call' in instruction_lower:
                    function_analysis['calls_other_functions'] = True
                
                if any(stack_op in instruction_lower for stack_op in ['push', 'pop', 'esp', 'rsp']):
                    function_analysis['uses_stack'] = True
                
                if 'ret' in instruction_lower and 'eax' in instruction_lower:
                    function_analysis['return_type'] = 'int'
                elif 'ret' in instruction_lower and 'void' in instruction_lower:
                    function_analysis['return_type'] = 'void'
            
            function_map[func_name] = function_analysis
        
        return function_map
    
    def _generate_c_patterns(self, control_flow_mapping: Dict[str, Any],
                           data_type_mapping: Dict[str, Any],
                           function_mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Generate C code patterns from mappings"""
        c_patterns = {
            'function_templates': {},
            'control_structures': [],
            'data_declarations': [],
            'complete_functions': {}
        }
        
        # Generate function templates
        for func_name, func_data in function_mapping.items():
            return_type = func_data.get('return_type', 'int')
            param_count = func_data.get('estimated_parameters', 0)
            
            if param_count == 0:
                template = f"{return_type} {func_name}(void) {{\n    /* Function body */\n    return 0;\n}}"
            else:
                params = ', '.join([f"int param{i}" for i in range(param_count)])
                template = f"{return_type} {func_name}({params}) {{\n    /* Function body */\n    return 0;\n}}"
            
            c_patterns['function_templates'][func_name] = template
        
        # Generate control structure patterns
        loops = control_flow_mapping.get('loops', [])
        for loop in loops:
            c_patterns['control_structures'].append(loop.get('c_construct', ''))
        
        conditionals = control_flow_mapping.get('conditionals', [])
        for conditional in conditionals:
            c_patterns['control_structures'].append(conditional.get('c_construct', ''))
        
        # Generate data declarations
        integers = data_type_mapping.get('integers', [])
        for int_data in integers:
            int_type = int_data.get('inferred_type', 'int')
            c_patterns['data_declarations'].append(f"{int_type} variable;")
        
        pointers = data_type_mapping.get('pointers', [])
        for ptr_data in pointers:
            ptr_type = ptr_data.get('pointer_type', 'void*')
            c_patterns['data_declarations'].append(f"{ptr_type} pointer;")
        
        return c_patterns
    
    def _calculate_mapping_confidence(self, c_patterns: Dict[str, Any]) -> float:
        """Calculate confidence score for ASM-to-C mapping"""
        confidence_factors = []
        
        # Factor in number of mapped functions
        function_count = len(c_patterns.get('function_templates', {}))
        if function_count > 0:
            confidence_factors.append(min(function_count * 0.1, 0.4))
        
        # Factor in control structures
        control_count = len(c_patterns.get('control_structures', []))
        if control_count > 0:
            confidence_factors.append(min(control_count * 0.05, 0.3))
        
        # Factor in data declarations
        data_count = len(c_patterns.get('data_declarations', []))
        if data_count > 0:
            confidence_factors.append(min(data_count * 0.02, 0.3))
        
        return min(sum(confidence_factors), 1.0)
    
    def _generate_reconstruction_hints(self, c_patterns: Dict[str, Any]) -> List[str]:
        """Generate hints for code reconstruction"""
        hints = []
        
        function_count = len(c_patterns.get('function_templates', {}))
        control_count = len(c_patterns.get('control_structures', []))
        data_count = len(c_patterns.get('data_declarations', []))
        
        if function_count > 5:
            hints.append("Multiple functions detected - consider modular design")
        
        if control_count > 10:
            hints.append("Complex control flow - consider using switch statements or function tables")
        
        if data_count > 20:
            hints.append("Many data declarations - consider using structures")
        
        if function_count == 0:
            hints.append("No clear functions detected - may need manual analysis")
        
        hints.append(f"Detected {function_count} functions, {control_count} control structures, {data_count} data elements")
        
        return hints