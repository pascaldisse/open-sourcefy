"""
Agent 2: Architecture Analysis
Performs detailed architecture-specific analysis and optimization detection.
"""

from typing import Dict, Any, List
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent2_ArchAnalysis(BaseAgent):
    """Agent 2: Architecture-specific analysis"""
    
    def __init__(self):
        super().__init__(
            agent_id=2,
            name="ArchAnalysis",
            dependencies=[1]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute architecture analysis"""
        # Get data from Agent 1
        agent1_result = context['agent_results'].get(1)
        if not agent1_result or agent1_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 1 (BinaryDiscovery) did not complete successfully"
            )

        try:
            binary_info = agent1_result.data
            arch_analysis = self._analyze_architecture(binary_info)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=arch_analysis,
                metadata={
                    'depends_on': [1],
                    'analysis_type': 'architecture_analysis'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Architecture analysis failed: {str(e)}"
            )

    def _analyze_architecture(self, binary_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed architecture analysis"""
        arch_info = binary_info.get('architecture', {})
        architecture = arch_info.get('architecture', 'Unknown')
        
        analysis = {
            'architecture': architecture,
            'calling_conventions': self._get_calling_conventions(architecture),
            'register_usage': self._get_register_usage(architecture),
            'instruction_sets': self._get_instruction_sets(architecture),
            'abi_info': self._get_abi_info(architecture),
            'stack_analysis': self._analyze_stack_usage(architecture),
            'optimization_hints': self._detect_optimization_hints(binary_info),
            'compiler_hints': self._detect_compiler_hints(binary_info)
        }
        
        return analysis

    def _get_calling_conventions(self, architecture: str) -> List[str]:
        """Get possible calling conventions for the architecture"""
        conventions = {
            'x86': ['cdecl', 'stdcall', 'fastcall', 'thiscall'],
            'x64': ['Microsoft x64', 'System V AMD64 ABI'],
            'ARM': ['AAPCS', 'AAPCS-VFP'],
            'ARM64': ['AAPCS64'],
            'Unknown': ['unknown']
        }
        return conventions.get(architecture, ['unknown'])

    def _get_register_usage(self, architecture: str) -> Dict[str, Any]:
        """Get register usage patterns for the architecture"""
        register_info = {
            'x86': {
                'general_purpose': ['eax', 'ebx', 'ecx', 'edx', 'esi', 'edi'],
                'special': ['esp', 'ebp', 'eip'],
                'calling_convention_registers': {
                    'return_value': 'eax',
                    'first_param': 'stack',
                    'second_param': 'stack'
                }
            },
            'x64': {
                'general_purpose': ['rax', 'rbx', 'rcx', 'rdx', 'rsi', 'rdi', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15'],
                'special': ['rsp', 'rbp', 'rip'],
                'calling_convention_registers': {
                    'return_value': 'rax',
                    'first_param': 'rcx',  # Windows x64
                    'second_param': 'rdx'
                }
            },
            'ARM': {
                'general_purpose': ['r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10', 'r11', 'r12'],
                'special': ['sp', 'lr', 'pc'],
                'calling_convention_registers': {
                    'return_value': 'r0',
                    'first_param': 'r0',
                    'second_param': 'r1'
                }
            },
            'ARM64': {
                'general_purpose': ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15'],
                'special': ['sp', 'lr', 'pc'],
                'calling_convention_registers': {
                    'return_value': 'x0',
                    'first_param': 'x0',
                    'second_param': 'x1'
                }
            }
        }
        return register_info.get(architecture, {'unknown': True})

    def _get_instruction_sets(self, architecture: str) -> List[str]:
        """Get available instruction sets for the architecture"""
        instruction_sets = {
            'x86': ['8086', 'MMX', 'SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE4.1', 'SSE4.2'],
            'x64': ['x86-64', 'SSE', 'SSE2', 'SSE3', 'SSSE3', 'SSE4.1', 'SSE4.2', 'AVX', 'AVX2'],
            'ARM': ['ARMv7', 'Thumb', 'Thumb-2', 'NEON', 'VFP'],
            'ARM64': ['ARMv8-A', 'NEON', 'SVE'],
            'Unknown': ['unknown']
        }
        return instruction_sets.get(architecture, ['unknown'])

    def _detect_optimization_hints(self, binary_info: Dict[str, Any]) -> Dict[str, Any]:
        """Detect optimization hints from binary characteristics"""
        optimization_hints = {
            'likely_optimization_level': 'unknown',
            'optimization_indicators': [],
            'confidence': 0.0
        }
        
        # Analyze PE-specific optimization hints
        if 'pe_analysis' in binary_info and 'error' not in binary_info['pe_analysis']:
            pe_data = binary_info['pe_analysis']
            
            # Check section characteristics for optimization hints
            sections = pe_data.get('sections', [])
            text_sections = [s for s in sections if '.text' in s.get('name', '')]
            
            if text_sections:
                text_section = text_sections[0]
                entropy = text_section.get('entropy', 0.0)
                
                # High entropy often indicates optimized/packed code
                if entropy > 7.0:
                    optimization_hints['optimization_indicators'].append('high_code_entropy')
                    optimization_hints['likely_optimization_level'] = 'O2_or_higher'
                    optimization_hints['confidence'] += 0.3
                elif entropy > 6.0:
                    optimization_hints['optimization_indicators'].append('moderate_code_entropy')
                    optimization_hints['likely_optimization_level'] = 'O1'
                    optimization_hints['confidence'] += 0.2
            
            # Check import table for optimization hints
            imports = pe_data.get('imports', [])
            runtime_functions = ['_CxxThrowException', '__security_check_cookie', '_guard_check_icall']
            
            for import_dll in imports:
                functions = import_dll.get('functions', [])
                for func in functions:
                    func_name = func.get('name', '') or ''
                    if func_name and any(runtime_func in func_name for runtime_func in runtime_functions):
                        optimization_hints['optimization_indicators'].append('runtime_checks_present')
                        optimization_hints['confidence'] += 0.1
        
        # Analyze ELF-specific optimization hints
        elif 'elf_analysis' in binary_info and 'error' not in binary_info['elf_analysis']:
            elf_data = binary_info['elf_analysis']
            
            # Check for debug information (indicates less optimization)
            sections = elf_data.get('sections', [])
            debug_sections = [s for s in sections if s.get('name', '').startswith('.debug')]
            
            if debug_sections:
                optimization_hints['optimization_indicators'].append('debug_info_present')
                optimization_hints['likely_optimization_level'] = 'O0_or_O1'
                optimization_hints['confidence'] += 0.4
            else:
                optimization_hints['optimization_indicators'].append('no_debug_info')
                optimization_hints['likely_optimization_level'] = 'O2_or_higher'
                optimization_hints['confidence'] += 0.3
        
        # Normalize confidence
        optimization_hints['confidence'] = min(optimization_hints['confidence'], 1.0)
        
        return optimization_hints

    def _detect_compiler_hints(self, binary_info: Dict[str, Any]) -> Dict[str, Any]:
        """Detect compiler-specific hints"""
        compiler_hints = {
            'likely_compiler': 'unknown',
            'compiler_version': 'unknown',
            'compiler_indicators': [],
            'confidence': 0.0
        }
        
        # Analyze PE-specific compiler hints
        if 'pe_analysis' in binary_info and 'error' not in binary_info['pe_analysis']:
            pe_data = binary_info['pe_analysis']
            
            # Check imports for compiler-specific runtime libraries
            imports = pe_data.get('imports', [])
            for import_dll in imports:
                dll_name = import_dll.get('dll', '').lower()
                
                if 'msvcr' in dll_name or 'msvcp' in dll_name:
                    compiler_hints['likely_compiler'] = 'MSVC'
                    compiler_hints['compiler_indicators'].append(f'imports_{dll_name}')
                    compiler_hints['confidence'] += 0.4
                    
                    # Try to extract version from DLL name
                    if 'msvcr120' in dll_name:
                        compiler_hints['compiler_version'] = 'Visual Studio 2013'
                    elif 'msvcr140' in dll_name:
                        compiler_hints['compiler_version'] = 'Visual Studio 2015+'
                        
                elif 'libgcc' in dll_name or 'libstdc++' in dll_name:
                    compiler_hints['likely_compiler'] = 'GCC'
                    compiler_hints['compiler_indicators'].append(f'imports_{dll_name}')
                    compiler_hints['confidence'] += 0.4
            
            # Check version info for compiler signatures
            version_info = pe_data.get('version_info', {})
            for key, value in version_info.items():
                if 'microsoft' in str(value).lower():
                    compiler_hints['likely_compiler'] = 'MSVC'
                    compiler_hints['compiler_indicators'].append('version_info_microsoft')
                    compiler_hints['confidence'] += 0.2
                elif 'gcc' in str(value).lower() or 'gnu' in str(value).lower():
                    compiler_hints['likely_compiler'] = 'GCC'
                    compiler_hints['compiler_indicators'].append('version_info_gnu')
                    compiler_hints['confidence'] += 0.2
        
        # Analyze ELF-specific compiler hints
        elif 'elf_analysis' in binary_info and 'error' not in binary_info['elf_analysis']:
            elf_data = binary_info['elf_analysis']
            
            # Check dynamic section for compiler hints
            symbols = elf_data.get('symbols', {})
            dynamic_symbols = symbols.get('dynamic', [])
            
            for symbol in dynamic_symbols:
                symbol_name = symbol.get('name', '') or ''
                if symbol_name and ('__gxx_personality' in symbol_name or '__gnu_' in symbol_name):
                    compiler_hints['likely_compiler'] = 'GCC'
                    compiler_hints['compiler_indicators'].append('gnu_symbols')
                    compiler_hints['confidence'] += 0.3
                elif symbol_name and ('__clang_' in symbol_name or '_ZN' in symbol_name):
                    compiler_hints['likely_compiler'] = 'Clang'
                    compiler_hints['compiler_indicators'].append('clang_symbols')
                    compiler_hints['confidence'] += 0.3
            
            # Check sections for compiler-specific sections
            sections = elf_data.get('sections', [])
            for section in sections:
                section_name = section.get('name', '')
                if '.gcc_except_table' in section_name:
                    compiler_hints['likely_compiler'] = 'GCC'
                    compiler_hints['compiler_indicators'].append('gcc_exception_table')
                    compiler_hints['confidence'] += 0.2
                elif '.llvm' in section_name:
                    compiler_hints['likely_compiler'] = 'Clang'
                    compiler_hints['compiler_indicators'].append('llvm_section')
                    compiler_hints['confidence'] += 0.2
        
        # Normalize confidence
        compiler_hints['confidence'] = min(compiler_hints['confidence'], 1.0)
        
        return compiler_hints

    def _get_abi_info(self, architecture: str) -> Dict[str, Any]:
        """Get ABI (Application Binary Interface) information"""
        abi_info = {
            'x86': {
                'pointer_size': 4,
                'alignment': 4,
                'stack_alignment': 4,
                'calling_convention': 'cdecl',
                'return_value_location': 'eax'
            },
            'x64': {
                'pointer_size': 8,
                'alignment': 8,
                'stack_alignment': 16,
                'calling_convention': 'Microsoft x64',
                'return_value_location': 'rax'
            },
            'ARM': {
                'pointer_size': 4,
                'alignment': 4,
                'stack_alignment': 8,
                'calling_convention': 'AAPCS',
                'return_value_location': 'r0'
            },
            'ARM64': {
                'pointer_size': 8,
                'alignment': 8,
                'stack_alignment': 16,
                'calling_convention': 'AAPCS64',
                'return_value_location': 'x0'
            }
        }
        return abi_info.get(architecture, {'unknown': True})

    def _analyze_stack_usage(self, architecture: str) -> Dict[str, Any]:
        """Analyze stack usage patterns"""
        stack_info = {
            'stack_grows_down': True,  # Most architectures
            'frame_pointer_register': None,
            'stack_pointer_register': None,
            'typical_frame_setup': []
        }
        
        if architecture == 'x86':
            stack_info.update({
                'frame_pointer_register': 'ebp',
                'stack_pointer_register': 'esp',
                'typical_frame_setup': ['push ebp', 'mov ebp, esp']
            })
        elif architecture == 'x64':
            stack_info.update({
                'frame_pointer_register': 'rbp',
                'stack_pointer_register': 'rsp',
                'typical_frame_setup': ['push rbp', 'mov rbp, rsp']
            })
        elif architecture in ['ARM', 'ARM64']:
            stack_info.update({
                'frame_pointer_register': 'fp',
                'stack_pointer_register': 'sp',
                'typical_frame_setup': ['push {fp, lr}', 'mov fp, sp']
            })
        
        return stack_info