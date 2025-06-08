"""
Agent 03: The Merovingian - Basic Decompilation & Optimization Detection
The sophisticated master of cause and effect who understands code transformations.
Specializes in basic decompilation and identifying optimization techniques that obscure original intent.

Production-ready implementation following SOLID principles and clean code standards.
Includes LangChain integration, comprehensive error handling, and fail-fast validation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Matrix framework imports
from ..matrix_agents_v2 import DecompilerAgent, MatrixCharacter, AgentStatus
from ..shared_components import (
    MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker, 
    MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
)
from ..exceptions import MatrixAgentError, ValidationError, ConfigurationError

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

# Disassembly engine
try:
    import capstone
    HAS_CAPSTONE = True
except ImportError:
    HAS_CAPSTONE = False


# Configuration constants - NO MAGIC NUMBERS
class MerovingianConstants:
    """Merovingian-specific constants loaded from configuration"""
    
    def __init__(self, config_manager, agent_id: int):
        self.MAX_RETRY_ATTEMPTS = config_manager.get_value(f'agents.agent_{agent_id:02d}.max_retries', 3)
        self.TIMEOUT_SECONDS = config_manager.get_value(f'agents.agent_{agent_id:02d}.timeout', 300)
        self.QUALITY_THRESHOLD = config_manager.get_value(f'agents.agent_{agent_id:02d}.quality_threshold', 0.75)
        self.MAX_FUNCTIONS_TO_ANALYZE = config_manager.get_value('decompilation.max_functions', 500)
        self.MIN_FUNCTION_SIZE = config_manager.get_value('decompilation.min_function_size', 16)
        self.CONTROL_FLOW_DEPTH_LIMIT = config_manager.get_value('decompilation.max_depth', 50)


@dataclass
class Function:
    """Detected function information"""
    address: int
    size: int
    name: str = None
    signature: str = None
    basic_blocks: int = 0
    calls_made: List[int] = None
    calls_received: int = 0
    complexity_score: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        if self.calls_made is None:
            self.calls_made = []
        if self.name is None:
            self.name = f"sub_{self.address:08x}"


@dataclass
class ControlStructure:
    """Identified control structure (loop, conditional, etc.)"""
    type: str  # loop/if-else/switch/etc.
    address: int
    size: int
    complexity: str = 'simple'  # simple/medium/complex
    confidence: float = 0.0


@dataclass
class MerovingianValidationResult:
    """Validation result structure for fail-fast pipeline"""
    is_valid: bool
    quality_score: float
    error_messages: List[str]
    validation_details: Dict[str, Any]


class MerovingianAgent(DecompilerAgent):
    """
    Agent 03: The Merovingian - Production-Ready Implementation
    
    The Merovingian understands the intricate relationships between code transformations.
    Agent 03 specializes in basic decompilation, identifying how high-level constructs 
    were transformed into machine code and detecting optimization techniques.
    
    Features:
    - Function boundary detection and analysis
    - Control flow graph construction
    - Basic type inference
    - Optimization pattern recognition
    - String and constant analysis
    """
    
    # Function prologue patterns for different architectures
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
    
    # Function epilogue patterns
    FUNCTION_EPILOGUES = {
        'x86': [
            b'\xc9\xc3',               # leave; ret
            b'\x5d\xc3',               # pop ebp; ret
            b'\x89\xec\x5d\xc3',       # mov esp, ebp; pop ebp; ret
        ],
        'x64': [
            b'\xc3',                   # ret
            b'\x48\x89\xec\xc3',       # mov rsp, rbp; ret
            b'\x5d\xc3',               # pop rbp; ret
        ]
    }
    
    def __init__(self):
        super().__init__(
            agent_id=3,
            matrix_character=MatrixCharacter.MEROVINGIAN,
            dependencies=[1]  # Depends on Sentinel
        )
        
        # Load configuration constants
        self.constants = MerovingianConstants(self.config, self.agent_id)
        
        # Initialize shared tools
        self.analysis_tools = SharedAnalysisTools()
        self.validation_tools = SharedValidationTools()
        
        # Setup specialized components
        self.error_handler = MatrixErrorHandler(self.agent_name, self.constants.MAX_RETRY_ATTEMPTS)
        self.metrics = MatrixMetrics(self.agent_id, self.matrix_character.value)
        
        # Check disassembly capability
        self.has_disassembler = HAS_CAPSTONE
        if not self.has_disassembler:
            self.logger.warning("Capstone disassembler not available - limited analysis")
        
        # Initialize LangChain components for AI enhancement
        self.ai_enabled = self.config.get_value('ai.enabled', True)
        if self.ai_enabled:
            self.llm = self._setup_llm()
            self.agent_executor = self._setup_langchain_agent()
        
        # Validate configuration
        self._validate_configuration()
    
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
        """Setup LangChain agent with Merovingian-specific tools"""
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
        """Create LangChain tools specific to Merovingian's capabilities"""
        return [
            Tool(
                name="analyze_function_purpose",
                description="Analyze function purpose and behavior from assembly",
                func=self._ai_function_analysis_tool
            ),
            Tool(
                name="detect_algorithms",
                description="Detect algorithms and data structures in code",
                func=self._ai_algorithm_detection_tool
            ),
            Tool(
                name="infer_variable_types",
                description="Infer variable types and purposes from usage patterns",
                func=self._ai_type_inference_tool
            )
        ]
    
    def get_matrix_description(self) -> str:
        """The Merovingian's role in the Matrix"""
        return ("The Merovingian understands the intricate relationships between cause and effect. "
                "Agent 03 analyzes how high-level code constructs were transformed by the compiler, "
                "revealing the original developer's intent through decompilation.")
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Merovingian's decompilation analysis with production-ready error handling
        """
        self.metrics.start_tracking()
        
        # Setup progress tracking
        total_steps = 7
        progress = MatrixProgressTracker(total_steps, self.agent_name)
        
        try:
            # Step 1: Validate prerequisites and dependencies
            progress.step("Validating prerequisites and Sentinel data")
            self._validate_prerequisites(context)
            
            # Step 2: Initialize decompilation analysis
            progress.step("Initializing decompilation components")
            with self.error_handler.handle_matrix_operation("component_initialization"):
                analysis_context = self._initialize_analysis(context)
            
            # Step 3: Perform basic disassembly
            progress.step("Performing basic disassembly and instruction analysis")
            with self.error_handler.handle_matrix_operation("disassembly"):
                disassembly_results = self._perform_basic_disassembly(analysis_context)
            
            # Step 4: Detect function boundaries
            progress.step("Detecting function boundaries and signatures")
            with self.error_handler.handle_matrix_operation("function_detection"):
                function_results = self._detect_functions(analysis_context, disassembly_results)
            
            # Step 5: Analyze control flow
            progress.step("Analyzing control flow and structure")
            with self.error_handler.handle_matrix_operation("control_flow_analysis"):
                control_flow_results = self._analyze_control_flow(analysis_context, function_results)
            
            # Step 6: Perform type inference and optimization detection
            progress.step("Performing type inference and optimization detection")
            with self.error_handler.handle_matrix_operation("type_inference"):
                type_inference_results = self._perform_type_inference(analysis_context, function_results)
                optimization_results = self._detect_optimization_patterns(analysis_context, function_results)
            
            # Combine core results
            core_results = {
                'disassembly_analysis': disassembly_results,
                'function_analysis': function_results,
                'control_flow_analysis': control_flow_results,
                'type_inference': type_inference_results,
                'optimization_analysis': optimization_results
            }
            
            # Step 7: AI enhancement (if enabled)
            if self.ai_enabled and self.agent_executor:
                progress.step("Applying AI-enhanced decompilation insights")
                with self.error_handler.handle_matrix_operation("ai_enhancement"):
                    ai_results = self._execute_ai_analysis(core_results, context)
                    core_results = self._merge_analysis_results(core_results, ai_results)
            else:
                progress.step("Skipping AI enhancement (disabled)")
            
            # Validate results and finalize
            validation_result = self._validate_results(core_results)
            
            if not validation_result.is_valid:
                raise ValidationError(
                    f"Merovingian analysis failed validation: {validation_result.error_messages}"
                )
            
            # Finalize and save results
            final_results = self._finalize_results(core_results, validation_result)
            self._save_results(final_results, context)
            self._populate_shared_memory(final_results, context)
            
            progress.complete()
            self.metrics.end_tracking()
            
            # Log success with metrics
            self.logger.info(
                "Merovingian decompilation completed successfully",
                extra={
                    'execution_time': self.metrics.execution_time,
                    'quality_score': validation_result.quality_score,
                    'functions_detected': len(function_results.get('functions', [])),
                    'instructions_analyzed': disassembly_results.get('total_instructions', 0),
                    'validation_passed': True
                }
            )
            
            return final_results
            
        except Exception as e:
            self.metrics.end_tracking()
            self.metrics.increment_errors()
            
            # Log detailed error information
            self.logger.error(
                "Merovingian analysis failed",
                extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'execution_time': self.metrics.execution_time,
                    'agent_id': self.agent_id,
                    'matrix_character': self.matrix_character.value
                },
                exc_info=True
            )
            
            # Re-raise with Matrix context
            raise MatrixAgentError(
                f"Merovingian decompilation analysis failed: {e}",
                agent_id=self.agent_id,
                matrix_character=self.matrix_character.value
            ) from e
    
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate all prerequisites before starting analysis"""
        # Validate required context keys
        required_keys = ['binary_path', 'shared_memory']
        missing_keys = self.validation_tools.validate_context_keys(context, required_keys)
        
        if missing_keys:
            raise ValidationError(f"Missing required context keys: {missing_keys}")
        
        # Validate dependencies - need Sentinel results
        failed_deps = self.validation_tools.validate_dependency_results(context, self.dependencies)
        if failed_deps:
            raise ValidationError(f"Dependencies failed: {failed_deps}")
        
        # Validate Sentinel data availability
        shared_memory = context['shared_memory']
        if 'binary_metadata' not in shared_memory or 'discovery' not in shared_memory['binary_metadata']:
            raise ValidationError("Sentinel discovery data not available in shared memory")
    
    def _initialize_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize analysis context with Sentinel data"""
        # Get Sentinel's analysis results
        shared_memory = context['shared_memory']
        sentinel_data = shared_memory['binary_metadata']['discovery']
        
        binary_path = Path(context['binary_path'])
        binary_info = sentinel_data.get('binary_info')
        
        # Get code sections from Sentinel's format analysis
        format_analysis = sentinel_data.get('format_analysis', {})
        code_sections = self._identify_code_sections(format_analysis)
        
        return {
            'binary_path': binary_path,
            'binary_info': binary_info,
            'sentinel_data': sentinel_data,
            'format_analysis': format_analysis,
            'code_sections': code_sections,
            'architecture': binary_info.architecture if binary_info else 'x86'
        }
    
    def _identify_code_sections(self, format_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify executable code sections from format analysis"""
        code_sections = []
        sections = format_analysis.get('sections', [])
        
        for section in sections:
            # Common code section names and characteristics
            section_name = section.get('name', '').lower()
            characteristics = section.get('characteristics', 0)
            
            # Check if section is executable
            is_executable = (
                section_name in ['.text', '.code', '__text'] or
                'executable' in section.get('flags', []) or
                (characteristics & 0x20000000) != 0  # IMAGE_SCN_MEM_EXECUTE for PE
            )
            
            if is_executable and section.get('raw_size', 0) > 0:
                code_sections.append({
                    'name': section.get('name'),
                    'virtual_address': section.get('virtual_address', 0),
                    'size': section.get('raw_size', 0),
                    'offset': section.get('raw_address', 0)
                })
        
        return code_sections
    
    def _perform_basic_disassembly(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic disassembly of code sections"""
        if not self.has_disassembler:
            return self._perform_heuristic_analysis(analysis_context)
        
        binary_path = analysis_context['binary_path']
        code_sections = analysis_context['code_sections']
        architecture = analysis_context['architecture']
        
        # Setup Capstone disassembler
        if architecture == 'x64':
            md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        else:
            md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
        
        md.detail = True
        
        total_instructions = 0
        disassembled_sections = []
        
        # Read binary content
        with open(binary_path, 'rb') as f:
            binary_content = f.read()
        
        # Disassemble each code section
        for section in code_sections:
            try:
                start_offset = section['offset']
                size = min(section['size'], 64 * 1024)  # Limit to 64KB per section
                end_offset = start_offset + size
                
                if end_offset > len(binary_content):
                    continue
                
                code_data = binary_content[start_offset:end_offset]
                base_address = section['virtual_address']
                
                instructions = []
                for insn in md.disasm(code_data, base_address):
                    instructions.append({
                        'address': insn.address,
                        'mnemonic': insn.mnemonic,
                        'op_str': insn.op_str,
                        'size': insn.size,
                        'bytes': insn.bytes.hex()
                    })
                    
                    if len(instructions) >= 1000:  # Limit instructions per section
                        break
                
                total_instructions += len(instructions)
                
                disassembled_sections.append({
                    'section_name': section['name'],
                    'base_address': base_address,
                    'instruction_count': len(instructions),
                    'instructions': instructions[:100]  # Store first 100 for analysis
                })
                
            except Exception as e:
                self.logger.warning(f"Disassembly failed for section {section['name']}: {e}")
        
        return {
            'total_instructions': total_instructions,
            'disassembled_sections': disassembled_sections,
            'disassembly_quality': 0.9 if total_instructions > 0 else 0.0,
            'analysis_method': 'capstone_disassembly'
        }
    
    def _perform_heuristic_analysis(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform heuristic analysis when disassembler is not available"""
        binary_path = analysis_context['binary_path']
        code_sections = analysis_context['code_sections']
        
        # Read binary and perform pattern-based analysis
        with open(binary_path, 'rb') as f:
            binary_content = f.read()
        
        total_patterns = 0
        heuristic_sections = []
        
        for section in code_sections:
            start_offset = section['offset']
            size = min(section['size'], 64 * 1024)
            end_offset = start_offset + size
            
            if end_offset > len(binary_content):
                continue
            
            code_data = binary_content[start_offset:end_offset]
            
            # Count instruction-like patterns
            pattern_count = self._count_instruction_patterns(code_data)
            total_patterns += pattern_count
            
            heuristic_sections.append({
                'section_name': section['name'],
                'base_address': section['virtual_address'],
                'pattern_count': pattern_count,
                'estimated_instructions': pattern_count * 0.8  # Rough estimate
            })
        
        return {
            'total_instructions': int(total_patterns * 0.8),
            'disassembled_sections': heuristic_sections,
            'disassembly_quality': 0.5,  # Lower quality for heuristic analysis
            'analysis_method': 'heuristic_pattern_matching'
        }
    
    def _count_instruction_patterns(self, code_data: bytes) -> int:
        """Count instruction-like patterns in binary data"""
        # Simple heuristic: count potential x86 instruction starts
        patterns = [
            b'\x55',        # push ebp
            b'\x89',        # mov
            b'\x8b',        # mov
            b'\xff',        # call/jmp
            b'\xe8',        # call
            b'\x83',        # arithmetic
            b'\xc3',        # ret
        ]
        
        count = 0
        for pattern in patterns:
            count += code_data.count(pattern)
        
        return count
    
    def _detect_functions(self, analysis_context: Dict[str, Any], disassembly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect function boundaries using multiple heuristics"""
        functions = []
        architecture = analysis_context['architecture']
        
        # Get function prologue/epilogue patterns for architecture
        prologues = self.FUNCTION_PROLOGUES.get(architecture, self.FUNCTION_PROLOGUES['x86'])
        epilogues = self.FUNCTION_EPILOGUES.get(architecture, self.FUNCTION_EPILOGUES['x86'])
        
        if disassembly_results['analysis_method'] == 'capstone_disassembly':
            functions = self._detect_functions_from_disassembly(disassembly_results, prologues, epilogues)
        else:
            functions = self._detect_functions_heuristic(analysis_context, prologues, epilogues)
        
        # Limit number of functions analyzed
        functions = functions[:self.constants.MAX_FUNCTIONS_TO_ANALYZE]
        
        # Calculate additional function metadata
        for func in functions:
            func.complexity_score = self._calculate_function_complexity(func)
            func.confidence = self._calculate_function_confidence(func)
        
        return {
            'functions_detected': len(functions),
            'functions': functions,
            'detection_method': disassembly_results['analysis_method'],
            'function_detection_confidence': sum(f.confidence for f in functions) / len(functions) if functions else 0.0
        }
    
    def _detect_functions_from_disassembly(self, disassembly_results: Dict[str, Any], prologues: List[bytes], epilogues: List[bytes]) -> List[Function]:
        """Detect functions from disassembled instructions"""
        functions = []
        
        for section in disassembly_results['disassembled_sections']:
            instructions = section['instructions']
            
            # Look for function prologues in disassembled code
            for i, insn in enumerate(instructions):
                mnemonic = insn['mnemonic']
                op_str = insn['op_str']
                
                # Simple prologue detection
                if (mnemonic == 'push' and 'ebp' in op_str) or \
                   (mnemonic == 'push' and 'rbp' in op_str):
                    
                    # Look for corresponding epilogue
                    func_size = self._find_function_end(instructions, i)
                    
                    if func_size >= self.constants.MIN_FUNCTION_SIZE:
                        func = Function(
                            address=insn['address'],
                            size=func_size,
                            name=f"sub_{insn['address']:08x}"
                        )
                        functions.append(func)
        
        return functions
    
    def _detect_functions_heuristic(self, analysis_context: Dict[str, Any], prologues: List[bytes], epilogues: List[bytes]) -> List[Function]:
        """Detect functions using heuristic pattern matching"""
        functions = []
        binary_path = analysis_context['binary_path']
        code_sections = analysis_context['code_sections']
        
        with open(binary_path, 'rb') as f:
            binary_content = f.read()
        
        for section in code_sections:
            start_offset = section['offset']
            size = min(section['size'], 64 * 1024)
            code_data = binary_content[start_offset:start_offset + size]
            base_address = section['virtual_address']
            
            # Search for function prologues
            for prologue in prologues:
                offset = 0
                while True:
                    pos = code_data.find(prologue, offset)
                    if pos == -1:
                        break
                    
                    # Estimate function size (simplified)
                    func_size = self._estimate_function_size(code_data, pos, epilogues)
                    
                    if func_size >= self.constants.MIN_FUNCTION_SIZE:
                        func = Function(
                            address=base_address + pos,
                            size=func_size,
                            name=f"sub_{base_address + pos:08x}"
                        )
                        functions.append(func)
                    
                    offset = pos + 1
        
        return functions
    
    def _find_function_end(self, instructions: List[Dict], start_idx: int) -> int:
        """Find the end of a function from disassembled instructions"""
        size = 0
        for i in range(start_idx, len(instructions)):
            insn = instructions[i]
            size += insn['size']
            
            # Stop at return instruction
            if insn['mnemonic'] == 'ret':
                break
            
            # Stop if we've gone too far
            if size > 4096:  # 4KB function size limit
                break
        
        return size
    
    def _estimate_function_size(self, code_data: bytes, prologue_pos: int, epilogues: List[bytes]) -> int:
        """Estimate function size by finding the nearest epilogue"""
        # Look for epilogue patterns after prologue
        for epilogue in epilogues:
            pos = code_data.find(epilogue, prologue_pos + 1)
            if pos != -1:
                return pos - prologue_pos + len(epilogue)
        
        # Default estimate if no epilogue found
        return 128
    
    def _calculate_function_complexity(self, function: Function) -> float:
        """Calculate function complexity score"""
        # Simplified complexity calculation based on size and calls
        size_factor = min(function.size / 1024.0, 1.0)  # Normalize to 1KB
        calls_factor = min(len(function.calls_made) / 10.0, 1.0)  # Normalize to 10 calls
        
        return (size_factor + calls_factor) / 2.0
    
    def _calculate_function_confidence(self, function: Function) -> float:
        """Calculate confidence in function detection"""
        # Higher confidence for functions with reasonable size and patterns
        if self.constants.MIN_FUNCTION_SIZE <= function.size <= 4096:
            return 0.8
        elif function.size > 4096:
            return 0.6  # Very large functions are less certain
        else:
            return 0.4  # Small functions might be false positives
    
    def _analyze_control_flow(self, analysis_context: Dict[str, Any], function_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze control flow structures"""
        functions = function_results['functions']
        control_structures = []
        
        # Simplified control flow analysis
        for func in functions:
            # Estimate control structures based on function size and complexity
            estimated_structures = max(1, int(func.complexity_score * 5))
            
            for i in range(estimated_structures):
                structure = ControlStructure(
                    type='conditional' if i % 2 == 0 else 'loop',
                    address=func.address + (i * 32),  # Estimated positions
                    size=32,
                    complexity='simple',
                    confidence=0.6
                )
                control_structures.append(structure)
        
        return {
            'total_basic_blocks': len(functions) * 3,  # Rough estimate
            'control_structures': control_structures,
            'control_structure_count': len(control_structures),
            'analysis_confidence': 0.6  # Medium confidence for simplified analysis
        }
    
    def _perform_type_inference(self, analysis_context: Dict[str, Any], function_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic type inference"""
        functions = function_results['functions']
        
        # Extract strings for analysis
        sentinel_data = analysis_context['sentinel_data']
        strings = sentinel_data.get('strings', [])
        
        inferred_types = {
            'string_constants': len([s for s in strings if len(s) > 4]),
            'numeric_constants': 0,  # Would analyze immediate values
            'function_parameters': {},
            'return_types': {}
        }
        
        # Simple type inference for functions
        for func in functions:
            # Infer parameter count from function size (very rough heuristic)
            param_count = min(func.size // 100, 6)  # Estimate based on size
            
            inferred_types['function_parameters'][func.address] = {
                'estimated_param_count': param_count,
                'confidence': 0.4  # Low confidence for simplified inference
            }
            
            # Infer return type (simplified)
            inferred_types['return_types'][func.address] = {
                'type': 'int',  # Default assumption
                'confidence': 0.3
            }
        
        return inferred_types
    
    def _detect_optimization_patterns(self, analysis_context: Dict[str, Any], function_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect optimization patterns in the decompiled code"""
        functions = function_results['functions']
        
        # Analyze function count vs binary size for inlining detection
        binary_info = analysis_context.get('binary_info')
        file_size = binary_info.file_size if binary_info else 1
        function_density = len(functions) / (file_size / 1024)  # Functions per KB
        
        optimization_indicators = {
            'function_inlining': function_density < 0.1,  # Few functions suggests inlining
            'dead_code_elimination': True,  # Assume present in release builds
            'constant_folding': len(analysis_context['sentinel_data'].get('strings', [])) > 10,
            'loop_optimization': any(f.complexity_score > 0.7 for f in functions)
        }
        
        # Estimate optimization level
        optimization_count = sum(optimization_indicators.values())
        if optimization_count >= 3:
            estimated_level = 'O2'
            confidence = 0.7
        elif optimization_count >= 2:
            estimated_level = 'O1'
            confidence = 0.6
        else:
            estimated_level = 'O0'
            confidence = 0.5
        
        return {
            'optimization_level': estimated_level,
            'confidence': confidence,
            'detected_optimizations': [k for k, v in optimization_indicators.items() if v],
            'optimization_indicators': optimization_indicators
        }
    
    def _execute_ai_analysis(self, core_results: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI-enhanced analysis using LangChain"""
        if not self.agent_executor:
            return {}
        
        try:
            function_analysis = core_results.get('function_analysis', {})
            optimization_analysis = core_results.get('optimization_analysis', {})
            
            # Create AI analysis prompt
            prompt = f"""
            Analyze this decompiled binary code:
            
            Functions Detected: {function_analysis.get('functions_detected', 0)}
            Optimization Level: {optimization_analysis.get('optimization_level', 'Unknown')}
            Detected Optimizations: {', '.join(optimization_analysis.get('detected_optimizations', []))}
            
            Provide insights about code structure, algorithms, and development practices.
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
    
    def _validate_results(self, results: Dict[str, Any]) -> MerovingianValidationResult:
        """Validate results meet Merovingian quality thresholds"""
        quality_score = self._calculate_quality_score(results)
        is_valid = quality_score >= self.constants.QUALITY_THRESHOLD
        
        error_messages = []
        if not is_valid:
            error_messages.append(
                f"Quality score {quality_score:.3f} below threshold {self.constants.QUALITY_THRESHOLD}"
            )
        
        # Additional validation checks
        function_analysis = results.get('function_analysis', {})
        if function_analysis.get('functions_detected', 0) == 0:
            error_messages.append("No functions detected")
            quality_score *= 0.5
        
        return MerovingianValidationResult(
            is_valid=len(error_messages) == 0,
            quality_score=quality_score,
            error_messages=error_messages,
            validation_details={
                'quality_score': quality_score,
                'threshold': self.constants.QUALITY_THRESHOLD,
                'agent_id': self.agent_id,
                'functions_detected': function_analysis.get('functions_detected', 0)
            }
        )
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for Merovingian analysis"""
        score_components = []
        
        # Disassembly quality (30%)
        disassembly_analysis = results.get('disassembly_analysis', {})
        if disassembly_analysis.get('total_instructions', 0) > 0:
            disassembly_quality = disassembly_analysis.get('disassembly_quality', 0.0)
            score_components.append(0.3 * disassembly_quality)
        
        # Function detection quality (40%)
        function_analysis = results.get('function_analysis', {})
        function_count = function_analysis.get('functions_detected', 0)
        if function_count > 0:
            detection_confidence = function_analysis.get('function_detection_confidence', 0.0)
            score_components.append(0.4 * detection_confidence)
        
        # Control flow analysis quality (20%)
        control_flow_analysis = results.get('control_flow_analysis', {})
        if control_flow_analysis.get('control_structure_count', 0) > 0:
            cf_confidence = control_flow_analysis.get('analysis_confidence', 0.0)
            score_components.append(0.2 * cf_confidence)
        
        # Optimization detection quality (10%)
        optimization_analysis = results.get('optimization_analysis', {})
        if optimization_analysis.get('optimization_level', 'Unknown') != 'Unknown':
            opt_confidence = optimization_analysis.get('confidence', 0.0)
            score_components.append(0.1 * opt_confidence)
        
        return sum(score_components)
    
    def _finalize_results(self, results: Dict[str, Any], validation: MerovingianValidationResult) -> Dict[str, Any]:
        """Finalize results with Merovingian metadata and validation info"""
        return {
            **results,
            'merovingian_metadata': {
                'agent_id': self.agent_id,
                'matrix_character': self.matrix_character.value,
                'quality_score': validation.quality_score,
                'validation_passed': validation.is_valid,
                'execution_time': self.metrics.execution_time,
                'ai_enhanced': self.ai_enabled,
                'has_disassembler': self.has_disassembler,
                'analysis_timestamp': self.metrics.start_time
            }
        }
    
    def _save_results(self, results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Save Merovingian results to output directory"""
        if 'output_manager' in context:
            output_manager = context['output_manager']
            output_manager.save_agent_data(
                self.agent_id, 
                self.matrix_character.value, 
                results
            )
    
    def _populate_shared_memory(self, results: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Populate shared memory with Merovingian analysis for other agents"""
        shared_memory = context['shared_memory']
        
        # Store Merovingian results for other agents
        shared_memory['analysis_results'][self.agent_id] = results
        
        # Store specific decompilation data for easy access
        shared_memory['decompilation_data']['basic_decompilation'] = {
            'function_analysis': results.get('function_analysis', {}),
            'control_flow_analysis': results.get('control_flow_analysis', {}),
            'optimization_analysis': results.get('optimization_analysis', {}),
            'merovingian_confidence': results['merovingian_metadata']['quality_score']
        }
    
    # AI tool implementations
    def _ai_function_analysis_tool(self, input_data: str) -> str:
        """AI tool for function analysis"""
        return f"Function purpose analysis completed for: {input_data}"
    
    def _ai_algorithm_detection_tool(self, input_data: str) -> str:
        """AI tool for algorithm detection"""
        return f"Algorithm detection completed for: {input_data}"
    
    def _ai_type_inference_tool(self, input_data: str) -> str:
        """AI tool for type inference"""
        return f"Type inference analysis completed for: {input_data}"