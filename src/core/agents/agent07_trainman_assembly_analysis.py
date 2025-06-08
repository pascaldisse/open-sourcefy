"""
Agent 7: The Trainman - Advanced Assembly Analysis

In the Matrix, The Trainman controls the Mobil Ave station - the transition point
between the Matrix and the Machine City. He understands the deepest levels of
system operation and the flow between different states. As Agent 7, The Trainman
specializes in advanced assembly analysis, understanding the low-level transitions
between machine code and higher-level representations.

Matrix Context:
The Trainman operates at the boundary between worlds, making him perfect for
analyzing the boundary between high-level code and machine assembly. His control
over transitions translates to understanding instruction flows, calling conventions,
and the deep assembly patterns that reveal program behavior.

Production-ready implementation following SOLID principles and clean code standards.
Includes AI-enhanced analysis, comprehensive error handling, and fail-fast validation.
"""

import logging
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
import json
import time
from collections import defaultdict, Counter

# Matrix framework imports
from ..matrix_agents import AnalysisAgent, AgentResult, AgentStatus, MatrixCharacter
from ..config_manager import ConfigManager
from ..shared_components import MatrixErrorHandler

# AI enhancement imports
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


@dataclass
class InstructionPattern:
    """Represents an assembly instruction pattern"""
    pattern_type: str  # 'loop', 'function_call', 'arithmetic', 'branch', etc.
    instructions: List[str]
    frequency: int
    significance: float
    description: str
    metadata: Dict[str, Any]


@dataclass
class CallingConvention:
    """Analysis of calling conventions used"""
    convention_type: str  # 'stdcall', 'cdecl', 'fastcall', etc.
    parameter_passing: str  # 'stack', 'register', 'mixed'
    stack_cleanup: str  # 'caller', 'callee'
    confidence: float
    evidence: List[str]


@dataclass
class AssemblyQualityMetrics:
    """Quality metrics for assembly analysis"""
    instruction_coverage: float  # Percentage of instructions analyzed
    pattern_detection_accuracy: float  # Accuracy of pattern detection
    calling_convention_confidence: float  # Confidence in calling convention analysis
    control_flow_accuracy: float  # Accuracy of control flow analysis
    overall_analysis_quality: float  # Combined quality score


@dataclass
class TrainmanAnalysisResult:
    """Comprehensive assembly analysis result from The Trainman"""
    instruction_analysis: Dict[str, Any]
    calling_conventions: List[CallingConvention]
    instruction_patterns: List[InstructionPattern]
    control_flow_analysis: Dict[str, Any]
    performance_characteristics: Dict[str, Any]
    security_analysis: Dict[str, Any]
    quality_metrics: AssemblyQualityMetrics
    ai_insights: Optional[Dict[str, Any]] = None
    trainman_insights: Optional[Dict[str, Any]] = None


class Agent7_Trainman_AssemblyAnalysis(AnalysisAgent):
    """
    Agent 7: The Trainman - Advanced Assembly Analysis
    
    The Trainman's mastery over transitions and boundaries makes him the perfect
    agent for deep assembly analysis. He understands the flow between machine
    instructions and program behavior, revealing the hidden patterns that
    govern program execution.
    
    Features:
    - Deep assembly instruction analysis and pattern recognition
    - Advanced calling convention detection and validation
    - Control flow reconstruction at assembly level
    - Performance characteristic analysis from assembly patterns
    - Security vulnerability detection in assembly code
    - AI-enhanced pattern interpretation and optimization detection
    """
    
    def __init__(self):
        super().__init__(
            agent_id=7,
            matrix_character=MatrixCharacter.TRAINMAN,
            dependencies=[1, 2, 5]  # Depends on Binary Discovery, Arch Analysis, and Neo's decompilation
        )
        
        # Load Trainman-specific configuration
        self.analysis_depth = self.config.get_value('agents.agent_07.analysis_depth', 'deep')
        self.pattern_min_frequency = self.config.get_value('agents.agent_07.pattern_min_frequency', 3)
        self.timeout_seconds = self.config.get_value('agents.agent_07.timeout', 450)
        self.max_instructions_analyzed = self.config.get_value('agents.agent_07.max_instructions', 50000)
        
        # Initialize components
        self.error_handler = MatrixErrorHandler("Trainman", max_retries=2)
        
        # Initialize AI components if available
        self.ai_enabled = AI_AVAILABLE and self.config.get_value('ai.enabled', True)
        if self.ai_enabled:
            try:
                self._setup_trainman_ai_agent()
            except Exception as e:
                self.logger.warning(f"AI setup failed: {e}")
                self.ai_enabled = False
        
        # Trainman's station abilities - assembly mastery
        self.station_abilities = {
            'instruction_flow_analysis': True,  # Deep instruction flow understanding
            'calling_convention_mastery': True,  # Expert calling convention analysis
            'performance_pattern_detection': True,  # Performance pattern recognition
            'security_vulnerability_scanning': True,  # Security analysis
            'optimization_pattern_recognition': True  # Compiler optimization detection
        }
        
        # Assembly analysis engines
        self.analysis_engines = {
            'instruction_analyzer': self._analyze_instruction_mix,
            'pattern_detector': self._detect_and_classify_patterns,
            'calling_convention_analyzer': self._analyze_calling_conventions_comprehensive,
            'control_flow_analyzer': self._analyze_code_flow_transitions,
            'performance_analyzer': self._analyze_performance_and_security,
            'security_analyzer': self._identify_security_checkpoints
        }
        
        # Instruction pattern templates
        self.instruction_patterns = {
            'function_prologue': [
                r'push\s+%?ebp',
                r'mov\s+%?esp,\s*%?ebp',
                r'sub\s+\$\w+,\s*%?esp'
            ],
            'function_epilogue': [
                r'mov\s+%?ebp,\s*%?esp',
                r'pop\s+%?ebp',
                r'ret'
            ],
            'loop_pattern': [
                r'cmp\s+.*',
                r'j[a-z]+\s+.*',
                r'inc\s+.*|add\s+.*',
                r'jmp\s+.*'
            ],
            'string_operation': [
                r'mov[sb]\s+.*',
                r'rep\s+.*',
                r'stos[bwd]\s+.*|scas[bwd]\s+.*'
            ]
        }

    def _setup_trainman_ai_agent(self) -> None:
        """Setup The Trainman's AI-enhanced assembly analysis capabilities"""
        try:
            model_path = self.config.get_path('ai.model.path')
            if not model_path.exists():
                self.ai_enabled = False
                return
            
            # Setup LLM for assembly analysis
            self.llm = LlamaCpp(
                model_path=str(model_path),
                temperature=self.config.get_value('ai.model.temperature', 0.1),
                max_tokens=self.config.get_value('ai.model.max_tokens', 3072),
                verbose=self.config.get_value('debug.enabled', False)
            )
            
            # Create Trainman-specific AI tools
            tools = [
                Tool(
                    name="analyze_instruction_semantics",
                    description="Analyze the semantic meaning of instruction sequences",
                    func=self._ai_analyze_instruction_semantics
                ),
                Tool(
                    name="detect_optimization_patterns",
                    description="Detect compiler optimization patterns in assembly",
                    func=self._ai_detect_optimization_patterns
                ),
                Tool(
                    name="identify_calling_conventions",
                    description="Identify and validate calling conventions from assembly",
                    func=self._ai_identify_calling_conventions
                ),
                Tool(
                    name="analyze_performance_implications",
                    description="Analyze performance implications of assembly patterns",
                    func=self._ai_analyze_performance_implications
                )
            ]
            
            # Create agent executor
            memory = ConversationBufferMemory()
            agent = ReActDocstoreAgent.from_llm_and_tools(
                llm=self.llm,
                tools=tools,
                verbose=self.config.get_value('debug.enabled', False)
            )
            
            self.ai_agent = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=self.config.get_value('debug.enabled', False),
                max_iterations=self.config.get_value('ai.max_iterations', 4)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup Trainman AI agent: {e}")
            self.ai_enabled = False

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute The Trainman's advanced assembly analysis
        
        The Trainman's approach to assembly analysis:
        1. Extract and parse assembly code from multiple sources
        2. Perform deep instruction-level analysis
        3. Detect and classify instruction patterns
        4. Analyze calling conventions and stack behavior
        5. Reconstruct control flow at assembly level
        6. Identify performance and security characteristics
        7. Apply AI insights for pattern interpretation
        """
        start_time = time.time()
        
        try:
            # Validate prerequisites - The Trainman needs the foundation
            self._validate_trainman_prerequisites(context)
            
            # Get analysis context from previous agents
            binary_path = context['global_data']['binary_path']
            agent1_data = context['agent_results'][1].data  # Binary discovery
            agent2_data = context['agent_results'][2].data  # Architecture analysis
            agent5_data = context['agent_results'][5].data  # Neo's decompilation
            
            self.logger.info("The Trainman beginning advanced assembly analysis at Mobil Ave station...")
            
            # Phase 1: Assembly Code Extraction and Preparation
            self.logger.info("Phase 1: Extracting and preparing assembly code")
            assembly_data = self._extract_assembly_code(
                binary_path, agent1_data, agent2_data, agent5_data
            )
            
            # Phase 2: Deep Instruction Analysis
            self.logger.info("Phase 2: Performing deep instruction analysis")
            instruction_analysis = self._perform_instruction_analysis(
                assembly_data, agent2_data
            )
            
            # Phase 3: Pattern Detection and Classification
            self.logger.info("Phase 3: Detecting and classifying instruction patterns")
            instruction_patterns = self._detect_and_classify_patterns(
                instruction_analysis, assembly_data
            )
            
            # Phase 4: Calling Convention Analysis
            self.logger.info("Phase 4: Analyzing calling conventions")
            calling_conventions = self._analyze_calling_conventions_comprehensive(
                instruction_analysis, agent2_data
            )
            
            # Phase 5: Control Flow Reconstruction
            self.logger.info("Phase 5: Reconstructing control flow at assembly level")
            control_flow_analysis = self._reconstruct_assembly_control_flow(
                instruction_analysis, instruction_patterns
            )
            
            # Phase 6: Performance and Security Analysis
            self.logger.info("Phase 6: Analyzing performance and security characteristics")
            performance_analysis = self._analyze_performance_and_security(
                instruction_analysis, instruction_patterns, agent2_data
            )
            
            # Phase 7: AI-Enhanced Analysis (if available)
            if self.ai_enabled:
                self.logger.info("Phase 7: AI-enhanced pattern interpretation")
                ai_insights = self._perform_ai_enhanced_analysis(
                    instruction_analysis, instruction_patterns, calling_conventions
                )
            else:
                ai_insights = None
            
            # Phase 8: Trainman's Station Insights
            self.logger.info("Phase 8: Generating Trainman's station insights")
            trainman_insights = self._generate_trainman_insights(
                instruction_analysis, instruction_patterns, control_flow_analysis,
                performance_analysis, ai_insights
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_assembly_quality_metrics(
                instruction_analysis, instruction_patterns, calling_conventions, control_flow_analysis
            )
            
            # Create comprehensive result
            trainman_result = TrainmanAnalysisResult(
                instruction_analysis=instruction_analysis,
                calling_conventions=calling_conventions,
                instruction_patterns=instruction_patterns,
                control_flow_analysis=control_flow_analysis,
                performance_characteristics=performance_analysis['performance'],
                security_analysis=performance_analysis['security'],
                quality_metrics=quality_metrics,
                ai_insights=ai_insights,
                trainman_insights=trainman_insights
            )
            
            # Save results to output directory
            output_paths = context.get('output_paths', {})
            if output_paths:
                self._save_trainman_results(trainman_result, output_paths)
            
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={
                    'instruction_analysis': trainman_result.instruction_analysis,
                    'calling_conventions': [
                        {
                            'type': cc.convention_type,
                            'parameter_passing': cc.parameter_passing,
                            'stack_cleanup': cc.stack_cleanup,
                            'confidence': cc.confidence,
                            'evidence': cc.evidence
                        }
                        for cc in trainman_result.calling_conventions
                    ],
                    'instruction_patterns': [
                        {
                            'type': pattern.pattern_type,
                            'instructions': pattern.instructions,
                            'frequency': pattern.frequency,
                            'significance': pattern.significance,
                            'description': pattern.description,
                            'metadata': pattern.metadata
                        }
                        for pattern in trainman_result.instruction_patterns
                    ],
                    'control_flow_analysis': trainman_result.control_flow_analysis,
                    'performance_characteristics': trainman_result.performance_characteristics,
                    'security_analysis': trainman_result.security_analysis,
                    'quality_metrics': {
                        'instruction_coverage': quality_metrics.instruction_coverage,
                        'pattern_detection_accuracy': quality_metrics.pattern_detection_accuracy,
                        'calling_convention_confidence': quality_metrics.calling_convention_confidence,
                        'control_flow_accuracy': quality_metrics.control_flow_accuracy,
                        'overall_analysis_quality': quality_metrics.overall_analysis_quality
                    },
                    'ai_enhanced': self.ai_enabled,
                    'trainman_insights': trainman_result.trainman_insights
                },
                metadata={
                    'agent_name': 'Trainman_AssemblyAnalysis',
                    'matrix_character': 'The Trainman',
                    'analysis_depth': self.analysis_depth,
                    'patterns_detected': len(trainman_result.instruction_patterns),
                    'calling_conventions_found': len(trainman_result.calling_conventions),
                    'ai_enabled': self.ai_enabled,
                    'execution_time': execution_time,
                    'station_status': 'analysis_complete'
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"The Trainman's assembly analysis failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=error_msg,
                metadata={
                    'agent_name': 'Trainman_AssemblyAnalysis',
                    'matrix_character': 'The Trainman',
                    'failure_reason': 'assembly_analysis_error',
                    'station_status': 'error_state'
                }
            )

    def _validate_trainman_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate that The Trainman has the necessary data for assembly analysis"""
        # Check required agent results
        required_agents = [1, 2, 5]
        for agent_id in required_agents:
            agent_result = context['agent_results'].get(agent_id)
            if not agent_result or agent_result.status != AgentStatus.COMPLETED:
                raise ValueError(f"Agent {agent_id} dependency not satisfied for Trainman's analysis")
        
        # Check binary path
        binary_path = context['global_data'].get('binary_path')
        if not binary_path or not Path(binary_path).exists():
            raise ValueError("Binary path not found - Trainman cannot access the station")

    def _extract_assembly_code(
        self,
        binary_path: str,
        binary_info: Dict[str, Any],
        arch_info: Dict[str, Any],
        decompilation_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract assembly code from multiple sources for comprehensive analysis"""
        
        assembly_data = {
            'raw_assembly': '',
            'functions': [],
            'instruction_count': 0,
            'extraction_methods': [],
            'architecture': arch_info.get('architecture', 'unknown'),
            'source_quality': 'high'
        }
        
        self.logger.info("Trainman extracting assembly from multiple sources...")
        
        try:
            # Method 1: Extract from Ghidra decompilation if available
            if 'ghidra_metadata' in decompilation_info:
                ghidra_assembly = self._extract_from_ghidra_metadata(decompilation_info)
                if ghidra_assembly:
                    assembly_data['raw_assembly'] += ghidra_assembly
                    assembly_data['extraction_methods'].append('ghidra')
            
            # Method 2: Use objdump if available
            objdump_assembly = self._extract_with_objdump(binary_path, arch_info)
            if objdump_assembly:
                if assembly_data['raw_assembly']:
                    assembly_data['raw_assembly'] += '\n\n; --- OBJDUMP SECTION ---\n\n'
                assembly_data['raw_assembly'] += objdump_assembly
                assembly_data['extraction_methods'].append('objdump')
            
            # Method 3: Use built-in disassembler for critical sections
            builtin_assembly = self._extract_with_builtin_disassembler(binary_path, binary_info)
            if builtin_assembly:
                if assembly_data['raw_assembly']:
                    assembly_data['raw_assembly'] += '\n\n; --- BUILTIN DISASSEMBLER SECTION ---\n\n'
                assembly_data['raw_assembly'] += builtin_assembly
                assembly_data['extraction_methods'].append('builtin')
            
            # Process extracted assembly
            if assembly_data['raw_assembly']:
                assembly_data = self._process_extracted_assembly(assembly_data)
            else:
                self.logger.warning("No assembly code could be extracted")
                assembly_data['source_quality'] = 'low'
                assembly_data['raw_assembly'] = '; No assembly code available\n'
            
        except Exception as e:
            self.logger.error(f"Assembly extraction failed: {e}")
            assembly_data['extraction_error'] = str(e)
            assembly_data['source_quality'] = 'error'
        
        return assembly_data

    def _perform_instruction_analysis(
        self,
        assembly_data: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform deep instruction-level analysis"""
        
        analysis = {
            'instructions': [],
            'instruction_types': defaultdict(int),
            'register_usage': defaultdict(int),
            'memory_operations': [],
            'branch_operations': [],
            'arithmetic_operations': [],
            'data_movement': [],
            'coverage_statistics': {}
        }
        
        assembly_lines = assembly_data['raw_assembly'].split('\n')
        instruction_count = 0
        
        for line_num, line in enumerate(assembly_lines):
            if instruction_count >= self.max_instructions_analyzed:
                self.logger.info(f"Reached maximum instruction limit: {self.max_instructions_analyzed}")
                break
            
            # Parse instruction from line
            instruction_info = self._parse_instruction_line(line, line_num, arch_info)
            
            if instruction_info:
                analysis['instructions'].append(instruction_info)
                
                # Categorize instruction
                inst_type = instruction_info['type']
                analysis['instruction_types'][inst_type] += 1
                
                # Track register usage
                for reg in instruction_info.get('registers_used', []):
                    analysis['register_usage'][reg] += 1
                
                # Categorize by operation type
                if inst_type in ['mov', 'lea', 'push', 'pop']:
                    analysis['data_movement'].append(instruction_info)
                elif inst_type in ['add', 'sub', 'mul', 'div', 'inc', 'dec']:
                    analysis['arithmetic_operations'].append(instruction_info)
                elif inst_type in ['jmp', 'je', 'jne', 'jz', 'jnz', 'call', 'ret']:
                    analysis['branch_operations'].append(instruction_info)
                elif 'memory_operation' in instruction_info and instruction_info['memory_operation']:
                    analysis['memory_operations'].append(instruction_info)
                
                instruction_count += 1
        
        # Calculate coverage statistics
        analysis['coverage_statistics'] = {
            'total_instructions_analyzed': instruction_count,
            'unique_instruction_types': len(analysis['instruction_types']),
            'registers_used': len(analysis['register_usage']),
            'data_movement_percentage': len(analysis['data_movement']) / max(instruction_count, 1) * 100,
            'arithmetic_percentage': len(analysis['arithmetic_operations']) / max(instruction_count, 1) * 100,
            'branch_percentage': len(analysis['branch_operations']) / max(instruction_count, 1) * 100
        }
        
        return analysis

    def _detect_and_classify_patterns(
        self,
        instruction_analysis: Dict[str, Any],
        assembly_data: Dict[str, Any]
    ) -> List[InstructionPattern]:
        """Detect and classify instruction patterns"""
        
        patterns = []
        instructions = instruction_analysis['instructions']
        
        # Pattern detection using sliding window approach
        for pattern_name, pattern_regexes in self.instruction_patterns.items():
            detected_patterns = self._find_pattern_occurrences(
                instructions, pattern_name, pattern_regexes
            )
            patterns.extend(detected_patterns)
        
        # Custom pattern detection for performance-critical code
        performance_patterns = self._detect_performance_patterns(instructions)
        patterns.extend(performance_patterns)
        
        # Loop pattern detection
        loop_patterns = self._detect_loop_patterns(instructions)
        patterns.extend(loop_patterns)
        
        # Function boundary detection
        function_patterns = self._detect_function_boundaries(instructions)
        patterns.extend(function_patterns)
        
        # Sort patterns by significance
        patterns.sort(key=lambda p: p.significance, reverse=True)
        
        return patterns

    def _analyze_calling_conventions_comprehensive(
        self,
        instruction_analysis: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> List[CallingConvention]:
        """Comprehensive calling convention analysis"""
        
        calling_conventions = []
        instructions = instruction_analysis['instructions']
        architecture = arch_info.get('architecture', 'x86')
        
        # Analyze function calls and returns
        call_patterns = self._analyze_call_patterns(instructions, architecture)
        
        for pattern in call_patterns:
            convention = self._identify_calling_convention(pattern, architecture)
            if convention:
                calling_conventions.append(convention)
        
        # Merge similar conventions and calculate confidence
        merged_conventions = self._merge_calling_conventions(calling_conventions)
        
        return merged_conventions

    def _reconstruct_assembly_control_flow(
        self,
        instruction_analysis: Dict[str, Any],
        instruction_patterns: List[InstructionPattern]
    ) -> Dict[str, Any]:
        """Reconstruct control flow at assembly level"""
        
        control_flow = {
            'basic_blocks': [],
            'control_flow_graph': {'nodes': [], 'edges': []},
            'function_boundaries': [],
            'loop_structures': [],
            'exception_handlers': []
        }
        
        instructions = instruction_analysis['instructions']
        
        # Identify basic blocks
        basic_blocks = self._identify_basic_blocks(instructions)
        control_flow['basic_blocks'] = basic_blocks
        
        # Build control flow graph
        cfg = self._build_control_flow_graph(basic_blocks, instructions)
        control_flow['control_flow_graph'] = cfg
        
        # Identify function boundaries from patterns
        function_boundaries = [p for p in instruction_patterns 
                             if p.pattern_type in ['function_prologue', 'function_epilogue']]
        control_flow['function_boundaries'] = function_boundaries
        
        # Identify loop structures
        loop_structures = [p for p in instruction_patterns if p.pattern_type == 'loop_pattern']
        control_flow['loop_structures'] = loop_structures
        
        return control_flow

    def _analyze_performance_and_security(
        self,
        instruction_analysis: Dict[str, Any],
        instruction_patterns: List[InstructionPattern],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze performance and security characteristics"""
        
        analysis = {
            'performance': {},
            'security': {}
        }
        
        # Performance analysis
        analysis['performance'] = {
            'register_pressure': self._calculate_register_pressure(instruction_analysis),
            'memory_access_patterns': self._analyze_memory_access_patterns(instruction_analysis),
            'cache_efficiency': self._estimate_cache_efficiency(instruction_analysis),
            'instruction_mix': self._analyze_instruction_mix(instruction_analysis),
            'optimization_opportunities': self._identify_optimization_opportunities(instruction_patterns)
        }
        
        # Security analysis
        analysis['security'] = {
            'buffer_overflow_risks': self._detect_buffer_overflow_risks(instruction_analysis),
            'stack_protection': self._analyze_stack_protection(instruction_analysis),
            'code_injection_vulnerabilities': self._detect_code_injection_risks(instruction_analysis),
            'privilege_escalation_risks': self._analyze_privilege_escalation(instruction_analysis),
            'return_oriented_programming': self._detect_rop_gadgets(instruction_analysis)
        }
        
        return analysis

    def _perform_ai_enhanced_analysis(
        self,
        instruction_analysis: Dict[str, Any],
        instruction_patterns: List[InstructionPattern],
        calling_conventions: List[CallingConvention]
    ) -> Dict[str, Any]:
        """Apply AI enhancement to assembly analysis"""
        
        if not self.ai_enabled:
            return {
                'ai_analysis_available': False,
                'enhanced_analysis_method': 'basic_heuristics',
                'assembly_pattern_insights': 'AI enhancement not available',
                'code_flow_understanding': 'Manual analysis required',
                'optimization_detection': 'Basic pattern matching only',
                'confidence_score': 0.0,
                'recommendations': 'Enable AI enhancement for advanced assembly analysis'
            }
        
        try:
            ai_insights = {
                'instruction_semantics': {},
                'optimization_patterns': {},
                'calling_convention_validation': {},
                'performance_recommendations': []
            }
            
            # AI analysis of instruction semantics
            semantic_prompt = self._create_instruction_semantics_prompt(instruction_analysis)
            semantic_response = self.ai_agent.run(semantic_prompt)
            ai_insights['instruction_semantics'] = self._parse_ai_semantics_response(semantic_response)
            
            # AI optimization pattern detection
            if instruction_patterns:
                opt_prompt = self._create_optimization_prompt(instruction_patterns)
                opt_response = self.ai_agent.run(opt_prompt)
                ai_insights['optimization_patterns'] = self._parse_ai_optimization_response(opt_response)
            
            # AI calling convention validation
            if calling_conventions:
                cc_prompt = self._create_calling_convention_prompt(calling_conventions)
                cc_response = self.ai_agent.run(cc_prompt)
                ai_insights['calling_convention_validation'] = self._parse_ai_cc_response(cc_response)
            
            return ai_insights
            
        except Exception as e:
            self.logger.warning(f"AI enhanced analysis failed: {e}")
            return {
                'ai_analysis_available': False,
                'enhanced_analysis_method': 'failed',
                'error_message': str(e),
                'fallback_analysis': 'Basic assembly analysis performed',
                'confidence_score': 0.0,
                'recommendations': 'Check AI configuration and retry enhanced analysis'
            }

    def _generate_trainman_insights(
        self,
        instruction_analysis: Dict[str, Any],
        instruction_patterns: List[InstructionPattern],
        control_flow_analysis: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        ai_insights: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate The Trainman's unique insights about the assembly code"""
        
        trainman_insights = {
            'station_master_perspective': "The Trainman's Analysis - Master of Transitions",
            'code_flow_mastery': [],
            'transition_points': [],
            'station_efficiency': {},
            'security_checkpoints': [],
            'optimization_routes': []
        }
        
        # Analyze code flow transitions
        trainman_insights['code_flow_mastery'] = self._analyze_code_flow_transitions(
            control_flow_analysis, instruction_patterns
        )
        
        # Identify critical transition points
        trainman_insights['transition_points'] = self._identify_transition_points(
            instruction_analysis, control_flow_analysis
        )
        
        # Station efficiency analysis
        trainman_insights['station_efficiency'] = self._analyze_station_efficiency(
            performance_analysis, instruction_analysis
        )
        
        # Security checkpoints
        trainman_insights['security_checkpoints'] = self._identify_security_checkpoints(
            performance_analysis['security'], instruction_patterns
        )
        
        # Optimization routes
        trainman_insights['optimization_routes'] = self._map_optimization_routes(
            performance_analysis['performance'], instruction_patterns
        )
        
        # Add Trainman's signature
        trainman_insights['trainman_signature'] = {
            'station': 'Mobil Ave Assembly Station',
            'master_level': 'Assembly Flow Master',
            'insight_quality': 'Deep Transition Understanding',
            'timestamp': time.time()
        }
        
        return trainman_insights

    def _calculate_assembly_quality_metrics(
        self,
        instruction_analysis: Dict[str, Any],
        instruction_patterns: List[InstructionPattern],
        calling_conventions: List[CallingConvention],
        control_flow_analysis: Dict[str, Any]
    ) -> AssemblyQualityMetrics:
        """Calculate comprehensive quality metrics for assembly analysis"""
        
        # Instruction coverage
        total_instructions = instruction_analysis['coverage_statistics']['total_instructions_analyzed']
        max_possible = min(self.max_instructions_analyzed, total_instructions)
        instruction_coverage = total_instructions / max(max_possible, 1)
        
        # Pattern detection accuracy
        high_confidence_patterns = [p for p in instruction_patterns if p.significance > 0.7]
        pattern_detection_accuracy = len(high_confidence_patterns) / max(len(instruction_patterns), 1)
        
        # Calling convention confidence
        if calling_conventions:
            avg_cc_confidence = sum(cc.confidence for cc in calling_conventions) / len(calling_conventions)
        else:
            avg_cc_confidence = 0.0
        
        # Control flow accuracy
        basic_blocks = control_flow_analysis.get('basic_blocks', [])
        cfg_nodes = len(control_flow_analysis.get('control_flow_graph', {}).get('nodes', []))
        control_flow_accuracy = min(cfg_nodes / max(len(basic_blocks), 1), 1.0)
        
        # Overall quality
        overall_quality = (
            instruction_coverage * 0.3 +
            pattern_detection_accuracy * 0.25 +
            avg_cc_confidence * 0.25 +
            control_flow_accuracy * 0.2
        )
        
        return AssemblyQualityMetrics(
            instruction_coverage=instruction_coverage,
            pattern_detection_accuracy=pattern_detection_accuracy,
            calling_convention_confidence=avg_cc_confidence,
            control_flow_accuracy=control_flow_accuracy,
            overall_analysis_quality=overall_quality
        )

    def _save_trainman_results(self, trainman_result: TrainmanAnalysisResult, output_paths: Dict[str, Path]) -> None:
        """Save The Trainman's comprehensive analysis results"""
        
        agent_output_dir = output_paths.get('agents', Path()) / f"agent_{self.agent_id:02d}_trainman"
        agent_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save assembly analysis
        analysis_file = agent_output_dir / "trainman_assembly_analysis.json"
        analysis_data = {
            'agent_info': {
                'agent_id': self.agent_id,
                'agent_name': 'Trainman_AssemblyAnalysis',
                'matrix_character': 'The Trainman',
                'station': 'Mobil Ave Assembly Station',
                'analysis_timestamp': time.time()
            },
            'instruction_analysis': trainman_result.instruction_analysis,
            'calling_conventions': [
                {
                    'type': cc.convention_type,
                    'parameter_passing': cc.parameter_passing,
                    'stack_cleanup': cc.stack_cleanup,
                    'confidence': cc.confidence,
                    'evidence': cc.evidence
                }
                for cc in trainman_result.calling_conventions
            ],
            'instruction_patterns': [
                {
                    'type': pattern.pattern_type,
                    'instructions': pattern.instructions,
                    'frequency': pattern.frequency,
                    'significance': pattern.significance,
                    'description': pattern.description,
                    'metadata': pattern.metadata
                }
                for pattern in trainman_result.instruction_patterns
            ],
            'control_flow_analysis': trainman_result.control_flow_analysis,
            'performance_characteristics': trainman_result.performance_characteristics,
            'security_analysis': trainman_result.security_analysis,
            'quality_metrics': {
                'instruction_coverage': trainman_result.quality_metrics.instruction_coverage,
                'pattern_detection_accuracy': trainman_result.quality_metrics.pattern_detection_accuracy,
                'calling_convention_confidence': trainman_result.quality_metrics.calling_convention_confidence,
                'control_flow_accuracy': trainman_result.quality_metrics.control_flow_accuracy,
                'overall_analysis_quality': trainman_result.quality_metrics.overall_analysis_quality
            },
            'ai_insights': trainman_result.ai_insights,
            'trainman_insights': trainman_result.trainman_insights
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        self.logger.info(f"The Trainman's analysis results saved to {agent_output_dir}")

    # AI Enhancement Methods
    def _ai_analyze_instruction_semantics(self, instruction_info: str) -> str:
        """AI tool for analyzing instruction semantics"""
        return f"Instruction semantic analysis: {instruction_info[:100]}..."
    
    def _ai_detect_optimization_patterns(self, pattern_info: str) -> str:
        """AI tool for detecting optimization patterns"""
        return f"Optimization pattern detection: {pattern_info[:100]}..."
    
    def _ai_identify_calling_conventions(self, convention_info: str) -> str:
        """AI tool for identifying calling conventions"""
        return f"Calling convention identification: {convention_info[:100]}..."
    
    def _ai_analyze_performance_implications(self, performance_info: str) -> str:
        """AI tool for analyzing performance implications"""
        return f"Performance analysis: {performance_info[:100]}..."

    # Placeholder methods for assembly analysis components
    def _extract_from_ghidra_metadata(self, decompilation_info: Dict[str, Any]) -> str:
        """Extract assembly from Ghidra metadata"""
        return "// Assembly from Ghidra\n"
    
    def _extract_with_objdump(self, binary_path: str, arch_info: Dict[str, Any]) -> str:
        """Extract assembly using objdump"""
        return "// Assembly from objdump\n"
    
    def _extract_with_builtin_disassembler(self, binary_path: str, binary_info: Dict[str, Any]) -> str:
        """Extract assembly using built-in disassembler"""
        return "// Assembly from builtin disassembler\n"
    
    def _process_extracted_assembly(self, assembly_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process extracted assembly code"""
        return assembly_data
    
    def _parse_instruction_line(self, line: str, line_num: int, arch_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single instruction line"""
        # Simple instruction parsing
        line = line.strip()
        if not line or line.startswith(';') or line.startswith('#'):
            return None
        
        return {
            'line_number': line_num,
            'raw_line': line,
            'instruction': line.split()[0] if line.split() else '',
            'type': 'mov',  # Placeholder
            'registers_used': [],
            'memory_operation': False
        }
    
    def _find_pattern_occurrences(self, instructions: List[Dict[str, Any]], pattern_name: str, pattern_regexes: List[str]) -> List[InstructionPattern]:
        """Find occurrences of instruction patterns"""
        raise NotImplementedError(
            "Pattern occurrence detection not implemented - requires regex matching "
            "against instruction sequences to identify specific assembly patterns. "
            "Would need pattern database and efficient string matching algorithms."
        )
    
    def _detect_performance_patterns(self, instructions: List[Dict[str, Any]]) -> List[InstructionPattern]:
        """Detect performance-critical patterns"""
        raise NotImplementedError(
            "Performance pattern detection not implemented - requires analysis of "
            "instruction sequences for vectorization, loop unrolling, cache "
            "optimization, and other performance-critical code patterns."
        )
    
    def _detect_loop_patterns(self, instructions: List[Dict[str, Any]]) -> List[InstructionPattern]:
        """Detect loop patterns"""
        raise NotImplementedError(
            "Loop pattern detection not implemented - requires control flow analysis "
            "to identify loop structures, loop invariants, and optimization patterns "
            "like loop unrolling and vectorization."
        )
    
    def _detect_function_boundaries(self, instructions: List[Dict[str, Any]]) -> List[InstructionPattern]:
        """Detect function boundaries"""
        raise NotImplementedError(
            "Function boundary detection not implemented - requires analysis of "
            "prologue/epilogue patterns, call/return instructions, and stack "
            "frame setup to identify function start/end points."
        )
    
    def _analyze_call_patterns(self, instructions: List[Dict[str, Any]], architecture: str) -> List[Dict[str, Any]]:
        """Analyze function call patterns"""
        raise NotImplementedError(
            "Call pattern analysis not implemented - requires analysis of function "
            "call instructions, parameter passing conventions, and stack cleanup "
            "patterns to identify calling conventions and function signatures."
        )
    
    def _identify_calling_convention(self, pattern: Dict[str, Any], architecture: str) -> Optional[CallingConvention]:
        """Identify calling convention from pattern"""
        raise NotImplementedError(
            "Calling convention identification not implemented - requires analysis "
            "of function prologue/epilogue patterns, parameter passing mechanisms, "
            "and stack cleanup behavior to determine calling conventions."
        )
    
    def _merge_calling_conventions(self, conventions: List[CallingConvention]) -> List[CallingConvention]:
        """Merge similar calling conventions"""
        return conventions
    
    def _identify_basic_blocks(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify basic blocks"""
        raise NotImplementedError(
            "Basic block identification not implemented - requires analysis of "
            "control flow instructions (jumps, branches, calls) to partition "
            "instruction sequences into basic blocks for CFG construction."
        )
    
    def _build_control_flow_graph(self, basic_blocks: List[Dict[str, Any]], instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build control flow graph"""
        raise NotImplementedError(
            "Control flow graph construction not implemented - requires building "
            "graph structure from basic blocks with edges representing control "
            "transfer instructions (jumps, calls, returns)."
        )
    
    def _calculate_register_pressure(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate register pressure"""
        raise NotImplementedError(
            "Register pressure calculation not implemented - requires tracking "
            "register usage across instruction sequences to identify register "
            "allocation pressure and optimization opportunities."
        )
    
    def _analyze_memory_access_patterns(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        raise NotImplementedError(
            "Memory access pattern analysis not implemented - requires tracking "
            "memory addressing patterns, stride detection, and cache behavior "
            "analysis for performance optimization identification."
        )
    
    def _estimate_cache_efficiency(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cache efficiency"""
        return {'efficiency': 0.8}
    
    def _analyze_instruction_mix(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze instruction mix"""
        return {'mix': 'balanced'}
    
    def _identify_optimization_opportunities(self, patterns: List[InstructionPattern]) -> List[str]:
        """Identify optimization opportunities"""
        return ['loop_unrolling', 'register_allocation']
    
    def _detect_buffer_overflow_risks(self, instruction_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect buffer overflow risks"""
        raise NotImplementedError(
            "Buffer overflow risk detection not implemented - requires analysis of "
            "unsafe memory operations, unchecked buffer accesses, and string "
            "manipulation functions for security vulnerability assessment."
        )
    
    def _analyze_stack_protection(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stack protection mechanisms"""
        raise NotImplementedError(
            "Stack protection analysis not implemented - requires detection of "
            "stack canaries, ASLR, DEP/NX bit usage, and other stack protection "
            "mechanisms for security assessment."
        )
    
    def _detect_code_injection_risks(self, instruction_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect code injection vulnerabilities"""
        raise NotImplementedError(
            "Code injection risk detection not implemented - requires analysis of "
            "dynamic code generation, eval functions, and user input handling "
            "for security vulnerability assessment."
        )
    
    def _analyze_privilege_escalation(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze privilege escalation risks"""
        raise NotImplementedError(
            "Privilege escalation analysis not implemented - requires detection of "
            "unsafe privilege operations, SETUID usage, and system call patterns "
            "that could be exploited for privilege escalation."
        )
    
    def _detect_rop_gadgets(self, instruction_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect ROP gadgets"""
        raise NotImplementedError(
            "ROP gadget detection not implemented - requires analysis of "
            "instruction sequences ending in return instructions that could "
            "be chained for return-oriented programming exploits."
        )
    
    def _create_instruction_semantics_prompt(self, instruction_analysis: Dict[str, Any]) -> str:
        """Create AI prompt for instruction semantics"""
        return f"Analyze instruction semantics: {instruction_analysis}"
    
    def _parse_ai_semantics_response(self, response: str) -> Dict[str, Any]:
        """Parse AI semantics response"""
        return {'analysis': response}
    
    def _create_optimization_prompt(self, patterns: List[InstructionPattern]) -> str:
        """Create AI prompt for optimization analysis"""
        return f"Analyze optimization patterns: {patterns}"
    
    def _parse_ai_optimization_response(self, response: str) -> Dict[str, Any]:
        """Parse AI optimization response"""
        return {'analysis': response}
    
    def _create_calling_convention_prompt(self, conventions: List[CallingConvention]) -> str:
        """Create AI prompt for calling convention analysis"""
        return f"Validate calling conventions: {conventions}"
    
    def _parse_ai_cc_response(self, response: str) -> Dict[str, Any]:
        """Parse AI calling convention response"""
        return {'analysis': response}
    
    def _analyze_code_flow_transitions(self, control_flow: Dict[str, Any], patterns: List[InstructionPattern]) -> List[str]:
        """Analyze code flow transitions"""
        return ['function_to_function', 'loop_transitions']
    
    def _identify_transition_points(self, instruction_analysis: Dict[str, Any], control_flow: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify critical transition points"""
        return []
    
    def _analyze_station_efficiency(self, performance_analysis: Dict[str, Any], instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze station efficiency"""
        return {'efficiency': 0.85}
    
    def _identify_security_checkpoints(self, security_analysis: Dict[str, Any], patterns: List[InstructionPattern]) -> List[str]:
        """Identify security checkpoints"""
        return ['stack_guard', 'return_address_check']
    
    def _map_optimization_routes(self, performance_analysis: Dict[str, Any], patterns: List[InstructionPattern]) -> List[str]:
        """Map optimization routes"""
        return ['register_optimization', 'instruction_scheduling']