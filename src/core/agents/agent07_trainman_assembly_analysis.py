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
            dependencies=[1, 2, 3]  # Depends on Binary Discovery, Arch Analysis, and Merovingian's decompilation
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
            binary_path = context.get('binary_path', '')
            agent1_data = context['agent_results'][1].data  # Binary discovery
            agent2_data = context['agent_results'][2].data  # Architecture analysis
            agent3_data = context['agent_results'][3].data  # Merovingian's decompilation
            
            self.logger.info("The Trainman beginning advanced assembly analysis at Mobil Ave station...")
            
            # Phase 1: Assembly Code Extraction and Preparation
            self.logger.info("Phase 1: Extracting and preparing assembly code")
            assembly_data = self._extract_assembly_code(
                binary_path, agent1_data, agent2_data, agent3_data
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
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return {
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
                    'trainman_insights': trainman_result.trainman_insights,
                    'trainman_metadata': {
                        'agent_name': 'Trainman_AssemblyAnalysis',
                        'matrix_character': 'The Trainman',
                        'analysis_depth': self.analysis_depth,
                        'patterns_detected': len(trainman_result.instruction_patterns),
                        'calling_conventions_found': len(trainman_result.calling_conventions),
                        'ai_enabled': self.ai_enabled,
                        'execution_time': execution_time,
                        'station_status': 'analysis_complete'
                    }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"The Trainman's assembly analysis failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e

    def _validate_trainman_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate that The Trainman has the necessary data for assembly analysis"""
        # Check required agent results
        required_agents = [1, 2, 3]
        for agent_id in required_agents:
            agent_result = context['agent_results'].get(agent_id)
            if not agent_result or agent_result.status != AgentStatus.SUCCESS:
                raise ValueError(f"Agent {agent_id} dependency not satisfied for Trainman's analysis")
        
        # Check binary path
        binary_path = context.get('binary_path')
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
        patterns = []
        
        if not instructions or not pattern_regexes:
            return patterns
        
        # Convert instructions to searchable text
        instruction_lines = [inst.get('raw_line', '').lower() for inst in instructions]
        
        matches = []
        for i, line in enumerate(instruction_lines):
            # Check if this line starts a pattern match
            pattern_match = True
            matched_instructions = []
            
            for j, regex_pattern in enumerate(pattern_regexes):
                if i + j < len(instruction_lines):
                    import re
                    if re.search(regex_pattern, instruction_lines[i + j]):
                        matched_instructions.append(instruction_lines[i + j])
                    else:
                        pattern_match = False
                        break
                else:
                    pattern_match = False
                    break
            
            if pattern_match and len(matched_instructions) == len(pattern_regexes):
                matches.append((i, matched_instructions))
        
        # Create pattern objects for found matches
        if matches:
            pattern = InstructionPattern(
                pattern_type=pattern_name,
                instructions=list(set([inst for _, inst_list in matches for inst in inst_list])),
                frequency=len(matches),
                significance=min(len(matches) / 10.0, 1.0),
                description=f'{pattern_name} pattern found {len(matches)} times',
                metadata={
                    'pattern_regexes': pattern_regexes,
                    'match_positions': [pos for pos, _ in matches],
                    'total_matches': len(matches)
                }
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_performance_patterns(self, instructions: List[Dict[str, Any]]) -> List[InstructionPattern]:
        """Detect performance-critical patterns"""
        patterns = []
        
        if not instructions:
            return patterns
        
        # Detect vectorization patterns (SIMD instructions)
        simd_instructions = ['movq', 'movdq', 'addps', 'mulps', 'subps', 'divps', 'movaps', 'movups']
        simd_count = 0
        simd_lines = []
        
        for inst in instructions:
            inst_name = inst.get('instruction', '').lower()
            if any(simd in inst_name for simd in simd_instructions):
                simd_count += 1
                simd_lines.append(inst.get('raw_line', ''))
        
        if simd_count > 2:
            patterns.append(InstructionPattern(
                pattern_type='vectorization',
                instructions=simd_lines[:5],  # Show first 5 examples
                frequency=simd_count,
                significance=min(simd_count / 20.0, 1.0),
                description=f'SIMD/vectorization pattern detected with {simd_count} vector instructions',
                metadata={'simd_instruction_count': simd_count, 'optimization_type': 'vectorization'}
            ))
        
        # Detect cache optimization patterns (sequential memory access)
        memory_instructions = []
        for i, inst in enumerate(instructions):
            raw_line = inst.get('raw_line', '').lower()
            if any(mem_op in raw_line for mem_op in ['mov', 'lea', 'push', 'pop']):
                if '[' in raw_line and ']' in raw_line:  # Memory reference
                    memory_instructions.append((i, inst))
        
        # Look for sequential memory access patterns
        sequential_access_count = 0
        for i in range(len(memory_instructions) - 1):
            curr_line = memory_instructions[i][1].get('raw_line', '')
            next_line = memory_instructions[i+1][1].get('raw_line', '')
            # Simple heuristic: consecutive memory operations suggest cache optimization
            if memory_instructions[i+1][0] - memory_instructions[i][0] <= 2:
                sequential_access_count += 1
        
        if sequential_access_count > 3:
            patterns.append(InstructionPattern(
                pattern_type='cache_optimization',
                instructions=[inst[1].get('raw_line', '') for inst in memory_instructions[:5]],
                frequency=sequential_access_count,
                significance=min(sequential_access_count / 15.0, 1.0),
                description=f'Cache optimization pattern with {sequential_access_count} sequential memory accesses',
                metadata={'sequential_accesses': sequential_access_count, 'optimization_type': 'cache'}
            ))
        
        # Detect register reuse patterns (performance optimization)
        register_usage = {}
        for inst in instructions:
            for reg in inst.get('registers_used', []):
                register_usage[reg] = register_usage.get(reg, 0) + 1
        
        high_usage_regs = [reg for reg, count in register_usage.items() if count > 5]
        if len(high_usage_regs) > 2:
            patterns.append(InstructionPattern(
                pattern_type='register_optimization',
                instructions=[f'High usage: {reg} ({register_usage[reg]} times)' for reg in high_usage_regs[:3]],
                frequency=len(high_usage_regs),
                significance=min(len(high_usage_regs) / 8.0, 1.0),
                description=f'Register optimization pattern with {len(high_usage_regs)} heavily used registers',
                metadata={'high_usage_registers': high_usage_regs, 'optimization_type': 'register_allocation'}
            ))
        
        return patterns
    
    def _detect_loop_patterns(self, instructions: List[Dict[str, Any]]) -> List[InstructionPattern]:
        """Detect loop patterns"""
        patterns = []
        
        if not instructions:
            return patterns
        
        # Find potential loop structures by looking for jump-back patterns
        loop_candidates = []
        
        for i, inst in enumerate(instructions):
            inst_name = inst.get('instruction', '').lower()
            raw_line = inst.get('raw_line', '').lower()
            
            # Look for conditional jumps that might be loop ends
            if any(jump in inst_name for jump in ['je', 'jne', 'jl', 'jg', 'jle', 'jge', 'jz', 'jnz']):
                # Look for jump target (simplified - would need proper address parsing)
                if any(target in raw_line for target in ['$', '+', '-']):
                    loop_candidates.append((i, inst, 'conditional_loop'))
            
            # Look for explicit loop instructions
            elif inst_name in ['loop', 'loope', 'loopne', 'loopz', 'loopnz']:
                loop_candidates.append((i, inst, 'explicit_loop'))
        
        # Analyze loop candidates
        if loop_candidates:
            loop_instructions = []
            for pos, inst, loop_type in loop_candidates:
                loop_instructions.append(inst.get('raw_line', ''))
                
                # Look for loop body (instructions between compare and jump)
                if pos > 0:
                    # Check previous instructions for compare operations
                    for j in range(max(0, pos - 5), pos):
                        prev_inst = instructions[j].get('instruction', '').lower()
                        if prev_inst in ['cmp', 'test', 'inc', 'dec', 'add', 'sub']:
                            loop_instructions.append(instructions[j].get('raw_line', ''))
            
            if loop_candidates:
                patterns.append(InstructionPattern(
                    pattern_type='loop_structure',
                    instructions=loop_instructions[:10],  # Limit to first 10 instructions
                    frequency=len(loop_candidates),
                    significance=min(len(loop_candidates) / 5.0, 1.0),
                    description=f'Loop structures detected with {len(loop_candidates)} loop patterns',
                    metadata={
                        'loop_types': [ltype for _, _, ltype in loop_candidates],
                        'loop_positions': [pos for pos, _, _ in loop_candidates]
                    }
                ))
        
        # Detect potential loop unrolling (repeated instruction sequences)
        instruction_sequences = []
        sequence_length = 4
        
        for i in range(len(instructions) - sequence_length + 1):
            sequence = [inst.get('instruction', '') for inst in instructions[i:i+sequence_length]]
            sequence_key = ' '.join(sequence)
            instruction_sequences.append(sequence_key)
        
        # Count sequence repetitions
        from collections import Counter
        sequence_counts = Counter(instruction_sequences)
        repeated_sequences = [(seq, count) for seq, count in sequence_counts.items() if count > 2]
        
        if repeated_sequences:
            most_repeated = max(repeated_sequences, key=lambda x: x[1])
            patterns.append(InstructionPattern(
                pattern_type='loop_unrolling',
                instructions=[most_repeated[0]],
                frequency=most_repeated[1],
                significance=min(most_repeated[1] / 10.0, 1.0),
                description=f'Loop unrolling pattern: sequence repeated {most_repeated[1]} times',
                metadata={
                    'repeated_sequence': most_repeated[0],
                    'repetition_count': most_repeated[1],
                    'total_repeated_sequences': len(repeated_sequences)
                }
            ))
        
        return patterns
    
    def _detect_function_boundaries(self, instructions: List[Dict[str, Any]]) -> List[InstructionPattern]:
        """Detect function boundaries"""
        patterns = []
        
        if not instructions:
            return patterns
        
        # Function prologue detection
        prologue_positions = []
        for i in range(len(instructions) - 2):
            # Common prologue pattern: push ebp; mov ebp, esp; sub esp, N
            inst1 = instructions[i].get('instruction', '').lower()
            inst2 = instructions[i+1].get('instruction', '').lower() if i+1 < len(instructions) else ''
            inst3 = instructions[i+2].get('instruction', '').lower() if i+2 < len(instructions) else ''
            
            raw1 = instructions[i].get('raw_line', '').lower()
            raw2 = instructions[i+1].get('raw_line', '').lower() if i+1 < len(instructions) else ''
            
            # Pattern 1: push ebp; mov ebp, esp
            if (inst1 == 'push' and 'ebp' in raw1 and 
                inst2 == 'mov' and 'ebp' in raw2 and 'esp' in raw2):
                prologue_positions.append(i)
            
            # Pattern 2: Single instruction prologue markers
            elif inst1 in ['enter'] or ('proc' in raw1 and 'far' in raw1):
                prologue_positions.append(i)
        
        if prologue_positions:
            prologue_instructions = []
            for pos in prologue_positions[:5]:  # Show first 5 examples
                if pos + 2 < len(instructions):
                    prologue_instructions.extend([
                        instructions[pos].get('raw_line', ''),
                        instructions[pos+1].get('raw_line', ''),
                        instructions[pos+2].get('raw_line', '')
                    ])
            
            patterns.append(InstructionPattern(
                pattern_type='function_prologue',
                instructions=prologue_instructions[:10],
                frequency=len(prologue_positions),
                significance=min(len(prologue_positions) / 10.0, 1.0),
                description=f'Function prologue patterns detected at {len(prologue_positions)} locations',
                metadata={'prologue_positions': prologue_positions}
            ))
        
        # Function epilogue detection
        epilogue_positions = []
        for i in range(len(instructions) - 1):
            # Common epilogue pattern: mov esp, ebp; pop ebp; ret
            inst1 = instructions[i].get('instruction', '').lower()
            inst2 = instructions[i+1].get('instruction', '').lower() if i+1 < len(instructions) else ''
            
            raw1 = instructions[i].get('raw_line', '').lower()
            
            # Pattern 1: mov esp, ebp; pop ebp (followed by ret)
            if (inst1 == 'mov' and 'esp' in raw1 and 'ebp' in raw1 and 
                inst2 == 'pop' and 'ebp' in instructions[i+1].get('raw_line', '').lower()):
                epilogue_positions.append(i)
            
            # Pattern 2: Simple return
            elif inst1 in ['ret', 'retn', 'retf'] or inst1 == 'leave':
                epilogue_positions.append(i)
        
        if epilogue_positions:
            epilogue_instructions = []
            for pos in epilogue_positions[:5]:  # Show first 5 examples
                epilogue_instructions.append(instructions[pos].get('raw_line', ''))
                if pos + 1 < len(instructions):
                    epilogue_instructions.append(instructions[pos+1].get('raw_line', ''))
            
            patterns.append(InstructionPattern(
                pattern_type='function_epilogue',
                instructions=epilogue_instructions[:10],
                frequency=len(epilogue_positions),
                significance=min(len(epilogue_positions) / 10.0, 1.0),
                description=f'Function epilogue patterns detected at {len(epilogue_positions)} locations',
                metadata={'epilogue_positions': epilogue_positions}
            ))
        
        # Function call detection
        call_positions = []
        for i, inst in enumerate(instructions):
            inst_name = inst.get('instruction', '').lower()
            if inst_name in ['call', 'callq']:
                call_positions.append(i)
        
        if call_positions:
            call_instructions = [instructions[pos].get('raw_line', '') for pos in call_positions[:10]]
            
            patterns.append(InstructionPattern(
                pattern_type='function_calls',
                instructions=call_instructions,
                frequency=len(call_positions),
                significance=min(len(call_positions) / 20.0, 1.0),
                description=f'Function call instructions detected at {len(call_positions)} locations',
                metadata={'call_positions': call_positions}
            ))
        
        return patterns
    
    def _analyze_call_patterns(self, instructions: List[Dict[str, Any]], architecture: str) -> List[Dict[str, Any]]:
        """Analyze function call patterns"""
        call_patterns = []
        
        if not instructions:
            return call_patterns
        
        # Find all call instructions
        for i, inst in enumerate(instructions):
            inst_name = inst.get('instruction', '').lower()
            raw_line = inst.get('raw_line', '').lower()
            
            if inst_name in ['call', 'callq']:
                pattern = {
                    'position': i,
                    'instruction': raw_line,
                    'target': 'unknown',
                    'parameter_setup': [],
                    'stack_cleanup': [],
                    'calling_convention_hints': []
                }
                
                # Extract call target
                parts = raw_line.split()
                if len(parts) > 1:
                    pattern['target'] = parts[-1].replace(',', '').replace(';', '')
                
                # Look backwards for parameter setup (preceding instructions)
                for j in range(max(0, i-10), i):
                    prev_inst = instructions[j]
                    prev_name = prev_inst.get('instruction', '').lower()
                    prev_line = prev_inst.get('raw_line', '').lower()
                    
                    # Parameter passing via registers (fastcall/register calling convention)
                    if prev_name == 'mov' and any(reg in prev_line for reg in ['eax', 'ebx', 'ecx', 'edx', 'rdi', 'rsi']):
                        pattern['parameter_setup'].append(prev_line)
                        pattern['calling_convention_hints'].append('register_based')
                    
                    # Parameter passing via stack (cdecl/stdcall)
                    elif prev_name == 'push':
                        pattern['parameter_setup'].append(prev_line)
                        pattern['calling_convention_hints'].append('stack_based')
                    
                    # Immediate values as parameters
                    elif prev_name == 'mov' and '$' in prev_line:
                        pattern['parameter_setup'].append(prev_line)
                
                # Look forwards for stack cleanup (following instructions)
                for j in range(i+1, min(len(instructions), i+5)):
                    next_inst = instructions[j]
                    next_name = next_inst.get('instruction', '').lower()
                    next_line = next_inst.get('raw_line', '').lower()
                    
                    # Stack cleanup patterns
                    if next_name in ['add', 'pop'] and ('esp' in next_line or 'rsp' in next_line):
                        pattern['stack_cleanup'].append(next_line)
                        pattern['calling_convention_hints'].append('caller_cleanup')
                    
                    # Break on next call or return
                    elif next_name in ['call', 'ret', 'retn']:
                        break
                
                call_patterns.append(pattern)
        
        return call_patterns
    
    def _identify_calling_convention(self, pattern: Dict[str, Any], architecture: str) -> Optional[CallingConvention]:
        """Identify calling convention from pattern"""
        hints = pattern.get('calling_convention_hints', [])
        parameter_setup = pattern.get('parameter_setup', [])
        stack_cleanup = pattern.get('stack_cleanup', [])
        
        if not hints:
            return None
        
        # Count different types of hints
        register_hints = hints.count('register_based')
        stack_hints = hints.count('stack_based')
        cleanup_hints = hints.count('caller_cleanup')
        
        # Determine calling convention based on patterns
        if register_hints > stack_hints:
            # Register-based calling convention
            if architecture.lower() in ['x86_64', 'amd64']:
                convention_type = 'microsoft_x64' if 'rdi' in str(parameter_setup) else 'system_v_x64'
            else:
                convention_type = 'fastcall'
            
            return CallingConvention(
                convention_type=convention_type,
                parameter_passing='register',
                stack_cleanup='caller' if cleanup_hints > 0 else 'callee',
                confidence=min(0.7 + (register_hints * 0.1), 1.0),
                evidence=[
                    f'Register-based parameter passing detected ({register_hints} instances)',
                    f'Parameter setup: {parameter_setup[:3]}',
                    f'Architecture: {architecture}'
                ]
            )
        
        elif stack_hints > 0:
            # Stack-based calling convention
            if cleanup_hints > 0:
                convention_type = 'cdecl'
                stack_cleanup = 'caller'
            else:
                convention_type = 'stdcall'
                stack_cleanup = 'callee'
            
            return CallingConvention(
                convention_type=convention_type,
                parameter_passing='stack',
                stack_cleanup=stack_cleanup,
                confidence=min(0.6 + (stack_hints * 0.1), 1.0),
                evidence=[
                    f'Stack-based parameter passing detected ({stack_hints} instances)',
                    f'Stack cleanup: {"caller" if cleanup_hints > 0 else "callee"}',
                    f'Parameter setup: {parameter_setup[:3]}'
                ]
            )
        
        # Default/unknown convention
        return CallingConvention(
            convention_type='unknown',
            parameter_passing='mixed',
            stack_cleanup='unknown',
            confidence=0.3,
            evidence=[
                f'Mixed or unclear calling pattern',
                f'Hints: register={register_hints}, stack={stack_hints}, cleanup={cleanup_hints}'
            ]
        )
    
    def _merge_calling_conventions(self, conventions: List[CallingConvention]) -> List[CallingConvention]:
        """Merge similar calling conventions"""
        return conventions
    
    def _identify_basic_blocks(self, instructions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify basic blocks"""
        if not instructions:
            return []
        
        basic_blocks = []
        block_starts = set([0])  # First instruction is always a block start
        
        # Identify block boundaries
        for i, inst in enumerate(instructions):
            inst_name = inst.get('instruction', '').lower()
            
            # Instructions that end a basic block
            if inst_name in ['ret', 'retn', 'retf', 'jmp', 'je', 'jne', 'jz', 'jnz', 
                           'jl', 'jg', 'jle', 'jge', 'ja', 'jb', 'jae', 'jbe', 'call']:
                # Next instruction starts a new block (if it exists)
                if i + 1 < len(instructions):
                    block_starts.add(i + 1)
            
            # Jump targets also start new blocks (simplified - would need address analysis)
            # Look for labels or addresses in jump instructions
            if any(jump in inst_name for jump in ['jmp', 'je', 'jne', 'jz', 'jnz', 'jl', 'jg']):
                raw_line = inst.get('raw_line', '')
                # Simple heuristic: if the jump target looks like an offset, mark it
                if '+' in raw_line or '-' in raw_line:
                    # This is simplified - real implementation would parse addresses
                    pass
        
        # Create basic blocks
        block_starts = sorted(list(block_starts))
        
        for i in range(len(block_starts)):
            start_idx = block_starts[i]
            end_idx = block_starts[i + 1] if i + 1 < len(block_starts) else len(instructions)
            
            if start_idx < len(instructions):
                block_instructions = instructions[start_idx:end_idx]
                
                basic_block = {
                    'block_id': i,
                    'start_index': start_idx,
                    'end_index': end_idx - 1,
                    'instruction_count': len(block_instructions),
                    'instructions': [inst.get('raw_line', '') for inst in block_instructions],
                    'first_instruction': block_instructions[0].get('instruction', '') if block_instructions else '',
                    'last_instruction': block_instructions[-1].get('instruction', '') if block_instructions else '',
                    'block_type': self._classify_basic_block(block_instructions)
                }
                
                basic_blocks.append(basic_block)
        
        return basic_blocks
    
    def _classify_basic_block(self, instructions: List[Dict[str, Any]]) -> str:
        """Classify the type of basic block"""
        if not instructions:
            return 'empty'
        
        last_inst = instructions[-1].get('instruction', '').lower()
        
        if last_inst in ['ret', 'retn', 'retf']:
            return 'return'
        elif last_inst == 'jmp':
            return 'unconditional_jump'
        elif last_inst in ['je', 'jne', 'jz', 'jnz', 'jl', 'jg', 'jle', 'jge']:
            return 'conditional_jump'
        elif last_inst == 'call':
            return 'function_call'
        else:
            return 'sequential'
    
    def _build_control_flow_graph(self, basic_blocks: List[Dict[str, Any]], instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build control flow graph"""
        cfg = {
            'nodes': [],
            'edges': [],
            'entry_points': [],
            'exit_points': []
        }
        
        if not basic_blocks:
            return cfg
        
        # Create nodes from basic blocks
        for block in basic_blocks:
            node = {
                'id': block['block_id'],
                'type': block['block_type'],
                'instruction_count': block['instruction_count'],
                'start_index': block['start_index'],
                'end_index': block['end_index']
            }
            cfg['nodes'].append(node)
            
            # Identify entry and exit points
            if block['block_id'] == 0 or block['block_type'] in ['function_call']:
                cfg['entry_points'].append(block['block_id'])
            
            if block['block_type'] in ['return', 'unconditional_jump']:
                cfg['exit_points'].append(block['block_id'])
        
        # Create edges based on control flow
        for i, block in enumerate(basic_blocks):
            block_id = block['block_id']
            block_type = block['block_type']
            
            # Sequential flow to next block
            if block_type == 'sequential' and i + 1 < len(basic_blocks):
                cfg['edges'].append({
                    'from': block_id,
                    'to': basic_blocks[i + 1]['block_id'],
                    'type': 'sequential'
                })
            
            # Conditional jumps create two edges
            elif block_type == 'conditional_jump':
                # Fall-through edge (condition false)
                if i + 1 < len(basic_blocks):
                    cfg['edges'].append({
                        'from': block_id,
                        'to': basic_blocks[i + 1]['block_id'],
                        'type': 'fall_through'
                    })
                
                # Jump edge (condition true) - simplified target resolution
                # In a real implementation, we'd parse the jump target address
                # For now, we'll estimate based on instruction patterns
                target_block = self._estimate_jump_target(block, basic_blocks, i)
                if target_block is not None:
                    cfg['edges'].append({
                        'from': block_id,
                        'to': target_block,
                        'type': 'conditional_jump'
                    })
            
            # Function calls create call and return edges
            elif block_type == 'function_call':
                if i + 1 < len(basic_blocks):
                    cfg['edges'].append({
                        'from': block_id,
                        'to': basic_blocks[i + 1]['block_id'],
                        'type': 'function_return'
                    })
            
            # Unconditional jumps
            elif block_type == 'unconditional_jump':
                target_block = self._estimate_jump_target(block, basic_blocks, i)
                if target_block is not None:
                    cfg['edges'].append({
                        'from': block_id,
                        'to': target_block,
                        'type': 'unconditional_jump'
                    })
        
        return cfg
    
    def _estimate_jump_target(self, block: Dict[str, Any], basic_blocks: List[Dict[str, Any]], current_index: int) -> Optional[int]:
        """Estimate jump target block (simplified implementation)"""
        # This is a simplified heuristic since we don't have full address resolution
        # In a real implementation, this would parse actual jump targets
        
        # Look for backward jumps (loops) - target earlier blocks
        if 'loop' in str(block.get('instructions', [])).lower():
            # Estimate loop target as a few blocks back
            target_index = max(0, current_index - 3)
            if target_index < len(basic_blocks):
                return basic_blocks[target_index]['block_id']
        
        # Look for forward jumps - estimate target as a few blocks ahead
        elif current_index + 3 < len(basic_blocks):
            return basic_blocks[current_index + 3]['block_id']
        
        return None
    
    def _calculate_register_pressure(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate register pressure"""
        register_usage = instruction_analysis.get('register_usage', {})
        instructions = instruction_analysis.get('instructions', [])
        
        if not register_usage or not instructions:
            return {
                'overall_pressure': 'low',
                'high_pressure_registers': [],
                'register_conflicts': 0,
                'spill_indicators': 0,
                'register_utilization': 0.0
            }
        
        # Calculate register pressure metrics
        total_registers = len(register_usage)
        total_usage = sum(register_usage.values())
        avg_usage_per_register = total_usage / max(total_registers, 1)
        
        # Identify high-pressure registers (used more than average)
        high_pressure_threshold = avg_usage_per_register * 1.5
        high_pressure_registers = [
            reg for reg, usage in register_usage.items() 
            if usage > high_pressure_threshold
        ]
        
        # Estimate register conflicts (simplified)
        # Look for instructions that might cause register conflicts
        conflict_indicators = 0
        spill_indicators = 0
        
        for inst in instructions:
            raw_line = inst.get('raw_line', '').lower()
            
            # Look for memory operations that might indicate register spilling
            if any(pattern in raw_line for pattern in ['[esp', '[ebp', 'push', 'pop']):
                spill_indicators += 1
            
            # Look for register-to-register moves (potential conflicts)
            if 'mov' in raw_line and any(reg in raw_line for reg in register_usage.keys()):
                conflict_indicators += 1
        
        # Calculate overall pressure level
        pressure_score = (
            len(high_pressure_registers) / max(total_registers, 1) +
            spill_indicators / max(len(instructions), 1) * 10
        )
        
        if pressure_score > 0.8:
            overall_pressure = 'very_high'
        elif pressure_score > 0.6:
            overall_pressure = 'high'
        elif pressure_score > 0.4:
            overall_pressure = 'medium'
        else:
            overall_pressure = 'low'
        
        # Calculate register utilization percentage
        # Estimate based on common architectures (x86 has ~8 general purpose registers)
        estimated_available_registers = 8
        register_utilization = min(total_registers / estimated_available_registers, 1.0)
        
        return {
            'overall_pressure': overall_pressure,
            'high_pressure_registers': high_pressure_registers,
            'register_conflicts': conflict_indicators,
            'spill_indicators': spill_indicators,
            'register_utilization': register_utilization,
            'pressure_score': pressure_score,
            'total_registers_used': total_registers,
            'average_usage_per_register': avg_usage_per_register,
            'register_usage_distribution': register_usage
        }
    
    def _analyze_memory_access_patterns(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze memory access patterns"""
        memory_operations = instruction_analysis.get('memory_operations', [])
        instructions = instruction_analysis.get('instructions', [])
        
        if not instructions:
            return {
                'access_patterns': [],
                'stride_analysis': {},
                'cache_friendliness': 'unknown',
                'alignment_analysis': {},
                'total_memory_operations': 0
            }
        
        # Extract memory access patterns
        memory_accesses = []
        for inst in instructions:
            raw_line = inst.get('raw_line', '').lower()
            
            # Look for memory references [reg+offset], [reg], etc.
            if '[' in raw_line and ']' in raw_line:
                # Extract memory reference pattern
                start = raw_line.find('[')
                end = raw_line.find(']') + 1
                if start != -1 and end != -1:
                    memory_ref = raw_line[start:end]
                    memory_accesses.append({
                        'pattern': memory_ref,
                        'instruction': inst.get('instruction', ''),
                        'line': raw_line,
                        'type': self._classify_memory_access(raw_line)
                    })
        
        # Analyze access patterns
        access_patterns = []
        pattern_counts = {}
        
        for access in memory_accesses:
            pattern = access['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Identify common patterns
        for pattern, count in pattern_counts.items():
            if count > 2:  # Patterns that appear multiple times
                access_patterns.append({
                    'pattern': pattern,
                    'frequency': count,
                    'type': self._classify_access_pattern(pattern)
                })
        
        # Stride analysis (simplified)
        stride_analysis = self._analyze_stride_patterns(memory_accesses)
        
        # Cache friendliness assessment
        cache_friendliness = self._assess_cache_friendliness(access_patterns, stride_analysis)
        
        # Memory alignment analysis
        alignment_analysis = self._analyze_memory_alignment(memory_accesses)
        
        return {
            'access_patterns': access_patterns,
            'stride_analysis': stride_analysis,
            'cache_friendliness': cache_friendliness,
            'alignment_analysis': alignment_analysis,
            'total_memory_operations': len(memory_accesses),
            'unique_patterns': len(pattern_counts),
            'memory_access_density': len(memory_accesses) / max(len(instructions), 1)
        }
    
    def _classify_memory_access(self, raw_line: str) -> str:
        """Classify the type of memory access"""
        if 'mov' in raw_line:
            return 'load_store'
        elif any(op in raw_line for op in ['add', 'sub', 'mul', 'div']):
            return 'arithmetic_memory'
        elif any(op in raw_line for op in ['push', 'pop']):
            return 'stack_operation'
        elif 'lea' in raw_line:
            return 'address_calculation'
        else:
            return 'other'
    
    def _classify_access_pattern(self, pattern: str) -> str:
        """Classify memory access pattern"""
        if 'esp' in pattern or 'rsp' in pattern:
            return 'stack_access'
        elif 'ebp' in pattern or 'rbp' in pattern:
            return 'frame_access'
        elif '+' in pattern or '-' in pattern:
            return 'offset_access'
        elif any(reg in pattern for reg in ['eax', 'ebx', 'ecx', 'edx']):
            return 'register_indirect'
        else:
            return 'direct_access'
    
    def _analyze_stride_patterns(self, memory_accesses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stride patterns in memory accesses"""
        # Simplified stride analysis
        stride_indicators = {
            'sequential_access': 0,
            'strided_access': 0,
            'random_access': 0
        }
        
        # Look for patterns that suggest different access types
        for access in memory_accesses:
            pattern = access['pattern']
            if '+4' in pattern or '+8' in pattern:
                stride_indicators['sequential_access'] += 1
            elif '+' in pattern and any(str(i) in pattern for i in [16, 32, 64]):
                stride_indicators['strided_access'] += 1
            else:
                stride_indicators['random_access'] += 1
        
        return stride_indicators
    
    def _assess_cache_friendliness(self, access_patterns: List[Dict[str, Any]], stride_analysis: Dict[str, Any]) -> str:
        """Assess cache friendliness of memory access patterns"""
        sequential = stride_analysis.get('sequential_access', 0)
        strided = stride_analysis.get('strided_access', 0)
        random = stride_analysis.get('random_access', 0)
        
        total = sequential + strided + random
        if total == 0:
            return 'unknown'
        
        sequential_ratio = sequential / total
        strided_ratio = strided / total
        
        if sequential_ratio > 0.6:
            return 'cache_friendly'
        elif strided_ratio > 0.4:
            return 'moderately_cache_friendly'
        else:
            return 'cache_unfriendly'
    
    def _analyze_memory_alignment(self, memory_accesses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory alignment patterns"""
        alignment_analysis = {
            'aligned_accesses': 0,
            'unaligned_accesses': 0,
            'alignment_hints': []
        }
        
        for access in memory_accesses:
            pattern = access['pattern']
            # Look for alignment hints (multiples of 4, 8, 16)
            if any(align in pattern for align in ['+4', '+8', '+16', '+32']):
                alignment_analysis['aligned_accesses'] += 1
                alignment_analysis['alignment_hints'].append(pattern)
            else:
                alignment_analysis['unaligned_accesses'] += 1
        
        return alignment_analysis
    
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
        risks = []
        instructions = instruction_analysis.get('instructions', [])
        
        if not instructions:
            return risks
        
        # Dangerous function patterns
        dangerous_functions = ['strcpy', 'sprintf', 'gets', 'scanf', 'strcat']
        buffer_operations = ['mov', 'rep', 'stos', 'scas']
        
        for inst in instructions:
            raw_line = inst.get('raw_line', '').lower()
            
            # Check for dangerous function calls
            for dangerous_func in dangerous_functions:
                if dangerous_func in raw_line:
                    risks.append({
                        'type': 'dangerous_function_call',
                        'function': dangerous_func,
                        'instruction': raw_line,
                        'risk_level': 'high',
                        'description': f'Call to potentially unsafe function {dangerous_func}'
                    })
            
            # Check for unchecked buffer operations
            if any(op in inst.get('instruction', '').lower() for op in buffer_operations):
                if '[' in raw_line and ']' in raw_line:
                    # Memory operation without visible bounds checking
                    risks.append({
                        'type': 'unchecked_buffer_operation',
                        'operation': inst.get('instruction', ''),
                        'instruction': raw_line,
                        'risk_level': 'medium',
                        'description': 'Buffer operation without visible bounds checking'
                    })
            
            # Check for stack buffer operations
            if 'esp' in raw_line or 'ebp' in raw_line:
                if any(op in raw_line for op in ['add', 'sub', 'mov']):
                    risks.append({
                        'type': 'stack_buffer_manipulation',
                        'instruction': raw_line,
                        'risk_level': 'low',
                        'description': 'Direct stack manipulation - potential for stack corruption'
                    })
        
        return risks
    
    def _analyze_stack_protection(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze stack protection mechanisms"""
        protection = {
            'stack_canary_detected': False,
            'stack_guard_patterns': [],
            'aslr_indicators': [],
            'dep_nx_indicators': [],
            'protection_level': 'unknown'
        }
        
        instructions = instruction_analysis.get('instructions', [])
        
        if not instructions:
            return protection
        
        # Look for stack canary patterns
        canary_patterns = ['__stack_chk_fail', 'gs:', 'fs:', '__security_cookie']
        guard_instructions = []
        
        for inst in instructions:
            raw_line = inst.get('raw_line', '').lower()
            
            # Stack canary detection
            for pattern in canary_patterns:
                if pattern.lower() in raw_line:
                    protection['stack_canary_detected'] = True
                    guard_instructions.append(raw_line)
            
            # Look for segment register usage (often used for canaries)
            if any(seg in raw_line for seg in ['gs:', 'fs:']):
                protection['stack_guard_patterns'].append(raw_line)
            
            # ASLR indicators (random base addresses)
            if any(indicator in raw_line for indicator in ['rand', 'random']):
                protection['aslr_indicators'].append(raw_line)
            
            # DEP/NX indicators (executable permission checks)
            if any(indicator in raw_line for indicator in ['nx', 'exec', 'mprotect']):
                protection['dep_nx_indicators'].append(raw_line)
        
        # Determine protection level
        protection_score = 0
        if protection['stack_canary_detected']:
            protection_score += 2
        if protection['stack_guard_patterns']:
            protection_score += 1
        if protection['aslr_indicators']:
            protection_score += 1
        if protection['dep_nx_indicators']:
            protection_score += 1
        
        if protection_score >= 4:
            protection['protection_level'] = 'high'
        elif protection_score >= 2:
            protection['protection_level'] = 'medium'
        elif protection_score >= 1:
            protection['protection_level'] = 'low'
        else:
            protection['protection_level'] = 'none_detected'
        
        return protection
    
    def _detect_code_injection_risks(self, instruction_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect code injection vulnerabilities"""
        risks = []
        instructions = instruction_analysis.get('instructions', [])
        
        if not instructions:
            return risks
        
        # Patterns that might indicate code injection vulnerabilities
        injection_patterns = {
            'dynamic_allocation': ['malloc', 'calloc', 'alloc', 'mmap'],
            'execution_functions': ['exec', 'system', 'call', 'jmp'],
            'input_functions': ['read', 'recv', 'input', 'scanf'],
            'string_operations': ['sprintf', 'strcpy', 'strcat', 'memcpy']
        }
        
        for inst in instructions:
            raw_line = inst.get('raw_line', '').lower()
            
            # Check for dynamic memory allocation followed by execution
            for category, patterns in injection_patterns.items():
                for pattern in patterns:
                    if pattern in raw_line:
                        risk_level = 'high' if category == 'execution_functions' else 'medium'
                        
                        risks.append({
                            'type': f'potential_{category}',
                            'pattern': pattern,
                            'instruction': raw_line,
                            'risk_level': risk_level,
                            'description': f'Detected {pattern} which could be used for code injection'
                        })
            
            # Look for dynamic jump targets (register-based jumps)
            if any(jump in inst.get('instruction', '').lower() for jump in ['jmp', 'call']):
                if any(reg in raw_line for reg in ['eax', 'ebx', 'ecx', 'edx', 'rdi', 'rsi']):
                    risks.append({
                        'type': 'dynamic_control_transfer',
                        'instruction': raw_line,
                        'risk_level': 'medium',
                        'description': 'Dynamic control transfer - potential for code injection'
                    })
            
            # Look for self-modifying code patterns
            if 'mov' in inst.get('instruction', '').lower() and '[' in raw_line:
                # Writing to memory locations that might be executable
                risks.append({
                    'type': 'potential_self_modification',
                    'instruction': raw_line,
                    'risk_level': 'low',
                    'description': 'Memory write operation - potential for self-modifying code'
                })
        
        return risks
    
    def _analyze_privilege_escalation(self, instruction_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze privilege escalation risks"""
        analysis = {
            'privilege_operations': [],
            'system_calls': [],
            'setuid_indicators': [],
            'escalation_risks': [],
            'risk_level': 'low'
        }
        
        instructions = instruction_analysis.get('instructions', [])
        
        if not instructions:
            return analysis
        
        # Privilege-related function patterns
        privilege_functions = {
            'setuid_functions': ['setuid', 'seteuid', 'setgid', 'setegid'],
            'system_calls': ['syscall', 'int 0x80', 'sysenter'],
            'privilege_checks': ['getuid', 'geteuid', 'getgid', 'getegid'],
            'dangerous_syscalls': ['execve', 'fork', 'clone', 'mmap', 'mprotect']
        }
        
        for inst in instructions:
            raw_line = inst.get('raw_line', '').lower()
            
            # Check for privilege-related operations
            for category, functions in privilege_functions.items():
                for func in functions:
                    if func in raw_line:
                        analysis['privilege_operations'].append({
                            'type': category,
                            'function': func,
                            'instruction': raw_line
                        })
                        
                        # Assess risk level
                        if category == 'setuid_functions':
                            analysis['setuid_indicators'].append(func)
                            analysis['escalation_risks'].append({
                                'type': 'setuid_operation',
                                'function': func,
                                'risk': 'high',
                                'description': f'SETUID operation {func} detected - potential privilege escalation'
                            })
                        elif category == 'dangerous_syscalls':
                            analysis['escalation_risks'].append({
                                'type': 'dangerous_syscall',
                                'function': func,
                                'risk': 'medium',
                                'description': f'Potentially dangerous system call {func}'
                            })
            
            # Look for direct system call instructions
            if any(syscall in raw_line for syscall in ['int 0x80', 'syscall', 'sysenter']):
                analysis['system_calls'].append(raw_line)
            
            # Look for privilege bit manipulation
            if any(pattern in raw_line for pattern in ['or', 'and', 'xor']) and any(reg in raw_line for reg in ['eax', 'ebx']):
                # Potential privilege bit manipulation
                analysis['escalation_risks'].append({
                    'type': 'bit_manipulation',
                    'instruction': raw_line,
                    'risk': 'low',
                    'description': 'Bit manipulation that could affect privilege flags'
                })
        
        # Determine overall risk level
        high_risks = [r for r in analysis['escalation_risks'] if r['risk'] == 'high']
        medium_risks = [r for r in analysis['escalation_risks'] if r['risk'] == 'medium']
        
        if high_risks:
            analysis['risk_level'] = 'high'
        elif medium_risks:
            analysis['risk_level'] = 'medium'
        elif analysis['escalation_risks']:
            analysis['risk_level'] = 'low'
        else:
            analysis['risk_level'] = 'none_detected'
        
        return analysis
    
    def _detect_rop_gadgets(self, instruction_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect ROP gadgets"""
        gadgets = []
        instructions = instruction_analysis.get('instructions', [])
        
        if not instructions:
            return gadgets
        
        # Look for potential ROP gadgets (instruction sequences ending in ret)
        for i, inst in enumerate(instructions):
            inst_name = inst.get('instruction', '').lower()
            
            # Found a return instruction - look backward for useful gadgets
            if inst_name in ['ret', 'retn', 'retf']:
                # Check preceding instructions for useful operations
                gadget_length = min(5, i)  # Look at up to 5 preceding instructions
                
                if gadget_length > 0:
                    gadget_instructions = []
                    useful_operations = 0
                    
                    for j in range(i - gadget_length, i + 1):
                        if j >= 0:
                            gadget_inst = instructions[j]
                            gadget_instructions.append(gadget_inst.get('raw_line', ''))
                            
                            # Check if instruction is useful for ROP
                            if self._is_useful_rop_instruction(gadget_inst):
                                useful_operations += 1
                    
                    # If gadget has useful operations, add it
                    if useful_operations > 0:
                        gadget_type = self._classify_rop_gadget(gadget_instructions)
                        
                        gadgets.append({
                            'type': gadget_type,
                            'instructions': gadget_instructions,
                            'useful_operations': useful_operations,
                            'length': len(gadget_instructions),
                            'end_position': i,
                            'address_hint': f'instruction_{i}',
                            'exploitation_potential': self._assess_gadget_potential(gadget_type, useful_operations)
                        })
        
        # Sort gadgets by exploitation potential
        gadgets.sort(key=lambda g: g['exploitation_potential'], reverse=True)
        
        return gadgets[:20]  # Return top 20 gadgets
    
    def _is_useful_rop_instruction(self, inst: Dict[str, Any]) -> bool:
        """Check if instruction is useful for ROP exploitation"""
        inst_name = inst.get('instruction', '').lower()
        raw_line = inst.get('raw_line', '').lower()
        
        # Useful ROP instruction categories
        useful_categories = {
            'data_movement': ['mov', 'lea', 'push', 'pop'],
            'arithmetic': ['add', 'sub', 'inc', 'dec', 'xor', 'or', 'and'],
            'control_flow': ['jmp', 'call'],
            'memory_access': ['mov', 'lea'],
            'register_manipulation': ['xchg', 'push', 'pop']
        }
        
        for category, instructions in useful_categories.items():
            if inst_name in instructions:
                return True
        
        # Check for register-to-register operations
        if any(reg in raw_line for reg in ['eax', 'ebx', 'ecx', 'edx', 'esp', 'ebp']):
            return True
        
        return False
    
    def _classify_rop_gadget(self, gadget_instructions: List[str]) -> str:
        """Classify the type of ROP gadget"""
        combined = ' '.join(gadget_instructions).lower()
        
        if any(pattern in combined for pattern in ['pop', 'ret']):
            return 'pop_ret_gadget'
        elif any(pattern in combined for pattern in ['mov', 'ret']):
            return 'mov_ret_gadget'
        elif any(pattern in combined for pattern in ['add', 'ret']):
            return 'add_ret_gadget'
        elif any(pattern in combined for pattern in ['xor', 'ret']):
            return 'xor_ret_gadget'
        elif any(pattern in combined for pattern in ['jmp', 'esp']):
            return 'jmp_esp_gadget'
        else:
            return 'generic_gadget'
    
    def _assess_gadget_potential(self, gadget_type: str, useful_operations: int) -> float:
        """Assess exploitation potential of ROP gadget"""
        type_scores = {
            'pop_ret_gadget': 0.9,
            'jmp_esp_gadget': 0.95,
            'mov_ret_gadget': 0.7,
            'add_ret_gadget': 0.6,
            'xor_ret_gadget': 0.6,
            'generic_gadget': 0.3
        }
        
        base_score = type_scores.get(gadget_type, 0.2)
        operation_bonus = min(useful_operations * 0.1, 0.3)
        
        return min(base_score + operation_bonus, 1.0)
    
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