"""
Matrix Agent Base Classes for Open-Sourcefy Matrix Architecture
Enhanced agent framework with LangChain integration and standardized patterns
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .config_manager import get_config_manager
from .ai_engine_interface import get_ai_engine, AIRequest, AIResponse
from .file_utils import OutputManager, JsonManager
from .binary_utils import BinaryInfo


class AgentStatus(Enum):
    """Agent execution status"""
    PENDING = "pending"
    INITIALIZING = "initializing" 
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class AgentType(Enum):
    """Matrix agent types"""
    MACHINE = "machine"           # Hardware-level operations
    PROGRAM = "program"           # AI software entities
    SOFTWARE_FRAGMENT = "fragment"  # Glitch entities
    OPERATOR = "operator"         # Human-machine interfaces
    AI_COMMANDER = "ai_commander" # Military AI
    COLLECTIVE_AI = "collective"  # Machine civilization


@dataclass
class AgentResult:
    """Standardized agent result structure"""
    agent_id: int
    agent_name: str
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def is_success(self) -> bool:
        """Check if agent execution was successful"""
        return self.status == AgentStatus.SUCCESS
    
    def get_quality_score(self) -> float:
        """Get quality score from metadata"""
        return self.metadata.get('quality_score', 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'status': self.status.value,
            'data': self.data,
            'metadata': self.metadata,
            'execution_time': self.execution_time,
            'error_message': self.error_message,
            'output_files': self.output_files,
            'timestamp': self.timestamp
        }


class MatrixAgentBase(ABC):
    """Base class for all Matrix agents"""
    
    def __init__(self, agent_id: int, agent_name: str, matrix_character: str,
                 agent_type: AgentType = AgentType.PROGRAM):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.matrix_character = matrix_character
        self.agent_type = agent_type
        
        # Configuration and utilities
        self.config = get_config_manager()
        self.ai_engine = get_ai_engine()
        self.output_manager = OutputManager()
        self.json_manager = JsonManager()
        
        # Logging
        self.logger = self._setup_logging()
        
        # Execution tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.status = AgentStatus.PENDING
        
        # Dependencies and prerequisites
        self.dependencies = self.get_dependencies()
        self.prerequisites = self.get_prerequisites()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(f"Matrix.{self.matrix_character}")
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                f'[{self.matrix_character}] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the agent's main functionality"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get agent description"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[int]:
        """Get list of agent IDs this agent depends on"""
        pass
    
    def get_prerequisites(self) -> List[str]:
        """Get list of prerequisites for this agent"""
        return []
    
    def validate_dependencies(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that all dependencies are satisfied"""
        issues = []
        agent_results = context.get('agent_results', {})
        
        for dep_id in self.dependencies:
            if dep_id not in agent_results:
                issues.append(f"Missing dependency: Agent {dep_id}")
            else:
                result = agent_results[dep_id]
                if isinstance(result, dict) and result.get('status') != 'success':
                    issues.append(f"Dependency failed: Agent {dep_id}")
                elif isinstance(result, AgentResult) and not result.is_success():
                    issues.append(f"Dependency failed: Agent {dep_id}")
        
        return len(issues) == 0, issues
    
    def validate_prerequisites(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate that all prerequisites are met"""
        issues = []
        
        for prerequisite in self.prerequisites:
            if prerequisite == "binary_file":
                binary_path = context.get('binary_path')
                if not binary_path or not Path(binary_path).exists():
                    issues.append("Binary file not found")
            
            elif prerequisite == "output_paths":
                if 'output_paths' not in context:
                    issues.append("Output paths not configured")
            
            elif prerequisite == "ghidra":
                ghidra_path = self.config.get_path('ghidra_home')
                if not ghidra_path or not Path(ghidra_path).exists():
                    issues.append("Ghidra not found")
        
        return len(issues) == 0, issues
    
    def run(self, context: Dict[str, Any]) -> AgentResult:
        """Run the agent with full lifecycle management"""
        self.logger.info(f"Starting {self.matrix_character} (Agent {self.agent_id})")
        self.start_time = time.time()
        self.status = AgentStatus.INITIALIZING
        
        try:
            # Validate dependencies
            deps_valid, dep_issues = self.validate_dependencies(context)
            if not deps_valid:
                self.status = AgentStatus.FAILED
                error_msg = f"Dependency validation failed: {', '.join(dep_issues)}"
                self.logger.error(error_msg)
                return self._create_error_result(error_msg)
            
            # Validate prerequisites
            prereq_valid, prereq_issues = self.validate_prerequisites(context)
            if not prereq_valid:
                self.status = AgentStatus.FAILED
                error_msg = f"Prerequisites not met: {', '.join(prereq_issues)}"
                self.logger.error(error_msg)
                return self._create_error_result(error_msg)
            
            # Execute agent
            self.status = AgentStatus.RUNNING
            result = self.execute(context)
            
            # Update timing
            self.end_time = time.time()
            result.execution_time = self.end_time - self.start_time
            
            # Save output
            self._save_agent_output(result, context)
            
            # Log result
            if result.is_success():
                self.logger.info(f"{self.matrix_character} completed successfully in {result.execution_time:.2f}s")
            else:
                self.logger.error(f"{self.matrix_character} failed: {result.error_message}")
            
            self.status = result.status
            return result
            
        except Exception as e:
            self.end_time = time.time()
            self.status = AgentStatus.FAILED
            error_msg = f"Unexpected error in {self.matrix_character}: {str(e)}"
            self.logger.exception(error_msg)
            return self._create_error_result(error_msg)
    
    def _create_error_result(self, error_message: str) -> AgentResult:
        """Create error result"""
        execution_time = 0.0
        if self.start_time:
            execution_time = time.time() - self.start_time
        
        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            status=AgentStatus.FAILED,
            error_message=error_message,
            execution_time=execution_time,
            metadata={'matrix_character': self.matrix_character}
        )
    
    def _save_agent_output(self, result: AgentResult, context: Dict[str, Any]) -> None:
        """Save agent output to standardized location"""
        output_paths = context.get('output_paths', {})
        if 'agents' in output_paths:
            try:
                self.output_manager.save_agent_output(
                    self.agent_id, 
                    self.agent_name.lower(),
                    result.to_dict(),
                    output_paths
                )
            except Exception as e:
                self.logger.warning(f"Failed to save agent output: {e}")
    
    def generate_ai_response(self, prompt: str, context: Optional[str] = None,
                           system_message: Optional[str] = None, **kwargs) -> AIResponse:
        """Generate AI response using configured engine"""
        # Add agent context to system message
        agent_context = f"You are {self.matrix_character}, {self.get_description()}"
        if system_message:
            system_message = f"{agent_context}\n\n{system_message}"
        else:
            system_message = agent_context
        
        request = AIRequest(
            prompt=prompt,
            context=context,
            system_message=system_message,
            **kwargs
        )
        
        return self.ai_engine.generate_response(request)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'matrix_character': self.matrix_character,
            'agent_type': self.agent_type.value,
            'description': self.get_description(),
            'dependencies': self.dependencies,
            'prerequisites': self.prerequisites,
            'status': self.status.value
        }


class AnalysisAgent(MatrixAgentBase):
    """Base class for analysis-focused Matrix agents"""
    
    def __init__(self, agent_id: int, agent_name: str, matrix_character: str):
        super().__init__(agent_id, agent_name, matrix_character, AgentType.PROGRAM)
    
    def analyze_with_ai(self, data: Any, analysis_type: str, 
                       additional_context: Optional[str] = None) -> Dict[str, Any]:
        """Perform AI-powered analysis"""
        # Prepare data for analysis
        if isinstance(data, (dict, list)):
            data_str = str(data)[:2000]  # Limit context size
        else:
            data_str = str(data)[:2000]
        
        # Build analysis prompt
        prompt = f"Analyze the following {analysis_type} data and provide insights:\n\n{data_str}"
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        
        # Generate AI response
        response = self.generate_ai_response(
            prompt=prompt,
            system_message=f"You are an expert in {analysis_type} analysis."
        )
        
        if response.success:
            return {
                'analysis_result': response.content,
                'confidence': 0.8,  # Default confidence
                'ai_metadata': response.metadata
            }
        else:
            self.logger.warning(f"AI analysis failed: {response.error}")
            return {
                'analysis_result': f"Analysis failed: {response.error}",
                'confidence': 0.0,
                'ai_metadata': {}
            }


class DecompilationAgent(MatrixAgentBase):
    """Base class for decompilation-focused Matrix agents"""
    
    def __init__(self, agent_id: int, agent_name: str, matrix_character: str):
        super().__init__(agent_id, agent_name, matrix_character, AgentType.PROGRAM)
    
    def decompile_with_ai(self, binary_info: BinaryInfo, functions: List[Dict[str, Any]],
                         context: Optional[str] = None) -> Dict[str, Any]:
        """Perform AI-powered decompilation enhancement"""
        # Build decompilation prompt
        prompt = f"""
Enhance the decompilation of a {binary_info.format.value} binary:

Binary Info:
- Architecture: {binary_info.architecture.value}
- Size: {binary_info.size} bytes
- Entry Point: 0x{binary_info.entry_point:08x} (if available)
- Functions: {len(functions)} identified

Functions to enhance:
"""
        
        # Add function details (limited to avoid token limits)
        for i, func in enumerate(functions[:5]):  # Limit to first 5 functions
            prompt += f"\nFunction {i+1}: {func.get('name', 'unknown')} at 0x{func.get('address', 0):08x}"
            if 'disassembly' in func:
                prompt += f"\nDisassembly snippet: {func['disassembly'][:200]}..."
        
        if context:
            prompt += f"\n\nAdditional context: {context}"
        
        prompt += "\n\nProvide enhanced decompilation with proper function names, variable types, and logic flow."
        
        # Generate AI response
        response = self.generate_ai_response(
            prompt=prompt,
            system_message="You are an expert reverse engineer and decompilation specialist."
        )
        
        if response.success:
            return {
                'enhanced_functions': response.content,
                'confidence': 0.75,
                'ai_metadata': response.metadata
            }
        else:
            self.logger.warning(f"AI decompilation enhancement failed: {response.error}")
            return {
                'enhanced_functions': "Enhancement failed",
                'confidence': 0.0,
                'ai_metadata': {}
            }


class CompilationAgent(MatrixAgentBase):
    """Base class for compilation-focused Matrix agents"""
    
    def __init__(self, agent_id: int, agent_name: str, matrix_character: str):
        super().__init__(agent_id, agent_name, matrix_character, AgentType.PROGRAM)
    
    def generate_build_script_with_ai(self, source_files: List[str], 
                                    target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate build script using AI"""
        prompt = f"""
Generate a build script for the following source files:

Source Files:
{chr(10).join(f"- {file}" for file in source_files)}

Target Information:
- Architecture: {target_info.get('architecture', 'x86')}
- Compiler: {target_info.get('compiler', 'msvc')}
- Optimization: {target_info.get('optimization', 'O2')}

Generate appropriate build files (CMakeLists.txt, Makefile, or .vcxproj) based on the target environment.
"""
        
        response = self.generate_ai_response(
            prompt=prompt,
            system_message="You are an expert build system engineer."
        )
        
        if response.success:
            return {
                'build_script': response.content,
                'confidence': 0.8,
                'ai_metadata': response.metadata
            }
        else:
            self.logger.warning(f"AI build script generation failed: {response.error}")
            return {
                'build_script': "# Build script generation failed",
                'confidence': 0.0,
                'ai_metadata': {}
            }


# Convenience functions for agent registration and discovery
_registered_agents: Dict[int, type] = {}


def register_matrix_agent(agent_id: int, agent_class: type):
    """Register a Matrix agent class"""
    _registered_agents[agent_id] = agent_class


def get_registered_agents() -> Dict[int, type]:
    """Get all registered agent classes"""
    return _registered_agents.copy()


def create_agent(agent_id: int, **kwargs) -> Optional[MatrixAgentBase]:
    """Create agent instance by ID"""
    agent_class = _registered_agents.get(agent_id)
    if agent_class:
        try:
            return agent_class(**kwargs)
        except Exception as e:
            logging.error(f"Failed to create agent {agent_id}: {e}")
            return None
    else:
        logging.error(f"Agent {agent_id} not registered")
        return None