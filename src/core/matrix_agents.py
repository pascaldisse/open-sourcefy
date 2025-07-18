"""
Enhanced Matrix Agent Base Classes for Phase 2 Refactor
Implements the 17-agent Matrix-themed architecture with LangChain integration
All agents are Matrix agents with centralized dependency management and LangChain support
"""

import abc
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config_manager import get_config_manager
from .ai_engine_interface import get_ai_engine

# LangChain imports - STRICT MODE: fail fast in production
try:
    from langchain.agents import AgentExecutor, ReActTextWorldAgent
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool
    from langchain.schema import AgentAction, AgentFinish
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Only allow import errors during testing/dependency checking
    import os
    if os.getenv('MATRIX_TESTING') == 'true':
        LANGCHAIN_AVAILABLE = False
        # Create stub classes for testing only
        class AgentExecutor: 
            def __init__(self, *args, **kwargs): pass
            @classmethod
            def from_agent_and_tools(cls, *args, **kwargs): return cls()
        class ReActTextWorldAgent: 
            def __init__(self, *args, **kwargs): pass
            @classmethod
            def from_llm_and_tools(cls, *args, **kwargs): return cls()
        class ConversationBufferMemory: 
            def __init__(self, *args, **kwargs): pass
        class Tool: 
            def __init__(self, *args, **kwargs): pass
    else:
        raise ImportError("LangChain is required for Matrix agents. Install with: pip install langchain")


class AgentStatus(Enum):
    """Agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class MatrixCharacter(Enum):
    """Matrix character types for themed agents"""
    DEUS_EX_MACHINA = "deus_ex_machina"  # Agent 0
    SENTINEL = "sentinel"                # Agent 1
    ARCHITECT = "architect"              # Agent 2
    MEROVINGIAN = "merovingian"          # Agent 3
    AGENT_SMITH = "agent_smith"          # Agent 4
    NEO = "neo"                          # Agent 5
    TWINS = "twins"                      # Agent 6
    TRAINMAN = "trainman"                # Agent 7
    KEYMAKER = "keymaker"                # Agent 8
    COMMANDER_LOCKE = "commander_locke"  # Agent 9
    MACHINE = "machine"                  # Agent 10
    ORACLE = "oracle"                    # Agent 11
    LINK = "link"                        # Agent 12
    AGENT_JOHNSON = "agent_johnson"      # Agent 13
    CLEANER = "cleaner"                  # Agent 14
    ANALYST = "analyst"                  # Agent 15
    AGENT_BROWN = "agent_brown"          # Agent 16


@dataclass
class AgentResult:
    """Standardized agent result structure"""
    agent_id: int
    agent_name: str
    matrix_character: str
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None


class MatrixAgent(abc.ABC):
    """Enhanced base class for all Matrix agents with LangChain integration and centralized dependencies"""
    
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter):
        self.agent_id = agent_id
        self.matrix_character = matrix_character
        self.agent_name = f"Agent{agent_id:02d}_{matrix_character.value.title()}"
        
        # Centralized dependency management - SINGLE SOURCE OF TRUTH
        self.dependencies = MATRIX_DEPENDENCIES.get(agent_id, [])
        self.status = AgentStatus.PENDING
        
        # Setup shared components
        self.logger = self._setup_logger()
        self.config = get_config_manager()
        self.ai_engine = get_ai_engine()
        self.output_manager = None  # Set during execution
        
        # LangChain integration - ALL AGENTS ARE LANGCHAIN AGENTS
        self.memory = ConversationBufferMemory()
        self.langchain_tools = self._setup_langchain_tools()
        self.langchain_agent = None  # Initialized during execution
        
        # Execution settings
        self.timeout = self.config.get_value(f'agents.agent_{agent_id:02d}.timeout', 300)
        self.max_retries = self.config.get_value(f'agents.agent_{agent_id:02d}.max_retries', 2)
        self.retry_count = 0

    def _setup_logger(self) -> logging.Logger:
        """Setup standardized logger for the agent"""
        logger = logging.getLogger(f"Matrix.{self.agent_name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[{self.matrix_character.value.upper()}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger

    def _setup_langchain_tools(self) -> List[Tool]:
        """Setup LangChain tools for this agent - to be extended by subclasses"""
        base_tools = [
            Tool(
                name="matrix_logger",
                description="Log Matrix operations and findings",
                func=lambda msg: self.logger.info(f"Matrix Log: {msg}")
            ),
            Tool(
                name="matrix_status",
                description="Get current Matrix agent status and progress",
                func=lambda: f"Agent {self.agent_id} ({self.matrix_character.value}) - Status: {self.status.value}"
            )
        ]
        return base_tools

    def _initialize_langchain_agent(self) -> AgentExecutor:
        """Initialize LangChain agent executor for this Matrix agent"""
        if not self.langchain_agent:
            if LANGCHAIN_AVAILABLE:
                # Create LangChain agent with Matrix-specific configuration
                agent = ReActDocstoreAgent.from_llm_and_tools(
                    llm=self.ai_engine,
                    tools=self.langchain_tools,
                    memory=self.memory
                )
                self.langchain_agent = AgentExecutor.from_agent_and_tools(
                    agent=agent,
                    tools=self.langchain_tools,
                    memory=self.memory,
                    verbose=True,
                    max_iterations=10
                )
            else:
                # Testing mode - create stub agent
                self.langchain_agent = AgentExecutor()
        return self.langchain_agent

    @abc.abstractmethod
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's specific Matrix task - to be implemented by subclasses"""
        pass

    def get_matrix_description(self) -> str:
        """Get Matrix character description - to be implemented by subclasses"""
        return f"{self.matrix_character.value.replace('_', ' ').title()} agent"

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if agent can execute based on dependencies"""
        agent_results = context.get('agent_results', {})
        
        for dep_id in self.dependencies:
            dep_result = agent_results.get(dep_id)
            if not dep_result or dep_result.status != AgentStatus.SUCCESS:
                return False
        return True

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Main execution wrapper with standardized error handling"""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.RUNNING
            self.logger.info(f"🎬 {self.get_matrix_description()} enters the Matrix...")
            
            # Setup output manager
            self.output_manager = context.get('output_manager')
            
            # Pre-execution validation
            try:
                validation_result = self._validate_prerequisites(context)
                if validation_result is False:
                    return self._create_failure_result("Prerequisites not satisfied", start_time)
            except Exception as e:
                return self._create_failure_result(f"Prerequisites validation failed: {str(e)}", start_time)
            
            # Execute Matrix task
            task_data = self.execute_matrix_task(context)
            
            # Post-execution processing
            self._post_process_results(task_data, context)
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ {self.matrix_character.value.title()} completed mission in {execution_time:.2f}s")
            
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                matrix_character=self.matrix_character.value,
                status=AgentStatus.SUCCESS,
                data=task_data,
                execution_time=execution_time,
                metadata=self._generate_metadata(context)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Matrix breach detected: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return self._create_failure_result(error_msg, start_time, execution_time)

    def _validate_prerequisites(self, context: Dict[str, Any]) -> bool:
        """Validate prerequisites for execution"""
        self.logger.debug(f"Validating prerequisites for Agent{self.agent_id:02d}")
        self.logger.debug(f"Available context keys: {list(context.keys())}")
        
        # Check dependencies
        for dep_id in self.dependencies:
            if not self._is_dependency_satisfied(dep_id, context):
                self.logger.error(f"Dependency Agent{dep_id:02d} not satisfied")
                return False
        
        # Check required context keys
        required_keys = self._get_required_context_keys()
        self.logger.debug(f"Required context keys: {required_keys}")
        for key in required_keys:
            if key not in context:
                self.logger.error(f"Required context key missing: {key}")
                return False
                
        self.logger.debug("All prerequisites satisfied")
        return True

    def _get_required_context_keys(self) -> List[str]:
        """Get required context keys - can be overridden by subclasses"""
        base_keys = ['binary_path', 'output_paths']
        if self.agent_id > 1:  # All agents except Sentinel need Agent 1 results
            base_keys.append('agent_results')
        return base_keys

    def _is_dependency_satisfied(self, dep_id: int, context: Dict[str, Any]) -> bool:
        """Check if dependency is satisfied - either from live results or cache"""
        # First check live agent results
        agent_results = context.get('agent_results', {})
        dep_result = agent_results.get(dep_id)
        if dep_result and dep_result.status == AgentStatus.SUCCESS:
            return True
            
        # If no live results, check cache files
        output_paths = context.get('output_paths', {})
        if output_paths and 'agents' in output_paths:
            from pathlib import Path
            import json
            
            # Check multiple possible cache locations for the dependency
            cache_locations = [
                Path(output_paths['agents']) / f"agent_{dep_id:02d}" / "cache.json",
                Path(output_paths['agents']) / f"agent_{dep_id:02d}" / f"agent_{dep_id:02d}_results.json",
                Path(output_paths['agents']) / f"agent_{dep_id:02d}" / "agent_result.json"
            ]
            
            # Also check agent-specific directories (e.g., agent_05_neo)
            from pathlib import Path
            import glob
            agents_dir = Path(output_paths['agents'])
            if agents_dir.exists():
                # Find any directory that starts with agent_{dep_id:02d}_
                agent_dirs = list(agents_dir.glob(f"agent_{dep_id:02d}_*"))
                for agent_dir in agent_dirs:
                    cache_locations.extend([
                        agent_dir / "agent_result.json",
                        agent_dir / "cache.json",
                        agent_dir / f"agent_{dep_id:02d}_results.json"
                    ])
            
            for cache_path in cache_locations:
                if cache_path.exists():
                    try:
                        with open(cache_path, 'r') as f:
                            cache_data = json.load(f)
                            # Check if cache indicates successful completion
                            if (cache_data.get('status') == 'SUCCESS' or 
                                cache_data.get('status') == 'success' or
                                cache_data.get('agent_status') == 'SUCCESS'):
                                self.logger.debug(f"Dependency Agent{dep_id:02d} satisfied via cache: {cache_path}")
                                return True
                    except (json.JSONDecodeError, Exception) as e:
                        self.logger.debug(f"Could not read cache file {cache_path}: {e}")
                        continue
        
        return False

    def _post_process_results(self, task_data: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Post-process results - can be overridden by subclasses"""
        # Save results to output directory if output manager available
        if self.output_manager:
            agent_output_dir = Path(context['output_paths']['agents']) / f"agent_{self.agent_id:02d}_{self.matrix_character.value}"
            agent_output_dir.mkdir(exist_ok=True)
            
            # Save main results
            self.output_manager.save_json(
                task_data, 
                agent_output_dir / f"{self.agent_name}_results.json"
            )

    def _generate_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate standard metadata for the result"""
        return {
            'matrix_character': self.matrix_character.value,
            'agent_version': '2.0',
            'binary_path': context.get('binary_path', ''),
            'dependencies_satisfied': [dep for dep in self.dependencies if self._is_dependency_satisfied(dep, context)],
            'timeout_used': self.timeout,
            'retry_count': self.retry_count
        }

    def _create_failure_result(self, error_msg: str, start_time: float, execution_time: float = None) -> AgentResult:
        """Create standardized failure result"""
        if execution_time is None:
            execution_time = time.time() - start_time
            
        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            matrix_character=self.matrix_character.value,
            status=AgentStatus.FAILED,
            data={},
            execution_time=execution_time,
            error_message=error_msg,
            metadata={'matrix_character': self.matrix_character.value}
        )

    @staticmethod
    def get_result_status(result: Any) -> str:
        """Get status from result object regardless of type"""
        if hasattr(result, 'status'):
            status = result.status
            # Convert enum to string if needed
            if hasattr(status, 'value'):
                return status.value
            return str(status)
        elif isinstance(result, dict):
            return result.get('status', 'unknown')
        else:
            return 'unknown'

    @staticmethod
    def get_result_data(result: Any) -> Dict[str, Any]:
        """Get data from result object regardless of type"""
        if hasattr(result, 'data'):
            return result.data if isinstance(result.data, dict) else {}
        elif isinstance(result, dict):
            return result.get('data', {})
        else:
            return {}

    @staticmethod
    def get_agent_data_safely(agent_data: Any, key: str) -> Any:
        """Safely get data from agent result, handling both dict and AgentResult objects"""
        if hasattr(agent_data, 'data'):
            if isinstance(agent_data.data, dict):
                return agent_data.data.get(key)
        elif isinstance(agent_data, dict):
            data = agent_data.get('data', {})
            if isinstance(data, dict):
                return data.get(key)
        return None

    @staticmethod
    def is_agent_successful(result: Any) -> bool:
        """Check if agent result indicates success"""
        status = MatrixAgent.get_result_status(result)
        return status in ['success', 'SUCCESS', AgentStatus.SUCCESS.value]

    def __str__(self) -> str:
        return f"{self.agent_name} ({self.matrix_character.value})"

    def __repr__(self) -> str:
        return f"<MatrixAgent(id={self.agent_id}, character={self.matrix_character.value}, status={self.status.value})>"


class AnalysisAgent(MatrixAgent):
    """Base class for analysis-focused agents (Phase B: Agents 1-8)"""
    
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter):
        super().__init__(agent_id, matrix_character)
        self.analysis_type = "binary_analysis"

    def _get_required_context_keys(self) -> List[str]:
        """Analysis agents require binary path and shared memory"""
        keys = super()._get_required_context_keys()
        keys.extend(['shared_memory'])
        return keys


class DecompilerAgent(MatrixAgent):
    """Base class for decompilation-focused agents"""
    
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter):
        super().__init__(agent_id, matrix_character)
        self.decompiler_type = "advanced"
        
        # Decompiler-specific settings
        self.ghidra_timeout = self.config.get_value('ghidra.timeout', 600)
        self.quality_threshold = self.config.get_value('decompilation.quality_threshold', 0.7)


class ReconstructionAgent(MatrixAgent):
    """Base class for reconstruction-focused agents (Phase C: Agents 9-16)"""
    
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter):
        super().__init__(agent_id, matrix_character)
        self.reconstruction_type = "advanced"


class ValidationAgent(MatrixAgent):
    """Base class for validation and testing agents"""
    
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter):
        super().__init__(agent_id, matrix_character)
        self.validation_type = "comprehensive"
        
        # Validation-specific settings
        self.quality_threshold = self.config.get_value('validation.quality_threshold', 0.75)
        self.completeness_threshold = self.config.get_value('validation.completeness_threshold', 0.7)


# Matrix Agent Dependencies - CENTRALIZED SINGLE SOURCE OF TRUTH
# NO DEPENDENCIES ANYWHERE ELSE - ALL MANAGED HERE
# RENUMBERED: Agents numbered according to execution order
MATRIX_DEPENDENCIES = {
    0: [],                    # Deus Ex Machina - Master orchestrator, no dependencies
    1: [],                    # Sentinel - Entry point, no dependencies
    2: [1],                   # Architect - Depends on Sentinel
    3: [1],                   # Merovingian - Depends on Sentinel  
    4: [1],                   # Agent Smith - Depends on Sentinel
    5: [1, 2, 3],             # Neo - Depends on Sentinel, Architect, and Merovingian
    6: [1, 2],                # Trainman - Depends on Sentinel and Architect (formerly 7)
    7: [1, 2],                # Keymaker - Depends on Sentinel and Architect (formerly 8)
    8: [1, 5, 6, 7],          # Commander Locke - Depends on Sentinel (import data) + Phase B agents (formerly 9)
    9: [5, 8],                # The Machine - Compilation depends on Neo and Commander Locke (formerly 10)
    10: [9],                  # Twins - Binary comparison AFTER compilation completes (formerly 6)
    11: [10],                 # Oracle - Depends on Twins comparison
    12: [5, 6, 7],            # Link - Depends on Phase B agents (formerly 12)
    13: [5, 6, 7],            # Agent Johnson - Depends on Phase B agents (formerly 13)
    14: [7, 8, 9],            # Cleaner - Depends on early Phase C agents
    15: [7, 8, 9],            # Analyst - Depends on early Phase C agents
    16: [14, 15]              # Agent Brown - Final validation
}


def get_matrix_execution_batches() -> List[List[int]]:
    """
    Calculate Matrix execution batches optimized for parallel processing
    Returns batches where agents can execute in parallel after dependencies are met
    """
    completed = set()
    batches = []
    
    while len(completed) < len(MATRIX_DEPENDENCIES):
        current_batch = []
        
        for agent_id, dependencies in MATRIX_DEPENDENCIES.items():
            if agent_id in completed:
                continue
                
            # Check if all dependencies are completed
            if all(dep in completed for dep in dependencies):
                current_batch.append(agent_id)
        
        if not current_batch:
            remaining = set(MATRIX_DEPENDENCIES.keys()) - completed
            raise ValueError(f"Cannot resolve Matrix dependencies for agents: {remaining}")
        
        batches.append(current_batch)
        completed.update(current_batch)
    
    return batches


def validate_matrix_dependencies() -> bool:
    """Validate Matrix dependency graph"""
    try:
        get_matrix_execution_batches()
        
        # Check that all dependencies reference valid agent IDs
        valid_ids = set(MATRIX_DEPENDENCIES.keys())
        for agent_id, dependencies in MATRIX_DEPENDENCIES.items():
            for dep in dependencies:
                if dep not in valid_ids:
                    raise ValueError(f"Agent {agent_id} depends on non-existent agent {dep}")
        
        return True
    except Exception as e:
        logging.error(f"Matrix dependency validation failed: {e}")
        return False