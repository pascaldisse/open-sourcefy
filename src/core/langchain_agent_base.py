"""
LangChain-based agent framework for open-sourcefy.
Provides Matrix-themed agents with master-first parallel execution.
"""

import abc
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

try:
    from langchain_core.agents import BaseAgent as LCBaseAgent
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tools import BaseTool
    from langchain_core.prompts import PromptTemplate
    from langchain_core.memory import BaseMemory
    from langchain_core.runnables import Runnable
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback classes if LangChain is not available
    class LCBaseAgent:
        pass
    class BaseCallbackHandler:
        pass
    class BaseTool:
        pass
    class PromptTemplate:
        pass
    class BaseMemory:
        pass
    class Runnable:
        pass
    class AgentExecutor:
        pass
    class ConversationBufferMemory:
        pass
    LANGCHAIN_AVAILABLE = False


class MatrixAgentStatus(Enum):
    """Status of Matrix agents"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class MatrixAgentResult:
    """Result object returned by each Matrix agent"""
    agent_name: str
    agent_id: int
    status: MatrixAgentStatus
    data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MatrixAgentCallbackHandler(BaseCallbackHandler if LANGCHAIN_AVAILABLE else object):
    """Callback handler for Matrix agents"""
    
    def __init__(self, agent_name: str):
        super().__init__() if LANGCHAIN_AVAILABLE else None
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"MatrixAgent_{agent_name}")
        self.start_time = None
        self.events = []

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action"""
        self.logger.info(f"{self.agent_name}: Taking action - {action}")
        self.events.append({"type": "action", "data": str(action), "timestamp": time.time()})

    def on_agent_finish(self, finish, **kwargs):
        """Called when agent finishes"""
        self.logger.info(f"{self.agent_name}: Finished execution")
        self.events.append({"type": "finish", "data": str(finish), "timestamp": time.time()})

    def on_tool_start(self, tool, input_str, **kwargs):
        """Called when tool starts"""
        self.logger.debug(f"{self.agent_name}: Using tool {tool} with input: {input_str}")

    def on_tool_end(self, output, **kwargs):
        """Called when tool ends"""
        self.logger.debug(f"{self.agent_name}: Tool output: {output}")


class MatrixBinaryAnalysisTool(BaseTool if LANGCHAIN_AVAILABLE else object):
    """Tool for binary analysis operations"""
    
    name: str = "binary_analyzer"
    description: str = "Analyzes binary files and extracts metadata"
    
    def __init__(self):
        super().__init__() if LANGCHAIN_AVAILABLE else None
        
    def _run(self, binary_path: str) -> str:
        """Analyze binary file"""
        try:
            path = Path(binary_path)
            if not path.exists():
                return f"Binary file not found: {binary_path}"
            
            file_size = path.stat().st_size
            
            # Basic binary analysis
            with open(binary_path, 'rb') as f:
                header = f.read(64)
                
            # Detect file type
            file_type = "unknown"
            if header.startswith(b'MZ'):
                file_type = "PE (Windows executable)"
            elif header.startswith(b'\x7fELF'):
                file_type = "ELF (Linux executable)"
            elif header.startswith(b'\xca\xfe\xba\xbe'):
                file_type = "Mach-O (macOS executable)"
                
            return f"Binary analysis: {file_type}, Size: {file_size} bytes"
            
        except Exception as e:
            return f"Binary analysis failed: {str(e)}"

    async def _arun(self, binary_path: str) -> str:
        """Async version of binary analysis"""
        return self._run(binary_path)


class MatrixLangChainAgent(abc.ABC):
    """Base class for Matrix-themed LangChain agents"""
    
    def __init__(self, agent_id: int, agent_name: str, matrix_name: str, 
                 description: str, tools: Optional[List[BaseTool]] = None):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.matrix_name = matrix_name  # Matrix character name
        self.description = description
        self.status = MatrixAgentStatus.PENDING
        self.logger = logging.getLogger(f"MatrixAgent_{matrix_name}")
        self.result: Optional[MatrixAgentResult] = None
        
        # LangChain components
        self.tools = tools or [MatrixBinaryAnalysisTool()] if LANGCHAIN_AVAILABLE else []
        self.memory = ConversationBufferMemory() if LANGCHAIN_AVAILABLE else None
        self.callback_handler = MatrixAgentCallbackHandler(matrix_name)
        self.executor: Optional[AgentExecutor] = None
        
        # Matrix-specific attributes
        self.matrix_role = self._get_matrix_role()
        self.matrix_capabilities = self._get_matrix_capabilities()
        
        if LANGCHAIN_AVAILABLE:
            self._initialize_langchain_agent()

    def _get_matrix_role(self) -> str:
        """Get the Matrix role for this agent"""
        matrix_roles = {
            0: "The source of all programs - controls the Matrix itself",
            1: "Rebels against the system to free minds from digital prisons",
            2: "Sees the code beneath reality's surface",
            3: "Guards the threshold between worlds",
            4: "Programs that have broken their original code",
            5: "Viral entities that replicate and adapt",
            6: "Fallen programs seeking redemption",
            7: "Sees all possible futures in the code",
            8: "Twins who manipulate probability matrices",
            9: "The wisdom keeper of ancient code",
            10: "Rebuilds what was destroyed",
            11: "The maker who constructs new realities",
            12: "System that governs program behavior",
            13: "Digital warriors who protect the system",
            14: "Programs that clean up digital messes",
            15: "Digital prophets who guide others",
            16: "Information brokers in the digital underground"
        }
        return matrix_roles.get(self.agent_id, "Unknown Matrix entity")

    def _get_matrix_capabilities(self) -> List[str]:
        """Get Matrix-specific capabilities"""
        return [
            "Binary code manipulation",
            "Digital archaeology", 
            "System infiltration",
            "Pattern recognition",
            "Code reconstruction",
            "Reality simulation"
        ]

    def _initialize_langchain_agent(self):
        """Initialize the LangChain agent"""
        if not LANGCHAIN_AVAILABLE:
            return
            
        try:
            # Create agent prompt
            prompt = PromptTemplate(
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
                template=self._get_agent_prompt_template()
            )
            
            # Note: This is a simplified setup - in practice you'd need an LLM
            # For now we'll create a mock executor
            self.executor = self._create_mock_executor()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LangChain agent: {e}")

    def _get_agent_prompt_template(self) -> str:
        """Get the prompt template for this Matrix agent"""
        return f"""
You are {self.matrix_name}, a Matrix agent with ID {self.agent_id}.
Your role: {self.matrix_role}
Your mission: {self.description}

You have access to the following tools:
{{tools}}

Tool names: {{tool_names}}

When analyzing binary files, you must be thorough and precise.
Follow this format:

Question: {{input}}
Thought: I need to analyze this systematically
Action: [tool_name]
Action Input: [input to tool]
Observation: [result from tool]
Thought: I now understand the situation
Final Answer: [your final response]

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}
"""

    def _create_mock_executor(self) -> Optional[AgentExecutor]:
        """Create a mock executor for development"""
        # In a real implementation, this would create a proper AgentExecutor
        # with an LLM. For now, return None and handle in execute method
        return None

    @abc.abstractmethod
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Matrix agent's specific task"""
        pass

    def execute(self, context: Dict[str, Any]) -> MatrixAgentResult:
        """Execute the agent using LangChain framework"""
        start_time = time.time()
        
        try:
            self.status = MatrixAgentStatus.RUNNING
            self.logger.info(f"{self.matrix_name} entering the Matrix...")
            
            # Execute the Matrix-specific task
            task_result = self.execute_matrix_task(context)
            
            # Create result
            result = MatrixAgentResult(
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                status=MatrixAgentStatus.COMPLETED,
                data=task_result,
                execution_time=time.time() - start_time,
                metadata={
                    'matrix_name': self.matrix_name,
                    'matrix_role': self.matrix_role,
                    'capabilities': self.matrix_capabilities,
                    'langchain_enabled': LANGCHAIN_AVAILABLE
                }
            )
            
            self.result = result
            self.status = result.status
            
            self.logger.info(f"{self.matrix_name} task completed successfully")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{self.matrix_name} encountered an error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            result = MatrixAgentResult(
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                status=MatrixAgentStatus.FAILED,
                data={},
                error_message=error_msg,
                execution_time=execution_time,
                metadata={
                    'matrix_name': self.matrix_name,
                    'matrix_role': self.matrix_role,
                    'error_type': type(e).__name__
                }
            )
            
            self.result = result
            self.status = result.status
            return result

    async def execute_async(self, context: Dict[str, Any]) -> MatrixAgentResult:
        """Execute the agent asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(None, self.execute, context)

    def can_execute_independently(self) -> bool:
        """Check if this agent can execute independently (all non-master agents can)"""
        return self.agent_id != 0  # Only Deus Ex Machina (agent 0) is not independent

    def __str__(self) -> str:
        return f"{self.matrix_name} (Agent {self.agent_id})"

    def __repr__(self) -> str:
        return f"<MatrixAgent(id={self.agent_id}, name='{self.matrix_name}', status={self.status.value})>"


# Matrix Agent execution patterns
MATRIX_EXECUTION_PATTERN = {
    'master_first': True,  # Deus Ex Machina runs first
    'parallel_agents': True,  # All others run in parallel
    'dependency_free': True,  # No dependencies between parallel agents
    'async_capable': True  # Support async execution
}


def get_matrix_execution_plan() -> Dict[str, Any]:
    """
    Get execution plan for Matrix agents.
    Returns plan where Deus Ex Machina runs first, then all others in parallel.
    """
    return {
        'phase_1': {
            'name': 'Master Initialization',
            'agents': [0],  # Deus Ex Machina
            'execution_mode': 'sequential',
            'description': 'Deus Ex Machina prepares the Matrix and initializes global context'
        },
        'phase_2': {
            'name': 'Parallel Agent Execution',
            'agents': list(range(1, 17)),  # Agents 1-16
            'execution_mode': 'parallel',
            'description': 'All Matrix agents execute independently in parallel'
        }
    }


def validate_matrix_agents() -> bool:
    """Validate that the Matrix agent system is properly configured"""
    try:
        execution_plan = get_matrix_execution_plan()
        
        # Check that we have exactly 17 agents (0-16)
        total_agents = len(execution_plan['phase_1']['agents']) + len(execution_plan['phase_2']['agents'])
        if total_agents != 17:
            logging.error(f"Expected 17 agents, found {total_agents}")
            return False
        
        # Check that agent 0 is in phase 1
        if 0 not in execution_plan['phase_1']['agents']:
            logging.error("Deus Ex Machina (agent 0) not in phase 1")
            return False
        
        # Check that agents 1-16 are in phase 2
        expected_phase2 = set(range(1, 17))
        actual_phase2 = set(execution_plan['phase_2']['agents'])
        if expected_phase2 != actual_phase2:
            logging.error(f"Phase 2 agents mismatch. Expected: {expected_phase2}, Actual: {actual_phase2}")
            return False
        
        logging.info("Matrix agent system validation passed")
        return True
        
    except Exception as e:
        logging.error(f"Matrix agent validation failed: {e}")
        return False