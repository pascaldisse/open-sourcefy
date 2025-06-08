"""
Base agent class and dependency system for open-sourcefy.
Provides common functionality for all 13 agents with dependency management.
"""

import abc
import logging
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum


class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AgentResult:
    """Result object returned by each agent"""
    agent_id: int
    status: AgentStatus
    data: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(abc.ABC):
    """Base class for all agents in the open-sourcefy system"""
    
    def __init__(self, agent_id: int, name: str, dependencies: List[int] = None):
        self.agent_id = agent_id
        self.name = name
        self.dependencies = dependencies or []
        self.status = AgentStatus.PENDING
        self.logger = logging.getLogger(f"Agent{agent_id}_{name}")
        self.result: Optional[AgentResult] = None
        self.retry_count = 0
        self.max_retries = 3

    @abc.abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute the agent's main functionality"""
        pass

    def pre_execute(self, context: Dict[str, Any]) -> bool:
        """Pre-execution validation and setup"""
        self.logger.info(f"Starting pre-execution for {self.name}")
        
        # Check if all dependencies are satisfied
        for dep_id in self.dependencies:
            if not self._is_dependency_satisfied(dep_id, context):
                self.logger.error(f"Dependency Agent{dep_id} not satisfied")
                return False
        
        return True

    def post_execute(self, result: AgentResult, context: Dict[str, Any]) -> None:
        """Post-execution cleanup and result processing"""
        self.logger.info(f"Completed execution for {self.name} with status: {result.status}")
        self.result = result
        self.status = result.status

    def run(self, context: Dict[str, Any]) -> AgentResult:
        """Main execution wrapper with error handling and timing"""
        start_time = time.time()
        
        try:
            self.status = AgentStatus.RUNNING
            self.logger.info(f"Starting execution of {self.name}")
            
            # Pre-execution checks
            if not self.pre_execute(context):
                result = AgentResult(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    data={},
                    error_message="Pre-execution validation failed"
                )
                self.post_execute(result, context)
                return result
            
            # Execute main functionality
            result = self.execute(context)
            result.execution_time = time.time() - start_time
            
            # Post-execution processing
            self.post_execute(result, context)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            result = AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=error_msg,
                execution_time=execution_time
            )
            
            self.post_execute(result, context)
            return result

    def _is_dependency_satisfied(self, dep_id: int, context: Dict[str, Any]) -> bool:
        """Check if a dependency agent has completed successfully"""
        agent_results = context.get('agent_results', {})
        dep_result = agent_results.get(dep_id)
        
        if dep_result is None:
            return False
            
        return dep_result.status == AgentStatus.COMPLETED

    def can_execute(self, context: Dict[str, Any]) -> bool:
        """Check if this agent can execute based on dependencies"""
        for dep_id in self.dependencies:
            if not self._is_dependency_satisfied(dep_id, context):
                return False
        return True

    def __str__(self) -> str:
        return f"Agent{self.agent_id}_{self.name}"

    def __repr__(self) -> str:
        return f"<Agent(id={self.agent_id}, name='{self.name}', status={self.status.value})>"


# Agent dependency mapping - defines execution order
AGENT_DEPENDENCIES = {
    1: [],                    # BinaryDiscovery - no dependencies
    2: [1],                   # ArchAnalysis - depends on BinaryDiscovery
    3: [1],                   # SmartErrorPatternMatching - depends on BinaryDiscovery
    4: [2],                   # BasicDecompiler - depends on ArchAnalysis
    5: [2],                   # BinaryStructureAnalyzer - depends on ArchAnalysis
    6: [4],                   # OptimizationMatcher - depends on BasicDecompiler
    7: [4, 5],               # AdvancedDecompiler - depends on BasicDecompiler and BinaryStructureAnalyzer
    8: [6],                   # BinaryDiffAnalyzer - depends on OptimizationMatcher
    9: [7],                   # AdvancedAssemblyAnalyzer - depends on AdvancedDecompiler
    10: [8, 9],              # ResourceReconstructor - depends on BinaryDiffAnalyzer and AdvancedAssemblyAnalyzer
    11: [10],                # GlobalReconstructor - depends on ResourceReconstructor
    12: [11],                # CompilationOrchestrator - depends on GlobalReconstructor
    13: [12],                # FinalValidator - depends on CompilationOrchestrator
    14: [7],                 # AdvancedGhidra - depends on AdvancedDecompiler
    15: [1, 2],              # MetadataAnalysis - depends on BinaryDiscovery and ArchAnalysis
    18: [11, 12],            # AdvancedBuildSystems - depends on GlobalReconstructor and CompilationOrchestrator
    19: [12, 18],            # BinaryComparison - depends on CompilationOrchestrator and AdvancedBuildSystems
    20: [18, 19]             # AutoTesting - depends on AdvancedBuildSystems and BinaryComparison
}


def get_execution_batches() -> List[List[int]]:
    """
    Calculate execution batches based on dependencies.
    Returns list of batches where each batch can be executed in parallel.
    """
    completed = set()
    batches = []
    
    while len(completed) < len(AGENT_DEPENDENCIES):
        current_batch = []
        
        for agent_id, dependencies in AGENT_DEPENDENCIES.items():
            if agent_id in completed:
                continue
                
            # Check if all dependencies are completed
            if all(dep in completed for dep in dependencies):
                current_batch.append(agent_id)
        
        if not current_batch:
            # Circular dependency or other issue
            remaining = set(AGENT_DEPENDENCIES.keys()) - completed
            raise ValueError(f"Cannot resolve dependencies for agents: {remaining}")
        
        batches.append(current_batch)
        completed.update(current_batch)
    
    return batches


def validate_dependencies() -> bool:
    """Validate that the dependency graph is valid (no cycles, all deps exist)"""
    try:
        # Check for circular dependencies by attempting to get execution batches
        get_execution_batches()
        
        # Check that all dependencies reference valid agent IDs
        valid_ids = set(AGENT_DEPENDENCIES.keys())
        for agent_id, dependencies in AGENT_DEPENDENCIES.items():
            for dep in dependencies:
                if dep not in valid_ids:
                    raise ValueError(f"Agent {agent_id} depends on non-existent agent {dep}")
        
        return True
    except Exception as e:
        logging.error(f"Dependency validation failed: {e}")
        return False