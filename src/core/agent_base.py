"""
Core Agent Base Classes for Open-Sourcefy Matrix System
Provides base classes and interfaces for all Matrix agents with production-ready architecture.

This module implements the foundation for the Matrix agent system, providing:
- Abstract base classes for different agent types
- Common agent functionality and interfaces
- Standardized result structures and status management
- Configuration-driven behavior and validation
- Comprehensive error handling and logging

Production-ready implementation following SOLID principles and clean code standards.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime

# Import shared components
from .config_manager import get_config_manager
from .exceptions import MatrixAgentError, ValidationError, ConfigurationError


class AgentStatus(Enum):
    """Agent execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AgentType(Enum):
    """Types of Matrix agents for specialized functionality"""
    ANALYSIS = "analysis"           # Data analysis and discovery
    DECOMPILER = "decompiler"      # Code decompilation
    RECONSTRUCTION = "reconstruction"  # Code reconstruction
    VALIDATION = "validation"       # Validation and testing
    ORCHESTRATOR = "orchestrator"   # Pipeline orchestration


@dataclass
class AgentMetadata:
    """Metadata structure for agent information"""
    agent_id: int
    agent_name: str
    agent_type: AgentType
    version: str = "1.0.0"
    description: str = ""
    dependencies: List[int] = field(default_factory=list)
    timeout_seconds: int = 300
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Standardized result structure for all Matrix agents"""
    agent_id: int
    status: AgentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    quality_score: float = 0.0
    confidence_score: float = 0.0


@dataclass
class ExecutionContext:
    """Execution context shared between agents"""
    binary_path: str
    output_dir: str
    global_data: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)
    agent_results: Dict[int, AgentResult] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    pipeline_start_time: Optional[datetime] = None


class AgentBase(ABC):
    """
    Abstract base class for all Matrix agents
    
    Provides common functionality including:
    - Configuration management
    - Logging and metrics
    - Error handling
    - Result standardization
    - Validation framework
    """
    
    def __init__(
        self, 
        agent_id: int, 
        agent_name: str, 
        agent_type: AgentType,
        dependencies: Optional[List[int]] = None
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.dependencies = dependencies or []
        
        # Initialize configuration
        self.config = get_config_manager()
        
        # Setup agent metadata
        self.metadata = AgentMetadata(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            dependencies=self.dependencies,
            timeout_seconds=self.config.get_value(f'agents.agent_{agent_id:02d}.timeout', 300),
            max_retries=self.config.get_value(f'agents.agent_{agent_id:02d}.max_retries', 3)
        )
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Execution tracking
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._execution_metrics: Dict[str, Any] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger_name = f"Agent{self.agent_id:02d}_{self.agent_name}"
        logger = logging.getLogger(logger_name)
        
        # Configure logger if not already configured
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            # Prevent log injection by sanitizing logger name
            if not logger_name.replace('_', '').replace('Agent', '').isalnum():
                raise ValueError(f"Invalid logger name: {logger_name}")
        
        return logger
    
    @abstractmethod
    def execute(self, context: ExecutionContext) -> AgentResult:
        """
        Main execution method for the agent
        
        Args:
            context: Execution context with shared data
            
        Returns:
            AgentResult with execution results and metadata
        """
        pass
    
    @abstractmethod
    def validate_prerequisites(self, context: ExecutionContext) -> bool:
        """
        Validate that agent prerequisites are met
        
        Args:
            context: Execution context to validate
            
        Returns:
            True if prerequisites are met, False otherwise
            
        Raises:
            ValidationError: If critical prerequisites are not met
        """
        pass
    
    def get_dependencies(self) -> List[int]:
        """Get list of agent IDs this agent depends on"""
        return self.dependencies.copy()
    
    def can_execute(self, context: ExecutionContext) -> bool:
        """
        Check if agent can execute in the current context
        
        Args:
            context: Current execution context
            
        Returns:
            True if agent can execute, False otherwise
        """
        try:
            # Check dependencies
            for dep_id in self.dependencies:
                if dep_id not in context.agent_results:
                    self.logger.debug(f"Dependency agent {dep_id} not yet executed")
                    return False
                
                dep_result = context.agent_results[dep_id]
                if dep_result.status != AgentStatus.COMPLETED:
                    self.logger.warning(f"Dependency agent {dep_id} failed: {dep_result.status}")
                    return False
            
            # Validate prerequisites
            return self.validate_prerequisites(context)
            
        except Exception as e:
            self.logger.error(f"Error checking execution readiness: {e}")
            return False
    
    def execute_with_tracking(self, context: ExecutionContext) -> AgentResult:
        """
        Execute agent with comprehensive tracking and error handling
        
        Args:
            context: Execution context
            
        Returns:
            AgentResult with execution results
        """
        self._start_time = datetime.now()
        start_time_float = time.time()
        
        self.logger.info(f"Starting execution for Agent {self.agent_id}: {self.agent_name}")
        
        try:
            # Validate prerequisites
            if not self.can_execute(context):
                return AgentResult(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    error_message="Prerequisites not met or dependencies failed",
                    metadata=self._get_execution_metadata()
                )
            
            # Execute the agent
            result = self.execute(context)
            
            # Ensure result has correct agent_id
            result.agent_id = self.agent_id
            result.start_time = self._start_time
            result.end_time = datetime.now()
            result.execution_time = time.time() - start_time_float
            
            # Add execution metadata
            result.metadata.update(self._get_execution_metadata())
            
            self.logger.info(
                f"Agent {self.agent_id} completed with status: {result.status.value} "
                f"(execution time: {result.execution_time:.2f}s)"
            )
            
            return result
            
        except ValidationError as e:
            self.logger.error(f"Validation error in Agent {self.agent_id}: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error_message=f"Validation error: {str(e)}",
                execution_time=time.time() - start_time_float,
                metadata=self._get_execution_metadata()
            )
            
        except Exception as e:
            self.logger.error(f"Unexpected error in Agent {self.agent_id}: {e}", exc_info=True)
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                error_message=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time_float,
                metadata=self._get_execution_metadata()
            )
        
        finally:
            self._end_time = datetime.now()
    
    def _get_execution_metadata(self) -> Dict[str, Any]:
        """Get execution metadata for the agent"""
        return {
            'agent_name': self.agent_name,
            'agent_type': self.agent_type.value,
            'dependencies': self.dependencies,
            'timeout_seconds': self.metadata.timeout_seconds,
            'max_retries': self.metadata.max_retries,
            'start_time': self._start_time.isoformat() if self._start_time else None,
            'end_time': self._end_time.isoformat() if self._end_time else None
        }
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information for the agent."""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_type': self.agent_type.value,
            'dependencies': self.dependencies.copy(),  # Return copy to prevent mutation
            'is_running': self._start_time is not None and self._end_time is None,
            'last_execution': self._end_time.isoformat() if self._end_time else None
        }


class AnalysisAgent(AgentBase):
    """
    Base class for analysis agents that perform binary analysis and data extraction
    
    Analysis agents are responsible for:
    - Binary format detection and parsing
    - Metadata extraction
    - Pattern recognition
    - Statistical analysis
    """
    
    def __init__(self, agent_id: int, agent_name: str, dependencies: Optional[List[int]] = None):
        super().__init__(agent_id, agent_name, AgentType.ANALYSIS, dependencies)
        
        # Analysis-specific configuration
        self.analysis_depth = self.config.get_value(f'agents.agent_{agent_id:02d}.analysis_depth', 'standard')
        self.quality_threshold = self.config.get_value(f'agents.agent_{agent_id:02d}.quality_threshold', 0.75)
    
    @abstractmethod
    def analyze(self, data: Any, context: ExecutionContext) -> Dict[str, Any]:
        """
        Perform analysis on the given data
        
        Args:
            data: Data to analyze
            context: Execution context
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def execute(self, context: ExecutionContext) -> AgentResult:
        """Execute analysis agent"""
        try:
            # Get binary path from context
            binary_path = Path(context.binary_path)
            if not binary_path.exists():
                raise ValidationError(f"Binary file not found: {binary_path}")
            
            # Perform analysis
            analysis_results = self.analyze(binary_path, context)
            
            # Calculate quality score
            quality_score = self._calculate_analysis_quality(analysis_results)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=analysis_results,
                quality_score=quality_score,
                confidence_score=analysis_results.get('confidence', 0.0)
            )
            
        except Exception as e:
            raise MatrixAgentError(f"Analysis failed in {self.agent_name}: {e}") from e
    
    def _calculate_analysis_quality(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for analysis results"""
        if not results:
            return 0.0
        
        # Basic quality scoring based on data completeness
        score_components = []
        
        # Check for key analysis components
        if 'format_detection' in results:
            score_components.append(0.3)
        if 'metadata' in results:
            score_components.append(0.2)
        if 'patterns' in results:
            score_components.append(0.2)
        if 'statistics' in results:
            score_components.append(0.2)
        if 'confidence' in results and results['confidence'] > 0.5:
            score_components.append(0.1)
        
        return sum(score_components)


class DecompilerAgent(AgentBase):
    """
    Base class for decompilation agents that convert binary code to source code
    
    Decompiler agents are responsible for:
    - Disassembly and decompilation
    - Function identification
    - Control flow analysis
    - Code reconstruction
    """
    
    def __init__(self, agent_id: int, agent_name: str, dependencies: Optional[List[int]] = None):
        super().__init__(agent_id, agent_name, AgentType.DECOMPILER, dependencies)
        
        # Decompiler-specific configuration
        self.decompilation_mode = self.config.get_value(f'agents.agent_{agent_id:02d}.mode', 'standard')
        self.function_threshold = self.config.get_value(f'agents.agent_{agent_id:02d}.function_threshold', 10)
    
    @abstractmethod
    def decompile(self, binary_data: Any, context: ExecutionContext) -> Dict[str, Any]:
        """
        Perform decompilation on binary data
        
        Args:
            binary_data: Binary data to decompile
            context: Execution context
            
        Returns:
            Dictionary containing decompilation results
        """
        pass
    
    def execute(self, context: ExecutionContext) -> AgentResult:
        """Execute decompilation agent"""
        try:
            # Get binary and analysis data
            binary_path = Path(context.binary_path)
            
            # Perform decompilation
            decompilation_results = self.decompile(binary_path, context)
            
            # Calculate quality score
            quality_score = self._calculate_decompilation_quality(decompilation_results)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=decompilation_results,
                quality_score=quality_score,
                confidence_score=decompilation_results.get('confidence', 0.0)
            )
            
        except Exception as e:
            raise MatrixAgentError(f"Decompilation failed in {self.agent_name}: {e}") from e
    
    def _calculate_decompilation_quality(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for decompilation results"""
        if not results:
            return 0.0
        
        score_components = []
        
        # Check for key decompilation components
        if 'functions' in results and len(results['functions']) > 0:
            score_components.append(0.4)
        if 'source_code' in results and results['source_code']:
            score_components.append(0.3)
        if 'control_flow' in results:
            score_components.append(0.2)
        if 'confidence' in results and results['confidence'] > 0.6:
            score_components.append(0.1)
        
        return sum(score_components)


class ReconstructionAgent(AgentBase):
    """
    Base class for reconstruction agents that rebuild and optimize source code
    
    Reconstruction agents are responsible for:
    - Code reconstruction and organization
    - Resource handling
    - Build system generation
    - Quality improvement
    """
    
    def __init__(self, agent_id: int, agent_name: str, dependencies: Optional[List[int]] = None):
        super().__init__(agent_id, agent_name, AgentType.RECONSTRUCTION, dependencies)
        
        # Reconstruction-specific configuration
        self.reconstruction_mode = self.config.get_value(f'agents.agent_{agent_id:02d}.mode', 'comprehensive')
        self.output_format = self.config.get_value(f'agents.agent_{agent_id:02d}.output_format', 'c')
    
    @abstractmethod
    def reconstruct(self, decompiled_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """
        Perform reconstruction on decompiled data
        
        Args:
            decompiled_data: Decompiled data to reconstruct
            context: Execution context
            
        Returns:
            Dictionary containing reconstruction results
        """
        pass
    
    def execute(self, context: ExecutionContext) -> AgentResult:
        """Execute reconstruction agent"""
        try:
            # Collect decompiled data from previous agents
            decompiled_data = self._collect_decompiled_data(context)
            
            # Perform reconstruction
            reconstruction_results = self.reconstruct(decompiled_data, context)
            
            # Calculate quality score
            quality_score = self._calculate_reconstruction_quality(reconstruction_results)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=reconstruction_results,
                quality_score=quality_score,
                confidence_score=reconstruction_results.get('confidence', 0.0)
            )
            
        except Exception as e:
            raise MatrixAgentError(f"Reconstruction failed in {self.agent_name}: {e}") from e
    
    def _collect_decompiled_data(self, context: ExecutionContext) -> Dict[str, Any]:
        """Collect decompiled data from previous agents"""
        decompiled_data = {}
        
        for agent_id, result in context.agent_results.items():
            if result.status == AgentStatus.COMPLETED and 'source_code' in result.data:
                decompiled_data[agent_id] = result.data
        
        return decompiled_data
    
    def _calculate_reconstruction_quality(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for reconstruction results"""
        if not results:
            return 0.0
        
        score_components = []
        
        # Check for key reconstruction components
        if 'reconstructed_code' in results:
            score_components.append(0.4)
        if 'build_system' in results:
            score_components.append(0.2)
        if 'resources' in results:
            score_components.append(0.2)
        if 'quality_improvements' in results:
            score_components.append(0.2)
        
        return sum(score_components)


class ValidationAgent(AgentBase):
    """
    Base class for validation agents that test and verify results
    
    Validation agents are responsible for:
    - Compilation testing
    - Functionality verification
    - Quality assessment
    - Final validation
    """
    
    def __init__(self, agent_id: int, agent_name: str, dependencies: Optional[List[int]] = None):
        super().__init__(agent_id, agent_name, AgentType.VALIDATION, dependencies)
        
        # Validation-specific configuration
        self.validation_mode = self.config.get_value(f'agents.agent_{agent_id:02d}.mode', 'comprehensive')
        self.pass_threshold = self.config.get_value(f'agents.agent_{agent_id:02d}.pass_threshold', 0.8)
    
    @abstractmethod
    def validate(self, reconstruction_data: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        """
        Perform validation on reconstruction data
        
        Args:
            reconstruction_data: Reconstruction data to validate
            context: Execution context
            
        Returns:
            Dictionary containing validation results
        """
        pass
    
    def execute(self, context: ExecutionContext) -> AgentResult:
        """Execute validation agent"""
        try:
            # Collect reconstruction data from previous agents
            reconstruction_data = self._collect_reconstruction_data(context)
            
            # Perform validation
            validation_results = self.validate(reconstruction_data, context)
            
            # Calculate quality score
            quality_score = self._calculate_validation_quality(validation_results)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=validation_results,
                quality_score=quality_score,
                confidence_score=validation_results.get('confidence', 0.0)
            )
            
        except Exception as e:
            raise MatrixAgentError(f"Validation failed in {self.agent_name}: {e}") from e
    
    def _collect_reconstruction_data(self, context: ExecutionContext) -> Dict[str, Any]:
        """Collect reconstruction data from previous agents"""
        reconstruction_data = {}
        
        for agent_id, result in context.agent_results.items():
            if result.status == AgentStatus.COMPLETED:
                reconstruction_data[agent_id] = result.data
        
        return reconstruction_data
    
    def _calculate_validation_quality(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for validation results"""
        if not results:
            return 0.0
        
        score_components = []
        
        # Check for key validation components
        if 'compilation_test' in results:
            score_components.append(0.3)
        if 'functionality_test' in results:
            score_components.append(0.3)
        if 'quality_assessment' in results:
            score_components.append(0.2)
        if 'validation_score' in results and results['validation_score'] > self.pass_threshold:
            score_components.append(0.2)
        
        return sum(score_components)


# Utility functions for agent management
def create_agent_result(
    agent_id: int,
    status: AgentStatus,
    data: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    **kwargs
) -> AgentResult:
    """
    Utility function to create standardized AgentResult objects
    
    Args:
        agent_id: ID of the agent
        status: Execution status
        data: Result data dictionary
        error_message: Error message if failed
        **kwargs: Additional metadata
        
    Returns:
        AgentResult object
    """
    return AgentResult(
        agent_id=agent_id,
        status=status,
        data=data or {},
        error_message=error_message,
        metadata=kwargs
    )


def validate_agent_dependencies(agents: List[AgentBase]) -> bool:
    """
    Validate that agent dependencies are properly configured
    
    Args:
        agents: List of agents to validate
        
    Returns:
        True if dependencies are valid, False otherwise
    """
    agent_ids = {agent.agent_id for agent in agents}
    
    for agent in agents:
        for dep_id in agent.dependencies:
            if dep_id not in agent_ids:
                return False
    
    return True


def sort_agents_by_dependencies(agents: List[AgentBase]) -> List[AgentBase]:
    """
    Sort agents by their dependencies for execution order
    
    Args:
        agents: List of agents to sort
        
    Returns:
        List of agents sorted by dependencies
    """
    # Simple topological sort
    sorted_agents = []
    remaining_agents = agents.copy()
    
    while remaining_agents:
        # Find agents with no unmet dependencies
        ready_agents = []
        for agent in remaining_agents:
            dependencies_met = all(
                dep_id in [a.agent_id for a in sorted_agents]
                for dep_id in agent.dependencies
            )
            if dependencies_met:
                ready_agents.append(agent)
        
        if not ready_agents:
            # Circular dependency or missing dependency
            break
        
        # Add ready agents to sorted list
        for agent in ready_agents:
            sorted_agents.append(agent)
            remaining_agents.remove(agent)
    
    return sorted_agents