"""
Matrix Execution Context for Agent Pipeline
Complete context management and sharing system for Matrix agents
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from .matrix_agents import AgentResult, AgentStatus
from .shared_components import SharedMemory, create_shared_memory, setup_output_structure
from .shared_utils import get_performance_monitor, get_error_handler


@dataclass
class MatrixExecutionContext:
    """
    Complete execution context for Matrix agent pipeline
    Provides global state sharing, agent results, and output management
    """
    
    # Core context data
    binary_path: Path
    output_paths: Dict[str, Path] = field(default_factory=dict)
    
    # Shared state management
    shared_memory: SharedMemory = field(default_factory=create_shared_memory)
    agent_results: Dict[int, AgentResult] = field(default_factory=dict)
    global_data: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline execution state
    execution_id: str = field(default_factory=lambda: f"matrix_{int(time.time())}")
    start_time: float = field(default_factory=time.time)
    pipeline_status: str = "initializing"
    current_batch: int = 0
    completed_agents: List[int] = field(default_factory=list)
    failed_agents: List[int] = field(default_factory=list)
    
    # Configuration and settings
    config: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "master_first_parallel"
    resource_profile: str = "standard"
    enable_ai_enhancement: bool = True
    
    # Monitoring and metrics
    performance_monitor: Any = field(default_factory=get_performance_monitor)
    error_handler: Any = field(default_factory=get_error_handler)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Thread safety
    _lock: Lock = field(default_factory=Lock)
    
    def __post_init__(self):
        """Initialize context after creation"""
        if not self.output_paths:
            # Create output structure using config manager for new path format
            from .config_manager import get_config_manager
            binary_name = self.binary_path.stem
            config_manager = get_config_manager()
            base_output = config_manager.get_output_path(binary_name)
            self.output_paths = setup_output_structure(base_output)
        
        # Initialize global data
        self.global_data.update({
            'execution_id': self.execution_id,
            'binary_path': str(self.binary_path),
            'binary_name': self.binary_path.name,
            'binary_size': self.binary_path.stat().st_size if self.binary_path.exists() else 0,
            'pipeline_start_time': self.start_time
        })
        
        # Initialize metrics
        self.metrics.update({
            'agents_completed': 0,
            'agents_failed': 0,
            'total_execution_time': 0.0,
            'quality_scores': {},
            'performance_stats': {}
        })
    
    def add_agent_result(self, result: AgentResult) -> None:
        """Thread-safe addition of agent result"""
        with self._lock:
            self.agent_results[result.agent_id] = result
            
            if result.status == AgentStatus.SUCCESS:
                if result.agent_id not in self.completed_agents:
                    self.completed_agents.append(result.agent_id)
                    self.metrics['agents_completed'] += 1
                
                # Update shared memory with agent data
                self._update_shared_memory(result)
                
            elif result.status == AgentStatus.FAILED:
                if result.agent_id not in self.failed_agents:
                    self.failed_agents.append(result.agent_id)
                    self.metrics['agents_failed'] += 1
            
            # Update quality scores if available
            if 'quality_score' in result.data:
                self.metrics['quality_scores'][f'agent_{result.agent_id:02d}'] = result.data['quality_score']
    
    def _update_shared_memory(self, result: AgentResult) -> None:
        """Update shared memory with agent result data"""
        agent_key = f'agent_{result.agent_id:02d}'
        
        # Update appropriate shared memory section based on agent type
        if result.agent_id == 1:  # Sentinel - binary metadata
            self.shared_memory.binary_metadata.update(result.data)
        elif result.agent_id in [2, 3, 4, 6, 7, 8]:  # Analysis agents
            self.shared_memory.analysis_results[agent_key] = result.data
        elif result.agent_id in [3, 5, 7]:  # Decompilation agents
            self.shared_memory.decompilation_data[agent_key] = result.data
        elif result.agent_id in [9, 10, 11, 12]:  # Reconstruction agents
            self.shared_memory.reconstruction_info[agent_key] = result.data
        elif result.agent_id in [13, 14, 15, 16]:  # Validation agents
            self.shared_memory.validation_status[agent_key] = result.data
    
    def get_agent_result(self, agent_id: int) -> Optional[AgentResult]:
        """Thread-safe retrieval of agent result"""
        with self._lock:
            return self.agent_results.get(agent_id)
    
    def is_agent_completed(self, agent_id: int) -> bool:
        """Check if agent completed successfully"""
        result = self.get_agent_result(agent_id)
        return result and result.status == AgentStatus.SUCCESS
    
    def are_dependencies_satisfied(self, dependencies: List[int]) -> bool:
        """Check if all dependencies are satisfied"""
        return all(self.is_agent_completed(dep_id) for dep_id in dependencies)
    
    def get_dependency_data(self, agent_id: int) -> Dict[str, Any]:
        """Get dependency data for an agent"""
        result = self.get_agent_result(agent_id)
        return result.data if result else {}
    
    def update_pipeline_status(self, status: str, batch_number: int = None) -> None:
        """Update pipeline execution status"""
        with self._lock:
            self.pipeline_status = status
            if batch_number is not None:
                self.current_batch = batch_number
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        with self._lock:
            current_time = time.time()
            total_time = current_time - self.start_time
            
            return {
                'execution_id': self.execution_id,
                'binary_path': str(self.binary_path),
                'pipeline_status': self.pipeline_status,
                'current_batch': self.current_batch,
                'total_execution_time': total_time,
                'agents_completed': len(self.completed_agents),
                'agents_failed': len(self.failed_agents),
                'completed_agents': sorted(self.completed_agents),
                'failed_agents': sorted(self.failed_agents),
                'success_rate': len(self.completed_agents) / max(1, len(self.completed_agents) + len(self.failed_agents)),
                'quality_scores': self.metrics['quality_scores'].copy(),
                'output_paths': {k: str(v) for k, v in self.output_paths.items()}
            }
    
    def validate_context(self) -> Dict[str, Any]:
        """Validate context integrity and completeness"""
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check binary path
        if not self.binary_path.exists():
            validation_results['valid'] = False
            validation_results['issues'].append(f"Binary file not found: {self.binary_path}")
        
        # Check output paths
        for path_type, path in self.output_paths.items():
            if not path.exists():
                validation_results['warnings'].append(f"Output path does not exist: {path_type} -> {path}")
        
        # Check shared memory structure
        if not isinstance(self.shared_memory, SharedMemory):
            validation_results['valid'] = False
            validation_results['issues'].append("Invalid shared memory structure")
        
        # Check for circular dependencies in agent results
        agent_ids = set(self.agent_results.keys())
        for agent_id in agent_ids:
            if agent_id < 1 or agent_id > 16:
                validation_results['warnings'].append(f"Unexpected agent ID: {agent_id}")
        
        return validation_results
    
    def export_context(self, export_path: Path) -> None:
        """Export context state to file for debugging"""
        export_data = {
            'execution_summary': self.get_execution_summary(),
            'shared_memory': {
                'binary_metadata': self.shared_memory.binary_metadata,
                'analysis_results': self.shared_memory.analysis_results,
                'decompilation_data': self.shared_memory.decompilation_data,
                'reconstruction_info': self.shared_memory.reconstruction_info,
                'validation_status': self.shared_memory.validation_status
            },
            'agent_results': {
                agent_id: {
                    'agent_id': result.agent_id,
                    'agent_name': result.agent_name,
                    'matrix_character': result.matrix_character,
                    'status': result.status.value,
                    'execution_time': result.execution_time,
                    'error_message': result.error_message,
                    'data_keys': list(result.data.keys()) if result.data else []
                }
                for agent_id, result in self.agent_results.items()
            },
            'global_data': self.global_data,
            'validation': self.validate_context()
        }
        
        import json
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def get_context_for_agent(self, agent_id: int) -> Dict[str, Any]:
        """Get context dictionary for agent execution"""
        return {
            'binary_path': self.binary_path,
            'output_paths': self.output_paths,
            'shared_memory': self.shared_memory,
            'agent_results': self.agent_results.copy(),
            'global_data': self.global_data.copy(),
            'execution_id': self.execution_id,
            'pipeline_status': self.pipeline_status,
            'config': self.config,
            'enable_ai_enhancement': self.enable_ai_enhancement,
            'performance_monitor': self.performance_monitor,
            'error_handler': self.error_handler
        }
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files and directories"""
        temp_path = self.output_paths.get('temp')
        if temp_path and temp_path.exists():
            import shutil
            try:
                shutil.rmtree(temp_path)
                self.output_paths['temp'].mkdir(exist_ok=True)  # Recreate empty temp dir
            except Exception as e:
                # Log error but don't fail - cleanup is best effort
                pass
    
    def __str__(self) -> str:
        return f"MatrixExecutionContext(id={self.execution_id}, binary={self.binary_path.name}, status={self.pipeline_status})"
    
    def __repr__(self) -> str:
        return (f"<MatrixExecutionContext(execution_id='{self.execution_id}', "
                f"binary='{self.binary_path.name}', completed={len(self.completed_agents)}, "
                f"failed={len(self.failed_agents)})>")


def create_matrix_execution_context(binary_path: Path, 
                                   config: Optional[Dict[str, Any]] = None,
                                   output_base: Optional[Path] = None) -> MatrixExecutionContext:
    """
    Factory function to create properly initialized MatrixExecutionContext
    
    Args:
        binary_path: Path to the binary file to analyze
        config: Optional configuration dictionary
        output_base: Optional base output directory
    
    Returns:
        Initialized MatrixExecutionContext
    """
    context = MatrixExecutionContext(binary_path=binary_path)
    
    if config:
        context.config.update(config)
    
    if output_base:
        context.output_paths = setup_output_structure(output_base)
    
    # Initialize performance monitoring
    context.performance_monitor.start_operation(
        f"matrix_pipeline_{context.execution_id}",
        metadata={'binary_path': str(binary_path)}
    )
    
    return context