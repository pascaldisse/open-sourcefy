"""
Matrix Pipeline Orchestrator
Orchestrates the Matrix Phase 4 pipeline with master-first execution model
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .shared_utils import LoggingUtils
from .config_manager import get_config_manager


class PipelineMode(Enum):
    """Pipeline execution modes"""
    FULL_PIPELINE = "full_pipeline"
    DECOMPILE_ONLY = "decompile_only" 
    ANALYZE_ONLY = "analyze_only"
    COMPILE_ONLY = "compile_only"
    VALIDATE_ONLY = "validate_only"
    CUSTOM_AGENTS = "custom_agents"


class MatrixExecutionMode(Enum):
    """Matrix execution modes"""
    MASTER_FIRST_PARALLEL = "master_first_parallel"
    SEQUENTIAL = "sequential"
    FULL_PARALLEL = "full_parallel"
    HYBRID = "hybrid"


@dataclass
class MatrixResourceLimits:
    """Resource limits for Matrix pipeline execution"""
    max_memory: str = "4G"
    max_cpu_percent: int = 80
    max_disk_io: str = "100MB/s"
    max_parallel_agents: int = 16
    timeout_agent: int = 300
    timeout_master: int = 600
    
    @classmethod
    def STANDARD(cls):
        """Standard resource profile"""
        return cls()
    
    @classmethod
    def HIGH_PERFORMANCE(cls):
        """High performance resource profile"""
        return cls(
            max_memory="8G",
            max_cpu_percent=95,
            max_disk_io="1GB/s",
            max_parallel_agents=32,
            timeout_agent=600,
            timeout_master=1200
        )
    
    @classmethod
    def CONSERVATIVE(cls):
        """Conservative resource profile"""
        return cls(
            max_memory="2G",
            max_cpu_percent=50,
            max_disk_io="50MB/s",
            max_parallel_agents=8,
            timeout_agent=180,
            timeout_master=300
        )


@dataclass
class PipelineConfig:
    """Configuration for Matrix pipeline orchestration"""
    pipeline_mode: PipelineMode = PipelineMode.FULL_PIPELINE
    execution_mode: MatrixExecutionMode = MatrixExecutionMode.MASTER_FIRST_PARALLEL
    resource_limits: MatrixResourceLimits = field(default_factory=MatrixResourceLimits.STANDARD)
    
    # Agent selection
    custom_agents: Optional[List[int]] = None
    exclude_agents: Optional[List[int]] = None
    
    # Execution parameters
    max_retries: int = 3
    continue_on_failure: bool = True
    validate_results: bool = True
    
    # Behavioral flags
    verbose: bool = False
    debug: bool = False
    save_reports: bool = True
    dry_run: bool = False
    profile_performance: bool = False


@dataclass
class MatrixPipelineResult:
    """Result from Matrix pipeline execution"""
    success: bool = False
    execution_time: float = 0.0
    total_agents: int = 0
    successful_agents: int = 0
    failed_agents: int = 0
    agent_results: Dict[int, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    report_path: Optional[str] = None


class MatrixPipelineOrchestrator:
    """
    Matrix Pipeline Orchestrator
    Implements master-first execution with parallel agent coordination
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.config_manager = get_config_manager()
        self.logger = LoggingUtils.setup_agent_logging(0, "matrix_orchestrator")
        
        # Pipeline state
        self.master_executed = False
        self.global_context = {}
        self.agent_results = {}
        self.execution_start_time = None
        
        # Agent selection
        self.selected_agents = self._determine_agent_selection()
        
    def _determine_agent_selection(self) -> List[int]:
        """Determine which agents to execute based on pipeline mode"""
        if self.config.custom_agents:
            return self.config.custom_agents
            
        mode_agents = {
            PipelineMode.FULL_PIPELINE: list(range(1, 21)),  # All agents 1-20
            PipelineMode.DECOMPILE_ONLY: [1, 2, 5, 7, 14],
            PipelineMode.ANALYZE_ONLY: [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15],
            PipelineMode.COMPILE_ONLY: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18],
            PipelineMode.VALIDATE_ONLY: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19],
            PipelineMode.CUSTOM_AGENTS: self.config.custom_agents or []
        }
        
        agents = mode_agents.get(self.config.pipeline_mode, [])
        
        # Apply exclusions
        if self.config.exclude_agents:
            agents = [a for a in agents if a not in self.config.exclude_agents]
            
        return sorted(agents)
    
    async def execute_pipeline(self, binary_path: str, output_dir: str) -> MatrixPipelineResult:
        """Execute the complete Matrix pipeline"""
        self.execution_start_time = time.time()
        
        try:
            self.logger.info("🔮 Initializing Matrix Pipeline Orchestration...")
            self.logger.info(f"Pipeline Mode: {self.config.pipeline_mode.value}")
            self.logger.info(f"Execution Mode: {self.config.execution_mode.value}")
            self.logger.info(f"Selected Agents: {self.selected_agents}")
            
            # Step 1: Execute Deus Ex Machina (Master Agent)
            if not await self._execute_master_agent(binary_path, output_dir):
                return MatrixPipelineResult(
                    success=False,
                    error_messages=["Master agent (Deus Ex Machina) execution failed"],
                    execution_time=time.time() - self.execution_start_time
                )
            
            # Step 2: Execute selected agents in parallel
            agent_results = await self._execute_parallel_agents()
            
            # Step 3: Generate final results
            result = self._generate_pipeline_result(agent_results)
            
            if self.config.save_reports:
                await self._save_execution_report(result, output_dir)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return MatrixPipelineResult(
                success=False,
                error_messages=[str(e)],
                execution_time=time.time() - self.execution_start_time
            )
    
    async def _execute_master_agent(self, binary_path: str, output_dir: str) -> bool:
        """Execute the Deus Ex Machina master agent"""
        self.logger.info("🤖 Executing Deus Ex Machina (Master Orchestrator)...")
        
        try:
            # Import master agent
            from .agents.agent00_deus_ex_machina import DeusExMachinaAgent
            
            master_agent = DeusExMachinaAgent()
            
            # Setup output directory structure
            output_path = Path(output_dir)
            output_paths = {
                'base': output_path,
                'agents': output_path / 'agents',
                'ghidra': output_path / 'ghidra',
                'compilation': output_path / 'compilation',
                'reports': output_path / 'reports',
                'logs': output_path / 'logs',
                'temp': output_path / 'temp',
                'tests': output_path / 'tests'
            }
            
            # Create output directories
            for path in output_paths.values():
                path.mkdir(parents=True, exist_ok=True)
            
            # Prepare master context
            master_context = {
                'binary_path': binary_path,
                'output_dir': output_dir,
                'output_paths': output_paths,
                'pipeline_config': self.config,
                'selected_agents': self.selected_agents,
                'execution_mode': self.config.execution_mode.value
            }
            
            # Execute master agent
            master_result = await master_agent.execute_async(master_context)
            
            if master_result.success:
                self.global_context = master_result.data
                self.master_executed = True
                self.logger.info("✅ Deus Ex Machina executed successfully")
                return True
            else:
                self.logger.error(f"❌ Deus Ex Machina failed: {master_result.error}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Master agent execution error: {e}")
            return False
    
    async def _execute_parallel_agents(self) -> Dict[int, Any]:
        """Execute selected agents in parallel"""
        self.logger.info(f"⚡ Executing {len(self.selected_agents)} agents in parallel...")
        
        # Create agent execution tasks
        tasks = []
        for agent_id in self.selected_agents:
            task = asyncio.create_task(
                self._execute_single_agent(agent_id),
                name=f"agent_{agent_id}"
            )
            tasks.append((agent_id, task))
        
        # Execute agents with resource limits
        agent_results = {}
        completed_count = 0
        
        for agent_id, task in tasks:
            try:
                result = await asyncio.wait_for(
                    task, 
                    timeout=self.config.resource_limits.timeout_agent
                )
                agent_results[agent_id] = result
                
                from .matrix_agents_v2 import AgentStatus
                if result.status == AgentStatus.SUCCESS:
                    completed_count += 1
                    self.logger.info(f"✅ Agent {agent_id} completed successfully")
                else:
                    error_msg = result.error_message or "Unknown error"
                    self.logger.warning(f"⚠️ Agent {agent_id} failed: {error_msg}")
                
            except asyncio.TimeoutError:
                self.logger.error(f"⏱️ Agent {agent_id} timed out")
                agent_results[agent_id] = self._create_timeout_result(agent_id)
                
            except Exception as e:
                self.logger.error(f"❌ Agent {agent_id} error: {e}")
                agent_results[agent_id] = self._create_error_result(agent_id, str(e))
        
        self.logger.info(f"🎯 Parallel execution complete: {completed_count}/{len(self.selected_agents)} successful")
        return agent_results
    
    async def _execute_single_agent(self, agent_id: int):
        """Execute a single agent with context"""
        try:
            # Get agent from agents registry
            from .agents import get_agent_by_id
            agent = get_agent_by_id(agent_id)
            
            # Prepare agent context
            agent_context = {
                **self.global_context,
                'agent_id': agent_id,
                'agent_results': self.agent_results.copy(),
                'pipeline_config': self.config
            }
            
            return agent.execute(agent_context)
            
        except Exception as e:
            self.logger.error(f"Agent {agent_id} execution failed: {e}")
            return self._create_error_result(agent_id, str(e))
    
    def _create_timeout_result(self, agent_id: int):
        """Create timeout result for agent"""
        from .matrix_agents_v2 import AgentResult, AgentStatus
        return AgentResult(
            agent_id=agent_id,
            agent_name=f"Agent{agent_id:02d}",
            matrix_character=f"agent_{agent_id:02d}",
            status=AgentStatus.FAILED,
            error_message=f"Agent {agent_id} execution timed out",
            execution_time=self.config.resource_limits.timeout_agent
        )
    
    def _create_error_result(self, agent_id: int, error_msg: str):
        """Create error result for agent"""
        from .matrix_agents_v2 import AgentResult, AgentStatus
        return AgentResult(
            agent_id=agent_id,
            agent_name=f"Agent{agent_id:02d}",
            matrix_character=f"agent_{agent_id:02d}",
            status=AgentStatus.FAILED,
            error_message=error_msg,
            execution_time=0.0
        )
    
    def _generate_pipeline_result(self, agent_results: Dict[int, Any]) -> MatrixPipelineResult:
        """Generate final pipeline result"""
        from .matrix_agents_v2 import AgentStatus
        
        successful_count = sum(1 for result in agent_results.values() 
                             if result.status == AgentStatus.SUCCESS)
        failed_count = len(agent_results) - successful_count
        
        # Collect error messages
        error_messages = []
        for agent_id, result in agent_results.items():
            if result.status != AgentStatus.SUCCESS:
                error_msg = result.error_message or f"Agent {agent_id} failed"
                error_messages.append(f"Agent {agent_id}: {error_msg}")
        
        # Calculate performance metrics
        execution_time = time.time() - self.execution_start_time
        
        return MatrixPipelineResult(
            success=failed_count == 0,
            execution_time=execution_time,
            total_agents=len(agent_results),
            successful_agents=successful_count,
            failed_agents=failed_count,
            agent_results=agent_results,
            error_messages=error_messages,
            performance_metrics={
                'total_execution_time': execution_time,
                'agent_count': len(agent_results),
                'success_rate': successful_count / len(agent_results) if agent_results else 0,
                'average_agent_time': sum(r.execution_time for r in agent_results.values()) / len(agent_results) if agent_results else 0
            }
        )
    
    async def _save_execution_report(self, result: MatrixPipelineResult, output_dir: str):
        """Save pipeline execution report"""
        try:
            report_path = Path(output_dir) / "reports" / "matrix_pipeline_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            
            report_data = {
                'pipeline_execution': {
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'timestamp': time.time(),
                    'config': {
                        'pipeline_mode': self.config.pipeline_mode.value,
                        'execution_mode': self.config.execution_mode.value,
                        'selected_agents': self.selected_agents
                    }
                },
                'agent_summary': {
                    'total_agents': result.total_agents,
                    'successful_agents': result.successful_agents,
                    'failed_agents': result.failed_agents,
                    'success_rate': result.performance_metrics.get('success_rate', 0)
                },
                'performance_metrics': result.performance_metrics,
                'errors': result.error_messages
            }
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            result.report_path = str(report_path)
            self.logger.info(f"📊 Pipeline report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline report: {e}")


async def execute_matrix_pipeline_orchestrated(
    binary_path: str,
    output_dir: str,
    config: Optional[PipelineConfig] = None
) -> MatrixPipelineResult:
    """
    Execute Matrix pipeline with orchestration
    Convenience function for external usage
    """
    if config is None:
        config = PipelineConfig()
    
    orchestrator = MatrixPipelineOrchestrator(config)
    return await orchestrator.execute_pipeline(binary_path, output_dir)


def create_pipeline_config(
    pipeline_mode: str = "full_pipeline",
    execution_mode: str = "master_first_parallel",
    resource_profile: str = "standard",
    **kwargs
) -> PipelineConfig:
    """Create pipeline configuration from string parameters"""
    
    # Convert string enums
    mode = PipelineMode(pipeline_mode)
    exec_mode = MatrixExecutionMode(execution_mode)
    
    # Get resource limits
    resource_map = {
        'standard': MatrixResourceLimits.STANDARD(),
        'high_performance': MatrixResourceLimits.HIGH_PERFORMANCE(),
        'conservative': MatrixResourceLimits.CONSERVATIVE()
    }
    resources = resource_map.get(resource_profile, MatrixResourceLimits.STANDARD())
    
    return PipelineConfig(
        pipeline_mode=mode,
        execution_mode=exec_mode,
        resource_limits=resources,
        **kwargs
    )