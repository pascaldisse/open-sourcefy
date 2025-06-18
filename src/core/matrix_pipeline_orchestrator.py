"""
Matrix Pipeline Orchestrator
Orchestrates the Matrix Phase 4 pipeline with master-first execution model
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .shared_utils import LoggingUtils
from .config_manager import get_config_manager
from .final_validation_orchestrator import FinalValidationOrchestrator


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
    enable_final_validation: bool = True
    use_cache: bool = True  # Enable caching by default


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
    
    # Final validation results (automatically added when compilation output exists)
    validation_report: Optional[Dict[str, Any]] = None
    final_validation_success: bool = False
    binary_match_percentage: float = 0.0


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
            PipelineMode.FULL_PIPELINE: list(range(1, 17)),  # All agents 1-16
            PipelineMode.DECOMPILE_ONLY: [1, 2, 3, 5, 7, 14],
            PipelineMode.ANALYZE_ONLY: [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15],
            PipelineMode.COMPILE_ONLY: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            PipelineMode.VALIDATE_ONLY: [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            PipelineMode.CUSTOM_AGENTS: self.config.custom_agents or []
        }
        
        agents = mode_agents.get(self.config.pipeline_mode, [])
        
        # Apply exclusions
        if self.config.exclude_agents:
            agents = [a for a in agents if a not in self.config.exclude_agents]
            
        return sorted(agents)
    
    def _check_agent_cache(self, agent_id: int, output_dir: str) -> Optional[Any]:
        """Check if agent results are cached and valid"""
        if not self.config.use_cache:
            return None
            
        try:
            # Check for agent output directory
            agent_output_dir = Path(output_dir) / 'agents' / f'agent_{agent_id:02d}_{self._get_agent_name(agent_id).lower()}'
            
            if not agent_output_dir.exists():
                self.logger.info(f"🔍 Agent {agent_id} cache miss: output directory not found at {agent_output_dir}")
                return None
            
            # Check for result file
            result_file = agent_output_dir / 'agent_result.json'
            if not result_file.exists():
                self.logger.info(f"🔍 Agent {agent_id} cache miss: result file not found at {result_file}")
                return None
            
            # Check for generated files (agent-specific)
            if not self._validate_agent_outputs(agent_id, agent_output_dir):
                self.logger.info(f"🔍 Agent {agent_id} cache miss: output validation failed")
                return None
            
            # Load cached result
            with open(result_file, 'r') as f:
                cached_data = json.load(f)
            
            # Validate cache format
            if not self._validate_cache_format(cached_data):
                self.logger.debug(f"🔍 Agent {agent_id} cache miss: invalid format")
                return None
            
            self.logger.info(f"✅ Agent {agent_id} cache hit: using cached results")
            return self._deserialize_agent_result(cached_data)
            
        except Exception as e:
            self.logger.debug(f"🔍 Agent {agent_id} cache miss: {e}")
            return None
    
    def _get_agent_name(self, agent_id: int) -> str:
        """Get agent name for cache directory"""
        agent_names = {
            1: "sentinel", 2: "architect", 3: "merovingian", 4: "agent_smith",
            5: "neo", 6: "twins", 7: "keymaker", 8: "keymaker", 9: "machine",
            10: "machine", 11: "oracle", 12: "link", 13: "johnson",
            14: "cleaner", 15: "analyst", 16: "brown"
        }
        return agent_names.get(agent_id, f"agent{agent_id}")
    
    def _validate_agent_outputs(self, agent_id: int, agent_output_dir: Path) -> bool:
        """Validate that agent outputs exist and are complete"""
        # Basic validation: just check if agent_result.json exists and is non-empty
        result_file = agent_output_dir / 'agent_result.json'
        if not result_file.exists() or result_file.stat().st_size == 0:
            self.logger.debug(f"Missing/empty result file: {result_file}")
            return False
        
        # Optional: check for any additional files that indicate successful completion
        # This is more lenient - we don't require specific files since agents may save different outputs
        
        # Count total files in the directory
        total_files = sum(1 for f in agent_output_dir.iterdir() if f.is_file())
        if total_files == 0:
            self.logger.debug(f"No files found in agent directory: {agent_output_dir}")
            return False
        
        self.logger.debug(f"Agent {agent_id} validation passed: {total_files} files in {agent_output_dir}")
        return True
    
    def _validate_cache_format(self, cached_data: dict) -> bool:
        """Validate cached result format"""
        required_keys = ['agent_id', 'status', 'data']
        return all(key in cached_data for key in required_keys)
    
    def _deserialize_agent_result(self, cached_data: dict):
        """Convert cached data back to AgentResult object"""
        from .matrix_agents import AgentResult, AgentStatus
        
        status_map = {
            'success': AgentStatus.SUCCESS,
            'failed': AgentStatus.FAILED,
            'error': AgentStatus.ERROR
        }
        
        return AgentResult(
            agent_id=cached_data['agent_id'],
            status=status_map.get(cached_data['status'], AgentStatus.ERROR),
            data=cached_data.get('data', {}),
            error_message=cached_data.get('error_message'),
            execution_time=cached_data.get('execution_time', 0.0)
        )
    
    def _save_agent_cache(self, agent_id: int, result, output_dir: str):
        """Save agent result to cache"""
        if not self.config.use_cache:
            return
            
        try:
            agent_output_dir = Path(output_dir) / 'agents' / f'agent_{agent_id:02d}_{self._get_agent_name(agent_id).lower()}'
            agent_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Serialize result
            cache_data = {
                'agent_id': agent_id,
                'status': result.status.value if hasattr(result.status, 'value') else str(result.status),
                'data': result.data if hasattr(result, 'data') else {},
                'error_message': getattr(result, 'error_message', None),
                'execution_time': getattr(result, 'execution_time', 0.0),
                'timestamp': time.time()
            }
            
            # Save to cache file
            result_file = agent_output_dir / 'agent_result.json'
            with open(result_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
            self.logger.debug(f"💾 Agent {agent_id} result cached to {result_file}")
            
        except Exception as e:
            self.logger.debug(f"⚠️ Failed to cache Agent {agent_id} result: {e}")
    
    async def execute_pipeline(self, binary_path: str, output_dir: str) -> MatrixPipelineResult:
        """Execute the complete Matrix pipeline"""
        self.execution_start_time = time.time()
        
        try:
            self.logger.info("🔮 Initializing Matrix Pipeline Orchestration...")
            self.logger.info(f"Pipeline Mode: {self.config.pipeline_mode.value}")
            self.logger.info(f"Execution Mode: {self.config.execution_mode.value}")
            self.logger.info(f"Selected Agents: {self.selected_agents}")
            self.logger.info(f"Cache Mode: {'Enabled' if self.config.use_cache else 'Disabled'}")
            
            # Step 1: Execute Deus Ex Machina (Master Agent)
            if not await self._execute_master_agent(binary_path, output_dir):
                return MatrixPipelineResult(
                    success=False,
                    error_messages=["Master agent (Deus Ex Machina) execution failed"],
                    execution_time=time.time() - self.execution_start_time
                )
            
            # Step 2: CRITICAL FIX - Always execute agents independently per rules.md
            # Rule #26: NO FAKE COMPILATION - Never simulate or mock compilation results
            # Rule #74: ALL OR NOTHING - Either execute fully or fail completely
            # Rule #82: STRICT SUCCESS CRITERIA - Only report success when all components work
            
            # Check if master agent attempted to bypass execution (VIOLATION)
            if 'agent_results' in self.global_context and self.global_context['agent_results']:
                # SECURITY FIX: Master agent bypass detected - this violates rules.md
                bypass_agents = list(self.global_context['agent_results'].keys())
                self.logger.error(f"🚨 RULES VIOLATION: Deus Ex Machina attempted to bypass agent execution for agents {bypass_agents}")
                self.logger.error("🚨 This violates rules.md Rule #26 (NO FAKE COMPILATION) and Rule #74 (ALL OR NOTHING)")
                self.logger.error("🚨 All agents must execute independently - no shortcuts allowed")
                
                # Clear fake results and force proper execution
                self.global_context['agent_results'] = {}
                self.logger.info("🔒 Cleared fake agent results, enforcing real execution")
            
            # Execute agents independently (ALWAYS - no exceptions)
            self.logger.info(f"🔥 Starting real agent execution for {len(self.selected_agents)} agents: {self.selected_agents}")
            agent_results = await self._execute_parallel_agents()
            
            # CRITICAL VALIDATION: Verify all agents attempted execution
            executed_agents = list(agent_results.keys())
            missing_agents = set(self.selected_agents) - set(executed_agents)
            
            if missing_agents:
                self.logger.error(f"🚨 EXECUTION FAILURE: Agents {sorted(missing_agents)} never attempted execution!")
                self.logger.error(f"🚨 Selected: {self.selected_agents}")
                self.logger.error(f"🚨 Executed: {executed_agents}")
                self.logger.error("🚨 This violates rules.md Rule #74 (ALL OR NOTHING)")
            
            # Log execution results summary
            successful_agents = [aid for aid, result in agent_results.items() 
                               if result.status.value == 'success']
            failed_agents = [aid for aid, result in agent_results.items() 
                           if result.status.value != 'success']
            
            self.logger.info(f"🎯 Agent execution summary:")
            self.logger.info(f"  - Selected: {len(self.selected_agents)} agents {self.selected_agents}")
            self.logger.info(f"  - Executed: {len(executed_agents)} agents {executed_agents}")
            self.logger.info(f"  - Successful: {len(successful_agents)} agents {successful_agents}")
            self.logger.info(f"  - Failed: {len(failed_agents)} agents {failed_agents}")
            if missing_agents:
                self.logger.error(f"  - Missing: {len(missing_agents)} agents {sorted(missing_agents)}")
            
            # Step 3: Execute final validation first (required for accurate success determination)
            final_validation_success = True
            validation_report = None
            final_validation_error = None
            
            if (self.config.enable_final_validation and 
                self._has_compilation_output(output_dir) and
                len([r for r in agent_results.values() if r.status.value == 'success']) == len(agent_results)):
                
                try:
                    final_validation_success, validation_report = await self._execute_final_validation_with_status(
                        self.global_context.get('binary_path', ''), output_dir)
                except Exception as e:
                    self.logger.error(f"Final validation failed: {e}")
                    final_validation_success = False
                    validation_report = None
                    final_validation_error = str(e)
            
            # Step 4: Generate final results (including final validation status)
            result = self._generate_pipeline_result(agent_results, final_validation_success, validation_report, final_validation_error)
            
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
        """Execute selected agents in dependency-based batches"""
        # Calculate execution batches based on dependencies
        execution_batches = self._calculate_execution_batches(self.selected_agents)
        
        self.logger.info(f"⚡ Executing {len(self.selected_agents)} agents in {len(execution_batches)} batches...")
        
        agent_results = {}
        total_completed = 0
        
        # Execute batches sequentially, agents within batches in parallel
        for batch_idx, batch_agents in enumerate(execution_batches):
            self.logger.info(f"🚀 Executing batch {batch_idx + 1}/{len(execution_batches)}: Agents {batch_agents}")
            
            # Pre-execution validation
            self.logger.debug(f"📋 Batch {batch_idx + 1} agent registry check:")
            for agent_id in batch_agents:
                try:
                    from .agents import get_agent_by_id
                    agent_class = get_agent_by_id(agent_id)
                    self.logger.debug(f"  ✅ Agent {agent_id}: {type(agent_class).__name__} available")
                except Exception as e:
                    self.logger.error(f"  ❌ Agent {agent_id}: MISSING - {e}")
            
            batch_results = await self._execute_agent_batch(batch_agents)
            
            # Update results and count successes
            batch_completed = 0
            for agent_id, result in batch_results.items():
                agent_results[agent_id] = result
                self.agent_results[agent_id] = result  # Update shared state for subsequent batches
                
                from .matrix_agents import AgentStatus
                if result.status == AgentStatus.SUCCESS:
                    batch_completed += 1
                    total_completed += 1
                    self.logger.info(f"✅ Agent {agent_id} completed successfully")
                else:
                    error_msg = result.error_message or "Unknown error"
                    self.logger.warning(f"⚠️ Agent {agent_id} failed: {error_msg}")
            
            self.logger.info(f"📊 Batch {batch_idx + 1} complete: {batch_completed}/{len(batch_agents)} successful")
            
            # Check if we should continue or abort on critical failures
            if not self.config.continue_on_failure and batch_completed < len(batch_agents):
                remaining_batches = len(execution_batches) - batch_idx - 1
                if remaining_batches > 0:
                    self.logger.warning(f"⚠️ Stopping execution due to failures (continue_on_failure=False)")
                    break
        
        self.logger.info(f"🎯 Batch execution complete: {total_completed}/{len(self.selected_agents)} successful")
        return agent_results
    
    def _calculate_execution_batches(self, selected_agents: List[int]) -> List[List[int]]:
        """Calculate execution batches for selected agents based on dependencies"""
        from .matrix_agents import MATRIX_DEPENDENCIES
        
        self.logger.info(f"🧮 Calculating execution batches for agents: {selected_agents}")
        
        # Filter dependencies to only include selected agents
        filtered_deps = {agent_id: [dep for dep in deps if dep in selected_agents] 
                        for agent_id, deps in MATRIX_DEPENDENCIES.items() 
                        if agent_id in selected_agents}
        
        self.logger.debug(f"📊 Filtered dependencies: {filtered_deps}")
        
        # Validate all selected agents have dependency definitions
        missing_deps = set(selected_agents) - set(filtered_deps.keys())
        if missing_deps:
            self.logger.warning(f"⚠️ Agents {sorted(missing_deps)} have no dependency definitions in MATRIX_DEPENDENCIES")
        
        completed = set()
        batches = []
        
        while len(completed) < len(selected_agents):
            current_batch = []
            
            self.logger.debug(f"🔄 Batch calculation iteration - completed: {sorted(completed)}")
            
            for agent_id in selected_agents:
                if agent_id in completed:
                    continue
                    
                # Check if all dependencies are completed
                dependencies = filtered_deps.get(agent_id, [])
                deps_satisfied = all(dep in completed for dep in dependencies)
                
                self.logger.debug(f"  Agent {agent_id}: deps={dependencies}, satisfied={deps_satisfied}")
                
                if deps_satisfied:
                    current_batch.append(agent_id)
            
            if not current_batch:
                remaining = set(selected_agents) - completed
                self.logger.error(f"🚨 Cannot resolve dependencies for agents: {remaining}")
                self.logger.error(f"🚨 Completed agents: {sorted(completed)}")
                self.logger.error(f"🚨 Dependency deadlock detected!")
                raise ValueError(f"Cannot resolve dependencies for agents: {remaining}")
            
            batches.append(sorted(current_batch))  # Sort for consistent ordering
            completed.update(current_batch)
            self.logger.debug(f"📦 Batch {len(batches)}: {sorted(current_batch)}")
        
        self.logger.info(f"✅ Batch calculation complete: {len(batches)} batches")
        for i, batch in enumerate(batches):
            self.logger.info(f"  Batch {i+1}: {batch}")
        
        return batches
    
    async def _execute_agent_batch(self, batch_agents: List[int]) -> Dict[int, Any]:
        """Execute a batch of agents in parallel"""
        if len(batch_agents) == 1:
            # Single agent - execute directly
            agent_id = batch_agents[0]
            result = await self._execute_single_agent(agent_id)
            return {agent_id: result}
        
        # Multiple agents - execute in parallel
        tasks = []
        for agent_id in batch_agents:
            task = asyncio.create_task(
                self._execute_single_agent(agent_id),
                name=f"agent_{agent_id}"
            )
            tasks.append((agent_id, task))
        
        batch_results = {}
        
        # Wait for all agents in batch to complete
        for agent_id, task in tasks:
            try:
                result = await asyncio.wait_for(
                    task, 
                    timeout=self.config.resource_limits.timeout_agent
                )
                batch_results[agent_id] = result
                
            except asyncio.TimeoutError:
                self.logger.error(f"⏱️ Agent {agent_id} timed out")
                batch_results[agent_id] = self._create_timeout_result(agent_id)
                
            except Exception as e:
                self.logger.error(f"❌ Agent {agent_id} error: {e}")
                batch_results[agent_id] = self._create_error_result(agent_id, str(e))
        
        return batch_results
    
    async def _execute_single_agent(self, agent_id: int):
        """Execute a single agent with context"""
        self.logger.info(f"🤖 Attempting to execute Agent {agent_id}...")
        
        # Check cache first (for --update mode or normal mode)
        output_dir = self.global_context.get('output_dir', '')
        self.logger.debug(f"🔍 Agent {agent_id} checking cache in: {output_dir}")
        cached_result = self._check_agent_cache(agent_id, output_dir)
        if cached_result is not None:
            self.logger.info(f"📦 Agent {agent_id} using cached result, skipping execution")
            # Ensure we return the cached result and skip execution completely
            return cached_result
        else:
            self.logger.info(f"🔍 Agent {agent_id} cache miss, proceeding with execution")
        
        try:
            # Get agent from agents registry
            from .agents import get_agent_by_id
            self.logger.debug(f"📦 Importing agent {agent_id} from registry...")
            agent = get_agent_by_id(agent_id)
            self.logger.debug(f"✅ Agent {agent_id} imported: {type(agent).__name__}")
            
            # Get current shared_memory state from global context or initialize
            current_shared_memory = self.global_context.get('shared_memory', {
                'binary_metadata': {},
                'analysis_results': {},
                'decompilation_data': {},
                'reconstruction_info': {},
                'validation_status': {}
            })
            
            # Ensure analysis_results section exists and is populated
            if 'analysis_results' not in current_shared_memory:
                current_shared_memory['analysis_results'] = {}
            
            # CRITICAL FIX: Copy completed agent results into shared_memory for dependency access
            # Store both as agent_id keys and formatted agent_XX keys for backward compatibility
            for completed_agent_id, completed_result in self.agent_results.items():
                # Store with agent_id as key for direct lookup
                current_shared_memory['analysis_results'][completed_agent_id] = completed_result.data if hasattr(completed_result, 'data') else completed_result
                # Store with formatted key for agents expecting agent_XX format
                agent_key = f'agent_{completed_agent_id:02d}'
                current_shared_memory['analysis_results'][agent_key] = completed_result.data if hasattr(completed_result, 'data') else completed_result
            
            # Update global context with updated shared_memory
            self.global_context['shared_memory'] = current_shared_memory
            
            # Prepare agent context with unified structure
            agent_context = {
                **self.global_context,
                'agent_id': agent_id,
                'agent_results': self.agent_results.copy(),  # CRITICAL: Pass ALL completed agent results
                'pipeline_config': self.config,
                'shared_memory': current_shared_memory,  # Use updated shared_memory
                # Add global_data alias for agents that expect it
                'global_data': {
                    'binary_path': self.global_context.get('binary_path'),
                    'output_dir': self.global_context.get('output_dir'),
                    'output_paths': self.global_context.get('output_paths', {})
                }
            }
            
            # CRITICAL FIX: Ensure agent can access dependency results
            self.logger.debug(f"Agent {agent_id} context contains agent_results for agents: {list(self.agent_results.keys())}")
            self.logger.debug(f"Agent {agent_id} shared_memory analysis_results keys: {list(current_shared_memory['analysis_results'].keys())}")
            
            self.logger.info(f"🔄 Executing Agent {agent_id} ({type(agent).__name__})...")
            result = agent.execute(agent_context)
            self.logger.info(f"✅ Agent {agent_id} execution completed with status: {result.status if hasattr(result, 'status') else 'unknown'}")
            
            # Save result to cache
            self._save_agent_cache(agent_id, result, output_dir)
            
            # Update global context with any shared memory changes from agent execution
            if 'shared_memory' in agent_context:
                self.global_context['shared_memory'] = agent_context['shared_memory']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent {agent_id} execution failed: {e}")
            return self._create_error_result(agent_id, str(e))
    
    def _create_timeout_result(self, agent_id: int):
        """Create timeout result for agent"""
        from .matrix_agents import AgentResult, AgentStatus
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
        from .matrix_agents import AgentResult, AgentStatus
        return AgentResult(
            agent_id=agent_id,
            agent_name=f"Agent{agent_id:02d}",
            matrix_character=f"agent_{agent_id:02d}",
            status=AgentStatus.FAILED,
            error_message=error_msg,
            execution_time=0.0
        )
    
    def _generate_pipeline_result(self, agent_results: Dict[int, Any], final_validation_success: bool = True, validation_report: dict = None, final_validation_error: str = None) -> MatrixPipelineResult:
        """Generate final pipeline result including final validation status"""
        from .matrix_agents import AgentStatus
        
        successful_count = sum(1 for result in agent_results.values() 
                             if result.status == AgentStatus.SUCCESS)
        failed_count = len(agent_results) - successful_count
        
        # Collect error messages
        error_messages = []
        for agent_id, result in agent_results.items():
            if result.status != AgentStatus.SUCCESS:
                error_msg = result.error_message or f"Agent {agent_id} failed"
                error_messages.append(f"Agent {agent_id}: {error_msg}")
        
        # Add final validation failure as error if applicable
        if not final_validation_success:
            if validation_report:
                # Failed with validation report (percentage mismatch)
                match_pct = validation_report.get('final_validation', {}).get('total_match_percentage', 0)
                validation_error = f"Final validation failed: {match_pct:.1f}% match (required: 95.0%). No partial success allowed per rules.md Rule #74: NO PARTIAL SUCCESS"
                error_messages.append(validation_error)
            elif final_validation_error:
                # Failed with exception
                error_messages.append(f"Final validation failed with error: {final_validation_error}. No partial success allowed per rules.md Rule #74: NO PARTIAL SUCCESS")
            else:
                # Failed for unknown reason
                error_messages.append("Final validation failed for unknown reason. No partial success allowed per rules.md Rule #74: NO PARTIAL SUCCESS")
        
        # Calculate performance metrics
        execution_time = time.time() - self.execution_start_time
        
        # CRITICAL FIX: Enhanced validation per rules.md
        # Rule #74: NO PARTIAL SUCCESS - Never report partial success when components fail
        # Rule #80: ALL OR NOTHING - Either execute fully or fail completely
        # Rule #82: STRICT SUCCESS CRITERIA - Only report success when all components work
        
        # Validate ALL selected agents actually executed
        expected_agents = set(self.selected_agents)
        executed_agents = set(agent_results.keys())
        missing_agents = expected_agents - executed_agents
        
        if missing_agents:
            error_messages.append(f"CRITICAL: Agents {sorted(missing_agents)} did not execute - violates Rule #74 (ALL OR NOTHING)")
            self.logger.error(f"🚨 Missing agent execution: {sorted(missing_agents)}")
        
        # Validate critical agents (1-5, 8-9, 11-13) executed successfully
        critical_agents = {1, 2, 3, 4, 5, 8, 9, 11, 12, 13}
        critical_in_selection = critical_agents.intersection(expected_agents)
        critical_failed = []
        
        for critical_agent in critical_in_selection:
            if critical_agent not in executed_agents:
                critical_failed.append(critical_agent)
            elif agent_results[critical_agent].status != AgentStatus.SUCCESS:
                critical_failed.append(critical_agent)
        
        if critical_failed:
            error_messages.append(f"CRITICAL: Core agents {sorted(critical_failed)} failed - pipeline cannot succeed per Rule #74")
            self.logger.error(f"🚨 Critical agent failures: {sorted(critical_failed)}")
        
        # Pipeline success requires: ALL agents executed + ALL succeeded + final validation success + NO critical failures
        agents_fully_executed = len(missing_agents) == 0
        agents_all_succeeded = failed_count == 0
        no_critical_failures = len(critical_failed) == 0
        
        overall_success = (agents_fully_executed and 
                          agents_all_succeeded and 
                          no_critical_failures and 
                          final_validation_success)
        
        result = MatrixPipelineResult(
            success=overall_success,
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
        
        # Add validation info to result
        if validation_report:
            result.validation_report = validation_report
            result.final_validation_success = final_validation_success
            result.binary_match_percentage = validation_report.get('final_validation', {}).get('total_match_percentage', 0)
        
        return result
    
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
            
            # Final validation was already executed before pipeline result generation
            # No need to run it again here
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline report: {e}")
    
    def _has_compilation_output(self, output_dir: str) -> bool:
        """Check if compilation output exists"""
        try:
            output_path = Path(output_dir)
            compilation_dir = output_path / "compilation"
            
            # Look for compiled binary
            bin_dirs = [
                compilation_dir / "bin" / "Release" / "Win32",
                compilation_dir / "bin" / "Debug" / "Win32",
                compilation_dir
            ]
            
            for bin_dir in bin_dirs:
                if bin_dir.exists():
                    exe_files = list(bin_dir.glob("*.exe"))
                    if exe_files:
                        return True
            
            return False
            
        except Exception:
            return False
    
    async def _execute_final_validation_with_status(self, binary_path: str, output_dir: str) -> Tuple[bool, dict]:
        """Execute final validation and return success status and report"""
        try:
            self.logger.info("🏆 Starting Final Validation for Perfect Binary Recompilation")
            
            # Find original and recompiled binaries
            original_binary = Path(binary_path)
            
            # Find recompiled binary
            output_path = Path(output_dir)
            compilation_dir = output_path / "compilation"
            
            recompiled_binary = None
            bin_dirs = [
                compilation_dir / "bin" / "Release" / "Win32",
                compilation_dir / "bin" / "Debug" / "Win32",
                compilation_dir
            ]
            
            for bin_dir in bin_dirs:
                if bin_dir.exists():
                    exe_files = list(bin_dir.glob("*.exe"))
                    if exe_files:
                        recompiled_binary = exe_files[0]
                        break
            
            if not recompiled_binary or not recompiled_binary.exists():
                self.logger.warning("No recompiled binary found for final validation")
                return False, None
            
            # Execute final validation
            validator = FinalValidationOrchestrator(self.config_manager)
            validation_report = await validator.execute_final_validation(
                original_binary, 
                recompiled_binary,
                output_path / "reports"
            )
            
            success = validation_report['final_validation']['success']
            match_pct = validation_report['final_validation']['total_match_percentage']
            
            # Log final status
            if success:
                self.logger.info(f"🎯 Final Validation: SUCCESS - {match_pct:.2f}% binary match achieved")
            else:
                self.logger.error(f"❌ Final Validation: FAILED - {match_pct:.2f}% binary match (required: 95.0%)")
            
            return success, validation_report
            
        except Exception as e:
            self.logger.error(f"Final validation failed: {e}")
            return False, None
    
    async def _execute_final_validation(self, binary_path: str, output_dir: str, result):
        """Execute final validation for perfect binary recompilation"""
        try:
            self.logger.info("🏆 Starting Final Validation for Perfect Binary Recompilation")
            
            # Find original and recompiled binaries
            original_binary = Path(binary_path)
            
            # Find recompiled binary
            output_path = Path(output_dir)
            compilation_dir = output_path / "compilation"
            
            recompiled_binary = None
            bin_dirs = [
                compilation_dir / "bin" / "Release" / "Win32",
                compilation_dir / "bin" / "Debug" / "Win32",
                compilation_dir
            ]
            
            for bin_dir in bin_dirs:
                if bin_dir.exists():
                    exe_files = list(bin_dir.glob("*.exe"))
                    if exe_files:
                        recompiled_binary = exe_files[0]
                        break
            
            if not recompiled_binary or not recompiled_binary.exists():
                self.logger.warning("No recompiled binary found for final validation")
                return
            
            # Execute final validation
            validator = FinalValidationOrchestrator(self.config_manager)
            validation_report = await validator.execute_final_validation(
                original_binary, 
                recompiled_binary,
                output_path / "reports"
            )
            
            # Update result with validation info
            result.validation_report = validation_report
            result.final_validation_success = validation_report['final_validation']['success']
            result.binary_match_percentage = validation_report['final_validation']['total_match_percentage']
            
            # Log final status
            match_pct = validation_report['final_validation']['total_match_percentage']
            if validation_report['final_validation']['success']:
                self.logger.info(f"🎯 Final Validation: SUCCESS - {match_pct:.2f}% binary match achieved")
            else:
                self.logger.info(f"⚠️ Final Validation: {match_pct:.2f}% binary match - improvement needed")
            
        except Exception as e:
            self.logger.error(f"Final validation failed: {e}")
            result.validation_report = None
            result.final_validation_success = False
            result.binary_match_percentage = 0.0


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