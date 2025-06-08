"""
Parallel execution framework for agent system.
Handles batch processing, error handling, and retry logic.
"""

import asyncio
import concurrent.futures
import logging
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .agent_base import BaseAgent, AgentResult, AgentStatus, get_execution_batches, AGENT_DEPENDENCIES


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass
class ExecutionConfig:
    """Configuration for agent execution"""
    mode: ExecutionMode = ExecutionMode.HYBRID
    max_parallel_agents: int = 6
    retry_enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_per_agent: Optional[float] = 300.0  # 5 minutes
    fail_fast: bool = False
    continue_on_failure: bool = True


@dataclass
class BatchResult:
    """Result of executing a batch of agents"""
    batch_id: int
    agent_results: Dict[int, AgentResult]
    success_count: int
    failure_count: int
    execution_time: float
    errors: List[str]


class ParallelExecutor:
    """Manages parallel execution of agents with dependency handling"""
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.logger = logging.getLogger("ParallelExecutor")
        self.agents: Dict[int, BaseAgent] = {}
        self.execution_context: Dict[str, Any] = {
            'agent_results': {},
            'global_data': {},
            'execution_start_time': None
        }

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for execution"""
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered {agent}")

    def register_agents(self, agents: List[BaseAgent]) -> None:
        """Register multiple agents"""
        for agent in agents:
            self.register_agent(agent)

    def execute_agents_in_batches(self, 
                                agent_ids: Optional[List[int]] = None,
                                context: Optional[Dict[str, Any]] = None) -> Dict[int, AgentResult]:
        """
        Execute agents in dependency-based batches.
        Returns mapping of agent_id -> AgentResult
        """
        self.execution_context['execution_start_time'] = time.time()
        
        if context:
            self.execution_context['global_data'].update(context)

        # Determine which agents to execute
        target_agents = agent_ids or list(self.agents.keys())
        missing_agents = set(target_agents) - set(self.agents.keys())
        if missing_agents:
            raise ValueError(f"Agents not registered: {missing_agents}")

        # Get execution batches
        try:
            all_batches = get_execution_batches()
            # Filter batches to only include target agents
            filtered_batches = []
            for batch in all_batches:
                filtered_batch = [aid for aid in batch if aid in target_agents]
                if filtered_batch:
                    filtered_batches.append(filtered_batch)
        except Exception as e:
            self.logger.error(f"Failed to calculate execution batches: {e}")
            raise

        self.logger.info(f"Executing {len(target_agents)} agents in {len(filtered_batches)} batches")
        
        total_results = {}
        
        for batch_id, agent_batch in enumerate(filtered_batches):
            self.logger.info(f"Starting batch {batch_id + 1}/{len(filtered_batches)}: {agent_batch}")
            
            batch_result = self._execute_batch(batch_id, agent_batch)
            
            # Update global results
            total_results.update(batch_result.agent_results)
            self.execution_context['agent_results'].update(batch_result.agent_results)
            
            # Handle batch failures
            if batch_result.failure_count > 0:
                self.logger.warning(f"Batch {batch_id} had {batch_result.failure_count} failures")
                
                if self.config.fail_fast:
                    self.logger.error("Fail-fast enabled, stopping execution")
                    break
                
                if not self.config.continue_on_failure:
                    failed_agents = [aid for aid, result in batch_result.agent_results.items() 
                                   if result.status == AgentStatus.FAILED]
                    self.logger.error(f"Stopping execution due to failed agents: {failed_agents}")
                    break

        total_time = time.time() - self.execution_context['execution_start_time']
        self.logger.info(f"Total execution completed in {total_time:.2f} seconds")
        
        return total_results

    def _execute_batch(self, batch_id: int, agent_ids: List[int]) -> BatchResult:
        """Execute a single batch of agents"""
        start_time = time.time()
        
        if self.config.mode == ExecutionMode.SEQUENTIAL:
            results = self._execute_sequential(agent_ids)
        elif self.config.mode == ExecutionMode.PARALLEL:
            results = self._execute_parallel(agent_ids)
        else:  # HYBRID
            results = self._execute_hybrid(agent_ids)
        
        execution_time = time.time() - start_time
        
        success_count = sum(1 for r in results.values() if r.status == AgentStatus.COMPLETED)
        failure_count = sum(1 for r in results.values() if r.status == AgentStatus.FAILED)
        
        errors = [r.error_message for r in results.values() 
                 if r.error_message and r.status == AgentStatus.FAILED]
        
        return BatchResult(
            batch_id=batch_id,
            agent_results=results,
            success_count=success_count,
            failure_count=failure_count,
            execution_time=execution_time,
            errors=errors
        )

    def _execute_sequential(self, agent_ids: List[int]) -> Dict[int, AgentResult]:
        """Execute agents sequentially"""
        results = {}
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            result = self._execute_agent_with_retry(agent)
            results[agent_id] = result
        return results

    def _execute_parallel(self, agent_ids: List[int]) -> Dict[int, AgentResult]:
        """Execute agents in parallel using ThreadPoolExecutor"""
        results = {}
        max_workers = min(len(agent_ids), self.config.max_parallel_agents)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_agent = {
                executor.submit(self._execute_agent_with_retry, self.agents[agent_id]): agent_id
                for agent_id in agent_ids
            }
            
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_agent)
                    results[agent_id] = result
                except concurrent.futures.TimeoutError:
                    self.logger.error(f"Agent {agent_id} timed out")
                    results[agent_id] = AgentResult(
                        agent_id=agent_id,
                        status=AgentStatus.FAILED,
                        data={},
                        error_message="Agent execution timed out"
                    )
                except Exception as e:
                    self.logger.error(f"Agent {agent_id} execution failed: {e}")
                    results[agent_id] = AgentResult(
                        agent_id=agent_id,
                        status=AgentStatus.FAILED,
                        data={},
                        error_message=str(e)
                    )
        
        return results

    def _execute_hybrid(self, agent_ids: List[int]) -> Dict[int, AgentResult]:
        """Execute agents with hybrid approach (small batches in parallel)"""
        results = {}
        batch_size = min(self.config.max_parallel_agents, len(agent_ids))
        
        for i in range(0, len(agent_ids), batch_size):
            batch = agent_ids[i:i + batch_size]
            batch_results = self._execute_parallel(batch)
            results.update(batch_results)
        
        return results

    def _execute_agent_with_retry(self, agent: BaseAgent) -> AgentResult:
        """Execute a single agent with retry logic"""
        last_result = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                result = agent.run(self.execution_context)
                
                if result.status == AgentStatus.COMPLETED:
                    if attempt > 0:
                        self.logger.info(f"Agent {agent.agent_id} succeeded on attempt {attempt + 1}")
                    return result
                
                last_result = result
                
                if not self.config.retry_enabled or attempt == self.config.max_retries:
                    break
                
                self.logger.warning(f"Agent {agent.agent_id} failed (attempt {attempt + 1}), retrying...")
                time.sleep(self.config.retry_delay)
                
            except Exception as e:
                self.logger.error(f"Agent {agent.agent_id} execution error: {e}")
                last_result = AgentResult(
                    agent_id=agent.agent_id,
                    status=AgentStatus.FAILED,
                    data={},
                    error_message=str(e)
                )
                
                if not self.config.retry_enabled or attempt == self.config.max_retries:
                    break
                
                time.sleep(self.config.retry_delay)
        
        return last_result or AgentResult(
            agent_id=agent.agent_id,
            status=AgentStatus.FAILED,
            data={},
            error_message="Unknown execution failure"
        )

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution results"""
        results = self.execution_context.get('agent_results', {})
        
        summary = {
            'total_agents': len(results),
            'completed': sum(1 for r in results.values() if r.status == AgentStatus.COMPLETED),
            'failed': sum(1 for r in results.values() if r.status == AgentStatus.FAILED),
            'total_time': 0.0,
            'agent_times': {},
            'errors': []
        }
        
        if self.execution_context.get('execution_start_time'):
            summary['total_time'] = time.time() - self.execution_context['execution_start_time']
        
        for agent_id, result in results.items():
            summary['agent_times'][agent_id] = result.execution_time
            if result.error_message:
                summary['errors'].append(f"Agent {agent_id}: {result.error_message}")
        
        return summary


# Convenience function for simple execution
def execute_agents_in_batches(agents: List[BaseAgent], 
                            config: ExecutionConfig = None,
                            context: Dict[str, Any] = None) -> Dict[int, AgentResult]:
    """
    Convenience function to execute agents in batches.
    
    Args:
        agents: List of agent instances to execute
        config: Execution configuration
        context: Additional context data
        
    Returns:
        Dictionary mapping agent_id to AgentResult
    """
    executor = ParallelExecutor(config)
    executor.register_agents(agents)
    return executor.execute_agents_in_batches(context=context)