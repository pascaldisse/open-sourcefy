"""
Matrix Parallel Executor
Advanced parallel execution engine for Matrix Phase 4 pipeline
"""

import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import psutil

from .shared_utils import LoggingUtils, PerformanceMonitor
from .config_manager import get_config_manager


class MatrixExecutionMode(Enum):
    """Matrix execution modes for parallel processing"""
    MASTER_FIRST_PARALLEL = "master_first_parallel"
    SEQUENTIAL = "sequential"
    FULL_PARALLEL = "full_parallel"
    HYBRID = "hybrid"
    THREADED = "threaded"
    PROCESS_POOL = "process_pool"


@dataclass
class MatrixResourceLimits:
    """Resource limits for Matrix execution"""
    max_memory: str = "4G"
    max_cpu_percent: int = 80
    max_disk_io: str = "100MB/s"
    max_parallel_agents: int = 16
    timeout_agent: int = 300
    timeout_master: int = 600
    max_threads: Optional[int] = None
    max_processes: Optional[int] = None
    
    def __post_init__(self):
        """Set default values based on system resources"""
        if self.max_threads is None:
            self.max_threads = min(self.max_parallel_agents, os.cpu_count() * 2)
        if self.max_processes is None:
            self.max_processes = min(self.max_parallel_agents, os.cpu_count())
    
    @classmethod
    def STANDARD(cls):
        return cls()
    
    @classmethod
    def HIGH_PERFORMANCE(cls):
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
        return cls(
            max_memory="2G",
            max_cpu_percent=50,
            max_disk_io="50MB/s",
            max_parallel_agents=8,
            timeout_agent=180,
            timeout_master=300
        )


@dataclass
class MatrixExecutionConfig:
    """Configuration for Matrix parallel execution"""
    execution_mode: MatrixExecutionMode = MatrixExecutionMode.MASTER_FIRST_PARALLEL
    resource_limits: MatrixResourceLimits = field(default_factory=MatrixResourceLimits.STANDARD)
    
    # Execution parameters
    max_retries: int = 3
    retry_delay: float = 1.0
    continue_on_failure: bool = True
    fail_fast: bool = False
    
    # Monitoring
    enable_monitoring: bool = True
    monitor_interval: float = 1.0
    log_progress: bool = True
    
    # Performance
    enable_profiling: bool = False
    memory_monitoring: bool = True
    cpu_monitoring: bool = True


@dataclass
class MatrixExecutionResult:
    """Result from Matrix parallel execution"""
    success: bool = False
    execution_time: float = 0.0
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    retried_tasks: int = 0
    task_results: Dict[str, Any] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)


class MatrixResourceMonitor:
    """Monitor system resources during Matrix execution"""
    
    def __init__(self, config: MatrixExecutionConfig):
        self.config = config
        self.logger = logging.getLogger("MatrixResourceMonitor")
        self.monitoring = False
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': [],
            'timestamps': []
        }
        
    def start_monitoring(self):
        """Start resource monitoring"""
        if not self.config.enable_monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.debug("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        self.logger.debug("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # CPU usage
                if self.config.cpu_monitoring:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    self.metrics['cpu_usage'].append(cpu_percent)
                
                # Memory usage
                if self.config.memory_monitoring:
                    memory = psutil.virtual_memory()
                    self.metrics['memory_usage'].append(memory.percent)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.metrics['disk_io'].append({
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    })
                
                # Network I/O
                net_io = psutil.net_io_counters()
                if net_io:
                    self.metrics['network_io'].append({
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv
                    })
                
                self.metrics['timestamps'].append(timestamp)
                
                time.sleep(self.config.monitor_interval)
                
            except Exception as e:
                self.logger.warning(f"Monitoring error: {e}")
                time.sleep(self.config.monitor_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary"""
        if not self.metrics['timestamps']:
            return {}
        
        summary = {}
        
        if self.metrics['cpu_usage']:
            summary['cpu'] = {
                'avg': sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']),
                'max': max(self.metrics['cpu_usage']),
                'min': min(self.metrics['cpu_usage'])
            }
        
        if self.metrics['memory_usage']:
            summary['memory'] = {
                'avg': sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']),
                'max': max(self.metrics['memory_usage']),
                'min': min(self.metrics['memory_usage'])
            }
        
        summary['duration'] = self.metrics['timestamps'][-1] - self.metrics['timestamps'][0]
        summary['samples'] = len(self.metrics['timestamps'])
        
        return summary


class MatrixParallelExecutor:
    """
    Matrix Parallel Executor
    Advanced parallel execution engine with resource management
    """
    
    def __init__(self, config: MatrixExecutionConfig):
        self.config = config
        self.logger = LoggingUtils.setup_agent_logging(0, "matrix_executor")
        self.resource_monitor = MatrixResourceMonitor(config)
        self.performance_monitor = PerformanceMonitor()
        
        # Execution state
        self.execution_start_time = None
        self.task_results = {}
        self.failed_tasks = set()
        self.retried_tasks = set()
        
        # Resource management
        self._thread_pool = None
        self._process_pool = None
    
    def __enter__(self):
        """Context manager entry"""
        self._initialize_pools()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._cleanup_pools()
    
    def _initialize_pools(self):
        """Initialize thread and process pools"""
        try:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config.resource_limits.max_threads
            )
            self._process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.resource_limits.max_processes
            )
            self.logger.debug("Executor pools initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize pools: {e}")
    
    def _cleanup_pools(self):
        """Cleanup thread and process pools"""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        self.logger.debug("Executor pools cleaned up")
    
    async def execute_pipeline(
        self,
        tasks: List[Callable],
        context: Dict[str, Any]
    ) -> MatrixExecutionResult:
        """
        Execute pipeline with Matrix parallel processing
        
        Args:
            tasks: List of callable tasks to execute
            context: Execution context shared across tasks
            
        Returns:
            MatrixExecutionResult with execution details
        """
        self.execution_start_time = time.time()
        
        try:
            self.logger.info(f"🔮 Matrix Parallel Executor starting {len(tasks)} tasks")
            self.logger.info(f"Execution Mode: {self.config.execution_mode.value}")
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Execute based on mode
            if self.config.execution_mode == MatrixExecutionMode.MASTER_FIRST_PARALLEL:
                result = await self._execute_master_first_parallel(tasks, context)
            elif self.config.execution_mode == MatrixExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(tasks, context)
            elif self.config.execution_mode == MatrixExecutionMode.FULL_PARALLEL:
                result = await self._execute_full_parallel(tasks, context)
            elif self.config.execution_mode == MatrixExecutionMode.THREADED:
                result = await self._execute_threaded(tasks, context)
            elif self.config.execution_mode == MatrixExecutionMode.PROCESS_POOL:
                result = await self._execute_process_pool(tasks, context)
            else:
                result = await self._execute_hybrid(tasks, context)
            
            # Stop monitoring and collect metrics
            self.resource_monitor.stop_monitoring()
            result.resource_usage = self.resource_monitor.get_summary()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return MatrixExecutionResult(
                success=False,
                error_messages=[str(e)],
                execution_time=time.time() - self.execution_start_time
            )
    
    async def _execute_master_first_parallel(
        self,
        tasks: List[Callable],
        context: Dict[str, Any]
    ) -> MatrixExecutionResult:
        """Execute master task first, then parallel execution"""
        
        # Execute master task (assumed to be first task)
        if tasks:
            master_task = tasks[0]
            remaining_tasks = tasks[1:]
            
            self.logger.info("🤖 Executing master task...")
            master_result = await self._execute_single_task(master_task, context, "master")
            
            if not master_result['success'] and not self.config.continue_on_failure:
                return MatrixExecutionResult(
                    success=False,
                    total_tasks=len(tasks),
                    failed_tasks=1,
                    error_messages=[f"Master task failed: {master_result.get('error', 'Unknown error')}"]
                )
            
            # Update context with master results
            context.update(master_result.get('data', {}))
            
            # Execute remaining tasks in parallel
            if remaining_tasks:
                self.logger.info(f"⚡ Executing {len(remaining_tasks)} tasks in parallel...")
                parallel_result = await self._execute_full_parallel(remaining_tasks, context)
                
                # Combine results
                parallel_result.total_tasks += 1
                if master_result['success']:
                    parallel_result.successful_tasks += 1
                else:
                    parallel_result.failed_tasks += 1
                
                parallel_result.task_results['master'] = master_result
                
                return parallel_result
        
        return MatrixExecutionResult(success=True, total_tasks=0)
    
    async def _execute_sequential(
        self,
        tasks: List[Callable],
        context: Dict[str, Any]
    ) -> MatrixExecutionResult:
        """Execute tasks sequentially"""
        successful_count = 0
        failed_count = 0
        task_results = {}
        error_messages = []
        
        for i, task in enumerate(tasks):
            task_id = f"task_{i}"
            self.logger.info(f"🔄 Executing task {i+1}/{len(tasks)}")
            
            result = await self._execute_single_task(task, context, task_id)
            task_results[task_id] = result
            
            if result['success']:
                successful_count += 1
                # Update context with task results
                context.update(result.get('data', {}))
            else:
                failed_count += 1
                error_messages.append(f"Task {i}: {result.get('error', 'Unknown error')}")
                
                if self.config.fail_fast:
                    break
        
        return MatrixExecutionResult(
            success=failed_count == 0,
            execution_time=time.time() - self.execution_start_time,
            total_tasks=len(tasks),
            successful_tasks=successful_count,
            failed_tasks=failed_count,
            task_results=task_results,
            error_messages=error_messages
        )
    
    async def _execute_full_parallel(
        self,
        tasks: List[Callable],
        context: Dict[str, Any]
    ) -> MatrixExecutionResult:
        """Execute all tasks in parallel"""
        
        # Create async tasks
        async_tasks = []
        for i, task in enumerate(tasks):
            task_id = f"task_{i}"
            async_task = asyncio.create_task(
                self._execute_single_task(task, context, task_id),
                name=task_id
            )
            async_tasks.append((task_id, async_task))
        
        # Execute with timeout
        task_results = {}
        successful_count = 0
        failed_count = 0
        error_messages = []
        
        for task_id, task in async_tasks:
            try:
                result = await asyncio.wait_for(
                    task,
                    timeout=self.config.resource_limits.timeout_agent
                )
                task_results[task_id] = result
                
                if result['success']:
                    successful_count += 1
                else:
                    failed_count += 1
                    error_messages.append(f"{task_id}: {result.get('error', 'Unknown error')}")
                    
            except asyncio.TimeoutError:
                failed_count += 1
                error_messages.append(f"{task_id}: Execution timeout")
                task_results[task_id] = {
                    'success': False,
                    'error': 'Execution timeout',
                    'execution_time': self.config.resource_limits.timeout_agent
                }
        
        return MatrixExecutionResult(
            success=failed_count == 0,
            execution_time=time.time() - self.execution_start_time,
            total_tasks=len(tasks),
            successful_tasks=successful_count,
            failed_tasks=failed_count,
            task_results=task_results,
            error_messages=error_messages
        )
    
    async def _execute_threaded(
        self,
        tasks: List[Callable],
        context: Dict[str, Any]
    ) -> MatrixExecutionResult:
        """Execute tasks using thread pool"""
        
        if not self._thread_pool:
            self._initialize_pools()
        
        loop = asyncio.get_event_loop()
        futures = []
        
        for i, task in enumerate(tasks):
            task_id = f"task_{i}"
            future = loop.run_in_executor(
                self._thread_pool,
                lambda t=task, c=context, tid=task_id: asyncio.run(self._execute_single_task(t, c, tid))
            )
            futures.append((task_id, future))
        
        # Collect results
        task_results = {}
        successful_count = 0
        failed_count = 0
        error_messages = []
        
        for task_id, future in futures:
            try:
                result = await future
                task_results[task_id] = result
                
                if result['success']:
                    successful_count += 1
                else:
                    failed_count += 1
                    error_messages.append(f"{task_id}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_count += 1
                error_msg = f"{task_id}: {str(e)}"
                error_messages.append(error_msg)
                task_results[task_id] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': 0.0
                }
        
        return MatrixExecutionResult(
            success=failed_count == 0,
            execution_time=time.time() - self.execution_start_time,
            total_tasks=len(tasks),
            successful_tasks=successful_count,
            failed_tasks=failed_count,
            task_results=task_results,
            error_messages=error_messages
        )
    
    async def _execute_process_pool(
        self,
        tasks: List[Callable],
        context: Dict[str, Any]
    ) -> MatrixExecutionResult:
        """Execute tasks using process pool"""
        
        if not self._process_pool:
            self._initialize_pools()
        
        loop = asyncio.get_event_loop()
        futures = []
        
        for i, task in enumerate(tasks):
            task_id = f"task_{i}"
            future = loop.run_in_executor(
                self._process_pool,
                task,
                context
            )
            futures.append((task_id, future))
        
        # Collect results (similar to threaded execution)
        return await self._collect_executor_results(futures)
    
    async def _execute_hybrid(
        self,
        tasks: List[Callable],
        context: Dict[str, Any]
    ) -> MatrixExecutionResult:
        """Execute tasks using hybrid approach"""
        # Determine optimal execution strategy based on task types
        cpu_intensive_tasks = []
        io_intensive_tasks = []
        
        for i, task in enumerate(tasks):
            # Simple heuristic: assume CPU-intensive if task has certain characteristics
            if hasattr(task, '_cpu_intensive') and task._cpu_intensive:
                cpu_intensive_tasks.append((i, task))
            else:
                io_intensive_tasks.append((i, task))
        
        # Execute CPU-intensive tasks in process pool
        # Execute I/O-intensive tasks in thread pool or async
        
        # For now, default to full parallel
        return await self._execute_full_parallel(tasks, context)
    
    async def _execute_single_task(
        self,
        task: Callable,
        context: Dict[str, Any],
        task_id: str
    ) -> Dict[str, Any]:
        """Execute a single task with error handling and retries"""
        
        start_time = time.time()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(task):
                    result = await task(context)
                else:
                    result = task(context)
                
                execution_time = time.time() - start_time
                
                return {
                    'success': True,
                    'data': result,
                    'execution_time': execution_time,
                    'attempts': attempt + 1,
                    'task_id': task_id
                }
                
            except Exception as e:
                self.logger.warning(f"Task {task_id} attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    execution_time = time.time() - start_time
                    return {
                        'success': False,
                        'error': str(e),
                        'execution_time': execution_time,
                        'attempts': attempt + 1,
                        'task_id': task_id
                    }
    
    async def _collect_executor_results(self, futures: List[Tuple[str, Any]]) -> MatrixExecutionResult:
        """Collect results from executor futures"""
        task_results = {}
        successful_count = 0
        failed_count = 0
        error_messages = []
        
        for task_id, future in futures:
            try:
                result = await future
                task_results[task_id] = {
                    'success': True,
                    'data': result,
                    'task_id': task_id
                }
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                error_msg = f"{task_id}: {str(e)}"
                error_messages.append(error_msg)
                task_results[task_id] = {
                    'success': False,
                    'error': str(e),
                    'task_id': task_id
                }
        
        return MatrixExecutionResult(
            success=failed_count == 0,
            execution_time=time.time() - self.execution_start_time,
            total_tasks=len(futures),
            successful_tasks=successful_count,
            failed_tasks=failed_count,
            task_results=task_results,
            error_messages=error_messages
        )


async def execute_matrix_pipeline(
    tasks: List[Callable],
    context: Dict[str, Any],
    config: Optional[MatrixExecutionConfig] = None
) -> MatrixExecutionResult:
    """
    Execute Matrix pipeline with parallel processing
    Convenience function for external usage
    """
    if config is None:
        config = MatrixExecutionConfig()
    
    with MatrixParallelExecutor(config) as executor:
        return await executor.execute_pipeline(tasks, context)