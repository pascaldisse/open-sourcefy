"""
Phase 2 Enhancement: Enhanced Parallel Executor with Performance Optimization
Integrates reliability, health monitoring, and advanced performance optimization.
"""

import asyncio
import concurrent.futures
import logging
import time
import threading
import psutil
import gc
import weakref
import sys
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .agent_base import BaseAgent, AgentResult, AgentStatus, get_execution_batches
from .agent_reliability import (
    AgentHealthMonitor, AdvancedRetryMechanism, FallbackManager,
    RetryConfig, HealthStatus, create_default_fallback_strategies
)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for agent execution"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PERFORMANCE_BASED = "performance_based"
    MEMORY_AWARE = "memory_aware"
    ADAPTIVE = "adaptive"


@dataclass
class ResourceLimits:
    """Resource limits for agent execution"""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_execution_time: Optional[float] = None
    max_concurrent_agents: Optional[int] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for agents"""
    agent_id: int
    execution_count: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    memory_usage_history: List[float] = field(default_factory=list)
    cpu_usage_history: List[float] = field(default_factory=list)
    success_rate: float = 1.0
    performance_score: float = 1.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class EnhancedExecutionConfig:
    """Enhanced configuration for agent execution"""
    # Basic execution settings
    max_parallel_agents: int = 6
    timeout_per_agent: Optional[float] = 300.0
    retry_enabled: bool = True
    continue_on_failure: bool = True
    
    # Performance optimization settings
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    dynamic_batch_sizing: bool = True
    memory_optimization: bool = True
    cpu_throttling: bool = True
    
    # Resource management
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    
    # Reliability settings
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    health_monitoring: bool = True
    fallback_enabled: bool = True
    
    # Performance monitoring
    performance_tracking: bool = True
    adaptive_timeouts: bool = True
    predictive_scheduling: bool = True


class MemoryManager:
    """Enhanced memory management for optimal performance"""
    
    def __init__(self, gc_threshold: float = 0.8, max_history_size: int = 1000):
        self.gc_threshold = gc_threshold  # Trigger GC when memory usage exceeds this ratio
        self.max_history_size = max_history_size
        self.memory_history = deque(maxlen=max_history_size)
        self.weak_references = weakref.WeakSet()
        self.last_gc_time = time.time()
        self.gc_interval = 30.0  # Minimum seconds between forced GC
        self.logger = logging.getLogger("MemoryManager")
    
    def monitor_memory(self) -> Dict[str, Any]:
        """Monitor current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            memory_stats = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': memory_percent,
                'system_memory_percent': system_memory.percent,
                'available_mb': system_memory.available / 1024 / 1024,
                'timestamp': time.time()
            }
            
            self.memory_history.append(memory_stats)
            return memory_stats
            
        except Exception as e:
            self.logger.warning(f"Failed to monitor memory: {e}")
            return {}
    
    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered"""
        current_time = time.time()
        
        # Don't trigger GC too frequently
        if current_time - self.last_gc_time < self.gc_interval:
            return False
        
        try:
            memory_stats = self.monitor_memory()
            memory_percent = memory_stats.get('memory_percent', 0)
            
            # Trigger GC if memory usage is high
            if memory_percent > self.gc_threshold * 100:
                return True
            
            # Also check if memory usage has grown significantly
            if len(self.memory_history) >= 10:
                recent_avg = sum(m.get('memory_percent', 0) for m in list(self.memory_history)[-5:]) / 5
                older_avg = sum(m.get('memory_percent', 0) for m in list(self.memory_history)[-10:-5]) / 5
                
                # If memory usage increased by more than 20%
                if recent_avg > older_avg * 1.2:
                    return True
                    
        except Exception as e:
            self.logger.warning(f"Failed to check GC trigger: {e}")
        
        return False
    
    def optimize_memory(self):
        """Perform memory optimization"""
        try:
            initial_stats = self.monitor_memory()
            
            # Force garbage collection
            collected = gc.collect()
            
            # Update last GC time
            self.last_gc_time = time.time()
            
            # Monitor results
            final_stats = self.monitor_memory()
            
            memory_freed = initial_stats.get('rss_mb', 0) - final_stats.get('rss_mb', 0)
            
            self.logger.info(
                f"Memory optimization completed: "
                f"collected {collected} objects, "
                f"freed {memory_freed:.2f} MB"
            )
            
            return {
                'objects_collected': collected,
                'memory_freed_mb': memory_freed,
                'initial_memory_mb': initial_stats.get('rss_mb', 0),
                'final_memory_mb': final_stats.get('rss_mb', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
            return {}
    
    def register_weak_reference(self, obj):
        """Register object for weak reference tracking"""
        try:
            self.weak_references.add(obj)
        except TypeError:
            # Object doesn't support weak references
            pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        if not self.memory_history:
            return {}
        
        recent_memory = list(self.memory_history)[-10:]
        
        return {
            'current_memory_mb': recent_memory[-1].get('rss_mb', 0) if recent_memory else 0,
            'peak_memory_mb': max(m.get('rss_mb', 0) for m in self.memory_history),
            'avg_memory_mb': sum(m.get('rss_mb', 0) for m in recent_memory) / len(recent_memory),
            'memory_trend': self._calculate_memory_trend(),
            'gc_count': gc.get_count(),
            'weak_refs_count': len(self.weak_references),
            'last_gc_time': self.last_gc_time
        }
    
    def _calculate_memory_trend(self) -> str:
        """Calculate memory usage trend"""
        if len(self.memory_history) < 5:
            return "insufficient_data"
        
        recent = list(self.memory_history)[-5:]
        older = list(self.memory_history)[-10:-5] if len(self.memory_history) >= 10 else recent[:3]
        
        recent_avg = sum(m.get('rss_mb', 0) for m in recent) / len(recent)
        older_avg = sum(m.get('rss_mb', 0) for m in older) / len(older)
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"


class ResourceMonitor:
    """Enhanced system resource monitoring during agent execution"""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.resource_history: List[Dict[str, Any]] = []
        self.memory_manager = MemoryManager()
        self.logger = logging.getLogger("ResourceMonitor")
        self._lock = threading.RLock()
    
    def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started resource monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_resource_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
    
    def _collect_resource_metrics(self) -> None:
        """Collect current resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'memory_available_mb': memory.available / (1024 * 1024),
                'memory_percent': memory.percent,
                'disk_used_percent': disk.percent,
                'active_threads': threading.active_count()
            }
            
            with self._lock:
                self.resource_history.append(metrics)
                # Keep only last 1000 entries
                if len(self.resource_history) > 1000:
                    self.resource_history = self.resource_history[-1000:]
                    
        except Exception as e:
            self.logger.warning(f"Failed to collect resource metrics: {e}")
    
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource usage"""
        with self._lock:
            if self.resource_history:
                return self.resource_history[-1].copy()
            else:
                return {}
    
    def is_resource_available(self, limits: ResourceLimits) -> bool:
        """Check if sufficient resources are available"""
        current = self.get_current_resources()
        if not current:
            return True  # Assume available if no data
        
        if limits.max_memory_mb and current.get('memory_used_mb', 0) > limits.max_memory_mb:
            return False
        
        if limits.max_cpu_percent and current.get('cpu_percent', 0) > limits.max_cpu_percent:
            return False
        
        return True


class PerformanceTracker:
    """Tracks performance metrics for agents"""
    
    def __init__(self):
        self.metrics: Dict[int, PerformanceMetrics] = {}
        self.logger = logging.getLogger("PerformanceTracker")
        self._lock = threading.RLock()
    
    def register_agent(self, agent_id: int) -> None:
        """Register an agent for performance tracking"""
        with self._lock:
            if agent_id not in self.metrics:
                self.metrics[agent_id] = PerformanceMetrics(agent_id=agent_id)
    
    def update_metrics(self, agent_id: int, result: AgentResult, 
                      memory_usage: float = 0.0, cpu_usage: float = 0.0) -> None:
        """Update performance metrics for an agent"""
        with self._lock:
            if agent_id not in self.metrics:
                self.register_agent(agent_id)
            
            metrics = self.metrics[agent_id]
            metrics.execution_count += 1
            
            # Update execution time metrics
            exec_time = result.execution_time
            metrics.total_execution_time += exec_time
            metrics.avg_execution_time = metrics.total_execution_time / metrics.execution_count
            metrics.min_execution_time = min(metrics.min_execution_time, exec_time)
            metrics.max_execution_time = max(metrics.max_execution_time, exec_time)
            
            # Update resource usage history
            if memory_usage > 0:
                metrics.memory_usage_history.append(memory_usage)
                if len(metrics.memory_usage_history) > 100:
                    metrics.memory_usage_history = metrics.memory_usage_history[-100:]
            
            if cpu_usage > 0:
                metrics.cpu_usage_history.append(cpu_usage)
                if len(metrics.cpu_usage_history) > 100:
                    metrics.cpu_usage_history = metrics.cpu_usage_history[-100:]
            
            # Calculate performance score
            metrics.performance_score = self._calculate_performance_score(metrics, result)
            metrics.last_updated = time.time()
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics, 
                                   latest_result: AgentResult) -> float:
        """Calculate performance score for an agent"""
        score = 1.0
        
        # Success rate factor
        if latest_result.status == AgentStatus.COMPLETED:
            # Improve success rate
            metrics.success_rate = min(1.0, metrics.success_rate + 0.1)
        else:
            # Decrease success rate
            metrics.success_rate = max(0.0, metrics.success_rate - 0.2)
        
        score *= metrics.success_rate
        
        # Execution time factor (faster is better)
        if metrics.avg_execution_time > 0:
            # Normalize based on expected execution time (30 seconds baseline)
            time_factor = min(1.0, 30.0 / metrics.avg_execution_time)
            score *= time_factor
        
        # Memory efficiency factor
        if metrics.memory_usage_history:
            avg_memory = sum(metrics.memory_usage_history) / len(metrics.memory_usage_history)
            # Penalize high memory usage (>500MB)
            if avg_memory > 500:
                memory_penalty = min(0.5, (avg_memory - 500) / 1000)
                score *= (1.0 - memory_penalty)
        
        return max(0.0, min(1.0, score))
    
    def get_agent_performance(self, agent_id: int) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a specific agent"""
        with self._lock:
            return self.metrics.get(agent_id)
    
    def get_top_performers(self, count: int = 5) -> List[int]:
        """Get list of top performing agents"""
        with self._lock:
            sorted_agents = sorted(
                self.metrics.items(),
                key=lambda x: x[1].performance_score,
                reverse=True
            )
            return [agent_id for agent_id, _ in sorted_agents[:count]]
    
    def estimate_execution_time(self, agent_id: int) -> float:
        """Estimate execution time for an agent"""
        with self._lock:
            metrics = self.metrics.get(agent_id)
            if metrics and metrics.execution_count > 0:
                return metrics.avg_execution_time
            else:
                return 30.0  # Default estimate


class LoadBalancer:
    """Intelligent load balancer for agent execution"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE):
        self.strategy = strategy
        self.current_loads: Dict[int, float] = defaultdict(float)
        self.logger = logging.getLogger("LoadBalancer")
        self._lock = threading.RLock()
    
    def assign_agents_to_workers(self, agent_ids: List[int], num_workers: int,
                                performance_tracker: PerformanceTracker,
                                resource_monitor: ResourceMonitor) -> List[List[int]]:
        """Assign agents to workers based on load balancing strategy"""
        if num_workers <= 0 or not agent_ids:
            return []
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_assignment(agent_ids, num_workers)
        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return self._performance_based_assignment(agent_ids, num_workers, performance_tracker)
        elif self.strategy == LoadBalancingStrategy.MEMORY_AWARE:
            return self._memory_aware_assignment(agent_ids, num_workers, performance_tracker)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return self._adaptive_assignment(agent_ids, num_workers, performance_tracker, resource_monitor)
        else:
            return self._round_robin_assignment(agent_ids, num_workers)
    
    def _round_robin_assignment(self, agent_ids: List[int], num_workers: int) -> List[List[int]]:
        """Simple round-robin assignment"""
        workers = [[] for _ in range(num_workers)]
        for i, agent_id in enumerate(agent_ids):
            workers[i % num_workers].append(agent_id)
        return workers
    
    def _performance_based_assignment(self, agent_ids: List[int], num_workers: int,
                                    performance_tracker: PerformanceTracker) -> List[List[int]]:
        """Assign based on agent performance history"""
        workers = [[] for _ in range(num_workers)]
        worker_loads = [0.0] * num_workers
        
        # Sort agents by estimated execution time (longest first)
        agent_times = []
        for agent_id in agent_ids:
            est_time = performance_tracker.estimate_execution_time(agent_id)
            agent_times.append((agent_id, est_time))
        
        agent_times.sort(key=lambda x: x[1], reverse=True)
        
        # Assign to least loaded worker
        for agent_id, est_time in agent_times:
            min_load_idx = min(range(num_workers), key=lambda i: worker_loads[i])
            workers[min_load_idx].append(agent_id)
            worker_loads[min_load_idx] += est_time
        
        return workers
    
    def _memory_aware_assignment(self, agent_ids: List[int], num_workers: int,
                               performance_tracker: PerformanceTracker) -> List[List[int]]:
        """Assign based on memory usage patterns"""
        workers = [[] for _ in range(num_workers)]
        worker_memory_loads = [0.0] * num_workers
        
        # Sort agents by memory usage
        agent_memory = []
        for agent_id in agent_ids:
            metrics = performance_tracker.get_agent_performance(agent_id)
            if metrics and metrics.memory_usage_history:
                avg_memory = sum(metrics.memory_usage_history) / len(metrics.memory_usage_history)
            else:
                avg_memory = 100.0  # Default estimate
            agent_memory.append((agent_id, avg_memory))
        
        agent_memory.sort(key=lambda x: x[1], reverse=True)
        
        # Assign to worker with least memory load
        for agent_id, memory_usage in agent_memory:
            min_memory_idx = min(range(num_workers), key=lambda i: worker_memory_loads[i])
            workers[min_memory_idx].append(agent_id)
            worker_memory_loads[min_memory_idx] += memory_usage
        
        return workers
    
    def _adaptive_assignment(self, agent_ids: List[int], num_workers: int,
                           performance_tracker: PerformanceTracker,
                           resource_monitor: ResourceMonitor) -> List[List[int]]:
        """Adaptive assignment based on current system state"""
        current_resources = resource_monitor.get_current_resources()
        
        # Choose strategy based on current system state
        if current_resources.get('memory_percent', 0) > 80:
            # High memory usage - use memory-aware assignment
            return self._memory_aware_assignment(agent_ids, num_workers, performance_tracker)
        elif current_resources.get('cpu_percent', 0) > 80:
            # High CPU usage - use performance-based assignment
            return self._performance_based_assignment(agent_ids, num_workers, performance_tracker)
        else:
            # Normal conditions - use round-robin
            return self._round_robin_assignment(agent_ids, num_workers)


class EnhancedParallelExecutor:
    """Enhanced parallel executor with performance optimization and reliability"""
    
    def __init__(self, config: EnhancedExecutionConfig = None):
        self.config = config or EnhancedExecutionConfig()
        self.logger = logging.getLogger("EnhancedParallelExecutor")
        
        # Core components
        self.agents: Dict[int, BaseAgent] = {}
        self.execution_context: Dict[str, Any] = {
            'agent_results': {},
            'global_data': {},
            'execution_start_time': None
        }
        
        # Enhanced components with memory management
        self.health_monitor = AgentHealthMonitor() if self.config.health_monitoring else None
        self.retry_mechanism = AdvancedRetryMechanism(self.config.retry_config)
        self.fallback_manager = FallbackManager() if self.config.fallback_enabled else None
        self.performance_tracker = PerformanceTracker() if self.config.performance_tracking else None
        self.resource_monitor = ResourceMonitor() if self.config.memory_optimization else None
        self.load_balancer = LoadBalancer(self.config.load_balancing_strategy)
        self.memory_manager = MemoryManager() if self.config.memory_optimization else None
        
        # Setup fallback strategies
        if self.fallback_manager:
            for strategy in create_default_fallback_strategies():
                self.fallback_manager.register_fallback_strategy(strategy)
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for execution"""
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        
        self.agents[agent.agent_id] = agent
        
        # Register with monitoring components
        if self.health_monitor:
            self.health_monitor.register_agent(agent)
        if self.performance_tracker:
            self.performance_tracker.register_agent(agent.agent_id)
        
        self.logger.info(f"Registered {agent}")
    
    def register_agents(self, agents: List[BaseAgent]) -> None:
        """Register multiple agents"""
        for agent in agents:
            self.register_agent(agent)
    
    def start_monitoring(self) -> None:
        """Start all monitoring components"""
        if self.health_monitor:
            self.health_monitor.start_monitoring()
        if self.resource_monitor:
            self.resource_monitor.start_monitoring()
        self.logger.info("Started monitoring components")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components"""
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        self.logger.info("Stopped monitoring components")
    
    def execute_agents_in_batches(self, 
                                agent_ids: Optional[List[int]] = None,
                                context: Optional[Dict[str, Any]] = None) -> Dict[int, AgentResult]:
        """Execute agents with enhanced reliability and performance optimization"""
        self.execution_context['execution_start_time'] = time.time()
        
        if context:
            self.execution_context['global_data'].update(context)
            # Ensure backward compatibility by providing key fields at top level
            for key in ['output_dir', 'output_paths', 'binary_path']:
                if key in context:
                    self.execution_context[key] = context[key]
        
        # Start monitoring
        self.start_monitoring()
        
        try:
            # Determine target agents
            target_agents = agent_ids or list(self.agents.keys())
            missing_agents = set(target_agents) - set(self.agents.keys())
            if missing_agents:
                raise ValueError(f"Agents not registered: {missing_agents}")
            
            # Get execution batches
            all_batches = get_execution_batches()
            filtered_batches = []
            for batch in all_batches:
                filtered_batch = [aid for aid in batch if aid in target_agents]
                if filtered_batch:
                    filtered_batches.append(filtered_batch)
            
            self.logger.info(f"Executing {len(target_agents)} agents in {len(filtered_batches)} batches")
            
            total_results = {}
            
            for batch_id, agent_batch in enumerate(filtered_batches):
                self.logger.info(f"Starting batch {batch_id + 1}/{len(filtered_batches)}: {agent_batch}")
                
                # Perform memory optimization before batch if needed
                if self.memory_manager and self.memory_manager.should_trigger_gc():
                    self.logger.info(f"Performing memory optimization before batch {batch_id + 1}")
                    optimization_result = self.memory_manager.optimize_memory()
                    self.logger.debug(f"Memory optimization result: {optimization_result}")
                
                # Execute batch with enhanced features
                batch_results = self._execute_enhanced_batch(batch_id, agent_batch)
                
                # Update global results
                total_results.update(batch_results)
                self.execution_context['agent_results'].update(batch_results)
                
                # Register objects for weak reference tracking
                if self.memory_manager:
                    for result in batch_results.values():
                        self.memory_manager.register_weak_reference(result)
                
                # Check for failures and apply fallbacks
                failed_agents = [aid for aid, result in batch_results.items() 
                               if result.status == AgentStatus.FAILED]
                
                if failed_agents and self.fallback_manager:
                    self.logger.info(f"Attempting fallbacks for failed agents: {failed_agents}")
                    for failed_agent_id in failed_agents:
                        fallback_result = self.fallback_manager.execute_fallback(
                            failed_agent_id, self.execution_context, self.agents
                        )
                        if fallback_result:
                            total_results[failed_agent_id] = fallback_result
                            self.execution_context['agent_results'][failed_agent_id] = fallback_result
            
            total_time = time.time() - self.execution_context['execution_start_time']
            self.logger.info(f"Enhanced execution completed in {total_time:.2f} seconds")
            
            # Final memory optimization
            if self.memory_manager:
                final_optimization = self.memory_manager.optimize_memory()
                self.logger.info(f"Final memory optimization: {final_optimization}")
            
            return total_results
            
        finally:
            self.stop_monitoring()
    
    def _execute_enhanced_batch(self, batch_id: int, agent_ids: List[int]) -> Dict[int, AgentResult]:
        """Execute a batch with enhanced features"""
        # Determine optimal worker count
        max_workers = self._calculate_optimal_workers(agent_ids)
        
        # Assign agents to workers using load balancing
        worker_assignments = self.load_balancer.assign_agents_to_workers(
            agent_ids, max_workers, self.performance_tracker, self.resource_monitor
        )
        
        self.logger.info(f"Executing batch {batch_id} with {max_workers} workers")
        
        results = {}
        
        # Execute workers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_agents = {}
            
            for worker_id, worker_agents in enumerate(worker_assignments):
                if worker_agents:  # Only submit if there are agents to execute
                    future = executor.submit(self._execute_worker, worker_id, worker_agents)
                    future_to_agents[future] = worker_agents
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_agents):
                worker_agents = future_to_agents[future]
                try:
                    worker_results = future.result(timeout=self.config.timeout_per_agent)
                    results.update(worker_results)
                except Exception as e:
                    self.logger.error(f"Worker failed for agents {worker_agents}: {e}")
                    # Create failed results for all agents in this worker
                    for agent_id in worker_agents:
                        results[agent_id] = AgentResult(
                            agent_id=agent_id,
                            status=AgentStatus.FAILED,
                            data={},
                            error_message=f"Worker execution failed: {str(e)}"
                        )
        
        return results
    
    def _execute_worker(self, worker_id: int, agent_ids: List[int]) -> Dict[int, AgentResult]:
        """Execute agents assigned to a worker"""
        results = {}
        
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                continue
            
            agent = self.agents[agent_id]
            
            # Check resource availability
            if (self.resource_monitor and self.config.resource_limits and 
                not self.resource_monitor.is_resource_available(self.config.resource_limits)):
                
                self.logger.warning(f"Insufficient resources for agent {agent_id}, waiting...")
                time.sleep(1.0)  # Brief wait for resources
            
            # Execute with enhanced retry mechanism
            start_time = time.time()
            result = self.retry_mechanism.execute_with_retry(
                agent, self.execution_context, self.health_monitor
            )
            
            # Track performance metrics
            if self.performance_tracker and self.resource_monitor:
                current_resources = self.resource_monitor.get_current_resources()
                memory_usage = current_resources.get('memory_used_mb', 0)
                cpu_usage = current_resources.get('cpu_percent', 0)
                
                self.performance_tracker.update_metrics(
                    agent_id, result, memory_usage, cpu_usage
                )
            
            results[agent_id] = result
            
            # Brief pause between agents to prevent resource contention
            if len(agent_ids) > 1:
                time.sleep(0.1)
        
        return results
    
    def _calculate_optimal_workers(self, agent_ids: List[int]) -> int:
        """Calculate optimal number of workers for the given agents"""
        base_workers = min(len(agent_ids), self.config.max_parallel_agents)
        
        if not self.config.dynamic_batch_sizing:
            return base_workers
        
        # Adjust based on system resources
        if self.resource_monitor:
            current_resources = self.resource_monitor.get_current_resources()
            
            # Reduce workers if high memory usage
            if current_resources.get('memory_percent', 0) > 85:
                base_workers = max(1, base_workers // 2)
            
            # Reduce workers if high CPU usage
            if current_resources.get('cpu_percent', 0) > 90:
                base_workers = max(1, base_workers // 2)
        
        return base_workers
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': time.time(),
            'configuration': {
                'max_parallel_agents': self.config.max_parallel_agents,
                'load_balancing_strategy': self.config.load_balancing_strategy.value,
                'health_monitoring_enabled': self.config.health_monitoring,
                'performance_tracking_enabled': self.config.performance_tracking
            }
        }
        
        # Add health metrics
        if self.health_monitor:
            report['health_metrics'] = self.health_monitor.get_health_report()
        
        # Add performance metrics
        if self.performance_tracker:
            report['performance_metrics'] = {
                agent_id: {
                    'execution_count': metrics.execution_count,
                    'avg_execution_time': metrics.avg_execution_time,
                    'success_rate': metrics.success_rate,
                    'performance_score': metrics.performance_score
                }
                for agent_id, metrics in self.performance_tracker.metrics.items()
            }
            
            report['top_performers'] = self.performance_tracker.get_top_performers()
        
        # Add resource metrics
        if self.resource_monitor:
            report['current_resources'] = self.resource_monitor.get_current_resources()
        
        # Add memory management statistics
        if self.memory_manager:
            report['memory_stats'] = self.memory_manager.get_memory_stats()
        
        return report
    
    def get_system_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive system optimization report"""
        return {
            'memory_optimization': {
                'enabled': self.memory_manager is not None,
                'stats': self.memory_manager.get_memory_stats() if self.memory_manager else {},
                'recommendations': self._get_memory_recommendations()
            },
            'performance_optimization': {
                'load_balancing_strategy': self.config.load_balancing_strategy.value,
                'parallel_agents': self.config.max_parallel_agents,
                'performance_metrics': self.get_performance_report() if self.performance_tracker else {}
            },
            'resource_management': {
                'monitoring_enabled': self.resource_monitor is not None,
                'current_resources': self.resource_monitor.get_current_resources() if self.resource_monitor else {},
                'limits': {
                    'max_memory_mb': self.config.resource_limits.max_memory_mb,
                    'max_cpu_percent': self.config.resource_limits.max_cpu_percent,
                    'max_execution_time': self.config.resource_limits.max_execution_time
                }
            },
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3)
            }
        }
    
    def _get_memory_recommendations(self) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if not self.memory_manager:
            recommendations.append("Enable memory optimization for better performance")
            return recommendations
        
        memory_stats = self.memory_manager.get_memory_stats()
        memory_trend = memory_stats.get('memory_trend', 'unknown')
        current_memory = memory_stats.get('current_memory_mb', 0)
        peak_memory = memory_stats.get('peak_memory_mb', 0)
        
        if memory_trend == 'increasing':
            recommendations.append("Memory usage is trending upward - consider more frequent garbage collection")
        
        if peak_memory > 1000:  # > 1GB
            recommendations.append("High peak memory usage detected - consider processing data in smaller batches")
        
        if current_memory > 500:  # > 500MB
            recommendations.append("Current memory usage is high - monitor for memory leaks")
        
        if not recommendations:
            recommendations.append("Memory usage is within normal parameters")
        
        return recommendations


# Convenience function for enhanced execution
def execute_agents_with_enhancements(agents: List[BaseAgent], 
                                   config: EnhancedExecutionConfig = None,
                                   context: Dict[str, Any] = None) -> Tuple[Dict[int, AgentResult], Dict[str, Any]]:
    """
    Execute agents with all Phase 2 enhancements.
    
    Returns:
        Tuple of (agent_results, performance_report)
    """
    executor = EnhancedParallelExecutor(config)
    executor.register_agents(agents)
    
    results = executor.execute_agents_in_batches(context=context)
    performance_report = executor.get_performance_report()
    
    return results, performance_report