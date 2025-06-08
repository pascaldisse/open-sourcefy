"""
Phase 2 Enhancement: Advanced Agent Reliability System
Implements robust retry mechanisms, health monitoring, and fallback strategies.
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .agent_base import BaseAgent, AgentResult, AgentStatus


class HealthStatus(Enum):
    """Agent health status indicators"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Different retry strategies for agents"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    CUSTOM = "custom"


@dataclass
class HealthMetrics:
    """Health metrics for an agent"""
    agent_id: int
    status: HealthStatus = HealthStatus.UNKNOWN
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_patterns: List[str] = field(default_factory=list)
    consecutive_failures: int = 0
    health_score: float = 1.0  # 0.0 to 1.0
    last_health_check: Optional[float] = None


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    custom_delay_func: Optional[Callable[[int], float]] = None
    retry_on_errors: List[str] = field(default_factory=lambda: ["timeout", "memory", "dependency"])
    critical_errors_no_retry: List[str] = field(default_factory=lambda: ["fatal", "corruption"])


@dataclass
class FallbackStrategy:
    """Fallback strategy for agent failures"""
    agent_id: int
    fallback_agents: List[int] = field(default_factory=list)
    simplified_mode: bool = True
    skip_non_critical: bool = True
    alternative_approach: Optional[Callable] = None
    rollback_on_failure: bool = True


class AgentHealthMonitor:
    """Real-time health monitoring for agents"""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.health_metrics: Dict[int, HealthMetrics] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger("AgentHealthMonitor")
        self._lock = threading.RLock()
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for health monitoring"""
        with self._lock:
            self.health_metrics[agent.agent_id] = HealthMetrics(agent_id=agent.agent_id)
        self.logger.info(f"Registered agent {agent.agent_id} for health monitoring")
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started agent health monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Stopped agent health monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_health_metrics()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def _update_health_metrics(self) -> None:
        """Update health metrics for all registered agents"""
        current_time = time.time()
        
        with self._lock:
            for agent_id, metrics in self.health_metrics.items():
                # Update system metrics
                try:
                    process = psutil.Process()
                    metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                    metrics.cpu_usage_percent = process.cpu_percent()
                except:
                    pass  # Process info not available
                
                # Calculate health score
                metrics.health_score = self._calculate_health_score(metrics)
                
                # Determine health status
                metrics.status = self._determine_health_status(metrics)
                metrics.last_health_check = current_time
    
    def _calculate_health_score(self, metrics: HealthMetrics) -> float:
        """Calculate overall health score for an agent"""
        score = 1.0
        
        # Factor in success/failure ratio
        total_executions = metrics.success_count + metrics.failure_count
        if total_executions > 0:
            success_ratio = metrics.success_count / total_executions
            score *= success_ratio
        
        # Penalize consecutive failures
        if metrics.consecutive_failures > 0:
            penalty = min(0.8, metrics.consecutive_failures * 0.1)
            score *= (1.0 - penalty)
        
        # Factor in execution time (penalize very slow agents)
        if metrics.avg_execution_time > 60:  # 1 minute threshold
            time_penalty = min(0.3, (metrics.avg_execution_time - 60) / 300)
            score *= (1.0 - time_penalty)
        
        # Factor in memory usage (penalize high memory usage)
        if metrics.memory_usage_mb > 1000:  # 1GB threshold
            memory_penalty = min(0.2, (metrics.memory_usage_mb - 1000) / 5000)
            score *= (1.0 - memory_penalty)
        
        return max(0.0, min(1.0, score))
    
    def _determine_health_status(self, metrics: HealthMetrics) -> HealthStatus:
        """Determine health status based on metrics"""
        if metrics.health_score >= 0.9:
            return HealthStatus.HEALTHY
        elif metrics.health_score >= 0.7:
            return HealthStatus.DEGRADED
        elif metrics.health_score >= 0.4:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def update_agent_result(self, agent_id: int, result: AgentResult) -> None:
        """Update metrics based on agent execution result"""
        with self._lock:
            if agent_id not in self.health_metrics:
                return
            
            metrics = self.health_metrics[agent_id]
            current_time = time.time()
            
            if result.status == AgentStatus.COMPLETED:
                metrics.success_count += 1
                metrics.last_success_time = current_time
                metrics.consecutive_failures = 0
            else:
                metrics.failure_count += 1
                metrics.last_failure_time = current_time
                metrics.consecutive_failures += 1
                
                # Track error patterns
                if result.error_message:
                    error_type = self._classify_error(result.error_message)
                    if error_type not in metrics.error_patterns:
                        metrics.error_patterns.append(error_type)
            
            # Update average execution time
            total_executions = metrics.success_count + metrics.failure_count
            if total_executions == 1:
                metrics.avg_execution_time = result.execution_time
            else:
                # Exponential moving average
                alpha = 0.3
                metrics.avg_execution_time = (alpha * result.execution_time + 
                                            (1 - alpha) * metrics.avg_execution_time)
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type for pattern tracking"""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower or "oom" in error_lower:
            return "memory"
        elif "dependency" in error_lower or "prerequisite" in error_lower:
            return "dependency"
        elif "file not found" in error_lower or "no such file" in error_lower:
            return "file_not_found"
        elif "permission" in error_lower or "access denied" in error_lower:
            return "permission"
        elif "network" in error_lower or "connection" in error_lower:
            return "network"
        else:
            return "unknown"
    
    def get_agent_health(self, agent_id: int) -> Optional[HealthMetrics]:
        """Get health metrics for a specific agent"""
        with self._lock:
            return self.health_metrics.get(agent_id)
    
    def get_all_health_metrics(self) -> Dict[int, HealthMetrics]:
        """Get health metrics for all agents"""
        with self._lock:
            return self.health_metrics.copy()
    
    def is_agent_healthy(self, agent_id: int) -> bool:
        """Check if an agent is healthy enough to execute"""
        metrics = self.get_agent_health(agent_id)
        if not metrics:
            return True  # Assume healthy if no data
        
        return metrics.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        with self._lock:
            report = {
                "timestamp": time.time(),
                "total_agents": len(self.health_metrics),
                "health_summary": {
                    "healthy": 0,
                    "degraded": 0,
                    "unhealthy": 0,
                    "critical": 0
                },
                "agents": {}
            }
            
            for agent_id, metrics in self.health_metrics.items():
                report["health_summary"][metrics.status.value] += 1
                report["agents"][agent_id] = {
                    "status": metrics.status.value,
                    "health_score": metrics.health_score,
                    "success_rate": (metrics.success_count / 
                                   (metrics.success_count + metrics.failure_count)
                                   if (metrics.success_count + metrics.failure_count) > 0 else 0),
                    "avg_execution_time": metrics.avg_execution_time,
                    "consecutive_failures": metrics.consecutive_failures,
                    "error_patterns": metrics.error_patterns
                }
            
            return report


class AdvancedRetryMechanism:
    """Advanced retry mechanism with multiple strategies"""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("AdvancedRetryMechanism")
    
    def execute_with_retry(self, agent: BaseAgent, context: Dict[str, Any], 
                          health_monitor: Optional[AgentHealthMonitor] = None) -> AgentResult:
        """Execute agent with advanced retry logic"""
        last_result = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Check health before execution (if monitor available)
                if health_monitor and not health_monitor.is_agent_healthy(agent.agent_id):
                    self.logger.warning(f"Agent {agent.agent_id} health check failed, attempting anyway...")
                
                # Execute agent
                result = agent.run(context)
                
                # Update health metrics
                if health_monitor:
                    health_monitor.update_agent_result(agent.agent_id, result)
                
                # Check if successful
                if result.status == AgentStatus.COMPLETED:
                    if attempt > 0:
                        self.logger.info(f"Agent {agent.agent_id} succeeded on attempt {attempt + 1}")
                    return result
                
                last_result = result
                
                # Check if we should retry
                if not self._should_retry(result, attempt):
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Agent {agent.agent_id} failed (attempt {attempt + 1}), "
                    f"retrying in {delay:.2f}s... Error: {result.error_message}"
                )
                time.sleep(delay)
                
            except Exception as e:
                error_message = str(e)
                self.logger.error(f"Agent {agent.agent_id} execution error: {error_message}")
                
                last_result = AgentResult(
                    agent_id=agent.agent_id,
                    status=AgentStatus.FAILED,
                    data={},
                    error_message=error_message
                )
                
                # Update health metrics
                if health_monitor:
                    health_monitor.update_agent_result(agent.agent_id, last_result)
                
                if not self._should_retry_exception(error_message, attempt):
                    break
                
                delay = self._calculate_delay(attempt)
                time.sleep(delay)
        
        return last_result or AgentResult(
            agent_id=agent.agent_id,
            status=AgentStatus.FAILED,
            data={},
            error_message="Maximum retries exceeded"
        )
    
    def _should_retry(self, result: AgentResult, attempt: int) -> bool:
        """Determine if we should retry based on result"""
        if attempt >= self.config.max_retries:
            return False
        
        if not result.error_message:
            return False
        
        # Check for critical errors that shouldn't be retried
        error_lower = result.error_message.lower()
        for critical_error in self.config.critical_errors_no_retry:
            if critical_error in error_lower:
                self.logger.info(f"Critical error detected, not retrying: {critical_error}")
                return False
        
        # Check if error type is in retry list
        for retry_error in self.config.retry_on_errors:
            if retry_error in error_lower:
                return True
        
        # Default retry behavior
        return True
    
    def _should_retry_exception(self, error_message: str, attempt: int) -> bool:
        """Determine if we should retry based on exception"""
        return self._should_retry(
            AgentResult(agent_id=0, status=AgentStatus.FAILED, data={}, error_message=error_message),
            attempt
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt"""
        if self.config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * (attempt + 1)
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** attempt)
        
        elif self.config.strategy == RetryStrategy.CUSTOM and self.config.custom_delay_func:
            delay = self.config.custom_delay_func(attempt)
        
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            import random
            jitter = delay * 0.1 * random.random()
            delay += jitter
        
        return delay


class FallbackManager:
    """Manages fallback strategies for failed agents"""
    
    def __init__(self):
        self.fallback_strategies: Dict[int, FallbackStrategy] = {}
        self.logger = logging.getLogger("FallbackManager")
    
    def register_fallback_strategy(self, strategy: FallbackStrategy) -> None:
        """Register a fallback strategy for an agent"""
        self.fallback_strategies[strategy.agent_id] = strategy
        self.logger.info(f"Registered fallback strategy for agent {strategy.agent_id}")
    
    def execute_fallback(self, failed_agent_id: int, context: Dict[str, Any],
                        available_agents: Dict[int, BaseAgent]) -> Optional[AgentResult]:
        """Execute fallback strategy for a failed agent"""
        if failed_agent_id not in self.fallback_strategies:
            self.logger.warning(f"No fallback strategy for agent {failed_agent_id}")
            return None
        
        strategy = self.fallback_strategies[failed_agent_id]
        self.logger.info(f"Executing fallback strategy for agent {failed_agent_id}")
        
        # Try fallback agents
        for fallback_agent_id in strategy.fallback_agents:
            if fallback_agent_id in available_agents:
                self.logger.info(f"Trying fallback agent {fallback_agent_id}")
                
                fallback_agent = available_agents[fallback_agent_id]
                
                # Create modified context for simplified mode
                fallback_context = context.copy()
                if strategy.simplified_mode:
                    fallback_context['simplified_mode'] = True
                    fallback_context['skip_non_critical'] = strategy.skip_non_critical
                
                try:
                    result = fallback_agent.run(fallback_context)
                    if result.status == AgentStatus.COMPLETED:
                        self.logger.info(f"Fallback agent {fallback_agent_id} succeeded")
                        # Mark as fallback result
                        result.metadata = result.metadata or {}
                        result.metadata['fallback_for'] = failed_agent_id
                        result.metadata['fallback_agent'] = fallback_agent_id
                        return result
                except Exception as e:
                    self.logger.warning(f"Fallback agent {fallback_agent_id} failed: {e}")
                    continue
        
        # Try alternative approach if available
        if strategy.alternative_approach:
            try:
                self.logger.info(f"Trying alternative approach for agent {failed_agent_id}")
                result = strategy.alternative_approach(context)
                if result and result.status == AgentStatus.COMPLETED:
                    result.metadata = result.metadata or {}
                    result.metadata['alternative_approach'] = True
                    return result
            except Exception as e:
                self.logger.warning(f"Alternative approach failed: {e}")
        
        self.logger.error(f"All fallback strategies failed for agent {failed_agent_id}")
        return None


def create_default_fallback_strategies() -> List[FallbackStrategy]:
    """Create default fallback strategies for common agent failures"""
    strategies = []
    
    # Agent 3 (Smart Error Pattern Matching) -> Agent 1 (Basic discovery)
    strategies.append(FallbackStrategy(
        agent_id=3,
        fallback_agents=[1],
        simplified_mode=True
    ))
    
    # Agent 7 (Advanced Decompiler) -> Agent 4 (Basic Decompiler)
    strategies.append(FallbackStrategy(
        agent_id=7,
        fallback_agents=[4],
        simplified_mode=True
    ))
    
    # Agent 9 (Advanced Assembly Analyzer) -> Agent 5 (Binary Structure Analyzer)
    strategies.append(FallbackStrategy(
        agent_id=9,
        fallback_agents=[5],
        simplified_mode=True
    ))
    
    return strategies