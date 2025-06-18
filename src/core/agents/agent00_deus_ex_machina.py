"""
Agent 00: Deus Ex Machina - Master Pipeline Orchestrator
The supreme overseer that coordinates all Matrix agents to achieve the impossible:
reconstructing compilable source code from binary executables.

Production-ready implementation following SOLID principles and clean code standards.
"""

import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Matrix framework imports
from ..matrix_agents import MatrixAgent, MatrixCharacter, AgentStatus, AgentResult
from ..shared_components import MatrixLogger, MatrixFileManager, MatrixValidator
from ..exceptions import MatrixAgentError, ValidationError
from ..config_manager import get_config_manager

# Import actual agent implementations
from .agent01_sentinel import SentinelAgent
from .agent02_architect import ArchitectAgent  
from .agent03_merovingian import MerovingianAgent
from .agent04_agent_smith import AgentSmithAgent
from .agent05_neo_advanced_decompiler import NeoAgent
from .agent06_trainman_assembly_analysis import Agent6_Trainman_AssemblyAnalysis
from .agent07_keymaker_resource_reconstruction import Agent7_Keymaker_ResourceReconstruction
from .agent08_commander_locke import Agent8_CommanderLocke
from .agent09_the_machine import Agent9_TheMachine
from .agent10_twins_binary_diff import Agent10_Twins_BinaryDiff
from .agent11_the_oracle import Agent11_TheOracle
from .agent12_link import Agent12_Link
from .agent13_agent_johnson import Agent13_AgentJohnson
from .agent14_the_cleaner import Agent14_TheCleaner
from .agent15_analyst import Agent15_Analyst
from .agent16_agent_brown import Agent16_AgentBrown

@dataclass
class PipelineExecutionPlan:
    """Plan for executing the Matrix pipeline"""
    agent_batches: List[List[int]]
    execution_mode: str
    total_agents: int
    estimated_time: float

@dataclass
class MasterAgentResult:
    """Result structure for master agent execution"""
    success: bool
    data: Dict[str, Any]
    error: Optional[str]

class DeusExMachinaAgent(MatrixAgent):
    """
    Master orchestrator that guides the entire Matrix pipeline execution.
    Coordinates all 16 agents to transform binary into source code.
    """
    
    def __init__(self):
        super().__init__(
            agent_id=0,
            matrix_character=MatrixCharacter.ARCHITECT  # Use Architect as closest character
        )
        self.agent_name = "Agent00_DeusExMachina"
        self.matrix_character_name = "deus_ex_machina"
        
        # Master orchestrator settings
        self.config = get_config_manager()
        self.execution_timeout = self.config.get_value('pipeline.master_timeout', 3600)
        self.max_parallel_agents = self.config.get_value('pipeline.max_parallel_agents', 4)
        
        # Pipeline coordination state
        self.execution_plan: Optional[PipelineExecutionPlan] = None
        self.agent_results: Dict[int, AgentResult] = {}
        self.execution_start_time = 0.0
        
        # Agent class mapping
        self.agent_classes = {
            1: SentinelAgent,
            2: ArchitectAgent,
            3: MerovingianAgent,
            4: AgentSmithAgent,
            5: NeoAgent,
            6: Agent6_Trainman_AssemblyAnalysis,
            7: Agent7_Keymaker_ResourceReconstruction,
            8: Agent8_CommanderLocke,
            9: Agent9_TheMachine,
            10: Agent10_Twins_BinaryDiff,
            11: Agent11_TheOracle,
            12: Agent12_Link,
            13: Agent13_AgentJohnson,
            14: Agent14_TheCleaner,
            15: Agent15_Analyst,
            16: Agent16_AgentBrown
        }
        
    def get_matrix_description(self) -> str:
        """Get description of the master orchestrator"""
        return "Deus Ex Machina - Supreme Matrix orchestrator that guides the impossible transformation"
    
    async def execute_async(self, context: Dict[str, Any]) -> 'MasterAgentResult':
        """Async execution method expected by the orchestrator"""
        try:
            result_data = self.execute_matrix_task(context)
            return MasterAgentResult(
                success=True,
                data=result_data,
                error=None
            )
        except Exception as e:
            self.logger.error(f"Master agent execution failed: {str(e)}")
            return MasterAgentResult(
                success=False,
                data={},
                error=str(e)
            )
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute pure coordination orchestration - NO AGENT EXECUTION
        
        STRICT COMPLIANCE: rules.md #26 (NO FAKE COMPILATION), #1-3 (NO FALLBACKS)
        Master agent ONLY coordinates - Pipeline Orchestrator handles execution
        """
        self.logger.info("🎭 Deus Ex Machina: Pure coordination orchestration initiated")
        
        # Validate prerequisites with STRICT MODE
        if not self._validate_pipeline_prerequisites(context):
            raise MatrixAgentError("STRICT MODE: Pipeline prerequisites not satisfied - IMMEDIATE FAILURE")
        
        # Create optimized execution plan
        self.execution_plan = self._create_execution_plan(context)
        self.logger.info(f"📋 Execution plan optimized: {len(self.execution_plan.agent_batches)} batches, "
                        f"{self.execution_plan.total_agents} agents")
        
        # Enhanced coordination analysis
        coordination_analysis = self._perform_coordination_analysis(context)
        
        # Pipeline health monitoring setup
        monitoring_config = self._setup_pipeline_monitoring(context)
        
        # Dependency optimization
        dependency_optimization = self._optimize_dependencies(context)
        
        # Resource allocation planning
        resource_allocation = self._plan_resource_allocation(context)
        
        self.logger.info("✨ Deus Ex Machina: Pure coordination complete - Ready for Pipeline Orchestrator")
        
        # STRICT COMPLIANCE: Only coordination data, NO execution results
        return {
            'coordination_mode': 'pure_orchestration',
            'execution_plan': {
                'agent_batches': self.execution_plan.agent_batches,
                'execution_mode': self.execution_plan.execution_mode,
                'total_agents': self.execution_plan.total_agents,
                'estimated_time': self.execution_plan.estimated_time
            },
            'coordination_analysis': coordination_analysis,
            'monitoring_config': monitoring_config,
            'dependency_optimization': dependency_optimization,
            'resource_allocation': resource_allocation,
            'orchestrator_handoff': {
                'ready_for_execution': True,
                'coordination_complete': True,
                'pipeline_validated': True
            },
            # Preserve critical context
            'binary_path': context['binary_path'],
            'output_paths': context['output_paths'],
            'shared_memory': {
                'coordination_metadata': {
                    'orchestrator': 'deus_ex_machina',
                    'plan_version': '2.0',
                    'optimization_level': 'maximum'
                },
                'binary_metadata': {},
                'analysis_results': {},
                'decompilation_data': {},
                'reconstruction_info': {},
                'validation_status': {}
            }
        }
    
    def _create_execution_plan(self, context: Dict[str, Any]) -> PipelineExecutionPlan:
        """Create optimized execution plan for Matrix agents"""
        selected_agents = context.get('selected_agents', list(range(1, 17)))
        execution_mode = context.get('execution_mode', 'master_first_parallel')
        
        # Organize agents into dependency-based batches
        agent_batches = self._organize_agent_batches(selected_agents)
        
        # Estimate execution time
        estimated_time = self._estimate_execution_time(selected_agents)
        
        return PipelineExecutionPlan(
            agent_batches=agent_batches,
            execution_mode=execution_mode,
            total_agents=len(selected_agents),
            estimated_time=estimated_time
        )
    
    def _organize_agent_batches(self, selected_agents: List[int]) -> List[List[int]]:
        """Organize agents into batches based on dependencies"""
        # Use centralized dependencies from matrix_agents.py - SINGLE SOURCE OF TRUTH
        from ..matrix_agents import MATRIX_DEPENDENCIES
        
        batches = []
        remaining_agents = set(selected_agents)
        
        while remaining_agents:
            # Find agents with satisfied dependencies
            current_batch = []
            for agent_id in sorted(remaining_agents):
                deps = MATRIX_DEPENDENCIES.get(agent_id, [])
                if all(dep not in remaining_agents for dep in deps):
                    current_batch.append(agent_id)
            
            if not current_batch:
                # Break dependency deadlock - take agent with fewest remaining deps
                agent_id = min(remaining_agents, 
                             key=lambda a: len([d for d in MATRIX_DEPENDENCIES.get(a, []) if d in remaining_agents]))
                current_batch.append(agent_id)
            
            batches.append(current_batch)
            remaining_agents -= set(current_batch)
        
        return batches
    
    def _estimate_execution_time(self, selected_agents: List[int]) -> float:
        """Estimate total pipeline execution time"""
        # Base time estimates per agent type
        base_times = {
            1: 30,   # Sentinel - Binary analysis
            2: 45,   # Architect - Architecture analysis
            3: 120,  # Merovingian - Basic decompilation
            4: 60,   # Agent Smith - Binary structure
            5: 300,  # Neo - Advanced decompilation
            6: 180,  # Twins - Binary diff
            7: 150,  # Trainman - Assembly analysis
            8: 120,  # Keymaker - Resource reconstruction
            9: 200,  # Commander Locke - Global reconstruction
            10: 180, # Machine - Compilation orchestration
            11: 90,  # Oracle - Validation
            12: 60,  # Link - Cross-reference
            13: 90,  # Agent Johnson - Security analysis
            14: 120, # Cleaner - Code cleanup
            15: 150, # Analyst - Metadata analysis
            16: 60   # Agent Brown - Final QA
        }
        
        return sum(base_times.get(agent_id, 60) for agent_id in selected_agents)
    
    def _validate_pipeline_prerequisites(self, context: Dict[str, Any]) -> bool:
        """Validate that pipeline can execute successfully"""
        required_keys = ['binary_path', 'output_paths', 'selected_agents']
        
        for key in required_keys:
            if key not in context:
                self.logger.error(f"Missing required context key: {key}")
                return False
        
        # Validate binary exists
        binary_path = Path(context['binary_path'])
        if not binary_path.exists():
            self.logger.error(f"Binary file not found: {binary_path}")
            return False
        
        # Validate output directory
        output_dir = Path(context['output_paths']['base'])
        if not output_dir.exists():
            self.logger.error(f"Output directory not found: {output_dir}")
            return False
        
        return True
    
    def _perform_coordination_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced coordination analysis for optimal pipeline execution"""
        analysis = {
            'agent_dependency_graph': self._analyze_agent_dependencies(),
            'execution_optimization': self._analyze_execution_optimization(context),
            'resource_requirements': self._analyze_resource_requirements(context),
            'parallel_execution_potential': self._analyze_parallelization_potential(),
            'critical_path_analysis': self._analyze_critical_path()
        }
        
        self.logger.info("🧠 Coordination analysis complete - Pipeline optimized")
        return analysis
    
    def _setup_pipeline_monitoring(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Setup comprehensive pipeline health monitoring"""
        monitoring = {
            'health_check_intervals': {
                'agent_status': 30,  # seconds
                'memory_usage': 60,
                'execution_progress': 15
            },
            'failure_detection': {
                'timeout_thresholds': self._calculate_timeout_thresholds(),
                'error_patterns': self._define_error_patterns(),
                'recovery_strategies': self._define_recovery_strategies()
            },
            'performance_metrics': {
                'execution_time_tracking': True,
                'memory_usage_tracking': True,
                'quality_score_tracking': True
            },
            'alert_conditions': self._define_alert_conditions()
        }
        
        self.logger.info("📊 Pipeline monitoring configured")
        return monitoring
    
    def _optimize_dependencies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize agent dependencies for maximum parallelization"""
        optimization = {
            'dependency_resolution': self._resolve_complex_dependencies(),
            'parallel_batches': self._optimize_parallel_batches(),
            'execution_order': self._optimize_execution_order(),
            'bottleneck_identification': self._identify_bottlenecks(),
            'load_balancing': self._plan_load_balancing()
        }
        
        self.logger.info("⚡ Dependency optimization complete")
        return optimization
    
    def _plan_resource_allocation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan optimal resource allocation for pipeline execution"""
        allocation = {
            'memory_allocation': self._plan_memory_allocation(context),
            'cpu_utilization': self._plan_cpu_utilization(context),
            'io_optimization': self._plan_io_optimization(context),
            'temporary_storage': self._plan_temporary_storage(context),
            'concurrent_agent_limits': self._calculate_concurrent_limits()
        }
        
        self.logger.info("🎯 Resource allocation planned")
        return allocation
    
    def _analyze_agent_dependencies(self) -> Dict[str, Any]:
        """Analyze agent dependency graph for optimization"""
        from ..matrix_agents import MATRIX_DEPENDENCIES
        
        graph = {
            'dependency_matrix': MATRIX_DEPENDENCIES,
            'critical_dependencies': self._identify_critical_dependencies(),
            'parallel_groups': self._identify_parallel_groups(),
            'dependency_depth': self._calculate_dependency_depth()
        }
        
        return graph
    
    def _analyze_execution_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze execution optimization opportunities"""
        optimization = {
            'parallel_potential': self._calculate_parallel_potential(context),
            'resource_utilization': self._analyze_resource_utilization(context),
            'execution_time_prediction': self._predict_execution_times(context),
            'bottleneck_analysis': self._analyze_potential_bottlenecks(context)
        }
        
        return optimization
    
    def _analyze_resource_requirements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource requirements for pipeline execution"""
        requirements = {
            'memory_requirements': self._calculate_memory_requirements(context),
            'cpu_requirements': self._calculate_cpu_requirements(context),
            'storage_requirements': self._calculate_storage_requirements(context),
            'network_requirements': self._calculate_network_requirements(context)
        }
        
        return requirements
    
    def _analyze_parallelization_potential(self) -> Dict[str, Any]:
        """Analyze potential for parallel execution"""
        potential = {
            'max_parallel_agents': self._calculate_max_parallel_agents(),
            'optimal_batch_size': self._calculate_optimal_batch_size(),
            'parallel_efficiency': self._calculate_parallel_efficiency(),
            'synchronization_points': self._identify_synchronization_points()
        }
        
        return potential
    
    def _analyze_critical_path(self) -> Dict[str, Any]:
        """Analyze critical path through agent dependencies"""
        critical_path = {
            'longest_path': self._find_longest_dependency_path(),
            'critical_agents': self._identify_critical_path_agents(),
            'path_optimization': self._suggest_path_optimizations(),
            'execution_time_impact': self._calculate_critical_path_impact()
        }
        
        return critical_path
    
    # REMOVED: Agent execution methods - STRICT COMPLIANCE with rules.md #26
    # Master agent ONLY coordinates - Pipeline Orchestrator handles all execution
    
    def _identify_critical_dependencies(self) -> List[Tuple[int, int]]:
        """Identify critical dependencies that cannot be parallelized"""
        from ..matrix_agents import MATRIX_DEPENDENCIES
        
        critical = []
        for agent_id, deps in MATRIX_DEPENDENCIES.items():
            for dep in deps:
                # Mark as critical if dependency chain is long
                if len(MATRIX_DEPENDENCIES.get(dep, [])) > 2:
                    critical.append((dep, agent_id))
        
        return critical
    
    def _identify_parallel_groups(self) -> List[List[int]]:
        """Identify groups of agents that can run in parallel"""
        from ..matrix_agents import MATRIX_DEPENDENCIES
        
        # Simple parallel grouping based on no shared dependencies
        all_agents = set(range(1, 17))
        parallel_groups = []
        
        while all_agents:
            current_group = []
            remaining = list(all_agents)
            
            for agent_id in remaining:
                # Check if this agent can be added to current group
                deps = set(MATRIX_DEPENDENCIES.get(agent_id, []))
                if not any(dep in current_group for dep in deps):
                    current_group.append(agent_id)
                    all_agents.remove(agent_id)
            
            if current_group:
                parallel_groups.append(current_group)
            else:
                # Break deadlock
                agent_id = all_agents.pop()
                parallel_groups.append([agent_id])
        
        return parallel_groups
    
    def _calculate_dependency_depth(self) -> Dict[int, int]:
        """Calculate dependency depth for each agent"""
        from ..matrix_agents import MATRIX_DEPENDENCIES
        
        depth = {}
        
        def get_depth(agent_id: int) -> int:
            if agent_id in depth:
                return depth[agent_id]
            
            deps = MATRIX_DEPENDENCIES.get(agent_id, [])
            if not deps:
                depth[agent_id] = 0
                return 0
            
            max_dep_depth = max(get_depth(dep) for dep in deps)
            depth[agent_id] = max_dep_depth + 1
            return depth[agent_id]
        
        for agent_id in range(1, 17):
            get_depth(agent_id)
        
        return depth
    
    # REMOVED: Pipeline abort logic - STRICT COMPLIANCE with rules.md #26
    # Coordination agent does not handle execution failures
    
    def _calculate_timeout_thresholds(self) -> Dict[int, int]:
        """Calculate timeout thresholds for each agent"""
        base_times = {
            1: 30, 2: 45, 3: 120, 4: 60, 5: 300, 6: 180, 7: 150, 8: 120,
            9: 200, 10: 180, 11: 90, 12: 60, 13: 90, 14: 120, 15: 150, 16: 60
        }
        
        # Add safety margin
        return {agent_id: time * 2 for agent_id, time in base_times.items()}
    
    def _define_error_patterns(self) -> List[str]:
        """Define error patterns for monitoring"""
        return [
            'MatrixAgentError',
            'ValidationError', 
            'TimeoutError',
            'MemoryError',
            'FileNotFoundError'
        ]
    
    def _define_recovery_strategies(self) -> Dict[str, str]:
        """Define recovery strategies for different failure types"""
        return {
            'timeout': 'extend_timeout_and_retry',
            'memory': 'reduce_batch_size',
            'validation': 'skip_optional_validation',
            'file_access': 'create_missing_directories'
        }
    
    def _define_alert_conditions(self) -> Dict[str, Any]:
        """Define conditions that trigger alerts"""
        return {
            'agent_timeout': {'threshold': 300, 'action': 'escalate'},
            'memory_usage': {'threshold': 0.85, 'action': 'reduce_load'},
            'error_rate': {'threshold': 0.1, 'action': 'pause_execution'},
            'quality_score': {'threshold': 0.6, 'action': 'review_required'}
        }
    
    def _generate_master_report(self, context: Dict[str, Any], 
                              orchestration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive coordination report"""
        coordination_time = time.time() - (getattr(self, 'execution_start_time', time.time()))
        
        return {
            'coordination_summary': {
                'total_agents_planned': len(context.get('selected_agents', [])),
                'coordination_mode': 'pure_orchestration',
                'coordination_time': coordination_time,
                'orchestrator_status': 'coordination_complete',
                'pipeline_optimized': True
            },
            'execution_plan_details': {
                'total_batches': len(self.execution_plan.agent_batches),
                'agent_batches': self.execution_plan.agent_batches,
                'estimated_total_time': self.execution_plan.estimated_time,
                'execution_mode': self.execution_plan.execution_mode,
                'optimization_level': 'maximum'
            },
            'coordination_results': orchestration_results,
            'pipeline_readiness': {
                'dependencies_resolved': True,
                'resources_allocated': True,
                'monitoring_configured': True,
                'ready_for_execution': True
            },
            'optimization_achievements': {
                'dependency_optimization': 'complete',
                'resource_allocation': 'optimized',
                'parallel_potential': 'maximized',
                'critical_path': 'analyzed'
            },
            'handoff_instructions': [
                'Pipeline coordination complete',
                'Execute via Matrix Pipeline Orchestrator',
                'All dependencies resolved and optimized',
                'Monitoring configuration ready',
                'Resource allocation planned'
            ]
        }
    
    # REMOVED: Execution metrics methods - STRICT COMPLIANCE with rules.md #26
    # Coordination agent does not track execution metrics
    
    def _calculate_parallel_potential(self, context: Dict[str, Any]) -> float:
        """Calculate parallel execution potential"""
        selected_agents = context.get('selected_agents', list(range(1, 17)))
        parallel_groups = self._identify_parallel_groups()
        
        total_agents = len(selected_agents)
        max_parallel = max(len(group) for group in parallel_groups) if parallel_groups else 1
        
        return min(1.0, max_parallel / total_agents)
    
    def _analyze_resource_utilization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze expected resource utilization"""
        return {
            'cpu_utilization_estimate': 0.75,
            'memory_utilization_estimate': 0.65,
            'io_utilization_estimate': 0.55,
            'network_utilization_estimate': 0.25
        }
    
    def _predict_execution_times(self, context: Dict[str, Any]) -> Dict[int, int]:
        """Predict execution times for agents"""
        base_times = {
            1: 30, 2: 45, 3: 120, 4: 60, 5: 300, 6: 180, 7: 150, 8: 120,
            9: 200, 10: 180, 11: 90, 12: 60, 13: 90, 14: 120, 15: 150, 16: 60
        }
        
        selected_agents = context.get('selected_agents', list(range(1, 17)))
        return {agent_id: base_times.get(agent_id, 60) for agent_id in selected_agents}
    
    def _analyze_potential_bottlenecks(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential execution bottlenecks"""
        bottlenecks = []
        
        # Identify high-dependency agents
        from ..matrix_agents import MATRIX_DEPENDENCIES
        for agent_id, deps in MATRIX_DEPENDENCIES.items():
            if len(deps) > 3:
                bottlenecks.append({
                    'agent_id': agent_id,
                    'type': 'high_dependency_count',
                    'dependency_count': len(deps),
                    'impact': 'delays_parallel_execution'
                })
        
        # Identify long-running agents
        predicted_times = self._predict_execution_times(context)
        for agent_id, time in predicted_times.items():
            if time > 200:
                bottlenecks.append({
                    'agent_id': agent_id,
                    'type': 'long_execution_time',
                    'estimated_time': time,
                    'impact': 'extends_critical_path'
                })
        
        return bottlenecks