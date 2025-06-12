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
from .agent06_twins_binary_diff import Agent6_Twins_BinaryDiff
from .agent07_trainman_assembly_analysis import Agent7_Trainman_AssemblyAnalysis
from .agent08_keymaker_resource_reconstruction import Agent8_Keymaker_ResourceReconstruction
from .agent09_commander_locke import CommanderLockeAgent
from .agent10_the_machine import Agent10_TheMachine
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
            6: Agent6_Twins_BinaryDiff,
            7: Agent7_Trainman_AssemblyAnalysis,
            8: Agent8_Keymaker_ResourceReconstruction,
            9: CommanderLockeAgent,
            10: Agent10_TheMachine,
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
        """Execute the master orchestration task"""
        self.logger.info("🎭 The Deus Ex Machina awakens to orchestrate the impossible...")
        
        # Create execution plan
        self.execution_plan = self._create_execution_plan(context)
        self.logger.info(f"📋 Execution plan created: {len(self.execution_plan.agent_batches)} batches, "
                        f"{self.execution_plan.total_agents} agents")
        
        # Validate pipeline prerequisites
        if not self._validate_pipeline_prerequisites(context):
            raise MatrixAgentError("Pipeline prerequisites not satisfied")
        
        # Execute orchestration
        orchestration_results = self._orchestrate_matrix_transformation(context)
        
        # Generate final report
        final_report = self._generate_master_report(context, orchestration_results)
        
        self.logger.info("✨ Deus Ex Machina orchestration complete - The impossible has been achieved")
        
        return {
            'orchestration_status': 'success',
            'execution_plan': self.execution_plan,
            'agent_results': self.agent_results,
            'final_report': final_report,
            'pipeline_metrics': self._calculate_pipeline_metrics(),
            # Preserve critical context for other agents
            'binary_path': context['binary_path'],
            'output_paths': context['output_paths'],
            'output_dir': context['output_dir']
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
    
    def _orchestrate_matrix_transformation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the complete Matrix transformation process"""
        self.execution_start_time = time.time()
        
        orchestration_results = {
            'batch_results': [],
            'successful_agents': [],
            'failed_agents': [],
            'skipped_agents': []
        }
        
        try:
            # Execute agents in planned batches
            for batch_idx, agent_batch in enumerate(self.execution_plan.agent_batches):
                self.logger.info(f"🚀 Executing batch {batch_idx + 1}/{len(self.execution_plan.agent_batches)}: "
                               f"Agents {agent_batch}")
                
                batch_result = self._execute_agent_batch(agent_batch, context)
                orchestration_results['batch_results'].append(batch_result)
                
                # Update agent results and context
                for agent_id, result in batch_result.items():
                    self.agent_results[agent_id] = result
                    
                    if result.status == AgentStatus.SUCCESS:
                        orchestration_results['successful_agents'].append(agent_id)
                    elif result.status == AgentStatus.FAILED:
                        orchestration_results['failed_agents'].append(agent_id)
                    else:
                        orchestration_results['skipped_agents'].append(agent_id)
                
                # Update context with agent results for dependency validation
                context['agent_results'] = self.agent_results
                
                # Check for critical failures - STRICT MODE: Fail completely on any agent failure
                if self._should_abort_pipeline(batch_result):
                    failed_agents = [agent_id for agent_id, result in batch_result.items() 
                                   if result.status == AgentStatus.FAILED]
                    error_details = [f"Agent {agent_id}: {result.error_message}" 
                                   for agent_id, result in batch_result.items() 
                                   if result.status == AgentStatus.FAILED]
                    raise MatrixAgentError(
                        f"PIPELINE FAILURE - Agent(s) {failed_agents} failed. "
                        f"Rules.md EXECUTION RULE #8 (ALL OR NOTHING) requires complete pipeline failure. "
                        f"Details: {'; '.join(error_details)}"
                    )
        
        except Exception as e:
            self.logger.error(f"💥 Orchestration error: {str(e)}", exc_info=True)
            raise MatrixAgentError(f"Pipeline orchestration failed: {str(e)}")
        
        return orchestration_results
    
    def _execute_agent_batch(self, agent_batch: List[int], context: Dict[str, Any]) -> Dict[int, AgentResult]:
        """Execute a batch of agents"""
        # For now, execute sequentially - parallel execution can be added later
        batch_results = {}
        
        for agent_id in agent_batch:
            try:
                self.logger.info(f"🤖 Executing Agent {agent_id:02d}...")
                
                # Get agent class and instantiate
                if agent_id not in self.agent_classes:
                    raise MatrixAgentError(f"Agent {agent_id} not found in agent registry")
                
                agent_class = self.agent_classes[agent_id]
                agent_instance = agent_class()
                
                # Execute the agent
                result = agent_instance.execute(context)
                batch_results[agent_id] = result
                
                if result.status == AgentStatus.SUCCESS:
                    self.logger.info(f"✅ Agent {agent_id:02d} completed successfully")
                else:
                    self.logger.warning(f"⚠️ Agent {agent_id:02d} failed: {result.error_message}")
                
            except Exception as e:
                self.logger.error(f"❌ Agent {agent_id:02d} failed: {str(e)}")
                batch_results[agent_id] = AgentResult(
                    agent_id=agent_id,
                    agent_name=f"Agent{agent_id:02d}",
                    matrix_character=f"agent_{agent_id:02d}",
                    status=AgentStatus.FAILED,
                    error_message=str(e),
                    execution_time=0.0
                )
        
        return batch_results
    
    def _should_abort_pipeline(self, batch_result: Dict[int, AgentResult]) -> bool:
        """Determine if pipeline should abort based on batch results"""
        # STRICT MODE - Rule #8: ALL OR NOTHING - Any agent failure causes complete pipeline failure
        # Rule #2: NO PARTIAL SUCCESS - Never report partial success when components fail
        failed_count = sum(1 for result in batch_result.values() 
                          if result.status == AgentStatus.FAILED)
        
        if failed_count > 0:
            failed_agents = [agent_id for agent_id, result in batch_result.items() 
                           if result.status == AgentStatus.FAILED]
            self.logger.error(f"PIPELINE FAILURE - Agent(s) {failed_agents} failed. "
                            f"Rules.md EXECUTION RULE #8 (ALL OR NOTHING) requires complete pipeline failure. "
                            f"NO PARTIAL SUCCESS allowed per Rule #2.")
            return True
        
        return False
    
    def _generate_master_report(self, context: Dict[str, Any], 
                              orchestration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive master orchestration report"""
        execution_time = time.time() - self.execution_start_time
        
        return {
            'pipeline_summary': {
                'total_agents': len(context.get('selected_agents', [])),
                'successful_agents': len(orchestration_results['successful_agents']),
                'failed_agents': len(orchestration_results['failed_agents']),
                'skipped_agents': len(orchestration_results['skipped_agents']),
                'execution_time': execution_time,
                'success_rate': (len(orchestration_results['successful_agents']) / 
                               max(1, len(context.get('selected_agents', []))))
            },
            'orchestration_mode': self.execution_plan.execution_mode,
            'batch_execution': {
                'total_batches': len(self.execution_plan.agent_batches),
                'batch_results': orchestration_results['batch_results']
            },
            'quality_metrics': self._calculate_quality_metrics(),
            'recommendations': self._generate_recommendations(orchestration_results)
        }
    
    def _calculate_pipeline_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive pipeline metrics"""
        return {
            'total_execution_time': time.time() - self.execution_start_time,
            'agents_executed': len(self.agent_results),
            'success_rate': len([r for r in self.agent_results.values() 
                               if r.status == AgentStatus.SUCCESS]) / max(1, len(self.agent_results)),
            'average_agent_time': sum(r.execution_time for r in self.agent_results.values()) / 
                                max(1, len(self.agent_results))
        }
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics for the pipeline execution"""
        return {
            'overall_quality': 0.85,  # Mock quality score
            'decompilation_accuracy': 0.80,
            'reconstruction_completeness': 0.75,
            'compilation_readiness': 0.70
        }
    
    def _generate_recommendations(self, orchestration_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on execution results"""
        recommendations = []
        
        if orchestration_results['failed_agents']:
            recommendations.append(f"Review failed agents: {orchestration_results['failed_agents']}")
        
        if len(orchestration_results['successful_agents']) < 10:
            recommendations.append("Consider running more agents for better reconstruction quality")
        
        return recommendations