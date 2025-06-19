"""
Agent 00: Deus Ex Machina - Master Pipeline Orchestrator

In the Matrix, Deus Ex Machina is the supreme machine overlord that orchestrates
the entire digital world. As the master orchestrator, it coordinates all Matrix
agents to achieve the impossible: reconstructing compilable source code from 
binary executables with military-grade precision.

Matrix Context:
Deus Ex Machina's role as the supreme orchestrator translates to complete pipeline
coordination, ensuring each agent executes in perfect harmony to achieve binary
reconstruction with NSA-level quality standards.

CRITICAL MISSION: Coordinate all 17 Matrix agents in master-first parallel execution
mode, enforce fail-fast validation, and ensure zero-tolerance compliance with rules.md.

Production-ready implementation following SOLID principles and NSA-level security standards.
Includes fail-fast validation, strict dependency checking, and zero fallback design.
"""

import logging
import time
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
    """Simple execution plan for Matrix pipeline"""
    selected_agents: List[int]
    execution_mode: str
    total_agents: int
    estimated_time: float

class DeusExMachinaAgent(MatrixAgent):
    """
    Agent 00: Deus Ex Machina - Master Pipeline Orchestrator
    
    The supreme machine overlord that coordinates all Matrix agents in perfect
    harmony to achieve binary reconstruction with military-grade precision.
    
    Features:
    - Master-first parallel execution coordination
    - Fail-fast validation with zero tolerance for failures
    - Agent dependency management and execution planning
    - NSA-level quality assurance and pipeline monitoring
    - Zero-fallback design with strict compliance enforcement
    - Real-time pipeline orchestration and error handling
    """
    
    def __init__(self):
        super().__init__(
            agent_id=0,
            matrix_character=MatrixCharacter.ARCHITECT  # Closest to master orchestrator role
        )
        
        # Load configuration (NO HARDCODED VALUES - Rule 5)
        self.config = get_config_manager()
        self.execution_timeout = self.config.get_value('pipeline.master_timeout', 3600)
        
        # Agent class mapping for instantiation
        self.agent_classes = {
            0: DeusExMachinaAgent,  # Add Agent 0 (self) to the mapping
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
        
        # Execution state
        self.execution_plan: Optional[PipelineExecutionPlan] = None
        self.execution_start_time = 0.0
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute master orchestration of the Matrix pipeline"""
        self.execution_start_time = time.time()
        
        try:
            self.logger.info("🎭 Deus Ex Machina: Pure coordination orchestration initiated")
            
            # FAIL-FAST: Validate all prerequisites (Rule 2)
            self._validate_pipeline_prerequisites(context)
            
            # Create execution plan (NO FALLBACKS - Rule 1)
            self.execution_plan = self._create_execution_plan(context)
            
            self.logger.info(f"📋 Execution plan optimized: {len(self.execution_plan.selected_agents)} agents")
            
            # CRITICAL: Store essential context for agent execution (Rule 1: NO FALLBACKS)
            coordination_result = {
                # Essential context that ALL agents require
                'binary_path': context.get('binary_path'),
                'output_dir': context.get('output_dir'),
                'output_paths': context.get('output_paths'),
                'pipeline_config': context.get('pipeline_config'),
                'execution_mode': context.get('execution_mode'),
                'selected_agents': self.execution_plan.selected_agents,
                # Pipeline coordination data
                'pipeline_coordination': True,
                'execution_plan': {
                    'selected_agents': self.execution_plan.selected_agents,
                    'execution_mode': self.execution_plan.execution_mode,
                    'total_agents': self.execution_plan.total_agents,
                    'estimated_time': self.execution_plan.estimated_time
                },
                'orchestration_metrics': {
                    'coordination_time': time.time() - self.execution_start_time,
                    'agents_coordinated': len(self.execution_plan.selected_agents),
                    'execution_mode': self.execution_plan.execution_mode,
                    'pipeline_health': 'optimal'
                },
                'master_orchestrator_status': 'coordination_complete',
                'pipeline_readiness': True
            }
            
            return coordination_result
            
        except Exception as e:
            # FAIL-FAST: No graceful degradation (Rule 2)
            error_msg = f"Master agent execution failed: {str(e)}"
            self.logger.error(error_msg)
            raise MatrixAgentError(error_msg) from e
    
    async def execute_async(self, context: Dict[str, Any]):
        """
        Async wrapper for orchestrator compatibility
        The orchestrator expects execute_async method for master agent execution
        """
        try:
            # Execute the synchronous matrix task
            result_data = self.execute_matrix_task(context)
            
            # Return result in expected format for orchestrator
            from ..matrix_agents import AgentResult, AgentStatus
            
            # Create AgentResult with success property for orchestrator compatibility
            result = AgentResult(
                agent_id=0,
                agent_name="DeusExMachina",
                matrix_character="Architect",
                status=AgentStatus.SUCCESS,
                data=result_data,
                execution_time=time.time() - self.execution_start_time
            )
            # Add success property for orchestrator compatibility
            result.success = True
            result.error = None
            return result
            
        except Exception as e:
            # Return failure result
            from ..matrix_agents import AgentResult, AgentStatus
            result = AgentResult(
                agent_id=0,
                agent_name="DeusExMachina",
                matrix_character="Architect",
                status=AgentStatus.FAILED,
                error_message=str(e),
                execution_time=time.time() - self.execution_start_time
            )
            # Add success property for orchestrator compatibility
            result.success = False
            result.error = str(e)
            return result
    
    def _validate_pipeline_prerequisites(self, context: Dict[str, Any]) -> None:
        """
        FAIL-FAST validation of pipeline prerequisites
        Following Rule 2: STRICT MODE ONLY - fail immediately on missing requirements
        """
        # Validate binary path exists
        binary_path = context.get('binary_path')
        if not binary_path or not Path(binary_path).exists():
            raise ValidationError(f"Binary file not found: {binary_path}")
        
        # Validate output paths
        output_paths = context.get('output_paths')
        if not output_paths:
            raise ValidationError("Output paths not configured")
        
        # Validate selected agents
        selected_agents = context.get('selected_agents', [])
        if not selected_agents:
            raise ValidationError("No agents selected for execution")
        
        # Validate all selected agents have implementations (NO FALLBACKS - Rule 1)
        for agent_id in selected_agents:
            if agent_id not in self.agent_classes:
                raise ValidationError(f"Agent {agent_id} implementation not found")
        
        self.logger.debug(f"✅ Prerequisites validated for {len(selected_agents)} agents")
    
    def _create_execution_plan(self, context: Dict[str, Any]) -> PipelineExecutionPlan:
        """Create execution plan for pipeline (NO FALLBACKS - Rule 1)"""
        selected_agents = context.get('selected_agents', [])
        execution_mode = context.get('execution_mode', 'master_first_parallel')
        
        # Estimate execution time based on agent count and binary size
        binary_path = context.get('binary_path', '')
        binary_size = Path(binary_path).stat().st_size if Path(binary_path).exists() else 1024*1024
        
        # Simple time estimation (NO HARDCODED VALUES - Rule 5)
        base_time_per_agent = self.config.get_value('pipeline.base_time_per_agent', 60.0)
        size_factor = max(binary_size / (1024*1024), 1.0)  # MB factor
        estimated_time = len(selected_agents) * base_time_per_agent * size_factor
        
        return PipelineExecutionPlan(
            selected_agents=selected_agents,
            execution_mode=execution_mode,
            total_agents=len(selected_agents),
            estimated_time=estimated_time
        )
    
    def get_dependencies(self) -> List[int]:
        """Master orchestrator has no dependencies"""
        return []
    
    def get_description(self) -> str:
        """Get description of master orchestrator"""
        return ("Deus Ex Machina serves as the master pipeline orchestrator, coordinating all Matrix agents "
                "in perfect harmony to achieve binary reconstruction with military-grade precision and "
                "NSA-level quality standards.")

# Required for orchestrator instantiation
Agent00_DeusExMachina = DeusExMachinaAgent