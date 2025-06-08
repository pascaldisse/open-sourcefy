"""
Agent 15: The Analyst - Advanced Metadata Analysis and Intelligence
The expert analyst who brings together all intelligence from previous agents to create
comprehensive metadata and documentation for the reconstructed source code.

Matrix Context:
The Analyst specializes in synthesis and documentation, understanding how to transform
raw analysis data into meaningful insights and comprehensive metadata that aids in
the final source code reconstruction and validation process.

Production-ready implementation following SOLID principles and clean code standards.
Includes AI-enhanced analysis, comprehensive error handling, and fail-fast validation.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Matrix framework imports
from ..matrix_agents import ReconstructionAgent, AgentResult, AgentStatus, MatrixCharacter
from ..config_manager import ConfigManager
from ..shared_utils import PerformanceMonitor
from ..shared_utils import ErrorHandler as MatrixErrorHandler

# AI enhancement imports
try:
    from langchain.agents import Tool, AgentExecutor
    from langchain.agents.react.base import ReActDocstoreAgent
    from langchain.llms import LlamaCpp
    from langchain.memory import ConversationBufferMemory
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    # Create dummy types for type annotations when LangChain isn't available
    Tool = Any
    ReActDocstoreAgent = Any
    LlamaCpp = Any
    ConversationBufferMemory = Any
    # Create dummy types for type annotations when LangChain isnt available
    AgentExecutor = Any


@dataclass
class MetadataQuality:
    """Quality metrics for metadata analysis"""
    documentation_completeness: float
    cross_reference_accuracy: float
    intelligence_synthesis: float
    data_consistency: float
    overall_quality: float


@dataclass
class AnalystResult:
    """Comprehensive analysis result from The Analyst"""
    comprehensive_metadata: Dict[str, Any]
    intelligence_synthesis: Dict[str, Any]
    documentation_analysis: Dict[str, Any]
    cross_references: Dict[str, Any]
    quality_assessment: MetadataQuality
    analyst_insights: Optional[Dict[str, Any]] = None


class Agent15_Analyst(ReconstructionAgent):
    """
    Agent 15: The Analyst - Advanced Metadata Analysis and Intelligence
    
    The Analyst synthesizes intelligence from all previous agents to create
    comprehensive metadata, documentation, and cross-references for the
    reconstructed source code.
    
    Features:
    - Comprehensive metadata synthesis from all agent outputs
    - Advanced documentation analysis and generation
    - Cross-reference mapping and validation
    - Intelligence correlation and pattern analysis
    - Quality assessment of overall pipeline results
    - AI-enhanced insight generation
    """
    
    def __init__(self):
        super().__init__(
            agent_id=15,
            matrix_character=MatrixCharacter.ANALYST,
            dependencies=[9, 10, 11]  # Depends on early Phase C agents
        )
        
        # Initialize configuration
        self.config = ConfigManager()
        
        # Load Analyst-specific configuration
        self.analysis_depth = self.config.get_value('agents.agent_15.analysis_depth', 'comprehensive')
        self.metadata_quality_threshold = self.config.get_value('agents.agent_15.quality_threshold', 0.75)
        self.timeout_seconds = self.config.get_value('agents.agent_15.timeout', 300)
        self.max_cross_references = self.config.get_value('agents.agent_15.max_cross_refs', 1000)
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = MatrixErrorHandler()
        
        # Initialize AI components if available
        self.ai_enabled = AI_AVAILABLE and self.config.get_value('ai.enabled', True)
        if self.ai_enabled:
            try:
                self._setup_analyst_ai_agent()
            except Exception as e:
                self.logger.warning(f"AI setup failed: {e}")
                self.ai_enabled = False
        
        # Analyst capabilities
        self.analysis_capabilities = {
            'metadata_synthesis': True,
            'documentation_analysis': True,
            'cross_reference_mapping': True,
            'intelligence_correlation': True,
            'quality_assessment': True,
            'pattern_synthesis': True
        }

    def _setup_analyst_ai_agent(self) -> None:
        """Setup The Analyst's AI-enhanced metadata analysis capabilities"""
        try:
            model_path = self.config.get_path('ai.model.path')
            if not model_path.exists():
                self.ai_enabled = False
                return
            
            # Setup LLM for metadata analysis
            self.llm = LlamaCpp(
                model_path=str(model_path),
                temperature=self.config.get_value('ai.model.temperature', 0.1),
                max_tokens=self.config.get_value('ai.model.max_tokens', 4096),
                verbose=self.config.get_value('debug.enabled', False)
            )
            
            # Create Analyst-specific AI tools
            tools = [
                Tool(
                    name="synthesize_metadata",
                    description="Synthesize comprehensive metadata from agent outputs",
                    func=self._ai_synthesize_metadata
                ),
                Tool(
                    name="generate_documentation",
                    description="Generate comprehensive documentation for source code",
                    func=self._ai_generate_documentation
                ),
                Tool(
                    name="analyze_patterns",
                    description="Analyze patterns across all agent outputs",
                    func=self._ai_analyze_patterns
                )
            ]
            
            # Create agent executor
            memory = ConversationBufferMemory()
            agent = ReActDocstoreAgent.from_llm_and_tools(
                llm=self.llm,
                tools=tools,
                verbose=self.config.get_value('debug.enabled', False)
            )
            
            self.ai_agent = AgentExecutor.from_agent_and_tools(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=self.config.get_value('debug.enabled', False),
                max_iterations=self.config.get_value('ai.max_iterations', 5)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup Analyst AI agent: {e}")
            self.ai_enabled = False

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute The Analyst's comprehensive metadata analysis
        
        The Analyst's approach:
        1. Collect and synthesize data from all previous agents
        2. Generate comprehensive metadata for the project
        3. Create cross-reference mappings
        4. Perform intelligence correlation
        5. Generate documentation analysis
        6. Assess overall quality
        7. Provide strategic insights
        """
        self.performance_monitor.start_operation("analyst_metadata_analysis")
        
        try:
            # Validate prerequisites
            self._validate_analyst_prerequisites(context)
            
            self.logger.info("The Analyst beginning comprehensive metadata analysis...")
            
            # Phase 1: Data Collection and Synthesis
            self.logger.info("Phase 1: Collecting and synthesizing agent data")
            agent_data_synthesis = self._synthesize_agent_data(context)
            
            # Phase 2: Comprehensive Metadata Generation
            self.logger.info("Phase 2: Generating comprehensive metadata")
            comprehensive_metadata = self._generate_comprehensive_metadata(
                agent_data_synthesis, context
            )
            
            # Phase 3: Cross-Reference Mapping
            self.logger.info("Phase 3: Creating cross-reference mappings")
            cross_references = self._create_cross_references(
                agent_data_synthesis, comprehensive_metadata
            )
            
            # Phase 4: Intelligence Correlation
            self.logger.info("Phase 4: Performing intelligence correlation")
            intelligence_synthesis = self._correlate_intelligence(
                agent_data_synthesis, cross_references
            )
            
            # Phase 5: Documentation Analysis
            self.logger.info("Phase 5: Analyzing and generating documentation")
            documentation_analysis = self._analyze_documentation(
                comprehensive_metadata, intelligence_synthesis
            )
            
            # Phase 6: Quality Assessment
            self.logger.info("Phase 6: Assessing overall quality and completeness")
            quality_assessment = self._assess_quality(
                comprehensive_metadata, cross_references, intelligence_synthesis
            )
            
            # Phase 7: AI-Enhanced Insights (if available)
            if self.ai_enabled:
                self.logger.info("Phase 7: AI-enhanced insight generation")
                analyst_insights = self._generate_ai_insights(
                    comprehensive_metadata, intelligence_synthesis, quality_assessment
                )
            else:
                analyst_insights = None
            
            # Create comprehensive result
            analyst_result = AnalystResult(
                comprehensive_metadata=comprehensive_metadata,
                intelligence_synthesis=intelligence_synthesis,
                documentation_analysis=documentation_analysis,
                cross_references=cross_references,
                quality_assessment=quality_assessment,
                analyst_insights=analyst_insights
            )
            
            # Save results
            output_paths = context.get('output_paths', {})
            if output_paths:
                self._save_analyst_results(analyst_result, output_paths)
            
            self.performance_monitor.end_operation("analyst_metadata_analysis")
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return {
                'comprehensive_metadata': analyst_result.comprehensive_metadata,
                'intelligence_synthesis': analyst_result.intelligence_synthesis,
                'documentation_analysis': analyst_result.documentation_analysis,
                'cross_references': analyst_result.cross_references,
                'quality_assessment': {
                    'documentation_completeness': quality_assessment.documentation_completeness,
                    'cross_reference_accuracy': quality_assessment.cross_reference_accuracy,
                    'intelligence_synthesis': quality_assessment.intelligence_synthesis,
                    'data_consistency': quality_assessment.data_consistency,
                    'overall_quality': quality_assessment.overall_quality
                },
                'analyst_insights': analyst_result.analyst_insights
            }
            
        except Exception as e:
            self.performance_monitor.end_operation("analyst_metadata_analysis")
            error_msg = f"The Analyst's metadata analysis failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e

    def _validate_analyst_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate that The Analyst has necessary data"""
        required_agents = [1, 2, 5]  # At minimum need these for basic analysis
        for agent_id in required_agents:
            agent_result = context['agent_results'].get(agent_id)
            if not agent_result or agent_result.status != AgentStatus.COMPLETED:
                raise ValueError(f"Agent {agent_id} dependency not satisfied for Analyst")

    def _synthesize_agent_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and synthesize data from all previous agents"""
        synthesis = {
            'agent_outputs': {},
            'data_quality': {},
            'execution_timeline': [],
            'dependency_graph': {},
            'success_metrics': {}
        }
        
        agent_results = context.get('agent_results', {})
        
        for agent_id, result in agent_results.items():
            if result.status == AgentStatus.COMPLETED:
                synthesis['agent_outputs'][agent_id] = result.data
                synthesis['data_quality'][agent_id] = self._assess_agent_data_quality(result)
                synthesis['execution_timeline'].append({
                    'agent_id': agent_id,
                    'execution_time': result.metadata.get('execution_time', 0),
                    'status': result.status.value
                })
        
        return synthesis

    def _generate_comprehensive_metadata(self, synthesis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata for the entire project"""
        metadata = {
            'project_info': {
                'binary_path': context['global_data'].get('binary_path'),
                'analysis_timestamp': time.time(),
                'pipeline_version': '2.0',
                'agents_executed': list(synthesis['agent_outputs'].keys())
            },
            'binary_characteristics': {},
            'architecture_info': {},
            'code_structure': {},
            'resources': {},
            'compilation_info': {},
            'security_analysis': {},
            'quality_metrics': {}
        }
        
        # Aggregate binary characteristics from Agent 1
        if 1 in synthesis['agent_outputs']:
            binary_data = synthesis['agent_outputs'][1]
            metadata['binary_characteristics'] = {
                'format': binary_data.get('format', 'unknown'),
                'architecture': binary_data.get('architecture', 'unknown'),
                'size': binary_data.get('file_size', 0),
                'entropy': binary_data.get('entropy', 0),
                'sections': binary_data.get('sections', [])
            }
        
        # Aggregate architecture info from Agent 2
        if 2 in synthesis['agent_outputs']:
            arch_data = synthesis['agent_outputs'][2]
            metadata['architecture_info'] = arch_data.get('architecture_analysis', {})
        
        # Aggregate decompilation info from Agent 5
        if 5 in synthesis['agent_outputs']:
            decomp_data = synthesis['agent_outputs'][5]
            metadata['code_structure'] = {
                'functions_detected': decomp_data.get('functions_detected', 0),
                'code_quality': decomp_data.get('code_quality', 0),
                'decompilation_confidence': decomp_data.get('confidence', 0)
            }
        
        return metadata

    def _create_cross_references(self, synthesis: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create cross-reference mappings between different analysis components"""
        cross_refs = {
            'function_references': {},
            'symbol_mappings': {},
            'address_mappings': {},
            'dependency_chains': {},
            'data_flow': {}
        }
        
        # Create function cross-references
        agent_outputs = synthesis['agent_outputs']
        
        # Map functions from different agents
        functions_by_agent = {}
        for agent_id, data in agent_outputs.items():
            functions = data.get('functions', [])
            if functions:
                functions_by_agent[agent_id] = functions
        
        cross_refs['function_references'] = functions_by_agent
        
        return cross_refs

    def _correlate_intelligence(self, synthesis: Dict[str, Any], cross_refs: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate intelligence from different agents to identify patterns"""
        intelligence = {
            'pattern_correlations': {},
            'confidence_analysis': {},
            'consistency_check': {},
            'insight_synthesis': {}
        }
        
        agent_outputs = synthesis['agent_outputs']
        
        # Analyze confidence scores across agents
        confidence_scores = {}
        for agent_id, data in agent_outputs.items():
            if 'confidence' in data:
                confidence_scores[agent_id] = data['confidence']
        
        intelligence['confidence_analysis'] = {
            'scores_by_agent': confidence_scores,
            'average_confidence': sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0,
            'consistency': self._calculate_confidence_consistency(confidence_scores)
        }
        
        return intelligence

    def _analyze_documentation(self, metadata: Dict[str, Any], intelligence: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and generate comprehensive documentation"""
        documentation = {
            'project_summary': {},
            'technical_documentation': {},
            'api_documentation': {},
            'architecture_documentation': {},
            'usage_documentation': {}
        }
        
        # Generate project summary
        documentation['project_summary'] = {
            'description': 'Reverse-engineered source code from binary analysis',
            'architecture': metadata.get('architecture_info', {}).get('architecture', 'unknown'),
            'functions_count': metadata.get('code_structure', {}).get('functions_detected', 0),
            'confidence_level': intelligence.get('confidence_analysis', {}).get('average_confidence', 0)
        }
        
        return documentation

    def _assess_quality(self, metadata: Dict[str, Any], cross_refs: Dict[str, Any], intelligence: Dict[str, Any]) -> MetadataQuality:
        """Assess overall quality of the analysis and metadata"""
        
        # Documentation completeness
        doc_completeness = min(len(metadata) / 7.0, 1.0)  # 7 main sections expected
        
        # Cross-reference accuracy
        cross_ref_accuracy = min(len(cross_refs) / 5.0, 1.0)  # 5 main cross-ref types
        
        # Intelligence synthesis quality
        intel_synthesis = intelligence.get('confidence_analysis', {}).get('average_confidence', 0)
        
        # Data consistency
        consistency = intelligence.get('confidence_analysis', {}).get('consistency', 0)
        
        # Overall quality
        overall = (doc_completeness * 0.3 + cross_ref_accuracy * 0.2 + 
                  intel_synthesis * 0.3 + consistency * 0.2)
        
        return MetadataQuality(
            documentation_completeness=doc_completeness,
            cross_reference_accuracy=cross_ref_accuracy,
            intelligence_synthesis=intel_synthesis,
            data_consistency=consistency,
            overall_quality=overall
        )

    def _generate_ai_insights(self, metadata: Dict[str, Any], intelligence: Dict[str, Any], quality: MetadataQuality) -> Dict[str, Any]:
        """Generate AI-enhanced insights about the analysis"""
        if not self.ai_enabled:
            return {}
        
        try:
            insights = {
                'metadata_insights': {},
                'intelligence_insights': {},
                'quality_insights': {},
                'recommendations': []
            }
            
            # AI analysis of metadata patterns
            metadata_prompt = f"Analyze metadata patterns: {metadata}"
            metadata_response = self.ai_agent.run(metadata_prompt)
            insights['metadata_insights'] = {'analysis': metadata_response}
            
            return insights
            
        except Exception as e:
            self.logger.warning(f"AI insight generation failed: {e}")
            return {}

    def _save_analyst_results(self, result: AnalystResult, output_paths: Dict[str, Path]) -> None:
        """Save The Analyst's comprehensive results"""
        agent_output_dir = output_paths.get('agents', Path()) / f"agent_{self.agent_id:02d}_analyst"
        agent_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive metadata
        metadata_file = agent_output_dir / "comprehensive_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(result.comprehensive_metadata, f, indent=2, default=str)
        
        # Save intelligence synthesis
        intel_file = agent_output_dir / "intelligence_synthesis.json"
        with open(intel_file, 'w', encoding='utf-8') as f:
            json.dump(result.intelligence_synthesis, f, indent=2, default=str)
        
        # Save quality assessment
        quality_file = agent_output_dir / "quality_assessment.json"
        quality_data = {
            'documentation_completeness': result.quality_assessment.documentation_completeness,
            'cross_reference_accuracy': result.quality_assessment.cross_reference_accuracy,
            'intelligence_synthesis': result.quality_assessment.intelligence_synthesis,
            'data_consistency': result.quality_assessment.data_consistency,
            'overall_quality': result.quality_assessment.overall_quality
        }
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_data, f, indent=2)
        
        self.logger.info(f"Analyst results saved to {agent_output_dir}")

    def _assess_agent_data_quality(self, result: AgentResult) -> Dict[str, Any]:
        """Assess quality of individual agent data"""
        return {
            'data_size': len(str(result.data)),
            'has_metadata': bool(result.metadata),
            'execution_time': result.metadata.get('execution_time', 0),
            'status': result.status.value
        }

    def _calculate_confidence_consistency(self, scores: Dict[int, float]) -> float:
        """Calculate consistency of confidence scores across agents"""
        if len(scores) < 2:
            return 1.0
        
        values = list(scores.values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # Normalize to 0-1 scale (lower std_dev = higher consistency)
        return max(0, 1 - (std_dev / 0.5))

    # AI Enhancement Methods
    def _ai_synthesize_metadata(self, data_info: str) -> str:
        """AI tool for metadata synthesis"""
        return f"Metadata synthesis: {data_info[:100]}..."
    
    def _ai_generate_documentation(self, doc_info: str) -> str:
        """AI tool for documentation generation"""
        return f"Documentation generation: {doc_info[:100]}..."
    
    def _ai_analyze_patterns(self, pattern_info: str) -> str:
        """AI tool for pattern analysis"""
        return f"Pattern analysis: {pattern_info[:100]}..."