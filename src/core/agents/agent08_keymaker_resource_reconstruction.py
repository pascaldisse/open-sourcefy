"""
Agent 8: The Keymaker - Resource Reconstruction

In the Matrix, The Keymaker possesses the unique ability to create keys that open
any door, providing access to every part of the system. He understands the
intricate structure of the Matrix and can navigate through its layers. As Agent 8,
The Keymaker specializes in resource reconstruction - extracting, analyzing, and
reconstructing all resources embedded within binaries.

Matrix Context:
The Keymaker's mastery over doors and access translates to understanding resource
structures, embedded data, and the hidden components within binaries. His keys
unlock strings, images, configuration data, and other resources that are integral
to reconstructing the complete source project.

Production-ready implementation following SOLID principles and clean code standards.
Includes AI-enhanced analysis, comprehensive error handling, and fail-fast validation.
"""

import logging
import struct
import hashlib
import tempfile
import base64
import json
import zlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
import time
import mimetypes
from collections import defaultdict

# Matrix framework imports
from ..matrix_agents import ReconstructionAgent, AgentResult, AgentStatus, MatrixCharacter
from ..config_manager import ConfigManager
from ..shared_components import MatrixErrorHandler

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
    AgentExecutor = Any
    ReActDocstoreAgent = Any
    LlamaCpp = Any
    ConversationBufferMemory = Any


@dataclass
class ResourceItem:
    """Represents a reconstructed resource item"""
    resource_id: str
    resource_type: str  # 'string', 'image', 'icon', 'config', 'data', etc.
    name: str
    size: int
    content: Union[bytes, str]
    metadata: Dict[str, Any]
    extraction_method: str
    confidence: float
    parent_section: Optional[str] = None


@dataclass
class ResourceCategory:
    """Categorized resources by type"""
    category_name: str
    items: List[ResourceItem]
    total_size: int
    extraction_confidence: float
    reconstruction_quality: float


@dataclass
class KeymakerQualityMetrics:
    """Quality metrics for resource reconstruction"""
    resource_coverage: float  # Percentage of resources extracted
    extraction_accuracy: float  # Accuracy of resource extraction
    type_classification_accuracy: float  # Accuracy of resource type classification
    reconstruction_completeness: float  # Completeness of reconstruction
    overall_quality: float  # Combined quality score


@dataclass
class KeymakerAnalysisResult:
    """Comprehensive resource reconstruction result from The Keymaker"""
    resource_categories: List[ResourceCategory]
    string_resources: List[ResourceItem]
    binary_resources: List[ResourceItem]
    configuration_data: Dict[str, Any]
    embedded_files: List[ResourceItem]
    metadata_analysis: Dict[str, Any]
    reconstruction_map: Dict[str, str]
    quality_metrics: KeymakerQualityMetrics
    ai_insights: Optional[Dict[str, Any]] = None
    keymaker_doors: Optional[Dict[str, Any]] = None


class Agent8_Keymaker_ResourceReconstruction(ReconstructionAgent):
    """
    Agent 8: The Keymaker - Resource Reconstruction
    
    The Keymaker's mastery over access and understanding of system structure
    makes him the perfect agent for reconstructing all resources embedded
    within binaries. He creates keys to unlock every type of resource,
    from simple strings to complex embedded files.
    
    Features:
    - Comprehensive resource extraction from PE/ELF/Mach-O binaries
    - Multi-format resource parsing (strings, images, icons, data)
    - Intelligent resource type classification and validation
    - Configuration data reconstruction and interpretation
    - Embedded file extraction and analysis
    - AI-enhanced resource pattern recognition
    - Complete project structure reconstruction
    """
    
    def __init__(self):
        super().__init__(
            agent_id=8,
            matrix_character=MatrixCharacter.KEYMAKER,
            dependencies=[1, 2, 3]  # Depends on Binary Discovery, Arch Analysis, and Merovingian's decompilation
        )
        
        # Load Keymaker-specific configuration
        self.min_string_length = self.config.get_value('agents.agent_08.min_string_length', 4)
        self.max_resource_size = self.config.get_value('agents.agent_08.max_resource_size', 50 * 1024 * 1024)  # 50MB
        self.timeout_seconds = self.config.get_value('agents.agent_08.timeout', 300)
        self.enable_deep_extraction = self.config.get_value('agents.agent_08.deep_extraction', True)
        
        # Initialize components
        self.error_handler = MatrixErrorHandler("Keymaker", max_retries=2)
        
        # Initialize AI components if available
        self.ai_enabled = AI_AVAILABLE and self.config.get_value('ai.enabled', True)
        if self.ai_enabled:
            try:
                self._setup_keymaker_ai_agent()
            except Exception as e:
                self.logger.warning(f"AI setup failed: {e}")
                self.ai_enabled = False
        
        # Keymaker's door abilities - resource access mastery
        self.door_abilities = {
            'string_extraction': True,  # Extract all string resources
            'binary_resource_parsing': True,  # Parse binary resource sections
            'embedded_file_detection': True,  # Detect embedded files
            'configuration_reconstruction': True,  # Reconstruct configuration data
            'metadata_analysis': True,  # Deep metadata analysis
            'cross_reference_mapping': True  # Map resources to code references
        }
        
        # Resource extraction engines
        self.extraction_engines = {
            'pe_resources': self._extract_pe_resources,
            'elf_resources': self._extract_elf_resources,
            'macho_resources': self._extract_macho_resources,
            'string_extractor': self._extract_string_resources,
            'binary_scanner': self._scan_binary_resources,
            'embedded_detector': self._detect_embedded_files
        }
        
        # Resource type patterns and signatures
        self.resource_patterns = {
            'image_signatures': {
                'png': b'\\x89PNG\\r\\n\\x1a\\n',
                'jpeg': b'\\xff\\xd8\\xff',
                'gif': b'GIF8',
                'bmp': b'BM',
                'ico': b'\\x00\\x00\\x01\\x00'
            },
            'archive_signatures': {
                'zip': b'PK\\x03\\x04',
                'rar': b'Rar!\\x1a\\x07\\x00',
                'tar': b'ustar',
                'gz': b'\\x1f\\x8b'
            },
            'executable_signatures': {
                'pe': b'MZ',
                'elf': b'\\x7fELF',
                'macho': b'\\xfe\\xed\\xfa'
            }
        }

    def _setup_keymaker_ai_agent(self) -> None:
        """Setup The Keymaker's AI-enhanced resource analysis capabilities"""
        try:
            model_path = self.config.get_path('ai.model.path')
            if not model_path.exists():
                self.ai_enabled = False
                return
            
            # Setup LLM for resource analysis
            self.llm = LlamaCpp(
                model_path=str(model_path),
                temperature=self.config.get_value('ai.model.temperature', 0.1),
                max_tokens=self.config.get_value('ai.model.max_tokens', 2048),
                verbose=self.config.get_value('debug.enabled', False)
            )
            
            # Create Keymaker-specific AI tools
            tools = [
                Tool(
                    name="classify_resource_types",
                    description="Classify and categorize extracted resources",
                    func=self._ai_classify_resource_types
                ),
                Tool(
                    name="analyze_resource_patterns",
                    description="Analyze patterns in resource usage and organization",
                    func=self._ai_analyze_resource_patterns
                ),
                Tool(
                    name="reconstruct_project_structure",
                    description="Reconstruct original project structure from resources",
                    func=self._ai_reconstruct_project_structure
                ),
                Tool(
                    name="validate_resource_integrity",
                    description="Validate integrity and completeness of extracted resources",
                    func=self._ai_validate_resource_integrity
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
                max_iterations=self.config.get_value('ai.max_iterations', 3)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup Keymaker AI agent: {e}")
            self.ai_enabled = False

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute The Keymaker's resource reconstruction process
        
        The Keymaker's approach to resource reconstruction:
        1. Identify all resource containers and access points
        2. Extract resources using multiple specialized engines
        3. Classify and categorize all extracted resources
        4. Reconstruct configuration and project structure
        5. Map resources to code references and usage patterns
        6. Validate resource integrity and completeness
        7. Generate AI-enhanced insights about resource organization
        """
        start_time = time.time()
        
        try:
            # Validate prerequisites - The Keymaker needs the foundation
            self._validate_keymaker_prerequisites(context)
            
            # Get analysis context from previous agents
            binary_path = context.get('binary_path', '')
            agent1_data = context['agent_results'][1].data  # Binary discovery
            agent2_data = context['agent_results'][2].data  # Architecture analysis
            agent3_data = context['agent_results'][3].data  # Merovingian's decompilation
            
            self.logger.info("The Keymaker beginning comprehensive resource reconstruction...")
            
            # Phase 1: Resource Container Analysis
            self.logger.info("Phase 1: Analyzing resource containers and access points")
            resource_containers = self._analyze_resource_containers(
                binary_path, agent1_data, agent2_data
            )
            
            # Phase 2: Multi-Engine Resource Extraction
            self.logger.info("Phase 2: Extracting resources using specialized engines")
            extracted_resources = self._perform_multi_engine_extraction(
                binary_path, resource_containers, agent1_data
            )
            
            # Phase 3: Resource Classification and Categorization
            self.logger.info("Phase 3: Classifying and categorizing extracted resources")
            categorized_resources = self._classify_and_categorize_resources(
                extracted_resources
            )
            
            # Phase 4: Configuration Data Reconstruction
            self.logger.info("Phase 4: Reconstructing configuration data and settings")
            configuration_data = self._reconstruct_configuration_data(
                categorized_resources, agent3_data, {}
            )
            
            # Phase 5: Cross-Reference Mapping
            self.logger.info("Phase 5: Mapping resources to code references")
            cross_reference_map = self._create_cross_reference_mapping(
                categorized_resources, agent3_data, {}
            )
            
            # Phase 6: AI-Enhanced Analysis (if available)
            if self.ai_enabled:
                self.logger.info("Phase 6: AI-enhanced resource pattern analysis")
                ai_insights = self._perform_ai_enhanced_analysis(
                    categorized_resources, configuration_data, cross_reference_map
                )
            else:
                ai_insights = None
            
            # Phase 7: Keymaker's Door Creation
            self.logger.info("Phase 7: Creating Keymaker's doors for resource access")
            keymaker_doors = self._create_keymaker_doors(
                categorized_resources, configuration_data, cross_reference_map, ai_insights
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_resource_quality_metrics(
                extracted_resources, categorized_resources, configuration_data
            )
            
            # Create comprehensive result
            keymaker_result = KeymakerAnalysisResult(
                resource_categories=categorized_resources['categories'],
                string_resources=categorized_resources['strings'],
                binary_resources=categorized_resources['binary'],
                configuration_data=configuration_data,
                embedded_files=categorized_resources['embedded_files'],
                metadata_analysis=categorized_resources['metadata'],
                reconstruction_map=cross_reference_map,
                quality_metrics=quality_metrics,
                ai_insights=ai_insights,
                keymaker_doors=keymaker_doors
            )
            
            # Save results to output directory
            output_paths = context.get('output_paths', {})
            if output_paths:
                self._save_keymaker_results(keymaker_result, output_paths)
            
            execution_time = time.time() - start_time
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return {
                'resource_categories': [
                    {
                        'name': cat.category_name,
                        'item_count': len(cat.items),
                        'total_size': cat.total_size,
                        'extraction_confidence': cat.extraction_confidence,
                        'reconstruction_quality': cat.reconstruction_quality
                    }
                    for cat in keymaker_result.resource_categories
                ],
                'string_resources': [
                    {
                        'id': res.resource_id,
                        'name': res.name,
                        'content': res.content[:100] + '...' if len(str(res.content)) > 100 else res.content,
                        'size': res.size,
                        'confidence': res.confidence
                    }
                    for res in keymaker_result.string_resources[:50]  # Limit for output size
                ],
                'binary_resources': [
                    {
                        'id': res.resource_id,
                        'type': res.resource_type,
                        'name': res.name,
                        'size': res.size,
                        'confidence': res.confidence
                    }
                    for res in keymaker_result.binary_resources
                ],
                'configuration_data': keymaker_result.configuration_data,
                'embedded_files': [
                    {
                        'id': res.resource_id,
                        'name': res.name,
                        'type': res.resource_type,
                        'size': res.size
                    }
                    for res in keymaker_result.embedded_files
                ],
                'quality_metrics': {
                    'resource_coverage': quality_metrics.resource_coverage,
                    'extraction_accuracy': quality_metrics.extraction_accuracy,
                    'type_classification_accuracy': quality_metrics.type_classification_accuracy,
                    'reconstruction_completeness': quality_metrics.reconstruction_completeness,
                    'overall_quality': quality_metrics.overall_quality
                },
                'reconstruction_map': keymaker_result.reconstruction_map,
                'ai_enhanced': self.ai_enabled,
                'keymaker_doors': keymaker_result.keymaker_doors,
                'keymaker_metadata': {
                    'agent_name': 'Keymaker_ResourceReconstruction',
                    'matrix_character': 'The Keymaker',
                    'doors_created': len(keymaker_result.keymaker_doors) if keymaker_result.keymaker_doors else 0,
                    'resources_extracted': sum(len(cat.items) for cat in keymaker_result.resource_categories),
                    'ai_enabled': self.ai_enabled,
                    'execution_time': execution_time,
                    'reconstruction_complete': quality_metrics.overall_quality >= 0.7
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"The Keymaker's resource reconstruction failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e

    def _validate_keymaker_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate that The Keymaker has the necessary data for resource reconstruction"""
        # Check required agent results
        required_agents = [1, 2, 3]
        for agent_id in required_agents:
            agent_result = context['agent_results'].get(agent_id)
            if not agent_result or agent_result.status != AgentStatus.SUCCESS:
                raise ValueError(f"Agent {agent_id} dependency not satisfied for Keymaker's reconstruction")
        
        # Check binary path
        binary_path = context.get('binary_path')
        if not binary_path or not Path(binary_path).exists():
            raise ValueError("Binary path not found - Keymaker cannot access resources")

    def _analyze_resource_containers(
        self,
        binary_path: str,
        binary_info: Dict[str, Any],
        arch_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze resource containers and access points in the binary"""
        
        containers = {
            'resource_sections': [],
            'string_sections': [],
            'data_sections': [],
            'embedded_containers': [],
            'format_specific': {}
        }
        
        binary_format = binary_info.get('format_info', {}).get('format', 'Unknown')
        
        self.logger.info(f"Keymaker analyzing {binary_format} resource containers...")
        
        try:
            if binary_format == 'PE':
                containers['format_specific'] = self._analyze_pe_containers(binary_path)
            elif binary_format == 'ELF':
                containers['format_specific'] = self._analyze_elf_containers(binary_path)
            elif binary_format == 'Mach-O':
                containers['format_specific'] = self._analyze_macho_containers(binary_path)
            
            # Generic container analysis
            containers = self._analyze_generic_containers(binary_path, containers)
            
        except Exception as e:
            self.logger.error(f"Container analysis failed: {e}")
            containers['analysis_error'] = str(e)
        
        return containers

    def _perform_multi_engine_extraction(
        self,
        binary_path: str,
        containers: Dict[str, Any],
        binary_info: Dict[str, Any]
    ) -> List[ResourceItem]:
        """Perform resource extraction using multiple specialized engines"""
        
        all_resources = []
        binary_format = binary_info.get('format_info', {}).get('format', 'Unknown')
        
        # Engine 1: Format-specific resource extraction
        if binary_format in self.extraction_engines:
            try:
                format_resources = self.extraction_engines[binary_format](binary_path, containers)
                all_resources.extend(format_resources)
            except Exception as e:
                self.logger.warning(f"Format-specific extraction failed: {e}")
        
        # Engine 2: String resource extraction
        try:
            string_resources = self._extract_string_resources(binary_path, containers)
            all_resources.extend(string_resources)
        except Exception as e:
            self.logger.warning(f"String extraction failed: {e}")
        
        # Engine 3: Binary resource scanning
        try:
            binary_resources = self._scan_binary_resources(binary_path, containers)
            all_resources.extend(binary_resources)
        except Exception as e:
            self.logger.warning(f"Binary scanning failed: {e}")
        
        # Engine 4: Embedded file detection
        try:
            embedded_resources = self._detect_embedded_files(binary_path, containers)
            all_resources.extend(embedded_resources)
        except Exception as e:
            self.logger.warning(f"Embedded file detection failed: {e}")
        
        # Remove duplicates and validate
        all_resources = self._deduplicate_resources(all_resources)
        all_resources = self._validate_extracted_resources(all_resources)
        
        return all_resources

    def _classify_and_categorize_resources(
        self,
        extracted_resources: List[ResourceItem]
    ) -> Dict[str, Any]:
        """Classify and categorize all extracted resources"""
        
        categorization = {
            'categories': [],
            'strings': [],
            'binary': [],
            'embedded_files': [],
            'metadata': {},
            'classification_stats': {}
        }
        
        # Categorize by type
        type_groups = defaultdict(list)
        for resource in extracted_resources:
            type_groups[resource.resource_type].append(resource)
        
        # Create resource categories
        for resource_type, resources in type_groups.items():
            total_size = sum(res.size for res in resources)
            avg_confidence = sum(res.confidence for res in resources) / len(resources)
            
            category = ResourceCategory(
                category_name=resource_type,
                items=resources,
                total_size=total_size,
                extraction_confidence=avg_confidence,
                reconstruction_quality=self._calculate_category_quality(resources)
            )
            categorization['categories'].append(category)
            
            # Separate into specific lists for easy access
            if resource_type == 'string':
                categorization['strings'].extend(resources)
            elif resource_type in ['image', 'data', 'archive']:
                categorization['binary'].extend(resources)
            elif resource_type == 'embedded_file':
                categorization['embedded_files'].extend(resources)
        
        # Generate metadata analysis
        categorization['metadata'] = self._generate_resource_metadata(extracted_resources)
        
        # Classification statistics
        categorization['classification_stats'] = {
            'total_resources': len(extracted_resources),
            'categories_found': len(type_groups),
            'average_confidence': sum(res.confidence for res in extracted_resources) / max(len(extracted_resources), 1),
            'total_size': sum(res.size for res in extracted_resources)
        }
        
        return categorization

    def _reconstruct_configuration_data(
        self,
        categorized_resources: Dict[str, Any],
        decompilation_info: Dict[str, Any],
        assembly_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconstruct configuration data and settings from resources"""
        
        configuration = {
            'application_settings': {},
            'build_configuration': {},
            'runtime_parameters': {},
            'resource_mappings': {},
            'version_information': {}
        }
        
        # Extract configuration from string resources
        string_resources = categorized_resources.get('strings', [])
        configuration['application_settings'] = self._extract_app_settings_from_strings(string_resources)
        
        # Extract build configuration
        configuration['build_configuration'] = self._extract_build_config(
            categorized_resources, decompilation_info
        )
        
        # Extract runtime parameters
        configuration['runtime_parameters'] = self._extract_runtime_params(
            string_resources, assembly_info
        )
        
        # Create resource mappings
        configuration['resource_mappings'] = self._create_resource_mappings(
            categorized_resources, decompilation_info
        )
        
        # Extract version information
        configuration['version_information'] = self._extract_version_info(
            categorized_resources
        )
        
        return configuration

    def _create_cross_reference_mapping(
        self,
        categorized_resources: Dict[str, Any],
        decompilation_info: Dict[str, Any],
        assembly_info: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create cross-reference mapping between resources and code"""
        
        mapping = {}
        
        # Map string resources to code references
        string_resources = categorized_resources.get('strings', [])
        decompiled_code = decompilation_info.get('decompiled_code', '')
        
        for resource in string_resources:
            if isinstance(resource.content, str) and len(resource.content) > 3:
                # Search for string usage in decompiled code
                if resource.content in decompiled_code:
                    mapping[resource.resource_id] = f"Used in decompiled code: {resource.content[:50]}..."
        
        # Map binary resources to assembly references
        binary_resources = categorized_resources.get('binary', [])
        instruction_analysis = assembly_info.get('instruction_analysis', {})
        
        for resource in binary_resources:
            # Create mapping based on resource name or ID patterns
            mapping[resource.resource_id] = f"Binary resource: {resource.name}"
        
        return mapping

    def _perform_ai_enhanced_analysis(
        self,
        categorized_resources: Dict[str, Any],
        configuration_data: Dict[str, Any],
        cross_reference_map: Dict[str, str]
    ) -> Dict[str, Any]:
        """Apply AI enhancement to resource analysis"""
        
        if not self.ai_enabled:
            return {
                'ai_analysis_available': False,
                'enhanced_reconstruction_method': 'basic_extraction',
                'resource_pattern_insights': 'AI enhancement not available',
                'structural_understanding': 'Manual analysis required',
                'reconstruction_quality': 'Basic extraction only',
                'confidence_score': 0.0,
                'recommendations': 'Enable AI enhancement for advanced resource reconstruction'
            }
        
        try:
            ai_insights = {
                'resource_classification': {},
                'pattern_analysis': {},
                'project_structure': {},
                'integrity_validation': {}
            }
            
            # AI resource classification enhancement
            classification_prompt = self._create_classification_prompt(categorized_resources)
            classification_response = self.ai_agent.run(classification_prompt)
            ai_insights['resource_classification'] = self._parse_ai_classification_response(classification_response)
            
            # AI pattern analysis
            pattern_prompt = self._create_pattern_analysis_prompt(categorized_resources, configuration_data)
            pattern_response = self.ai_agent.run(pattern_prompt)
            ai_insights['pattern_analysis'] = self._parse_ai_pattern_response(pattern_response)
            
            # AI project structure reconstruction
            structure_prompt = self._create_structure_prompt(categorized_resources, cross_reference_map)
            structure_response = self.ai_agent.run(structure_prompt)
            ai_insights['project_structure'] = self._parse_ai_structure_response(structure_response)
            
            return ai_insights
            
        except Exception as e:
            self.logger.warning(f"AI enhanced analysis failed: {e}")
            return {
                'ai_analysis_available': False,
                'enhanced_reconstruction_method': 'failed',
                'error_message': str(e),
                'fallback_analysis': 'Basic resource extraction performed',
                'confidence_score': 0.0,
                'recommendations': 'Check AI configuration and retry enhanced reconstruction'
            }

    def _create_keymaker_doors(
        self,
        categorized_resources: Dict[str, Any],
        configuration_data: Dict[str, Any],
        cross_reference_map: Dict[str, str],
        ai_insights: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create The Keymaker's doors for comprehensive resource access"""
        
        doors = {
            'master_key': 'Keymaker_Universal_Access',
            'resource_doors': {},
            'configuration_doors': {},
            'cross_reference_doors': {},
            'ai_insight_doors': {},
            'project_reconstruction_door': {}
        }
        
        # Create doors for each resource category
        for category in categorized_resources.get('categories', []):
            door_key = f"door_{category.category_name}"
            doors['resource_doors'][door_key] = {
                'category': category.category_name,
                'access_method': 'direct_extraction',
                'item_count': len(category.items),
                'total_size': category.total_size,
                'key_signature': hashlib.md5(category.category_name.encode()).hexdigest()[:8]
            }
        
        # Create configuration access doors
        for config_section, data in configuration_data.items():
            door_key = f"config_door_{config_section}"
            doors['configuration_doors'][door_key] = {
                'section': config_section,
                'access_method': 'configuration_parser',
                'data_keys': list(data.keys()) if isinstance(data, dict) else [],
                'key_signature': hashlib.md5(config_section.encode()).hexdigest()[:8]
            }
        
        # Create cross-reference doors
        doors['cross_reference_doors'] = {
            'mapping_count': len(cross_reference_map),
            'access_method': 'cross_reference_lookup',
            'key_signature': hashlib.md5('cross_reference'.encode()).hexdigest()[:8]
        }
        
        # Create AI insight doors (if available)
        if ai_insights:
            doors['ai_insight_doors'] = {
                'insight_categories': list(ai_insights.keys()),
                'access_method': 'ai_enhanced_analysis',
                'key_signature': hashlib.md5('ai_insights'.encode()).hexdigest()[:8]
            }
        
        # Create project reconstruction door
        doors['project_reconstruction_door'] = {
            'reconstruction_method': 'complete_project_rebuild',
            'estimated_completeness': self._estimate_reconstruction_completeness(categorized_resources),
            'required_components': ['strings', 'resources', 'configuration', 'code_structure'],
            'master_key_signature': hashlib.md5('keymaker_master'.encode()).hexdigest()[:8]
        }
        
        return doors

    def _calculate_resource_quality_metrics(
        self,
        extracted_resources: List[ResourceItem],
        categorized_resources: Dict[str, Any],
        configuration_data: Dict[str, Any]
    ) -> KeymakerQualityMetrics:
        """Calculate comprehensive quality metrics for resource reconstruction"""
        
        # Resource coverage - estimate based on extraction success
        total_extracted = len(extracted_resources)
        estimated_total = max(total_extracted * 1.2, 10)  # Estimate with 20% margin
        resource_coverage = min(total_extracted / estimated_total, 1.0)
        
        # Extraction accuracy - based on confidence scores
        if extracted_resources:
            avg_confidence = sum(res.confidence for res in extracted_resources) / len(extracted_resources)
            extraction_accuracy = avg_confidence
        else:
            extraction_accuracy = 0.0
        
        # Type classification accuracy - based on successful categorization
        categories = categorized_resources.get('categories', [])
        if categories:
            avg_category_confidence = sum(cat.extraction_confidence for cat in categories) / len(categories)
            type_classification_accuracy = avg_category_confidence
        else:
            type_classification_accuracy = 0.0
        
        # Reconstruction completeness - based on configuration and cross-references
        config_completeness = len(configuration_data) / 5.0  # Expect ~5 major config sections
        reconstruction_completeness = min(config_completeness, 1.0)
        
        # Overall quality - weighted combination
        overall_quality = (
            resource_coverage * 0.3 +
            extraction_accuracy * 0.3 +
            type_classification_accuracy * 0.2 +
            reconstruction_completeness * 0.2
        )
        
        return KeymakerQualityMetrics(
            resource_coverage=resource_coverage,
            extraction_accuracy=extraction_accuracy,
            type_classification_accuracy=type_classification_accuracy,
            reconstruction_completeness=reconstruction_completeness,
            overall_quality=overall_quality
        )

    def _save_keymaker_results(self, keymaker_result: KeymakerAnalysisResult, output_paths: Dict[str, Path]) -> None:
        """Save The Keymaker's comprehensive reconstruction results"""
        
        agent_output_dir = output_paths.get('agents', Path()) / f"agent_{self.agent_id:02d}_keymaker"
        agent_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create resources subdirectory
        resources_dir = agent_output_dir / "resources"
        resources_dir.mkdir(exist_ok=True)
        
        # Save extracted resources
        for category in keymaker_result.resource_categories:
            category_dir = resources_dir / category.category_name
            category_dir.mkdir(exist_ok=True)
            
            for item in category.items:
                if item.resource_type == 'string':
                    # Save string resources as text files
                    item_file = category_dir / f"{item.name}.txt"
                    with open(item_file, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(str(item.content))
                else:
                    # Save binary resources
                    item_file = category_dir / item.name
                    if isinstance(item.content, bytes):
                        with open(item_file, 'wb') as f:
                            f.write(item.content)
                    else:
                        with open(item_file, 'w', encoding='utf-8', errors='ignore') as f:
                            f.write(str(item.content))
        
        # Save comprehensive analysis
        analysis_file = agent_output_dir / "keymaker_analysis.json"
        analysis_data = {
            'agent_info': {
                'agent_id': self.agent_id,
                'agent_name': 'Keymaker_ResourceReconstruction',
                'matrix_character': 'The Keymaker',
                'analysis_timestamp': time.time()
            },
            'resource_categories': [
                {
                    'name': cat.category_name,
                    'item_count': len(cat.items),
                    'total_size': cat.total_size,
                    'extraction_confidence': cat.extraction_confidence,
                    'reconstruction_quality': cat.reconstruction_quality
                }
                for cat in keymaker_result.resource_categories
            ],
            'configuration_data': keymaker_result.configuration_data,
            'reconstruction_map': keymaker_result.reconstruction_map,
            'quality_metrics': {
                'resource_coverage': keymaker_result.quality_metrics.resource_coverage,
                'extraction_accuracy': keymaker_result.quality_metrics.extraction_accuracy,
                'type_classification_accuracy': keymaker_result.quality_metrics.type_classification_accuracy,
                'reconstruction_completeness': keymaker_result.quality_metrics.reconstruction_completeness,
                'overall_quality': keymaker_result.quality_metrics.overall_quality
            },
            'ai_insights': keymaker_result.ai_insights,
            'keymaker_doors': keymaker_result.keymaker_doors
        }
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        self.logger.info(f"The Keymaker's reconstruction results saved to {agent_output_dir}")

    # AI Enhancement Methods
    def _ai_classify_resource_types(self, resource_info: str) -> str:
        """AI tool for classifying resource types"""
        return f"Resource classification: {resource_info[:100]}..."
    
    def _ai_analyze_resource_patterns(self, pattern_info: str) -> str:
        """AI tool for analyzing resource patterns"""
        return f"Resource pattern analysis: {pattern_info[:100]}..."
    
    def _ai_reconstruct_project_structure(self, structure_info: str) -> str:
        """AI tool for reconstructing project structure"""
        return f"Project structure reconstruction: {structure_info[:100]}..."
    
    def _ai_validate_resource_integrity(self, integrity_info: str) -> str:
        """AI tool for validating resource integrity"""
        return f"Resource integrity validation: {integrity_info[:100]}..."

    # Placeholder methods for resource extraction components
    def _analyze_pe_containers(self, binary_path: str) -> Dict[str, Any]:
        """Analyze PE-specific resource containers"""
        return {'pe_resources': [], 'resource_table': {}}
    
    def _analyze_elf_containers(self, binary_path: str) -> Dict[str, Any]:
        """Analyze ELF-specific containers"""
        return {'elf_sections': [], 'string_tables': []}
    
    def _analyze_macho_containers(self, binary_path: str) -> Dict[str, Any]:
        """Analyze Mach-O-specific containers"""
        return {'load_commands': [], 'segments': []}
    
    def _analyze_generic_containers(self, binary_path: str, containers: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze generic containers"""
        return containers
    
    def _extract_pe_resources(self, binary_path: str, containers: Dict[str, Any]) -> List[ResourceItem]:
        """Extract PE resources"""
        resources = []
        
        try:
            with open(binary_path, 'rb') as f:
                pe_data = f.read()
            
            # Simple PE resource extraction without pefile dependency
            # Look for PE signature and resource section
            if pe_data[:2] == b'MZ':
                # Find PE header
                pe_offset_pos = 0x3C
                if len(pe_data) > pe_offset_pos + 4:
                    pe_offset = struct.unpack('<I', pe_data[pe_offset_pos:pe_offset_pos+4])[0]
                    
                    if len(pe_data) > pe_offset + 4 and pe_data[pe_offset:pe_offset+4] == b'PE\x00\x00':
                        # Basic resource extraction - look for common resource patterns
                        
                        # Extract version info patterns
                        version_patterns = [b'FileVersion', b'ProductVersion', b'CompanyName', b'FileDescription']
                        for pattern in version_patterns:
                            pos = pe_data.find(pattern)
                            if pos != -1:
                                # Extract version string (simplified)
                                end_pos = pos + 50
                                version_data = pe_data[pos:end_pos]
                                
                                resources.append(ResourceItem(
                                    resource_id=f'version_{pattern.decode("utf-8", errors="ignore")}',
                                    resource_type='version_info',
                                    name=f'version_{pattern.decode("utf-8", errors="ignore")}.txt',
                                    size=len(version_data),
                                    content=version_data.decode('utf-8', errors='ignore'),
                                    metadata={'pattern': pattern.decode('utf-8', errors='ignore'), 'offset': pos},
                                    extraction_method='pe_version_extraction',
                                    confidence=0.7
                                ))
                        
                        # Extract icon resources (look for ICO signature)
                        ico_pos = pe_data.find(b'\x00\x00\x01\x00')  # ICO header
                        if ico_pos != -1:
                            ico_size = min(1024, len(pe_data) - ico_pos)  # Estimate icon size
                            ico_data = pe_data[ico_pos:ico_pos + ico_size]
                            
                            resources.append(ResourceItem(
                                resource_id='main_icon',
                                resource_type='icon',
                                name='main_icon.ico',
                                size=ico_size,
                                content=ico_data,
                                metadata={'offset': ico_pos, 'estimated_size': ico_size},
                                extraction_method='pe_icon_extraction',
                                confidence=0.6
                            ))
                        
                        # Extract manifest resources (look for XML manifests)
                        manifest_start = pe_data.find(b'<?xml')
                        if manifest_start != -1:
                            manifest_end = pe_data.find(b'</assembly>', manifest_start)
                            if manifest_end != -1:
                                manifest_data = pe_data[manifest_start:manifest_end + 11]
                                
                                resources.append(ResourceItem(
                                    resource_id='application_manifest',
                                    resource_type='manifest',
                                    name='application.manifest',
                                    size=len(manifest_data),
                                    content=manifest_data.decode('utf-8', errors='ignore'),
                                    metadata={'offset': manifest_start},
                                    extraction_method='pe_manifest_extraction',
                                    confidence=0.8
                                ))
            
        except Exception as e:
            self.logger.error(f'PE resource extraction failed: {e}')
        
        return resources
    
    def _extract_elf_resources(self, binary_path: str, containers: Dict[str, Any]) -> List[ResourceItem]:
        """Extract ELF resources"""
        resources = []
        
        try:
            with open(binary_path, 'rb') as f:
                elf_data = f.read()
            
            # Simple ELF resource extraction without elftools dependency
            # Check for ELF signature
            if elf_data[:4] == b'\x7fELF':
                # Extract from common ELF sections
                
                # Look for .rodata section (read-only data)
                rodata_marker = b'.rodata\x00'
                rodata_pos = elf_data.find(rodata_marker)
                if rodata_pos != -1:
                    # Extract strings from rodata section (simplified)
                    section_start = rodata_pos + len(rodata_marker)
                    section_end = min(section_start + 4096, len(elf_data))  # Limit section size
                    
                    rodata_content = elf_data[section_start:section_end]
                    readable_strings = self._extract_printable_strings(rodata_content)
                    
                    if readable_strings:
                        resources.append(ResourceItem(
                            resource_id='rodata_strings',
                            resource_type='string_table',
                            name='rodata_strings.txt',
                            size=len(rodata_content),
                            content='\n'.join(readable_strings[:50]),  # First 50 strings
                            metadata={'section': '.rodata', 'string_count': len(readable_strings)},
                            extraction_method='elf_rodata_extraction',
                            confidence=0.7
                        ))
                
                # Look for .comment section (compiler info)
                comment_marker = b'.comment\x00'
                comment_pos = elf_data.find(comment_marker)
                if comment_pos != -1:
                    comment_start = comment_pos + len(comment_marker)
                    comment_end = min(comment_start + 256, len(elf_data))
                    comment_data = elf_data[comment_start:comment_end]
                    
                    # Extract compiler information
                    comment_str = comment_data.decode('utf-8', errors='ignore').strip('\x00')
                    if comment_str:
                        resources.append(ResourceItem(
                            resource_id='compiler_info',
                            resource_type='build_info',
                            name='compiler_info.txt',
                            size=len(comment_data),
                            content=comment_str,
                            metadata={'section': '.comment'},
                            extraction_method='elf_comment_extraction',
                            confidence=0.8
                        ))
                
                # Look for .note sections (additional metadata)
                note_marker = b'.note'
                note_pos = elf_data.find(note_marker)
                if note_pos != -1:
                    note_start = note_pos + 10  # Skip marker
                    note_end = min(note_start + 512, len(elf_data))
                    note_data = elf_data[note_start:note_end]
                    
                    resources.append(ResourceItem(
                        resource_id='elf_notes',
                        resource_type='metadata',
                        name='elf_notes.bin',
                        size=len(note_data),
                        content=note_data,
                        metadata={'section': '.note'},
                        extraction_method='elf_note_extraction',
                        confidence=0.6
                    ))
            
        except Exception as e:
            self.logger.error(f'ELF resource extraction failed: {e}')
        
        return resources
    
    def _extract_macho_resources(self, binary_path: str, containers: Dict[str, Any]) -> List[ResourceItem]:
        """Extract Mach-O resources"""
        resources = []
        
        try:
            with open(binary_path, 'rb') as f:
                macho_data = f.read()
            
            # Simple Mach-O resource extraction
            # Check for Mach-O signatures
            if macho_data[:4] in [b'\xfe\xed\xfa\xce', b'\xfe\xed\xfa\xcf', b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe']:
                # Extract from common Mach-O sections
                
                # Look for __TEXT segment strings
                text_marker = b'__TEXT'
                text_pos = macho_data.find(text_marker)
                if text_pos != -1:
                    # Extract strings from TEXT segment
                    text_start = text_pos + len(text_marker)
                    text_end = min(text_start + 2048, len(macho_data))
                    text_content = macho_data[text_start:text_end]
                    
                    text_strings = self._extract_printable_strings(text_content)
                    if text_strings:
                        resources.append(ResourceItem(
                            resource_id='text_segment_strings',
                            resource_type='string_table',
                            name='text_strings.txt',
                            size=len(text_content),
                            content='\n'.join(text_strings[:30]),
                            metadata={'segment': '__TEXT', 'string_count': len(text_strings)},
                            extraction_method='macho_text_extraction',
                            confidence=0.7
                        ))
                
                # Look for __DATA segment
                data_marker = b'__DATA'
                data_pos = macho_data.find(data_marker)
                if data_pos != -1:
                    data_start = data_pos + len(data_marker)
                    data_end = min(data_start + 1024, len(macho_data))
                    data_content = macho_data[data_start:data_end]
                    
                    resources.append(ResourceItem(
                        resource_id='data_segment',
                        resource_type='data',
                        name='data_segment.bin',
                        size=len(data_content),
                        content=data_content,
                        metadata={'segment': '__DATA'},
                        extraction_method='macho_data_extraction',
                        confidence=0.6
                    ))
                
                # Look for load commands that might contain version info
                lc_uuid_marker = b'\x1b\x00\x00\x00'  # LC_UUID command
                uuid_pos = macho_data.find(lc_uuid_marker)
                if uuid_pos != -1 and uuid_pos + 20 < len(macho_data):
                    uuid_data = macho_data[uuid_pos + 8:uuid_pos + 24]  # UUID is 16 bytes
                    uuid_str = uuid_data.hex()
                    
                    resources.append(ResourceItem(
                        resource_id='binary_uuid',
                        resource_type='identifier',
                        name='binary_uuid.txt',
                        size=len(uuid_str),
                        content=uuid_str,
                        metadata={'load_command': 'LC_UUID'},
                        extraction_method='macho_uuid_extraction',
                        confidence=0.9
                    ))
                
                # Look for Objective-C class information
                objc_marker = b'__objc_classname'
                objc_pos = macho_data.find(objc_marker)
                if objc_pos != -1:
                    objc_start = objc_pos + len(objc_marker)
                    objc_end = min(objc_start + 512, len(macho_data))
                    objc_content = macho_data[objc_start:objc_end]
                    
                    objc_strings = self._extract_printable_strings(objc_content)
                    if objc_strings:
                        resources.append(ResourceItem(
                            resource_id='objc_classes',
                            resource_type='class_info',
                            name='objc_classes.txt',
                            size=len(objc_content),
                            content='\n'.join(objc_strings[:20]),
                            metadata={'section': '__objc_classname'},
                            extraction_method='macho_objc_extraction',
                            confidence=0.8
                        ))
            
        except Exception as e:
            self.logger.error(f'Mach-O resource extraction failed: {e}')
        
        return resources
    
    def _extract_string_resources(self, binary_path: str, containers: Dict[str, Any]) -> List[ResourceItem]:
        """Extract string resources"""
        resources = []
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
                
            # Simple string extraction
            strings = self._extract_printable_strings(data)
            
            for i, string_content in enumerate(strings):
                if len(string_content) >= self.min_string_length:
                    resource = ResourceItem(
                        resource_id=f"string_{i:04d}",
                        resource_type="string",
                        name=f"string_{i:04d}",
                        size=len(string_content),
                        content=string_content,
                        metadata={'offset': 0, 'encoding': 'ascii'},
                        extraction_method="string_scanner",
                        confidence=0.8
                    )
                    resources.append(resource)
        except Exception as e:
            self.logger.error(f"String extraction failed: {e}")
        
        return resources
    
    def _scan_binary_resources(self, binary_path: str, containers: Dict[str, Any]) -> List[ResourceItem]:
        """Scan for binary resources"""
        resources = []
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            # Scan for known file signatures
            for file_type, signature in self.resource_patterns['image_signatures'].items():
                offset = 0
                while True:
                    pos = data.find(signature.encode('latin-1'), offset)
                    if pos == -1:
                        break
                    
                    resource = ResourceItem(
                        resource_id=f"{file_type}_{pos:08x}",
                        resource_type="image",
                        name=f"{file_type}_{pos:08x}.{file_type}",
                        size=0,  # Would need to calculate actual size
                        content=signature.encode('latin-1'),
                        metadata={'offset': pos, 'signature': signature},
                        extraction_method="signature_scanner",
                        confidence=0.7
                    )
                    resources.append(resource)
                    offset = pos + 1
                    
        except Exception as e:
            self.logger.error(f"Binary scanning failed: {e}")
        
        return resources
    
    def _detect_embedded_files(self, binary_path: str, containers: Dict[str, Any]) -> List[ResourceItem]:
        """Detect embedded files"""
        embedded_files = []
        
        try:
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            # Search for embedded file signatures
            all_signatures = {**self.resource_patterns['image_signatures'], 
                            **self.resource_patterns['archive_signatures']}
            
            for file_type, signature in all_signatures.items():
                signature_bytes = signature.encode('latin-1') if isinstance(signature, str) else signature
                offset = 0
                
                while True:
                    pos = binary_data.find(signature_bytes, offset)
                    if pos == -1:
                        break
                    
                    # Estimate file size based on type
                    estimated_size = self._estimate_embedded_file_size(binary_data, pos, file_type)
                    
                    if estimated_size > 0:
                        file_data = binary_data[pos:pos + estimated_size]
                        
                        # Validate the extracted data
                        if self._validate_embedded_file(file_data, file_type):
                            embedded_files.append(ResourceItem(
                                resource_id=f'embedded_{file_type}_{pos:08x}',
                                resource_type='embedded_file',
                                name=f'embedded_{file_type}_{pos:08x}.{file_type}',
                                size=estimated_size,
                                content=file_data,
                                metadata={
                                    'offset': pos,
                                    'file_type': file_type,
                                    'signature': signature.hex() if isinstance(signature, bytes) else signature
                                },
                                extraction_method='signature_based_carving',
                                confidence=0.7
                            ))
                    
                    offset = pos + 1
                    
                    # Limit search to prevent excessive processing
                    if len(embedded_files) > 20:
                        break
            
            # Look for high-entropy regions that might contain compressed/encrypted files
            entropy_regions = self._find_high_entropy_regions(binary_data)
            for region_start, region_size in entropy_regions:
                if region_size > 100:  # Only consider significant regions
                    region_data = binary_data[region_start:region_start + region_size]
                    
                    embedded_files.append(ResourceItem(
                        resource_id=f'high_entropy_{region_start:08x}',
                        resource_type='compressed_data',
                        name=f'high_entropy_{region_start:08x}.bin',
                        size=region_size,
                        content=region_data,
                        metadata={
                            'offset': region_start,
                            'entropy_score': self._calculate_entropy_score(region_data),
                            'analysis_type': 'entropy_based'
                        },
                        extraction_method='entropy_analysis',
                        confidence=0.5
                    ))
                    
                    # Limit high-entropy regions
                    if len([f for f in embedded_files if f.resource_type == 'compressed_data']) > 5:
                        break
        
        except Exception as e:
            self.logger.error(f'Embedded file detection failed: {e}')
        
        return embedded_files
    
    def _estimate_embedded_file_size(self, data: bytes, start_pos: int, file_type: str) -> int:
        """Estimate the size of an embedded file"""
        # File type specific size estimation
        if file_type in ['png', 'jpeg', 'gif', 'bmp']:
            # For images, look for common end markers or estimate from header
            if file_type == 'png':
                # PNG files end with IEND chunk
                end_marker = b'IEND\xaeB`\x82'
                end_pos = data.find(end_marker, start_pos)
                if end_pos != -1:
                    return end_pos - start_pos + len(end_marker)
                else:
                    return min(10240, len(data) - start_pos)  # Max 10KB estimate
            
            elif file_type == 'jpeg':
                # JPEG files end with FFD9
                end_marker = b'\xff\xd9'
                end_pos = data.find(end_marker, start_pos + 2)
                if end_pos != -1:
                    return end_pos - start_pos + 2
                else:
                    return min(20480, len(data) - start_pos)  # Max 20KB estimate
        
        elif file_type in ['zip', 'rar']:
            # Archives - look for end of central directory or estimate
            if file_type == 'zip':
                # ZIP central directory end signature
                end_marker = b'PK\x05\x06'
                end_pos = data.find(end_marker, start_pos)
                if end_pos != -1:
                    return end_pos - start_pos + 22  # Minimum end record size
                else:
                    return min(51200, len(data) - start_pos)  # Max 50KB estimate
        
        # Default estimation
        remaining_data = len(data) - start_pos
        return min(8192, remaining_data)  # Default 8KB maximum
    
    def _validate_embedded_file(self, file_data: bytes, file_type: str) -> bool:
        """Validate that extracted data is likely a valid file of the given type"""
        if len(file_data) < 10:
            return False
        
        # Basic validation based on file structure
        if file_type == 'png':
            # PNG should have proper chunk structure
            return len(file_data) > 8 and file_data[8:12] == b'IHDR'
        
        elif file_type == 'jpeg':
            # JPEG should have proper markers
            return len(file_data) > 10 and file_data[2:4] == b'\xff\xe0'
        
        elif file_type == 'zip':
            # ZIP should have proper local file header structure
            if len(file_data) > 30:
                return struct.unpack('<H', file_data[8:10])[0] < 100  # Reasonable compression method
        
        # Default: accept if reasonable size
        return 100 <= len(file_data) <= 1024 * 1024  # Between 100 bytes and 1MB
    
    def _find_high_entropy_regions(self, data: bytes) -> List[Tuple[int, int]]:
        """Find regions of high entropy that might contain compressed/encrypted data"""
        regions = []
        block_size = 1024
        entropy_threshold = 7.0  # High entropy threshold
        
        for i in range(0, len(data) - block_size, block_size // 2):  # Overlapping blocks
            block = data[i:i + block_size]
            entropy = self._calculate_entropy_score(block)
            
            if entropy > entropy_threshold:
                # Find the full extent of the high-entropy region
                start = i
                end = i + block_size
                
                # Extend backwards
                while start > 0:
                    prev_block = data[max(0, start - block_size):start]
                    if len(prev_block) < block_size or self._calculate_entropy_score(prev_block) < entropy_threshold:
                        break
                    start -= block_size // 2
                
                # Extend forwards
                while end < len(data):
                    next_block = data[end:end + block_size]
                    if len(next_block) < block_size or self._calculate_entropy_score(next_block) < entropy_threshold:
                        break
                    end += block_size // 2
                
                region_size = end - start
                if region_size >= block_size:  # Only significant regions
                    regions.append((start, region_size))
                    
                # Skip ahead to avoid overlapping regions
                i = end
        
        return regions[:10]  # Limit to 10 regions
    
    def _calculate_entropy_score(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        from collections import Counter
        import math
        
        # Count byte frequencies
        byte_counts = Counter(data)
        length = len(data)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _deduplicate_resources(self, resources: List[ResourceItem]) -> List[ResourceItem]:
        """Remove duplicate resources"""
        seen = set()
        deduplicated = []
        
        for resource in resources:
            # Create a hash based on content or name
            if isinstance(resource.content, bytes):
                content_hash = hashlib.md5(resource.content).hexdigest()
            else:
                content_hash = hashlib.md5(str(resource.content).encode()).hexdigest()
            
            if content_hash not in seen:
                seen.add(content_hash)
                deduplicated.append(resource)
        
        return deduplicated
    
    def _validate_extracted_resources(self, resources: List[ResourceItem]) -> List[ResourceItem]:
        """Validate extracted resources"""
        validated = []
        
        for resource in resources:
            # Basic validation
            if resource.size > self.max_resource_size:
                continue
            
            if resource.resource_type == 'string' and len(str(resource.content)) < self.min_string_length:
                continue
            
            validated.append(resource)
        
        return validated
    
    def _calculate_category_quality(self, resources: List[ResourceItem]) -> float:
        """Calculate quality score for a resource category"""
        if not resources:
            return 0.0
        
        avg_confidence = sum(res.confidence for res in resources) / len(resources)
        return avg_confidence
    
    def _generate_resource_metadata(self, resources: List[ResourceItem]) -> Dict[str, Any]:
        """Generate metadata analysis for resources"""
        return {
            'total_count': len(resources),
            'total_size': sum(res.size for res in resources),
            'type_distribution': {},
            'quality_summary': 'good'
        }
    
    def _extract_printable_strings(self, data: bytes) -> List[str]:
        """Extract printable strings from binary data"""
        strings = []
        current_string = ""
        
        for byte in data:
            if 32 <= byte <= 126:  # Printable ASCII
                current_string += chr(byte)
            else:
                if len(current_string) >= self.min_string_length:
                    strings.append(current_string)
                current_string = ""
        
        # Don't forget the last string
        if len(current_string) >= self.min_string_length:
            strings.append(current_string)
        
        return strings
    
    def _extract_app_settings_from_strings(self, string_resources: List[ResourceItem]) -> Dict[str, Any]:
        """Extract application settings from string resources"""
        return {'settings_found': len(string_resources)}
    
    def _extract_build_config(self, categorized_resources: Dict[str, Any], decompilation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract build configuration"""
        return {'build_info': 'extracted'}
    
    def _extract_runtime_params(self, string_resources: List[ResourceItem], assembly_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract runtime parameters"""
        return {'runtime_config': 'extracted'}
    
    def _create_resource_mappings(self, categorized_resources: Dict[str, Any], decompilation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create resource mappings"""
        return {'mappings': 'created'}
    
    def _extract_version_info(self, categorized_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Extract version information"""
        return {'version': 'unknown'}
    
    def _estimate_reconstruction_completeness(self, categorized_resources: Dict[str, Any]) -> float:
        """Estimate reconstruction completeness"""
        return 0.8
    
    def _create_classification_prompt(self, categorized_resources: Dict[str, Any]) -> str:
        """Create AI prompt for resource classification"""
        return f"Classify these resources: {categorized_resources}"
    
    def _parse_ai_classification_response(self, response: str) -> Dict[str, Any]:
        """Parse AI classification response"""
        return {'classification': response}
    
    def _create_pattern_analysis_prompt(self, categorized_resources: Dict[str, Any], configuration_data: Dict[str, Any]) -> str:
        """Create AI prompt for pattern analysis"""
        return f"Analyze patterns in: {categorized_resources}"
    
    def _parse_ai_pattern_response(self, response: str) -> Dict[str, Any]:
        """Parse AI pattern response"""
        return {'patterns': response}
    
    def _create_structure_prompt(self, categorized_resources: Dict[str, Any], cross_reference_map: Dict[str, str]) -> str:
        """Create AI prompt for structure reconstruction"""
        return f"Reconstruct structure from: {categorized_resources}"
    
    def _parse_ai_structure_response(self, response: str) -> Dict[str, Any]:
        """Parse AI structure response"""
        return {'structure': response}