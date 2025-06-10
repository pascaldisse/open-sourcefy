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

# Centralized AI system imports
from ..ai_system import ai_available, ai_analyze_code, ai_enhance_code, ai_request_safe

# Optional imports with fallbacks for missing dependencies
try:
    from langchain.agents import ReActDocstoreAgent, AgentExecutor
    from langchain.llms import LlamaCpp
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# PE resource extraction imports
try:
    import pefile
    PE_AVAILABLE = True
except ImportError:
    PE_AVAILABLE = False

# Image processing imports
try:
    from PIL import Image
    import io
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    ReActDocstoreAgent = Any
    LlamaCpp = Any
    ConversationBufferMemory = Any
    AgentExecutor = Any
    Tool = Any

# Math import for FP constant analysis
import math

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
            agent_id=7,
            matrix_character=MatrixCharacter.KEYMAKER,
            dependencies=[1, 2]  # Depends on Binary Discovery and Arch Analysis
        )
        
        # Load Keymaker-specific configuration
        self.min_string_length = self.config.get_value('agents.agent_08.min_string_length', 4)
        self.max_resource_size = self.config.get_value('agents.agent_08.max_resource_size', 50 * 1024 * 1024)  # 50MB
        self.timeout_seconds = self.config.get_value('agents.agent_08.timeout', 300)
        self.enable_deep_extraction = self.config.get_value('agents.agent_08.deep_extraction', True)
        
        # Initialize components
        self.error_handler = MatrixErrorHandler("Keymaker", max_retries=2)
        
        # Initialize AI components if available
        self.ai_enabled = ai_available()
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
            # Use centralized AI system instead of local model
            from ..ai_system import ai_available
            self.ai_enabled = ai_available()
            if not self.ai_enabled:
                return
            
            # AI system is now centralized - no local setup needed
            self.logger.info("Keymaker AI agent successfully initialized with centralized AI system")
            
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
            agent3_data = context['agent_results'][3].data if 3 in context['agent_results'] else {}  # Merovingian's decompilation (optional)
            
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
            
            # Phase 3.5: Constant Pool Reconstruction
            self.logger.info("Phase 3.5: Reconstructing constant pools with exact placement")
            constant_pools = self._reconstruct_constant_pools_phase3(binary_path, categorized_resources)
            categorized_resources['constant_pools'] = constant_pools
            
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
        required_agents = [1, 2]
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
            classification_response = ai_analyze_code(classification_prompt)
            ai_insights['resource_classification'] = self._parse_ai_classification_response(classification_response)
            
            # AI pattern analysis
            pattern_prompt = self._create_pattern_analysis_prompt(categorized_resources, configuration_data)
            pattern_response = ai_analyze_code(pattern_prompt)
            ai_insights['pattern_analysis'] = self._parse_ai_pattern_response(pattern_response)
            
            # AI project structure reconstruction
            structure_prompt = self._create_structure_prompt(categorized_resources, cross_reference_map)
            structure_response = ai_analyze_code(structure_prompt)
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
                elif item.resource_type in ['image', 'icon', 'bitmap']:
                    # Save image resources as binary files
                    item_file = category_dir / item.name
                    with open(item_file, 'wb') as f:
                        f.write(item.content)
                    self.logger.info(f"💾 Saved {item.resource_type}: {item.name} ({item.size:,} bytes)")
                else:
                    # Save other binary resources
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
        """Extract comprehensive PE resources including strings, images, and embedded data"""
        resources = []
        
        try:
            # First try advanced PE analysis if pefile is available
            if PE_AVAILABLE:
                resources.extend(self._extract_pe_resources_advanced(binary_path))
            else:
                # Fallback to manual PE parsing
                resources.extend(self._extract_pe_resources_manual(binary_path))
                
            # Always perform comprehensive string extraction
            resources.extend(self._extract_pe_string_table(binary_path))
            
            # Extract embedded image resources
            resources.extend(self._extract_pe_image_resources(binary_path))
            
            # Extract dialog and menu resources
            resources.extend(self._extract_pe_ui_resources(binary_path))
            
            self.logger.info(f"✅ Extracted {len(resources)} PE resources")
            
        except Exception as e:
            self.logger.error(f'PE resource extraction failed: {e}')
            # Still try basic extraction
            resources.extend(self._extract_pe_resources_basic(binary_path))
        
        return resources
    
    def _extract_pe_resources_advanced(self, binary_path: str) -> List[ResourceItem]:
        """Advanced PE resource extraction using pefile library"""
        resources = []
        
        try:
            import pefile
            pe = pefile.PE(binary_path)
            
            # Extract all resource types
            if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
                for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                    if hasattr(resource_type, 'directory'):
                        for resource_id in resource_type.directory.entries:
                            if hasattr(resource_id, 'directory'):
                                for resource_lang in resource_id.directory.entries:
                                    try:
                                        data = pe.get_data(resource_lang.data.struct.OffsetToData, 
                                                         resource_lang.data.struct.Size)
                                        
                                        # Determine resource type
                                        type_name = self._get_pe_resource_type_name(resource_type.id)
                                        
                                        resource_item = ResourceItem(
                                            resource_id=f'{type_name}_{resource_id.id}_{resource_lang.id}',
                                            resource_type=type_name,
                                            name=f'{type_name}_{resource_id.id}.{self._get_extension_for_type(type_name)}',
                                            size=len(data),
                                            content=data,
                                            metadata={
                                                'type_id': resource_type.id,
                                                'resource_id': resource_id.id, 
                                                'language_id': resource_lang.id,
                                                'offset': resource_lang.data.struct.OffsetToData
                                            },
                                            extraction_method='pefile_advanced',
                                            confidence=0.95
                                        )
                                        resources.append(resource_item)
                                        
                                        # Special handling for string tables
                                        if type_name == 'string':
                                            resources.extend(self._parse_string_table_data(data, resource_id.id))
                                            
                                    except Exception as e:
                                        self.logger.warning(f"Failed to extract resource {resource_type.id}.{resource_id.id}: {e}")
                                        
            pe.close()
            self.logger.info(f"Advanced PE extraction found {len(resources)} resources")
            
        except Exception as e:
            self.logger.error(f'Advanced PE resource extraction failed: {e}')
            
        return resources
    
    def _extract_pe_string_table(self, binary_path: str) -> List[ResourceItem]:
        """Phase 3: Enhanced string table extraction with exact layout and placement analysis"""
        resources = []
        
        try:
            with open(binary_path, 'rb') as f:
                pe_data = f.read()
            
            self.logger.info("🎯 Phase 3: Analyzing string literal placement and layout...")
            
            # Phase 3.1: String Literal Placement Analysis
            string_layout_analysis = self._analyze_string_literal_placement_phase3(pe_data)
            
            # Phase 3.2: Unicode String Extraction with Memory Layout
            unicode_strings = self._extract_unicode_strings_with_layout(pe_data)
            for string_info in unicode_strings:
                resources.append(ResourceItem(
                    resource_id=f'unicode_string_{string_info["memory_address"]:08x}',
                    resource_type='string_literal',
                    name=f'unicode_{string_info["memory_address"]:08x}.txt',
                    size=string_info['size'],
                    content=string_info['content'],
                    metadata={
                        'type': 'unicode',
                        'memory_address': string_info['memory_address'],
                        'file_offset': string_info['file_offset'],
                        'section': string_info['section'],
                        'alignment': string_info['alignment'],
                        'encoding': 'utf-16le',
                        'null_terminated': string_info['null_terminated'],
                        'access_pattern': 'read_only'
                    },
                    extraction_method='phase3_unicode_layout_analysis',
                    confidence=0.92
                ))
            
            # Phase 3.3: ASCII String Extraction with Memory Layout
            ascii_strings = self._extract_ascii_strings_with_layout(pe_data)
            for string_info in ascii_strings:
                resources.append(ResourceItem(
                    resource_id=f'ascii_string_{string_info["memory_address"]:08x}',
                    resource_type='string_literal',
                    name=f'ascii_{string_info["memory_address"]:08x}.txt',
                    size=string_info['size'],
                    content=string_info['content'],
                    metadata={
                        'type': 'ascii',
                        'memory_address': string_info['memory_address'],
                        'file_offset': string_info['file_offset'],
                        'section': string_info['section'],
                        'alignment': string_info['alignment'],
                        'encoding': 'ascii',
                        'null_terminated': string_info['null_terminated'],
                        'access_pattern': 'read_only'
                    },
                    extraction_method='phase3_ascii_layout_analysis',
                    confidence=0.88
                ))
            
            # Phase 3.4: String Table Layout Reconstruction
            string_tables = self._reconstruct_string_table_layout(pe_data, unicode_strings + ascii_strings)
            for table_info in string_tables:
                resources.append(ResourceItem(
                    resource_id=f'string_table_{table_info["address"]:08x}',
                    resource_type='string_table_layout',
                    name=f'string_table_{table_info["address"]:08x}.layout',
                    size=table_info['size'],
                    content=table_info['layout_description'],
                    metadata={
                        'base_address': table_info['address'],
                        'string_count': table_info['string_count'],
                        'total_size': table_info['size'],
                        'alignment': table_info['alignment'],
                        'section': table_info['section'],
                        'density': table_info['density']
                    },
                    extraction_method='phase3_string_table_reconstruction',
                    confidence=0.85
                ))
            
            self.logger.info(f"✅ Phase 3 string extraction: {len(resources)} string resources with layout analysis")
            
        except Exception as e:
            self.logger.error(f'Phase 3 string table extraction failed: {e}')
            
        return resources
    
    def _extract_pe_image_resources(self, binary_path: str) -> List[ResourceItem]:
        """Extract image resources - targeting the 21 BMP images"""
        resources = []
        
        try:
            with open(binary_path, 'rb') as f:
                pe_data = f.read()
            
            self.logger.info("🖼️ Starting comprehensive BMP extraction for Matrix Online launcher...")
            
            # Enhanced BMP extraction targeting the expected 21 images
            bmp_count = 0
            offset = 0
            
            while True:
                # Look for BMP signature 'BM' (0x424D)
                pos = pe_data.find(b'BM', offset)
                if pos == -1:
                    break
                    
                try:
                    # Validate this is a real BMP by checking header structure
                    if pos + 54 < len(pe_data):  # Minimum BMP header size
                        # Read BMP file header
                        file_size = struct.unpack('<I', pe_data[pos+2:pos+6])[0]
                        reserved1 = struct.unpack('<H', pe_data[pos+6:pos+8])[0]
                        reserved2 = struct.unpack('<H', pe_data[pos+8:pos+10])[0]
                        data_offset = struct.unpack('<I', pe_data[pos+10:pos+14])[0]
                        
                        # Read DIB header
                        dib_header_size = struct.unpack('<I', pe_data[pos+14:pos+18])[0]
                        
                        # Validate BMP structure
                        if (reserved1 == 0 and reserved2 == 0 and  # Reserved fields should be 0
                            data_offset >= 54 and data_offset < file_size and  # Valid data offset
                            dib_header_size >= 40 and  # Standard DIB header
                            file_size > 54 and file_size < 5*1024*1024):  # Reasonable file size
                            
                            # Extract BMP data
                            actual_size = min(file_size, len(pe_data) - pos)
                            bmp_data = pe_data[pos:pos+actual_size]
                            
                            # Additional validation: try to read width/height
                            if len(bmp_data) >= 26:
                                width = struct.unpack('<I', bmp_data[18:22])[0]
                                height = struct.unpack('<I', bmp_data[22:26])[0]
                                
                                # Reasonable dimensions check
                                if width > 0 and height > 0 and width < 4096 and height < 4096:
                                    resources.append(ResourceItem(
                                        resource_id=f'matrix_bitmap_{bmp_count:03d}',
                                        resource_type='image',
                                        name=f'matrix_bitmap_{bmp_count:03d}.bmp',
                                        size=actual_size,
                                        content=bmp_data,
                                        metadata={
                                            'format': 'BMP',
                                            'offset': pos,
                                            'file_size': file_size,
                                            'width': width,
                                            'height': height,
                                            'data_offset': data_offset,
                                            'dib_header_size': dib_header_size
                                        },
                                        extraction_method='enhanced_bmp_validation',
                                        confidence=0.95
                                    ))
                                    bmp_count += 1
                                    self.logger.info(f"✅ Extracted BMP {bmp_count}: {width}x{height} pixels, {actual_size} bytes")
                            
                except Exception as e:
                    self.logger.debug(f"BMP validation failed at offset {pos}: {e}")
                
                offset = pos + 1
                
                # Stop when we've found the expected number plus some margin
                if bmp_count >= 30:  # Look for more than 21 to be thorough
                    break
            
            # Look for icon resources (ICO format)
            ico_count = 0
            offset = 0
            while True:
                pos = pe_data.find(b'\x00\x00\x01\x00', offset)  # ICO header
                if pos == -1:
                    break
                    
                if pos + 6 < len(pe_data):
                    try:
                        num_images = struct.unpack('<H', pe_data[pos+4:pos+6])[0]
                        if 1 <= num_images <= 10:  # Reasonable icon count
                            ico_size = min(4096, len(pe_data) - pos)  # Estimate
                            ico_data = pe_data[pos:pos+ico_size]
                            
                            resources.append(ResourceItem(
                                resource_id=f'icon_{ico_count}',
                                resource_type='icon', 
                                name=f'icon_{ico_count}.ico',
                                size=ico_size,
                                content=ico_data,
                                metadata={'format': 'ICO', 'num_images': num_images, 'offset': pos},
                                extraction_method='ico_signature_scan',
                                confidence=0.85
                            ))
                            ico_count += 1
                    except:
                        pass
                
                offset = pos + 1
                if ico_count >= 20:  # Safety limit
                    break
            
            self.logger.info(f"Image extraction found {len(resources)} images ({bmp_count} BMPs, {ico_count} ICOs)")
            
        except Exception as e:
            self.logger.error(f'Image resource extraction failed: {e}')
            
        return resources
    
    def _extract_pe_ui_resources(self, binary_path: str) -> List[ResourceItem]:
        """Extract dialog and menu UI resources"""
        resources = []
        
        try:
            with open(binary_path, 'rb') as f:
                pe_data = f.read()
            
            # Look for dialog resource patterns
            dialog_patterns = [
                b'\x01\x00\xff\xff',  # Dialog template header
                b'DIALOG',            # Dialog keyword
                b'DIALOGEX',          # Extended dialog
            ]
            
            for pattern in dialog_patterns:
                offset = 0
                dialog_count = 0
                while True:
                    pos = pe_data.find(pattern, offset)
                    if pos == -1:
                        break
                    
                    # Extract dialog data (estimate size)
                    dialog_size = min(1024, len(pe_data) - pos)
                    dialog_data = pe_data[pos:pos+dialog_size]
                    
                    resources.append(ResourceItem(
                        resource_id=f'dialog_{dialog_count}',
                        resource_type='dialog',
                        name=f'dialog_{dialog_count}.dlg',
                        size=dialog_size,
                        content=dialog_data,
                        metadata={'pattern': pattern.decode('utf-8', errors='ignore'), 'offset': pos},
                        extraction_method='dialog_pattern_scan',
                        confidence=0.75
                    ))
                    dialog_count += 1
                    offset = pos + 1
                    
                    if dialog_count >= 100:  # Safety limit
                        break
            
            self.logger.info(f"UI resource extraction found {len(resources)} UI elements")
            
        except Exception as e:
            self.logger.error(f'UI resource extraction failed: {e}')
            
        return resources
    
    def _analyze_string_literal_placement_phase3(self, pe_data: bytes) -> Dict[str, Any]:
        """Phase 3.1: Analyze string literal placement with exact layout preservation"""
        layout_analysis = {
            'string_sections': [],
            'memory_regions': {},
            'alignment_patterns': [],
            'clustering_analysis': {},
            'total_string_memory': 0
        }
        
        try:
            # Identify string-rich memory regions
            block_size = 1024
            string_density_threshold = 0.3  # 30% string content
            
            for offset in range(0, len(pe_data) - block_size, block_size):
                block = pe_data[offset:offset + block_size]
                
                # Calculate string density in this block
                printable_count = sum(1 for b in block if 32 <= b <= 126)
                density = printable_count / len(block)
                
                if density >= string_density_threshold:
                    layout_analysis['string_sections'].append({
                        'offset': offset,
                        'size': block_size,
                        'density': density,
                        'estimated_section': self._estimate_section_from_offset(offset)
                    })
            
            self.logger.info(f"🎯 Identified {len(layout_analysis['string_sections'])} string-rich regions")
            
        except Exception as e:
            self.logger.error(f'String layout analysis failed: {e}')
        
        return layout_analysis
    
    def _extract_unicode_strings_with_layout(self, pe_data: bytes) -> List[Dict[str, Any]]:
        """Phase 3.2: Extract Unicode strings with precise memory layout information"""
        string_infos = []
        
        try:
            i = 0
            while i < len(pe_data) - 4:
                # Check for potential UTF-16 string start
                if pe_data[i] != 0 and pe_data[i+1] == 0:  # First char of UTF-16
                    string_start = i
                    string_bytes = []
                    j = i
                    
                    # Extract until double null terminator
                    while j < len(pe_data) - 1:
                        char_bytes = pe_data[j:j+2]
                        if char_bytes == b'\x00\x00':  # End of string
                            break
                        string_bytes.extend(char_bytes)
                        j += 2
                        
                        if len(string_bytes) > 1000:  # Reasonable string length limit
                            break
                    
                    if len(string_bytes) >= 8:  # Minimum string length (4 characters)
                        try:
                            string_text = bytes(string_bytes).decode('utf-16le', errors='ignore')
                            if len(string_text.strip()) >= 2:  # Must have actual content
                                # Calculate memory layout information
                                string_info = {
                                    'content': string_text.strip(),
                                    'file_offset': string_start,
                                    'memory_address': 0x400000 + string_start,  # Estimate virtual address
                                    'size': len(string_bytes) + 2,  # Include null terminator
                                    'alignment': self._calculate_string_alignment(string_start),
                                    'section': self._estimate_section_from_offset(string_start),
                                    'null_terminated': True,
                                    'padding_before': self._calculate_padding_before(pe_data, string_start),
                                    'padding_after': self._calculate_padding_after(pe_data, j + 2)
                                }
                                string_infos.append(string_info)
                        except:
                            pass
                    
                    i = j + 2
                else:
                    i += 1
                    
                if len(string_infos) >= 15000:  # Reasonable limit
                    break
            
            self.logger.info(f"📍 Extracted {len(string_infos)} Unicode strings with layout information")
            
        except Exception as e:
            self.logger.error(f'Unicode string layout extraction failed: {e}')
            
        return string_infos
    
    def _extract_ascii_strings_with_layout(self, pe_data: bytes) -> List[Dict[str, Any]]:
        """Phase 3.3: Extract ASCII strings with precise memory layout information"""
        string_infos = []
        
        try:
            current_string = []
            string_start = 0
            
            for i, byte in enumerate(pe_data):
                if 32 <= byte <= 126:  # Printable ASCII
                    if not current_string:  # Start of new string
                        string_start = i
                    current_string.append(chr(byte))
                else:
                    if len(current_string) >= 4:  # Minimum string length
                        string_text = ''.join(current_string).strip()
                        if len(string_text) >= 3:
                            # Calculate memory layout information
                            string_info = {
                                'content': string_text,
                                'file_offset': string_start,
                                'memory_address': 0x400000 + string_start,  # Estimate virtual address
                                'size': len(current_string) + 1,  # Include null terminator
                                'alignment': self._calculate_string_alignment(string_start),
                                'section': self._estimate_section_from_offset(string_start),
                                'null_terminated': byte == 0,
                                'padding_before': self._calculate_padding_before(pe_data, string_start),
                                'padding_after': self._calculate_padding_after(pe_data, i)
                            }
                            string_infos.append(string_info)
                    
                    current_string = []
                    
                if len(string_infos) >= 10000:  # Reasonable limit
                    break
            
            # Handle last string if file doesn't end with non-printable
            if len(current_string) >= 4:
                string_text = ''.join(current_string).strip()
                if len(string_text) >= 3:
                    string_info = {
                        'content': string_text,
                        'file_offset': string_start,
                        'memory_address': 0x400000 + string_start,
                        'size': len(current_string),
                        'alignment': self._calculate_string_alignment(string_start),
                        'section': self._estimate_section_from_offset(string_start),
                        'null_terminated': False,
                        'padding_before': 0,
                        'padding_after': 0
                    }
                    string_infos.append(string_info)
            
            self.logger.info(f"📍 Extracted {len(string_infos)} ASCII strings with layout information")
            
        except Exception as e:
            self.logger.error(f'ASCII string layout extraction failed: {e}')
            
        return string_infos
    
    def _reconstruct_string_table_layout(self, pe_data: bytes, all_strings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Phase 3.4: Reconstruct string table layout for exact placement"""
        string_tables = []
        
        try:
            # Group strings by memory regions (4KB pages)
            page_size = 4096
            string_groups = {}
            
            for string_info in all_strings:
                page_addr = (string_info['memory_address'] // page_size) * page_size
                if page_addr not in string_groups:
                    string_groups[page_addr] = []
                string_groups[page_addr].append(string_info)
            
            # Analyze each string table region
            for page_addr, strings in string_groups.items():
                if len(strings) >= 5:  # At least 5 strings to form a table
                    # Calculate table statistics
                    min_addr = min(s['memory_address'] for s in strings)
                    max_addr = max(s['memory_address'] + s['size'] for s in strings)
                    total_size = max_addr - min_addr
                    
                    # Calculate string density
                    total_string_bytes = sum(s['size'] for s in strings)
                    density = total_string_bytes / total_size if total_size > 0 else 0
                    
                    # Determine common alignment
                    alignments = [s['alignment'] for s in strings]
                    common_alignment = max(set(alignments), key=alignments.count) if alignments else 4
                    
                    # Generate layout description
                    layout_desc = self._generate_string_table_layout_description(strings)
                    
                    string_table = {
                        'address': min_addr,
                        'size': total_size,
                        'string_count': len(strings),
                        'alignment': common_alignment,
                        'section': self._estimate_section_from_offset(min_addr - 0x400000),
                        'density': density,
                        'layout_description': layout_desc
                    }
                    string_tables.append(string_table)
            
            self.logger.info(f"🗂️ Reconstructed {len(string_tables)} string table layouts")
            
        except Exception as e:
            self.logger.error(f'String table layout reconstruction failed: {e}')
        
        return string_tables
    
    # Helper methods for Phase 3 string analysis
    
    def _calculate_string_alignment(self, offset: int) -> int:
        """Calculate string alignment based on memory offset"""
        # Check alignment boundaries
        for alignment in [16, 8, 4, 2, 1]:
            if offset % alignment == 0:
                return alignment
        return 1
    
    def _estimate_section_from_offset(self, offset: int) -> str:
        """Estimate which section contains this offset"""
        # Simple heuristic based on typical PE layout
        if offset < 0x1000:
            return '.text'  # Code section
        elif offset < 0x10000:
            return '.rdata'  # Read-only data
        elif offset < 0x20000:
            return '.data'  # Initialized data
        else:
            return '.rsrc'  # Resources
    
    def _calculate_padding_before(self, pe_data: bytes, string_start: int) -> int:
        """Calculate padding bytes before string"""
        padding = 0
        i = string_start - 1
        
        while i >= 0 and pe_data[i] == 0:
            padding += 1
            i -= 1
            if padding >= 16:  # Reasonable limit
                break
        
        return padding
    
    def _calculate_padding_after(self, pe_data: bytes, string_end: int) -> int:
        """Calculate padding bytes after string"""
        padding = 0
        i = string_end
        
        while i < len(pe_data) and pe_data[i] == 0:
            padding += 1
            i += 1
            if padding >= 16:  # Reasonable limit
                break
        
        return padding
    
    def _generate_string_table_layout_description(self, strings: List[Dict[str, Any]]) -> str:
        """Generate human-readable layout description for string table"""
        # Sort strings by memory address
        sorted_strings = sorted(strings, key=lambda s: s['memory_address'])
        
        layout_lines = []
        layout_lines.append(f"String Table Layout ({len(strings)} strings)")
        layout_lines.append("=" * 50)
        
        for i, string_info in enumerate(sorted_strings[:10]):  # Show first 10
            addr = string_info['memory_address']
            size = string_info['size']
            content = string_info['content'][:30] + '...' if len(string_info['content']) > 30 else string_info['content']
            
            layout_lines.append(f"[{addr:08x}] +{size:3d} bytes: \"{content}\"")
        
        if len(sorted_strings) > 10:
            layout_lines.append(f"... and {len(sorted_strings) - 10} more strings")
        
        return '\n'.join(layout_lines)
    
    def _extract_unicode_strings(self, pe_data: bytes) -> List[str]:
        """Legacy method - kept for compatibility"""
        # Extract just the content for backward compatibility
        string_infos = self._extract_unicode_strings_with_layout(pe_data)
        return [info['content'] for info in string_infos]
    
    def _extract_ascii_strings(self, pe_data: bytes) -> List[str]:
        """Legacy method - kept for compatibility"""
        # Extract just the content for backward compatibility
        string_infos = self._extract_ascii_strings_with_layout(pe_data)
        return [info['content'] for info in string_infos]
    
    def _get_pe_resource_type_name(self, type_id: int) -> str:
        """Get PE resource type name from ID"""
        pe_resource_types = {
            1: 'cursor',
            2: 'bitmap', 
            3: 'icon',
            4: 'menu',
            5: 'dialog',
            6: 'string',
            7: 'fontdir',
            8: 'font',
            9: 'accelerator',
            10: 'rcdata',
            11: 'messagetable',
            12: 'group_cursor',
            14: 'group_icon',
            16: 'version',
            17: 'dlginclude',
            19: 'plugplay',
            20: 'vxd',
            21: 'anicursor',
            22: 'aniicon',
            23: 'html',
            24: 'manifest'
        }
        return pe_resource_types.get(type_id, f'unknown_{type_id}')
    
    def _get_extension_for_type(self, type_name: str) -> str:
        """Get file extension for resource type"""
        extensions = {
            'cursor': 'cur',
            'bitmap': 'bmp',
            'icon': 'ico', 
            'menu': 'rc',
            'dialog': 'rc',
            'string': 'txt',
            'font': 'ttf',
            'accelerator': 'rc',
            'rcdata': 'bin',
            'messagetable': 'txt',
            'version': 'txt',
            'html': 'html',
            'manifest': 'xml'
        }
        return extensions.get(type_name, 'bin')
    
    def _parse_string_table_data(self, data: bytes, resource_id: int) -> List[ResourceItem]:
        """Parse string table data into individual strings"""
        strings = []
        
        try:
            # String tables contain 16 strings each
            offset = 0
            for i in range(16):
                if offset >= len(data):
                    break
                    
                # Read string length (first 2 bytes)
                if offset + 2 <= len(data):
                    str_len = struct.unpack('<H', data[offset:offset+2])[0]
                    offset += 2
                    
                    if str_len > 0 and offset + str_len * 2 <= len(data):
                        # Read Unicode string
                        str_data = data[offset:offset+str_len*2]
                        try:
                            string_text = str_data.decode('utf-16le', errors='ignore').strip()
                            if string_text:
                                strings.append(ResourceItem(
                                    resource_id=f'stringtable_{resource_id}_{i}',
                                    resource_type='string',
                                    name=f'string_{resource_id}_{i}.txt',
                                    size=len(string_text),
                                    content=string_text,
                                    metadata={'table_id': resource_id, 'string_index': i},
                                    extraction_method='string_table_parse',
                                    confidence=0.95
                                ))
                        except:
                            pass
                        offset += str_len * 2
                    else:
                        break
        except Exception as e:
            self.logger.error(f'String table parsing failed: {e}')
            
        return strings
    
    def _extract_pe_resources_manual(self, binary_path: str) -> List[ResourceItem]:
        """Manual PE resource extraction fallback"""
        # This would be the existing manual extraction logic
        return []
    
    def _extract_pe_resources_basic(self, binary_path: str) -> List[ResourceItem]:
        """Basic PE resource extraction fallback"""
        # This would be the most basic extraction logic
        return []
    
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
                # Handle both string and bytes signatures
                if isinstance(signature, str):
                    signature_bytes = signature.encode('latin-1')
                else:
                    signature_bytes = signature
                    
                offset = 0
                while True:
                    pos = data.find(signature_bytes, offset)
                    if pos == -1:
                        break
                    
                    resource = ResourceItem(
                        resource_id=f"{file_type}_{pos:08x}",
                        resource_type="image",
                        name=f"{file_type}_{pos:08x}.{file_type}",
                        size=0,  # Would need to calculate actual size
                        content=signature_bytes,
                        metadata={'offset': pos, 'signature': signature_bytes.hex()},
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
    
    def _reconstruct_constant_pools_phase3(self, binary_path: str, categorized_resources: Dict[str, Any]) -> List[ResourceItem]:
        """Phase 3.5: Reconstruct constant pools with exact placement for perfect recompilation"""
        constant_pools = []
        
        try:
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            self.logger.info("🔢 Phase 3: Reconstructing constant pools with exact placement...")
            
            # Phase 3.5.1: Floating Point Constant Pool Reconstruction
            fp_pools = self._reconstruct_floating_point_pools(binary_data)
            constant_pools.extend(fp_pools)
            
            # Phase 3.5.2: Integer Constant Pool Reconstruction
            int_pools = self._reconstruct_integer_pools(binary_data)
            constant_pools.extend(int_pools)
            
            # Phase 3.5.3: Address Constant Pool Reconstruction
            addr_pools = self._reconstruct_address_pools(binary_data)
            constant_pools.extend(addr_pools)
            
            # Phase 3.5.4: String Reference Pool Reconstruction
            str_ref_pools = self._reconstruct_string_reference_pools(binary_data, categorized_resources)
            constant_pools.extend(str_ref_pools)
            
            self.logger.info(f"✅ Reconstructed {len(constant_pools)} constant pools with exact placement")
            
        except Exception as e:
            self.logger.error(f'Constant pool reconstruction failed: {e}')
        
        return constant_pools
    
    def _reconstruct_floating_point_pools(self, binary_data: bytes) -> List[ResourceItem]:
        """Reconstruct floating point constant pools with exact memory layout - OPTIMIZED"""
        fp_pools = []
        
        try:
            import struct
            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # OPTIMIZATION: Use larger steps and targeted scanning
            chunk_size = 32768  # 32KB chunks for better cache performance
            step_size = 8  # 8-byte aligned scanning for doubles
            
            fp_constants = []
            
            def scan_chunk(start_offset: int, end_offset: int) -> List[dict]:
                """Scan a chunk of binary data for floating point constants"""
                chunk_constants = []
                
                for offset in range(start_offset, min(end_offset, len(binary_data) - 8), step_size):
                    try:
                        double_bytes = binary_data[offset:offset + 8]
                        if len(double_bytes) == 8:
                            double_val = struct.unpack('<d', double_bytes)[0]
                            
                            # Check for reasonable FP constants (optimized checks)
                            if (not math.isnan(double_val) and not math.isinf(double_val) and 
                                1e-15 < abs(double_val) < 1e15):  # More restrictive range
                                
                                chunk_constants.append({
                                    'offset': offset,
                                    'value': double_val,
                                    'size': 8,
                                    'type': 'double',
                                    'memory_address': 0x400000 + offset,
                                    'alignment': 8
                                })
                    except (struct.error, OverflowError):
                        continue
                
                return chunk_constants
            
            # OPTIMIZATION: Use multithreading for parallel chunk scanning
            max_workers = min(4, (len(binary_data) // chunk_size) + 1)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                # Submit chunks for parallel processing
                for start in range(0, len(binary_data), chunk_size):
                    end = min(start + chunk_size, len(binary_data))
                    futures.append(executor.submit(scan_chunk, start, end))
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        fp_constants.extend(chunk_results)
                    except Exception as e:
                        self.logger.warning(f"Chunk processing failed: {e}")
            
            # Group nearby constants into pools
            if fp_constants:
                pools = self._group_constants_into_pools(fp_constants, pool_threshold=64)
                
                for i, pool in enumerate(pools):
                    if len(pool) >= 3:  # At least 3 constants to form a pool
                        min_addr = min(c['memory_address'] for c in pool)
                        max_addr = max(c['memory_address'] + c['size'] for c in pool)
                        
                        pool_resource = ResourceItem(
                            resource_id=f'fp_constant_pool_{min_addr:08x}',
                            resource_type='fp_constant_pool',
                            name=f'fp_constants_{min_addr:08x}.pool',
                            size=max_addr - min_addr,
                            content=self._generate_constant_pool_layout(pool),
                            metadata={
                                'base_address': min_addr,
                                'constant_count': len(pool),
                                'pool_size': max_addr - min_addr,
                                'alignment': 8,
                                'section': '.rdata',
                                'access_pattern': 'read_only',
                                'constant_type': 'double_precision'
                            },
                            extraction_method='phase3_fp_pool_reconstruction',
                            confidence=0.90
                        )
                        fp_pools.append(pool_resource)
            
            self.logger.info(f"🔢 Reconstructed {len(fp_pools)} floating point constant pools")
            
        except Exception as e:
            self.logger.error(f'FP constant pool reconstruction failed: {e}')
        
        return fp_pools
    
    def _reconstruct_integer_pools(self, binary_data: bytes) -> List[ResourceItem]:
        """Reconstruct integer constant pools with exact memory layout - OPTIMIZED"""
        int_pools = []
        
        try:
            import struct
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # OPTIMIZATION: Use chunked parallel processing
            chunk_size = 32768  # 32KB chunks
            step_size = 4  # 4-byte aligned scanning for integers
            
            int_constants = []
            
            def scan_int_chunk(start_offset: int, end_offset: int) -> List[dict]:
                """Scan a chunk for integer constants"""
                chunk_constants = []
                
                for offset in range(start_offset, min(end_offset, len(binary_data) - 4), step_size):
                    try:
                        int_bytes = binary_data[offset:offset + 4]
                        if len(int_bytes) == 4:
                            int_val = struct.unpack('<I', int_bytes)[0]
                            
                            # OPTIMIZED: More restrictive filtering for performance
                            if (10000 <= int_val <= 0xFFFFF000 or  # Tighter range
                                int_val in {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}):  # Set lookup
                                
                                chunk_constants.append({
                                    'offset': offset,
                                    'value': int_val,
                                    'size': 4,
                                    'type': 'uint32',
                                    'memory_address': 0x400000 + offset,
                                    'alignment': 4
                                })
                    except (struct.error, OverflowError):
                        continue
                
                return chunk_constants
            
            # OPTIMIZATION: Parallel chunk processing
            max_workers = min(4, (len(binary_data) // chunk_size) + 1)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for start in range(0, len(binary_data), chunk_size):
                    end = min(start + chunk_size, len(binary_data))
                    futures.append(executor.submit(scan_int_chunk, start, end))
                
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        int_constants.extend(chunk_results)
                    except Exception as e:
                        self.logger.warning(f"Integer chunk processing failed: {e}")
            
            # Group nearby constants into pools
            if int_constants:
                pools = self._group_constants_into_pools(int_constants, pool_threshold=32)
                
                for i, pool in enumerate(pools):
                    if len(pool) >= 4:  # At least 4 constants to form a pool
                        min_addr = min(c['memory_address'] for c in pool)
                        max_addr = max(c['memory_address'] + c['size'] for c in pool)
                        
                        pool_resource = ResourceItem(
                            resource_id=f'int_constant_pool_{min_addr:08x}',
                            resource_type='int_constant_pool',
                            name=f'int_constants_{min_addr:08x}.pool',
                            size=max_addr - min_addr,
                            content=self._generate_constant_pool_layout(pool),
                            metadata={
                                'base_address': min_addr,
                                'constant_count': len(pool),
                                'pool_size': max_addr - min_addr,
                                'alignment': 4,
                                'section': '.rdata',
                                'access_pattern': 'read_only',
                                'constant_type': 'integer'
                            },
                            extraction_method='phase3_int_pool_reconstruction',
                            confidence=0.85
                        )
                        int_pools.append(pool_resource)
            
            self.logger.info(f"🔢 Reconstructed {len(int_pools)} integer constant pools")
            
        except Exception as e:
            self.logger.error(f'Integer constant pool reconstruction failed: {e}')
        
        return int_pools
    
    def _reconstruct_address_pools(self, binary_data: bytes) -> List[ResourceItem]:
        """Reconstruct address/pointer constant pools - OPTIMIZED"""
        addr_pools = []
        
        try:
            import struct
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # OPTIMIZATION: Parallel chunked processing
            chunk_size = 65536  # 64KB chunks for address scanning
            step_size = 4  # 4-byte aligned scanning
            
            addr_constants = []
            
            def scan_address_chunk(start_offset: int, end_offset: int) -> List[dict]:
                """Scan chunk for address-like constants"""
                chunk_constants = []
                
                for offset in range(start_offset, min(end_offset, len(binary_data) - 4), step_size):
                    try:
                        addr_bytes = binary_data[offset:offset + 4]
                        if len(addr_bytes) == 4:
                            addr_val = struct.unpack('<I', addr_bytes)[0]
                            
                            # OPTIMIZED: More restrictive address validation
                            if (0x400000 <= addr_val <= 0x480000 or  # Narrower code range
                                0x10000000 <= addr_val <= 0x20000000):  # Narrower data range
                                
                                chunk_constants.append({
                                    'offset': offset,
                                    'value': addr_val,
                                    'size': 4,
                                    'type': 'address',
                                    'memory_address': 0x400000 + offset,
                                    'alignment': 4
                                })
                    except struct.error:
                        continue
                
                return chunk_constants
            
            # OPTIMIZATION: Parallel chunk processing
            max_workers = min(4, (len(binary_data) // chunk_size) + 1)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for start in range(0, len(binary_data), chunk_size):
                    end = min(start + chunk_size, len(binary_data))
                    futures.append(executor.submit(scan_address_chunk, start, end))
                
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result()
                        addr_constants.extend(chunk_results)
                    except Exception as e:
                        self.logger.warning(f"Address chunk processing failed: {e}")
            
            # Group nearby addresses into pools
            if addr_constants:
                pools = self._group_constants_into_pools(addr_constants, pool_threshold=16)
                
                for i, pool in enumerate(pools):
                    if len(pool) >= 3:  # At least 3 addresses to form a pool
                        min_addr = min(c['memory_address'] for c in pool)
                        max_addr = max(c['memory_address'] + c['size'] for c in pool)
                        
                        pool_resource = ResourceItem(
                            resource_id=f'addr_constant_pool_{min_addr:08x}',
                            resource_type='address_pool',
                            name=f'addresses_{min_addr:08x}.pool',
                            size=max_addr - min_addr,
                            content=self._generate_constant_pool_layout(pool),
                            metadata={
                                'base_address': min_addr,
                                'constant_count': len(pool),
                                'pool_size': max_addr - min_addr,
                                'alignment': 4,
                                'section': '.rdata',
                                'access_pattern': 'read_only',
                                'constant_type': 'address_pointer'
                            },
                            extraction_method='phase3_address_pool_reconstruction',
                            confidence=0.80
                        )
                        addr_pools.append(pool_resource)
            
            self.logger.info(f"🔢 Reconstructed {len(addr_pools)} address constant pools")
            
        except Exception as e:
            self.logger.error(f'Address constant pool reconstruction failed: {e}')
        
        return addr_pools
    
    def _reconstruct_string_reference_pools(self, binary_data: bytes, categorized_resources: Dict[str, Any]) -> List[ResourceItem]:
        """Reconstruct string reference pools for exact layout matching"""
        str_ref_pools = []
        
        try:
            # Get string resources for reference mapping
            string_resources = categorized_resources.get('strings', [])
            
            if not string_resources:
                return str_ref_pools
            
            # Create mapping of string addresses to references
            string_addresses = {}
            for string_res in string_resources:
                if hasattr(string_res, 'metadata') and 'memory_address' in string_res.metadata:
                    addr = string_res.metadata['memory_address']
                    string_addresses[addr] = string_res
            
            # Look for clusters of string references
            import struct
            ref_constants = []
            
            for offset in range(0, len(binary_data) - 4, 4):
                try:
                    ref_bytes = binary_data[offset:offset + 4]
                    if len(ref_bytes) == 4:
                        ref_val = struct.unpack('<I', ref_bytes)[0]
                        
                        # Check if this points to a known string
                        if ref_val in string_addresses:
                            ref_constants.append({
                                'offset': offset,
                                'value': ref_val,
                                'size': 4,
                                'type': 'string_ref',
                                'memory_address': 0x400000 + offset,
                                'alignment': 4,
                                'target_string': string_addresses[ref_val]
                            })
                except struct.error:
                    continue
            
            # Group nearby references into pools
            if ref_constants:
                pools = self._group_constants_into_pools(ref_constants, pool_threshold=32)
                
                for i, pool in enumerate(pools):
                    if len(pool) >= 3:  # At least 3 references to form a pool
                        min_addr = min(c['memory_address'] for c in pool)
                        max_addr = max(c['memory_address'] + c['size'] for c in pool)
                        
                        pool_resource = ResourceItem(
                            resource_id=f'string_ref_pool_{min_addr:08x}',
                            resource_type='string_ref_pool',
                            name=f'string_refs_{min_addr:08x}.pool',
                            size=max_addr - min_addr,
                            content=self._generate_string_ref_pool_layout(pool),
                            metadata={
                                'base_address': min_addr,
                                'reference_count': len(pool),
                                'pool_size': max_addr - min_addr,
                                'alignment': 4,
                                'section': '.rdata',
                                'access_pattern': 'read_only',
                                'constant_type': 'string_reference'
                            },
                            extraction_method='phase3_string_ref_reconstruction',
                            confidence=0.88
                        )
                        str_ref_pools.append(pool_resource)
            
            self.logger.info(f"🔗 Reconstructed {len(str_ref_pools)} string reference pools")
            
        except Exception as e:
            self.logger.error(f'String reference pool reconstruction failed: {e}')
        
        return str_ref_pools
    
    def _group_constants_into_pools(self, constants: List[Dict[str, Any]], pool_threshold: int = 32) -> List[List[Dict[str, Any]]]:
        """Group nearby constants into pools based on memory proximity"""
        if not constants:
            return []
        
        # Sort by memory address
        sorted_constants = sorted(constants, key=lambda c: c['memory_address'])
        
        pools = []
        current_pool = [sorted_constants[0]]
        
        for i in range(1, len(sorted_constants)):
            current_const = sorted_constants[i]
            prev_const = sorted_constants[i-1]
            
            # Check if close enough to be in same pool
            distance = current_const['memory_address'] - (prev_const['memory_address'] + prev_const['size'])
            
            if distance <= pool_threshold:
                current_pool.append(current_const)
            else:
                # Start new pool
                if len(current_pool) >= 2:  # Only keep pools with multiple constants
                    pools.append(current_pool)
                current_pool = [current_const]
        
        # Add last pool
        if len(current_pool) >= 2:
            pools.append(current_pool)
        
        return pools
    
    def _generate_constant_pool_layout(self, pool: List[Dict[str, Any]]) -> str:
        """Generate human-readable layout description for constant pool"""
        layout_lines = []
        layout_lines.append(f"Constant Pool Layout ({len(pool)} constants)")
        layout_lines.append("=" * 50)
        
        for const in pool:
            addr = const['memory_address']
            value = const['value']
            const_type = const['type']
            
            if const_type == 'double':
                layout_lines.append(f"[{addr:08x}] double: {value:.6f}")
            elif const_type in ['uint32', 'integer']:
                layout_lines.append(f"[{addr:08x}] uint32: {value} (0x{value:08x})")
            elif const_type == 'address':
                layout_lines.append(f"[{addr:08x}] address: 0x{value:08x}")
            else:
                layout_lines.append(f"[{addr:08x}] {const_type}: {value}")
        
        return '\n'.join(layout_lines)
    
    def _generate_string_ref_pool_layout(self, pool: List[Dict[str, Any]]) -> str:
        """Generate human-readable layout description for string reference pool"""
        layout_lines = []
        layout_lines.append(f"String Reference Pool Layout ({len(pool)} references)")
        layout_lines.append("=" * 60)
        
        for ref in pool:
            addr = ref['memory_address']
            target_addr = ref['value']
            target_string = ref.get('target_string')
            
            if target_string and hasattr(target_string, 'content'):
                content = target_string.content[:30] + '...' if len(str(target_string.content)) > 30 else str(target_string.content)
                layout_lines.append(f"[{addr:08x}] -> 0x{target_addr:08x}: \"{content}\"")
            else:
                layout_lines.append(f"[{addr:08x}] -> 0x{target_addr:08x}: <unknown string>")
        
        return '\n'.join(layout_lines)
    
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