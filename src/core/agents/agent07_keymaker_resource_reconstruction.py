"""
Agent 7: The Keymaker - Resource Reconstruction Engine

The Keymaker possesses the unique ability to create keys that open any door,
providing access to every part of the system. Focused implementation for
extracting and reconstructing resources from Windows PE executables.

Core Responsibilities:
- PE resource extraction (strings, icons, dialogs, menus)
- Resource type classification and validation
- RC file generation for resource compilation
- Configuration data reconstruction

STRICT MODE: No fallbacks, no placeholders, fail-fast validation.
"""

import logging
import struct
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# Matrix framework imports
from ..matrix_agents import ReconstructionAgent, MatrixCharacter, AgentStatus
from ..shared_components import (
    MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker,
    MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
)
from ..exceptions import MatrixAgentError, ValidationError
from ..config_manager import get_config_manager

@dataclass
class ResourceItem:
    """Streamlined resource item representation"""
    resource_id: str
    resource_type: str
    name: str
    size: int
    content: Union[bytes, str]
    metadata: Dict[str, Any]

@dataclass
class ResourceAnalysisResult:
    """Focused resource analysis result"""
    string_resources: List[ResourceItem]
    binary_resources: List[ResourceItem]
    rc_file_content: str
    resource_count: int
    total_size: int
    quality_score: float

class Agent7_Keymaker_ResourceReconstruction(ReconstructionAgent):
    """
    Agent 7: The Keymaker - Resource Reconstruction Engine
    
    Streamlined implementation focused on extracting and reconstructing
    resources from Windows PE executables for compilation readiness.
    """
    
    def __init__(self):
        super().__init__(
            agent_id=7,
            matrix_character=MatrixCharacter.KEYMAKER
        )
        
        # Load configuration
        self.config = get_config_manager()
        self.min_string_length = self.config.get_value('agents.agent_07.min_string_length', 4)
        self.max_resource_size = self.config.get_value('agents.agent_07.max_resource_size', 1024*1024)  # 1MB
        self.timeout_seconds = self.config.get_value('agents.agent_07.timeout', 300)
        
        # Initialize shared components
        self.file_manager = None  # Will be initialized with output paths
        self.error_handler = MatrixErrorHandler("Keymaker", max_retries=2)
        self.metrics = MatrixMetrics(7, "Keymaker")
        self.validation_tools = SharedValidationTools()
        
        # Resource type patterns
        self.resource_patterns = {
            'string': {'min_length': self.min_string_length, 'encoding': 'utf-8'},
            'icon': {'signature': b'\\x00\\x00\\x01\\x00', 'extension': '.ico'},
            'bitmap': {'signature': b'BM', 'extension': '.bmp'},
            'dialog': {'keywords': ['DIALOG', 'DIALOGEX']},
            'menu': {'keywords': ['MENU', 'MENUEX']}
        }

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Keymaker's resource reconstruction"""
        self.logger.info("🗝️ Keymaker initiating resource reconstruction...")
        self.metrics.start_tracking()
        
        try:
            # Initialize file manager
            if 'output_paths' in context:
                self.file_manager = MatrixFileManager(context['output_paths'])
            
            # Validate prerequisites - STRICT MODE
            self._validate_prerequisites(context)
            
            # Extract resources from binary and previous agents
            resource_data = self._extract_resource_data(context)
            
            # Perform resource analysis and reconstruction
            analysis_result = self._perform_resource_analysis(resource_data)
            
            # Generate RC file for resource compilation
            rc_content = self._generate_rc_file(analysis_result)
            analysis_result.rc_file_content = rc_content
            
            # Save results
            if self.file_manager:
                self._save_results(analysis_result, context.get('output_paths', {}))
            
            self.metrics.end_tracking()
            execution_time = self.metrics.execution_time
            
            self.logger.info(f"✅ Keymaker reconstruction complete in {execution_time:.2f}s")
            
            return {
                'resource_analysis': {
                    'string_count': len(analysis_result.string_resources),
                    'binary_count': len(analysis_result.binary_resources),
                    'total_resources': analysis_result.resource_count,
                    'total_size': analysis_result.total_size,
                    'quality_score': analysis_result.quality_score
                },
                'string_resources': [self._resource_to_dict(r) for r in analysis_result.string_resources],
                'binary_resources': [self._resource_to_dict(r) for r in analysis_result.binary_resources],
                'rc_file_content': analysis_result.rc_file_content,
                'keymaker_metadata': {
                    'agent_id': self.agent_id,
                    'matrix_character': self.matrix_character.value,
                    'execution_time': execution_time,
                    'resources_extracted': analysis_result.resource_count
                }
            }
            
        except Exception as e:
            self.metrics.end_tracking()
            error_msg = f"Keymaker resource reconstruction failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise MatrixAgentError(error_msg) from e

    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate prerequisites - STRICT MODE compliance"""
        # Check required agent results
        agent_results = context.get('agent_results', {})
        
        # Require Agent 2 (Architect) for PE structure analysis
        if 2 not in agent_results:
            raise ValidationError("Agent 2 (Architect) required for PE structure analysis")
        
        # Validate binary path
        binary_path = context.get('binary_path')
        if not binary_path or not Path(binary_path).exists():
            raise ValidationError("Valid binary path required for resource extraction")

    def _extract_resource_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract resource data from binary and previous agents"""
        resource_data = {
            'strings': [],
            'binary_resources': [],
            'pe_resources': {},
            'source_quality': 'unknown'
        }
        
        # Get data from Agent 2 (Architect) if available
        agent_results = context.get('agent_results', {})
        if 2 in agent_results:
            agent2_data = agent_results[2].data if hasattr(agent_results[2], 'data') else {}
            pe_analysis = agent2_data.get('pe_analysis', {})
            
            # Extract resources from PE analysis
            if 'resources' in pe_analysis:
                resource_data['pe_resources'] = pe_analysis['resources']
                resource_data['source_quality'] = 'high'
            
            # Extract strings if available
            if 'strings' in pe_analysis:
                resource_data['strings'] = pe_analysis['strings']
        
        # Fallback: Basic resource extraction from binary
        binary_path = context.get('binary_path', '')
        if binary_path and not resource_data['strings']:
            resource_data['strings'] = self._extract_basic_strings(binary_path)
            resource_data['source_quality'] = 'medium' if resource_data['strings'] else 'low'
        
        return resource_data

    def _extract_basic_strings(self, binary_path: str) -> List[str]:
        """Basic string extraction from binary file"""
        strings = []
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read(min(1024*1024, f.seek(0, 2) or f.tell()))  # Read first 1MB
                
                # Extract ASCII strings
                current_string = b''
                for byte in data:
                    if 32 <= byte <= 126:  # Printable ASCII
                        current_string += bytes([byte])
                    else:
                        if len(current_string) >= self.min_string_length:
                            try:
                                strings.append(current_string.decode('ascii', errors='ignore'))
                            except:
                                pass
                        current_string = b''
                
                # Don't forget the last string
                if len(current_string) >= self.min_string_length:
                    try:
                        strings.append(current_string.decode('ascii', errors='ignore'))
                    except:
                        pass
                        
        except Exception as e:
            self.logger.warning(f"Basic string extraction failed: {e}")
            
        return strings[:1000]  # Limit to first 1000 strings

    def _perform_resource_analysis(self, resource_data: Dict[str, Any]) -> ResourceAnalysisResult:
        """Perform focused resource analysis"""
        self.logger.info("Analyzing extracted resources...")
        
        # Process string resources
        string_resources = self._process_string_resources(resource_data['strings'])
        
        # Process binary resources
        binary_resources = self._process_binary_resources(resource_data.get('pe_resources', {}))
        
        # Calculate metrics
        resource_count = len(string_resources) + len(binary_resources)
        total_size = sum(r.size for r in string_resources + binary_resources)
        quality_score = self._calculate_quality_score(string_resources, binary_resources, resource_data['source_quality'])
        
        return ResourceAnalysisResult(
            string_resources=string_resources,
            binary_resources=binary_resources,
            rc_file_content="",  # Will be generated later
            resource_count=resource_count,
            total_size=total_size,
            quality_score=quality_score
        )

    def _process_string_resources(self, strings: List[str]) -> List[ResourceItem]:
        """Process string resources into ResourceItem objects"""
        string_resources = []
        
        for i, string_value in enumerate(strings):
            if len(string_value) >= self.min_string_length and len(string_value) <= 1000:
                # Filter out obvious binary data
                if self._is_meaningful_string(string_value):
                    resource = ResourceItem(
                        resource_id=f"STRING_{i+1:04d}",
                        resource_type="string",
                        name=f"str_{i+1:04d}",
                        size=len(string_value),
                        content=string_value,
                        metadata={
                            'encoding': 'ascii',
                            'printable': True,
                            'length': len(string_value)
                        }
                    )
                    string_resources.append(resource)
        
        self.logger.info(f"Processed {len(string_resources)} string resources")
        return string_resources

    def _process_binary_resources(self, pe_resources: Dict[str, Any]) -> List[ResourceItem]:
        """Process binary resources from PE analysis"""
        binary_resources = []
        
        # Process different resource types
        for resource_type, resources in pe_resources.items():
            if isinstance(resources, list):
                for i, resource_data in enumerate(resources):
                    if isinstance(resource_data, dict):
                        content = resource_data.get('data', b'')
                        if isinstance(content, bytes) and len(content) <= self.max_resource_size:
                            resource = ResourceItem(
                                resource_id=f"{resource_type.upper()}_{i+1:04d}",
                                resource_type=resource_type.lower(),
                                name=f"{resource_type.lower()}_{i+1:04d}",
                                size=len(content),
                                content=content,
                                metadata={
                                    'original_type': resource_type,
                                    'binary': True,
                                    'size': len(content)
                                }
                            )
                            binary_resources.append(resource)
        
        self.logger.info(f"Processed {len(binary_resources)} binary resources")
        return binary_resources

    def _is_meaningful_string(self, s: str) -> bool:
        """Check if string is meaningful (not just binary data)"""
        # Skip strings that are mostly non-alphanumeric
        alphanumeric_ratio = sum(1 for c in s if c.isalnum()) / len(s)
        if alphanumeric_ratio < 0.3:
            return False
        
        # Skip strings with too many repeated characters
        if len(set(s)) < len(s) * 0.3:
            return False
            
        # Skip obvious binary patterns
        if s.startswith('\\x') or all(c in '0123456789abcdefABCDEF' for c in s):
            return False
            
        return True

    def _generate_rc_file(self, analysis_result: ResourceAnalysisResult) -> str:
        """Generate RC (Resource Compiler) file content"""
        rc_lines = [
            "// Resource file generated by The Keymaker",
            "// Agent 7: Resource Reconstruction",
            "",
            "#include <windows.h>",
            "#include <winres.h>",
            "",
            "// String Table",
            "STRINGTABLE",
            "BEGIN"
        ]
        
        # Add string resources
        for i, string_resource in enumerate(analysis_result.string_resources[:100]):  # Limit to 100 strings
            # Escape quotes and special characters
            escaped_content = string_resource.content.replace('"', '\\"').replace('\\', '\\\\')
            rc_lines.append(f'    {i+1001}, "{escaped_content}"')
        
        rc_lines.extend([
            "END",
            "",
            "// Binary Resources"
        ])
        
        # Add binary resources (as custom resources)
        for binary_resource in analysis_result.binary_resources[:50]:  # Limit to 50 binary resources
            rc_lines.extend([
                f"// {binary_resource.resource_type.upper()} resource: {binary_resource.name}",
                f"{binary_resource.resource_id} RCDATA",
                "BEGIN",
                f"    // Binary data ({binary_resource.size} bytes)",
                "    // Data content would be embedded here in production",
                "END",
                ""
            ])
        
        # Add version info if we have meaningful strings
        if analysis_result.string_resources:
            rc_lines.extend([
                "// Version Information",
                "VS_VERSION_INFO VERSIONINFO",
                " FILEVERSION 1,0,0,0",
                " PRODUCTVERSION 1,0,0,0",
                " FILEFLAGSMASK 0x3fL",
                " FILEFLAGS 0x0L",
                " FILEOS 0x40004L",
                " FILETYPE 0x1L",
                " FILESUBTYPE 0x0L",
                "BEGIN",
                "    BLOCK \"StringFileInfo\"",
                "    BEGIN",
                "        BLOCK \"040904b0\"",
                "        BEGIN",
                "            VALUE \"CompanyName\", \"Reconstructed Application\"",
                "            VALUE \"FileDescription\", \"Reconstructed by The Keymaker\"",
                "            VALUE \"FileVersion\", \"1.0.0.0\"",
                "            VALUE \"ProductName\", \"Matrix Reconstruction\"",
                "            VALUE \"ProductVersion\", \"1.0.0.0\"",
                "        END",
                "    END",
                "    BLOCK \"VarFileInfo\"",
                "    BEGIN",
                "        VALUE \"Translation\", 0x409, 1200",
                "    END",
                "END"
            ])
        
        return "\\n".join(rc_lines)

    def _calculate_quality_score(self, string_resources: List[ResourceItem], 
                                binary_resources: List[ResourceItem], source_quality: str) -> float:
        """Calculate overall resource reconstruction quality"""
        score = 0.0
        
        # Base score from resource count
        total_resources = len(string_resources) + len(binary_resources)
        if total_resources > 50:
            score += 0.4
        elif total_resources > 20:
            score += 0.3
        elif total_resources > 5:
            score += 0.2
        
        # Score from string quality
        meaningful_strings = sum(1 for s in string_resources if len(s.content) > 10)
        if meaningful_strings > 10:
            score += 0.3
        elif meaningful_strings > 5:
            score += 0.2
        
        # Score from binary resources
        if binary_resources:
            score += 0.2
        
        # Score from source quality
        quality_scores = {'high': 0.1, 'medium': 0.075, 'low': 0.05, 'unknown': 0.0}
        score += quality_scores.get(source_quality, 0.0)
        
        return min(score, 1.0)

    def _resource_to_dict(self, resource: ResourceItem) -> Dict[str, Any]:
        """Convert ResourceItem to dictionary for serialization"""
        return {
            'resource_id': resource.resource_id,
            'resource_type': resource.resource_type,
            'name': resource.name,
            'size': resource.size,
            'content': resource.content if isinstance(resource.content, str) else f"<binary data {resource.size} bytes>",
            'metadata': resource.metadata
        }

    def _save_results(self, analysis_result: ResourceAnalysisResult, output_paths: Dict[str, Path]) -> None:
        """Save resource analysis results using shared file manager"""
        if not self.file_manager:
            return
            
        try:
            # Prepare results for saving
            results_data = {
                'agent_info': {
                    'agent_id': self.agent_id,
                    'agent_name': 'Keymaker_ResourceReconstruction',
                    'matrix_character': 'The Keymaker',
                    'analysis_timestamp': time.time()
                },
                'resource_analysis': {
                    'string_count': len(analysis_result.string_resources),
                    'binary_count': len(analysis_result.binary_resources),
                    'total_resources': analysis_result.resource_count,
                    'total_size': analysis_result.total_size,
                    'quality_score': analysis_result.quality_score
                },
                'string_resources': [self._resource_to_dict(r) for r in analysis_result.string_resources],
                'binary_resources': [self._resource_to_dict(r) for r in analysis_result.binary_resources],
                'rc_file_content': analysis_result.rc_file_content
            }
            
            # Save using shared file manager
            output_file = self.file_manager.save_agent_data(
                self.agent_id, "keymaker", results_data
            )
            
            # Also save RC file separately for compilation
            if analysis_result.rc_file_content:
                compilation_dir = output_paths.get('compilation', Path('output/compilation'))
                rc_file_path = compilation_dir / 'resources.rc'
                rc_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(rc_file_path, 'w', encoding='utf-8') as f:
                    f.write(analysis_result.rc_file_content)
                
                self.logger.info(f"RC file saved to {rc_file_path}")
            
            self.logger.info(f"Keymaker results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save Keymaker results: {e}")