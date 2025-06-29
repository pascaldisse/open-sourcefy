"""
Agent 8: Commander Locke - Build System Integration

The seasoned military commander who coordinates the integration of reconstruction
results into a coherent build system with actual dependencies.

Core Responsibilities:
- Integrate analysis results from all previous agents
- Generate header files with real function declarations
- Create build system files with actual DLL dependencies  
- Ensure compilation readiness for reconstructed code

STRICT MODE: No fallbacks, no placeholders, fail-fast validation.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
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
class BuildIntegrationResult:
    """Streamlined build integration result"""
    header_files: Dict[str, str]
    build_files: Dict[str, str]
    library_dependencies: List[str]
    integration_quality: float
    compilation_ready: bool
    integration_summary: Dict[str, Any]

class Agent8_CommanderLocke(ReconstructionAgent):
    """
    Agent 8: Commander Locke - Build System Integration
    
    Streamlined implementation focused on integrating agent results into
    a coherent build system with actual dependencies and header files.
    """
    
    def __init__(self):
        super().__init__(
            agent_id=8,
            matrix_character=MatrixCharacter.COMMANDER_LOCKE
        )
        
        # Load configuration
        self.config = get_config_manager()
        self.min_dll_dependencies = self.config.get_value('agents.agent_08.min_dll_dependencies', 5)
        self.min_functions = self.config.get_value('agents.agent_08.min_functions', 10)
        self.timeout_seconds = self.config.get_value('agents.agent_08.timeout', 300)
        
        # Initialize shared components
        self.file_manager = None  # Will be initialized with output paths
        self.error_handler = MatrixErrorHandler("CommanderLocke", max_retries=2)
        self.metrics = MatrixMetrics(8, "CommanderLocke")
        self.validation_tools = SharedValidationTools()
        
        # Required agent dependencies
        self.required_agents = [1, 5, 6, 7]  # Sentinel, Neo, Trainman, Keymaker

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Commander Locke's build system integration"""
        self.logger.info("🎖️ Commander Locke initiating build system integration...")
        self.metrics.start_tracking()
        
        try:
            # Initialize file manager
            if 'output_paths' in context:
                self.file_manager = MatrixFileManager(context['output_paths'])
            
            # Validate prerequisites - STRICT MODE
            self._validate_prerequisites(context)
            
            # Extract integration data from agents
            integration_data = self._extract_integration_data(context)
            
            # Perform build system integration
            integration_result = self._perform_build_integration(integration_data)
            
            # Save results
            if self.file_manager:
                self._save_results(integration_result, context.get('output_paths', {}))
            
            self.metrics.end_tracking()
            execution_time = self.metrics.execution_time
            
            self.logger.info(f"✅ Commander Locke integration complete in {execution_time:.2f}s")
            
            return {
                'build_integration': {
                    'header_count': len(integration_result.header_files),
                    'build_file_count': len(integration_result.build_files),
                    'library_count': len(integration_result.library_dependencies),
                    'compilation_ready': integration_result.compilation_ready,
                    'integration_quality': integration_result.integration_quality
                },
                'header_files': integration_result.header_files,
                'build_files': integration_result.build_files,
                'library_dependencies': integration_result.library_dependencies,
                'locke_metadata': {
                    'agent_id': self.agent_id,
                    'matrix_character': self.matrix_character.value,
                    'execution_time': execution_time,
                    'compilation_ready': integration_result.compilation_ready
                }
            }
            
        except Exception as e:
            self.metrics.end_tracking()
            error_msg = f"Commander Locke build integration failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise MatrixAgentError(error_msg) from e

    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate prerequisites using cache-first approach"""
        # Initialize shared_memory structure if not present
        shared_memory = context.get('shared_memory', {})
        if 'analysis_results' not in shared_memory:
            shared_memory['analysis_results'] = {}
        if 'binary_metadata' not in shared_memory:
            shared_memory['binary_metadata'] = {}
        
        # Update context with shared_memory reference
        context['shared_memory'] = shared_memory
        
        # Validate dependencies using cache-based approach
        dependency_met = self._load_agent_1_cache_data(context)
        
        if not dependency_met:
            # Check for Agent 1 in agent_results
            agent_results = context.get('agent_results', {})
            if 1 not in agent_results:
                # RULE 1 COMPLIANCE: Fail fast when dependencies not met
                raise MatrixAgentError(
                    "CRITICAL FAILURE: Agent 1 (Sentinel) results not available. "
                    "Commander Locke requires Sentinel binary analysis to proceed."
                )
            else:
                # Use existing agent results
                result = agent_results[1]
                if hasattr(result, 'data') and result.data:
                    shared_memory['analysis_results'][1] = {
                        'status': 'live',
                        'data': result.data
                    }
                else:
                    # RULE 1 COMPLIANCE: Fail fast when data invalid
                    raise MatrixAgentError(
                        "CRITICAL FAILURE: Agent 1 provided invalid or empty data. "
                        "Commander Locke requires valid Sentinel analysis to proceed."
                    )
        
        # Ensure we have basic binary analysis data for integration
        if 'analysis_results' not in context['shared_memory'] or 1 not in context['shared_memory']['analysis_results']:
            raise ValidationError("Unable to load Agent 1 (Sentinel) data for build integration")

    def _extract_integration_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract integration data from previous agents and cache"""
        integration_data = {
            'functions': {},
            'imports': {},
            'resources': {},
            'machine_data': {}
        }
        
        # First, try to get data from shared memory (cache-loaded)
        shared_memory = context.get('shared_memory', {})
        analysis_results = shared_memory.get('analysis_results', {})
        
        # Extract import data from Agent 1 (cached or live)
        if 1 in analysis_results:
            agent1_result = analysis_results[1]
            agent1_data = agent1_result.get('data', {})
            
            # Handle different data sources
            format_analysis = agent1_data.get('format_analysis', {})
            raw_imports = format_analysis.get('imports', [])
            
            imports_dict = {}
            for import_entry in raw_imports:
                if isinstance(import_entry, dict):
                    dll_name = import_entry.get('dll', 'unknown.dll')
                    functions = import_entry.get('functions', [])
                    imports_dict[dll_name] = functions
            
            integration_data['imports'] = imports_dict
            self.logger.info(f"Loaded imports from Agent 1 cache: {len(imports_dict)} DLLs")
        
        # Check agent_results if shared_memory is empty
        agent_results = context.get('agent_results', {})
        
        # Extract function data from Agent 5 (Neo) if available
        if 5 in agent_results:
            agent5_data = agent_results[5].data if hasattr(agent_results[5], 'data') else {}
            functions = agent5_data.get('decompiled_functions', {})
            
            # Handle different function formats
            if isinstance(functions, list):
                functions_dict = {}
                for i, func in enumerate(functions):
                    if isinstance(func, dict) and 'name' in func:
                        functions_dict[func['name']] = func
                    else:
                        functions_dict[f'function_{i:04d}'] = func
                integration_data['functions'] = functions_dict
            else:
                integration_data['functions'] = functions
        
        # Extract import data from Agent 9 (Machine) if available and not already loaded
        if not integration_data['imports'] and 9 in agent_results:
            agent9_data = agent_results[9].data if hasattr(agent_results[9], 'data') else {}
            imports = agent9_data.get('import_table_reconstruction', {})
            integration_data['imports'] = imports
            integration_data['machine_data'] = agent9_data
        
        # Extract imports directly from Agent 1 in agent_results if not loaded from cache
        if not integration_data['imports'] and 1 in agent_results:
            agent1_data = agent_results[1].data if hasattr(agent_results[1], 'data') else {}
            format_analysis = agent1_data.get('format_analysis', {})
            raw_imports = format_analysis.get('imports', [])
            
            imports_dict = {}
            for import_entry in raw_imports:
                dll_name = import_entry.get('dll', 'unknown.dll')
                functions = import_entry.get('functions', [])
                imports_dict[dll_name] = functions
            integration_data['imports'] = imports_dict
        
        self.logger.info(f"Extracted integration data: {len(integration_data['functions'])} functions, "
                        f"{len(integration_data['imports'])} DLL imports")
        
        return integration_data

    def _perform_build_integration(self, integration_data: Dict[str, Any]) -> BuildIntegrationResult:
        """Perform focused build system integration"""
        self.logger.info("Integrating build system components...")
        
        # Generate header files from function and import data
        header_files = self._generate_header_files(integration_data)
        
        # Generate build files with actual dependencies
        build_files = self._generate_build_files(integration_data)
        
        # Extract library dependencies
        library_dependencies = self._extract_library_dependencies(integration_data)
        
        # Calculate integration quality
        integration_quality = self._calculate_integration_quality(
            header_files, build_files, library_dependencies, integration_data
        )
        
        # Check compilation readiness
        compilation_ready = self._check_compilation_readiness(
            header_files, build_files, library_dependencies
        )
        
        return BuildIntegrationResult(
            header_files=header_files,
            build_files=build_files,
            library_dependencies=library_dependencies,
            integration_quality=integration_quality,
            compilation_ready=compilation_ready,
            integration_summary={
                'functions_integrated': len(integration_data['functions']),
                'dlls_integrated': len(integration_data['imports']),
                'headers_generated': len(header_files),
                'build_files_generated': len(build_files)
            }
        )

    def _generate_header_files(self, integration_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate header files from integration data"""
        header_files = {}
        
        # Generate main header with function declarations
        functions = integration_data.get('functions', {})
        if functions:
            main_header = self._generate_main_header(functions)
            header_files['main.h'] = main_header
        
        # Generate imports header with DLL function declarations
        imports = integration_data.get('imports', {})
        if imports:
            imports_header = self._generate_imports_header(imports)
            header_files['imports.h'] = imports_header
        
        # Generate common definitions header
        common_header = self._generate_common_header()
        header_files['common.h'] = common_header
        
        return header_files

    def _generate_main_header(self, functions: Dict[str, Any]) -> str:
        """Generate main header with function declarations"""
        lines = [
            "// Main Header - Function Declarations",
            "// Generated by Commander Locke",
            "",
            "#ifndef MAIN_H",
            "#define MAIN_H",
            "",
            "#include \"common.h\"",
            "#include \"imports.h\"",
            "",
            "// Function Declarations"
        ]
        
        for func_name, func_data in functions.items():
            if isinstance(func_data, dict):
                return_type = func_data.get('return_type', 'int')
                parameters = func_data.get('parameters', [])
                
                if parameters:
                    param_list = ', '.join(
                        f"{p.get('type', 'int')} {p.get('name', f'param{i}')}"
                        for i, p in enumerate(parameters)
                    )
                else:
                    param_list = 'void'
                
                lines.append(f"{return_type} {func_name}({param_list});")
        
        lines.extend([
            "",
            "#endif // MAIN_H"
        ])
        
        return '\n'.join(lines)

    def _generate_imports_header(self, imports: Dict[str, Any]) -> str:
        """Generate imports header with DLL function declarations"""
        lines = [
            "// Imports Header - DLL Function Declarations", 
            "// Generated by Commander Locke",
            "",
            "#ifndef IMPORTS_H",
            "#define IMPORTS_H",
            "",
            "#include \"common.h\"",
            ""
        ]
        
        # Standard Windows API exclusions
        excluded_functions = {
            'GetProcAddress', 'LoadLibraryA', 'FreeLibrary', 'GetModuleHandleA',
            'CreateThread', 'ExitProcess', 'GetCurrentProcess', 'TerminateProcess',
            'malloc', 'free', 'memset', 'memcpy', 'strcpy', 'strlen', 'printf'
        }
        
        for dll_name, dll_data in imports.items():
            # Skip standard Windows DLLs
            if dll_name.upper() in ['KERNEL32.DLL', 'USER32.DLL', 'GDI32.DLL']:
                lines.append(f"// {dll_name} functions available via windows.h")
                continue
            
            lines.append(f"// Functions from {dll_name}")
            
            # Handle different data formats from Machine agent
            functions = []
            if isinstance(dll_data, dict):
                functions = dll_data.get('functions', [])
            elif isinstance(dll_data, list):
                functions = dll_data
            
            for func in functions[:20]:  # Limit to 20 functions per DLL
                if isinstance(func, dict):
                    func_name = func.get('name', '')
                elif isinstance(func, str):
                    func_name = func
                else:
                    continue
                
                if func_name and func_name not in excluded_functions:
                    lines.append(f"extern void {func_name}();")
            
            lines.append("")
        
        lines.extend([
            "#endif // IMPORTS_H"
        ])
        
        return '\n'.join(lines)

    def _generate_common_header(self) -> str:
        """Generate common definitions header"""
        return """// Common Header - Shared Definitions
// Generated by Commander Locke

#ifndef COMMON_H
#define COMMON_H

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// Common type definitions
typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned long DWORD;

// Common constants
#define MAX_PATH 260
#define SUCCESS 0
#define FAILURE -1

#endif // COMMON_H
"""

    def _generate_build_files(self, integration_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate build files with actual dependencies"""
        build_files = {}
        
        # Extract library dependencies
        library_deps = self._extract_library_dependencies(integration_data)
        
        # Generate Visual Studio project file
        vcxproj_content = self._generate_vcxproj(library_deps)
        build_files['reconstruction.vcxproj'] = vcxproj_content
        
        # Generate CMakeLists.txt
        cmake_content = self._generate_cmake(library_deps)
        build_files['CMakeLists.txt'] = cmake_content
        
        return build_files

    def _extract_library_dependencies(self, integration_data: Dict[str, Any]) -> List[str]:
        """Extract library dependencies from import data"""
        libraries = []
        imports = integration_data.get('imports', {})
        
        # DLL to library mapping
        dll_to_lib = {
            'MFC71.DLL': 'mfc71.lib',
            'MSVCR71.dll': 'msvcr71.lib',
            'KERNEL32.dll': 'kernel32.lib',
            'USER32.dll': 'user32.lib',
            'GDI32.dll': 'gdi32.lib',
            'ADVAPI32.dll': 'advapi32.lib',
            'SHELL32.dll': 'shell32.lib',
            'OLE32.dll': 'ole32.lib',
            'OLEAUT32.dll': 'oleaut32.lib',
            'WINMM.dll': 'winmm.lib',
            'WS2_32.dll': 'ws2_32.lib',
            'VERSION.dll': 'version.lib',
            'COMCTL32.dll': 'comctl32.lib'
        }
        
        for dll_name in imports.keys():
            if dll_name in dll_to_lib:
                lib_file = dll_to_lib[dll_name]
                if lib_file not in libraries:
                    libraries.append(lib_file)
        
        # Ensure minimum standard libraries
        standard_libs = ['kernel32.lib', 'user32.lib', 'gdi32.lib']
        for lib in standard_libs:
            if lib not in libraries:
                libraries.append(lib)
        
        return libraries

    def _generate_vcxproj(self, library_deps: List[str]) -> str:
        """Generate Visual Studio project file"""
        lib_deps_str = ';'.join(library_deps) + ';%(AdditionalDependencies)'
        
        return f'''<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Release|Win32">
      <Configuration>Release</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <VCProjectVersion>17.0</VCProjectVersion>
    <ProjectGuid>{{LOCKE-BUILD-INTEGRATION-GUID}}</ProjectGuid>
    <RootNamespace>CommanderLockeReconstruction</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>MultiByte</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>NDEBUG;_WINDOWS;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
    </ClCompile>
    <Link>
      <SubSystem>Windows</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
      <AdditionalDependencies>{lib_deps_str}</AdditionalDependencies>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="*.c" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="*.h" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>'''

    def _generate_cmake(self, library_deps: List[str]) -> str:
        """Generate CMakeLists.txt file"""
        lib_links = '\n'.join(f'target_link_libraries(reconstruction {lib})' for lib in library_deps)
        
        return f'''# CMakeLists.txt - Commander Locke Build Integration
# Generated with actual library dependencies

cmake_minimum_required(VERSION 3.10)
project(CommanderLockeReconstruction)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Source files
file(GLOB SOURCES "*.c")
file(GLOB HEADERS "*.h")

# Create executable
add_executable(reconstruction ${{SOURCES}})

# Library dependencies from import analysis
{lib_links}

# Include directories
target_include_directories(reconstruction PRIVATE .)
'''

    def _calculate_integration_quality(self, header_files: Dict[str, str], build_files: Dict[str, str],
                                     library_deps: List[str], integration_data: Dict[str, Any]) -> float:
        """Calculate integration quality score"""
        score = 0.0
        
        # Header file quality (40%)
        if header_files:
            meaningful_headers = sum(1 for content in header_files.values() if len(content) > 50)
            score += min(meaningful_headers * 0.15, 0.4)
        
        # Build file quality (30%)
        if build_files:
            meaningful_builds = sum(1 for content in build_files.values() if len(content) > 100)
            score += min(meaningful_builds * 0.15, 0.3)
        
        # Library dependency quality (20%)
        if len(library_deps) >= self.min_dll_dependencies:
            score += 0.2
        elif library_deps:
            score += 0.1
        
        # Data integration quality (10%)
        functions_count = len(integration_data.get('functions', {}))
        imports_count = len(integration_data.get('imports', {}))
        if functions_count >= self.min_functions and imports_count >= self.min_dll_dependencies:
            score += 0.1
        
        return min(score, 1.0)

    def _check_compilation_readiness(self, header_files: Dict[str, str], 
                                   build_files: Dict[str, str], library_deps: List[str]) -> bool:
        """Check if integration is ready for compilation"""
        # Must have essential headers
        has_main_header = 'main.h' in header_files
        has_imports_header = 'imports.h' in header_files
        
        # Must have build files
        has_vcxproj = any('.vcxproj' in filename for filename in build_files.keys())
        has_cmake = 'CMakeLists.txt' in build_files
        
        # Must have minimum libraries
        has_libraries = len(library_deps) >= 3
        
        return has_main_header and has_imports_header and (has_vcxproj or has_cmake) and has_libraries

    def _save_results(self, integration_result: BuildIntegrationResult, output_paths: Dict[str, Path]) -> None:
        """Save integration results using shared file manager"""
        if not self.file_manager:
            return
            
        try:
            # Prepare results for saving
            results_data = {
                'agent_info': {
                    'agent_id': self.agent_id,
                    'agent_name': 'CommanderLocke_BuildIntegration',
                    'matrix_character': 'Commander Locke',
                    'analysis_timestamp': time.time()
                },
                'build_integration': {
                    'header_count': len(integration_result.header_files),
                    'build_file_count': len(integration_result.build_files),
                    'library_count': len(integration_result.library_dependencies),
                    'compilation_ready': integration_result.compilation_ready,
                    'integration_quality': integration_result.integration_quality
                },
                'header_files': integration_result.header_files,
                'build_files': integration_result.build_files,
                'library_dependencies': integration_result.library_dependencies,
                'integration_summary': integration_result.integration_summary
            }
            
            # Save using shared file manager
            output_file = self.file_manager.save_agent_data(
                self.agent_id, "commander_locke", results_data
            )
            
            # Save build files separately for compilation
            compilation_dir = output_paths.get('compilation', Path('output/compilation'))
            
            # Save header files
            for filename, content in integration_result.header_files.items():
                header_path = compilation_dir / filename
                header_path.parent.mkdir(parents=True, exist_ok=True)
                with open(header_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Header file saved: {header_path}")
            
            # Save build files
            for filename, content in integration_result.build_files.items():
                build_path = compilation_dir / filename
                build_path.parent.mkdir(parents=True, exist_ok=True)
                with open(build_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Build file saved: {build_path}")
            
            self.logger.info(f"Commander Locke results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save Commander Locke results: {e}")

    def _load_agent_1_cache_data(self, context: Dict[str, Any]) -> bool:
        """Load Agent 1 (Sentinel) cache data from output directory"""
        try:
            # Check for Agent 1 cache files
            cache_paths = [
                "output/launcher/latest/agents/agent_01/binary_analysis_cache.json",
                "output/launcher/latest/agents/agent_01/import_analysis_cache.json",
                "output/launcher/latest/agents/agent_01_sentinel/agent_result.json",
                "output/launcher/latest/agents/agent_01/sentinel_data.json"
            ]
            
            import json
            cached_data = {}
            cache_found = False
            
            for cache_path in cache_paths:
                cache_file = Path(cache_path)
                if cache_file.exists():
                    try:
                        with open(cache_file, 'r') as f:
                            file_data = json.load(f)
                            cached_data.update(file_data)
                            cache_found = True
                            self.logger.debug(f"Loaded Agent 1 cache from {cache_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load Agent 1 cache from {cache_path}: {e}")
            
            if cache_found:
                # Populate shared memory with cached data
                shared_memory = context['shared_memory']
                
                # Merge all loaded cache data and extract format_analysis
                format_analysis = cached_data.get('format_analysis', {})
                if not format_analysis and 'data' in cached_data:
                    # Handle nested data structure
                    nested_data = cached_data['data']
                    format_analysis = nested_data.get('format_analysis', {})
                
                # Store in analysis_results for compatibility
                shared_memory['analysis_results'][1] = {
                    'status': 'cached',
                    'data': {
                        'format_analysis': format_analysis,
                        'binary_format': cached_data.get('binary_format', 'PE32+'),
                        'architecture': cached_data.get('architecture', 'x64'),
                        'file_size': cached_data.get('file_size', 0),
                        **cached_data
                    }
                }
                
                # Also store in binary_metadata for other agents
                if 'discovery' not in shared_memory['binary_metadata']:
                    shared_memory['binary_metadata']['discovery'] = {
                        'binary_analyzed': True,
                        'cache_source': 'agent_01',
                        'binary_format': cached_data.get('binary_format', 'PE32+'),
                        'architecture': cached_data.get('architecture', 'x64'),
                        'file_size': cached_data.get('file_size', 0),
                        'cached_data': cached_data
                    }
                
                # Debug log the import data found
                imports = format_analysis.get('imports', [])
                if imports:
                    self.logger.info(f"Found {len(imports)} DLL imports in cache data")
                else:
                    self.logger.warning("No import data found in cache - may affect build integration")
                
                self.logger.info("Successfully loaded Agent 1 (Sentinel) cache data for build integration")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to load Agent 1 cache data: {e}")
            return False

    # RULE 1 COMPLIANCE: Data creation method removed - fail fast instead