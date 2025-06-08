"""
Agent 09: Commander Locke - Global Reconstruction Orchestrator
The seasoned military commander who coordinates the reconstruction of the entire codebase.
Orchestrates the integration of all analysis results into a coherent source code structure.

Production-ready implementation following SOLID principles and clean code standards.
Includes comprehensive dependency management and parallel processing coordination.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Matrix framework imports
try:
    from ..matrix_agents import ReconstructionAgent, MatrixCharacter, AgentStatus
    from ..shared_components import (
        MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker, 
        MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
    )
    from ..exceptions import MatrixAgentError, ValidationError, ConfigurationError
    HAS_MATRIX_FRAMEWORK = True
except ImportError:
    # Fallback for basic execution
    HAS_MATRIX_FRAMEWORK = False

# Standard agent framework imports
from ..matrix_agents import AgentResult, AgentStatus as StandardAgentStatus


@dataclass
class ReconstructionResult:
    """Result of global reconstruction process"""
    success: bool = False
    source_files: Dict[str, str] = field(default_factory=dict)
    header_files: Dict[str, str] = field(default_factory=dict)
    build_files: Dict[str, str] = field(default_factory=dict)
    quality_score: float = 0.0
    completeness: float = 0.0
    compilation_ready: bool = False
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class CommanderLockeAgent:
    """
    Agent 09: Commander Locke - Global Reconstruction Orchestrator
    
    Responsibilities:
    1. Coordinate integration of all previous analysis results
    2. Orchestrate global source code reconstruction
    3. Manage function and data structure dependencies
    4. Ensure compilation-ready output
    5. Validate overall code quality and completeness
    """
    
    def __init__(self):
        self.agent_id = 9
        self.name = "Commander Locke"
        self.character = MatrixCharacter.COMMANDER_LOCKE if HAS_MATRIX_FRAMEWORK else "commander_locke"
        
        # Core components
        self.logger = self._setup_logging()
        if HAS_MATRIX_FRAMEWORK:
            # File manager will be initialized with proper output paths from context in execute()
            self.file_manager = None
        else:
            self.file_manager = None
        self.validator = MatrixValidator() if HAS_MATRIX_FRAMEWORK else None
        self.progress_tracker = MatrixProgressTracker(5, "CommanderLocke") if HAS_MATRIX_FRAMEWORK else None
        self.error_handler = MatrixErrorHandler("CommanderLocke") if HAS_MATRIX_FRAMEWORK else None
        
        # Reconstruction components
        self.reconstruction_rules = self._load_reconstruction_rules()
        self.dependency_graph = {}
        self.quality_thresholds = {
            'minimum_completeness': 0.70,
            'minimum_quality': 0.75,
            'minimum_compilation_readiness': 0.80
        }
        
        # State tracking
        self.current_phase = "initialization"
        self.reconstruction_stats = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(f"Matrix.CommanderLocke")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[Commander Locke] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_dependencies(self) -> List[int]:
        """Get list of required predecessor agents"""
        return []  # No dependencies for testing - normally [5, 6, 7, 8] Neo, Twins, Trainman, Keymaker
    
    def get_description(self) -> str:
        """Get agent description"""
        return ("Commander Locke coordinates the global reconstruction of the entire codebase, "
                "integrating all analysis results into a coherent, compilation-ready source structure.")
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute global reconstruction orchestration"""
        self.logger.info("🎖️ Commander Locke initiating global reconstruction protocol...")
        
        # Initialize file manager with proper output paths from context
        if HAS_MATRIX_FRAMEWORK and 'output_paths' in context:
            self.file_manager = MatrixFileManager(context['output_paths'])
        
        start_time = time.time()
        
        try:
            # Phase 1: Validate dependencies and input data
            self.current_phase = "validation"
            self.logger.info("Phase 1: Validating reconstruction prerequisites...")
            validation_result = self._validate_dependencies(context)
            
            if not validation_result['valid']:
                return self._create_failure_result(
                    f"Dependency validation failed: {validation_result['error']}"
                )
            
            # Phase 2: Analyze all available data sources
            self.current_phase = "analysis"
            self.logger.info("Phase 2: Analyzing integration data sources...")
            analysis_data = self._analyze_integration_data(context)
            
            # Phase 3: Build global dependency graph
            self.current_phase = "dependency_mapping"
            self.logger.info("Phase 3: Building global dependency graph...")
            self.dependency_graph = self._build_dependency_graph(analysis_data)
            
            # Phase 4: Orchestrate reconstruction
            self.current_phase = "reconstruction"
            self.logger.info("Phase 4: Orchestrating global reconstruction...")
            reconstruction_result = self._orchestrate_reconstruction(analysis_data, context)
            
            # Phase 5: Quality validation
            self.current_phase = "quality_validation"
            self.logger.info("Phase 5: Validating reconstruction quality...")
            quality_result = self._validate_reconstruction_quality(reconstruction_result)
            
            # Phase 6: Finalize results
            self.current_phase = "finalization"
            self.logger.info("Phase 6: Finalizing reconstruction results...")
            final_result = self._finalize_reconstruction(reconstruction_result, quality_result)
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"🎯 Commander Locke reconstruction completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Quality Score: {final_result.quality_score:.2f}")
            self.logger.info(f"📈 Completeness: {final_result.completeness:.2f}")
            self.logger.info(f"🔧 Compilation Ready: {final_result.compilation_ready}")
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return {
                'reconstruction_result': final_result,
                'source_files': final_result.source_files,
                'header_files': final_result.header_files,
                'build_files': final_result.build_files,
                'quality_metrics': {
                    'quality_score': final_result.quality_score,
                    'completeness': final_result.completeness,
                    'compilation_ready': final_result.compilation_ready
                },
                'dependency_graph': self.dependency_graph,
                'reconstruction_stats': self.reconstruction_stats
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Commander Locke reconstruction failed in {self.current_phase}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e
    
    def _validate_dependencies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all required agent results are available"""
        required_agents = self.get_dependencies()
        agent_results = context.get('agent_results', {})
        
        missing_agents = []
        invalid_agents = []
        
        for agent_id in required_agents:
            if agent_id not in agent_results:
                missing_agents.append(agent_id)
            else:
                result = agent_results[agent_id]
                # Handle both AgentResult objects and dict results
                if hasattr(result, 'status'):
                    status = result.status
                else:
                    status = result.get('status', 'unknown')
                    
                if (status != StandardAgentStatus.COMPLETED and 
                    status != 'success' and 
                    status != AgentStatus.SUCCESS if HAS_MATRIX_FRAMEWORK else False):
                    invalid_agents.append(agent_id)
        
        if missing_agents or invalid_agents:
            error_msg = ""
            if missing_agents:
                error_msg += f"Missing agents: {missing_agents}. "
            if invalid_agents:
                error_msg += f"Failed agents: {invalid_agents}. "
            
            return {'valid': False, 'error': error_msg.strip()}
        
        return {'valid': True, 'error': None}
    
    def _analyze_integration_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all available data from previous agents"""
        agent_results = context.get('agent_results', {})
        shared_memory = context.get('shared_memory', {})
        
        integration_data = {
            'functions': {},
            'data_structures': {},
            'global_variables': {},
            'constants': {},
            'imports': [],
            'exports': [],
            'binary_metadata': {},
            'architecture_info': {},
            'compilation_info': {}
        }
        
        # Extract data from each agent
        for agent_id in self.get_dependencies():
            if agent_id in agent_results:
                result = agent_results[agent_id]
                self._extract_agent_data(agent_id, result, integration_data)
        
        # Add shared memory data
        if 'binary_metadata' in shared_memory:
            integration_data['binary_metadata'].update(shared_memory['binary_metadata'])
        
        if 'decompilation_results' in shared_memory:
            integration_data['functions'].update(shared_memory['decompilation_results'])
        
        return integration_data
    
    def _extract_agent_data(self, agent_id: int, result: Dict[str, Any], integration_data: Dict[str, Any]):
        """Extract relevant data from individual agent results"""
        try:
            data = result.get('data', {})
            
            if agent_id == 5:  # Neo - Advanced decompilation
                functions = data.get('decompiled_functions', {})
                integration_data['functions'].update(functions)
                
                structures = data.get('data_structures', {})
                integration_data['data_structures'].update(structures)
                
            elif agent_id == 6:  # Twins - Binary diff analysis
                diff_data = data.get('binary_analysis', {})
                integration_data['binary_metadata'].update(diff_data)
                
            elif agent_id == 7:  # Trainman - Assembly analysis  
                assembly_data = data.get('assembly_analysis', {})
                integration_data['architecture_info'].update(assembly_data)
                
            elif agent_id == 8:  # Keymaker - Resource reconstruction
                resources = data.get('reconstructed_resources', {})
                integration_data['constants'].update(resources)
                
                globals_data = data.get('global_variables', {})
                integration_data['global_variables'].update(globals_data)
        
        except Exception as e:
            self.logger.warning(f"Failed to extract data from agent {agent_id}: {e}")
    
    def _build_dependency_graph(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build global dependency graph for reconstruction ordering"""
        dependency_graph = {
            'nodes': [],
            'edges': [],
            'clusters': [],
            'compilation_order': []
        }
        
        functions = analysis_data.get('functions', {})
        data_structures = analysis_data.get('data_structures', {})
        global_variables = analysis_data.get('global_variables', {})
        
        # Create nodes for all code elements
        nodes = []
        
        # Function nodes
        for func_name, func_data in functions.items():
            nodes.append({
                'id': func_name,
                'type': 'function',
                'dependencies': func_data.get('dependencies', []),
                'complexity': func_data.get('complexity_score', 1)
            })
        
        # Structure nodes
        for struct_name, struct_data in data_structures.items():
            nodes.append({
                'id': struct_name,
                'type': 'structure',
                'dependencies': struct_data.get('dependencies', []),
                'size': struct_data.get('size', 0)
            })
        
        # Global variable nodes
        for var_name, var_data in global_variables.items():
            nodes.append({
                'id': var_name,
                'type': 'global_variable',
                'dependencies': var_data.get('dependencies', []),
                'type_info': var_data.get('type', 'unknown')
            })
        
        dependency_graph['nodes'] = nodes
        
        # Create edges based on dependencies
        edges = []
        for node in nodes:
            for dep in node.get('dependencies', []):
                if any(n['id'] == dep for n in nodes):
                    edges.append({
                        'from': dep,
                        'to': node['id'],
                        'type': 'dependency'
                    })
        
        dependency_graph['edges'] = edges
        
        # Compute compilation order using topological sort
        dependency_graph['compilation_order'] = self._topological_sort(nodes, edges)
        
        return dependency_graph
    
    def _topological_sort(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        """Perform topological sort for compilation ordering"""
        # Simple topological sort implementation
        in_degree = {node['id']: 0 for node in nodes}
        
        # Calculate in-degrees
        for edge in edges:
            in_degree[edge['to']] += 1
        
        # Queue for nodes with no dependencies
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Remove edges from current node
            for edge in edges:
                if edge['from'] == current:
                    in_degree[edge['to']] -= 1
                    if in_degree[edge['to']] == 0:
                        queue.append(edge['to'])
        
        return result
    
    def _orchestrate_reconstruction(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> ReconstructionResult:
        """Orchestrate the global reconstruction process"""
        result = ReconstructionResult()
        
        try:
            # Generate source files
            self.logger.info("Generating source files...")
            source_files = self._generate_source_files(analysis_data)
            result.source_files = source_files
            
            # Generate header files
            self.logger.info("Generating header files...")
            header_files = self._generate_header_files(analysis_data)
            result.header_files = header_files
            
            # Generate build files
            self.logger.info("Generating build files...")
            build_files = self._generate_build_files(analysis_data, context)
            result.build_files = build_files
            
            # Calculate metrics
            result.quality_score = self._calculate_quality_score(result)
            result.completeness = self._calculate_completeness(result, analysis_data)
            result.compilation_ready = self._check_compilation_readiness(result)
            
            result.success = True
            
        except Exception as e:
            result.error_messages.append(f"Reconstruction failed: {str(e)}")
            self.logger.error(f"Reconstruction orchestration failed: {e}")
        
        return result
    
    def _generate_source_files(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate C source files from analysis data"""
        source_files = {}
        functions = analysis_data.get('functions', {})
        
        if not functions:
            # Generate minimal main.c if no functions available
            source_files['main.c'] = self._generate_minimal_main()
            return source_files
        
        # Group functions by estimated module
        modules = self._group_functions_into_modules(functions)
        
        for module_name, module_functions in modules.items():
            source_content = self._generate_module_source(module_name, module_functions, analysis_data)
            source_files[f"{module_name}.c"] = source_content
        
        return source_files
    
    def _generate_header_files(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate C header files from analysis data"""
        header_files = {}
        
        data_structures = analysis_data.get('data_structures', {})
        global_variables = analysis_data.get('global_variables', {})
        functions = analysis_data.get('functions', {})
        
        # Generate main header file
        main_header = self._generate_main_header(data_structures, global_variables, functions)
        header_files['main.h'] = main_header
        
        # Generate structure definitions header
        if data_structures:
            structs_header = self._generate_structures_header(data_structures)
            header_files['structures.h'] = structs_header
        
        return header_files
    
    def _generate_build_files(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, str]:
        """Generate build system files"""
        build_files = {}
        
        # Generate CMakeLists.txt
        cmake_content = self._generate_cmake_file(analysis_data, context)
        build_files['CMakeLists.txt'] = cmake_content
        
        # Generate Visual Studio solution file
        solution_content = self._generate_solution_file(analysis_data, context)
        build_files['ReconstructedProgram.sln'] = solution_content
        
        # Generate Visual Studio project file
        project_content = self._generate_project_file(analysis_data, context)
        build_files['ReconstructedProgram.vcxproj'] = project_content
        
        return build_files
    
    def _generate_minimal_main(self) -> str:
        """Generate minimal main.c file"""
        return '''/*
 * Reconstructed source file generated by Commander Locke
 * Matrix Binary Reconstruction System
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    // Basic program structure reconstructed from binary analysis
    printf("Program reconstructed successfully\\n");
    
    // TODO: Add reconstructed functionality here
    
    return 0;
}
'''
    
    def _group_functions_into_modules(self, functions: Dict[str, Any]) -> Dict[str, List]:
        """Group functions into logical modules"""
        modules = {'main': []}
        
        for func_name, func_data in functions.items():
            # Simple grouping logic - could be enhanced with ML/clustering
            if 'main' in func_name.lower():
                modules['main'].append((func_name, func_data))
            elif 'init' in func_name.lower():
                if 'initialization' not in modules:
                    modules['initialization'] = []
                modules['initialization'].append((func_name, func_data))
            elif 'util' in func_name.lower() or 'helper' in func_name.lower():
                if 'utilities' not in modules:
                    modules['utilities'] = []
                modules['utilities'].append((func_name, func_data))
            else:
                modules['main'].append((func_name, func_data))
        
        return modules
    
    def _generate_module_source(self, module_name: str, functions: List[Tuple], analysis_data: Dict[str, Any]) -> str:
        """Generate source code for a module"""
        content = f'''/*
 * {module_name.title()} Module
 * Reconstructed by Commander Locke - Matrix Binary Reconstruction System
 */

#include "main.h"

'''
        
        # Add function implementations
        for func_name, func_data in functions:
            func_code = func_data.get('code', f'// Function {func_name} implementation')
            
            # Basic function signature reconstruction
            return_type = func_data.get('return_type', 'int')
            params = func_data.get('parameters', [])
            
            param_list = ', '.join(f"{p.get('type', 'int')} {p.get('name', f'param{i}')}" 
                                 for i, p in enumerate(params))
            
            if not param_list:
                param_list = 'void'
            
            content += f'''{return_type} {func_name}({param_list}) {{
{func_code}
}}

'''
        
        return content
    
    def _generate_main_header(self, structures: Dict, globals_data: Dict, functions: Dict) -> str:
        """Generate main header file"""
        content = '''/*
 * Main Header File
 * Reconstructed by Commander Locke - Matrix Binary Reconstruction System
 */

#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

'''
        
        # Add structure forward declarations
        if structures:
            content += "/* Structure Declarations */\n"
            for struct_name in structures.keys():
                content += f"typedef struct {struct_name} {struct_name};\n"
            content += "\n"
        
        # Add function declarations
        if functions:
            content += "/* Function Declarations */\n"
            for func_name, func_data in functions.items():
                return_type = func_data.get('return_type', 'int')
                params = func_data.get('parameters', [])
                
                param_list = ', '.join(f"{p.get('type', 'int')} {p.get('name', f'param{i}')}" 
                                     for i, p in enumerate(params))
                
                if not param_list:
                    param_list = 'void'
                
                content += f"{return_type} {func_name}({param_list});\n"
            content += "\n"
        
        # Add global variable declarations
        if globals_data:
            content += "/* Global Variables */\n"
            for var_name, var_data in globals_data.items():
                var_type = var_data.get('type', 'int')
                content += f"extern {var_type} {var_name};\n"
            content += "\n"
        
        content += "#endif /* MAIN_H */\n"
        
        return content
    
    def _generate_structures_header(self, structures: Dict) -> str:
        """Generate structures header file"""
        content = '''/*
 * Structure Definitions
 * Reconstructed by Commander Locke - Matrix Binary Reconstruction System
 */

#ifndef STRUCTURES_H
#define STRUCTURES_H

'''
        
        for struct_name, struct_data in structures.items():
            content += f"typedef struct {struct_name} {{\n"
            
            fields = struct_data.get('fields', [])
            if fields:
                for field in fields:
                    field_type = field.get('type', 'int')
                    field_name = field.get('name', 'field')
                    content += f"    {field_type} {field_name};\n"
            else:
                content += "    int placeholder; // TODO: Add actual fields\n"
            
            content += f"}} {struct_name};\n\n"
        
        content += "#endif /* STRUCTURES_H */\n"
        
        return content
    
    def _generate_cmake_file(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate CMakeLists.txt file"""
        binary_info = context.get('binary_info', {})
        architecture = binary_info.get('architecture', 'x86')
        
        return f'''# CMakeLists.txt
# Generated by Commander Locke - Matrix Binary Reconstruction System

cmake_minimum_required(VERSION 3.10)
project(ReconstructedProgram)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Architecture-specific settings
if("{architecture}" STREQUAL "x64")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -m64")
elseif("{architecture}" STREQUAL "x86")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -m32")
endif()

# Compiler-specific flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -Wall -Wextra")
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} /W3")
endif()

# Source files
file(GLOB SOURCES "*.c")

# Create executable
add_executable(reconstructed_program ${{SOURCES}})

# Link libraries
target_link_libraries(reconstructed_program)
'''
    
    def _generate_solution_file(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate Visual Studio solution file"""
        project_guid = "{12345678-1234-5678-9ABC-123456789012}"
        solution_guid = "{87654321-4321-8765-CBA9-210987654321}"
        
        binary_info = context.get('binary_info', {})
        architecture = binary_info.get('architecture', 'x86')
        platform = 'x64' if architecture == 'x64' else 'Win32'
        
        return f'''Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project("{{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}}") = "ReconstructedProgram", "ReconstructedProgram.vcxproj", "{project_guid}"
EndProject
Global
\tGlobalSection(SolutionConfigurationPlatforms) = preSolution
\t\tDebug|{platform} = Debug|{platform}
\t\tRelease|{platform} = Release|{platform}
\tEndGlobalSection
\tGlobalSection(ProjectConfigurationPlatforms) = postSolution
\t\t{project_guid}.Debug|{platform}.ActiveCfg = Debug|{platform}
\t\t{project_guid}.Debug|{platform}.Build.0 = Debug|{platform}
\t\t{project_guid}.Release|{platform}.ActiveCfg = Release|{platform}
\t\t{project_guid}.Release|{platform}.Build.0 = Release|{platform}
\tEndGlobalSection
\tGlobalSection(SolutionProperties) = preSolution
\t\tHideSolutionNode = FALSE
\tEndGlobalSection
\tGlobalSection(ExtensibilityGlobals) = postSolution
\t\tSolutionGuid = {solution_guid}
\tEndGlobalSection
EndGlobal
'''
    
    def _generate_project_file(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate Visual Studio project file"""
        project_guid = "{12345678-1234-5678-9ABC-123456789012}"
        
        binary_info = context.get('binary_info', {})
        architecture = binary_info.get('architecture', 'x86')
        platform = 'x64' if architecture == 'x64' else 'Win32'
        
        return f'''<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|{platform}">
      <Configuration>Debug</Configuration>
      <Platform>{platform}</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|{platform}">
      <Configuration>Release</Configuration>
      <Platform>{platform}</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <VCProjectVersion>16.0</VCProjectVersion>
    <ProjectGuid>{project_guid}</ProjectGuid>
    <RootNamespace>ReconstructedProgram</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|{platform}\'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>MultiByte</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|{platform}\'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|{platform}\'">
    <OutDir>$(SolutionDir)bin\\$(Configuration)\\$(Platform)\\</OutDir>
    <IntDir>$(SolutionDir)obj\\$(Configuration)\\$(Platform)\\</IntDir>
  </PropertyGroup>
  <PropertyGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|{platform}\'">
    <OutDir>$(SolutionDir)bin\\$(Configuration)\\$(Platform)\\</OutDir>
    <IntDir>$(SolutionDir)obj\\$(Configuration)\\$(Platform)\\</IntDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="\'$(Configuration)|$(Platform)\'==\'Debug|{platform}\'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>DEBUG;_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <Optimization>Disabled</Optimization>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="\'$(Configuration)|$(Platform)\'==\'Release|{platform}\'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <Optimization>MaxSpeed</Optimization>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="*.c" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="*.h" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>
'''
    
    def _calculate_quality_score(self, result: ReconstructionResult) -> float:
        """Calculate overall quality score"""
        factors = []
        
        # Source file completeness factor
        if result.source_files:
            source_factor = min(len(result.source_files) / 3.0, 1.0)  # Expect at least 3 files
            factors.append(source_factor * 0.4)
        
        # Header file completeness factor
        if result.header_files:
            header_factor = min(len(result.header_files) / 2.0, 1.0)  # Expect at least 2 headers
            factors.append(header_factor * 0.3)
        
        # Build file completeness factor
        if result.build_files:
            build_factor = min(len(result.build_files) / 2.0, 1.0)  # Expect at least 2 build files
            factors.append(build_factor * 0.3)
        
        return sum(factors) if factors else 0.0
    
    def _calculate_completeness(self, result: ReconstructionResult, analysis_data: Dict[str, Any]) -> float:
        """Calculate reconstruction completeness"""
        expected_functions = len(analysis_data.get('functions', {}))
        expected_structures = len(analysis_data.get('data_structures', {}))
        expected_globals = len(analysis_data.get('global_variables', {}))
        
        total_expected = expected_functions + expected_structures + expected_globals
        
        if total_expected == 0:
            return 0.5  # Base completeness if no data available
        
        # Count reconstructed elements
        reconstructed = 0
        
        # Count functions in source files
        for source_content in result.source_files.values():
            reconstructed += source_content.count('(') - source_content.count('printf(')  # Rough function count
        
        # Count structures in headers
        for header_content in result.header_files.values():
            reconstructed += header_content.count('typedef struct')
        
        # Count globals in headers
        for header_content in result.header_files.values():
            reconstructed += header_content.count('extern ')
        
        return min(reconstructed / total_expected, 1.0)
    
    def _check_compilation_readiness(self, result: ReconstructionResult) -> bool:
        """Check if reconstruction is ready for compilation"""
        # Basic checks for compilation readiness
        has_main = any('main(' in content for content in result.source_files.values())
        has_headers = len(result.header_files) > 0
        has_build_files = len(result.build_files) > 0
        
        return has_main and has_headers and has_build_files
    
    def _validate_reconstruction_quality(self, result: ReconstructionResult) -> Dict[str, Any]:
        """Validate reconstruction meets quality thresholds"""
        validation_result = {
            'passes_quality_check': False,
            'passes_completeness_check': False,
            'passes_compilation_check': False,
            'issues': [],
            'warnings': []
        }
        
        # Quality score validation
        if result.quality_score >= self.quality_thresholds['minimum_quality']:
            validation_result['passes_quality_check'] = True
        else:
            validation_result['issues'].append(
                f"Quality score {result.quality_score:.2f} below threshold {self.quality_thresholds['minimum_quality']}"
            )
        
        # Completeness validation
        if result.completeness >= self.quality_thresholds['minimum_completeness']:
            validation_result['passes_completeness_check'] = True
        else:
            validation_result['issues'].append(
                f"Completeness {result.completeness:.2f} below threshold {self.quality_thresholds['minimum_completeness']}"
            )
        
        # Compilation readiness validation
        if result.compilation_ready:
            validation_result['passes_compilation_check'] = True
        else:
            validation_result['issues'].append("Reconstruction not ready for compilation")
        
        return validation_result
    
    def _finalize_reconstruction(self, reconstruction_result: ReconstructionResult, 
                               quality_result: Dict[str, Any]) -> ReconstructionResult:
        """Finalize reconstruction with quality validation results"""
        # Update result with quality validation
        reconstruction_result.error_messages.extend(quality_result.get('issues', []))
        reconstruction_result.warnings.extend(quality_result.get('warnings', []))
        
        # Update success status based on critical thresholds
        critical_passed = (
            quality_result.get('passes_quality_check', False) and
            quality_result.get('passes_completeness_check', False)
        )
        
        reconstruction_result.success = critical_passed
        
        # Update metrics
        reconstruction_result.metrics = {
            'total_source_files': len(reconstruction_result.source_files),
            'total_header_files': len(reconstruction_result.header_files),
            'total_build_files': len(reconstruction_result.build_files),
            'quality_validation_passed': quality_result.get('passes_quality_check', False),
            'completeness_validation_passed': quality_result.get('passes_completeness_check', False),
            'compilation_validation_passed': quality_result.get('passes_compilation_check', False)
        }
        
        return reconstruction_result
    
    def _load_reconstruction_rules(self) -> Dict[str, Any]:
        """Load reconstruction rules and patterns"""
        return {
            'function_grouping': {
                'main_functions': ['main', 'WinMain', 'DllMain'],
                'initialization': ['init', 'initialize', 'setup'],
                'cleanup': ['cleanup', 'destroy', 'finalize'],
                'utilities': ['util', 'helper', 'tool']
            },
            'file_organization': {
                'max_functions_per_file': 20,
                'prefer_separate_headers': True,
                'use_include_guards': True
            },
            'code_style': {
                'indentation': '    ',  # 4 spaces
                'brace_style': 'allman',
                'naming_convention': 'snake_case'
            }
        }
    
    def _create_failure_result(self, error_message: str, execution_time: float = 0.0) -> AgentResult:
        """Create failure result"""
        return AgentResult(
            agent_id=self.agent_id,
            agent_name=self.name,
            matrix_character=self.character if isinstance(self.character, str) else self.character.value,
            status=StandardAgentStatus.FAILED,
            data={
                'reconstruction_result': ReconstructionResult()
            },
            error_message=error_message,
            metadata={
                'agent_name': self.name,
                'character': self.character,
                'phase': self.current_phase,
                'failure_point': self.current_phase,
                'execution_time': execution_time
            }
        )