"""
Agent 10: The Machine - Compilation Orchestration + Build Systems
Orchestrates compilation processes and manages complex build systems for reconstructed code.
"""

import os
import json
import subprocess
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent10_TheMachine(BaseAgent):
    """Agent 10: The Machine - Compilation orchestration and build systems"""
    
    def __init__(self):
        super().__init__(
            agent_id=10,
            name="TheMachine",
            dependencies=[8, 9]  # Depends on resource reconstruction and assembly analysis
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute compilation orchestration"""
        # Gather dependencies - can work with partial results
        agent8_result = context['agent_results'].get(8)  # Resource reconstruction
        agent9_result = context['agent_results'].get(9)  # Global reconstruction
        
        # Also check for source code from other agents
        available_sources = self._gather_available_sources(context['agent_results'])
        
        if not available_sources:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="No source code available for compilation orchestration"
            )

        try:
            # Analyze available source code and determine build requirements
            build_analysis = self._analyze_build_requirements(available_sources, context)
            
            # Generate comprehensive build system
            build_system = self._generate_build_system(build_analysis, context)
            
            # Orchestrate compilation process
            compilation_results = self._orchestrate_compilation(build_system, context)
            
            # Validate and optimize build process
            optimized_build = self._optimize_build_process(compilation_results, build_system)
            
            machine_result = {
                'build_analysis': build_analysis,
                'build_system': build_system,
                'compilation_results': compilation_results,
                'optimized_build': optimized_build,
                'build_orchestration': self._create_build_orchestration(optimized_build),
                'machine_metrics': self._calculate_machine_metrics(compilation_results)
            }
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=machine_result,
                metadata={
                    'depends_on': [8, 9],
                    'analysis_type': 'compilation_orchestration',
                    'build_systems_detected': len(build_system.get('detected_systems', [])),
                    'compilation_success_rate': compilation_results.get('success_rate', 0.0),
                    'machine_efficiency': optimized_build.get('efficiency_score', 0.0)
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"The Machine compilation orchestration failed: {str(e)}"
            )

    def _gather_available_sources(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Gather all available source code from various agents"""
        sources = {
            'source_files': {},
            'header_files': {},
            'resource_files': {},
            'build_files': {},
            'main_function': None,
            'dependencies': [],
            'compilation_units': []
        }
        
        # Gather from Global Reconstructor (Agent 9)
        if 9 in all_results and hasattr(all_results[9], 'status') and all_results[9].status == AgentStatus.COMPLETED:
            global_data = all_results[9].data
            if isinstance(global_data, dict):
                reconstructed_source = global_data.get('reconstructed_source', {})
                if isinstance(reconstructed_source, dict):
                    sources['source_files'].update(reconstructed_source.get('source_files', {}))
                    sources['header_files'].update(reconstructed_source.get('header_files', {}))
                    sources['main_function'] = reconstructed_source.get('main_function')
                
                # Get build configuration
                build_config = global_data.get('build_configuration', {})
                if build_config:
                    sources['build_files'].update(build_config.get('build_files', {}))
                    sources['dependencies'].extend(build_config.get('dependencies', []))
        
        # Gather from Resource Reconstructor (Agent 8)
        if 8 in all_results and hasattr(all_results[8], 'status') and all_results[8].status == AgentStatus.COMPLETED:
            resource_data = all_results[8].data
            if isinstance(resource_data, dict):
                sources['resource_files'].update(resource_data.get('resource_files', {}))
        
        # Gather from Advanced Decompiler (Agent 7)
        if 7 in all_results and hasattr(all_results[7], 'status') and all_results[7].status == AgentStatus.COMPLETED:
            decompiler_data = all_results[7].data
            if isinstance(decompiler_data, dict):
                enhanced_functions = decompiler_data.get('enhanced_functions', {})
                for func_name, func_data in enhanced_functions.items():
                    if isinstance(func_data, dict) and func_data.get('code'):
                        sources['source_files'][f"{func_name}.c"] = func_data['code']
        
        # Gather from Basic Decompiler (Agent 4)
        if 4 in all_results and hasattr(all_results[4], 'status') and all_results[4].status == AgentStatus.COMPLETED:
            basic_data = all_results[4].data
            if isinstance(basic_data, dict):
                decompiled_functions = basic_data.get('decompiled_functions', {})
                for func_name, func_data in decompiled_functions.items():
                    if isinstance(func_data, dict) and func_data.get('code'):
                        # Only add if not already present from advanced decompiler
                        file_name = f"{func_name}.c"
                        if file_name not in sources['source_files']:
                            sources['source_files'][file_name] = func_data['code']
        
        return sources

    def _analyze_build_requirements(self, sources: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze build requirements from available sources"""
        analysis = {
            'project_type': 'executable',
            'target_platform': 'windows',
            'architecture': 'x86',
            'compiler_requirements': [],
            'linker_requirements': [],
            'dependencies': [],
            'include_paths': [],
            'library_paths': [],
            'compilation_flags': [],
            'optimization_level': 'O2',
            'debug_symbols': True,
            'static_linking': False,
            'build_complexity': 'simple'
        }
        
        # Analyze source files for requirements
        source_files = sources.get('source_files', {})
        header_files = sources.get('header_files', {})
        
        # Detect project type
        if any('WinMain' in content for content in source_files.values()):
            analysis['project_type'] = 'windows_gui'
        elif any('DllMain' in content for content in source_files.values()):
            analysis['project_type'] = 'dynamic_library'
        elif sources.get('main_function'):
            analysis['project_type'] = 'console_application'
        
        # Analyze includes and dependencies
        all_content = ' '.join(source_files.values()) + ' '.join(header_files.values())
        
        # Windows-specific dependencies
        if '#include <windows.h>' in all_content:
            analysis['dependencies'].extend(['kernel32.lib', 'user32.lib'])
        if '#include <winsock2.h>' in all_content:
            analysis['dependencies'].extend(['ws2_32.lib', 'wsock32.lib'])
        if 'printf' in all_content or '#include <stdio.h>' in all_content:
            analysis['dependencies'].extend(['msvcrt.lib'])
        
        # Determine architecture from context
        if 2 in context.get('agent_results', {}):
            arch_result = context['agent_results'][2]
            if hasattr(arch_result, 'data') and isinstance(arch_result.data, dict):
                arch_info = arch_result.data.get('architecture', {})
                if isinstance(arch_info, dict):
                    analysis['architecture'] = arch_info.get('architecture', 'x86')
                elif isinstance(arch_info, str):
                    analysis['architecture'] = arch_info
        
        # Set compiler requirements based on analysis
        if analysis['target_platform'] == 'windows':
            analysis['compiler_requirements'] = ['msvc', 'gcc-mingw']
            if analysis['architecture'] == 'x64':
                analysis['compilation_flags'].extend(['/MACHINE:X64', '-m64'])
            else:
                analysis['compilation_flags'].extend(['/MACHINE:X86', '-m32'])
        
        # Determine build complexity
        num_files = len(source_files) + len(header_files)
        if num_files > 20:
            analysis['build_complexity'] = 'complex'
        elif num_files > 5:
            analysis['build_complexity'] = 'moderate'
        
        return analysis

    def _generate_build_system(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive build system configuration"""
        build_system = {
            'detected_systems': [],
            'primary_system': 'cmake',
            'build_files': {},
            'build_configurations': {},
            'compilation_commands': {},
            'linking_commands': {},
            'build_scripts': {},
            'automated_build': {}
        }
        
        # Generate CMakeLists.txt
        cmake_content = self._generate_cmake_file(analysis)
        build_system['build_files']['CMakeLists.txt'] = cmake_content
        build_system['detected_systems'].append('cmake')
        
        # Generate Makefile
        makefile_content = self._generate_makefile(analysis)
        build_system['build_files']['Makefile'] = makefile_content
        build_system['detected_systems'].append('make')
        
        # Generate Visual Studio project files
        if analysis['target_platform'] == 'windows':
            vcxproj_content = self._generate_vcxproj_file(analysis)
            build_system['build_files']['project.vcxproj'] = vcxproj_content
            build_system['detected_systems'].append('msbuild')
        
        # Generate build configurations
        build_system['build_configurations'] = {
            'debug': {
                'optimization': 'Od' if analysis['target_platform'] == 'windows' else 'O0',
                'debug_symbols': True,
                'defines': ['DEBUG', '_DEBUG'],
                'runtime_checks': True
            },
            'release': {
                'optimization': analysis['optimization_level'],
                'debug_symbols': False,
                'defines': ['NDEBUG', 'RELEASE'],
                'runtime_checks': False
            },
            'relwithdebinfo': {
                'optimization': 'O2',
                'debug_symbols': True,
                'defines': ['NDEBUG'],
                'runtime_checks': False
            }
        }
        
        # Generate compilation commands
        build_system['compilation_commands'] = self._generate_compilation_commands(analysis)
        
        # Generate linking commands
        build_system['linking_commands'] = self._generate_linking_commands(analysis)
        
        # Generate automated build scripts
        build_system['build_scripts'] = self._generate_build_scripts(analysis)
        
        return build_system

    def _generate_cmake_file(self, analysis: Dict[str, Any]) -> str:
        """Generate CMakeLists.txt file"""
        project_name = "ReconstructedProject"
        
        cmake_content = f"""cmake_minimum_required(VERSION 3.10)
project({project_name})

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set architecture
if("{analysis['architecture']}" STREQUAL "x64")
    set(CMAKE_GENERATOR_PLATFORM x64)
else()
    set(CMAKE_GENERATOR_PLATFORM Win32)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${{CMAKE_BINARY_DIR}}/lib)

# Compiler-specific options
if(MSVC)
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} /W3")
    set(CMAKE_C_FLAGS_DEBUG "${{CMAKE_C_FLAGS_DEBUG}} /Od /Zi")
    set(CMAKE_C_FLAGS_RELEASE "${{CMAKE_C_FLAGS_RELEASE}} /O2 /DNDEBUG")
else()
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -Wall -Wextra")
    set(CMAKE_C_FLAGS_DEBUG "${{CMAKE_C_FLAGS_DEBUG}} -g -O0")
    set(CMAKE_C_FLAGS_RELEASE "${{CMAKE_C_FLAGS_RELEASE}} -O2 -DNDEBUG")
endif()

# Source files
file(GLOB_RECURSE SOURCES "src/*.c")
file(GLOB_RECURSE HEADERS "src/*.h")

# Include directories
include_directories(src)

"""
        
        # Add target based on project type
        if analysis['project_type'] == 'dynamic_library':
            cmake_content += f"""# Create shared library
add_library({project_name} SHARED ${{SOURCES}})
"""
        elif analysis['project_type'] == 'static_library':
            cmake_content += f"""# Create static library
add_library({project_name} STATIC ${{SOURCES}})
"""
        else:
            cmake_content += f"""# Create executable
add_executable({project_name} ${{SOURCES}})
"""
        
        # Add dependencies
        if analysis['dependencies']:
            cmake_content += "\n# Link libraries\n"
            for dep in analysis['dependencies']:
                if dep.endswith('.lib'):
                    lib_name = dep[:-4]  # Remove .lib extension
                    cmake_content += f"target_link_libraries({project_name} {lib_name})\n"
        
        # Windows-specific settings
        if analysis['target_platform'] == 'windows':
            cmake_content += f"""
# Windows-specific settings
if(WIN32)
    target_compile_definitions({project_name} PRIVATE WIN32_LEAN_AND_MEAN)
    if("{analysis['project_type']}" STREQUAL "windows_gui")
        set_target_properties({project_name} PROPERTIES
            WIN32_EXECUTABLE TRUE
        )
    endif()
endif()
"""
        
        return cmake_content

    def _generate_makefile(self, analysis: Dict[str, Any]) -> str:
        """Generate Makefile"""
        compiler = 'gcc' if analysis['target_platform'] != 'windows' else 'cl'
        
        makefile = f"""# Generated Makefile for reconstructed project
CC = {compiler}
"""
        
        # Set flags based on compiler
        if compiler == 'gcc':
            makefile += f"""CFLAGS = -Wall -Wextra -std=c99
CFLAGS_DEBUG = -g -O0 -DDEBUG
CFLAGS_RELEASE = -O2 -DNDEBUG
LDFLAGS = {' '.join(f'-l{dep[:-4]}' for dep in analysis['dependencies'] if dep.endswith('.lib'))}
"""
        else:
            makefile += f"""CFLAGS = /W3 /std:c99
CFLAGS_DEBUG = /Od /Zi /DDEBUG
CFLAGS_RELEASE = /O2 /DNDEBUG
LDFLAGS = {' '.join(analysis['dependencies'])}
"""
        
        makefile += f"""
# Target executable
TARGET = reconstructed_project
SRCDIR = src
OBJDIR = obj

# Source files (will be populated by build system)
SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

# Default target
all: $(TARGET)

# Debug build
debug: CFLAGS += $(CFLAGS_DEBUG)
debug: $(TARGET)

# Release build
release: CFLAGS += $(CFLAGS_RELEASE)
release: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Install (placeholder)
install: $(TARGET)
	@echo "Installation not configured"

.PHONY: all debug release clean install
"""
        
        return makefile

    def _generate_vcxproj_file(self, analysis: Dict[str, Any]) -> str:
        """Generate Visual Studio project file"""
        project_guid = "{" + "12345678-1234-5678-9ABC-123456789012" + "}"
        
        if analysis['architecture'] == 'x64':
            platform = 'x64'
            platform_toolset = 'v143'
        else:
            platform = 'Win32'
            platform_toolset = 'v143'
        
        config_type = 'Application'
        if analysis['project_type'] == 'dynamic_library':
            config_type = 'DynamicLibrary'
        elif analysis['project_type'] == 'static_library':
            config_type = 'StaticLibrary'
        
        vcxproj = f"""<?xml version="1.0" encoding="utf-8"?>
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
    <RootNamespace>ReconstructedProject</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|{platform}'" Label="Configuration">
    <ConfigurationType>{config_type}</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>{platform_toolset}</PlatformToolset>
    <CharacterSet>MultiByte</CharacterSet>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|{platform}'" Label="Configuration">
    <ConfigurationType>{config_type}</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>{platform_toolset}</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|{platform}'">
    <OutDir>$(SolutionDir)bin\\$(Configuration)\\$(Platform)\\</OutDir>
    <IntDir>$(SolutionDir)obj\\$(Configuration)\\$(Platform)\\</IntDir>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|{platform}'">
    <OutDir>$(SolutionDir)bin\\$(Configuration)\\$(Platform)\\</OutDir>
    <IntDir>$(SolutionDir)obj\\$(Configuration)\\$(Platform)\\</IntDir>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|{platform}'">
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
      <AdditionalDependencies>{';'.join(analysis['dependencies'])};%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
  </ItemDefinitionGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|{platform}'">
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
      <AdditionalDependencies>{';'.join(analysis['dependencies'])};%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="src\\*.c" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="src\\*.h" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>"""
        
        return vcxproj

    def _generate_compilation_commands(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate compilation commands for different compilers"""
        commands = {}
        
        # GCC commands
        gcc_base = ['gcc', '-std=c99', '-Wall', '-Wextra']
        if analysis['architecture'] == 'x64':
            gcc_base.append('-m64')
        else:
            gcc_base.append('-m32')
        
        commands['gcc'] = {
            'debug': gcc_base + ['-g', '-O0', '-DDEBUG'],
            'release': gcc_base + ['-O2', '-DNDEBUG'],
            'includes': ['-Isrc'],
            'output': ['-o', 'reconstructed_project']
        }
        
        # MSVC commands
        msvc_base = ['cl', '/std:c11', '/W3']
        if analysis['architecture'] == 'x64':
            msvc_base.append('/MACHINE:X64')
        else:
            msvc_base.append('/MACHINE:X86')
        
        commands['msvc'] = {
            'debug': msvc_base + ['/Od', '/Zi', '/DDEBUG'],
            'release': msvc_base + ['/O2', '/DNDEBUG'],
            'includes': ['/Isrc'],
            'output': ['/Fe:reconstructed_project.exe']
        }
        
        # Clang commands
        clang_base = ['clang', '-std=c99', '-Wall', '-Wextra']
        if analysis['architecture'] == 'x64':
            clang_base.append('-m64')
        else:
            clang_base.append('-m32')
        
        commands['clang'] = {
            'debug': clang_base + ['-g', '-O0', '-DDEBUG'],
            'release': clang_base + ['-O2', '-DNDEBUG'],
            'includes': ['-Isrc'],
            'output': ['-o', 'reconstructed_project']
        }
        
        return commands

    def _generate_linking_commands(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate linking commands"""
        commands = {}
        
        # Base libraries
        libs = analysis['dependencies']
        
        # GCC/Clang linking
        gcc_libs = [f"-l{lib[:-4]}" if lib.endswith('.lib') else f"-l{lib}" for lib in libs]
        commands['gcc'] = gcc_libs
        commands['clang'] = gcc_libs
        
        # MSVC linking
        commands['msvc'] = [f"/DEFAULTLIB:{lib}" for lib in libs if lib.endswith('.lib')]
        
        return commands

    def _generate_build_scripts(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate automated build scripts"""
        scripts = {}
        
        # Windows batch script
        batch_script = f"""@echo off
echo Building reconstructed project...

REM Create directories
if not exist "bin" mkdir bin
if not exist "obj" mkdir obj

REM Build with different configurations
echo.
echo === Debug Build ===
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

echo.
echo === Release Build ===
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

echo.
echo === Testing builds ===
if exist "build-debug\\bin\\Debug\\ReconstructedProject.exe" (
    echo Debug build: SUCCESS
) else (
    echo Debug build: FAILED
)

if exist "build-release\\bin\\Release\\ReconstructedProject.exe" (
    echo Release build: SUCCESS
) else (
    echo Release build: FAILED
)

echo.
echo Build complete!
pause
"""
        scripts['build.bat'] = batch_script
        
        # Linux shell script
        shell_script = f"""#!/bin/bash
echo "Building reconstructed project..."

# Create directories
mkdir -p bin obj

# Build with different configurations
echo
echo "=== Debug Build ==="
cmake -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

echo
echo "=== Release Build ==="
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release

echo
echo "=== Testing builds ==="
if [ -f "build-debug/bin/ReconstructedProject" ]; then
    echo "Debug build: SUCCESS"
else
    echo "Debug build: FAILED"
fi

if [ -f "build-release/bin/ReconstructedProject" ]; then
    echo "Release build: SUCCESS"
else
    echo "Release build: FAILED"
fi

echo
echo "Build complete!"
"""
        scripts['build.sh'] = shell_script
        
        return scripts

    def _orchestrate_compilation(self, build_system: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the compilation process"""
        results = {
            'attempted_builds': [],
            'successful_builds': [],
            'failed_builds': [],
            'build_outputs': {},
            'error_logs': {},
            'success_rate': 0.0,
            'compilation_time': {},
            'binary_outputs': {}
        }
        
        # Try different build systems
        build_systems_to_try = ['cmake', 'make', 'msbuild']
        
        for build_system_name in build_systems_to_try:
            if build_system_name in build_system['detected_systems']:
                build_result = self._attempt_build(build_system_name, build_system, context)
                results['attempted_builds'].append(build_system_name)
                
                if build_result['success']:
                    results['successful_builds'].append(build_system_name)
                    results['build_outputs'][build_system_name] = build_result['output']
                    results['compilation_time'][build_system_name] = build_result['time']
                    if build_result.get('binary_path'):
                        results['binary_outputs'][build_system_name] = build_result['binary_path']
                else:
                    results['failed_builds'].append(build_system_name)
                    results['error_logs'][build_system_name] = build_result['error']
        
        # Calculate success rate
        if results['attempted_builds']:
            results['success_rate'] = len(results['successful_builds']) / len(results['attempted_builds'])
        
        return results

    def _attempt_build(self, build_system: str, build_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to build using specified build system"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'time': 0.0,
            'binary_path': None
        }
        
        try:
            # Create temporary build directory
            output_dir = context.get('output_paths', {}).get('compilation', 'output/compilation')
            os.makedirs(output_dir, exist_ok=True)
            
            import time
            start_time = time.time()
            
            if build_system == 'cmake':
                result = self._build_with_cmake(output_dir, build_config)
            elif build_system == 'make':
                result = self._build_with_make(output_dir, build_config)
            elif build_system == 'msbuild':
                result = self._build_with_msbuild(output_dir, build_config)
            
            result['time'] = time.time() - start_time
            
        except Exception as e:
            result['error'] = f"Build system {build_system} failed: {str(e)}"
        
        return result

    def _build_with_cmake(self, output_dir: str, build_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build using CMake"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'binary_path': None
        }
        
        try:
            # Write CMakeLists.txt
            cmake_file = os.path.join(output_dir, 'CMakeLists.txt')
            with open(cmake_file, 'w') as f:
                f.write(build_config['build_files']['CMakeLists.txt'])
            
            # Configure with CMake
            build_dir = os.path.join(output_dir, 'build')
            configure_cmd = ['cmake', '-B', build_dir, '-S', output_dir]
            
            configure_result = subprocess.run(
                configure_cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if configure_result.returncode != 0:
                result['error'] = f"CMake configure failed: {configure_result.stderr}"
                return result
            
            # Build with CMake
            build_cmd = ['cmake', '--build', build_dir]
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            result['output'] = configure_result.stdout + "\n" + build_result.stdout
            
            if build_result.returncode == 0:
                result['success'] = True
                # Try to find the built binary
                for root, dirs, files in os.walk(build_dir):
                    for file in files:
                        if file.endswith('.exe') or (os.name != 'nt' and os.access(os.path.join(root, file), os.X_OK)):
                            result['binary_path'] = os.path.join(root, file)
                            break
            else:
                result['error'] = f"CMake build failed: {build_result.stderr}"
            
        except subprocess.TimeoutExpired:
            result['error'] = "CMake build timed out"
        except Exception as e:
            result['error'] = f"CMake build error: {str(e)}"
        
        return result

    def _build_with_make(self, output_dir: str, build_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build using Make"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'binary_path': None
        }
        
        try:
            # Write Makefile
            makefile = os.path.join(output_dir, 'Makefile')
            with open(makefile, 'w') as f:
                f.write(build_config['build_files']['Makefile'])
            
            # Run make
            make_cmd = ['make', '-C', output_dir]
            make_result = subprocess.run(
                make_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            result['output'] = make_result.stdout
            
            if make_result.returncode == 0:
                result['success'] = True
                # Look for binary output
                binary_path = os.path.join(output_dir, 'reconstructed_project')
                if os.path.exists(binary_path):
                    result['binary_path'] = binary_path
            else:
                result['error'] = f"Make build failed: {make_result.stderr}"
            
        except subprocess.TimeoutExpired:
            result['error'] = "Make build timed out"
        except Exception as e:
            result['error'] = f"Make build error: {str(e)}"
        
        return result

    def _build_with_msbuild(self, output_dir: str, build_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build using MSBuild"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'binary_path': None
        }
        
        try:
            # Write project file
            proj_file = os.path.join(output_dir, 'project.vcxproj')
            with open(proj_file, 'w') as f:
                f.write(build_config['build_files']['project.vcxproj'])
            
            # Run MSBuild
            msbuild_cmd = ['msbuild', proj_file, '/p:Configuration=Release']
            msbuild_result = subprocess.run(
                msbuild_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            result['output'] = msbuild_result.stdout
            
            if msbuild_result.returncode == 0:
                result['success'] = True
                # Look for binary output
                bin_dir = os.path.join(output_dir, 'bin', 'Release')
                if os.path.exists(bin_dir):
                    for file in os.listdir(bin_dir):
                        if file.endswith('.exe'):
                            result['binary_path'] = os.path.join(bin_dir, file)
                            break
            else:
                result['error'] = f"MSBuild failed: {msbuild_result.stderr}"
            
        except subprocess.TimeoutExpired:
            result['error'] = "MSBuild timed out"
        except Exception as e:
            result['error'] = f"MSBuild error: {str(e)}"
        
        return result

    def _optimize_build_process(self, compilation_results: Dict[str, Any], build_system: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the build process based on results"""
        optimization = {
            'recommended_system': None,
            'optimization_suggestions': [],
            'parallel_build_options': {},
            'cache_strategies': {},
            'efficiency_score': 0.0,
            'performance_metrics': {}
        }
        
        # Determine best build system
        successful_builds = compilation_results.get('successful_builds', [])
        if successful_builds:
            # Prefer CMake, then Make, then MSBuild
            if 'cmake' in successful_builds:
                optimization['recommended_system'] = 'cmake'
            elif 'make' in successful_builds:
                optimization['recommended_system'] = 'make'
            else:
                optimization['recommended_system'] = successful_builds[0]
        
        # Generate optimization suggestions
        optimization['optimization_suggestions'] = [
            "Enable parallel compilation (-j flag for make, /MP for MSBuild)",
            "Use precompiled headers for large projects",
            "Implement incremental builds",
            "Configure build caching (ccache, sccache)",
            "Optimize linker settings for faster linking"
        ]
        
        # Calculate efficiency score
        success_rate = compilation_results.get('success_rate', 0.0)
        avg_time = 0.0
        if compilation_results.get('compilation_time'):
            avg_time = sum(compilation_results['compilation_time'].values()) / len(compilation_results['compilation_time'])
        
        # Efficiency score based on success rate and speed (lower time is better)
        time_score = max(0.0, 1.0 - (avg_time / 300.0))  # Normalize around 5 minutes
        optimization['efficiency_score'] = (success_rate + time_score) / 2.0
        
        return optimization

    def _create_build_orchestration(self, optimized_build: Dict[str, Any]) -> Dict[str, Any]:
        """Create build orchestration plan"""
        orchestration = {
            'build_pipeline': [],
            'parallel_stages': {},
            'dependency_order': [],
            'quality_gates': {},
            'automation_scripts': {},
            'ci_cd_integration': {}
        }
        
        # Define build pipeline stages
        orchestration['build_pipeline'] = [
            'pre_build_validation',
            'dependency_resolution',
            'source_compilation',
            'linking',
            'post_build_validation',
            'artifact_generation'
        ]
        
        # Define parallel stages
        orchestration['parallel_stages'] = {
            'source_compilation': [
                'compile_main_sources',
                'compile_resource_files',
                'generate_headers'
            ],
            'validation': [
                'syntax_validation',
                'semantic_validation',
                'security_scanning'
            ]
        }
        
        # Quality gates
        orchestration['quality_gates'] = {
            'compilation_success': {'threshold': 1.0, 'required': True},
            'warning_count': {'threshold': 10, 'required': False},
            'code_coverage': {'threshold': 0.7, 'required': False},
            'security_issues': {'threshold': 0, 'required': True}
        }
        
        return orchestration

    def _calculate_machine_metrics(self, compilation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for The Machine's performance"""
        metrics = {
            'build_success_rate': compilation_results.get('success_rate', 0.0),
            'average_build_time': 0.0,
            'systems_attempted': len(compilation_results.get('attempted_builds', [])),
            'systems_successful': len(compilation_results.get('successful_builds', [])),
            'efficiency_rating': 'unknown',
            'reliability_score': 0.0,
            'automation_level': 0.9  # High automation in The Machine
        }
        
        # Calculate average build time
        if compilation_results.get('compilation_time'):
            metrics['average_build_time'] = sum(compilation_results['compilation_time'].values()) / len(compilation_results['compilation_time'])
        
        # Calculate reliability score
        if metrics['systems_attempted'] > 0:
            metrics['reliability_score'] = metrics['systems_successful'] / metrics['systems_attempted']
        
        # Determine efficiency rating
        if metrics['build_success_rate'] >= 0.8:
            metrics['efficiency_rating'] = 'excellent'
        elif metrics['build_success_rate'] >= 0.6:
            metrics['efficiency_rating'] = 'good'
        elif metrics['build_success_rate'] >= 0.4:
            metrics['efficiency_rating'] = 'fair'
        else:
            metrics['efficiency_rating'] = 'poor'
        
        return metrics

    def get_description(self) -> str:
        """Get description of The Machine agent"""
        return "The Machine orchestrates compilation processes and manages complex build systems for reconstructed code"

    def get_dependencies(self) -> List[int]:
        """Get dependencies for The Machine"""
        return [8, 9]  # Resource reconstruction and global reconstruction