#!/usr/bin/env python3
"""
Build System Automation Script
Automates build system tasks like CMake generation, MSBuild project creation, and compilation testing.
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import uuid


def setup_logging():
    """Setup logging for build system automation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class BuildSystemAutomation:
    """Automates build system generation and compilation testing."""
    
    def __init__(self, output_root: Path):
        self.logger = setup_logging()
        self.output_root = Path(output_root).resolve()
        self.compilation_dir = self.output_root / "compilation"
        
        # Ensure output directory exists and is valid
        self._validate_output_directory()
    
    def _validate_output_directory(self):
        """Validate that we're working within the output directory."""
        if not str(self.output_root).endswith('output'):
            if 'output' not in str(self.output_root):
                raise ValueError(f"Output root must be under 'output' directory, got: {self.output_root}")
        
        self.compilation_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Build system working in: {self.compilation_dir}")
    
    def generate_cmake_project(self, project_name: str, source_files: List[Path], 
                             include_dirs: List[Path] = None, 
                             libraries: List[str] = None) -> Path:
        """Generate CMakeLists.txt file for the project."""
        if include_dirs is None:
            include_dirs = []
        if libraries is None:
            libraries = []
        
        cmake_content = self._create_cmake_content(project_name, source_files, include_dirs, libraries)
        cmake_file = self.compilation_dir / "CMakeLists.txt"
        
        with open(cmake_file, 'w') as f:
            f.write(cmake_content)
        
        self.logger.info(f"Generated CMakeLists.txt: {cmake_file}")
        return cmake_file
    
    def _create_cmake_content(self, project_name: str, source_files: List[Path],
                            include_dirs: List[Path], libraries: List[str]) -> str:
        """Create CMakeLists.txt content."""
        content = f"""# Generated CMakeLists.txt for {project_name}
cmake_minimum_required(VERSION 3.16)
project({project_name} LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} /W3")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} /W3")
else()
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -Wall -Wextra")
    set(CMAKE_C_FLAGS "${{CMAKE_C_FLAGS}} -Wall -Wextra")
endif()

# Find required packages
find_package(Threads REQUIRED)

"""
        
        # Add source files
        if source_files:
            content += "# Source files\n"
            content += "set(SOURCES\n"
            for src_file in source_files:
                # Make path relative to compilation directory
                rel_path = src_file.relative_to(self.compilation_dir) if src_file.is_absolute() else src_file
                content += f"    {rel_path}\n"
            content += ")\n\n"
        
        # Add include directories
        if include_dirs:
            content += "# Include directories\n"
            content += "set(INCLUDE_DIRS\n"
            for inc_dir in include_dirs:
                rel_path = inc_dir.relative_to(self.compilation_dir) if inc_dir.is_absolute() else inc_dir
                content += f"    {rel_path}\n"
            content += ")\n\n"
        
        # Create executable
        if source_files:
            content += f"# Create executable\n"
            content += f"add_executable(${{PROJECT_NAME}} ${{SOURCES}})\n\n"
            
            # Set include directories
            if include_dirs:
                content += f"# Set include directories\n"
                content += f"target_include_directories(${{PROJECT_NAME}} PRIVATE ${{INCLUDE_DIRS}})\n\n"
            
            # Link libraries
            content += f"# Link libraries\n"
            content += f"target_link_libraries(${{PROJECT_NAME}} Threads::Threads)\n"
            
            # Platform-specific libraries
            content += f"""
# Platform-specific libraries
if(WIN32)
    target_link_libraries(${{PROJECT_NAME}} kernel32 user32 gdi32 winspool comdlg32 advapi32 shell32 ole32 oleaut32 uuid odbc32 odbccp32)
elseif(UNIX)
    target_link_libraries(${{PROJECT_NAME}} m dl)
endif()

"""
            
            # Additional libraries
            for lib in libraries:
                content += f"target_link_libraries(${{PROJECT_NAME}} {lib})\n"
        
        return content
    
    def generate_msbuild_project(self, project_name: str, source_files: List[Path],
                                include_dirs: List[Path] = None) -> Path:
        """Generate MSBuild project file (.vcxproj)."""
        if include_dirs is None:
            include_dirs = []
        
        # Generate project GUID
        project_guid = str(uuid.uuid4()).upper()
        
        vcxproj_content = self._create_msbuild_content(project_name, source_files, include_dirs, project_guid)
        vcxproj_file = self.compilation_dir / f"{project_name}.vcxproj"
        
        with open(vcxproj_file, 'w', encoding='utf-8') as f:
            f.write(vcxproj_content)
        
        # Generate solution file
        sln_content = self._create_solution_content(project_name, project_guid)
        sln_file = self.compilation_dir / f"{project_name}.sln"
        
        with open(sln_file, 'w', encoding='utf-8') as f:
            f.write(sln_content)
        
        self.logger.info(f"Generated MSBuild project: {vcxproj_file}")
        self.logger.info(f"Generated solution file: {sln_file}")
        
        return vcxproj_file
    
    def _create_msbuild_content(self, project_name: str, source_files: List[Path],
                              include_dirs: List[Path], project_guid: str) -> str:
        """Create MSBuild project content."""
        content = f"""<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  
  <!-- Configuration -->
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  
  <!-- Global Properties -->
  <PropertyGroup Label="Globals">
    <ProjectGuid>{{{project_guid}}}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>{project_name}</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  
  <!-- Import Default Props -->
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  
  <!-- Configuration Properties -->
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  
  <!-- Import Cpp Props -->
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  
  <!-- Extension Settings -->
  <ImportGroup Label="ExtensionSettings">
  </ImportGroup>
  
  <!-- Shared -->
  <ImportGroup Label="Shared">
  </ImportGroup>
  
  <!-- Property Sheets -->
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <Import Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  <ImportGroup Label="PropertySheets" Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <Import Project="$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props" Condition="exists('$(UserRootDir)\\Microsoft.Cpp.$(Platform).user.props')" Label="LocalAppDataPlatform" />
  </ImportGroup>
  
  <!-- User Macros -->
  <PropertyGroup />
  
  <!-- Property Groups -->
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <LinkIncremental>true</LinkIncremental>
"""
        
        # Add include directories
        if include_dirs:
            content += "    <IncludePath>"
            include_paths = []
            for inc_dir in include_dirs:
                rel_path = inc_dir.relative_to(self.compilation_dir) if inc_dir.is_absolute() else inc_dir
                include_paths.append(str(rel_path).replace('/', '\\'))
            content += ";".join(include_paths) + ";$(IncludePath)</IncludePath>\n"
        
        content += """  </PropertyGroup>
  
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <LinkIncremental>false</LinkIncremental>
  </PropertyGroup>
  
  <!-- Item Definition Groups -->
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>true</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  
  <!-- Source Files -->
  <ItemGroup>
"""
        
        # Add source files
        for src_file in source_files:
            rel_path = src_file.relative_to(self.compilation_dir) if src_file.is_absolute() else src_file
            file_ext = src_file.suffix.lower()
            
            if file_ext in ['.c']:
                content += f'    <ClCompile Include="{rel_path}" />\n'
            elif file_ext in ['.cpp', '.cxx', '.cc']:
                content += f'    <ClCompile Include="{rel_path}" />\n'
            elif file_ext in ['.h', '.hpp', '.hxx']:
                content += f'    <ClInclude Include="{rel_path}" />\n'
        
        content += """  </ItemGroup>
  
  <!-- Import Targets -->
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
  
  <!-- Extension Targets -->
  <ImportGroup Label="ExtensionTargets">
  </ImportGroup>
  
</Project>"""
        
        return content
    
    def _create_solution_content(self, project_name: str, project_guid: str) -> str:
        """Create Visual Studio solution content."""
        solution_guid = str(uuid.uuid4()).upper()
        
        content = f"""
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project("{{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}}") = "{project_name}", "{project_name}.vcxproj", "{{{project_guid}}}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|x64 = Debug|x64
		Release|x64 = Release|x64
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{{{project_guid}}}.Debug|x64.ActiveCfg = Debug|x64
		{{{project_guid}}}.Debug|x64.Build.0 = Debug|x64
		{{{project_guid}}}.Release|x64.ActiveCfg = Release|x64
		{{{project_guid}}}.Release|x64.Build.0 = Release|x64
	EndGlobalSection
	GlobalSection(SolutionProperties) = preSolution
		HideSolutionNode = FALSE
	EndGlobalSection
	GlobalSection(ExtensibilityGlobals) = postSolution
		SolutionGuid = {{{solution_guid}}}
	EndGlobalSection
EndGlobal
"""
        return content
    
    def generate_makefile(self, project_name: str, source_files: List[Path],
                         include_dirs: List[Path] = None, libraries: List[str] = None) -> Path:
        """Generate Makefile for Unix systems."""
        if include_dirs is None:
            include_dirs = []
        if libraries is None:
            libraries = []
        
        makefile_content = self._create_makefile_content(project_name, source_files, include_dirs, libraries)
        makefile_path = self.compilation_dir / "Makefile"
        
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        
        self.logger.info(f"Generated Makefile: {makefile_path}")
        return makefile_path
    
    def _create_makefile_content(self, project_name: str, source_files: List[Path],
                               include_dirs: List[Path], libraries: List[str]) -> str:
        """Create Makefile content."""
        # Determine compiler based on file extensions
        has_cpp = any(src.suffix.lower() in ['.cpp', '.cxx', '.cc'] for src in source_files)
        compiler = "CXX = g++" if has_cpp else "CC = gcc"
        
        content = f"""# Generated Makefile for {project_name}

# Compiler settings
{compiler}
CFLAGS = -Wall -Wextra -std=c11
CXXFLAGS = -Wall -Wextra -std=c++17

# Project settings
TARGET = {project_name}
"""
        
        # Add include directories
        if include_dirs:
            content += "INCLUDES = "
            for inc_dir in include_dirs:
                rel_path = inc_dir.relative_to(self.compilation_dir) if inc_dir.is_absolute() else inc_dir
                content += f"-I{rel_path} "
            content += "\n"
        else:
            content += "INCLUDES = \n"
        
        # Add libraries
        if libraries:
            content += "LIBS = "
            for lib in libraries:
                if lib.startswith('-'):
                    content += f"{lib} "
                else:
                    content += f"-l{lib} "
            content += "\n"
        else:
            content += "LIBS = -lm -ldl -lpthread\n"
        
        # Add source files
        content += "\n# Source files\n"
        content += "SOURCES = "
        for src_file in source_files:
            rel_path = src_file.relative_to(self.compilation_dir) if src_file.is_absolute() else src_file
            content += f"{rel_path} "
        content += "\n"
        
        # Object files
        content += "\n# Object files\n"
        content += "OBJECTS = $(SOURCES:.c=.o)\n"
        content += "OBJECTS := $(OBJECTS:.cpp=.o)\n"
        
        # Build rules
        content += f"""
# Build rules
all: $(TARGET)

$(TARGET): $(OBJECTS)
\t{'$(CXX)' if has_cpp else '$(CC)'} $(OBJECTS) -o $(TARGET) $(LIBS)

.c.o:
\t$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.cpp.o:
\t$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
\trm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
"""
        
        return content
    
    def test_compilation(self, build_system: str = 'auto') -> Dict:
        """Test compilation using the specified build system."""
        if build_system == 'auto':
            if platform.system() == 'Windows':
                build_system = 'msbuild'
            else:
                build_system = 'make'
        
        compilation_result = {
            'build_system': build_system,
            'success': False,
            'output': '',
            'errors': '',
            'warnings': [],
            'executable_created': False
        }
        
        try:
            if build_system == 'cmake':
                result = self._test_cmake_compilation()
            elif build_system == 'msbuild':
                result = self._test_msbuild_compilation()
            elif build_system == 'make':
                result = self._test_make_compilation()
            else:
                raise ValueError(f"Unsupported build system: {build_system}")
            
            compilation_result.update(result)
            
        except Exception as e:
            compilation_result['errors'] = str(e)
            self.logger.error(f"Compilation test failed: {e}")
        
        return compilation_result
    
    def _test_cmake_compilation(self) -> Dict:
        """Test CMake-based compilation."""
        build_dir = self.compilation_dir / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Configure with CMake
        configure_cmd = ["cmake", "..", "-G", "Unix Makefiles"]
        if platform.system() == 'Windows':
            configure_cmd = ["cmake", "..", "-G", "Visual Studio 17 2022", "-A", "x64"]
        
        self.logger.info(f"Running CMake configure: {' '.join(configure_cmd)}")
        configure_result = subprocess.run(configure_cmd, cwd=build_dir, 
                                        capture_output=True, text=True, timeout=60)
        
        if configure_result.returncode != 0:
            return {
                'success': False,
                'output': configure_result.stdout,
                'errors': configure_result.stderr
            }
        
        # Build with CMake
        build_cmd = ["cmake", "--build", ".", "--config", "Release"]
        self.logger.info(f"Running CMake build: {' '.join(build_cmd)}")
        build_result = subprocess.run(build_cmd, cwd=build_dir,
                                    capture_output=True, text=True, timeout=300)
        
        # Check if executable was created
        executable_patterns = ["*", "*.exe", "Release/*", "Release/*.exe"]
        executable_created = False
        for pattern in executable_patterns:
            executables = list(build_dir.glob(pattern))
            if executables:
                executable_created = True
                break
        
        return {
            'success': build_result.returncode == 0,
            'output': configure_result.stdout + "\n" + build_result.stdout,
            'errors': configure_result.stderr + "\n" + build_result.stderr,
            'executable_created': executable_created
        }
    
    def _test_msbuild_compilation(self) -> Dict:
        """Test MSBuild compilation."""
        # Find solution file
        solution_files = list(self.compilation_dir.glob("*.sln"))
        if not solution_files:
            return {
                'success': False,
                'errors': 'No solution file found'
            }
        
        solution_file = solution_files[0]
        
        # Build with MSBuild
        build_cmd = ["msbuild", str(solution_file), "/p:Configuration=Release", "/p:Platform=x64"]
        self.logger.info(f"Running MSBuild: {' '.join(build_cmd)}")
        
        try:
            build_result = subprocess.run(build_cmd, cwd=self.compilation_dir,
                                        capture_output=True, text=True, timeout=300)
            
            # Check if executable was created
            executable_patterns = ["x64/Release/*.exe", "Release/*.exe", "*.exe"]
            executable_created = False
            for pattern in executable_patterns:
                executables = list(self.compilation_dir.glob(pattern))
                if executables:
                    executable_created = True
                    break
            
            return {
                'success': build_result.returncode == 0,
                'output': build_result.stdout,
                'errors': build_result.stderr,
                'executable_created': executable_created
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'errors': 'MSBuild timed out'
            }
        except FileNotFoundError:
            return {
                'success': False,
                'errors': 'MSBuild not found - install Visual Studio Build Tools'
            }
    
    def _test_make_compilation(self) -> Dict:
        """Test Make compilation."""
        # Check if Makefile exists
        makefile = self.compilation_dir / "Makefile"
        if not makefile.exists():
            return {
                'success': False,
                'errors': 'No Makefile found'
            }
        
        # Build with Make
        build_cmd = ["make", "-j4"]  # Use 4 parallel jobs
        self.logger.info(f"Running Make: {' '.join(build_cmd)}")
        
        try:
            build_result = subprocess.run(build_cmd, cwd=self.compilation_dir,
                                        capture_output=True, text=True, timeout=300)
            
            # Check if executable was created
            executables = list(self.compilation_dir.glob("*"))
            executable_created = any(f.is_file() and os.access(f, os.X_OK) for f in executables)
            
            return {
                'success': build_result.returncode == 0,
                'output': build_result.stdout,
                'errors': build_result.stderr,
                'executable_created': executable_created
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'errors': 'Make timed out'
            }
        except FileNotFoundError:
            return {
                'success': False,
                'errors': 'Make not found - install build-essential'
            }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build System Automation')
    parser.add_argument('--output-dir', required=True, help='Output directory (must be under /output/)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate CMake
    cmake_parser = subparsers.add_parser('cmake', help='Generate CMake project')
    cmake_parser.add_argument('--project-name', required=True, help='Project name')
    cmake_parser.add_argument('--sources', nargs='+', required=True, help='Source files')
    cmake_parser.add_argument('--includes', nargs='*', default=[], help='Include directories')
    cmake_parser.add_argument('--libraries', nargs='*', default=[], help='Libraries to link')
    
    # Generate MSBuild
    msbuild_parser = subparsers.add_parser('msbuild', help='Generate MSBuild project')
    msbuild_parser.add_argument('--project-name', required=True, help='Project name')
    msbuild_parser.add_argument('--sources', nargs='+', required=True, help='Source files')
    msbuild_parser.add_argument('--includes', nargs='*', default=[], help='Include directories')
    
    # Generate Makefile
    make_parser = subparsers.add_parser('makefile', help='Generate Makefile')
    make_parser.add_argument('--project-name', required=True, help='Project name')
    make_parser.add_argument('--sources', nargs='+', required=True, help='Source files')
    make_parser.add_argument('--includes', nargs='*', default=[], help='Include directories')
    make_parser.add_argument('--libraries', nargs='*', default=[], help='Libraries to link')
    
    # Test compilation
    test_parser = subparsers.add_parser('test', help='Test compilation')
    test_parser.add_argument('--build-system', choices=['auto', 'cmake', 'msbuild', 'make'], 
                           default='auto', help='Build system to use')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        automation = BuildSystemAutomation(args.output_dir)
        
        if args.command == 'cmake':
            source_paths = [Path(src) for src in args.sources]
            include_paths = [Path(inc) for inc in args.includes]
            cmake_file = automation.generate_cmake_project(
                args.project_name, source_paths, include_paths, args.libraries
            )
            print(f"Generated CMake project: {cmake_file}")
            
        elif args.command == 'msbuild':
            source_paths = [Path(src) for src in args.sources]
            include_paths = [Path(inc) for inc in args.includes]
            project_file = automation.generate_msbuild_project(
                args.project_name, source_paths, include_paths
            )
            print(f"Generated MSBuild project: {project_file}")
            
        elif args.command == 'makefile':
            source_paths = [Path(src) for src in args.sources]
            include_paths = [Path(inc) for inc in args.includes]
            makefile = automation.generate_makefile(
                args.project_name, source_paths, include_paths, args.libraries
            )
            print(f"Generated Makefile: {makefile}")
            
        elif args.command == 'test':
            result = automation.test_compilation(args.build_system)
            print(f"Compilation test using {result['build_system']}:")
            print(f"Success: {result['success']}")
            print(f"Executable created: {result['executable_created']}")
            if result['errors']:
                print(f"Errors:\n{result['errors']}")
            if result['output']:
                print(f"Output:\n{result['output']}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()