"""
Agent 18: Advanced Build Systems Generator
Provides multi-compiler support and advanced build system generation.
Phase 4: Build Systems & Production Readiness
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent18_AdvancedBuildSystems(BaseAgent):
    """Agent 18: Advanced build systems with multi-compiler support"""
    
    def __init__(self):
        super().__init__(
            agent_id=18,
            name="AdvancedBuildSystems",
            dependencies=[11, 12]  # Depends on GlobalReconstructor and CompilationOrchestrator
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute advanced build system generation"""
        agent11_result = context['agent_results'].get(11)
        agent12_result = context['agent_results'].get(12)
        
        if not agent11_result or agent11_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 11 (GlobalReconstructor) did not complete successfully"
            )

        try:
            global_reconstruction = agent11_result.data
            compilation_data = agent12_result.data if agent12_result else {}
            
            build_result = self._generate_advanced_build_systems(
                global_reconstruction, compilation_data, context
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=build_result,
                metadata={
                    'depends_on': [11, 12],
                    'analysis_type': 'advanced_build_systems'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Advanced build systems generation failed: {str(e)}"
            )

    def _generate_advanced_build_systems(self, global_reconstruction: Dict[str, Any], 
                                       compilation_data: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate advanced build systems with multi-compiler support"""
        result = {
            'detected_compilers': {},
            'generated_build_files': {},
            'compilation_attempts': {},
            'best_compiler': None,
            'cmake_generated': False,
            'makefile_generated': False,
            'ninja_generated': False,
            'vs_solution_generated': False,
            'successful_compilers': [],
            'failed_compilers': {},
            'overall_success': False
        }
        
        # Get output directory structure
        base_output_dir = context.get('output_dir', 'output')
        output_paths = context.get('output_paths', {})
        compilation_dir = output_paths.get('compilation', os.path.join(base_output_dir, 'compilation'))
        
        # Ensure compilation directory exists
        os.makedirs(compilation_dir, exist_ok=True)
        
        try:
            # Detect available compilers
            result['detected_compilers'] = self._detect_compilers()
            
            # Generate different build systems
            result['cmake_generated'] = self._generate_cmake_files(
                global_reconstruction, compilation_dir
            )
            result['makefile_generated'] = self._generate_makefile(
                global_reconstruction, compilation_dir
            )
            result['ninja_generated'] = self._generate_ninja_build(
                global_reconstruction, compilation_dir
            )
            result['vs_solution_generated'] = self._generate_vs_solution(
                global_reconstruction, compilation_dir
            )
            
            # Attempt compilation with each available compiler
            result['compilation_attempts'] = self._attempt_multi_compiler_build(
                result['detected_compilers'], compilation_dir, global_reconstruction
            )
            
            # Determine best compiler
            result['best_compiler'] = self._determine_best_compiler(
                result['compilation_attempts']
            )
            
            # Track successful compilers
            for compiler, attempt in result['compilation_attempts'].items():
                if attempt['success']:
                    result['successful_compilers'].append(compiler)
                else:
                    result['failed_compilers'][compiler] = attempt['errors']
            
            result['overall_success'] = len(result['successful_compilers']) > 0
            
        except Exception as e:
            result['error'] = str(e)
            result['overall_success'] = False
        
        return result

    def _detect_compilers(self) -> Dict[str, Dict[str, Any]]:
        """Detect available compilers on the system"""
        compilers = {}
        
        # Check for MSVC (Visual Studio)
        msvc_info = self._detect_msvc()
        if msvc_info['available']:
            compilers['msvc'] = msvc_info
        
        # Check for MinGW
        mingw_info = self._detect_mingw()
        if mingw_info['available']:
            compilers['mingw'] = mingw_info
        
        # Check for Clang
        clang_info = self._detect_clang()
        if clang_info['available']:
            compilers['clang'] = clang_info
        
        # Check for GCC (if on WSL or Linux)
        gcc_info = self._detect_gcc()
        if gcc_info['available']:
            compilers['gcc'] = gcc_info
        
        return compilers

    def _detect_msvc(self) -> Dict[str, Any]:
        """Detect Microsoft Visual Studio compiler"""
        info = {
            'available': False,
            'version': None,
            'path': None,
            'msbuild_path': None,
            'devenv_path': None,
            'priority': 1  # Highest priority for Windows
        }
        
        try:
            # Check environment variables first
            vs_install_dir = os.environ.get('VSINSTALLDIR')
            msbuild_path = os.environ.get('MSBUILD_PATH')
            devenv_path = os.environ.get('DEVENV_PATH')
            
            if vs_install_dir and msbuild_path and os.path.exists(msbuild_path):
                info['available'] = True
                info['path'] = vs_install_dir
                info['msbuild_path'] = msbuild_path
                info['devenv_path'] = devenv_path
                info['version'] = 'Environment'
                return info
            
            # Common Visual Studio paths
            vs_paths = [
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\",
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\",
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\",
                "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\",
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Enterprise\\",
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\",
                "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\"
            ]
            
            for vs_path in vs_paths:
                msbuild = os.path.join(vs_path, 'MSBuild\\Current\\Bin\\MSBuild.exe')
                if os.path.exists(msbuild):
                    info['available'] = True
                    info['path'] = vs_path
                    info['msbuild_path'] = msbuild
                    info['devenv_path'] = os.path.join(vs_path, 'Common7\\IDE\\devenv.exe')
                    info['version'] = '2022' if '2022' in vs_path else '2019'
                    break
                    
        except Exception:
            pass
        
        return info

    def _detect_mingw(self) -> Dict[str, Any]:
        """Detect MinGW compiler"""
        info = {
            'available': False,
            'version': None,
            'path': None,
            'gcc_path': None,
            'priority': 2
        }
        
        try:
            # Try to run gcc --version
            result = subprocess.run(['gcc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'mingw' in result.stdout.lower():
                info['available'] = True
                info['version'] = result.stdout.split('\n')[0]
                
                # Try to find gcc path
                which_result = subprocess.run(['where', 'gcc'], 
                                            capture_output=True, text=True, timeout=10)
                if which_result.returncode == 0:
                    info['gcc_path'] = which_result.stdout.strip().split('\n')[0]
                    info['path'] = os.path.dirname(info['gcc_path'])
                    
        except Exception:
            pass
        
        return info

    def _detect_clang(self) -> Dict[str, Any]:
        """Detect Clang compiler"""
        info = {
            'available': False,
            'version': None,
            'path': None,
            'clang_path': None,
            'priority': 3
        }
        
        try:
            # Try to run clang --version
            result = subprocess.run(['clang', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info['available'] = True
                info['version'] = result.stdout.split('\n')[0]
                
                # Try to find clang path
                which_result = subprocess.run(['where', 'clang'], 
                                            capture_output=True, text=True, timeout=10)
                if which_result.returncode == 0:
                    info['clang_path'] = which_result.stdout.strip().split('\n')[0]
                    info['path'] = os.path.dirname(info['clang_path'])
                    
        except Exception:
            pass
        
        return info

    def _detect_gcc(self) -> Dict[str, Any]:
        """Detect GCC compiler (for WSL/Linux)"""
        info = {
            'available': False,
            'version': None,
            'path': None,
            'gcc_path': None,
            'priority': 4
        }
        
        try:
            # Try to run gcc --version (but not MinGW)
            result = subprocess.run(['gcc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and 'mingw' not in result.stdout.lower():
                info['available'] = True
                info['version'] = result.stdout.split('\n')[0]
                
                # Try to find gcc path
                which_result = subprocess.run(['which', 'gcc'], 
                                            capture_output=True, text=True, timeout=10)
                if which_result.returncode == 0:
                    info['gcc_path'] = which_result.stdout.strip()
                    info['path'] = os.path.dirname(info['gcc_path'])
                    
        except Exception:
            pass
        
        return info

    def _generate_cmake_files(self, global_reconstruction: Dict[str, Any], 
                            output_dir: str) -> bool:
        """Generate CMakeLists.txt file"""
        try:
            cmake_content = self._create_cmake_content(global_reconstruction)
            cmake_path = os.path.join(output_dir, 'CMakeLists.txt')
            
            with open(cmake_path, 'w') as f:
                f.write(cmake_content)
            
            return True
        except Exception:
            return False

    def _create_cmake_content(self, global_reconstruction: Dict[str, Any]) -> str:
        """Create CMakeLists.txt content"""
        project_name = "launcher-new"
        
        # Find source files
        source_files = []
        reconstructed_source = global_reconstruction.get('reconstructed_source', {})
        src_files = reconstructed_source.get('source_files', {})
        
        for filename in src_files.keys():
            if filename.endswith('.c'):
                source_files.append(f"src/{filename}")
        
        if not source_files:
            source_files = ["src/main.c"]
        
        cmake_content = f"""cmake_minimum_required(VERSION 3.10)
project({project_name})

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
{chr(10).join(f'    {src}' for src in source_files)}
)

# Create executable
add_executable({project_name} ${{SOURCES}})

# Compiler-specific settings
if(MSVC)
    target_compile_options({project_name} PRIVATE /W3)
    target_compile_definitions({project_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options({project_name} PRIVATE -Wall -Wextra)
endif()

# Windows-specific settings
if(WIN32)
    set_target_properties({project_name} PROPERTIES
        WIN32_EXECUTABLE TRUE
    )
endif()

# Installation
install(TARGETS {project_name} DESTINATION bin)
"""
        return cmake_content

    def _generate_makefile(self, global_reconstruction: Dict[str, Any], 
                         output_dir: str) -> bool:
        """Generate Makefile"""
        try:
            makefile_content = self._create_makefile_content(global_reconstruction)
            makefile_path = os.path.join(output_dir, 'Makefile')
            
            with open(makefile_path, 'w') as f:
                f.write(makefile_content)
            
            return True
        except Exception:
            return False

    def _create_makefile_content(self, global_reconstruction: Dict[str, Any]) -> str:
        """Create Makefile content"""
        project_name = "launcher-new"
        
        # Find source files
        source_files = []
        reconstructed_source = global_reconstruction.get('reconstructed_source', {})
        src_files = reconstructed_source.get('source_files', {})
        
        for filename in src_files.keys():
            if filename.endswith('.c'):
                source_files.append(f"src/{filename}")
        
        if not source_files:
            source_files = ["src/main.c"]
        
        obj_files = [src.replace('.c', '.o').replace('src/', 'obj/') for src in source_files]
        
        makefile_content = f"""# Makefile for {project_name}

CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -Iinclude
TARGET = {project_name}.exe
SRCDIR = src
OBJDIR = obj
SOURCES = {' '.join(source_files)}
OBJECTS = {' '.join(obj_files)}

# Default target
all: $(TARGET)

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Compile source files to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)

# Rebuild everything
rebuild: clean all

.PHONY: all clean rebuild
"""
        return makefile_content

    def _generate_ninja_build(self, global_reconstruction: Dict[str, Any], 
                            output_dir: str) -> bool:
        """Generate build.ninja file"""
        try:
            ninja_content = self._create_ninja_content(global_reconstruction)
            ninja_path = os.path.join(output_dir, 'build.ninja')
            
            with open(ninja_path, 'w') as f:
                f.write(ninja_content)
            
            return True
        except Exception:
            return False

    def _create_ninja_content(self, global_reconstruction: Dict[str, Any]) -> str:
        """Create build.ninja content"""
        project_name = "launcher-new"
        
        # Find source files
        source_files = []
        reconstructed_source = global_reconstruction.get('reconstructed_source', {})
        src_files = reconstructed_source.get('source_files', {})
        
        for filename in src_files.keys():
            if filename.endswith('.c'):
                source_files.append(f"src/{filename}")
        
        if not source_files:
            source_files = ["src/main.c"]
        
        obj_files = [src.replace('.c', '.o').replace('src/', 'obj/') for src in source_files]
        
        ninja_content = f"""# Ninja build file for {project_name}

cflags = -Wall -Wextra -std=c99 -Iinclude

rule cc
  command = gcc $cflags -c $in -o $out
  description = Compiling $in

rule link
  command = gcc $in -o $out
  description = Linking $out

# Object files
{chr(10).join(f'build {obj}: cc {src}' for obj, src in zip(obj_files, source_files))}

# Executable
build {project_name}.exe: link {' '.join(obj_files)}

default {project_name}.exe
"""
        return ninja_content

    def _generate_vs_solution(self, global_reconstruction: Dict[str, Any], 
                            output_dir: str) -> bool:
        """Generate Visual Studio solution file"""
        try:
            # Generate .sln file
            sln_content = self._create_sln_content()
            sln_path = os.path.join(output_dir, 'launcher-new.sln')
            
            with open(sln_path, 'w') as f:
                f.write(sln_content)
            
            return True
        except Exception:
            return False

    def _create_sln_content(self) -> str:
        """Create Visual Studio solution content"""
        project_guid = "{12345678-1234-1234-1234-123456789012}"
        sln_guid = "{87654321-4321-4321-4321-210987654321}"
        
        sln_content = f"""
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project("{{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}}") = "launcher-new", "launcher-new.vcxproj", "{project_guid}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|x64 = Debug|x64
		Debug|x86 = Debug|x86
		Release|x64 = Release|x64
		Release|x86 = Release|x86
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{project_guid}.Debug|x64.ActiveCfg = Debug|x64
		{project_guid}.Debug|x64.Build.0 = Debug|x64
		{project_guid}.Debug|x86.ActiveCfg = Debug|Win32
		{project_guid}.Debug|x86.Build.0 = Debug|Win32
		{project_guid}.Release|x64.ActiveCfg = Release|x64
		{project_guid}.Release|x64.Build.0 = Release|x64
		{project_guid}.Release|x86.ActiveCfg = Release|Win32
		{project_guid}.Release|x86.Build.0 = Release|Win32
	EndGlobalSection
	GlobalSection(SolutionProperties) = preSolution
		HideSolutionNode = FALSE
	EndGlobalSection
	GlobalSection(ExtensibilityGlobals) = postSolution
		SolutionGuid = {sln_guid}
	EndGlobalSection
EndGlobal
"""
        return sln_content

    def _attempt_multi_compiler_build(self, compilers: Dict[str, Dict[str, Any]], 
                                    compilation_dir: str, 
                                    global_reconstruction: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Attempt compilation with multiple compilers"""
        results = {}
        
        # Sort compilers by priority
        sorted_compilers = sorted(compilers.items(), 
                                key=lambda x: x[1]['priority'])
        
        for compiler_name, compiler_info in sorted_compilers:
            try:
                if compiler_name == 'msvc':
                    result = self._build_with_msvc(compilation_dir, compiler_info)
                elif compiler_name == 'mingw':
                    result = self._build_with_mingw(compilation_dir, compiler_info)
                elif compiler_name == 'clang':
                    result = self._build_with_clang(compilation_dir, compiler_info)
                elif compiler_name == 'gcc':
                    result = self._build_with_gcc(compilation_dir, compiler_info)
                else:
                    result = {'success': False, 'errors': ['Unknown compiler']}
                
                results[compiler_name] = result
                
                # If successful, we can continue to test others or stop here
                if result['success']:
                    print(f"Successfully compiled with {compiler_name}")
                    
            except Exception as e:
                results[compiler_name] = {
                    'success': False,
                    'errors': [f"Exception during {compiler_name} compilation: {str(e)}"]
                }
        
        return results

    def _build_with_msvc(self, compilation_dir: str, compiler_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build with MSVC (reuse existing Agent 12 logic)"""
        result = {'success': False, 'errors': [], 'binary_path': None}
        
        try:
            # Use MSBuild from compiler info
            msbuild_path = compiler_info.get('msbuild_path')
            if not msbuild_path or not os.path.exists(msbuild_path):
                result['errors'].append("MSBuild path not found")
                return result
            
            # Change to compilation directory
            original_cwd = os.getcwd()
            os.chdir(compilation_dir)
            
            # Look for existing .vcxproj file
            vcxproj_files = [f for f in os.listdir('.') if f.endswith('.vcxproj')]
            if not vcxproj_files:
                result['errors'].append("No .vcxproj file found")
                return result
            
            project_file = vcxproj_files[0]
            
            # Run MSBuild
            cmd = [msbuild_path, project_file, '/p:Configuration=Release', '/p:Platform=Win32']
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if process.returncode == 0:
                # Look for generated binary
                for binary in ['launcher-new.exe', 'Release\\launcher-new.exe']:
                    if os.path.exists(binary):
                        result['success'] = True
                        result['binary_path'] = os.path.join(compilation_dir, binary)
                        break
            else:
                result['errors'].append(f"MSBuild failed: {process.stderr}")
                
        except Exception as e:
            result['errors'].append(f"MSVC compilation error: {str(e)}")
        finally:
            os.chdir(original_cwd)
        
        return result

    def _build_with_mingw(self, compilation_dir: str, compiler_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build with MinGW"""
        result = {'success': False, 'errors': [], 'binary_path': None}
        
        try:
            original_cwd = os.getcwd()
            os.chdir(compilation_dir)
            
            # Use Makefile if available, otherwise direct compilation
            if os.path.exists('Makefile'):
                cmd = ['make']
            else:
                # Direct compilation
                src_files = []
                if os.path.exists('src'):
                    for f in os.listdir('src'):
                        if f.endswith('.c'):
                            src_files.append(f'src/{f}')
                
                if not src_files:
                    result['errors'].append("No source files found")
                    return result
                
                cmd = ['gcc', '-Wall', '-Wextra', '-std=c99', '-Iinclude'] + src_files + ['-o', 'launcher-new.exe']
            
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if process.returncode == 0 and os.path.exists('launcher-new.exe'):
                result['success'] = True
                result['binary_path'] = os.path.join(compilation_dir, 'launcher-new.exe')
            else:
                result['errors'].append(f"MinGW compilation failed: {process.stderr}")
                
        except Exception as e:
            result['errors'].append(f"MinGW compilation error: {str(e)}")
        finally:
            os.chdir(original_cwd)
        
        return result

    def _build_with_clang(self, compilation_dir: str, compiler_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build with Clang"""
        result = {'success': False, 'errors': [], 'binary_path': None}
        
        try:
            original_cwd = os.getcwd()
            os.chdir(compilation_dir)
            
            # Direct compilation with clang
            src_files = []
            if os.path.exists('src'):
                for f in os.listdir('src'):
                    if f.endswith('.c'):
                        src_files.append(f'src/{f}')
            
            if not src_files:
                result['errors'].append("No source files found")
                return result
            
            cmd = ['clang', '-Wall', '-Wextra', '-std=c99', '-Iinclude'] + src_files + ['-o', 'launcher-new.exe']
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if process.returncode == 0 and os.path.exists('launcher-new.exe'):
                result['success'] = True
                result['binary_path'] = os.path.join(compilation_dir, 'launcher-new.exe')
            else:
                result['errors'].append(f"Clang compilation failed: {process.stderr}")
                
        except Exception as e:
            result['errors'].append(f"Clang compilation error: {str(e)}")
        finally:
            os.chdir(original_cwd)
        
        return result

    def _build_with_gcc(self, compilation_dir: str, compiler_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build with GCC (non-MinGW)"""
        result = {'success': False, 'errors': [], 'binary_path': None}
        
        try:
            original_cwd = os.getcwd()
            os.chdir(compilation_dir)
            
            # Use Makefile if available, otherwise direct compilation
            if os.path.exists('Makefile'):
                cmd = ['make']
            else:
                # Direct compilation
                src_files = []
                if os.path.exists('src'):
                    for f in os.listdir('src'):
                        if f.endswith('.c'):
                            src_files.append(f'src/{f}')
                
                if not src_files:
                    result['errors'].append("No source files found")
                    return result
                
                cmd = ['gcc', '-Wall', '-Wextra', '-std=c99', '-Iinclude'] + src_files + ['-o', 'launcher-new']
            
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            binary_name = 'launcher-new' if not cmd[0] == 'make' else 'launcher-new.exe'
            if process.returncode == 0 and os.path.exists(binary_name):
                result['success'] = True
                result['binary_path'] = os.path.join(compilation_dir, binary_name)
            else:
                result['errors'].append(f"GCC compilation failed: {process.stderr}")
                
        except Exception as e:
            result['errors'].append(f"GCC compilation error: {str(e)}")
        finally:
            os.chdir(original_cwd)
        
        return result

    def _determine_best_compiler(self, compilation_attempts: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Determine the best compiler based on success and priority"""
        successful_compilers = []
        
        for compiler, attempt in compilation_attempts.items():
            if attempt['success']:
                successful_compilers.append(compiler)
        
        if not successful_compilers:
            return None
        
        # Priority order: MSVC > MinGW > Clang > GCC
        priority_order = ['msvc', 'mingw', 'clang', 'gcc']
        
        for compiler in priority_order:
            if compiler in successful_compilers:
                return compiler
        
        return successful_compilers[0]  # Fallback to first successful