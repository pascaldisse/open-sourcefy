"""
Agent 18: Advanced Build Systems Generator
Provides multi-compiler support and advanced build system generation.
Phase 4: Build Systems & Production Readiness
"""

import os
import json
import subprocess
import tempfile
import logging
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
        self.logger = logging.getLogger(f"Agent{self.agent_id}.{self.name}")

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
            
            # Generate Visual Studio solution only (Windows-only build process)
            result['vs_solution_generated'] = self._generate_vs_solution(
                global_reconstruction, compilation_dir
            )
            result['cmake_generated'] = False  # Disabled - Windows MSBuild only
            result['makefile_generated'] = False  # Disabled - Windows MSBuild only
            result['ninja_generated'] = False  # Disabled - Windows MSBuild only
            
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
        """Detect Visual Studio MSBuild only (Windows-only build process)"""
        compilers = {}
        
        # Only check for MSVC (Visual Studio) - Windows MSBuild only
        msvc_info = self._detect_msvc()
        if msvc_info['available']:
            compilers['msvc'] = msvc_info
        
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
        """Attempt compilation with Visual Studio MSBuild only"""
        results = {}
        
        # Only attempt MSVC compilation (Windows-only build process)
        for compiler_name, compiler_info in compilers.items():
            try:
                if compiler_name == 'msvc':
                    result = self._build_with_msvc(compilation_dir, compiler_info)
                    results[compiler_name] = result
                    
                    if result['success']:
                        self.logger.info(f"Successfully compiled with {compiler_name}")
                else:
                    # Skip non-MSVC compilers
                    continue
                    
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




    def _determine_best_compiler(self, compilation_attempts: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Determine the best compiler (MSVC only for Windows-only build process)"""
        for compiler, attempt in compilation_attempts.items():
            if compiler == 'msvc' and attempt['success']:
                return 'msvc'
        
        # No successful MSVC compilation
        return None