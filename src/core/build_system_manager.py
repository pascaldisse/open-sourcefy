"""
Central Build System Manager for Open-Sourcefy Matrix Pipeline

This module provides the AUTHORITATIVE build system configuration
for the entire Matrix pipeline. All agents MUST use this module
for build operations - NO FALLBACKS OR ALTERNATIVE PATHS ALLOWED.

PERMANENT CONFIGURATION: Visual Studio 2022 Preview with MSBuild

CRITICAL: NO FALLBACKS, NO ALTERNATIVES, NO DEGRADATION
STRICT MODE ONLY - FAIL FAST WHEN TOOLS ARE MISSING
NEVER USE FALLBACK PATHS OR MOCK IMPLEMENTATIONS
"""

import os
import yaml
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BuildConfiguration:
    """Central build configuration data structure"""
    compiler_x64: str
    compiler_x86: str
    linker_x64: str
    linker_x86: str
    msbuild_path: str
    include_dirs: List[str]
    library_dirs_x64: List[str]
    library_dirs_x86: List[str]
    default_flags: List[str]
    release_flags: List[str]
    debug_flags: List[str]
    linker_flags: List[str]


class BuildSystemManager:
    """
    Central Build System Manager - AUTHORITATIVE build configuration
    
    This class provides the single source of truth for all build operations
    in the Matrix pipeline. All agents must use this class for compilation.
    
    CRITICAL: NO FALLBACKS, NO ALTERNATIVES, NO DEGRADATION
    NO FALLBACKS - Only uses configured Visual Studio 2022 Preview paths.
    NEVER USE FALLBACK PATHS, MOCK IMPLEMENTATIONS, OR WORKAROUNDS
    STRICT MODE ONLY - FAIL FAST WHEN TOOLS ARE MISSING
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.build_config_path = self.project_root / "build_config.yaml"
        
        # Load and validate configuration
        self.config = self._load_build_config()
        self.build_config = self._parse_build_config()
        
        # Validate all tools exist
        self._validate_build_tools()
        
        self.logger.info(f"✅ Build System Manager initialized with VS2022 Preview")
    
    def _load_build_config(self) -> Dict:
        """Load central build configuration - REQUIRED"""
        if not self.build_config_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: build_config.yaml not found at {self.build_config_path}. "
                "The central build configuration is required for Matrix pipeline operation."
            )
        
        try:
            with open(self.build_config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"✅ Loaded build configuration from {self.build_config_path}")
            return config
            
        except Exception as e:
            raise RuntimeError(f"Failed to load build_config.yaml: {e}")
    
    def _parse_build_config(self) -> BuildConfiguration:
        """Parse configuration into structured format"""
        build_system = self.config['build_system']
        vs_config = build_system['visual_studio']
        msbuild_config = build_system['msbuild']
        compilation_config = self.config['compilation']
        
        return BuildConfiguration(
            compiler_x64=vs_config['compiler']['x64'],
            compiler_x86=vs_config['compiler']['x86'],
            linker_x64=vs_config['linker']['x64'],
            linker_x86=vs_config['linker']['x86'],
            msbuild_path=msbuild_config['vs2022_path'],
            include_dirs=vs_config['includes'],
            library_dirs_x64=vs_config['libraries']['x64'],
            library_dirs_x86=vs_config['libraries']['x86'],
            default_flags=compilation_config['default_flags'],
            release_flags=compilation_config['release_flags'],
            debug_flags=compilation_config['debug_flags'],
            linker_flags=compilation_config['linker_flags']
        )
    
    def _validate_build_tools(self) -> None:
        """Validate build tools - strict mode only"""
        validation_errors = []
        
        # Check compilers
        if not Path(self.build_config.compiler_x64).exists():
            validation_errors.append(f"x64 Compiler not found: {self.build_config.compiler_x64}")
        
        if not Path(self.build_config.compiler_x86).exists():
            validation_errors.append(f"x86 Compiler not found: {self.build_config.compiler_x86}")
        
        # Check linkers
        if not Path(self.build_config.linker_x64).exists():
            validation_errors.append(f"x64 Linker not found: {self.build_config.linker_x64}")
        
        # Check MSBuild
        if not Path(self.build_config.msbuild_path).exists():
            validation_errors.append(f"MSBuild not found: {self.build_config.msbuild_path}")
        
        # Check include directories
        for include_dir in self.build_config.include_dirs:
            if not Path(include_dir).exists():
                validation_errors.append(f"Include directory not found: {include_dir}")
        
        # Check library directories
        for lib_dir in self.build_config.library_dirs_x64:
            if not Path(lib_dir).exists():
                validation_errors.append(f"x64 Library directory not found: {lib_dir}")
        
        if validation_errors:
            error_msg = "BUILD SYSTEM VALIDATION FAILED:\n" + "\n".join(validation_errors)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        else:
            self.logger.info("✅ All build tools validated successfully")
    
    
    def get_compiler_path(self, architecture: str = "x64") -> str:
        """Get compiler path for specified architecture"""
        if architecture.lower() == "x64":
            return self.build_config.compiler_x64
        elif architecture.lower() == "x86":
            return self.build_config.compiler_x86
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def get_linker_path(self, architecture: str = "x64") -> str:
        """Get linker path for specified architecture"""
        if architecture.lower() == "x64":
            return self.build_config.linker_x64
        elif architecture.lower() == "x86":
            return self.build_config.linker_x86
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def get_msbuild_path(self) -> str:
        """Get MSBuild path - ALWAYS VS2022 Preview"""
        return self.build_config.msbuild_path

    def _find_rc_compiler(self) -> Optional[str]:
        """Find Windows Resource Compiler (rc.exe) in VS2022 installation"""
        try:
            # rc.exe is typically in the same directory as cl.exe
            vc_tools_base = "/mnt/c/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.44.35207"
            
            # Try x64 host tools first
            rc_x64 = f"{vc_tools_base}/bin/Hostx64/x64/rc.exe"
            if Path(rc_x64).exists():
                return rc_x64
            
            # Try x86 host tools
            rc_x86 = f"{vc_tools_base}/bin/Hostx86/x86/rc.exe"
            if Path(rc_x86).exists():
                return rc_x86
            
            # Try Windows SDK location (alternative)
            sdk_paths = [
                "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/rc.exe",
                "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64/rc.exe"
            ]
            
            for sdk_rc in sdk_paths:
                if Path(sdk_rc).exists():
                    return sdk_rc
            
            self.logger.warning("Resource compiler (rc.exe) not found in VS2022 installation")
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding resource compiler: {e}")
            return None
    
    def get_include_dirs(self) -> List[str]:
        """Get all include directories"""
        return self.build_config.include_dirs.copy()
    
    def get_library_dirs(self, architecture: str = "x64") -> List[str]:
        """Get library directories for specified architecture"""
        if architecture.lower() == "x64":
            return self.build_config.library_dirs_x64.copy()
        elif architecture.lower() == "x86":
            return self.build_config.library_dirs_x86.copy()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def compile_source(self, source_file: Path, output_file: Path, 
                      architecture: str = "x64", configuration: str = "Release") -> Tuple[bool, str]:
        """
        Compile source file using VS2022 MSVC - PRIMARY COMPILATION METHOD
        
        Args:
            source_file: Path to source file (.c or .cpp)
            output_file: Path for output executable
            architecture: Target architecture ("x64" or "x86")
            configuration: Build configuration ("Release" or "Debug")
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        import shutil
        import tempfile
        
        try:
            # If source is in /tmp/, copy to Windows-accessible location
            if str(source_file).startswith('/tmp/'):
                # Create a temporary directory in Windows space
                win_temp_dir = Path('/mnt/c/temp')
                win_temp_dir.mkdir(exist_ok=True)
                
                # Copy source file to Windows temp directory
                temp_source = win_temp_dir / source_file.name
                shutil.copy2(source_file, temp_source)
                
                # Update output path to Windows temp directory
                temp_output = win_temp_dir / output_file.name
                
                self.logger.info(f"Copied source to Windows-accessible location: {temp_source}")
                
                # Use the Windows-accessible paths
                actual_source = temp_source
                actual_output = temp_output
            else:
                actual_source = source_file
                actual_output = output_file
            
            compiler_path = self.get_compiler_path(architecture)
            # Use WSL path for executable, Windows paths for arguments
            
            # Build compiler command with WSL path for executable
            cmd = [compiler_path]
            cmd.extend(self.build_config.default_flags)
            
            # Add configuration-specific flags
            if configuration.lower() == "release":
                cmd.extend(self.build_config.release_flags)
            else:
                cmd.extend(self.build_config.debug_flags)
            
            # Add include directories (convert to Windows paths)
            for include_dir in self.build_config.include_dirs:
                win_include_path = self._convert_wsl_path_to_windows(Path(include_dir))
                cmd.append(f"/I\"{win_include_path}\"")
            
            # Convert WSL paths to Windows paths for the compiler
            source_win_path = self._convert_wsl_path_to_windows(actual_source)
            output_win_path = self._convert_wsl_path_to_windows(actual_output)
            
            # Add source file and output
            cmd.append(source_win_path)
            cmd.append(f"/Fe{output_win_path}")
            
            # Add linker options (library directories)
            cmd.append("/link")
            for lib_dir in self.get_library_dirs(architecture):
                win_lib_path = self._convert_wsl_path_to_windows(Path(lib_dir))
                cmd.append(f"/LIBPATH:\"{win_lib_path}\"")
            
            self.logger.info(f"Compiling with VS2022: {actual_source} -> {actual_output}")
            self.logger.info(f"Compiler command: {' '.join(cmd)}")
            
            # Execute compilation with Windows environment variables
            self.logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Set up Windows environment for the compiler
            env = os.environ.copy()
            env.update({
                'INCLUDE': ';'.join([self._convert_wsl_path_to_windows(Path(inc)) for inc in self.build_config.include_dirs]),
                'LIB': ';'.join([self._convert_wsl_path_to_windows(Path(lib)) for lib in self.get_library_dirs(architecture)]),
                'PATH': env.get('PATH', '') + ';' + self._convert_wsl_path_to_windows(Path(compiler_path).parent)
            })
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes for large resource compilation
                cwd=actual_output.parent,
                env=env
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Compilation successful: {actual_output}")
                
                # If we used temporary files, copy the result back
                if str(source_file).startswith('/tmp/') and actual_output.exists():
                    shutil.copy2(actual_output, output_file)
                    self.logger.info(f"Copied result back to: {output_file}")
                
                return True, result.stdout
            else:
                error_output = result.stderr or result.stdout or f"Exit code: {result.returncode}"
                self.logger.error(f"❌ Compilation failed: {error_output}")
                return False, error_output
                
        except Exception as e:
            error_msg = f"Compilation error: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def build_with_msbuild(self, project_file: Path, configuration: str = "Release", 
                          platform: str = "x64") -> Tuple[bool, str]:
        """
        Build project using MSBuild - strict mode only
        
        Args:
            project_file: Path to .vcxproj file
            configuration: Build configuration
            platform: Target platform
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            msbuild_path = self.get_msbuild_path()
            
            # Convert paths to Windows format for MSBuild
            win_msbuild_path = self._convert_wsl_path_to_windows(msbuild_path)
            
            # CRITICAL FIX: Ensure project_file is absolute before conversion
            if not os.path.isabs(str(project_file)):
                abs_project_file = os.path.abspath(str(project_file))
                self.logger.info(f"🔧 Converting relative to absolute: {project_file} → {abs_project_file}")
                project_file = Path(abs_project_file)
            
            win_project_path = self._convert_wsl_path_to_windows(str(project_file))
            
            # DEBUG: Check if project file actually exists
            project_exists = os.path.exists(project_file)
            self.logger.info(f"🔍 DEBUG: Project file exists: {project_exists} at {project_file}")
            self.logger.info(f"🔍 DEBUG: WSL project path: {project_file}")
            self.logger.info(f"🔍 DEBUG: Windows project path: {win_project_path}")
            
            # Additional validation
            if not project_exists:
                error_msg = f"Project file does not exist at WSL path: {project_file}"
                self.logger.error(error_msg)
                return False, error_msg
            
            # Execute MSBuild directly from WSL using the WSL path
            cmd = [
                msbuild_path,  # Use WSL path directly
                win_project_path,  # Project file in Windows format
                f"/p:Configuration={configuration}",
                f"/p:Platform={platform}",
                "/verbosity:minimal",
                "/nologo"
            ]
            
            self.logger.info(f"Building with MSBuild: {project_file}")
            self.logger.info(f"MSBuild command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes for MSBuild with resources
                cwd=str(project_file.parent)
            )
            
            if result.returncode == 0:
                self.logger.info("✅ MSBuild successful")
                return True, result.stdout
            else:
                error_output = result.stderr or result.stdout or f"Exit code: {result.returncode}"
                self.logger.error(f"❌ MSBuild failed: {error_output}")
                return False, error_output
                
        except Exception as e:
            error_msg = f"MSBuild error: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    
    def create_vcxproj(self, project_name: str, source_files: List[Path], 
                      output_dir: Path, architecture: str = "x64") -> Path:
        """
        Create Visual Studio project file with proper VS2022 configuration
        
        Args:
            project_name: Name of the project
            source_files: List of source files to include
            output_dir: Directory for project file
            architecture: Target architecture
            
        Returns:
            Path to created .vcxproj file
        """
        import uuid
        
        project_guid = str(uuid.uuid4()).upper()
        vcxproj_path = output_dir / f"{project_name}.vcxproj"
        
        # Generate VS2022 compatible project file
        project_content = self._generate_vcxproj_content(
            project_name, project_guid, source_files, architecture
        )
        
        with open(vcxproj_path, 'w', encoding='utf-8') as f:
            f.write(project_content)
        
        self.logger.info(f"✅ Created VS2022 project: {vcxproj_path}")
        return vcxproj_path
    
    def _generate_vcxproj_content(self, project_name: str, project_guid: str,
                                 source_files: List[Path], architecture: str) -> str:
        """Generate VS2022 compatible .vcxproj content"""
        
        platform_config = "x64" if architecture.lower() == "x64" else "Win32"
        
        # Source files XML
        source_items = "\n".join([
            f'    <ClCompile Include="{src.name}" />' 
            for src in source_files
        ])
        
        content = f'''<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|{platform_config}">
      <Configuration>Debug</Configuration>
      <Platform>{platform_config}</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|{platform_config}">
      <Configuration>Release</Configuration>
      <Platform>{platform_config}</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  
  <PropertyGroup Label="Globals">
    <VCProjectVersion>17.0</VCProjectVersion>
    <ProjectGuid>{{{project_guid}}}</ProjectGuid>
    <RootNamespace>{project_name}</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
    <PlatformToolset>v143</PlatformToolset>
  </PropertyGroup>
  
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|{platform_config}'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <CharacterSet>MultiByte</CharacterSet>
  </PropertyGroup>
  
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|{platform_config}'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>MultiByte</CharacterSet>
  </PropertyGroup>
  
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|{platform_config}'">
    <OutDir>$(SolutionDir)bin\\$(Configuration)\\</OutDir>
    <IntDir>$(SolutionDir)obj\\$(Configuration)\\</IntDir>
  </PropertyGroup>
  
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|{platform_config}'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>MaxSpeed</Optimization>
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
      <GenerateDebugInformation>false</GenerateDebugInformation>
    </Link>
  </ItemDefinitionGroup>
  
  <ItemGroup>
{source_items}
  </ItemGroup>
  
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>'''
        
        return content
    
    def _convert_wsl_path_to_windows(self, path) -> str:
        """Convert WSL path to Windows path for the compiler"""
        if isinstance(path, str):
            path_str = path
        else:
            path_str = str(path.absolute())
        
        # Convert /mnt/c/ to C:\
        if path_str.startswith('/mnt/c/'):
            win_path = 'C:' + path_str[6:].replace('/', '\\')
            return win_path
        elif path_str.startswith('/mnt/'):
            # Handle other drive letters
            drive_letter = path_str[5]
            win_path = drive_letter.upper() + ':' + path_str[6:].replace('/', '\\')
            return win_path
        elif path_str.startswith('/tmp/'):
            # Convert /tmp/ paths to Windows temp directory using WSL interop
            # WSL makes /tmp accessible via the UNC path
            win_path = '\\\\wsl$\\Ubuntu' + path_str.replace('/', '\\')
            return win_path
        else:
            # Already a Windows path or relative path
            return path_str.replace('/', '\\')


# Global build system manager instance
_build_manager = None


def get_build_manager() -> BuildSystemManager:
    """Get the global build system manager instance"""
    global _build_manager
    if _build_manager is None:
        _build_manager = BuildSystemManager()
    return _build_manager


# Convenience functions for agents
def compile_source_file(source_file: Path, output_file: Path, 
                       architecture: str = "x64", configuration: str = "Release") -> Tuple[bool, str]:
    """Compile source file - PRIMARY METHOD FOR ALL AGENTS"""
    return get_build_manager().compile_source(source_file, output_file, architecture, configuration)


def build_msbuild_project(project_file: Path, configuration: str = "Release", 
                         platform: str = "x64") -> Tuple[bool, str]:
    """Build MSBuild project - ALTERNATIVE METHOD FOR AGENTS"""
    return get_build_manager().build_with_msbuild(project_file, configuration, platform)


def create_vs_project(project_name: str, source_files: List[Path], 
                     output_dir: Path, architecture: str = "x64") -> Path:
    """Create Visual Studio project - PROJECT GENERATION FOR AGENTS"""
    return get_build_manager().create_vcxproj(project_name, source_files, output_dir, architecture)


def get_compiler_path(architecture: str = "x64") -> str:
    """Get compiler path - FOR AGENT REFERENCE"""
    return get_build_manager().get_compiler_path(architecture)


def get_msbuild_path() -> str:
    """Get MSBuild path - FOR AGENT REFERENCE"""
    return get_build_manager().get_msbuild_path()


if __name__ == "__main__":
    # Test build system manager
    try:
        manager = BuildSystemManager()
        print("✅ Build System Manager validation successful")
        print(f"Compiler (x64): {manager.get_compiler_path('x64')}")
        print(f"MSBuild: {manager.get_msbuild_path()}")
    except Exception as e:
        print(f"❌ Build System validation failed: {e}")