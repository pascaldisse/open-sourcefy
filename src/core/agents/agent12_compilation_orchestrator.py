"""
Agent 12: Visual Studio MSBuild Compilation Orchestrator
Orchestrates compilation using Visual Studio MSBuild from environment variables.

Environment Variables:
- MSBUILD_PATH: Path to MSBuild.exe
- DEVENV_PATH: Path to devenv.exe  
- VSINSTALLDIR: Visual Studio installation directory

Defaults to Visual Studio 2022 Preview paths if environment variables not set.
"""

import os
import subprocess
import tempfile
from typing import Dict, Any, List
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent12_CompilationOrchestrator(BaseAgent):
    """Agent 12: Visual Studio MSBuild compilation orchestration"""
    
    def __init__(self):
        super().__init__(
            agent_id=12,
            name="CompilationOrchestrator",
            dependencies=[11]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute compilation orchestration"""
        agent11_result = context['agent_results'].get(11)
        if not agent11_result or agent11_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 11 (GlobalReconstructor) did not complete successfully"
            )

        try:
            global_reconstruction = agent11_result.data
            
            # NEW: Perform early source code validation before compilation
            early_validation = self._perform_early_source_validation(global_reconstruction)
            if not early_validation['validation_passed']:
                return AgentResult(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    data={'early_validation_failed': True, 'validation_details': early_validation},
                    error_message=f"COMPILATION ABORTED: Source validation failed - {early_validation.get('primary_issue', 'Source code quality insufficient for compilation')}"
                )
            
            compilation_result = self._orchestrate_compilation(global_reconstruction, context)
            
            # NEW: Add validation results to compilation output
            compilation_result['early_validation'] = early_validation
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=compilation_result,
                metadata={
                    'depends_on': [11],
                    'analysis_type': 'compilation_orchestration'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Compilation orchestration failed: {str(e)}"
            )

    def _orchestrate_compilation(self, global_reconstruction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate the MSBuild-only compilation process"""
        result = {
            'compilation_attempts': [],
            'successful_compilation': False,
            'generated_binary': None,
            'final_status': 'not_attempted',
            'source_files_written': False,
            'build_files_created': False
        }
        
        # Get output directory structure - use the ACTUAL requested output directory
        requested_output_dir = context.get('output_dir', 'output')
        output_paths = context.get('output_paths', {})
        
        # Use the correct requested output directory, not hardcoded 'output'
        compilation_dir = output_paths.get('compilation', os.path.join(requested_output_dir, 'compilation'))
        
        # Debug: Ensure we're using the correct base directory
        # Configuration determined from context
        
        # Convert to absolute path to avoid any relative path issues
        compilation_dir = os.path.abspath(compilation_dir)
        
        # Ensure compilation directory exists
        os.makedirs(compilation_dir, exist_ok=True)
        
        # Using determined compilation directory
        
        # Write source files first
        try:
            self._write_source_files(global_reconstruction, compilation_dir)
            result['source_files_written'] = True
        except Exception as e:
            result['final_status'] = 'source_write_failed'
            return result
        
        # Create MSBuild project files
        try:
            self._create_msbuild_project(compilation_dir, "launcher-new")
            result['build_files_created'] = True
        except Exception as e:
            result['final_status'] = 'build_file_creation_failed'
            return result
        
        # Attempt compilation with MSBuild first, then fallback to gcc
        compilation_attempt = self._attempt_msbuild_compilation(compilation_dir, global_reconstruction, context)
        result['compilation_attempts'].append(compilation_attempt)
        
        # If MSBuild fails, try gcc fallback
        if not compilation_attempt['success']:
            gcc_attempt = self._attempt_gcc_compilation(compilation_dir, global_reconstruction, context)
            result['compilation_attempts'].append(gcc_attempt)
            compilation_attempt = gcc_attempt  # Use gcc result for final evaluation
        
        if compilation_attempt['success']:
            # NEW: Analyze compilation output to detect dummy code
            output_analysis = self._analyze_compilation_output(compilation_attempt, global_reconstruction)
            result['output_analysis'] = output_analysis
            
            if output_analysis['is_meaningful_compilation']:
                result['successful_compilation'] = True
                result['generated_binary'] = compilation_attempt['binary_path']
                result['final_status'] = 'success'
            else:
                result['successful_compilation'] = False
                result['final_status'] = 'dummy_code_detected'
                result['compilation_errors'] = [f"COMPILATION VALIDATION FAILED: {output_analysis['rejection_reason']}"]
        else:
            result['final_status'] = 'failed'
            result['compilation_errors'] = compilation_attempt['errors']
        
        return result

    def _write_source_files(self, global_reconstruction: Dict[str, Any], output_dir: str) -> None:
        """Write source files to the output directory"""
        if not isinstance(global_reconstruction, dict):
            raise ValueError("Global reconstruction data must be a dictionary")
            
        reconstructed_source = global_reconstruction.get('reconstructed_source', {})
        
        # Create subdirectories
        src_dir = os.path.join(output_dir, 'src')
        include_dir = os.path.join(output_dir, 'include')
        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(include_dir, exist_ok=True)
        
        # Write source files
        source_files = reconstructed_source.get('source_files', {})
        if isinstance(source_files, dict):
            for filename, content in source_files.items():
                if isinstance(content, str):
                    file_path = os.path.join(src_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(content)
        
        # Write header files
        header_files = reconstructed_source.get('header_files', {})
        if isinstance(header_files, dict):
            for filename, content in header_files.items():
                if isinstance(content, str):
                    file_path = os.path.join(include_dir, filename)
                    with open(file_path, 'w') as f:
                        f.write(content)

    def _create_msbuild_project(self, project_dir: str, target_name: str) -> None:
        """Create MSBuild project file (.vcxproj)"""
        
        # Find all C source files
        src_dir = os.path.join(project_dir, 'src')
        source_files = []
        if os.path.exists(src_dir):
            for file in os.listdir(src_dir):
                if file.endswith('.c'):
                    source_files.append(os.path.join('src', file))
        
        # If no source files, create a minimal main.c
        if not source_files:
            main_c_path = os.path.join(src_dir, 'main.c')
            with open(main_c_path, 'w') as f:
                f.write('#include <stdio.h>\nint main() { printf("Hello World"); return 0; }\n')
            source_files.append('src\\main.c')
        
        # Create the .vcxproj file
        vcxproj_content = f"""<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" ToolsVersion="15.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Release|Win32">
      <Configuration>Release</Configuration>
      <Platform>Win32</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  <PropertyGroup Label="Globals">
    <ProjectGuid>{{12345678-1234-1234-1234-123456789012}}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>{target_name}</RootNamespace>
    <WindowsTargetPlatformVersion>$(LatestTargetPlatformVersion)</WindowsTargetPlatformVersion>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.Default.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>false</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <WholeProgramOptimization>true</WholeProgramOptimization>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.props" />
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <LinkIncremental>false</LinkIncremental>
    <IncludePath>include;$(IncludePath)</IncludePath>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|Win32'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <Optimization>MaxSpeed</Optimization>
      <FunctionLevelLinking>true</FunctionLevelLinking>
      <IntrinsicFunctions>true</IntrinsicFunctions>
      <PreprocessorDefinitions>WIN32;NDEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
    </ClCompile>
    <Link>
      <SubSystem>Console</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>"""
        
        # Add source files
        for src_file in source_files:
            vcxproj_content += f'\n    <ClCompile Include="{src_file}" />'
        
        vcxproj_content += """
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>"""
        
        # Write the project file
        vcxproj_path = os.path.join(project_dir, f"{target_name}.vcxproj")
        with open(vcxproj_path, 'w') as f:
            f.write(vcxproj_content)

    def _attempt_msbuild_compilation(self, project_dir: str, global_reconstruction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt compilation using ONLY MSBuild from the specified path"""
        result = {
            'success': False,
            'binary_path': None,
            'stdout': '',
            'stderr': '',
            'errors': [],
            'warnings': []
        }
        
        try:
            # Change to project directory
            original_cwd = os.getcwd()
            project_dir_abs = os.path.abspath(project_dir)
            # Changing to project directory
            os.chdir(project_dir_abs)
            
            target_name = "launcher-new"
            
            # USE VISUAL STUDIO MSBUILD FROM ENVIRONMENT
            # Using Visual Studio MSBuild compilation
            
            # Get paths from environment variables
            vs_install_dir = os.environ.get('VSINSTALLDIR', 'C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\')
            msbuild_path = os.environ.get('MSBUILD_PATH', os.path.join(vs_install_dir, 'MSBuild\\Current\\Bin\\MSBuild.exe'))
            devenv_path = os.environ.get('DEVENV_PATH', os.path.join(vs_install_dir, 'Common7\\IDE\\devenv.exe'))
            
            # MSBuild environment configured
            
            # Create PowerShell script that uses MSBuild from environment paths
            ps_script = f"""
# MSBuild compilation using Visual Studio MSBuild from environment
Write-Host "Starting Visual Studio MSBuild compilation..."

# Get MSBuild path from environment or use default
$msbuildPath = $env:MSBUILD_PATH
if (-not $msbuildPath) {{
    $vsInstallDir = $env:VSINSTALLDIR
    if (-not $vsInstallDir) {{
        $vsInstallDir = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\"
    }}
    $msbuildPath = Join-Path $vsInstallDir "MSBuild\\Current\\Bin\\MSBuild.exe"
}}

$devenvPath = $env:DEVENV_PATH
if (-not $devenvPath) {{
    $vsInstallDir = $env:VSINSTALLDIR
    if (-not $vsInstallDir) {{
        $vsInstallDir = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\"
    }}
    $devenvPath = Join-Path $vsInstallDir "Common7\\IDE\\devenv.exe"
}}

if (-not (Test-Path $msbuildPath)) {{
    Write-Host "ERROR: MSBuild.exe not found at path: $msbuildPath"
    Write-Host "Set MSBUILD_PATH environment variable or ensure Visual Studio is installed"
    exit 1
}}

Write-Host "Found Visual Studio MSBuild at: $msbuildPath"

# Check available Windows SDK versions and use the latest available
$sdkPath = "C:\\Program Files (x86)\\Windows Kits\\10\\Include"
$selectedSdk = $null
if (Test-Path $sdkPath) {{
    $sdkVersions = Get-ChildItem $sdkPath | Where-Object {{ $_.PSIsContainer }} | Select-Object -ExpandProperty Name
    Write-Host "Available Windows SDK versions: $($sdkVersions -join ', ')"
    if ($sdkVersions.Count -gt 0) {{
        $selectedSdk = $sdkVersions | Sort-Object {{ [Version]($_ -replace '[^\\d.]', '') }} -Descending | Select-Object -First 1
        Write-Host "Using Windows SDK version: $selectedSdk"
    }}
}}

# If no SDK found, try common locations
if (-not $selectedSdk) {{
    $commonSdks = @("10.0.22621.0", "10.0.19041.0", "10.0.18362.0", "10.0.17763.0")
    foreach ($sdk in $commonSdks) {{
        $testPath = "C:\\Program Files (x86)\\Windows Kits\\10\\Include\\$sdk"
        if (Test-Path $testPath) {{
            $selectedSdk = $sdk
            Write-Host "Found Windows SDK: $selectedSdk"
            break
        }}
    }}
}}

# Check if project file exists
if (-not (Test-Path "{target_name}.vcxproj")) {{
    Write-Host "ERROR: Project file {target_name}.vcxproj not found"
    exit 1
}}

try {{
    Write-Host "Building project with Visual Studio MSBuild..."
    
    # Try compilation with fallback SDK versions
    $compilationSucceeded = $false
    $attemptedVersions = @()
    
    # First attempt: Auto-detection (let MSBuild choose)
    Write-Host "Attempt 1: Auto-detection - Let MSBuild choose SDK version"
    Write-Host "Command: & `"$msbuildPath`" `"{target_name}.vcxproj`" /p:Configuration=Release /p:Platform=Win32"
    & "$msbuildPath" "{target_name}.vcxproj" /p:Configuration=Release /p:Platform=Win32 /p:UseEnv=true
    $attemptedVersions += "auto-detection"
    
    # Check if compilation succeeded
    if ($LASTEXITCODE -eq 0) {{
        $compilationSucceeded = $true
        Write-Host "SUCCESS: Compilation succeeded with auto-detection"
    }} else {{
        Write-Host "Auto-detection failed with exit code: $LASTEXITCODE"
        
        # Second attempt: Try with detected SDK if we found one
        if ($selectedSdk) {{
            Write-Host "Attempt 2: Using detected SDK version $selectedSdk"
            Write-Host "Command: & `"$msbuildPath`" `"{target_name}.vcxproj`" /p:Configuration=Release /p:Platform=Win32 /p:WindowsTargetPlatformVersion=$selectedSdk"
            & "$msbuildPath" "{target_name}.vcxproj" /p:Configuration=Release /p:Platform=Win32 /p:UseEnv=true /p:WindowsTargetPlatformVersion=$selectedSdk
            $attemptedVersions += $selectedSdk
            
            if ($LASTEXITCODE -eq 0) {{
                $compilationSucceeded = $true
                Write-Host "SUCCESS: Compilation succeeded with SDK $selectedSdk"
            }} else {{
                Write-Host "SDK $selectedSdk failed with exit code: $LASTEXITCODE"
            }}
        }}
        
        # Third attempt: Try common SDK versions
        if (-not $compilationSucceeded) {{
            $fallbackSdks = @("10.0.22621.0", "10.0.19041.0", "10.0.18362.0", "10.0.17763.0")
            foreach ($fallbackSdk in $fallbackSdks) {{
                Write-Host "Attempt: Trying fallback SDK version $fallbackSdk"
                Write-Host "Command: & `"$msbuildPath`" `"{target_name}.vcxproj`" /p:Configuration=Release /p:Platform=Win32 /p:WindowsTargetPlatformVersion=$fallbackSdk"
                & "$msbuildPath" "{target_name}.vcxproj" /p:Configuration=Release /p:Platform=Win32 /p:UseEnv=true /p:WindowsTargetPlatformVersion=$fallbackSdk
                $attemptedVersions += $fallbackSdk
                
                if ($LASTEXITCODE -eq 0) {{
                    $compilationSucceeded = $true
                    Write-Host "SUCCESS: Compilation succeeded with fallback SDK $fallbackSdk"
                    break
                }} else {{
                    Write-Host "Fallback SDK $fallbackSdk failed with exit code: $LASTEXITCODE"
                }}
            }}
        }}
    }}
    
    if (-not $compilationSucceeded) {{
        Write-Host "ERROR: All compilation attempts failed"
        Write-Host "Attempted versions: $($attemptedVersions -join ', ')"
        exit 1
    }}
    
    # Check multiple possible output locations
    $possiblePaths = @(
        "Release\\{target_name}.exe",
        "Win32\\Release\\{target_name}.exe", 
        "{target_name}.exe",
        "Release\\{target_name}.exe"
    )
    
    $foundBinary = $null
    foreach ($path in $possiblePaths) {{
        if (Test-Path $path) {{
            $foundBinary = $path
            break
        }}
    }}
    
    if ($foundBinary) {{
        # Copy to root directory
        Copy-Item -Path $foundBinary -Destination ".\\{target_name}.exe" -Force
        Write-Host "Successfully compiled with Visual Studio MSBuild"
        $binaryFile = Get-Item "{target_name}.exe"
        Write-Host "Binary created: $($binaryFile.FullName)"
        Write-Host "Binary size: $($binaryFile.Length) bytes"
    }} else {{
        Write-Host "ERROR: MSBuild completed but binary not found in any expected location"
        Write-Host "Checked paths: $($possiblePaths -join ', ')"
        exit 1
    }}
}} catch {{
    Write-Host "ERROR: Visual Studio MSBuild compilation failed: $_"
    exit 1
}}
"""
            
            # Write PowerShell script
            ps_file = os.path.join(project_dir_abs, 'build.ps1')
            # Writing PowerShell build script
            with open(ps_file, 'w') as f:
                f.write(ps_script)
            
            # Verify script was written
            if os.path.exists(ps_file):
                # PowerShell script created successfully
                pass
            else:
                result['errors'].append(f"Failed to create PowerShell script at {ps_file}")
                return result
            
            # Convert Linux/WSL paths to Windows paths for PowerShell
            def wsl_to_windows_path(linux_path):
                """Convert WSL/Linux path to Windows path"""
                if linux_path.startswith('/mnt/'):
                    # Extract drive letter and path
                    path_parts = linux_path[5:].split('/', 1)
                    if len(path_parts) >= 1:
                        drive = path_parts[0].upper()
                        rest_path = path_parts[1] if len(path_parts) > 1 else ""
                        windows_path = f"{drive}:\\{rest_path.replace('/', '\\')}"
                        return windows_path
                return linux_path
            
            # Convert paths to Windows format
            ps_file_abs = os.path.abspath(ps_file)
            ps_file_windows = wsl_to_windows_path(ps_file_abs)
            project_dir_windows = wsl_to_windows_path(project_dir_abs)
            
            # Executing PowerShell build script
            
            # Verify Windows directory exists before running PowerShell
            if not os.path.exists(project_dir_abs):
                result['errors'].append(f"Project directory does not exist: {project_dir_abs}")
                return result
            
            # Use Linux path for cwd (subprocess), Windows path for PowerShell file argument
            process = subprocess.run([
                'powershell.exe', '-ExecutionPolicy', 'Bypass', 
                '-File', ps_file_windows
            ], capture_output=True, text=True, timeout=300, cwd=project_dir_abs)
            
            result['stdout'] = process.stdout
            result['stderr'] = process.stderr
            
            # Check if compilation actually succeeded by looking for binary AND checking return code
            binary_found = False
            binary_path = None
            for binary in [f"{target_name}.exe", "launcher-new.exe", "launcher.exe", "main.exe"]:
                if os.path.exists(binary):
                    binary_path = os.path.join(project_dir, binary)
                    binary_found = True
                    break
            
            # Success only if both returncode is 0 AND binary was created
            if process.returncode == 0 and binary_found:
                result['success'] = True
                result['binary_path'] = binary_path
            else:
                result['success'] = False
                result['errors'] = self._parse_compilation_errors(process.stderr + process.stdout)
                result['warnings'] = self._parse_compilation_warnings(process.stderr + process.stdout)
                if process.returncode != 0:
                    result['errors'].append(f"MSBuild exited with code {process.returncode}")
                if not binary_found:
                    result['errors'].append("No executable binary was generated")
            
        except subprocess.TimeoutExpired:
            result['errors'].append("Compilation timed out")
        except Exception as e:
            result['errors'].append(f"Compilation failed: {str(e)}")
        finally:
            os.chdir(original_cwd)
        
        return result

    def _parse_compilation_errors(self, stderr: str) -> List[str]:
        """Parse compilation errors from stderr"""
        errors = []
        for line in stderr.split('\n'):
            if 'error' in line.lower():
                errors.append(line.strip())
        return errors

    def _parse_compilation_warnings(self, stderr: str) -> List[str]:
        """Parse compilation warnings from stderr"""
        warnings = []
        for line in stderr.split('\n'):
            if 'warning' in line.lower():
                warnings.append(line.strip())
        return warnings

    def _is_pe_binary(self, binary_path: str) -> bool:
        """Check if the binary is a Windows PE executable"""
        if not binary_path or not os.path.exists(binary_path):
            return False
            
        try:
            with open(binary_path, 'rb') as f:
                # Read first few bytes to check for PE signature
                header = f.read(64)
                if len(header) < 64:
                    return False
                    
                # Check for MZ signature (DOS header)
                if header[:2] != b'MZ':
                    return False
                    
                # Get PE header offset
                pe_offset = int.from_bytes(header[60:64], 'little')
                
                # Read PE signature
                f.seek(pe_offset)
                pe_sig = f.read(4)
                
                # Check for PE signature
                return pe_sig == b'PE\x00\x00'
                
        except Exception as e:
            return binary_path.lower().endswith('.exe')  # Fallback to extension check
    
    def _perform_early_source_validation(self, global_reconstruction: Dict[str, Any]) -> Dict[str, Any]:
        """Perform early validation of source code before attempting compilation"""
        validation = {
            'validation_passed': False,
            'primary_issue': '',
            'source_files_count': 0,
            'total_code_lines': 0,
            'meaningful_functions': 0,
            'has_main_function': False,
            'syntax_issues': [],
            'size_issues': [],
            'content_quality_score': 0.0
        }
        
        try:
            # Extract source files
            reconstructed_source = global_reconstruction.get('reconstructed_source', {})
            source_files = reconstructed_source.get('source_files', {})
            
            if not isinstance(source_files, dict) or len(source_files) == 0:
                validation['primary_issue'] = 'No source files generated'
                return validation
            
            validation['source_files_count'] = len(source_files)
            
            total_code_lines = 0
            meaningful_functions = 0
            has_main = False
            all_syntax_issues = []
            all_size_issues = []
            
            for filename, content in source_files.items():
                if not isinstance(content, str):
                    continue
                    
                # Basic size validation
                if len(content.strip()) < 20:  # Very small files
                    all_size_issues.append(f"{filename}: File too small ({len(content)} chars)")
                    continue
                
                # Count code lines
                code_lines = len([line for line in content.split('\n') 
                                if line.strip() and not line.strip().startswith('//')])
                total_code_lines += code_lines
                
                # Check for main function
                if 'int main' in content or 'void main' in content:
                    has_main = True
                
                # Count meaningful functions
                import re
                functions = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', content)
                meaningful_functions += len([f for f in functions if f not in ['if', 'while', 'for']])
                
                # Basic syntax validation
                open_braces = content.count('{')
                close_braces = content.count('}')
                if open_braces != close_braces:
                    all_syntax_issues.append(f"{filename}: Unbalanced braces ({open_braces} open, {close_braces} close)")
                
                # Check for obvious placeholder content
                if 'Hello World' in content and len(content) < 100:
                    all_size_issues.append(f"{filename}: Appears to be placeholder Hello World program")
                
                if 'TODO' in content and content.count('TODO') > 2:
                    all_size_issues.append(f"{filename}: Contains multiple TODO placeholders")
            
            validation['total_code_lines'] = total_code_lines
            validation['meaningful_functions'] = meaningful_functions
            validation['has_main_function'] = has_main
            validation['syntax_issues'] = all_syntax_issues
            validation['size_issues'] = all_size_issues
            
            # Calculate content quality score
            quality_factors = []
            
            if total_code_lines >= 10:  # At least 10 lines of code
                quality_factors.append(0.3)
            if meaningful_functions >= 1:  # At least 1 meaningful function
                quality_factors.append(0.25)
            if has_main:  # Has main function
                quality_factors.append(0.2)
            if len(all_syntax_issues) == 0:  # No syntax issues
                quality_factors.append(0.15)
            if len(all_size_issues) <= 1:  # Minimal size issues
                quality_factors.append(0.1)
            
            validation['content_quality_score'] = sum(quality_factors)
            
            # Determine if validation passes
            critical_issues = []
            
            if total_code_lines < 5:
                critical_issues.append(f"Insufficient code content ({total_code_lines} lines)")
            
            if meaningful_functions == 0:
                critical_issues.append("No meaningful functions detected")
            
            if len(all_syntax_issues) > 0:
                critical_issues.append(f"Syntax errors detected: {len(all_syntax_issues)}")
            
            if validation['content_quality_score'] < 0.4:  # Require at least 40% quality
                critical_issues.append(f"Content quality too low ({validation['content_quality_score']:.1%})")
            
            # STRICT validation criteria for compilation
            if len(critical_issues) == 0:
                validation['validation_passed'] = True
            else:
                validation['validation_passed'] = False
                validation['primary_issue'] = '; '.join(critical_issues)
                
        except Exception as e:
            validation['validation_passed'] = False
            validation['primary_issue'] = f"Validation failed: {str(e)}"
        
        return validation
    
    def _analyze_compilation_output(self, compilation_attempt: Dict[str, Any], global_reconstruction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze compilation output to detect if we compiled meaningful code or just dummy code"""
        analysis = {
            'is_meaningful_compilation': False,
            'rejection_reason': '',
            'binary_size': 0,
            'compilation_warnings': 0,
            'source_complexity_indicators': {},
            'binary_analysis': {},
            'quality_score': 0.0
        }
        
        try:
            binary_path = compilation_attempt.get('binary_path')
            if not binary_path or not os.path.exists(binary_path):
                analysis['rejection_reason'] = 'Binary file not found or inaccessible'
                return analysis
            
            # Analyze binary size
            binary_size = os.path.getsize(binary_path)
            analysis['binary_size'] = binary_size
            
            # Basic size validation - too small likely means dummy code
            if binary_size < 10000:  # Less than 10KB is suspicious for reconstructed code
                analysis['rejection_reason'] = f'Binary too small ({binary_size} bytes) - likely dummy/placeholder code'
                return analysis
            
            # Check if binary is too large (may indicate compilation errors)
            if binary_size > 50000000:  # More than 50MB is suspicious
                analysis['rejection_reason'] = f'Binary unreasonably large ({binary_size} bytes) - possible compilation issues'
                return analysis
            
            # Analyze compilation warnings/errors
            stdout = compilation_attempt.get('stdout', '')
            stderr = compilation_attempt.get('stderr', '')
            
            warning_indicators = ['warning', 'Warning', 'WARNING']
            error_indicators = ['error', 'Error', 'ERROR']
            
            warnings_count = sum(stdout.count(w) + stderr.count(w) for w in warning_indicators)
            errors_count = sum(stdout.count(e) + stderr.count(e) for e in error_indicators)
            
            analysis['compilation_warnings'] = warnings_count
            
            # Too many warnings suggest poor code quality
            if warnings_count > 20:
                analysis['rejection_reason'] = f'Excessive compilation warnings ({warnings_count}) - poor code quality'
                return analysis
            
            # Any compilation errors that didn't fail the build suggest linker/runtime issues
            if errors_count > 0:
                analysis['rejection_reason'] = f'Compilation errors present ({errors_count}) despite successful build'
                return analysis
            
            # Analyze source code complexity
            source_complexity = self._analyze_source_complexity(global_reconstruction)
            analysis['source_complexity_indicators'] = source_complexity
            
            # Check if source indicates dummy code
            if source_complexity['appears_dummy']:
                analysis['rejection_reason'] = f"Source code analysis indicates dummy/placeholder code: {source_complexity['dummy_indicators']}"
                return analysis
            
            # Perform basic binary content analysis
            binary_content_analysis = self._analyze_binary_content(binary_path)
            analysis['binary_analysis'] = binary_content_analysis
            
            # Check for placeholder strings in binary
            if binary_content_analysis.get('has_placeholder_strings', False):
                analysis['rejection_reason'] = 'Binary contains placeholder strings (Hello World, TODO, etc.)'
                return analysis
            
            # Calculate quality score
            quality_factors = []
            
            # Size factor (reasonable size gets points)
            if 15000 <= binary_size <= 10000000:  # 15KB to 10MB is reasonable
                quality_factors.append(0.3)
            
            # Warning factor (few warnings is good)
            if warnings_count <= 5:
                quality_factors.append(0.2)
            elif warnings_count <= 10:
                quality_factors.append(0.1)
            
            # Source complexity factor
            if source_complexity['complexity_score'] >= 0.5:
                quality_factors.append(0.3)
            elif source_complexity['complexity_score'] >= 0.3:
                quality_factors.append(0.15)
            
            # Binary content factor
            if not binary_content_analysis.get('has_placeholder_strings', True):
                quality_factors.append(0.2)
            
            analysis['quality_score'] = sum(quality_factors)
            
            # Final determination
            if analysis['quality_score'] >= 0.6:  # Require 60% quality
                analysis['is_meaningful_compilation'] = True
            else:
                analysis['rejection_reason'] = f'Overall quality score too low ({analysis["quality_score"]:.1%}) - likely dummy code'
                
        except Exception as e:
            analysis['rejection_reason'] = f'Output analysis failed: {str(e)}'
        
        return analysis
    
    def _analyze_source_complexity(self, global_reconstruction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze source code complexity to detect dummy code"""
        complexity = {
            'appears_dummy': False,
            'dummy_indicators': [],
            'complexity_score': 0.0,
            'total_functions': 0,
            'meaningful_functions': 0,
            'total_lines': 0,
            'code_lines': 0
        }
        
        try:
            reconstructed_source = global_reconstruction.get('reconstructed_source', {})
            source_files = reconstructed_source.get('source_files', {})
            
            total_functions = 0
            meaningful_functions = 0
            total_lines = 0
            code_lines = 0
            dummy_indicators = []
            
            for filename, content in source_files.items():
                if not isinstance(content, str):
                    continue
                
                lines = content.split('\n')
                total_lines += len(lines)
                
                # Count actual code lines
                file_code_lines = len([line for line in lines 
                                     if line.strip() and not line.strip().startswith('//')])
                code_lines += file_code_lines
                
                # Check for dummy indicators
                if 'Hello World' in content:
                    dummy_indicators.append(f'{filename}: Contains Hello World')
                
                if 'printf("Hello' in content and file_code_lines < 10:
                    dummy_indicators.append(f'{filename}: Simple Hello World program')
                
                if content.count('TODO') > 2:
                    dummy_indicators.append(f'{filename}: Multiple TODO placeholders')
                
                if 'return 0;' in content and file_code_lines < 5:
                    dummy_indicators.append(f'{filename}: Minimal return 0 program')
                
                # Count functions
                import re
                functions = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', content)
                total_functions += len(functions)
                
                # Count meaningful functions (not just main with return 0)
                for func in functions:
                    if func == 'main':
                        # Check if main is more than just return 0
                        main_match = re.search(r'int\s+main\s*\([^)]*\)\s*\{([^}]*)\}', content, re.DOTALL)
                        if main_match:
                            main_body = main_match.group(1).strip()
                            if len(main_body.split('\n')) > 2 or 'printf' not in main_body:
                                meaningful_functions += 1
                    else:
                        meaningful_functions += 1
            
            complexity['total_functions'] = total_functions
            complexity['meaningful_functions'] = meaningful_functions  
            complexity['total_lines'] = total_lines
            complexity['code_lines'] = code_lines
            complexity['dummy_indicators'] = dummy_indicators
            
            # Calculate complexity score
            score_factors = []
            
            if meaningful_functions >= 2:  # At least 2 meaningful functions
                score_factors.append(0.4)
            elif meaningful_functions >= 1:
                score_factors.append(0.2)
            
            if code_lines >= 20:  # At least 20 lines of code
                score_factors.append(0.3)
            elif code_lines >= 10:
                score_factors.append(0.15)
            
            if len(dummy_indicators) == 0:  # No dummy indicators
                score_factors.append(0.3)
            elif len(dummy_indicators) <= 1:
                score_factors.append(0.15)
            
            complexity['complexity_score'] = sum(score_factors)
            
            # Determine if appears dummy
            if len(dummy_indicators) >= 2 or complexity['complexity_score'] < 0.3:
                complexity['appears_dummy'] = True
                
        except Exception as e:
            complexity['appears_dummy'] = True
            complexity['dummy_indicators'] = [f'Complexity analysis failed: {str(e)}']
        
        return complexity
    
    def _analyze_binary_content(self, binary_path: str) -> Dict[str, Any]:
        """Analyze binary content for placeholder strings and other indicators"""
        analysis = {
            'has_placeholder_strings': False,
            'placeholder_strings_found': [],
            'binary_entropy': 0.0,
            'suspicious_patterns': []
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read a sample of the binary (first 64KB should be enough for string analysis)
                sample_size = min(65536, os.path.getsize(binary_path))
                binary_data = f.read(sample_size)
            
            # Convert to string for text analysis (ignore encoding errors)
            try:
                text_content = binary_data.decode('utf-8', errors='ignore')
            except:
                text_content = str(binary_data)
            
            # Check for common placeholder strings
            placeholder_patterns = [
                'Hello World',
                'Hello, World',
                'TODO',
                'FIXME',
                'Placeholder',
                'Test program',
                'Generated program',
                'Reconstructed program'
            ]
            
            found_placeholders = []
            for pattern in placeholder_patterns:
                if pattern.lower() in text_content.lower():
                    found_placeholders.append(pattern)
            
            analysis['placeholder_strings_found'] = found_placeholders
            analysis['has_placeholder_strings'] = len(found_placeholders) > 0
            
            # Calculate basic entropy (complexity measure)
            if len(binary_data) > 0:
                import math
                byte_counts = [0] * 256
                for byte in binary_data:
                    byte_counts[byte] += 1
                
                entropy = 0.0
                for count in byte_counts:
                    if count > 0:
                        probability = count / len(binary_data)
                        entropy -= probability * math.log2(probability)
                
                analysis['binary_entropy'] = entropy
                
                # Low entropy suggests simple/dummy programs
                if entropy < 4.0:  # Typical threshold for meaningful binary content
                    analysis['suspicious_patterns'].append(f'Low binary entropy ({entropy:.2f}) suggests simple program')
            
        except Exception as e:
            analysis['suspicious_patterns'].append(f'Binary analysis failed: {str(e)}')
        
        return analysis
    
    def _attempt_gcc_compilation(self, project_dir: str, global_reconstruction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt compilation using GCC as fallback when MSBuild fails"""
        result = {
            'success': False,
            'binary_path': None,
            'stdout': '',
            'stderr': '',
            'errors': [],
            'warnings': []
        }
        
        try:
            # Change to project directory
            original_cwd = os.getcwd()
            project_dir_abs = os.path.abspath(project_dir)
            os.chdir(project_dir_abs)
            
            target_name = "launcher-new"
            # Using GCC compilation as fallback
            
            # Find all C source files
            src_dir = os.path.join(project_dir_abs, 'src')
            source_files = []
            if os.path.exists(src_dir):
                for file in os.listdir(src_dir):
                    if file.endswith('.c'):
                        source_files.append(os.path.join('src', file))
            
            if not source_files:
                result['errors'].append("No C source files found for compilation")
                return result
            
            # Build GCC command
            gcc_cmd = [
                'gcc',
                '-Wall', '-Wextra', '-std=c99',
                '-Iinclude',  # Include directory
                '-o', f'{target_name}.exe',
            ] + source_files
            
            # Executing GCC compilation command
            
            # Run GCC compilation
            process = subprocess.run(
                gcc_cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_dir_abs
            )
            
            result['stdout'] = process.stdout
            result['stderr'] = process.stderr
            
            # Check if compilation succeeded
            binary_path = os.path.join(project_dir_abs, f'{target_name}.exe')
            if process.returncode == 0 and os.path.exists(binary_path):
                result['success'] = True
                result['binary_path'] = binary_path
                # GCC compilation successful
            else:
                result['success'] = False
                result['errors'] = self._parse_compilation_errors(process.stderr)
                result['warnings'] = self._parse_compilation_warnings(process.stderr)
                if process.returncode != 0:
                    result['errors'].append(f"GCC exited with code {process.returncode}")
                if not os.path.exists(binary_path):
                    result['errors'].append("No executable binary was generated by GCC")
        
        except subprocess.TimeoutExpired:
            result['errors'].append("GCC compilation timed out")
        except Exception as e:
            result['errors'].append(f"GCC compilation failed: {str(e)}")
        finally:
            os.chdir(original_cwd)
        
        return result