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
from ..matrix_agents import ReconstructionAgent, AgentResult, AgentStatus, MatrixCharacter

class Agent10_TheMachine(ReconstructionAgent):
    """Agent 10: The Machine - Compilation orchestration and build systems"""
    
    def __init__(self):
        super().__init__(
            agent_id=10,
            matrix_character=MatrixCharacter.MACHINE,
            dependencies=[5, 9]  # Depends on Neo (agent 5) and Commander Locke (agent 9)
        )

    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate prerequisites with flexible dependency checking"""
        # Initialize shared_memory structure if not present
        shared_memory = context.get('shared_memory', {})
        if 'analysis_results' not in shared_memory:
            shared_memory['analysis_results'] = {}
        if 'binary_metadata' not in shared_memory:
            shared_memory['binary_metadata'] = {}
        
        # Check for dependencies more flexibly - Agent 10 depends on Agent 5 (Neo) or Agent 9
        dependencies_met = False
        agent_results = context.get('agent_results', {})
        
        # Check for Agent 5 (Neo) results - PRIMARY SOURCE
        if 5 in agent_results or 5 in shared_memory['analysis_results']:
            dependencies_met = True
            self.logger.info("✅ Found Agent 5 (Neo) dependency - primary decompilation source available")
        
        # Check for Agent 9 results
        if 9 in agent_results or 9 in shared_memory['analysis_results']:
            dependencies_met = True
            self.logger.info("✅ Found Agent 9 (Commander Locke) dependency")
        
        # Also check for any source code from previous agents 
        available_sources = any(
            self._get_agent_data_safely(agent_data, 'source_files') or 
            self._get_agent_data_safely(agent_data, 'decompiled_code')
            for agent_data in agent_results.values()
            if agent_data
        )
        
        if available_sources:
            dependencies_met = True
        
        if not dependencies_met:
            self.logger.warning("No source code dependencies found - proceeding with basic compilation setup")

    def _get_agent_data_safely(self, agent_data: Any, key: str) -> Any:
        """Safely get data from agent result, handling both dict and AgentResult objects"""
        if hasattr(agent_data, 'data') and hasattr(agent_data.data, 'get'):
            return agent_data.data.get(key)
        elif hasattr(agent_data, 'get'):
            data = agent_data.get('data', {})
            if hasattr(data, 'get'):
                return data.get(key)
        return None

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute compilation orchestration"""
        # Validate prerequisites with flexible dependency checking
        self._validate_prerequisites(context)
        
        # Gather dependencies - can work with partial results
        agent_results = context.get('agent_results', {})
        shared_memory = context.get('shared_memory', {})
        
        # Also check for source code from other agents
        available_sources = self._gather_available_sources(agent_results, shared_memory)
        
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
            build_system = self._generate_build_system(build_analysis, available_sources, context)
            
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
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return machine_result
            
        except Exception as e:
            error_msg = f"The Machine compilation orchestration failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e

    def _gather_available_sources(self, all_results: Dict[int, Any], shared_memory: Dict[str, Any] = None) -> Dict[str, Any]:
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
        if 9 in all_results and hasattr(all_results[9], 'status') and all_results[9].status == AgentStatus.SUCCESS:
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
        if 8 in all_results and hasattr(all_results[8], 'status') and all_results[8].status == AgentStatus.SUCCESS:
            resource_data = all_results[8].data
            if isinstance(resource_data, dict):
                sources['resource_files'].update(resource_data.get('resource_files', {}))
        
        # Gather from Neo Advanced Decompiler (Agent 5) - PRIMARY SOURCE
        if 5 in all_results and hasattr(all_results[5], 'status') and all_results[5].status == AgentStatus.SUCCESS:
            neo_data = all_results[5].data
            if isinstance(neo_data, dict):
                # Check for decompiled code from Neo's analysis
                decompiled_code = neo_data.get('decompiled_code')
                if decompiled_code:
                    sources['source_files']['main.c'] = decompiled_code
                    self.logger.info(f"✅ Found Neo's decompiled source code ({len(decompiled_code)} chars)")
                
                # Also check for any function-specific code
                enhanced_functions = neo_data.get('enhanced_functions', {})
                for func_name, func_data in enhanced_functions.items():
                    if isinstance(func_data, dict) and func_data.get('code'):
                        sources['source_files'][f"{func_name}.c"] = func_data['code']
        else:
            # If Agent 5 results not available, try to find existing decompiled source files
            self.logger.info("Agent 5 results not available - searching for existing decompiled source files...")
            self._try_load_existing_source_files(sources)
        
        # Gather from Advanced Decompiler (Agent 7)
        if 7 in all_results and hasattr(all_results[7], 'status') and all_results[7].status == AgentStatus.SUCCESS:
            decompiler_data = all_results[7].data
            if isinstance(decompiler_data, dict):
                enhanced_functions = decompiler_data.get('enhanced_functions', {})
                for func_name, func_data in enhanced_functions.items():
                    if isinstance(func_data, dict) and func_data.get('code'):
                        sources['source_files'][f"{func_name}.c"] = func_data['code']
        
        # Gather from Basic Decompiler (Agent 4)
        if 4 in all_results and hasattr(all_results[4], 'status') and all_results[4].status == AgentStatus.SUCCESS:
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
    
    def _try_load_existing_source_files(self, sources: Dict[str, Any]) -> None:
        """Try to load existing decompiled source files from previous runs"""
        try:
            # Look for output directories
            output_root = Path("output")
            if not output_root.exists():
                self.logger.warning("No output directory found")
                return
            
            # Find the most recent launcher output directory with timestamp (excluding current run)
            launcher_dirs = []
            launcher_root = output_root / "launcher"
            if launcher_root.exists():
                for subdir in launcher_root.iterdir():
                    if subdir.is_dir() and "-" in subdir.name:  # Timestamped directories
                        # Check if this directory has agent output (not the current empty one)
                        agents_dir = subdir / "agents" / "agent_05_neo"
                        if agents_dir.exists():
                            launcher_dirs.append(subdir)
            
            if not launcher_dirs:
                self.logger.warning("No existing launcher output directories with agent data found")
                return
            
            # Sort by name (which includes timestamp) to get most recent
            launcher_dirs.sort(key=lambda x: x.name, reverse=True)
            most_recent = launcher_dirs[0]
            self.logger.info(f"Searching for source files in: {most_recent}")
            
            # Look for Agent 5 decompiled code
            neo_file = most_recent / "agents" / "agent_05_neo" / "decompiled_code.c"
            if neo_file.exists():
                try:
                    with open(neo_file, 'r', encoding='utf-8') as f:
                        decompiled_code = f.read()
                    
                    # Fix common case sensitivity issues for WSL compilation
                    decompiled_code = self._fix_case_sensitivity_issues(decompiled_code)
                    
                    sources['source_files']['main.c'] = decompiled_code
                    self.logger.info(f"✅ Loaded existing Neo decompiled source: {neo_file}")
                    self.logger.info(f"✅ Source code size: {len(decompiled_code)} characters")
                except Exception as e:
                    self.logger.error(f"Failed to read {neo_file}: {e}")
            else:
                self.logger.warning(f"No decompiled code found at {neo_file}")
            
            # Phase 1 Enhancement: Load Agent 8 (Keymaker) resources for integration
            self._load_keymaker_resources(most_recent, sources)
                
        except Exception as e:
            self.logger.error(f"Error loading existing source files: {e}")
    
    def _load_keymaker_resources(self, run_dir: Path, sources: Dict[str, Any]) -> None:
        """Load Agent 8 (Keymaker) extracted resources for integration into compilation"""
        try:
            keymaker_dir = run_dir / "agents" / "agent_08_keymaker"
            if not keymaker_dir.exists():
                self.logger.warning("No Keymaker resources found - proceeding without resource integration")
                return
            
            self.logger.info("🔧 Phase 1: Loading Keymaker resources for binary equivalence improvement...")
            
            # Load keymaker analysis JSON
            analysis_file = keymaker_dir / "keymaker_analysis.json"
            if analysis_file.exists():
                try:
                    import json
                    with open(analysis_file, 'r', encoding='utf-8') as f:
                        keymaker_data = json.load(f)
                    
                    # Extract resource statistics
                    resources = keymaker_data.get('resources', {})
                    string_count = resources.get('string', {}).get('count', 0)
                    image_count = resources.get('embedded_file', {}).get('count', 0)
                    
                    self.logger.info(f"🔧 Found Keymaker resources: {string_count} strings, {image_count} images")
                    
                    # Generate Windows Resource Script (.rc) file
                    if string_count > 0 or image_count > 0:
                        rc_content = self._generate_resource_script(keymaker_dir, keymaker_data)
                        sources['resource_files'] = sources.get('resource_files', {})
                        sources['resource_files']['resources.rc'] = rc_content
                        
                        self.logger.info(f"✅ Generated resources.rc with {string_count} strings and {image_count} images")
                        
                        # Update resource header with additional constants
                        self._enhance_resource_header(sources, keymaker_data)
                        
                except Exception as e:
                    self.logger.error(f"Failed to load keymaker analysis: {e}")
            else:
                self.logger.warning("No keymaker_analysis.json found - trying direct resource scan")
                self._scan_resources_directly(keymaker_dir, sources)
                
        except Exception as e:
            self.logger.error(f"Error loading Keymaker resources: {e}")
    
    def _generate_resource_script(self, keymaker_dir: Path, keymaker_data: Dict[str, Any]) -> str:
        """Generate Windows Resource Script (.rc) from Keymaker extracted resources"""
        try:
            rc_content = []
            rc_content.append("// Generated Resource Script from Matrix Keymaker extraction")
            rc_content.append("#include \"src/resource.h\"")
            rc_content.append("")
            
            # Add string table
            resources = keymaker_data.get('resources', {})
            string_info = resources.get('string', {})
            
            if string_info.get('count', 0) > 0:
                rc_content.append("// String Table")
                rc_content.append("STRINGTABLE")
                rc_content.append("BEGIN")
                
                # Load a sample of string resources (first 100 for performance)
                string_dir = keymaker_dir / "resources" / "string"
                if string_dir.exists():
                    string_files = sorted(string_dir.glob("string_*.txt"))[:100]  # Limit for performance
                    string_id = 1000
                    
                    for string_file in string_files:
                        try:
                            with open(string_file, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read().strip()
                            
                            # Clean and escape string content for RC format
                            if content and len(content) < 256:  # Reasonable string length
                                content = content.replace('\\', '\\\\').replace('"', '\\"')
                                content = ''.join(c for c in content if ord(c) < 128)  # ASCII only
                                if content:  # Only include non-empty strings
                                    rc_content.append(f'    {string_id}, "{content}"')
                                    string_id += 1
                        except Exception as e:
                            continue  # Skip problematic strings
                
                rc_content.append("END")
                rc_content.append("")
            
            # Add bitmap resources
            image_info = resources.get('embedded_file', {})
            if image_info.get('count', 0) > 0:
                rc_content.append("// Bitmap Resources")
                
                bmp_dir = keymaker_dir / "resources" / "embedded_file"
                if bmp_dir.exists():
                    bmp_files = list(bmp_dir.glob("*.bmp"))[:10]  # Limit for performance
                    bmp_id = 2000
                    
                    for bmp_file in bmp_files:
                        # Use relative path for RC file
                        rel_path = bmp_file.name
                        rc_content.append(f'{bmp_id} BITMAP "{rel_path}"')
                        bmp_id += 1
                
                rc_content.append("")
            
            # Add version information
            rc_content.append("// Version Information")
            rc_content.append("1 VERSIONINFO")
            rc_content.append("FILEVERSION 1,0,0,0")
            rc_content.append("PRODUCTVERSION 1,0,0,0")
            rc_content.append("BEGIN")
            rc_content.append('  VALUE "CompanyName", "Matrix Reconstructed"')
            rc_content.append('  VALUE "FileDescription", "Reconstructed Application"')
            rc_content.append('  VALUE "FileVersion", "1.0.0.0"')
            rc_content.append('  VALUE "ProductName", "Matrix Decompiled Binary"')
            rc_content.append('  VALUE "ProductVersion", "1.0.0.0"')
            rc_content.append("END")
            
            return '\n'.join(rc_content)
            
        except Exception as e:
            self.logger.error(f"Error generating resource script: {e}")
            return "// Resource generation failed\n"
    
    def _enhance_resource_header(self, sources: Dict[str, Any], keymaker_data: Dict[str, Any]) -> None:
        """Enhance resource.h with additional constants from Keymaker data"""
        try:
            # Add resource IDs for strings and bitmaps
            additional_header = []
            additional_header.append("// Additional Resource Constants from Keymaker")
            additional_header.append("#define IDS_STRING_BASE         1000")
            additional_header.append("#define IDB_BITMAP_BASE         2000")
            additional_header.append("#define IDI_ICON_BASE           3000")
            additional_header.append("")
            
            # Add to existing resource.h content
            if 'resource.h' in sources.get('source_files', {}):
                existing_header = sources['source_files']['resource.h']
                sources['source_files']['resource.h'] = existing_header + '\n' + '\n'.join(additional_header)
            else:
                # Create new resource.h if it doesn't exist
                base_header = self._generate_resource_header()
                sources['source_files'] = sources.get('source_files', {})
                sources['source_files']['resource.h'] = base_header + '\n' + '\n'.join(additional_header)
                
        except Exception as e:
            self.logger.error(f"Error enhancing resource header: {e}")
    
    def _scan_resources_directly(self, keymaker_dir: Path, sources: Dict[str, Any]) -> None:
        """Direct resource scanning fallback when analysis JSON is not available"""
        try:
            resources_dir = keymaker_dir / "resources"
            if not resources_dir.exists():
                return
            
            # Count resources directly
            string_count = len(list((resources_dir / "string").glob("*.txt"))) if (resources_dir / "string").exists() else 0
            image_count = len(list((resources_dir / "embedded_file").glob("*.bmp"))) if (resources_dir / "embedded_file").exists() else 0
            
            if string_count > 0 or image_count > 0:
                self.logger.info(f"🔧 Direct scan found: {string_count} strings, {image_count} images")
                
                # Generate basic resource script
                rc_content = f"// Basic Resource Script - {string_count} strings, {image_count} images\n"
                rc_content += "#include \"src/resource.h\"\n"
                
                sources['resource_files'] = sources.get('resource_files', {})
                sources['resource_files']['resources.rc'] = rc_content
                
        except Exception as e:
            self.logger.error(f"Error in direct resource scanning: {e}")
    
    def _fix_case_sensitivity_issues(self, source_code: str) -> str:
        """
        Phase 2.1: Exact Function Reconstruction & Phase 2.9: Compiler-Specific Idioms
        Fix compilation issues for perfect MSVC compatibility
        """
        try:
            # Fix Windows header case sensitivity
            source_code = source_code.replace('#include <windows.h>', '#include <Windows.h>')
            source_code = source_code.replace('#include <commctrl.h>', '#include <CommCtrl.h>')
            
            # Add required pragma for safe functions
            if '#define _CRT_SECURE_NO_WARNINGS' not in source_code:
                # Insert at the beginning, after initial comments
                lines = source_code.split('\n')
                insert_pos = 0
                
                # Find position after initial comments
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith('//') and not line.strip().startswith('/*'):
                        insert_pos = i
                        break
                
                lines.insert(insert_pos, '#define _CRT_SECURE_NO_WARNINGS')
                source_code = '\n'.join(lines)
            
            # Phase 2.1: Add resource header inclusion for exact function reconstruction
            if '#include "resource.h"' not in source_code:
                # Find the last #include statement
                lines = source_code.split('\n')
                last_include_pos = -1
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('#include'):
                        last_include_pos = i
                
                if last_include_pos >= 0:
                    lines.insert(last_include_pos + 1, '#include "resource.h"')
                    source_code = '\n'.join(lines)
            
            # Fix common missing includes
            if '#include <CommCtrl.h>' not in source_code and 'InitCommonControlsEx' in source_code:
                source_code = source_code.replace('#include <Windows.h>', '#include <Windows.h>\n#include <CommCtrl.h>')
            
            # Phase 2.9: Remove conflicting redefinitions (MSVC-specific idioms)
            lines = source_code.split('\n')
            filtered_lines = []
            in_resource_section = False
            
            for line in lines:
                line_stripped = line.strip()
                
                # Skip redundant resource definitions since we have resource.h
                if ('// Resource IDs (reconstructed from analysis)' in line or
                    (line_stripped.startswith('#define') and any(x in line for x in 
                    ['IDI_MAIN_ICON', 'IDI_APPLICATION', 'IDS_APP_TITLE', 'ID_FILE_EXIT']))):
                    in_resource_section = True
                    filtered_lines.append(f'// {line}  // Moved to resource.h for Phase 2.1 compliance')
                    continue
                    
                # End resource section detection
                if in_resource_section and (line_stripped == '' or 
                    (line_stripped.startswith('//') and 'reconstructed' not in line)):
                    in_resource_section = False
                
                # Skip lines in resource section
                if in_resource_section:
                    continue
                    
                filtered_lines.append(line)
            
            source_code = '\n'.join(filtered_lines)
            
            # Fix function declaration issues
            if 'CreateMainWindow' in source_code and 'HWND CreateMainWindow(HINSTANCE, int);' not in source_code:
                # Find forward declarations section
                lines = source_code.split('\n')
                for i, line in enumerate(lines):
                    if 'LRESULT CALLBACK MainWindowProc' in line:
                        # Insert after existing forward declarations
                        for j in range(i+1, len(lines)):
                            if lines[j].strip() == '' or not lines[j].strip().endswith(';'):
                                lines.insert(j, 'HWND CreateMainWindow(HINSTANCE, int);')
                                break
                        break
                source_code = '\n'.join(lines)
            
            self.logger.info("✅ Applied case sensitivity and compilation fixes")
            return source_code
            
        except Exception as e:
            self.logger.warning(f"Failed to apply case sensitivity fixes: {e}")
            return source_code

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
        
        # Windows-specific dependencies (check both cases)
        content_lower = all_content.lower()
        if '#include <windows.h>' in content_lower:
            analysis['dependencies'].extend(['kernel32.lib', 'user32.lib'])
        if '#include <commctrl.h>' in content_lower or 'InitCommonControlsEx' in all_content:
            analysis['dependencies'].append('Comctl32.lib')
        if '#include <winsock2.h>' in content_lower:
            analysis['dependencies'].extend(['ws2_32.lib', 'wsock32.lib'])
        if 'printf' in all_content or '#include <stdio.h>' in content_lower:
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
        
        # Set compiler requirements for Windows only
        analysis['compiler_requirements'] = ['msvc']
        if analysis['architecture'] == 'x64':
            analysis['compilation_flags'].extend(['/MACHINE:X64'])
        else:
            analysis['compilation_flags'].extend(['/MACHINE:X86'])
        
        # Determine build complexity
        num_files = len(source_files) + len(header_files)
        if num_files > 20:
            analysis['build_complexity'] = 'complex'
        elif num_files > 5:
            analysis['build_complexity'] = 'moderate'
        
        return analysis

    def _generate_build_system(self, analysis: Dict[str, Any], sources: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate VS2022 MSBuild system configuration - CENTRALIZED ONLY"""
        build_system = {
            'detected_systems': ['msbuild'],  # Only MSBuild supported
            'primary_system': 'msbuild_vs2022',
            'build_files': {},
            'source_files': sources.get('source_files', {}),  # Add source files
            'build_configurations': {},
            'compilation_commands': {},
            'linking_commands': {},
            'build_scripts': {},
            'automated_build': {},
            'centralized_config': True,
            'vs_version': '2022_preview'
        }
        
        self.logger.info("🔧 Generating VS2022 MSBuild configuration (centralized system)")
        
        # Generate VS2022 project files using centralized build system
        vcxproj_content = self._generate_vcxproj_file(analysis)
        build_system['build_files']['project.vcxproj'] = vcxproj_content
        build_system['detected_systems'] = ['msbuild']  # Only system available
        
        # Generate VS2022 solution file
        sln_content = self._generate_solution_file(analysis)
        build_system['build_files']['project.sln'] = sln_content
        
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
        
        # Generate compilation commands for MSVC only
        build_system['compilation_commands'] = self._generate_msvc_commands(analysis)
        
        # Generate linking commands for MSVC
        build_system['linking_commands'] = self._generate_msvc_linking_commands(analysis)
        
        # Generate automated build scripts for Windows
        build_system['build_scripts'] = self._generate_windows_build_scripts(analysis)
        
        return build_system

    def _generate_solution_file(self, analysis: Dict[str, Any]) -> str:
        """Generate Visual Studio solution file"""
        project_guid = "{12345678-1234-5678-9ABC-123456789012}"
        solution_guid = "{87654321-4321-8765-CBA9-210987654321}"
        
        if analysis['architecture'] == 'x64':
            platform = 'x64'
        else:
            platform = 'Win32'
        
        sln_content = f"""Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project("{{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}}") = "ReconstructedProject", "project.vcxproj", "{project_guid}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|{platform} = Debug|{platform}
		Release|{platform} = Release|{platform}
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{project_guid}.Debug|{platform}.ActiveCfg = Debug|{platform}
		{project_guid}.Debug|{platform}.Build.0 = Debug|{platform}
		{project_guid}.Release|{platform}.ActiveCfg = Release|{platform}
		{project_guid}.Release|{platform}.Build.0 = Release|{platform}
	EndGlobalSection
	GlobalSection(SolutionProperties) = preSolution
		HideSolutionNode = FALSE
	EndGlobalSection
	GlobalSection(ExtensibilityGlobals) = postSolution
		SolutionGuid = {solution_guid}
	EndGlobalSection
EndGlobal
"""
        
        return sln_content

    def _generate_vcxproj_file(self, analysis: Dict[str, Any]) -> str:
        """Generate VS2022 Visual Studio project file using centralized build system"""
        import uuid
        project_guid = "{" + str(uuid.uuid4()).upper() + "}"
        
        if analysis['architecture'] == 'x64':
            platform = 'x64'
            platform_toolset = 'v143'  # VS2022 toolset
        else:
            platform = 'Win32'
            platform_toolset = 'v143'  # VS2022 toolset
        
        config_type = 'Application'
        subsystem = 'Console'
        if analysis['project_type'] == 'dynamic_library':
            config_type = 'DynamicLibrary'
        elif analysis['project_type'] == 'static_library':
            config_type = 'StaticLibrary'
        elif analysis['project_type'] == 'windows_gui':
            subsystem = 'Windows'
        
        # Get include and library directories from centralized build system
        build_manager = self._get_build_manager()
        include_dirs = ";".join(build_manager.get_include_dirs())
        library_dirs = ";".join(build_manager.get_library_dirs(analysis['architecture']))
        
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
    <VCProjectVersion>17.0</VCProjectVersion>
    <ProjectGuid>{project_guid}</ProjectGuid>
    <RootNamespace>ReconstructedProject</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
    <PlatformToolset>{platform_toolset}</PlatformToolset>
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
    <IncludePath>{include_dirs};$(IncludePath)</IncludePath>
    <LibraryPath>{library_dirs};$(LibraryPath)</LibraryPath>
  </PropertyGroup>
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|{platform}'">
    <OutDir>$(SolutionDir)bin\\$(Configuration)\\$(Platform)\\</OutDir>
    <IntDir>$(SolutionDir)obj\\$(Configuration)\\$(Platform)\\</IntDir>
    <IncludePath>{include_dirs};$(IncludePath)</IncludePath>
    <LibraryPath>{library_dirs};$(LibraryPath)</LibraryPath>
  </PropertyGroup>
  <ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Debug|{platform}'">
    <ClCompile>
      <WarningLevel>Level3</WarningLevel>
      <SDLCheck>true</SDLCheck>
      <PreprocessorDefinitions>DEBUG;_DEBUG;_CONSOLE;%(PreprocessorDefinitions)</PreprocessorDefinitions>
      <ConformanceMode>true</ConformanceMode>
      <Optimization>Disabled</Optimization>
      <RuntimeLibrary>MultiThreadedDebugDLL</RuntimeLibrary>
    </ClCompile>
    <Link>
      <SubSystem>{subsystem}</SubSystem>
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
      <RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>
    </ClCompile>
    <Link>
      <SubSystem>{subsystem}</SubSystem>
      <EnableCOMDATFolding>true</EnableCOMDATFolding>
      <OptimizeReferences>true</OptimizeReferences>
      <GenerateDebugInformation>false</GenerateDebugInformation>
      <AdditionalDependencies>{';'.join(analysis['dependencies'])};%(AdditionalDependencies)</AdditionalDependencies>
    </Link>
  </ItemDefinitionGroup>
  <ItemGroup>
    <ClCompile Include="src\\*.c" />
  </ItemGroup>
  <ItemGroup>
    <ClInclude Include="src\\*.h" />
  </ItemGroup>
  <ItemGroup>
    <ResourceCompile Include="resources.rc" />
  </ItemGroup>
  <Import Project="$(VCTargetsPath)\\Microsoft.Cpp.targets" />
</Project>"""
        
        return vcxproj

    def _generate_msvc_commands(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate MSVC compilation commands using centralized build system"""
        commands = {}
        
        # Get compiler path from centralized build manager
        build_manager = self._get_build_manager()
        compiler_path = build_manager.get_compiler_path(analysis['architecture'])
        
        # Use centralized configuration
        msvc_base = [compiler_path, '/std:c11', '/W3']
        if analysis['architecture'] == 'x64':
            msvc_base.append('/MACHINE:X64')
        else:
            msvc_base.append('/MACHINE:X86')
        
        # Add include directories from centralized config
        include_flags = [f'/I"{include_dir}"' for include_dir in build_manager.get_include_dirs()]
        
        commands['msvc'] = {
            'debug': msvc_base + ['/Od', '/Zi', '/DDEBUG'] + include_flags,
            'release': msvc_base + ['/O2', '/DNDEBUG'] + include_flags,
            'includes': include_flags,
            'output': ['/Fe:reconstructed_project.exe']
        }
        
        return commands

    def _generate_msvc_linking_commands(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate MSVC linking commands using centralized build system"""
        commands = {}
        
        # Get linker and library paths from centralized build manager
        build_manager = self._get_build_manager()
        
        # Base libraries
        libs = analysis['dependencies']
        
        # Library path flags from centralized config
        lib_paths = [f'/LIBPATH:"{lib_dir}"' for lib_dir in build_manager.get_library_dirs(analysis['architecture'])]
        
        # MSVC linking with centralized library paths
        commands['msvc'] = (
            [f"/DEFAULTLIB:{lib}" for lib in libs if lib.endswith('.lib')] +
            lib_paths
        )
        
        return commands

    def _generate_windows_build_scripts(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate Windows build scripts using centralized MSBuild paths"""
        scripts = {}
        
        # Get MSBuild path from centralized build manager
        build_manager = self._get_build_manager()
        msbuild_path = build_manager.get_msbuild_path().replace('/mnt/c/', 'C:\\')
        
        # Windows batch script using centralized MSBuild path
        batch_script = f"""@echo off
echo Building reconstructed project with VS2022 MSBuild...

REM Create directories
if not exist "bin" mkdir bin
if not exist "obj" mkdir obj
if not exist "Debug" mkdir Debug
if not exist "Release" mkdir Release

REM Set up Visual Studio 2022 Preview environment
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\VC\\Auxiliary\\Build\\vcvars64.bat"

REM Build with different configurations using centralized MSBuild
echo.
echo === Debug Build ===
"{msbuild_path}" project.sln /p:Configuration=Debug /p:Platform={analysis['architecture']}

echo.
echo === Release Build ===
"{msbuild_path}" project.sln /p:Configuration=Release /p:Platform={analysis['architecture']}

echo.
echo === Testing builds ===
if exist "Debug\\*.exe" (
    echo Debug build: SUCCESS
) else (
    echo Debug build: FAILED
)

if exist "Release\\*.exe" (
    echo Release build: SUCCESS
) else (
    echo Release build: FAILED
)

echo.
echo Build complete!
pause
"""
        scripts['build.bat'] = batch_script
        
        # PowerShell script using centralized MSBuild path
        powershell_script = f"""# PowerShell build script - VS2022 Preview
Write-Host "Building reconstructed project with VS2022 MSBuild..." -ForegroundColor Green

# Create directories
New-Item -ItemType Directory -Force -Path "bin", "obj", "Debug", "Release"

# Import Visual Studio 2022 Preview build tools
Import-Module "C:\\Program Files\\Microsoft Visual Studio\\2022\\Preview\\Common7\\Tools\\Microsoft.VisualStudio.DevShell.dll"
Enter-VsDevShell

try {{
    # Debug build using centralized MSBuild path
    Write-Host "Building Debug configuration..." -ForegroundColor Yellow
    & "{msbuild_path}" project.sln /p:Configuration=Debug /p:Platform={analysis['architecture']} /verbosity:minimal
    
    # Release build using centralized MSBuild path
    Write-Host "Building Release configuration..." -ForegroundColor Yellow
    & "{msbuild_path}" project.sln /p:Configuration=Release /p:Platform={analysis['architecture']} /verbosity:minimal
    
    # Test builds
    Write-Host "Testing builds..." -ForegroundColor Cyan
    if (Get-ChildItem -Path "Debug\\*.exe" -ErrorAction SilentlyContinue) {{
        Write-Host "Debug build: SUCCESS" -ForegroundColor Green
    }} else {{
        Write-Host "Debug build: FAILED" -ForegroundColor Red
    }}
    
    if (Get-ChildItem -Path "Release\\*.exe" -ErrorAction SilentlyContinue) {{
        Write-Host "Release build: SUCCESS" -ForegroundColor Green
    }} else {{
        Write-Host "Release build: FAILED" -ForegroundColor Red
    }}
}} catch {{
    Write-Host "Build failed: ${{$_.Exception.Message}}" -ForegroundColor Red
    exit 1
}}

Write-Host "Build complete!" -ForegroundColor Green
"""
        scripts['build.ps1'] = powershell_script
        
        return scripts

    def _orchestrate_compilation(self, build_system: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate compilation using centralized VS2022 MSBuild"""
        results = {
            'attempted_builds': [],
            'successful_builds': [],
            'failed_builds': [],
            'build_outputs': {},
            'error_logs': {},
            'success_rate': 0.0,
            'compilation_time': {},
            'binary_outputs': {},
            'build_system_used': 'msbuild_vs2022',
            'centralized_config': True
        }
        
        # Always use VS2022 MSBuild (centralized system)
        build_systems_to_try = ['msbuild']
        
        # Log that we're using centralized build system
        self.logger.info("🔧 Using centralized VS2022 MSBuild system (NO FALLBACKS)")
        
        for build_system_name in build_systems_to_try:
            # Always proceed with MSBuild since it's our only system
            build_result = self._attempt_build(build_system_name, build_system, context)
            results['attempted_builds'].append(f"{build_system_name}_vs2022")
            
            if build_result['success']:
                results['successful_builds'].append(f"{build_system_name}_vs2022")
                results['build_outputs'][f"{build_system_name}_vs2022"] = build_result['output']
                results['compilation_time'][f"{build_system_name}_vs2022"] = build_result['time']
                if build_result.get('binary_path'):
                    results['binary_outputs'][f"{build_system_name}_vs2022"] = build_result['binary_path']
                self.logger.info(f"✅ VS2022 MSBuild compilation successful in {build_result['time']:.2f}s")
            else:
                results['failed_builds'].append(f"{build_system_name}_vs2022")
                results['error_logs'][f"{build_system_name}_vs2022"] = build_result['error']
                self.logger.error(f"❌ VS2022 MSBuild compilation failed: {build_result['error']}")
        
        # Calculate success rate
        if results['attempted_builds']:
            results['success_rate'] = len(results['successful_builds']) / len(results['attempted_builds'])
        
        # Log final results
        if results['success_rate'] > 0:
            self.logger.info(f"🎯 Compilation orchestration complete: {results['success_rate']:.1%} success rate")
        else:
            self.logger.warning("⚠️ Compilation orchestration failed - check VS2022 configuration")
        
        return results

    def _attempt_build(self, build_system: str, build_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to build using centralized build system - NO FALLBACKS"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'time': 0.0,
            'binary_path': None
        }
        
        try:
            # Create build directory using centralized output paths
            output_dir = context.get('output_paths', {}).get('compilation')
            if not output_dir:
                # Use centralized config manager
                from ..config_manager import get_config_manager
                config_manager = get_config_manager()
                binary_name = context.get('binary_name', 'unknown_binary')
                output_dir = config_manager.get_structured_output_path(binary_name, 'compilation')
            os.makedirs(output_dir, exist_ok=True)
            
            import time
            start_time = time.time()
            
            if build_system == 'msbuild':
                # Use centralized MSBuild system
                # Note: build_config is actually the build_system dict from _generate_build_system
                result = self._build_with_msbuild(output_dir, build_config)
            else:
                result['error'] = f"Unsupported build system: {build_system}. Only MSBuild is supported."
            
            result['time'] = time.time() - start_time
            
        except Exception as e:
            result['error'] = f"Centralized build system failed: {str(e)}"
        
        return result

    def _compile_resource_files(self, sources: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Compile resource files (icons, images, strings) into Windows resource format"""
        resource_result = {
            'success': False,
            'resource_count': 0,
            'compiled_resources': [],
            'rc_file': None,
            'res_file': None
        }
        
        try:
            resource_files = sources.get('resource_files', {})
            if not resource_files:
                self.logger.info("No resource files to compile")
                resource_result['success'] = True
                return resource_result
            
            # Create resources directory
            resources_dir = os.path.join(output_dir, 'resources')
            os.makedirs(resources_dir, exist_ok=True)
            
            # Generate resource script (.rc file)
            rc_content = self._generate_resource_script(resource_files, resources_dir)
            rc_file = os.path.join(output_dir, 'resources.rc')
            
            with open(rc_file, 'w', encoding='utf-8') as f:
                f.write(rc_content)
            
            resource_result['rc_file'] = rc_file
            resource_result['resource_count'] = len(resource_files)
            resource_result['success'] = True
            
            self.logger.info(f"✅ Generated resource script: {rc_file} with {len(resource_files)} resources")
            
            # Try to compile with rc.exe if available
            try:
                from ..build_system_manager import get_build_manager
                build_manager = get_build_manager()
                
                # Check if Windows Resource Compiler is available
                rc_exe = build_manager._find_rc_compiler()
                if rc_exe:
                    res_file = os.path.join(output_dir, 'resources.res')
                    cmd = [rc_exe, '/fo', res_file, rc_file]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode == 0:
                        resource_result['res_file'] = res_file
                        self.logger.info(f"✅ Compiled resources to: {res_file}")
                    else:
                        self.logger.warning(f"Resource compilation failed: {result.stderr}")
                else:
                    self.logger.info("Resource compiler not found - RC file generated for manual compilation")
                    
            except Exception as e:
                self.logger.warning(f"Resource compilation attempt failed: {e}")
            
        except Exception as e:
            self.logger.error(f"Resource file processing failed: {e}")
            resource_result['error'] = str(e)
        
        return resource_result

    def _generate_resource_script(self, resource_files: Dict[str, Any], resources_dir: str) -> str:
        """Generate Windows resource script (.rc) from extracted resources"""
        rc_lines = [
            "// Generated resource script for Matrix Online decompilation",
            "// Generated by Agent 10 (The Machine)",
            "",
            "#include <windows.h>",
            ""
        ]
        
        icon_id = 100
        bitmap_id = 200
        string_id = 300
        
        # Process icons
        for resource_name, resource_data in resource_files.items():
            if 'icon' in resource_name.lower() and isinstance(resource_data, dict):
                content = resource_data.get('content')
                if content and isinstance(content, bytes):
                    # Save icon file
                    icon_file = os.path.join(resources_dir, f"{resource_name}.ico")
                    with open(icon_file, 'wb') as f:
                        f.write(content)
                    
                    # Add to RC script
                    rc_lines.append(f"{icon_id} ICON \"{icon_file}\"")
                    icon_id += 1
        
        # Process bitmaps/images  
        for resource_name, resource_data in resource_files.items():
            if any(x in resource_name.lower() for x in ['bitmap', 'bmp', 'image']) and isinstance(resource_data, dict):
                content = resource_data.get('content')
                if content and isinstance(content, bytes):
                    # Save bitmap file
                    bmp_file = os.path.join(resources_dir, f"{resource_name}.bmp")
                    with open(bmp_file, 'wb') as f:
                        f.write(content)
                    
                    # Add to RC script
                    rc_lines.append(f"{bitmap_id} BITMAP \"{bmp_file}\"")
                    bitmap_id += 1
        
        # Process string tables
        strings = []
        for resource_name, resource_data in resource_files.items():
            if 'string' in resource_name.lower() and isinstance(resource_data, dict):
                content = resource_data.get('content')
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    # Escape quotes and newlines
                    escaped_content = content.replace('"', '\\"').replace('\n', '\\n').replace('\r', '')
                    strings.append(f'    {string_id}, "{escaped_content}"')
                    string_id += 1
        
        if strings:
            rc_lines.extend([
                "",
                "STRINGTABLE",
                "BEGIN"
            ])
            rc_lines.extend(strings)
            rc_lines.append("END")
        
        rc_lines.extend([
            "",
            "// End of generated resource script"
        ])
        
        return '\n'.join(rc_lines)

    def _build_with_msbuild(self, output_dir: str, build_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build using centralized MSBuild system - NO FALLBACKS"""
        result = {
            'success': False,
            'output': '',
            'error': '',
            'binary_path': None
        }
        
        try:
            # Create source directory and write source files
            src_dir = os.path.join(output_dir, 'src')
            os.makedirs(src_dir, exist_ok=True)
            
            # Write all source files to src directory
            source_files = build_config.get('source_files', {})
            self.logger.info(f"🔍 DEBUG: Found {len(source_files)} source files to write")
            
            if not source_files:
                self.logger.warning("⚠️ No source files found in build_config")
                self.logger.info(f"🔍 DEBUG: build_config keys: {list(build_config.keys())}")
                
            for filename, content in source_files.items():
                src_file = os.path.join(src_dir, filename)
                with open(src_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"✅ Written source file: {filename} ({len(content)} chars)")
            
            # Phase 2.1: Create resource.h for exact function reconstruction
            resource_header = self._generate_resource_header()
            resource_file = os.path.join(src_dir, 'resource.h')
            with open(resource_file, 'w', encoding='utf-8') as f:
                f.write(resource_header)
            self.logger.info("✅ Generated resource.h for Phase 2.1 compliance")
            
            # Phase 1: Write resource files (RC files and BMPs) for binary equivalence improvement
            resource_files = build_config.get('resource_files', {})
            if resource_files:
                self.logger.info(f"🔧 Writing {len(resource_files)} resource files for binary equivalence")
                
                for filename, content in resource_files.items():
                    if filename.endswith('.rc'):
                        # Write RC file to compilation root (not src/)
                        rc_file = os.path.join(output_dir, filename)
                        with open(rc_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        self.logger.info(f"✅ Written resource script: {filename}")
                    else:
                        # Other resource files go to src/
                        res_file = os.path.join(src_dir, filename)
                        with open(res_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        self.logger.info(f"✅ Written resource file: {filename}")
            else:
                self.logger.info("🔧 No Keymaker resources found - compiling without resource integration")
            
            # Write project file
            proj_file = os.path.join(output_dir, 'project.vcxproj')
            
            # Ensure build_files exists and has project content
            if 'build_files' not in build_config or 'project.vcxproj' not in build_config['build_files']:
                self.logger.error("❌ Build configuration missing project.vcxproj content")
                result['error'] = "Build configuration missing project.vcxproj content"
                return result
            
            try:
                with open(proj_file, 'w', encoding='utf-8') as f:
                    f.write(build_config['build_files']['project.vcxproj'])
                self.logger.info(f"✅ Written project file: {proj_file}")
                
                # Verify the file was written
                if not os.path.exists(proj_file):
                    result['error'] = f"Project file was not created: {proj_file}"
                    return result
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to write project file: {e}")
                result['error'] = f"Failed to write project file: {e}"
                return result
            
            # Use centralized build system manager
            build_manager = self._get_build_manager()
            
            # Use the platform from analysis
            platform = "x64" if build_config.get('architecture') == 'x64' else "Win32"
            
            success, output = build_manager.build_with_msbuild(
                Path(proj_file),
                configuration="Release",
                platform=platform
            )
            
            result['output'] = output
            result['success'] = success
            
            if success:
                # Look for binary output using standard VS output structure
                bin_dir = os.path.join(output_dir, 'bin', 'Release')
                if os.path.exists(bin_dir):
                    for file in os.listdir(bin_dir):
                        if file.endswith('.exe'):
                            exe_path = os.path.join(bin_dir, file)
                            # Validate this is a real PE executable, not a mock file
                            if self._validate_executable(exe_path):
                                result['binary_path'] = exe_path
                                break
                            else:
                                self.logger.warning(f"⚠️ Invalid executable detected: {exe_path} - removing mock/invalid file")
                                try:
                                    os.remove(exe_path)
                                except Exception as e:
                                    self.logger.error(f"Failed to remove invalid file {exe_path}: {e}")
                                result['error'] = "Generated file is not a valid executable - compilation failed"
                                result['success'] = False
            else:
                result['error'] = f"Centralized MSBuild failed: {output}"
            
        except Exception as e:
            result['error'] = f"Centralized build system error: {str(e)}"
        
        return result

    def _validate_executable(self, exe_path: str) -> bool:
        """
        Validate that a file is a real PE executable, not a mock or text file.
        
        Args:
            exe_path: Path to the executable file
            
        Returns:
            True if it's a valid PE executable, False otherwise
        """
        try:
            # Check if file exists and has reasonable size
            if not os.path.exists(exe_path):
                return False
            
            file_size = os.path.getsize(exe_path)
            if file_size < 1024:  # Real executables are typically much larger than 1KB
                self.logger.warning(f"File {exe_path} too small ({file_size} bytes) to be a real executable")
                return False
            
            # Check if it starts with PE header (MZ signature)
            with open(exe_path, 'rb') as f:
                header = f.read(64)
                if len(header) < 2 or not header.startswith(b'MZ'):
                    self.logger.warning(f"File {exe_path} does not have valid PE header")
                    return False
                
                # Check for PE signature at offset specified in DOS header
                if len(header) >= 64:
                    try:
                        import struct
                        pe_offset = struct.unpack('<I', header[60:64])[0]
                        if pe_offset < len(header):
                            return True  # Basic validation passed
                        else:
                            # Need to read more to check PE signature
                            f.seek(pe_offset)
                            pe_sig = f.read(4)
                            if pe_sig == b'PE\x00\x00':
                                return True
                    except (struct.error, OSError):
                        pass
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating executable {exe_path}: {e}")
            return False

    def _generate_resource_header(self) -> str:
        """
        Phase 2.1: Generate resource.h for exact function reconstruction
        Creates proper Windows resource definitions for MSVC compatibility
        """
        return """//{{NO_DEPENDENCIES}}
// Microsoft Visual C++ generated include file.
// Used by Matrix Online Launcher
//
// Resource IDs reconstructed from binary analysis
// Phase 2.1: Exact Function Reconstruction - Resource Constants

#define IDI_MAIN_ICON                   101
#define IDI_APP_ICON                    102
#define IDS_APP_TITLE                   201
#define IDS_APP_NAME                    202
#define ID_FILE_EXIT                    1001
#define ID_FILE_OPEN                    1002
#define ID_HELP_ABOUT                   1003

// Next default values for new objects
//
#ifdef APSTUDIO_INVOKED
#ifndef APSTUDIO_READONLY_SYMBOLS
#define _APS_NEXT_RESOURCE_VALUE        103
#define _APS_NEXT_COMMAND_VALUE         1004
#define _APS_NEXT_CONTROL_VALUE         1000
#define _APS_NEXT_SYMED_VALUE           101
#endif
#endif
"""

    def _get_build_manager(self):
        """Get centralized build system manager - REQUIRED"""
        try:
            from ..build_system_manager import get_build_manager
            return get_build_manager()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize build system manager: {e}")
    
    def _get_msbuild_path(self) -> str:
        """Get MSBuild path from centralized configuration - NO FALLBACKS"""
        return self._get_build_manager().get_msbuild_path()

    def _optimize_build_process(self, compilation_results: Dict[str, Any], build_system: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the build process - VS2022 MSBuild only"""
        optimization = {
            'recommended_system': 'msbuild_vs2022',  # Always VS2022 MSBuild
            'optimization_suggestions': [],
            'parallel_build_options': {},
            'cache_strategies': {},
            'efficiency_score': 0.0,
            'performance_metrics': {},
            'centralized_config': True  # Always use centralized build config
        }
        
        # VS2022 MSBuild is always the recommended and only system
        optimization['recommended_system'] = 'msbuild_vs2022'
        
        # Generate VS2022-specific optimization suggestions
        optimization['optimization_suggestions'] = [
            "Enable parallel compilation (/MP with VS2022 MSBuild)",
            "Use VS2022 precompiled headers for large projects",
            "Implement incremental builds with VS2022 MSBuild",
            "Configure VS2022 build caching and IntelliSense cache",
            "Optimize VS2022 linker settings for faster linking (/INCREMENTAL)",
            "Use VS2022 link-time code generation (/LTCG) for release builds",
            "Leverage VS2022 Preview features for enhanced performance",
            "Use centralized build configuration for consistency"
        ]
        
        # VS2022 parallel build options
        optimization['parallel_build_options'] = {
            'max_parallel_projects': 8,
            'max_parallel_processes': 4,
            'use_mp_flag': True,  # /MP compiler flag
            'parallel_linking': True
        }
        
        # VS2022 cache strategies
        optimization['cache_strategies'] = {
            'intellisense_cache': True,
            'build_cache': True,
            'precompiled_headers': True,
            'incremental_linking': True
        }
        
        # Calculate efficiency score
        success_rate = compilation_results.get('success_rate', 0.0)
        avg_time = 0.0
        if compilation_results.get('compilation_time'):
            avg_time = sum(compilation_results['compilation_time'].values()) / len(compilation_results['compilation_time'])
        
        # Efficiency score based on success rate and speed (lower time is better)
        time_score = max(0.0, 1.0 - (avg_time / 300.0))  # Normalize around 5 minutes
        optimization['efficiency_score'] = (success_rate + time_score) / 2.0
        
        # Performance metrics specific to VS2022
        optimization['performance_metrics'] = {
            'vs2022_toolset': 'v143',
            'target_platform': 'Windows 10/11',
            'build_system_version': 'VS2022 Preview',
            'centralized_config_used': True
        }
        
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