#!/usr/bin/env python3
"""
Pipeline Execution Helper Script
Automates pipeline execution tasks and provides utilities for pipeline management.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import argparse


def setup_logging(debug: bool = False):
    """Setup logging for pipeline helper."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class PipelineHelper:
    """Helper class for pipeline execution and management."""
    
    def __init__(self, project_root: Path, debug: bool = False):
        self.project_root = Path(project_root).resolve()
        self.logger = setup_logging(debug)
        self.output_root = self.project_root / "output"
        self.scripts_dir = self.project_root / "scripts"
        
        # Ensure output directory exists
        self.output_root.mkdir(exist_ok=True)
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate environment using the environment validator script."""
        env_validator = self.scripts_dir / "environment_validator.py"
        
        if not env_validator.exists():
            self.logger.error(f"Environment validator not found: {env_validator}")
            return {"valid": False, "error": "Validator script not found"}
        
        try:
            result = subprocess.run([
                sys.executable, str(env_validator), "--json"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                validation_data = json.loads(result.stdout)
                self.logger.info("✅ Environment validation completed successfully")
                return validation_data
            else:
                self.logger.error(f"Environment validation failed: {result.stderr}")
                return {"valid": False, "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            self.logger.error("Environment validation timed out")
            return {"valid": False, "error": "Validation timeout"}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse validation output: {e}")
            return {"valid": False, "error": "JSON parse error"}
        except Exception as e:
            self.logger.error(f"Environment validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def create_output_structure(self, session_name: str = None) -> Path:
        """Create output directory structure for a pipeline run."""
        if session_name is None:
            session_name = time.strftime("%Y%m%d_%H%M%S")
        
        session_dir = self.output_root / session_name
        
        # Use file operations script to create structure
        file_ops_script = self.scripts_dir / "file_operations.py"
        
        if file_ops_script.exists():
            try:
                result = subprocess.run([
                    sys.executable, str(file_ops_script), 
                    "create-structure", str(session_dir)
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.logger.info(f"✅ Created output structure: {session_dir}")
                    return session_dir
                else:
                    self.logger.warning(f"File operations script failed: {result.stderr}")
            except Exception as e:
                self.logger.warning(f"Failed to use file operations script: {e}")
        
        # Fallback: create structure manually
        directories = ['agents', 'ghidra', 'compilation', 'reports', 'logs', 'temp', 'tests']
        for dir_name in directories:
            (session_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output structure: {session_dir}")
        return session_dir
    
    def run_pipeline(self, binary_path: Path, output_dir: Path = None, 
                    agents: List[int] = None, mode: str = "full", 
                    debug: bool = False) -> Dict[str, Any]:
        """Run the pipeline with specified parameters."""
        if output_dir is None:
            output_dir = self.create_output_structure()
        
        # Validate binary exists
        if not binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {binary_path}")
        
        # Build command
        main_script = self.project_root / "main.py"
        if not main_script.exists():
            raise FileNotFoundError(f"Main script not found: {main_script}")
        
        cmd = [sys.executable, str(main_script), str(binary_path)]
        cmd.extend(["--output-dir", str(output_dir)])
        
        if agents:
            cmd.extend(["--agents", ",".join(map(str, agents))])
        
        if mode != "full":
            mode_map = {
                "decompile": "--decompile-only",
                "analyze": "--analyze-only", 
                "compile": "--compile-only",
                "validate": "--validate-only"
            }
            if mode in mode_map:
                cmd.append(mode_map[mode])
        
        if debug:
            cmd.extend(["--debug", "--log-level", "DEBUG"])
        
        self.logger.info(f"Running pipeline: {' '.join(cmd)}")
        
        # Execute pipeline
        start_time = time.time()
        try:
            result = subprocess.run(cmd, cwd=self.project_root,
                                  capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            execution_time = time.time() - start_time
            
            pipeline_result = {
                "success": result.returncode == 0,
                "execution_time": execution_time,
                "output_dir": str(output_dir),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
            if result.returncode == 0:
                self.logger.info(f"✅ Pipeline completed successfully in {execution_time:.2f}s")
            else:
                self.logger.error(f"❌ Pipeline failed with return code {result.returncode}")
                
            return pipeline_result
            
        except subprocess.TimeoutExpired:
            self.logger.error("Pipeline execution timed out")
            return {
                "success": False,
                "error": "Pipeline timeout",
                "execution_time": time.time() - start_time,
                "output_dir": str(output_dir)
            }
    
    def analyze_pipeline_results(self, output_dir: Path) -> Dict[str, Any]:
        """Analyze pipeline results and generate summary."""
        analysis = {
            "output_dir": str(output_dir),
            "agents_completed": 0,
            "agents_failed": 0,
            "compilation_attempted": False,
            "compilation_successful": False,
            "files_generated": 0,
            "total_size_mb": 0.0,
            "errors": [],
            "warnings": []
        }
        
        if not output_dir.exists():
            analysis["errors"].append("Output directory does not exist")
            return analysis
        
        # Analyze agent results
        agents_dir = output_dir / "agents"
        if agents_dir.exists():
            for agent_dir in agents_dir.iterdir():
                if agent_dir.is_dir():
                    # Look for success indicators
                    result_files = list(agent_dir.glob("*.json"))
                    if result_files:
                        try:
                            with open(result_files[0]) as f:
                                result_data = json.load(f)
                                if result_data.get("status") == "completed":
                                    analysis["agents_completed"] += 1
                                else:
                                    analysis["agents_failed"] += 1
                        except:
                            analysis["agents_failed"] += 1
                    else:
                        analysis["agents_failed"] += 1
        
        # Analyze compilation results
        compilation_dir = output_dir / "compilation"
        if compilation_dir.exists():
            analysis["compilation_attempted"] = True
            
            # Look for build files
            build_files = (
                list(compilation_dir.glob("*.sln")) +
                list(compilation_dir.glob("CMakeLists.txt")) +
                list(compilation_dir.glob("Makefile"))
            )
            
            if build_files:
                # Look for executables
                executables = (
                    list(compilation_dir.glob("*.exe")) +
                    list(compilation_dir.rglob("*.exe")) +
                    [f for f in compilation_dir.rglob("*") if f.is_file() and os.access(f, os.X_OK)]
                )
                analysis["compilation_successful"] = len(executables) > 0
        
        # Count files and calculate size
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                analysis["files_generated"] += 1
                analysis["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
        
        # Analyze logs for errors and warnings
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.log"):
                try:
                    with open(log_file) as f:
                        content = f.read()
                        analysis["errors"].extend([
                            line.strip() for line in content.split('\n') 
                            if 'ERROR' in line or 'Exception' in line
                        ])
                        analysis["warnings"].extend([
                            line.strip() for line in content.split('\n')
                            if 'WARNING' in line or 'WARN' in line
                        ])
                except:
                    pass
        
        return analysis
    
    def test_compilation(self, output_dir: Path, build_system: str = "auto") -> Dict[str, Any]:
        """Test compilation of generated code."""
        compilation_dir = output_dir / "compilation"
        
        if not compilation_dir.exists():
            return {
                "success": False,
                "error": "No compilation directory found"
            }
        
        # Use build system automation script
        build_script = self.scripts_dir / "build_system_automation.py"
        
        if not build_script.exists():
            return {
                "success": False,
                "error": "Build system automation script not found"
            }
        
        try:
            result = subprocess.run([
                sys.executable, str(build_script),
                "--output-dir", str(output_dir),
                "test", "--build-system", build_system
            ], capture_output=True, text=True, timeout=300)
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr,
                "build_system": build_system
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Compilation test timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def cleanup_old_outputs(self, max_age_days: int = 7) -> List[str]:
        """Clean up old output directories."""
        if not self.output_root.exists():
            return []
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        cleaned_dirs = []
        
        for item in self.output_root.iterdir():
            if item.is_dir():
                dir_age = current_time - item.stat().st_mtime
                if dir_age > max_age_seconds:
                    try:
                        import shutil
                        shutil.rmtree(item)
                        cleaned_dirs.append(str(item))
                        self.logger.info(f"Cleaned up old output directory: {item}")
                    except Exception as e:
                        self.logger.error(f"Failed to clean up {item}: {e}")
        
        return cleaned_dirs
    
    def generate_report(self, output_dir: Path, report_file: Path = None) -> Path:
        """Generate comprehensive pipeline report."""
        if report_file is None:
            report_file = output_dir / "reports" / "pipeline_summary.json"
        
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Analyze results
        analysis = self.analyze_pipeline_results(output_dir)
        
        # Add metadata
        report = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_analysis": analysis,
            "environment_info": {
                "platform": sys.platform,
                "python_version": sys.version,
                "working_directory": str(self.project_root)
            }
        }
        
        # Save report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Generated pipeline report: {report_file}")
        return report_file


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Pipeline Execution Helper')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate environment
    validate_parser = subparsers.add_parser('validate-env', help='Validate environment')
    
    # Run pipeline
    run_parser = subparsers.add_parser('run', help='Run pipeline')
    run_parser.add_argument('binary', help='Binary file to analyze')
    run_parser.add_argument('--output-dir', help='Output directory')
    run_parser.add_argument('--agents', help='Comma-separated list of agents to run')
    run_parser.add_argument('--mode', choices=['full', 'decompile', 'analyze', 'compile', 'validate'],
                          default='full', help='Pipeline mode')
    
    # Analyze results
    analyze_parser = subparsers.add_parser('analyze', help='Analyze pipeline results')
    analyze_parser.add_argument('output_dir', help='Output directory to analyze')
    
    # Test compilation
    test_parser = subparsers.add_parser('test-compile', help='Test compilation')
    test_parser.add_argument('output_dir', help='Output directory with compilation files')
    test_parser.add_argument('--build-system', choices=['auto', 'cmake', 'msbuild', 'make'],
                           default='auto', help='Build system to use')
    
    # Cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old outputs')
    cleanup_parser.add_argument('--max-age', type=int, default=7, help='Max age in days')
    
    # Generate report
    report_parser = subparsers.add_parser('report', help='Generate pipeline report')
    report_parser.add_argument('output_dir', help='Output directory to report on')
    report_parser.add_argument('--report-file', help='Report output file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        helper = PipelineHelper(args.project_root, args.debug)
        
        if args.command == 'validate-env':
            result = helper.validate_environment()
            if result.get('valid', False):
                print("✅ Environment validation passed")
            else:
                print(f"❌ Environment validation failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.command == 'run':
            binary_path = Path(args.binary)
            output_dir = Path(args.output_dir) if args.output_dir else None
            agents = [int(x) for x in args.agents.split(',')] if args.agents else None
            
            result = helper.run_pipeline(binary_path, output_dir, agents, args.mode, args.debug)
            
            if result['success']:
                print(f"✅ Pipeline completed successfully")
                print(f"Output directory: {result['output_dir']}")
                print(f"Execution time: {result['execution_time']:.2f}s")
            else:
                print(f"❌ Pipeline failed")
                if 'error' in result:
                    print(f"Error: {result['error']}")
                if result.get('stderr'):
                    print(f"Stderr: {result['stderr']}")
                sys.exit(1)
        
        elif args.command == 'analyze':
            analysis = helper.analyze_pipeline_results(Path(args.output_dir))
            print(f"Pipeline Analysis for {args.output_dir}:")
            print(f"  Agents completed: {analysis['agents_completed']}")
            print(f"  Agents failed: {analysis['agents_failed']}")
            print(f"  Compilation attempted: {analysis['compilation_attempted']}")
            print(f"  Compilation successful: {analysis['compilation_successful']}")
            print(f"  Files generated: {analysis['files_generated']}")
            print(f"  Total size: {analysis['total_size_mb']:.2f} MB")
            print(f"  Errors: {len(analysis['errors'])}")
            print(f"  Warnings: {len(analysis['warnings'])}")
        
        elif args.command == 'test-compile':
            result = helper.test_compilation(Path(args.output_dir), args.build_system)
            if result['success']:
                print(f"✅ Compilation test passed using {result['build_system']}")
            else:
                print(f"❌ Compilation test failed: {result.get('error', 'Unknown error')}")
                if result.get('errors'):
                    print(f"Build errors:\n{result['errors']}")
        
        elif args.command == 'cleanup':
            cleaned = helper.cleanup_old_outputs(args.max_age)
            print(f"Cleaned up {len(cleaned)} old output directories")
            for cleaned_dir in cleaned:
                print(f"  {cleaned_dir}")
        
        elif args.command == 'report':
            report_file = Path(args.report_file) if args.report_file else None
            report_path = helper.generate_report(Path(args.output_dir), report_file)
            print(f"Generated report: {report_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()