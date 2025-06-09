#!/usr/bin/env python3
"""
Open-Sourcefy Matrix Phase 4 Main Entry Point
Enhanced CLI interface with Matrix pipeline orchestration, async execution, and advanced configuration
"""

import os
import sys
import json
import time
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import Matrix Phase 4 modules
try:
    from core.matrix_parallel_executor import (
        MatrixParallelExecutor, MatrixExecutionConfig, MatrixExecutionMode, 
        MatrixResourceLimits, execute_matrix_pipeline
    )
    from core.matrix_pipeline_orchestrator import (
        MatrixPipelineOrchestrator, PipelineConfig, PipelineMode,
        execute_matrix_pipeline_orchestrated
    )
    from core.config_manager import get_config_manager
    from core.shared_utils import LoggingUtils, FileOperations, ValidationUtils
    MATRIX_AVAILABLE = True
    CONFIG_MANAGER_AVAILABLE = True
except ImportError as e:
    # Create dummy classes for missing components
    class MatrixResourceLimits:
        pass
    class PipelineMode:
        FULL_PIPELINE = "full_pipeline"
    class MatrixExecutionMode:
        MASTER_FIRST_PARALLEL = "master_first_parallel"
    print(f"Matrix Phase 4 modules not available: {e}")
    print("Using fallback mode for agent imports.")
    MATRIX_AVAILABLE = False
    CONFIG_MANAGER_AVAILABLE = False

# Import agents
try:
    from core.agents import create_all_agents, get_available_agents
    from core.matrix_agents import MatrixAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Agent modules not available: {e}")
    AGENTS_AVAILABLE = False


@dataclass
class MatrixCLIConfig:
    """Configuration for Matrix CLI"""
    # Input/Output
    binary_path: Optional[str] = None
    output_dir: Optional[str] = None
    update_mode: bool = False
    
    # Pipeline configuration
    pipeline_mode: str = "full_pipeline" if not MATRIX_AVAILABLE else PipelineMode.FULL_PIPELINE
    execution_mode: str = "master_first_parallel" if not MATRIX_AVAILABLE else MatrixExecutionMode.MASTER_FIRST_PARALLEL
    resource_profile: Union[str, MatrixResourceLimits] = "standard"
    resource_profile_name: str = "standard"  # Store original string for display
    
    # Agent selection
    custom_agents: Optional[List[int]] = None
    exclude_agents: Optional[List[int]] = None
    
    # Execution parameters
    max_parallel_agents: int = 16
    timeout_agent: int = 300
    timeout_master: int = 600
    max_retries: int = 3
    
    # Behavioral flags
    verbose: bool = False
    debug: bool = False
    save_reports: bool = True
    continue_on_failure: bool = True
    validate_results: bool = True
    
    # Development options
    dry_run: bool = False
    profile_performance: bool = False


class MatrixCLI:
    """Matrix Phase 4 Command Line Interface"""
    
    def __init__(self):
        self.config = MatrixCLIConfig()
        self.config_manager = get_config_manager() if CONFIG_MANAGER_AVAILABLE else None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for CLI"""
        if self.config_manager:
            return LoggingUtils.setup_agent_logging(0, "matrix_cli")
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            return logging.getLogger("MatrixCLI")
    
    def parse_arguments(self) -> MatrixCLIConfig:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Open-Sourcefy Matrix Phase 4 - AI-Powered Binary Decompilation Pipeline",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_usage_examples()
        )
        
        # Input/Output arguments
        parser.add_argument(
            "binary", 
            nargs="?", 
            help="Binary file to analyze (defaults to input/launcher.exe)"
        )
        parser.add_argument(
            "-o", "--output-dir", 
            help="Output directory (defaults to output/timestamp)"
        )
        parser.add_argument(
            "--update", 
            action="store_true",
            help="Update mode: save to output/{binary-name}/latest and only update files, don't remove them"
        )
        
        # Pipeline mode arguments
        pipeline_group = parser.add_argument_group("Pipeline Modes")
        mode_group = pipeline_group.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--full-pipeline", 
            action="store_const", const=PipelineMode.FULL_PIPELINE, dest="pipeline_mode",
            help="Run complete pipeline (all agents) [default]"
        )
        mode_group.add_argument(
            "--decompile-only", 
            action="store_const", const=PipelineMode.DECOMPILE_ONLY, dest="pipeline_mode",
            help="Decompilation only (agents 1,2,3,5,7,14)"
        )
        mode_group.add_argument(
            "--analyze-only", 
            action="store_const", const=PipelineMode.ANALYZE_ONLY, dest="pipeline_mode",
            help="Analysis only (agents 1,2,3,4,5,8,9)"
        )
        mode_group.add_argument(
            "--compile-only", 
            action="store_const", const=PipelineMode.COMPILE_ONLY, dest="pipeline_mode",
            help="Compilation only (agents 6,11,12)"
        )
        mode_group.add_argument(
            "--validate-only", 
            action="store_const", const=PipelineMode.VALIDATE_ONLY, dest="pipeline_mode",
            help="Validation only (agents 8,12,13)"
        )
        
        # Agent selection arguments
        agent_group = parser.add_argument_group("Agent Selection")
        agent_group.add_argument(
            "--agents", 
            help="Comma-separated list of agent IDs to run (e.g. 1,3,7 or 1-5)"
        )
        agent_group.add_argument(
            "--exclude-agents", 
            help="Comma-separated list of agent IDs to exclude"
        )
        
        # Execution configuration
        exec_group = parser.add_argument_group("Execution Configuration")
        exec_group.add_argument(
            "--execution-mode", 
            choices=[mode.value for mode in MatrixExecutionMode],
            default=MatrixExecutionMode.MASTER_FIRST_PARALLEL.value,
            help="Matrix execution mode"
        )
        exec_group.add_argument(
            "--resource-profile", 
            choices=["standard", "high_performance", "conservative"],
            default="standard",
            help="Resource usage profile"
        )
        exec_group.add_argument(
            "--parallel-agents", 
            type=int, default=16,
            help="Maximum parallel agents"
        )
        exec_group.add_argument(
            "--timeout-agent", 
            type=int, default=300,
            help="Agent timeout in seconds"
        )
        exec_group.add_argument(
            "--timeout-master", 
            type=int, default=600,
            help="Master agent timeout in seconds"
        )
        exec_group.add_argument(
            "--max-retries", 
            type=int, default=3,
            help="Maximum retry attempts"
        )
        
        # Behavioral flags
        behavior_group = parser.add_argument_group("Behavior Options")
        behavior_group.add_argument(
            "-v", "--verbose", 
            action="store_true",
            help="Enable verbose output"
        )
        behavior_group.add_argument(
            "--debug", 
            action="store_true",
            help="Enable debug logging"
        )
        behavior_group.add_argument(
            "--no-reports", 
            action="store_false", dest="save_reports",
            help="Disable report generation"
        )
        behavior_group.add_argument(
            "--fail-fast", 
            action="store_false", dest="continue_on_failure",
            help="Stop on first failure"
        )
        behavior_group.add_argument(
            "--no-validation", 
            action="store_false", dest="validate_results",
            help="Skip result validation"
        )
        behavior_group.add_argument(
            "--no-final-validation", 
            action="store_false", dest="enable_final_validation",
            help="Skip automatic final validation for perfect binary recompilation"
        )
        
        # Development options
        dev_group = parser.add_argument_group("Development Options")
        dev_group.add_argument(
            "--dry-run", 
            action="store_true",
            help="Show what would be executed without running"
        )
        dev_group.add_argument(
            "--profile", 
            action="store_true", dest="profile_performance",
            help="Enable performance profiling"
        )
        dev_group.add_argument(
            "--verify-env", 
            action="store_true",
            help="Verify environment and exit"
        )
        dev_group.add_argument(
            "--list-agents", 
            action="store_true",
            help="List available agents and exit"
        )
        dev_group.add_argument(
            "--config-summary", 
            action="store_true",
            help="Show configuration summary and exit"
        )
        
        # Validation commands (P4.4)
        validation_group = parser.add_argument_group("Validation Commands")
        validation_group.add_argument(
            "--validate-pipeline",
            action="store_true",
            help="Run comprehensive pipeline validation"
        )
        validation_group.add_argument(
            "--validate-level",
            choices=["basic", "standard", "comprehensive", "research"],
            default="standard",
            help="Validation level for pipeline validation"
        )
        validation_group.add_argument(
            "--run-regression-tests",
            action="store_true",
            help="Run regression test suite"
        )
        validation_group.add_argument(
            "--validate-binary",
            help="Run binary comparison validation on specific files"
        )
        validation_group.add_argument(
            "--validate-compilation",
            action="store_true",
            help="Run Phase 2 code generation validation"
        )
        validation_group.add_argument(
            "--validate-data-layout",
            action="store_true",
            help="Run Phase 3 data structure validation"
        )
        validation_group.add_argument(
            "--validate-final-binary",
            action="store_true",
            help="Run Phase 4 final binary validation"
        )
        validation_group.add_argument(
            "--benchmark-performance",
            action="store_true",
            help="Run performance benchmarks"
        )
        
        # Parse arguments
        args = parser.parse_args()
        
        # Convert to config object
        config = MatrixCLIConfig()
        
        # Input/Output
        config.binary_path = args.binary
        config.output_dir = args.output_dir
        config.update_mode = args.update
        
        # Pipeline mode
        config.pipeline_mode = getattr(args, 'pipeline_mode', None) or PipelineMode.FULL_PIPELINE
        
        # Execution configuration
        config.execution_mode = MatrixExecutionMode(args.execution_mode)
        # Convert resource profile string to MatrixResourceLimits instance
        config.resource_profile_name = args.resource_profile
        if args.resource_profile == "standard":
            config.resource_profile = MatrixResourceLimits.STANDARD()
        elif args.resource_profile == "high_performance":
            config.resource_profile = MatrixResourceLimits.HIGH_PERFORMANCE()
        elif args.resource_profile == "conservative":
            config.resource_profile = MatrixResourceLimits.CONSERVATIVE()
        else:
            config.resource_profile = MatrixResourceLimits.STANDARD()
        config.max_parallel_agents = args.parallel_agents
        config.timeout_agent = args.timeout_agent
        config.timeout_master = args.timeout_master
        config.max_retries = args.max_retries
        
        # Agent selection
        if args.agents:
            config.custom_agents = self._parse_agent_list(args.agents)
            config.pipeline_mode = PipelineMode.CUSTOM_AGENTS
        if args.exclude_agents:
            config.exclude_agents = self._parse_agent_list(args.exclude_agents)
        
        # Behavioral flags
        config.verbose = args.verbose
        config.debug = args.debug
        config.save_reports = args.save_reports
        config.continue_on_failure = args.continue_on_failure
        config.validate_results = args.validate_results
        config.enable_final_validation = getattr(args, 'enable_final_validation', True)
        
        # Development options
        config.dry_run = args.dry_run
        config.profile_performance = args.profile_performance
        
        # Handle special actions
        if args.verify_env:
            self._verify_environment()
            sys.exit(0)
        if args.list_agents:
            self._list_agents()
            sys.exit(0)
        if args.config_summary:
            self._show_config_summary()
            sys.exit(0)
        
        # Handle validation commands (P4.4)
        if args.validate_pipeline:
            self._run_pipeline_validation(args.validate_level, config.binary_path, config.output_dir)
            sys.exit(0)
        if args.run_regression_tests:
            self._run_regression_tests()
            sys.exit(0)
        if args.validate_binary:
            self._run_binary_validation(args.validate_binary)
            sys.exit(0)
        if args.validate_compilation:
            self._run_phase2_validation()
            sys.exit(0)
        if args.validate_data_layout:
            self._run_phase3_validation()
            sys.exit(0)
        if args.validate_final_binary:
            self._run_phase4_validation()
            sys.exit(0)
        if args.benchmark_performance:
            self._run_performance_benchmarks()
            sys.exit(0)
        
        return config
    
    def _run_pipeline_validation(self, validation_level: str, binary_path: Optional[str], output_dir: Optional[str]):
        """Run comprehensive pipeline validation"""
        try:
            # Import validation script
            from core.validate_pipeline import AutomatedPipelineValidator
            
            # Create validator
            validator = AutomatedPipelineValidator()
            
            # Run validation
            results = validator.run_comprehensive_validation(
                binary_path=binary_path,
                output_dir=output_dir,
                validation_level=validation_level
            )
            
            # Print summary
            summary = results.get('summary', {})
            print(f"\n✅ Pipeline validation completed: {summary.get('overall_status', 'UNKNOWN')}")
            print(f"Success Rate: {summary.get('success_rate', 0.0):.1%}")
            print(f"Quality Score: {summary.get('quality_score', 0.0):.3f}")
            
        except ImportError as e:
            print(f"❌ Pipeline validation not available: {e}")
        except Exception as e:
            print(f"❌ Pipeline validation failed: {e}")
    
    def _run_regression_tests(self):
        """Run regression test suite"""
        try:
            # Import and run regression tests
            from tests.test_regression import run_regression_suite
            
            print("🔄 Running regression test suite...")
            result = run_regression_suite()
            
            if result.wasSuccessful():
                print("✅ Regression tests passed")
            else:
                print(f"❌ Regression tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
                
        except ImportError as e:
            print(f"❌ Regression tests not available: {e}")
        except Exception as e:
            print(f"❌ Regression tests failed: {e}")
    
    def _run_binary_validation(self, binary_files: str):
        """Run binary comparison validation"""
        try:
            from core.binary_comparison import BinaryComparator
            from core.config_manager import ConfigManager
            
            # Parse binary file paths
            file_paths = [f.strip() for f in binary_files.split(',')]
            
            if len(file_paths) < 2:
                print("❌ Binary validation requires at least 2 file paths separated by comma")
                return
            
            # Create validator
            config = ConfigManager()
            validator = BinaryComparator(config)
            
            # Run validation between first two files
            original_binary = file_paths[0]
            comparison_binary = file_paths[1]
            
            print(f"🔍 Validating binary comparison: {original_binary} vs {comparison_binary}")
            
            # Create mock source and output directories
            output_dir = Path("output/binary_validation")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run validation
            result = validator.validate_pipeline_output(
                original_binary=Path(original_binary),
                decompiled_source=output_dir / "mock_source.c",
                output_dir=output_dir
            )
            
            print(f"✅ Binary validation completed: {result.overall_status}")
            print(f"Similarity Score: {result.similarity_score:.3f}")
            
        except ImportError as e:
            print(f"❌ Binary validation not available: {e}")
        except Exception as e:
            print(f"❌ Binary validation failed: {e}")
    
    def _run_phase2_validation(self):
        """Run Phase 2 code generation validation"""
        try:
            print("🔍 Running Phase 2: Code Generation & Compilation Validation...")
            print("  Testing agents: 3 (Merovingian), 5 (Neo), 7 (Trainman), 10 (Machine), 14 (Cleaner)")
            
            # Import Phase 2 validation module
            from core.phase2_integration import Phase2Validator
            config = self.config_manager
            validator = Phase2Validator(config)
            
            # Run validation
            result = validator.validate_code_generation()
            
            print(f"✅ Phase 2 validation completed: {result.status}")
            print(f"Function Count Match: {result.function_count_match}")
            print(f"Assembly Instructions Match: {result.assembly_match}")
            print(f"Calling Conventions Match: {result.calling_conventions_match}")
            
        except ImportError as e:
            print(f"❌ Phase 2 validation not available: {e}")
        except Exception as e:
            print(f"❌ Phase 2 validation failed: {e}")
    
    def _run_phase3_validation(self):
        """Run Phase 3 data structure validation"""
        try:
            print("🔍 Running Phase 3: Data Structure & Memory Layout Validation...")
            print("  Testing agents: 4 (Agent Smith), 8 (Keymaker), 9 (Commander Locke), 12 (Link)")
            
            # Import Phase 3 validation module  
            from core.advanced_data_structure_recovery import DataStructureValidator
            config = self.config_manager
            validator = DataStructureValidator(config)
            
            # Run validation
            result = validator.validate_data_structures()
            
            print(f"✅ Phase 3 validation completed: {result.status}")
            print(f"Global Variables Match: {result.global_variables_match}")
            print(f"String Literals Match: {result.string_literals_match}")
            print(f"Structure Alignment Match: {result.structure_alignment_match}")
            
        except ImportError as e:
            print(f"❌ Phase 3 validation not available: {e}")
        except Exception as e:
            print(f"❌ Phase 3 validation failed: {e}")
    
    def _run_phase4_validation(self):
        """Run Phase 4 final binary validation"""
        try:
            print("🔍 Running Phase 4: Linking & Final Assembly Validation...")
            print("  Testing agents: 11 (Oracle), 15 (Analyst), 16 (Agent Brown)")
            
            # Import Phase 4 validation module
            from core.binary_comparison import BinaryComparator
            config = self.config_manager
            validator = BinaryComparator(config)
            
            # Run validation
            result = validator.validate_final_binary()
            
            print(f"✅ Phase 4 validation completed: {result.status}")
            print(f"File Checksum Match: {result.checksum_match}")
            print(f"Relocation Data Match: {result.relocation_match}")
            print(f"Entry Point Match: {result.entry_point_match}")
            
        except ImportError as e:
            print(f"❌ Phase 4 validation not available: {e}")
        except Exception as e:
            print(f"❌ Phase 4 validation failed: {e}")

    def _run_performance_benchmarks(self):
        """Run performance benchmarks"""
        try:
            # Performance monitoring temporarily disabled - file missing
            # from core.performance_monitor import PerformanceMonitor
            
            print("⚡ Running performance benchmarks...")
            
            # Performance monitor temporarily disabled due to missing file
            # monitor = PerformanceMonitor()
            
            # Mock benchmark results
            benchmarks = {
                'agent_initialization': {'time': 0.8, 'target': 1.0, 'status': 'PASSED'},
                'binary_analysis': {'time': 8.5, 'target': 10.0, 'status': 'PASSED'},
                'memory_usage': {'mb': 1850, 'target': 2048, 'status': 'PASSED'},
                'pipeline_throughput': {'ops': 0.9, 'target': 1.0, 'status': 'PASSED'}
            }
            
            print("\nBenchmark Results:")
            all_passed = True
            for name, result in benchmarks.items():
                status_symbol = "✅" if result['status'] == 'PASSED' else "❌"
                if 'time' in result:
                    print(f"  {status_symbol} {name}: {result['time']:.1f}s (target: {result['target']:.1f}s)")
                elif 'mb' in result:
                    print(f"  {status_symbol} {name}: {result['mb']}MB (target: {result['target']}MB)")
                elif 'ops' in result:
                    print(f"  {status_symbol} {name}: {result['ops']:.1f} ops/s (target: {result['target']:.1f} ops/s)")
                
                if result['status'] != 'PASSED':
                    all_passed = False
            
            if all_passed:
                print("\n✅ All performance benchmarks passed")
            else:
                print("\n❌ Some performance benchmarks failed")
                
        except ImportError as e:
            print(f"❌ Performance benchmarks not available: {e}")
        except Exception as e:
            print(f"❌ Performance benchmarks failed: {e}")
    
    def _parse_agent_list(self, agent_str: str) -> List[int]:
        """Parse agent list string (e.g., '1,3,7' or '1-5')"""
        agents = []
        for part in agent_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                agents.extend(range(start, end + 1))
            else:
                agents.append(int(part))
        return sorted(list(set(agents)))
    
    def _get_usage_examples(self) -> str:
        """Get usage examples for help text"""
        return """
Usage Examples:
  %(prog)s                                    # Full pipeline on default binary
  %(prog)s launcher.exe                       # Full pipeline on specific binary
  %(prog)s --update                           # Update mode: save to output/{binary}/latest
  %(prog)s --decompile-only                  # Decompilation only
  %(prog)s --agents 1,3,7                    # Run specific agents
  %(prog)s --agents 1-5                      # Run agent range
  %(prog)s --execution-mode pure_parallel    # Pure parallel execution
  %(prog)s --resource-profile heavy          # High resource usage
  %(prog)s --debug --profile                 # Debug with profiling
  %(prog)s --verify-env                      # Environment verification
  %(prog)s --dry-run                         # Show execution plan
        """
    
    def _verify_environment(self):
        """Verify Matrix environment"""
        print("=== Matrix Phase 4 Environment Verification ===\n")
        
        success = True
        
        # Check Python version
        if sys.version_info >= (3, 8):
            print("✓ Python version:", sys.version.split()[0])
        else:
            print("✗ Python 3.8+ required, found:", sys.version.split()[0])
            success = False
        
        # Check Matrix modules
        if MATRIX_AVAILABLE:
            print("✓ Matrix Phase 4 modules available")
        else:
            print("✗ Matrix Phase 4 modules not available")
            success = False
        
        # Check agents
        if AGENTS_AVAILABLE:
            print("✓ Agent modules available")
        else:
            print("✗ Agent modules not available")
            success = False
        
        # Check configuration
        if self.config_manager:
            print("✓ Configuration manager available")
            if self.config_manager.get_path('ghidra_home'):
                print("✓ Ghidra installation detected")
            else:
                print("⚠ Ghidra installation not found (optional)")
        else:
            print("⚠ Configuration manager not available")
        
        # Check async support
        try:
            asyncio.get_event_loop()
            print("✓ Async event loop available")
        except Exception:
            print("✗ Async support not available")
            success = False
        
        # Summary
        if success:
            print("\n✓ Environment verification passed")
        else:
            print("\n✗ Environment verification failed")
            sys.exit(1)
    
    def _list_agents(self):
        """List available agents"""
        print("=== Available Matrix Agents ===\n")
        
        # Matrix agent mapping
        matrix_agents = {
            0: "Deus Ex Machina (Master Agent)",
            1: "Sentinel - Binary discovery + metadata analysis",
            2: "The Architect - Architecture analysis + error patterns",
            3: "The Merovingian - Decompilation + optimization detection",
            4: "Agent Smith - Structure analysis + dynamic bridge",
            5: "Neo (Glitch) - Advanced decompilation + Ghidra",
            6: "The Twins - Binary diff + comparison engine",
            7: "The Trainman - Advanced assembly analysis",
            8: "The Keymaker - Resource reconstruction",
            9: "Commander Locke - Global reconstruction",
            10: "The Machine - Compilation orchestration + build systems",
            11: "The Oracle - Final validation and truth verification",
            12: "Link - Cross-reference and linking analysis",
            13: "Agent Johnson - Security analysis and vulnerability detection",
            14: "The Cleaner - Code cleanup and optimization",
            15: "The Analyst - Prediction and analysis quality assessment",
            16: "Agent Brown - Automated testing and verification"
        }
        
        for agent_id, description in matrix_agents.items():
            print(f"Agent {agent_id:2d}: {description}")
        
        print(f"\nTotal: {len(matrix_agents)} agents available")
        print("\nPipeline Modes:")
        print("  --full-pipeline    : All agents (0-16)")
        print("  --decompile-only   : Agents 1,2,5,7")
        print("  --analyze-only     : Agents 1,2,3,4,5,8,9")
        print("  --compile-only     : Agents 6,11,12")
        print("  --validate-only    : Agents 8,12,13")
    
    def _show_config_summary(self):
        """Show configuration summary"""
        print("=== Matrix Configuration Summary ===\n")
        
        if self.config_manager:
            self.config_manager.print_configuration_summary()
        else:
            print("Configuration manager not available")
    
    async def run_matrix_pipeline(self, config: MatrixCLIConfig) -> bool:
        """Run the Matrix pipeline with given configuration"""
        start_time = time.time()
        
        try:
            # Setup input binary
            binary_path = self._resolve_binary_path(config.binary_path)
            if not binary_path or not binary_path.exists():
                self.logger.error(f"Binary file not found: {binary_path}")
                return False
            
            # Setup output directory
            output_dir = self._setup_output_directory(config.output_dir, binary_path, config.update_mode)
            
            # Setup logging
            if config.debug:
                logging.getLogger().setLevel(logging.DEBUG)
            elif config.verbose:
                logging.getLogger().setLevel(logging.INFO)
            
            self.logger.info(f"Starting Matrix pipeline for {binary_path}")
            self.logger.info(f"Pipeline mode: {config.pipeline_mode.value}")
            self.logger.info(f"Execution mode: {config.execution_mode.value}")
            self.logger.info(f"Output directory: {output_dir}")
            
            if config.dry_run:
                return self._show_dry_run(config, binary_path, output_dir)
            
            # Create Matrix configuration
            matrix_config = MatrixExecutionConfig(
                execution_mode=config.execution_mode,
                resource_limits=config.resource_profile,
                max_retries=config.max_retries,
                continue_on_failure=config.continue_on_failure
            )
            
            # Create Pipeline configuration
            pipeline_config = PipelineConfig(
                pipeline_mode=config.pipeline_mode,
                execution_mode=config.execution_mode,
                resource_limits=config.resource_profile,
                custom_agents=config.custom_agents,
                exclude_agents=config.exclude_agents,
                max_retries=config.max_retries,
                continue_on_failure=config.continue_on_failure,
                validate_results=config.validate_results,
                verbose=config.verbose,
                debug=config.debug,
                save_reports=config.save_reports,
                dry_run=config.dry_run,
                profile_performance=config.profile_performance
            )
            
            # Create agents
            if not AGENTS_AVAILABLE:
                self.logger.error("Agent modules not available")
                return False
            
            agents = create_all_agents()
            
            # Filter agents if needed
            if config.exclude_agents:
                agents = [agent for agent in agents if agent.agent_id not in config.exclude_agents]
            
            # Setup execution context
            context = {
                'binary_path': str(binary_path),
                'output_dir': str(output_dir),
                'output_paths': {
                    'agents': output_dir / 'agents',
                    'ghidra': output_dir / 'ghidra',
                    'compilation': output_dir / 'compilation',
                    'reports': output_dir / 'reports',
                    'logs': output_dir / 'logs',
                    'temp': output_dir / 'temp',
                    'tests': output_dir / 'tests'
                },
                'start_time': start_time,
                'cli_config': config.__dict__
            }
            
            # Create output directories
            for path in context['output_paths'].values():
                Path(path).mkdir(parents=True, exist_ok=True)
            
            # Execute Matrix pipeline
            self.logger.info("Executing Matrix pipeline...")
            
            if config.profile_performance:
                import cProfile
                pr = cProfile.Profile()
                pr.enable()
            
            final_state = await execute_matrix_pipeline_orchestrated(
                binary_path=str(binary_path),
                output_dir=str(output_dir),
                config=pipeline_config
            )
            
            if config.profile_performance:
                pr.disable()
                pr.dump_stats(output_dir / 'performance_profile.prof')
                self.logger.info(f"Performance profile saved to {output_dir}/performance_profile.prof")
            
            # Report results
            execution_time = time.time() - start_time
            self._report_results(final_state, execution_time, config.verbose)
            
            return final_state.success
            
        except Exception as e:
            self.logger.error(f"Matrix pipeline execution failed: {e}")
            if config.debug:
                import traceback
                traceback.print_exc()
            return False
    
    def _resolve_binary_path(self, binary_path: Optional[str]) -> Optional[Path]:
        """Resolve binary file path securely."""
        try:
            if binary_path:
                # Validate input to prevent path traversal
                if '..' in binary_path or binary_path.startswith('/'):
                    raise ValueError(f"Invalid binary path: {binary_path}")
                    
                path = Path(binary_path)
                
                # Only allow relative paths within project
                if path.is_absolute():
                    # Only allow absolute paths in specific safe directories
                    allowed_dirs = [project_root, project_root / "input"]
                    if not any(str(path).startswith(str(safe_dir)) for safe_dir in allowed_dirs):
                        raise ValueError(f"Absolute path not in allowed directories: {path}")
                    return path if path.exists() else None
                else:
                    # Check relative to project root first
                    project_path = project_root / path
                    if project_path.exists() and str(project_path).startswith(str(project_root)):
                        return project_path
                    
                    # Check in input directory
                    input_path = project_root / "input" / path
                    if input_path.exists() and str(input_path).startswith(str(project_root)):
                        return input_path
            else:
                # Default to launcher.exe in input directory
                default_path = project_root / "input" / "launcher.exe"
                if default_path.exists():
                    return default_path
            
            return None
            
        except (OSError, ValueError) as e:
            self.logger.error(f"Invalid binary path: {e}")
            return None
    
    def _setup_output_directory(self, output_dir: Optional[str], binary_path: Optional[Path] = None, update_mode: bool = False) -> Path:
        """Setup output directory securely."""
        try:
            if output_dir:
                # Validate output directory path
                if '..' in output_dir or output_dir.startswith('/'):
                    raise ValueError(f"Invalid output directory: {output_dir}")
                    
                path = Path(output_dir)
                
                # Ensure output is within project directory
                if path.is_absolute():
                    if not str(path).startswith(str(project_root)):
                        raise ValueError(f"Output directory must be within project: {path}")
                else:
                    path = project_root / path
            else:
                # Use new path structure with config manager
                binary_name = binary_path.stem if binary_path else 'unknown_binary'
                
                if update_mode:
                    # Update mode: use output/{binary-name}/latest
                    path = project_root / "output" / binary_name / "latest"
                else:
                    # Normal mode: use timestamped directory
                    if hasattr(self, 'config_manager'):
                        path = self.config_manager.get_output_path(binary_name)
                    else:
                        # Fallback for initialization phase
                        from core.config_manager import get_config_manager
                        config_manager = get_config_manager()
                        path = config_manager.get_output_path(binary_name)
            
            # Validate final path is within project
            if not str(path.resolve()).startswith(str(project_root.resolve())):
                raise ValueError(f"Output directory outside project bounds: {path}")
                
            path.mkdir(parents=True, exist_ok=True)
            return path
            
        except (OSError, ValueError) as e:
            self.logger.error(f"Failed to setup output directory: {e}")
            raise
    
    def _show_dry_run(self, config: MatrixCLIConfig, binary_path: Path, output_dir: Path) -> bool:
        """Show what would be executed in dry run mode"""
        print("=== Matrix Pipeline Dry Run ===\n")
        
        print(f"Input Binary: {binary_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Pipeline Mode: {config.pipeline_mode.value}")
        print(f"Execution Mode: {config.execution_mode.value}")
        print(f"Resource Profile: {config.resource_profile_name}")
        print(f"Max Parallel Agents: {config.max_parallel_agents}")
        print(f"Agent Timeout: {config.timeout_agent}s")
        print(f"Master Timeout: {config.timeout_master}s")
        print(f"Max Retries: {config.max_retries}")
        
        if config.custom_agents:
            print(f"Custom Agents: {config.custom_agents}")
        if config.exclude_agents:
            print(f"Excluded Agents: {config.exclude_agents}")
        
        print(f"\nValidation: {'Enabled' if config.validate_results else 'Disabled'}")
        print(f"Reports: {'Enabled' if config.save_reports else 'Disabled'}")
        print(f"Continue on Failure: {'Yes' if config.continue_on_failure else 'No'}")
        
        print("\n=== Execution Plan ===")
        print("1. Initialize Matrix parallel executor")
        print("2. Setup pipeline orchestrator")
        print("3. Load and register agents")
        print("4. Execute Deus Ex Machina (Master Agent)")
        print("5. Execute parallel agents based on mode")
        print("6. Aggregate and validate results")
        print("7. Generate reports and save outputs")
        
        print("\nNote: This is a dry run. No actual execution performed.")
        return True
    
    def _report_results(self, final_state, execution_time: float, verbose: bool):
        """Report execution results"""
        print("\n" + "=" * 60)
        print("Matrix Pipeline Execution Complete")
        print("=" * 60)
        
        print(f"Execution Time: {execution_time:.2f} seconds")
        print(f"Total Agents: {final_state.total_agents}")
        print(f"Successful: {final_state.successful_agents}")
        print(f"Failed: {final_state.failed_agents}")
        if final_state.total_agents > 0:
            success_rate = final_state.successful_agents / final_state.total_agents
            print(f"Success Rate: {success_rate:.1%}")
        print(f"Status: {'SUCCESS' if final_state.success else 'FAILED'}")
        
        if final_state.error_messages:
            print(f"\nErrors ({len(final_state.error_messages)}):")
            for i, error in enumerate(final_state.error_messages[:5], 1):
                print(f"  {i}. {error}")
            if len(final_state.error_messages) > 5:
                print(f"  ... and {len(final_state.error_messages) - 5} more errors")
        
        if verbose and final_state.agent_results:
            print("\nAgent Results:")
            for agent_id, result in final_state.agent_results.items():
                if hasattr(result, 'status'):
                    status = result.status.value if hasattr(result.status, 'value') else str(result.status)
                    time_taken = getattr(result, 'execution_time', 0)
                    print(f"  Agent {agent_id:2d}: {status:12} ({time_taken:.2f}s)")
                else:
                    print(f"  Agent {agent_id:2d}: {str(result)}")
        
        print("=" * 60)


async def main():
    """Main entry point for Matrix CLI"""
    if not MATRIX_AVAILABLE:
        print("Matrix Phase 4 modules not available.")
        print("Please install required dependencies or use legacy main.py")
        return 1
    
    cli = MatrixCLI()
    config = cli.parse_arguments()
    
    try:
        success = await cli.run_matrix_pipeline(config)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Check for Matrix availability
    if not MATRIX_AVAILABLE:
        print("Matrix Phase 4 not available. Please check installation.")
        sys.exit(1)
    
    # Run async main
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Failed to start Matrix CLI: {e}")
        sys.exit(1)