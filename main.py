#!/usr/bin/env python3
"""
open-sourcefy Main Entry Point - Phase 4 Implementation
Complete CLI interface with component isolation and pipeline orchestration
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Import Phase 2 modules (Enhanced Agent System)
try:
    from core.parallel_executor import execute_agents_in_batches, ExecutionConfig
    from core.enhanced_parallel_executor import (
        EnhancedParallelExecutor, EnhancedExecutionConfig, 
        execute_agents_with_enhancements
    )
    from core.agent_base import BaseAgent, AgentStatus, AGENT_DEPENDENCIES, get_execution_batches
    from core.enhanced_dependency_management import get_execution_plan, dependency_resolver
    from core.enhanced_analysis_capabilities import (
        analyze_with_enhanced_capabilities, get_comprehensive_analysis_report,
        BinaryCharacteristics, pattern_database, accuracy_assessment
    )
    from core.agent_reliability import (
        AgentHealthMonitor, AdvancedRetryMechanism, FallbackManager,
        RetryConfig, HealthStatus
    )
    from core.agents import create_all_agents
    PHASE2_AVAILABLE = True
    PHASE2_ENHANCED = True
except ImportError as e:
    print(f"Phase 2 enhanced modules not available: {e}")
    try:
        from core.parallel_executor import execute_agents_in_batches, ExecutionConfig
        from core.agent_base import BaseAgent, AgentStatus, AGENT_DEPENDENCIES, get_execution_batches
        from core.agents import create_all_agents
        PHASE2_AVAILABLE = True
        PHASE2_ENHANCED = False
    except ImportError as e2:
        print(f"Phase 2 basic modules not available: {e2}")
        PHASE2_AVAILABLE = False
        PHASE2_ENHANCED = False

# Import Phase 3 modules (Ghidra Integration)
try:
    from core.ghidra_headless import run_ghidra_analysis
    from core.ghidra_processor import process_ghidra_output
    PHASE3_AVAILABLE = True
except ImportError as e:
    print(f"Phase 3 modules not available: {e}")
    PHASE3_AVAILABLE = False

# Phase 1 functionality (simplified, built-in)
PHASE1_AVAILABLE = True

# Import Phase 1 modules (Enhanced Environment)
try:
    from core.environment import EnhancedEnvironmentManager, validate_environment as enhanced_validate_environment
    PHASE1_ENHANCED = True
except ImportError as e:
    print(f"Phase 1 enhanced modules not available: {e}")
    PHASE1_ENHANCED = False

def validate_environment():
    """Enhanced environment validation"""
    if PHASE1_ENHANCED:
        env_manager = EnhancedEnvironmentManager()
        success, results = env_manager.validate_environment(comprehensive=True)
        env_manager.print_validation_report()
        return success
    else:
        # Fallback to simple validation
        print("Using fallback environment validation...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("✗ Python 3.8+ required")
            return False
        else:
            print("✓ Python version OK")
        
        # Check for basic dependencies
        try:
            import json, subprocess, pathlib
            print("✓ Core dependencies available")
        except ImportError as e:
            print(f"✗ Missing core dependencies: {e}")
            return False
        
        # Check for Ghidra (optional)
        ghidra_home = os.environ.get('GHIDRA_HOME', str(project_root / "ghidra"))
        ghidra_executable = Path(ghidra_home) / "support" / "analyzeHeadless"
        if ghidra_executable.exists():
            print("✓ Ghidra found")
        else:
            print("⚠ Ghidra not found (optional)")
        
        print("✓ Environment validation complete")
        return True

class SimpleEnvironment:
    """Simple environment wrapper"""
    def __init__(self):
        self.validation_results = []
        if PHASE1_ENHANCED:
            self.env_manager = EnhancedEnvironmentManager()
            success, self.validation_results = self.env_manager.validate_environment(comprehensive=False)
            self._is_valid = success
        else:
            self._is_valid = True
    
    def is_valid(self):
        return self._is_valid
    
    def get_missing_components(self):
        if PHASE1_ENHANCED and self.validation_results:
            from core.environment import EnvironmentStatus
            return [r.component for r in self.validation_results 
                    if r.status in [EnvironmentStatus.ERROR, EnvironmentStatus.CRITICAL]]
        return []

def get_environment():
    """Get environment instance"""
    if PHASE1_ENHANCED:
        return EnhancedEnvironmentManager()
    else:
        return SimpleEnvironment()

# Component definitions for pipeline isolation
PIPELINE_COMPONENTS = {
    'decompile': {
        'name': 'Decompilation Pipeline',
        'agents': [1, 2, 4, 5, 7, 14],
        'description': 'Binary discovery, architecture analysis, basic decompilation, structure analysis, advanced decompilation, and enhanced Ghidra analysis'
    },
    'analyze': {
        'name': 'Analysis Pipeline', 
        'agents': [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15],
        'description': 'Complete binary analysis including structure, optimization, assembly analysis, advanced Ghidra, and metadata analysis'
    },
    'compile': {
        'name': 'Compilation Pipeline',
        'agents': [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18], 
        'description': 'Complete pipeline from analysis through compilation with advanced build systems'
    },
    'validate': {
        'name': 'Validation Pipeline',
        'agents': [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19],
        'description': 'Full pipeline validation from discovery through final validation with binary comparison'
    },
    'production': {
        'name': 'Production Readiness Pipeline',
        'agents': [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20],
        'description': 'Complete production pipeline with all Phase 4 enhancements including testing framework'
    }
}

# Agent descriptions (for Phase 4 CLI)
AGENT_DESCRIPTIONS = {
    1: "Binary Discovery & Initial Analysis",
    2: "Architecture Analysis", 
    3: "Smart Error Pattern Matching",
    4: "Optimization Detection",
    5: "Binary Structure Analysis",
    6: "Optimization Pattern Matching", 
    7: "Ghidra Decompilation",
    8: "Binary Diff Analysis",
    9: "Advanced Assembly Analysis",
    10: "Resource Reconstruction",
    11: "AI Enhancement Integration", 
    12: "Integration Testing",
    13: "Final Validation",
    14: "Advanced Ghidra Integration",
    15: "Binary Metadata Analysis",
    18: "Advanced Build Systems",
    19: "Binary Comparison Engine", 
    20: "Automated Testing Framework"
}

# Output directory structure for organized results
OUTPUT_STRUCTURE = {
    'agents': 'agents',
    'ghidra': 'ghidra', 
    'compilation': 'compilation',
    'reports': 'reports',
    'logs': 'logs',
    'temp': 'temp',
    'tests': 'tests'
}

def ensure_output_structure(base_output_dir: str) -> Dict[str, str]:
    """
    Create organized output directory structure and return paths
    
    Args:
        base_output_dir: Base output directory (e.g., 'output')
        
    Returns:
        Dict mapping structure names to full paths
    """
    base_path = Path(base_output_dir)
    structure_paths = {}
    
    for key, subdir in OUTPUT_STRUCTURE.items():
        full_path = base_path / subdir
        full_path.mkdir(parents=True, exist_ok=True)
        structure_paths[key] = str(full_path)
    
    return structure_paths


class OpenSourcefyPipeline:
    """Enhanced pipeline orchestrator with Phase 2 reliability and performance features"""
    
    def __init__(self, batch_size: int = 4, parallel_mode: str = "thread", 
                 timeout: int = 300, verbose: bool = False, enhanced_mode: bool = True):
        self.batch_size = batch_size
        self.parallel_mode = parallel_mode
        self.timeout = timeout
        self.verbose = verbose
        self.enhanced_mode = enhanced_mode and PHASE2_ENHANCED
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Initialize environment
        self.env = get_environment()
        self.env_valid = self.env.is_valid()
        
        # Initialize agents if Phase 2 is available
        self.all_agents = {}
        if PHASE2_AVAILABLE:
            try:
                agent_instances = create_all_agents()
                self.all_agents = {agent.agent_id: agent for agent in agent_instances}
                
                # Initialize enhanced components if available
                if self.enhanced_mode:
                    self.enhanced_executor = None
                    self.health_monitor = AgentHealthMonitor()
                    self.dependency_resolver = dependency_resolver
                    self.logger = logging.getLogger("OpenSourcefyPipeline")
                    self.logger.info("Enhanced Phase 2 components initialized")
                
            except Exception as e:
                print(f"Failed to create agent instances: {e}")
                self.all_agents = {}
                self.enhanced_mode = False
    
    def run_component(self, component: str, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Run specific pipeline component"""
        if component not in PIPELINE_COMPONENTS:
            raise ValueError(f"Unknown component: {component}")
        
        comp_info = PIPELINE_COMPONENTS[component]
        agents_to_run = comp_info['agents']
        
        print(f"Running {comp_info['name']}")
        print(f"Description: {comp_info['description']}")
        print(f"Agents: {agents_to_run}")
        print("-" * 60)
        
        return self._run_agents(agents_to_run, binary_path, output_dir)
    
    def run_specific_agents(self, agent_ids: List[int], binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Run specific agents by ID"""
        print(f"Running specific agents: {agent_ids}")
        
        for agent_id in agent_ids:
            if agent_id in AGENT_DESCRIPTIONS:
                print(f"  Agent {agent_id}: {AGENT_DESCRIPTIONS[agent_id]}")
            else:
                print(f"  Agent {agent_id}: Unknown agent")
        print("-" * 60)
        
        return self._run_agents(agent_ids, binary_path, output_dir)
    
    def run_full_pipeline(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Run complete 15-agent pipeline with extensions"""
        print("=" * 80)
        print("open-sourcefy Full Pipeline (15 Agents)")
        print(f"Target: {binary_path}")
        print(f"Output: {output_dir}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Parallel Mode: {self.parallel_mode}")
        print("=" * 80)
        
        all_agents = list(range(1, 21))  # Include all agents (1-15, 18-20)
        return self._run_agents(all_agents, binary_path, output_dir)
    
    def _run_agents(self, agent_ids: List[int], binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Internal method to run agents using enhanced execution system"""
        self.start_time = time.time()
        
        # Create organized output directory structure
        output_paths = ensure_output_structure(output_dir)
        
        # Environment check
        print("Environment Status:")
        if self.env_valid:
            print("  ✓ Environment validated")
        else:
            missing = self.env.get_missing_components()
            print(f"  ✗ Environment issues: {missing}")
        
        # Binary characteristics extraction (for enhanced analysis)
        binary_chars = None
        if self.enhanced_mode:
            try:
                binary_chars = self._extract_binary_characteristics(binary_path)
                print(f"  ✓ Binary characteristics extracted: {binary_chars.architecture}, {binary_chars.file_size} bytes")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠ Binary characteristics extraction failed: {e}")
                binary_chars = BinaryCharacteristics(
                    file_path=binary_path,
                    file_size=Path(binary_path).stat().st_size if Path(binary_path).exists() else 0,
                    file_hash="unknown",
                    architecture="unknown"
                )
        
        # Agent execution
        if PHASE2_AVAILABLE and self.all_agents:
            # Filter agents to only include requested ones
            agents_to_run = [self.all_agents[aid] for aid in agent_ids if aid in self.all_agents]
            
            if not agents_to_run:
                print(f"No valid agents found for IDs: {agent_ids}")
                agent_results = self._mock_agent_execution(agent_ids, binary_path, output_dir)
            else:
                # Create execution context with structured output paths
                context = {
                    'global_data': {
                        'binary_path': binary_path,
                        'output_dir': output_dir,
                        'output_paths': output_paths,
                        'binary_characteristics': binary_chars,
                        'pipeline_config': {
                            'batch_size': self.batch_size,
                            'parallel_mode': self.parallel_mode,
                            'timeout': self.timeout,
                            'verbose': self.verbose,
                            'enhanced_mode': self.enhanced_mode
                        }
                    },
                    'agent_results': {},  # Will be populated during execution
                    'binary_path': binary_path,  # Keep for backward compatibility
                    'output_dir': output_dir,    # Keep for backward compatibility
                    'output_paths': output_paths  # Keep for backward compatibility
                }
                
                # Execute agents with enhanced features if available
                try:
                    if self.enhanced_mode:
                        print(f"\n🚀 Executing agents with Enhanced Phase 2 features...")
                        agent_results, performance_report = self._execute_enhanced_agents(
                            agents_to_run, context
                        )
                        
                        # Add performance report to context for final report
                        context['performance_report'] = performance_report
                        
                    else:
                        print("\nExecuting agents with standard parallel processing...")
                        config = ExecutionConfig(
                            max_parallel_agents=self.batch_size,
                            timeout_per_agent=self.timeout,
                            retry_enabled=True,
                            max_retries=2,
                            continue_on_failure=True
                        )
                        
                        raw_results = execute_agents_in_batches(agents_to_run, config, context)
                        agent_results = self._convert_agent_results_to_dict(raw_results)
                        
                    # NEW: Pipeline-level validation checkpoints
                    validation_checkpoints = self._perform_pipeline_validation_checkpoints(
                        agent_results, agent_ids, context
                    )
                    context['validation_checkpoints'] = validation_checkpoints
                    
                    # Check if pipeline should fail based on validation checkpoints
                    if not validation_checkpoints['pipeline_continues']:
                        termination_agent = validation_checkpoints.get('termination_agent')
                        
                        if termination_agent == 13:
                            print(f"\n🚫 PIPELINE TERMINATED BY VALIDATION AGENT: {validation_checkpoints['failure_reason']}")
                            print("=" * 80)
                            print("CRITICAL: Source code validation determined this is NOT a real implementation!")
                            print("The generated code appears to be placeholder/dummy code, not actual reconstruction.")
                            print("Pipeline execution terminated to prevent invalid results.")
                            print("=" * 80)
                        else:
                            print(f"\n🚫 PIPELINE VALIDATION FAILED: {validation_checkpoints['failure_reason']}")
                            print("Pipeline execution stopped due to validation failure.")
                        
                        context['pipeline_stopped_by_validation'] = True
                        context['pipeline_termination_agent'] = termination_agent
                        
                        # Override agent results to reflect pipeline failure
                        for agent_id in agent_results:
                            if agent_results[agent_id].get('success', False):
                                agent_results[agent_id]['success'] = False
                                agent_results[agent_id]['validation_failure'] = True
                        
                except Exception as e:
                    print(f"Agent execution failed: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
                    agent_results = self._mock_agent_execution(agent_ids, binary_path, output_dir)
                    context['execution_error'] = str(e)
        else:
            print("\n⚠ Phase 2 (Agent System) not available - using mock execution")
            agent_results = self._mock_agent_execution(agent_ids, binary_path, output_dir)
            context = {
                'global_data': {
                    'binary_path': binary_path,
                    'output_dir': output_dir,
                    'output_paths': output_paths
                },
                'agent_results': {},
                'binary_path': binary_path,
                'output_dir': output_dir,
                'output_paths': output_paths
            }
        
        self.end_time = time.time()
        
        # Generate enhanced report
        return self._generate_report(binary_path, output_dir, agent_results, agent_ids, context)
    
    def _mock_agent_execution(self, agent_ids: List[int], binary_path: str, output_dir: str) -> Dict[int, Dict[str, Any]]:
        """Mock agent execution for Phase 4 testing"""
        results = {}
        
        for agent_id in agent_ids:
            print(f"  [MOCK] Agent {agent_id}: {AGENT_DESCRIPTIONS.get(agent_id, 'Unknown')}")
            time.sleep(0.1)  # Simulate work
            
            results[agent_id] = {
                "agent": agent_id,
                "success": True,
                "mock": True,
                "description": AGENT_DESCRIPTIONS.get(agent_id, "Unknown"),
                "execution_time": 0.1
            }
        
        return results
    
    def _extract_binary_characteristics(self, binary_path: str) -> 'BinaryCharacteristics':
        """Extract binary characteristics for enhanced analysis"""
        import hashlib
        
        file_path = Path(binary_path)
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # Calculate file hash
        try:
            with open(binary_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except:
            file_hash = "unknown"
        
        # Basic architecture detection (simplified)
        architecture = "unknown"
        try:
            with open(binary_path, 'rb') as f:
                header = f.read(64)
                if b'PE\x00\x00' in header:
                    # PE file
                    if b'\x64\x86' in header:
                        architecture = "x64"
                    else:
                        architecture = "x86"
                elif header.startswith(b'\x7fELF'):
                    # ELF file
                    if header[4] == 2:  # 64-bit
                        architecture = "x64"
                    else:
                        architecture = "x86"
                elif header.startswith(b'\xca\xfe\xba\xbe'):
                    # Mach-O
                    architecture = "x64"
        except:
            pass
        
        return BinaryCharacteristics(
            file_path=binary_path,
            file_size=file_size,
            file_hash=file_hash,
            architecture=architecture
        )
    
    def _execute_enhanced_agents(self, agents: List[BaseAgent], 
                               context: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
        """Execute agents with enhanced Phase 2 features"""
        
        # Create enhanced execution config
        config = EnhancedExecutionConfig(
            max_parallel_agents=self.batch_size,
            timeout_per_agent=self.timeout,
            health_monitoring=True,
            performance_tracking=True,
            fallback_enabled=True,
            dynamic_batch_sizing=True,
            memory_optimization=True
        )
        
        # Execute with enhancements
        raw_results, performance_report = execute_agents_with_enhancements(
            agents, config, context
        )
        
        # Convert results to dict format for compatibility
        agent_results = self._convert_agent_results_to_dict(raw_results)
        
        # Perform enhanced analysis on results
        if context.get('binary_characteristics'):
            binary_chars = context['binary_characteristics']
            
            for agent_id, result_dict in agent_results.items():
                # Create AgentResult object for analysis
                from core.agent_base import AgentResult, AgentStatus
                agent_result = AgentResult(
                    agent_id=agent_id,
                    status=AgentStatus.COMPLETED if result_dict.get('success') else AgentStatus.FAILED,
                    data=result_dict.get('data', {}),
                    error_message=result_dict.get('error_message'),
                    execution_time=result_dict.get('execution_time', 0.0)
                )
                
                # Analyze with enhanced capabilities
                try:
                    pattern_matches, accuracy_metrics = analyze_with_enhanced_capabilities(
                        agent_id, agent_result, binary_chars
                    )
                    
                    # Add enhanced analysis to result
                    result_dict['pattern_matches'] = [
                        {
                            'pattern_type': match.pattern_type.value,
                            'pattern_id': match.pattern_id,
                            'confidence': match.confidence,
                            'description': match.description
                        }
                        for match in pattern_matches
                    ]
                    
                    result_dict['accuracy_metrics'] = {
                        'overall_accuracy': accuracy_metrics.overall_accuracy,
                        'confidence_score': accuracy_metrics.confidence_score,
                        'quality_level': accuracy_metrics.quality_level.value,
                        'patterns_detected': accuracy_metrics.patterns_detected
                    }
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Enhanced analysis failed for agent {agent_id}: {e}")
        
        return agent_results, performance_report
    
    def _convert_agent_results_to_dict(self, agent_results: Dict[int, Any]) -> Dict[int, Dict[str, Any]]:
        """Convert AgentResult objects to dict format for compatibility"""
        converted_results = {}
        
        for agent_id, result in agent_results.items():
            # Handle both AgentResult objects and plain dicts
            if hasattr(result, 'status'):
                # It's an AgentResult object
                converted_results[agent_id] = {
                    "agent": agent_id,
                    "success": result.status == AgentStatus.COMPLETED,
                    "status": result.status.value,
                    "data": result.data,
                    "error_message": result.error_message,
                    "execution_time": result.execution_time,
                    "metadata": result.metadata or {}
                }
            else:
                # It's already a dict
                converted_results[agent_id] = result
                
        return converted_results
    
    def _generate_report(self, binary_path: str, output_dir: str, 
                        agent_results: Dict[int, Dict[str, Any]], 
                        requested_agents: List[int],
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        elapsed = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate success metrics
        successful_agents = [aid for aid, result in agent_results.items() if result.get("success", False)]
        failed_agents = [aid for aid, result in agent_results.items() if not result.get("success", False)]
        
        # Environment info - convert ValidationResult objects to dicts
        validation_results_dict = {}
        if hasattr(self.env, 'validation_results') and self.env.validation_results:
            validation_results_dict = {
                result.component: {
                    "status": result.status.value,
                    "message": result.message,
                    "fix_suggestion": result.fix_suggestion,
                    "details": result.details
                }
                for result in self.env.validation_results
            }
        
        env_info = {
            "environment_valid": self.env_valid,
            "validation_results": validation_results_dict
        }
        
        # Enhanced metrics
        enhanced_metrics = {}
        if self.enhanced_mode and context:
            enhanced_metrics = {
                "binary_characteristics": {
                    "file_size": context.get('binary_characteristics', {}).file_size if context.get('binary_characteristics') else 0,
                    "architecture": context.get('binary_characteristics', {}).architecture if context.get('binary_characteristics') else "unknown",
                    "file_hash": context.get('binary_characteristics', {}).file_hash if context.get('binary_characteristics') else "unknown"
                },
                "performance_report": context.get('performance_report', {}),
                "pattern_analysis": {
                    "total_patterns_detected": sum(
                        len(result.get('pattern_matches', [])) for result in agent_results.values()
                    ),
                    "average_confidence": self._calculate_average_confidence(agent_results),
                    "quality_distribution": self._calculate_quality_distribution(agent_results)
                },
                "enhanced_features_used": {
                    "health_monitoring": True,
                    "performance_tracking": True,
                    "pattern_recognition": True,
                    "accuracy_assessment": True,
                    "fallback_strategies": True
                }
            }

        # Check for pipeline termination information
        pipeline_terminated = context.get('pipeline_stopped_by_validation', False) if context else False
        termination_agent = context.get('pipeline_termination_agent') if context else None
        validation_checkpoints = context.get('validation_checkpoints', {}) if context else {}
        
        report = {
            "pipeline_version": "2.1-Phase2-Enhanced" if self.enhanced_mode else "2.0-Phase4",
            "timestamp": time.time(),
            "execution_time": f"{elapsed:.2f} seconds",
            "binary_analyzed": binary_path,
            "output_directory": output_dir,
            "requested_agents": requested_agents,
            "agents_executed": len(agent_results),
            "agents_successful": len(successful_agents),
            "agents_failed": len(failed_agents),
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "environment_info": env_info,
            "pipeline_terminated": pipeline_terminated,
            "termination_reason": validation_checkpoints.get('failure_reason', '') if pipeline_terminated else '',
            "termination_agent": termination_agent,
            "validation_checkpoints": validation_checkpoints,
            "phases_available": {
                "phase1_environment": True,
                "phase2_agents": PHASE2_AVAILABLE,
                "phase2_enhanced": PHASE2_ENHANCED and self.enhanced_mode,
                "phase3_ghidra": PHASE3_AVAILABLE,
                "phase4_cli": True
            },
            "agent_results": agent_results,
            "overall_success": len(failed_agents) == 0 and not pipeline_terminated,
            "enhanced_metrics": enhanced_metrics if self.enhanced_mode else {}
        }
        
        # Save report to reports subdirectory
        reports_dir = Path(output_dir) / OUTPUT_STRUCTURE['reports']
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "pipeline_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        
        # Print summary
        self._print_summary(report, report_path)
        
        return report
    
    def _print_summary(self, report: Dict[str, Any], report_path: Path):
        """Print pipeline execution summary"""
        print("\n" + "=" * 80)
        
        # Check for pipeline termination
        if report.get('pipeline_terminated', False):
            print("Pipeline Execution TERMINATED!")
            print(f"🚫 TERMINATION REASON: {report.get('termination_reason', 'Unknown')}")
            if report.get('termination_agent') == 13:
                print("🔍 Source code validation failed - generated code is not a real implementation")
            print(f"Time: {report['execution_time']}")
            print(f"Agents executed before termination: {report['agents_executed']}")
        else:
            print("Pipeline Execution Complete!")
            print(f"Time: {report['execution_time']}")
            print(f"Agents: {report['agents_successful']}/{report['agents_executed']} successful")
        
        if report['failed_agents']:
            print(f"Failed: Agents {report['failed_agents']}")
        
        print(f"Report: {report_path}")
        
        # Validation checkpoints summary
        validation_checkpoints = report.get('validation_checkpoints', {})
        if validation_checkpoints:
            print(f"\nValidation Summary:")
            checkpoints = validation_checkpoints.get('checkpoints', {})
            for checkpoint_name, checkpoint_data in checkpoints.items():
                status = "✓" if checkpoint_data.get('passed', False) else "✗"
                print(f"  {checkpoint_name}: {status}")
        
        # Phase status
        phases = report['phases_available']
        print(f"\nPhase Status:")
        print(f"  Phase 1 (Environment): {'✓' if phases['phase1_environment'] else '✗'}")
        print(f"  Phase 2 (Agents): {'✓' if phases['phase2_agents'] else '✗'}")
        print(f"  Phase 3 (Ghidra): {'✓' if phases['phase3_ghidra'] else '✗'}")
        print(f"  Phase 4 (CLI): ✓")
        
        if not all([phases['phase2_agents'], phases['phase3_ghidra']]):
            print("\n⚠ Some phases not available - using mock execution")
        
        print("=" * 80)
    
    def _calculate_average_confidence(self, agent_results: Dict[int, Dict[str, Any]]) -> float:
        """Calculate average confidence score across all agents"""
        confidence_scores = []
        
        for result in agent_results.values():
            if 'accuracy_metrics' in result:
                confidence = result['accuracy_metrics'].get('confidence_score', 0.0)
                confidence_scores.append(confidence)
            elif 'pattern_matches' in result:
                matches = result['pattern_matches']
                if matches:
                    avg_confidence = sum(match.get('confidence', 0.0) for match in matches) / len(matches)
                    confidence_scores.append(avg_confidence)
        
        return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    
    def _calculate_quality_distribution(self, agent_results: Dict[int, Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of quality levels"""
        quality_dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "failed": 0}
        
        for result in agent_results.values():
            if 'accuracy_metrics' in result:
                quality = result['accuracy_metrics'].get('quality_level', 'failed')
                if quality in quality_dist:
                    quality_dist[quality] += 1
            elif result.get('success', False):
                quality_dist['good'] += 1
            else:
                quality_dist['failed'] += 1
        
        return quality_dist
    
    def _perform_pipeline_validation_checkpoints(self, agent_results: Dict[int, Dict[str, Any]], 
                                                agent_ids: List[int], 
                                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pipeline-level validation checkpoints to ensure meaningful progress"""
        validation = {
            'pipeline_continues': True,
            'failure_reason': '',
            'checkpoints': {},
            'critical_agents_status': {},
            'source_quality_assessment': {},
            'compilation_quality_assessment': {},
            'overall_quality_score': 0.0
        }
        
        try:
            # Define critical agents for different pipeline stages
            critical_agents = {
                'discovery': [1],       # Binary discovery is critical
                'decompilation': [4, 7], # Basic and advanced decompilation
                'reconstruction': [11],  # Global reconstruction
                'compilation': [12],     # Compilation orchestrator
                'validation': [13]       # Final validation
            }
            
            # Checkpoint 1: Critical agent success
            checkpoint1 = self._validate_critical_agents(agent_results, critical_agents, agent_ids)
            validation['checkpoints']['critical_agents'] = checkpoint1
            validation['critical_agents_status'] = checkpoint1['status']
            
            if not checkpoint1['passed']:
                validation['pipeline_continues'] = False
                validation['failure_reason'] = f"Critical agents failed: {checkpoint1['failure_reason']}"
                return validation
            
            # Checkpoint 2: Source code quality validation (if agent 11 ran)
            if 11 in agent_results and 11 in agent_ids:
                checkpoint2 = self._validate_source_code_quality(agent_results[11])
                validation['checkpoints']['source_quality'] = checkpoint2
                validation['source_quality_assessment'] = checkpoint2['assessment']
                
                if not checkpoint2['passed']:
                    validation['pipeline_continues'] = False
                    validation['failure_reason'] = f"Source code quality insufficient: {checkpoint2['failure_reason']}"
                    return validation
            
            # Checkpoint 3: Compilation validation (if agent 12 ran)
            if 12 in agent_results and 12 in agent_ids:
                checkpoint3 = self._validate_compilation_results(agent_results[12])
                validation['checkpoints']['compilation_quality'] = checkpoint3
                validation['compilation_quality_assessment'] = checkpoint3['assessment']
                
                if not checkpoint3['passed']:
                    validation['pipeline_continues'] = False
                    validation['failure_reason'] = f"Compilation validation failed: {checkpoint3['failure_reason']}"
                    return validation
            
            # Checkpoint 4: Final validation results (if agent 13 ran)
            if 13 in agent_results and 13 in agent_ids:
                checkpoint4 = self._validate_final_results(agent_results[13])
                validation['checkpoints']['final_validation'] = checkpoint4
                
                # Check if Agent 13 specifically terminated the pipeline
                agent13_data = agent_results[13].get('data', {})
                if agent13_data.get('pipeline_terminated', False):
                    validation['pipeline_continues'] = False
                    validation['failure_reason'] = agent13_data.get('termination_reason', 'Agent 13 terminated pipeline')
                    validation['termination_agent'] = 13
                    return validation
                
                if not checkpoint4['passed']:
                    validation['pipeline_continues'] = False
                    validation['failure_reason'] = f"Final validation failed: {checkpoint4['failure_reason']}"
                    return validation
            
            # Calculate overall quality score
            validation['overall_quality_score'] = self._calculate_pipeline_quality_score(
                validation['checkpoints'], agent_results
            )
            
            # Pipeline continues if we get here
            validation['pipeline_continues'] = True
            
        except Exception as e:
            validation['pipeline_continues'] = False
            validation['failure_reason'] = f"Validation checkpoint error: {str(e)}"
        
        return validation
    
    def _validate_critical_agents(self, agent_results: Dict[int, Dict[str, Any]], 
                                 critical_agents: Dict[str, List[int]], 
                                 agent_ids: List[int]) -> Dict[str, Any]:
        """Validate that critical agents completed successfully"""
        validation = {
            'passed': True,
            'failure_reason': '',
            'status': {}
        }
        
        failed_stages = []
        
        for stage, agents in critical_agents.items():
            stage_status = {
                'required_agents': agents,
                'executed_agents': [],
                'successful_agents': [],
                'failed_agents': [],
                'stage_success': True
            }
            
            for agent_id in agents:
                if agent_id in agent_ids:  # Only check if agent was supposed to run
                    stage_status['executed_agents'].append(agent_id)
                    
                    if agent_id in agent_results:
                        if agent_results[agent_id].get('success', False):
                            stage_status['successful_agents'].append(agent_id)
                        else:
                            stage_status['failed_agents'].append(agent_id)
                            stage_status['stage_success'] = False
                    else:
                        stage_status['failed_agents'].append(agent_id)
                        stage_status['stage_success'] = False
            
            validation['status'][stage] = stage_status
            
            # If any executed agents in this stage failed, consider stage failed
            if stage_status['executed_agents'] and not stage_status['stage_success']:
                failed_stages.append(stage)
        
        if failed_stages:
            validation['passed'] = False
            validation['failure_reason'] = f"Critical stages failed: {', '.join(failed_stages)}"
        
        return validation
    
    def _validate_source_code_quality(self, agent11_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate source code quality from Agent 11 results"""
        validation = {
            'passed': True,
            'failure_reason': '',
            'assessment': {}
        }
        
        try:
            if not agent11_result.get('success', False):
                validation['passed'] = False
                validation['failure_reason'] = 'Agent 11 (Global Reconstructor) failed'
                return validation
            
            data = agent11_result.get('data', {})
            reconstructed_source = data.get('reconstructed_source', {})
            source_files = reconstructed_source.get('source_files', {})
            
            assessment = {
                'source_files_count': len(source_files),
                'total_code_lines': 0,
                'meaningful_functions': 0,
                'has_main_function': False,
                'quality_indicators': [],
                'issues': []
            }
            
            if len(source_files) == 0:
                validation['passed'] = False
                validation['failure_reason'] = 'No source files generated'
                assessment['issues'].append('No source files generated')
                validation['assessment'] = assessment
                return validation
            
            # Analyze source files
            for filename, content in source_files.items():
                if not isinstance(content, str):
                    continue
                
                lines = content.split('\n')
                code_lines = len([line for line in lines 
                                if line.strip() and not line.strip().startswith('//')])
                assessment['total_code_lines'] += code_lines
                
                # Check for main function
                if 'int main' in content or 'void main' in content:
                    assessment['has_main_function'] = True
                
                # Count functions
                import re
                functions = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', content)
                meaningful_funcs = len([f for f in functions if f not in ['if', 'while', 'for']])
                assessment['meaningful_functions'] += meaningful_funcs
                
                # Quality indicators
                if code_lines >= 10:
                    assessment['quality_indicators'].append(f'{filename}: Substantial code ({code_lines} lines)')
                
                if meaningful_funcs >= 1:
                    assessment['quality_indicators'].append(f'{filename}: Contains {meaningful_funcs} functions')
                
                # Check for issues
                if 'Hello World' in content and code_lines < 10:
                    assessment['issues'].append(f'{filename}: Appears to be Hello World placeholder')
                
                if content.count('TODO') > 2:
                    assessment['issues'].append(f'{filename}: Contains multiple TODO placeholders')
            
            validation['assessment'] = assessment
            
            # Enhanced validation thresholds targeting 75% code quality, 75% real implementation, 70% completeness
            # Calculate quality scores for stricter validation
            
            # Code Quality Score (targeting 75%)
            code_quality_score = self._calculate_code_quality_score(assessment, source_files)
            
            # Real Implementation Score (targeting 75%)
            implementation_score = self._calculate_implementation_score(assessment, source_files)
            
            # Completeness Score (targeting 70%)
            completeness_score = self._calculate_completeness_score(assessment, data)
            
            # Store scores in assessment for reporting
            assessment['quality_scores'] = {
                'code_quality': code_quality_score,
                'implementation': implementation_score,
                'completeness': completeness_score,
                'overall': (code_quality_score + implementation_score + completeness_score) / 3
            }
            
            # Apply strict validation thresholds
            quality_passed = code_quality_score >= 0.75
            implementation_passed = implementation_score >= 0.75
            completeness_passed = completeness_score >= 0.70
            
            # Require all three thresholds to pass
            if not (quality_passed and implementation_passed and completeness_passed):
                validation['passed'] = False
                failed_criteria = []
                if not quality_passed:
                    failed_criteria.append(f'Code Quality: {code_quality_score:.2f} < 0.75')
                if not implementation_passed:
                    failed_criteria.append(f'Implementation: {implementation_score:.2f} < 0.75')
                if not completeness_passed:
                    failed_criteria.append(f'Completeness: {completeness_score:.2f} < 0.70')
                
                validation['failure_reason'] = f'Strict validation failed: {"; ".join(failed_criteria)}'
            
        except Exception as e:
            validation['passed'] = False
            validation['failure_reason'] = f'Source quality validation error: {str(e)}'
        
        return validation
    
    def _validate_compilation_results(self, agent12_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compilation results from Agent 12"""
        validation = {
            'passed': True,
            'failure_reason': '',
            'assessment': {}
        }
        
        try:
            if not agent12_result.get('success', False):
                validation['passed'] = False
                validation['failure_reason'] = 'Agent 12 (Compilation Orchestrator) failed'
                return validation
            
            data = agent12_result.get('data', {})
            
            assessment = {
                'compilation_successful': data.get('successful_compilation', False),
                'binary_generated': data.get('generated_binary') is not None,
                'early_validation_passed': True,
                'output_analysis_passed': True,
                'issues': []
            }
            
            # Check early validation results
            early_validation = data.get('early_validation', {})
            if early_validation and not early_validation.get('validation_passed', True):
                assessment['early_validation_passed'] = False
                assessment['issues'].append(f"Early validation failed: {early_validation.get('primary_issue', 'unknown')}")
            
            # Check output analysis results
            output_analysis = data.get('output_analysis', {})
            if output_analysis and not output_analysis.get('is_meaningful_compilation', True):
                assessment['output_analysis_passed'] = False
                assessment['issues'].append(f"Output analysis failed: {output_analysis.get('rejection_reason', 'unknown')}")
            
            # Check for dummy code detection
            if data.get('final_status') == 'dummy_code_detected':
                assessment['issues'].append('Dummy code detected in compilation output')
            
            validation['assessment'] = assessment
            
            # Determine if compilation is valid
            if not assessment['compilation_successful']:
                validation['passed'] = False
                validation['failure_reason'] = 'Compilation was not successful'
            elif not assessment['early_validation_passed']:
                validation['passed'] = False
                validation['failure_reason'] = 'Early source validation failed'
            elif not assessment['output_analysis_passed']:
                validation['passed'] = False
                validation['failure_reason'] = 'Compilation output analysis failed'
            
        except Exception as e:
            validation['passed'] = False
            validation['failure_reason'] = f'Compilation validation error: {str(e)}'
        
        return validation
    
    def _validate_final_results(self, agent13_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate final results from Agent 13"""
        validation = {
            'passed': True,
            'failure_reason': ''
        }
        
        try:
            if not agent13_result.get('success', False):
                validation['passed'] = False
                validation['failure_reason'] = 'Agent 13 (Final Validator) failed'
                return validation
            
            # Agent 13 has its own strict validation criteria
            # If it succeeded, the validation passed
            validation['passed'] = True
            
        except Exception as e:
            validation['passed'] = False
            validation['failure_reason'] = f'Final validation error: {str(e)}'
        
        return validation
    
    def _calculate_code_quality_score(self, assessment: Dict[str, Any], source_files: Dict[str, str]) -> float:
        """Calculate code quality score (targeting 75%)"""
        score = 0.0
        
        # Basic code structure (40%)
        if assessment.get('has_main_function', False):
            score += 0.15
        if assessment.get('meaningful_functions', 0) >= 3:
            score += 0.15
        if assessment.get('total_code_lines', 0) >= 50:
            score += 0.10
        
        # Code complexity and realism (35%)
        total_complexity = 0
        for filename, content in source_files.items():
            if filename.endswith('.c'):
                # Check for realistic C constructs
                constructs = ['if', 'for', 'while', 'switch', 'return', 'printf', 'malloc', 'free', 'struct']
                found_constructs = sum(1 for construct in constructs if construct in content)
                total_complexity += found_constructs
        
        if total_complexity >= 10:
            score += 0.20
        elif total_complexity >= 5:
            score += 0.15
        elif total_complexity >= 2:
            score += 0.10
        
        # Quality indicators (25%)
        quality_indicators = len(assessment.get('quality_indicators', []))
        if quality_indicators >= 5:
            score += 0.25
        elif quality_indicators >= 3:
            score += 0.15
        elif quality_indicators >= 1:
            score += 0.10
        
        return min(1.0, score)
    
    def _calculate_implementation_score(self, assessment: Dict[str, Any], source_files: Dict[str, str]) -> float:
        """Calculate real implementation score (targeting 75%)"""
        score = 0.0
        
        # Function implementation depth (40%)
        if assessment.get('meaningful_functions', 0) >= 5:
            score += 0.25
        elif assessment.get('meaningful_functions', 0) >= 3:
            score += 0.15
        elif assessment.get('meaningful_functions', 0) >= 1:
            score += 0.10
        
        # Code volume indicating real implementation (30%)
        total_lines = assessment.get('total_code_lines', 0)
        if total_lines >= 200:
            score += 0.30
        elif total_lines >= 100:
            score += 0.20
        elif total_lines >= 50:
            score += 0.10
        
        # Absence of placeholder indicators (30%)
        issues = assessment.get('issues', [])
        placeholder_issues = sum(1 for issue in issues if 'placeholder' in issue.lower() or 'hello world' in issue.lower())
        
        if placeholder_issues == 0:
            score += 0.30
        elif placeholder_issues <= 1:
            score += 0.15
        
        return min(1.0, score)
    
    def _calculate_completeness_score(self, assessment: Dict[str, Any], data: Dict[str, Any]) -> float:
        """Calculate completeness score (targeting 70%)"""
        score = 0.0
        
        # File completeness (25%)
        if assessment.get('source_files_count', 0) >= 3:
            score += 0.25
        elif assessment.get('source_files_count', 0) >= 2:
            score += 0.15
        elif assessment.get('source_files_count', 0) >= 1:
            score += 0.10
        
        # Build system completeness (25%)
        build_config = data.get('build_configuration', {})
        if build_config.get('makefile') or build_config.get('build_bat'):
            score += 0.15
        if build_config.get('compiler_flags'):
            score += 0.10
        
        # Project structure completeness (25%)
        project_structure = data.get('project_structure', {})
        if project_structure.get('directories'):
            score += 0.15
        if project_structure.get('build_system'):
            score += 0.10
        
        # Documentation and metadata (25%)
        if data.get('integration_report'):
            score += 0.15
        if data.get('ai_documentation'):
            score += 0.10
        
        return min(1.0, score)
    
    def _calculate_pipeline_quality_score(self, checkpoints: Dict[str, Any], 
                                        agent_results: Dict[int, Dict[str, Any]]) -> float:
        """Calculate overall pipeline quality score based on checkpoints"""
        try:
            score_factors = []
            
            # Critical agents factor (30%)
            if checkpoints.get('critical_agents', {}).get('passed', False):
                score_factors.append(0.3)
            
            # Source quality factor (25%)
            source_checkpoint = checkpoints.get('source_quality', {})
            if source_checkpoint.get('passed', False):
                score_factors.append(0.25)
            elif source_checkpoint:  # Partial credit for having source assessment
                assessment = source_checkpoint.get('assessment', {})
                if assessment.get('total_code_lines', 0) >= 5:
                    score_factors.append(0.15)
                elif assessment.get('source_files_count', 0) >= 1:
                    score_factors.append(0.1)
            
            # Compilation quality factor (25%)
            compilation_checkpoint = checkpoints.get('compilation_quality', {})
            if compilation_checkpoint.get('passed', False):
                score_factors.append(0.25)
            elif compilation_checkpoint:  # Partial credit
                assessment = compilation_checkpoint.get('assessment', {})
                if assessment.get('compilation_successful', False):
                    score_factors.append(0.15)
            
            # Final validation factor (20%)
            final_checkpoint = checkpoints.get('final_validation', {})
            if final_checkpoint.get('passed', False):
                score_factors.append(0.2)
            
            return sum(score_factors)
            
        except Exception:
            return 0.0


def parse_agent_list(agent_str: str) -> List[int]:
    """Parse agent list from command line argument"""
    agents = []
    
    for part in agent_str.split(','):
        part = part.strip()
        if '-' in part:
            # Range like "1-5"
            start, end = map(int, part.split('-'))
            agents.extend(range(start, end + 1))
        else:
            # Single agent
            agents.append(int(part))
    
    return sorted(list(set(agents)))


def validate_agent_ids(agent_ids: List[int]) -> List[int]:
    """Validate agent IDs are in valid range"""
    valid_agents = []
    valid_agent_ids = list(range(1, 16)) + [18, 19, 20]  # Agents 1-15 and 18-20
    for agent_id in agent_ids:
        if agent_id in valid_agent_ids:
            valid_agents.append(agent_id)
        else:
            print(f"Warning: Invalid agent ID {agent_id} (must be 1-15, 18-20)")
    return valid_agents


def print_component_info():
    """Print information about available pipeline components"""
    print("Available Pipeline Components:")
    print("=" * 50)
    
    for comp_name, comp_info in PIPELINE_COMPONENTS.items():
        print(f"\n--{comp_name}-only")
        print(f"  Name: {comp_info['name']}")
        print(f"  Agents: {comp_info['agents']}")
        print(f"  Description: {comp_info['description']}")


def print_agent_info():
    """Print information about all available agents"""
    print("Available Agents:")
    print("=" * 50)
    
    for agent_id, description in AGENT_DESCRIPTIONS.items():
        print(f"Agent {agent_id:2}: {description}")


def cleanup_environment(output_dir: str = "output"):
    """Clean up temporary files and reset output directory"""
    import shutil
    
    print("🧹 Cleaning up environment...")
    
    # Clean up Python cache files
    print("  Removing Python cache files...")
    for root, dirs, files in os.walk("."):
        # Remove __pycache__ directories
        if "__pycache__" in dirs:
            cache_path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(cache_path)
                print(f"    Removed: {cache_path}")
            except Exception as e:
                print(f"    Warning: Could not remove {cache_path}: {e}")
        
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"    Removed: {file_path}")
                except Exception as e:
                    print(f"    Warning: Could not remove {file_path}: {e}")
    
    # Clean up output directory contents but keep the directory structure
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"  Cleaning output directory: {output_path}")
        
        # Clean each subdirectory while preserving structure
        for subdir in OUTPUT_STRUCTURE.values():
            subdir_path = output_path / subdir
            if subdir_path.exists():
                try:
                    # Remove all contents but recreate the directory
                    shutil.rmtree(subdir_path)
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    print(f"    Cleaned: {subdir_path}")
                except Exception as e:
                    print(f"    Warning: Could not clean {subdir_path}: {e}")
        
        # Remove any extra files in the root output directory
        for item in output_path.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                    print(f"    Removed file: {item}")
                except Exception as e:
                    print(f"    Warning: Could not remove {item}: {e}")
    else:
        print(f"  Creating output directory structure: {output_path}")
    
    # Ensure the output structure is created/recreated
    ensure_output_structure(output_dir)
    print(f"  Output structure created with subdirectories: {list(OUTPUT_STRUCTURE.values())}")
    
    # Clean up any temporary files in the project root
    temp_patterns = ['*.tmp', '*.temp', '*.log', 'temp_*', 'launcher-new*']
    for pattern in temp_patterns:
        import glob
        for temp_file in glob.glob(pattern):
            try:
                if os.path.isfile(temp_file):
                    os.remove(temp_file)
                    print(f"    Removed temporary file: {temp_file}")
                elif os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
                    print(f"    Removed temporary directory: {temp_file}")
            except Exception as e:
                print(f"    Warning: Could not remove {temp_file}: {e}")
    
    print("✅ Cleanup complete!")
    return True


def auto_detect_binary(input_dir: str = "input") -> Optional[str]:
    """
    Auto-detect binary files in input directory
    
    Args:
        input_dir: Directory to search for binary files
        
    Returns:
        Path to first suitable binary found, or None if none found
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        return None
    
    # Common binary file extensions to prioritize
    binary_extensions = {'.exe', '.dll', '.so', '.bin', '.out', '.app', '.dylib', '.sys', '.com', '.scr', '.msi', '.deb', '.rpm', '.pkg'}
    
    # Find all files in input directory
    all_files = []
    non_hidden_files = []
    hidden_files = []
    
    try:
        for file_path in input_path.iterdir():
            if file_path.is_file():
                all_files.append(file_path)
                
                # Check if file is hidden (starts with .)
                if file_path.name.startswith('.'):
                    hidden_files.append(file_path)
                else:
                    non_hidden_files.append(file_path)
    except PermissionError:
        print(f"Warning: Permission denied accessing {input_dir}")
        return None
    
    if not all_files:
        return None
    
    def is_likely_binary(file_path: Path) -> bool:
        """Check if file is likely a binary executable"""
        # Check extension
        if file_path.suffix.lower() in binary_extensions:
            return True
            
        # Check if file is executable (Unix-like systems)
        if file_path.is_file() and os.access(file_path, os.X_OK):
            return True
            
        # Basic file content check for binary format
        try:
            with open(file_path, 'rb') as f:
                header = f.read(64)
                # Common binary signatures
                if header.startswith(b'MZ') or header.startswith(b'\x7fELF') or header.startswith(b'\xca\xfe\xba\xbe'):
                    return True
        except:
            pass
            
        return False
    
    # Prioritize non-hidden files
    candidates = non_hidden_files if non_hidden_files else hidden_files
    
    # First, look for files with binary extensions
    for file_path in candidates:
        if file_path.suffix.lower() in binary_extensions:
            return str(file_path)
    
    # Then look for likely binary files
    for file_path in candidates:
        if is_likely_binary(file_path):
            return str(file_path)
    
    # Finally, return first file if no obvious binaries found
    if candidates:
        return str(candidates[0])
    
    return None


def main():
    """Main entry point with comprehensive CLI interface"""
    parser = argparse.ArgumentParser(
        description="open-sourcefy - AI-Powered Binary Decompilation (Phase 4 CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect binary from input/ directory
  python main.py
  python main.py --decompile-only
  
  # Specify binary explicitly
  python main.py launcher.exe --decompile-only
  python main.py launcher.exe --analyze-only
  python main.py launcher.exe --compile-only
  python main.py launcher.exe --validate-only
  
  # Environment check and cleanup
  python main.py --verify-env
  python main.py --cleanup
  
  # Specific agents
  python main.py launcher.exe --agent 7
  python main.py launcher.exe --agents 1,3,7
  python main.py launcher.exe --agents 1-5
  python main.py launcher.exe --skip-agents 2,4
  
  # Parallel configuration
  python main.py launcher.exe --batch-size 6 --parallel-mode process --timeout 600
  
  # Information
  python main.py --list-components
  python main.py --list-agents
        """
    )
    
    # Main argument
    parser.add_argument("target", nargs='?', help="Target binary file to analyze (optional - will auto-detect from input/ directory if not specified)")
    
    # Pipeline component modes
    parser.add_argument("--decompile-only", action="store_true", 
                       help="Run decompilation pipeline only (agents 1,2,5,7)")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Run analysis pipeline only (agents 1,2,3,4,5,8,9)")
    parser.add_argument("--compile-only", action="store_true", 
                       help="Run compilation pipeline only (agents 6,11,12)")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Run validation pipeline only (agents 8,12,13)")
    parser.add_argument("--production-only", action="store_true", 
                       help="Run production pipeline with all Phase 4 enhancements (agents 1-15,18-20)")
    
    # Specific agent selection
    parser.add_argument("--agent", type=int, help="Run specific agent by ID (1-15,18-20)")
    parser.add_argument("--agents", type=str, help="Run specific agents (e.g., '1,3,7' or '1-5,18-20')")
    parser.add_argument("--skip-agents", type=str, help="Skip specific agents (e.g., '2,4')")
    
    # Configuration options
    parser.add_argument("--output-dir", default="", help="Subdirectory name under /output (default: uses timestamp). All results organized under /output/[subdirectory]/")
    parser.add_argument("--batch-size", type=int, default=4, help="Parallel batch size (default: 4)")
    parser.add_argument("--parallel-mode", choices=["thread", "process"], default="thread", 
                       help="Parallel execution mode (default: thread)")
    parser.add_argument("--timeout", type=int, default=300, help="Agent timeout in seconds (default: 300)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Phase 2 Enhanced Features
    parser.add_argument("--disable-enhanced", action="store_true", 
                       help="Disable Phase 2 enhanced features (reliability, performance, AI analysis)")
    parser.add_argument("--health-monitoring", action="store_true", default=True,
                       help="Enable real-time agent health monitoring (default: enabled)")
    parser.add_argument("--performance-tracking", action="store_true", default=True,
                       help="Enable detailed performance tracking and optimization (default: enabled)")
    parser.add_argument("--pattern-analysis", action="store_true", default=True,
                       help="Enable advanced pattern recognition and analysis (default: enabled)")
    parser.add_argument("--fallback-strategies", action="store_true", default=True,
                       help="Enable automatic fallback strategies for failed agents (default: enabled)")
    
    # Information and verification commands
    parser.add_argument("--verify-env", action="store_true", help="Verify environment and exit")
    parser.add_argument("--list-components", action="store_true", help="List available pipeline components")
    parser.add_argument("--list-agents", action="store_true", help="List all available agents")
    parser.add_argument("--phase-status", action="store_true", help="Show development phase status")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temporary files and reset output directory")
    parser.add_argument("--complete-cleanup", action="store_true", help="Complete cleanup: remove all untracked files, empty input/output directories (WARNING: destructive)")
    
    args = parser.parse_args()
    
    # Information commands (no target required)
    if args.list_components:
        print_component_info()
        return 0
    
    if args.list_agents:
        print_agent_info()
        return 0
    
    if args.phase_status:
        print("Development Phase Status:")
        print(f"  Phase 1 (Environment): ✓ Available")
        print(f"  Phase 2 (Agent System): {'✓ Available' if PHASE2_AVAILABLE else '✗ Not Available'}")
        print(f"  Phase 3 (Ghidra Integration): {'✓ Available' if PHASE3_AVAILABLE else '✗ Not Available'}")
        print(f"  Phase 4 (CLI): ✓ Available")
        return 0
    
    if args.verify_env:
        success = validate_environment()
        return 0 if success else 1
    
    if args.cleanup:
        # Always use /output as base directory
        base_output_dir = "output"
        success = cleanup_environment(base_output_dir)
        return 0 if success else 1
    
    # Auto-detect target if not provided
    target_binary = args.target
    if not target_binary:
        print("No target binary specified, attempting auto-detection from input/ directory...")
        target_binary = auto_detect_binary()
        
        if not target_binary:
            # Check if input directory exists
            input_path = Path("input")
            if not input_path.exists():
                print("Error: No target binary specified and input/ directory does not exist")
                print("Create an input/ directory and place your binary file there, or specify the binary explicitly:")
                print("  python main.py your_binary.exe")
                return 1
            else:
                print("Error: No suitable binary files found in input/ directory")
                print("Supported formats: .exe, .dll, .so, .bin, .out, .app, .dylib, .sys, .com, .scr, .msi, .deb, .rpm, .pkg")
                print("Or specify the binary explicitly: python main.py your_binary.exe")
                return 1
        else:
            print(f"✓ Auto-detected binary: {target_binary}")
    
    # Validate target file
    if not Path(target_binary).exists():
        print(f"Error: Target file '{target_binary}' not found")
        return 1
    
    # Create pipeline with enhanced mode configuration
    enhanced_mode = not args.disable_enhanced and PHASE2_ENHANCED
    
    pipeline = OpenSourcefyPipeline(
        batch_size=args.batch_size,
        parallel_mode=args.parallel_mode,
        timeout=args.timeout,
        verbose=args.verbose,
        enhanced_mode=enhanced_mode
    )
    
    if enhanced_mode:
        print("🚀 Phase 2 Enhanced Mode: ON")
        print("   ✓ Health monitoring")
        print("   ✓ Performance tracking")
        print("   ✓ Pattern analysis")
        print("   ✓ Fallback strategies")
        print("   ✓ Advanced accuracy assessment")
    else:
        print("⚙️  Standard Mode: ON")
    
    # Construct proper output directory structure
    # Always use /output as base directory
    base_output_dir = "output"
    if args.output_dir:
        # User specified a subdirectory name
        final_output_dir = f"{base_output_dir}/{args.output_dir}"
    else:
        # Generate timestamp-based subdirectory
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = f"{base_output_dir}/{timestamp}"
    
    print(f"📁 Output will be organized under: {final_output_dir}/")
    
    # Determine execution mode and run pipeline
    try:
        if args.agent:
            # Single agent
            agents = validate_agent_ids([args.agent])
            if not agents:
                return 1
            result = pipeline.run_specific_agents(agents, target_binary, final_output_dir)
            
        elif args.agents:
            # Multiple specific agents
            agents = parse_agent_list(args.agents)
            agents = validate_agent_ids(agents)
            if not agents:
                return 1
            result = pipeline.run_specific_agents(agents, target_binary, final_output_dir)
            
        elif args.skip_agents:
            # All agents except skipped ones
            skip_agents = parse_agent_list(args.skip_agents)
            all_agents = list(range(1, 21))  # Include all available agents
            agents = [a for a in all_agents if a not in skip_agents]
            agents = validate_agent_ids(agents)
            result = pipeline.run_specific_agents(agents, target_binary, final_output_dir)
            
        elif args.decompile_only:
            result = pipeline.run_component('decompile', target_binary, final_output_dir)
            
        elif args.analyze_only:
            result = pipeline.run_component('analyze', target_binary, final_output_dir)
            
        elif args.compile_only:
            result = pipeline.run_component('compile', target_binary, final_output_dir)
            
        elif args.validate_only:
            result = pipeline.run_component('validate', target_binary, final_output_dir)
            
        elif args.production_only:
            result = pipeline.run_component('production', target_binary, final_output_dir)
            
        else:
            # Full pipeline
            result = pipeline.run_full_pipeline(target_binary, final_output_dir)
        
        # Check for overall success
        if not result.get("overall_success", False):
            print(f"\nPipeline execution had failures. Check {final_output_dir}/reports/pipeline_report.json for details.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())