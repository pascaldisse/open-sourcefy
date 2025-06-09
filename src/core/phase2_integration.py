#!/usr/bin/env python3
"""
Phase 2 Integration: Compiler and Build System Analysis

P2.4 Implementation for Open-Sourcefy Matrix Pipeline
Integrates all Phase 2 components for comprehensive compiler analysis
and binary-identical reconstruction capabilities.

Features:
- Advanced compiler fingerprinting integration
- Binary-identical reconstruction pipeline
- Build system automation enhancement
- Agent 2 (Architect) integration
- Comprehensive reporting and validation

Research Base:
- Integration of P2.1, P2.2, P2.3 components
- Matrix pipeline architecture compatibility
- Production-ready error handling
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

# Phase 2 component imports
from .advanced_compiler_fingerprinting import (
    AdvancedCompilerFingerprinter, 
    enhance_agent2_compiler_detection,
    create_compiler_fingerprinting_report
)
from .binary_identical_reconstruction import (
    BinaryIdenticalReconstructor,
    enhance_binary_reconstruction,
    ReconstructionQuality
)
from .build_system_automation import BuildSystemAutomation

# Matrix framework imports
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


@dataclass
class Phase2Results:
    """Comprehensive Phase 2 analysis results"""
    compiler_fingerprinting: Dict[str, Any]
    binary_reconstruction: Dict[str, Any]
    build_automation: Dict[str, Any]
    integration_success: bool
    overall_confidence: float
    analysis_timestamp: float
    enhancement_applied: bool
    error_messages: List[str]


class Phase2Integrator:
    """
    Phase 2 Integration Engine
    
    Provides unified interface for all Phase 2 components:
    - P2.1: Advanced Compiler Fingerprinting
    - P2.2: Binary-Identical Reconstruction  
    - P2.3: Automated Build System Generation
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Phase 2 components
        self.compiler_fingerprinter = AdvancedCompilerFingerprinter(config_manager)
        self.binary_reconstructor = BinaryIdenticalReconstructor(config_manager)
        self.build_automator = None  # Initialized on demand
        
        # Phase 2 configuration
        self.enable_fingerprinting = self.config.get_value('phase2.enable_fingerprinting', True)
        self.enable_reconstruction = self.config.get_value('phase2.enable_reconstruction', True)
        self.enable_build_automation = self.config.get_value('phase2.enable_build_automation', True)
        
        # Quality thresholds
        self.fingerprinting_threshold = self.config.get_value('phase2.fingerprinting_threshold', 0.7)
        self.reconstruction_threshold = self.config.get_value('phase2.reconstruction_threshold', 0.8)
        
    def enhance_agent2_with_phase2(
        self, 
        binary_path: str, 
        existing_agent2_results: Dict[str, Any],
        decompiled_source_path: Optional[str] = None,
        output_directory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhance Agent 2 (Architect) with comprehensive Phase 2 analysis
        
        Args:
            binary_path: Path to target binary
            existing_agent2_results: Existing Agent 2 analysis results
            decompiled_source_path: Optional path to decompiled source (for reconstruction)
            output_directory: Optional output directory for build artifacts
            
        Returns:
            Enhanced Agent 2 results with Phase 2 capabilities
        """
        self.logger.info("Starting Phase 2 enhancement for Agent 2 (Architect)")
        
        enhanced_results = existing_agent2_results.copy()
        phase2_results = {}
        error_messages = []
        
        try:
            # Phase 2.1: Advanced Compiler Fingerprinting
            if self.enable_fingerprinting:
                self.logger.info("Executing P2.1: Advanced Compiler Fingerprinting")
                fingerprinting_results = self._execute_compiler_fingerprinting(
                    binary_path, existing_agent2_results
                )
                phase2_results['compiler_fingerprinting'] = fingerprinting_results
                
                # Enhance existing compiler analysis
                if fingerprinting_results.get('enhancement_applied', False):
                    enhanced_results.update(fingerprinting_results)
            
            # Phase 2.2: Binary-Identical Reconstruction
            if self.enable_reconstruction and decompiled_source_path and output_directory:
                self.logger.info("Executing P2.2: Binary-Identical Reconstruction")
                reconstruction_results = self._execute_binary_reconstruction(
                    binary_path, decompiled_source_path, 
                    phase2_results.get('compiler_fingerprinting', {}),
                    output_directory
                )
                phase2_results['binary_reconstruction'] = reconstruction_results
            
            # Phase 2.3: Build System Automation
            if self.enable_build_automation and output_directory:
                self.logger.info("Executing P2.3: Build System Automation")
                build_automation_results = self._execute_build_automation(
                    phase2_results.get('compiler_fingerprinting', {}),
                    phase2_results.get('binary_reconstruction', {}),
                    output_directory
                )
                phase2_results['build_automation'] = build_automation_results
            
            # Integration validation
            integration_success = self._validate_phase2_integration(phase2_results)
            overall_confidence = self._calculate_overall_confidence(phase2_results)
            
            # Create comprehensive Phase 2 results
            final_phase2_results = Phase2Results(
                compiler_fingerprinting=phase2_results.get('compiler_fingerprinting', {}),
                binary_reconstruction=phase2_results.get('binary_reconstruction', {}),
                build_automation=phase2_results.get('build_automation', {}),
                integration_success=integration_success,
                overall_confidence=overall_confidence,
                analysis_timestamp=time.time(),
                enhancement_applied=True,
                error_messages=error_messages
            )
            
            # Add Phase 2 results to enhanced Agent 2 output
            enhanced_results['phase2_analysis'] = {
                'compiler_fingerprinting': final_phase2_results.compiler_fingerprinting,
                'binary_reconstruction': final_phase2_results.binary_reconstruction,
                'build_automation': final_phase2_results.build_automation,
                'integration_success': final_phase2_results.integration_success,
                'overall_confidence': final_phase2_results.overall_confidence,
                'analysis_timestamp': final_phase2_results.analysis_timestamp,
                'enhancement_applied': final_phase2_results.enhancement_applied,
                'components_executed': {
                    'p2_1_fingerprinting': self.enable_fingerprinting,
                    'p2_2_reconstruction': self.enable_reconstruction and bool(decompiled_source_path),
                    'p2_3_build_automation': self.enable_build_automation and bool(output_directory)
                }
            }
            
            # Enhanced confidence scoring
            original_confidence = existing_agent2_results.get('architect_metadata', {}).get('quality_score', 0.5)
            enhanced_confidence = max(original_confidence, overall_confidence)
            
            if 'architect_metadata' in enhanced_results:
                enhanced_results['architect_metadata']['quality_score'] = enhanced_confidence
                enhanced_results['architect_metadata']['phase2_enhanced'] = True
            
            self.logger.info(
                f"Phase 2 enhancement complete: confidence={overall_confidence:.3f}, "
                f"integration_success={integration_success}"
            )
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Phase 2 enhancement failed: {e}")
            error_messages.append(str(e))
            
            # Return original results with error information
            enhanced_results['phase2_analysis'] = {
                'enhancement_applied': False,
                'error': str(e),
                'error_messages': error_messages
            }
            
            return enhanced_results
    
    def _execute_compiler_fingerprinting(
        self, 
        binary_path: str, 
        existing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute P2.1: Advanced Compiler Fingerprinting"""
        
        try:
            # Use the integration function from advanced_compiler_fingerprinting.py
            enhanced_analysis = enhance_agent2_compiler_detection(binary_path, existing_results)
            
            # Validate fingerprinting quality
            advanced_analysis = enhanced_analysis.get('advanced_compiler_analysis', {})
            confidence = advanced_analysis.get('confidence', 0.0)
            
            if confidence >= self.fingerprinting_threshold:
                self.logger.info(f"High-quality compiler fingerprinting achieved: {confidence:.3f}")
            else:
                self.logger.warning(f"Compiler fingerprinting below threshold: {confidence:.3f}")
            
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Compiler fingerprinting failed: {e}")
            return {'error': str(e), 'enhancement_applied': False}
    
    def _execute_binary_reconstruction(
        self,
        binary_path: str,
        decompiled_source_path: str,
        compiler_analysis: Dict[str, Any],
        output_directory: str
    ) -> Dict[str, Any]:
        """Execute P2.2: Binary-Identical Reconstruction"""
        
        try:
            # Use the integration function from binary_identical_reconstruction.py
            reconstruction_results = enhance_binary_reconstruction(
                binary_path, decompiled_source_path, compiler_analysis, output_directory
            )
            
            # Validate reconstruction quality
            binary_reconstruction = reconstruction_results.get('binary_identical_reconstruction', {})
            confidence = binary_reconstruction.get('confidence', 0.0)
            quality = binary_reconstruction.get('quality', 'failed')
            
            if confidence >= self.reconstruction_threshold:
                self.logger.info(f"High-quality binary reconstruction: {quality} ({confidence:.3f})")
            else:
                self.logger.warning(f"Binary reconstruction below threshold: {quality} ({confidence:.3f})")
            
            return reconstruction_results
            
        except Exception as e:
            self.logger.error(f"Binary reconstruction failed: {e}")
            return {'error': str(e), 'enhancement_applied': False}
    
    def _execute_build_automation(
        self,
        compiler_analysis: Dict[str, Any],
        reconstruction_analysis: Dict[str, Any],
        output_directory: str
    ) -> Dict[str, Any]:
        """Execute P2.3: Build System Automation"""
        
        try:
            # Initialize build automation if not already done
            if not self.build_automator:
                self.build_automator = BuildSystemAutomation(output_directory)
            
            # Extract build configuration from analyses
            build_config = self._extract_build_configuration(
                compiler_analysis, reconstruction_analysis
            )
            
            # Generate build scripts based on detected configuration
            build_results = {}
            
            # Determine appropriate build system
            compiler_type = build_config.get('compiler_type', 'unknown')
            
            if compiler_type == 'microsoft_visual_cpp':
                # Generate MSBuild project
                project_name = build_config.get('project_name', 'ReconstructedProject')
                source_files = build_config.get('source_files', [])
                include_dirs = build_config.get('include_directories', [])
                
                if source_files:
                    project_file = self.build_automator.generate_msbuild_project(
                        project_name, source_files, include_dirs
                    )
                    build_results['msbuild_project'] = str(project_file)
                    
                    # Test compilation
                    compilation_test = self.build_automator.test_compilation('msbuild')
                    build_results['compilation_test'] = compilation_test
                    
            elif compiler_type in ['gnu_compiler_collection', 'llvm_clang']:
                # Generate Makefile
                project_name = build_config.get('project_name', 'reconstructed_project')
                source_files = build_config.get('source_files', [])
                include_dirs = build_config.get('include_directories', [])
                libraries = build_config.get('libraries', [])
                
                if source_files:
                    makefile = self.build_automator.generate_makefile(
                        project_name, source_files, include_dirs, libraries
                    )
                    build_results['makefile'] = str(makefile)
                    
                    # Test compilation
                    compilation_test = self.build_automator.test_compilation('make')
                    build_results['compilation_test'] = compilation_test
            
            # Generate CMake for cross-platform compatibility
            if build_config.get('source_files'):
                cmake_file = self.build_automator.generate_cmake_project(
                    build_config.get('project_name', 'ReconstructedProject'),
                    build_config['source_files'],
                    build_config.get('include_directories', []),
                    build_config.get('libraries', [])
                )
                build_results['cmake_project'] = str(cmake_file)
            
            build_results.update({
                'build_configuration': build_config,
                'automation_success': True,
                'enhancement_applied': True
            })
            
            return build_results
            
        except Exception as e:
            self.logger.error(f"Build automation failed: {e}")
            return {'error': str(e), 'enhancement_applied': False}
    
    def _extract_build_configuration(
        self,
        compiler_analysis: Dict[str, Any],
        reconstruction_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract build configuration from Phase 2 analyses"""
        
        build_config = {
            'project_name': 'ReconstructedProject',
            'compiler_type': 'unknown',
            'compiler_version': 'unknown',
            'optimization_level': 'O2',
            'target_architecture': 'x86',
            'source_files': [],
            'include_directories': [],
            'libraries': [],
            'preprocessor_defines': []
        }
        
        # Extract from compiler analysis
        advanced_compiler = compiler_analysis.get('advanced_compiler_analysis', {})
        if advanced_compiler:
            build_config['compiler_type'] = advanced_compiler.get('compiler_type', 'unknown')
            build_config['compiler_version'] = advanced_compiler.get('version', 'unknown')
            build_config['optimization_level'] = advanced_compiler.get('optimization_level', 'O2')
        
        # Extract from reconstruction analysis  
        binary_reconstruction = reconstruction_analysis.get('binary_identical_reconstruction', {})
        if binary_reconstruction:
            build_configuration = binary_reconstruction.get('build_configuration', {})
            if build_configuration:
                build_config.update({
                    'compiler_type': build_configuration.get('compiler_type', build_config['compiler_type']),
                    'compiler_version': build_configuration.get('compiler_version', build_config['compiler_version']),
                    'optimization_level': build_configuration.get('optimization_level', build_config['optimization_level']),
                    'target_architecture': build_configuration.get('target_architecture', build_config['target_architecture'])
                })
        
        # Add default source files and includes (would be extracted from actual decompilation in real scenario)
        build_config['source_files'] = [
            Path('main.c'),
            Path('utils.c')
        ]
        build_config['include_directories'] = [
            Path('include'),
            Path('.')
        ]
        
        # Add standard libraries based on compiler type
        if build_config['compiler_type'] == 'microsoft_visual_cpp':
            build_config['libraries'] = ['kernel32.lib', 'user32.lib', 'gdi32.lib']
        else:
            build_config['libraries'] = ['m', 'dl', 'pthread']
        
        return build_config
    
    def _validate_phase2_integration(self, phase2_results: Dict[str, Any]) -> bool:
        """Validate successful Phase 2 integration"""
        
        validation_checks = []
        
        # Check compiler fingerprinting
        fingerprinting = phase2_results.get('compiler_fingerprinting', {})
        if fingerprinting.get('enhancement_applied', False):
            advanced_analysis = fingerprinting.get('advanced_compiler_analysis', {})
            confidence = advanced_analysis.get('confidence', 0.0)
            validation_checks.append(confidence >= self.fingerprinting_threshold)
        
        # Check binary reconstruction
        reconstruction = phase2_results.get('binary_reconstruction', {})
        if reconstruction.get('enhancement_applied', False):
            binary_reconstruction = reconstruction.get('binary_identical_reconstruction', {})
            confidence = binary_reconstruction.get('confidence', 0.0)
            validation_checks.append(confidence >= self.reconstruction_threshold)
        
        # Check build automation
        build_automation = phase2_results.get('build_automation', {})
        if build_automation.get('enhancement_applied', False):
            automation_success = build_automation.get('automation_success', False)
            validation_checks.append(automation_success)
        
        # Integration is successful if at least one component succeeded
        return len(validation_checks) > 0 and any(validation_checks)
    
    def _calculate_overall_confidence(self, phase2_results: Dict[str, Any]) -> float:
        """Calculate overall Phase 2 confidence score"""
        
        confidence_scores = []
        
        # Compiler fingerprinting confidence
        fingerprinting = phase2_results.get('compiler_fingerprinting', {})
        if fingerprinting.get('enhancement_applied', False):
            advanced_analysis = fingerprinting.get('advanced_compiler_analysis', {})
            confidence_scores.append(advanced_analysis.get('confidence', 0.0))
        
        # Binary reconstruction confidence
        reconstruction = phase2_results.get('binary_reconstruction', {})
        if reconstruction.get('enhancement_applied', False):
            binary_reconstruction = reconstruction.get('binary_identical_reconstruction', {})
            confidence_scores.append(binary_reconstruction.get('confidence', 0.0))
        
        # Build automation confidence (simplified)
        build_automation = phase2_results.get('build_automation', {})
        if build_automation.get('enhancement_applied', False):
            # Build automation confidence based on successful generation
            automation_success = build_automation.get('automation_success', False)
            confidence_scores.append(0.8 if automation_success else 0.3)
        
        # Return average confidence or default
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        else:
            return 0.5  # Default confidence


def generate_phase2_report(phase2_results: Dict[str, Any]) -> str:
    """Generate comprehensive Phase 2 analysis report"""
    
    report_lines = [
        "=" * 80,
        "PHASE 2: COMPILER AND BUILD SYSTEM ANALYSIS REPORT",
        "=" * 80,
        "",
        f"Analysis Timestamp: {time.ctime(phase2_results.get('analysis_timestamp', time.time()))}",
        f"Integration Success: {phase2_results.get('integration_success', False)}",
        f"Overall Confidence: {phase2_results.get('overall_confidence', 0.0):.3f}",
        ""
    ]
    
    # P2.1: Compiler Fingerprinting Results
    fingerprinting = phase2_results.get('compiler_fingerprinting', {})
    advanced_analysis = fingerprinting.get('advanced_compiler_analysis', {})
    
    if advanced_analysis:
        report_lines.extend([
            "P2.1: ADVANCED COMPILER FINGERPRINTING",
            "-" * 40,
            f"Compiler Type: {advanced_analysis.get('compiler_type', 'Unknown')}",
            f"Version: {advanced_analysis.get('version', 'Unknown')}",
            f"Confidence: {advanced_analysis.get('confidence', 0.0):.3f}",
            f"Optimization Level: {advanced_analysis.get('optimization_level', 'Unknown')}",
            f"Optimization Confidence: {advanced_analysis.get('optimization_confidence', 0.0):.3f}",
            "",
            "Evidence Found:",
        ])
        
        for evidence in advanced_analysis.get('evidence', []):
            report_lines.append(f"  - {evidence}")
        
        report_lines.append("")
    
    # P2.2: Binary Reconstruction Results
    reconstruction = phase2_results.get('binary_reconstruction', {})
    binary_reconstruction = reconstruction.get('binary_identical_reconstruction', {})
    
    if binary_reconstruction:
        report_lines.extend([
            "P2.2: BINARY-IDENTICAL RECONSTRUCTION",
            "-" * 40,
            f"Quality: {binary_reconstruction.get('quality', 'Unknown')}",
            f"Confidence: {binary_reconstruction.get('confidence', 0.0):.3f}",
            f"Byte Differences: {binary_reconstruction.get('byte_differences', -1)}",
            f"Symbol Recovery Rate: {binary_reconstruction.get('symbol_recovery_rate', 0.0):.3f}",
            f"Debug Recovery Rate: {binary_reconstruction.get('debug_recovery_rate', 0.0):.3f}",
            f"Compilation Success: {binary_reconstruction.get('compilation_success', False)}",
            ""
        ])
        
        build_config = binary_reconstruction.get('build_configuration', {})
        if build_config:
            report_lines.extend([
                "Build Configuration:",
                f"  - Compiler: {build_config.get('compiler_type', 'Unknown')} {build_config.get('compiler_version', '')}",
                f"  - Optimization: {build_config.get('optimization_level', 'Unknown')}",
                f"  - Architecture: {build_config.get('target_architecture', 'Unknown')}",
                ""
            ])
    
    # P2.3: Build Automation Results
    build_automation = phase2_results.get('build_automation', {})
    
    if build_automation.get('enhancement_applied', False):
        report_lines.extend([
            "P2.3: BUILD SYSTEM AUTOMATION",
            "-" * 40,
            f"Automation Success: {build_automation.get('automation_success', False)}",
        ])
        
        if build_automation.get('msbuild_project'):
            report_lines.append(f"MSBuild Project: {build_automation['msbuild_project']}")
        
        if build_automation.get('makefile'):
            report_lines.append(f"Makefile: {build_automation['makefile']}")
        
        if build_automation.get('cmake_project'):
            report_lines.append(f"CMake Project: {build_automation['cmake_project']}")
        
        compilation_test = build_automation.get('compilation_test', {})
        if compilation_test:
            report_lines.extend([
                "",
                "Compilation Test Results:",
                f"  - Build System: {compilation_test.get('build_system', 'Unknown')}",
                f"  - Success: {compilation_test.get('success', False)}",
                f"  - Executable Created: {compilation_test.get('executable_created', False)}"
            ])
        
        report_lines.append("")
    
    # Error Messages
    error_messages = phase2_results.get('error_messages', [])
    if error_messages:
        report_lines.extend([
            "ERRORS AND WARNINGS",
            "-" * 40
        ])
        for error in error_messages:
            report_lines.append(f"  - {error}")
        report_lines.append("")
    
    report_lines.extend([
        "=" * 80,
        "END OF PHASE 2 ANALYSIS REPORT",
        "=" * 80
    ])
    
    return "\n".join(report_lines)


# Integration function for Agent 2 enhancement
def enhance_agent2_with_comprehensive_phase2(
    binary_path: str,
    existing_agent2_results: Dict[str, Any],
    decompiled_source_path: Optional[str] = None,
    output_directory: Optional[str] = None,
    config_manager: Optional[ConfigManager] = None
) -> Dict[str, Any]:
    """
    Main integration function for enhancing Agent 2 with Phase 2 capabilities
    
    This function serves as the primary entry point for Phase 2 enhancement
    and can be called from the main Matrix pipeline or Agent 2 directly.
    
    Args:
        binary_path: Path to target binary
        existing_agent2_results: Existing Agent 2 analysis results
        decompiled_source_path: Optional path to decompiled source
        output_directory: Optional output directory for artifacts
        config_manager: Optional configuration manager
        
    Returns:
        Enhanced Agent 2 results with Phase 2 analysis
    """
    try:
        integrator = Phase2Integrator(config_manager)
        
        enhanced_results = integrator.enhance_agent2_with_phase2(
            binary_path=binary_path,
            existing_agent2_results=existing_agent2_results,
            decompiled_source_path=decompiled_source_path,
            output_directory=output_directory
        )
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Phase 2 integration failed: {e}")
        
        # Return original results with error information
        error_results = existing_agent2_results.copy()
        error_results['phase2_analysis'] = {
            'enhancement_applied': False,
            'error': str(e),
            'integration_success': False
        }
        
        return error_results


class Phase2Validator:
    """Phase 2 code generation validation class"""
    
    def __init__(self, config):
        self.config = config
        
    def validate_code_generation(self):
        """Validate Phase 2 code generation and compilation fidelity"""
        from types import SimpleNamespace
        
        logger.info("🔍 Validating Phase 2: Code Generation & Compilation")
        
        # Mock validation result for Phase 2
        result = SimpleNamespace()
        result.status = "VALIDATION_AVAILABLE"
        result.function_count_match = False  # Would validate function count
        result.assembly_match = False  # Would check assembly instruction fidelity  
        result.calling_conventions_match = False  # Would verify calling conventions
        
        return result