#!/usr/bin/env python3
"""
Test Phase 2 Integration: Compiler and Build System Analysis

Tests for P2.1, P2.2, P2.3, and P2.4 integration with Matrix pipeline
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from core.phase2_integration import (
        Phase2Integrator,
        enhance_agent2_with_comprehensive_phase2,
        generate_phase2_report
    )
    from core.advanced_compiler_fingerprinting import (
        AdvancedCompilerFingerprinter,
        CompilerType,
        OptimizationLevel
    )
    from core.binary_identical_reconstruction import (
        BinaryIdenticalReconstructor,
        ReconstructionQuality
    )
    from core.build_system_automation import BuildSystemAutomation
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Import error: {e}")


class TestPhase2Integration:
    """Test suite for Phase 2 integration functionality"""
    
    @pytest.fixture
    def sample_binary_path(self):
        """Create a sample binary file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
            # Write minimal PE header
            f.write(b'MZ')  # DOS header
            f.write(b'\x00' * 58)  # DOS header padding
            f.write(b'\x80\x00\x00\x00')  # PE offset
            f.write(b'\x00' * (0x80 - 64))  # Padding to PE offset
            f.write(b'PE\x00\x00')  # PE signature
            # COFF header
            f.write(b'\x64\x86')  # Machine (x64)
            f.write(b'\x06\x00')  # NumberOfSections
            f.write(b'\x00\x00\x00\x00')  # TimeDateStamp
            f.write(b'\x00\x00\x00\x00')  # PointerToSymbolTable
            f.write(b'\x00\x00\x00\x00')  # NumberOfSymbols
            f.write(b'\xf0\x00')  # SizeOfOptionalHeader
            f.write(b'\x22\x00')  # Characteristics
            # Add some binary content
            f.write(b'\x00' * 1000)
            return f.name
    
    @pytest.fixture
    def sample_decompiled_source(self):
        """Create sample decompiled source code"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write('''
#include <stdio.h>
#include <windows.h>

int main(int argc, char* argv[]) {
    printf("Hello, World!\\n");
    return 0;
}

void helper_function(void) {
    // Helper function implementation
}
''')
            return f.name
    
    @pytest.fixture
    def sample_agent2_results(self):
        """Sample Agent 2 (Architect) results"""
        return {
            'compiler_analysis': {
                'toolchain': 'MSVC',
                'confidence': 0.8,
                'evidence': ['MSVC pattern match']
            },
            'optimization_analysis': {
                'level': 'O2',
                'confidence': 0.7,
                'detected_patterns': ['function_inlining']
            },
            'abi_analysis': {
                'calling_convention': 'Microsoft x64',
                'stack_alignment': 16
            },
            'architect_metadata': {
                'agent_id': 2,
                'quality_score': 0.75,
                'execution_time': 1.5
            }
        }
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_phase2_integrator_initialization(self):
        """Test Phase2Integrator initialization"""
        integrator = Phase2Integrator()
        
        assert integrator is not None
        assert integrator.compiler_fingerprinter is not None
        assert integrator.binary_reconstructor is not None
        assert integrator.enable_fingerprinting is True
        assert integrator.enable_reconstruction is True
        assert integrator.enable_build_automation is True
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_compiler_fingerprinting_integration(self, sample_binary_path, sample_agent2_results):
        """Test P2.1: Advanced Compiler Fingerprinting integration"""
        integrator = Phase2Integrator()
        
        # Mock compiler fingerprinting
        with patch.object(integrator.compiler_fingerprinter, 'analyze_compiler_fingerprint') as mock_analyze:
            mock_signature = Mock()
            mock_signature.compiler_type = CompilerType.MSVC
            mock_signature.version = "2019"
            mock_signature.confidence = 0.85
            mock_signature.evidence = ["Rich header match", "MSVC signature"]
            mock_signature.optimization_level = OptimizationLevel.O2
            mock_signature.optimization_confidence = 0.8
            mock_signature.rich_header_info = None
            
            mock_analyze.return_value = mock_signature
            
            # Execute fingerprinting
            result = integrator._execute_compiler_fingerprinting(
                sample_binary_path, sample_agent2_results
            )
            
            # Validate results
            assert result is not None
            assert 'advanced_compiler_analysis' in result
            advanced_analysis = result['advanced_compiler_analysis']
            assert advanced_analysis['compiler_type'] == 'microsoft_visual_cpp'
            assert advanced_analysis['confidence'] >= 0.8
            assert 'evidence' in advanced_analysis
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_binary_reconstruction_integration(self, sample_binary_path, sample_decompiled_source, temp_output_dir):
        """Test P2.2: Binary-Identical Reconstruction integration"""
        integrator = Phase2Integrator()
        
        compiler_analysis = {
            'advanced_compiler_analysis': {
                'compiler_type': 'microsoft_visual_cpp',
                'version': '2019',
                'confidence': 0.85
            }
        }
        
        # Mock binary reconstruction
        with patch.object(integrator.binary_reconstructor, 'reconstruct_binary_identical') as mock_reconstruct:
            mock_result = Mock()
            mock_result.quality = ReconstructionQuality.FUNCTIONALLY_IDENTICAL
            mock_result.confidence = 0.88
            mock_result.original_hash = "abc123"
            mock_result.reconstructed_hash = "def456"
            mock_result.byte_differences = 42
            mock_result.symbol_recovery_rate = 0.85
            mock_result.debug_info_recovery_rate = 0.60
            mock_result.compilation_success = True
            mock_result.build_config = Mock()
            mock_result.build_config.compiler_type = 'microsoft_visual_cpp'
            mock_result.build_config.compiler_version = '2019'
            mock_result.build_config.optimization_level = 'O2'
            mock_result.build_config.target_architecture = 'x64'
            mock_result.metrics = {'compilation_attempts': 2}
            
            mock_reconstruct.return_value = mock_result
            
            # Execute reconstruction
            result = integrator._execute_binary_reconstruction(
                sample_binary_path, sample_decompiled_source, compiler_analysis, temp_output_dir
            )
            
            # Validate results
            assert result is not None
            assert 'binary_identical_reconstruction' in result
            reconstruction = result['binary_identical_reconstruction']
            assert reconstruction['quality'] == 'functional'
            assert reconstruction['confidence'] >= 0.8
            assert reconstruction['compilation_success'] is True
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_build_automation_integration(self, temp_output_dir):
        """Test P2.3: Build System Automation integration"""
        integrator = Phase2Integrator()
        
        compiler_analysis = {
            'advanced_compiler_analysis': {
                'compiler_type': 'microsoft_visual_cpp',
                'version': '2019'
            }
        }
        
        reconstruction_analysis = {
            'binary_identical_reconstruction': {
                'build_configuration': {
                    'compiler_type': 'microsoft_visual_cpp',
                    'optimization_level': 'O2',
                    'target_architecture': 'x64'
                }
            }
        }
        
        # Mock build automation
        with patch('core.phase2_integration.BuildSystemAutomation') as mock_build_class:
            mock_build_automator = Mock()
            mock_build_automator.generate_msbuild_project.return_value = Path(temp_output_dir) / 'project.vcxproj'
            mock_build_automator.generate_cmake_project.return_value = Path(temp_output_dir) / 'CMakeLists.txt'
            mock_build_automator.test_compilation.return_value = {
                'build_system': 'msbuild',
                'success': True,
                'executable_created': True
            }
            
            mock_build_class.return_value = mock_build_automator
            
            # Execute build automation
            result = integrator._execute_build_automation(
                compiler_analysis, reconstruction_analysis, temp_output_dir
            )
            
            # Validate results
            assert result is not None
            assert result['automation_success'] is True
            assert 'build_configuration' in result
            assert 'compilation_test' in result
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_comprehensive_phase2_enhancement(self, sample_binary_path, sample_decompiled_source, sample_agent2_results, temp_output_dir):
        """Test comprehensive Phase 2 enhancement of Agent 2"""
        
        # Mock all components
        with patch('core.phase2_integration.AdvancedCompilerFingerprinter') as mock_fingerprinter_class, \
             patch('core.phase2_integration.BinaryIdenticalReconstructor') as mock_reconstructor_class, \
             patch('core.phase2_integration.BuildSystemAutomation') as mock_build_class:
            
            # Setup fingerprinter mock
            mock_fingerprinter = Mock()
            mock_signature = Mock()
            mock_signature.compiler_type = CompilerType.MSVC
            mock_signature.version = "2019"
            mock_signature.confidence = 0.85
            mock_signature.evidence = ["Rich header match"]
            mock_signature.optimization_level = OptimizationLevel.O2
            mock_signature.optimization_confidence = 0.8
            mock_signature.rich_header_info = None
            mock_fingerprinter.analyze_compiler_fingerprint.return_value = mock_signature
            mock_fingerprinter_class.return_value = mock_fingerprinter
            
            # Setup reconstructor mock
            mock_reconstructor = Mock()
            mock_result = Mock()
            mock_result.quality = ReconstructionQuality.FUNCTIONALLY_IDENTICAL
            mock_result.confidence = 0.88
            mock_result.original_hash = "abc123"
            mock_result.reconstructed_hash = "def456"
            mock_result.byte_differences = 42
            mock_result.symbol_recovery_rate = 0.85
            mock_result.debug_info_recovery_rate = 0.60
            mock_result.compilation_success = True
            mock_result.build_config = Mock()
            mock_result.build_config.compiler_type = 'microsoft_visual_cpp'
            mock_result.build_config.compiler_version = '2019'
            mock_result.build_config.optimization_level = 'O2'
            mock_result.build_config.target_architecture = 'x64'
            mock_result.metrics = {}
            mock_result.error_messages = []
            mock_reconstructor.reconstruct_binary_identical.return_value = mock_result
            mock_reconstructor_class.return_value = mock_reconstructor
            
            # Setup build automation mock
            mock_build_automator = Mock()
            mock_build_automator.generate_msbuild_project.return_value = Path(temp_output_dir) / 'project.vcxproj'
            mock_build_automator.generate_cmake_project.return_value = Path(temp_output_dir) / 'CMakeLists.txt'
            mock_build_automator.test_compilation.return_value = {
                'build_system': 'msbuild',
                'success': True,
                'executable_created': True
            }
            mock_build_class.return_value = mock_build_automator
            
            # Execute comprehensive enhancement
            enhanced_results = enhance_agent2_with_comprehensive_phase2(
                binary_path=sample_binary_path,
                existing_agent2_results=sample_agent2_results,
                decompiled_source_path=sample_decompiled_source,
                output_directory=temp_output_dir
            )
            
            # Validate comprehensive results
            assert enhanced_results is not None
            assert 'phase2_analysis' in enhanced_results
            
            phase2_analysis = enhanced_results['phase2_analysis']
            assert phase2_analysis['enhancement_applied'] is True
            assert phase2_analysis['integration_success'] is True
            assert phase2_analysis['overall_confidence'] > 0.7
            
            # Verify all components were executed
            components = phase2_analysis['components_executed']
            assert components['p2_1_fingerprinting'] is True
            assert components['p2_2_reconstruction'] is True
            assert components['p2_3_build_automation'] is True
            
            # Verify enhanced confidence
            assert enhanced_results['architect_metadata']['quality_score'] >= sample_agent2_results['architect_metadata']['quality_score']
            assert enhanced_results['architect_metadata']['phase2_enhanced'] is True
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_phase2_report_generation(self):
        """Test Phase 2 report generation"""
        
        phase2_results = {
            'compiler_fingerprinting': {
                'advanced_compiler_analysis': {
                    'compiler_type': 'microsoft_visual_cpp',
                    'version': '2019',
                    'confidence': 0.85,
                    'optimization_level': 'O2',
                    'optimization_confidence': 0.8,
                    'evidence': ['Rich header match', 'MSVC signature']
                }
            },
            'binary_reconstruction': {
                'binary_identical_reconstruction': {
                    'quality': 'functional',
                    'confidence': 0.88,
                    'byte_differences': 42,
                    'symbol_recovery_rate': 0.85,
                    'debug_recovery_rate': 0.60,
                    'compilation_success': True,
                    'build_configuration': {
                        'compiler_type': 'microsoft_visual_cpp',
                        'compiler_version': '2019',
                        'optimization_level': 'O2',
                        'target_architecture': 'x64'
                    }
                }
            },
            'build_automation': {
                'enhancement_applied': True,
                'automation_success': True,
                'msbuild_project': '/tmp/project.vcxproj',
                'cmake_project': '/tmp/CMakeLists.txt',
                'compilation_test': {
                    'build_system': 'msbuild',
                    'success': True,
                    'executable_created': True
                }
            },
            'integration_success': True,
            'overall_confidence': 0.84,
            'analysis_timestamp': 1640995200.0,
            'error_messages': []
        }
        
        report = generate_phase2_report(phase2_results)
        
        # Validate report content
        assert report is not None
        assert 'PHASE 2: COMPILER AND BUILD SYSTEM ANALYSIS REPORT' in report
        assert 'P2.1: ADVANCED COMPILER FINGERPRINTING' in report
        assert 'P2.2: BINARY-IDENTICAL RECONSTRUCTION' in report
        assert 'P2.3: BUILD SYSTEM AUTOMATION' in report
        assert 'microsoft_visual_cpp' in report
        assert 'Confidence: 0.850' in report
        assert 'Quality: functional' in report
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_error_handling(self, sample_binary_path, sample_agent2_results):
        """Test error handling in Phase 2 integration"""
        
        # Test with invalid binary path
        with patch('core.phase2_integration.AdvancedCompilerFingerprinter') as mock_class:
            mock_fingerprinter = Mock()
            mock_fingerprinter.analyze_compiler_fingerprint.side_effect = Exception("Analysis failed")
            mock_class.return_value = mock_fingerprinter
            
            enhanced_results = enhance_agent2_with_comprehensive_phase2(
                binary_path="nonexistent_file.exe",
                existing_agent2_results=sample_agent2_results
            )
            
            # Should return original results with error information
            assert enhanced_results is not None
            assert 'phase2_analysis' in enhanced_results
            assert enhanced_results['phase2_analysis']['enhancement_applied'] is False
            assert 'error' in enhanced_results['phase2_analysis']
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_validation_thresholds(self):
        """Test Phase 2 validation thresholds"""
        
        integrator = Phase2Integrator()
        
        # Test with high-quality results
        high_quality_results = {
            'compiler_fingerprinting': {
                'enhancement_applied': True,
                'advanced_compiler_analysis': {
                    'confidence': 0.9
                }
            },
            'binary_reconstruction': {
                'enhancement_applied': True,
                'binary_identical_reconstruction': {
                    'confidence': 0.85
                }
            }
        }
        
        validation = integrator._validate_phase2_integration(high_quality_results)
        assert validation is True
        
        confidence = integrator._calculate_overall_confidence(high_quality_results)
        assert confidence > 0.8
        
        # Test with low-quality results
        low_quality_results = {
            'compiler_fingerprinting': {
                'enhancement_applied': True,
                'advanced_compiler_analysis': {
                    'confidence': 0.3
                }
            }
        }
        
        validation = integrator._validate_phase2_integration(low_quality_results)
        assert validation is False
        
        confidence = integrator._calculate_overall_confidence(low_quality_results)
        assert confidence < 0.5


if __name__ == '__main__':
    # Run tests if imports are available
    if IMPORTS_AVAILABLE:
        pytest.main([__file__, '-v'])
    else:
        print("Phase 2 integration tests skipped - required imports not available")
        print("This is expected if running outside the full Matrix pipeline environment")