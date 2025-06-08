"""
Phase 1 Integration Module for Advanced Binary Deobfuscation Pipeline

Integrates Phase 1 deobfuscation components with the existing Matrix pipeline:
- Entropy analysis integration
- CFG reconstruction integration
- Packer detection integration
- Obfuscation detection integration

Enhances Agent 1 (Sentinel) with NSA-level binary analysis capabilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .entropy_analyzer import EntropyAnalyzer, PackedSectionDetector
from .cfg_reconstructor import CFGReconstructor, AdvancedControlFlowAnalyzer
from .packer_detector import PackerDetector, UnpackingEngine
from .obfuscation_detector import ObfuscationDetector, AntiAnalysisDetector


@dataclass
class Phase1AnalysisResult:
    """Comprehensive Phase 1 analysis result."""
    entropy_analysis: Dict[str, Any]
    cfg_analysis: Dict[str, Any]
    packer_analysis: Dict[str, Any]
    obfuscation_analysis: Dict[str, Any]
    integration_metadata: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float


class Phase1Integrator:
    """
    Main integration class for Phase 1 deobfuscation capabilities.
    
    Provides a unified interface for all Phase 1 components
    and integrates them with the existing Matrix pipeline.
    """
    
    def __init__(self, config_manager=None):
        """Initialize Phase 1 integrator with all components."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize all Phase 1 components
        self.entropy_analyzer = EntropyAnalyzer(config_manager)
        self.packed_section_detector = PackedSectionDetector(self.entropy_analyzer)
        self.cfg_reconstructor = CFGReconstructor(config_manager)
        self.packer_detector = PackerDetector(config_manager)
        self.unpacking_engine = UnpackingEngine(config_manager)
        self.obfuscation_detector = ObfuscationDetector(config_manager)
        self.anti_analysis_detector = AntiAnalysisDetector(config_manager)
        
        self.logger.info("Phase 1 deobfuscation integration initialized")
    
    def perform_comprehensive_analysis(self, binary_path: Path) -> Phase1AnalysisResult:
        """
        Perform comprehensive Phase 1 analysis on binary.
        
        Args:
            binary_path: Path to binary file to analyze
            
        Returns:
            Phase1AnalysisResult with complete analysis
        """
        try:
            self.logger.info(f"Starting comprehensive Phase 1 analysis for {binary_path}")
            
            # Phase 1.1: Entropy Analysis
            self.logger.debug("Performing entropy analysis")
            entropy_results = self._perform_entropy_analysis(binary_path)
            
            # Phase 1.2: Packer Detection
            self.logger.debug("Performing packer detection")
            packer_results = self._perform_packer_analysis(binary_path)
            
            # Phase 1.3: Obfuscation Detection
            self.logger.debug("Performing obfuscation detection")
            obfuscation_results = self._perform_obfuscation_analysis(binary_path)
            
            # Phase 1.4: CFG Reconstruction (conditional on previous results)
            self.logger.debug("Performing CFG reconstruction")
            cfg_results = self._perform_cfg_analysis(binary_path, packer_results, obfuscation_results)
            
            # Phase 1.5: Integration and Recommendations
            self.logger.debug("Generating integration results")
            recommendations = self._generate_recommendations(
                entropy_results, packer_results, obfuscation_results, cfg_results
            )
            
            confidence_score = self._calculate_overall_confidence(
                entropy_results, packer_results, obfuscation_results, cfg_results
            )
            
            # Create comprehensive result
            result = Phase1AnalysisResult(
                entropy_analysis=entropy_results,
                cfg_analysis=cfg_results,
                packer_analysis=packer_results,
                obfuscation_analysis=obfuscation_results,
                integration_metadata={
                    'analysis_timestamp': self._get_timestamp(),
                    'binary_path': str(binary_path),
                    'binary_size': binary_path.stat().st_size if binary_path.exists() else 0,
                    'phase1_version': '1.0.0'
                },
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
            self.logger.info(f"Phase 1 analysis complete. Confidence: {confidence_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive Phase 1 analysis: {e}")
            return self._create_error_result(str(e))
    
    def _perform_entropy_analysis(self, binary_path: Path) -> Dict[str, Any]:
        """Perform entropy-based analysis."""
        try:
            # Read binary data
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            # Overall entropy analysis
            overall_entropy = self.entropy_analyzer.analyze_binary_data(binary_data)
            
            # Section-wise analysis (if PE file)
            section_analysis = []
            try:
                section_analysis = self.packed_section_detector.analyze_pe_file(binary_path)
            except Exception as e:
                self.logger.debug(f"PE section analysis failed: {e}")
            
            # Packer signature detection
            packer_signatures = {}
            try:
                packer_signatures = self.packed_section_detector.detect_packer_signatures(binary_path)
            except Exception as e:
                self.logger.debug(f"Packer signature detection failed: {e}")
            
            return {
                'overall_entropy': overall_entropy.__dict__,
                'section_analysis': [s.__dict__ for s in section_analysis],
                'packer_signatures': packer_signatures,
                'analysis_status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in entropy analysis: {e}")
            return {
                'analysis_status': 'error',
                'error': str(e)
            }
    
    def _perform_packer_analysis(self, binary_path: Path) -> Dict[str, Any]:
        """Perform packer detection and analysis."""
        try:
            # Comprehensive packer detection
            detection_result = self.packer_detector.detect_packer(binary_path)
            
            # Attempt unpacking if packer detected
            unpacking_result = None
            if detection_result.packer_detected and detection_result.unpacking_difficulty in ['easy', 'medium']:
                try:
                    unpacking_result = self.unpacking_engine.unpack_binary(binary_path)
                except Exception as e:
                    self.logger.debug(f"Unpacking attempt failed: {e}")
                    unpacking_result = {'success': False, 'error': str(e)}
            
            return {
                'detection_result': detection_result.__dict__,
                'unpacking_result': unpacking_result,
                'analysis_status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in packer analysis: {e}")
            return {
                'analysis_status': 'error',
                'error': str(e)
            }
    
    def _perform_obfuscation_analysis(self, binary_path: Path) -> Dict[str, Any]:
        """Perform obfuscation detection and analysis."""
        try:
            # Comprehensive obfuscation detection
            obfuscation_result = self.obfuscation_detector.analyze_obfuscation(binary_path)
            
            # Anti-analysis detection
            anti_analysis_result = self.anti_analysis_detector.detect_anti_analysis(binary_path)
            
            return {
                'obfuscation_result': obfuscation_result.__dict__,
                'anti_analysis_result': anti_analysis_result,
                'analysis_status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in obfuscation analysis: {e}")
            return {
                'analysis_status': 'error',
                'error': str(e)
            }
    
    def _perform_cfg_analysis(self, binary_path: Path, packer_results: Dict, obfuscation_results: Dict) -> Dict[str, Any]:
        """Perform CFG reconstruction with context from previous analyses."""
        try:
            # Determine if CFG analysis should be performed
            should_analyze_cfg = self._should_perform_cfg_analysis(packer_results, obfuscation_results)
            
            if not should_analyze_cfg:
                return {
                    'cfg_result': None,
                    'analysis_status': 'skipped',
                    'reason': 'Binary too obfuscated or packed for meaningful CFG analysis'
                }
            
            # Perform CFG reconstruction
            cfg_result = self.cfg_reconstructor.reconstruct_cfg(binary_path)
            
            return {
                'cfg_result': {
                    'basic_blocks_count': len(cfg_result.basic_blocks),
                    'indirect_jumps_count': len(cfg_result.indirect_jumps),
                    'switch_statements_count': len(cfg_result.switch_statements),
                    'exception_handlers_count': len(cfg_result.exception_handlers),
                    'self_modifying_regions_count': len(cfg_result.self_modifying_regions),
                    'analysis_quality': cfg_result.analysis_quality,
                    'coverage_percentage': cfg_result.coverage_percentage,
                    'metadata': cfg_result.metadata
                },
                'analysis_status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Error in CFG analysis: {e}")
            return {
                'analysis_status': 'error',
                'error': str(e)
            }
    
    def _should_perform_cfg_analysis(self, packer_results: Dict, obfuscation_results: Dict) -> bool:
        """Determine if CFG analysis should be performed based on previous results."""
        try:
            # Skip if heavily packed
            detection_result = packer_results.get('detection_result', {})
            if (detection_result.get('packer_detected', False) and
                detection_result.get('unpacking_difficulty') in ['hard', 'very_hard']):
                return False
            
            # Skip if heavily obfuscated
            obfuscation_result = obfuscation_results.get('obfuscation_result', {})
            if obfuscation_result.get('obfuscation_level') in ['heavy', 'extreme']:
                return False
            
            # Check for VM obfuscation
            indicators = obfuscation_result.get('indicators', [])
            for indicator in indicators:
                if (isinstance(indicator, dict) and 
                    indicator.get('obfuscation_type') == 'virtual_machine' and
                    indicator.get('severity') == 'critical'):
                    return False
            
            return True
            
        except Exception:
            # If we can't determine, default to performing analysis
            return True
    
    def _generate_recommendations(self, entropy_results: Dict, packer_results: Dict,
                                obfuscation_results: Dict, cfg_results: Dict) -> List[str]:
        """Generate comprehensive recommendations based on all analyses."""
        recommendations = []
        
        try:
            # Entropy-based recommendations
            entropy_analysis = entropy_results.get('overall_entropy', {})
            if entropy_analysis.get('classification') == 'packed':
                recommendations.append("High entropy detected - binary likely packed or encrypted")
                recommendations.append("Consider unpacking before further analysis")
            
            # Packer-based recommendations
            detection_result = packer_results.get('detection_result', {})
            if detection_result.get('packer_detected', False):
                packer_name = detection_result.get('packer_name', 'Unknown')
                difficulty = detection_result.get('unpacking_difficulty', 'unknown')
                
                recommendations.append(f"{packer_name} packer detected (difficulty: {difficulty})")
                
                recommended_tools = detection_result.get('recommended_tools', [])
                if recommended_tools:
                    recommendations.append(f"Recommended tools: {', '.join(recommended_tools)}")
                
                # Check if unpacking was attempted
                unpacking_result = packer_results.get('unpacking_result')
                if unpacking_result and unpacking_result.get('success'):
                    recommendations.append("Automatic unpacking successful - use unpacked binary for analysis")
                elif unpacking_result:
                    recommendations.append("Automatic unpacking failed - manual unpacking required")
            
            # Obfuscation-based recommendations
            obfuscation_result = obfuscation_results.get('obfuscation_result', {})
            if obfuscation_result.get('obfuscated', False):
                obfuscation_level = obfuscation_result.get('obfuscation_level', 'unknown')
                difficulty = obfuscation_result.get('estimated_difficulty', 'unknown')
                
                recommendations.append(f"Obfuscation detected (level: {obfuscation_level}, difficulty: {difficulty})")
                
                deobf_recommendations = obfuscation_result.get('deobfuscation_recommendations', [])
                recommendations.extend(deobf_recommendations)
                
                # Anti-analysis recommendations
                anti_analysis = obfuscation_results.get('anti_analysis_result', {})
                if anti_analysis.get('anti_analysis_detected', False):
                    bypass_recommendations = anti_analysis.get('bypass_recommendations', [])
                    recommendations.extend(bypass_recommendations)
            
            # CFG-based recommendations
            cfg_result = cfg_results.get('cfg_result')
            if cfg_result:
                quality = cfg_result.get('analysis_quality', 0)
                coverage = cfg_result.get('coverage_percentage', 0)
                
                if quality < 0.5:
                    recommendations.append("Low CFG reconstruction quality - consider advanced analysis techniques")
                
                if coverage < 50:
                    recommendations.append("Low code coverage in CFG analysis - binary may have dynamic code generation")
                
                indirect_jumps = cfg_result.get('indirect_jumps_count', 0)
                if indirect_jumps > 20:
                    recommendations.append("High number of indirect jumps detected - possible obfuscation or virtualization")
            elif cfg_results.get('analysis_status') == 'skipped':
                recommendations.append("CFG analysis skipped due to heavy obfuscation/packing")
                recommendations.append("Recommend deobfuscation/unpacking before CFG analysis")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Binary appears to be clean - proceed with standard analysis")
            
            # Add prioritized recommendations
            recommendations.insert(0, "=== PRIORITY RECOMMENDATIONS ===")
            
            if detection_result.get('packer_detected', False):
                recommendations.insert(1, "1. Address packing first before other analysis")
            elif obfuscation_result.get('obfuscated', False):
                recommendations.insert(1, "1. Address obfuscation before detailed analysis")
            else:
                recommendations.insert(1, "1. Binary ready for detailed analysis")
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Error generating recommendations - manual analysis required"]
        
        return recommendations
    
    def _calculate_overall_confidence(self, entropy_results: Dict, packer_results: Dict,
                                    obfuscation_results: Dict, cfg_results: Dict) -> float:
        """Calculate overall confidence score for Phase 1 analysis."""
        try:
            confidence_scores = []
            
            # Entropy analysis confidence
            entropy_analysis = entropy_results.get('overall_entropy', {})
            if entropy_analysis.get('confidence'):
                confidence_scores.append(entropy_analysis['confidence'])
            
            # Packer detection confidence
            detection_result = packer_results.get('detection_result', {})
            if detection_result.get('confidence'):
                confidence_scores.append(detection_result['confidence'])
            
            # Obfuscation detection confidence
            obfuscation_result = obfuscation_results.get('obfuscation_result', {})
            indicators = obfuscation_result.get('indicators', [])
            if indicators:
                obf_confidence = sum(i.get('confidence', 0) for i in indicators if isinstance(i, dict)) / len(indicators)
                confidence_scores.append(obf_confidence)
            
            # CFG analysis confidence
            cfg_result = cfg_results.get('cfg_result')
            if cfg_result and cfg_result.get('analysis_quality'):
                confidence_scores.append(cfg_result['analysis_quality'])
            
            # Calculate weighted average
            if confidence_scores:
                return sum(confidence_scores) / len(confidence_scores)
            else:
                return 0.5  # Default confidence
        
        except Exception:
            return 0.0
    
    def _create_error_result(self, error_msg: str) -> Phase1AnalysisResult:
        """Create error result for failed analysis."""
        return Phase1AnalysisResult(
            entropy_analysis={'analysis_status': 'error', 'error': error_msg},
            cfg_analysis={'analysis_status': 'error', 'error': error_msg},
            packer_analysis={'analysis_status': 'error', 'error': error_msg},
            obfuscation_analysis={'analysis_status': 'error', 'error': error_msg},
            integration_metadata={'error': error_msg},
            recommendations=[f"Analysis failed: {error_msg}"],
            confidence_score=0.0
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def enhance_agent1_analysis(self, binary_path: Path, existing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance Agent 1 (Sentinel) analysis with Phase 1 capabilities.
        
        Args:
            binary_path: Path to binary being analyzed
            existing_analysis: Existing Agent 1 analysis results
            
        Returns:
            Enhanced analysis results with Phase 1 data
        """
        try:
            self.logger.info("Enhancing Agent 1 analysis with Phase 1 capabilities")
            
            # Perform comprehensive Phase 1 analysis
            phase1_results = self.perform_comprehensive_analysis(binary_path)
            
            # Merge with existing analysis
            enhanced_analysis = existing_analysis.copy()
            enhanced_analysis['phase1_deobfuscation'] = {
                'entropy_analysis': phase1_results.entropy_analysis,
                'packer_analysis': phase1_results.packer_analysis,
                'obfuscation_analysis': phase1_results.obfuscation_analysis,
                'cfg_analysis': phase1_results.cfg_analysis,
                'recommendations': phase1_results.recommendations,
                'confidence_score': phase1_results.confidence_score
            }
            
            # Update overall analysis confidence
            original_confidence = existing_analysis.get('confidence_score', 0.5)
            phase1_confidence = phase1_results.confidence_score
            
            # Weighted average with higher weight on Phase 1 if it's more confident
            if phase1_confidence > original_confidence:
                enhanced_confidence = (phase1_confidence * 0.7) + (original_confidence * 0.3)
            else:
                enhanced_confidence = (phase1_confidence * 0.3) + (original_confidence * 0.7)
            
            enhanced_analysis['confidence_score'] = enhanced_confidence
            enhanced_analysis['enhanced_with_phase1'] = True
            
            self.logger.info(f"Agent 1 enhancement complete. Confidence: {enhanced_confidence:.2f}")
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Error enhancing Agent 1 analysis: {e}")
            # Return original analysis if enhancement fails
            existing_analysis['phase1_enhancement_error'] = str(e)
            return existing_analysis


# Factory function for easy integration
def create_phase1_integrator(config_manager=None) -> Phase1Integrator:
    """Create configured Phase 1 integrator instance."""
    return Phase1Integrator(config_manager)


# Helper function for Agent 1 integration
def enhance_agent1_with_phase1(binary_path: Path, agent1_results: Dict[str, Any], 
                              config_manager=None) -> Dict[str, Any]:
    """
    Helper function to enhance Agent 1 results with Phase 1 analysis.
    
    Args:
        binary_path: Path to binary file
        agent1_results: Existing Agent 1 analysis results
        config_manager: Configuration manager instance
        
    Returns:
        Enhanced Agent 1 results with Phase 1 deobfuscation analysis
    """
    integrator = create_phase1_integrator(config_manager)
    return integrator.enhance_agent1_analysis(binary_path, agent1_results)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    integrator = Phase1Integrator()
    
    print("Phase 1 Integration Module loaded successfully")
    print("Available capabilities:")
    print("  - Entropy analysis and packed section detection")
    print("  - Advanced CFG reconstruction with indirect jump resolution")
    print("  - Comprehensive packer detection (UPX, Themida, VMProtect, etc.)")
    print("  - Obfuscation detection (CFF, VM obfuscation, anti-analysis)")
    print("  - Original Entry Point (OEP) analysis")
    print("  - Integrated recommendations and confidence scoring")
    print("  - Agent 1 (Sentinel) enhancement integration")