"""
Phase 1 Enhanced Integration Module

Integrates all Phase 1 advanced deobfuscation enhancements:
- Advanced anti-obfuscation techniques
- ML-enhanced control flow graph reconstruction  
- Modern packer detection system
- Comprehensive analysis orchestration

This module provides a unified interface for all Phase 1 enhancements
while maintaining compatibility with the existing deobfuscation framework.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

# Import existing Phase 1 components
from .phase1_integration import Phase1Integrator, Phase1AnalysisResult
from .entropy_analyzer import EntropyAnalyzer
from .cfg_reconstructor import AdvancedControlFlowAnalyzer
from .obfuscation_detector import ObfuscationDetector
from .packer_detector import PackerDetector

# Import new enhanced modules
from .advanced_anti_obfuscation import AdvancedAntiObfuscation, AdvancedDeobfuscationResult
from .ml_enhanced_cfg import MLEnhancedCFGReconstructor, MLEnhancedCFGResult
from .modern_packer_detection import ModernPackerDetector, ModernPackerResult


class EnhancementLevel(Enum):
    """Levels of Phase 1 enhancement."""
    BASIC = "basic"
    ADVANCED = "advanced"
    ML_ENHANCED = "ml_enhanced"
    COMPREHENSIVE = "comprehensive"


@dataclass
class Phase1EnhancedResult:
    """Comprehensive result of Phase 1 enhanced analysis."""
    # Basic Phase 1 results
    basic_analysis: Phase1AnalysisResult
    
    # Enhanced analysis results
    advanced_obfuscation: Optional[AdvancedDeobfuscationResult] = None
    ml_cfg_analysis: Optional[MLEnhancedCFGResult] = None
    modern_packer_analysis: Optional[ModernPackerResult] = None
    
    # Performance metrics
    execution_time: float = 0.0
    enhancement_level: EnhancementLevel = EnhancementLevel.BASIC
    
    # Combined confidence scores
    overall_confidence: float = 0.0
    component_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    enhanced_recommendations: List[str] = field(default_factory=list)
    
    # Quality metrics
    enhancement_quality: Dict[str, Any] = field(default_factory=dict)


class Phase1EnhancedIntegrator:
    """
    Enhanced Phase 1 Integrator with advanced deobfuscation capabilities.
    
    Provides a unified interface for all Phase 1 enhancements while maintaining
    backward compatibility with existing systems.
    """
    
    def __init__(self, config_manager=None):
        """Initialize enhanced Phase 1 integrator."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize base Phase 1 integrator
        self.base_integrator = Phase1Integrator(config_manager)
        
        # Initialize enhanced components
        self.advanced_anti_obfuscation = AdvancedAntiObfuscation(config_manager)
        self.ml_cfg_reconstructor = MLEnhancedCFGReconstructor(config_manager)
        self.modern_packer_detector = ModernPackerDetector(config_manager)
        
        # Configuration for enhancement levels
        self._configure_enhancement_levels()
        
        self.logger.info("Phase 1 Enhanced Integrator initialized with advanced capabilities")

    def _configure_enhancement_levels(self):
        """Configure different enhancement levels based on configuration."""
        # Default configuration
        self.enhancement_config = {
            EnhancementLevel.BASIC: {
                'use_advanced_obfuscation': False,
                'use_ml_cfg': False,
                'use_modern_packer': False,
                'timeout_multiplier': 1.0
            },
            EnhancementLevel.ADVANCED: {
                'use_advanced_obfuscation': True,
                'use_ml_cfg': False,
                'use_modern_packer': True,
                'timeout_multiplier': 2.0
            },
            EnhancementLevel.ML_ENHANCED: {
                'use_advanced_obfuscation': True,
                'use_ml_cfg': True,
                'use_modern_packer': True,
                'timeout_multiplier': 3.0
            },
            EnhancementLevel.COMPREHENSIVE: {
                'use_advanced_obfuscation': True,
                'use_ml_cfg': True,
                'use_modern_packer': True,
                'timeout_multiplier': 5.0,
                'deep_analysis': True
            }
        }
        
        # Override with configuration if available
        if self.config:
            enhancement_level = self.config.get_value('phase1.enhancement_level', 'advanced')
            self.default_enhancement_level = EnhancementLevel(enhancement_level)
        else:
            self.default_enhancement_level = EnhancementLevel.ADVANCED

    def perform_enhanced_analysis(self, binary_path: Path, 
                                enhancement_level: Optional[EnhancementLevel] = None) -> Phase1EnhancedResult:
        """
        Perform enhanced Phase 1 analysis with specified enhancement level.
        
        Args:
            binary_path: Path to binary file
            enhancement_level: Level of enhancement to apply
            
        Returns:
            Phase1EnhancedResult with comprehensive analysis
        """
        start_time = time.time()
        
        # Use default enhancement level if not specified
        if enhancement_level is None:
            enhancement_level = self.default_enhancement_level
        
        self.logger.info(f"Starting Phase 1 enhanced analysis: {binary_path}")
        self.logger.info(f"Enhancement level: {enhancement_level.value}")
        
        try:
            # Phase 1: Basic Phase 1 analysis
            self.logger.info("Performing basic Phase 1 analysis...")
            basic_result = self.base_integrator.perform_comprehensive_analysis(binary_path)
            
            # Initialize enhanced result
            enhanced_result = Phase1EnhancedResult(
                basic_analysis=basic_result,
                enhancement_level=enhancement_level
            )
            
            # Get enhancement configuration
            config = self.enhancement_config[enhancement_level]
            
            # Phase 2: Advanced anti-obfuscation analysis (if enabled)
            if config['use_advanced_obfuscation']:
                self.logger.info("Performing advanced anti-obfuscation analysis...")
                try:
                    advanced_obfuscation = self.advanced_anti_obfuscation.analyze_advanced_obfuscation(binary_path)
                    enhanced_result.advanced_obfuscation = advanced_obfuscation
                    enhanced_result.component_confidences['advanced_obfuscation'] = advanced_obfuscation.confidence_score
                except Exception as e:
                    self.logger.error(f"Advanced obfuscation analysis failed: {e}")
            
            # Phase 3: ML-enhanced CFG reconstruction (if enabled)
            if config['use_ml_cfg']:
                self.logger.info("Performing ML-enhanced CFG analysis...")
                try:
                    base_cfg = basic_result.cfg_analysis if hasattr(basic_result, 'cfg_analysis') else {}
                    ml_cfg_result = self.ml_cfg_reconstructor.enhance_cfg_with_ml(binary_path, base_cfg)
                    enhanced_result.ml_cfg_analysis = ml_cfg_result
                    
                    # Calculate ML CFG confidence
                    if ml_cfg_result.confidence_map:
                        avg_confidence = sum(ml_cfg_result.confidence_map.values()) / len(ml_cfg_result.confidence_map)
                        enhanced_result.component_confidences['ml_cfg'] = avg_confidence
                except Exception as e:
                    self.logger.error(f"ML-enhanced CFG analysis failed: {e}")
            
            # Phase 4: Modern packer detection (if enabled)
            if config['use_modern_packer']:
                self.logger.info("Performing modern packer detection...")
                try:
                    modern_packer_result = self.modern_packer_detector.detect_modern_packers(binary_path)
                    enhanced_result.modern_packer_analysis = modern_packer_result
                    
                    # Use overall detection confidence
                    packer_confidence = modern_packer_result.confidence_scores.get('overall_detection', 0.0)
                    enhanced_result.component_confidences['modern_packer'] = packer_confidence
                except Exception as e:
                    self.logger.error(f"Modern packer detection failed: {e}")
            
            # Phase 5: Calculate overall confidence and recommendations
            enhanced_result.overall_confidence = self._calculate_overall_confidence(enhanced_result)
            enhanced_result.enhanced_recommendations = self._generate_enhanced_recommendations(enhanced_result)
            enhanced_result.enhancement_quality = self._assess_enhancement_quality(enhanced_result)
            
            # Record execution time
            enhanced_result.execution_time = time.time() - start_time
            
            self.logger.info(f"Phase 1 enhanced analysis complete in {enhanced_result.execution_time:.2f}s")
            self.logger.info(f"Overall confidence: {enhanced_result.overall_confidence:.2f}")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Phase 1 enhanced analysis failed: {e}")
            # Return basic result with error information
            execution_time = time.time() - start_time
            return Phase1EnhancedResult(
                basic_analysis=basic_result if 'basic_result' in locals() else Phase1AnalysisResult(),
                enhancement_level=enhancement_level,
                execution_time=execution_time,
                enhanced_recommendations=[f"Enhancement failed: {e}"]
            )

    def enhance_agent1_with_phase1_enhancements(self, binary_path: Path, 
                                              existing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance Agent 1 (Sentinel) analysis with Phase 1 enhancements.
        
        This method provides the enhanced integration interface for Agent 1.
        """
        self.logger.info("Enhancing Agent 1 analysis with Phase 1 enhancements")
        
        try:
            # Perform enhanced analysis
            enhanced_result = self.perform_enhanced_analysis(binary_path)
            
            # Merge with existing analysis
            enhanced_analysis = existing_analysis.copy()
            
            # Add enhanced Phase 1 data
            enhanced_analysis['phase1_enhanced'] = {
                'enhancement_level': enhanced_result.enhancement_level.value,
                'execution_time': enhanced_result.execution_time,
                'overall_confidence': enhanced_result.overall_confidence,
                'component_confidences': enhanced_result.component_confidences,
                'enhanced_recommendations': enhanced_result.enhanced_recommendations,
                'enhancement_quality': enhanced_result.enhancement_quality
            }
            
            # Add specific enhancement results
            if enhanced_result.advanced_obfuscation:
                enhanced_analysis['advanced_obfuscation'] = {
                    'obfuscation_techniques': [t.value for t in enhanced_result.advanced_obfuscation.obfuscation_techniques],
                    'control_flow_patterns': len(enhanced_result.advanced_obfuscation.control_flow_patterns),
                    'vm_patterns': len(enhanced_result.advanced_obfuscation.vm_patterns),
                    'confidence': enhanced_result.advanced_obfuscation.confidence_score,
                    'recommendations': enhanced_result.advanced_obfuscation.recommendations
                }
            
            if enhanced_result.ml_cfg_analysis:
                enhanced_analysis['ml_cfg_enhancements'] = {
                    'indirect_jump_predictions': len(enhanced_result.ml_cfg_analysis.indirect_jump_predictions),
                    'switch_reconstructions': len(enhanced_result.ml_cfg_analysis.switch_reconstructions),
                    'exception_handlers': len(enhanced_result.ml_cfg_analysis.exception_handlers),
                    'confidence_map_size': len(enhanced_result.ml_cfg_analysis.confidence_map)
                }
            
            if enhanced_result.modern_packer_analysis:
                enhanced_analysis['modern_packer_detection'] = {
                    'detected_packers': [p.packer_type.value for p in enhanced_result.modern_packer_analysis.detected_packers],
                    'multi_layer_protection': enhanced_result.modern_packer_analysis.multi_layer_analysis is not None,
                    'custom_signatures': len(enhanced_result.modern_packer_analysis.custom_signatures),
                    'unpacking_strategies': enhanced_result.modern_packer_analysis.unpacking_strategies,
                    'confidence_scores': enhanced_result.modern_packer_analysis.confidence_scores
                }
            
            # Update overall confidence
            if 'confidence' in enhanced_analysis:
                enhanced_analysis['confidence'] = max(
                    enhanced_analysis['confidence'],
                    enhanced_result.overall_confidence
                )
            else:
                enhanced_analysis['confidence'] = enhanced_result.overall_confidence
            
            self.logger.info("Agent 1 enhancement completed successfully")
            return enhanced_analysis
            
        except Exception as e:
            self.logger.error(f"Agent 1 enhancement failed: {e}")
            # Return original analysis if enhancement fails
            return existing_analysis

    def _calculate_overall_confidence(self, result: Phase1EnhancedResult) -> float:
        """Calculate overall confidence score from all components."""
        confidences = []
        
        # Base Phase 1 confidence
        if hasattr(result.basic_analysis, 'confidence'):
            confidences.append(result.basic_analysis.confidence)
        
        # Component confidences
        confidences.extend(result.component_confidences.values())
        
        # Calculate weighted average
        if confidences:
            # Weight recent/advanced techniques higher
            weights = [1.0] + [2.0] * (len(confidences) - 1)  # Higher weight for enhancements
            weighted_sum = sum(c * w for c, w in zip(confidences, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        
        return 0.0

    def _generate_enhanced_recommendations(self, result: Phase1EnhancedResult) -> List[str]:
        """Generate enhanced recommendations based on all analysis components."""
        recommendations = []
        
        # Base recommendations
        if hasattr(result.basic_analysis, 'recommendations'):
            recommendations.extend(result.basic_analysis.recommendations)
        
        # Advanced obfuscation recommendations
        if result.advanced_obfuscation:
            recommendations.extend(result.advanced_obfuscation.recommendations)
        
        # ML CFG recommendations
        if result.ml_cfg_analysis and result.ml_cfg_analysis.indirect_jump_predictions:
            recommendations.append("Apply ML-enhanced indirect jump resolution for improved CFG accuracy")
            
            if result.ml_cfg_analysis.switch_reconstructions:
                recommendations.append("Use ML-detected switch statement reconstructions for better analysis")
        
        # Modern packer recommendations
        if result.modern_packer_analysis:
            recommendations.extend(result.modern_packer_analysis.unpacking_strategies)
        
        # Enhancement-specific recommendations
        if result.enhancement_level == EnhancementLevel.COMPREHENSIVE:
            recommendations.extend([
                "Consider applying specialized deobfuscation tools for detected techniques",
                "Use multi-pass analysis with different tool combinations",
                "Validate results with dynamic analysis techniques"
            ])
        
        # Quality-based recommendations
        if result.overall_confidence < 0.6:
            recommendations.append("Low confidence detected - consider manual analysis verification")
        
        return list(set(recommendations))  # Remove duplicates

    def _assess_enhancement_quality(self, result: Phase1EnhancedResult) -> Dict[str, Any]:
        """Assess the quality of enhancements applied."""
        quality_assessment = {
            'enhancement_success_rate': 0.0,
            'component_coverage': 0.0,
            'analysis_depth': 'basic',
            'recommendations_quality': 'standard'
        }
        
        # Calculate enhancement success rate
        total_enhancements = 3  # Max possible enhancements
        successful_enhancements = 0
        
        if result.advanced_obfuscation:
            successful_enhancements += 1
        if result.ml_cfg_analysis:
            successful_enhancements += 1
        if result.modern_packer_analysis:
            successful_enhancements += 1
        
        quality_assessment['enhancement_success_rate'] = successful_enhancements / total_enhancements
        
        # Calculate component coverage
        total_components = len(result.component_confidences)
        if total_components > 0:
            avg_confidence = sum(result.component_confidences.values()) / total_components
            quality_assessment['component_coverage'] = avg_confidence
        
        # Determine analysis depth
        if result.enhancement_level == EnhancementLevel.COMPREHENSIVE:
            quality_assessment['analysis_depth'] = 'comprehensive'
        elif result.enhancement_level == EnhancementLevel.ML_ENHANCED:
            quality_assessment['analysis_depth'] = 'ml_enhanced'
        elif result.enhancement_level == EnhancementLevel.ADVANCED:
            quality_assessment['analysis_depth'] = 'advanced'
        
        # Assess recommendations quality
        rec_count = len(result.enhanced_recommendations)
        if rec_count > 10:
            quality_assessment['recommendations_quality'] = 'comprehensive'
        elif rec_count > 5:
            quality_assessment['recommendations_quality'] = 'detailed'
        
        return quality_assessment

    def get_enhancement_capabilities(self) -> Dict[str, Any]:
        """Get information about available enhancement capabilities."""
        return {
            'available_enhancements': {
                'advanced_anti_obfuscation': True,
                'ml_enhanced_cfg': True,
                'modern_packer_detection': True
            },
            'enhancement_levels': [level.value for level in EnhancementLevel],
            'default_level': self.default_enhancement_level.value,
            'supported_techniques': {
                'control_flow_flattening_reversal': True,
                'virtual_machine_detection': True,
                'ml_indirect_jump_resolution': True,
                'modern_packer_signatures': True,
                'multi_layer_protection_analysis': True
            }
        }

    def validate_enhancement_configuration(self) -> Dict[str, Any]:
        """Validate the current enhancement configuration."""
        validation_result = {
            'configuration_valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        try:
            # Check if enhancement components are available
            if not hasattr(self, 'advanced_anti_obfuscation'):
                validation_result['warnings'].append("Advanced anti-obfuscation component not available")
            
            if not hasattr(self, 'ml_cfg_reconstructor'):
                validation_result['warnings'].append("ML CFG reconstructor not available")
            
            if not hasattr(self, 'modern_packer_detector'):
                validation_result['warnings'].append("Modern packer detector not available")
            
            # Check configuration validity
            if self.default_enhancement_level not in self.enhancement_config:
                validation_result['configuration_valid'] = False
                validation_result['warnings'].append(f"Invalid default enhancement level: {self.default_enhancement_level}")
            
            # Generate recommendations
            if validation_result['warnings']:
                validation_result['recommendations'].append("Consider installing missing dependencies for full enhancement capabilities")
            
        except Exception as e:
            validation_result['configuration_valid'] = False
            validation_result['warnings'].append(f"Configuration validation failed: {e}")
        
        return validation_result


def create_phase1_enhanced_integrator(config_manager=None) -> Phase1EnhancedIntegrator:
    """
    Factory function to create Phase 1 enhanced integrator.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Phase1EnhancedIntegrator: Configured integrator instance
    """
    return Phase1EnhancedIntegrator(config_manager)


# Backward compatibility helper
def enhance_agent1_with_phase1_advanced(binary_path: Path, existing_analysis: Dict[str, Any], 
                                       config_manager=None) -> Dict[str, Any]:
    """
    Backward compatibility function for enhancing Agent 1 with advanced Phase 1 techniques.
    
    This function provides the same interface as the original Phase 1 integration
    but with enhanced capabilities.
    """
    integrator = create_phase1_enhanced_integrator(config_manager)
    return integrator.enhance_agent1_with_phase1_enhancements(binary_path, existing_analysis)