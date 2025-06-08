"""
Machine Learning Components for Open-Sourcefy Matrix Pipeline

This module provides AI-powered analysis capabilities including:
- Function classification and naming
- Code pattern recognition  
- Quality assessment and scoring
- Neural network-based code mapping
- Semantic analysis for reverse engineering
"""

from typing import Dict, Any, List, Optional
import logging

__version__ = "2.0.0"
__all__ = [
    'FunctionClassifier', 'PatternEngine', 'QualityScorer', 
    'NeuralMapper', 'VariableNamer', 'SemanticAnalyzer'
]

# Optional ML dependencies
try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Import ML components
from .function_classifier import FunctionClassifier
from .pattern_engine import PatternEngine  
from .quality_scorer import QualityScorer
from .neural_mapper import NeuralMapper
from .variable_namer import VariableNamer
from .semantic_analyzer import SemanticAnalyzer

logger = logging.getLogger(__name__)


def check_ml_dependencies() -> Dict[str, bool]:
    """Check availability of ML dependencies"""
    return {
        'sklearn': SKLEARN_AVAILABLE,
        'numpy': NUMPY_AVAILABLE, 
        'pytorch': PYTORCH_AVAILABLE
    }


def get_ml_capabilities() -> List[str]:
    """Get list of available ML capabilities based on dependencies"""
    capabilities = []
    
    if SKLEARN_AVAILABLE:
        capabilities.extend([
            'function_classification',
            'pattern_recognition', 
            'quality_scoring'
        ])
    
    if NUMPY_AVAILABLE:
        capabilities.extend([
            'statistical_analysis',
            'feature_extraction'
        ])
        
    if PYTORCH_AVAILABLE:
        capabilities.extend([
            'neural_mapping',
            'deep_learning_analysis'
        ])
    
    return capabilities


def create_ml_analyzer(analyzer_type: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Factory function to create ML analyzer instances
    
    Args:
        analyzer_type: Type of analyzer ('function_classifier', 'pattern_engine', etc.)
        config: Optional configuration dictionary
        
    Returns:
        Configured ML analyzer instance
        
    Raises:
        ValueError: If analyzer type is not supported
        ImportError: If required dependencies are missing
    """
    config = config or {}
    
    if analyzer_type == 'function_classifier':
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for function classification")
        return FunctionClassifier(config)
        
    elif analyzer_type == 'pattern_engine':
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for pattern recognition")
        return PatternEngine(config)
        
    elif analyzer_type == 'quality_scorer':
        return QualityScorer(config)
        
    elif analyzer_type == 'neural_mapper':
        if not PYTORCH_AVAILABLE:
            raise ImportError("pytorch required for neural mapping")
        return NeuralMapper(config)
        
    elif analyzer_type == 'variable_namer':
        return VariableNamer(config)
        
    elif analyzer_type == 'semantic_analyzer':
        return SemanticAnalyzer(config)
        
    else:
        raise ValueError(f"Unknown analyzer type: {analyzer_type}")


class MLAnalysisEngine:
    """
    Unified ML analysis engine that coordinates multiple ML components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize available analyzers
        self.analyzers = {}
        self._initialize_analyzers()
        
    def _initialize_analyzers(self):
        """Initialize available ML analyzers based on dependencies"""
        capabilities = get_ml_capabilities()
        
        try:
            if 'function_classification' in capabilities:
                self.analyzers['function_classifier'] = create_ml_analyzer('function_classifier', self.config)
                
            if 'pattern_recognition' in capabilities:
                self.analyzers['pattern_engine'] = create_ml_analyzer('pattern_engine', self.config)
                
            if 'quality_scoring' in capabilities:
                self.analyzers['quality_scorer'] = create_ml_analyzer('quality_scorer', self.config)
                
            self.analyzers['variable_namer'] = create_ml_analyzer('variable_namer', self.config)
            self.analyzers['semantic_analyzer'] = create_ml_analyzer('semantic_analyzer', self.config)
            
            if 'neural_mapping' in capabilities:
                self.analyzers['neural_mapper'] = create_ml_analyzer('neural_mapper', self.config)
                
        except Exception as e:
            self.logger.warning(f"Failed to initialize some ML analyzers: {e}")
    
    def analyze_code(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive ML analysis on code data
        
        Args:
            code_data: Dictionary containing code to analyze
            
        Returns:
            Dictionary with ML analysis results
        """
        results = {
            'ml_analysis_version': __version__,
            'available_analyzers': list(self.analyzers.keys()),
            'analysis_results': {}
        }
        
        # Run each available analyzer
        for analyzer_name, analyzer in self.analyzers.items():
            try:
                self.logger.info(f"Running {analyzer_name} analysis...")
                analysis_result = analyzer.analyze(code_data)
                results['analysis_results'][analyzer_name] = analysis_result
                
            except Exception as e:
                self.logger.error(f"ML analysis failed for {analyzer_name}: {e}")
                results['analysis_results'][analyzer_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def get_quality_metrics(self, code_data: Dict[str, Any]) -> Dict[str, float]:
        """Get quality metrics for code"""
        if 'quality_scorer' in self.analyzers:
            try:
                return self.analyzers['quality_scorer'].get_quality_metrics(code_data)
            except Exception as e:
                self.logger.error(f"Quality scoring failed: {e}")
        
        return {'overall_quality': 0.0}
    
    def classify_functions(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify functions using ML"""
        if 'function_classifier' in self.analyzers:
            try:
                return self.analyzers['function_classifier'].classify_functions(functions)
            except Exception as e:
                self.logger.error(f"Function classification failed: {e}")
        
        return functions  # Return unchanged if classifier unavailable
    
    def detect_patterns(self, code_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect code patterns using ML"""
        if 'pattern_engine' in self.analyzers:
            try:
                return self.analyzers['pattern_engine'].detect_patterns(code_data)
            except Exception as e:
                self.logger.error(f"Pattern detection failed: {e}")
        
        return []
    
    def suggest_variable_names(self, variables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest meaningful variable names"""
        if 'variable_namer' in self.analyzers:
            try:
                return self.analyzers['variable_namer'].suggest_names(variables)
            except Exception as e:
                self.logger.error(f"Variable naming failed: {e}")
        
        return variables


# Convenience function for simple ML analysis
def analyze_with_ml(code_data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function for ML analysis
    
    Args:
        code_data: Code data to analyze
        config: Optional ML configuration
        
    Returns:
        ML analysis results
    """
    engine = MLAnalysisEngine(config)
    return engine.analyze_code(code_data)


# Log ML capabilities on import
capabilities = get_ml_capabilities()
if capabilities:
    logger.info(f"ML capabilities available: {', '.join(capabilities)}")
else:
    logger.warning("No ML dependencies available - ML features disabled")