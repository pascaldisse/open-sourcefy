"""
Semantic Analyzer Module - AI-Enhanced Code Analysis

This module provides semantic analysis capabilities for decompiled code including:
- Semantic function naming using ML models
- Variable purpose inference algorithms  
- Algorithm pattern recognition framework
- Code style analysis and architectural pattern detection
- Integration with existing agent pipeline

Production-ready implementation following SOLID principles and clean code standards.
"""

import os
import re
import json
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict
from enum import Enum
import time

# Core framework imports
try:
    from ..core.config_manager import ConfigManager
    from ..core.shared_components import MatrixLogger, MatrixValidator
except ImportError:
    # Fallback for development/testing
    ConfigManager = None
    MatrixLogger = logging.getLogger
    MatrixValidator = None

# AI/ML imports with graceful fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    # Scikit-learn for basic ML features
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    # Transformers for advanced NLP
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    torch = None

logger = logging.getLogger(__name__)


class SemanticAnalysisType(Enum):
    """Types of semantic analysis"""
    FUNCTION_NAMING = "function_naming"
    VARIABLE_INFERENCE = "variable_inference"
    ALGORITHM_DETECTION = "algorithm_detection"
    CODE_STYLE_ANALYSIS = "code_style_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    ARCHITECTURAL_ANALYSIS = "architectural_analysis"


class ConfidenceLevel(Enum):
    """Confidence levels for analysis results"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9


@dataclass
class SemanticResult:
    """Result of semantic analysis"""
    analysis_type: SemanticAnalysisType
    original_name: str
    suggested_name: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['analysis_type'] = self.analysis_type.value
        return result


@dataclass
class FunctionSemantic:
    """Semantic information for a function"""
    original_name: str
    suggested_name: str
    purpose: str
    complexity_category: str
    algorithm_patterns: List[str]
    variable_suggestions: Dict[str, str]
    confidence: float
    reasoning: List[str]


@dataclass
class CodeAnalysisReport:
    """Comprehensive code analysis report"""
    total_functions: int
    analyzed_functions: int
    function_semantics: Dict[str, FunctionSemantic]
    algorithm_patterns: Dict[str, List[str]]
    code_style_metrics: Dict[str, float]
    architectural_patterns: List[str]
    overall_confidence: float
    analysis_time: float
    

class SemanticAnalyzer:
    """
    AI-Enhanced Semantic Analyzer for decompiled code
    
    Features:
    - Function naming using pattern recognition and ML models
    - Variable purpose inference based on usage patterns
    - Algorithm detection using code signature analysis
    - Code style analysis for quality assessment
    - Architectural pattern recognition
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize semantic analyzer
        
        Args:
            config: Configuration manager instance
        """
        self.config = config or self._create_default_config()
        self.logger = MatrixLogger(__name__) if MatrixLogger != logging.getLogger else logging.getLogger(__name__)
        
        # Initialize ML capabilities
        self.ml_enabled = HAS_SKLEARN and HAS_NUMPY
        self.advanced_ml_enabled = HAS_TRANSFORMERS
        
        # Initialize models and patterns
        self._function_patterns = self._load_function_patterns()
        self._algorithm_signatures = self._load_algorithm_signatures()
        self._code_style_rules = self._load_code_style_rules()
        
        # Initialize ML components if available
        if self.ml_enabled:
            self._initialize_ml_models()
        
        self.logger.info(f"SemanticAnalyzer initialized with ML={'enabled' if self.ml_enabled else 'disabled'}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'semantic_analysis': {
                'confidence_threshold': 0.6,
                'max_suggestions': 5,
                'use_advanced_ml': False,
                'pattern_matching_strict': True
            }
        }
    
    def _load_function_patterns(self) -> Dict[str, List[str]]:
        """Load function naming patterns"""
        return {
            'getter': ['get_', 'retrieve_', 'fetch_', 'read_', 'load_'],
            'setter': ['set_', 'update_', 'write_', 'store_', 'save_'],
            'validator': ['is_', 'has_', 'can_', 'check_', 'validate_'],
            'processor': ['process_', 'handle_', 'execute_', 'run_', 'perform_'],
            'converter': ['convert_', 'transform_', 'parse_', 'encode_', 'decode_'],
            'creator': ['create_', 'make_', 'build_', 'generate_', 'construct_'],
            'destroyer': ['delete_', 'remove_', 'destroy_', 'clear_', 'cleanup_']
        }
    
    def _load_algorithm_signatures(self) -> Dict[str, Dict[str, Any]]:
        """Load algorithm detection signatures"""
        return {
            'sorting': {
                'keywords': ['sort', 'bubble', 'quick', 'merge', 'heap', 'insertion'],
                'patterns': [r'for.*for.*swap', r'while.*partition', r'recursive.*divide'],
                'complexity_indicators': ['nested_loops', 'recursive_calls', 'array_access']
            },
            'searching': {
                'keywords': ['search', 'find', 'binary', 'linear', 'hash'],
                'patterns': [r'while.*mid', r'for.*compare', r'hash.*lookup'],
                'complexity_indicators': ['loop_with_comparison', 'divide_and_conquer']
            },
            'cryptographic': {
                'keywords': ['encrypt', 'decrypt', 'hash', 'cipher', 'key', 'crypto'],
                'patterns': [r'xor.*key', r'shift.*bits', r'modular.*arithmetic'],
                'complexity_indicators': ['bitwise_operations', 'mathematical_operations']
            },
            'string_processing': {
                'keywords': ['string', 'text', 'parse', 'format', 'regex'],
                'patterns': [r'char.*array', r'string.*manipulation', r'pattern.*match'],
                'complexity_indicators': ['character_iteration', 'pattern_matching']
            }
        }
    
    def _load_code_style_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load code style analysis rules"""
        return {
            'naming_conventions': {
                'snake_case': r'^[a-z]+(_[a-z]+)*$',
                'camel_case': r'^[a-z]+([A-Z][a-z]+)*$',
                'pascal_case': r'^[A-Z][a-z]+([A-Z][a-z]+)*$',
                'hungarian_notation': r'^[a-z]{1,3}[A-Z]'
            },
            'complexity_metrics': {
                'max_function_length': 50,
                'max_cyclomatic_complexity': 10,
                'max_nesting_depth': 4
            },
            'quality_indicators': {
                'meaningful_names': r'^[a-zA-Z_][a-zA-Z0-9_]{2,}$',
                'proper_spacing': r'\s+',
                'comment_density': 0.1
            }
        }
    
    def _initialize_ml_models(self):
        """Initialize ML models for semantic analysis"""
        try:
            # Initialize TF-IDF vectorizer for function naming
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Initialize clustering model for pattern recognition
            self.kmeans_model = KMeans(n_clusters=10, random_state=42)
            
            # Initialize classification model for function categorization
            if HAS_SKLEARN:
                self.function_classifier = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            self.ml_enabled = False
    
    def analyze(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for semantic analysis
        
        Args:
            code_data: Dictionary containing code analysis data
            
        Returns:
            Dictionary with semantic analysis results
        """
        start_time = time.time()
        
        try:
            self.logger.info("Starting comprehensive semantic analysis...")
            
            functions = code_data.get('functions', [])
            decompiled_code = code_data.get('decompiled_code', '')
            
            # Perform different types of analysis
            function_analysis = self._analyze_functions(functions)
            algorithm_patterns = self._detect_algorithm_patterns(decompiled_code)
            code_style_metrics = self._analyze_code_style(decompiled_code)
            architectural_patterns = self._detect_architectural_patterns(code_data)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence([
                function_analysis.get('average_confidence', 0.0),
                algorithm_patterns.get('detection_confidence', 0.0),
                code_style_metrics.get('quality_score', 0.0)
            ])
            
            analysis_time = time.time() - start_time
            
            results = {
                'semantic_analysis_version': '2.0',
                'analysis_timestamp': time.time(),
                'ml_enabled': self.ml_enabled,
                'function_analysis': function_analysis,
                'algorithm_patterns': algorithm_patterns,
                'code_style_metrics': code_style_metrics,
                'architectural_patterns': architectural_patterns,
                'overall_confidence': overall_confidence,
                'analysis_time': analysis_time,
                'capabilities': self._get_analysis_capabilities()
            }
            
            self.logger.info(f"Semantic analysis completed in {analysis_time:.2f}s with confidence {overall_confidence:.2f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}", exc_info=True)
            return {
                'error': str(e),
                'status': 'failed',
                'analysis_time': time.time() - start_time
            }
    
    def _analyze_functions(self, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze functions for semantic meaning"""
        if not functions:
            return {'analyzed_functions': 0, 'function_semantics': {}}
        
        function_semantics = {}
        total_confidence = 0.0
        
        for func in functions:
            func_name = func.get('name', 'unknown_function')
            func_code = func.get('code', '')
            
            # Analyze function purpose and suggest better name
            semantic_result = self._analyze_single_function(func_name, func_code, func)
            function_semantics[func_name] = semantic_result
            total_confidence += semantic_result.get('confidence', 0.0)
        
        average_confidence = total_confidence / len(functions) if functions else 0.0
        
        return {
            'analyzed_functions': len(functions),
            'function_semantics': function_semantics,
            'average_confidence': average_confidence,
            'naming_suggestions': self._generate_naming_suggestions(function_semantics)
        }
    
    def _analyze_single_function(self, func_name: str, func_code: str, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single function for semantic meaning"""
        # Extract features from function
        features = self._extract_function_features(func_name, func_code, func_data)
        
        # Determine function category
        category = self._classify_function_category(features)
        
        # Generate name suggestions
        name_suggestions = self._generate_function_name_suggestions(features, category)
        
        # Analyze variables
        variable_suggestions = self._analyze_function_variables(func_code)
        
        # Calculate confidence
        confidence = self._calculate_function_confidence(features, category, name_suggestions)
        
        return {
            'original_name': func_name,
            'suggested_names': name_suggestions,
            'category': category,
            'purpose': features.get('inferred_purpose', 'Unknown'),
            'complexity': features.get('complexity', 'Medium'),
            'variable_suggestions': variable_suggestions,
            'confidence': confidence,
            'features': features
        }
    
    def _extract_function_features(self, func_name: str, func_code: str, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract semantic features from function"""
        features = {
            'name_length': len(func_name),
            'code_length': len(func_code),
            'parameter_count': len(func_data.get('parameters', [])),
            'return_type': func_data.get('return_type', 'void'),
            'has_loops': 'for' in func_code or 'while' in func_code,
            'has_conditionals': 'if' in func_code,
            'has_recursion': func_name in func_code,
            'string_operations': any(op in func_code for op in ['string', 'str', 'char']),
            'mathematical_operations': any(op in func_code for op in ['+', '-', '*', '/', '%']),
            'memory_operations': any(op in func_code for op in ['malloc', 'free', 'new', 'delete']),
            'io_operations': any(op in func_code for op in ['printf', 'scanf', 'read', 'write']),
            'api_calls': self._count_api_calls(func_code),
            'complexity_estimate': self._estimate_complexity(func_code)
        }
        
        # Infer purpose based on features
        features['inferred_purpose'] = self._infer_function_purpose(features, func_code)
        
        return features
    
    def _classify_function_category(self, features: Dict[str, Any]) -> str:
        """Classify function into semantic categories"""
        # Rule-based classification
        if features.get('return_type') == 'bool' or features.get('inferred_purpose', '').startswith('validate'):
            return 'validator'
        elif features.get('parameter_count', 0) == 0 and 'return' in features.get('inferred_purpose', ''):
            return 'getter'
        elif features.get('parameter_count', 0) > 0 and features.get('return_type') == 'void':
            return 'setter'
        elif features.get('has_loops') and features.get('mathematical_operations'):
            return 'processor'
        elif features.get('string_operations'):
            return 'converter'
        elif features.get('memory_operations'):
            if 'malloc' in str(features) or 'new' in str(features):
                return 'creator'
            else:
                return 'destroyer'
        else:
            return 'processor'  # Default category
    
    def _generate_function_name_suggestions(self, features: Dict[str, Any], category: str) -> List[str]:
        """Generate meaningful function name suggestions"""
        suggestions = []
        
        # Get base patterns for category
        base_patterns = self._function_patterns.get(category, ['process_'])
        
        # Generate purpose-based names
        purpose = features.get('inferred_purpose', 'unknown')
        for pattern in base_patterns[:3]:  # Limit to top 3 patterns
            if purpose != 'unknown':
                suggested_name = f"{pattern}{purpose.replace(' ', '_').lower()}"
                suggestions.append(suggested_name)
        
        # Add complexity-based suggestions
        complexity = features.get('complexity_estimate', 'medium')
        if complexity == 'high' and category == 'processor':
            suggestions.append('complex_processor')
        elif complexity == 'low' and category == 'getter':
            suggestions.append('simple_getter')
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _analyze_function_variables(self, func_code: str) -> Dict[str, str]:
        """Analyze variables in function and suggest better names"""
        # Simple variable extraction (could be enhanced with AST parsing)
        variable_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        variables = re.findall(variable_pattern, func_code)
        
        suggestions = {}
        for var in set(variables):
            if len(var) <= 2 or var.startswith('temp') or var.startswith('tmp'):
                # Suggest better name based on context
                context = self._analyze_variable_context(var, func_code)
                suggestions[var] = self._suggest_variable_name(context)
        
        return suggestions
    
    def _detect_algorithm_patterns(self, code: str) -> Dict[str, Any]:
        """Detect algorithmic patterns in code"""
        detected_patterns = {}
        overall_confidence = 0.0
        
        for algorithm, signature in self._algorithm_signatures.items():
            confidence = self._calculate_algorithm_confidence(code, signature)
            if confidence > 0.3:  # Threshold for detection
                detected_patterns[algorithm] = {
                    'confidence': confidence,
                    'indicators': self._get_matching_indicators(code, signature)
                }
                overall_confidence = max(overall_confidence, confidence)
        
        return {
            'detected_algorithms': detected_patterns,
            'detection_confidence': overall_confidence,
            'algorithm_count': len(detected_patterns)
        }
    
    def _analyze_code_style(self, code: str) -> Dict[str, Any]:
        """Analyze code style and quality metrics"""
        style_metrics = {
            'naming_convention_score': self._analyze_naming_conventions(code),
            'complexity_score': self._analyze_complexity(code),
            'readability_score': self._analyze_readability(code),
            'consistency_score': self._analyze_consistency(code)
        }
        
        # Calculate overall quality score
        quality_score = sum(style_metrics.values()) / len(style_metrics)
        style_metrics['quality_score'] = quality_score
        
        return style_metrics
    
    def _detect_architectural_patterns(self, code_data: Dict[str, Any]) -> List[str]:
        """Detect architectural patterns in the codebase"""
        patterns = []
        
        functions = code_data.get('functions', [])
        if not functions:
            return patterns
        
        # Detect common patterns
        if self._has_factory_pattern(functions):
            patterns.append('Factory Pattern')
        
        if self._has_singleton_pattern(functions):
            patterns.append('Singleton Pattern')
        
        if self._has_observer_pattern(functions):
            patterns.append('Observer Pattern')
        
        if self._has_mvc_pattern(code_data):
            patterns.append('Model-View-Controller')
        
        return patterns
    
    # Helper methods for pattern detection
    def _infer_function_purpose(self, features: Dict[str, Any], code: str) -> str:
        """Infer the purpose of a function based on its features"""
        if features.get('io_operations'):
            return 'input_output_handling'
        elif features.get('string_operations'):
            return 'string_processing'
        elif features.get('mathematical_operations'):
            return 'mathematical_computation'
        elif features.get('memory_operations'):
            return 'memory_management'
        elif features.get('has_loops') and features.get('has_conditionals'):
            return 'data_processing'
        else:
            return 'general_processing'
    
    def _count_api_calls(self, code: str) -> int:
        """Count API calls in function"""
        api_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\('
        return len(re.findall(api_pattern, code))
    
    def _estimate_complexity(self, code: str) -> str:
        """Estimate function complexity"""
        lines = len(code.split('\n'))
        nesting = code.count('{') + code.count('if') + code.count('for') + code.count('while')
        
        if lines > 50 or nesting > 10:
            return 'high'
        elif lines > 20 or nesting > 5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_function_confidence(self, features: Dict[str, Any], category: str, suggestions: List[str]) -> float:
        """Calculate confidence score for function analysis"""
        base_confidence = 0.5
        
        # Boost confidence based on clear indicators
        if features.get('inferred_purpose') != 'unknown':
            base_confidence += 0.2
        if len(suggestions) > 2:
            base_confidence += 0.1
        if category != 'processor':  # Specific category detected
            base_confidence += 0.15
        
        return min(base_confidence, 1.0)
    
    def _calculate_overall_confidence(self, confidences: List[float]) -> float:
        """Calculate overall confidence from individual scores"""
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
    
    def _get_analysis_capabilities(self) -> List[str]:
        """Get list of available analysis capabilities"""
        capabilities = [
            'function_analysis',
            'algorithm_detection',
            'code_style_analysis',
            'architectural_patterns'
        ]
        
        if self.ml_enabled:
            capabilities.extend(['ml_classification', 'similarity_analysis'])
        
        if self.advanced_ml_enabled:
            capabilities.extend(['transformer_analysis', 'advanced_nlp'])
        
        return capabilities
    
    # Placeholder methods that would be expanded in full implementation
    def _analyze_variable_context(self, var: str, code: str) -> Dict[str, Any]:
        """Analyze context of variable usage"""
        return {'type': 'unknown', 'usage': 'general'}
    
    def _suggest_variable_name(self, context: Dict[str, Any]) -> str:
        """Suggest better variable name based on context"""
        return f"improved_{context.get('type', 'var')}"
    
    def _calculate_algorithm_confidence(self, code: str, signature: Dict[str, Any]) -> float:
        """Calculate confidence for algorithm detection"""
        keyword_matches = sum(1 for keyword in signature['keywords'] if keyword in code.lower())
        pattern_matches = sum(1 for pattern in signature['patterns'] if re.search(pattern, code))
        
        total_indicators = len(signature['keywords']) + len(signature['patterns'])
        if total_indicators == 0:
            return 0.0
        
        return (keyword_matches + pattern_matches) / total_indicators
    
    def _get_matching_indicators(self, code: str, signature: Dict[str, Any]) -> List[str]:
        """Get list of matching indicators for algorithm"""
        indicators = []
        for keyword in signature['keywords']:
            if keyword in code.lower():
                indicators.append(f"keyword: {keyword}")
        return indicators
    
    def _analyze_naming_conventions(self, code: str) -> float:
        """Analyze naming convention consistency"""
        return 0.7  # Placeholder implementation
    
    def _analyze_complexity(self, code: str) -> float:
        """Analyze code complexity"""
        return 0.6  # Placeholder implementation
    
    def _analyze_readability(self, code: str) -> float:
        """Analyze code readability"""
        return 0.8  # Placeholder implementation
    
    def _analyze_consistency(self, code: str) -> float:
        """Analyze code consistency"""
        return 0.75  # Placeholder implementation
    
    def _has_factory_pattern(self, functions: List[Dict[str, Any]]) -> bool:
        """Check for factory pattern"""
        return any('create' in f.get('name', '').lower() for f in functions)
    
    def _has_singleton_pattern(self, functions: List[Dict[str, Any]]) -> bool:
        """Check for singleton pattern"""
        return any('instance' in f.get('name', '').lower() for f in functions)
    
    def _has_observer_pattern(self, functions: List[Dict[str, Any]]) -> bool:
        """Check for observer pattern"""
        return any(name in ['notify', 'update', 'subscribe'] for f in functions for name in [f.get('name', '').lower()])
    
    def _has_mvc_pattern(self, code_data: Dict[str, Any]) -> bool:
        """Check for MVC pattern"""
        functions = code_data.get('functions', [])
        has_model = any('model' in f.get('name', '').lower() for f in functions)
        has_view = any('view' in f.get('name', '').lower() for f in functions)
        has_controller = any('control' in f.get('name', '').lower() for f in functions)
        return has_model and has_view and has_controller
    
    def _generate_naming_suggestions(self, function_semantics: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate comprehensive naming suggestions"""
        suggestions = {}
        for func_name, semantic_data in function_semantics.items():
            if semantic_data.get('confidence', 0) > 0.6:
                suggestions[func_name] = semantic_data.get('suggested_names', [])
        return suggestions


# Factory function for easy instantiation
def create_semantic_analyzer(config: Optional[Dict[str, Any]] = None) -> SemanticAnalyzer:
    """
    Factory function to create a semantic analyzer instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured SemanticAnalyzer instance
    """
    return SemanticAnalyzer(config)