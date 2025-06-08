"""
ML-Based Code Quality Scoring Engine
Advanced machine learning system for comprehensive code quality assessment.
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    
import re
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
from collections import defaultdict, Counter
import math


class QualityMetric(Enum):
    """Code quality metrics"""
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    SECURITY = "security"
    COMPLEXITY = "complexity"
    DOCUMENTATION = "documentation"
    STANDARDS_COMPLIANCE = "standards_compliance"
    TESTABILITY = "testability"
    REUSABILITY = "reusability"


class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 75-89%
    ACCEPTABLE = "acceptable"   # 60-74%
    POOR = "poor"              # 40-59%
    VERY_POOR = "very_poor"    # 0-39%


class CodeSmell(Enum):
    """Types of code smells"""
    LONG_FUNCTION = "long_function"
    COMPLEX_CONDITIONAL = "complex_conditional"
    DEEP_NESTING = "deep_nesting"
    DUPLICATE_CODE = "duplicate_code"
    MAGIC_NUMBERS = "magic_numbers"
    POOR_NAMING = "poor_naming"
    TOO_MANY_PARAMETERS = "too_many_parameters"
    EXCESSIVE_COUPLING = "excessive_coupling"
    DEAD_CODE = "dead_code"
    INCONSISTENT_STYLE = "inconsistent_style"


@dataclass
class QualityIssue:
    """Represents a code quality issue"""
    smell_type: CodeSmell
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    location: str
    suggestion: str
    impact_score: float


@dataclass
class QualityAssessment:
    """Complete quality assessment result"""
    overall_score: float
    quality_level: QualityLevel
    metric_scores: Dict[QualityMetric, float]
    issues: List[QualityIssue]
    recommendations: List[str]
    strengths: List[str]
    complexity_analysis: Dict[str, Any]
    patterns_detected: List[str]
    confidence: float


class CodeAnalyzer:
    """Analyzes code for quality metrics"""
    
    def __init__(self):
        self.complexity_thresholds = self._build_complexity_thresholds()
        self.quality_patterns = self._build_quality_patterns()
        self.security_patterns = self._build_security_patterns()
        self.performance_patterns = self._build_performance_patterns()
        
    def _build_complexity_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Build thresholds for complexity metrics"""
        return {
            'cyclomatic_complexity': {
                'excellent': 5,
                'good': 10,
                'acceptable': 15,
                'poor': 25
            },
            'function_length': {
                'excellent': 20,
                'good': 50,
                'acceptable': 100,
                'poor': 200
            },
            'nesting_depth': {
                'excellent': 2,
                'good': 3,
                'acceptable': 4,
                'poor': 6
            },
            'parameter_count': {
                'excellent': 3,
                'good': 5,
                'acceptable': 7,
                'poor': 10
            }
        }
    
    def _build_quality_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for quality assessment"""
        return {
            'good_patterns': [
                r'\/\*.*\*\/',  # C-style comments
                r'\/\/.*',      # C++-style comments
                r'const\s+\w+', # Constants usage
                r'static\s+\w+', # Static declarations
                r'return\s+\w+;', # Clear returns
                r'if\s*\(\s*\w+\s*!=\s*NULL\s*\)', # Null checks
                r'assert\s*\(', # Assertions
                r'#define\s+\w+', # Macro definitions
            ],
            'poor_patterns': [
                r'goto\s+\w+',  # Goto statements
                r'\d{3,}',      # Magic numbers (3+ digits)
                r'\/\*.*TODO.*\*\/', # TODO comments
                r'\/\/.*FIXME.*',    # FIXME comments
                r'\/\/.*HACK.*',     # HACK comments
                r'printf\s*\([^)]*\);', # Debug prints
                r'malloc\s*\([^)]*\).*free', # Manual memory management
                r'strcpy\s*\(', # Unsafe string functions
            ],
            'security_risks': [
                r'gets\s*\(',   # Unsafe gets
                r'strcpy\s*\(', # Unsafe strcpy
                r'sprintf\s*\(', # Unsafe sprintf
                r'scanf\s*\(',  # Unsafe scanf
                r'system\s*\(', # System calls
                r'exec\s*\(',   # Exec calls
                r'eval\s*\(',   # Eval calls
            ]
        }
    
    def _build_security_patterns(self) -> Dict[str, List[str]]:
        """Build security-specific patterns"""
        return {
            'buffer_overflow_risks': [
                r'strcpy\s*\(',
                r'strcat\s*\(',
                r'sprintf\s*\(',
                r'gets\s*\(',
                r'scanf\s*\([^)]*%s',
            ],
            'injection_risks': [
                r'system\s*\(',
                r'exec\s*\(',
                r'popen\s*\(',
                r'sql.*\+.*\+',  # String concatenation in SQL
            ],
            'memory_safety': [
                r'malloc\s*\(',
                r'free\s*\(',
                r'realloc\s*\(',
                r'calloc\s*\(',
                r'memcpy\s*\(',
                r'memmove\s*\(',
            ],
            'crypto_issues': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'des\s*\(',
                r'rc4\s*\(',
            ]
        }
    
    def _build_performance_patterns(self) -> Dict[str, List[str]]:
        """Build performance-related patterns"""
        return {
            'inefficient_patterns': [
                r'strlen\s*\([^)]*\)\s*==\s*0',  # strlen for empty check
                r'for\s*\([^;]*strlen\s*\([^)]*\)',  # strlen in loop condition
                r'malloc\s*\(\s*1\s*\)',  # Byte-by-byte allocation
                r'pow\s*\([^,]*,\s*2\s*\)',  # Using pow for squaring
            ],
            'optimization_opportunities': [
                r'\/\*.*OPTIMIZE.*\*\/',
                r'\/\/.*TODO.*optimize',
                r'\/\/.*slow.*',
                r'\/\/.*performance.*',
            ],
            'good_performance': [
                r'inline\s+\w+',
                r'const\s+\w+',
                r'register\s+\w+',
                r'restrict\s+\w+',
                r'__builtin_',
            ]
        }
    
    def analyze_code_structure(self, code_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze structural aspects of code"""
        analysis = {
            'functions': [],
            'total_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'code_lines': 0,
            'average_function_length': 0,
            'max_function_length': 0,
            'total_functions': 0,
            'complexity_distribution': {},
            'naming_quality': 0.0
        }
        
        # Analyze decompiled functions
        functions = code_data.get('decompiled_functions', {})
        if functions:
            function_lengths = []
            complexities = []
            
            for func_name, func_data in functions.items():
                func_analysis = self._analyze_function_structure(func_name, func_data)
                analysis['functions'].append(func_analysis)
                
                function_lengths.append(func_analysis['line_count'])
                complexities.append(func_analysis['cyclomatic_complexity'])
            
            analysis['total_functions'] = len(functions)
            analysis['average_function_length'] = (np.mean(function_lengths) if NUMPY_AVAILABLE and function_lengths 
                                                      else (sum(function_lengths) / len(function_lengths) if function_lengths else 0))
            analysis['max_function_length'] = max(function_lengths) if function_lengths else 0
            
            # Complexity distribution
            analysis['complexity_distribution'] = {
                'low': len([c for c in complexities if c <= 5]),
                'medium': len([c for c in complexities if 6 <= c <= 10]),
                'high': len([c for c in complexities if 11 <= c <= 20]),
                'very_high': len([c for c in complexities if c > 20])
            }
            
            # Naming quality assessment
            analysis['naming_quality'] = self._assess_naming_quality(functions)
        
        return analysis
    
    def _analyze_function_structure(self, func_name: str, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze structure of individual function"""
        instructions = func_data.get('instructions', [])
        if isinstance(instructions, str):
            instructions = instructions.split('\n')
        
        code_lines = [line for line in instructions if line.strip() and not line.strip().startswith(';')]
        
        analysis = {
            'name': func_name,
            'line_count': len(code_lines),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(code_lines),
            'nesting_depth': self._estimate_nesting_depth(code_lines),
            'parameter_count': self._estimate_parameter_count(code_lines),
            'return_count': self._count_returns(code_lines),
            'loop_count': self._count_loops(code_lines),
            'branch_count': self._count_branches(code_lines),
            'function_calls': self._count_function_calls(code_lines),
            'comment_density': self._calculate_comment_density(instructions),
            'variable_count': self._estimate_variable_count(code_lines)
        }
        
        return analysis
    
    def _calculate_cyclomatic_complexity(self, code_lines: List[str]) -> int:
        """Calculate cyclomatic complexity"""
        # Count decision points
        decision_keywords = ['if', 'else', 'while', 'for', 'switch', 'case', 'catch', '&&', '||']
        
        complexity = 1  # Base complexity
        
        for line in code_lines:
            line_lower = line.lower().strip()
            for keyword in decision_keywords:
                if keyword in line_lower:
                    if keyword in ['&&', '||']:
                        complexity += line_lower.count(keyword)
                    else:
                        complexity += 1
        
        return complexity
    
    def _estimate_nesting_depth(self, code_lines: List[str]) -> int:
        """Estimate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        
        for line in code_lines:
            line_stripped = line.strip()
            
            # Count opening braces or control structures
            if any(keyword in line_stripped.lower() for keyword in ['if', 'while', 'for', 'switch']):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            
            # Count closing patterns (simplified)
            if 'ret' in line_stripped.lower() or '}' in line_stripped:
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _estimate_parameter_count(self, code_lines: List[str]) -> int:
        """Estimate parameter count from function prologue"""
        param_count = 0
        
        # Look for parameter access patterns in first 20 lines
        for line in code_lines[:20]:
            if 'ebp+' in line.lower() or 'rbp+' in line.lower():
                # Extract offset to estimate parameter
                offset_match = re.search(r'[er]bp\+(\d+)', line.lower())
                if offset_match:
                    offset = int(offset_match.group(1))
                    if offset >= 8:  # Parameters typically start at ebp+8
                        param_index = (offset - 8) // 4
                        param_count = max(param_count, param_index + 1)
        
        return min(param_count, 10)  # Cap at reasonable number
    
    def _count_returns(self, code_lines: List[str]) -> int:
        """Count return statements"""
        return sum(1 for line in code_lines if 'ret' in line.lower())
    
    def _count_loops(self, code_lines: List[str]) -> int:
        """Count loop structures"""
        loop_count = 0
        labels = {}
        
        # Find labels
        for i, line in enumerate(code_lines):
            if ':' in line:
                label = line.split(':')[0].strip()
                labels[label] = i
        
        # Find backward jumps (loops)
        for i, line in enumerate(code_lines):
            if any(jump in line.lower() for jump in ['jmp', 'je', 'jne', 'jl', 'jg']):
                parts = line.split()
                if len(parts) >= 2:
                    target = parts[-1]
                    if target in labels and labels[target] < i:
                        loop_count += 1
        
        return loop_count
    
    def _count_branches(self, code_lines: List[str]) -> int:
        """Count conditional branches"""
        branch_keywords = ['je', 'jne', 'jz', 'jnz', 'jl', 'jg', 'jle', 'jge']
        return sum(1 for line in code_lines if any(branch in line.lower() for branch in branch_keywords))
    
    def _count_function_calls(self, code_lines: List[str]) -> int:
        """Count function calls"""
        return sum(1 for line in code_lines if 'call' in line.lower())
    
    def _calculate_comment_density(self, all_lines: List[str]) -> float:
        """Calculate comment density"""
        comment_lines = sum(1 for line in all_lines if line.strip().startswith(';') or '//' in line)
        total_lines = len([line for line in all_lines if line.strip()])
        
        return comment_lines / total_lines if total_lines > 0 else 0.0
    
    def _estimate_variable_count(self, code_lines: List[str]) -> int:
        """Estimate number of variables used"""
        # Look for stack variable access patterns
        var_offsets = set()
        
        for line in code_lines:
            # Look for stack variable patterns
            if 'ebp-' in line.lower() or 'rbp-' in line.lower():
                offset_match = re.search(r'[er]bp-(\d+)', line.lower())
                if offset_match:
                    var_offsets.add(offset_match.group(1))
        
        return len(var_offsets)
    
    def _assess_naming_quality(self, functions: Dict[str, Any]) -> float:
        """Assess quality of function names"""
        if not functions:
            return 0.0
        
        total_score = 0.0
        
        for func_name in functions.keys():
            score = self._score_function_name(func_name)
            total_score += score
        
        return total_score / len(functions)
    
    def _score_function_name(self, func_name: str) -> float:
        """Score individual function name quality"""
        if not func_name:
            return 0.0
        
        score = 0.5  # Base score
        
        # Penalty for auto-generated names
        if func_name.startswith(('sub_', 'func_', 'FUN_', 'unnamed')):
            score = 0.1
        elif func_name.startswith(('main', 'init', 'cleanup')):
            score = 0.9
        elif len(func_name) >= 4 and re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', func_name):
            score = 0.8
        elif len(func_name) >= 2:
            score = 0.6
        
        return score


class QualityScorer:
    """ML-based code quality scoring system"""
    
    def __init__(self):
        self.logger = logging.getLogger("QualityScorer")
        self.analyzer = CodeAnalyzer()
        self.metric_weights = self._initialize_metric_weights()
        self.scoring_cache = {}
    
    def _initialize_metric_weights(self) -> Dict[QualityMetric, float]:
        """Initialize weights for different quality metrics"""
        return {
            QualityMetric.READABILITY: 0.20,
            QualityMetric.MAINTAINABILITY: 0.18,
            QualityMetric.COMPLEXITY: 0.15,
            QualityMetric.RELIABILITY: 0.12,
            QualityMetric.PERFORMANCE: 0.10,
            QualityMetric.SECURITY: 0.10,
            QualityMetric.STANDARDS_COMPLIANCE: 0.08,
            QualityMetric.TESTABILITY: 0.04,
            QualityMetric.REUSABILITY: 0.03
        }
    
    def assess_code_quality(self, code_data: Dict[str, Any], context: Dict[str, Any] = None) -> QualityAssessment:
        """Perform comprehensive code quality assessment"""
        cache_key = self._create_cache_key(code_data)
        if cache_key in self.scoring_cache:
            return self.scoring_cache[cache_key]
        
        try:
            # Analyze code structure
            structure_analysis = self.analyzer.analyze_code_structure(code_data)
            
            # Calculate individual metric scores
            metric_scores = {}
            for metric in QualityMetric:
                score = self._calculate_metric_score(metric, code_data, structure_analysis)
                metric_scores[metric] = score
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(metric_scores)
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Detect issues and code smells
            issues = self._detect_quality_issues(code_data, structure_analysis)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metric_scores, issues, structure_analysis)
            
            # Identify strengths
            strengths = self._identify_strengths(metric_scores, structure_analysis)
            
            # Detect patterns
            patterns = self._detect_code_patterns(code_data)
            
            # Calculate confidence
            confidence = self._calculate_assessment_confidence(code_data, structure_analysis)
            
            assessment = QualityAssessment(
                overall_score=overall_score,
                quality_level=quality_level,
                metric_scores=metric_scores,
                issues=issues,
                recommendations=recommendations,
                strengths=strengths,
                complexity_analysis=structure_analysis,
                patterns_detected=patterns,
                confidence=confidence
            )
            
            # Cache result
            self.scoring_cache[cache_key] = assessment
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return QualityAssessment(
                overall_score=0.0,
                quality_level=QualityLevel.VERY_POOR,
                metric_scores={metric: 0.0 for metric in QualityMetric},
                issues=[],
                recommendations=[f"Assessment error: {str(e)}"],
                strengths=[],
                complexity_analysis={},
                patterns_detected=[],
                confidence=0.0
            )
    
    def _create_cache_key(self, code_data: Dict[str, Any]) -> str:
        """Create cache key for quality assessment"""
        key_data = str(code_data)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _calculate_metric_score(self, metric: QualityMetric, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> float:
        """Calculate score for individual quality metric"""
        
        if metric == QualityMetric.READABILITY:
            return self._score_readability(code_data, structure_analysis)
        elif metric == QualityMetric.MAINTAINABILITY:
            return self._score_maintainability(code_data, structure_analysis)
        elif metric == QualityMetric.COMPLEXITY:
            return self._score_complexity(structure_analysis)
        elif metric == QualityMetric.RELIABILITY:
            return self._score_reliability(code_data, structure_analysis)
        elif metric == QualityMetric.PERFORMANCE:
            return self._score_performance(code_data, structure_analysis)
        elif metric == QualityMetric.SECURITY:
            return self._score_security(code_data)
        elif metric == QualityMetric.STANDARDS_COMPLIANCE:
            return self._score_standards_compliance(code_data, structure_analysis)
        elif metric == QualityMetric.TESTABILITY:
            return self._score_testability(structure_analysis)
        elif metric == QualityMetric.REUSABILITY:
            return self._score_reusability(structure_analysis)
        else:
            return 0.5  # Default score
    
    def _score_readability(self, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> float:
        """Score code readability"""
        score = 0.5  # Base score
        
        # Naming quality
        naming_quality = structure_analysis.get('naming_quality', 0.0)
        score += (naming_quality - 0.5) * 0.4
        
        # Comment density
        functions = structure_analysis.get('functions', [])
        if functions:
            densities = [f.get('comment_density', 0) for f in functions]
            avg_comment_density = (np.mean(densities) if NUMPY_AVAILABLE 
                                  else (sum(densities) / len(densities) if densities else 0))
            if 0.1 <= avg_comment_density <= 0.3:  # Optimal comment density
                score += 0.2
            elif avg_comment_density > 0.3:
                score += 0.1  # Too many comments can be bad
        
        # Function length
        avg_length = structure_analysis.get('average_function_length', 0)
        if avg_length <= 20:
            score += 0.2
        elif avg_length <= 50:
            score += 0.1
        elif avg_length > 100:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_maintainability(self, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> float:
        """Score code maintainability"""
        score = 0.5
        
        # Function count and size distribution
        total_functions = structure_analysis.get('total_functions', 0)
        if total_functions > 0:
            avg_length = structure_analysis.get('average_function_length', 0)
            max_length = structure_analysis.get('max_function_length', 0)
            
            if avg_length <= 30 and max_length <= 100:
                score += 0.3
            elif avg_length <= 50 and max_length <= 200:
                score += 0.1
            else:
                score -= 0.2
        
        # Complexity distribution
        complexity_dist = structure_analysis.get('complexity_distribution', {})
        total_complex = complexity_dist.get('high', 0) + complexity_dist.get('very_high', 0)
        total_funcs = sum(complexity_dist.values()) if complexity_dist else 1
        
        complex_ratio = total_complex / total_funcs
        if complex_ratio <= 0.1:
            score += 0.2
        elif complex_ratio <= 0.2:
            score += 0.1
        else:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _score_complexity(self, structure_analysis: Dict[str, Any]) -> float:
        """Score based on code complexity"""
        score = 0.8  # Start with good score
        
        functions = structure_analysis.get('functions', [])
        if not functions:
            return 0.5
        
        # Analyze complexity metrics
        complexities = [f.get('cyclomatic_complexity', 1) for f in functions]
        avg_complexity = (np.mean(complexities) if NUMPY_AVAILABLE and complexities 
                         else (sum(complexities) / len(complexities) if complexities else 1))
        max_complexity = max(complexities)
        
        # Penalty for high complexity
        if avg_complexity > 15:
            score -= 0.4
        elif avg_complexity > 10:
            score -= 0.2
        elif avg_complexity > 5:
            score -= 0.1
        
        if max_complexity > 25:
            score -= 0.3
        elif max_complexity > 15:
            score -= 0.1
        
        # Nesting depth penalty
        max_nesting = max([f.get('nesting_depth', 0) for f in functions])
        if max_nesting > 4:
            score -= 0.2
        elif max_nesting > 3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_reliability(self, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> float:
        """Score code reliability"""
        score = 0.6
        
        # Check for error handling patterns
        code_text = str(code_data)
        
        # Positive patterns
        if re.search(r'error|exception|check|validate', code_text, re.IGNORECASE):
            score += 0.2
        
        if re.search(r'null.*check|assert', code_text, re.IGNORECASE):
            score += 0.1
        
        # Negative patterns
        if re.search(r'goto', code_text, re.IGNORECASE):
            score -= 0.2
        
        if re.search(r'malloc.*free', code_text, re.IGNORECASE):
            # Manual memory management increases risk
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_performance(self, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> float:
        """Score performance characteristics"""
        score = 0.6
        
        code_text = str(code_data)
        
        # Check for performance patterns
        for pattern_type, patterns in self.analyzer.performance_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, code_text, re.IGNORECASE))
                
                if pattern_type == 'inefficient_patterns':
                    score -= min(0.3, matches * 0.1)
                elif pattern_type == 'good_performance':
                    score += min(0.2, matches * 0.05)
        
        return max(0.0, min(1.0, score))
    
    def _score_security(self, code_data: Dict[str, Any]) -> float:
        """Score security aspects"""
        score = 0.8  # Start with good security score
        
        code_text = str(code_data)
        
        # Check for security vulnerabilities
        for vuln_type, patterns in self.analyzer.security_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, code_text, re.IGNORECASE))
                if matches > 0:
                    if vuln_type == 'buffer_overflow_risks':
                        score -= min(0.4, matches * 0.1)
                    elif vuln_type == 'injection_risks':
                        score -= min(0.3, matches * 0.1)
                    elif vuln_type == 'crypto_issues':
                        score -= min(0.2, matches * 0.05)
        
        return max(0.0, min(1.0, score))
    
    def _score_standards_compliance(self, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> float:
        """Score compliance with coding standards"""
        score = 0.6
        
        # Naming conventions
        naming_quality = structure_analysis.get('naming_quality', 0.0)
        score += (naming_quality - 0.5) * 0.4
        
        # Function size standards
        avg_length = structure_analysis.get('average_function_length', 0)
        if avg_length <= 50:
            score += 0.2
        elif avg_length > 100:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _score_testability(self, structure_analysis: Dict[str, Any]) -> float:
        """Score how testable the code is"""
        score = 0.5
        
        functions = structure_analysis.get('functions', [])
        if functions:
            # Smaller functions are more testable
            avg_length = structure_analysis.get('average_function_length', 0)
            if avg_length <= 30:
                score += 0.3
            elif avg_length <= 50:
                score += 0.1
            
            # Lower complexity is more testable
            complexities = [f.get('cyclomatic_complexity', 1) for f in functions]
            avg_complexity = np.mean(complexities)
            if avg_complexity <= 5:
                score += 0.2
            elif avg_complexity <= 10:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_reusability(self, structure_analysis: Dict[str, Any]) -> float:
        """Score code reusability"""
        score = 0.5
        
        # Well-named functions are more reusable
        naming_quality = structure_analysis.get('naming_quality', 0.0)
        score += (naming_quality - 0.5) * 0.3
        
        # Moderate-sized functions are more reusable
        avg_length = structure_analysis.get('average_function_length', 0)
        if 10 <= avg_length <= 50:
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_overall_score(self, metric_scores: Dict[QualityMetric, float]) -> float:
        """Calculate weighted overall quality score"""
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in metric_scores.items():
            weight = self.metric_weights.get(metric, 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level from overall score"""
        if overall_score >= 0.9:
            return QualityLevel.EXCELLENT
        elif overall_score >= 0.75:
            return QualityLevel.GOOD
        elif overall_score >= 0.6:
            return QualityLevel.ACCEPTABLE
        elif overall_score >= 0.4:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _detect_quality_issues(self, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> List[QualityIssue]:
        """Detect specific code quality issues"""
        issues = []
        
        # Check function length issues
        functions = structure_analysis.get('functions', [])
        for func in functions:
            length = func.get('line_count', 0)
            if length > 100:
                issues.append(QualityIssue(
                    smell_type=CodeSmell.LONG_FUNCTION,
                    severity='high' if length > 200 else 'medium',
                    description=f"Function '{func.get('name', 'unknown')}' is {length} lines long",
                    location=func.get('name', 'unknown'),
                    suggestion="Consider breaking down into smaller functions",
                    impact_score=min(1.0, length / 200.0)
                ))
            
            complexity = func.get('cyclomatic_complexity', 1)
            if complexity > 15:
                issues.append(QualityIssue(
                    smell_type=CodeSmell.COMPLEX_CONDITIONAL,
                    severity='high' if complexity > 25 else 'medium',
                    description=f"Function '{func.get('name', 'unknown')}' has complexity {complexity}",
                    location=func.get('name', 'unknown'),
                    suggestion="Simplify conditional logic and reduce branching",
                    impact_score=min(1.0, complexity / 25.0)
                ))
            
            nesting = func.get('nesting_depth', 0)
            if nesting > 3:
                issues.append(QualityIssue(
                    smell_type=CodeSmell.DEEP_NESTING,
                    severity='medium' if nesting > 4 else 'low',
                    description=f"Function '{func.get('name', 'unknown')}' has nesting depth {nesting}",
                    location=func.get('name', 'unknown'),
                    suggestion="Reduce nesting by using early returns or helper functions",
                    impact_score=min(1.0, nesting / 6.0)
                ))
            
            params = func.get('parameter_count', 0)
            if params > 5:
                issues.append(QualityIssue(
                    smell_type=CodeSmell.TOO_MANY_PARAMETERS,
                    severity='medium' if params > 7 else 'low',
                    description=f"Function '{func.get('name', 'unknown')}' has {params} parameters",
                    location=func.get('name', 'unknown'),
                    suggestion="Consider using a struct or reducing parameter count",
                    impact_score=min(1.0, params / 10.0)
                ))
        
        # Check naming issues
        naming_quality = structure_analysis.get('naming_quality', 0.0)
        if naming_quality < 0.5:
            issues.append(QualityIssue(
                smell_type=CodeSmell.POOR_NAMING,
                severity='medium',
                description=f"Poor naming quality score: {naming_quality:.2f}",
                location='global',
                suggestion="Improve function and variable naming conventions",
                impact_score=1.0 - naming_quality
            ))
        
        return issues
    
    def _generate_recommendations(self, metric_scores: Dict[QualityMetric, float], issues: List[QualityIssue], structure_analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Low-scoring metrics
        for metric, score in metric_scores.items():
            if score < 0.6:
                if metric == QualityMetric.COMPLEXITY:
                    recommendations.append("Reduce code complexity by breaking down large functions")
                elif metric == QualityMetric.READABILITY:
                    recommendations.append("Improve code readability with better naming and comments")
                elif metric == QualityMetric.MAINTAINABILITY:
                    recommendations.append("Enhance maintainability by reducing function sizes and complexity")
                elif metric == QualityMetric.SECURITY:
                    recommendations.append("Address security vulnerabilities in buffer handling and input validation")
                elif metric == QualityMetric.PERFORMANCE:
                    recommendations.append("Optimize performance by addressing inefficient patterns")
        
        # Issue-specific recommendations
        high_severity_issues = [issue for issue in issues if issue.severity == 'high']
        if high_severity_issues:
            recommendations.append(f"Address {len(high_severity_issues)} high-severity code quality issues")
        
        # Structure-specific recommendations
        avg_length = structure_analysis.get('average_function_length', 0)
        if avg_length > 50:
            recommendations.append("Consider breaking down large functions for better maintainability")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _identify_strengths(self, metric_scores: Dict[QualityMetric, float], structure_analysis: Dict[str, Any]) -> List[str]:
        """Identify code strengths"""
        strengths = []
        
        # High-scoring metrics
        for metric, score in metric_scores.items():
            if score >= 0.8:
                if metric == QualityMetric.COMPLEXITY:
                    strengths.append("Well-controlled code complexity")
                elif metric == QualityMetric.READABILITY:
                    strengths.append("Good code readability and structure")
                elif metric == QualityMetric.SECURITY:
                    strengths.append("Strong security practices")
                elif metric == QualityMetric.PERFORMANCE:
                    strengths.append("Performance-optimized code patterns")
        
        # Structure strengths
        complexity_dist = structure_analysis.get('complexity_distribution', {})
        if complexity_dist.get('low', 0) > complexity_dist.get('high', 0):
            strengths.append("Most functions have low complexity")
        
        naming_quality = structure_analysis.get('naming_quality', 0.0)
        if naming_quality >= 0.8:
            strengths.append("Excellent naming conventions")
        
        return strengths
    
    def _detect_code_patterns(self, code_data: Dict[str, Any]) -> List[str]:
        """Detect notable code patterns"""
        patterns = []
        code_text = str(code_data)
        
        # Design patterns
        if re.search(r'singleton|factory|observer|strategy', code_text, re.IGNORECASE):
            patterns.append("Design patterns detected")
        
        # Error handling
        if re.search(r'try.*catch|error.*handling', code_text, re.IGNORECASE):
            patterns.append("Error handling implemented")
        
        # Memory management
        if re.search(r'malloc.*free|new.*delete', code_text, re.IGNORECASE):
            patterns.append("Manual memory management")
        
        # Optimization
        if re.search(r'inline|const|register', code_text, re.IGNORECASE):
            patterns.append("Performance optimizations")
        
        return patterns
    
    def _calculate_assessment_confidence(self, code_data: Dict[str, Any], structure_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the assessment"""
        confidence = 0.5
        
        # More functions = higher confidence
        total_functions = structure_analysis.get('total_functions', 0)
        if total_functions >= 10:
            confidence += 0.3
        elif total_functions >= 5:
            confidence += 0.2
        elif total_functions >= 1:
            confidence += 0.1
        
        # Presence of analysis data
        if structure_analysis.get('functions'):
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def get_scorer_stats(self) -> Dict[str, Any]:
        """Get quality scorer statistics"""
        return {
            'supported_metrics': [m.value for m in QualityMetric],
            'quality_levels': [l.value for l in QualityLevel],
            'code_smells': [s.value for s in CodeSmell],
            'cached_assessments': len(self.scoring_cache),
            'metric_weights': {m.value: w for m, w in self.metric_weights.items()}
        }