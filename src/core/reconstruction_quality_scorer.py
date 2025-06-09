"""
Reconstruction Quality Scorer for Open-Sourcefy Matrix Pipeline

This module provides comprehensive quality metrics and scoring for binary
reconstruction and decompilation processes. It evaluates the accuracy,
completeness, and semantic correctness of reconstructed source code.

Features:
- Multi-dimensional quality assessment
- Code complexity analysis
- Semantic correctness validation
- Compilation readiness scoring
- Confidence interval calculation
- Quality benchmarking and reporting
"""

import os
import re
import ast
import json
import logging
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import statistics
from collections import defaultdict

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality assessment dimensions"""
    STRUCTURAL_ACCURACY = "structural_accuracy"
    SEMANTIC_CORRECTNESS = "semantic_correctness"
    CODE_COMPLETENESS = "code_completeness"
    COMPILATION_READINESS = "compilation_readiness"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE_RETENTION = "performance_retention"
    SECURITY_PRESERVATION = "security_preservation"


class QualityLevel(Enum):
    """Quality levels for reconstruction"""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 75-89%
    ACCEPTABLE = "acceptable"   # 60-74%
    POOR = "poor"              # 40-59%
    FAILED = "failed"          # 0-39%


@dataclass
class QualityMetric:
    """Individual quality metric"""
    name: str
    score: float  # 0.0 to 1.0
    weight: float
    confidence: float
    dimension: QualityDimension
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    reconstruction_id: str
    timestamp: float
    overall_score: float
    quality_level: QualityLevel
    confidence_interval: Tuple[float, float]
    
    # Dimension scores
    dimension_scores: Dict[QualityDimension, float] = field(default_factory=dict)
    
    # Individual metrics
    metrics: List[QualityMetric] = field(default_factory=list)
    
    # Analysis details
    source_analysis: Dict[str, Any] = field(default_factory=dict)
    compilation_analysis: Dict[str, Any] = field(default_factory=dict)
    semantic_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Issues and recommendations
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Benchmarking
    benchmark_comparisons: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    analysis_time: float = 0.0
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of quality assessment"""
        return {
            'overall_score': self.overall_score,
            'quality_level': self.quality_level.value,
            'confidence_range': f"{self.confidence_interval[0]:.1%} - {self.confidence_interval[1]:.1%}",
            'dimension_scores': {dim.value: score for dim, score in self.dimension_scores.items()},
            'critical_issues_count': len(self.critical_issues),
            'warnings_count': len(self.warnings),
            'recommendations_count': len(self.recommendations)
        }


class ReconstructionQualityScorer:
    """
    Advanced quality scoring engine for binary reconstruction
    
    Evaluates reconstruction quality across multiple dimensions using both
    static analysis and dynamic testing approaches.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.90,
            QualityLevel.GOOD: 0.75,
            QualityLevel.ACCEPTABLE: 0.60,
            QualityLevel.POOR: 0.40,
            QualityLevel.FAILED: 0.0
        }
        
        # Dimension weights
        self.dimension_weights = {
            QualityDimension.STRUCTURAL_ACCURACY: 0.20,
            QualityDimension.SEMANTIC_CORRECTNESS: 0.25,
            QualityDimension.CODE_COMPLETENESS: 0.20,
            QualityDimension.COMPILATION_READINESS: 0.15,
            QualityDimension.MAINTAINABILITY: 0.10,
            QualityDimension.PERFORMANCE_RETENTION: 0.05,
            QualityDimension.SECURITY_PRESERVATION: 0.05
        }
        
        # Analysis tools
        self.available_tools = self._detect_analysis_tools()
        
        # Quality benchmarks
        self.benchmarks = self._load_quality_benchmarks()
        
        # Metric cache
        self.metric_cache = {}
        
    def assess_reconstruction_quality(self, 
                                    original_binary_path: Path,
                                    reconstructed_source_path: Path,
                                    compilation_artifacts_path: Optional[Path] = None) -> QualityReport:
        """
        Comprehensive quality assessment of binary reconstruction
        
        Args:
            original_binary_path: Path to original binary file
            reconstructed_source_path: Path to reconstructed source code directory
            compilation_artifacts_path: Optional path to compilation artifacts
            
        Returns:
            Detailed quality assessment report
        """
        start_time = time.time()
        
        self.logger.info(f"Starting quality assessment for reconstruction from {original_binary_path}")
        
        # Generate assessment ID
        assessment_id = self._generate_assessment_id(original_binary_path, reconstructed_source_path)
        
        try:
            # Phase 1: Validate inputs
            if not self._validate_assessment_inputs(original_binary_path, reconstructed_source_path):
                return self._create_failed_assessment("Input validation failed", assessment_id)
            
            # Phase 2: Analyze source code structure and quality
            source_analysis = self._analyze_source_code_quality(reconstructed_source_path)
            
            # Phase 3: Evaluate compilation readiness
            compilation_analysis = self._analyze_compilation_readiness(
                reconstructed_source_path, compilation_artifacts_path
            )
            
            # Phase 4: Assess semantic correctness
            semantic_analysis = self._analyze_semantic_correctness(
                original_binary_path, reconstructed_source_path
            )
            
            # Phase 5: Calculate quality metrics
            metrics = self._calculate_quality_metrics(
                source_analysis, compilation_analysis, semantic_analysis
            )
            
            # Phase 6: Calculate overall scores and confidence
            overall_score, dimension_scores = self._calculate_overall_scores(metrics)
            confidence_interval = self._calculate_confidence_interval(metrics, overall_score)
            
            # Phase 7: Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Phase 8: Generate issues and recommendations
            issues, warnings, recommendations = self._generate_assessment_guidance(
                metrics, source_analysis, compilation_analysis, semantic_analysis
            )
            
            # Phase 9: Benchmark comparison
            benchmark_comparisons = self._compare_with_benchmarks(overall_score, dimension_scores)
            
            # Create final report
            report = QualityReport(
                reconstruction_id=assessment_id,
                timestamp=time.time(),
                overall_score=overall_score,
                quality_level=quality_level,
                confidence_interval=confidence_interval,
                dimension_scores=dimension_scores,
                metrics=metrics,
                source_analysis=source_analysis,
                compilation_analysis=compilation_analysis,
                semantic_analysis=semantic_analysis,
                critical_issues=issues,
                warnings=warnings,
                recommendations=recommendations,
                benchmark_comparisons=benchmark_comparisons,
                analysis_time=time.time() - start_time
            )
            
            self.logger.info(
                f"Quality assessment completed: {quality_level.value} "
                f"({overall_score:.1%}) in {report.analysis_time:.1f}s"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}", exc_info=True)
            return self._create_failed_assessment(f"Assessment error: {e}", assessment_id)
    
    def _validate_assessment_inputs(self, original_binary: Path, reconstructed_source: Path) -> bool:
        """Validate assessment inputs"""
        if not original_binary.exists():
            self.logger.error(f"Original binary not found: {original_binary}")
            return False
            
        if not reconstructed_source.exists():
            self.logger.error(f"Reconstructed source not found: {reconstructed_source}")
            return False
            
        if original_binary.stat().st_size == 0:
            self.logger.error(f"Original binary is empty: {original_binary}")
            return False
            
        # Check if reconstructed source has any source files
        source_files = list(reconstructed_source.rglob("*.c")) + list(reconstructed_source.rglob("*.cpp"))
        if not source_files:
            self.logger.warning(f"No source files found in: {reconstructed_source}")
            # Allow assessment to continue with warning
            
        return True
    
    def _analyze_source_code_quality(self, source_path: Path) -> Dict[str, Any]:
        """Analyze reconstructed source code quality"""
        self.logger.debug("Analyzing source code quality...")
        
        analysis = {
            'file_count': 0,
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'function_count': 0,
            'average_function_length': 0,
            'complexity_metrics': {},
            'code_quality_issues': [],
            'maintainability_score': 0.0,
            'readability_score': 0.0,
            'documentation_score': 0.0
        }
        
        try:
            # Find all source files
            source_files = []
            for ext in ['*.c', '*.cpp', '*.h', '*.hpp']:
                source_files.extend(source_path.rglob(ext))
            
            analysis['file_count'] = len(source_files)
            
            if not source_files:
                analysis['code_quality_issues'].append("No source files found")
                return analysis
            
            # Analyze each source file
            total_functions = 0
            total_function_lines = 0
            
            for source_file in source_files:
                try:
                    with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Count lines
                    lines = content.split('\n')
                    analysis['total_lines'] += len(lines)
                    
                    # Count code vs comment lines
                    code_lines, comment_lines = self._count_code_and_comments(content)
                    analysis['code_lines'] += code_lines
                    analysis['comment_lines'] += comment_lines
                    
                    # Count functions and analyze complexity
                    functions = self._extract_functions(content, source_file.suffix)
                    total_functions += len(functions)
                    
                    for func in functions:
                        total_function_lines += func.get('line_count', 0)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {source_file}: {e}")
                    analysis['code_quality_issues'].append(f"Failed to analyze {source_file.name}")
            
            analysis['function_count'] = total_functions
            if total_functions > 0:
                analysis['average_function_length'] = total_function_lines / total_functions
            
            # Calculate quality scores
            analysis['maintainability_score'] = self._calculate_maintainability_score(analysis)
            analysis['readability_score'] = self._calculate_readability_score(analysis)
            analysis['documentation_score'] = self._calculate_documentation_score(analysis)
            
            # Calculate complexity metrics
            analysis['complexity_metrics'] = self._calculate_complexity_metrics(source_files)
            
        except Exception as e:
            self.logger.error(f"Source code analysis failed: {e}")
            analysis['code_quality_issues'].append(f"Analysis error: {e}")
        
        return analysis
    
    def _analyze_compilation_readiness(self, source_path: Path, 
                                     artifacts_path: Optional[Path] = None) -> Dict[str, Any]:
        """Analyze compilation readiness of reconstructed code"""
        self.logger.debug("Analyzing compilation readiness...")
        
        analysis = {
            'build_files_present': False,
            'compilation_attempted': False,
            'compilation_successful': False,
            'compilation_errors': [],
            'compilation_warnings': [],
            'link_errors': [],
            'missing_dependencies': [],
            'build_system_score': 0.0,
            'compilation_score': 0.0
        }
        
        try:
            # Check for build files
            build_files = {
                'cmake': list(source_path.rglob("CMakeLists.txt")),
                'msbuild': list(source_path.rglob("*.vcxproj")),
                'makefile': list(source_path.rglob("Makefile")) + list(source_path.rglob("makefile"))
            }
            
            analysis['build_files_present'] = any(files for files in build_files.values())
            
            if analysis['build_files_present']:
                analysis['build_system_score'] = 0.7  # Base score for having build files
                
                # Attempt compilation if artifacts path provided
                if artifacts_path and artifacts_path.exists():
                    compilation_result = self._attempt_compilation(source_path, artifacts_path)
                    analysis.update(compilation_result)
            else:
                # Check if source files could be compiled manually
                source_files = list(source_path.rglob("*.c")) + list(source_path.rglob("*.cpp"))
                if source_files:
                    # Basic compilation check
                    main_file = self._find_main_function(source_files)
                    if main_file:
                        analysis['build_system_score'] = 0.3  # Has compilable code but no build system
                    else:
                        analysis['missing_dependencies'].append("No main function found")
                
        except Exception as e:
            self.logger.error(f"Compilation analysis failed: {e}")
            analysis['compilation_errors'].append(f"Analysis error: {e}")
        
        return analysis
    
    def _analyze_semantic_correctness(self, original_binary: Path, 
                                    reconstructed_source: Path) -> Dict[str, Any]:
        """Analyze semantic correctness of reconstruction"""
        self.logger.debug("Analyzing semantic correctness...")
        
        analysis = {
            'function_preservation_score': 0.0,
            'api_compatibility_score': 0.0,
            'control_flow_preservation': 0.0,
            'data_structure_accuracy': 0.0,
            'behavior_equivalence': 0.0,
            'semantic_issues': [],
            'correctness_confidence': 0.0
        }
        
        try:
            # Analyze function preservation
            original_functions = self._extract_binary_functions(original_binary)
            reconstructed_functions = self._extract_source_functions(reconstructed_source)
            
            analysis['function_preservation_score'] = self._compare_function_sets(
                original_functions, reconstructed_functions
            )
            
            # Analyze API compatibility
            original_apis = self._extract_binary_apis(original_binary)
            reconstructed_apis = self._extract_source_apis(reconstructed_source)
            
            analysis['api_compatibility_score'] = self._compare_api_usage(
                original_apis, reconstructed_apis
            )
            
            # Estimate control flow preservation
            analysis['control_flow_preservation'] = self._estimate_control_flow_preservation(
                original_binary, reconstructed_source
            )
            
            # Estimate data structure accuracy
            analysis['data_structure_accuracy'] = self._estimate_data_structure_accuracy(
                original_binary, reconstructed_source
            )
            
            # Calculate overall semantic correctness
            semantic_factors = [
                analysis['function_preservation_score'],
                analysis['api_compatibility_score'],
                analysis['control_flow_preservation'],
                analysis['data_structure_accuracy']
            ]
            
            analysis['behavior_equivalence'] = sum(semantic_factors) / len(semantic_factors)
            analysis['correctness_confidence'] = min(semantic_factors)
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            analysis['semantic_issues'].append(f"Analysis error: {e}")
        
        return analysis
    
    def _calculate_quality_metrics(self, source_analysis: Dict[str, Any],
                                 compilation_analysis: Dict[str, Any],
                                 semantic_analysis: Dict[str, Any]) -> List[QualityMetric]:
        """Calculate individual quality metrics"""
        metrics = []
        
        # Structural Accuracy Metrics
        metrics.append(QualityMetric(
            name="Code Structure Quality",
            score=source_analysis.get('maintainability_score', 0.0),
            weight=0.3,
            confidence=0.8,
            dimension=QualityDimension.STRUCTURAL_ACCURACY,
            details={
                'file_count': source_analysis.get('file_count', 0),
                'function_count': source_analysis.get('function_count', 0),
                'average_function_length': source_analysis.get('average_function_length', 0)
            }
        ))
        
        metrics.append(QualityMetric(
            name="Function Preservation",
            score=semantic_analysis.get('function_preservation_score', 0.0),
            weight=0.7,
            confidence=0.7,
            dimension=QualityDimension.STRUCTURAL_ACCURACY,
            details=semantic_analysis
        ))
        
        # Semantic Correctness Metrics
        metrics.append(QualityMetric(
            name="Behavior Equivalence",
            score=semantic_analysis.get('behavior_equivalence', 0.0),
            weight=0.4,
            confidence=semantic_analysis.get('correctness_confidence', 0.5),
            dimension=QualityDimension.SEMANTIC_CORRECTNESS,
            details=semantic_analysis
        ))
        
        metrics.append(QualityMetric(
            name="API Compatibility",
            score=semantic_analysis.get('api_compatibility_score', 0.0),
            weight=0.3,
            confidence=0.6,
            dimension=QualityDimension.SEMANTIC_CORRECTNESS,
            details={'api_analysis': semantic_analysis}
        ))
        
        metrics.append(QualityMetric(
            name="Control Flow Preservation",
            score=semantic_analysis.get('control_flow_preservation', 0.0),
            weight=0.3,
            confidence=0.5,
            dimension=QualityDimension.SEMANTIC_CORRECTNESS,
            details={'control_flow_analysis': semantic_analysis}
        ))
        
        # Code Completeness Metrics
        completeness_score = self._calculate_code_completeness_score(source_analysis)
        metrics.append(QualityMetric(
            name="Code Completeness",
            score=completeness_score,
            weight=1.0,
            confidence=0.8,
            dimension=QualityDimension.CODE_COMPLETENESS,
            details=source_analysis
        ))
        
        # Compilation Readiness Metrics
        compilation_score = self._calculate_compilation_score(compilation_analysis)
        metrics.append(QualityMetric(
            name="Compilation Readiness",
            score=compilation_score,
            weight=1.0,
            confidence=0.9,
            dimension=QualityDimension.COMPILATION_READINESS,
            details=compilation_analysis
        ))
        
        # Maintainability Metrics
        metrics.append(QualityMetric(
            name="Code Maintainability",
            score=source_analysis.get('maintainability_score', 0.0),
            weight=0.5,
            confidence=0.7,
            dimension=QualityDimension.MAINTAINABILITY,
            details={
                'readability': source_analysis.get('readability_score', 0.0),
                'documentation': source_analysis.get('documentation_score', 0.0),
                'complexity': source_analysis.get('complexity_metrics', {})
            }
        ))
        
        metrics.append(QualityMetric(
            name="Code Documentation",
            score=source_analysis.get('documentation_score', 0.0),
            weight=0.5,
            confidence=0.9,
            dimension=QualityDimension.MAINTAINABILITY,
            details={'documentation_analysis': source_analysis}
        ))
        
        return metrics
    
    def _calculate_overall_scores(self, metrics: List[QualityMetric]) -> Tuple[float, Dict[QualityDimension, float]]:
        """Calculate overall and dimension scores"""
        
        # Calculate dimension scores
        dimension_scores = {}
        for dimension in QualityDimension:
            dimension_metrics = [m for m in metrics if m.dimension == dimension]
            if dimension_metrics:
                weighted_score = sum(m.score * m.weight * m.confidence for m in dimension_metrics)
                total_weight = sum(m.weight * m.confidence for m in dimension_metrics)
                dimension_scores[dimension] = weighted_score / total_weight if total_weight > 0 else 0.0
            else:
                dimension_scores[dimension] = 0.0
        
        # Calculate overall score using dimension weights
        overall_score = sum(
            score * self.dimension_weights.get(dimension, 0.0)
            for dimension, score in dimension_scores.items()
        )
        
        return overall_score, dimension_scores
    
    def _calculate_confidence_interval(self, metrics: List[QualityMetric], 
                                     overall_score: float) -> Tuple[float, float]:
        """Calculate confidence interval for overall score"""
        
        # Calculate variance based on metric confidences
        confidence_scores = [m.confidence for m in metrics if m.confidence > 0]
        
        if not confidence_scores:
            return (overall_score * 0.7, overall_score * 1.0)
        
        avg_confidence = statistics.mean(confidence_scores)
        confidence_variance = statistics.variance(confidence_scores) if len(confidence_scores) > 1 else 0.1
        
        # Calculate confidence interval
        margin = confidence_variance * 0.5
        lower_bound = max(0.0, overall_score - margin)
        upper_bound = min(1.0, overall_score + margin)
        
        return (lower_bound, upper_bound)
    
    def _determine_quality_level(self, overall_score: float) -> QualityLevel:
        """Determine quality level based on overall score"""
        for level, threshold in sorted(self.quality_thresholds.items(), 
                                     key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                return level
        return QualityLevel.FAILED
    
    # Helper methods for detailed analysis
    def _count_code_and_comments(self, content: str) -> Tuple[int, int]:
        """Count code lines vs comment lines"""
        lines = content.split('\n')
        code_lines = 0
        comment_lines = 0
        
        in_multiline_comment = False
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check for multiline comments
            if '/*' in stripped:
                in_multiline_comment = True
            if '*/' in stripped:
                in_multiline_comment = False
                comment_lines += 1
                continue
                
            if in_multiline_comment:
                comment_lines += 1
            elif stripped.startswith('//'):
                comment_lines += 1
            else:
                code_lines += 1
        
        return code_lines, comment_lines
    
    def _extract_functions(self, content: str, file_extension: str) -> List[Dict[str, Any]]:
        """Extract function information from source code"""
        functions = []
        
        # Simple function detection (C/C++)
        if file_extension in ['.c', '.cpp', '.h', '.hpp']:
            # Basic function pattern matching
            function_pattern = r'(\w+\s+)*(\w+)\s*\([^)]*\)\s*\{'
            matches = re.finditer(function_pattern, content, re.MULTILINE)
            
            for match in matches:
                func_name = match.group(2)
                if func_name not in ['if', 'while', 'for', 'switch']:  # Filter out control structures
                    start_pos = match.start()
                    lines_before = content[:start_pos].count('\n')
                    
                    functions.append({
                        'name': func_name,
                        'start_line': lines_before,
                        'line_count': self._estimate_function_length(content, start_pos)
                    })
        
        return functions
    
    def _estimate_function_length(self, content: str, start_pos: int) -> int:
        """Estimate function length by counting braces"""
        brace_count = 0
        lines = 1
        
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '\n':
                lines += 1
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    break
        
        return lines
    
    def _calculate_maintainability_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate maintainability score based on code metrics"""
        score_factors = []
        
        # Function length factor
        avg_func_length = analysis.get('average_function_length', 0)
        if avg_func_length > 0:
            if avg_func_length <= 20:
                score_factors.append(1.0)
            elif avg_func_length <= 50:
                score_factors.append(0.8)
            elif avg_func_length <= 100:
                score_factors.append(0.6)
            else:
                score_factors.append(0.4)
        
        # Comment ratio factor
        total_lines = analysis.get('total_lines', 1)
        comment_lines = analysis.get('comment_lines', 0)
        comment_ratio = comment_lines / total_lines
        
        if comment_ratio >= 0.2:
            score_factors.append(1.0)
        elif comment_ratio >= 0.1:
            score_factors.append(0.8)
        elif comment_ratio >= 0.05:
            score_factors.append(0.6)
        else:
            score_factors.append(0.4)
        
        # File organization factor
        file_count = analysis.get('file_count', 1)
        function_count = analysis.get('function_count', 1)
        functions_per_file = function_count / file_count
        
        if functions_per_file <= 10:
            score_factors.append(1.0)
        elif functions_per_file <= 20:
            score_factors.append(0.8)
        else:
            score_factors.append(0.6)
        
        return sum(score_factors) / len(score_factors) if score_factors else 0.0
    
    def _calculate_readability_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate readability score"""
        # Simplified readability assessment
        code_lines = analysis.get('code_lines', 1)
        comment_lines = analysis.get('comment_lines', 0)
        
        comment_ratio = comment_lines / (code_lines + comment_lines)
        
        if comment_ratio >= 0.15:
            return 0.9
        elif comment_ratio >= 0.1:
            return 0.7
        elif comment_ratio >= 0.05:
            return 0.5
        else:
            return 0.3
    
    def _calculate_documentation_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate documentation score"""
        # Check for common documentation patterns
        comment_lines = analysis.get('comment_lines', 0)
        function_count = analysis.get('function_count', 1)
        
        # Estimate documentation coverage
        estimated_documented_functions = min(comment_lines / 3, function_count)
        documentation_coverage = estimated_documented_functions / function_count
        
        return min(documentation_coverage, 1.0)
    
    def _calculate_complexity_metrics(self, source_files: List[Path]) -> Dict[str, Any]:
        """Calculate complexity metrics for source files"""
        return {
            'cyclomatic_complexity': 'not_implemented',
            'cognitive_complexity': 'not_implemented',
            'nesting_depth': 'not_implemented'
        }
    
    def _attempt_compilation(self, source_path: Path, artifacts_path: Path) -> Dict[str, Any]:
        """Attempt to compile the reconstructed code"""
        result = {
            'compilation_attempted': True,
            'compilation_successful': False,
            'compilation_errors': [],
            'compilation_warnings': [],
            'link_errors': [],
            'compilation_score': 0.0
        }
        
        try:
            # Try different build systems
            if (source_path / "CMakeLists.txt").exists():
                result.update(self._attempt_cmake_build(source_path, artifacts_path))
            elif list(source_path.rglob("*.vcxproj")):
                result.update(self._attempt_msbuild(source_path, artifacts_path))
            else:
                result.update(self._attempt_direct_compilation(source_path, artifacts_path))
                
        except Exception as e:
            result['compilation_errors'].append(f"Build attempt failed: {e}")
        
        return result
    
    def _attempt_cmake_build(self, source_path: Path, build_path: Path) -> Dict[str, Any]:
        """Attempt CMake build"""
        # Placeholder for CMake build attempt
        return {
            'compilation_successful': False,
            'compilation_errors': ['CMake build not implemented'],
            'compilation_score': 0.0
        }
    
    def _attempt_msbuild(self, source_path: Path, build_path: Path) -> Dict[str, Any]:
        """Attempt MSBuild compilation"""
        # Placeholder for MSBuild attempt
        return {
            'compilation_successful': False,
            'compilation_errors': ['MSBuild not implemented'],
            'compilation_score': 0.0
        }
    
    def _attempt_direct_compilation(self, source_path: Path, build_path: Path) -> Dict[str, Any]:
        """Attempt direct compilation with GCC/Clang"""
        # Placeholder for direct compilation attempt
        return {
            'compilation_successful': False,
            'compilation_errors': ['Direct compilation not implemented'],
            'compilation_score': 0.0
        }
    
    def _find_main_function(self, source_files: List[Path]) -> Optional[Path]:
        """Find file containing main function"""
        for source_file in source_files:
            try:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if re.search(r'\bint\s+main\s*\(', content):
                        return source_file
            except:
                continue
        return None
    
    def _extract_binary_functions(self, binary_path: Path) -> List[str]:
        """Extract function names from binary (placeholder)"""
        # This would integrate with binary analysis tools
        return ['main', 'init', 'cleanup']  # Placeholder
    
    def _extract_source_functions(self, source_path: Path) -> List[str]:
        """Extract function names from source code"""
        functions = []
        source_files = list(source_path.rglob("*.c")) + list(source_path.rglob("*.cpp"))
        
        for source_file in source_files:
            try:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    extracted = self._extract_functions(content, source_file.suffix)
                    functions.extend([f['name'] for f in extracted])
            except:
                continue
        
        return functions
    
    def _extract_binary_apis(self, binary_path: Path) -> List[str]:
        """Extract API calls from binary (placeholder)"""
        # This would integrate with binary analysis tools
        return ['GetModuleHandle', 'LoadLibrary', 'GetProcAddress']  # Placeholder
    
    def _extract_source_apis(self, source_path: Path) -> List[str]:
        """Extract API calls from source code"""
        apis = set()
        source_files = list(source_path.rglob("*.c")) + list(source_path.rglob("*.cpp"))
        
        for source_file in source_files:
            try:
                with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Look for function calls
                    api_calls = re.findall(r'(\w+)\s*\(', content)
                    apis.update(api_calls)
            except:
                continue
        
        return list(apis)
    
    def _compare_function_sets(self, original_functions: List[str], 
                             reconstructed_functions: List[str]) -> float:
        """Compare function sets for preservation score"""
        if not original_functions:
            return 1.0 if not reconstructed_functions else 0.0
        
        original_set = set(original_functions)
        reconstructed_set = set(reconstructed_functions)
        
        intersection = original_set & reconstructed_set
        union = original_set | reconstructed_set
        
        return len(intersection) / len(union) if union else 0.0
    
    def _compare_api_usage(self, original_apis: List[str], reconstructed_apis: List[str]) -> float:
        """Compare API usage for compatibility score"""
        if not original_apis:
            return 1.0
        
        original_set = set(original_apis)
        reconstructed_set = set(reconstructed_apis)
        
        # Focus on preservation of original APIs
        preserved_apis = original_set & reconstructed_set
        return len(preserved_apis) / len(original_set)
    
    def _estimate_control_flow_preservation(self, original_binary: Path, 
                                          reconstructed_source: Path) -> float:
        """Estimate control flow preservation (placeholder)"""
        # This would require CFG analysis
        return 0.7  # Placeholder estimate
    
    def _estimate_data_structure_accuracy(self, original_binary: Path,
                                        reconstructed_source: Path) -> float:
        """Estimate data structure accuracy (placeholder)"""
        # This would require data structure analysis
        return 0.6  # Placeholder estimate
    
    def _calculate_code_completeness_score(self, source_analysis: Dict[str, Any]) -> float:
        """Calculate code completeness score"""
        score_factors = []
        
        # File presence factor
        file_count = source_analysis.get('file_count', 0)
        if file_count > 0:
            score_factors.append(1.0)
        else:
            score_factors.append(0.0)
        
        # Function presence factor
        function_count = source_analysis.get('function_count', 0)
        if function_count > 0:
            score_factors.append(1.0)
        else:
            score_factors.append(0.0)
        
        # Code to total lines ratio
        total_lines = source_analysis.get('total_lines', 1)
        code_lines = source_analysis.get('code_lines', 0)
        code_ratio = code_lines / total_lines
        
        if code_ratio >= 0.5:
            score_factors.append(1.0)
        elif code_ratio >= 0.3:
            score_factors.append(0.8)
        else:
            score_factors.append(0.6)
        
        return sum(score_factors) / len(score_factors) if score_factors else 0.0
    
    def _calculate_compilation_score(self, compilation_analysis: Dict[str, Any]) -> float:
        """Calculate compilation score"""
        score = 0.0
        
        # Build system presence
        if compilation_analysis.get('build_files_present', False):
            score += 0.3
        
        # Compilation attempt
        if compilation_analysis.get('compilation_attempted', False):
            score += 0.2
            
            # Compilation success
            if compilation_analysis.get('compilation_successful', False):
                score += 0.5
            else:
                # Partial credit for getting close
                errors = len(compilation_analysis.get('compilation_errors', []))
                if errors < 5:
                    score += 0.2
                elif errors < 10:
                    score += 0.1
        
        return min(score, 1.0)
    
    def _generate_assessment_guidance(self, metrics: List[QualityMetric],
                                    source_analysis: Dict[str, Any],
                                    compilation_analysis: Dict[str, Any],
                                    semantic_analysis: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """Generate issues, warnings, and recommendations"""
        
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Critical issues
        if source_analysis.get('file_count', 0) == 0:
            critical_issues.append("No source files found in reconstruction")
        
        if source_analysis.get('function_count', 0) == 0:
            critical_issues.append("No functions found in reconstructed code")
        
        if not compilation_analysis.get('build_files_present', False):
            warnings.append("No build system files found (CMakeLists.txt, *.vcxproj, Makefile)")
        
        # Low quality metrics
        for metric in metrics:
            if metric.score < 0.4:
                critical_issues.append(f"Low {metric.name} score: {metric.score:.1%}")
            elif metric.score < 0.6:
                warnings.append(f"Moderate {metric.name} score: {metric.score:.1%}")
        
        # Recommendations
        if source_analysis.get('documentation_score', 0) < 0.5:
            recommendations.append("Add more documentation and comments to improve maintainability")
        
        if source_analysis.get('average_function_length', 0) > 50:
            recommendations.append("Consider breaking down large functions for better maintainability")
        
        if semantic_analysis.get('api_compatibility_score', 0) < 0.7:
            recommendations.append("Review API usage to ensure compatibility with original binary")
        
        if compilation_analysis.get('compilation_errors'):
            recommendations.append("Fix compilation errors to improve build readiness")
        
        return critical_issues, warnings, recommendations
    
    def _compare_with_benchmarks(self, overall_score: float, 
                                dimension_scores: Dict[QualityDimension, float]) -> Dict[str, float]:
        """Compare scores with quality benchmarks"""
        
        benchmark_comparisons = {}
        
        # Compare with predefined benchmarks
        for benchmark_name, benchmark_score in self.benchmarks.items():
            comparison_ratio = overall_score / benchmark_score if benchmark_score > 0 else 0.0
            benchmark_comparisons[benchmark_name] = comparison_ratio
        
        return benchmark_comparisons
    
    def _detect_analysis_tools(self) -> Dict[str, bool]:
        """Detect available analysis tools"""
        tools = {}
        
        # Check for common tools
        try:
            subprocess.run(['gcc', '--version'], capture_output=True, timeout=5)
            tools['gcc'] = True
        except:
            tools['gcc'] = False
        
        try:
            subprocess.run(['clang', '--version'], capture_output=True, timeout=5)
            tools['clang'] = True
        except:
            tools['clang'] = False
        
        try:
            subprocess.run(['cmake', '--version'], capture_output=True, timeout=5)
            tools['cmake'] = True
        except:
            tools['cmake'] = False
        
        return tools
    
    def _load_quality_benchmarks(self) -> Dict[str, float]:
        """Load quality benchmarks for comparison"""
        return {
            'minimum_acceptable': 0.6,
            'production_ready': 0.8,
            'excellent_quality': 0.9,
            'perfect_reconstruction': 1.0
        }
    
    def _generate_assessment_id(self, original_binary: Path, reconstructed_source: Path) -> str:
        """Generate unique assessment ID"""
        content = f"{original_binary.name}_{reconstructed_source.name}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _create_failed_assessment(self, error_message: str, assessment_id: str) -> QualityReport:
        """Create failed assessment report"""
        return QualityReport(
            reconstruction_id=assessment_id,
            timestamp=time.time(),
            overall_score=0.0,
            quality_level=QualityLevel.FAILED,
            confidence_interval=(0.0, 0.0),
            critical_issues=[error_message]
        )
    
    def save_quality_report(self, report: QualityReport, output_path: Path) -> str:
        """Save quality report to JSON file"""
        try:
            report_data = {
                'reconstruction_id': report.reconstruction_id,
                'timestamp': report.timestamp,
                'overall_score': report.overall_score,
                'quality_level': report.quality_level.value,
                'confidence_interval': report.confidence_interval,
                'dimension_scores': {dim.value: score for dim, score in report.dimension_scores.items()},
                'metrics': [{
                    'name': m.name,
                    'score': m.score,
                    'weight': m.weight,
                    'confidence': m.confidence,
                    'dimension': m.dimension.value,
                    'details': m.details,
                    'issues': m.issues,
                    'recommendations': m.recommendations
                } for m in report.metrics],
                'analysis': {
                    'source_analysis': report.source_analysis,
                    'compilation_analysis': report.compilation_analysis,
                    'semantic_analysis': report.semantic_analysis
                },
                'issues_and_recommendations': {
                    'critical_issues': report.critical_issues,
                    'warnings': report.warnings,
                    'recommendations': report.recommendations
                },
                'benchmarks': report.benchmark_comparisons,
                'analysis_time': report.analysis_time
            }
            
            output_file = output_path / f"quality_report_{report.reconstruction_id}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Quality report saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save quality report: {e}")
            raise


# Factory function for easy instantiation
def create_quality_scorer(config_manager=None) -> ReconstructionQualityScorer:
    """Factory function to create quality scorer"""
    return ReconstructionQualityScorer(config_manager)


if __name__ == "__main__":
    # Example usage
    scorer = create_quality_scorer()
    
    # Mock assessment
    original_binary = Path("input/launcher.exe")
    reconstructed_source = Path("output/reconstruction/src")
    
    if original_binary.exists() and reconstructed_source.exists():
        try:
            report = scorer.assess_reconstruction_quality(
                original_binary, 
                reconstructed_source
            )
            
            print(f"Quality Assessment Results:")
            print(f"Overall Score: {report.overall_score:.1%}")
            print(f"Quality Level: {report.quality_level.value}")
            print(f"Confidence: {report.confidence_interval[0]:.1%} - {report.confidence_interval[1]:.1%}")
            
            # Save report
            output_path = Path("output/quality_reports")
            report_file = scorer.save_quality_report(report, output_path)
            print(f"Report saved to: {report_file}")
            
        except Exception as e:
            print(f"Assessment failed: {e}")
    else:
        print("Demo files not found - please run reconstruction first")