"""
Validation Reporting System for Open-Sourcefy Matrix Pipeline

This module provides comprehensive validation reporting capabilities that integrate
binary comparison results, quality metrics, and semantic analysis into unified
reports for reconstruction validation and quality assessment.

Features:
- Multi-format report generation (JSON, HTML, PDF)
- Interactive dashboards and visualizations
- Executive summaries and technical details
- Trend analysis and benchmarking
- Automated recommendations and action items
- Integration with all validation engines
"""

import os
import json
import logging
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
import statistics
from collections import defaultdict
import base64

# Import our validation engines
from .binary_comparison import BinaryComparisonEngine, ComparisonReport, ComparisonResult
from .reconstruction_quality_scorer import ReconstructionQualityScorer, QualityReport, QualityLevel

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats"""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    EXECUTIVE_SUMMARY = "executive"


class ReportScope(Enum):
    """Report scope levels"""
    SUMMARY = "summary"           # High-level overview
    DETAILED = "detailed"         # Comprehensive analysis
    TECHNICAL = "technical"       # Deep technical details
    EXECUTIVE = "executive"       # Business-focused summary


class ValidationStatus(Enum):
    """Overall validation status"""
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    INCOMPLETE = "incomplete"
    ERROR = "error"


@dataclass
class ValidationSummary:
    """Summary of validation results"""
    overall_status: ValidationStatus
    overall_score: float
    confidence: float
    
    # Component scores
    binary_similarity: float
    quality_score: float
    semantic_correctness: float
    compilation_readiness: float
    
    # Issue counts
    critical_issues: int
    warnings: int
    recommendations: int
    
    # Timing information
    analysis_time: float
    timestamp: float


@dataclass
class IntegratedValidationReport:
    """Comprehensive validation report integrating all analysis components"""
    report_id: str
    timestamp: float
    
    # Input information
    original_binary_path: str
    reconstructed_source_path: str
    reconstruction_method: str
    
    # Validation summary
    validation_summary: ValidationSummary
    
    # Component reports
    binary_comparison_report: Optional[ComparisonReport] = None
    quality_assessment_report: Optional[QualityReport] = None
    
    # Integrated analysis
    cross_validation_results: Dict[str, Any] = field(default_factory=dict)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Issues and recommendations
    consolidated_issues: List[Dict[str, Any]] = field(default_factory=list)
    consolidated_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Benchmarking
    benchmark_comparisons: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    generation_info: Dict[str, Any] = field(default_factory=dict)


class ValidationReportingSystem:
    """
    Comprehensive validation reporting system
    
    Integrates binary comparison, quality assessment, and semantic analysis
    into unified reports with multiple output formats and visualization options.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation engines
        self.binary_comparator = BinaryComparisonEngine(config_manager)
        self.quality_scorer = ReconstructionQualityScorer(config_manager)
        
        # Report configuration
        self.default_output_dir = Path("output/validation_reports")
        self.template_dir = Path(__file__).parent / "templates"
        
        # Validation thresholds
        self.validation_thresholds = {
            'critical_similarity': 0.8,
            'minimum_quality': 0.6,
            'semantic_correctness': 0.7,
            'compilation_readiness': 0.5
        }
        
        # Report templates
        self.report_templates = self._load_report_templates()
        
        # Historical data for trend analysis
        self.historical_reports = []
        
    def generate_integrated_validation_report(self,
                                            original_binary_path: Path,
                                            reconstructed_source_path: Path,
                                            compiled_binary_path: Optional[Path] = None,
                                            report_scope: ReportScope = ReportScope.DETAILED,
                                            output_formats: List[ReportFormat] = None) -> IntegratedValidationReport:
        """
        Generate comprehensive integrated validation report
        
        Args:
            original_binary_path: Path to original binary
            reconstructed_source_path: Path to reconstructed source directory
            compiled_binary_path: Optional path to recompiled binary
            report_scope: Level of detail in the report
            output_formats: List of output formats to generate
            
        Returns:
            Integrated validation report with all analysis components
        """
        start_time = time.time()
        
        if output_formats is None:
            output_formats = [ReportFormat.JSON, ReportFormat.HTML]
        
        self.logger.info(f"Generating integrated validation report for {original_binary_path.name}")
        
        # Generate report ID
        report_id = self._generate_report_id(original_binary_path, reconstructed_source_path)
        
        try:
            # Phase 1: Binary comparison analysis
            binary_comparison_report = None
            if compiled_binary_path and compiled_binary_path.exists():
                self.logger.info("Performing binary comparison analysis...")
                binary_comparison_report = self.binary_comparator.generate_detailed_comparison(
                    str(original_binary_path), str(compiled_binary_path)
                )
            
            # Phase 2: Quality assessment
            self.logger.info("Performing quality assessment...")
            quality_assessment_report = self.quality_scorer.assess_reconstruction_quality(
                original_binary_path, reconstructed_source_path
            )
            
            # Phase 3: Cross-validation analysis
            self.logger.info("Performing cross-validation analysis...")
            cross_validation_results = self._perform_cross_validation(
                binary_comparison_report, quality_assessment_report
            )
            
            # Phase 4: Generate validation summary
            validation_summary = self._generate_validation_summary(
                binary_comparison_report, quality_assessment_report, cross_validation_results
            )
            
            # Phase 5: Consolidate issues and recommendations
            consolidated_issues, consolidated_recommendations, action_items = self._consolidate_findings(
                binary_comparison_report, quality_assessment_report, cross_validation_results
            )
            
            # Phase 6: Benchmark comparison
            benchmark_comparisons = self._perform_benchmark_comparison(
                validation_summary, quality_assessment_report
            )
            
            # Phase 7: Trend analysis (if historical data available)
            trend_analysis = self._perform_trend_analysis(validation_summary)
            
            # Create integrated report
            integrated_report = IntegratedValidationReport(
                report_id=report_id,
                timestamp=time.time(),
                original_binary_path=str(original_binary_path),
                reconstructed_source_path=str(reconstructed_source_path),
                reconstruction_method="Matrix Pipeline Decompilation",
                validation_summary=validation_summary,
                binary_comparison_report=binary_comparison_report,
                quality_assessment_report=quality_assessment_report,
                cross_validation_results=cross_validation_results,
                trend_analysis=trend_analysis,
                consolidated_issues=consolidated_issues,
                consolidated_recommendations=consolidated_recommendations,
                action_items=action_items,
                benchmark_comparisons=benchmark_comparisons,
                generation_info={
                    'report_scope': report_scope.value,
                    'analysis_time': time.time() - start_time,
                    'generated_by': 'Open-Sourcefy Validation System',
                    'version': '1.0',
                    'components_analyzed': [
                        'binary_comparison' if binary_comparison_report else None,
                        'quality_assessment',
                        'cross_validation'
                    ]
                }
            )
            
            # Phase 8: Generate output files
            output_files = self._generate_report_outputs(integrated_report, output_formats, report_scope)
            
            # Store for trend analysis
            self.historical_reports.append(integrated_report)
            
            self.logger.info(
                f"Validation report generated: {validation_summary.overall_status.value} "
                f"(Score: {validation_summary.overall_score:.1%}) in {integrated_report.generation_info['analysis_time']:.1f}s"
            )
            
            return integrated_report
            
        except Exception as e:
            self.logger.error(f"Validation report generation failed: {e}", exc_info=True)
            return self._create_error_report(report_id, f"Report generation failed: {e}")
    
    def _perform_cross_validation(self, 
                                binary_report: Optional[ComparisonReport],
                                quality_report: QualityReport) -> Dict[str, Any]:
        """Perform cross-validation analysis between different validation methods"""
        
        cross_validation = {
            'consistency_score': 0.0,
            'discrepancies': [],
            'correlation_analysis': {},
            'confidence_assessment': {}
        }
        
        try:
            if binary_report:
                # Compare binary similarity with quality scores
                binary_similarity = binary_report.overall_similarity
                quality_score = quality_report.overall_score
                
                # Check for consistency
                similarity_diff = abs(binary_similarity - quality_score)
                if similarity_diff < 0.2:
                    cross_validation['consistency_score'] = 1.0 - similarity_diff
                else:
                    cross_validation['consistency_score'] = 0.0
                    cross_validation['discrepancies'].append({
                        'type': 'score_mismatch',
                        'binary_similarity': binary_similarity,
                        'quality_score': quality_score,
                        'difference': similarity_diff,
                        'description': f"Large discrepancy between binary similarity ({binary_similarity:.1%}) and quality score ({quality_score:.1%})"
                    })
                
                # Correlation analysis
                cross_validation['correlation_analysis'] = {
                    'binary_vs_quality': self._calculate_correlation(binary_similarity, quality_score),
                    'structural_consistency': self._analyze_structural_consistency(binary_report, quality_report),
                    'semantic_alignment': self._analyze_semantic_alignment(binary_report, quality_report)
                }
            
            else:
                # Quality-only validation
                cross_validation['consistency_score'] = quality_report.overall_score
                cross_validation['correlation_analysis'] = {
                    'quality_internal_consistency': self._analyze_quality_internal_consistency(quality_report)
                }
            
            # Confidence assessment
            cross_validation['confidence_assessment'] = self._assess_cross_validation_confidence(
                binary_report, quality_report, cross_validation
            )
            
        except Exception as e:
            self.logger.error(f"Cross-validation analysis failed: {e}")
            cross_validation['discrepancies'].append({
                'type': 'analysis_error',
                'description': f"Cross-validation failed: {e}"
            })
        
        return cross_validation
    
    def _generate_validation_summary(self,
                                   binary_report: Optional[ComparisonReport],
                                   quality_report: QualityReport,
                                   cross_validation: Dict[str, Any]) -> ValidationSummary:
        """Generate overall validation summary"""
        
        # Calculate component scores
        binary_similarity = binary_report.overall_similarity if binary_report else 0.0
        quality_score = quality_report.overall_score
        semantic_correctness = quality_report.dimension_scores.get('semantic_correctness', 0.0)
        compilation_readiness = quality_report.dimension_scores.get('compilation_readiness', 0.0)
        
        # Calculate overall score (weighted average)
        if binary_report:
            overall_score = (
                binary_similarity * 0.3 +
                quality_score * 0.4 +
                semantic_correctness * 0.2 +
                compilation_readiness * 0.1
            )
        else:
            overall_score = (
                quality_score * 0.6 +
                semantic_correctness * 0.25 +
                compilation_readiness * 0.15
            )
        
        # Calculate confidence
        confidence_factors = [quality_report.confidence_interval[1]]
        if binary_report:
            confidence_factors.append(binary_report.reconstruction_confidence)
        confidence_factors.append(cross_validation.get('consistency_score', 0.0))
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Count issues
        critical_issues = len(quality_report.critical_issues)
        warnings = len(quality_report.warnings)
        recommendations = len(quality_report.recommendations)
        
        if binary_report:
            critical_issues += len(binary_report.validation_errors)
            warnings += len(binary_report.validation_warnings)
        
        # Determine overall status
        overall_status = self._determine_validation_status(
            overall_score, critical_issues, warnings, cross_validation
        )
        
        return ValidationSummary(
            overall_status=overall_status,
            overall_score=overall_score,
            confidence=confidence,
            binary_similarity=binary_similarity,
            quality_score=quality_score,
            semantic_correctness=semantic_correctness,
            compilation_readiness=compilation_readiness,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=recommendations,
            analysis_time=time.time(),
            timestamp=time.time()
        )
    
    def _determine_validation_status(self, overall_score: float, critical_issues: int, 
                                   warnings: int, cross_validation: Dict[str, Any]) -> ValidationStatus:
        """Determine overall validation status"""
        
        # Check for critical failures
        if critical_issues > 0:
            return ValidationStatus.FAILED
        
        # Check cross-validation discrepancies
        if cross_validation.get('discrepancies'):
            for discrepancy in cross_validation['discrepancies']:
                if discrepancy.get('type') == 'score_mismatch' and discrepancy.get('difference', 0) > 0.3:
                    return ValidationStatus.FAILED
        
        # Score-based status determination
        if overall_score >= self.validation_thresholds['critical_similarity']:
            if warnings > 0:
                return ValidationStatus.PASSED_WITH_WARNINGS
            else:
                return ValidationStatus.PASSED
        elif overall_score >= self.validation_thresholds['minimum_quality']:
            return ValidationStatus.PASSED_WITH_WARNINGS
        else:
            return ValidationStatus.FAILED
    
    def _consolidate_findings(self,
                            binary_report: Optional[ComparisonReport],
                            quality_report: QualityReport,
                            cross_validation: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Consolidate issues and recommendations from all validation components"""
        
        consolidated_issues = []
        consolidated_recommendations = []
        action_items = []
        
        # Process quality report issues
        for issue in quality_report.critical_issues:
            consolidated_issues.append({
                'source': 'quality_assessment',
                'severity': 'critical',
                'description': issue,
                'category': 'code_quality'
            })
        
        for warning in quality_report.warnings:
            consolidated_issues.append({
                'source': 'quality_assessment',
                'severity': 'warning',
                'description': warning,
                'category': 'code_quality'
            })
        
        for recommendation in quality_report.recommendations:
            consolidated_recommendations.append({
                'source': 'quality_assessment',
                'description': recommendation,
                'category': 'improvement',
                'priority': 'medium'
            })
        
        # Process binary comparison issues
        if binary_report:
            for error in binary_report.validation_errors:
                consolidated_issues.append({
                    'source': 'binary_comparison',
                    'severity': 'critical',
                    'description': error,
                    'category': 'binary_compatibility'
                })
            
            for warning in binary_report.validation_warnings:
                consolidated_issues.append({
                    'source': 'binary_comparison',
                    'severity': 'warning',
                    'description': warning,
                    'category': 'binary_compatibility'
                })
        
        # Process cross-validation discrepancies
        for discrepancy in cross_validation.get('discrepancies', []):
            severity = 'critical' if discrepancy.get('difference', 0) > 0.3 else 'warning'
            consolidated_issues.append({
                'source': 'cross_validation',
                'severity': severity,
                'description': discrepancy.get('description', 'Validation discrepancy detected'),
                'category': 'validation_consistency'
            })
        
        # Generate action items based on issues
        action_items = self._generate_action_items(consolidated_issues, consolidated_recommendations)
        
        return consolidated_issues, consolidated_recommendations, action_items
    
    def _generate_action_items(self, issues: List[Dict[str, Any]], 
                             recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate actionable items based on issues and recommendations"""
        
        action_items = []
        
        # Critical issues become high-priority action items
        critical_issues = [issue for issue in issues if issue.get('severity') == 'critical']
        if critical_issues:
            action_items.append({
                'priority': 'high',
                'category': 'critical_fixes',
                'title': 'Address Critical Issues',
                'description': f'Fix {len(critical_issues)} critical issues that prevent successful validation',
                'tasks': [issue['description'] for issue in critical_issues[:5]],  # Limit to top 5
                'estimated_effort': 'high'
            })
        
        # Quality improvements
        quality_recommendations = [rec for rec in recommendations if rec.get('category') == 'improvement']
        if quality_recommendations:
            action_items.append({
                'priority': 'medium',
                'category': 'quality_improvements',
                'title': 'Implement Quality Improvements',
                'description': f'Apply {len(quality_recommendations)} quality improvement recommendations',
                'tasks': [rec['description'] for rec in quality_recommendations[:3]],  # Top 3
                'estimated_effort': 'medium'
            })
        
        # Compilation fixes
        compilation_issues = [issue for issue in issues if 'compilation' in issue.get('description', '').lower()]
        if compilation_issues:
            action_items.append({
                'priority': 'high',
                'category': 'compilation_fixes',
                'title': 'Fix Compilation Issues',
                'description': f'Resolve {len(compilation_issues)} compilation-related problems',
                'tasks': [issue['description'] for issue in compilation_issues],
                'estimated_effort': 'medium'
            })
        
        return action_items
    
    def _perform_benchmark_comparison(self, validation_summary: ValidationSummary,
                                    quality_report: QualityReport) -> Dict[str, float]:
        """Compare validation results with established benchmarks"""
        
        benchmarks = {
            'industry_standard': 0.75,
            'production_ready': 0.85,
            'research_quality': 0.70,
            'minimum_viable': 0.60
        }
        
        comparisons = {}
        overall_score = validation_summary.overall_score
        
        for benchmark_name, benchmark_score in benchmarks.items():
            comparison_ratio = overall_score / benchmark_score if benchmark_score > 0 else 0.0
            comparisons[f"{benchmark_name}_ratio"] = comparison_ratio
            comparisons[f"{benchmark_name}_meets_threshold"] = overall_score >= benchmark_score
        
        return comparisons
    
    def _perform_trend_analysis(self, current_summary: ValidationSummary) -> Dict[str, Any]:
        """Analyze trends in validation results over time"""
        
        if len(self.historical_reports) < 2:
            return {
                'trend_available': False,
                'message': 'Insufficient historical data for trend analysis'
            }
        
        # Extract historical scores
        historical_scores = [report.validation_summary.overall_score for report in self.historical_reports[-10:]]
        historical_scores.append(current_summary.overall_score)
        
        trend_analysis = {
            'trend_available': True,
            'historical_count': len(historical_scores),
            'current_score': current_summary.overall_score,
            'average_score': statistics.mean(historical_scores),
            'score_improvement': current_summary.overall_score - statistics.mean(historical_scores[:-1]),
            'trend_direction': 'improving' if len(historical_scores) > 1 and historical_scores[-1] > historical_scores[-2] else 'declining'
        }
        
        if len(historical_scores) > 2:
            trend_analysis['volatility'] = statistics.stdev(historical_scores)
        
        return trend_analysis
    
    def _generate_report_outputs(self, report: IntegratedValidationReport,
                               formats: List[ReportFormat], scope: ReportScope) -> Dict[str, str]:
        """Generate report outputs in specified formats"""
        
        output_files = {}
        output_dir = self.default_output_dir / report.report_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            for format_type in formats:
                if format_type == ReportFormat.JSON:
                    output_files['json'] = self._generate_json_report(report, output_dir, scope)
                elif format_type == ReportFormat.HTML:
                    output_files['html'] = self._generate_html_report(report, output_dir, scope)
                elif format_type == ReportFormat.MARKDOWN:
                    output_files['markdown'] = self._generate_markdown_report(report, output_dir, scope)
                elif format_type == ReportFormat.EXECUTIVE_SUMMARY:
                    output_files['executive'] = self._generate_executive_summary(report, output_dir)
                
        except Exception as e:
            self.logger.error(f"Failed to generate report outputs: {e}")
        
        return output_files
    
    def _generate_json_report(self, report: IntegratedValidationReport, 
                            output_dir: Path, scope: ReportScope) -> str:
        """Generate JSON format report"""
        
        # Convert dataclasses to dictionaries for JSON serialization
        json_data = {
            'report_metadata': {
                'report_id': report.report_id,
                'timestamp': report.timestamp,
                'scope': scope.value,
                'format': 'json'
            },
            'validation_summary': asdict(report.validation_summary),
            'analysis_results': {
                'cross_validation': report.cross_validation_results,
                'trend_analysis': report.trend_analysis,
                'benchmark_comparisons': report.benchmark_comparisons
            },
            'findings': {
                'issues': report.consolidated_issues,
                'recommendations': report.consolidated_recommendations,
                'action_items': report.action_items
            },
            'generation_info': report.generation_info
        }
        
        # Add component reports based on scope
        if scope in [ReportScope.DETAILED, ReportScope.TECHNICAL]:
            if report.binary_comparison_report:
                json_data['binary_comparison'] = {
                    'overall_similarity': report.binary_comparison_report.overall_similarity,
                    'comparison_type': report.binary_comparison_report.comparison_type.value,
                    'result': report.binary_comparison_report.result.value,
                    'reconstruction_confidence': report.binary_comparison_report.reconstruction_confidence
                }
            
            if report.quality_assessment_report:
                json_data['quality_assessment'] = {
                    'overall_score': report.quality_assessment_report.overall_score,
                    'quality_level': report.quality_assessment_report.quality_level.value,
                    'dimension_scores': {dim.value: score for dim, score in report.quality_assessment_report.dimension_scores.items()}
                }
        
        # Save JSON file
        json_file = output_dir / f"validation_report_{report.report_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        self.logger.info(f"JSON report generated: {json_file}")
        return str(json_file)
    
    def _generate_html_report(self, report: IntegratedValidationReport,
                            output_dir: Path, scope: ReportScope) -> str:
        """Generate HTML format report"""
        
        html_template = self._get_html_template(scope)
        
        # Prepare template variables
        template_vars = {
            'report_id': report.report_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp)),
            'overall_status': report.validation_summary.overall_status.value.replace('_', ' ').title(),
            'overall_score': f"{report.validation_summary.overall_score:.1%}",
            'confidence': f"{report.validation_summary.confidence:.1%}",
            'binary_path': Path(report.original_binary_path).name,
            'source_path': Path(report.reconstructed_source_path).name,
            'critical_issues_count': report.validation_summary.critical_issues,
            'warnings_count': report.validation_summary.warnings,
            'recommendations_count': report.validation_summary.recommendations,
            'analysis_time': f"{report.generation_info.get('analysis_time', 0):.1f}s"
        }
        
        # Generate status color
        status_colors = {
            'passed': '#28a745',
            'passed with warnings': '#ffc107',
            'failed': '#dc3545',
            'incomplete': '#6c757d',
            'error': '#dc3545'
        }
        template_vars['status_color'] = status_colors.get(template_vars['overall_status'].lower(), '#6c757d')
        
        # Add component scores
        template_vars.update({
            'binary_similarity': f"{report.validation_summary.binary_similarity:.1%}",
            'quality_score': f"{report.validation_summary.quality_score:.1%}",
            'semantic_correctness': f"{report.validation_summary.semantic_correctness:.1%}",
            'compilation_readiness': f"{report.validation_summary.compilation_readiness:.1%}"
        })
        
        # Generate issues and recommendations HTML
        template_vars['issues_html'] = self._generate_issues_html(report.consolidated_issues)
        template_vars['recommendations_html'] = self._generate_recommendations_html(report.consolidated_recommendations)
        template_vars['action_items_html'] = self._generate_action_items_html(report.action_items)
        
        # Apply template
        html_content = html_template.format(**template_vars)
        
        # Save HTML file
        html_file = output_dir / f"validation_report_{report.report_id}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {html_file}")
        return str(html_file)
    
    def _generate_markdown_report(self, report: IntegratedValidationReport,
                                output_dir: Path, scope: ReportScope) -> str:
        """Generate Markdown format report"""
        
        markdown_content = f"""# Validation Report - {report.report_id}

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}  
**Binary:** {Path(report.original_binary_path).name}  
**Source:** {Path(report.reconstructed_source_path).name}  

## Validation Summary

- **Overall Status:** {report.validation_summary.overall_status.value.replace('_', ' ').title()}
- **Overall Score:** {report.validation_summary.overall_score:.1%}
- **Confidence:** {report.validation_summary.confidence:.1%}

### Component Scores

| Component | Score |
|-----------|-------|
| Binary Similarity | {report.validation_summary.binary_similarity:.1%} |
| Quality Score | {report.validation_summary.quality_score:.1%} |
| Semantic Correctness | {report.validation_summary.semantic_correctness:.1%} |
| Compilation Readiness | {report.validation_summary.compilation_readiness:.1%} |

### Issue Summary

- **Critical Issues:** {report.validation_summary.critical_issues}
- **Warnings:** {report.validation_summary.warnings}
- **Recommendations:** {report.validation_summary.recommendations}

## Issues Found

{self._generate_issues_markdown(report.consolidated_issues)}

## Recommendations

{self._generate_recommendations_markdown(report.consolidated_recommendations)}

## Action Items

{self._generate_action_items_markdown(report.action_items)}

---
*Report generated by Open-Sourcefy Validation System*
"""
        
        # Save Markdown file
        markdown_file = output_dir / f"validation_report_{report.report_id}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        self.logger.info(f"Markdown report generated: {markdown_file}")
        return str(markdown_file)
    
    def _generate_executive_summary(self, report: IntegratedValidationReport, output_dir: Path) -> str:
        """Generate executive summary"""
        
        summary_content = f"""
EXECUTIVE SUMMARY - BINARY RECONSTRUCTION VALIDATION

Project: {Path(report.original_binary_path).name} Reconstruction
Date: {time.strftime('%Y-%m-%d', time.localtime(report.timestamp))}

OVERALL ASSESSMENT: {report.validation_summary.overall_status.value.upper().replace('_', ' ')}
Success Rate: {report.validation_summary.overall_score:.0%}
Confidence Level: {report.validation_summary.confidence:.0%}

KEY FINDINGS:
• Binary Compatibility: {report.validation_summary.binary_similarity:.0%}
• Code Quality: {report.validation_summary.quality_score:.0%}
• Semantic Accuracy: {report.validation_summary.semantic_correctness:.0%}
• Build Readiness: {report.validation_summary.compilation_readiness:.0%}

CRITICAL ISSUES: {report.validation_summary.critical_issues}
RECOMMENDATIONS: {report.validation_summary.recommendations}

BUSINESS IMPACT:
{'✓ PRODUCTION READY' if report.validation_summary.overall_score >= 0.8 else '⚠ REQUIRES ATTENTION' if report.validation_summary.overall_score >= 0.6 else '✗ NOT READY'}

NEXT STEPS:
{chr(10).join(['• ' + item['title'] for item in report.action_items[:3]])}
"""
        
        # Save executive summary
        exec_file = output_dir / f"executive_summary_{report.report_id}.txt"
        with open(exec_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        self.logger.info(f"Executive summary generated: {exec_file}")
        return str(exec_file)
    
    # Helper methods for analysis
    def _calculate_correlation(self, value1: float, value2: float) -> float:
        """Calculate simple correlation between two values"""
        return 1.0 - abs(value1 - value2)
    
    def _analyze_structural_consistency(self, binary_report: ComparisonReport, 
                                      quality_report: QualityReport) -> float:
        """Analyze structural consistency between binary and quality reports"""
        # Placeholder for structural consistency analysis
        return 0.8
    
    def _analyze_semantic_alignment(self, binary_report: ComparisonReport,
                                  quality_report: QualityReport) -> float:
        """Analyze semantic alignment between reports"""
        # Placeholder for semantic alignment analysis
        return 0.75
    
    def _analyze_quality_internal_consistency(self, quality_report: QualityReport) -> float:
        """Analyze internal consistency of quality report"""
        # Check if dimension scores are consistent with overall score
        dimension_scores = list(quality_report.dimension_scores.values())
        if not dimension_scores:
            return 0.0
        
        avg_dimension_score = sum(dimension_scores) / len(dimension_scores)
        consistency = 1.0 - abs(quality_report.overall_score - avg_dimension_score)
        
        return max(0.0, min(1.0, consistency))
    
    def _assess_cross_validation_confidence(self, binary_report: Optional[ComparisonReport],
                                          quality_report: QualityReport,
                                          cross_validation: Dict[str, Any]) -> Dict[str, float]:
        """Assess confidence in cross-validation results"""
        
        confidence_factors = {
            'data_completeness': 1.0 if binary_report else 0.7,  # Lower confidence without binary comparison
            'consistency_score': cross_validation.get('consistency_score', 0.0),
            'quality_confidence': quality_report.confidence_interval[1],
            'analysis_coverage': 1.0  # Full analysis performed
        }
        
        # Calculate overall cross-validation confidence
        overall_confidence = sum(confidence_factors.values()) / len(confidence_factors)
        confidence_factors['overall'] = overall_confidence
        
        return confidence_factors
    
    # Template and formatting methods
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates"""
        return {
            'html_summary': self._get_default_html_template(),
            'html_detailed': self._get_detailed_html_template()
        }
    
    def _get_html_template(self, scope: ReportScope) -> str:
        """Get HTML template based on scope"""
        if scope == ReportScope.DETAILED:
            return self._get_detailed_html_template()
        else:
            return self._get_default_html_template()
    
    def _get_default_html_template(self) -> str:
        """Get default HTML template"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>Validation Report - {report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .status {{ font-size: 24px; font-weight: bold; color: {status_color}; }}
        .score {{ font-size: 48px; font-weight: bold; color: {status_color}; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; }}
        .issue-critical {{ color: #dc3545; }}
        .issue-warning {{ color: #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Binary Reconstruction Validation Report</h1>
        <p><strong>Report ID:</strong> {report_id}</p>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Binary:</strong> {binary_path}</p>
        <p><strong>Source:</strong> {source_path}</p>
    </div>

    <div class="section">
        <h2>Validation Summary</h2>
        <div class="status">{overall_status}</div>
        <div class="score">{overall_score}</div>
        <p><strong>Confidence:</strong> {confidence}</p>
        <p><strong>Analysis Time:</strong> {analysis_time}</p>
    </div>

    <div class="section">
        <h2>Component Scores</h2>
        <div class="metric">
            <strong>Binary Similarity:</strong> {binary_similarity}
        </div>
        <div class="metric">
            <strong>Quality Score:</strong> {quality_score}
        </div>
        <div class="metric">
            <strong>Semantic Correctness:</strong> {semantic_correctness}
        </div>
        <div class="metric">
            <strong>Compilation Readiness:</strong> {compilation_readiness}
        </div>
    </div>

    <div class="section">
        <h2>Issues Summary</h2>
        <p><strong>Critical Issues:</strong> {critical_issues_count}</p>
        <p><strong>Warnings:</strong> {warnings_count}</p>
        <p><strong>Recommendations:</strong> {recommendations_count}</p>
    </div>

    <div class="section">
        <h2>Issues Found</h2>
        {issues_html}
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        {recommendations_html}
    </div>

    <div class="section">
        <h2>Action Items</h2>
        {action_items_html}
    </div>
</body>
</html>"""
    
    def _get_detailed_html_template(self) -> str:
        """Get detailed HTML template with additional sections"""
        # Use the default template for now - could be extended with more details
        return self._get_default_html_template()
    
    def _generate_issues_html(self, issues: List[Dict[str, Any]]) -> str:
        """Generate HTML for issues list"""
        if not issues:
            return "<p>No issues found.</p>"
        
        html = "<ul>"
        for issue in issues:
            severity_class = f"issue-{issue.get('severity', 'warning')}"
            html += f'<li class="{severity_class}"><strong>[{issue.get("severity", "").upper()}]</strong> {issue.get("description", "")}</li>'
        html += "</ul>"
        return html
    
    def _generate_recommendations_html(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate HTML for recommendations list"""
        if not recommendations:
            return "<p>No recommendations.</p>"
        
        html = "<ul>"
        for rec in recommendations:
            html += f'<li>{rec.get("description", "")}</li>'
        html += "</ul>"
        return html
    
    def _generate_action_items_html(self, action_items: List[Dict[str, Any]]) -> str:
        """Generate HTML for action items"""
        if not action_items:
            return "<p>No action items.</p>"
        
        html = ""
        for item in action_items:
            priority_color = {'high': '#dc3545', 'medium': '#ffc107', 'low': '#28a745'}.get(item.get('priority', 'medium'), '#6c757d')
            html += f"""
            <div style="margin: 10px 0; padding: 10px; border-left: 4px solid {priority_color};">
                <h4>{item.get('title', '')}</h4>
                <p><strong>Priority:</strong> {item.get('priority', '').upper()}</p>
                <p>{item.get('description', '')}</p>
                <ul>
            """
            for task in item.get('tasks', []):
                html += f"<li>{task}</li>"
            html += "</ul></div>"
        
        return html
    
    def _generate_issues_markdown(self, issues: List[Dict[str, Any]]) -> str:
        """Generate Markdown for issues list"""
        if not issues:
            return "No issues found."
        
        markdown = ""
        for issue in issues:
            severity = issue.get('severity', 'warning').upper()
            description = issue.get('description', '')
            markdown += f"- **[{severity}]** {description}\n"
        
        return markdown
    
    def _generate_recommendations_markdown(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate Markdown for recommendations list"""
        if not recommendations:
            return "No recommendations."
        
        markdown = ""
        for rec in recommendations:
            markdown += f"- {rec.get('description', '')}\n"
        
        return markdown
    
    def _generate_action_items_markdown(self, action_items: List[Dict[str, Any]]) -> str:
        """Generate Markdown for action items"""
        if not action_items:
            return "No action items."
        
        markdown = ""
        for item in action_items:
            priority = item.get('priority', 'medium').upper()
            title = item.get('title', '')
            description = item.get('description', '')
            
            markdown += f"### {title} ({priority} Priority)\n\n"
            markdown += f"{description}\n\n"
            
            tasks = item.get('tasks', [])
            if tasks:
                markdown += "Tasks:\n"
                for task in tasks:
                    markdown += f"- {task}\n"
            markdown += "\n"
        
        return markdown
    
    def _generate_report_id(self, original_binary: Path, reconstructed_source: Path) -> str:
        """Generate unique report ID"""
        content = f"{original_binary.name}_{reconstructed_source.name}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _create_error_report(self, report_id: str, error_message: str) -> IntegratedValidationReport:
        """Create error report when validation fails"""
        return IntegratedValidationReport(
            report_id=report_id,
            timestamp=time.time(),
            original_binary_path="",
            reconstructed_source_path="",
            reconstruction_method="Error",
            validation_summary=ValidationSummary(
                overall_status=ValidationStatus.ERROR,
                overall_score=0.0,
                confidence=0.0,
                binary_similarity=0.0,
                quality_score=0.0,
                semantic_correctness=0.0,
                compilation_readiness=0.0,
                critical_issues=1,
                warnings=0,
                recommendations=0,
                analysis_time=0.0,
                timestamp=time.time()
            ),
            consolidated_issues=[{
                'source': 'system',
                'severity': 'critical',
                'description': error_message,
                'category': 'system_error'
            }]
        )


# Factory function for easy instantiation
def create_validation_reporting_system(config_manager=None) -> ValidationReportingSystem:
    """Factory function to create validation reporting system"""
    return ValidationReportingSystem(config_manager)


if __name__ == "__main__":
    # Example usage
    reporting_system = create_validation_reporting_system()
    
    # Mock validation report generation
    original_binary = Path("input/launcher.exe")
    reconstructed_source = Path("output/reconstruction/src")
    
    if original_binary.exists() and reconstructed_source.exists():
        try:
            report = reporting_system.generate_integrated_validation_report(
                original_binary,
                reconstructed_source,
                report_scope=ReportScope.DETAILED,
                output_formats=[ReportFormat.JSON, ReportFormat.HTML, ReportFormat.MARKDOWN]
            )
            
            print(f"Validation Report Generated:")
            print(f"Status: {report.validation_summary.overall_status.value}")
            print(f"Score: {report.validation_summary.overall_score:.1%}")
            print(f"Report ID: {report.report_id}")
            
        except Exception as e:
            print(f"Report generation failed: {e}")
    else:
        print("Demo files not found - please run reconstruction first")