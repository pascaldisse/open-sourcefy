"""
Phase 2 Enhancement: Enhanced Analysis Capabilities and Accuracy Metrics
Advanced pattern recognition, machine learning integration, and quality assessment.
"""

import logging
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Fallback implementations
    class np:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0.0
        
        @staticmethod
        def std(data):
            if not data:
                return 0.0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        @staticmethod
        def min(data):
            return min(data) if data else 0.0
        
        @staticmethod
        def max(data):
            return max(data) if data else 0.0
        
        @staticmethod
        def var(data):
            if not data or len(data) <= 1:
                return 0.0
            mean_val = sum(data) / len(data)
            return sum((x - mean_val) ** 2 for x in data) / len(data)
        
        @staticmethod
        def exp(x):
            import math
            return math.exp(x)

from .agent_base import AgentResult, AgentStatus


class AnalysisQuality(Enum):
    """Analysis quality levels"""
    EXCELLENT = "excellent"    # 95-100% accuracy
    GOOD = "good"             # 85-94% accuracy
    FAIR = "fair"             # 70-84% accuracy
    POOR = "poor"             # 50-69% accuracy
    FAILED = "failed"         # <50% accuracy


class PatternType(Enum):
    """Types of patterns that can be detected"""
    COMPILER_SIGNATURE = "compiler_signature"
    OPTIMIZATION_PATTERN = "optimization_pattern"
    ERROR_PATTERN = "error_pattern"
    STRUCTURE_PATTERN = "structure_pattern"
    BEHAVIORAL_PATTERN = "behavioral_pattern"
    SECURITY_PATTERN = "security_pattern"


@dataclass
class PatternMatch:
    """Represents a detected pattern"""
    pattern_type: PatternType
    pattern_id: str
    confidence: float
    location: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)


@dataclass
class AccuracyMetrics:
    """Comprehensive accuracy metrics for analysis"""
    agent_id: int
    overall_accuracy: float = 0.0
    pattern_detection_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    confidence_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    quality_level: AnalysisQuality = AnalysisQuality.FAILED
    timestamp: float = field(default_factory=time.time)
    
    # Detailed metrics
    patterns_detected: int = 0
    patterns_verified: int = 0
    patterns_false_positive: int = 0
    patterns_missed: int = 0
    execution_efficiency: float = 0.0
    memory_efficiency: float = 0.0


@dataclass
class BinaryCharacteristics:
    """Comprehensive binary characteristics for enhanced analysis"""
    file_path: str
    file_size: int
    file_hash: str
    architecture: str
    compiler: Optional[str] = None
    compiler_version: Optional[str] = None
    optimization_level: Optional[str] = None
    debug_info: bool = False
    stripped: bool = False
    packed: bool = False
    encrypted: bool = False
    obfuscated: bool = False
    
    # Advanced characteristics
    entry_points: List[str] = field(default_factory=list)
    import_libraries: List[str] = field(default_factory=list)
    export_functions: List[str] = field(default_factory=list)
    string_patterns: List[str] = field(default_factory=list)
    code_complexity: float = 0.0
    data_entropy: float = 0.0
    
    # Analysis metadata
    analysis_timestamp: float = field(default_factory=time.time)
    analysis_version: str = "2.0"


class PatternDatabase:
    """Advanced pattern database with machine learning integration"""
    
    def __init__(self):
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_statistics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.learning_data: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("PatternDatabase")
        self._initialize_patterns()
    
    def _initialize_patterns(self) -> None:
        """Initialize with known patterns"""
        
        # Compiler signature patterns
        self.add_pattern("gcc_pattern_1", PatternType.COMPILER_SIGNATURE, {
            "signature": "gcc version",
            "indicators": ["__libc_start_main", "_start", "frame_dummy"],
            "confidence_boost": 0.8,
            "metadata": {"compiler": "gcc", "family": "gnu"}
        })
        
        self.add_pattern("msvc_pattern_1", PatternType.COMPILER_SIGNATURE, {
            "signature": "Microsoft Visual C++",
            "indicators": ["mainCRTStartup", "_CRT_INIT", "DllMainCRTStartup"],
            "confidence_boost": 0.9,
            "metadata": {"compiler": "msvc", "family": "microsoft"}
        })
        
        # Optimization patterns
        self.add_pattern("loop_unrolling", PatternType.OPTIMIZATION_PATTERN, {
            "signature": "unrolled loop",
            "indicators": ["repeated instruction sequences", "reduced branch count"],
            "confidence_boost": 0.7,
            "metadata": {"optimization": "loop_unrolling", "level": "O2+"}
        })
        
        self.add_pattern("dead_code_elimination", PatternType.OPTIMIZATION_PATTERN, {
            "signature": "removed unreachable code",
            "indicators": ["missing debug info", "compressed control flow"],
            "confidence_boost": 0.6,
            "metadata": {"optimization": "dead_code_elimination", "level": "O1+"}
        })
        
        # Error patterns
        self.add_pattern("stack_overflow", PatternType.ERROR_PATTERN, {
            "signature": "stack overflow",
            "indicators": ["recursive calls", "large stack allocations"],
            "confidence_boost": 0.8,
            "metadata": {"error_type": "runtime", "severity": "high"}
        })
        
        # Security patterns
        self.add_pattern("stack_canary", PatternType.SECURITY_PATTERN, {
            "signature": "stack protection",
            "indicators": ["__stack_chk_fail", "gs register usage"],
            "confidence_boost": 0.9,
            "metadata": {"security_feature": "stack_canary", "protection_level": "medium"}
        })
    
    def add_pattern(self, pattern_id: str, pattern_type: PatternType, 
                   pattern_data: Dict[str, Any]) -> None:
        """Add a new pattern to the database"""
        self.patterns[pattern_id] = {
            "type": pattern_type,
            "data": pattern_data,
            "created": time.time(),
            "usage_count": 0,
            "success_rate": 0.0
        }
        self.logger.debug(f"Added pattern: {pattern_id} ({pattern_type.value})")
    
    def match_patterns(self, analysis_data: Dict[str, Any], 
                      binary_chars: BinaryCharacteristics) -> List[PatternMatch]:
        """Match patterns against analysis data"""
        matches = []
        
        for pattern_id, pattern_info in self.patterns.items():
            pattern_data = pattern_info["data"]
            pattern_type = pattern_info["type"]
            
            # Calculate match confidence
            confidence = self._calculate_pattern_confidence(
                pattern_data, analysis_data, binary_chars
            )
            
            if confidence > 0.5:  # Minimum confidence threshold
                match = PatternMatch(
                    pattern_type=pattern_type,
                    pattern_id=pattern_id,
                    confidence=confidence,
                    location=analysis_data.get("location", "unknown"),
                    description=pattern_data.get("signature", ""),
                    metadata=pattern_data.get("metadata", {}),
                    evidence=self._collect_evidence(pattern_data, analysis_data)
                )
                matches.append(match)
                
                # Update pattern statistics
                self.patterns[pattern_id]["usage_count"] += 1
        
        return sorted(matches, key=lambda x: x.confidence, reverse=True)
    
    def _calculate_pattern_confidence(self, pattern_data: Dict[str, Any],
                                    analysis_data: Dict[str, Any],
                                    binary_chars: BinaryCharacteristics) -> float:
        """Calculate confidence score for pattern match"""
        base_confidence = 0.0
        
        # Check for signature match
        signature = pattern_data.get("signature", "")
        if signature and self._text_similarity(signature, str(analysis_data)) > 0.7:
            base_confidence += 0.4
        
        # Check for indicators
        indicators = pattern_data.get("indicators", [])
        if indicators:
            indicator_matches = 0
            for indicator in indicators:
                if self._find_indicator(indicator, analysis_data):
                    indicator_matches += 1
            
            indicator_confidence = indicator_matches / len(indicators)
            base_confidence += indicator_confidence * 0.4
        
        # Apply confidence boost
        confidence_boost = pattern_data.get("confidence_boost", 0.0)
        base_confidence += confidence_boost * 0.2
        
        # Adjust based on binary characteristics
        base_confidence = self._adjust_for_binary_characteristics(
            base_confidence, pattern_data, binary_chars
        )
        
        return min(1.0, max(0.0, base_confidence))
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple Jaccard similarity for now
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _find_indicator(self, indicator: str, data: Dict[str, Any]) -> bool:
        """Check if an indicator is present in the data"""
        indicator_lower = indicator.lower()
        data_str = json.dumps(data).lower()
        return indicator_lower in data_str
    
    def _adjust_for_binary_characteristics(self, confidence: float,
                                         pattern_data: Dict[str, Any],
                                         binary_chars: BinaryCharacteristics) -> float:
        """Adjust confidence based on binary characteristics"""
        
        # Adjust for compiler patterns
        pattern_metadata = pattern_data.get("metadata", {})
        if "compiler" in pattern_metadata:
            expected_compiler = pattern_metadata["compiler"]
            if binary_chars.compiler and binary_chars.compiler.lower() == expected_compiler.lower():
                confidence *= 1.2
            elif binary_chars.compiler and binary_chars.compiler.lower() != expected_compiler.lower():
                confidence *= 0.8
        
        # Adjust for optimization patterns
        if "optimization" in pattern_metadata:
            if binary_chars.optimization_level:
                opt_level = pattern_metadata.get("level", "")
                if opt_level in binary_chars.optimization_level:
                    confidence *= 1.1
        
        # Adjust for architecture-specific patterns
        if "architecture" in pattern_metadata:
            expected_arch = pattern_metadata["architecture"]
            if binary_chars.architecture.lower() == expected_arch.lower():
                confidence *= 1.1
            else:
                confidence *= 0.9
        
        return confidence
    
    def _collect_evidence(self, pattern_data: Dict[str, Any], 
                         analysis_data: Dict[str, Any]) -> List[str]:
        """Collect evidence for pattern match"""
        evidence = []
        
        # Add signature evidence
        signature = pattern_data.get("signature", "")
        if signature:
            evidence.append(f"Signature match: {signature}")
        
        # Add indicator evidence
        indicators = pattern_data.get("indicators", [])
        for indicator in indicators:
            if self._find_indicator(indicator, analysis_data):
                evidence.append(f"Indicator found: {indicator}")
        
        return evidence
    
    def learn_from_results(self, pattern_matches: List[PatternMatch], 
                          verified_results: Dict[str, bool]) -> None:
        """Learn from verification results to improve pattern matching"""
        
        for match in pattern_matches:
            pattern_id = match.pattern_id
            was_correct = verified_results.get(pattern_id, False)
            
            # Update pattern success rate
            if pattern_id in self.patterns:
                pattern_info = self.patterns[pattern_id]
                usage_count = pattern_info["usage_count"]
                current_rate = pattern_info["success_rate"]
                
                # Update using exponential moving average
                alpha = 0.1
                new_rate = alpha * (1.0 if was_correct else 0.0) + (1 - alpha) * current_rate
                pattern_info["success_rate"] = new_rate
                
                # Store learning data
                self.learning_data.append({
                    "pattern_id": pattern_id,
                    "confidence": match.confidence,
                    "was_correct": was_correct,
                    "timestamp": time.time()
                })
                
                # Keep only recent learning data
                if len(self.learning_data) > 10000:
                    self.learning_data = self.learning_data[-5000:]


class AccuracyAssessment:
    """Comprehensive accuracy assessment system"""
    
    def __init__(self):
        self.assessments: Dict[int, List[AccuracyMetrics]] = defaultdict(list)
        self.baseline_metrics: Dict[int, AccuracyMetrics] = {}
        self.logger = logging.getLogger("AccuracyAssessment")
    
    def assess_agent_accuracy(self, agent_id: int, result: AgentResult,
                            ground_truth: Optional[Dict[str, Any]] = None,
                            pattern_matches: List[PatternMatch] = None) -> AccuracyMetrics:
        """Assess accuracy of an agent's analysis"""
        
        metrics = AccuracyMetrics(agent_id=agent_id)
        
        if result.status != AgentStatus.COMPLETED:
            metrics.quality_level = AnalysisQuality.FAILED
            return metrics
        
        # Calculate overall accuracy
        metrics.overall_accuracy = self._calculate_overall_accuracy(result, ground_truth)
        
        # Calculate pattern detection accuracy
        if pattern_matches:
            metrics.patterns_detected = len(pattern_matches)
            metrics.pattern_detection_accuracy = self._calculate_pattern_accuracy(
                pattern_matches, ground_truth
            )
        
        # Calculate confidence score
        metrics.confidence_score = self._calculate_confidence_score(result, pattern_matches)
        
        # Calculate completeness score
        metrics.completeness_score = self._calculate_completeness_score(result, ground_truth)
        
        # Calculate consistency score
        metrics.consistency_score = self._calculate_consistency_score(agent_id, result)
        
        # Calculate efficiency metrics
        metrics.execution_efficiency = self._calculate_execution_efficiency(result)
        metrics.memory_efficiency = self._calculate_memory_efficiency(result)
        
        # Determine quality level
        metrics.quality_level = self._determine_quality_level(metrics)
        
        # Store assessment
        self.assessments[agent_id].append(metrics)
        
        # Keep only recent assessments
        if len(self.assessments[agent_id]) > 100:
            self.assessments[agent_id] = self.assessments[agent_id][-50:]
        
        self.logger.info(f"Agent {agent_id} accuracy: {metrics.overall_accuracy:.2f} ({metrics.quality_level.value})")
        
        return metrics
    
    def _calculate_overall_accuracy(self, result: AgentResult, 
                                  ground_truth: Optional[Dict[str, Any]]) -> float:
        """Calculate overall accuracy score"""
        if not ground_truth:
            # Use heuristic assessment
            return self._heuristic_accuracy_assessment(result)
        
        # Compare with ground truth
        correct_predictions = 0
        total_predictions = 0
        
        result_data = result.data
        
        for key, expected_value in ground_truth.items():
            if key in result_data:
                actual_value = result_data[key]
                if self._values_match(actual_value, expected_value):
                    correct_predictions += 1
                total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _heuristic_accuracy_assessment(self, result: AgentResult) -> float:
        """Heuristic accuracy assessment when no ground truth available"""
        score = 0.5  # Base score
        
        data = result.data
        
        # Check data completeness
        expected_keys = {
            1: ["binary_info", "format", "architecture"],
            2: ["architecture_details", "instruction_set"],
            3: ["error_patterns", "potential_issues"],
            4: ["decompiled_code", "functions"],
            5: ["sections", "memory_layout"],
            6: ["optimizations", "compiler_flags"],
            7: ["advanced_decompilation", "structures"],
            8: ["differences", "changes"],
            9: ["assembly_analysis", "instruction_patterns"],
            10: ["resources", "reconstructed_data"],
            11: ["global_analysis", "cross_references"],
            12: ["compilation_info", "build_commands"],
            13: ["validation_results", "confidence"]
        }
        
        agent_id = result.agent_id
        if agent_id in expected_keys:
            expected = expected_keys[agent_id]
            found_keys = sum(1 for key in expected if key in data)
            completeness = found_keys / len(expected)
            score += completeness * 0.3
        
        # Check data quality indicators
        if "confidence" in data and isinstance(data["confidence"], (int, float)):
            confidence = data["confidence"]
            if isinstance(confidence, (int, float)) and 0 <= confidence <= 1:
                score += confidence * 0.2
        
        # Check for error indicators
        if "errors" in data and data["errors"]:
            score -= 0.1
        
        # Check execution time (penalize very slow or very fast executions)
        exec_time = result.execution_time
        if 1.0 <= exec_time <= 300.0:  # Reasonable time range
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _values_match(self, actual: Any, expected: Any, tolerance: float = 0.1) -> bool:
        """Check if two values match within tolerance"""
        if type(actual) != type(expected):
            return str(actual).lower() == str(expected).lower()
        
        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= tolerance * abs(expected)
        
        if isinstance(actual, str) and isinstance(expected, str):
            return actual.lower() == expected.lower()
        
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            if len(actual) != len(expected):
                return False
            return all(self._values_match(a, e, tolerance) for a, e in zip(actual, expected))
        
        return actual == expected
    
    def _calculate_pattern_accuracy(self, pattern_matches: List[PatternMatch],
                                  ground_truth: Optional[Dict[str, Any]]) -> float:
        """Calculate pattern detection accuracy"""
        if not pattern_matches:
            return 0.0
        
        if not ground_truth or "patterns" not in ground_truth:
            # Use confidence-based assessment
            total_confidence = sum(match.confidence for match in pattern_matches)
            return total_confidence / len(pattern_matches)
        
        # Compare with ground truth patterns
        expected_patterns = set(ground_truth["patterns"])
        detected_patterns = set(match.pattern_id for match in pattern_matches)
        
        true_positives = len(detected_patterns.intersection(expected_patterns))
        false_positives = len(detected_patterns - expected_patterns)
        false_negatives = len(expected_patterns - detected_patterns)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1 score
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0.0
    
    def _calculate_confidence_score(self, result: AgentResult, 
                                  pattern_matches: List[PatternMatch] = None) -> float:
        """Calculate confidence score for the analysis"""
        
        # Base confidence from result data
        confidence = result.data.get("confidence", 0.5)
        
        # Adjust based on pattern match confidence
        if pattern_matches:
            pattern_confidence = sum(match.confidence for match in pattern_matches) / len(pattern_matches)
            confidence = (confidence + pattern_confidence) / 2
        
        # Adjust based on execution success
        if result.status == AgentStatus.COMPLETED and not result.error_message:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_completeness_score(self, result: AgentResult,
                                    ground_truth: Optional[Dict[str, Any]]) -> float:
        """Calculate completeness score"""
        
        if not ground_truth:
            # Heuristic based on data richness
            data = result.data
            if not data:
                return 0.0
            
            # Count non-empty values
            non_empty_count = 0
            total_count = 0
            
            for key, value in data.items():
                total_count += 1
                if value is not None and value != "" and value != []:
                    non_empty_count += 1
            
            return non_empty_count / total_count if total_count > 0 else 0.0
        
        # Compare with ground truth
        expected_keys = set(ground_truth.keys())
        actual_keys = set(result.data.keys())
        
        found_keys = len(expected_keys.intersection(actual_keys))
        return found_keys / len(expected_keys) if expected_keys else 0.0
    
    def _calculate_consistency_score(self, agent_id: int, result: AgentResult) -> float:
        """Calculate consistency score with previous results"""
        
        if agent_id not in self.assessments or len(self.assessments[agent_id]) < 2:
            return 1.0  # Perfect consistency for first result
        
        recent_assessments = self.assessments[agent_id][-5:]  # Last 5 assessments
        
        # Calculate variance in accuracy scores
        accuracies = [assessment.overall_accuracy for assessment in recent_assessments]
        if len(accuracies) > 1:
            variance = np.var(accuracies)
            # Convert variance to consistency score (lower variance = higher consistency)
            consistency = max(0.0, 1.0 - variance * 4)  # Scale variance
        else:
            consistency = 1.0
        
        return consistency
    
    def _calculate_execution_efficiency(self, result: AgentResult) -> float:
        """Calculate execution efficiency score"""
        
        # Expected execution times for each agent (in seconds)
        expected_times = {
            1: 5, 2: 10, 3: 15, 4: 30, 5: 20, 6: 25, 7: 60,
            8: 40, 9: 35, 10: 20, 11: 25, 12: 30, 13: 10
        }
        
        expected_time = expected_times.get(result.agent_id, 30)
        actual_time = result.execution_time
        
        if actual_time <= 0:
            return 0.0
        
        # Efficiency decreases as actual time exceeds expected time
        if actual_time <= expected_time:
            return 1.0
        else:
            # Exponential decay
            efficiency = np.exp(-(actual_time - expected_time) / expected_time)
            return max(0.0, efficiency)
    
    def _calculate_memory_efficiency(self, result: AgentResult) -> float:
        """Calculate memory efficiency score"""
        
        # Use metadata if available
        metadata = result.metadata or {}
        memory_usage = metadata.get("memory_usage_mb", 0)
        
        if memory_usage <= 0:
            return 1.0  # Assume efficient if no data
        
        # Expected memory usage (in MB)
        expected_memory = 512  # 512MB baseline
        
        if memory_usage <= expected_memory:
            return 1.0
        else:
            # Efficiency decreases with high memory usage
            efficiency = expected_memory / memory_usage
            return max(0.0, efficiency)
    
    def _determine_quality_level(self, metrics: AccuracyMetrics) -> AnalysisQuality:
        """Determine overall quality level"""
        
        # Weighted average of different metrics
        weights = {
            "overall_accuracy": 0.4,
            "confidence_score": 0.2,
            "completeness_score": 0.2,
            "consistency_score": 0.1,
            "execution_efficiency": 0.1
        }
        
        weighted_score = (
            metrics.overall_accuracy * weights["overall_accuracy"] +
            metrics.confidence_score * weights["confidence_score"] +
            metrics.completeness_score * weights["completeness_score"] +
            metrics.consistency_score * weights["consistency_score"] +
            metrics.execution_efficiency * weights["execution_efficiency"]
        )
        
        if weighted_score >= 0.95:
            return AnalysisQuality.EXCELLENT
        elif weighted_score >= 0.85:
            return AnalysisQuality.GOOD
        elif weighted_score >= 0.70:
            return AnalysisQuality.FAIR
        elif weighted_score >= 0.50:
            return AnalysisQuality.POOR
        else:
            return AnalysisQuality.FAILED
    
    def get_agent_statistics(self, agent_id: int) -> Dict[str, Any]:
        """Get comprehensive statistics for an agent"""
        
        if agent_id not in self.assessments:
            return {"agent_id": agent_id, "assessments_count": 0}
        
        assessments = self.assessments[agent_id]
        
        if not assessments:
            return {"agent_id": agent_id, "assessments_count": 0}
        
        # Calculate statistics
        accuracies = [a.overall_accuracy for a in assessments]
        confidence_scores = [a.confidence_score for a in assessments]
        quality_levels = [a.quality_level for a in assessments]
        
        stats = {
            "agent_id": agent_id,
            "assessments_count": len(assessments),
            "accuracy": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
                "latest": accuracies[-1] if accuracies else 0.0
            },
            "confidence": {
                "mean": np.mean(confidence_scores),
                "std": np.std(confidence_scores),
                "latest": confidence_scores[-1] if confidence_scores else 0.0
            },
            "quality_distribution": dict(Counter(q.value for q in quality_levels)),
            "trends": {
                "improving": self._calculate_trend(accuracies),
                "stable": self._is_stable(accuracies)
            }
        }
        
        return stats
    
    def _calculate_trend(self, values: List[float]) -> bool:
        """Calculate if values show improving trend"""
        if len(values) < 3:
            return False
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        return slope > 0.01  # Positive trend threshold
    
    def _is_stable(self, values: List[float], threshold: float = 0.05) -> bool:
        """Check if values are stable (low variance)"""
        if len(values) < 2:
            return True
        
        return np.std(values) < threshold


# Global instances
pattern_database = PatternDatabase()
accuracy_assessment = AccuracyAssessment()


def analyze_with_enhanced_capabilities(agent_id: int, result: AgentResult,
                                     binary_chars: BinaryCharacteristics,
                                     ground_truth: Optional[Dict[str, Any]] = None) -> Tuple[List[PatternMatch], AccuracyMetrics]:
    """
    Analyze result with enhanced capabilities and accuracy assessment.
    
    Returns:
        Tuple of (pattern_matches, accuracy_metrics)
    """
    
    # Pattern matching
    pattern_matches = pattern_database.match_patterns(result.data, binary_chars)
    
    # Accuracy assessment
    accuracy_metrics = accuracy_assessment.assess_agent_accuracy(
        agent_id, result, ground_truth, pattern_matches
    )
    
    return pattern_matches, accuracy_metrics


def get_comprehensive_analysis_report(agent_ids: List[int]) -> Dict[str, Any]:
    """Generate comprehensive analysis report for multiple agents"""
    
    report = {
        "timestamp": time.time(),
        "agents_analyzed": len(agent_ids),
        "pattern_database": {
            "total_patterns": len(pattern_database.patterns),
            "pattern_types": list(set(p["type"].value for p in pattern_database.patterns.values())),
            "learning_data_points": len(pattern_database.learning_data)
        },
        "agent_statistics": {},
        "overall_quality": {
            "excellent": 0,
            "good": 0,
            "fair": 0,
            "poor": 0,
            "failed": 0
        }
    }
    
    for agent_id in agent_ids:
        stats = accuracy_assessment.get_agent_statistics(agent_id)
        report["agent_statistics"][agent_id] = stats
        
        # Update overall quality distribution
        if "quality_distribution" in stats:
            for quality, count in stats["quality_distribution"].items():
                if quality in report["overall_quality"]:
                    report["overall_quality"][quality] += count
    
    return report