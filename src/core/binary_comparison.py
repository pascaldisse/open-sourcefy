"""
Binary Comparison Module
Advanced binary comparison and validation testing for the Matrix pipeline
"""

import hashlib
import logging
import struct
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .config_manager import get_config_manager
from .shared_components import MatrixLogger


class ComparisonType(Enum):
    """Types of binary comparison"""
    EXACT_MATCH = "exact_match"
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    SEMANTIC = "semantic"


class ComparisonResult(Enum):
    """Binary comparison results"""
    IDENTICAL = "identical"
    EQUIVALENT = "equivalent"
    SIMILAR = "similar"
    DIFFERENT = "different"
    ERROR = "error"


@dataclass
class BinaryMetrics:
    """Metrics extracted from binary analysis"""
    file_size: int = 0
    entry_point: int = 0
    section_count: int = 0
    function_count: int = 0
    import_count: int = 0
    export_count: int = 0
    entropy: float = 0.0
    hash_md5: str = ""
    hash_sha256: str = ""
    
    # PE-specific metrics
    pe_characteristics: int = 0
    pe_subsystem: int = 0
    pe_machine_type: int = 0
    
    # Section information
    sections: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Comprehensive binary comparison report"""
    comparison_id: str
    timestamp: float
    comparison_type: ComparisonType
    result: ComparisonResult
    similarity_score: float
    
    # File information
    original_binary: str
    compared_binary: str
    
    # Metrics comparison
    original_metrics: BinaryMetrics
    compared_metrics: BinaryMetrics
    
    # Detailed analysis
    structural_differences: List[Dict[str, Any]] = field(default_factory=list)
    functional_differences: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    comparison_time: float = 0.0
    
    # Summary
    differences_summary: Dict[str, Any] = field(default_factory=dict)


class BinaryComparator:
    """
    Advanced binary comparison engine for validation testing
    
    Provides multiple levels of binary comparison:
    - Exact byte-for-byte comparison
    - Structural comparison (headers, sections, imports/exports)
    - Functional comparison (control flow, function signatures)
    - Semantic comparison (behavior equivalence)
    """
    
    def __init__(self):
        self.config = get_config_manager()
        self.logger = MatrixLogger.get_logger("binary_comparator")
        
        # Comparison thresholds
        self.similarity_threshold = self.config.get_value('comparison.similarity_threshold', 0.85)
        self.structural_weight = self.config.get_value('comparison.structural_weight', 0.4)
        self.functional_weight = self.config.get_value('comparison.functional_weight', 0.6)
        
    def compare_binaries(
        self,
        original_path: str,
        compared_path: str,
        comparison_type: ComparisonType = ComparisonType.STRUCTURAL
    ) -> float:
        """
        Compare two binaries and return similarity score
        
        Args:
            original_path: Path to original binary
            compared_path: Path to compared binary
            comparison_type: Type of comparison to perform
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"Comparing binaries: {Path(original_path).name} vs {Path(compared_path).name}")
            
            # Generate comparison report
            report = self._generate_comparison_report(original_path, compared_path, comparison_type)
            
            report.comparison_time = time.time() - start_time
            
            self.logger.info(
                f"Binary comparison completed: {report.result.value} "
                f"(Similarity: {report.similarity_score:.3f})"
            )
            
            return report.similarity_score
            
        except Exception as e:
            self.logger.error(f"Binary comparison failed: {e}", exc_info=True)
            return 0.0
    
    def generate_detailed_comparison(
        self,
        original_path: str,
        compared_path: str,
        comparison_type: ComparisonType = ComparisonType.FUNCTIONAL
    ) -> ComparisonReport:
        """
        Generate detailed comparison report
        
        Args:
            original_path: Path to original binary
            compared_path: Path to compared binary
            comparison_type: Type of comparison to perform
            
        Returns:
            Detailed comparison report
        """
        try:
            return self._generate_comparison_report(original_path, compared_path, comparison_type)
        except Exception as e:
            self.logger.error(f"Failed to generate comparison report: {e}", exc_info=True)
            # Return error report
            return ComparisonReport(
                comparison_id="error",
                timestamp=time.time(),
                comparison_type=comparison_type,
                result=ComparisonResult.ERROR,
                similarity_score=0.0,
                original_binary=original_path,
                compared_binary=compared_path,
                original_metrics=BinaryMetrics(),
                compared_metrics=BinaryMetrics()
            )
    
    def _generate_comparison_report(
        self,
        original_path: str,
        compared_path: str,
        comparison_type: ComparisonType
    ) -> ComparisonReport:
        """Generate comprehensive comparison report"""
        
        # Generate comparison ID
        comparison_id = hashlib.md5(f"{original_path}_{compared_path}_{time.time()}".encode()).hexdigest()[:8]
        
        # Extract metrics from both binaries
        original_metrics = self._extract_binary_metrics(original_path)
        compared_metrics = self._extract_binary_metrics(compared_path)
        
        # Initialize report
        report = ComparisonReport(
            comparison_id=comparison_id,
            timestamp=time.time(),
            comparison_type=comparison_type,
            result=ComparisonResult.DIFFERENT,
            similarity_score=0.0,
            original_binary=original_path,
            compared_binary=compared_path,
            original_metrics=original_metrics,
            compared_metrics=compared_metrics
        )
        
        # Perform comparison based on type
        if comparison_type == ComparisonType.EXACT_MATCH:
            similarity_score = self._compare_exact_match(original_metrics, compared_metrics)
        elif comparison_type == ComparisonType.STRUCTURAL:
            similarity_score = self._compare_structural(original_metrics, compared_metrics, report)
        elif comparison_type == ComparisonType.FUNCTIONAL:
            similarity_score = self._compare_functional(original_metrics, compared_metrics, report)
        elif comparison_type == ComparisonType.SEMANTIC:
            similarity_score = self._compare_semantic(original_metrics, compared_metrics, report)
        else:
            similarity_score = self._compare_structural(original_metrics, compared_metrics, report)
        
        report.similarity_score = similarity_score
        
        # Determine result based on similarity score
        if similarity_score >= 0.99:
            report.result = ComparisonResult.IDENTICAL
        elif similarity_score >= self.similarity_threshold:
            report.result = ComparisonResult.EQUIVALENT
        elif similarity_score >= 0.5:
            report.result = ComparisonResult.SIMILAR
        else:
            report.result = ComparisonResult.DIFFERENT
        
        # Generate differences summary
        report.differences_summary = self._generate_differences_summary(original_metrics, compared_metrics)
        
        return report
    
    def _extract_binary_metrics(self, binary_path: str) -> BinaryMetrics:
        """Extract comprehensive metrics from binary file"""
        
        binary_file = Path(binary_path)
        if not binary_file.exists():
            self.logger.warning(f"Binary file not found: {binary_path}")
            return BinaryMetrics()
        
        try:
            metrics = BinaryMetrics()
            
            # Basic file metrics
            metrics.file_size = binary_file.stat().st_size
            
            # Calculate hashes
            with open(binary_file, 'rb') as f:
                content = f.read()
                metrics.hash_md5 = hashlib.md5(content).hexdigest()
                metrics.hash_sha256 = hashlib.sha256(content).hexdigest()
                metrics.entropy = self._calculate_entropy(content)
            
            # Check if this is a PE file
            if self._is_pe_file(binary_path):
                self._extract_pe_metrics(binary_path, metrics)
            else:
                # For non-PE files or mock binaries, use basic analysis
                metrics.section_count = 1
                metrics.function_count = max(1, metrics.file_size // 1000)  # Rough estimate
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to extract metrics from {binary_path}: {e}")
            return BinaryMetrics()
    
    def _is_pe_file(self, binary_path: str) -> bool:
        """Check if file is a valid PE executable"""
        try:
            with open(binary_path, 'rb') as f:
                # Check MZ header
                mz_header = f.read(2)
                if mz_header != b'MZ':
                    return False
                
                # Get PE offset
                f.seek(60)
                pe_offset_bytes = f.read(4)
                if len(pe_offset_bytes) < 4:
                    return False
                
                pe_offset = struct.unpack('<L', pe_offset_bytes)[0]
                
                # Check PE signature
                f.seek(pe_offset)
                pe_signature = f.read(4)
                return pe_signature == b'PE\x00\x00'
                
        except Exception as e:
            self.logger.debug(f"PE check failed for {binary_path}: {e}")
            return False
    
    def _extract_pe_metrics(self, binary_path: str, metrics: BinaryMetrics):
        """Extract PE-specific metrics"""
        try:
            with open(binary_path, 'rb') as f:
                # Read DOS header
                f.seek(60)
                pe_offset = struct.unpack('<L', f.read(4))[0]
                
                # Read PE header
                f.seek(pe_offset + 4)  # Skip PE signature
                coff_header = f.read(20)
                
                if len(coff_header) >= 20:
                    machine, num_sections, _, _, _, _, characteristics = struct.unpack('<HHIIIHH', coff_header)
                    
                    metrics.pe_machine_type = machine
                    metrics.pe_characteristics = characteristics
                    metrics.section_count = num_sections
                
                # Read optional header for additional info
                f.seek(pe_offset + 24)
                opt_header_size = struct.unpack('<H', f.read(2))[0]
                
                if opt_header_size >= 68:
                    f.seek(pe_offset + 24 + 2)  # Skip size field
                    magic = struct.unpack('<H', f.read(2))[0]
                    
                    if magic == 0x10b:  # PE32
                        f.seek(pe_offset + 24 + 16)
                        metrics.entry_point = struct.unpack('<L', f.read(4))[0]
                        f.seek(pe_offset + 24 + 68)
                        metrics.pe_subsystem = struct.unpack('<H', f.read(2))[0]
                    elif magic == 0x20b:  # PE32+
                        f.seek(pe_offset + 24 + 16)
                        metrics.entry_point = struct.unpack('<L', f.read(4))[0]
                        f.seek(pe_offset + 24 + 84)
                        metrics.pe_subsystem = struct.unpack('<H', f.read(2))[0]
                
                # Estimate function count based on file size and complexity
                metrics.function_count = max(1, metrics.file_size // 5000)
                
        except Exception as e:
            self.logger.debug(f"PE metrics extraction failed: {e}")
            # Set defaults
            metrics.section_count = 1
            metrics.function_count = 1
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of binary data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    def _compare_exact_match(self, original: BinaryMetrics, compared: BinaryMetrics) -> float:
        """Perform exact byte-for-byte comparison"""
        # Check if hashes match
        if original.hash_sha256 == compared.hash_sha256:
            return 1.0
        else:
            return 0.0
    
    def _compare_structural(self, original: BinaryMetrics, compared: BinaryMetrics, report: ComparisonReport) -> float:
        """Perform structural comparison of binary elements"""
        
        score_components = []
        
        # File size similarity (with tolerance)
        if original.file_size > 0:
            size_ratio = min(original.file_size, compared.file_size) / max(original.file_size, compared.file_size)
            score_components.append(('file_size', size_ratio, 0.1))
        
        # Section count similarity
        if original.section_count > 0:
            section_ratio = min(original.section_count, compared.section_count) / max(original.section_count, compared.section_count)
            score_components.append(('section_count', section_ratio, 0.2))
        
        # Entry point similarity (for PE files)
        if original.entry_point > 0 and compared.entry_point > 0:
            if original.entry_point == compared.entry_point:
                entry_score = 1.0
            else:
                # Allow some variance in entry point
                entry_diff = abs(original.entry_point - compared.entry_point)
                entry_score = max(0.0, 1.0 - (entry_diff / max(original.entry_point, compared.entry_point)))
            score_components.append(('entry_point', entry_score, 0.15))
        
        # PE characteristics similarity
        if original.pe_characteristics > 0 and compared.pe_characteristics > 0:
            if original.pe_characteristics == compared.pe_characteristics:
                char_score = 1.0
            else:
                # Compare bit-by-bit for characteristics
                char_xor = original.pe_characteristics ^ compared.pe_characteristics
                char_score = 1.0 - (bin(char_xor).count('1') / 16)  # 16 bits max
            score_components.append(('pe_characteristics', char_score, 0.1))
        
        # Machine type similarity
        if original.pe_machine_type > 0 and compared.pe_machine_type > 0:
            machine_score = 1.0 if original.pe_machine_type == compared.pe_machine_type else 0.0
            score_components.append(('machine_type', machine_score, 0.15))
        
        # Function count similarity
        if original.function_count > 0:
            func_ratio = min(original.function_count, compared.function_count) / max(original.function_count, compared.function_count)
            score_components.append(('function_count', func_ratio, 0.2))
        
        # Entropy similarity
        if original.entropy > 0:
            entropy_diff = abs(original.entropy - compared.entropy)
            entropy_score = max(0.0, 1.0 - (entropy_diff / 8.0))  # Max entropy is ~8
            score_components.append(('entropy', entropy_score, 0.1))
        
        # Calculate weighted score
        if score_components:
            total_score = sum(score * weight for _, score, weight in score_components)
            total_weight = sum(weight for _, _, weight in score_components)
            final_score = total_score / total_weight if total_weight > 0 else 0.0
        else:
            final_score = 0.0
        
        # Record structural differences
        for component, score, weight in score_components:
            if score < 0.9:  # Significant difference
                report.structural_differences.append({
                    'component': component,
                    'similarity': score,
                    'weight': weight,
                    'original_value': getattr(original, component, None),
                    'compared_value': getattr(compared, component, None)
                })
        
        return final_score
    
    def _compare_functional(self, original: BinaryMetrics, compared: BinaryMetrics, report: ComparisonReport) -> float:
        """Perform functional comparison (combines structural with function analysis)"""
        
        # Start with structural comparison
        structural_score = self._compare_structural(original, compared, report)
        
        # Add functional elements
        functional_components = []
        
        # Import/Export similarity (if available)
        if original.imports and compared.imports:
            import_intersection = set(original.imports) & set(compared.imports)
            import_union = set(original.imports) | set(compared.imports)
            import_similarity = len(import_intersection) / len(import_union) if import_union else 0.0
            functional_components.append(('imports', import_similarity, 0.3))
        
        if original.exports and compared.exports:
            export_intersection = set(original.exports) & set(compared.exports)
            export_union = set(original.exports) | set(compared.exports)
            export_similarity = len(export_intersection) / len(export_union) if export_union else 0.0
            functional_components.append(('exports', export_similarity, 0.2))
        
        # Calculate functional score
        if functional_components:
            func_total = sum(score * weight for _, score, weight in functional_components)
            func_weight = sum(weight for _, _, weight in functional_components)
            functional_score = func_total / func_weight if func_weight > 0 else 0.0
        else:
            functional_score = structural_score  # Fall back to structural
        
        # Combine structural and functional scores
        combined_score = (structural_score * self.structural_weight + 
                         functional_score * self.functional_weight)
        
        # Record functional differences
        for component, score, weight in functional_components:
            if score < 0.8:
                report.functional_differences.append({
                    'component': component,
                    'similarity': score,
                    'weight': weight
                })
        
        return combined_score
    
    def _compare_semantic(self, original: BinaryMetrics, compared: BinaryMetrics, report: ComparisonReport) -> float:
        """Perform semantic comparison (behavior equivalence)"""
        
        # For now, semantic comparison falls back to functional comparison
        # In a full implementation, this would involve:
        # - Control flow graph comparison
        # - API call sequence analysis
        # - Dynamic behavior analysis
        
        functional_score = self._compare_functional(original, compared, report)
        
        # Add semantic adjustments based on available metrics
        semantic_adjustments = []
        
        # Subsystem compatibility
        if original.pe_subsystem > 0 and compared.pe_subsystem > 0:
            subsystem_match = 1.0 if original.pe_subsystem == compared.pe_subsystem else 0.7
            semantic_adjustments.append(subsystem_match)
        
        # Apply semantic adjustments
        if semantic_adjustments:
            semantic_factor = sum(semantic_adjustments) / len(semantic_adjustments)
            semantic_score = functional_score * semantic_factor
        else:
            semantic_score = functional_score
        
        return semantic_score
    
    def _generate_differences_summary(self, original: BinaryMetrics, compared: BinaryMetrics) -> Dict[str, Any]:
        """Generate summary of key differences"""
        
        differences = {}
        
        # Size differences
        if original.file_size != compared.file_size:
            size_diff = compared.file_size - original.file_size
            size_change = (size_diff / original.file_size * 100) if original.file_size > 0 else 0
            differences['file_size'] = {
                'original': original.file_size,
                'compared': compared.file_size,
                'difference': size_diff,
                'percentage_change': size_change
            }
        
        # Section count differences
        if original.section_count != compared.section_count:
            differences['section_count'] = {
                'original': original.section_count,
                'compared': compared.section_count,
                'difference': compared.section_count - original.section_count
            }
        
        # Entry point differences
        if original.entry_point != compared.entry_point:
            differences['entry_point'] = {
                'original': hex(original.entry_point) if original.entry_point else '0x0',
                'compared': hex(compared.entry_point) if compared.entry_point else '0x0'
            }
        
        # Hash differences (always different unless identical)
        if original.hash_sha256 != compared.hash_sha256:
            differences['content_hash'] = {
                'original': original.hash_sha256[:16] + '...',
                'compared': compared.hash_sha256[:16] + '...',
                'identical': False
            }
        
        return differences
    
    def save_comparison_report(self, report: ComparisonReport, output_path: str) -> str:
        """Save comparison report to JSON file"""
        try:
            import json
            
            report_data = {
                'comparison_id': report.comparison_id,
                'timestamp': report.timestamp,
                'comparison_type': report.comparison_type.value,
                'result': report.result.value,
                'similarity_score': report.similarity_score,
                'comparison_time': report.comparison_time,
                'files': {
                    'original': report.original_binary,
                    'compared': report.compared_binary
                },
                'metrics': {
                    'original': self._metrics_to_dict(report.original_metrics),
                    'compared': self._metrics_to_dict(report.compared_metrics)
                },
                'differences': {
                    'structural': report.structural_differences,
                    'functional': report.functional_differences,
                    'summary': report.differences_summary
                }
            }
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Comparison report saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save comparison report: {e}")
            raise
    
    def _metrics_to_dict(self, metrics: BinaryMetrics) -> Dict[str, Any]:
        """Convert BinaryMetrics to dictionary"""
        return {
            'file_size': metrics.file_size,
            'entry_point': hex(metrics.entry_point) if metrics.entry_point else '0x0',
            'section_count': metrics.section_count,
            'function_count': metrics.function_count,
            'import_count': metrics.import_count,
            'export_count': metrics.export_count,
            'entropy': metrics.entropy,
            'hash_md5': metrics.hash_md5,
            'hash_sha256': metrics.hash_sha256,
            'pe_characteristics': hex(metrics.pe_characteristics) if metrics.pe_characteristics else '0x0',
            'pe_subsystem': metrics.pe_subsystem,
            'pe_machine_type': hex(metrics.pe_machine_type) if metrics.pe_machine_type else '0x0',
            'sections': metrics.sections,
            'imports': metrics.imports[:10],  # Limit for readability
            'exports': metrics.exports[:10]   # Limit for readability
        }


def create_binary_comparator() -> BinaryComparator:
    """Factory function to create binary comparator"""
    return BinaryComparator()


if __name__ == "__main__":
    # Example usage
    comparator = create_binary_comparator()
    
    # Mock comparison
    original_binary = "original.exe"
    recompiled_binary = "recompiled.exe"
    
    try:
        similarity = comparator.compare_binaries(original_binary, recompiled_binary)
        print(f"Similarity Score: {similarity:.3f}")
        
        # Generate detailed report
        report = comparator.generate_detailed_comparison(
            original_binary, 
            recompiled_binary, 
            ComparisonType.FUNCTIONAL
        )
        
        print(f"Comparison Result: {report.result.value}")
        print(f"Structural Differences: {len(report.structural_differences)}")
        print(f"Functional Differences: {len(report.functional_differences)}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")