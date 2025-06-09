"""
Advanced Entropy Analysis Module for Binary Deobfuscation

Implements Shannon entropy analysis for:
- Packed section detection (threshold >5.9)
- Encrypted content identification
- Code vs data classification
- Compression detection

Part of Phase 1: Foundational Analysis and Deobfuscation
"""

import math
import struct
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import pefile


@dataclass
class EntropyResult:
    """Result of entropy analysis on binary data."""
    entropy: float
    confidence: float
    classification: str
    packed_probability: float
    encrypted_probability: float
    metadata: Dict[str, Any]


@dataclass
class SectionAnalysis:
    """Analysis result for a binary section."""
    name: str
    offset: int
    size: int
    entropy: float
    classification: str
    packed: bool
    encrypted: bool
    suspicious: bool


class EntropyAnalyzer:
    """
    Advanced entropy analyzer for binary deobfuscation.
    
    Uses Shannon entropy to identify:
    - Packed sections (high entropy >5.9)
    - Code sections (medium entropy 4.5-6.5)
    - Data sections (low entropy <4.5)
    - Encrypted content (very high entropy >7.5)
    """
    
    # Classification thresholds based on research
    PACKED_THRESHOLD = 5.9
    ENCRYPTED_THRESHOLD = 7.5
    CODE_MIN_ENTROPY = 4.5
    CODE_MAX_ENTROPY = 6.5
    
    def __init__(self, config_manager=None):
        """Initialize entropy analyzer with configuration."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Configurable thresholds
        if self.config:
            self.PACKED_THRESHOLD = self.config.get_value('entropy.packed_threshold', 5.9)
            self.ENCRYPTED_THRESHOLD = self.config.get_value('entropy.encrypted_threshold', 7.5)
            self.CODE_MIN_ENTROPY = self.config.get_value('entropy.code_min', 4.5)
            self.CODE_MAX_ENTROPY = self.config.get_value('entropy.code_max', 6.5)
    
    def calculate_shannon_entropy(self, data: bytes) -> float:
        """
        Calculate Shannon entropy for binary data.
        
        Args:
            data: Binary data to analyze
            
        Returns:
            Float between 0.0 (no entropy) and 8.0 (maximum entropy)
            
        Raises:
            ValueError: If data is empty
        """
        if not data:
            raise ValueError("Data cannot be empty for entropy calculation")
        
        # Count byte frequencies
        byte_counts = Counter(data)
        data_len = len(data)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in byte_counts.values():
            if count > 0:
                probability = count / data_len
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def analyze_block_entropy(self, data: bytes, block_size: int = 256) -> List[float]:
        """
        Calculate entropy for blocks within binary data.
        
        Args:
            data: Binary data to analyze
            block_size: Size of analysis blocks in bytes
            
        Returns:
            List of entropy values for each block
        """
        if block_size <= 0:
            raise ValueError("Block size must be positive")
        
        entropies = []
        for i in range(0, len(data), block_size):
            block = data[i:i + block_size]
            if len(block) >= 32:  # Minimum block size for meaningful entropy
                entropy = self.calculate_shannon_entropy(block)
                entropies.append(entropy)
        
        return entropies
    
    def classify_entropy(self, entropy: float) -> Tuple[str, float, bool, bool]:
        """
        Classify data based on entropy value.
        
        Args:
            entropy: Calculated entropy value
            
        Returns:
            Tuple of (classification, confidence, is_packed, is_encrypted)
        """
        if entropy > self.ENCRYPTED_THRESHOLD:
            return ("encrypted", 0.95, True, True)
        elif entropy > self.PACKED_THRESHOLD:
            return ("packed", 0.85, True, False)
        elif self.CODE_MIN_ENTROPY <= entropy <= self.CODE_MAX_ENTROPY:
            return ("code", 0.75, False, False)
        elif entropy < self.CODE_MIN_ENTROPY:
            return ("data", 0.70, False, False)
        else:
            return ("unknown", 0.50, False, False)
    
    def analyze_binary_data(self, data: bytes) -> EntropyResult:
        """
        Perform comprehensive entropy analysis on binary data.
        
        Args:
            data: Binary data to analyze
            
        Returns:
            EntropyResult with detailed analysis
        """
        try:
            # Calculate overall entropy
            overall_entropy = self.calculate_shannon_entropy(data)
            
            # Classify based on entropy
            classification, confidence, is_packed, is_encrypted = self.classify_entropy(overall_entropy)
            
            # Calculate probabilities
            packed_prob = max(0.0, (overall_entropy - self.PACKED_THRESHOLD) / (8.0 - self.PACKED_THRESHOLD))
            encrypted_prob = max(0.0, (overall_entropy - self.ENCRYPTED_THRESHOLD) / (8.0 - self.ENCRYPTED_THRESHOLD))
            
            # Block-wise analysis for additional insights
            block_entropies = self.analyze_block_entropy(data)
            
            metadata = {
                'data_size': len(data),
                'block_entropies': block_entropies,
                'entropy_variance': self._calculate_variance(block_entropies) if block_entropies else 0.0,
                'high_entropy_blocks': sum(1 for e in block_entropies if e > self.PACKED_THRESHOLD),
                'analysis_timestamp': self._get_timestamp()
            }
            
            return EntropyResult(
                entropy=overall_entropy,
                confidence=confidence,
                classification=classification,
                packed_probability=packed_prob,
                encrypted_probability=encrypted_prob,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in entropy analysis: {e}")
            return EntropyResult(
                entropy=0.0,
                confidence=0.0,
                classification="error",
                packed_probability=0.0,
                encrypted_probability=0.0,
                metadata={'error': str(e)}
            )
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of entropy values."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()


class PackedSectionDetector:
    """
    Specialized detector for packed sections in PE binaries.
    
    Analyzes PE sections using entropy analysis to identify:
    - Packed executable sections
    - Compressed resources
    - Obfuscated code sections
    - Potential malware packers
    """
    
    def __init__(self, entropy_analyzer: EntropyAnalyzer = None):
        """Initialize with entropy analyzer."""
        self.entropy_analyzer = entropy_analyzer or EntropyAnalyzer()
        self.logger = logging.getLogger(__name__)
    
    def analyze_pe_file(self, pe_path: Path) -> List[SectionAnalysis]:
        """
        Analyze all sections in a PE file for packing indicators.
        
        Args:
            pe_path: Path to PE file to analyze
            
        Returns:
            List of SectionAnalysis results for each section
        """
        try:
            with open(pe_path, 'rb') as f:
                pe_data = f.read()
            
            pe = pefile.PE(data=pe_data, fast_load=True)
            section_analyses = []
            
            for section in pe.sections:
                # Extract section data
                section_data = section.get_data()
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                
                if len(section_data) < 32:  # Skip tiny sections
                    continue
                
                # Analyze entropy
                entropy_result = self.entropy_analyzer.analyze_binary_data(section_data)
                
                # Determine if section is suspicious
                is_suspicious = self._is_section_suspicious(section, entropy_result)
                
                analysis = SectionAnalysis(
                    name=section_name,
                    offset=section.PointerToRawData,
                    size=len(section_data),
                    entropy=entropy_result.entropy,
                    classification=entropy_result.classification,
                    packed=entropy_result.packed_probability > 0.7,
                    encrypted=entropy_result.encrypted_probability > 0.7,
                    suspicious=is_suspicious
                )
                
                section_analyses.append(analysis)
                
                self.logger.debug(f"Section {section_name}: entropy={entropy_result.entropy:.2f}, "
                                f"classification={entropy_result.classification}")
            
            return section_analyses
            
        except Exception as e:
            self.logger.error(f"Error analyzing PE file {pe_path}: {e}")
            return []
    
    def _is_section_suspicious(self, section, entropy_result: EntropyResult) -> bool:
        """
        Determine if a section is suspicious based on multiple factors.
        
        Args:
            section: PE section object
            entropy_result: Entropy analysis result
            
        Returns:
            True if section appears suspicious
        """
        # High entropy in executable sections is suspicious
        if (section.Characteristics & 0x20000000) and entropy_result.entropy > 6.5:  # IMAGE_SCN_MEM_EXECUTE
            return True
        
        # Very high entropy anywhere is suspicious
        if entropy_result.entropy > 7.5:
            return True
        
        # Large sections with high entropy
        if len(section.get_data()) > 10000 and entropy_result.entropy > 6.0:
            return True
        
        # Entropy variance indicating mixed content
        variance = entropy_result.metadata.get('entropy_variance', 0.0)
        if variance > 2.0:
            return True
        
        return False
    
    def detect_packer_signatures(self, pe_path: Path) -> Dict[str, Any]:
        """
        Detect common packer signatures using entropy patterns.
        
        Args:
            pe_path: Path to PE file
            
        Returns:
            Dictionary with packer detection results
        """
        try:
            section_analyses = self.analyze_pe_file(pe_path)
            
            # Analyze patterns
            high_entropy_sections = [s for s in section_analyses if s.entropy > 6.5]
            executable_packed = [s for s in section_analyses 
                               if s.packed and 'text' in s.name.lower()]
            
            # Common packer indicators
            indicators = {
                'high_entropy_sections': len(high_entropy_sections),
                'packed_executable_sections': len(executable_packed),
                'total_sections': len(section_analyses),
                'max_entropy': max(s.entropy for s in section_analyses) if section_analyses else 0.0,
                'suspicious_sections': [s.name for s in section_analyses if s.suspicious]
            }
            
            # Determine likely packer
            likely_packer = self._identify_packer_type(indicators, section_analyses)
            
            return {
                'packer_detected': len(high_entropy_sections) > 0,
                'likely_packer': likely_packer,
                'confidence': self._calculate_packer_confidence(indicators),
                'indicators': indicators,
                'section_analyses': section_analyses
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting packer signatures: {e}")
            return {
                'packer_detected': False,
                'likely_packer': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _identify_packer_type(self, indicators: Dict, sections: List[SectionAnalysis]) -> str:
        """Identify likely packer type based on analysis."""
        # UPX pattern: high entropy .UPX sections
        upx_sections = [s for s in sections if 'UPX' in s.name.upper()]
        if upx_sections:
            return 'UPX'
        
        # Themida pattern: very high entropy, many sections
        if indicators['max_entropy'] > 7.5 and indicators['total_sections'] > 8:
            return 'Themida'
        
        # VMProtect pattern: moderate high entropy with specific section names
        vmp_indicators = [s for s in sections if any(name in s.name.lower() 
                         for name in ['.vmp', 'vprotect'])]
        if vmp_indicators:
            return 'VMProtect'
        
        # Generic packer
        if indicators['high_entropy_sections'] > 1:
            return 'Generic Packer'
        
        return 'Unknown'
    
    def _calculate_packer_confidence(self, indicators: Dict) -> float:
        """Calculate confidence score for packer detection."""
        confidence = 0.0
        
        # High entropy sections increase confidence
        confidence += min(indicators['high_entropy_sections'] * 0.3, 0.6)
        
        # Packed executable sections are strong indicators
        confidence += indicators['packed_executable_sections'] * 0.4
        
        # Very high max entropy is a strong indicator
        if indicators['max_entropy'] > 7.5:
            confidence += 0.3
        elif indicators['max_entropy'] > 6.5:
            confidence += 0.2
        
        return min(confidence, 1.0)


# Factory function for easy instantiation
def create_entropy_analyzer(config_manager=None) -> EntropyAnalyzer:
    """Create configured entropy analyzer instance."""
    return EntropyAnalyzer(config_manager)


def create_packed_section_detector(config_manager=None) -> PackedSectionDetector:
    """Create configured packed section detector instance."""
    entropy_analyzer = create_entropy_analyzer(config_manager)
    return PackedSectionDetector(entropy_analyzer)


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    analyzer = EntropyAnalyzer()
    
    # Test with sample data
    test_data = b"A" * 1000  # Low entropy data
    result = analyzer.analyze_binary_data(test_data)
    print(f"Test data entropy: {result.entropy:.2f}, classification: {result.classification}")
    
    # Test with random data (high entropy)
    import os
    random_data = os.urandom(1000)
    result = analyzer.analyze_binary_data(random_data)
    print(f"Random data entropy: {result.entropy:.2f}, classification: {result.classification}")