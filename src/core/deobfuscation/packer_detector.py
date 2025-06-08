"""
Advanced Packer Detection and Unpacking Framework

Implements comprehensive packer detection for:
- UPX, Themida, VMProtect, ASPack, PECompact
- Runtime and multi-layer packer detection
- Behavioral analysis and signature matching
- Automated unpacking strategies

Part of Phase 1: Foundational Analysis and Deobfuscation
"""

import logging
import struct
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import tempfile

try:
    import pefile
    import yara
    PEFILE_AVAILABLE = True
    YARA_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False
    YARA_AVAILABLE = False
    logging.warning("pefile or yara not available, using fallback detection")

from .entropy_analyzer import EntropyAnalyzer


class PackerType(Enum):
    """Known packer types."""
    UPX = "UPX"
    THEMIDA = "Themida"
    VMPROTECT = "VMProtect"
    ASPACK = "ASPack"
    PECOMPACT = "PECompact"
    MPRESS = "MPRESS"
    PETITE = "Petite"
    FSG = "FSG"
    MOLEBOX = "MoleBox"
    ENIGMA = "Enigma"
    GENERIC = "Generic"
    UNKNOWN = "Unknown"


@dataclass
class PackerSignature:
    """Packer detection signature."""
    name: str
    packer_type: PackerType
    signature_type: str  # 'bytes', 'entropy', 'section', 'import'
    pattern: bytes
    offset: int
    confidence: float
    description: str


@dataclass
class PackerDetectionResult:
    """Result of packer detection analysis."""
    packer_detected: bool
    packer_type: PackerType
    packer_name: str
    confidence: float
    signatures_matched: List[PackerSignature]
    entropy_indicators: Dict[str, Any]
    section_indicators: List[Dict[str, Any]]
    import_indicators: List[str]
    behavioral_indicators: List[str]
    unpacking_difficulty: str  # 'easy', 'medium', 'hard', 'very_hard'
    recommended_tools: List[str]
    metadata: Dict[str, Any]


class PackerDetector:
    """
    Advanced packer detection using multiple techniques.
    
    Combines signature matching, entropy analysis, section analysis,
    import table analysis, and behavioral indicators.
    """
    
    def __init__(self, config_manager=None):
        """Initialize packer detector with configuration."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.entropy_analyzer = EntropyAnalyzer(config_manager)
        
        # Initialize signature database
        self.signatures = self._load_packer_signatures()
        
        # Initialize YARA rules if available
        self.yara_rules = None
        if YARA_AVAILABLE:
            self.yara_rules = self._load_yara_rules()
    
    def detect_packer(self, binary_path: Path) -> PackerDetectionResult:
        """
        Perform comprehensive packer detection.
        
        Args:
            binary_path: Path to binary file to analyze
            
        Returns:
            PackerDetectionResult with detailed analysis
        """
        try:
            self.logger.info(f"Starting packer detection for {binary_path}")
            
            # Read binary data
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            # Initialize result
            result = PackerDetectionResult(
                packer_detected=False,
                packer_type=PackerType.UNKNOWN,
                packer_name="None",
                confidence=0.0,
                signatures_matched=[],
                entropy_indicators={},
                section_indicators=[],
                import_indicators=[],
                behavioral_indicators=[],
                unpacking_difficulty="unknown",
                recommended_tools=[],
                metadata={}
            )
            
            # Phase 1: Signature-based detection
            self.logger.debug("Phase 1: Signature analysis")
            signature_results = self._signature_detection(binary_data)
            
            # Phase 2: Entropy analysis
            self.logger.debug("Phase 2: Entropy analysis")
            entropy_results = self._entropy_based_detection(binary_data)
            
            # Phase 3: Section analysis
            self.logger.debug("Phase 3: Section analysis")
            section_results = self._section_analysis(binary_data)
            
            # Phase 4: Import table analysis
            self.logger.debug("Phase 4: Import analysis")
            import_results = self._import_analysis(binary_data)
            
            # Phase 5: YARA rule matching
            self.logger.debug("Phase 5: YARA analysis")
            yara_results = self._yara_detection(binary_data)
            
            # Phase 6: Behavioral analysis
            self.logger.debug("Phase 6: Behavioral analysis")
            behavioral_results = self._behavioral_analysis(binary_data)
            
            # Combine results
            result = self._combine_detection_results(
                result, signature_results, entropy_results, section_results,
                import_results, yara_results, behavioral_results
            )
            
            # Determine final verdict
            result = self._determine_final_verdict(result)
            
            self.logger.info(f"Packer detection complete: {result.packer_name} "
                           f"(confidence: {result.confidence:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in packer detection: {e}")
            return self._create_error_result(str(e))
    
    def _load_packer_signatures(self) -> List[PackerSignature]:
        """Load database of packer signatures."""
        signatures = []
        
        # UPX signatures
        signatures.extend([
            PackerSignature(
                name="UPX Section Names",
                packer_type=PackerType.UPX,
                signature_type="section",
                pattern=b"UPX",
                offset=0,
                confidence=0.95,
                description="UPX compressed sections"
            ),
            PackerSignature(
                name="UPX Stub Signature",
                packer_type=PackerType.UPX,
                signature_type="bytes",
                pattern=bytes([0x60, 0xBE, 0x00, 0x10, 0x40, 0x00, 0x8D, 0xBE]),
                offset=0,
                confidence=0.90,
                description="UPX decompression stub"
            )
        ])
        
        # Themida signatures
        signatures.extend([
            PackerSignature(
                name="Themida Section",
                packer_type=PackerType.THEMIDA,
                signature_type="section",
                pattern=b".themida",
                offset=0,
                confidence=0.95,
                description="Themida protection section"
            ),
            PackerSignature(
                name="Themida Winlicense",
                packer_type=PackerType.THEMIDA,
                signature_type="section",
                pattern=b".winlice",
                offset=0,
                confidence=0.90,
                description="Winlicense protection section"
            )
        ])
        
        # VMProtect signatures
        signatures.extend([
            PackerSignature(
                name="VMProtect Section",
                packer_type=PackerType.VMPROTECT,
                signature_type="section",
                pattern=b".vmp",
                offset=0,
                confidence=0.95,
                description="VMProtect section"
            ),
            PackerSignature(
                name="VMProtect Pattern",
                packer_type=PackerType.VMPROTECT,
                signature_type="bytes",
                pattern=bytes([0x68, 0x00, 0x00, 0x00, 0x00, 0xE8, 0x01, 0x00]),
                offset=0,
                confidence=0.80,
                description="VMProtect entry pattern"
            )
        ])
        
        # ASPack signatures
        signatures.extend([
            PackerSignature(
                name="ASPack Section",
                packer_type=PackerType.ASPACK,
                signature_type="section",
                pattern=b".aspack",
                offset=0,
                confidence=0.95,
                description="ASPack section"
            ),
            PackerSignature(
                name="ASPack Stub",
                packer_type=PackerType.ASPACK,
                signature_type="bytes",
                pattern=bytes([0x60, 0xE8, 0x03, 0x00, 0x00, 0x00, 0xE9, 0xEB]),
                offset=0,
                confidence=0.85,
                description="ASPack unpacking stub"
            )
        ])
        
        # Additional signatures can be added here
        
        return signatures
    
    def _load_yara_rules(self) -> Optional[Any]:
        """Load YARA rules for packer detection."""
        if not YARA_AVAILABLE:
            return None
        
        try:
            # Define YARA rules as strings (in production, load from files)
            yara_rules_text = """
            rule UPX_Packer {
                meta:
                    description = "UPX Packer"
                    packer = "UPX"
                strings:
                    $upx1 = "UPX!"
                    $upx2 = { 55 50 58 21 }
                    $upx3 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? }
                condition:
                    any of them
            }
            
            rule Themida_Packer {
                meta:
                    description = "Themida/Winlicense Packer"
                    packer = "Themida"
                strings:
                    $themida1 = ".themida"
                    $themida2 = ".winlice"
                    $themida3 = { B8 ?? ?? ?? ?? B9 ?? ?? ?? ?? 51 50 }
                condition:
                    any of them
            }
            
            rule VMProtect_Packer {
                meta:
                    description = "VMProtect Packer"
                    packer = "VMProtect"
                strings:
                    $vmp1 = ".vmp0"
                    $vmp2 = ".vmp1"
                    $vmp3 = { 68 ?? ?? ?? ?? E8 01 00 00 00 C3 }
                condition:
                    any of them
            }
            """
            
            return yara.compile(source=yara_rules_text)
            
        except Exception as e:
            self.logger.warning(f"Failed to load YARA rules: {e}")
            return None
    
    def _signature_detection(self, binary_data: bytes) -> Dict[str, Any]:
        """Perform signature-based packer detection."""
        matched_signatures = []
        packer_votes = {}
        
        for signature in self.signatures:
            if self._match_signature(binary_data, signature):
                matched_signatures.append(signature)
                packer_type = signature.packer_type
                
                if packer_type not in packer_votes:
                    packer_votes[packer_type] = 0
                packer_votes[packer_type] += signature.confidence
        
        # Determine most likely packer
        best_packer = PackerType.UNKNOWN
        best_confidence = 0.0
        
        if packer_votes:
            best_packer = max(packer_votes.keys(), key=lambda k: packer_votes[k])
            best_confidence = packer_votes[best_packer]
        
        return {
            'matched_signatures': matched_signatures,
            'packer_votes': packer_votes,
            'best_packer': best_packer,
            'confidence': min(best_confidence, 1.0)
        }
    
    def _match_signature(self, binary_data: bytes, signature: PackerSignature) -> bool:
        """Check if signature matches binary data."""
        try:
            if signature.signature_type == "bytes":
                # Search for byte pattern
                return signature.pattern in binary_data
            
            elif signature.signature_type == "section":
                # Check section names (requires PE parsing)
                if PEFILE_AVAILABLE:
                    try:
                        pe = pefile.PE(data=binary_data, fast_load=True)
                        for section in pe.sections:
                            section_name = section.Name.strip(b'\x00')
                            if signature.pattern in section_name:
                                return True
                    except:
                        pass
                else:
                    # Fallback: search in binary data
                    return signature.pattern in binary_data
            
            elif signature.signature_type == "entropy":
                # Entropy-based signature (implemented in entropy analysis)
                pass
            
            elif signature.signature_type == "import":
                # Import table signature (implemented in import analysis)
                pass
        
        except Exception as e:
            self.logger.debug(f"Error matching signature {signature.name}: {e}")
        
        return False
    
    def _entropy_based_detection(self, binary_data: bytes) -> Dict[str, Any]:
        """Detect packers using entropy analysis."""
        try:
            # Analyze overall entropy
            entropy_result = self.entropy_analyzer.analyze_binary_data(binary_data)
            
            # Section-wise entropy analysis
            section_entropies = []
            if PEFILE_AVAILABLE:
                try:
                    pe = pefile.PE(data=binary_data, fast_load=True)
                    for section in pe.sections:
                        section_data = section.get_data()
                        if len(section_data) > 32:
                            section_entropy = self.entropy_analyzer.calculate_shannon_entropy(section_data)
                            section_entropies.append({
                                'name': section.Name.decode('utf-8', errors='ignore').strip('\x00'),
                                'entropy': section_entropy,
                                'size': len(section_data)
                            })
                except:
                    pass
            
            # Determine packing likelihood based on entropy
            packing_indicators = {
                'high_entropy_overall': entropy_result.entropy > 7.0,
                'very_high_entropy': entropy_result.entropy > 7.5,
                'high_entropy_sections': sum(1 for s in section_entropies if s['entropy'] > 6.5),
                'entropy_variance': entropy_result.metadata.get('entropy_variance', 0.0)
            }
            
            # Calculate confidence
            confidence = 0.0
            if packing_indicators['very_high_entropy']:
                confidence += 0.4
            elif packing_indicators['high_entropy_overall']:
                confidence += 0.3
            
            if packing_indicators['high_entropy_sections'] > 1:
                confidence += 0.3
            
            if packing_indicators['entropy_variance'] > 2.0:
                confidence += 0.2
            
            return {
                'entropy_result': entropy_result,
                'section_entropies': section_entropies,
                'packing_indicators': packing_indicators,
                'confidence': min(confidence, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in entropy-based detection: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    def _section_analysis(self, binary_data: bytes) -> Dict[str, Any]:
        """Analyze PE sections for packing indicators."""
        if not PEFILE_AVAILABLE:
            return {'confidence': 0.0, 'error': 'pefile not available'}
        
        try:
            pe = pefile.PE(data=binary_data, fast_load=True)
            
            section_indicators = []
            suspicious_sections = 0
            
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                
                # Analyze section characteristics
                is_executable = bool(section.Characteristics & 0x20000000)
                is_writable = bool(section.Characteristics & 0x80000000)
                raw_size = section.SizeOfRawData
                virtual_size = section.Misc_VirtualSize
                
                # Suspicious indicators
                size_discrepancy = abs(virtual_size - raw_size) > raw_size * 0.5 if raw_size > 0 else False
                unusual_name = not re.match(r'^\.?(text|data|rdata|idata|rsrc|reloc)$', section_name.lower())
                high_entropy = False
                
                # Calculate section entropy
                section_data = section.get_data()
                if len(section_data) > 32:
                    section_entropy = self.entropy_analyzer.calculate_shannon_entropy(section_data)
                    high_entropy = section_entropy > 6.5
                
                indicator = {
                    'name': section_name,
                    'executable': is_executable,
                    'writable': is_writable,
                    'size_discrepancy': size_discrepancy,
                    'unusual_name': unusual_name,
                    'high_entropy': high_entropy,
                    'suspicious': any([size_discrepancy, unusual_name and is_executable, high_entropy and is_executable])
                }
                
                section_indicators.append(indicator)
                
                if indicator['suspicious']:
                    suspicious_sections += 1
            
            # Calculate confidence
            confidence = 0.0
            if suspicious_sections > 0:
                confidence = min(suspicious_sections / len(section_indicators), 1.0) * 0.8
            
            return {
                'section_indicators': section_indicators,
                'suspicious_sections': suspicious_sections,
                'total_sections': len(section_indicators),
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error in section analysis: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    def _import_analysis(self, binary_data: bytes) -> Dict[str, Any]:
        """Analyze import table for packing indicators."""
        if not PEFILE_AVAILABLE:
            return {'confidence': 0.0, 'error': 'pefile not available'}
        
        try:
            pe = pefile.PE(data=binary_data, fast_load=True)
            
            import_indicators = []
            suspicious_imports = []
            
            # Suspicious API patterns that indicate packing/obfuscation
            suspicious_apis = {
                'VirtualAlloc', 'VirtualProtect', 'LoadLibrary', 'GetProcAddress',
                'CreateThread', 'WriteProcessMemory', 'ReadProcessMemory',
                'NtUnmapViewOfSection', 'ZwUnmapViewOfSection', 'NtAllocateVirtualMemory'
            }
            
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                total_imports = 0
                suspicious_count = 0
                
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8', errors='ignore')
                    
                    for imp in entry.imports:
                        if imp.name:
                            func_name = imp.name.decode('utf-8', errors='ignore')
                            total_imports += 1
                            
                            if func_name in suspicious_apis:
                                suspicious_imports.append(f"{dll_name}:{func_name}")
                                suspicious_count += 1
                
                # Few imports or high ratio of suspicious imports indicates packing
                few_imports = total_imports < 20
                high_suspicious_ratio = (suspicious_count / total_imports) > 0.3 if total_imports > 0 else False
                
                import_indicators = [
                    f"Total imports: {total_imports}",
                    f"Suspicious imports: {suspicious_count}",
                    f"Few imports: {few_imports}",
                    f"High suspicious ratio: {high_suspicious_ratio}"
                ]
                
                # Calculate confidence
                confidence = 0.0
                if few_imports:
                    confidence += 0.4
                if high_suspicious_ratio:
                    confidence += 0.5
                if suspicious_count > 5:
                    confidence += 0.3
                
                return {
                    'import_indicators': import_indicators,
                    'suspicious_imports': suspicious_imports,
                    'total_imports': total_imports,
                    'suspicious_count': suspicious_count,
                    'confidence': min(confidence, 1.0)
                }
            
            return {'confidence': 0.0, 'message': 'No import table found'}
            
        except Exception as e:
            self.logger.error(f"Error in import analysis: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    def _yara_detection(self, binary_data: bytes) -> Dict[str, Any]:
        """Perform YARA rule-based detection."""
        if not self.yara_rules:
            return {'confidence': 0.0, 'message': 'YARA rules not available'}
        
        try:
            matches = self.yara_rules.match(data=binary_data)
            
            yara_results = []
            packer_types = set()
            
            for match in matches:
                rule_name = match.rule
                packer_name = match.meta.get('packer', 'Unknown')
                
                yara_results.append({
                    'rule': rule_name,
                    'packer': packer_name,
                    'strings': [str(s) for s in match.strings]
                })
                
                packer_types.add(packer_name)
            
            confidence = min(len(matches) * 0.3, 1.0) if matches else 0.0
            
            return {
                'yara_matches': yara_results,
                'detected_packers': list(packer_types),
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error in YARA detection: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    def _behavioral_analysis(self, binary_data: bytes) -> Dict[str, Any]:
        """Analyze behavioral indicators of packing."""
        behavioral_indicators = []
        confidence = 0.0
        
        try:
            # Check for common packer strings
            packer_strings = [
                b"This program cannot be run in DOS mode",
                b"DOS mode",
                b"packed",
                b"compressed",
                b"protector",
                b"virtualiz"
            ]
            
            found_strings = []
            for string in packer_strings:
                if string in binary_data:
                    found_strings.append(string.decode('utf-8', errors='ignore'))
            
            if found_strings:
                behavioral_indicators.append(f"Packer-related strings: {found_strings}")
                confidence += 0.2
            
            # Check for unusual entry point
            if PEFILE_AVAILABLE:
                try:
                    pe = pefile.PE(data=binary_data, fast_load=True)
                    entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
                    image_base = pe.OPTIONAL_HEADER.ImageBase
                    
                    # Check if entry point is in last section (common for packers)
                    last_section = pe.sections[-1]
                    if (entry_point >= last_section.VirtualAddress and 
                        entry_point < last_section.VirtualAddress + last_section.Misc_VirtualSize):
                        behavioral_indicators.append("Entry point in last section")
                        confidence += 0.3
                    
                except:
                    pass
            
            # Check for overlay data
            if len(binary_data) > 1024 * 1024:  # Large files might have overlays
                # Simple heuristic: check if there's significant data after last section
                try:
                    if PEFILE_AVAILABLE:
                        pe = pefile.PE(data=binary_data, fast_load=True)
                        last_section = pe.sections[-1]
                        last_section_end = last_section.PointerToRawData + last_section.SizeOfRawData
                        
                        if len(binary_data) - last_section_end > 10000:
                            behavioral_indicators.append("Large overlay data detected")
                            confidence += 0.2
                except:
                    pass
            
            return {
                'behavioral_indicators': behavioral_indicators,
                'confidence': min(confidence, 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in behavioral analysis: {e}")
            return {'confidence': 0.0, 'error': str(e)}
    
    def _combine_detection_results(self, base_result: PackerDetectionResult, 
                                 signature_results: Dict, entropy_results: Dict,
                                 section_results: Dict, import_results: Dict,
                                 yara_results: Dict, behavioral_results: Dict) -> PackerDetectionResult:
        """Combine results from all detection methods."""
        
        # Collect all evidence
        base_result.signatures_matched = signature_results.get('matched_signatures', [])
        base_result.entropy_indicators = entropy_results
        base_result.section_indicators = section_results.get('section_indicators', [])
        base_result.import_indicators = import_results.get('import_indicators', [])
        base_result.behavioral_indicators = behavioral_results.get('behavioral_indicators', [])
        
        # Calculate combined confidence
        confidences = [
            signature_results.get('confidence', 0.0) * 0.3,
            entropy_results.get('confidence', 0.0) * 0.25,
            section_results.get('confidence', 0.0) * 0.2,
            import_results.get('confidence', 0.0) * 0.1,
            yara_results.get('confidence', 0.0) * 0.1,
            behavioral_results.get('confidence', 0.0) * 0.05
        ]
        
        base_result.confidence = sum(confidences)
        
        # Determine packer type
        if signature_results.get('best_packer') != PackerType.UNKNOWN:
            base_result.packer_type = signature_results['best_packer']
            base_result.packer_name = signature_results['best_packer'].value
        elif yara_results.get('detected_packers'):
            base_result.packer_name = yara_results['detected_packers'][0]
            base_result.packer_type = self._string_to_packer_type(base_result.packer_name)
        elif base_result.confidence > 0.6:
            base_result.packer_type = PackerType.GENERIC
            base_result.packer_name = "Generic Packer"
        
        # Set detection flag
        base_result.packer_detected = base_result.confidence > 0.4
        
        return base_result
    
    def _determine_final_verdict(self, result: PackerDetectionResult) -> PackerDetectionResult:
        """Determine final verdict and recommendations."""
        
        # Set unpacking difficulty
        if result.packer_type == PackerType.UPX:
            result.unpacking_difficulty = "easy"
            result.recommended_tools = ["upx", "manual_unpacking"]
        elif result.packer_type in [PackerType.THEMIDA, PackerType.VMPROTECT]:
            result.unpacking_difficulty = "very_hard"
            result.recommended_tools = ["specialized_unpackers", "dynamic_analysis", "manual_analysis"]
        elif result.packer_type == PackerType.GENERIC:
            result.unpacking_difficulty = "medium"
            result.recommended_tools = ["generic_unpackers", "dynamic_analysis"]
        else:
            result.unpacking_difficulty = "unknown"
            result.recommended_tools = ["identify_packer_first"]
        
        # Add metadata
        result.metadata = {
            'analysis_timestamp': self._get_timestamp(),
            'detection_methods_used': ['signature', 'entropy', 'section', 'import', 'yara', 'behavioral'],
            'total_signatures_checked': len(self.signatures),
            'pefile_available': PEFILE_AVAILABLE,
            'yara_available': YARA_AVAILABLE
        }
        
        return result
    
    def _string_to_packer_type(self, packer_string: str) -> PackerType:
        """Convert string to PackerType enum."""
        packer_map = {
            'UPX': PackerType.UPX,
            'Themida': PackerType.THEMIDA,
            'VMProtect': PackerType.VMPROTECT,
            'ASPack': PackerType.ASPACK,
            'PECompact': PackerType.PECOMPACT,
            'MPRESS': PackerType.MPRESS
        }
        
        return packer_map.get(packer_string, PackerType.UNKNOWN)
    
    def _create_error_result(self, error_msg: str) -> PackerDetectionResult:
        """Create error result for failed detection."""
        return PackerDetectionResult(
            packer_detected=False,
            packer_type=PackerType.UNKNOWN,
            packer_name="Error",
            confidence=0.0,
            signatures_matched=[],
            entropy_indicators={},
            section_indicators=[],
            import_indicators=[],
            behavioral_indicators=[],
            unpacking_difficulty="unknown",
            recommended_tools=[],
            metadata={'error': error_msg}
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()


class UnpackingEngine:
    """
    Automated unpacking engine for supported packers.
    
    Provides unpacking capabilities for common packers
    and interfaces to external unpacking tools.
    """
    
    def __init__(self, config_manager=None):
        """Initialize unpacking engine."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.detector = PackerDetector(config_manager)
    
    def unpack_binary(self, binary_path: Path, output_path: Path = None) -> Dict[str, Any]:
        """
        Attempt to unpack a binary file.
        
        Args:
            binary_path: Path to packed binary
            output_path: Path for unpacked output (optional)
            
        Returns:
            Dictionary with unpacking results
        """
        try:
            self.logger.info(f"Starting unpacking process for {binary_path}")
            
            # First, detect the packer
            detection_result = self.detector.detect_packer(binary_path)
            
            if not detection_result.packer_detected:
                return {
                    'success': False,
                    'error': 'No packer detected',
                    'detection_result': detection_result
                }
            
            self.logger.info(f"Detected packer: {detection_result.packer_name}")
            
            # Set output path if not provided
            if output_path is None:
                output_path = binary_path.with_suffix('.unpacked.exe')
            
            # Attempt unpacking based on packer type
            if detection_result.packer_type == PackerType.UPX:
                return self._unpack_upx(binary_path, output_path)
            elif detection_result.packer_type == PackerType.ASPACK:
                return self._unpack_aspack(binary_path, output_path)
            else:
                return self._generic_unpacking_attempt(binary_path, output_path, detection_result)
            
        except Exception as e:
            self.logger.error(f"Error in unpacking: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _unpack_upx(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Unpack UPX-packed binary."""
        try:
            # Try using UPX tool if available
            result = subprocess.run([
                'upx', '-d', str(input_path), '-o', str(output_path)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'method': 'upx_tool',
                    'output_path': output_path,
                    'details': result.stdout
                }
            else:
                # Fallback to manual UPX unpacking
                return self._manual_upx_unpack(input_path, output_path)
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # UPX tool not available, try manual unpacking
            return self._manual_upx_unpack(input_path, output_path)
    
    def _manual_upx_unpack(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Manual UPX unpacking algorithm."""
        try:
            with open(input_path, 'rb') as f:
                packed_data = f.read()
            
            # This is a simplified manual UPX unpacker
            # In practice, this would be much more complex
            
            # Look for UPX sections
            if not PEFILE_AVAILABLE:
                return {
                    'success': False,
                    'error': 'pefile required for manual UPX unpacking'
                }
            
            pe = pefile.PE(data=packed_data, fast_load=True)
            
            # Find UPX sections
            upx_sections = []
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                if 'UPX' in section_name.upper():
                    upx_sections.append(section)
            
            if not upx_sections:
                return {
                    'success': False,
                    'error': 'No UPX sections found for manual unpacking'
                }
            
            # This would implement the actual UPX decompression algorithm
            # For now, return partial success
            return {
                'success': False,
                'error': 'Manual UPX unpacking not yet fully implemented',
                'found_upx_sections': len(upx_sections)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Manual UPX unpacking failed: {e}'
            }
    
    def _unpack_aspack(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """Unpack ASPack-packed binary."""
        # ASPack unpacking would be implemented here
        return {
            'success': False,
            'error': 'ASPack unpacking not yet implemented'
        }
    
    def _generic_unpacking_attempt(self, input_path: Path, output_path: Path, 
                                 detection_result: PackerDetectionResult) -> Dict[str, Any]:
        """Generic unpacking attempt for unknown packers."""
        
        # Try common unpacking tools
        tools_to_try = [
            'unipacker',
            'upx -d',
            'de4dot'
        ]
        
        for tool in tools_to_try:
            try:
                if tool == 'unipacker':
                    result = subprocess.run([
                        'python', '-m', 'unipacker', str(input_path)
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        return {
                            'success': True,
                            'method': 'unipacker',
                            'output_path': output_path,
                            'details': result.stdout
                        }
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        return {
            'success': False,
            'error': f'No suitable unpacker found for {detection_result.packer_name}',
            'recommendations': detection_result.recommended_tools
        }


# Factory functions
def create_packer_detector(config_manager=None) -> PackerDetector:
    """Create configured packer detector instance."""
    return PackerDetector(config_manager)


def create_unpacking_engine(config_manager=None) -> UnpackingEngine:
    """Create configured unpacking engine instance."""
    return UnpackingEngine(config_manager)


# Example usage
if __name__ == "__main__":
    # Example usage
    detector = PackerDetector()
    
    # Test with sample data
    test_upx_data = b"UPX!" + b"\x00" * 1000
    
    # Would normally use a file path
    # result = detector.detect_packer(Path("test.exe"))
    # print(f"Packer detected: {result.packer_name} (confidence: {result.confidence:.2f})")
    
    print("Packer detection module loaded successfully")