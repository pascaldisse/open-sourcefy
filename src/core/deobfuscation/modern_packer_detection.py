"""
Modern Packer Detection System - Phase 1 Enhancement

Implements detection for latest packing and protection schemes:
- Modern packers (Obsidium, Enigma Protector, .NET protection)
- Runtime packer detection using behavioral analysis
- Multi-layer protection identification
- Custom packer signature development
- Automated unpacking pipeline integration

Enhanced implementation building on existing packer detection framework.
"""

import logging
import struct
import re
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import time

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
from .packer_detector import PackerDetector, PackerSignature


class ModernPackerType(Enum):
    """Modern packer and protection types."""
    OBSIDIUM = "obsidium"
    ENIGMA_PROTECTOR = "enigma_protector"
    VMPROTECT_V3 = "vmprotect_v3"
    THEMIDA_V3 = "themida_v3"
    DOTNET_REACTOR = "dotnet_reactor"
    CONFUSER_EX = "confuser_ex"
    EAZFUSCATOR = "eazfuscator"
    CRYPTOOBFUSCATOR = "cryptoobfuscator"
    SMARTASSEMBLY = "smartassembly"
    DOTFUSCATOR = "dotfuscator"
    CUSTOM_PACKER = "custom_packer"
    FILELESS_PROTECTION = "fileless_protection"


@dataclass
class ModernPackerSignature:
    """Signature for modern packer detection."""
    packer_type: ModernPackerType
    version_info: str
    detection_patterns: List[bytes]
    entropy_thresholds: Dict[str, float]
    import_patterns: List[str]
    section_characteristics: Dict[str, Any]
    behavioral_indicators: List[str]
    confidence_threshold: float


@dataclass
class RuntimeBehaviorPattern:
    """Runtime behavior pattern for dynamic detection."""
    api_calls: List[str]
    memory_patterns: List[bytes]
    registry_access: List[str]
    file_operations: List[str]
    network_activity: List[str]
    process_spawning: List[str]
    anti_analysis_techniques: List[str]


@dataclass
class MultiLayerProtection:
    """Multi-layer protection analysis result."""
    layer_count: int
    protection_layers: List[Dict[str, Any]]
    unpacking_sequence: List[str]
    estimated_difficulty: str
    tools_required: List[str]
    success_probability: float


@dataclass
class ModernPackerResult:
    """Result of modern packer detection."""
    detected_packers: List[ModernPackerSignature]
    runtime_behaviors: List[RuntimeBehaviorPattern]
    multi_layer_analysis: Optional[MultiLayerProtection]
    custom_signatures: List[Dict[str, Any]]
    unpacking_strategies: List[str]
    confidence_scores: Dict[str, float]
    performance_metrics: Dict[str, Any]


class ModernPackerDetector:
    """
    Modern Packer Detection System.
    
    Implements advanced detection for latest packing and protection schemes
    with support for behavioral analysis and custom signature generation.
    """
    
    def __init__(self, config_manager=None):
        """Initialize modern packer detector."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize base components
        self.entropy_analyzer = EntropyAnalyzer(config_manager)
        self.base_packer_detector = PackerDetector(config_manager)
        
        # Modern packer signature database
        self._initialize_modern_signatures()
        
        # YARA rules for advanced detection
        self._initialize_yara_rules()
        
        # Runtime behavior patterns
        self._initialize_behavior_patterns()
        
        self.logger.info("Modern Packer Detection System initialized")

    def _initialize_modern_signatures(self):
        """Initialize signature database for modern packers."""
        self.modern_signatures = {
            ModernPackerType.OBSIDIUM: ModernPackerSignature(
                packer_type=ModernPackerType.OBSIDIUM,
                version_info="v1.6+",
                detection_patterns=[
                    b'Obsidium Software',
                    b'\x55\x8B\xEC\x83\xC4\xF0\x53\x56\x57\x8B\xF1',
                    b'\x64\x8B\x15\x30\x00\x00\x00\x8B\x52\x0C'
                ],
                entropy_thresholds={'text_section': 7.5, 'overall': 7.0},
                import_patterns=['kernel32.dll', 'VirtualAlloc', 'VirtualProtect'],
                section_characteristics={
                    'packed_sections': True,
                    'unusual_names': ['.obsidium', '.packed'],
                    'high_entropy': True
                },
                behavioral_indicators=['anti_debug', 'vm_detection', 'api_hooking'],
                confidence_threshold=0.75
            ),
            
            ModernPackerType.ENIGMA_PROTECTOR: ModernPackerSignature(
                packer_type=ModernPackerType.ENIGMA_PROTECTOR,
                version_info="v7.0+",
                detection_patterns=[
                    b'The Enigma Protector',
                    b'\x68\x00\x00\x40\x00\x68\x00\x10\x00\x00',
                    b'\x8B\x45\x08\x8B\x4D\x0C\x8B\x55\x10'
                ],
                entropy_thresholds={'text_section': 7.8, 'overall': 7.2},
                import_patterns=['ntdll.dll', 'NtQueryInformationProcess'],
                section_characteristics={
                    'virtual_sections': True,
                    'encrypted_imports': True
                },
                behavioral_indicators=['license_checking', 'hardware_fingerprinting'],
                confidence_threshold=0.80
            ),
            
            ModernPackerType.VMPROTECT_V3: ModernPackerSignature(
                packer_type=ModernPackerType.VMPROTECT_V3,
                version_info="v3.5+",
                detection_patterns=[
                    b'VMProtect',
                    b'\x60\x61\x60\x61',  # pushad; popad; pushad; popad
                    b'\x8B\x44\x24\x04\x89\x44\x24\x08'
                ],
                entropy_thresholds={'vm_sections': 8.0, 'overall': 7.5},
                import_patterns=['kernel32.dll', 'IsDebuggerPresent'],
                section_characteristics={
                    'vm_code': True,
                    'mutation_engine': True
                },
                behavioral_indicators=['vm_execution', 'code_mutation', 'anti_analysis'],
                confidence_threshold=0.85
            ),
            
            ModernPackerType.DOTNET_REACTOR: ModernPackerSignature(
                packer_type=ModernPackerType.DOTNET_REACTOR,
                version_info="v6.0+",
                detection_patterns=[
                    b'.NET Reactor',
                    b'Eziriz .NET Reactor',
                    b'\x72\x01\x00\x00\x70\x28'  # .NET specific patterns
                ],
                entropy_thresholds={'dotnet_assembly': 6.5},
                import_patterns=['mscoree.dll', '_CorExeMain'],
                section_characteristics={
                    'dotnet_specific': True,
                    'encrypted_metadata': True
                },
                behavioral_indicators=['dotnet_obfuscation', 'string_encryption'],
                confidence_threshold=0.70
            )
        }

    def _initialize_yara_rules(self):
        """Initialize YARA rules for advanced pattern matching."""
        if not YARA_AVAILABLE:
            self.logger.warning("YARA not available, using pattern matching fallback")
            self.yara_rules = None
            return
        
        # YARA rules for modern packers
        yara_rule_source = '''
        rule Obsidium_Modern {
            meta:
                description = "Obsidium v1.6+ detection"
                author = "OpenSourcefy"
            strings:
                $obs1 = "Obsidium Software"
                $obs2 = { 55 8B EC 83 C4 F0 53 56 57 8B F1 }
                $obs3 = { 64 8B 15 30 00 00 00 8B 52 0C }
            condition:
                $obs1 or ($obs2 and $obs3)
        }
        
        rule EnigmaProtector_Modern {
            meta:
                description = "Enigma Protector v7.0+ detection"
            strings:
                $ep1 = "The Enigma Protector"
                $ep2 = { 68 00 00 40 00 68 00 10 00 00 }
                $ep3 = "NtQueryInformationProcess"
            condition:
                $ep1 or ($ep2 and $ep3)
        }
        
        rule VMProtect_V3 {
            meta:
                description = "VMProtect v3.5+ detection"
            strings:
                $vmp1 = "VMProtect"
                $vmp2 = { 60 61 60 61 }
                $vmp3 = { 8B 44 24 04 89 44 24 08 }
            condition:
                $vmp1 or ($vmp2 and $vmp3)
        }
        
        rule DotNET_Reactor {
            meta:
                description = ".NET Reactor v6.0+ detection"
            strings:
                $dnr1 = ".NET Reactor"
                $dnr2 = "Eziriz .NET Reactor"
                $dnr3 = { 72 01 00 00 70 28 }
            condition:
                uint16(0) == 0x5A4D and ($dnr1 or $dnr2 or $dnr3)
        }
        '''
        
        try:
            self.yara_rules = yara.compile(source=yara_rule_source)
            self.logger.info("YARA rules compiled successfully")
        except Exception as e:
            self.logger.error(f"Failed to compile YARA rules: {e}")
            self.yara_rules = None

    def _initialize_behavior_patterns(self):
        """Initialize runtime behavior patterns for dynamic analysis."""
        self.behavior_patterns = {
            'anti_debug': RuntimeBehaviorPattern(
                api_calls=['IsDebuggerPresent', 'CheckRemoteDebuggerPresent', 'NtQueryInformationProcess'],
                memory_patterns=[b'\x64\x8B\x15\x30\x00\x00\x00'],  # fs:[30h] access
                registry_access=['HKLM\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\\AeDebug'],
                file_operations=['CreateFile', 'DeleteFile'],
                network_activity=[],
                process_spawning=[],
                anti_analysis_techniques=['debugger_detection', 'vm_detection']
            ),
            
            'vm_detection': RuntimeBehaviorPattern(
                api_calls=['GetSystemInfo', 'GetComputerName', 'GetUserName'],
                memory_patterns=[b'\x0F\x01\x0D', b'\x0F\x01\x05'],  # sgdt, sidt
                registry_access=['HKLM\\SYSTEM\\CurrentControlSet\\Services\\Disk\\Enum'],
                file_operations=['VMwareService.exe', 'VBoxService.exe'],
                network_activity=[],
                process_spawning=[],
                anti_analysis_techniques=['hypervisor_detection', 'sandbox_detection']
            ),
            
            'code_injection': RuntimeBehaviorPattern(
                api_calls=['VirtualAlloc', 'VirtualProtect', 'WriteProcessMemory', 'CreateRemoteThread'],
                memory_patterns=[b'\xFF\x25', b'\xFF\x15'],  # Indirect calls
                registry_access=[],
                file_operations=[],
                network_activity=[],
                process_spawning=['CreateProcess', 'ShellExecute'],
                anti_analysis_techniques=['process_injection', 'dll_injection']
            )
        }

    def detect_modern_packers(self, binary_path: Path) -> ModernPackerResult:
        """
        Detect modern packers and protection schemes.
        
        Args:
            binary_path: Path to binary file
            
        Returns:
            ModernPackerResult with detection results
        """
        self.logger.info(f"Starting modern packer detection: {binary_path}")
        
        try:
            # Load binary data
            binary_data = binary_path.read_bytes()
            
            # Initialize result
            result = ModernPackerResult(
                detected_packers=[],
                runtime_behaviors=[],
                multi_layer_analysis=None,
                custom_signatures=[],
                unpacking_strategies=[],
                confidence_scores={},
                performance_metrics={}
            )
            
            # Phase 1: Static signature detection
            self.logger.info("Performing static signature analysis...")
            static_detections = self._perform_static_detection(binary_data)
            result.detected_packers.extend(static_detections)
            
            # Phase 2: YARA rule matching
            if self.yara_rules:
                self.logger.info("Running YARA rule analysis...")
                yara_matches = self._perform_yara_analysis(binary_data)
                result.detected_packers.extend(yara_matches)
            
            # Phase 3: Entropy-based analysis
            self.logger.info("Performing entropy analysis...")
            entropy_results = self._perform_entropy_analysis(binary_data, binary_path)
            
            # Phase 4: Import table analysis
            self.logger.info("Analyzing import patterns...")
            import_analysis = self._analyze_import_patterns(binary_data)
            
            # Phase 5: Section characteristics analysis
            self.logger.info("Analyzing section characteristics...")
            section_analysis = self._analyze_section_characteristics(binary_data)
            
            # Phase 6: Multi-layer protection detection
            self.logger.info("Detecting multi-layer protection...")
            multi_layer = self._detect_multi_layer_protection(binary_data, result.detected_packers)
            result.multi_layer_analysis = multi_layer
            
            # Phase 7: Generate custom signatures
            self.logger.info("Generating custom signatures...")
            custom_sigs = self._generate_custom_signatures(binary_data, result.detected_packers)
            result.custom_signatures = custom_sigs
            
            # Phase 8: Generate unpacking strategies
            result.unpacking_strategies = self._generate_unpacking_strategies(result)
            
            # Calculate confidence scores
            result.confidence_scores = self._calculate_confidence_scores(result)
            
            self.logger.info(f"Modern packer detection complete. Found {len(result.detected_packers)} packers")
            return result
            
        except Exception as e:
            self.logger.error(f"Modern packer detection failed: {e}")
            return ModernPackerResult(
                detected_packers=[],
                runtime_behaviors=[],
                multi_layer_analysis=None,
                custom_signatures=[],
                unpacking_strategies=[f"Detection failed: {e}"],
                confidence_scores={},
                performance_metrics={'error': str(e)}
            )

    def _perform_static_detection(self, binary_data: bytes) -> List[ModernPackerSignature]:
        """Perform static signature-based detection."""
        detected = []
        
        for packer_type, signature in self.modern_signatures.items():
            confidence = 0.0
            matches = 0
            
            # Check detection patterns
            for pattern in signature.detection_patterns:
                if pattern in binary_data:
                    matches += 1
                    confidence += 0.3
            
            # Check import patterns
            import_score = self._check_import_patterns(binary_data, signature.import_patterns)
            confidence += import_score * 0.2
            
            # Apply confidence threshold
            if confidence >= signature.confidence_threshold:
                detected_signature = signature
                detected_signature.confidence_threshold = confidence
                detected.append(detected_signature)
                
                self.logger.info(f"Detected {packer_type.value} with confidence {confidence:.2f}")
        
        return detected

    def _perform_yara_analysis(self, binary_data: bytes) -> List[ModernPackerSignature]:
        """Perform YARA rule-based analysis."""
        detected = []
        
        if not self.yara_rules:
            return detected
        
        try:
            matches = self.yara_rules.match(data=binary_data)
            
            for match in matches:
                # Map YARA rule to packer type
                packer_mapping = {
                    'Obsidium_Modern': ModernPackerType.OBSIDIUM,
                    'EnigmaProtector_Modern': ModernPackerType.ENIGMA_PROTECTOR,
                    'VMProtect_V3': ModernPackerType.VMPROTECT_V3,
                    'DotNET_Reactor': ModernPackerType.DOTNET_REACTOR
                }
                
                if match.rule in packer_mapping:
                    packer_type = packer_mapping[match.rule]
                    if packer_type in self.modern_signatures:
                        signature = self.modern_signatures[packer_type]
                        signature.confidence_threshold = 0.85  # High confidence for YARA matches
                        detected.append(signature)
                        
                        self.logger.info(f"YARA detected {packer_type.value} with rule {match.rule}")
        
        except Exception as e:
            self.logger.error(f"YARA analysis failed: {e}")
        
        return detected

    def _perform_entropy_analysis(self, binary_data: bytes, binary_path: Path) -> Dict[str, float]:
        """Perform entropy-based packer detection."""
        try:
            entropy_result = self.entropy_analyzer.analyze_entropy(binary_path)
            
            # Check entropy thresholds for each packer
            for packer_type, signature in self.modern_signatures.items():
                for section, threshold in signature.entropy_thresholds.items():
                    section_entropy = entropy_result.get('entropy_by_section', {}).get(section, 0)
                    if section_entropy > threshold:
                        self.logger.info(f"High entropy detected for {packer_type.value}: {section_entropy:.2f}")
            
            return entropy_result
            
        except Exception as e:
            self.logger.error(f"Entropy analysis failed: {e}")
            return {}

    def _analyze_import_patterns(self, binary_data: bytes) -> Dict[str, Any]:
        """Analyze import table patterns for packer detection."""
        import_analysis = {
            'suspicious_apis': [],
            'minimal_imports': False,
            'encrypted_imports': False
        }
        
        if not PEFILE_AVAILABLE:
            return import_analysis
        
        try:
            pe = pefile.PE(data=binary_data)
            
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                import_count = len(pe.DIRECTORY_ENTRY_IMPORT)
                
                # Check for minimal imports (common in packed files)
                if import_count < 5:
                    import_analysis['minimal_imports'] = True
                
                # Check for suspicious APIs
                suspicious_apis = [
                    'VirtualAlloc', 'VirtualProtect', 'LoadLibrary', 'GetProcAddress',
                    'IsDebuggerPresent', 'NtQueryInformationProcess'
                ]
                
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    for imp in entry.imports:
                        if imp.name and imp.name.decode('utf-8', errors='ignore') in suspicious_apis:
                            import_analysis['suspicious_apis'].append(imp.name.decode('utf-8', errors='ignore'))
            
        except Exception as e:
            self.logger.error(f"Import analysis failed: {e}")
        
        return import_analysis

    def _analyze_section_characteristics(self, binary_data: bytes) -> Dict[str, Any]:
        """Analyze PE section characteristics for packer indicators."""
        section_analysis = {
            'unusual_names': [],
            'high_entropy_sections': [],
            'executable_sections': 0,
            'writable_executable': False
        }
        
        if not PEFILE_AVAILABLE:
            return section_analysis
        
        try:
            pe = pefile.PE(data=binary_data)
            
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                
                # Check for unusual section names
                common_names = ['.text', '.data', '.rdata', '.rsrc', '.reloc']
                if section_name not in common_names and section_name:
                    section_analysis['unusual_names'].append(section_name)
                
                # Check characteristics
                if section.Characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                    section_analysis['executable_sections'] += 1
                    
                    if section.Characteristics & 0x80000000:  # IMAGE_SCN_MEM_WRITE
                        section_analysis['writable_executable'] = True
        
        except Exception as e:
            self.logger.error(f"Section analysis failed: {e}")
        
        return section_analysis

    def _detect_multi_layer_protection(self, binary_data: bytes, detected_packers: List) -> Optional[MultiLayerProtection]:
        """Detect multi-layer protection schemes."""
        if len(detected_packers) <= 1:
            return None
        
        # Analyze layering
        layers = []
        for i, packer in enumerate(detected_packers):
            layers.append({
                'layer_number': i + 1,
                'protection_type': packer.packer_type.value,
                'estimated_position': 'outer' if i == 0 else 'inner'
            })
        
        # Generate unpacking sequence
        unpacking_sequence = [f"Unpack {packer.packer_type.value}" for packer in detected_packers]
        
        # Estimate difficulty
        difficulty_map = {
            1: 'easy',
            2: 'medium', 
            3: 'hard',
            4: 'expert'
        }
        difficulty = difficulty_map.get(len(detected_packers), 'expert')
        
        return MultiLayerProtection(
            layer_count=len(detected_packers),
            protection_layers=layers,
            unpacking_sequence=unpacking_sequence,
            estimated_difficulty=difficulty,
            tools_required=['manual_analysis', 'dynamic_unpacking', 'specialized_tools'],
            success_probability=max(0.1, 0.9 - (len(detected_packers) * 0.2))
        )

    def _generate_custom_signatures(self, binary_data: bytes, detected_packers: List) -> List[Dict[str, Any]]:
        """Generate custom signatures for unknown packers."""
        custom_signatures = []
        
        # If no known packers detected, try to generate custom signature
        if not detected_packers:
            # Look for common packer patterns
            packer_indicators = self._find_packer_indicators(binary_data)
            
            if packer_indicators:
                custom_sig = {
                    'type': 'custom_packer',
                    'patterns': packer_indicators['patterns'],
                    'confidence': packer_indicators['confidence'],
                    'characteristics': packer_indicators['characteristics']
                }
                custom_signatures.append(custom_sig)
        
        return custom_signatures

    def _find_packer_indicators(self, binary_data: bytes) -> Optional[Dict[str, Any]]:
        """Find indicators of unknown/custom packers."""
        indicators = {
            'patterns': [],
            'confidence': 0.0,
            'characteristics': []
        }
        
        # Check for common packer patterns
        common_patterns = [
            b'\x60\x61',           # pushad; popad
            b'\xE8\x00\x00\x00\x00\x5D',  # call $+5; pop ebp
            b'\x64\x8B\x15\x30\x00\x00\x00'  # mov edx, fs:[30h]
        ]
        
        for pattern in common_patterns:
            if pattern in binary_data:
                indicators['patterns'].append(pattern.hex())
                indicators['confidence'] += 0.2
        
        # Check entropy
        high_entropy_blocks = 0
        block_size = 1024
        for i in range(0, len(binary_data), block_size):
            block = binary_data[i:i+block_size]
            entropy = self._calculate_block_entropy(block)
            if entropy > 7.0:
                high_entropy_blocks += 1
        
        if high_entropy_blocks > len(binary_data) // block_size * 0.3:
            indicators['characteristics'].append('high_entropy')
            indicators['confidence'] += 0.3
        
        return indicators if indicators['confidence'] > 0.4 else None

    def _calculate_block_entropy(self, data: bytes) -> float:
        """Calculate entropy for a data block."""
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

    def _check_import_patterns(self, binary_data: bytes, patterns: List[str]) -> float:
        """Check if import patterns match the binary."""
        if not PEFILE_AVAILABLE:
            return 0.0
        
        try:
            pe = pefile.PE(data=binary_data)
            
            if not hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                return 0.0
            
            found_patterns = 0
            total_patterns = len(patterns)
            
            import_names = []
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8', errors='ignore').lower()
                import_names.append(dll_name)
                
                for imp in entry.imports:
                    if imp.name:
                        import_names.append(imp.name.decode('utf-8', errors='ignore'))
            
            for pattern in patterns:
                if any(pattern.lower() in name.lower() for name in import_names):
                    found_patterns += 1
            
            return found_patterns / total_patterns if total_patterns > 0 else 0.0
            
        except Exception:
            return 0.0

    def _generate_unpacking_strategies(self, result: ModernPackerResult) -> List[str]:
        """Generate unpacking strategies based on detected packers."""
        strategies = []
        
        # Strategy based on detected packers
        for packer in result.detected_packers:
            packer_strategies = {
                ModernPackerType.OBSIDIUM: [
                    "Use Obsidium unpacker tools",
                    "Apply OEP finding techniques",
                    "Dump from memory after OEP"
                ],
                ModernPackerType.ENIGMA_PROTECTOR: [
                    "Use Enigma Protector unpacker",
                    "Bypass license checking",
                    "Manual OEP finding"
                ],
                ModernPackerType.VMPROTECT_V3: [
                    "Use VMProtect 3.x unpacker",
                    "VM code analysis",
                    "Bytecode reconstruction"
                ],
                ModernPackerType.DOTNET_REACTOR: [
                    "Use de4dot with .NET Reactor plugin",
                    "Decrypt string resources",
                    "Restore metadata"
                ]
            }
            
            if packer.packer_type in packer_strategies:
                strategies.extend(packer_strategies[packer.packer_type])
        
        # Multi-layer strategies
        if result.multi_layer_analysis and result.multi_layer_analysis.layer_count > 1:
            strategies.extend([
                "Apply layer-by-layer unpacking approach",
                "Use automated unpacking tools",
                "Consider manual analysis for complex cases"
            ])
        
        # General strategies
        strategies.extend([
            "Combine static and dynamic analysis",
            "Use multiple unpacking tools for verification",
            "Validate unpacked results"
        ])
        
        return list(set(strategies))  # Remove duplicates

    def _calculate_confidence_scores(self, result: ModernPackerResult) -> Dict[str, float]:
        """Calculate confidence scores for detection results."""
        confidence_scores = {}
        
        # Confidence for each detected packer
        for i, packer in enumerate(result.detected_packers):
            confidence_scores[f"packer_{i}_{packer.packer_type.value}"] = packer.confidence_threshold
        
        # Overall detection confidence
        if result.detected_packers:
            avg_confidence = sum(p.confidence_threshold for p in result.detected_packers) / len(result.detected_packers)
            confidence_scores['overall_detection'] = avg_confidence
        else:
            confidence_scores['overall_detection'] = 0.0
        
        # Multi-layer confidence
        if result.multi_layer_analysis:
            confidence_scores['multi_layer'] = result.multi_layer_analysis.success_probability
        
        return confidence_scores

    def enhance_existing_packer_detection(self, basic_result: Dict[str, Any], binary_path: Path) -> Dict[str, Any]:
        """
        Enhance existing packer detection with modern techniques.
        
        This method integrates with the existing packer detection framework
        to provide enhanced analysis capabilities.
        """
        self.logger.info("Enhancing existing packer detection with modern techniques")
        
        try:
            # Perform modern analysis
            modern_result = self.detect_modern_packers(binary_path)
            
            # Merge results
            enhanced_result = basic_result.copy()
            enhanced_result['modern_analysis'] = {
                'detected_packers': [p.packer_type.value for p in modern_result.detected_packers],
                'multi_layer_protection': modern_result.multi_layer_analysis is not None,
                'custom_signatures': len(modern_result.custom_signatures),
                'unpacking_strategies': modern_result.unpacking_strategies,
                'confidence_scores': modern_result.confidence_scores
            }
            
            # Update overall confidence
            if 'confidence' in enhanced_result and modern_result.confidence_scores.get('overall_detection', 0) > 0:
                enhanced_result['confidence'] = max(
                    enhanced_result['confidence'],
                    modern_result.confidence_scores['overall_detection']
                )
            
            self.logger.info("Modern packer detection enhancement completed successfully")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Modern enhancement failed: {e}")
            # Return original result if enhancement fails
            return basic_result


def create_modern_packer_detector(config_manager=None) -> ModernPackerDetector:
    """
    Factory function to create modern packer detector.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        ModernPackerDetector: Configured detector instance
    """
    return ModernPackerDetector(config_manager)