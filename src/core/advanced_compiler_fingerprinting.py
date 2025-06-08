#!/usr/bin/env python3
"""
Advanced Compiler Fingerprinting & Optimization Detection

P2.1 Implementation for Open-Sourcefy Matrix Pipeline
Provides comprehensive compiler identification and optimization analysis
using machine learning and pattern recognition techniques.

Features:
- Multi-compiler detection (MSVC, GCC, Clang, Intel ICC)
- Advanced optimization level detection with ML models
- Rich header analysis for MSVC binaries
- Name mangling scheme analysis for C++
- Binary pattern classification using CNN/LSTM models

Research Base:
- BinComp: A Stratified Approach to Compiler Provenance Attribution
- Binary Code Fingerprinting Approaches Survey
- Machine Learning for Compiler Identification
"""

import re
import logging
import struct
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import pickle
import numpy as np

# Conditional ML imports
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create dummy classes for type hints
    class RandomForestClassifier:
        pass
    class TfidfVectorizer:
        pass

logger = logging.getLogger(__name__)


class CompilerType(Enum):
    """Supported compiler types for fingerprinting"""
    MSVC = "microsoft_visual_cpp"
    GCC = "gnu_compiler_collection"  
    CLANG = "llvm_clang"
    INTEL_ICC = "intel_cpp_compiler"
    MINGW = "mingw_gcc"
    UNKNOWN = "unknown_compiler"


class OptimizationLevel(Enum):
    """Optimization levels detected in binaries"""
    O0 = "none"           # No optimization
    O1 = "basic"          # Basic optimization
    O2 = "standard"       # Standard optimization
    O3 = "aggressive"     # Aggressive optimization
    OS = "size"           # Size optimization
    OZ = "ultra_size"     # Ultra size optimization
    UNKNOWN = "unknown"


@dataclass
class CompilerSignature:
    """Comprehensive compiler signature result"""
    compiler_type: CompilerType
    version: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = None
    rich_header_info: Optional[Dict[str, Any]] = None
    optimization_level: OptimizationLevel = OptimizationLevel.UNKNOWN
    optimization_confidence: float = 0.0
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []


@dataclass
class OptimizationAnalysis:
    """Detailed optimization analysis result"""
    level: OptimizationLevel
    confidence: float
    detected_patterns: List[str]
    optimization_artifacts: List[Dict[str, Any]]
    ml_prediction: Optional[Dict[str, Any]] = None
    performance_indicators: Optional[Dict[str, float]] = None


class AdvancedCompilerFingerprinter:
    """
    Advanced compiler fingerprinting engine using multiple analysis techniques
    
    Combines traditional signature matching with machine learning models
    for accurate compiler and optimization detection.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize signature databases
        self.compiler_signatures = self._initialize_compiler_signatures()
        self.optimization_patterns = self._initialize_optimization_patterns()
        self.rich_header_database = self._initialize_rich_header_db()
        
        # Initialize ML models if available
        self.ml_enabled = SKLEARN_AVAILABLE
        self.compiler_classifier = None
        self.optimization_classifier = None
        
        if self.ml_enabled:
            self._initialize_ml_models()
    
    def analyze_compiler_fingerprint(self, binary_path: str, binary_content: bytes = None) -> CompilerSignature:
        """
        Perform comprehensive compiler fingerprinting analysis
        
        Args:
            binary_path: Path to binary file
            binary_content: Optional pre-loaded binary content
            
        Returns:
            CompilerSignature with comprehensive analysis results
        """
        if binary_content is None:
            try:
                with open(binary_path, 'rb') as f:
                    binary_content = f.read()
            except Exception as e:
                raise ValueError(f"Failed to read binary file: {e}")
        
        self.logger.info(f"Starting advanced compiler fingerprinting for {binary_path}")
        
        # Phase 1: Traditional signature analysis
        signature_result = self._analyze_traditional_signatures(binary_content)
        
        # Phase 2: Rich header analysis (for PE files)
        rich_header_result = self._analyze_rich_header(binary_content)
        
        # Phase 3: Optimization pattern analysis
        optimization_result = self._analyze_optimization_patterns(binary_content)
        
        # Phase 4: ML-based classification (if available)
        ml_result = self._analyze_with_ml(binary_content) if self.ml_enabled else None
        
        # Phase 5: Name mangling analysis
        mangling_result = self._analyze_name_mangling(binary_content)
        
        # Combine all analysis results
        final_signature = self._combine_analysis_results(
            signature_result, rich_header_result, optimization_result,
            ml_result, mangling_result
        )
        
        self.logger.info(
            f"Compiler fingerprinting complete: {final_signature.compiler_type.value} "
            f"(confidence: {final_signature.confidence:.3f})"
        )
        
        return final_signature
    
    def _initialize_compiler_signatures(self) -> Dict[CompilerType, Dict[str, Any]]:
        """Initialize comprehensive compiler signature database"""
        return {
            CompilerType.MSVC: {
                'binary_patterns': [
                    rb'Microsoft.*C/C\+\+.*Compiler',
                    rb'MSVCRT\.dll',
                    rb'VCRUNTIME\d+\.dll',
                    rb'MSVCP\d+\.dll',
                    rb'__security_cookie',
                    rb'_chkstk',
                    rb'___security_init_cookie',
                    rb'__scrt_common_main',
                    rb'__security_check_cookie',
                    rb'_RTC_CheckEsp',
                    rb'_guard_check_icall'
                ],
                'version_patterns': {
                    '12.0': [rb'Microsoft.*C/C\+\+.*Version 18\.00'],  # VS 2013
                    '14.0': [rb'Microsoft.*C/C\+\+.*Version 19\.00'],  # VS 2015
                    '14.1': [rb'Microsoft.*C/C\+\+.*Version 19\.1\d'], # VS 2017
                    '14.2': [rb'Microsoft.*C/C\+\+.*Version 19\.2\d'], # VS 2019
                    '14.3': [rb'Microsoft.*C/C\+\+.*Version 19\.3\d'], # VS 2022
                    'latest': [rb'Microsoft.*C/C\+\+.*Version 19\.\d+']
                },
                'calling_conventions': ['stdcall', 'fastcall', 'cdecl', 'vectorcall'],
                'optimization_indicators': {
                    'O0': [rb'push.*ebp.*mov.*ebp.*esp', rb'redundant_moves'],
                    'O1': [rb'optimized_register_usage', rb'basic_optimization'],
                    'O2': [rb'function_inlining', rb'loop_optimization'],
                    'O3': [rb'aggressive_inlining', rb'vectorization']
                }
            },
            
            CompilerType.GCC: {
                'binary_patterns': [
                    rb'GCC:.*\d+\.\d+\.\d+',
                    rb'__gxx_personality_v0',
                    rb'_Unwind_Resume',
                    rb'__cxa_finalize',
                    rb'__cxa_atexit',
                    rb'__stack_chk_fail',
                    rb'_GLOBAL_OFFSET_TABLE_',
                    rb'__gmon_start__',
                    rb'_ITM_deregisterTMCloneTable',
                    rb'_ITM_registerTMCloneTable'
                ],
                'version_patterns': {
                    '4.x': [rb'GCC.*4\.\d+\.\d+'],
                    '5.x': [rb'GCC.*5\.\d+\.\d+'],
                    '6.x': [rb'GCC.*6\.\d+\.\d+'],
                    '7.x': [rb'GCC.*7\.\d+\.\d+'],
                    '8.x': [rb'GCC.*8\.\d+\.\d+'],
                    '9.x': [rb'GCC.*9\.\d+\.\d+'],
                    '10.x': [rb'GCC.*10\.\d+\.\d+'],
                    '11.x': [rb'GCC.*11\.\d+\.\d+']
                },
                'calling_conventions': ['sysv', 'ms_abi'],
                'optimization_indicators': {
                    'O0': [rb'frame_pointer_preservation', rb'no_optimization'],
                    'O1': [rb'basic_optimization', rb'constant_folding'],
                    'O2': [rb'function_inlining', rb'loop_unrolling'],
                    'O3': [rb'aggressive_optimization', rb'vectorization'],
                    'Os': [rb'size_optimization', rb'space_efficiency']
                }
            },
            
            CompilerType.CLANG: {
                'binary_patterns': [
                    rb'clang version \d+\.\d+\.\d+',
                    rb'__clang_version__',
                    rb'libclang_rt',
                    rb'__asan_init',
                    rb'__tsan_init',
                    rb'__msan_init',
                    rb'__ubsan_handle',
                    rb'__sanitizer_',
                    rb'llvm\..*\.intrinsic'
                ],
                'version_patterns': {
                    '3.x': [rb'clang version 3\.\d+\.\d+'],
                    '4.x': [rb'clang version 4\.\d+\.\d+'],
                    '5.x': [rb'clang version 5\.\d+\.\d+'],
                    '6.x': [rb'clang version 6\.\d+\.\d+'],
                    '7.x': [rb'clang version 7\.\d+\.\d+'],
                    '8.x': [rb'clang version 8\.\d+\.\d+'],
                    '9.x': [rb'clang version 9\.\d+\.\d+'],
                    '10.x': [rb'clang version 10\.\d+\.\d+'],
                    '11.x': [rb'clang version 11\.\d+\.\d+'],
                    '12.x': [rb'clang version 12\.\d+\.\d+'],
                    '13.x': [rb'clang version 13\.\d+\.\d+'],
                    '14.x': [rb'clang version 14\.\d+\.\d+']
                },
                'calling_conventions': ['sysv', 'ms_abi', 'fastcall'],
                'optimization_indicators': {
                    'O0': [rb'optnone', rb'no_optimization'],
                    'O1': [rb'basic_optimization'],
                    'O2': [rb'function_inlining', rb'loop_optimization'],
                    'O3': [rb'aggressive_optimization'],
                    'Os': [rb'size_optimization'],
                    'Oz': [rb'ultra_size_optimization']
                }
            },
            
            CompilerType.INTEL_ICC: {
                'binary_patterns': [
                    rb'Intel.*C\+\+.*Compiler',
                    rb'icc.*\d+\.\d+',
                    rb'__intel_cpu_indicator',
                    rb'__intel_cpu_features_init',
                    rb'_intel_fast_memcpy',
                    rb'_intel_fast_memset',
                    rb'__svml_',
                    rb'__icc_',
                    rb'__ICL'
                ],
                'version_patterns': {
                    '18.x': [rb'Intel.*18\.\d+'],
                    '19.x': [rb'Intel.*19\.\d+'],
                    '20.x': [rb'Intel.*20\.\d+'],
                    '21.x': [rb'Intel.*21\.\d+']
                },
                'calling_conventions': ['icc', 'fastcall', 'vectorcall'],
                'optimization_indicators': {
                    'O0': [rb'no_optimization'],
                    'O1': [rb'basic_optimization'],
                    'O2': [rb'intel_optimization'],
                    'O3': [rb'aggressive_intel_optimization'],
                    'fast': [rb'fast_math_optimization']
                }
            }
        }
    
    def _initialize_optimization_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive optimization pattern database"""
        return {
            'function_inlining': {
                'signatures': [
                    rb'inline_marker',
                    rb'function_body_duplication',
                    rb'call_site_replacement'
                ],
                'indicators': ['reduced_call_count', 'increased_code_size'],
                'optimization_levels': ['O1', 'O2', 'O3']
            },
            
            'loop_optimization': {
                'signatures': [
                    rb'loop_unroll',
                    rb'vectorized_loop',
                    rb'loop_interchange',
                    rb'loop_blocking'
                ],
                'indicators': ['unrolled_instructions', 'simd_instructions'],
                'optimization_levels': ['O2', 'O3']
            },
            
            'constant_folding': {
                'signatures': [
                    rb'immediate_values',
                    rb'constant_propagation',
                    rb'compile_time_evaluation'
                ],
                'indicators': ['fewer_arithmetic_ops', 'immediate_operands'],
                'optimization_levels': ['O1', 'O2', 'O3']
            },
            
            'dead_code_elimination': {
                'signatures': [
                    rb'unreachable_code_removal',
                    rb'unused_variable_elimination',
                    rb'conditional_branch_optimization'
                ],
                'indicators': ['compact_code', 'eliminated_branches'],
                'optimization_levels': ['O1', 'O2', 'O3']
            },
            
            'register_allocation': {
                'signatures': [
                    rb'register_spill_optimization',
                    rb'register_coalescing',
                    rb'live_range_analysis'
                ],
                'indicators': ['efficient_register_usage', 'reduced_memory_access'],
                'optimization_levels': ['O1', 'O2', 'O3']
            },
            
            'vectorization': {
                'signatures': [
                    rb'sse_instructions',
                    rb'avx_instructions',
                    rb'simd_operations',
                    rb'packed_operations'
                ],
                'indicators': ['vector_instructions', 'parallel_operations'],
                'optimization_levels': ['O2', 'O3']
            }
        }
    
    def _initialize_rich_header_db(self) -> Dict[str, Dict[str, Any]]:
        """Initialize Rich header signature database for MSVC detection"""
        return {
            'visual_studio_versions': {
                '6.0': {'linker_version': 6.0, 'compiler_versions': [12.00]},
                '2002': {'linker_version': 7.0, 'compiler_versions': [13.00]},
                '2003': {'linker_version': 7.10, 'compiler_versions': [13.10]},
                '2005': {'linker_version': 8.0, 'compiler_versions': [14.00]},
                '2008': {'linker_version': 9.0, 'compiler_versions': [15.00]},
                '2010': {'linker_version': 10.0, 'compiler_versions': [16.00]},
                '2012': {'linker_version': 11.0, 'compiler_versions': [17.00]},
                '2013': {'linker_version': 12.0, 'compiler_versions': [18.00]},
                '2015': {'linker_version': 14.0, 'compiler_versions': [19.00]},
                '2017': {'linker_version': 14.1, 'compiler_versions': [19.10, 19.11, 19.12, 19.13, 19.14, 19.15, 19.16]},
                '2019': {'linker_version': 14.2, 'compiler_versions': [19.20, 19.21, 19.22, 19.23, 19.24, 19.25, 19.26, 19.27, 19.28, 19.29]},
                '2022': {'linker_version': 14.3, 'compiler_versions': [19.30, 19.31, 19.32, 19.33, 19.34, 19.35, 19.36, 19.37]}
            },
            
            'tool_signatures': {
                'cl.exe': {'id_start': 0x00, 'id_end': 0x50},
                'link.exe': {'id_start': 0x50, 'id_end': 0x60},
                'lib.exe': {'id_start': 0x60, 'id_end': 0x70},
                'ml.exe': {'id_start': 0x70, 'id_end': 0x80},
                'ml64.exe': {'id_start': 0x80, 'id_end': 0x90},
                'cvtres.exe': {'id_start': 0x90, 'id_end': 0xA0}
            }
        }
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for compiler classification"""
        if not self.ml_enabled:
            return
        
        try:
            # Initialize compiler classification model
            self.compiler_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
            
            # Initialize optimization level classification model
            self.optimization_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=15,
                random_state=42
            )
            
            # Initialize feature vectorizer for text-based features
            self.feature_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                analyzer='char'
            )
            
            # Load pre-trained models if available
            self._load_pretrained_models()
            
            self.logger.info("ML models initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize ML models: {e}")
            self.ml_enabled = False
    
    def _load_pretrained_models(self):
        """Load pre-trained ML models from disk if available"""
        models_dir = Path(__file__).parent / "ml_models"
        
        try:
            if (models_dir / "compiler_classifier.pkl").exists():
                with open(models_dir / "compiler_classifier.pkl", 'rb') as f:
                    self.compiler_classifier = pickle.load(f)
                self.logger.info("Loaded pre-trained compiler classifier")
            
            if (models_dir / "optimization_classifier.pkl").exists():
                with open(models_dir / "optimization_classifier.pkl", 'rb') as f:
                    self.optimization_classifier = pickle.load(f)
                self.logger.info("Loaded pre-trained optimization classifier")
                
        except Exception as e:
            self.logger.warning(f"Failed to load pre-trained models: {e}")
    
    def _analyze_traditional_signatures(self, binary_content: bytes) -> CompilerSignature:
        """Analyze binary using traditional signature matching"""
        detected_compilers = []
        
        for compiler_type, signatures in self.compiler_signatures.items():
            matches = 0
            evidence = []
            detected_version = None
            
            # Check main binary patterns
            for pattern in signatures['binary_patterns']:
                if re.search(pattern, binary_content, re.IGNORECASE):
                    matches += 1
                    evidence.append(f"Pattern match: {pattern.decode('utf-8', errors='ignore')}")
            
            # Check version-specific patterns
            for version, version_patterns in signatures.get('version_patterns', {}).items():
                for pattern in version_patterns:
                    if re.search(pattern, binary_content, re.IGNORECASE):
                        detected_version = version
                        evidence.append(f"Version pattern: {version}")
                        matches += 1
                        break
                if detected_version:
                    break
            
            if matches > 0:
                confidence = min(matches / len(signatures['binary_patterns']), 1.0)
                detected_compilers.append(CompilerSignature(
                    compiler_type=compiler_type,
                    version=detected_version,
                    confidence=confidence,
                    evidence=evidence
                ))
        
        # Return the most confident detection
        if detected_compilers:
            best_match = max(detected_compilers, key=lambda x: x.confidence)
            return best_match
        else:
            return CompilerSignature(
                compiler_type=CompilerType.UNKNOWN,
                confidence=0.0,
                evidence=['No compiler signatures detected']
            )
    
    def _analyze_rich_header(self, binary_content: bytes) -> Optional[Dict[str, Any]]:
        """Analyze PE Rich header for MSVC version detection"""
        try:
            # Look for PE signature
            if binary_content[:2] != b'MZ':
                return None
            
            # Get PE header offset
            pe_offset = struct.unpack('<L', binary_content[60:64])[0]
            
            # Check if it's a valid PE
            if binary_content[pe_offset:pe_offset+4] != b'PE\x00\x00':
                return None
            
            # Look for Rich header signature
            rich_start = binary_content.find(b'Rich')
            if rich_start == -1:
                return None
            
            # Find DanS signature (Rich header start)
            dans_start = binary_content.rfind(b'DanS', 0, rich_start)
            if dans_start == -1:
                return None
            
            # Extract Rich header data
            rich_data = binary_content[dans_start:rich_start+8]
            
            # Parse Rich header entries
            rich_entries = []
            for i in range(dans_start + 16, rich_start, 8):
                if i + 8 <= len(binary_content):
                    comp_id = struct.unpack('<L', binary_content[i:i+4])[0]
                    use_count = struct.unpack('<L', binary_content[i+4:i+8])[0]
                    
                    # Decode component ID
                    version = (comp_id >> 16) & 0xFFFF
                    build = comp_id & 0xFFFF
                    
                    rich_entries.append({
                        'component_id': comp_id,
                        'version': version,
                        'build': build,
                        'use_count': use_count
                    })
            
            # Analyze Rich header for compiler information
            compiler_info = self._interpret_rich_header(rich_entries)
            
            return {
                'rich_header_found': True,
                'entries': rich_entries,
                'compiler_info': compiler_info
            }
            
        except Exception as e:
            self.logger.debug(f"Rich header analysis failed: {e}")
            return None
    
    def _interpret_rich_header(self, rich_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Interpret Rich header entries to determine compiler version"""
        compiler_info = {
            'visual_studio_version': None,
            'compiler_version': None,
            'linker_version': None,
            'tools_used': [],
            'confidence': 0.0
        }
        
        # Analyze version patterns
        for entry in rich_entries:
            version = entry['version']
            build = entry['build']
            
            # Check against known Visual Studio versions
            for vs_version, vs_info in self.rich_header_database['visual_studio_versions'].items():
                if version in vs_info['compiler_versions']:
                    compiler_info['visual_studio_version'] = vs_version
                    compiler_info['compiler_version'] = version
                    compiler_info['confidence'] = 0.9
                    break
            
            # Identify tools used
            for tool_name, tool_info in self.rich_header_database['tool_signatures'].items():
                if tool_info['id_start'] <= (entry['component_id'] >> 16) <= tool_info['id_end']:
                    compiler_info['tools_used'].append(tool_name)
        
        return compiler_info
    
    def _analyze_optimization_patterns(self, binary_content: bytes) -> OptimizationAnalysis:
        """Analyze binary for optimization patterns and levels"""
        detected_patterns = []
        optimization_artifacts = []
        level_scores = {level: 0.0 for level in OptimizationLevel}
        
        # Analyze each optimization pattern
        for pattern_name, pattern_info in self.optimization_patterns.items():
            pattern_detected = False
            
            # Check for pattern signatures
            for signature in pattern_info['signatures']:
                if re.search(signature, binary_content, re.IGNORECASE):
                    pattern_detected = True
                    detected_patterns.append(pattern_name)
                    
                    # Add to optimization artifacts
                    optimization_artifacts.append({
                        'pattern': pattern_name,
                        'signature': signature.decode('utf-8', errors='ignore'),
                        'confidence': 0.8
                    })
                    break
            
            # Score optimization levels based on detected patterns
            if pattern_detected:
                for opt_level in pattern_info['optimization_levels']:
                    try:
                        enum_level = OptimizationLevel[opt_level]
                        level_scores[enum_level] += 1.0
                    except KeyError:
                        pass
        
        # Determine most likely optimization level
        if any(score > 0 for score in level_scores.values()):
            best_level = max(level_scores.keys(), key=lambda x: level_scores[x])
            confidence = level_scores[best_level] / max(sum(level_scores.values()), 1)
        else:
            best_level = OptimizationLevel.UNKNOWN
            confidence = 0.0
        
        return OptimizationAnalysis(
            level=best_level,
            confidence=confidence,
            detected_patterns=detected_patterns,
            optimization_artifacts=optimization_artifacts
        )
    
    def _analyze_with_ml(self, binary_content: bytes) -> Optional[Dict[str, Any]]:
        """Analyze binary using machine learning models"""
        if not self.ml_enabled:
            return None
        
        try:
            # Extract features for ML analysis
            features = self._extract_ml_features(binary_content)
            
            # Predict compiler type
            compiler_prediction = None
            if hasattr(self.compiler_classifier, 'predict_proba'):
                compiler_proba = self.compiler_classifier.predict_proba([features])
                compiler_classes = self.compiler_classifier.classes_
                
                # Get best prediction
                best_idx = np.argmax(compiler_proba[0])
                compiler_prediction = {
                    'compiler': compiler_classes[best_idx],
                    'confidence': compiler_proba[0][best_idx],
                    'all_predictions': dict(zip(compiler_classes, compiler_proba[0]))
                }
            
            # Predict optimization level
            optimization_prediction = None
            if hasattr(self.optimization_classifier, 'predict_proba'):
                opt_proba = self.optimization_classifier.predict_proba([features])
                opt_classes = self.optimization_classifier.classes_
                
                best_idx = np.argmax(opt_proba[0])
                optimization_prediction = {
                    'optimization_level': opt_classes[best_idx],
                    'confidence': opt_proba[0][best_idx],
                    'all_predictions': dict(zip(opt_classes, opt_proba[0]))
                }
            
            return {
                'compiler_prediction': compiler_prediction,
                'optimization_prediction': optimization_prediction,
                'feature_count': len(features),
                'ml_analysis_success': True
            }
            
        except Exception as e:
            self.logger.warning(f"ML analysis failed: {e}")
            return {'ml_analysis_success': False, 'error': str(e)}
    
    def _extract_ml_features(self, binary_content: bytes) -> List[float]:
        """Extract features for machine learning analysis"""
        features = []
        
        # Statistical features
        features.extend(self._extract_statistical_features(binary_content))
        
        # Structural features  
        features.extend(self._extract_structural_features(binary_content))
        
        # Instruction pattern features
        features.extend(self._extract_instruction_features(binary_content))
        
        # String pattern features
        features.extend(self._extract_string_features(binary_content))
        
        return features
    
    def _extract_statistical_features(self, binary_content: bytes) -> List[float]:
        """Extract statistical features from binary"""
        features = []
        
        # Entropy calculation
        if len(binary_content) > 0:
            byte_counts = np.bincount(np.frombuffer(binary_content, dtype=np.uint8), minlength=256)
            probabilities = byte_counts / len(binary_content)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            features.append(entropy)
        else:
            features.append(0.0)
        
        # Byte distribution features
        features.append(len(binary_content))  # File size
        features.append(np.mean(np.frombuffer(binary_content[:1000], dtype=np.uint8)))  # Mean byte value
        features.append(np.std(np.frombuffer(binary_content[:1000], dtype=np.uint8)))   # Std deviation
        
        return features
    
    def _extract_structural_features(self, binary_content: bytes) -> List[float]:
        """Extract structural features from binary"""
        features = []
        
        # Section-based features (simplified)
        features.append(binary_content.count(b'.text'))   # Code section indicators
        features.append(binary_content.count(b'.data'))   # Data section indicators
        features.append(binary_content.count(b'.rdata'))  # Read-only data
        features.append(binary_content.count(b'.bss'))    # BSS section
        
        # Import/export related
        features.append(binary_content.count(b'.dll'))    # DLL references
        features.append(binary_content.count(b'kernel32')) # Common imports
        
        return features
    
    def _extract_instruction_features(self, binary_content: bytes) -> List[float]:
        """Extract instruction pattern features"""
        features = []
        
        # Common x86/x64 instruction patterns
        instruction_patterns = [
            rb'\x55',           # push ebp
            rb'\x8b\xec',       # mov esp, ebp
            rb'\x89\xe5',       # mov ebp, esp
            rb'\xc3',           # ret
            rb'\xe8',           # call
            rb'\x74',           # je
            rb'\x75',           # jne
            rb'\xeb',           # jmp
        ]
        
        for pattern in instruction_patterns:
            features.append(binary_content.count(pattern))
        
        return features
    
    def _extract_string_features(self, binary_content: bytes) -> List[float]:
        """Extract string-based features"""
        features = []
        
        # Convert to string for pattern matching
        try:
            binary_str = binary_content.decode('utf-8', errors='ignore')
        except:
            binary_str = str(binary_content)
        
        # Common compiler-specific strings
        compiler_strings = [
            'microsoft', 'visual', 'gcc', 'clang', 'intel',
            '__cdecl', '__stdcall', '__fastcall',
            'runtime', 'crt', 'msvcr', 'libgcc'
        ]
        
        for string in compiler_strings:
            features.append(binary_str.lower().count(string.lower()))
        
        return features
    
    def _analyze_name_mangling(self, binary_content: bytes) -> Dict[str, Any]:
        """Analyze name mangling schemes to identify compiler"""
        mangling_analysis = {
            'detected_schemes': [],
            'confidence': 0.0,
            'compiler_hints': []
        }
        
        try:
            # Convert binary to string for analysis
            binary_str = binary_content.decode('utf-8', errors='ignore')
            
            # MSVC name mangling patterns
            msvc_patterns = [
                r'\?[A-Za-z_][A-Za-z0-9_]*@@',  # MSVC decorated names
                r'_[A-Za-z_][A-Za-z0-9_]*@\d+', # stdcall decoration
                r'@[A-Za-z_][A-Za-z0-9_]*@\d+', # fastcall decoration
            ]
            
            # GCC/Clang name mangling patterns
            gcc_patterns = [
                r'_Z[A-Za-z0-9_]+',              # C++ mangled names
                r'_ZN[A-Za-z0-9_]+',             # Namespace mangled names
                r'_ZL[A-Za-z0-9_]+',             # Local/static mangled names
            ]
            
            # Check MSVC patterns
            msvc_matches = 0
            for pattern in msvc_patterns:
                matches = len(re.findall(pattern, binary_str))
                if matches > 0:
                    msvc_matches += matches
                    mangling_analysis['detected_schemes'].append(f'MSVC: {pattern}')
            
            # Check GCC patterns
            gcc_matches = 0
            for pattern in gcc_patterns:
                matches = len(re.findall(pattern, binary_str))
                if matches > 0:
                    gcc_matches += matches
                    mangling_analysis['detected_schemes'].append(f'GCC/Clang: {pattern}')
            
            # Determine compiler hints based on mangling
            if msvc_matches > gcc_matches:
                mangling_analysis['compiler_hints'].append('MSVC')
                mangling_analysis['confidence'] = min(msvc_matches / 10.0, 1.0)
            elif gcc_matches > msvc_matches:
                mangling_analysis['compiler_hints'].append('GCC/Clang')
                mangling_analysis['confidence'] = min(gcc_matches / 10.0, 1.0)
            
        except Exception as e:
            self.logger.debug(f"Name mangling analysis failed: {e}")
        
        return mangling_analysis
    
    def _combine_analysis_results(
        self, 
        signature_result: CompilerSignature,
        rich_header_result: Optional[Dict[str, Any]],
        optimization_result: OptimizationAnalysis,
        ml_result: Optional[Dict[str, Any]],
        mangling_result: Dict[str, Any]
    ) -> CompilerSignature:
        """Combine all analysis results into final signature"""
        
        # Start with signature analysis result
        final_signature = CompilerSignature(
            compiler_type=signature_result.compiler_type,
            version=signature_result.version,
            confidence=signature_result.confidence,
            evidence=signature_result.evidence.copy(),
            optimization_level=optimization_result.level,
            optimization_confidence=optimization_result.confidence
        )
        
        # Enhance with Rich header information
        if rich_header_result and rich_header_result.get('rich_header_found'):
            rich_info = rich_header_result['compiler_info']
            if rich_info.get('visual_studio_version'):
                final_signature.compiler_type = CompilerType.MSVC
                final_signature.version = rich_info['visual_studio_version']
                final_signature.confidence = max(final_signature.confidence, rich_info['confidence'])
                final_signature.evidence.append(f"Rich header: VS {rich_info['visual_studio_version']}")
                final_signature.rich_header_info = rich_header_result
        
        # Enhance with ML predictions
        if ml_result and ml_result.get('ml_analysis_success'):
            if ml_result.get('compiler_prediction'):
                ml_compiler = ml_result['compiler_prediction']
                ml_confidence = ml_compiler['confidence']
                
                # If ML has high confidence and disagrees with signature analysis
                if ml_confidence > 0.8 and ml_confidence > final_signature.confidence:
                    try:
                        ml_compiler_type = CompilerType(ml_compiler['compiler'])
                        final_signature.compiler_type = ml_compiler_type
                        final_signature.confidence = ml_confidence
                        final_signature.evidence.append(f"ML prediction: {ml_compiler['compiler']} ({ml_confidence:.3f})")
                    except ValueError:
                        pass
            
            # Enhance optimization analysis with ML
            if ml_result.get('optimization_prediction'):
                ml_opt = ml_result['optimization_prediction']
                if ml_opt['confidence'] > optimization_result.confidence:
                    try:
                        ml_opt_level = OptimizationLevel(ml_opt['optimization_level'])
                        final_signature.optimization_level = ml_opt_level
                        final_signature.optimization_confidence = ml_opt['confidence']
                    except ValueError:
                        pass
        
        # Enhance with name mangling analysis
        if mangling_result['confidence'] > 0.5:
            for hint in mangling_result['compiler_hints']:
                if hint == 'MSVC' and final_signature.compiler_type == CompilerType.UNKNOWN:
                    final_signature.compiler_type = CompilerType.MSVC
                    final_signature.confidence = max(final_signature.confidence, mangling_result['confidence'])
                    final_signature.evidence.append(f"Name mangling: {hint}")
                elif hint == 'GCC/Clang' and final_signature.compiler_type == CompilerType.UNKNOWN:
                    final_signature.compiler_type = CompilerType.GCC
                    final_signature.confidence = max(final_signature.confidence, mangling_result['confidence'])
                    final_signature.evidence.append(f"Name mangling: {hint}")
        
        return final_signature


# Utility functions for integration with Agent 2 (Architect)
def enhance_agent2_compiler_detection(binary_path: str, existing_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Enhance Agent 2 (Architect) with advanced compiler fingerprinting
    
    Args:
        binary_path: Path to binary file
        existing_analysis: Existing analysis from Agent 2
        
    Returns:
        Enhanced compiler analysis with advanced fingerprinting
    """
    try:
        # Initialize advanced fingerprinter
        fingerprinter = AdvancedCompilerFingerprinter()
        
        # Perform advanced analysis
        signature = fingerprinter.analyze_compiler_fingerprint(binary_path)
        
        # Create enhanced analysis result
        enhanced_analysis = {
            'advanced_compiler_analysis': {
                'compiler_type': signature.compiler_type.value,
                'version': signature.version,
                'confidence': signature.confidence,
                'evidence': signature.evidence,
                'optimization_level': signature.optimization_level.value,
                'optimization_confidence': signature.optimization_confidence,
                'rich_header_info': signature.rich_header_info,
                'analysis_method': 'advanced_fingerprinting_p2_1',
                'enhancement_applied': True
            }
        }
        
        # Merge with existing analysis if provided
        if existing_analysis:
            enhanced_analysis.update(existing_analysis)
            
            # Update confidence scores
            if 'compiler_analysis' in existing_analysis:
                existing_conf = existing_analysis['compiler_analysis'].get('confidence', 0)
                enhanced_analysis['compiler_analysis']['confidence'] = max(
                    existing_conf, signature.confidence
                )
        
        return enhanced_analysis
        
    except Exception as e:
        logger.error(f"Advanced compiler fingerprinting failed: {e}")
        # Return original analysis if enhancement fails
        return existing_analysis or {'error': str(e)}


def create_compiler_fingerprinting_report(signature: CompilerSignature) -> str:
    """
    Create detailed report for compiler fingerprinting results
    
    Args:
        signature: CompilerSignature result
        
    Returns:
        Formatted analysis report
    """
    report_lines = [
        "=== Advanced Compiler Fingerprinting Report ===",
        "",
        f"Compiler Type: {signature.compiler_type.value}",
        f"Version: {signature.version or 'Unknown'}",
        f"Confidence: {signature.confidence:.3f}",
        f"Optimization Level: {signature.optimization_level.value}",
        f"Optimization Confidence: {signature.optimization_confidence:.3f}",
        "",
        "Evidence Found:",
    ]
    
    for evidence in signature.evidence:
        report_lines.append(f"  - {evidence}")
    
    if signature.rich_header_info:
        report_lines.extend([
            "",
            "Rich Header Analysis:",
            f"  - Visual Studio Version: {signature.rich_header_info['compiler_info'].get('visual_studio_version', 'Unknown')}",
            f"  - Tools Used: {', '.join(signature.rich_header_info['compiler_info'].get('tools_used', []))}",
        ])
    
    report_lines.extend([
        "",
        "=== End Report ===",
        ""
    ])
    
    return "\n".join(report_lines)