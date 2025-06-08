"""
Advanced Deobfuscation Module for NSA-Level Binary Analysis

This module implements Phase 1 of the advanced binary decompilation pipeline,
focusing on foundational analysis and deobfuscation capabilities.

Components:
- Entropy analysis for packed section detection
- Control flow graph reconstruction
- Packer detection and unpacking
- Anti-obfuscation techniques
- Virtual machine obfuscation detection
"""

from .entropy_analyzer import EntropyAnalyzer, PackedSectionDetector
from .cfg_reconstructor import CFGReconstructor, AdvancedControlFlowAnalyzer
from .packer_detector import PackerDetector, UnpackingEngine
from .obfuscation_detector import ObfuscationDetector, AntiAnalysisDetector

__all__ = [
    'EntropyAnalyzer',
    'PackedSectionDetector', 
    'CFGReconstructor',
    'AdvancedControlFlowAnalyzer',
    'PackerDetector',
    'UnpackingEngine',
    'ObfuscationDetector',
    'AntiAnalysisDetector'
]

__version__ = '1.0.0'
__author__ = 'Open-Sourcefy Advanced Deobfuscation Team'