"""
Agent implementations for open-sourcefy binary analysis and decompilation.
"""

from .agent01_binary_discovery import Agent1_BinaryDiscovery
from .agent02_arch_analysis import Agent2_ArchAnalysis
from .agent03_smart_error_pattern_matching import Agent3_SmartErrorPatternMatching
from .agent04_basic_decompiler import Agent4_BasicDecompiler
from .agent05_binary_structure_analyzer import Agent5_BinaryStructureAnalyzer
from .agent06_optimization_matcher import Agent6_OptimizationMatcher
from .agent07_advanced_decompiler import Agent7_AdvancedDecompiler
from .agent08_binary_diff_analyzer import Agent8_BinaryDiffAnalyzer
from .agent09_advanced_assembly_analyzer import Agent9_AdvancedAssemblyAnalyzer
from .agent10_resource_reconstructor import Agent10_ResourceReconstructor
from .agent11_global_reconstructor import Agent11_GlobalReconstructor
from .agent12_compilation_orchestrator import Agent12_CompilationOrchestrator
from .agent13_final_validator import Agent13_FinalValidator
from .agent14_advanced_ghidra import Agent14_AdvancedGhidra
from .agent15_metadata_analysis import Agent15_MetadataAnalysis
from .agent16_dynamic_bridge import Agent16_DynamicBridge
from .agent18_advanced_build_systems import Agent18_AdvancedBuildSystems
from .agent19_binary_comparison import Agent19_BinaryComparison
from .agent20_auto_testing import Agent20_AutoTesting

__all__ = [
    'Agent1_BinaryDiscovery',
    'Agent2_ArchAnalysis', 
    'Agent3_SmartErrorPatternMatching',
    'Agent4_BasicDecompiler',
    'Agent5_BinaryStructureAnalyzer',
    'Agent6_OptimizationMatcher',
    'Agent7_AdvancedDecompiler',
    'Agent8_BinaryDiffAnalyzer',
    'Agent9_AdvancedAssemblyAnalyzer',
    'Agent10_ResourceReconstructor',
    'Agent11_GlobalReconstructor',
    'Agent12_CompilationOrchestrator',
    'Agent13_FinalValidator',
    'Agent14_AdvancedGhidra',
    'Agent15_MetadataAnalysis',
    'Agent16_DynamicBridge',
    'Agent18_AdvancedBuildSystems',
    'Agent19_BinaryComparison',
    'Agent20_AutoTesting'
]

def create_all_agents():
    """Factory function to create all agents with proper dependencies"""
    agents = [
        Agent1_BinaryDiscovery(),
        Agent2_ArchAnalysis(),
        Agent3_SmartErrorPatternMatching(),
        Agent4_BasicDecompiler(),
        Agent5_BinaryStructureAnalyzer(),
        Agent6_OptimizationMatcher(),
        Agent7_AdvancedDecompiler(),
        Agent8_BinaryDiffAnalyzer(),
        Agent9_AdvancedAssemblyAnalyzer(),
        Agent10_ResourceReconstructor(),
        Agent11_GlobalReconstructor(),
        Agent12_CompilationOrchestrator(),
        Agent13_FinalValidator(),
        Agent14_AdvancedGhidra(),
        Agent15_MetadataAnalysis(),
        Agent16_DynamicBridge(),
        Agent18_AdvancedBuildSystems(),
        Agent19_BinaryComparison(),
        Agent20_AutoTesting()
    ]
    return agents