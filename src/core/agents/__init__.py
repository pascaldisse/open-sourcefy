"""
Matrix Agents - Complete 16-agent Matrix-themed architecture
Contains all production-ready implementations of agents 1-16
"""

from typing import Dict, Type, Any

# Import Matrix agents individually with error handling
MATRIX_AGENTS = {}
failed_imports = []

# Agent 1: Sentinel
try:
    from .agent01_sentinel import SentinelAgent
    MATRIX_AGENTS[1] = SentinelAgent
except ImportError as e:
    failed_imports.append(f"Agent 1 (Sentinel): {e}")

# Agent 2: Architect  
try:
    from .agent02_architect import ArchitectAgent
    MATRIX_AGENTS[2] = ArchitectAgent
except ImportError as e:
    failed_imports.append(f"Agent 2 (Architect): {e}")

# Agent 3: Merovingian
try:
    from .agent03_merovingian import MerovingianAgent
    MATRIX_AGENTS[3] = MerovingianAgent
except ImportError as e:
    failed_imports.append(f"Agent 3 (Merovingian): {e}")

# Agent 4: Agent Smith
try:
    from .agent04_agent_smith import AgentSmithAgent
    MATRIX_AGENTS[4] = AgentSmithAgent
except ImportError as e:
    failed_imports.append(f"Agent 4 (Agent Smith): {e}")

# Agent 5: Neo
try:
    from .agent05_neo_advanced_decompiler import NeoAgent
    MATRIX_AGENTS[5] = NeoAgent
except ImportError as e:
    failed_imports.append(f"Agent 5 (Neo): {e}")

# Agent 6: Trainman (reordered from Agent 7)
try:
    from .agent07_trainman_assembly_analysis import Agent7_Trainman_AssemblyAnalysis
    MATRIX_AGENTS[6] = Agent7_Trainman_AssemblyAnalysis
except ImportError as e:
    failed_imports.append(f"Agent 6 (Trainman): {e}")

# Agent 7: Keymaker (reordered from Agent 8)
try:
    from .agent08_keymaker_resource_reconstruction import Agent8_Keymaker_ResourceReconstruction
    MATRIX_AGENTS[7] = Agent8_Keymaker_ResourceReconstruction
except ImportError as e:
    failed_imports.append(f"Agent 7 (Keymaker): {e}")

# Agent 8: Commander Locke (reordered from Agent 9)
try:
    from .agent09_commander_locke import CommanderLockeAgent
    MATRIX_AGENTS[8] = CommanderLockeAgent
except ImportError as e:
    failed_imports.append(f"Agent 8 (Commander Locke): {e}")

# Agent 9: The Machine (reordered from Agent 10)
try:
    from .agent10_the_machine import Agent10_TheMachine
    MATRIX_AGENTS[9] = Agent10_TheMachine
except ImportError as e:
    failed_imports.append(f"Agent 9 (The Machine): {e}")

# Agent 10: Twins (reordered from Agent 6)
try:
    from .agent06_twins_binary_diff import Agent6_Twins_BinaryDiff
    MATRIX_AGENTS[10] = Agent6_Twins_BinaryDiff
except ImportError as e:
    failed_imports.append(f"Agent 10 (Twins): {e}")

# Agent 11: The Oracle
try:
    from .agent11_the_oracle import Agent11_TheOracle
    MATRIX_AGENTS[11] = Agent11_TheOracle
except ImportError as e:
    failed_imports.append(f"Agent 11 (The Oracle): {e}")

# Agent 12: Link
try:
    from .agent12_link import Agent12_Link
    MATRIX_AGENTS[12] = Agent12_Link
except ImportError as e:
    failed_imports.append(f"Agent 12 (Link): {e}")

# Agent 13: Agent Johnson
try:
    from .agent13_agent_johnson import Agent13_AgentJohnson
    MATRIX_AGENTS[13] = Agent13_AgentJohnson
except ImportError as e:
    failed_imports.append(f"Agent 13 (Agent Johnson): {e}")

# Agent 14: The Cleaner
try:
    from .agent14_the_cleaner import Agent14_TheCleaner
    MATRIX_AGENTS[14] = Agent14_TheCleaner
except ImportError as e:
    failed_imports.append(f"Agent 14 (The Cleaner): {e}")

# Agent 15: Analyst
try:
    from .agent15_analyst import Agent15_Analyst
    MATRIX_AGENTS[15] = Agent15_Analyst
except ImportError as e:
    failed_imports.append(f"Agent 15 (Analyst): {e}")

# Agent 16: Agent Brown
try:
    from .agent16_agent_brown import Agent16_AgentBrown
    MATRIX_AGENTS[16] = Agent16_AgentBrown
except ImportError as e:
    failed_imports.append(f"Agent 16 (Agent Brown): {e}")

# Report failed imports
if failed_imports:
    print("Warning: Some Matrix agents failed to import:")
    for failure in failed_imports:
        print(f"  - {failure}")

# Agent metadata for system integration (only for working agents)
AGENT_METADATA = {}

# Only add metadata for successfully imported agents
if 1 in MATRIX_AGENTS:
    AGENT_METADATA[1] = {
        'name': 'Sentinel',
        'character': 'sentinel',
        'description': 'Binary discovery and metadata analysis',
        'dependencies': []
    }

if 2 in MATRIX_AGENTS:
    AGENT_METADATA[2] = {
        'name': 'Architect',
        'character': 'architect',
        'description': 'Architecture analysis and design patterns',
        'dependencies': [1]
    }

if 3 in MATRIX_AGENTS:
    AGENT_METADATA[3] = {
        'name': 'Merovingian',
        'character': 'merovingian',
        'description': 'Basic decompilation and control flow',
        'dependencies': [1, 2]
    }

if 4 in MATRIX_AGENTS:
    AGENT_METADATA[4] = {
        'name': 'AgentSmith',
        'character': 'agent_smith',
        'description': 'Binary structure analysis and replication',
        'dependencies': [1, 2]
    }

if 5 in MATRIX_AGENTS:
    AGENT_METADATA[5] = {
        'name': 'Neo',
        'character': 'neo',
        'description': 'Advanced decompilation and code generation',
        'dependencies': [2, 3, 4]
    }

if 6 in MATRIX_AGENTS:
    AGENT_METADATA[6] = {
        'name': 'Trainman',
        'character': 'trainman',
        'description': 'Advanced assembly analysis and transportation',
        'dependencies': [1, 2]
    }

if 7 in MATRIX_AGENTS:
    AGENT_METADATA[7] = {
        'name': 'Keymaker',
        'character': 'keymaker',
        'description': 'Resource reconstruction and access management',
        'dependencies': [1, 2]
    }

if 8 in MATRIX_AGENTS:
    AGENT_METADATA[8] = {
        'name': 'CommanderLocke',
        'character': 'commander_locke',
        'description': 'Global reconstruction orchestration',
        'dependencies': [5, 6, 7]
    }

if 9 in MATRIX_AGENTS:
    AGENT_METADATA[9] = {
        'name': 'TheMachine',
        'character': 'the_machine',
        'description': 'Compilation orchestration and build systems',
        'dependencies': [8]
    }

if 10 in MATRIX_AGENTS:
    AGENT_METADATA[10] = {
        'name': 'Twins',
        'character': 'twins',
        'description': 'Binary differential analysis and comparison',
        'dependencies': [1, 2, 5]
    }

if 11 in MATRIX_AGENTS:
    AGENT_METADATA[11] = {
        'name': 'TheOracle',
        'character': 'the_oracle',
        'description': 'Final validation and truth verification',
        'dependencies': [10]
    }

if 12 in MATRIX_AGENTS:
    AGENT_METADATA[12] = {
        'name': 'Link',
        'character': 'link',
        'description': 'Cross-reference and linking analysis',
        'dependencies': [5, 6, 7]
    }

if 13 in MATRIX_AGENTS:
    AGENT_METADATA[13] = {
        'name': 'AgentJohnson',
        'character': 'agent_johnson',
        'description': 'Security analysis and vulnerability detection',
        'dependencies': [5, 6, 7]
    }

if 14 in MATRIX_AGENTS:
    AGENT_METADATA[14] = {
        'name': 'TheCleaner',
        'character': 'the_cleaner',
        'description': 'Code cleanup and optimization',
        'dependencies': [8, 9, 10]
    }

if 15 in MATRIX_AGENTS:
    AGENT_METADATA[15] = {
        'name': 'Analyst',
        'character': 'analyst',
        'description': 'Advanced metadata analysis and intelligence synthesis',
        'dependencies': [8, 9, 10]
    }

if 16 in MATRIX_AGENTS:
    AGENT_METADATA[16] = {
        'name': 'AgentBrown',
        'character': 'agent_brown',
        'description': 'Final quality assurance and optimization',
        'dependencies': [14, 15]
    }


def get_available_agents() -> Dict[int, Type]:
    """Get dictionary of available agent IDs mapped to their classes"""
    return MATRIX_AGENTS.copy()


def get_agent_by_id(agent_id: int):
    """Get agent class by ID"""
    if agent_id in MATRIX_AGENTS:
        return MATRIX_AGENTS[agent_id]()
    else:
        raise ValueError(f"Agent {agent_id} not found in Matrix agents")


def get_implementation_status() -> Dict[int, bool]:
    """Get implementation status of all Matrix agents"""
    return {agent_id: agent_id in MATRIX_AGENTS for agent_id in range(1, 21)}


def create_all_agents():
    """Create instances of all available Matrix agents"""
    agents = {}
    for agent_id, agent_class in MATRIX_AGENTS.items():
        try:
            agents[agent_id] = agent_class()
        except Exception as e:
            print(f"Warning: Failed to create agent {agent_id}: {e}")
    return agents


def get_decompile_agents():
    """Get agents used for decompilation pipeline"""
    # Based on CLAUDE.md: --decompile-only uses agents 1,2,5,6,14 (6 is now Trainman, was 7)
    decompile_agent_ids = [1, 2, 5, 6, 14]
    return {aid: MATRIX_AGENTS[aid] for aid in decompile_agent_ids if aid in MATRIX_AGENTS}


# Build __all__ list dynamically based on successful imports
__all__ = [
    'MATRIX_AGENTS',
    'AGENT_METADATA', 
    'get_available_agents',
    'get_agent_by_id',
    'get_implementation_status',
    'create_all_agents',
    'get_decompile_agents'
]

# Add successfully imported agent classes to __all__
if 1 in MATRIX_AGENTS:
    __all__.append('SentinelAgent')
if 2 in MATRIX_AGENTS:
    __all__.append('ArchitectAgent')
if 3 in MATRIX_AGENTS:
    __all__.append('MerovingianAgent')
if 4 in MATRIX_AGENTS:
    __all__.append('AgentSmithAgent')
if 5 in MATRIX_AGENTS:
    __all__.append('NeoAgent')
if 6 in MATRIX_AGENTS:
    __all__.append('Agent7_Trainman_AssemblyAnalysis')
if 7 in MATRIX_AGENTS:
    __all__.append('Agent8_Keymaker_ResourceReconstruction')
if 8 in MATRIX_AGENTS:
    __all__.append('CommanderLockeAgent')
if 9 in MATRIX_AGENTS:
    __all__.append('Agent10_TheMachine')
if 10 in MATRIX_AGENTS:
    __all__.append('Agent6_Twins_BinaryDiff')
if 11 in MATRIX_AGENTS:
    __all__.append('Agent11_TheOracle')
if 12 in MATRIX_AGENTS:
    __all__.append('Agent12_Link')
if 13 in MATRIX_AGENTS:
    __all__.append('Agent13_AgentJohnson')
if 14 in MATRIX_AGENTS:
    __all__.append('Agent14_TheCleaner')
if 15 in MATRIX_AGENTS:
    __all__.append('Agent15_Analyst')
if 16 in MATRIX_AGENTS:
    __all__.append('Agent16_AgentBrown')