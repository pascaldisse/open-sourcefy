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
    from .agent05_neo_advanced_decompiler import Agent5_Neo_AdvancedDecompiler as NeoAgent
    MATRIX_AGENTS[5] = NeoAgent
except ImportError as e:
    failed_imports.append(f"Agent 5 (Neo): {e}")

# Agent 9: Commander Locke  
try:
    from .agent09_commander_locke import CommanderLockeAgent
    MATRIX_AGENTS[9] = CommanderLockeAgent
except ImportError as e:
    failed_imports.append(f"Agent 9 (Commander Locke): {e}")

# Agent 12: Link
try:
    from .agent12_link import Agent12_Link
    MATRIX_AGENTS[12] = Agent12_Link
except ImportError as e:
    failed_imports.append(f"Agent 12 (Link): {e}")

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

if 9 in MATRIX_AGENTS:
    AGENT_METADATA[9] = {
        'name': 'CommanderLocke',
        'character': 'commander_locke',
        'description': 'Global reconstruction orchestration',
        'dependencies': [1, 2, 3, 4, 5]
    }

if 12 in MATRIX_AGENTS:
    AGENT_METADATA[12] = {
        'name': 'Link',
        'character': 'link',
        'description': 'Cross-reference and linking analysis',
        'dependencies': [1, 2]
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
    return {agent_id: agent_id in MATRIX_AGENTS for agent_id in range(1, 17)}


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
    # Based on CLAUDE.md: --decompile-only uses agents 1,2,5,7,14
    decompile_agent_ids = [1, 2, 5, 7, 14]
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
if 9 in MATRIX_AGENTS:
    __all__.append('CommanderLockeAgent')
if 12 in MATRIX_AGENTS:
    __all__.append('Agent12_Link')