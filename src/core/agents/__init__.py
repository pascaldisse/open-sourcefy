"""
Matrix Agents - Complete 16-agent Matrix-themed architecture
Contains all production-ready implementations of agents 1-16
"""

from typing import Dict, Type, Any

# Import Matrix agents
try:
    from .agent01_sentinel import SentinelAgent
    from .agent02_architect import ArchitectAgent  
    from .agent03_merovingian import MerovingianAgent
    from .agent04_agent_smith import AgentSmithAgent
    from .agent05_neo_advanced_decompiler import Agent5_Neo_AdvancedDecompiler as NeoAgent
    from .agent06_twins_binary_diff import Agent6_Twins_BinaryDiff as TwinsAgent
    from .agent07_trainman_assembly_analysis import Agent7_Trainman_AssemblyAnalysis as TrainmanAgent
    from .agent08_keymaker_resource_reconstruction import Agent8_Keymaker_ResourceReconstruction as KeymakerAgent
    from .agent09_commander_locke import CommanderLockeAgent
    from .agent10_the_machine import Agent10_TheMachine
    from .agent11_the_oracle import Agent11_TheOracle
    from .agent12_link import Agent12_Link
    from .agent13_agent_johnson import Agent13_AgentJohnson
    from .agent14_the_cleaner import Agent14_TheCleaner
    from .agent15_analyst import Agent15_Analyst
    from .agent16_agent_brown import Agent16_AgentBrown
    
    # Map agent IDs to classes
    MATRIX_AGENTS = {
        1: SentinelAgent,
        2: ArchitectAgent,
        3: MerovingianAgent,
        4: AgentSmithAgent,
        5: NeoAgent,
        6: TwinsAgent,
        7: TrainmanAgent,
        8: KeymakerAgent,
        9: CommanderLockeAgent,
        10: Agent10_TheMachine,
        11: Agent11_TheOracle,
        12: Agent12_Link,
        13: Agent13_AgentJohnson,
        14: Agent14_TheCleaner,
        15: Agent15_Analyst,
        16: Agent16_AgentBrown
    }
    
    # Agent metadata for system integration
    AGENT_METADATA = {
        1: {
            'name': 'Sentinel',
            'character': 'sentinel',
            'description': 'Binary discovery and metadata analysis',
            'dependencies': []
        },
        2: {
            'name': 'Architect',
            'character': 'architect',
            'description': 'Architecture analysis and design patterns',
            'dependencies': [1]
        },
        3: {
            'name': 'Merovingian',
            'character': 'merovingian',
            'description': 'Basic decompilation and control flow',
            'dependencies': [1, 2]
        },
        4: {
            'name': 'AgentSmith',
            'character': 'agent_smith',
            'description': 'Binary structure analysis and replication',
            'dependencies': [1, 2]
        },
        5: {
            'name': 'Neo',
            'character': 'neo',
            'description': 'Advanced decompilation and code generation',
            'dependencies': [2, 3, 4]
        },
        6: {
            'name': 'Twins',
            'character': 'twins',
            'description': 'Binary differential analysis',
            'dependencies': [5]
        },
        7: {
            'name': 'Trainman',
            'character': 'trainman',
            'description': 'Assembly analysis and transportation',
            'dependencies': [5, 6]
        },
        8: {
            'name': 'Keymaker',
            'character': 'keymaker',
            'description': 'Resource reconstruction and access',
            'dependencies': [5, 7]
        },
        9: {
            'name': 'CommanderLocke',
            'character': 'commander_locke',
            'description': 'Global reconstruction orchestration',
            'dependencies': [5, 6, 7, 8]
        },
        10: {
            'name': 'TheMachine',
            'character': 'machine',
            'description': 'Compilation orchestration and build systems',
            'dependencies': [8, 9]
        },
        11: {
            'name': 'TheOracle',
            'character': 'oracle',
            'description': 'Final validation and truth verification',
            'dependencies': [10]
        },
        12: {
            'name': 'Link',
            'character': 'link',
            'description': 'Cross-reference and linking analysis',
            'dependencies': [11]
        },
        13: {
            'name': 'AgentJohnson',
            'character': 'agent_johnson',
            'description': 'Security analysis and vulnerability detection',
            'dependencies': [12]
        },
        14: {
            'name': 'TheCleaner',
            'character': 'cleaner',
            'description': 'Code cleanup and optimization',
            'dependencies': [13]
        },
        15: {
            'name': 'Analyst',
            'character': 'analyst',
            'description': 'Advanced metadata analysis and intelligence synthesis',
            'dependencies': [1, 2, 5, 10, 11, 13, 14]
        },
        16: {
            'name': 'AgentBrown',
            'character': 'agent_brown',
            'description': 'Final quality assurance and optimization',
            'dependencies': [1, 2, 5, 10, 11, 13, 14, 15]
        }
    }
    
except ImportError as e:
    print(f"Warning: Failed to import some Matrix agents: {e}")
    MATRIX_AGENTS = {}
    AGENT_METADATA = {}


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


__all__ = [
    'MATRIX_AGENTS',
    'AGENT_METADATA', 
    'get_available_agents',
    'get_agent_by_id',
    'get_implementation_status',
    'create_all_agents',
    'get_decompile_agents',
    'SentinelAgent',
    'ArchitectAgent', 
    'MerovingianAgent',
    'AgentSmithAgent',
    'NeoAgent',
    'TwinsAgent',
    'TrainmanAgent',
    'KeymakerAgent',
    'CommanderLockeAgent',
    'Agent10_TheMachine',
    'Agent11_TheOracle', 
    'Agent12_Link',
    'Agent13_AgentJohnson',
    'Agent14_TheCleaner',
    'Agent15_Analyst',
    'Agent16_AgentBrown'
]