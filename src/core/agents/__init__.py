"""
Matrix Agents v2 - Enhanced Matrix-themed agent architecture
Contains the production-ready implementations of agents 10-14
"""

from typing import Dict, Type, Any

# Import Matrix agents
try:
    from .agent10_the_machine import Agent10_TheMachine
    from .agent11_the_oracle import Agent11_TheOracle
    from .agent12_link import Agent12_Link
    from .agent13_agent_johnson import Agent13_AgentJohnson
    from .agent14_the_cleaner import Agent14_TheCleaner
    
    # Map agent IDs to classes
    MATRIX_AGENTS = {
        10: Agent10_TheMachine,
        11: Agent11_TheOracle,
        12: Agent12_Link,
        13: Agent13_AgentJohnson,
        14: Agent14_TheCleaner
    }
    
    # Agent metadata for system integration
    AGENT_METADATA = {
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
    return {agent_id: agent_id in MATRIX_AGENTS for agent_id in range(10, 15)}


# Legacy compatibility classes
SentinelAgent = None
ArchitectAgent = None
MerovingianAgent = None
AgentSmithAgent = None

__all__ = [
    'MATRIX_AGENTS',
    'AGENT_METADATA', 
    'get_available_agents',
    'get_agent_by_id',
    'get_implementation_status',
    'Agent10_TheMachine',
    'Agent11_TheOracle', 
    'Agent12_Link',
    'Agent13_AgentJohnson',
    'Agent14_TheCleaner'
]