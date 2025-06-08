"""
Matrix Agent initialization module for open-sourcefy
New 16-agent Matrix-themed architecture
"""

# Import Matrix master agent
from .agent00_deus_ex_machina import Agent00_DeusExMachina

# Matrix Agent Registry (16 agents + 1 master)
# To be implemented in Phase B and C
MATRIX_AGENT_REGISTRY = {
    0: Agent00_DeusExMachina,  # Master Orchestrator - implemented
    # 1: Sentinel - Binary discovery + metadata analysis
    # 2: The Architect - Architecture analysis + error pattern matching  
    # 3: The Merovingian - Basic decompilation + optimization detection
    # 4: Agent Smith - Binary structure analysis + dynamic bridge
    # 5: Neo (Glitch) - Advanced decompilation + Ghidra integration
    # 6: The Twins - Binary diff analysis + comparison engine
    # 7: The Trainman - Advanced assembly analysis
    # 8: The Keymaker - Resource reconstruction
    # 9: Commander Locke - Global reconstruction + AI enhancement
    # 10: The Machine - Compilation orchestration + build systems
    # 11: The Oracle - Final validation and truth verification
    # 12: Link - Cross-reference and linking analysis
    # 13: Agent Johnson - Security analysis and vulnerability detection
    # 14: The Cleaner - Code cleanup and optimization
    # 15: The Analyst - Quality assessment and prediction
    # 16: Agent Brown - Automated testing and verification
}

def get_matrix_agent_class(agent_id: int):
    """Get Matrix agent class by ID"""
    return MATRIX_AGENT_REGISTRY.get(agent_id)

def get_available_matrix_agents():
    """Get list of available Matrix agent IDs"""
    return list(MATRIX_AGENT_REGISTRY.keys())

def create_matrix_agent(agent_id: int):
    """Create Matrix agent instance by ID"""
    agent_class = get_matrix_agent_class(agent_id)
    if agent_class:
        return agent_class()
    else:
        raise ValueError(f"Unknown Matrix agent ID: {agent_id}")

def get_matrix_agent_info():
    """Get information about the Matrix agent system"""
    return {
        'total_agents': 17,  # 16 + 1 master
        'implemented_agents': len([v for v in MATRIX_AGENT_REGISTRY.values() if v is not None]),
        'master_agent': 'Agent 0: Deus Ex Machina',
        'parallel_agents': 16,
        'execution_model': 'master_first_parallel'
    }

__all__ = [
    'MATRIX_AGENT_REGISTRY', 
    'get_matrix_agent_class', 
    'get_available_matrix_agents', 
    'create_matrix_agent',
    'get_matrix_agent_info',
    'Agent00_DeusExMachina'
]