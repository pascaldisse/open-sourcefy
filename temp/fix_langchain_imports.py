#!/usr/bin/env python3
"""
Quick script to fix LangChain import issues across agents
"""
import os
from pathlib import Path

# List of agent files that need fixing
agent_files = [
    "src/core/agents/agent01_sentinel.py",
    "src/core/agents/agent05_neo_advanced_decompiler.py", 
    "src/core/agents/agent06_twins_binary_diff.py",
    "src/core/agents/agent07_trainman_assembly_analysis.py",
    "src/core/agents/agent08_keymaker_resource_reconstruction.py",
    "src/core/agents/agent15_analyst.py",
    "src/core/agents/agent16_agent_brown.py"
]

langchain_import_block = '''
# LangChain imports (conditional)
try:
    from langchain.agents import AgentExecutor, ReActDocstoreAgent
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create dummy classes for type hints when LangChain not available
    class AgentExecutor:
        pass
    class ConversationBufferMemory:
        pass
    class ReActDocstoreAgent:
        pass
'''

def fix_agent_file(file_path):
    """Fix LangChain imports in an agent file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if already fixed
        if 'LANGCHAIN_AVAILABLE' in content:
            print(f"✓ {file_path} already fixed")
            return True
            
        # Find AI system import line
        ai_import_patterns = [
            "from ..ai_system import",
            "from ..ai_engine_interface import",
            "from ..anthropic_ai_interface import"
        ]
        
        for pattern in ai_import_patterns:
            if pattern in content:
                # Insert LangChain imports after AI imports
                content = content.replace(pattern, pattern + langchain_import_block, 1)
                break
        else:
            print(f"✗ Could not find AI import in {file_path}")
            return False
        
        # Update _setup_langchain_agent methods
        if '_setup_langchain_agent' in content:
            # Add LANGCHAIN_AVAILABLE check
            content = content.replace(
                'if not self.ai_enabled or not self.llm:',
                'if not self.ai_enabled or not self.llm or not LANGCHAIN_AVAILABLE:'
            )
            content = content.replace(
                'if False:  # Disabled',
                'if not LANGCHAIN_AVAILABLE:  # Disabled when not available'
            )
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(content)
            
        print(f"✓ Fixed {file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix all agent files"""
    fixed_count = 0
    
    for agent_file in agent_files:
        if os.path.exists(agent_file):
            if fix_agent_file(agent_file):
                fixed_count += 1
        else:
            print(f"✗ File not found: {agent_file}")
    
    print(f"\nFixed {fixed_count}/{len(agent_files)} agent files")

if __name__ == "__main__":
    main()