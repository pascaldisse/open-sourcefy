#!/usr/bin/env python3
"""
Matrix Pipeline Helper Script

This script provides utilities for Matrix pipeline development, testing, and debugging.
It helps developers quickly test individual agents, validate the pipeline, and generate reports.

Usage:
    python prompts/matrix_pipeline_helper.py --help
    python prompts/matrix_pipeline_helper.py --test-agent 1
    python prompts/matrix_pipeline_helper.py --validate-env
    python prompts/matrix_pipeline_helper.py --dry-run
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_single_agent(agent_id: int):
    """Test a single agent with the Matrix launcher binary."""
    print(f"🔧 Testing Agent {agent_id}...")
    
    # Import here to avoid circular imports
    try:
        from core.config_manager import ConfigManager
        from core.matrix_execution_context import MatrixExecutionContext
        from core.agent_base import get_agent_by_id
        
        config = ConfigManager()
        context = MatrixExecutionContext(
            binary_path=project_root / "input" / "launcher.exe",
            output_dir=project_root / "output" / "test_agent",
            config=config
        )
        
        agent = get_agent_by_id(agent_id, config)
        if agent:
            print(f"✅ Agent {agent_id} ({agent.__class__.__name__}) loaded successfully")
            result = agent.execute(context)
            print(f"📊 Result: {result.status} - {result.message}")
        else:
            print(f"❌ Agent {agent_id} not found")
            
    except Exception as e:
        print(f"❌ Error testing agent {agent_id}: {e}")

def validate_environment():
    """Validate the development environment."""
    print("🔍 Validating Matrix environment...")
    
    # Check required directories
    required_dirs = ['input', 'output', 'src', 'ghidra', 'tests', 'docs', 'temp', 'prompts']
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
    
    # Check for launcher binary
    launcher_path = project_root / "input" / "launcher.exe"
    if launcher_path.exists():
        print(f"✅ Matrix launcher binary found ({launcher_path.stat().st_size} bytes)")
    else:
        print("❌ Matrix launcher binary not found in input/")
    
    # Check Python dependencies
    try:
        import pefile
        import yaml
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
    
    # Check Ghidra installation
    ghidra_path = project_root / "ghidra"
    if (ghidra_path / "ghidraRun").exists() or (ghidra_path / "ghidraRun.bat").exists():
        print("✅ Ghidra installation found")
    else:
        print("❌ Ghidra installation not found")

def show_dry_run():
    """Show what a pipeline execution would do without actually running it."""
    print("🎭 Matrix Pipeline Dry Run")
    print("=" * 50)
    
    # Show agent execution order
    agent_batches = [
        [0],  # Master
        [1],  # Foundation
        [2, 3, 4],  # Core Analysis
        [5, 6, 7, 8],  # Advanced Analysis
        [9, 12, 13],  # Reconstruction
        [10],  # Sequential 1
        [11],  # Sequential 2
        [14, 15, 16]  # Final Validation
    ]
    
    for i, batch in enumerate(agent_batches):
        if i == 0:
            print(f"Master: Agent {batch[0]} (Deus Ex Machina)")
        elif len(batch) == 1:
            print(f"Sequential: Agent {batch[0]}")
        else:
            print(f"Batch {i}: Agents {', '.join(map(str, batch))} (parallel)")
    
    print("\n📂 Output structure:")
    print("output/[timestamp]/")
    print("├── agents/     # Agent-specific results")
    print("├── ghidra/     # Ghidra decompilation")
    print("├── compilation/# MSBuild artifacts")
    print("├── reports/    # Pipeline reports")
    print("├── logs/       # Execution logs")
    print("├── temp/       # Temporary files")
    print("└── tests/      # Generated tests")

def generate_agent_status():
    """Generate a status report of all Matrix agents."""
    print("🕵️ Matrix Agent Status Report")
    print("=" * 50)
    
    # Agent definitions with Matrix themes
    agents = {
        0: "Deus Ex Machina (Master Orchestrator)",
        1: "Sentinel (Binary Discovery)",
        2: "The Architect (Architecture Analysis)",
        3: "The Merovingian (Basic Decompilation)",
        4: "Agent Smith (Binary Structure Analysis)",
        5: "Neo (Advanced Decompilation)",
        6: "The Twins (Binary Differential Analysis)",
        7: "The Trainman (Advanced Assembly Analysis)",
        8: "The Keymaker (Resource Reconstruction)",
        9: "Commander Locke (Global Reconstruction)",
        10: "Link (Cross-reference Analysis)",
        11: "The Oracle (Validation)",
        12: "The Machine (Compilation)",
        13: "Agent Johnson (Security Analysis)",
        14: "The Cleaner (Code Cleanup)",
        15: "The Analyst (Metadata Analysis)",
        16: "Agent Brown (Quality Assurance)"
    }
    
    implemented_agents = [0, 1, 2, 3, 4]  # Currently implemented
    
    for agent_id, description in agents.items():
        status = "✅ Implemented" if agent_id in implemented_agents else "🚧 Planned"
        print(f"Agent {agent_id:2d}: {description:<40} {status}")

def main():
    parser = argparse.ArgumentParser(
        description="Matrix Pipeline Helper - Development utilities for the Open-Sourcefy Matrix system"
    )
    
    parser.add_argument("--test-agent", type=int, metavar="ID",
                       help="Test a specific agent by ID (0-16)")
    parser.add_argument("--validate-env", action="store_true",
                       help="Validate the development environment")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show pipeline execution plan without running")
    parser.add_argument("--agent-status", action="store_true",
                       help="Show Matrix agent implementation status")
    
    args = parser.parse_args()
    
    if args.test_agent is not None:
        test_single_agent(args.test_agent)
    elif args.validate_env:
        validate_environment()
    elif args.dry_run:
        show_dry_run()
    elif args.agent_status:
        generate_agent_status()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()