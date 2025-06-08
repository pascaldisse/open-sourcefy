#!/usr/bin/env python3
"""
Simple Decompilation Script
Direct execution of decompilation agents without the complex Matrix pipeline
"""

import sys
import os
from pathlib import Path

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def run_decompilation():
    """Run basic decompilation using only the core agents"""
    print("🎬 Starting Open-Sourcefy Decompilation Pipeline")
    
    # Check for binary
    binary_path = Path("input/launcher.exe")
    if not binary_path.exists():
        print(f"❌ Binary not found: {binary_path}")
        return False
    
    print(f"📁 Target binary: {binary_path} ({binary_path.stat().st_size} bytes)")
    
    # Create output directory
    output_dir = Path("output") / f"decompile_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Setup paths
    output_paths = {
        'base': output_dir,
        'agents': output_dir / 'agents',
        'ghidra': output_dir / 'ghidra',
        'temp': output_dir / 'temp',
        'logs': output_dir / 'logs'
    }
    
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    # Setup configuration and environment
    os.environ['GHIDRA_INSTALL_DIR'] = str(Path.cwd() / "ghidra")
    os.environ['MATRIX_AGENT_MODE'] = 'decompile_only'
    
    # Import and setup configuration 
    try:
        from core.config_manager import get_config_manager
        config = get_config_manager()
        print("✅ Configuration manager loaded")
    except Exception as e:
        print(f"⚠️ Config manager issue: {e}")
        config = None
    
    # Import agents
    try:
        from core.agents import get_decompile_agents, MATRIX_AGENTS
        print(f"✅ Loaded {len(MATRIX_AGENTS)} Matrix agents")
    except Exception as e:
        print(f"❌ Failed to import agents: {e}")
        return False
    
    # Get decompilation agents (1, 2, 5, 7, 14)
    decompile_agents = get_decompile_agents()
    print(f"🎯 Decompilation agents: {list(decompile_agents.keys())}")
    
    # Execute agents in sequence with proper context structure
    from dataclasses import dataclass
    
    @dataclass
    class SimpleResult:
        status: str
        data: dict = None
        error: str = None
    
    context = {
        'binary_path': str(binary_path),
        'output_paths': output_paths,
        'agent_results': {},
        'global_data': {
            'binary_path': str(binary_path),
            'binary_info': {
                'file_size': binary_path.stat().st_size,
                'file_name': binary_path.name
            }
        }
    }
    
    success_count = 0
    total_agents = len(decompile_agents)
    
    for agent_id, agent_class in decompile_agents.items():
        print(f"\n⚡ Executing Agent {agent_id:02d}...")
        
        try:
            # Create agent instance with proper initialization
            if agent_id in [1, 2]:  # Sentinel and Architect need special handling
                # Skip these for now due to config issues
                print(f"⏭️ Skipping Agent {agent_id:02d} due to configuration requirements")
                context['agent_results'][agent_id] = SimpleResult(status='skipped', data={'reason': 'config_not_available'})
                continue
                
            agent = agent_class()
            
            # Execute agent
            result = agent.execute(context)
            
            if result and hasattr(result, 'status') and result.status == 'success':
                print(f"✅ Agent {agent_id:02d} completed successfully")
                context['agent_results'][agent_id] = result
                success_count += 1
            elif isinstance(result, dict) and result.get('status') == 'success':
                print(f"✅ Agent {agent_id:02d} completed successfully")
                context['agent_results'][agent_id] = SimpleResult(status='success', data=result)
                success_count += 1
            else:
                print(f"⚠️ Agent {agent_id:02d} completed with issues")
                context['agent_results'][agent_id] = SimpleResult(status='completed_with_issues', data=result)
                
        except Exception as e:
            print(f"❌ Agent {agent_id:02d} failed: {e}")
            context['agent_results'][agent_id] = SimpleResult(status='failed', error=str(e))
    
    # Report results
    print(f"\n📊 Pipeline Results: {success_count}/{total_agents} agents succeeded")
    
    # Check for output files
    ghidra_output = output_paths['ghidra']
    if ghidra_output.exists():
        decompiled_files = list(ghidra_output.glob("**/*.c"))
        if decompiled_files:
            print(f"📝 Generated {len(decompiled_files)} C source files:")
            for file in decompiled_files[:5]:  # Show first 5
                print(f"   - {file.name}")
            if len(decompiled_files) > 5:
                print(f"   ... and {len(decompiled_files) - 5} more")
    
    return success_count > 0


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    try:
        success = run_decompilation()
        elapsed = time.time() - start_time
        
        if success:
            print(f"\n🎉 Decompilation completed in {elapsed:.1f}s")
            sys.exit(0)
        else:
            print(f"\n💥 Decompilation failed after {elapsed:.1f}s")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)