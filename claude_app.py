#!/usr/bin/env python3
"""
Claude Code App - Starts Claude Code with rules.md, tasks.md, and claude.md as input
"""

import asyncio
import os
import sys
from pathlib import Path
from claude_code_sdk import query, ClaudeCodeOptions, CLINotFoundError, ProcessError


async def read_file_content(file_path: Path) -> str:
    """Read file content if it exists, return empty string otherwise."""
    try:
        return file_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"Warning: {file_path} not found")
        return ""
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


async def main():
    """Main application entry point."""
    # Define input files
    current_dir = Path.cwd()
    rules_file = current_dir / "rules.md"
    tasks_file = current_dir / "tasks.md"
    claude_file = current_dir / "claude.md"
    
    print("Claude Code App - Starting...")
    print(f"Working directory: {current_dir}")
    
    # Read input files
    print("Reading input files...")
    rules_content = await read_file_content(rules_file)
    tasks_content = await read_file_content(tasks_file)
    claude_content = await read_file_content(claude_file)
    
    # Build system prompt from files
    system_prompt_parts = []
    
    if rules_content:
        system_prompt_parts.append(f"# Rules\n{rules_content}")
    
    if claude_content:
        system_prompt_parts.append(f"# Claude Configuration\n{claude_content}")
    
    system_prompt = "\n\n".join(system_prompt_parts) if system_prompt_parts else None
    
    # Build initial prompt from tasks
    initial_prompt = tasks_content if tasks_content else "Hello! I'm ready to help with your coding tasks."
    
    # Configure Claude Code options
    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        cwd=str(current_dir),
        permission_mode='acceptEdits'
    )
    
    print("\nStarting Claude Code session...")
    print("=" * 50)
    
    try:
        async for message in query(prompt=initial_prompt, options=options):
            print(message, end='', flush=True)
            
    except CLINotFoundError:
        print("\nError: Claude Code CLI not found.")
        print("Please install it with: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)
    except ProcessError as e:
        print(f"\nError running Claude Code: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())