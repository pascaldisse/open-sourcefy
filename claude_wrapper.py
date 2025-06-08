#!/usr/bin/env python3
"""
Simple Claude CLI wrapper for non-interactive usage
"""
import sys
import subprocess
import json
import tempfile
import os

def call_claude_cli(prompt):
    """Call Claude CLI and return response"""
    try:
        # Create a temporary file with the prompt
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            temp_file = f.name
        
        try:
            # Try different command variations
            commands_to_try = [
                ['claude', '--print', '--output-format', 'text', prompt],
                ['claude', '--print', prompt],
                ['claude', prompt]
            ]
            
            for cmd in commands_to_try:
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=15,
                        env=os.environ.copy()
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        return {'success': True, 'content': result.stdout.strip()}
                    elif result.stderr:
                        print(f"Command {cmd[0]} failed: {result.stderr}", file=sys.stderr)
                
                except subprocess.TimeoutExpired:
                    print(f"Command {cmd[0]} timed out", file=sys.stderr)
                except Exception as e:
                    print(f"Command {cmd[0]} error: {e}", file=sys.stderr)
            
            return {'success': False, 'error': 'All Claude CLI commands failed'}
            
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python claude_wrapper.py 'your prompt here'")
        sys.exit(1)
    
    prompt = ' '.join(sys.argv[1:])
    result = call_claude_cli(prompt)
    
    if result['success']:
        print(result['content'])
    else:
        print(f"Error: {result['error']}", file=sys.stderr)
        sys.exit(1)