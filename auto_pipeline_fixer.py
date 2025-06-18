#!/usr/bin/env python3
"""
Automated Pipeline Fixer - Continuous Pipeline Execution with Claude Code SDK
Uses git worktree for isolated Claude Code instances as recommended.
Implements ZERO TOLERANCE rules - does not stop until pipeline is fixed.
"""

import asyncio
import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from claude_code_sdk import query, ClaudeCodeOptions, CLINotFoundError, ProcessError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_pipeline_fixer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoPipelineFixer:
    """
    Automated pipeline fixer with Claude Code SDK integration.
    Implements ZERO TOLERANCE - never stops until success.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir).resolve()
        self.worktree_dir = self.base_dir / "worktrees"
        self.main_pipeline_script = self.base_dir / "main.py"
        self.rules_file = self.base_dir / "rules.md"
        self.claude_file = self.base_dir / "CLAUDE.md"
        self.tasks_file = self.base_dir / "tasks.md"
        
        # Pipeline state tracking
        self.attempt_count = 0
        self.max_attempts = None  # INFINITE ATTEMPTS - ZERO TOLERANCE
        self.current_worktree = None
        self.pipeline_success = False
        
        # Claude Code configuration
        self.claude_options = None
        self._setup_claude_options()
        
    def _setup_claude_options(self):
        """Setup Claude Code SDK options from project files."""
        logger.info("Setting up Claude Code SDK options...")
        
        # Read project configuration files
        rules_content = self._read_file_safe(self.rules_file)
        claude_content = self._read_file_safe(self.claude_file)
        tasks_content = self._read_file_safe(self.tasks_file)
        
        # Build system prompt from rules and claude config
        system_prompt_parts = []
        if rules_content:
            system_prompt_parts.append(f"# ABSOLUTE PROJECT RULES\n{rules_content}")
        if claude_content:
            system_prompt_parts.append(f"# PROJECT CONFIGURATION\n{claude_content}")
            
        system_prompt = "\n\n".join(system_prompt_parts) if system_prompt_parts else None
        
        # Configure Claude Code with ZERO TOLERANCE settings
        self.claude_options = ClaudeCodeOptions(
            system_prompt=system_prompt,
            permission_mode='acceptEdits',
            max_turns=None,  # No limit - continue until fixed
            cwd=str(self.base_dir)
        )
        
        logger.info("Claude Code SDK configured with ZERO TOLERANCE settings")
    
    def _read_file_safe(self, file_path: Path) -> str:
        """Safely read file content, return empty string if not found."""
        try:
            return file_path.read_text(encoding='utf-8')
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return ""
    
    def _run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                    timeout: Optional[int] = None) -> Tuple[bool, str, str]:
        """Run command and return success, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.base_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def setup_git_worktree(self) -> Path:
        """Setup git worktree for isolated Claude Code instance."""
        self.attempt_count += 1
        worktree_name = f"claude-fix-attempt-{self.attempt_count}-{int(time.time())}"
        worktree_path = self.worktree_dir / worktree_name
        
        logger.info(f"Setting up git worktree: {worktree_name}")
        
        # Ensure worktree directory exists
        self.worktree_dir.mkdir(exist_ok=True)
        
        # Create git worktree
        success, stdout, stderr = self._run_command([
            "git", "worktree", "add", str(worktree_path), "HEAD"
        ])
        
        if not success:
            logger.error(f"Failed to create worktree: {stderr}")
            # ZERO TOLERANCE - try alternative approach
            logger.info("Attempting direct directory copy as fallback...")
            import shutil
            try:
                shutil.copytree(self.base_dir, worktree_path, 
                              ignore=shutil.ignore_patterns('worktrees', '.git', 'venv', 'matrix_venv'))
                logger.info(f"Created worktree copy at {worktree_path}")
            except Exception as e:
                logger.error(f"Failed to create worktree copy: {e}")
                raise
        else:
            logger.info(f"Git worktree created successfully: {worktree_path}")
            
        self.current_worktree = worktree_path
        return worktree_path
    
    def cleanup_worktree(self, worktree_path: Path):
        """Clean up git worktree after use."""
        if not worktree_path.exists():
            return
            
        logger.info(f"Cleaning up worktree: {worktree_path}")
        
        # Remove git worktree
        success, stdout, stderr = self._run_command([
            "git", "worktree", "remove", str(worktree_path), "--force"
        ])
        
        if not success:
            logger.warning(f"Git worktree removal failed: {stderr}")
            # Force cleanup
            import shutil
            try:
                shutil.rmtree(worktree_path)
                logger.info("Force removed worktree directory")
            except Exception as e:
                logger.error(f"Failed to force remove worktree: {e}")
    
    def run_pipeline(self, worktree_path: Path) -> Tuple[bool, str, str]:
        """Run the main pipeline in the specified worktree."""
        logger.info(f"Running pipeline in worktree: {worktree_path}")
        
        pipeline_script = worktree_path / "main.py"
        if not pipeline_script.exists():
            return False, "", f"Pipeline script not found: {pipeline_script}"
        
        # Run pipeline with extended timeout for complex operations
        success, stdout, stderr = self._run_command([
            sys.executable, "main.py"
        ], cwd=worktree_path, timeout=3600)  # 1 hour timeout
        
        return success, stdout, stderr
    
    async def fix_pipeline_with_claude(self, worktree_path: Path, 
                                     error_output: str) -> bool:
        """Use Claude Code SDK to fix pipeline issues."""
        logger.info("Invoking Claude Code SDK to fix pipeline issues...")
        
        # Build detailed prompt for Claude
        fix_prompt = f"""
CRITICAL PIPELINE FAILURE - ZERO TOLERANCE FIXING REQUIRED

The open-sourcefy binary decompilation pipeline has failed. You must fix ALL issues with absolute compliance to rules.md.

## FAILURE DETAILS:
{error_output}

## CURRENT ATTEMPT: {self.attempt_count}

## MANDATORY REQUIREMENTS:
1. Read rules.md - ALL RULES ARE ABSOLUTE AND NON-NEGOTIABLE
2. Analyze the pipeline failure in detail
3. Fix Agent 1 (Sentinel) import table reconstruction if needed
4. Fix Agent 9 (The Machine) data flow issues if needed  
5. Resolve any build system or compilation issues
6. Ensure VS2022 Preview compatibility
7. Fix MFC 7.1 compatibility issues
8. Test the fixes thoroughly
9. Ensure the pipeline achieves 5.27MB binary reconstruction

## ZERO TOLERANCE POLICY:
- NO FALLBACKS - fix the real issue
- NO MOCK IMPLEMENTATIONS - real solutions only
- NO PARTIAL SUCCESS - complete fix required
- FAIL FAST on missing tools/dependencies
- NSA-level security standards mandatory

## SUCCESS CRITERIA:
- Pipeline completes successfully
- Binary reconstruction achieves target size
- All tests pass
- Zero tolerance compliance maintained

Begin fixing immediately. Do not stop until the pipeline is completely functional.
"""

        try:
            # Update Claude options with current worktree
            worktree_options = ClaudeCodeOptions(
                system_prompt=self.claude_options.system_prompt,
                permission_mode='acceptEdits',
                max_turns=None,
                cwd=str(worktree_path)
            )
            
            # Stream Claude Code execution
            logger.info("Starting Claude Code fixing session...")
            async for message in query(prompt=fix_prompt, options=worktree_options):
                # Log Claude's progress (truncated for readability)
                if hasattr(message, 'content') and message.content:
                    content_preview = str(message.content)[:200] + "..." if len(str(message.content)) > 200 else str(message.content)
                    logger.info(f"Claude: {content_preview}")
            
            logger.info("Claude Code fixing session completed")
            return True
            
        except CLINotFoundError:
            logger.error("Claude Code CLI not found - cannot proceed with fixing")
            return False
        except ProcessError as e:
            logger.error(f"Claude Code process error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during Claude fixing: {e}")
            return False
    
    def copy_fixes_to_main(self, worktree_path: Path):
        """Copy successful fixes from worktree back to main directory."""
        logger.info("Copying successful fixes back to main directory...")
        
        # Key files/directories to sync back
        sync_paths = [
            "src/",
            "main.py",
            "build_config.yaml",
            "config.yaml"
        ]
        
        for path_str in sync_paths:
            src_path = worktree_path / path_str
            dest_path = self.base_dir / path_str
            
            if src_path.exists():
                try:
                    if src_path.is_file():
                        import shutil
                        shutil.copy2(src_path, dest_path)
                        logger.info(f"Copied file: {path_str}")
                    elif src_path.is_dir():
                        import shutil
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(src_path, dest_path)
                        logger.info(f"Copied directory: {path_str}")
                except Exception as e:
                    logger.error(f"Failed to copy {path_str}: {e}")
    
    async def run_continuous_pipeline(self):
        """
        Main execution loop - NEVER STOPS until pipeline is fixed.
        Implements ZERO TOLERANCE policy.
        """
        logger.info("=" * 80)
        logger.info("STARTING AUTOMATED PIPELINE FIXER - ZERO TOLERANCE MODE")
        logger.info("WILL NOT STOP UNTIL PIPELINE IS COMPLETELY FUNCTIONAL")
        logger.info("=" * 80)
        
        while not self.pipeline_success:
            try:
                # Setup isolated worktree for this attempt
                worktree_path = self.setup_git_worktree()
                
                logger.info(f"ATTEMPT #{self.attempt_count} - Using worktree: {worktree_path}")
                
                # Run pipeline
                success, stdout, stderr = self.run_pipeline(worktree_path)
                
                if success:
                    logger.info("🎉 PIPELINE SUCCESS! Copying fixes to main directory...")
                    self.copy_fixes_to_main(worktree_path)
                    self.pipeline_success = True
                    logger.info("✅ AUTOMATED PIPELINE FIXING COMPLETED SUCCESSFULLY!")
                    break
                else:
                    logger.error(f"❌ PIPELINE FAILED - Attempt #{self.attempt_count}")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    
                    # Use Claude Code SDK to fix the issues
                    error_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
                    fixing_success = await self.fix_pipeline_with_claude(worktree_path, error_output)
                    
                    if fixing_success:
                        logger.info("Claude Code fixing completed - retesting pipeline...")
                        # Re-test the pipeline after Claude's fixes
                        success, stdout, stderr = self.run_pipeline(worktree_path)
                        
                        if success:
                            logger.info("🎉 PIPELINE FIXED BY CLAUDE! Copying fixes...")
                            self.copy_fixes_to_main(worktree_path)
                            self.pipeline_success = True
                            logger.info("✅ CLAUDE SUCCESSFULLY FIXED THE PIPELINE!")
                            break
                        else:
                            logger.warning("Pipeline still failing after Claude fixes - continuing...")
                    else:
                        logger.error("Claude Code fixing failed - will retry with new worktree")
                
                # Cleanup current worktree
                self.cleanup_worktree(worktree_path)
                
                # Brief pause before next attempt
                logger.info(f"Waiting 30 seconds before attempt #{self.attempt_count + 1}...")
                await asyncio.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("User interrupted - cleaning up...")
                if self.current_worktree:
                    self.cleanup_worktree(self.current_worktree)
                break
            except Exception as e:
                logger.error(f"Unexpected error in attempt #{self.attempt_count}: {e}")
                if self.current_worktree:
                    self.cleanup_worktree(self.current_worktree)
                
                # ZERO TOLERANCE - continue despite errors
                logger.info("Continuing despite error - ZERO TOLERANCE POLICY")
                await asyncio.sleep(60)  # Longer pause for unexpected errors
        
        logger.info("=" * 80)
        if self.pipeline_success:
            logger.info("🎉 AUTOMATED PIPELINE FIXING COMPLETED SUCCESSFULLY!")
            logger.info(f"Total attempts required: {self.attempt_count}")
        else:
            logger.info("Pipeline fixing interrupted by user")
        logger.info("=" * 80)


async def main():
    """Main entry point for automated pipeline fixer."""
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path.cwd()
    
    logger.info(f"Starting automated pipeline fixer for: {base_dir}")
    
    # Validate base directory
    if not base_dir.exists():
        logger.error(f"Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Check for required files
    required_files = ["main.py", "rules.md", "CLAUDE.md"]
    missing_files = [f for f in required_files if not (base_dir / f).exists()]
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        sys.exit(1)
    
    # Initialize and run the automated fixer
    fixer = AutoPipelineFixer(base_dir)
    await fixer.run_continuous_pipeline()


if __name__ == "__main__":
    asyncio.run(main())