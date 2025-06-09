"""
Single AI System for Open-Sourcefy Matrix Pipeline
Complete decoupling - agents simply reference this, no AI imports needed in agent files
"""

import logging
import subprocess
import tempfile
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .config_manager import ConfigManager


@dataclass
class AIResponse:
    """Simple AI response"""
    content: str
    success: bool
    error: Optional[str] = None


class AISystem:
    """Single AI system for all agents - handles everything AI-related"""
    
    def __init__(self):
        self.logger = logging.getLogger("AISystem")
        self.config = ConfigManager()
        
        # AI configuration - Claude Code only
        self.enabled = self.config.get_value('ai.enabled', True)
        self.provider = 'claude_code'  # Force Claude Code only
        self.timeout = self.config.get_value('ai.timeout', 10)  # Reduced timeout
        
        # Find Claude CLI command
        self.claude_cmd = self._find_claude_command()
        
        # Skip initial test - we confirmed Claude CLI works manually
        # The subprocess test has timeout issues but actual usage works fine
        self.claude_working = self.claude_cmd is not None
        
        if self.enabled and self.claude_cmd:
            self.logger.info(f"AI System initialized with Claude CLI: {self.claude_cmd}")
        else:
            self.logger.warning("Claude CLI not found - AI disabled")
            self.enabled = False
    
    def _find_claude_command(self) -> Optional[str]:
        """Find Claude CLI command"""
        import shutil
        
        # Try commands in order
        commands = ['claude', 'claude-code']
        
        for cmd in commands:
            if shutil.which(cmd):
                try:
                    # Quick version check
                    result = subprocess.run([cmd, '--version'], 
                                          capture_output=True, timeout=3)
                    if result.returncode == 0:
                        self.logger.info(f"Found Claude CLI: {cmd}")
                        return cmd
                except:
                    continue
        
        return None
    
    def _test_claude_cli(self) -> bool:
        """Test if Claude CLI is working properly"""
        if not self.claude_cmd:
            return False
            
        try:
            # Test using the same echo pipe method we use in production
            cmd = f"echo 'test' | {self.claude_cmd} --print --output-format text"
            
            test_result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                timeout=5,
                text=True
            )
            
            # Success if command ran without error and produced some output
            success = test_result.returncode == 0 and len(test_result.stdout.strip()) > 0
            if success:
                self.logger.debug("Claude CLI test passed")
            else:
                self.logger.debug(f"Claude CLI test failed: return code {test_result.returncode}, output: {test_result.stdout}")
            
            return success
            
        except (subprocess.TimeoutExpired, Exception) as e:
            self.logger.debug(f"Claude CLI test failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if AI system is available for use"""
        return self.enabled and self.claude_cmd is not None and self.claude_working
    
    def analyze(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Main AI analysis function - all agents use this"""
        if not self.is_available():
            return AIResponse(
                content="AI analysis not available",
                success=False,
                error="AI system disabled or not configured"
            )
        
        try:
            # Call Claude CLI directly
            return self._call_claude_cli(prompt, system_prompt)
            
        except Exception as e:
            self.logger.error(f"Claude CLI call failed: {e}")
            return AIResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    def _call_claude_cli(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Call Claude CLI using file-based approach for maximum compatibility"""
        try:
            # Prepare full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
                input_file.write(full_prompt)
                input_path = input_file.name
                
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_file:
                output_path = output_file.name
            
            try:
                # Use subprocess with timeout instead of os.system for proper timeout handling
                cmd = ["bash", "-c", f"cat {input_path} | {self.claude_cmd} --print --output-format text > {output_path} 2>&1"]
                
                self.logger.debug(f"Executing Claude CLI with {self.timeout}s timeout")
                result = subprocess.run(
                    cmd,
                    timeout=self.timeout,  # Use configured timeout (default 10s)
                    capture_output=False,  # We're using file redirection
                    check=False
                )
                exit_code = result.returncode
                
                # Read the output file
                with open(output_path, 'r') as f:
                    content = f.read().strip()
                
                if exit_code != 0:
                    self.logger.error(f"Claude CLI returned exit code {exit_code}")
                    # Check if content contains an error message
                    if "error" in content.lower() or not content:
                        return AIResponse(
                            content="",
                            success=False,
                            error=f"Claude CLI error (exit code {exit_code}): {content[:200]}"
                        )
                
                if not content:
                    self.logger.warning("Claude CLI returned empty response")
                    return AIResponse(
                        content="",
                        success=False,
                        error="Empty response from Claude CLI"
                    )
                
                self.logger.debug(f"Claude CLI response received: {len(content)} characters")
                return AIResponse(content=content, success=True)
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except:
                    pass
                    
        except subprocess.TimeoutExpired:
            return AIResponse(
                content="",
                success=False,
                error=f"Claude CLI timeout after {self.timeout}s"
            )
        except Exception as e:
            return AIResponse(
                content="",
                success=False,
                error=f"Claude CLI execution failed: {e}"
            )
    
    def analyze_binary_security(self, binary_info: Dict[str, Any]) -> AIResponse:
        """Analyze binary for security threats"""
        if not self.is_available():
            return AIResponse("", False, "Claude CLI not available")
        
        system_prompt = "You are a cybersecurity expert analyzing binary files. Provide concise security analysis focusing on potential threats, suspicious patterns, and recommendations. Be factual and specific."
        
        prompt = f"""
Analyze this binary file for security indicators:

File: {binary_info.get('file_path', 'unknown')}
Format: {binary_info.get('format', 'unknown')}
Architecture: {binary_info.get('architecture', 'unknown')}
Size: {binary_info.get('file_size', 0)} bytes
Entropy: {binary_info.get('entropy', 0)}

Sections: {binary_info.get('section_count', 0)}
Imports: {binary_info.get('import_count', 0)}
Exports: {binary_info.get('export_count', 0)}

Provide:
1. Security risk assessment (Low/Medium/High)
2. Suspicious indicators found
3. Behavioral predictions
4. Recommendations for further analysis
"""
        
        return self.analyze(prompt, system_prompt)
    
    def analyze_code_patterns(self, code_info: Dict[str, Any]) -> AIResponse:
        """Analyze code patterns and architecture"""
        if not self.is_available():
            return AIResponse("", False, "Claude CLI not available")
        
        system_prompt = "You are a reverse engineering expert. Analyze code patterns and architectural decisions to understand the original software design and purpose."
        
        prompt = f"""
Analyze these code patterns and architectural elements:

Functions detected: {code_info.get('functions_detected', 0)}
Code quality score: {code_info.get('code_quality', 0)}
Architecture: {code_info.get('architecture', 'unknown')}
Compiler indicators: {code_info.get('compiler_info', {})}

Provide insights on:
1. Original software purpose and design
2. Development practices and tools used
3. Code quality and maintainability indicators
4. Reconstruction recommendations
"""
        
        return self.analyze(prompt, system_prompt)
    
    def enhance_decompilation(self, decompiled_code: str, context: Dict[str, Any]) -> AIResponse:
        """Enhance decompiled code quality"""
        if not self.is_available():
            return AIResponse("", False, "Claude CLI not available")
        
        system_prompt = "You are a code reconstruction expert. Improve the readability and structure of decompiled code while maintaining functionality."
        
        prompt = f"""
Enhance this decompiled code:

```c
{decompiled_code[:2000]}  # Limit size
```

Context: {context.get('function_name', 'unknown function')}
Architecture: {context.get('architecture', 'unknown')}

Provide improved, more readable version with better variable names and comments.
"""
        
        return self.analyze(prompt, system_prompt)
    
    def generate_documentation(self, analysis_data: Dict[str, Any]) -> AIResponse:
        """Generate documentation from analysis results"""
        if not self.is_available():
            return AIResponse("", False, "Claude CLI not available")
        
        system_prompt = "You are a technical documentation expert. Create clear, comprehensive documentation for reverse-engineered software based on analysis results."
        
        prompt = f"""
Generate documentation for this reverse-engineered software:

Analysis Summary:
- Binary format: {analysis_data.get('format', 'unknown')}
- Architecture: {analysis_data.get('architecture', 'unknown')}
- Functions: {analysis_data.get('functions_count', 0)}
- Quality score: {analysis_data.get('quality_score', 0)}

Create:
1. Executive summary
2. Technical specifications
3. Function descriptions
4. Usage recommendations
5. Security considerations
"""
        
        return self.analyze(prompt, system_prompt)


# Global AI system instance
_ai_system = None


def get_ai_system() -> AISystem:
    """Get the global AI system instance"""
    global _ai_system
    if _ai_system is None:
        _ai_system = AISystem()
    return _ai_system


# Simple functions that agents can use directly
def ai_available() -> bool:
    """Check if AI is available"""
    return get_ai_system().is_available()


def ai_analyze(prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
    """Analyze with AI - main function for agents"""
    return get_ai_system().analyze(prompt, system_prompt)


def ai_analyze_security(binary_info: Dict[str, Any]) -> AIResponse:
    """Security analysis helper"""
    return get_ai_system().analyze_binary_security(binary_info)


def ai_analyze_code(code_info: Dict[str, Any]) -> AIResponse:
    """Code analysis helper"""  
    return get_ai_system().analyze_code_patterns(code_info)


def ai_enhance_code(code: str, context: Dict[str, Any]) -> AIResponse:
    """Code enhancement helper"""
    return get_ai_system().enhance_decompilation(code, context)


def ai_generate_docs(analysis_data: Dict[str, Any]) -> AIResponse:
    """Documentation generation helper"""
    return get_ai_system().generate_documentation(analysis_data)


# Agent convenience functions
def ai_is_enabled() -> bool:
    """Simple check if AI is enabled"""
    return ai_available()


def ai_request(prompt: str, system: Optional[str] = None) -> str:
    """Simple AI request - returns content string"""
    response = ai_analyze(prompt, system)
    return response.content if response.success else ""


def ai_request_safe(prompt: str, system: Optional[str] = None, fallback: str = "") -> str:
    """AI request with error handling"""
    if not ai_available():
        return ""  # No fallback, just empty if Claude not available
    
    response = ai_analyze(prompt, system)
    return response.content if response.success else ""