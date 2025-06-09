"""
Centralized AI Setup for Matrix Pipeline
Single configuration point for all AI integrations - easy to modify and maintain
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

from .config_manager import ConfigManager


class AIProvider(Enum):
    """Available AI providers"""
    CLAUDE_CODE = "claude_code"  # Claude Code CLI using Max subscription
    ANTHROPIC = "anthropic"      # Direct API access
    OPENAI = "openai" 
    LOCAL_LLM = "local_llm"
    DISABLED = "disabled"


@dataclass
class AIConfig:
    """AI configuration settings - all configurable, no hardcoded values"""
    provider: AIProvider
    model: str
    api_key_env: str
    temperature: float
    max_tokens: int
    timeout: int
    enabled: bool
    base_url: Optional[str] = None
    system_prompt_template: Optional[str] = None


class AISetup:
    """Centralized AI setup and configuration management"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger("AISetup")
        self.ai_config = self._load_ai_config()
        self.ai_interface = None
        
        # Initialize AI interface - strict mode only
        self._initialize_ai_interface()
    
    def _load_ai_config(self) -> AIConfig:
        """Load AI configuration from config manager"""
        # Get provider - strict mode only
        provider_str = self.config.get_value('ai.provider', 'claude_code')
        provider = AIProvider(provider_str)
        
        # Load all configuration from config manager
        return AIConfig(
            provider=provider,
            model=self.config.get_value('ai.model', self._get_default_model(provider)),
            api_key_env=self.config.get_value('ai.api_key_env', self._get_default_api_key_env(provider)),
            temperature=self.config.get_value('ai.temperature', 0.1),
            max_tokens=self.config.get_value('ai.max_tokens', 4096),
            timeout=self.config.get_value('ai.timeout', 30),
            enabled=self.config.get_value('ai.enabled', True) and provider != AIProvider.DISABLED,
            base_url=self.config.get_value('ai.base_url', None),
            system_prompt_template=self.config.get_value('ai.system_prompt_template', None)
        )
    
    def _get_default_model(self, provider: AIProvider) -> str:
        """Get default model for provider"""
        defaults = {
            AIProvider.CLAUDE_CODE: "claude-3-5-sonnet",  # Claude Code uses subscription model
            AIProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            AIProvider.OPENAI: "gpt-4",
            AIProvider.LOCAL_LLM: "llama-2-7b-chat",
            AIProvider.DISABLED: "none"
        }
        return defaults.get(provider, "none")
    
    def _get_default_api_key_env(self, provider: AIProvider) -> str:
        """Get default API key environment variable for provider"""
        defaults = {
            AIProvider.CLAUDE_CODE: "NONE",  # Claude Code uses subscription, no API key needed
            AIProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            AIProvider.OPENAI: "OPENAI_API_KEY", 
            AIProvider.LOCAL_LLM: "LOCAL_LLM_API_KEY",
            AIProvider.DISABLED: "NONE"
        }
        return defaults.get(provider, "NONE")
    
    def _initialize_ai_interface(self):
        """Initialize AI interface - strict mode only"""
        if self.ai_config.provider == AIProvider.CLAUDE_CODE:
            self.ai_interface = self._setup_claude_code()
        elif self.ai_config.provider == AIProvider.ANTHROPIC:
            self.ai_interface = self._setup_anthropic()
        elif self.ai_config.provider == AIProvider.OPENAI:
            self.ai_interface = self._setup_openai()
        elif self.ai_config.provider == AIProvider.LOCAL_LLM:
            self.ai_interface = self._setup_local_llm()
        else:
            raise ValueError(f"Unsupported AI provider: {self.ai_config.provider}")
    
    def _setup_claude_code(self):
        """Setup Claude Code CLI interface - WSL/Linux compatible"""
        import subprocess
        import shutil
        import os
        
        # Determine the correct command for the environment
        claude_cmd = self._get_claude_command()
        
        # Test claude command accessibility - strict validation
        result = subprocess.run([claude_cmd, '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI ({claude_cmd}) returned non-zero exit code: {result.returncode}")
        
        self.logger.info(f"Claude CLI initialized using command: {claude_cmd}")
        return ClaudeCodeInterface(self.ai_config, claude_cmd)
    
    def _get_claude_command(self):
        """Get Claude CLI command - strict mode only"""
        import shutil
        
        # Only claude-code is supported - no alternatives
        cmd = 'claude-code'
        if not shutil.which(cmd):
            raise ImportError(
                f"Required Claude CLI '{cmd}' not found. Install with:\n"
                "npm install -g @anthropic-ai/claude-code"
            )
        
        self.logger.info(f"Using required Claude CLI command: {cmd}")
        return cmd
    
    
    def _setup_anthropic(self):
        """Setup Anthropic Claude interface"""
        try:
            import anthropic
            
            # Check for API key
            api_key = os.getenv(self.ai_config.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {self.ai_config.api_key_env}")
            
            client = anthropic.Anthropic(api_key=api_key)
            self.logger.info(f"Anthropic AI initialized with model: {self.ai_config.model}")
            return AnthropicInterface(client, self.ai_config)
            
        except ImportError:
            raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
    
    def _setup_openai(self):
        """Setup OpenAI interface"""
        try:
            import openai
            
            # Check for API key
            api_key = os.getenv(self.ai_config.api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {self.ai_config.api_key_env}")
            
            client = openai.OpenAI(api_key=api_key, base_url=self.ai_config.base_url)
            self.logger.info(f"OpenAI initialized with model: {self.ai_config.model}")
            return OpenAIInterface(client, self.ai_config)
            
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
    
    def _setup_local_llm(self):
        """Setup local LLM interface (LangChain/Ollama/etc)"""
        try:
            from langchain.llms import LlamaCpp
            
            model_path = self.config.get_path('ai.model.path')
            if not model_path or not model_path.exists():
                raise ValueError(f"Local LLM model not found at: {model_path}")
            
            llm = LlamaCpp(
                model_path=str(model_path),
                temperature=self.ai_config.temperature,
                max_tokens=self.ai_config.max_tokens,
                verbose=self.config.get_value('debug.enabled', False)
            )
            
            self.logger.info(f"Local LLM initialized with model: {model_path}")
            return LocalLLMInterface(llm, self.ai_config)
            
        except ImportError:
            raise ImportError("LangChain library not installed. Install with: pip install langchain")
    
    def get_ai_interface(self):
        """Get the initialized AI interface"""
        return self.ai_interface
    
    def is_enabled(self) -> bool:
        """Check if AI is enabled and ready"""
        return self.ai_config.enabled and self.ai_interface is not None
    
    def get_config(self) -> AIConfig:
        """Get AI configuration"""
        return self.ai_config


@dataclass
class AIResponse:
    """Standardized AI response format"""
    content: str
    model: str
    usage: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    provider: Optional[str] = None


class AIInterface:
    """Base AI interface - common methods for all providers"""
    
    def __init__(self, client, config: AIConfig):
        self.client = client
        self.config = config
        self.logger = logging.getLogger(f"AIInterface.{config.provider.value}")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Generate response - must be implemented by subclasses"""
        raise NotImplementedError
    
    def analyze_binary_security(self, binary_info: Dict[str, Any]) -> AIResponse:
        """Analyze binary for security indicators"""
        system_prompt = self._get_security_system_prompt()
        prompt = self._format_security_prompt(binary_info)
        return self.generate_response(prompt, system_prompt)
    
    def analyze_code_patterns(self, code_analysis: Dict[str, Any]) -> AIResponse:
        """Analyze code patterns and architecture"""
        system_prompt = self._get_code_analysis_system_prompt()
        prompt = self._format_code_analysis_prompt(code_analysis)
        return self.generate_response(prompt, system_prompt)
    
    def generate_documentation(self, analysis_results: Dict[str, Any]) -> AIResponse:
        """Generate comprehensive documentation"""
        system_prompt = self._get_documentation_system_prompt()
        prompt = self._format_documentation_prompt(analysis_results)
        return self.generate_response(prompt, system_prompt)
    
    def _get_security_system_prompt(self) -> str:
        """Get system prompt for security analysis"""
        return (self.config.system_prompt_template or 
                "You are a cybersecurity expert analyzing binary files. "
                "Provide concise security analysis focusing on potential threats, "
                "suspicious patterns, and recommendations. Be factual and specific.")
    
    def _get_code_analysis_system_prompt(self) -> str:
        """Get system prompt for code analysis"""
        return ("You are a reverse engineering expert. Analyze code patterns "
                "and architectural decisions to understand the original software design and purpose.")
    
    def _get_documentation_system_prompt(self) -> str:
        """Get system prompt for documentation generation"""
        return ("You are a technical documentation expert. Create clear, "
                "comprehensive documentation for reverse-engineered software based on analysis results.")
    
    def _format_security_prompt(self, binary_info: Dict[str, Any]) -> str:
        """Format prompt for security analysis"""
        return f"""
        Analyze this binary file for security indicators:
        
        File: {binary_info.get('file_path', 'unknown')}
        Format: {binary_info.get('format_type', 'unknown')}
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
    
    def _format_code_analysis_prompt(self, code_analysis: Dict[str, Any]) -> str:
        """Format prompt for code analysis"""
        return f"""
        Analyze these code patterns and architectural elements:
        
        Functions detected: {code_analysis.get('functions_detected', 0)}
        Code quality score: {code_analysis.get('code_quality', 0)}
        Architecture: {code_analysis.get('architecture', 'unknown')}
        Compiler indicators: {code_analysis.get('compiler_info', {})}
        
        Provide insights on:
        1. Original software purpose and design
        2. Development practices and tools used
        3. Code quality and maintainability indicators
        4. Reconstruction recommendations
        """
    
    def _format_documentation_prompt(self, analysis_results: Dict[str, Any]) -> str:
        """Format prompt for documentation generation"""
        return f"""
        Generate documentation for this reverse-engineered software:
        
        Analysis Summary:
        - Binary format: {analysis_results.get('format', 'unknown')}
        - Architecture: {analysis_results.get('architecture', 'unknown')}
        - Functions: {analysis_results.get('functions_count', 0)}
        - Quality score: {analysis_results.get('quality_score', 0)}
        
        Create:
        1. Executive summary
        2. Technical specifications
        3. Function descriptions
        4. Usage recommendations
        5. Security considerations
        """


class ClaudeCodeInterface(AIInterface):
    """Claude Code CLI interface using Max subscription - WSL compatible"""
    
    def __init__(self, config: AIConfig, claude_cmd: str = 'claude-code'):
        super().__init__(None, config)  # No client needed for CLI
        self.claude_cmd = claude_cmd
        self.logger.info(f"Claude CLI interface initialized with command: {claude_cmd}")
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Generate response using Claude Code CLI"""
        import subprocess
        import tempfile
        import json
        
        try:
            # Prepare the full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            # Create temporary file for the prompt
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                tmp_file.write(full_prompt)
                tmp_file_path = tmp_file.name
            
            try:
                # Call Claude CLI with the correct arguments based on command
                if 'claude-code' in self.claude_cmd or 'claude-skip' in self.claude_cmd:
                    # Use claude-code/claude-skip arguments  
                    cmd = [
                        self.claude_cmd, 
                        '--prompt-file', tmp_file_path,
                        '--max-tokens', str(self.config.max_tokens),
                        '--temperature', str(self.config.temperature),
                        '--output-format', 'json'
                    ]
                else:
                    # Use claude command arguments (newer version)
                    cmd = [
                        self.claude_cmd,
                        '--print',
                        '--output-format', 'json',
                        '--model', self.config.model,
                        full_prompt  # Pass prompt directly as argument
                    ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr or f"Claude CLI ({self.claude_cmd}) failed"
                    return AIResponse(
                        content="", model=self.config.model, usage={}, success=False,
                        error=error_msg, provider="claude_code"
                    )
                
                # Parse response
                try:
                    response_data = json.loads(result.stdout)
                    content = response_data.get('content', result.stdout)
                    usage = response_data.get('usage', {})
                except json.JSONDecodeError:
                    # Fallback to plain text if JSON parsing fails
                    content = result.stdout.strip()
                    usage = {}
                
                return AIResponse(
                    content=content,
                    model=self.config.model,
                    usage=usage,
                    success=True,
                    provider="claude_code"
                )
                
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(tmp_file_path)
                except OSError:
                    pass
                
        except subprocess.TimeoutExpired:
            return AIResponse(
                content="", model=self.config.model, usage={}, success=False,
                error=f"Claude CLI ({self.claude_cmd}) timeout", provider="claude_code"
            )
        except Exception as e:
            self.logger.error(f"Claude Code CLI call failed: {e}")
            return AIResponse(
                content="", model=self.config.model, usage={}, success=False,
                error=str(e), provider="claude_code"
            )


class AnthropicInterface(AIInterface):
    """Anthropic Claude API interface"""
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Generate response using Anthropic Claude"""
        try:
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": messages
            }
            
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            
            content = ""
            if response.content:
                content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            
            return AIResponse(
                content=content,
                model=self.config.model,
                usage=response.usage.__dict__ if hasattr(response, 'usage') else {},
                success=True,
                provider="anthropic"
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            return AIResponse(
                content="", model=self.config.model, usage={}, success=False, 
                error=str(e), provider="anthropic"
            )


class OpenAIInterface(AIInterface):
    """OpenAI GPT interface"""
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Generate response using OpenAI GPT"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return AIResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=response.usage.__dict__ if hasattr(response, 'usage') else {},
                success=True,
                provider="openai"
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return AIResponse(
                content="", model=self.config.model, usage={}, success=False,
                error=str(e), provider="openai"
            )


class LocalLLMInterface(AIInterface):
    """Local LLM interface (LangChain/Ollama)"""
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Generate response using local LLM"""
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.client(full_prompt)
            
            return AIResponse(
                content=response,
                model=self.config.model,
                usage={},
                success=True,
                provider="local_llm"
            )
            
        except Exception as e:
            self.logger.error(f"Local LLM call failed: {e}")
            return AIResponse(
                content="", model=self.config.model, usage={}, success=False,
                error=str(e), provider="local_llm"
            )


# Global AI setup instance - initialized on first import
_ai_setup = None


def get_ai_setup(config_manager: Optional[ConfigManager] = None) -> AISetup:
    """Get global AI setup instance"""
    global _ai_setup
    if _ai_setup is None:
        _ai_setup = AISetup(config_manager)
    return _ai_setup


def get_ai_interface():
    """Get AI interface - convenience function"""
    setup = get_ai_setup()
    return setup.get_ai_interface()


def is_ai_enabled() -> bool:
    """Check if AI is enabled and ready"""
    setup = get_ai_setup()
    return setup.is_enabled()


# Backward compatibility functions for existing agent code
def create_ai_tools(ai_interface) -> List:
    """Create AI tools for backward compatibility"""
    if not ai_interface:
        return []
    
    class AITool:
        def __init__(self, name: str, description: str, func):
            self.name = name
            self.description = description
            self.func = func
        
        def run(self, input_text: str) -> str:
            response = self.func(input_text)
            return response.content if hasattr(response, 'content') else str(response)
    
    return [
        AITool("binary_security_analysis", "Analyze binary for security threats", 
               lambda x: ai_interface.analyze_binary_security({})),
        AITool("code_pattern_analysis", "Analyze code patterns and architecture",
               lambda x: ai_interface.analyze_code_patterns({})),
        AITool("documentation_generation", "Generate comprehensive documentation",
               lambda x: ai_interface.generate_documentation({}))
    ]


class AIAgentExecutor:
    """Agent executor for backward compatibility"""
    
    def __init__(self, ai_interface, tools=None):
        self.ai_interface = ai_interface
        self.tools = tools or []
    
    def run(self, prompt: str) -> str:
        """Execute with prompt"""
        if not self.ai_interface:
            return "AI interface not available"
        
        response = self.ai_interface.generate_response(prompt)
        return response.content if response.success else f"Error: {response.error}"