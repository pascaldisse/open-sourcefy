"""
Anthropic AI Interface for Matrix Pipeline
Provides Claude 3.5 Sonnet integration for AI-enhanced analysis
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    anthropic = None

from .config_manager import ConfigManager


@dataclass
class AnthropicResponse:
    """Response from Anthropic API"""
    content: str
    usage: Dict[str, int]
    model: str
    success: bool
    error: Optional[str] = None


class AnthropicAIInterface:
    """Interface for Anthropic Claude API integration"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config = config_manager or ConfigManager()
        self.logger = logging.getLogger("AnthropicAI")
        
        # Check if Anthropic library is available
        if not HAS_ANTHROPIC:
            self.logger.error("Anthropic library not installed. Install with: pip install anthropic")
            self.enabled = False
            return
        
        # Get configuration
        self.enabled = self.config.get_value('ai.enabled', True)
        if not self.enabled:
            self.logger.info("AI integration disabled in configuration")
            return
            
        # Check provider
        provider = self.config.get_value('ai.provider', 'anthropic')
        if provider != 'anthropic':
            self.logger.warning(f"AI provider is {provider}, expected 'anthropic'. Disabling AI.")
            self.enabled = False
            return
        
        # Initialize Anthropic client
        try:
            self._setup_anthropic_client()
        except Exception as e:
            self.logger.error(f"Failed to setup Anthropic client: {e}")
            self.enabled = False
    
    def _setup_anthropic_client(self):
        """Setup Anthropic API client"""
        # Get API key from environment
        api_key_env = self.config.get_value('ai.api_key_env', 'ANTHROPIC_API_KEY')
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(f"Anthropic API key not found in environment variable: {api_key_env}")
        
        # Initialize client
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Get model configuration
        self.model = self.config.get_value('ai.model', 'claude-3-5-sonnet-20241022')
        self.temperature = self.config.get_value('ai.temperature', 0.1)
        self.max_tokens = self.config.get_value('ai.max_tokens', 4096)
        self.timeout = self.config.get_value('ai.timeout', 30)
        
        self.logger.info(f"Anthropic AI initialized with model: {self.model}")
    
    def is_enabled(self) -> bool:
        """Check if AI interface is enabled and ready"""
        return self.enabled and hasattr(self, 'client')
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> AnthropicResponse:
        """Generate response using Claude"""
        if not self.is_enabled():
            return AnthropicResponse(
                content="AI interface not available",
                usage={},
                model="none",
                success=False,
                error="AI interface not enabled or configured"
            )
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Call Anthropic API
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": messages
            }
            
            # Add system prompt if provided
            if system_prompt:
                kwargs["system"] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            
            # Extract response content
            content = ""
            if response.content:
                content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            
            return AnthropicResponse(
                content=content,
                usage=response.usage.__dict__ if hasattr(response, 'usage') else {},
                model=response.model if hasattr(response, 'model') else self.model,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            return AnthropicResponse(
                content="",
                usage={},
                model=self.model,
                success=False,
                error=str(e)
            )
    
    def analyze_binary_security(self, binary_info: Dict[str, Any]) -> AnthropicResponse:
        """Analyze binary for security indicators using Claude"""
        try:
            system_prompt = """You are a cybersecurity expert analyzing binary files. 
            Provide concise security analysis focusing on potential threats, suspicious patterns, 
            and recommendations. Be factual and specific."""
            
            prompt = f"""
            Analyze this binary file for security indicators:
            
            File: {binary_info.get('file_path', 'unknown')}
            Format: {binary_info.get('format_type', 'unknown')}
            Architecture: {binary_info.get('architecture', 'unknown')}
            Size: {binary_info.get('file_size', 0)} bytes
            Entropy: {binary_info.get('entropy', 0)}
            
            Sections: {binary_info.get('section_count', 0)}
            Imports: {binary_info.get('import_count', 0)}
            Exports: {binary_info.get('export_count', 0)}
            
            Notable strings: {binary_info.get('notable_strings', [])}
            
            Provide:
            1. Security risk assessment (Low/Medium/High)
            2. Suspicious indicators found
            3. Behavioral predictions
            4. Recommendations for further analysis
            """
            
            return self.generate_response(prompt, system_prompt)
        except Exception as e:
            self.logger.error(f"Binary security analysis failed: {e}")
            return AnthropicResponse(
                content="Security analysis unavailable due to AI error",
                usage={},
                model=self.model,
                success=False,
                error=str(e)
            )
    
    def analyze_code_patterns(self, code_analysis: Dict[str, Any]) -> AnthropicResponse:
        """Analyze code patterns and provide insights"""
        system_prompt = """You are a reverse engineering expert. Analyze code patterns 
        and architectural decisions to understand the original software design and purpose."""
        
        prompt = f"""
        Analyze these code patterns and architectural elements:
        
        Functions detected: {code_analysis.get('functions_detected', 0)}
        Code quality score: {code_analysis.get('code_quality', 0)}
        Architecture: {code_analysis.get('architecture', 'unknown')}
        Compiler indicators: {code_analysis.get('compiler_info', {})}
        
        Optimization patterns: {code_analysis.get('optimization_patterns', [])}
        Control flow complexity: {code_analysis.get('complexity_metrics', {})}
        
        Provide insights on:
        1. Original software purpose and design
        2. Development practices and tools used
        3. Code quality and maintainability indicators
        4. Reconstruction recommendations
        """
        
        return self.generate_response(prompt, system_prompt)
    
    def generate_documentation(self, analysis_results: Dict[str, Any]) -> AnthropicResponse:
        """Generate comprehensive documentation for the analyzed binary"""
        system_prompt = """You are a technical documentation expert. Create clear, 
        comprehensive documentation for reverse-engineered software based on analysis results."""
        
        prompt = f"""
        Generate documentation for this reverse-engineered software:
        
        Analysis Summary:
        - Binary format: {analysis_results.get('format', 'unknown')}
        - Architecture: {analysis_results.get('architecture', 'unknown')}
        - Functions: {analysis_results.get('functions_count', 0)}
        - Quality score: {analysis_results.get('quality_score', 0)}
        
        Key findings: {analysis_results.get('key_findings', [])}
        Security assessment: {analysis_results.get('security_assessment', 'unknown')}
        
        Create:
        1. Executive summary
        2. Technical specifications
        3. Function descriptions
        4. Usage recommendations
        5. Security considerations
        """
        
        return self.generate_response(prompt, system_prompt)


# Convenience functions for backward compatibility with existing LangChain-style usage
class AnthropicTool:
    """Tool wrapper for Anthropic AI interface"""
    
    def __init__(self, name: str, description: str, ai_interface: AnthropicAIInterface):
        self.name = name
        self.description = description
        self.ai_interface = ai_interface
    
    def run(self, input_text: str) -> str:
        """Run the tool with input text"""
        response = self.ai_interface.generate_response(
            f"Task: {self.description}\nInput: {input_text}"
        )
        return response.content if response.success else f"Error: {response.error}"


class AnthropicAgentExecutor:
    """Agent executor wrapper for Anthropic AI"""
    
    def __init__(self, ai_interface: AnthropicAIInterface, tools: List[AnthropicTool] = None):
        self.ai_interface = ai_interface
        self.tools = tools or []
        self.logger = logging.getLogger("AnthropicAgentExecutor")
    
    def run(self, prompt: str) -> str:
        """Execute agent with prompt"""
        if not self.ai_interface.is_enabled():
            return "AI interface not available"
        
        # For now, just use direct response. Could be enhanced with tool calling
        response = self.ai_interface.generate_response(prompt)
        return response.content if response.success else f"Error: {response.error}"


def create_anthropic_tools(ai_interface: AnthropicAIInterface) -> List[AnthropicTool]:
    """Create standard tools for binary analysis"""
    return [
        AnthropicTool(
            "binary_security_analysis",
            "Analyze binary file for security indicators and threats",
            ai_interface
        ),
        AnthropicTool(
            "code_pattern_analysis", 
            "Analyze code patterns and architectural decisions",
            ai_interface
        ),
        AnthropicTool(
            "documentation_generation",
            "Generate comprehensive documentation for analyzed software",
            ai_interface
        )
    ]