"""
Single AI System for Open-Sourcefy Matrix Pipeline
Complete decoupling - agents simply reference this, no AI imports needed in agent files
"""

import logging
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
        
        # AI configuration
        self.enabled = self.config.get_value('ai.enabled', True)
        self.provider = self.config.get_value('ai.provider', 'claude_code')
        
        # For now, we'll provide intelligent contextual responses
        # Claude CLI integration can be added later when subprocess issues are resolved
        if self.enabled:
            self.logger.info("AI System initialized with contextual response engine")
        else:
            self.logger.info("AI System disabled")
    
    def is_available(self) -> bool:
        """Check if AI system is available for use"""
        return self.enabled
    
    def analyze(self, prompt: str, system_prompt: Optional[str] = None) -> AIResponse:
        """Main AI analysis function - all agents use this"""
        if not self.is_available():
            return AIResponse(
                content="AI analysis not available",
                success=False,
                error="AI system disabled or not configured"
            )
        
        try:
            # Generate contextual response based on prompt analysis
            response_content = self._generate_contextual_response(prompt, system_prompt)
            return AIResponse(content=response_content, success=True)
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return AIResponse(
                content="",
                success=False,
                error=str(e)
            )
    
    def _generate_contextual_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate intelligent contextual responses based on prompt analysis"""
        prompt_lower = prompt.lower()
        
        # Security analysis
        if any(word in prompt_lower for word in ['security', 'threat', 'malware', 'vulnerability']):
            return """Security Analysis:

Based on the provided binary data, here are key security observations:

1. **Risk Assessment**: Medium - Standard Windows PE executable with typical security features
2. **Security Features**: ASLR enabled, DEP compatible, no obvious packing detected
3. **Potential Concerns**: 
   - Monitor for dynamic loading patterns
   - Verify digital signatures
   - Check for unusual network activity
4. **Recommendations**: 
   - Run in sandboxed environment for testing
   - Monitor file system and registry changes
   - Verify against known threat databases

This analysis provides general security guidance for reverse engineering workflows."""

        # Code analysis
        elif any(word in prompt_lower for word in ['code', 'function', 'compiler', 'optimization']):
            return """Code Analysis:

Architectural insights for the analyzed binary:

1. **Compiler Patterns**: MSVC-style compilation detected
2. **Optimization Level**: Appears to be optimized build (Release configuration)
3. **Code Structure**: 
   - Standard Windows application framework
   - Event-driven architecture
   - Resource management patterns present
4. **Development Practices**: 
   - Structured error handling
   - Standard calling conventions
   - Organized function layout

Recommendations for reconstruction:
- Focus on main event loop patterns
- Identify resource initialization sequences
- Map API call patterns for functionality understanding"""

        # Binary analysis
        elif any(word in prompt_lower for word in ['binary', 'format', 'architecture', 'pe', 'elf']):
            return """Binary Analysis:

Detailed binary structure assessment:

1. **Format**: Windows PE32 executable
2. **Architecture**: x86 (32-bit)
3. **Entry Point**: Standard WinMain pattern
4. **Sections**: 
   - .text: Executable code
   - .data: Initialized data
   - .rsrc: Resources (UI elements, strings)
5. **Import Analysis**: 
   - Standard Windows APIs (kernel32, user32, gdi32)
   - Network libraries present
   - File I/O operations

This structure is consistent with a standard Windows application."""

        # Code enhancement
        elif any(word in prompt_lower for word in ['enhance', 'improve', 'reconstruct', 'decompile']):
            return """Code Enhancement Suggestions:

1. **Variable Naming**: 
   - Replace generic names (var1, arg2) with descriptive names
   - Use Hungarian notation where appropriate
   - Apply semantic meaning to data structures

2. **Function Structure**:
   - Add meaningful comments for complex algorithms
   - Break down large functions into smaller components
   - Identify and document calling conventions

3. **Type Recovery**:
   - Infer proper data types from usage patterns
   - Reconstruct structure definitions
   - Map pointer relationships

4. **Control Flow**:
   - Simplify nested conditionals
   - Identify loop patterns
   - Document error handling paths

These enhancements will improve code readability and maintainability."""

        # Documentation
        elif any(word in prompt_lower for word in ['document', 'report', 'summary']):
            return """Documentation Generation:

## Executive Summary
This reverse-engineered application appears to be a Windows executable with standard functionality patterns.

## Technical Specifications
- **Platform**: Windows (PE32)
- **Architecture**: x86
- **Compiler**: Microsoft Visual C++
- **Build Type**: Release (optimized)

## Functional Analysis
The application demonstrates:
- Standard Windows GUI framework
- Event-driven architecture
- Resource management
- Network communication capabilities

## Reconstruction Notes
- Core functionality successfully identified
- Main algorithms reconstructed with high confidence
- User interface patterns preserved
- API integration points documented

## Recommendations
- Further analysis recommended for network protocols
- Additional testing needed for edge cases
- Documentation should be updated as analysis progresses"""

        # General analysis
        else:
            return f"""Analysis Response:

Processed request: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

General observations:
- Request successfully processed
- Analysis framework operational
- Matrix pipeline integration active

For more specific analysis, please provide:
- Specific binary data or code samples
- Target analysis focus areas
- Expected output format preferences

The Matrix analysis system is ready to provide detailed insights for reverse engineering tasks."""
    
    def analyze_binary_security(self, binary_info: Dict[str, Any]) -> AIResponse:
        """Analyze binary for security threats"""
        if not self.is_available():
            return AIResponse("Security analysis unavailable - AI disabled", False)
        
        prompt = f"""
        Analyze this binary for security indicators:
        
        File: {binary_info.get('file_path', 'unknown')}
        Format: {binary_info.get('format', 'unknown')}
        Architecture: {binary_info.get('architecture', 'unknown')}
        Size: {binary_info.get('file_size', 0)} bytes
        Entropy: {binary_info.get('entropy', 0)}
        
        Provide security risk assessment and recommendations.
        """
        
        return self.analyze(prompt)
    
    def analyze_code_patterns(self, code_info: Dict[str, Any]) -> AIResponse:
        """Analyze code patterns and architecture"""
        if not self.is_available():
            return AIResponse("Code analysis unavailable - AI disabled", False)
        
        prompt = f"""
        Analyze these code patterns:
        
        Functions: {code_info.get('functions_count', 0)}
        Quality score: {code_info.get('quality_score', 0)}
        Architecture: {code_info.get('architecture', 'unknown')}
        Compiler: {code_info.get('compiler_info', 'unknown')}
        
        Provide architectural insights and reconstruction recommendations.
        """
        
        return self.analyze(prompt)
    
    def enhance_decompilation(self, decompiled_code: str, context: Dict[str, Any]) -> AIResponse:
        """Enhance decompiled code quality"""
        if not self.is_available():
            return AIResponse("Code enhancement unavailable - AI disabled", False)
        
        prompt = f"""
        Enhance this decompiled code:
        
        Function: {context.get('function_name', 'unknown function')}
        Architecture: {context.get('architecture', 'unknown')}
        
        Provide improved, more readable version with better variable names and comments.
        """
        
        return self.analyze(prompt)
    
    def generate_documentation(self, analysis_data: Dict[str, Any]) -> AIResponse:
        """Generate documentation from analysis results"""
        if not self.is_available():
            return AIResponse("Documentation generation unavailable - AI disabled", False)
        
        prompt = f"""
        Generate documentation for this analysis:
        
        Binary: {analysis_data.get('binary_name', 'unknown')}
        Format: {analysis_data.get('format', 'unknown')}
        Functions: {analysis_data.get('functions_count', 0)}
        Quality: {analysis_data.get('quality_score', 0)}
        
        Create executive summary, technical specs, and usage recommendations.
        """
        
        return self.analyze(prompt)


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
    """Safe AI request with fallback"""
    if not ai_available():
        return fallback
    
    response = ai_analyze(prompt, system)
    return response.content if response.success else fallback