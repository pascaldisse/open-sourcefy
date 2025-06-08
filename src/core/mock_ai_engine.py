"""
Mock AI Engine Implementation for Open-Sourcefy Matrix Pipeline
Provides fallback AI functionality when real AI providers are unavailable
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .ai_engine_interface import AIEngineInterface, AIEngineType, AIRequest, AIResponse
from .ai_setup import AIConfig, AIInterface, AIResponse as SetupAIResponse


class MockAIEngine(AIEngineInterface):
    """Mock AI engine that provides realistic responses without external dependencies"""
    
    def __init__(self, engine_type: AIEngineType, config: Dict[str, Any]):
        super().__init__(engine_type, config)
        self.mock_responses = self._load_mock_responses()
        self.call_count = 0
    
    def initialize(self) -> bool:
        """Initialize mock AI engine - always succeeds"""
        self._initialized = True
        self.logger.info("Mock AI engine initialized successfully")
        return True
    
    def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate mock response based on request type"""
        start_time = time.time()
        self.call_count += 1
        
        # Determine response type based on prompt content
        response_content = self._get_mock_response(request.prompt, request.operation_type)
        
        processing_time = time.time() - start_time
        
        return AIResponse(
            content=response_content,
            success=True,
            usage={'prompt_tokens': len(request.prompt.split()), 'completion_tokens': len(response_content.split())},
            metadata={
                'mock_engine': True,
                'call_count': self.call_count,
                'operation_type': request.operation_type
            },
            processing_time=processing_time
        )
    
    def validate_connection(self) -> bool:
        """Validate connection - always returns True for mock"""
        return True
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get mock engine information"""
        return {
            'engine_type': 'mock',
            'version': '1.0.0',
            'capabilities': ['analysis', 'decompilation', 'security_assessment'],
            'call_count': self.call_count,
            'status': 'operational'
        }
    
    def _load_mock_responses(self) -> Dict[str, str]:
        """Load realistic mock responses for different analysis types"""
        return {
            'binary_analysis': """
            Based on the binary analysis, this appears to be a Windows PE executable with the following characteristics:
            
            **Security Assessment**: Medium Risk
            - Standard Windows executable with common library dependencies
            - No obvious obfuscation or packing detected
            - Import table suggests standard GUI application functionality
            
            **Architecture Analysis**: 
            - Target Architecture: x86/x64 Windows
            - Compiler: Likely Microsoft Visual C++ based on runtime dependencies
            - Build Configuration: Release build with optimizations
            
            **Behavioral Predictions**:
            - GUI application with network capabilities
            - File system access for configuration and data storage
            - Registry access for settings persistence
            
            **Recommendations**:
            1. Monitor network connections during dynamic analysis
            2. Check for configuration files in AppData directories
            3. Analyze string references for functionality insights
            """,
            
            'decompilation': """
            **Function Analysis Summary**:
            
            ```c
            // Main application entry point
            int main(int argc, char* argv[]) {
                initialize_application();
                setup_ui_components();
                
                while (application_running) {
                    process_messages();
                    update_display();
                    handle_user_input();
                }
                
                cleanup_resources();
                return 0;
            }
            
            // Network initialization function
            bool initialize_network() {
                if (WSAStartup(MAKEWORD(2,2), &wsaData) != 0) {
                    return false;
                }
                
                connection_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
                return (connection_socket != INVALID_SOCKET);
            }
            ```
            
            **Quality Assessment**: Good (75% confidence)
            - Function boundaries clearly identified
            - Control flow reconstruction successful
            - Variable types inferred from usage patterns
            - Some optimized code patterns reversed successfully
            """,
            
            'code_quality': """
            **Code Quality Analysis**:
            
            **Overall Score**: 7.2/10
            
            **Strengths**:
            - Well-structured function organization
            - Consistent error handling patterns
            - Proper resource management
            - Clear separation of concerns
            
            **Areas for Improvement**:
            - Some magic numbers should be constants
            - Error messages could be more descriptive
            - Memory allocation patterns could be optimized
            
            **Architecture Insights**:
            - Follows standard Windows application patterns
            - Uses common design patterns (Observer, Factory)
            - Good modularity between UI and business logic
            """,
            
            'security_analysis': """
            **Security Analysis Report**:
            
            **Risk Level**: MEDIUM
            
            **Vulnerabilities Identified**:
            1. **Buffer Usage**: Some string operations may be vulnerable to overflow
            2. **Network Security**: Limited input validation on network data
            3. **File Access**: Unrestricted file system access in some functions
            
            **Security Features Detected**:
            - DEP/NX compatibility enabled
            - ASLR support present
            - Stack canaries in debug builds
            
            **Recommendations**:
            1. Implement bounds checking for all string operations
            2. Add input sanitization for network communications
            3. Use secure file access patterns with proper validation
            4. Consider code signing for distribution
            """,
            
            'general': """
            Analysis completed successfully. The binary exhibits standard characteristics 
            for its type and architecture. Key findings have been documented and are 
            available for further processing by subsequent agents in the Matrix pipeline.
            
            Confidence Level: High (85%)
            Processing Quality: Good
            Next Steps: Proceed with detailed analysis using specialized agents.
            """
        }
    
    def _get_mock_response(self, prompt: str, operation_type: str) -> str:
        """Get appropriate mock response based on prompt content"""
        prompt_lower = prompt.lower()
        
        # Determine response type based on keywords in prompt
        if any(keyword in prompt_lower for keyword in ['security', 'vulnerability', 'threat', 'malware']):
            return self.mock_responses['security_analysis']
        elif any(keyword in prompt_lower for keyword in ['decompile', 'code', 'function', 'assembly']):
            return self.mock_responses['decompilation']
        elif any(keyword in prompt_lower for keyword in ['quality', 'architecture', 'pattern']):
            return self.mock_responses['code_quality']
        elif any(keyword in prompt_lower for keyword in ['binary', 'format', 'pe', 'elf']):
            return self.mock_responses['binary_analysis']
        else:
            return self.mock_responses['general']


class MockAIInterface(AIInterface):
    """Mock implementation of AIInterface for fallback when real AI is unavailable"""
    
    def __init__(self, config: AIConfig):
        # Create a mock client object
        class MockClient:
            pass
        
        super().__init__(MockClient(), config)
        self.mock_engine = MockAIEngine(AIEngineType.MOCK, config.__dict__)
        self.mock_engine.initialize()
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> SetupAIResponse:
        """Generate mock response"""
        try:
            # Create AI request
            ai_request = AIRequest(
                prompt=prompt,
                operation_type="analysis",
                metadata={'system_prompt': system_prompt}
            )
            
            # Get response from mock engine
            mock_response = self.mock_engine.generate_response(ai_request)
            
            return SetupAIResponse(
                content=mock_response.content,
                model="mock-ai-v1.0",
                usage=mock_response.usage,
                success=mock_response.success,
                provider="mock"
            )
            
        except Exception as e:
            self.logger.error(f"Mock AI response generation failed: {e}")
            return SetupAIResponse(
                content="Mock AI analysis completed with basic heuristics.",
                model="mock-ai-v1.0",
                usage={},
                success=True,
                provider="mock"
            )


def create_mock_ai_setup(config_manager=None) -> 'MockAISetup':
    """Create mock AI setup for testing and fallback"""
    from .config_manager import ConfigManager
    
    if config_manager is None:
        config_manager = ConfigManager()
    
    return MockAISetup(config_manager)


class MockAISetup:
    """Mock AI setup that always works"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger("MockAISetup")
        
        # Create mock AI config
        from .ai_setup import AIConfig, AIProvider
        self.ai_config = AIConfig(
            provider=AIProvider.DISABLED,  # Use disabled but provide mock interface
            model="mock-ai-v1.0",
            api_key_env="NONE",
            temperature=0.1,
            max_tokens=4096,
            timeout=30,
            enabled=True  # Always enabled for mock
        )
        
        # Create mock AI interface
        self.ai_interface = MockAIInterface(self.ai_config)
        self.logger.info("Mock AI setup initialized successfully")
    
    def get_ai_interface(self):
        """Get mock AI interface"""
        return self.ai_interface
    
    def is_enabled(self) -> bool:
        """Mock AI is always enabled"""
        return True
    
    def get_config(self):
        """Get mock AI config"""
        return self.ai_config


# Register mock engine with factory
from .ai_engine_interface import AIEngineFactory
AIEngineFactory.register_engine(AIEngineType.MOCK, MockAIEngine)