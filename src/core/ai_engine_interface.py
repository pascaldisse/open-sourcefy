"""
AI Engine Interface for Open-Sourcefy Matrix Pipeline
Abstract interface for AI engines with pluggable implementations
"""

import abc
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json


class AIEngineType(Enum):
    """AI Engine type enumeration"""
    LANGCHAIN = "langchain"
    MOCK = "mock"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


class AIRole(Enum):
    """AI role types for different agent functions"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    MATRIX_AGENT = "matrix_agent"


@dataclass
class AIMessage:
    """AI message container"""
    role: AIRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'role': self.role.value,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AIMessage':
        """Create from dictionary"""
        return cls(
            role=AIRole(data['role']),
            content=data['content'],
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', time.time())
        )


@dataclass
class AIRequest:
    """AI request container"""
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[AIMessage] = field(default_factory=list)
    agent_name: str = ""
    operation_type: str = "analysis"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AIResponse:
    """AI response container"""
    content: str
    success: bool = True
    error_message: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'content': self.content,
            'success': self.success,
            'error_message': self.error_message,
            'usage': self.usage,
            'metadata': self.metadata,
            'processing_time': self.processing_time
        }


class AIEngineInterface(abc.ABC):
    """Abstract interface for AI engines"""
    
    def __init__(self, engine_type: AIEngineType, config: Dict[str, Any]):
        self.engine_type = engine_type
        self.config = config
        self.logger = logging.getLogger(f"AIEngine_{engine_type.value}")
        self._initialized = False
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize the AI engine"""
        pass
    
    @abc.abstractmethod
    def generate_response(self, request: AIRequest) -> AIResponse:
        """Generate response from AI engine"""
        pass
    
    @abc.abstractmethod
    def validate_connection(self) -> bool:
        """Validate connection to AI service"""
        pass
    
    @abc.abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if engine is initialized"""
        return self._initialized
    
    def shutdown(self):
        """Shutdown the AI engine"""
        self._initialized = False
        self.logger.info(f"AI engine {self.engine_type.value} shut down")


class AIEngineFactory:
    """Factory for creating AI engine instances"""
    
    _engines: Dict[AIEngineType, type] = {}
    
    @classmethod
    def register_engine(cls, engine_type: AIEngineType, engine_class: type):
        """Register an AI engine implementation"""
        cls._engines[engine_type] = engine_class
    
    @classmethod
    def create_engine(cls, engine_type: AIEngineType, config: Dict[str, Any]) -> AIEngineInterface:
        """Create AI engine instance"""
        if engine_type not in cls._engines:
            raise ValueError(f"Unknown AI engine type: {engine_type}")
        
        engine_class = cls._engines[engine_type]
        return engine_class(engine_type, config)
    
    @classmethod
    def list_available_engines(cls) -> List[AIEngineType]:
        """List available engine types"""
        return list(cls._engines.keys())


class AIEngineManager:
    """Manager for AI engine lifecycle and operations"""
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger("AIEngineManager")
        self.default_config = default_config or {}
        self._engines: Dict[str, AIEngineInterface] = {}
        self._active_engine: Optional[AIEngineInterface] = None
    
    def create_engine(self, engine_type: Union[str, AIEngineType], 
                     engine_name: str = "default", 
                     config: Optional[Dict[str, Any]] = None) -> AIEngineInterface:
        """Create and register AI engine"""
        if isinstance(engine_type, str):
            engine_type = AIEngineType(engine_type)
        
        # Merge configs
        final_config = {**self.default_config}
        if config:
            final_config.update(config)
        
        # Create engine
        engine = AIEngineFactory.create_engine(engine_type, final_config)
        
        # Initialize engine
        if engine.initialize():
            self._engines[engine_name] = engine
            if self._active_engine is None:
                self._active_engine = engine
            self.logger.info(f"Created and initialized AI engine: {engine_name} ({engine_type.value})")
            return engine
        else:
            raise RuntimeError(f"Failed to initialize AI engine: {engine_name}")
    
    def get_engine(self, engine_name: str = "default") -> Optional[AIEngineInterface]:
        """Get AI engine by name"""
        return self._engines.get(engine_name)
    
    def set_active_engine(self, engine_name: str):
        """Set active AI engine"""
        if engine_name not in self._engines:
            raise ValueError(f"Engine not found: {engine_name}")
        
        self._active_engine = self._engines[engine_name]
        self.logger.info(f"Set active AI engine: {engine_name}")
    
    def get_active_engine(self) -> Optional[AIEngineInterface]:
        """Get active AI engine"""
        return self._active_engine
    
    def generate_response(self, request: AIRequest, 
                         engine_name: Optional[str] = None) -> AIResponse:
        """Generate response using specified or active engine"""
        engine = self.get_engine(engine_name) if engine_name else self._active_engine
        
        if not engine:
            return AIResponse(
                content="",
                success=False,
                error_message="No AI engine available"
            )
        
        if not engine.is_initialized():
            return AIResponse(
                content="",
                success=False,
                error_message=f"AI engine not initialized: {engine.engine_type.value}"
            )
        
        try:
            return engine.generate_response(request)
        except Exception as e:
            self.logger.error(f"AI engine error: {e}")
            return AIResponse(
                content="",
                success=False,
                error_message=str(e)
            )
    
    def shutdown_all(self):
        """Shutdown all AI engines"""
        for name, engine in self._engines.items():
            try:
                engine.shutdown()
                self.logger.info(f"Shut down AI engine: {name}")
            except Exception as e:
                self.logger.error(f"Error shutting down engine {name}: {e}")
        
        self._engines.clear()
        self._active_engine = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all engines"""
        status = {
            'total_engines': len(self._engines),
            'active_engine': None,
            'engines': {}
        }
        
        if self._active_engine:
            # Find active engine name
            for name, engine in self._engines.items():
                if engine == self._active_engine:
                    status['active_engine'] = name
                    break
        
        for name, engine in self._engines.items():
            status['engines'][name] = {
                'type': engine.engine_type.value,
                'initialized': engine.is_initialized(),
                'info': engine.get_engine_info()
            }
        
        return status


class PromptTemplate:
    """Template for AI prompts with variable substitution"""
    
    def __init__(self, template: str, variables: Optional[List[str]] = None):
        self.template = template
        self.variables = variables or []
        self._extract_variables()
    
    def _extract_variables(self):
        """Extract variables from template"""
        import re
        variables = re.findall(r'\{(\w+)\}', self.template)
        self.variables = list(set(variables))
    
    def format(self, **kwargs) -> str:
        """Format template with variables"""
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing template variables: {missing_vars}")
        
        return self.template.format(**kwargs)
    
    def get_variables(self) -> List[str]:
        """Get list of template variables"""
        return self.variables.copy()


class MatrixPromptLibrary:
    """Library of Matrix-themed prompts for agents"""
    
    BINARY_ANALYSIS = PromptTemplate("""
🔮 Matrix Agent {agent_name} - Binary Analysis Protocol

MISSION: Analyze the binary file "{binary_path}" with Matrix precision.

BINARY METADATA:
- File Size: {file_size}
- Architecture: {architecture}
- Format: {binary_format}

ANALYSIS FOCUS:
{analysis_focus}

MATRIX DIRECTIVE: Process this binary through your unique Matrix capabilities and provide detailed analysis. Structure your response as actionable intelligence for the Matrix pipeline.

CONTEXT DATA:
{context_data}

Execute your Matrix protocol now.
    """)
    
    CODE_DECOMPILATION = PromptTemplate("""
🤖 Matrix Agent {agent_name} - Code Decompilation Protocol

MISSION: Decompile the provided assembly code into readable source code.

ASSEMBLY INPUT:
```assembly
{assembly_code}
```

ARCHITECTURE: {architecture}
FUNCTION CONTEXT: {function_context}

MATRIX DIRECTIVE: Use your Matrix code-sight abilities to reconstruct the original source code. Focus on:
1. Function structure and logic flow
2. Variable identification and typing
3. Control flow reconstruction
4. Optimization pattern reversal

ADDITIONAL CONTEXT:
{additional_context}

Reconstruct the Matrix code now.
    """)
    
    SYSTEM_ANALYSIS = PromptTemplate("""
🔮 Matrix System Analysis - Agent {agent_name}

ANALYSIS TARGET: {target_description}

DATA STREAMS:
{data_streams}

ANALYSIS PARAMETERS:
- Depth Level: {analysis_depth}
- Focus Areas: {focus_areas}
- Quality Threshold: {quality_threshold}

MATRIX PROTOCOL: Analyze the provided data streams using your Matrix analytical capabilities. Identify patterns, anomalies, and actionable insights.

CONTEXT MATRIX:
{context_matrix}

Initialize analysis sequence now.
    """)
    
    @classmethod
    def get_template(cls, template_name: str) -> PromptTemplate:
        """Get prompt template by name"""
        template_map = {
            'binary_analysis': cls.BINARY_ANALYSIS,
            'code_decompilation': cls.CODE_DECOMPILATION,
            'system_analysis': cls.SYSTEM_ANALYSIS
        }
        
        if template_name not in template_map:
            raise ValueError(f"Unknown template: {template_name}")
        
        return template_map[template_name]
    
    @classmethod
    def list_templates(cls) -> List[str]:
        """List available template names"""
        return ['binary_analysis', 'code_decompilation', 'system_analysis']


class AIConversationManager:
    """Manager for AI conversation context and history"""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.conversations: Dict[str, List[AIMessage]] = {}
        self.logger = logging.getLogger("AIConversationManager")
    
    def start_conversation(self, conversation_id: str, 
                          system_prompt: Optional[str] = None) -> str:
        """Start a new conversation"""
        self.conversations[conversation_id] = []
        
        if system_prompt:
            self.add_message(conversation_id, AIMessage(
                role=AIRole.SYSTEM,
                content=system_prompt,
                metadata={'type': 'system_initialization'}
            ))
        
        self.logger.debug(f"Started conversation: {conversation_id}")
        return conversation_id
    
    def add_message(self, conversation_id: str, message: AIMessage):
        """Add message to conversation"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append(message)
        
        # Trim history if needed
        if len(self.conversations[conversation_id]) > self.max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history:]
    
    def get_conversation(self, conversation_id: str) -> List[AIMessage]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])
    
    def get_context_for_request(self, conversation_id: str) -> List[AIMessage]:
        """Get conversation context for AI request"""
        return self.get_conversation(conversation_id)
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation history"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            self.logger.debug(f"Cleared conversation: {conversation_id}")
    
    def export_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Export conversation to dictionary"""
        messages = self.get_conversation(conversation_id)
        return {
            'conversation_id': conversation_id,
            'message_count': len(messages),
            'messages': [msg.to_dict() for msg in messages]
        }


# Utility functions
def create_matrix_request(agent_name: str, prompt: str, 
                         context: Optional[Dict[str, Any]] = None,
                         operation_type: str = "analysis") -> AIRequest:
    """Create AI request with Matrix theming"""
    return AIRequest(
        prompt=prompt,
        context=context or {},
        agent_name=agent_name,
        operation_type=operation_type,
        metadata={
            'matrix_themed': True,
            'agent_role': 'matrix_agent'
        }
    )


def validate_ai_response(response: AIResponse, 
                        min_length: int = 10) -> bool:
    """Validate AI response quality"""
    if not response.success:
        return False
    
    if len(response.content.strip()) < min_length:
        return False
    
    # Check for common AI failure patterns
    failure_patterns = [
        "I cannot", "I am unable", "I don't have access",
        "I'm sorry", "I apologize", "I cannot help"
    ]
    
    content_lower = response.content.lower()
    if any(pattern in content_lower for pattern in failure_patterns):
        return False
    
    return True


def extract_code_from_response(response: str) -> List[str]:
    """Extract code blocks from AI response"""
    import re
    
    # Match code blocks with various formats
    patterns = [
        r'```[\w]*\n(.*?)\n```',  # Triple backticks
        r'`([^`]+)`',              # Single backticks
        r'<code>(.*?)</code>',     # HTML code tags
    ]
    
    code_blocks = []
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        code_blocks.extend(matches)
    
    return [block.strip() for block in code_blocks if block.strip()]


# Global instances
_ai_manager: Optional[AIEngineManager] = None
_conversation_manager: Optional[AIConversationManager] = None


def get_ai_manager() -> AIEngineManager:
    """Get global AI engine manager"""
    global _ai_manager
    if _ai_manager is None:
        _ai_manager = AIEngineManager()
    return _ai_manager


def get_conversation_manager() -> AIConversationManager:
    """Get global conversation manager"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = AIConversationManager()
    return _conversation_manager


def get_ai_engine() -> AIEngineManager:
    """Get global AI engine manager (alias for get_ai_manager)"""
    return get_ai_manager()