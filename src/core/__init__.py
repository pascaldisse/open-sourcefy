"""
Open-Sourcefy Matrix Pipeline Core Module
New 16-agent Matrix-themed architecture with Phase A infrastructure
"""

# Phase A - Core Infrastructure
from .config_manager import ConfigManager, get_config_manager
from .binary_utils import BinaryAnalyzer, BinaryInfo, BinaryFormat, Architecture
from .file_utils import FileManager, JsonFileHandler, YamlFileHandler, PathUtils
from .shared_utils import (
    PerformanceMonitor, MatrixLogger, DataValidator, 
    RetryHelper, SystemUtils, ProgressTracker, ErrorHandler
)
from .ai_engine_interface import (
    AIEngineInterface, AIEngineManager, AIRequest, AIResponse,
    MatrixPromptLibrary, get_ai_manager, get_conversation_manager
)

# LangChain agent framework
from .langchain_agent_base import MatrixLangChainAgent
from .matrix_agent_base import MatrixAgentBase

__version__ = "2.0.0-matrix"
__author__ = "Open-Sourcefy Matrix Team"

__all__ = [
    # Configuration
    "ConfigManager", "get_config_manager",
    
    # Binary Analysis
    "BinaryAnalyzer", "BinaryInfo", "BinaryFormat", "Architecture",
    
    # File Operations
    "FileManager", "JsonFileHandler", "YamlFileHandler", "PathUtils",
    
    # Utilities
    "PerformanceMonitor", "MatrixLogger", "DataValidator", 
    "RetryHelper", "SystemUtils", "ProgressTracker", "ErrorHandler",
    
    # AI Engine
    "AIEngineInterface", "AIEngineManager", "AIRequest", "AIResponse",
    "MatrixPromptLibrary", "get_ai_manager", "get_conversation_manager",
    
    # Agent Framework
    "MatrixLangChainAgent", "MatrixAgentBase"
]