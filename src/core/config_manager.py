"""
Configuration Management System for Open-Sourcefy Matrix Pipeline
Handles environment variables, YAML/JSON config loading, and default value hierarchies
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum

# Required imports - strict mode only
import yaml
import json


class ConfigSource(Enum):
    """Configuration source priority"""
    ENVIRONMENT = "environment"
    CONFIG_FILE = "config_file"
    DEFAULT = "default"


@dataclass
class ConfigValue:
    """Configuration value with source tracking"""
    value: Any
    source: ConfigSource
    description: str = ""
    

class ConfigManager:
    """Configuration manager with hierarchical value resolution"""
    
    def __init__(self, config_file: Optional[str] = None, project_root: Optional[Path] = None):
        self.logger = logging.getLogger("ConfigManager")
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config_file = config_file or self._find_config_file()
        self.config_values: Dict[str, ConfigValue] = {}
        
        # Load configuration in priority order
        self._load_default_config()
        self._load_config_file()
        self._load_environment_variables()
        self._resolve_paths()
        
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations"""
        possible_files = [
            self.project_root / "matrix_config.yaml",
            self.project_root / "matrix_config.yml", 
            self.project_root / "config.yaml",
            self.project_root / "config.yml",
            self.project_root / "matrix_config.json",
            self.project_root / "config.json"
        ]
        
        for config_path in possible_files:
            if config_path.exists():
                self.logger.info(f"Found configuration file: {config_path}")
                return str(config_path)
        
        self.logger.info("No configuration file found, using environment variables and defaults")
        return None
    
    def _load_default_config(self):
        """Load default configuration values"""
        defaults = {
            # Pipeline defaults
            "pipeline.parallel_mode": ("thread", "Parallel execution mode (thread/process)"),
            "pipeline.batch_size": (4, "Number of agents to run in parallel"),
            "pipeline.timeout_agent": (-1, "Default agent timeout in seconds (-1 for unlimited)"),
            "pipeline.timeout_ghidra": (-1, "Ghidra analysis timeout in seconds (-1 for unlimited)"),
            "pipeline.timeout_compilation": (900, "Compilation timeout in seconds"),
            "pipeline.max_retries": (3, "Maximum retry attempts for failed agents"),
            "pipeline.memory_limit_per_agent": ("512MB", "Memory limit per agent"),
            
            # Ghidra defaults
            "ghidra.max_memory": ("4G", "Maximum memory for Ghidra JVM"),
            "ghidra.timeout": (-1, "Ghidra operation timeout (-1 for unlimited)"),
            
            # AI Engine defaults
            "ai_engine.provider": ("langchain", "AI engine provider"),
            "ai_engine.model": ("gpt-3.5-turbo", "AI model to use"),
            "ai_engine.temperature": (0.1, "AI model temperature"),
            "ai_engine.max_tokens": (2048, "Maximum tokens per AI request"),
            
            # Path defaults
            "paths.default_output_dir": ("output", "Default output directory"),
            "paths.temp_dir": ("temp", "Temporary files directory"),
            "paths.log_dir": ("logs", "Log files directory"),
            "paths.timestamp_format": ("%Y%m%d-%H%M%S", "Timestamp format for output directories"),
            
            # Matrix defaults
            "matrix.master_agent": ("deus_ex_machina", "Master agent name"),
            "matrix.parallel_agents": (16, "Number of parallel agents"),
            "matrix.execution_mode": ("master_first_parallel", "Execution mode"),
            "matrix.agent_independence": (True, "Enable agent independence"),
            "matrix.shared_memory_enabled": (True, "Enable shared memory between agents"),
            
            # Neo Agent defaults
            "agents.agent_05.enable_multithreading": (True, "Enable multithreading for Neo agent"),
            "agents.agent_05.max_threads": (4, "Maximum worker threads for Neo agent")
        }
        
        for key, (value, description) in defaults.items():
            self.config_values[key] = ConfigValue(value, ConfigSource.DEFAULT, description)
            
    def _load_config_file(self):
        """Load configuration from YAML or JSON file"""
        if not self.config_file or not Path(self.config_file).exists():
            return
            
        with open(self.config_file, 'r') as f:
            if self.config_file.endswith(('.yaml', '.yml')):
                data = yaml.safe_load(f)
            elif self.config_file.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {self.config_file}")
        
        self._flatten_config_dict(data, ConfigSource.CONFIG_FILE)
        self.logger.info(f"Loaded configuration from {self.config_file}")
    
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Pipeline environment variables
            "MATRIX_PARALLEL_MODE": "pipeline.parallel_mode",
            "MATRIX_BATCH_SIZE": "pipeline.batch_size",
            "MATRIX_AGENT_TIMEOUT": "pipeline.timeout_agent",
            "MATRIX_GHIDRA_TIMEOUT": "pipeline.timeout_ghidra", 
            "MATRIX_COMPILATION_TIMEOUT": "pipeline.timeout_compilation",
            "MATRIX_MAX_RETRIES": "pipeline.max_retries",
            "MATRIX_MEMORY_LIMIT": "pipeline.memory_limit_per_agent",
            
            # Ghidra environment variables
            "GHIDRA_HOME": "ghidra.install_path",
            "GHIDRA_MAX_MEMORY": "ghidra.max_memory",
            "GHIDRA_TIMEOUT": "ghidra.timeout",
            "JAVA_HOME": "ghidra.java_path",
            
            # AI Engine environment variables
            "AI_PROVIDER": "ai_engine.provider",
            "AI_MODEL": "ai_engine.model",
            "AI_TEMPERATURE": "ai_engine.temperature",
            "AI_MAX_TOKENS": "ai_engine.max_tokens",
            "AI_API_KEY": "ai_engine.api_key",
            "AI_BASE_URL": "ai_engine.base_url",
            
            # Path environment variables
            "MATRIX_OUTPUT_DIR": "paths.default_output_dir",
            "MATRIX_TEMP_DIR": "paths.temp_dir",
            "MATRIX_LOG_DIR": "paths.log_dir",
            
            # Matrix environment variables
            "MATRIX_EXECUTION_MODE": "matrix.execution_mode",
            "MATRIX_PARALLEL_AGENTS": "matrix.parallel_agents",
            
            # Neo Agent environment variables
            "NEO_ENABLE_MULTITHREADING": "agents.agent_05.enable_multithreading",
            "NEO_MAX_THREADS": "agents.agent_05.max_threads"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Type conversion
                converted_value = self._convert_env_value(env_value, config_key)
                self.config_values[config_key] = ConfigValue(
                    converted_value, 
                    ConfigSource.ENVIRONMENT,
                    f"Set from environment variable {env_var}"
                )
    
    def _flatten_config_dict(self, data: Dict[str, Any], source: ConfigSource, prefix: str = ""):
        """Flatten nested dictionary with dot notation"""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_config_dict(value, source, full_key)
            else:
                self.config_values[full_key] = ConfigValue(value, source, f"From {source.value}")
    
    def _convert_env_value(self, value: str, config_key: str) -> Any:
        """Convert environment variable string to appropriate type with validation."""
        # Sanitize input value
        if not isinstance(value, str):
            raise ValueError(f"Invalid environment value type: {type(value)}")
            
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion for known integer fields
        int_fields = ['batch_size', 'timeout_agent', 'timeout_ghidra', 'timeout_compilation', 
                     'max_retries', 'timeout', 'max_tokens', 'parallel_agents']
        if any(field in config_key for field in int_fields):
            try:
                int_val = int(value)
                # Allow -1 for unlimited timeouts
                if 'timeout' in config_key and int_val == -1:
                    return -1
                return int_val
            except ValueError:
                pass
        
        # Float conversion for known float fields
        float_fields = ['temperature']
        if any(field in config_key for field in float_fields):
            try:
                return float(value)
            except ValueError:
                pass
        
        return value
    
    def _resolve_paths(self):
        """Resolve and validate file system paths"""
        # Resolve Ghidra installation path
        ghidra_path = self.get_value("ghidra.install_path")
        if not ghidra_path:
            # Try to auto-detect Ghidra
            ghidra_path = self._auto_detect_ghidra()
            if ghidra_path:
                self.config_values["ghidra.install_path"] = ConfigValue(
                    ghidra_path, ConfigSource.DEFAULT, "Auto-detected Ghidra installation"
                )
        
        # Set headless script path if Ghidra found
        if ghidra_path:
            headless_script = Path(ghidra_path) / "support" / "analyzeHeadless"
            if headless_script.exists():
                self.config_values["ghidra.headless_script"] = ConfigValue(
                    str(headless_script), ConfigSource.DEFAULT, "Auto-detected Ghidra headless script"
                )
    
    def _auto_detect_ghidra(self) -> Optional[str]:
        """Auto-detect Ghidra installation safely with path validation."""
        # SECURITY FIX: Validate all paths to prevent directory traversal
        try:
            project_root_resolved = self.project_root.resolve()
            
            # Only check relative to project root for security
            ghidra_relative = project_root_resolved / "ghidra"
            possible_locations = []
            
            # Validate that ghidra path is actually under project root
            if str(ghidra_relative.resolve()).startswith(str(project_root_resolved)):
                possible_locations.append(ghidra_relative)
            
            # Add system paths only if explicitly allowed via environment
            if os.environ.get('MATRIX_ALLOW_SYSTEM_GHIDRA', '').lower() == 'true':
                # Validate system paths to prevent path traversal
                system_paths = [
                    Path("C:\\ghidra"),
                    Path("C:\\Program Files\\ghidra"),
                    Path("C:\\Program Files (x86)\\ghidra")
                ]
                for path in system_paths:
                    try:
                        resolved_path = path.resolve()
                        # Only allow paths under standard Windows directories
                        if str(resolved_path).startswith("C:\\"):
                            possible_locations.append(resolved_path)
                    except (OSError, ValueError):
                        # Skip invalid paths
                        continue
            
            for location in possible_locations:
                if location and location.exists():
                    try:
                        # Validate headless script path
                        headless = location / "support" / "analyzeHeadless"
                        headless_resolved = headless.resolve()
                        
                        # Ensure headless script is under the ghidra directory
                        if (str(headless_resolved).startswith(str(location.resolve())) 
                            and headless.exists()):
                            self.logger.info(f"Auto-detected Ghidra at: {location}")
                            return str(location)
                    except (OSError, ValueError):
                        # Skip invalid or dangerous paths
                        continue
            
        except (OSError, ValueError) as e:
            self.logger.warning(f"Path validation error during Ghidra detection: {e}")
        
        return None
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key with validation."""
        # Validate key format to prevent injection
        if not isinstance(key, str) or not key.replace('.', '').replace('_', '').isalnum():
            raise ValueError(f"Invalid configuration key format: {key}")
            
        config_value = self.config_values.get(key)
        return config_value.value if config_value else default
    
    def get_config_value(self, key: str) -> Optional[ConfigValue]:
        """Get configuration value object with metadata"""
        return self.config_values.get(key)
    
    def get_timeout(self, component: str, default: int = -1) -> int:
        """Get timeout for specific component (-1 means unlimited)"""
        timeout_mapping = {
            "agent": "pipeline.timeout_agent",
            "ghidra": "pipeline.timeout_ghidra", 
            "compilation": "pipeline.timeout_compilation",
            "advanced_decompiler": "pipeline.timeout_ghidra"
        }
        
        timeout_key = timeout_mapping.get(component, "pipeline.timeout_agent")
        timeout_value = self.get_value(timeout_key, default)
        # Return -1 for unlimited timeout, otherwise return the configured value
        return timeout_value if timeout_value != -1 else -1
    
    def get_limit(self, limit_type: str, default: Any = None) -> Any:
        """Get limit value for specific type"""
        limit_mapping = {
            "batch_size": "pipeline.batch_size",
            "max_retries": "pipeline.max_retries",
            "memory": "pipeline.memory_limit_per_agent",
            "parallel_agents": "matrix.parallel_agents"
        }
        
        limit_key = limit_mapping.get(limit_type)
        return self.get_value(limit_key, default) if limit_key else default
    
    def get_path(self, path_type: str, default: Optional[str] = None) -> Optional[str]:
        """Get path for specific type"""
        path_mapping = {
            "ghidra_home": "ghidra.install_path",
            "default_output_dir": "paths.default_output_dir",
            "temp_dir": "paths.temp_dir",
            "log_dir": "paths.log_dir",
            "headless_script": "ghidra.headless_script"
        }
        
        path_key = path_mapping.get(path_type)
        return self.get_value(path_key, default) if path_key else default
    
    def get_output_structure(self) -> Dict[str, str]:
        """Get output directory structure"""
        return {
            'agents': 'agents',
            'ghidra': 'ghidra', 
            'compilation': 'compilation',
            'reports': 'reports',
            'logs': 'logs',
            'temp': 'temp',
            'tests': 'tests',
            'docs': 'docs'
        }
    
    def get_output_path(self, binary_name: str, timestamp: Optional[str] = None) -> Path:
        """
        Get output path with new structure: output/{binary_name}/{yyyymmdd-hhmmss}/
        
        Args:
            binary_name: Name of the binary being analyzed
            timestamp: Optional timestamp, if not provided, current time is used
            
        Returns:
            Path object for the output directory
        """
        import time
        from pathlib import Path
        
        if timestamp is None:
            timestamp_format = self.get_value('paths.timestamp_format', '%Y%m%d-%H%M%S')
            timestamp = time.strftime(timestamp_format)
        
        base_output_dir = self.get_path('default_output_dir', 'output')
        return Path(base_output_dir) / binary_name / timestamp
    
    def get_structured_output_path(self, binary_name: str, subdir: str, timestamp: Optional[str] = None) -> Path:
        """
        Get structured output path for specific subdirectory
        
        Args:
            binary_name: Name of the binary being analyzed
            subdir: Subdirectory name (agents, ghidra, compilation, etc.)
            timestamp: Optional timestamp, if not provided, current time is used
            
        Returns:
            Path object for the specific output subdirectory
        """
        base_path = self.get_output_path(binary_name, timestamp)
        structure = self.get_output_structure()
        
        if subdir not in structure:
            raise ValueError(f"Invalid subdirectory: {subdir}. Valid options: {list(structure.keys())}")
            
        return base_path / structure[subdir]
    
    def print_configuration_summary(self):
        """Print configuration summary for debugging"""
        print("=" * 60)
        print("Open-Sourcefy Configuration Summary")
        print("=" * 60)
        
        categories = {
            "Pipeline": ["pipeline.parallel_mode", "pipeline.batch_size", "pipeline.timeout_agent"],
            "Ghidra": ["ghidra.install_path", "ghidra.max_memory", "ghidra.timeout"],
            "AI Engine": ["ai_engine.provider", "ai_engine.model", "ai_engine.temperature"],
            "Paths": ["paths.default_output_dir", "paths.temp_dir", "paths.log_dir"],
            "Matrix": ["matrix.execution_mode", "matrix.parallel_agents", "matrix.agent_independence"]
        }
        
        for category, keys in categories.items():
            print(f"\n{category}:")
            for key in keys:
                config_val = self.get_config_value(key)
                if config_val:
                    source_indicator = {"environment": "🌍", "config_file": "📄", "default": "⚙️"}
                    indicator = source_indicator.get(config_val.source.value, "❓")
                    print(f"  {indicator} {key}: {config_val.value}")
        
        print("\n" + "=" * 60)


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: Optional[str] = None, project_root: Optional[Path] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file, project_root)
    return _config_manager


def reset_config_manager():
    """Reset global configuration manager (for testing)"""
    global _config_manager
    _config_manager = None