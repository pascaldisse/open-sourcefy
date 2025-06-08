"""
Centralized Configuration Management for Open-Sourcefy
Phase 1 Implementation: Configuration Management
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime

# Third-party imports with fallbacks
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ConfigFormat(Enum):
    """Supported configuration file formats"""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


@dataclass
class EnvironmentConfig:
    """Environment-related configuration"""
    # Core paths
    project_root: str = ""
    ghidra_home: str = ""
    vs_base: str = ""
    sdk_base: str = ""
    msvc_version: str = "14.44.35207"
    
    # Performance settings
    max_memory_gb: int = 8
    max_parallel_agents: int = 4
    timeout_seconds: int = 1800
    
    # Logging
    log_level: str = "INFO"
    log_file: str = ""
    
    # Validation
    strict_validation: bool = False
    auto_fix_environment: bool = True


@dataclass
class PipelineConfig:
    """Pipeline execution configuration"""
    target_accuracy: float = 0.99
    max_iterations: int = 50
    batch_size: int = 4
    parallel_mode: str = "thread"  # thread, process
    timeout_per_agent: int = 300
    retry_enabled: bool = True
    max_retries: int = 3
    continue_on_failure: bool = True
    
    # Output configuration
    save_intermediate: bool = True
    generate_reports: bool = True
    compress_output: bool = False


@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    enabled: bool = True
    timeout: int = 300
    retry_count: int = 3
    memory_limit: str = "2G"
    cpu_limit: float = 1.0
    priority: int = 1
    dependencies: List[int] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GhidraConfig:
    """Ghidra-specific configuration"""
    enabled: bool = True
    timeout: int = 1800
    memory_limit: str = "8G"
    script_timeout: int = 600
    analysis_timeout: int = 1200
    headless_args: List[str] = field(default_factory=lambda: [
        "-noanalysis", "-scriptTimeout", "600"
    ])
    custom_scripts_path: str = ""
    project_cleanup: bool = True


@dataclass
class CompilationConfig:
    """Compilation-related configuration"""
    enabled: bool = True
    compiler: str = "msvc"  # msvc, gcc, clang
    optimization_level: str = "O2"
    target_architecture: str = "x86"
    debug_symbols: bool = False
    warnings_as_errors: bool = False
    custom_flags: List[str] = field(default_factory=list)
    library_paths: List[str] = field(default_factory=list)
    include_paths: List[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    """Validation and testing configuration"""
    enabled: bool = True
    accuracy_threshold: float = 0.99
    performance_threshold: float = 0.95
    memory_threshold_mb: int = 8192
    timeout_threshold: int = 3600
    
    # Test configuration
    run_unit_tests: bool = True
    run_integration_tests: bool = True
    run_performance_tests: bool = False
    generate_coverage_report: bool = False


@dataclass
class OpenSourcefyConfig:
    """Complete Open-Sourcefy configuration"""
    # Configuration metadata
    version: str = "1.0.0"
    created_date: str = ""
    last_modified: str = ""
    
    # Component configurations
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    ghidra: GhidraConfig = field(default_factory=GhidraConfig)
    compilation: CompilationConfig = field(default_factory=CompilationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Agent configurations (keyed by agent ID)
    agents: Dict[int, AgentConfig] = field(default_factory=dict)
    
    # Custom configurations
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        self.last_modified = datetime.now().isoformat()
        
        # Initialize default agent configurations if empty
        if not self.agents:
            self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default configurations for all 13 agents"""
        agent_defaults = {
            1: AgentConfig(timeout=60, memory_limit="512M"),    # Binary Discovery
            2: AgentConfig(timeout=120, memory_limit="1G"),     # Architecture Analysis
            3: AgentConfig(timeout=300, memory_limit="1G"),     # Error Pattern Matching
            4: AgentConfig(timeout=240, memory_limit="1G"),     # Optimization Detection
            5: AgentConfig(timeout=180, memory_limit="2G"),     # Binary Structure
            6: AgentConfig(timeout=300, memory_limit="2G"),     # Optimization Matching
            7: AgentConfig(timeout=1800, memory_limit="8G"),    # Ghidra Decompilation
            8: AgentConfig(timeout=600, memory_limit="4G"),     # Binary Diff Analysis
            9: AgentConfig(timeout=900, memory_limit="4G"),     # Assembly Analysis
            10: AgentConfig(timeout=300, memory_limit="2G"),    # Resource Reconstruction
            11: AgentConfig(timeout=1200, memory_limit="6G"),   # AI Enhancement
            12: AgentConfig(timeout=1800, memory_limit="4G"),   # Integration Testing
            13: AgentConfig(timeout=600, memory_limit="2G"),    # Final Validation
        }
        
        for agent_id, config in agent_defaults.items():
            self.agents[agent_id] = config


class ConfigurationManager:
    """Centralized configuration management system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.logger = logging.getLogger("ConfigManager")
        
        # Configuration storage
        self.config: OpenSourcefyConfig = OpenSourcefyConfig()
        self.config_file_path: Optional[Path] = None
        self.config_format: ConfigFormat = ConfigFormat.YAML if YAML_AVAILABLE else ConfigFormat.JSON
        
        # Configuration search paths
        self.config_search_paths = [
            self.project_root / "config.yaml",
            self.project_root / "config.yml", 
            self.project_root / "config.json",
            self.project_root / ".opensourcefy.yaml",
            self.project_root / ".opensourcefy.yml",
            self.project_root / ".opensourcefy.json",
            Path.home() / ".opensourcefy" / "config.yaml",
            Path.home() / ".opensourcefy" / "config.json"
        ]
        
        # Load configuration
        self._load_configuration()
        self._apply_environment_overrides()
    
    def _load_configuration(self):
        """Load configuration from files and environment"""
        # Try to find and load existing configuration file
        config_loaded = False
        
        for config_path in self.config_search_paths:
            if config_path.exists():
                try:
                    self._load_config_file(config_path)
                    self.config_file_path = config_path
                    config_loaded = True
                    self.logger.info(f"Loaded configuration from {config_path}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        if not config_loaded:
            self.logger.info("No configuration file found, using defaults")
            # Set project root in default config
            self.config.environment.project_root = str(self.project_root)
    
    def _load_config_file(self, config_path: Path):
        """Load configuration from a specific file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to load YAML configuration files")
                data = yaml.safe_load(f)
                self.config_format = ConfigFormat.YAML
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
                self.config_format = ConfigFormat.JSON
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Merge loaded data with default configuration
        self._merge_config_data(data)
    
    def _merge_config_data(self, data: Dict[str, Any]):
        """Merge loaded configuration data with current config"""
        if not data:
            return
        
        # Update metadata
        if 'version' in data:
            self.config.version = data['version']
        if 'created_date' in data:
            self.config.created_date = data['created_date']
        
        # Update component configurations
        for component in ['environment', 'pipeline', 'ghidra', 'compilation', 'validation']:
            if component in data and isinstance(data[component], dict):
                component_config = getattr(self.config, component)
                self._update_dataclass_from_dict(component_config, data[component])
        
        # Update agent configurations
        if 'agents' in data and isinstance(data['agents'], dict):
            for agent_id_str, agent_data in data['agents'].items():
                try:
                    agent_id = int(agent_id_str)
                    if agent_id not in self.config.agents:
                        self.config.agents[agent_id] = AgentConfig()
                    self._update_dataclass_from_dict(self.config.agents[agent_id], agent_data)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid agent ID in config: {agent_id_str}: {e}")
        
        # Update custom configurations
        if 'custom' in data and isinstance(data['custom'], dict):
            self.config.custom.update(data['custom'])
    
    def _update_dataclass_from_dict(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass object from dictionary data"""
        for key, value in data.items():
            if hasattr(obj, key):
                # Get the field type for validation
                field_type = type(getattr(obj, key))
                
                # Handle type conversion
                try:
                    if field_type == bool and isinstance(value, str):
                        # Convert string to boolean
                        value = value.lower() in ['true', '1', 'yes', 'on']
                    elif field_type == int and isinstance(value, str):
                        value = int(value)
                    elif field_type == float and isinstance(value, (str, int)):
                        value = float(value)
                    elif field_type == list and isinstance(value, str):
                        # Convert comma-separated string to list
                        value = [item.strip() for item in value.split(',')]
                    
                    setattr(obj, key, value)
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to set {key} = {value}: {e}")
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            # Environment configuration
            'OPENSOURCEFY_PROJECT_ROOT': ('environment', 'project_root'),
            'GHIDRA_HOME': ('environment', 'ghidra_home'),
            'VS_BASE': ('environment', 'vs_base'),
            'SDK_BASE': ('environment', 'sdk_base'),
            'MSVC_VERSION': ('environment', 'msvc_version'),
            'OPENSOURCEFY_MAX_MEMORY': ('environment', 'max_memory_gb'),
            'OPENSOURCEFY_PARALLEL_JOBS': ('environment', 'max_parallel_agents'),
            'OPENSOURCEFY_TIMEOUT': ('environment', 'timeout_seconds'),
            'OPENSOURCEFY_LOG_LEVEL': ('environment', 'log_level'),
            'OPENSOURCEFY_LOG_FILE': ('environment', 'log_file'),
            
            # Pipeline configuration
            'OPENSOURCEFY_TARGET_ACCURACY': ('pipeline', 'target_accuracy'),
            'OPENSOURCEFY_BATCH_SIZE': ('pipeline', 'batch_size'),
            'OPENSOURCEFY_PARALLEL_MODE': ('pipeline', 'parallel_mode'),
            'OPENSOURCEFY_MAX_RETRIES': ('pipeline', 'max_retries'),
            
            # Ghidra configuration
            'OPENSOURCEFY_GHIDRA_TIMEOUT': ('ghidra', 'timeout'),
            'OPENSOURCEFY_GHIDRA_MEMORY': ('ghidra', 'memory_limit'),
        }
        
        for env_var, (component, field) in env_mappings.items():
            if env_var in os.environ:
                try:
                    component_obj = getattr(self.config, component)
                    current_value = getattr(component_obj, field)
                    env_value = os.environ[env_var]
                    
                    # Convert to appropriate type
                    if isinstance(current_value, bool):
                        env_value = env_value.lower() in ['true', '1', 'yes', 'on']
                    elif isinstance(current_value, int):
                        env_value = int(env_value)
                    elif isinstance(current_value, float):
                        env_value = float(env_value)
                    
                    setattr(component_obj, field, env_value)
                    self.logger.debug(f"Applied environment override: {env_var} -> {component}.{field} = {env_value}")
                
                except (ValueError, TypeError, AttributeError) as e:
                    self.logger.warning(f"Failed to apply environment override {env_var}: {e}")
    
    def get_config(self) -> OpenSourcefyConfig:
        """Get the current configuration"""
        return self.config
    
    def get_environment_config(self) -> EnvironmentConfig:
        """Get environment configuration"""
        return self.config.environment
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Get pipeline configuration"""
        return self.config.pipeline
    
    def get_agent_config(self, agent_id: int) -> Optional[AgentConfig]:
        """Get configuration for specific agent"""
        return self.config.agents.get(agent_id)
    
    def get_ghidra_config(self) -> GhidraConfig:
        """Get Ghidra configuration"""
        return self.config.ghidra
    
    def get_compilation_config(self) -> CompilationConfig:
        """Get compilation configuration"""
        return self.config.compilation
    
    def get_validation_config(self) -> ValidationConfig:
        """Get validation configuration"""
        return self.config.validation
    
    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get custom configuration value"""
        return self.config.custom.get(key, default)
    
    def set_config_value(self, component: str, field: str, value: Any):
        """Set a configuration value"""
        if hasattr(self.config, component):
            component_obj = getattr(self.config, component)
            if hasattr(component_obj, field):
                setattr(component_obj, field, value)
                self.config.last_modified = datetime.now().isoformat()
                self.logger.info(f"Updated configuration: {component}.{field} = {value}")
            else:
                raise AttributeError(f"Field '{field}' not found in component '{component}'")
        else:
            raise AttributeError(f"Component '{component}' not found")
    
    def set_agent_config(self, agent_id: int, config: AgentConfig):
        """Set configuration for specific agent"""
        self.config.agents[agent_id] = config
        self.config.last_modified = datetime.now().isoformat()
        self.logger.info(f"Updated agent {agent_id} configuration")
    
    def set_custom_config(self, key: str, value: Any):
        """Set custom configuration value"""
        self.config.custom[key] = value
        self.config.last_modified = datetime.now().isoformat()
        self.logger.info(f"Set custom configuration: {key} = {value}")
    
    def save_configuration(self, output_path: Optional[Path] = None, 
                         format: Optional[ConfigFormat] = None):
        """Save current configuration to file"""
        if output_path is None:
            output_path = self.config_file_path or (self.project_root / "config.yaml")
        
        if format is None:
            format = self.config_format
        
        # Convert configuration to dictionary
        config_dict = self._config_to_dict()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        with open(output_path, 'w', encoding='utf-8') as f:
            if format == ConfigFormat.YAML:
                if not YAML_AVAILABLE:
                    raise ImportError("PyYAML is required to save YAML configuration files")
                yaml.dump(config_dict, f, default_flow_style=False, indent=2, sort_keys=True)
            elif format == ConfigFormat.JSON:
                json.dump(config_dict, f, indent=2, sort_keys=True)
            else:
                raise ValueError(f"Unsupported configuration format: {format}")
        
        self.logger.info(f"Configuration saved to {output_path}")
        self.config_file_path = output_path
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {
            'version': self.config.version,
            'created_date': self.config.created_date,
            'last_modified': self.config.last_modified,
            'environment': asdict(self.config.environment),
            'pipeline': asdict(self.config.pipeline),
            'ghidra': asdict(self.config.ghidra),
            'compilation': asdict(self.config.compilation),
            'validation': asdict(self.config.validation),
            'agents': {str(k): asdict(v) for k, v in self.config.agents.items()},
            'custom': self.config.custom
        }
        
        return config_dict
    
    def create_config_template(self, output_path: Optional[Path] = None,
                             include_comments: bool = True):
        """Create a configuration template with comments"""
        if output_path is None:
            output_path = self.project_root / "config.template.yaml"
        
        # Create template configuration with default values
        template_config = OpenSourcefyConfig()
        
        if include_comments and YAML_AVAILABLE:
            # Create YAML with comments
            template_content = self._create_yaml_template_with_comments()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
        else:
            # Save as regular YAML/JSON
            config_dict = asdict(template_config)
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_path.suffix.lower() in ['.yaml', '.yml']:
                    if YAML_AVAILABLE:
                        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                    else:
                        json.dump(config_dict, f, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration template created at {output_path}")
    
    def _create_yaml_template_with_comments(self) -> str:
        """Create YAML template with detailed comments"""
        template = f"""# Open-Sourcefy Configuration File
# Version: {self.config.version}
# Generated: {datetime.now().isoformat()}

# Project version and metadata
version: "{self.config.version}"
created_date: "{self.config.created_date}"
last_modified: "{self.config.last_modified}"

# Environment configuration
environment:
  # Core paths (use absolute paths or leave empty for auto-detection)
  project_root: "{self.config.environment.project_root}"
  ghidra_home: "{self.config.environment.ghidra_home}"
  vs_base: "{self.config.environment.vs_base}"
  sdk_base: "{self.config.environment.sdk_base}"
  msvc_version: "{self.config.environment.msvc_version}"
  
  # Performance settings
  max_memory_gb: {self.config.environment.max_memory_gb}      # Maximum memory usage
  max_parallel_agents: {self.config.environment.max_parallel_agents}  # Number of parallel agents
  timeout_seconds: {self.config.environment.timeout_seconds}    # Global timeout
  
  # Logging configuration
  log_level: "{self.config.environment.log_level}"     # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: "{self.config.environment.log_file}"       # Empty for console only
  
  # Validation settings
  strict_validation: {str(self.config.environment.strict_validation).lower()}
  auto_fix_environment: {str(self.config.environment.auto_fix_environment).lower()}

# Pipeline execution configuration
pipeline:
  target_accuracy: {self.config.pipeline.target_accuracy}    # Target reproduction accuracy (0.0-1.0)
  max_iterations: {self.config.pipeline.max_iterations}      # Maximum refinement iterations
  batch_size: {self.config.pipeline.batch_size}              # Parallel execution batch size
  parallel_mode: "{self.config.pipeline.parallel_mode}"      # thread or process
  timeout_per_agent: {self.config.pipeline.timeout_per_agent} # Timeout per agent in seconds
  retry_enabled: {str(self.config.pipeline.retry_enabled).lower()}
  max_retries: {self.config.pipeline.max_retries}
  continue_on_failure: {str(self.config.pipeline.continue_on_failure).lower()}
  
  # Output settings
  save_intermediate: {str(self.config.pipeline.save_intermediate).lower()}
  generate_reports: {str(self.config.pipeline.generate_reports).lower()}
  compress_output: {str(self.config.pipeline.compress_output).lower()}

# Ghidra configuration
ghidra:
  enabled: {str(self.config.ghidra.enabled).lower()}
  timeout: {self.config.ghidra.timeout}                    # Ghidra analysis timeout
  memory_limit: "{self.config.ghidra.memory_limit}"        # Memory limit for Ghidra
  script_timeout: {self.config.ghidra.script_timeout}      # Script execution timeout
  analysis_timeout: {self.config.ghidra.analysis_timeout}  # Analysis timeout
  headless_args: {self.config.ghidra.headless_args}        # Additional Ghidra arguments
  custom_scripts_path: "{self.config.ghidra.custom_scripts_path}"
  project_cleanup: {str(self.config.ghidra.project_cleanup).lower()}

# Compilation configuration
compilation:
  enabled: {str(self.config.compilation.enabled).lower()}
  compiler: "{self.config.compilation.compiler}"           # msvc, gcc, clang
  optimization_level: "{self.config.compilation.optimization_level}"
  target_architecture: "{self.config.compilation.target_architecture}"
  debug_symbols: {str(self.config.compilation.debug_symbols).lower()}
  warnings_as_errors: {str(self.config.compilation.warnings_as_errors).lower()}
  custom_flags: {self.config.compilation.custom_flags}
  library_paths: {self.config.compilation.library_paths}
  include_paths: {self.config.compilation.include_paths}

# Validation configuration
validation:
  enabled: {str(self.config.validation.enabled).lower()}
  accuracy_threshold: {self.config.validation.accuracy_threshold}
  performance_threshold: {self.config.validation.performance_threshold}
  memory_threshold_mb: {self.config.validation.memory_threshold_mb}
  timeout_threshold: {self.config.validation.timeout_threshold}
  
  # Testing configuration
  run_unit_tests: {str(self.config.validation.run_unit_tests).lower()}
  run_integration_tests: {str(self.config.validation.run_integration_tests).lower()}
  run_performance_tests: {str(self.config.validation.run_performance_tests).lower()}
  generate_coverage_report: {str(self.config.validation.generate_coverage_report).lower()}

# Agent-specific configurations
agents:"""
        
        # Add agent configurations
        for agent_id in range(1, 14):
            if agent_id in self.config.agents:
                agent_config = self.config.agents[agent_id]
                template += f"""
  {agent_id}:  # Agent {agent_id}
    enabled: {str(agent_config.enabled).lower()}
    timeout: {agent_config.timeout}
    retry_count: {agent_config.retry_count}
    memory_limit: "{agent_config.memory_limit}"
    cpu_limit: {agent_config.cpu_limit}
    priority: {agent_config.priority}
    dependencies: {agent_config.dependencies}
    custom_settings: {agent_config.custom_settings}"""
        
        template += """

# Custom configurations (add your own key-value pairs here)
custom: {}
"""
        
        return template
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration and return any issues"""
        issues = []
        
        # Validate environment configuration
        env_config = self.config.environment
        if env_config.max_memory_gb < 1:
            issues.append("Environment: max_memory_gb must be at least 1")
        if env_config.max_parallel_agents < 1:
            issues.append("Environment: max_parallel_agents must be at least 1")
        if env_config.timeout_seconds < 60:
            issues.append("Environment: timeout_seconds should be at least 60")
        
        # Validate pipeline configuration
        pipeline_config = self.config.pipeline
        if not (0.0 <= pipeline_config.target_accuracy <= 1.0):
            issues.append("Pipeline: target_accuracy must be between 0.0 and 1.0")
        if pipeline_config.batch_size < 1:
            issues.append("Pipeline: batch_size must be at least 1")
        if pipeline_config.parallel_mode not in ["thread", "process"]:
            issues.append("Pipeline: parallel_mode must be 'thread' or 'process'")
        
        # Validate agent configurations
        for agent_id, agent_config in self.config.agents.items():
            if not (1 <= agent_id <= 13):
                issues.append(f"Agent {agent_id}: invalid agent ID (must be 1-13)")
            if agent_config.timeout < 10:
                issues.append(f"Agent {agent_id}: timeout should be at least 10 seconds")
            if agent_config.retry_count < 0:
                issues.append(f"Agent {agent_id}: retry_count cannot be negative")
        
        return issues
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = OpenSourcefyConfig()
        self.config.environment.project_root = str(self.project_root)
        self.logger.info("Configuration reset to defaults")


# Global configuration manager instance
_global_config_manager: Optional[ConfigurationManager] = None


def get_config_manager() -> ConfigurationManager:
    """Get or create global configuration manager"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigurationManager()
    return _global_config_manager


def get_config() -> OpenSourcefyConfig:
    """Get current configuration"""
    return get_config_manager().get_config()


def get_environment_config() -> EnvironmentConfig:
    """Get environment configuration"""
    return get_config_manager().get_environment_config()


def get_pipeline_config() -> PipelineConfig:
    """Get pipeline configuration"""
    return get_config_manager().get_pipeline_config()


def get_agent_config(agent_id: int) -> Optional[AgentConfig]:
    """Get agent configuration"""
    return get_config_manager().get_agent_config(agent_id)


if __name__ == "__main__":
    # Test configuration management
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Management Test")
    parser.add_argument("--create-template", action="store_true", 
                       help="Create configuration template")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate current configuration")
    parser.add_argument("--show", action="store_true", 
                       help="Show current configuration")
    
    args = parser.parse_args()
    
    config_manager = ConfigurationManager()
    
    if args.create_template:
        config_manager.create_config_template()
        print("Configuration template created")
    
    if args.validate:
        issues = config_manager.validate_configuration()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid")
    
    if args.show:
        config = config_manager.get_config()
        print("Current Configuration:")
        print(json.dumps(config_manager._config_to_dict(), indent=2))