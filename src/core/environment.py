"""
Enhanced Environment Management for Open-Sourcefy
Phase 1 Implementation: Environment Optimization & Infrastructure
"""

import os
import sys
import logging
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

# Third-party imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class EnvironmentStatus(Enum):
    """Environment validation status"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of environment validation check"""
    component: str
    status: EnvironmentStatus
    message: str
    details: Dict[str, Any] = None
    fix_suggestion: Optional[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class EnvironmentConfig:
    """Environment configuration container"""
    # Core paths
    project_root: Path
    ghidra_home: Optional[Path] = None
    vs_base: Optional[Path] = None
    sdk_base: Optional[Path] = None
    
    # Compiler settings
    msvc_version: Optional[str] = None
    compiler_flags: List[str] = None
    
    # Performance settings
    max_memory_gb: int = 8
    max_parallel_agents: int = 4
    timeout_seconds: int = 1800
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    def __post_init__(self):
        if self.compiler_flags is None:
            self.compiler_flags = []


class EnhancedEnvironmentManager:
    """Enhanced environment management for Open-Sourcefy"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config = self._load_configuration()
        self.logger = self._setup_logging()
        self.validation_results: List[ValidationResult] = []
        
    def _load_configuration(self) -> EnvironmentConfig:
        """Load configuration from environment variables and config files"""
        config = EnvironmentConfig(project_root=self.project_root)
        
        # Load from environment variables
        config.ghidra_home = self._get_path_env("GHIDRA_HOME", self.project_root / "ghidra")
        config.vs_base = self._get_path_env("VS_BASE")
        config.sdk_base = self._get_path_env("SDK_BASE")
        config.msvc_version = os.environ.get("MSVC_VERSION", "14.44.35207")
        
        # Performance settings from environment
        config.max_memory_gb = int(os.environ.get("OPENSOURCEFY_MAX_MEMORY", "8"))
        config.max_parallel_agents = int(os.environ.get("OPENSOURCEFY_PARALLEL_JOBS", "4"))
        config.timeout_seconds = int(os.environ.get("OPENSOURCEFY_TIMEOUT", "1800"))
        config.log_level = os.environ.get("OPENSOURCEFY_LOG_LEVEL", "INFO")
        
        # Try to load from config file
        config_file = self.project_root / "config.yaml"
        if config_file.exists() and YAML_AVAILABLE:
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    self._merge_file_config(config, file_config)
            except Exception as e:
                # Config file exists but can't be loaded - not critical
                pass
                
        return config
    
    def _get_path_env(self, env_var: str, default: Optional[Path] = None) -> Optional[Path]:
        """Get path from environment variable with proper handling"""
        value = os.environ.get(env_var)
        if value:
            path = Path(value)
            # Ensure Windows paths only
            if platform.system() != "Windows":
                raise ValueError("Only Windows environment is supported")
            return path
        return default
    
    def _merge_file_config(self, config: EnvironmentConfig, file_config: Dict):
        """Merge configuration from file into environment config"""
        if "environment" in file_config:
            env_config = file_config["environment"]
            if "ghidra_home" in env_config:
                config.ghidra_home = Path(env_config["ghidra_home"])
            if "max_memory_gb" in env_config:
                config.max_memory_gb = env_config["max_memory_gb"]
            if "max_parallel_agents" in env_config:
                config.max_parallel_agents = env_config["max_parallel_agents"]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup enhanced logging"""
        logger = logging.getLogger("Environment")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler if specified
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(console_formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def validate_environment(self, comprehensive: bool = True) -> Tuple[bool, List[ValidationResult]]:
        """Perform comprehensive environment validation"""
        self.logger.info("Starting environment validation...")
        self.validation_results = []
        
        # Core validation
        self._validate_python_environment()
        self._validate_core_dependencies()
        self._validate_project_structure()
        
        if comprehensive:
            # Extended validation
            self._validate_ghidra_installation()
            self._validate_compiler_environment()
            self._validate_system_resources()
            self._validate_network_connectivity()
        
        # Analyze results
        has_critical_errors = any(r.status == EnvironmentStatus.CRITICAL for r in self.validation_results)
        has_errors = any(r.status == EnvironmentStatus.ERROR for r in self.validation_results)
        
        success = not has_critical_errors and not has_errors
        
        self.logger.info(f"Environment validation completed. Success: {success}")
        return success, self.validation_results
    
    def _validate_python_environment(self):
        """Validate Python environment"""
        # Python version
        if sys.version_info >= (3, 8):
            self._add_result("Python Version", EnvironmentStatus.VALID, 
                           f"Python {sys.version_info.major}.{sys.version_info.minor} is supported")
        else:
            self._add_result("Python Version", EnvironmentStatus.CRITICAL,
                           f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported",
                           fix_suggestion="Upgrade to Python 3.8 or higher")
        
        # Python path
        python_path = sys.executable
        self._add_result("Python Executable", EnvironmentStatus.VALID,
                        f"Python executable: {python_path}")
    
    def _validate_core_dependencies(self):
        """Validate core Python dependencies"""
        core_deps = [
            ("json", "JSON support"),
            ("subprocess", "Process execution"),
            ("pathlib", "Path handling"),
            ("threading", "Threading support"),
            ("multiprocessing", "Multiprocessing support")
        ]
        
        for module, description in core_deps:
            try:
                __import__(module)
                self._add_result(f"Dependency: {module}", EnvironmentStatus.VALID,
                               f"{description} available")
            except ImportError:
                self._add_result(f"Dependency: {module}", EnvironmentStatus.CRITICAL,
                               f"{description} not available",
                               fix_suggestion=f"Install or update Python to include {module}")
        
        # Optional dependencies
        optional_deps = [
            ("psutil", "System monitoring", "pip install psutil"),
            ("yaml", "YAML configuration", "pip install PyYAML"),
            ("pefile", "PE file analysis", "pip install pefile"),
            ("capstone", "Disassembly engine", "pip install capstone")
        ]
        
        for module, description, install_cmd in optional_deps:
            try:
                __import__(module)
                self._add_result(f"Optional: {module}", EnvironmentStatus.VALID,
                               f"{description} available")
            except ImportError:
                self._add_result(f"Optional: {module}", EnvironmentStatus.WARNING,
                               f"{description} not available",
                               fix_suggestion=install_cmd)
    
    def _validate_project_structure(self):
        """Validate project structure"""
        required_paths = [
            ("Project Root", self.project_root),
            ("Source Directory", self.project_root / "src"),
            ("Docs Directory", self.project_root / "docs"),
            ("Main Script", self.project_root / "main.py"),
            ("Requirements", self.project_root / "requirements.txt")
        ]
        
        for name, path in required_paths:
            if path.exists():
                self._add_result(f"Structure: {name}", EnvironmentStatus.VALID,
                               f"{name} exists at {path}")
            else:
                self._add_result(f"Structure: {name}", EnvironmentStatus.ERROR,
                               f"{name} not found at {path}",
                               fix_suggestion=f"Create missing {name}")
    
    def _validate_ghidra_installation(self):
        """Validate Ghidra installation"""
        if not self.config.ghidra_home:
            self._add_result("Ghidra", EnvironmentStatus.WARNING,
                           "Ghidra home not configured",
                           fix_suggestion="Set GHIDRA_HOME environment variable")
            return
        
        ghidra_path = self.config.ghidra_home
        if not ghidra_path.exists():
            self._add_result("Ghidra", EnvironmentStatus.ERROR,
                           f"Ghidra not found at {ghidra_path}",
                           fix_suggestion="Install Ghidra or update GHIDRA_HOME")
            return
        
        # Check for essential Ghidra components
        headless_analyzer = ghidra_path / "support" / "analyzeHeadless"
        if headless_analyzer.exists():
            self._add_result("Ghidra Headless", EnvironmentStatus.VALID,
                           "Ghidra headless analyzer found")
        else:
            self._add_result("Ghidra Headless", EnvironmentStatus.ERROR,
                           "Ghidra headless analyzer not found")
        
        # Check Java for Ghidra
        try:
            result = subprocess.run(["java", "-version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self._add_result("Java for Ghidra", EnvironmentStatus.VALID,
                               "Java runtime available for Ghidra")
            else:
                self._add_result("Java for Ghidra", EnvironmentStatus.ERROR,
                               "Java runtime not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._add_result("Java for Ghidra", EnvironmentStatus.ERROR,
                           "Java runtime not found",
                           fix_suggestion="Install Java 17 or higher")
    
    def _validate_compiler_environment(self):
        """Validate compiler environment for Windows/MSVC only"""
        if platform.system() != "Windows":
            self._add_result("Compiler Environment", EnvironmentStatus.CRITICAL,
                           "Non-Windows environment detected - only Windows with VS MSBuild is supported",
                           fix_suggestion="Use Windows environment with Visual Studio installed")
            return
        
        # Check for MSVC paths (Windows environment only)
        if self.config.vs_base:
            vs_path = self.config.vs_base
            if vs_path.exists():
                self._add_result("Visual Studio", EnvironmentStatus.VALID,
                               f"Visual Studio found at {vs_path}")
                
                # Check for specific compiler
                if self.config.msvc_version:
                    compiler_path = vs_path / "VC" / "Tools" / "MSVC" / self.config.msvc_version / "bin" / "Hostx64" / "x86" / "cl.exe"
                    if compiler_path.exists():
                        self._add_result("MSVC Compiler", EnvironmentStatus.VALID,
                                       "MSVC compiler found")
                    else:
                        self._add_result("MSVC Compiler", EnvironmentStatus.ERROR,
                                       f"MSVC compiler not found at {compiler_path}")
            else:
                self._add_result("Visual Studio", EnvironmentStatus.ERROR,
                               f"Visual Studio not found at {vs_path}")
        else:
            self._add_result("Visual Studio", EnvironmentStatus.WARNING,
                           "Visual Studio path not configured")
    
    def _validate_system_resources(self):
        """Validate system resources"""
        if not PSUTIL_AVAILABLE:
            self._add_result("System Resources", EnvironmentStatus.WARNING,
                           "psutil not available - system resource validation skipped")
            return
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= 8:
            self._add_result("System Memory", EnvironmentStatus.VALID,
                           f"Sufficient memory: {memory_gb:.1f}GB available")
        elif memory_gb >= 4:
            self._add_result("System Memory", EnvironmentStatus.WARNING,
                           f"Limited memory: {memory_gb:.1f}GB available",
                           fix_suggestion="Consider upgrading to 8GB+ for optimal performance")
        else:
            self._add_result("System Memory", EnvironmentStatus.ERROR,
                           f"Insufficient memory: {memory_gb:.1f}GB available",
                           fix_suggestion="Upgrade to at least 4GB RAM")
        
        # CPU check
        cpu_count = psutil.cpu_count()
        self._add_result("CPU Cores", EnvironmentStatus.VALID,
                        f"{cpu_count} CPU cores available")
        
        # Disk space check
        disk = psutil.disk_usage(str(self.project_root))
        disk_gb = disk.free / (1024**3)
        
        if disk_gb >= 10:
            self._add_result("Disk Space", EnvironmentStatus.VALID,
                           f"Sufficient disk space: {disk_gb:.1f}GB free")
        else:
            self._add_result("Disk Space", EnvironmentStatus.WARNING,
                           f"Limited disk space: {disk_gb:.1f}GB free",
                           fix_suggestion="Free up disk space for temporary files")
    
    def _validate_network_connectivity(self):
        """Validate network connectivity (optional)"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            self._add_result("Network Connectivity", EnvironmentStatus.VALID,
                           "Network connectivity available")
        except (socket.error, OSError):
            self._add_result("Network Connectivity", EnvironmentStatus.WARNING,
                           "Network connectivity not available",
                           fix_suggestion="Check network connection for online resources")
    
    def _add_result(self, component: str, status: EnvironmentStatus, 
                   message: str, fix_suggestion: Optional[str] = None,
                   details: Optional[Dict] = None):
        """Add validation result"""
        result = ValidationResult(
            component=component,
            status=status,
            message=message,
            fix_suggestion=fix_suggestion,
            details=details or {}
        )
        self.validation_results.append(result)
        
        # Log based on status
        if status == EnvironmentStatus.VALID:
            self.logger.debug(f"✓ {component}: {message}")
        elif status == EnvironmentStatus.WARNING:
            self.logger.warning(f"⚠ {component}: {message}")
        elif status == EnvironmentStatus.ERROR:
            self.logger.error(f"✗ {component}: {message}")
        elif status == EnvironmentStatus.CRITICAL:
            self.logger.critical(f"✗ {component}: {message}")
    
    def is_valid(self) -> bool:
        """Check if environment is valid based on last validation"""
        if not self.validation_results:
            # No validation has been run yet, perform basic validation
            success, _ = self.validate_environment(comprehensive=False)
            return success
        
        # Check for critical errors or errors in existing validation results
        has_critical_errors = any(r.status == EnvironmentStatus.CRITICAL for r in self.validation_results)
        has_errors = any(r.status == EnvironmentStatus.ERROR for r in self.validation_results)
        
        return not has_critical_errors and not has_errors
    
    def get_missing_components(self) -> List[str]:
        """Get list of missing components from validation results"""
        if not self.validation_results:
            return []
        
        return [r.component for r in self.validation_results 
                if r.status in [EnvironmentStatus.ERROR, EnvironmentStatus.CRITICAL]]

    def print_validation_report(self):
        """Print comprehensive validation report"""
        print("\n" + "="*60)
        print("Environment Validation Report")
        print("="*60)
        
        status_counts = {status: 0 for status in EnvironmentStatus}
        
        for result in self.validation_results:
            status_counts[result.status] += 1
            
            # Status icon
            icon = {
                EnvironmentStatus.VALID: "✓",
                EnvironmentStatus.WARNING: "⚠",
                EnvironmentStatus.ERROR: "✗",
                EnvironmentStatus.CRITICAL: "✗"
            }[result.status]
            
            print(f"{icon} {result.component}: {result.message}")
            if result.fix_suggestion:
                print(f"  → Fix: {result.fix_suggestion}")
        
        print("\n" + "-"*60)
        print("Summary:")
        for status, count in status_counts.items():
            if count > 0:
                print(f"  {status.value.title()}: {count}")
        
        # Overall status
        if status_counts[EnvironmentStatus.CRITICAL] > 0:
            print("\n🔴 CRITICAL ISSUES FOUND - System not ready for use")
        elif status_counts[EnvironmentStatus.ERROR] > 0:
            print("\n🟠 ERRORS FOUND - Some features may not work")
        elif status_counts[EnvironmentStatus.WARNING] > 0:
            print("\n🟡 WARNINGS FOUND - System ready with limitations")
        else:
            print("\n🟢 ALL CHECKS PASSED - System ready for use")
        
        print("="*60)
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get optimized performance configuration"""
        config = {
            "max_memory_gb": self.config.max_memory_gb,
            "max_parallel_agents": self.config.max_parallel_agents,
            "timeout_seconds": self.config.timeout_seconds
        }
        
        # Adjust based on system capabilities
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            # Adjust memory limit based on available memory
            recommended_memory = min(self.config.max_memory_gb, int(memory_gb * 0.8))
            config["max_memory_gb"] = recommended_memory
            
            # Adjust parallel agents based on CPU cores
            recommended_agents = min(self.config.max_parallel_agents, max(1, cpu_count - 1))
            config["max_parallel_agents"] = recommended_agents
        
        return config
    
    def create_config_template(self, output_path: Optional[Path] = None):
        """Create configuration template file"""
        if not YAML_AVAILABLE:
            self.logger.warning("PyYAML not available - cannot create config template")
            return
        
        template = {
            "environment": {
                "ghidra_home": str(self.config.ghidra_home) if self.config.ghidra_home else "/path/to/ghidra",
                "max_memory_gb": 8,
                "max_parallel_agents": 4,
                "timeout_seconds": 1800,
                "log_level": "INFO"
            },
            "pipeline": {
                "target_accuracy": 0.99,
                "max_iterations": 50,
                "batch_size": 4
            },
            "agents": {
                "ghidra": {
                    "timeout": 1800,
                    "memory_limit": "8G"
                }
            }
        }
        
        output_path = output_path or self.project_root / "config.template.yaml"
        
        with open(output_path, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration template created at {output_path}")


# Convenience functions for backward compatibility
def validate_environment() -> bool:
    """Simple environment validation for backward compatibility"""
    env_manager = EnhancedEnvironmentManager()
    success, _ = env_manager.validate_environment(comprehensive=False)
    return success


def get_environment():
    """Get environment manager instance"""
    return EnhancedEnvironmentManager()


def print_environment_report():
    """Print comprehensive environment report"""
    env_manager = EnhancedEnvironmentManager()
    env_manager.validate_environment(comprehensive=True)
    env_manager.print_validation_report()


if __name__ == "__main__":
    # Command-line interface for environment validation
    import argparse
    
    parser = argparse.ArgumentParser(description="Open-Sourcefy Environment Validation")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Perform comprehensive validation")
    parser.add_argument("--create-config", action="store_true",
                       help="Create configuration template")
    
    args = parser.parse_args()
    
    env_manager = EnhancedEnvironmentManager()
    
    if args.create_config:
        env_manager.create_config_template()
    
    if args.comprehensive:
        env_manager.validate_environment(comprehensive=True)
    else:
        env_manager.validate_environment(comprehensive=False)
    
    env_manager.print_validation_report()