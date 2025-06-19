# Configuration Guide

Complete guide for configuring Open-Sourcefy Matrix pipeline settings, build systems, and environment variables.

## Configuration Files Overview

Open-Sourcefy uses multiple configuration files for different aspects of the system:

- **`config.yaml`**: Main application configuration
- **`build_config.yaml`**: Build system and tool paths
- **`CLAUDE.md`**: Project command center and development settings
- **Environment Variables**: Runtime configuration and API keys

## Main Configuration (config.yaml)

### Default Configuration Structure

```yaml
# Main application settings
application:
  name: "Open-Sourcefy Matrix Pipeline"
  version: "2.0"
  debug_mode: false
  log_level: "INFO"

# Agent execution settings
agents:
  timeout: 300  # seconds
  retry_count: 2
  parallel_execution: true
  max_parallel_agents: 4
  quality_threshold: 0.75
  fail_fast: true

# Pipeline execution settings
pipeline:
  execution_mode: "production"  # production, development, debug
  validation_level: "comprehensive"  # basic, standard, comprehensive
  cache_results: true
  cleanup_temp_files: true
  
# Ghidra integration settings
ghidra:
  enabled: true
  headless_timeout: 600
  custom_scripts: true
  decompilation_timeout: 60
  analysis_timeout: 300
  java_heap_size: "4G"

# Output configuration
output:
  structured_dirs: true
  compression: false
  cleanup_temp: true
  preserve_logs: true
  max_output_size: "10G"

# AI integration settings
ai:
  enabled: true
  provider: "anthropic"
  timeout: 30
  retry_attempts: 3
  fallback_mode: true

# Security settings
security:
  input_validation: true
  secure_temp_files: true
  sanitize_paths: true
  restrict_file_access: true
```

### Agent-Specific Configuration

```yaml
agents:
  # Global settings
  timeout: 300
  retry_count: 2
  
  # Per-agent settings
  agent_1:
    name: "Sentinel"
    timeout: 120
    import_analysis_depth: "comprehensive"
    security_scan_level: "high"
    
  agent_5:
    name: "Neo"
    ghidra_timeout: 600
    decompilation_quality: "high"
    type_inference: true
    
  agent_9:
    name: "Commander Locke"
    compilation_timeout: 900
    build_system: "msbuild"
    optimization_level: "O2"
    
  agent_15:
    name: "Analyst"
    analysis_depth: "comprehensive_enhanced"
    quality_threshold: 0.85
    predictive_analysis: true
    documentation_automation: true
```

## Build System Configuration (build_config.yaml)

### Visual Studio 2022 Preview Configuration

```yaml
build_system:
  # Primary build system
  type: "visual_studio"
  version: "2022_preview"
  
  # Visual Studio paths
  visual_studio:
    installation_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview"
    edition: "Preview"
    version: "17.0"
    
  # MSBuild configuration
  msbuild:
    path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/MSBuild/Current/Bin/MSBuild.exe"
    version: "17.0"
    platform_toolset: "v143"
    windows_sdk_version: "10.0.22000.0"

# Build tools configuration
build_tools:
  # Compiler (cl.exe)
  cl_exe_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.40.33807/bin/Hostx64/x64/cl.exe"
  
  # Resource Compiler (rc.exe)
  rc_exe_path: "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64/rc.exe"
  
  # Library Tool (lib.exe)
  lib_exe_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.40.33807/bin/Hostx64/x64/lib.exe"
  
  # Manifest Tool (mt.exe)
  mt_exe_path: "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64/mt.exe"
  
  # Linker (link.exe)
  link_exe_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.40.33807/bin/Hostx64/x64/link.exe"

# Compilation settings
compilation:
  # Target platform
  target_platform: "x64"
  configuration: "Release"
  
  # Compiler flags
  compiler_flags:
    - "/O2"          # Optimize for speed
    - "/GL"          # Whole program optimization
    - "/MD"          # Multithreaded DLL runtime
    - "/EHsc"        # C++ exception handling
    - "/W3"          # Warning level 3
    
  # Linker flags
  linker_flags:
    - "/LTCG"        # Link-time code generation
    - "/OPT:REF"     # Eliminate unreferenced functions
    - "/OPT:ICF"     # Identical COMDAT folding
    - "/SUBSYSTEM:CONSOLE"  # Console application
    - "/MACHINE:X64" # Target x64 architecture

# Validation settings
validation:
  verify_paths_on_startup: true
  test_compilation: true
  fail_on_missing_tools: true
  strict_mode: true  # No fallback paths
```

### Alternative Build System Support

```yaml
# CMake configuration (alternative/additional)
cmake:
  enabled: true
  path: "C:/Program Files/CMake/bin/cmake.exe"
  generator: "Visual Studio 17 2022"
  architecture: "x64"
  
# Ninja build system (fast builds)
ninja:
  enabled: false
  path: "C:/tools/ninja/ninja.exe"
  
# MinGW-w64 (GCC for Windows)
mingw:
  enabled: false
  gcc_path: "C:/mingw64/bin/gcc.exe"
  gxx_path: "C:/mingw64/bin/g++.exe"
```

## Environment Variables

### Required Variables

```bash
# AI Integration (optional but recommended)
export ANTHROPIC_API_KEY="your_api_key_here"

# Ghidra Integration
export GHIDRA_HOME="/path/to/ghidra"
export JAVA_HOME="/path/to/java"

# Windows Build Tools (if not in standard locations)
export VS2022_PATH="C:/Program Files/Microsoft Visual Studio/2022/Preview"
export WINDOWS_SDK_PATH="C:/Program Files (x86)/Windows Kits/10"
```

### Optional Debug Variables

```bash
# Debug and development
export MATRIX_DEBUG=true
export MATRIX_VERBOSE=true
export MATRIX_PROFILE=true

# AI configuration
export MATRIX_AI_ENABLED=true
export MATRIX_AI_TIMEOUT=30
export MATRIX_AI_FALLBACK=true

# Pipeline behavior
export MATRIX_FAIL_FAST=true
export MATRIX_CACHE_ENABLED=true
export MATRIX_PARALLEL_AGENTS=4

# Custom paths
export MATRIX_TEMP_DIR="/custom/temp/path"
export MATRIX_OUTPUT_DIR="/custom/output/path"
export MATRIX_CONFIG_PATH="/custom/config/path"
```

### Performance Tuning Variables

```bash
# Memory management
export MATRIX_MEMORY_LIMIT="16G"
export MATRIX_JAVA_HEAP="4G"
export MATRIX_GHIDRA_MEMORY="8G"

# CPU utilization
export MATRIX_MAX_THREADS=8
export MATRIX_PARALLEL_COMPILATION=true

# I/O optimization
export MATRIX_ASYNC_IO=true
export MATRIX_BUFFER_SIZE="64KB"
```

## Platform-Specific Configuration

### Windows Configuration

```yaml
# Windows-specific settings
platform:
  type: "windows"
  version: "10"  # or "11"
  
windows:
  # Console settings
  console_encoding: "utf-8"
  enable_ansi_colors: true
  
  # Path settings
  use_short_paths: false
  max_path_length: 260
  
  # Security settings
  execution_policy: "restricted"
  require_admin: false
```

### Linux/WSL Configuration

```yaml
# Linux/WSL settings
platform:
  type: "linux"
  wsl_version: "2"  # if running under WSL
  
linux:
  # Wine configuration for Windows tools
  wine:
    enabled: true
    prefix: "~/.wine_openSourcefy"
    windows_version: "win10"
    
  # Alternative tools
  alternatives:
    use_mono: true  # For .NET functionality
    use_mingw: true  # For Windows compilation
    
  # Path translation
  wsl_path_translation: true
  windows_drive_mapping: "/mnt/c"
```

### macOS Configuration

```yaml
# macOS settings
platform:
  type: "macos"
  version: "12.0"
  
macos:
  # Xcode tools
  xcode_tools_path: "/Applications/Xcode.app/Contents/Developer"
  
  # Homebrew tools
  homebrew_prefix: "/opt/homebrew"
  
  # Compatibility layers
  use_parallels: false
  use_vmware: false
```

## Advanced Configuration

### Performance Optimization

```yaml
performance:
  # Execution optimization
  enable_jit_compilation: true
  preload_libraries: true
  optimize_memory_usage: true
  
  # Caching strategy
  cache:
    enable_result_cache: true
    cache_directory: "~/.openSourcefy/cache"
    max_cache_size: "5G"
    cache_retention_days: 30
    
  # Parallel processing
  parallel:
    max_agents_parallel: 4
    enable_numa_awareness: true
    cpu_affinity: "auto"
    
  # I/O optimization
  io:
    use_async_io: true
    buffer_size: 65536
    prefetch_enabled: true
```

### Security Configuration

```yaml
security:
  # Input validation
  validation:
    strict_path_validation: true
    sanitize_file_names: true
    validate_binary_signatures: true
    
  # Execution security
  execution:
    sandbox_enabled: false  # Requires additional setup
    restrict_network_access: false
    limit_file_system_access: true
    
  # Logging security
  logging:
    sanitize_logs: true
    exclude_sensitive_data: true
    log_access_attempts: true
```

### Logging Configuration

```yaml
logging:
  # Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: "INFO"
  
  # Log destinations
  destinations:
    console: true
    file: true
    syslog: false
    
  # File logging
  file:
    path: "logs/openSourcefy.log"
    max_size: "100MB"
    backup_count: 5
    rotation: "daily"
    
  # Format
  format:
    include_timestamp: true
    include_agent_id: true
    include_thread_id: true
    
  # Agent-specific logging
  agents:
    enable_per_agent_logs: true
    log_directory: "logs/agents"
    debug_level_agents: [1, 5, 9]  # Extra debug for specific agents
```

## Configuration Validation

### Automatic Validation

```bash
# Validate all configuration
python main.py --validate-config

# Validate specific configuration files
python main.py --validate-config --config-file build_config.yaml

# Check environment variables
python main.py --validate-env

# Test configuration with dry run
python main.py --dry-run --debug
```

### Manual Validation

```python
from src.core.config_manager import ConfigManager

# Load and validate configuration
config = ConfigManager()
validation_result = config.validate_configuration()

if not validation_result.is_valid:
    print("Configuration validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
    
# Check specific paths
build_config = config.get_build_config()
if not build_config.validate_paths():
    print("Build tool paths are invalid")
```

### Configuration Templates

#### Minimal Configuration
```yaml
# Minimal working configuration
agents:
  timeout: 300
  
output:
  structured_dirs: true
  
build_system:
  type: "visual_studio"
  visual_studio:
    installation_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview"
```

#### Development Configuration
```yaml
# Development-focused configuration
application:
  debug_mode: true
  log_level: "DEBUG"
  
agents:
  timeout: 600  # Longer timeouts for debugging
  fail_fast: false
  
pipeline:
  execution_mode: "development"
  validation_level: "comprehensive"
  cache_results: false  # Always fresh results
  
logging:
  level: "DEBUG"
  agents:
    enable_per_agent_logs: true
    debug_level_agents: [1, 5, 9, 15, 16]
```

#### Production Configuration
```yaml
# Production-optimized configuration
application:
  debug_mode: false
  log_level: "INFO"
  
agents:
  timeout: 300
  fail_fast: true
  parallel_execution: true
  max_parallel_agents: 8
  
performance:
  enable_jit_compilation: true
  optimize_memory_usage: true
  cache:
    enable_result_cache: true
    max_cache_size: "10G"
    
security:
  validation:
    strict_path_validation: true
    validate_binary_signatures: true
  logging:
    sanitize_logs: true
    exclude_sensitive_data: true
```

## Troubleshooting Configuration

### Common Configuration Issues

#### Path Problems
```yaml
# Wrong paths (common mistakes)
build_tools:
  cl_exe_path: "C:/Program Files/Microsoft Visual Studio/2019/Community/..."  # Wrong version
  rc_exe_path: "C:/Program Files (x86)/Windows Kits/8.1/..."  # Old SDK

# Correct paths
build_tools:
  cl_exe_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.40.33807/bin/Hostx64/x64/cl.exe"
  rc_exe_path: "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64/rc.exe"
```

#### Permission Issues
```bash
# Check file permissions
ls -la config.yaml
chmod 644 config.yaml  # Read/write for owner, read for others

# Check directory permissions
ls -la logs/
chmod 755 logs/  # Full access for owner, read/execute for others
```

#### Environment Variable Issues
```bash
# Check if variables are set
echo $ANTHROPIC_API_KEY
echo $GHIDRA_HOME

# Set variables properly
export ANTHROPIC_API_KEY="sk-..."  # Ensure quotes for special characters
export GHIDRA_HOME="/opt/ghidra"   # No trailing slash
```

### Configuration Debugging

```bash
# Show current configuration
python main.py --config-summary

# Validate configuration with detailed output
python main.py --validate-config --verbose

# Test specific configuration sections
python main.py --test-build-config
python main.py --test-agent-config
python main.py --test-ai-config
```

---

**Related**: [[Getting Started|Getting-Started]] - Initial setup and installation  
**Next**: [[Troubleshooting]] - Configuration problem resolution