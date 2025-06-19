# User Guide

Complete guide for using Open-Sourcefy to decompile Windows PE executables into compilable C source code.

## Overview

Open-Sourcefy provides multiple ways to analyze and decompile binary executables through its 17-agent Matrix pipeline. This guide covers all usage scenarios from basic analysis to advanced customization.

## Basic Usage

### Quick Analysis
```bash
# Analyze default binary (launcher.exe in input/ directory)
python main.py

# Analyze specific binary
python main.py path/to/binary.exe

# Analyze with custom output directory
python main.py binary.exe --output-dir my_analysis
```

### Pipeline Modes

#### Full Pipeline (Recommended)
```bash
# Run all 17 agents for complete analysis
python main.py --full-pipeline

# Full pipeline with debug logging
python main.py --full-pipeline --debug
```

#### Specialized Modes
```bash
# Decompilation only (agents 1,2,5,7,14)
python main.py --decompile-only

# Analysis without compilation (agents 1,2,3,4,5,6,7,8,9,14,15)
python main.py --analyze-only

# Compilation testing only (agents 1,2,4,5,6,7,8,9,10,11,12,13)
python main.py --compile-only
```

### Agent Selection

#### Individual Agents
```bash
# Run single agent
python main.py --agents 1

# Run multiple specific agents
python main.py --agents 1,3,7,15

# Run agent ranges
python main.py --agents 1-5
python main.py --agents 10-16
```

#### Agent Information
```bash
# List all available agents
python main.py --list-agents

# Show agent dependencies
python main.py --show-dependencies

# Display agent descriptions
python main.py --describe-agents
```

## Advanced Usage

### Environment and Configuration

#### Environment Validation
```bash
# Verify all dependencies and paths
python main.py --verify-env

# Show current configuration summary
python main.py --config-summary

# Validate specific components
python main.py --verify-ghidra
python main.py --verify-build-system
```

#### Development Options
```bash
# Dry run (show execution plan without running)
python main.py --dry-run

# Debug mode with comprehensive logging
python main.py --debug --profile

# Benchmark mode with performance metrics
python main.py --benchmark --profile
```

### Pipeline Validation

#### Quality Validation
```bash
# Basic pipeline validation
python main.py --validate-pipeline basic

# Comprehensive validation with all checks
python main.py --validate-pipeline comprehensive

# Agent-specific validation
python main.py --validate-agent 5
```

#### Update and Maintenance
```bash
# Update mode for incremental development
python main.py --update

# Force reprocessing (ignore cached results)
python main.py --force-reprocess

# Clean temporary files
python main.py --clean-temp
```

## Understanding Output

### Output Directory Structure

When you run Open-Sourcefy, it creates a structured output directory:

```
output/{binary_name}/{timestamp}/
├── agents/              # Individual agent outputs
│   ├── agent_01_sentinel/
│   ├── agent_02_architect/
│   └── ... (all agents)
├── ghidra/              # Ghidra decompilation results
│   ├── launcher.exe.gzf    # Ghidra project file
│   ├── decompiled/         # Decompiled C code
│   └── analysis/           # Analysis results
├── compilation/         # Generated source and build files
│   ├── src/               # Reconstructed C source code
│   ├── include/           # Header files
│   ├── resources/         # Extracted resources
│   └── CMakeLists.txt     # Build configuration
├── reports/             # Pipeline execution reports
│   ├── execution_report.json
│   ├── quality_report.json
│   └── agent_summary.json
└── logs/                # Detailed execution logs
    ├── pipeline.log
    ├── agents/           # Per-agent logs
    └── debug/            # Debug information
```

### Key Output Files

#### Comprehensive Reports
- **`reports/execution_report.json`**: Complete pipeline execution summary
- **`reports/quality_report.json`**: Quality metrics and validation results
- **`reports/agent_summary.json`**: Individual agent performance and results

#### Generated Source Code
- **`compilation/src/`**: Reconstructed C source files
- **`compilation/include/`**: Generated header files
- **`compilation/resources/`**: Extracted and converted resources

#### Build Files
- **`compilation/CMakeLists.txt`**: CMake build configuration
- **`compilation/Makefile`**: Generated Makefile for compilation
- **`compilation/project.vcxproj`**: Visual Studio project file

### Success Indicators

#### Pipeline Success
Look for these indicators of successful execution:

```json
{
  "pipeline_status": "SUCCESS",
  "agents_completed": 16,
  "agents_failed": 0,
  "overall_quality": 0.85,
  "compilation_status": "SUCCESS",
  "binary_size_accuracy": 0.8336
}
```

#### Output Quality Metrics
- **Compilation Success**: Generated code compiles without errors
- **Size Accuracy**: ~83% of original binary size achieved
- **Function Recovery**: 538+ functions successfully identified
- **Resource Extraction**: Icons, dialogs, strings extracted

## Working with Results

### Compilation Testing

#### Using Generated Build Files
```bash
# Navigate to compilation directory
cd output/launcher/20250619-123456/compilation/

# Compile using CMake
cmake .
make

# Or using Visual Studio (Windows)
msbuild project.vcxproj /p:Configuration=Release
```

#### Verification
```bash
# Compare binary sizes
ls -la original_binary.exe
ls -la output/compilation/launcher.exe

# Test functionality (if applicable)
./output/compilation/launcher.exe
```

### Code Review

#### Generated Source Analysis
- **Main Function**: Look in `src/main.c` for program entry point
- **Function Definitions**: Individual functions in `src/functions/`
- **Data Structures**: Type definitions in `include/types.h`
- **Resources**: Converted resources in `resources/`

#### Quality Assessment
- **Code Comments**: Auto-generated documentation
- **Function Signatures**: Type-inferred parameters and returns
- **Control Flow**: Reconstructed program logic
- **Error Handling**: Identified exception patterns

## Customization

### Configuration Files

#### Main Configuration (`config.yaml`)
```yaml
agents:
  timeout: 300
  retry_count: 2
  parallel_execution: true

ghidra:
  headless_timeout: 600
  custom_scripts: true
  decompilation_timeout: 60

output:
  structured_dirs: true
  compression: false
  cleanup_temp: true
```

#### Build Configuration (`build_config.yaml`)
```yaml
build_system:
  visual_studio:
    version: "2022_preview"
    installation_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview"

build_tools:
  rc_exe_path: "C:/Program Files (x86)/Windows Kits/10/bin/x64/rc.exe"
  lib_exe_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.XX.XXXXX/bin/Hostx64/x64/lib.exe"
```

### Environment Variables

#### Required Variables
```bash
# AI functionality (optional but recommended)
export ANTHROPIC_API_KEY=your_api_key_here

# Ghidra integration
export GHIDRA_HOME=/path/to/ghidra
export JAVA_HOME=/path/to/java
```

#### Debug Variables
```bash
# Enable debug logging
export MATRIX_DEBUG=true

# Enable AI-enhanced analysis
export MATRIX_AI_ENABLED=true

# Custom temporary directory
export MATRIX_TEMP_DIR=/custom/temp/path
```

## Performance Optimization

### System Requirements
- **Memory**: 16GB+ RAM for optimal performance
- **CPU**: Multi-core processor recommended for parallel execution
- **Storage**: SSD recommended for improved I/O performance
- **Network**: Stable connection for AI service integration

### Performance Tuning
```bash
# Optimize for speed (fewer quality checks)
python main.py --fast

# Optimize for quality (more thorough analysis)
python main.py --thorough

# Parallel processing control
python main.py --max-parallel 4

# Memory usage control
python main.py --max-memory 8G
```

## Troubleshooting

### Common Issues

#### Pipeline Failures
```bash
# Check agent logs
cat output/binary/timestamp/logs/agents/agent_XX.log

# Run with debug logging
python main.py --debug

# Validate specific agent
python main.py --validate-agent X
```

#### Compilation Issues
```bash
# Verify build environment
python main.py --verify-build-system

# Check generated code
less output/binary/timestamp/compilation/src/main.c

# Test minimal compilation
cd output/binary/timestamp/compilation
gcc -c src/main.c
```

#### Resource Issues
```bash
# Monitor memory usage
python main.py --profile --debug

# Clean temporary files
python main.py --clean-temp

# Check disk space
df -h output/
```

### Getting Help

#### Log Analysis
- **Pipeline Logs**: `output/{binary}/{timestamp}/logs/pipeline.log`
- **Agent Logs**: `output/{binary}/{timestamp}/logs/agents/`
- **Debug Logs**: `output/{binary}/{timestamp}/logs/debug/`

#### Support Resources
- **Issues**: [GitHub Issues](https://github.com/pascaldisse/open-sourcefy/issues)
- **Documentation**: [[Troubleshooting]] for detailed problem resolution
- **Architecture**: [[Architecture Overview|Architecture-Overview]] for system understanding

---

**Next**: [[Configuration Guide|Configuration-Guide]] - Advanced configuration options  
**Related**: [[Troubleshooting]] - Problem resolution guide