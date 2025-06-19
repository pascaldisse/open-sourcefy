# Getting Started with Open-Sourcefy

This guide will help you set up and run the Open-Sourcefy Matrix pipeline for binary decompilation.

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11 64-bit (Linux/WSL supported with limitations)
- **Memory**: 16GB+ RAM recommended for AI processing
- **Storage**: 5GB+ free space for pipeline operations
- **Python**: Python 3.9+ required

### Required Software
- **Visual Studio 2022 Preview**: Required for compilation (Windows only)
- **Java JDK 11+**: Required for Ghidra integration
- **Git**: For repository management

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/pascaldisse/open-sourcefy.git
cd open-sourcefy
```

### 2. Install Python Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Verify environment setup
python main.py --verify-env

# Check configuration
python main.py --config-summary
```

### 4. Download Ghidra (Optional)
```bash
# Download Ghidra 10.3+ from NSA GitHub
# Extract to preferred location
# Set GHIDRA_HOME environment variable
export GHIDRA_HOME=/path/to/ghidra
```

## Quick Start

### Basic Binary Analysis
```bash
# Analyze default binary (launcher.exe)
python main.py

# Analyze specific binary
python main.py path/to/binary.exe

# Full pipeline with all agents
python main.py --full-pipeline
```

### Pipeline Modes
```bash
# Decompilation only
python main.py --decompile-only

# Analysis without compilation
python main.py --analyze-only

# Compilation testing
python main.py --compile-only

# Debug mode with detailed logging
python main.py --debug --profile
```

### Agent Selection
```bash
# Run specific agents
python main.py --agents 1,3,7

# Run agent ranges
python main.py --agents 1-5

# List available agents
python main.py --list-agents
```

## Configuration

### Build System Configuration
Edit `build_config.yaml` to configure build tools:

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
```bash
# Required for AI functionality
export ANTHROPIC_API_KEY=your_api_key_here

# Optional debug settings
export MATRIX_DEBUG=true
export MATRIX_AI_ENABLED=true
export GHIDRA_HOME=/path/to/ghidra
export JAVA_HOME=/path/to/java
```

## Understanding Output

### Output Structure
```
output/{binary_name}/{timestamp}/
├── agents/          # Agent-specific outputs
├── ghidra/          # Decompilation results
├── compilation/     # MSBuild artifacts
├── reports/         # Pipeline reports
└── logs/            # Execution logs
```

### Key Output Files
- **comprehensive_metadata.json**: Complete analysis summary
- **execution_report.json**: Pipeline execution details
- **reconstructed_source/**: Generated C source code
- **build_files/**: MSBuild project files

## Verification

### Test Pipeline Success
```bash
# Run comprehensive tests
python -m unittest discover tests -v

# Verify specific functionality
python main.py --validate-pipeline basic

# Check system status
python main.py --verify-env
```

### Expected Results
- **Pipeline Success Rate**: 100% (16/16 agents)
- **Binary Output Size**: ~4.3MB for launcher.exe
- **Compilation Success**: Generated code should compile with VS2022
- **Size Accuracy**: ~83% of original binary size

## Common Issues

### Windows-Specific Issues
- **VS2022 Not Found**: Ensure Visual Studio 2022 Preview is installed
- **Build Tools Missing**: Install Windows SDK and MSVC build tools
- **Path Issues**: Verify all paths in build_config.yaml are correct

### Linux/WSL Issues
- **Limited Compilation**: Some Windows-specific tools unavailable
- **Path Translation**: Windows paths may need adjustment
- **Tool Emulation**: Some tools run through Wine/emulation

### Performance Issues
- **Memory Usage**: Ensure 16GB+ RAM for full AI processing
- **Disk Space**: Pipeline can generate several GB of temporary files
- **CPU Usage**: AI processing is CPU-intensive

## Next Steps

After successful installation:

1. **[[Run Your First Analysis|User-Guide#first-analysis]]**
2. **[[Understand the Architecture|Architecture-Overview]]**
3. **[[Explore Agent Capabilities|Agent-Documentation]]**
4. **[[Configure Advanced Settings|Configuration-Guide]]**

## Support

- **Issues**: [GitHub Issues](https://github.com/pascaldisse/open-sourcefy/issues)
- **Documentation**: [[Home]] for complete wiki navigation
- **Troubleshooting**: [[Troubleshooting]] for common problems

---

**Next**: [[User Guide|User-Guide]] - Learn how to use Open-Sourcefy effectively