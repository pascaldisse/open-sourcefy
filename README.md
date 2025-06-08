# Open-Sourcefy

**Advanced AI-Powered Binary Decompilation & Reconstruction System**

Open-Sourcefy is a sophisticated reverse engineering framework designed to reconstruct compilable source code from binary executables through intelligent decompilation and multi-agent analysis.

## Overview

This project implements a **15-agent AI-enhanced pipeline** with **Ghidra integration** and **machine learning capabilities** to transform binary executables back into readable, compilable C source code. The system is currently in active development with a focus on the **Matrix Online launcher.exe** binary as a primary test case.

### Key Features

- **🤖 15-Agent AI Pipeline**: Specialized agents for comprehensive binary analysis and reconstruction
- **🧠 Machine Learning Framework**: Pattern recognition, function classification, and code quality assessment
- **🔍 Ghidra Integration**: Automated headless decompilation with custom script support
- **⚡ Parallel Processing**: Multi-threaded and multi-process execution modes
- **🔄 Error Recovery**: Retry mechanisms and robust error handling
- **📊 Comprehensive Analysis**: Binary structure, optimization patterns, and resource reconstruction
- **🔧 Structured Output**: Organized output directory structure with detailed reports

## Architecture

### Pipeline Stages

The system operates through a structured multi-agent pipeline:

1. **Discovery & Analysis** (Agents 1-5): Binary format detection, architecture analysis, structure parsing, and optimization detection
2. **Decompilation & Processing** (Agents 6-10): Ghidra-based decompilation, pattern matching, and resource reconstruction
3. **Enhancement & Integration** (Agents 11-15): Code enhancement, build system generation, and comprehensive validation
4. **Extended Capabilities** (Agents 16-20): Advanced build systems, binary comparison, and automated testing

### Agent System

| Agent | Name | Purpose | Status |
|-------|------|---------|--------|
| 1 | Binary Discovery | Initial binary analysis and metadata extraction | 🔄 Active Development |
| 2 | Architecture Analysis | x86/x64 architecture and calling convention analysis | 🔄 Active Development |
| 3 | Smart Error Pattern Matching | ML-based error detection and pattern matching | 🔄 Active Development |
| 4 | Basic Decompiler | Initial decompilation and function identification | 🔄 Active Development |
| 5 | Binary Structure Analyzer | PE/ELF/Mach-O structure parsing and analysis | 🔄 Active Development |
| 6 | Optimization Matcher | Compiler optimization pattern detection and reversal | 🔄 Active Development |
| 7 | Advanced Decompiler | Ghidra-based comprehensive decompilation | ✅ Recently Enhanced |
| 8 | Binary Diff Analyzer | Binary difference analysis and validation | 🔄 Active Development |
| 9 | Advanced Assembly Analyzer | Deep assembly instruction analysis | 🔄 Active Development |
| 10 | Resource Reconstructor | Resource section extraction and reconstruction | 🔄 Active Development |
| 11 | Global Reconstructor | Code organization and structure reconstruction | ✅ Recently Enhanced |
| 12 | Compilation Orchestrator | Build system generation and compilation testing | 🔄 Active Development |
| 13 | Final Validator | Quality assurance and binary reproduction validation | 🔄 Active Development |
| 14 | Advanced Ghidra | Enhanced Ghidra capabilities and script integration | ✅ Recently Enhanced |
| 15 | Metadata Analysis | Comprehensive metadata extraction and analysis | ✅ Recently Enhanced |

## Installation

### Prerequisites

- **Python 3.8+** with pip package manager
- **Java 17+** (for Ghidra integration)
- **Microsoft Visual C++ Build Tools** (for compilation testing)
- **8GB+ RAM** (recommended for AI-enhanced analysis)
- **WSL/Linux environment** (for optimal performance)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/open-sourcefy.git
   cd open-sourcefy
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify environment and dependencies:**
   ```bash
   python main.py --verify-env
   ```

## Usage

### Basic Usage

```bash
# Full AI-enhanced pipeline with all 13 agents
python main.py launcher.exe

# Specific pipeline stages with AI enhancements
python main.py launcher.exe --decompile-only    # Agents 1,2,5,7
python main.py launcher.exe --analyze-only      # Agents 1,2,3,4,5,8,9
python main.py launcher.exe --compile-only      # Agents 6,11,12

# Individual agent execution
python main.py launcher.exe --agent 7           # Ghidra decompilation
python main.py launcher.exe --agents 1,3,7      # Multiple agents

# Enhanced parallel execution with AI monitoring
python main.py launcher.exe --batch-size 6 --parallel-mode process
```

### Advanced Options

```bash
# Enhanced timeout and retry configuration
python main.py launcher.exe --timeout 1800 --max-retries 5

# Advanced parallel processing modes
python main.py launcher.exe --parallel-mode process --batch-size 8

# AI enhancement configuration
python main.py launcher.exe --enable-ai-naming --enable-quality-assessment

# Environment validation and health monitoring
python main.py --verify-env --detailed

# Agent range execution
python main.py launcher.exe --agents 1-5  # Execute agents 1 through 5
```

## Target: Matrix Online Launcher

The system is being developed and tested with the **Matrix Online launcher.exe**:

- **File Size**: 5.3MB
- **Architecture**: x86 PE32
- **Compiler**: Microsoft Visual C++ 7.1 (MSVC .NET 2003)
- **Functions**: 2,099+ identified functions
- **Development Goal**: High-quality source reconstruction

### Current Development Status

**Overall Quality**: 66.5% (Phase 3 - Active Development)
- **Code Quality**: 30% (needs improvement)
- **Analysis Accuracy**: 60%
- **Agent Success Rate**: 16/16 agents functional (see CLAUDE.md for details)

## Documentation

### Core Documentation
- **[Project Documentation](docs/project_documentation.md)** - Comprehensive usage guide with examples and AI features
- **[Completion Report](docs/completion_report.md)** - 100% completion status report
- **[Ghidra Integration Guide](docs/ghidra_integration.md)** - Detailed Ghidra setup and usage

### Development Files
- **[CLAUDE.md](CLAUDE.md)** - Development instructions and project configuration
- **[requirements.txt](docs/requirements.txt)** - Python package dependencies

### Agent System Reference
The system includes 15 core agents plus extension agents (16-20) for comprehensive binary analysis:

**Discovery & Analysis (Agents 1-5)**
- Agent 1: Binary Discovery and metadata extraction
- Agent 2: Architecture analysis (x86/x64 focus)
- Agent 3: Smart error pattern matching with ML
- Agent 4: Basic decompilation and function identification
- Agent 5: Binary structure analysis (PE/ELF/Mach-O)

**Processing & Decompilation (Agents 6-10)**
- Agent 6: Optimization pattern matching and reversal
- Agent 7: Advanced Ghidra decompilation
- Agent 8: Binary difference analysis
- Agent 9: Advanced assembly instruction analysis
- Agent 10: Resource section reconstruction

**Enhancement & Validation (Agents 11-15)**
- Agent 11: Global code reconstruction and organization
- Agent 12: Compilation orchestration and build systems
- Agent 13: Final validation and quality assurance
- Agent 14: Advanced Ghidra capabilities and scripting
- Agent 15: Comprehensive metadata analysis

**Extension Agents (16-20)**
- Agent 16: Dynamic analysis bridge
- Agent 18: Advanced build system support
- Agent 19: Binary comparison and analysis
- Agent 20: Automated testing frameworks

## Project Status

### Current Development Phase

- ✅ **Phase 1**: Basic pipeline infrastructure and agent framework (Complete)
- ✅ **Phase 2**: Core agent implementations with basic functionality (Complete)
- 🔄 **Phase 3**: Dummy code removal and error handling improvement (In Progress)
- 📋 **Phase 4**: Full feature implementation and production readiness (Planned)

### Development Progress

- ✅ **Agent Framework**: 15 core agents + 5 extension agents implemented
- ✅ **Pipeline System**: Multi-stage execution with dependency management
- ✅ **CLI Interface**: Comprehensive command-line interface
- 🔄 **Code Quality**: Active cleanup of dummy implementations
- 🔄 **ML Integration**: Pattern recognition and quality assessment framework
- 📋 **Advanced Features**: Planned for Phase 4 implementation

### Current Quality Metrics

- **Overall Quality**: 66.5% (improving through Phase 3)
- **Code Quality**: 30% (needs improvement - active cleanup in progress)
- **Analysis Accuracy**: 60% (target: 80%+)
- **Agent Success Rate**: 16/16 agents functional
- **Pipeline Reliability**: Good (structured error handling)

## Development

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Testing

```bash
# Run full test suite
python -m pytest tests/

# Test specific agent
python -m pytest tests/test_agent_specific.py

# Integration tests
python main.py test --integration
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Enable development mode
export OPENsourcefy_DEV=1

# Run in debug mode
python main.py pipeline --binary binary.exe --output ./output --debug
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Ghidra**: NSA's reverse engineering framework
- **Matrix Online**: Testing target and inspiration
- **AI Research Community**: For advanced decompilation techniques

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/open-sourcefy/issues)
- **Documentation**: [Wiki](https://github.com/yourusername/open-sourcefy/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/open-sourcefy/discussions)

---

**Open-Sourcefy** - Transforming binaries back to source code with AI precision.