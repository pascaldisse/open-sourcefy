# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-Sourcefy is an AI-powered binary decompilation system that reconstructs compilable C source code from Windows PE executables using a 16-agent pipeline with Ghidra integration. The primary test target is the Matrix Online launcher.exe binary.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment and dependencies
python main.py --verify-env

# Clean up temporary files and reset output
python main.py --cleanup
```

### Running the Pipeline
```bash
# Full pipeline (auto-detects binary from input/ directory)
python main.py

# Specific binary
python main.py launcher.exe

# Pipeline components
python main.py launcher.exe --decompile-only     # Agents 1,2,5,7,14
python main.py launcher.exe --analyze-only       # Agents 1,2,3,4,5,6,7,8,9,14,15
python main.py launcher.exe --compile-only       # Agents 1,2,4,5,6,7,8,9,10,11,12,18
python main.py launcher.exe --validate-only      # Agents 1,2,4,5,6,7,8,9,10,11,12,13,19

# Specific agents
python main.py launcher.exe --agent 7            # Single agent
python main.py launcher.exe --agents 1,3,7       # Multiple agents
python main.py launcher.exe --agents 1-5         # Agent ranges

# Parallel execution
python main.py launcher.exe --batch-size 6 --parallel-mode process --timeout 600
```

### Testing and Validation
```bash
# Environment validation with detailed output
python main.py --verify-env

# List available components and agents
python main.py --list-components
python main.py --list-agents

# Phase status check
python main.py --phase-status
```

## Architecture Overview

### Agent-Based Pipeline System

The system uses a **16-agent pipeline** with dependency-based execution:

**Core Agents (1-15)**:
- **Discovery & Analysis** (1-5): Binary format detection, architecture analysis, structure parsing
- **Decompilation** (6-7, 14): Optimization matching, Ghidra decompilation, enhanced analysis  
- **Processing** (8-10): Binary diff analysis, assembly analysis, resource reconstruction
- **Reconstruction** (11): Global code reconstruction with AI enhancement
- **Compilation & Validation** (12-13): MSBuild orchestration, final validation
- **Metadata** (15): Comprehensive metadata extraction

**Extension Agents** (18-20):
- **Advanced Build Systems** (18): Extended compilation support
- **Binary Comparison** (19): Binary difference analysis
- **Automated Testing** (20): Test framework generation

### Execution Pipeline

1. **Dependency Resolution**: Agents organized into batches based on dependencies
2. **Parallel Execution**: Configurable parallel processing within batches  
3. **Context Sharing**: Global execution context passed between agents
4. **Validation Checkpoints**: Pipeline-level validation at critical stages
5. **Quality Assessment**: Built-in quality scoring and validation thresholds

### Configuration System

- **Centralized Config**: `core/config_manager.py` manages all configuration
- **Environment Detection**: Auto-discovery of Ghidra, Visual Studio, compilers
- **Agent-Specific Settings**: Individual timeout, memory, retry configurations
- **Output Structure**: Organized subdirectories (agents/, ghidra/, compilation/, reports/, logs/, temp/, tests/)

### Key Components

**Agent Framework** (`core/agent_base.py`):
- Base class for all agents with standardized interfaces
- Dependency management and execution ordering
- Result objects with status, data, and metadata

**Parallel Executor** (`core/parallel_executor.py`):
- Manages concurrent agent execution
- Configurable execution modes (thread/process)
- Resource management and timeout handling

**Ghidra Integration** (`core/ghidra_headless.py`):
- Automated headless decompilation
- Custom script management for enhanced accuracy
- Quality assessment and confidence scoring

**Pipeline Orchestrator** (`main.py` - `OpenSourcefyPipeline`):
- Main execution controller
- Component-based pipeline modes
- Comprehensive reporting and validation

### Output Organization

All output is organized under `output/[timestamp]/`:
```
output/
├── agents/          # Agent-specific analysis outputs
├── ghidra/          # Ghidra decompilation results
├── compilation/     # MSBuild artifacts and generated source
├── reports/         # Pipeline execution reports
├── logs/            # Execution logs and debug information
├── temp/            # Temporary files (auto-cleaned)
└── tests/           # Generated test files
```

## Development Guidelines

### Agent Development

When creating or modifying agents:

1. **Inherit from BaseAgent** (`core/agent_base.py`)
2. **Implement required methods**: `execute()`, `get_description()`, `get_dependencies()`
3. **Use AgentResult objects** for standardized return values
4. **Access context data** via `context.get('agent_results', {})` for previous agent outputs
5. **Use output_paths** from context for organized file placement

### Configuration

- **Environment variables**: Check `core/config_manager.py` for available settings
- **Tool paths**: Use relative path resolution from project root
- **Output structure**: Always use `context['output_paths']` for file placement

### Quality Validation

The system implements strict validation thresholds:
- **Code Quality**: 75% threshold for meaningful code structure
- **Implementation Score**: 75% threshold for real vs placeholder code  
- **Completeness**: 70% threshold for project completeness

### Dependencies

**Required**:
- Python 3.8+
- Java 17+ (for Ghidra)
- Ghidra (included in ghidra/ directory)

**Optional**:
- Microsoft Visual C++ Build Tools (for compilation testing)
- WSL/Linux environment (for optimal performance)

### Current Development Status

- **Overall Quality**: 66.5% (Phase 3 completion)
- **Agent Success Rate**: 16/16 agents functional
- **Primary Target**: Matrix Online launcher.exe (5.3MB, x86 PE32, MSVC .NET 2003)

### Testing Approach

- The system includes built-in validation at multiple stages
- Agent results are validated for quality and completeness
- Pipeline can terminate early if validation thresholds are not met
- Use `--verify-env` to check environment setup before running

### Ghidra Integration

- Ghidra path resolution: Uses relative path from project root (`ghidra/`)
- Custom scripts located in agent implementations
- Headless analysis with quality assessment
- Temporary project management and cleanup