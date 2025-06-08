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

### Matrix Agent-Based Pipeline System

The system uses a **4-agent Matrix pipeline** with production-ready, fail-fast architecture:

**Phase 1 - Foundation** (Agent 1):
- **Sentinel**: Binary discovery, metadata analysis, and security scanning
- Multi-format support (PE/ELF/Mach-O), hash calculation, entropy analysis
- LangChain AI integration for threat detection and binary insights

**Phase 2 - Core Analysis** (Agents 2-4):
- **The Architect**: Architecture analysis, compiler detection, optimization pattern recognition
- **The Merovingian**: Basic decompilation, function detection, control flow analysis
- **Agent Smith**: Binary structure analysis, data extraction, dynamic bridge preparation

**Future Phases** (Agents 5-16):
- **Phase 3**: Advanced decompilation and reconstruction (Neo, Twins, Trainman, Keymaker, Link)
- **Phase 4**: Final validation and compilation (Commander Locke, Machine, Oracle, Agent Johnson, Cleaner, Analyst, Agent Brown)

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

**Matrix Agent Framework** (`core/matrix_agents_v2.py`):
- Production-ready base classes with Matrix-themed architecture
- LangChain integration for AI-enhanced analysis
- Fail-fast validation with quality thresholds
- Comprehensive error handling and logging
- Shared components for code reuse and abstraction

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

### Matrix Agent Development

When creating or modifying Matrix agents:

1. **Inherit from MatrixAgentV2** (`core/matrix_agents_v2.py`)
2. **Follow Matrix naming conventions** (use Matrix character themes)
3. **Implement required methods**: `execute_matrix_task()`, `get_matrix_description()`
4. **Use fail-fast validation** with quality thresholds at each step
5. **Integrate LangChain tools** for AI-enhanced analysis capabilities
6. **Follow SOLID principles** with no hardcoded values or absolute paths
7. **Use shared components** for logging, file management, and validation
8. **Implement comprehensive error handling** with MatrixErrorHandler

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

- **Matrix Phase 2 Completion**: 4/4 core agents implemented ✅
- **Agent Success Rate**: 4/4 Matrix agents functional (Sentinel, Architect, Merovingian, Agent Smith)
- **Primary Target**: Matrix Online launcher.exe (5.3MB, x86 PE32, MSVC .NET 2003)
- **Architecture**: Production-ready with LangChain AI integration and fail-fast validation
- **Next Phases**: Agents 5-16 (Neo through Agent Brown) planned for future implementation

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