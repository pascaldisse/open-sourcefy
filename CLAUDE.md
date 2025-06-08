# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-Sourcefy is an AI-powered binary decompilation system that reconstructs compilable C source code from Windows PE executables using a 17-agent Matrix pipeline with Ghidra integration. 

**WINDOWS ONLY SYSTEM**: This system exclusively supports Windows PE executables and requires Visual Studio/MSBuild for compilation. Linux/macOS platforms and other binary formats (ELF/Mach-O) are not supported.

The primary test target is the Matrix Online launcher.exe binary.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment and dependencies
python3 main.py --verify-env

# List available agents
python3 main.py --list-agents
```

### Running the Pipeline
```bash
# Full pipeline (auto-detects binary from input/ directory)
python3 main.py

# Specific binary
python3 main.py launcher.exe

# Pipeline modes
python3 main.py --full-pipeline              # All agents (0-16)
python3 main.py --decompile-only             # Agents 1,2,5,7,14
python3 main.py --analyze-only               # Agents 1,2,3,4,5,6,7,8,9,14,15
python3 main.py --compile-only               # Agents 1,2,4,5,6,7,8,9,10,11,12,18
python3 main.py --validate-only              # Agents 1,2,4,5,6,7,8,9,10,11,12,13,19

# Specific agents
python3 main.py --agents 1                   # Single agent
python3 main.py --agents 1,3,7               # Multiple agents
python3 main.py --agents 1-5                 # Agent ranges

# Execution modes
python3 main.py --execution-mode master_first_parallel    # Default
python3 main.py --execution-mode pure_parallel           # Pure parallel
python3 main.py --execution-mode sequential              # Sequential

# Resource profiles
python3 main.py --resource-profile standard              # Default
python3 main.py --resource-profile high_performance      # High resource usage
python3 main.py --resource-profile conservative         # Conservative usage

# Development options
python3 main.py --dry-run                    # Show execution plan
python3 main.py --debug --profile            # Debug with profiling
```

### Testing and Validation
```bash
# Environment validation
python3 main.py --verify-env

# Configuration summary
python3 main.py --config-summary

# List available agents and modes
python3 main.py --list-agents
```

## Architecture Overview

### Matrix Agent Pipeline System

The system implements a **17-agent Matrix pipeline** with master-first execution and production-ready architecture:

**Agent 0 - Master Orchestrator**:
- **Deus Ex Machina**: Supreme orchestrator that coordinates the entire pipeline
- Creates execution plans, manages agent batches, validates prerequisites
- Generates comprehensive reports and performance metrics

**Phase 1 - Foundation** (Agent 1):
- **Sentinel**: Binary discovery, metadata analysis, and security scanning
- Multi-format support (PE/ELF/Mach-O), hash calculation, entropy analysis
- LangChain AI integration for enhanced threat detection

**Phase 2 - Core Analysis** (Agents 2-4):
- **The Architect**: Architecture analysis, compiler detection, optimization patterns
- **The Merovingian**: Basic decompilation, function detection, control flow analysis
- **Agent Smith**: Binary structure analysis, data extraction, dynamic bridge preparation

**Phase 3 - Advanced Analysis** (Agents 5-12):
- **Neo**: Advanced decompilation with Ghidra integration
- **The Twins**: Binary differential analysis and comparison
- **The Trainman**: Advanced assembly analysis and transportation
- **The Keymaker**: Resource reconstruction and access management
- **Commander Locke**: Global reconstruction orchestration
- **The Machine**: Compilation orchestration and build systems
- **The Oracle**: Final validation and truth verification
- **Link**: Cross-reference and linking analysis

**Phase 4 - Final Validation** (Agents 13-16):
- **Agent Johnson**: Security analysis and vulnerability detection
- **The Cleaner**: Code cleanup and optimization
- **The Analyst**: Advanced metadata analysis and intelligence synthesis
- **Agent Brown**: Final quality assurance and optimization

### Execution Model

**Master-First Parallel Execution**:
1. **Master Agent (Agent 0)** coordinates the entire pipeline
2. **Dependency-Based Batching**: Agents organized into execution batches based on dependencies
3. **Parallel Execution**: Agents within batches execute in parallel with timeout management
4. **Context Sharing**: Global execution context passed between agents with shared memory
5. **Fail-Fast Validation**: Quality thresholds enforced at each stage

**Dependency Structure**:
```
Agent 1 (Sentinel) → No dependencies
Agents 2,3,4 → Depend on Agent 1
Agents 5,6,7,8 → Depend on Agents 1,2
Agents 9,12,13 → Depend on Agents 5,6,7,8
Agent 10 → Depends on Agent 9
Agent 11 → Depends on Agent 10
Agents 14,15 → Depend on Agents 9,10,11
Agent 16 → Depends on Agents 14,15
```

### Core System Components

**Pipeline Orchestrator** (`core/matrix_pipeline_orchestrator.py`):
- Master-first execution with parallel agent coordination
- Comprehensive configuration management and resource limits
- Async execution with timeout and error handling
- Report generation and performance metrics

**Matrix Agent Framework** (`core/matrix_agents_v2.py`):
- Production-ready base classes with Matrix-themed architecture
- Standardized agent result structures and status management
- Specialized base classes: AnalysisAgent, DecompilerAgent, ReconstructionAgent, ValidationAgent
- Comprehensive dependency mapping for all 17 agents

**Shared Components** (`core/shared_components.py`):
- MatrixLogger: Enhanced logging with Matrix-themed formatting
- MatrixFileManager: Standardized file operations
- MatrixValidator: Common validation functions
- MatrixProgressTracker: Progress tracking with ETA calculation
- MatrixErrorHandler: Standardized error handling with retry logic
- SharedAnalysisTools: Entropy calculation and pattern detection

**Configuration Manager** (`core/config_manager.py`):
- Hierarchical configuration with environment variables, YAML/JSON config files
- Auto-detection of tools (Ghidra, Java, Visual Studio)
- Agent-specific settings and resource limits
- No hardcoded values - fully configurable system

**CLI Interface** (`main.py`):
- Advanced CLI with comprehensive argument parsing
- Multiple execution modes and resource profiles
- Async pipeline execution with performance profiling
- Dry-run mode for execution planning

### Output Organization

All output is organized under `output/[timestamp]/`:
```
output/20250608_HHMMSS/
├── agents/          # Agent-specific analysis outputs
│   ├── agent_01_sentinel/
│   ├── agent_02_architect/
│   └── ...
├── ghidra/          # Ghidra decompilation results
├── compilation/     # MSBuild artifacts and generated source
├── reports/         # Pipeline execution reports
│   └── matrix_pipeline_report.json
├── logs/            # Execution logs and debug information
├── temp/            # Temporary files (auto-cleaned)
└── tests/           # Generated test files
```

## Development Guidelines

### Matrix Agent Development

When creating or modifying Matrix agents:

1. **Inherit from appropriate base class**:
   - `AnalysisAgent` for analysis-focused agents (Agents 1-8)
   - `DecompilerAgent` for decompilation agents (Agent 3, 5, 7)
   - `ReconstructionAgent` for reconstruction agents (Agents 9-16)
   - `ValidationAgent` for validation agents (Agents 11, 13, 16)

2. **Follow Matrix naming conventions**:
   - Use Matrix character themes for agent names
   - Implement `get_matrix_description()` with character-appropriate descriptions

3. **Implement required methods**:
   - `execute_matrix_task(context)`: Main agent logic
   - `_validate_prerequisites(context)`: Input validation
   - `_get_required_context_keys()`: Required context dependencies

4. **Use shared components**:
   - Import from `shared_components` for common functionality
   - Use `MatrixLogger`, `MatrixFileManager`, `MatrixValidator`
   - Leverage `SharedAnalysisTools` and `SharedValidationTools`

5. **Follow SOLID principles**:
   - No hardcoded values or absolute paths
   - Use configuration manager for all settings
   - Implement comprehensive error handling
   - Use dependency injection for external tools

6. **Quality validation**:
   - Implement fail-fast validation with quality thresholds
   - Use `ValidationError` for prerequisite failures
   - Return structured data with confidence scores

### Configuration

**Environment Variables**:
- `GHIDRA_HOME`: Ghidra installation directory
- `JAVA_HOME`: Java installation directory
- `MATRIX_DEBUG`: Enable debug logging
- `MATRIX_AI_ENABLED`: Enable LangChain AI features

**Configuration Files**:
- Support for YAML and JSON configuration files
- Hierarchical configuration: env vars > config files > defaults
- Agent-specific timeouts, retries, and resource limits

**Tool Detection**:
- Automatic detection of Ghidra, Java, Visual Studio
- Relative path resolution from project root
- Graceful degradation when tools are unavailable

### Quality Validation

The system implements strict validation thresholds:
- **Code Quality**: 75% threshold for meaningful code structure
- **Implementation Score**: 75% threshold for real vs placeholder code
- **Completeness**: 70% threshold for project completeness
- **Binary Analysis Confidence**: Minimum confidence scores for format detection

### Dependencies

**Required**:
- Python 3.8+ (async/await support required)
- Java 17+ (for Ghidra integration)

**Included**:
- Ghidra 11.0.3 (in ghidra/ directory)
- Matrix agent implementations (0-4 complete)

**Optional**:
- Microsoft Visual C++ Build Tools (for compilation testing)
- LangChain libraries (for AI-enhanced analysis)
- pefile, elftools, macholib (for binary parsing)

### Current Implementation Status

**✅ Production-Ready Infrastructure**:
- Master orchestrator (Agent 0) fully implemented
- Matrix agent framework complete
- Configuration management system operational
- CLI interface with comprehensive options
- Shared components and utilities complete

**✅ Implemented Agents**:
- Agent 0: Deus Ex Machina (Master Orchestrator)
- Agent 1: Sentinel (Binary Discovery & Metadata Analysis)
- Agent 2: The Architect (Architecture Analysis) 
- Agent 3: The Merovingian (Basic Decompilation)
- Agent 4: Agent Smith (Binary Structure Analysis)

**🚧 Planned Agents** (Agents 5-16):
- Framework established, ready for implementation
- Dependency structure defined
- Base classes and patterns available

**📊 System Status**:
- **Architecture**: Production-ready with comprehensive error handling
- **Primary Target**: Matrix Online launcher.exe (5.3MB, x86 PE32, MSVC .NET 2003)
- **Execution Model**: Master-first parallel with dependency batching
- **AI Integration**: LangChain support for enhanced analysis
- **Quality Assurance**: Fail-fast validation with quality thresholds

### Testing Approach

**Built-in Validation**:
- Agent results validated for quality and completeness
- Pipeline can terminate early if validation thresholds not met
- Comprehensive error handling and retry logic
- Performance metrics and execution reports

**Environment Verification**:
```bash
python3 main.py --verify-env    # Check all dependencies
python3 main.py --dry-run       # Preview execution plan
python3 main.py --debug         # Detailed logging
```

**Testing Commands**:
```bash
# Single agent testing
python3 main.py --agents 1

# Core analysis testing  
python3 main.py --agents 1-4

# Decompilation pipeline testing
python3 main.py --decompile-only
```

## System Requirements

### **Windows Requirements (MANDATORY)**
- **Operating System**: Windows 10/11 (64-bit)
- **Visual Studio**: 2019 or 2022 with MSVC compiler
- **MSBuild**: Included with Visual Studio
- **Architecture**: x86/x64 Windows executables only

### **Core Dependencies**
- **Python**: 3.8+ (Windows version)
- **Java**: 17+ (for Ghidra integration)
- **Ghidra**: 11.0.3 (included in project)
- **MSVC Compiler**: cl.exe must be in PATH

### **Unsupported Platforms**
❌ **Linux/Unix**: Not supported
❌ **macOS**: Not supported  
❌ **ELF binaries**: Not supported
❌ **Mach-O binaries**: Not supported
❌ **GCC/Clang**: Not supported
❌ **Make/CMake**: Not supported

### Ghidra Integration

**Installation**: Ghidra 11.0.3 included in `ghidra/` directory
**Detection**: Automatic path resolution from project root
**Usage**: Headless analysis with custom scripts
**Quality**: Assessment and confidence scoring
**Management**: Temporary project creation and cleanup

**Custom Scripts**: Located in agent implementations for enhanced accuracy
**Integration Points**: Agents 3, 5, 7, 14 leverage Ghidra for decompilation tasks