# Open-Sourcefy Matrix System Overview for Claude

## Project Mission
Open-Sourcefy is an AI-powered binary decompilation system that reconstructs compilable C source code from Windows PE executables using a 17-agent Matrix pipeline with Ghidra integration.

**WINDOWS ONLY SYSTEM**: This system exclusively supports Windows PE executables and requires Visual Studio/MSBuild for compilation.

## Matrix Agent Pipeline Architecture

### Core Philosophy - The Matrix Framework
- **17-Agent Matrix Pipeline**: Specialized Matrix characters work in dependency-based batches
- **Master-First Execution**: Agent 0 (Deus Ex Machina) orchestrates the entire pipeline
- **Production-Ready**: NSA-level code quality with comprehensive error handling
- **Windows-Focused**: Optimized for Windows PE executables and MSVC compilation

### Matrix Agent Pipeline Overview

```
Binary Input → Matrix Agent Flow → Compilable Source Code Output

Agent 0: Deus Ex Machina (Master Orchestrator)
         ↓ Coordinates entire pipeline
Agent 1: Sentinel (Binary Discovery & Security Scanning)
         ↓
Batch 1: Agents 2,3,4 (Parallel Execution)
├── Agent 2: The Architect (Architecture Analysis)
├── Agent 3: The Merovingian (Basic Decompilation)  
└── Agent 4: Agent Smith (Binary Structure Analysis)
         ↓
Batch 2: Agents 5,6,7,8 (Advanced Analysis)
├── Agent 5: Neo (Advanced Decompilation with Ghidra)
├── Agent 6: The Twins (Binary Differential Analysis)
├── Agent 7: The Trainman (Advanced Assembly Analysis)
└── Agent 8: The Keymaker (Resource Reconstruction)
         ↓
Batch 3: Agents 9,12,13 (Reconstruction & Compilation)
├── Agent 9: Commander Locke (Global Reconstruction)
├── Agent 12: The Machine (Compilation Orchestration)
└── Agent 13: The Oracle (Final Validation)
         ↓
Sequential: Agents 10,11 (Dependency Chain)
Agent 10: → Agent 11: (Cross-reference and linking)
         ↓
Final Batch: Agents 14,15,16 (Quality Assurance)
├── Agent 14: Agent Johnson (Security Analysis)
├── Agent 15: The Cleaner (Code Cleanup)
└── Agent 16: The Analyst (Final Intelligence)
```

## Current Project Status

### Implementation Status
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

### Quality Metrics (Current)
- **Architecture**: Production-ready with comprehensive error handling
- **Primary Target**: Matrix Online launcher.exe (5.3MB, x86 PE32, MSVC .NET 2003)
- **Execution Model**: Master-first parallel with dependency batching
- **AI Integration**: LangChain support for enhanced analysis
- **Quality Assurance**: Fail-fast validation with quality thresholds

## Technical Stack

### Core Technologies
- **Python 3.9+**: Main implementation language
- **Ghidra**: Primary decompilation engine (headless mode)
- **Binary Analysis**: pefile, pyelftools, macholib for format parsing
- **Machine Learning**: scikit-learn for pattern recognition
- **Build Systems**: MSBuild, CMake integration

### Project Structure
```
open-sourcefy/
├── input/                   # Input binary files for analysis
├── output/                  # Pipeline execution results and artifacts
├── src/                     # Source code and core system
│   ├── core/               # Core framework components
│   │   ├── agents/         # Matrix agent implementations (0-16)
│   │   ├── config_manager.py # Configuration management
│   │   ├── matrix_pipeline_orchestrator.py # Master orchestrator
│   │   ├── shared_components.py # Shared utilities
│   │   └── agent_base.py   # Base classes and interfaces
│   ├── ml/                 # Machine learning components
│   └── utils/              # Pure utility functions
├── tests/                  # Test suites and validation scripts
├── docs/                   # Project documentation and analysis reports
├── ghidra/                 # Ghidra installation and custom scripts
├── temp/                   # Temporary files and development artifacts
├── prompts/                # AI prompts and pipeline instructions
├── venv/                   # Python virtual environment
├── main.py                 # Primary CLI entry point
├── requirements.txt        # Python dependencies
└── config.yaml             # Main configuration file

output/[timestamp]/         # Structured output directory
├── agents/                 # Agent-specific results
├── ghidra/                # Ghidra analysis outputs
├── compilation/           # Generated source code and MSBuild artifacts
├── reports/              # Pipeline reports and execution summaries
├── logs/                # Execution logs and debug information
├── temp/               # Temporary files (auto-cleaned)
└── tests/             # Generated test files
```

## Key Features

### 1. Multi-Format Binary Support
- **PE (Windows)**: Full PE32/PE32+ support with resource extraction
- **ELF (Linux)**: Complete ELF analysis with symbol parsing
- **Mach-O (macOS)**: Basic Mach-O support

### 2. Advanced Analysis Capabilities
- **Ghidra Integration**: Automated headless decompilation
- **Pattern Recognition**: ML-based compiler optimization detection
- **Control Flow Analysis**: Function call graph reconstruction
- **Type Inference**: Data structure and variable type recovery

### 3. Code Reconstruction
- **Function Recovery**: Signature inference and body reconstruction
- **Resource Extraction**: Icons, strings, dialogs, and other resources
- **Build System Generation**: CMake and MSBuild project files
- **Quality Validation**: Compilation testing and accuracy assessment

## Common Issues and Solutions

### Current Known Issues
1. **NotImplementedError Exceptions**: Many advanced features throw NotImplementedError
   - **Solution**: Follow implementation_fixing.md prompt for systematic implementation
   
2. **Dummy Code**: Some agents return placeholder data
   - **Solution**: Use agent_cleanup.md prompt to identify and fix
   
3. **Build Dependencies**: Complex dependency chains between agents
   - **Solution**: Follow dependency order and implement foundational agents first

### Development Best Practices
- **Test-Driven Development**: Write tests before implementing features
- **Incremental Implementation**: Focus on core functionality before advanced features
- **Real-World Testing**: Use actual binary samples for validation
- **Cross-Platform Compatibility**: Test on Windows, Linux, and macOS

## Usage Examples

### Basic Analysis
```bash
# Analyze a Windows executable
python main.py target.exe

# Custom output directory
python main.py target.exe --output-dir my_analysis

# Results will be in structured subdirectories:
# my_analysis/agents/     - Agent-specific outputs
# my_analysis/ghidra/     - Ghidra decompilation  
# my_analysis/compilation/ - Generated source code
# my_analysis/reports/    - Pipeline summary
```

### Advanced Configuration
```python
# Pipeline configuration
{
    "agents": {
        "timeout": 300,
        "retry_count": 2,
        "parallel_execution": True
    },
    "ghidra": {
        "headless_timeout": 600,
        "custom_scripts": True,
        "decompilation_timeout": 60
    },
    "output": {
        "structured_dirs": True,
        "compression": False,
        "cleanup_temp": True
    }
}
```

## Development Priorities

### Immediate (This Sprint)
1. ✅ Remove all dummy code from agents
2. 🔄 Implement core binary format parsing
3. 📋 Enhance Ghidra integration
4. 📋 Add comprehensive error handling

### Short Term (Next 2-4 weeks)  
1. 📋 Implement missing NotImplementedError functions
2. 📋 Add ML-based pattern recognition
3. 📋 Enhance code reconstruction quality
4. 📋 Implement automated testing framework

### Long Term (1-3 months)
1. 📋 Advanced optimization reversal
2. 📋 Multi-platform build system support  
3. 📋 Real-time analysis capabilities
4. 📋 Web interface and API

## Integration Points

### External Tool Dependencies
- **Ghidra**: Must be installed and GHIDRA_HOME set
- **Build Tools**: MSBuild (Windows), GCC/Clang (Linux/macOS)
- **Python Libraries**: See requirements.txt for complete list

### Agent Communication
- **Context Passing**: Agents communicate via shared context dictionary
- **Result Storage**: AgentResult objects with status, data, and metadata
- **Dependency Management**: Automatic dependency resolution and execution order

### Output Integration
- **Structured Output**: All results organized in logical subdirectories
- **JSON Reports**: Machine-readable analysis summaries
- **Source Code**: Human-readable reconstructed code
- **Build Files**: CMake/MSBuild files for compilation testing

## Success Metrics and Goals

### Technical Goals
- **95%+ Pipeline Reliability**: Robust error handling and recovery
- **80%+ Compilation Success**: Generated code compiles successfully  
- **90%+ Code Quality**: Well-documented, tested, maintainable code
- **<5 minute Analysis Time**: Efficient processing for typical binaries

### Business Goals  
- **Production Ready**: Suitable for commercial reverse engineering workflows
- **Cross-Platform**: Works reliably on Windows, Linux, and macOS
- **Extensible**: Easy to add new agents and analysis capabilities
- **User Friendly**: Clear documentation and intuitive interfaces

## Getting Help

### Documentation
- **README.md**: Basic usage and setup instructions
- **DOCUMENTATION.md**: Detailed technical documentation
- **docs/**: Agent-specific documentation and architecture details

### Development Prompts
- **prompts/agent_cleanup.md**: For cleaning up dummy code and refactoring
- **prompts/implementation_fixing.md**: For implementing missing functionality
- **prompts/pipeline_compilation.md**: For fixing compilation and build issues

### Common Commands
```bash
# Run full pipeline (auto-detects binary from input/ directory)
python3 main.py

# Specific binary
python3 main.py launcher.exe

# Pipeline modes
python3 main.py --full-pipeline              # All agents (0-16)
python3 main.py --decompile-only             # Agents 1,2,5,7,14
python3 main.py --analyze-only               # Agents 1,2,3,4,5,6,7,8,9,14,15
python3 main.py --compile-only               # Agents 1,2,4,5,6,7,8,9,10,11,12,18

# Specific agents
python3 main.py --agents 1                   # Single agent
python3 main.py --agents 1,3,7               # Multiple agents
python3 main.py --agents 1-5                 # Agent ranges

# Development options
python3 main.py --dry-run                    # Show execution plan
python3 main.py --debug --profile            # Debug with profiling
python3 main.py --verify-env                 # Environment validation
python3 main.py --list-agents                # List available agents
```