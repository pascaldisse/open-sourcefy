# Open-Sourcefy System Overview for Claude

## Project Mission
Open-Sourcefy is an advanced binary reverse engineering and source code reconstruction system that analyzes compiled executables and attempts to reconstruct readable, compilable source code using multi-agent AI analysis.

## System Architecture

### Core Philosophy
- **Multi-Agent Pipeline**: 15 specialized agents work in dependency order to analyze different aspects of binaries
- **Incremental Analysis**: Each agent builds upon previous agents' findings
- **Real Tool Integration**: Leverages industry-standard tools (Ghidra, disassemblers, compilers)
- **Quality-Focused**: Emphasizes accuracy and compilability over speed

### Agent Pipeline Overview

```
Binary Input → Agent Flow → Source Code Output

Agent 1: Binary Discovery (PE/ELF/Mach-O detection)
         ↓
Agent 2: Architecture Analysis (x86/x64/ARM detection)  
         ↓
Agent 3: Smart Error Pattern Matching (ML-based analysis)
         ↓
Agent 4: Basic Decompiler (Initial disassembly)
         ↓
Agent 5: Binary Structure Analyzer (Sections, imports, exports)
         ↓
Agent 6: Optimization Matcher (Compiler optimization detection)
         ↓  
Agent 7: Advanced Decompiler (Ghidra integration)
         ↓
Agent 8: Binary Diff Analyzer (Comparison analysis)
         ↓
Agent 9: Advanced Assembly Analyzer (Deep instruction analysis)
         ↓
Agent 10: Resource Reconstructor (Icons, strings, dialogs)
         ↓
Agent 11: Global Reconstructor (Code organization)
         ↓
Agent 12: Compilation Orchestrator (Build system generation)
         ↓
Agent 13: Final Validator (Quality assurance)
         ↓
Agent 14: Advanced Ghidra (Enhanced analysis)
         ↓
Agent 15: Metadata Analysis (Comprehensive metadata extraction)
```

## Current Project Status

### Development Phases
**Phase 1** (✅ Complete): Basic pipeline infrastructure and agent framework
**Phase 2** (✅ Complete): Core agent implementations with basic functionality  
**Phase 3** (🔄 In Progress): Dummy code removal and error handling improvement
**Phase 4** (📋 Planned): Full feature implementation and production readiness

### Quality Metrics (Current)
- **Overall Quality**: 66.5%
- **Code Quality**: 30% (needs improvement)
- **Analysis Accuracy**: 60% 
- **Agent Success Rate**: 16/16 agents functional
- **Pipeline Reliability**: Good (structured error handling)

## Technical Stack

### Core Technologies
- **Python 3.9+**: Main implementation language
- **Ghidra**: Primary decompilation engine (headless mode)
- **Binary Analysis**: pefile, pyelftools, macholib for format parsing
- **Machine Learning**: scikit-learn for pattern recognition
- **Build Systems**: MSBuild, CMake integration

### Project Structure
```
src/
├── core/
│   ├── agents/           # 15 analysis agents
│   ├── agent_base.py     # Base agent class and framework
│   ├── parallel_executor.py  # Agent execution management
│   ├── ghidra_processor.py   # Ghidra integration
│   └── config_manager.py     # Configuration management
├── ml/
│   └── pattern_engine.py     # ML-based pattern recognition
└── utils/                    # Utility functions

output/                   # Structured output directory
├── agents/              # Agent-specific results
├── ghidra/             # Ghidra analysis outputs
├── compilation/        # Generated source code
├── reports/           # Pipeline reports
├── logs/             # Execution logs
└── temp/            # Temporary files
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
# Run full pipeline
python main.py target.exe

# Run specific agent for debugging  
python -m src.core.agents.agent01_binary_discovery target.exe

# Generate test report
python -m src.core.testing.run_tests --binary target.exe

# Validate pipeline health
python -m src.core.validation.pipeline_health_check
```