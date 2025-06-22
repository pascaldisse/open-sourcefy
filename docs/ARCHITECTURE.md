# Open-Sourcefy Architecture Documentation

## Executive Summary

Open-Sourcefy is a production-grade AI-powered binary decompilation system that reconstructs 100% functionally identical C source code from Windows PE executables using a 17-agent Matrix pipeline with NSA-level security standards.

**MISSION CRITICAL**: Transform Windows PE executables into compilable C source code with binary-identical reconstruction capability and zero tolerance for functional deviations.

## System Architecture Overview

### Core Design Principles

1. **100% Functional Identity**: Reconstructed binaries must be functionally identical to originals
2. **Binary-Level Precision**: Exact PE structure preservation with timestamp-only differences
3. **Rule-Based Compliance**: Absolute adherence to rules.md without exceptions
4. **Self-Correcting System**: Continuous validation and auto-fix until perfection achieved
5. **NSA-Level Security**: Military-grade security and quality standards

### Platform Requirements (ABSOLUTE)

- **Windows 10/11 64-bit ONLY**: No alternative platforms supported
- **Visual Studio 2022 Preview**: Exclusive build system (no fallbacks)
- **Python 3.8+**: Core runtime environment
- **16GB+ RAM**: Memory requirements for large binary processing
- **SSD Storage**: High-speed I/O for binary analysis operations

## Matrix Agent Pipeline Architecture

### Agent Hierarchy and Dependencies

```
MASTER ORCHESTRATOR:
├── Agent 0: Deus Ex Machina (Pipeline Coordination)
    └── Controls execution flow and inter-agent communication

FOUNDATION PHASE (Sequential Execution):
├── Agent 1: Sentinel (Binary Analysis & Import Recovery)
│   ├── PE format validation and metadata extraction
│   ├── Import table analysis and DLL dependency mapping
│   └── Security scanning and threat assessment
├── Agent 2: Architect (PE Structure & Resource Extraction)
│   ├── Compiler toolchain detection and optimization analysis
│   ├── ABI and build system characteristic analysis
│   └── Architectural blueprint generation
├── Agent 3: Merovingian (Advanced Pattern Recognition)
│   ├── Assembly pattern recognition and categorization
│   └── Code structure pattern analysis
└── Agent 4: Agent Smith (Code Flow Analysis)
    ├── Memory layout and address mapping analysis
    ├── Data structure identification and categorization
    └── Dynamic analysis instrumentation preparation

ADVANCED ANALYSIS PHASE (Parallel Execution):
├── Agent 5: Neo (Advanced Decompilation Engine)
│   ├── Ghidra integration for advanced decompilation
│   ├── Function signature analysis and recovery
│   └── Advanced semantic analysis
├── Agent 6: Trainman (Assembly Analysis)
│   ├── Assembly instruction analysis and optimization
│   └── Low-level code pattern recognition
├── Agent 7: Keymaker (Resource Reconstruction)
│   ├── Resource extraction and categorization
│   └── Icon, string, and metadata reconstruction
└── Agent 8: Commander Locke (Build System Integration)
    ├── VS2022 build file generation (vcxproj, CMakeLists.txt)
    ├── Library dependency integration
    └── Header file generation and organization

RECONSTRUCTION PHASE (Sequential Execution):
├── Agent 9: The Machine (Compilation & Resource Integration)
│   ├── C source code compilation with Rule 12 compliance
│   ├── Resource compilation (RC.EXE integration)
│   └── Binary generation and validation
├── Agent 10: Twins (Binary Diff & Validation)
│   ├── Binary-level comparison between original and reconstructed
│   ├── Structural integrity validation
│   └── Functional equivalence verification
├── Agent 11: Oracle (Semantic Analysis)
│   ├── High-level semantic analysis and validation
│   └── Code quality and correctness verification
└── Agent 12: Link (Code Integration)
    ├── Final code integration and linking
    └── Dependency resolution and optimization

FINAL PROCESSING PHASE (Parallel Execution):
├── Agent 13: Agent Johnson (Quality Assurance)
│   ├── Comprehensive quality validation
│   └── Performance and security assessment
├── Agent 14: Cleaner (Code Cleanup)
│   ├── Code formatting and optimization
│   └── Final cleanup and beautification
├── Agent 15: Analyst (Final Validation)
│   ├── Final analysis and validation
│   └── Comprehensive reporting
└── Agent 16: Agent Brown (Output Generation)
    ├── Final output generation and packaging
    └── Report generation and documentation
```

### Inter-Agent Communication Protocol

```python
# Agent Data Flow Architecture
class AgentDataFlow:
    """
    Defines standardized data flow between Matrix agents
    """
    
    # Sequential Dependencies
    SEQUENTIAL_FLOW = {
        1: [],                    # Sentinel (Foundation)
        2: [1],                   # Architect depends on Sentinel
        3: [1, 2],               # Merovingian depends on Sentinel + Architect
        4: [1, 2, 3],            # Smith depends on Foundation Phase
        
        5: [1, 2, 3, 4],         # Neo depends on Foundation Phase
        6: [1, 2, 3, 4],         # Trainman depends on Foundation Phase
        7: [1, 2, 3, 4],         # Keymaker depends on Foundation Phase
        8: [1, 2, 3, 4],         # Locke depends on Foundation Phase
        
        9: [1, 2, 3, 4, 5, 6, 7, 8],   # Machine depends on all previous
        10: [9],                 # Twins depends on Machine output
        11: [9],                 # Oracle depends on Machine output
        12: [9, 10, 11],         # Link depends on Reconstruction Phase
        
        13: [12],                # Johnson depends on Link
        14: [12],                # Cleaner depends on Link
        15: [12],                # Analyst depends on Link
        16: [12, 13, 14, 15]     # Brown depends on Final Processing
    }
```

## Binary Reconstruction Methodology

### Phase 1: Deep Binary Analysis
1. **PE Structure Decomposition**: Complete PE header, section, and resource analysis
2. **Import Table Recovery**: Full DLL dependency mapping with function resolution
3. **Code Flow Analysis**: Control flow graph generation and optimization detection
4. **Resource Extraction**: Complete resource section extraction (icons, strings, metadata)

### Phase 2: Decompilation and Pattern Recognition
1. **Ghidra Integration**: Professional-grade decompilation using NSA's Ghidra
2. **Assembly Pattern Recognition**: Advanced pattern matching for compiler artifacts
3. **Function Signature Recovery**: Complete function prototype reconstruction
4. **Data Structure Analysis**: Complex data structure identification and reconstruction

### Phase 3: Source Code Generation
1. **Rule 12 Compliance**: Never edit source code - fix compiler/build system instead
2. **Header Generation**: Complete header file ecosystem (main.h, imports.h, common.h)
3. **Build System Integration**: VS2022 project files with exact library dependencies
4. **Resource Compilation**: RC file generation with raw resource integration

### Phase 4: Binary Reconstruction and Validation
1. **Compilation Pipeline**: Multi-stage compilation with error correction
2. **Binary Comparison**: Byte-level analysis with diff detection
3. **Functional Validation**: Runtime behavior verification
4. **Self-Correction Loop**: Automated fix application until 100% success

## Quality Assurance Framework

### Success Metrics (MANDATORY)
- **Functional Identity**: 100% functional equivalence required
- **Size Accuracy**: 99%+ size matching (timestamps excluded)
- **Resource Preservation**: 95%+ resource accuracy
- **Compilation Success**: Zero compilation errors tolerated
- **Security Compliance**: NSA-level security standards

### Validation Pipeline
```python
class ValidationPipeline:
    """
    Comprehensive validation framework
    """
    
    def validate_reconstruction(self):
        """Complete reconstruction validation"""
        validations = [
            self.validate_pe_structure(),      # PE format compliance
            self.validate_import_tables(),     # Import table accuracy
            self.validate_resource_sections(), # Resource preservation
            self.validate_code_sections(),     # Code section integrity
            self.validate_functionality(),     # Runtime behavior
            self.validate_size_accuracy(),     # Size matching
            self.validate_security()           # Security compliance
        ]
        return all(validations)
```

## File System Architecture

```
open-sourcefy/
├── src/                              # Core source code
│   ├── core/
│   │   ├── matrix_pipeline_orchestrator.py    # Master pipeline controller
│   │   ├── agents/                             # 17 Matrix agents (0-16)
│   │   │   ├── agent00_deus_ex_machina.py     # Master orchestrator
│   │   │   ├── agent01_sentinel.py            # Binary analysis
│   │   │   ├── agent02_architect.py           # PE structure
│   │   │   ├── agent03_merovingian.py         # Pattern recognition
│   │   │   ├── agent04_agent_smith.py         # Code flow analysis
│   │   │   ├── agent05_neo.py                 # Decompilation engine
│   │   │   ├── agent06_trainman.py            # Assembly analysis
│   │   │   ├── agent07_keymaker.py            # Resource reconstruction
│   │   │   ├── agent08_commander_locke.py     # Build integration
│   │   │   ├── agent09_the_machine.py         # Compilation system
│   │   │   ├── agent10_twins.py               # Binary diff validation
│   │   │   ├── agent11_oracle.py              # Semantic analysis
│   │   │   ├── agent12_link.py                # Code integration
│   │   │   ├── agent13_agent_johnson.py       # Quality assurance
│   │   │   ├── agent14_cleaner.py             # Code cleanup
│   │   │   ├── agent15_analyst.py             # Final validation
│   │   │   └── agent16_agent_brown.py         # Output generation
│   │   ├── shared_components.py               # Shared agent framework
│   │   ├── config_manager.py                  # Configuration management
│   │   ├── build_system_manager.py            # VS2022 integration
│   │   └── exceptions.py                      # Error handling
│   └── utils/                                 # Utility modules
├── input/                                     # Input binaries
├── output/                                    # Pipeline outputs
│   └── {binary_name}/
│       └── {timestamp}/
│           ├── agents/                        # Agent-specific outputs
│           ├── ghidra/                        # Decompilation results
│           ├── compilation/                   # Build artifacts
│           ├── reports/                       # Pipeline reports
│           └── logs/                          # Execution logs
├── tests/                                     # Comprehensive test suite
├── docs/                                      # Documentation
├── prompts/                                   # AI prompt templates
├── main.py                                    # Pipeline entry point
├── rules.md                                   # Absolute project rules
├── CLAUDE.md                                  # Command center
└── build_config.yaml                         # Build configuration
```

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary development language
- **Visual Studio 2022 Preview**: Exclusive build system
- **Ghidra**: NSA's reverse engineering framework
- **Windows SDK**: PE analysis and compilation tools
- **LangChain**: AI agent coordination framework

### Key Dependencies
```yaml
production_dependencies:
  - pefile: PE format analysis
  - lief: Binary manipulation
  - ghidra_bridge: Ghidra integration
  - langchain: AI agent framework
  - pyyaml: Configuration management
  - psutil: System monitoring
  - requests: AI API integration

development_dependencies:
  - pytest: Testing framework
  - black: Code formatting
  - mypy: Type checking
  - coverage: Test coverage
```

## Security Architecture

### NSA-Level Security Standards
1. **Input Sanitization**: All binary inputs validated and sandboxed
2. **Secure File Handling**: Temporary files encrypted and properly cleaned
3. **Access Control**: Strict permission validation throughout pipeline
4. **No Credential Exposure**: Zero secrets in logs, output, or temporary files
5. **Fail-Safe Design**: Secure failure modes with comprehensive error handling

### Threat Model
- **Malicious Binaries**: Comprehensive malware detection and sandboxing
- **Code Injection**: Input validation and sanitization at all levels
- **Data Exfiltration**: Secure temporary file handling and cleanup
- **System Compromise**: Minimal privilege execution with sandboxing

## Performance Architecture

### Optimization Strategies
- **Parallel Agent Execution**: Multi-threaded agent processing where possible
- **Intelligent Caching**: Agent output caching for incremental improvements
- **Memory Management**: Efficient memory usage for large binary processing
- **I/O Optimization**: High-speed SSD operations with async processing

### Scalability Design
- **Modular Architecture**: Each agent independently scalable
- **Resource Management**: Dynamic resource allocation based on binary size
- **Pipeline Optimization**: Intelligent batch processing and dependency management

## Integration Points

### External Tool Integration
```python
# Build System Integration
VS2022_INTEGRATION = {
    "cl_exe": "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/*/bin/Hostx64/x64/cl.exe",
    "rc_exe": "C:/Program Files (x86)/Windows Kits/10/bin/*/x64/rc.exe",
    "msbuild": "C:/Program Files/Microsoft Visual Studio/2022/Preview/MSBuild/Current/Bin/MSBuild.exe"
}

# Ghidra Integration
GHIDRA_INTEGRATION = {
    "headless_analyzer": "{GHIDRA_HOME}/support/analyzeHeadless",
    "script_manager": "src/ghidra_scripts/",
    "project_manager": "output/{binary}/ghidra/"
}
```

## Error Handling and Recovery

### Comprehensive Error Management
1. **Fail-Fast Design**: Immediate termination on critical errors
2. **Self-Correction Loop**: Automated error detection and correction
3. **Comprehensive Logging**: Detailed error tracking and analysis
4. **Recovery Strategies**: Multiple fallback mechanisms for each component

### Critical Error Codes
- **E001**: Missing VS2022 Preview installation
- **E002**: Invalid build configuration
- **E003**: Insufficient system resources
- **E004**: Agent prerequisite validation failure
- **E005**: Binary diff validation failure
- **E006**: Compilation pipeline failure
- **E007**: Resource reconstruction failure
- **E008**: Import table reconstruction failure

## Monitoring and Observability

### Performance Monitoring
- **Agent Execution Times**: Individual agent performance tracking
- **Memory Usage**: Real-time memory consumption monitoring
- **Success Rates**: Pipeline and agent success rate tracking
- **Quality Metrics**: Reconstruction accuracy and quality assessment

### Comprehensive Reporting
```python
class PipelineReporter:
    """
    Comprehensive pipeline reporting system
    """
    
    def generate_pipeline_report(self):
        return {
            "execution_summary": self.get_execution_summary(),
            "agent_performance": self.get_agent_performance(),
            "quality_metrics": self.get_quality_metrics(),
            "binary_analysis": self.get_binary_analysis(),
            "reconstruction_accuracy": self.get_reconstruction_accuracy(),
            "security_assessment": self.get_security_assessment()
        }
```

This architecture document serves as the definitive guide for the Open-Sourcefy system, ensuring 100% functional identity in binary reconstruction while maintaining NSA-level security and quality standards.