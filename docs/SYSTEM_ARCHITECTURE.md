# Open-Sourcefy System Architecture

## Overview

Open-Sourcefy is a production-grade AI-powered binary decompilation system that reconstructs compilable C source code from Windows PE executables using a 17-agent Matrix pipeline with Ghidra integration.

## Core Principles

- **STRICT MODE ONLY**: No fallbacks, no alternatives, no graceful degradation
- **WINDOWS EXCLUSIVE**: Windows PE executables with Visual Studio/MSBuild compilation
- **ZERO TOLERANCE**: Fail fast when tools are missing - never degrade gracefully
- **PRODUCTION READY**: NSA-level security, >90% test coverage, SOLID principles

## System Components

### 1. Matrix Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATRIX PIPELINE (17 AGENTS)                  │
├─────────────────────────────────────────────────────────────────┤
│ Agent 0: Deus Ex Machina (Master Orchestrator)                 │
├─────────────────────────────────────────────────────────────────┤
│ FOUNDATION AGENTS (1-4):                                       │
│ • Agent 1: Sentinel - Binary Analysis & Import Table Recovery  │
│ • Agent 2: Architect - PE Structure & Resource Extraction      │
│ • Agent 3: Merovingian - Advanced Analysis                     │
│ • Agent 4: Agent Smith - Code Flow Analysis                    │
├─────────────────────────────────────────────────────────────────┤
│ ADVANCED ANALYSIS AGENTS (5-8):                                │
│ • Agent 5: Neo - Advanced Decompiler                          │
│ • Agent 6: Trainman - Assembly Analysis                       │
│ • Agent 7: Keymaker - Resource Reconstruction                 │
│ • Agent 8: Commander Locke - Build System Integration         │
├─────────────────────────────────────────────────────────────────┤
│ RECONSTRUCTION AGENTS (9-12):                                  │
│ • Agent 9: The Machine - Resource Compilation                 │
│ • Agent 10: Twins - Binary Diff & Validation                  │
│ • Agent 11: Oracle - Semantic Analysis                        │
│ • Agent 12: Link - Code Integration                           │
├─────────────────────────────────────────────────────────────────┤
│ FINAL PROCESSING AGENTS (13-16):                               │
│ • Agent 13: Agent Johnson - Quality Assurance                 │
│ • Agent 14: Cleaner - Code Cleanup                            │
│ • Agent 15: Analyst - Final Validation                        │
│ • Agent 16: Agent Brown - Output Generation                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Core System Components

```
src/core/
├── agents/                          # 17 Matrix Agents (0-16)
├── matrix_pipeline_orchestrator.py  # Master Pipeline Controller
├── matrix_agents_v2.py             # Agent Framework & Base Classes
├── config_manager.py               # Configuration Management
├── build_system_manager.py         # VS2022 Build Integration
├── shared_components.py            # Shared Agent Components
├── ghidra_processor.py             # Ghidra 11.0.3 Integration
├── ai_system.py                    # AI Engine Interface
└── exceptions.py                   # Error Handling System
```

### 3. Data Flow Architecture

```
INPUT → GHIDRA → MATRIX PIPELINE → BUILD SYSTEM → VALIDATION → OUTPUT
  ↓       ↓           ↓              ↓            ↓          ↓
PE.EXE → C CODE → RESOURCES → MSBuild → TESTS → RECONSTRUCTED.EXE
```

## Agent Specifications

### Agent 0: Deus Ex Machina (Master Orchestrator)
- **Purpose**: Master control and coordination
- **Input**: Target PE executable
- **Output**: Orchestrated pipeline execution
- **Critical Functions**:
  - Pipeline initialization and coordination
  - Agent dependency management
  - Error propagation and recovery
  - Quality gate enforcement

### Foundation Agents (1-4)

#### Agent 1: Sentinel
- **Purpose**: Binary analysis and import table recovery
- **Critical Issue**: Import table mismatch (538→5 DLLs)
- **Input**: PE executable
- **Output**: Import table, function signatures, DLL dependencies
- **Key Functions**:
  - PE header analysis
  - Import table reconstruction
  - MFC 7.1 signature detection
  - Ordinal resolution

#### Agent 2: Architect
- **Purpose**: PE structure and resource extraction
- **Input**: PE executable, Sentinel output
- **Output**: Resources, structure analysis
- **Key Functions**:
  - Resource section extraction
  - Icon/bitmap extraction
  - Version info recovery
  - Manifest processing

#### Agent 3: Merovingian
- **Purpose**: Advanced analysis and pattern recognition
- **Input**: PE structure, binary data
- **Output**: Code patterns, algorithms
- **Key Functions**:
  - Algorithm identification
  - Code pattern analysis
  - Obfuscation detection
  - Compiler fingerprinting

#### Agent 4: Agent Smith
- **Purpose**: Code flow analysis
- **Input**: Disassembly, structure data
- **Output**: Control flow graphs, function boundaries
- **Key Functions**:
  - Control flow reconstruction
  - Function identification
  - Call graph generation
  - Dead code elimination

### Advanced Analysis Agents (5-8)

#### Agent 5: Neo
- **Purpose**: Advanced decompilation
- **Input**: Binary code, control flows
- **Output**: C source code (readable main)
- **Key Functions**:
  - High-level C reconstruction
  - Variable type inference
  - Function signature recovery
  - Meaningful name generation

#### Agent 6: Trainman
- **Purpose**: Assembly analysis
- **Input**: Raw assembly
- **Output**: Assembly annotations, optimizations
- **Key Functions**:
  - Instruction pattern analysis
  - Optimization detection
  - Register usage analysis
  - Stack frame reconstruction

#### Agent 7: Keymaker
- **Purpose**: Resource reconstruction
- **Input**: Extracted resources
- **Output**: RC files, resource headers
- **Key Functions**:
  - RC file generation
  - Resource compilation
  - String table reconstruction
  - Icon/bitmap integration

#### Agent 8: Commander Locke
- **Purpose**: Build system integration
- **Input**: Source code, resources
- **Output**: VS project files, build configuration
- **Key Functions**:
  - VS2022 project generation
  - MSBuild configuration
  - Dependency management
  - Compilation orchestration

### Reconstruction Agents (9-12)

#### Agent 9: The Machine
- **Purpose**: Resource compilation
- **Input**: RC files, resources
- **Output**: Compiled resource files (.res)
- **Key Functions**:
  - RC.EXE compilation
  - Resource linking
  - Binary resource generation
  - MFC 7.1 compatibility

#### Agent 10: Twins
- **Purpose**: Binary diff and validation
- **Input**: Original binary, reconstructed binary
- **Output**: Diff analysis, validation report
- **Key Functions**:
  - Binary comparison
  - Functionality validation
  - Import table verification
  - Size/structure analysis

#### Agent 11: Oracle
- **Purpose**: Semantic analysis
- **Input**: Source code, binary behavior
- **Output**: Semantic annotations, optimizations
- **Key Functions**:
  - Semantic code analysis
  - Behavior verification
  - Logic optimization
  - Code quality assessment

#### Agent 12: Link
- **Purpose**: Code integration
- **Input**: Multiple code components
- **Output**: Integrated source code
- **Key Functions**:
  - Component integration
  - Dependency resolution
  - Code merging
  - Final assembly

### Final Processing Agents (13-16)

#### Agent 13: Agent Johnson
- **Purpose**: Quality assurance
- **Input**: Integrated code
- **Output**: QA report, compliance verification
- **Key Functions**:
  - Code quality validation
  - Standards compliance
  - Security assessment
  - Performance analysis

#### Agent 14: Cleaner
- **Purpose**: Code cleanup
- **Input**: Raw generated code
- **Output**: Clean, formatted code
- **Key Functions**:
  - Code formatting
  - Comment generation
  - Dead code removal
  - Style normalization

#### Agent 15: Analyst
- **Purpose**: Final validation
- **Input**: Clean code, resources
- **Output**: Final validation report
- **Key Functions**:
  - Comprehensive testing
  - Regression validation
  - Performance benchmarking
  - Success rate analysis

#### Agent 16: Agent Brown
- **Purpose**: Output generation
- **Input**: Validated code and resources
- **Output**: Final deliverables
- **Key Functions**:
  - Final package generation
  - Documentation creation
  - Archive preparation
  - Deployment packaging

## Build System Integration

### Visual Studio 2022 Preview (EXCLUSIVE)
- **Compiler**: cl.exe (configured paths only)
- **MSBuild**: MSBuild.exe (no fallbacks)
- **SDK**: Windows SDK (required)
- **No Alternatives**: Single build path, strict validation

### Resource Compilation Pipeline
```
RC Files → RC.EXE → .RES Files → LINK.EXE → Final Binary
```

## Configuration Management

### Centralized Configuration
- `config.yaml`: Main configuration
- `build_config.yaml`: Build system paths
- Environment validation on startup
- No hardcoded values allowed

### Path Management
- Absolute paths only
- No relative path alternatives
- Strict path validation
- Configured tools only

## Error Handling

### Fail-Fast Philosophy
- Immediate failure on missing tools
- No graceful degradation
- No alternative code paths
- Strict prerequisite validation

### Error Categories
1. **FATAL**: Missing required tools/dependencies
2. **CRITICAL**: Agent execution failures
3. **WARNING**: Quality threshold violations
4. **INFO**: Progress and status updates

## Quality Assurance

### Testing Strategy
- **Unit Tests**: >90% coverage requirement
- **Integration Tests**: Pipeline validation
- **Regression Tests**: Binary comparison
- **Performance Tests**: Execution time benchmarks

### Validation Criteria
- Binary functionality match
- Import table completeness
- Resource integrity
- Compilation success

## Security Architecture

### NSA-Level Security
- Zero hardcoded credentials
- Secure temporary file handling
- Memory cleanup procedures
- Access control validation

### Threat Model
- Malicious binary protection
- Code injection prevention
- Resource manipulation detection
- Build system isolation

## Performance Optimization

### Parallel Execution
- Agent-level parallelization
- Resource compilation optimization
- I/O operation batching
- Memory usage optimization

### Scalability
- Agent isolation
- Resource pooling
- Caching strategies
- Load balancing

## Monitoring & Observability

### Logging Framework
- Structured logging
- Agent-specific logs
- Performance metrics
- Error tracking

### Metrics Collection
- Pipeline success rates
- Agent execution times
- Resource usage
- Quality scores

## Deployment Architecture

### Production Environment
- Windows Server 2022
- Visual Studio 2022 Preview
- Ghidra 11.0.3
- Python 3.11+

### Container Support
- Windows containers only
- VS Build Tools integration
- Ghidra headless mode
- Resource compilation support

## Known Issues & Solutions

### Import Table Mismatch (PRIMARY BOTTLENECK)
- **Issue**: 538→5 DLL reduction, 64.3% discrepancy
- **Impact**: 25% validation failure
- **Solution**: Agent 9 data flow repair, MFC 7.1 integration
- **Expected**: 60% → 85% success rate improvement

### MFC 7.1 Compatibility
- **Issue**: VS2022 incompatible with MFC 7.1
- **Solution**: Alternative build approach research
- **Status**: Implementation ready

## Maintenance & Updates

### Version Control
- Git-based workflow
- Branch protection rules
- Mandatory code review
- Automated testing

### Documentation Standards
- Architecture documentation
- Agent specifications
- API documentation
- Deployment guides

## Future Enhancements

### Planned Features
- Multi-compiler support research
- Advanced obfuscation handling
- Machine learning integration
- Cloud deployment options

### Research Areas
- Binary similarity analysis
- Advanced packing detection
- Automated testing generation
- Performance optimization