# CLAUDE.md - Project Command Center

🚨 **MANDATORY READING**: Read rules.md IMMEDIATELY before any work. ALL RULES ARE ABSOLUTE AND NON-NEGOTIABLE. 🚨

## Project Overview

**Open-Sourcefy** is a production-grade AI-powered binary decompilation system that reconstructs compilable C source code from Windows PE executables using a 17-agent Matrix pipeline with Ghidra integration and NSA-level security standards.

**MISSION**: Transform Windows PE executables into compilable C source code with military-grade precision and zero tolerance for failures.

## CRITICAL SYSTEM SPECIFICATIONS

### Platform Requirements (ABSOLUTE)
- **WINDOWS EXCLUSIVE**: Windows 10/11 64-bit ONLY
- **NO FALLBACKS**: Zero alternatives, no graceful degradation
- **VISUAL STUDIO 2022 PREVIEW**: Exclusive build system (no alternatives)
- **FAIL-FAST**: Immediate termination on missing requirements

### Architecture Overview
- **17-Agent Matrix Pipeline**: Comprehensive agent framework with production-ready infrastructure
- **Master-First Execution**: Agent 0 (Deus Ex Machina) coordinates all operations
- **Zero-Fallback Design**: Single correct implementation path only
- **NSA-Level Security**: Zero tolerance for vulnerabilities

## Development Commands

### Essential Operations
```bash
# Full pipeline execution
python3 main.py

# Environment validation (MANDATORY before work)
python3 main.py --verify-env

# Update mode (incremental development)
python3 main.py --update

# Agent-specific testing
python3 main.py --agents 1,3,7

# Pipeline validation
python3 main.py --validate-pipeline basic
```

### Testing Commands
```bash
# Comprehensive test suite
python3 -m unittest discover tests -v

# Agent-specific testing
python3 -m unittest tests.test_agent_individual -v

# Integration testing
python3 main.py --validate-pipeline comprehensive
```

### Build System Commands
```bash
# Validate VS2022 Preview installation
python3 main.py --verify-env

# Debug build configuration
python3 main.py --debug --profile

# Configuration summary
python3 main.py --config-summary
```

## Agent Architecture

### Matrix Agent Hierarchy

```
MASTER ORCHESTRATOR:
├── Agent 0: Deus Ex Machina (Pipeline Coordination)

FOUNDATION PHASE (Sequential):
├── Agent 1: Sentinel (Binary Analysis & Import Recovery) 🚨 CRITICAL FIX NEEDED
├── Agent 2: Architect (PE Structure & Resource Extraction)
├── Agent 3: Merovingian (Advanced Pattern Recognition)
└── Agent 4: Agent Smith (Code Flow Analysis)

ADVANCED ANALYSIS PHASE (Parallel):
├── Agent 5: Neo (Advanced Decompilation Engine)
├── Agent 6: Trainman (Assembly Analysis)
├── Agent 7: Keymaker (Resource Reconstruction)
└── Agent 8: Commander Locke (Build System Integration)

RECONSTRUCTION PHASE (Sequential):
├── Agent 9: The Machine (Resource Compilation) 🚨 CRITICAL FIX NEEDED
├── Agent 10: Twins (Binary Diff & Validation)
├── Agent 11: Oracle (Semantic Analysis)
└── Agent 12: Link (Code Integration)

FINAL PROCESSING PHASE (Parallel):
├── Agent 13: Agent Johnson (Quality Assurance)
├── Agent 14: Cleaner (Code Cleanup)
├── Agent 15: Analyst (Final Validation)
└── Agent 16: Agent Brown (Output Generation)
```

### Critical Issues & Solutions

#### PRIMARY BOTTLENECK: Import Table Mismatch
- **Issue**: Original binary imports 538 functions from 14 DLLs, reconstruction only includes 5
- **Impact**: ~25% pipeline failure rate
- **Root Cause**: Agent 9 (The Machine) ignores rich import data from Agent 1 (Sentinel)
- **Solution Status**: ✅ Research complete, implementation ready
- **Expected Impact**: 60% → 85% success rate improvement

#### Agent 1 (Sentinel) - Import Table Recovery
**CRITICAL FIX REQUIRED**: Enhance import table reconstruction
- MFC 7.1 signature detection and resolution
- Ordinal-to-function name mapping
- Complete DLL dependency analysis
- Rich header processing for compiler metadata

#### Agent 9 (The Machine) - Resource Compilation
**CRITICAL FIX REQUIRED**: Repair data flow from Agent 1
- Utilize comprehensive import analysis from Sentinel
- Generate complete function declarations for all 538 imports
- Update VS project with all 14 DLL dependencies
- Handle MFC 7.1 compatibility requirements

## File Structure & Protection

### CRITICAL PROTECTION RULES
```
NEVER DELETE ANYTHING IN:
├── prompts/              # Critical project documentation
├── src/core/agents/      # Matrix agent implementations
├── main.py              # Pipeline entry point
├── rules.md             # Absolute project rules
├── CLAUDE.md            # This file
└── build_config.yaml    # Build system configuration
```

### Project Structure
```
src/core/
├── matrix_pipeline_orchestrator.py  # Master Pipeline Controller
├── agents/                          # 17 Matrix Agents (0-16)
│   ├── agent00_deus_ex_machina.py   # Master Orchestrator
│   ├── agent01_sentinel.py          # Binary Analysis (FIX NEEDED)
│   ├── agent02_architect.py         # PE Structure Analysis
│   ├── agent09_the_machine.py       # Resource Compilation (FIX NEEDED)
│   └── ... (all 17 agents)
├── config_manager.py               # Configuration Management
├── build_system_manager.py         # VS2022 Build Integration
├── shared_components.py            # Shared Agent Framework
└── exceptions.py                   # Error Handling System

output/{binary_name}/{timestamp}/
├── agents/          # Agent-specific outputs
├── ghidra/          # Decompilation results
├── compilation/     # MSBuild artifacts
├── reports/         # Pipeline reports
└── logs/            # Execution logs

tests/               # Comprehensive test suite (>90% coverage)
docs/                # Architecture and specification documentation
```

## Configuration Management

### Build System Configuration (build_config.yaml)
```yaml
build_system:
  visual_studio:
    version: "2022_preview"
    installation_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview"
    cl_exe_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.XX.XXXXX/bin/Hostx64/x64/cl.exe"
    msbuild_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/MSBuild/Current/Bin/MSBuild.exe"
  
  # NO FALLBACK PATHS - CONFIGURED PATHS ONLY
  strict_mode: true
  fail_on_missing_tools: true
```

### Environment Variables
```bash
# Required for AI functionality
ANTHROPIC_API_KEY=your_api_key_here

# Optional debug settings
MATRIX_DEBUG=true
MATRIX_AI_ENABLED=true
GHIDRA_HOME=/path/to/ghidra
JAVA_HOME=/path/to/java
```

## Quality Standards

### Code Quality Requirements (MANDATORY)
- **NSA-Level Security**: Zero tolerance for vulnerabilities
- **SOLID Principles**: Architectural compliance enforced
- **Test Coverage**: >90% requirement (automatically validated)
- **No Hardcoded Values**: All configuration external
- **Error Handling**: Fail-fast with comprehensive validation

### Performance Standards
- **Memory Optimization**: Designed for 16GB+ systems
- **Execution Time**: Pipeline completion under 30 minutes
- **Success Rate**: Target 85% binary reconstruction accuracy
- **Quality Thresholds**: 75% minimum at each pipeline stage

### Security Standards
- **Input Sanitization**: All inputs validated and sanitized
- **Secure File Handling**: Temporary files properly managed
- **Access Control**: Strict permission validation
- **No Credential Exposure**: Zero secrets in logs or output

## Development Workflow

### Pre-Development Checklist
1. ✅ Read rules.md (MANDATORY)
2. ✅ Validate environment: `python3 main.py --verify-env`
3. ✅ Check current todo list status
4. ✅ Review agent documentation for target work
5. ✅ Ensure VS2022 Preview installation

### Development Process
1. **Rule Compliance**: Every change must comply with rules.md
2. **Testing**: Comprehensive test suite maintained at all times
3. **Documentation**: Update relevant docs for any changes
4. **Validation**: Run full test suite before completion
5. **Configuration**: Never hardcode values, use config files

### Git Workflow
```bash
# Check current status
git status

# Stage and commit (with required format)
git add .
git commit -m "Your descriptive message

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Never push without explicit user request
```

## Agent Development Guidelines

### Matrix Agent Implementation Standards
1. **Inheritance**: Use appropriate base class from `shared_components`
2. **Naming**: Follow Matrix conventions exactly (no deviations)
3. **Methods**: Implement `execute_matrix_task()` and `_validate_prerequisites()`
4. **Configuration**: Use `shared_components` for all shared functionality
5. **Validation**: Quality thresholds enforced at each stage

### Agent Refactor Priorities

#### HIGH PRIORITY (CRITICAL FIXES)
1. **Agent 1 (Sentinel)**: Import table reconstruction enhancement
2. **Agent 9 (The Machine)**: Data flow repair from Sentinel
3. **Core Pipeline**: MFC 7.1 compatibility integration

#### MEDIUM PRIORITY (OPTIMIZATION)
4. **Agent 0 (Deus Ex Machina)**: Enhanced coordination algorithms
5. **Agents 2-4**: Foundation phase optimization
6. **Agents 5-8**: Advanced analysis enhancement
7. **Agents 10-12**: Reconstruction optimization
8. **Agents 13-16**: Final processing refinement

#### LOW PRIORITY (ENHANCEMENT)
9. **Build System**: Enhanced VS2022 integration
10. **Testing**: Expanded test coverage and validation
11. **Documentation**: Comprehensive API documentation
12. **Performance**: Pipeline execution optimization

## Troubleshooting Guide

### Common Issues

#### Import Table Mismatch (PRIMARY ISSUE)
- **Symptoms**: Binary validation fails, missing DLL dependencies
- **Root Cause**: Agent 9 data flow from Agent 1
- **Solution**: Implement comprehensive import table fix
- **Files**: `agent01_sentinel.py`, `agent09_the_machine.py`

#### Build System Failures
- **Symptoms**: Compilation errors, missing tools
- **Root Cause**: VS2022 Preview path configuration
- **Solution**: Validate `build_config.yaml` paths
- **Command**: `python3 main.py --verify-env`

#### Agent Execution Failures
- **Symptoms**: Agent prerequisite validation failures
- **Root Cause**: Missing dependencies or configuration errors
- **Solution**: Check logs in `output/{binary}/logs/`
- **Debug**: `python3 main.py --debug`

### Critical Error Codes
- **E001**: Missing VS2022 Preview installation
- **E002**: Invalid build_config.yaml configuration
- **E003**: Insufficient system resources
- **E004**: Agent prerequisite validation failure
- **E005**: Import table reconstruction failure

## Resource Links

### Documentation
- **System Architecture**: docs/SYSTEM_ARCHITECTURE.md
- **Absolute Rules**: rules.md
- **Agent Documentation**: src/core/agents/
- **API Reference**: docs/ (when generated)

### External Resources
- **Ghidra Documentation**: ghidra/docs/
- **VS2022 Preview**: Microsoft Visual Studio documentation
- **MFC 7.1 Reference**: Legacy Microsoft documentation

## Current Status Summary

### Implementation Status
- **Agent Implementation**: 17 agent files implemented (Agent 00 + Agents 1-16) ✅
- **Pipeline Success Rate**: 16/16 agents achieving 100% success rate (latest run successful)
- **Code Quality**: Production-ready with NSA-level standards ✅
- **Testing Framework**: Comprehensive test suite with quality validation ✅
- **Documentation**: Comprehensive and up-to-date ✅

### Critical Work Items
1. **URGENT**: Fix Agent 1 import table reconstruction
2. **URGENT**: Repair Agent 9 data flow from Agent 1
3. **HIGH**: Implement MFC 7.1 compatibility layer
4. **MEDIUM**: Optimize pipeline execution performance
5. **LOW**: Enhance documentation and testing coverage

---

## FINAL REMINDERS

**🚨 ABSOLUTE COMPLIANCE**: All work must comply with rules.md without exception
**⚡ ZERO TOLERANCE**: No fallbacks, no alternatives, no compromises
**🛡️ NSA STANDARDS**: Military-grade security and quality required
**🎯 MISSION FOCUS**: Windows PE to compilable C source code transformation

**SUCCESS METRICS**: 85% pipeline success rate with binary-identical reconstruction capability