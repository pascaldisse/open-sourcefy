# CLAUDE.md

🚨 **OBLIGATORY READING OF rules.md FILE OR DEATH SENTENCE** 🚨
READ /mnt/c/Users/pascaldisse/Downloads/open-sourcefy/rules.md IMMEDIATELY BEFORE ANY WORK
ALL RULES IN rules.md ARE ABSOLUTE AND NON-NEGOTIABLE

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Open-Sourcefy is an AI-powered binary decompilation system that reconstructs compilable C source code from Windows PE executables using a 17-agent Matrix pipeline with Ghidra integration.

**WINDOWS ONLY SYSTEM**: Exclusively supports Windows PE executables with Visual Studio/MSBuild compilation.
**Primary Target**: Matrix Online launcher.exe binary

### Current Status (December 2024)
- **Pipeline Validation**: ~60% overall success rate
- **Major Bottleneck**: Import table mismatch (64.3% discrepancy, 538→5 DLLs)
- **Architecture**: 17-agent Matrix pipeline with master-first parallel execution
- **Recent Research**: Import table reconstruction solutions identified

## Development Commands

### Basic Operations
```bash
# Run full pipeline
python3 main.py

# Test specific agents
python3 main.py --agents 1,3,7

# Update mode (incremental development)  
python3 main.py --update

# Environment validation
python3 main.py --verify-env
```

### Testing
```bash
# Run all tests
python3 -m unittest discover tests -v

# Pipeline validation
python3 main.py --validate-pipeline basic
```

## Architecture Overview

### 17-Agent Matrix Pipeline
- **Agent 0**: Deus Ex Machina (Master Orchestrator)
- **Agents 1-4**: Foundation & Core Analysis (Sentinel, Architect, Merovingian, Agent Smith)
- **Agents 5-12**: Advanced Analysis & Reconstruction (Neo, Twins, Trainman, Keymaker, Locke, Machine, Oracle, Link)
- **Agents 13-16**: Final Processing & QA (Johnson, Cleaner, Analyst, Brown)

### Key Components
- **Pipeline Orchestrator**: `core/matrix_pipeline_orchestrator.py`
- **Agent Framework**: `core/matrix_agents_v2.py`
- **Configuration**: `core/config_manager.py`
- **Build System**: `core/build_system_manager.py` (VS2022 Preview only)

### Output Structure
```
output/{binary_name}/{timestamp}/
├── agents/          # Agent-specific outputs
├── ghidra/          # Decompilation results
├── compilation/     # MSBuild artifacts
├── reports/         # Pipeline reports
└── logs/            # Execution logs
```

## Critical Issues & Solutions

### Import Table Mismatch (PRIMARY BOTTLENECK)
**Problem**: Original binary imports 538 functions from 14 DLLs, reconstruction only includes 5 basic DLLs
**Impact**: ~25% validation failure
**Solution Status**: ✅ Research complete, implementation ready

**Key Findings**:
- MFC 7.1 signatures available from [VS2003 Documentation](https://www.microsoft.com/en-us/download/details.aspx?id=55979)
- Ordinal resolution via `dumpbin /exports MFC71.DLL`
- VS2022 incompatible with MFC 7.1 (requires alternative build approach)
- Agent 9 ignores rich import data from Agent 1

**Fix Strategy**: 
1. Repair Agent 9 data flow (uses Sentinel import analysis)
2. Generate comprehensive function declarations
3. Update VS project with all 14 DLL dependencies
4. Handle MFC 7.1 compatibility requirements

**Expected Impact**: 60% → 85% pipeline validation

## Configuration

### Environment Variables
```bash
GHIDRA_HOME=/path/to/ghidra
JAVA_HOME=/path/to/java
MATRIX_DEBUG=true
MATRIX_AI_ENABLED=true
```

### Build System (FIXED CONFIGURATION)
- **Compiler**: VS2022 Preview cl.exe (centralized paths only)
- **MSBuild**: VS2022 Preview MSBuild
- **No Fallbacks**: Only configured paths in build_config.yaml used
- **Agent Integration**: All build operations via centralized build manager

### File Structure
```
src/core/
├── agents/              # 17 Matrix agents (0-16)
├── matrix_pipeline_orchestrator.py
├── matrix_agents_v2.py
├── config_manager.py
└── shared_components.py

tests/                   # Unit and integration tests
docs/                    # Analysis reports
ghidra/                  # Ghidra 11.0.3 installation
```

## Quality Standards

- **SOLID Principles**: Mandatory
- **No Hardcoded Values**: All configuration external
- **NSA-Level Security**: Zero tolerance for vulnerabilities
- **Comprehensive Testing**: >90% coverage requirement
- **Agent Validation**: Quality thresholds enforced at each stage

## Development Guidelines

### Matrix Agent Development
1. Inherit from appropriate base class (AnalysisAgent, DecompilerAgent, etc.)
2. Follow Matrix naming conventions
3. Implement required methods: `execute_matrix_task()`, `_validate_prerequisites()`
4. Use shared components from `shared_components`
5. No hardcoded paths or values

### Testing Approach
- **Framework**: Python unittest (not pytest)
- **Categories**: Integration, unit, validation, quality assurance
- **Built-in Validation**: Agent results validated for quality/completeness

## File Protection Rules

### CRITICAL PROTECTION
**NEVER DELETE ANYTHING IN prompts/ DIRECTORY**
- Contains critical project documentation
- NEVER use deletion commands on prompts/ files
- Update/organize only, never delete

### Protection Hierarchy
1. **CRITICAL**: prompts/, CLAUDE.md, main.py, src/core/
2. **IMPORTANT**: docs/, requirements.txt, configuration files
3. **SAFE**: temp/, output/, logs/, cache/

## Git Workflow

```bash
# Check status
git status

# Stage and commit
git add .
git commit -m "Your message

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

**Key Resources**:
- Import table fixes: `IMPORT_TABLE_FIX_STRATEGIES.md`
- Research answers: `IMPORT_TABLE_RESEARCH_QUESTIONS.md`
- Agent documentation: `src/core/agents/`
- Validation reports: `output/*/reports/`