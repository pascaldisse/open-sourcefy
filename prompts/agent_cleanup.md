# Project-Wide Cleanup and Refactoring Prompt
## Rules.md Compliant System Cleanup

## Objective
Clean up the entire open-sourcefy project according to STRICT rules.md compliance by removing ALL verbose, mock, and overly complex code. Refactor, rename, and merge where needed to create a clean, production-ready system that follows NSA-level security standards.

## CRITICAL OUTPUT REQUIREMENT
⚠️ **ALL OUTPUT MUST GO TO `/output` DIRECTORY ONLY** ⚠️
- NO files should be created outside of `/output/` or its subdirectories
- NO temporary files in project root, `/tmp`, or other locations
- ALL agent results, logs, reports, and analysis data MUST be in `/output/`
- Use structured subdirectories under `/output/` as defined in CLAUDE.md
- Ensure all file paths are relative to `/output/` base directory

## RULES.MD COMPLIANCE REQUIREMENTS

🚨 **ABSOLUTE REQUIREMENTS FROM RULES.MD** 🚨

### NO MOCK IMPLEMENTATIONS
- **ZERO TOLERANCE**: No mock agents, mock AI engines, or mock implementations
- **Real AI Only**: Use actual Claude AI system from ai_system.py 
- **No Fallbacks**: Remove all mock fallback systems and dummy engines
- **Production Only**: All code must be production-ready or properly raise NotImplementedError

### NSA-LEVEL SECURITY STANDARDS
- **Zero Vulnerabilities**: No security holes, exposed secrets, or unsafe operations
- **Input Validation**: All inputs must be validated and sanitized
- **Error Handling**: Comprehensive error handling with fail-fast validation
- **No Debug Code**: Remove all debug prints, temporary files outside /output/, insecure logging

### VS2022 PREVIEW ONLY
- **No Fallbacks**: Remove all VS2019, VS2017, or other compiler fallbacks
- **Centralized Paths**: Only use build_config.yaml configured paths
- **MSBuild Only**: No alternative build systems (remove CMake, Make, etc.)

### PRODUCTION-READY CODE ONLY
- **No Placeholders**: Remove all TODO comments, placeholder functions, dummy returns
- **Real Implementation**: All functions must either work or raise NotImplementedError with detailed explanation
- **Quality Metrics**: All code must meet production quality standards
- **Clean Architecture**: Follow SOLID principles, proper error handling, comprehensive logging

## Tasks to Complete

### 1. MANDATORY: Remove ALL Mock/Verbose/Complex Code
**Target Areas**:
- **Agents**: `src/core/agents/` - Remove placeholder implementations
- **Core Modules**: `src/core/` - Fix incomplete functionality  
- **Utilities**: `src/utils/` - Remove stub implementations
- **ML Components**: `src/ml/` - Replace placeholder algorithms
- **Configuration**: Root level config files with dummy values
- **Utils**: Any utility modules with placeholder functionality

**RULES.MD VIOLATIONS TO REMOVE**:
- **Mock Implementations**: Any class/function containing "Mock", "Dummy", "Fake", "Stub"
- **Verbose Code**: Overly complex implementations when simple ones suffice
- **Debug Code**: Print statements, debug flags, development-only features  
- **Fallback Systems**: Alternative implementations for missing dependencies
- **Complex Test Mocks**: Overly elaborate test fixtures and mock systems
- **Placeholder Values**: Hardcoded confidence scores, fake data structures
- **TODO Comments**: Unimplemented features without clear implementation plans

**Replacement Strategy**: Replace with `NotImplementedError` exceptions that clearly explain:
- What functionality is missing
- What would be required to implement it properly  
- Any dependencies or prerequisites needed
- Estimated complexity/effort level

**Example Transformation**:
```python
# BEFORE (dummy code)
def _analyze_control_flow(self, data):
    return {
        'control_flow_graphs': {},
        'basic_blocks': {},
        'confidence': 0.3
    }

# AFTER (proper error)
def _analyze_control_flow(self, data):
    raise NotImplementedError(
        "Control flow analysis not implemented - requires CFG construction "
        "and basic block identification algorithms"
    )
```

### 2. CRITICAL: Test Suite Rules.md Compliance
**IMMEDIATE PRIORITY**: Fix test suite to align with rules.md

**Test Violations to Fix**:
- **❌ Mock AI Engines**: Remove all `test_mock_ai_engine_available()` and mock AI components
- **❌ Fallback Testing**: Remove tests that validate fallback systems
- **❌ Complex Mock Systems**: Simplify overly elaborate test mocking
- **❌ Verbose Test Code**: Reduce overly complex test implementations

**Required Test Updates**:
- **✅ Real AI Testing**: Update tests to use actual `ai_system.py` Claude integration
- **✅ Production Testing**: Test actual agent implementations, not mocks
- **✅ Clean Validation**: Test real output validation using AI-enhanced analysis
- **✅ Simple Mocking**: Use minimal, necessary mocking only

**Test File Priority Order**:
1. `tests/test_agent_output_validation.py` - ✅ ALREADY FIXED 
2. `tests/test_missing_agent_validation.py` - ✅ ALREADY FIXED
3. `tests/test_phase4_comprehensive.py` - ✅ ALREADY FIXED
4. All other test files - NEEDS REVIEW

### 3. Project-Wide Obsolete Code Removal
**Target Areas**:
- **Unused Import Statements**: Across all Python files
- **Dead Code Paths**: Functions/classes that are never called
- **Commented-Out Code**: Remove old code blocks left as comments
- **Debug Statements**: Remove development-only print statements and debug code
- **Duplicate Documentation**: Remove redundant README files and documentation
- **Old Test Files**: Remove obsolete or non-functional test files
- **Unused Configuration**: Remove unused config sections and parameters
- **Legacy Files**: Remove old build files, batch files, or utilities no longer used
- **Placeholder Files**: Remove empty or minimal placeholder files
- **Backup Files**: Remove .backup, .old, or temporary files

### 3. Refactor Similar/Duplicate Functionality Across Project
**Core Module Consolidation**:
- **Binary Format Parsing**: Consolidate PE/ELF/Mach-O logic into shared utilities
- **Ghidra Integration**: Create unified Ghidra interface in `src/core/ghidra_processor.py`
- **File I/O Operations**: Standardize file reading/writing patterns project-wide
- **Error Handling**: Unify error handling approaches across all modules
- **Logging**: Consolidate logging configuration and patterns
- **Configuration Management**: Centralize config handling in `config_manager.py`

**Agent Consolidation**:
- **Similar Analysis Logic**: Merge overlapping analysis functions between agents
- **Shared Data Structures**: Create common data models for agent communication
- **Duplicate Utilities**: Move shared helper functions to `src/utils/`

**Documentation Consolidation**:
- **Multiple README files**: Merge redundant documentation
- **Scattered documentation**: Centralize in `docs/` directory
- **Inconsistent formatting**: Standardize markdown formatting project-wide

### 4. Project Structure Optimization
**Directory Reorganization**:
- **Test Directories**: Consolidate multiple test directories (`test-*`, `*-test/`)
- **Output Directories**: Ensure ALL output goes to `/output/` with structured subdirectories:
  - `/output/agents/` - Agent-specific analysis outputs
  - `/output/ghidra/` - Ghidra decompilation results and projects  
  - `/output/compilation/` - Compilation artifacts and generated source
  - `/output/reports/` - Pipeline execution reports and summaries
  - `/output/logs/` - Execution logs and debug information
  - `/output/temp/` - Temporary files (auto-cleaned)
- **Documentation**: Centralize all docs in `docs/` directory
- **Configuration**: Centralize config files in logical locations
- **Utils**: Organize utility modules in appropriate directories

**Module Reorganization**:
- **Agent Dependencies**: Simplify agent dependency chains where possible
- **Core Module Separation**: Ensure clean separation between core, agents, ml, and utils
- **Import Structure**: Fix circular imports and optimize import paths

### 5. Standardize Code Structure Project-Wide
**Agent Standardization**:
Ensure all agents follow consistent patterns:
```python
class AgentXX_Name(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_id=XX,
            name="Name",
            dependencies=[list_of_dependencies]
        )
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        # Dependency validation
        # Main execution logic
        # Error handling
        # Return AgentResult
    
    def _helper_method(self, params):
        # Either real implementation or NotImplementedError
```

**Core Module Standardization**:
```python
# Consistent module structure
"""
Module docstring explaining purpose and usage
"""

import standard_libraries
import third_party_libraries
from project_modules import local_imports

class ClassName:
    """Clear class documentation"""
    
    def __init__(self):
        """Clear constructor documentation"""
        pass
    
    def public_method(self, params):
        """Clear method documentation with type hints"""
        pass
    
    def _private_method(self, params):
        """Private method documentation"""
        pass
```

**File Structure Standardization**:
- **Consistent file headers**: License, author, purpose
- **Consistent import ordering**: stdlib, third-party, local
- **Type hints**: Add type hints throughout project
- **Error handling**: Consistent exception handling patterns

### 6. Project-Wide Documentation and Naming
**Documentation Standardization**:
- **Update Docstrings**: Ensure all classes and methods have clear, accurate docstrings
- **README Consolidation**: Merge multiple README files into coherent documentation
- **API Documentation**: Generate comprehensive API docs from docstrings
- **User Guide**: Create/update user-facing documentation
- **Developer Guide**: Create/update contributor documentation

**Naming Consistency**:
- **File Naming**: Ensure consistent file naming conventions (snake_case for Python)
- **Class Naming**: Consistent PascalCase for classes
- **Function Naming**: Consistent snake_case for functions
- **Variable Naming**: Consistent and descriptive variable names
- **Constant Naming**: Consistent UPPER_CASE for constants

**Comment Cleanup**:
- **Remove Misleading Comments**: Remove comments that suggest functionality exists when it doesn't
- **Update Outdated Comments**: Fix comments that no longer match the code
- **Add Missing Comments**: Add comments for complex logic sections
- **Remove Redundant Comments**: Remove comments that just repeat what code does

## Specific Areas to Focus On

### PRIORITY 1: Rules.md Compliance Enforcement

#### 1. **Elite Refactored Agents** (`src/core/agents/`)
**✅ PRODUCTION-READY AGENTS** (Rules.md Compliant):
- **Agent 0**: Deus Ex Machina (Master Orchestrator) - ✅ ELITE REFACTOR
- **Agent 14**: The Cleaner (Security & Cleanup) - ✅ ELITE REFACTOR  
- **Agent 15**: Analyst (Intelligence Synthesis) - ✅ ELITE REFACTOR
- **Agent 16**: Agent Brown (QA Validation) - ✅ ELITE REFACTOR

**🚧 NEEDS RULES.MD COMPLIANCE REVIEW**:
- **Agent 1**: Sentinel (Binary Discovery) - Contains some verbose code
- **Agent 2**: Architect (Architecture Analysis) - Some complex implementations
- **Agent 3**: Merovingian (Function Detection) - Has placeholder patterns
- **Agent 4**: Agent Smith (Structure Analysis) - Contains mock elements
- **Agent 5**: Neo (Advanced Decompilation) - Needs Ghidra integration cleanup

**❌ MISSING IMPLEMENTATIONS** (Need Agent Framework):
- Agents 6-13: Need complete implementation using production framework

#### 2. **Core Infrastructure** (`src/core/`)
- **agent_base.py**: Base agent framework
- **parallel_executor.py**: Agent execution management
- **enhanced_parallel_executor.py**: Enhanced execution logic
- **ghidra_processor.py**: Ghidra integration
- **ghidra_headless.py**: Headless Ghidra operations
- **config_manager.py**: Configuration management
- **environment.py**: Environment setup
- **error_handler.py**: Error handling framework
- **performance_monitor.py**: Performance monitoring
- **ai_enhancement.py**: AI enhancement capabilities

#### 3. **Machine Learning** (`src/ml/`)
- **pattern_engine.py**: ML-based pattern recognition

#### 4. **Utilities** (`src/utils/`)
- All utility modules and helper functions

#### 5. **Root Level Files**
- **main.py**: Main entry point
- **requirements.txt**: Dependencies
- Multiple documentation files (README.md, DOCUMENTATION.md, etc.)
- Configuration files

#### 6. **Documentation Cleanup** (`docs/` and root level)
- Multiple README files
- Agent-specific documentation
- Architecture documentation
- Integration guides

#### 7. **Test and Output Directories**
- **test-*/** directories (multiple scattered test directories)
- **output/** directory structure
- **build/** and **dist/** directories
- Temporary directories

### Common Dummy Code Patterns to Remove Project-Wide:
```python
# Hardcoded confidence scores
'confidence': 0.75

# Empty placeholder returns
return {}
return []

# Placeholder messages
'analysis': 'This would require implementation'

# TODO comments without plans
# TODO: Add real implementation

# Fake data structures
'functions': [],
'imports': {},
'exports': {}

# Placeholder function implementations (from recent linter changes detected)
def _enhance_function_signature(self, ...):
    return {
        'name': func_name,
        'parameters': [],
        'return_type': 'unknown',
        'calling_convention': 'unknown',
        'confidence': 0.5
    }

# Generated placeholder functions
def _generate_init_function(self, name, index):
    return f"// Placeholder function {name}"

# Hardcoded file size logic
if file_size > 10 * 1024 * 1024:  # Arbitrary threshold

# Placeholder analysis results  
'analysis_quality': 'unknown'
'program_analysis': 'Placeholder analysis result'
```

## Success Criteria

### Code Quality Metrics
- [ ] **Zero Dummy Code**: All placeholder implementations replaced with proper NotImplementedError exceptions
- [ ] **No Hardcoded Values**: Remove all hardcoded confidence scores, thresholds, and fake data
- [ ] **Consistent Error Handling**: Unified error handling patterns across entire project
- [ ] **Clean Codebase**: All dead/obsolete code removed project-wide
- [ ] **Consolidated Functionality**: Duplicate code merged into shared utilities
- [ ] **Complete Documentation**: Clear, accurate docstrings for all public methods and classes
- [ ] **Naming Consistency**: Consistent naming conventions across entire project

### Structural Improvements
- [ ] **Organized Directory Structure**: Logical organization of files and directories
- [ ] **Centralized Configuration**: All configuration consolidated in appropriate locations
- [ ] **Clean Dependencies**: Optimal import structure with no circular dependencies
- [ ] **Type Safety**: Type hints added throughout project
- [ ] **Test Organization**: Test files properly organized and functional

### Documentation Quality
- [ ] **Unified Documentation**: Single source of truth for each type of documentation
- [ ] **Complete API Docs**: Comprehensive API documentation generated from docstrings
- [ ] **User Guides**: Clear user-facing documentation
- [ ] **Developer Guides**: Comprehensive contributor documentation

## Implementation Strategy

### Phase 1: IMMEDIATE Rules.md Compliance (Priority 1)
1. **Mock Code Elimination**: Remove ALL mock implementations system-wide
2. **Test Suite Compliance**: Update tests to use real AI system only
3. **Verbose Code Reduction**: Simplify overly complex implementations
4. **Debug Code Removal**: Remove debug prints, temp files outside /output/

### Phase 2: Agent System Cleanup (Priority 2)  
1. **Elite Agent Validation**: Verify Agents 0,14,15,16 meet NSA-level standards
2. **Production Agent Fixes**: Clean up Agents 1-5 to production standards
3. **Missing Agent Framework**: Implement Agents 6-13 using production patterns
4. **Integration Testing**: Validate cleaned agents work with pipeline

### Phase 3: Infrastructure Compliance (Priority 3)
1. **AI System Cleanup**: Ensure only real Claude AI integration exists
2. **Build System Cleanup**: Remove non-VS2022 Preview fallbacks  
3. **Configuration Cleanup**: Centralize all paths in build_config.yaml
4. **Security Validation**: NSA-level security review of all components

### Phase 4: Final Production Readiness (Priority 4)
1. **Quality Assurance**: Comprehensive quality metrics validation
2. **Documentation Update**: Update all docs to reflect clean architecture
3. **Performance Testing**: Validate cleaned system performance
4. **Deployment Readiness**: Final production deployment validation

## Automated Tools and Commands

**Note**: Manual validation required for these cleanup tasks.

### Environment Validation (AUTOMATED)
```bash
# Use automation script for environment validation
python3 main.py --verify-env
python3 main.py --verify-env --setup-help  # Get setup instructions
python3 main.py --verify-env --json        # JSON output for scripting
```

### File Operations (AUTOMATED)
```bash
# Use automation script for file operations
mkdir -p output/cleanup_session
find output/temp -type f -mtime +1 -delete  # Clean temp files older than 1 day
find . -name '*.py' | wc -l  # Count Python files
find src -name '*.py'
```

### Manual Finding Dummy Code (when automation isn't sufficient)
```bash
# Search for common dummy patterns
grep -r "confidence.*0\." src/
grep -r "return {}" src/
grep -r "TODO.*implementation" src/
grep -r "placeholder" src/
grep -r "not implemented" src/

# Find hardcoded values
grep -r "0\.75\|0\.8\|0\.5" src/ | grep confidence

# Find empty/minimal functions
grep -A 5 -B 2 "def.*:" src/ | grep -A 5 "return {}\|return \[\]"
```

### Code Quality Checks
```bash
# Check for unused imports
python -m pyflakes src/

# Check code style
python -m flake8 src/

# Check type hints
python -m mypy src/

# Find duplicate code
python -m pylint src/ --disable=all --enable=duplicate-code
```

### Project Structure Analysis (AUTOMATED + Manual)
```bash
# Use automation for comprehensive analysis
find src -name '*.py' -exec wc -l {} +  # Count lines in source files

# Manual commands for specific needs
find . -name "*.py" -exec basename {} \; | sort | uniq -d

# Analyze import dependencies
python -c "
import ast
import os
for root, dirs, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            print(f'{root}/{file}')
"

# Count lines of code by module
find src/ -name "*.py" -exec wc -l {} + | sort -n
```

## Implementation Notes
- **Preserve Working Code**: Ensure any real, working implementations are preserved
- **Test Integration**: Test that cleaned modules still integrate properly with the pipeline
- **Incremental Approach**: Work systematically through the project to avoid breaking changes
- **Documentation Updates**: Update documentation as code is cleaned and refactored
- **Version Control**: Commit changes incrementally to track progress and enable rollbacks
- **OUTPUT DIRECTORY ENFORCEMENT**: Verify all code writes ONLY to `/output/` directory:
  - Update all hardcoded paths to use `/output/` base
  - Fix any code that writes to project root or other directories
  - Ensure temp files are created in `/output/temp/`
  - Validate all agent context uses proper output paths