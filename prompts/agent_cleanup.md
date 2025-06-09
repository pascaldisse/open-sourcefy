# Project-Wide Cleanup and Refactoring Prompt

## Objective
Clean up the entire open-sourcefy project by removing dummy code, refactoring implementations, removing obsolete code/scripts, fixing inconsistencies, and optimizing the overall codebase structure.

## CRITICAL OUTPUT REQUIREMENT
⚠️ **ALL OUTPUT MUST GO TO `/output` DIRECTORY ONLY** ⚠️
- NO files should be created outside of `/output/` or its subdirectories
- NO temporary files in project root, `/tmp`, or other locations
- ALL agent results, logs, reports, and analysis data MUST be in `/output/`
- Use structured subdirectories under `/output/` as defined in CLAUDE.md
- Ensure all file paths are relative to `/output/` base directory

## Tasks to Complete

### 1. Project-Wide Dummy Code Removal
**Target Areas**:
- **Agents**: `src/core/agents/` - Remove placeholder implementations
- **Core Modules**: `src/core/` - Fix incomplete functionality  
- **Utilities**: `src/utils/` - Remove stub implementations
- **ML Components**: `src/ml/` - Replace placeholder algorithms
- **Configuration**: Root level config files with dummy values
- **Utils**: Any utility modules with placeholder functionality

**Search Patterns**: Look for functions that return:
- Hardcoded placeholder values (`confidence: 0.75`)
- Empty dictionaries/lists (`return {}`, `return []`)
- Dummy data structures with fake information
- TODO comments without implementation plans
- Placeholder strings (`'placeholder'`, `'not implemented'`)

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

### 2. Project-Wide Obsolete Code Removal
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

### Core Project Areas (Priority Order)

#### 1. **Agent System** (`src/core/agents/`)
**Status Overview**:
- ✅ **Agent 7 (AdvancedDecompiler)**: Recently cleaned up (but may need further work)
- ✅ **Agent 11 (GlobalReconstructor)**: Recently cleaned up (but may need further work)
- ✅ **Agent 14 (AdvancedGhidra)**: Recently cleaned up
- ✅ **Agent 15 (MetadataAnalysis)**: Recently cleaned up

**Remaining Agents to Review**:
- Agent 1 (BinaryDiscovery)
- Agent 2 (ArchitectureAnalysis) 
- Agent 3 (SmartErrorPatternMatching)
- Agent 4 (BasicDecompiler)
- Agent 5 (BinaryStructureAnalyzer)
- Agent 6 (OptimizationMatcher)
- Agent 8 (BinaryDiffAnalyzer)
- Agent 9 (AdvancedAssemblyAnalyzer)
- Agent 10 (ResourceReconstructor)
- Agent 12 (CompilationOrchestrator)
- Agent 13 (FinalValidator)

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

### Phase 1: Assessment and Planning (Day 1)
1. **Full Project Scan**: Use automated tools to identify all dummy code patterns
2. **Dependency Analysis**: Map out all module dependencies and potential circular imports
3. **Duplication Detection**: Identify all duplicate functionality across the project
4. **Documentation Audit**: Catalog all documentation files and identify consolidation opportunities

### Phase 2: Core Infrastructure Cleanup (Days 2-3)
1. **Base Framework**: Clean up agent_base.py and core infrastructure
2. **Shared Utilities**: Consolidate common functionality in src/utils/
3. **Configuration**: Centralize all configuration management
4. **Error Handling**: Implement consistent error handling patterns

### Phase 3: Agent System Cleanup (Days 4-6)
1. **Work Through Agents 1-15**: Systematic cleanup of each agent
2. **Agent Communication**: Standardize agent result formats and context passing
3. **Dependencies**: Optimize agent dependency chains
4. **Testing**: Ensure cleaned agents still integrate with pipeline

### Phase 4: Documentation and Structure (Days 7-8)
1. **Documentation Consolidation**: Merge and organize all documentation
2. **Directory Reorganization**: Optimize project structure
3. **Final Validation**: Comprehensive testing of cleaned codebase
4. **Quality Assurance**: Code quality metrics and standards compliance

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