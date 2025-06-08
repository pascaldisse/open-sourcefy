# Task Phase 2.1: Agent 02 - The Architect Implementation

## Agent Implementation Task: Agent 02 - The Architect

**Phase**: 2 (Core Analysis)
**Priority**: P1 - High
**Dependencies**: Agent 01 (Sentinel)
**Estimated Time**: 2-3 hours

### Character Profile
- **Name**: The Architect
- **Role**: Architecture analysis and error pattern matching
- **Personality**: Precise, mathematical, systematic designer of binary structures
- **Matrix Context**: The Architect designed the Matrix itself, understanding its fundamental structures. Agent 02 analyzes the architectural patterns within binaries, identifying compilation methods, optimization levels, and structural anomalies that reveal the binary's construction blueprint.

### Technical Requirements
- **Base Class**: `AnalysisAgent` (from `matrix_agents_v2.py`)
- **Dependencies**: Agent 01 (Sentinel) - requires binary metadata
- **Input Requirements**: 
  - Sentinel's binary metadata and format analysis
  - Binary file for deep architectural analysis
  - Shared memory with binary_metadata populated
- **Output Requirements**: 
  - Compiler identification and version detection
  - Optimization level analysis (-O0, -O1, -O2, -O3)
  - Calling convention detection
  - ABI (Application Binary Interface) analysis
  - Build system identification (CMake, MSBuild, etc.)
  - Error pattern recognition and classification
- **Quality Metrics**: 
  - Compiler detection accuracy: >85%
  - Optimization level detection: >80%
  - Pattern matching precision: >75%

### Implementation Steps

1. **Initialize Architect Analysis**
   - Access Sentinel's binary metadata from shared memory
   - Setup compiler signature databases
   - Initialize pattern matching engines

2. **Compiler Analysis**
   - Identify compiler toolchain (GCC, Clang, MSVC, ICC)
   - Determine compiler version from signatures
   - Detect debug information presence (PDB, DWARF)

3. **Optimization Analysis**
   - Analyze instruction patterns for optimization levels
   - Detect inline function patterns
   - Identify loop unrolling and vectorization
   - Recognize dead code elimination patterns

4. **ABI and Calling Convention Analysis**
   - Determine calling conventions (cdecl, stdcall, fastcall)
   - Analyze stack frame structures
   - Identify exception handling mechanisms
   - Detect TLS (Thread Local Storage) usage

5. **Error Pattern Recognition**
   - Scan for common compilation errors embedded in binary
   - Identify debugging artifacts and development patterns
   - Detect test code remnants and assertion patterns
   - Analyze error handling structures

### Detailed Implementation Requirements

#### Compiler Signature Database
```python
COMPILER_SIGNATURES = {
    'MSVC': {
        'versions': {
            '6.0': {'patterns': [b'Microsoft (R) 32-bit C/C++'], 'year': 1998},
            '7.0': {'patterns': [b'Microsoft (R) C/C++ Compiler'], 'year': 2002},
            '7.1': {'patterns': [b'Microsoft (R) C/C++ Optimizing Compiler Version 13'], 'year': 2003},
            # ... more versions
        }
    },
    'GCC': {
        'versions': {
            '4.9': {'patterns': [b'GCC: (GNU) 4.9'], 'optimizations': ['O2', 'O3']},
            # ... more versions
        }
    }
}
```

#### Required Output Structure
```python
architect_results = {
    'compiler_analysis': {
        'toolchain': 'MSVC',  # MSVC/GCC/Clang/ICC/Unknown
        'version': '7.1',
        'confidence': 0.92,
        'debug_info': {
            'present': True,
            'format': 'PDB',  # PDB/DWARF/STABS/None
            'path': 'C:\\path\\to\\debug.pdb'
        }
    },
    'optimization_analysis': {
        'level': 'O2',  # O0/O1/O2/O3/Os/Oz/Unknown
        'confidence': 0.85,
        'detected_patterns': [
            'inline_functions',
            'loop_unrolling', 
            'dead_code_elimination',
            'constant_folding'
        ],
        'optimization_artifacts': [...]
    },
    'abi_analysis': {
        'calling_convention': 'cdecl',  # cdecl/stdcall/fastcall/vectorcall
        'stack_alignment': 16,
        'exception_handling': 'SEH',  # SEH/C++EH/None
        'tls_usage': False,
        'rtti_enabled': True
    },
    'build_system_analysis': {
        'build_tool': 'MSBuild',  # MSBuild/CMake/Make/Ninja/Unknown
        'configuration': 'Release',  # Debug/Release/MinSizeRel/RelWithDebInfo
        'target_platform': 'Win32',
        'subsystem': 'Console'  # Console/Windows/DLL
    },
    'pattern_analysis': {
        'error_patterns': [
            {'type': 'assert', 'count': 15, 'confidence': 0.8},
            {'type': 'exception', 'count': 8, 'confidence': 0.9}
        ],
        'development_artifacts': [
            'debug_strings',
            'test_functions',
            'profiling_hooks'
        ],
        'code_quality_indicators': {
            'error_handling_coverage': 0.75,
            'assertion_density': 0.12,
            'exception_safety': 'basic'
        }
    },
    'architectural_insights': {
        'complexity_assessment': 'medium',  # low/medium/high
        'maintainability_score': 0.68,
        'architectural_style': 'procedural',  # procedural/oop/functional/mixed
        'design_patterns': ['singleton', 'factory']
    }
}
```

### Integration Requirements

#### With Sentinel (Agent 01)
```python
def _get_sentinel_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Extract required data from Sentinel's analysis"""
    shared_memory = context['shared_memory']
    sentinel_data = shared_memory['binary_metadata'].get('discovery', {})
    
    required_fields = ['binary_info', 'format_analysis', 'security_analysis']
    for field in required_fields:
        if field not in sentinel_data:
            raise ValueError(f"Sentinel data missing required field: {field}")
    
    return sentinel_data
```

### Error Handling Requirements

- **Missing Sentinel Data**: Clear error with guidance
- **Unsupported Architecture**: Partial analysis with warnings  
- **Corrupted Sections**: Skip problematic sections, continue analysis
- **Unknown Compiler**: Generic analysis with confidence scores
- **Pattern Matching Failures**: Fallback to heuristic analysis

### Testing Requirements

#### Unit Tests
```python
def test_architect_msvc_detection():
    """Test MSVC compiler detection and version identification"""
    
def test_architect_optimization_analysis():
    """Test optimization level detection"""
    
def test_architect_calling_convention():
    """Test calling convention detection"""
    
def test_architect_pattern_matching():
    """Test error pattern recognition"""
    
def test_architect_integration_with_sentinel():
    """Test integration with Sentinel data"""
```

### Files to Create/Modify

#### New Files
- `/src/core/agents_v2/agent02_architect.py` - Main agent implementation
- `/tests/test_agent02_architect.py` - Unit tests

#### Files to Update  
- `/src/core/agents_v2/__init__.py` - Add Architect import
- `/docs/matrix_agent_implementation_tasks.md` - Mark Task 2.1 complete

### Success Criteria

- [ ] **Functional**: Successfully analyzes compiler and optimization patterns
- [ ] **Quality**: Meets >85% accuracy on compiler detection
- [ ] **Integration**: Properly consumes Sentinel data
- [ ] **Testing**: Achieves >80% test coverage
- [ ] **Performance**: Completes analysis in <60 seconds

### Dependencies for Phase 3

The Architect's analysis will be used by:
- Agent 05 (Neo) - for advanced decompilation guidance
- Agent 06 (Twins) - for comparison baseline
- Agent 07 (Trainman) - for assembly analysis context

This agent provides critical architectural understanding that guides all subsequent analysis phases.