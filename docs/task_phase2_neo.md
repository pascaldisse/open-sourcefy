# Task Phase 2.4: Agent 05 - Neo (Glitch) Implementation

## Agent Implementation Task: Agent 05 - Neo (Glitch)

**Phase**: 2 (Core Analysis)
**Priority**: P1 - High
**Dependencies**: Agent 01 (Sentinel), Agent 02 (Architect)
**Estimated Time**: 3-4 hours

### Character Profile
- **Name**: Neo (The Glitch)
- **Role**: Advanced decompilation and Ghidra integration
- **Personality**: Intuitive, sees beyond the code, breaks conventional patterns
- **Matrix Context**: Neo sees the Matrix code as it truly is. Agent 05 leverages advanced decompilation techniques and Ghidra integration to reconstruct high-level code structures, transcending the limitations of basic disassembly to reveal the original developer's intent.

### Technical Requirements
- **Base Class**: `DecompilerAgent` (from `matrix_agents_v2.py`)
- **Dependencies**: Agent 01 (Sentinel), Agent 02 (Architect) - requires architecture and binary metadata
- **Input Requirements**: 
  - Sentinel's binary format and structure data
  - Architect's compiler and optimization analysis
  - Binary file for Ghidra processing
  - Shared memory with complete binary metadata
- **Output Requirements**: 
  - High-quality decompiled C/C++ code
  - Function reconstruction with proper signatures
  - Variable and type recovery
  - Control structure reconstruction
  - Comment generation and code annotation
  - Integration with Ghidra's analysis results
- **Quality Metrics**: 
  - Decompilation accuracy: >85%
  - Function signature recovery: >80%
  - Type inference accuracy: >75%
  - Compilability score: >70%

### Implementation Steps

1. **Initialize Ghidra Integration**
   - Setup headless Ghidra environment
   - Configure Ghidra scripts for enhanced analysis
   - Initialize decompilation pipeline
   - Setup architecture-specific processors

2. **Advanced Binary Analysis**
   - Import binary into Ghidra project
   - Apply Architect's compiler analysis for optimization
   - Run comprehensive analysis suite
   - Extract function information and control flow

3. **High-Level Code Reconstruction**
   - Decompile functions using Ghidra's decompiler
   - Enhance decompilation with architecture insights
   - Reconstruct data types and structures
   - Recover variable names and purposes

4. **Code Quality Enhancement**
   - Apply post-processing to improve readability
   - Generate meaningful comments and documentation
   - Reconstruct higher-level constructs (classes, objects)
   - Optimize code organization and structure

5. **Integration and Validation**
   - Validate decompiled code quality
   - Cross-reference with original binary structures
   - Prepare results for compilation testing
   - Generate comprehensive analysis reports

### Detailed Implementation Requirements

#### Ghidra Integration Configuration
```python
GHIDRA_CONFIG = {
    'ghidra_home': '/path/to/ghidra',
    'project_directory': 'output/ghidra_projects',
    'analysis_timeout': 600,  # 10 minutes
    'decompilation_timeout': 300,  # 5 minutes per function
    'custom_scripts': [
        'enhanced_function_analysis.java',
        'type_recovery_script.java',
        'string_analysis_script.java'
    ],
    'analyzers': [
        'Decompiler Parameter ID',
        'Function Start Search',
        'Aggressive Instruction Finder',
        'Demangler Microsoft',
        'Stack'
    ]
}
```

#### Required Output Structure
```python
neo_results = {
    'ghidra_analysis': {
        'project_status': 'success',
        'analysis_time': 347.5,
        'functions_analyzed': 127,
        'functions_decompiled': 119,
        'decompilation_success_rate': 0.937,
        'analysis_quality': 'high',
        'ghidra_version': '10.3.1',
        'processor_module': 'x86:LE:32:default'
    },
    'decompiled_functions': [
        {
            'address': 0x401000,
            'name': 'main',
            'signature': 'int main(int argc, char** argv)',
            'decompiled_code': '''int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <filename>\\n", argv[0]);
        return 1;
    }
    
    FILE* file = fopen(argv[1], "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\\n", argv[1]);
        return 2;
    }
    
    process_file(file);
    fclose(file);
    return 0;
}''',
            'quality_score': 0.92,
            'complexity_level': 'low',
            'local_variables': [
                {'name': 'file', 'type': 'FILE*', 'confidence': 0.95}
            ],
            'function_calls': ['printf', 'fopen', 'fprintf', 'process_file', 'fclose'],
            'control_structures': ['if-else', 'sequential']
        }
    ],
    'type_recovery': {
        'recovered_types': [
            {
                'name': 'ProcessorConfig',
                'type': 'struct',
                'definition': '''typedef struct {
    char name[64];
    int processor_id;
    bool enabled;
    void (*process_func)(void*);
} ProcessorConfig;''',
                'confidence': 0.87,
                'member_count': 4,
                'size': 76
            }
        ],
        'function_signatures': [
            {
                'address': 0x401200,
                'original': 'sub_401200',
                'recovered': 'int process_file(FILE* input)',
                'confidence': 0.83,
                'parameter_count': 1,
                'return_type': 'int'
            }
        ],
        'global_variables': [
            {
                'address': 0x431000,
                'name': 'g_config',
                'type': 'ProcessorConfig',
                'confidence': 0.91
            }
        ]
    },
    'code_reconstruction': {
        'source_files': [
            {
                'filename': 'main.c',
                'path': 'output/decompiled/main.c',
                'function_count': 8,
                'line_count': 247,
                'quality_score': 0.89,
                'compilable': True,
                'includes': ['#include <stdio.h>', '#include <stdlib.h>', '#include <string.h>']
            },
            {
                'filename': 'processor.c',
                'path': 'output/decompiled/processor.c',
                'function_count': 12,
                'line_count': 456,
                'quality_score': 0.76,
                'compilable': True,
                'includes': ['#include "processor.h"', '#include <memory.h>']
            }
        ],
        'header_files': [
            {
                'filename': 'processor.h',
                'path': 'output/decompiled/processor.h',
                'type_definitions': 5,
                'function_declarations': 12,
                'constants': 8
            }
        ],
        'total_lines_of_code': 1247,
        'estimated_original_lines': 1450,
        'recovery_percentage': 0.86
    },
    'analysis_enhancement': {
        'string_analysis': {
            'format_strings_identified': 15,
            'constant_strings': 67,
            'unicode_strings': 8,
            'obfuscated_strings': 2
        },
        'algorithm_detection': [
            {
                'name': 'quicksort',
                'location': 0x402000,
                'confidence': 0.92,
                'variant': 'optimized'
            },
            {
                'name': 'crc32',
                'location': 0x403000,
                'confidence': 0.87,
                'table_driven': True
            }
        ],
        'design_patterns': [
            {
                'pattern': 'factory',
                'location': 0x404000,
                'confidence': 0.78,
                'implementation': 'function_pointer_table'
            }
        ],
        'api_usage_analysis': {
            'win32_apis': 45,
            'crt_functions': 23,
            'custom_apis': 8,
            'security_sensitive': ['CreateFile', 'RegOpenKey', 'LoadLibrary']
        }
    },
    'neo_insights': {
        'code_quality_assessment': {
            'overall_quality': 'high',  # low/medium/high/excellent
            'maintainability': 0.82,
            'readability': 0.88,
            'documentation_level': 0.65
        },
        'reverse_engineering_success': {
            'function_recovery': 0.937,
            'type_recovery': 0.78,
            'logic_reconstruction': 0.84,
            'overall_success': 0.85
        },
        'glitch_discoveries': [
            {
                'type': 'hidden_functionality',
                'description': 'Debug mode accessible via environment variable',
                'location': 0x405000,
                'significance': 'medium'
            },
            {
                'type': 'unused_code',
                'description': 'Legacy authentication system still present',
                'location': 0x406000,
                'significance': 'high'
            }
        ],
        'matrix_anomalies': [
            'non_standard_calling_convention',
            'custom_encryption_routine',
            'embedded_configuration_data'
        ],
        'transcendence_level': 'high',  # How well Neo has "seen through" the binary
        'awakening_insights': [
            'Original codebase appears to be C++ compiled as C',
            'Multiple development phases evident in code style',
            'Significant debugging infrastructure removed in release'
        ]
    }
}
```

### Integration Requirements

#### With Sentinel (Agent 01) and Architect (Agent 02)
```python
def _integrate_previous_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate Sentinel and Architect data for enhanced decompilation"""
    shared_memory = context['shared_memory']
    
    # Get Sentinel's binary metadata
    sentinel_data = shared_memory['binary_metadata']['discovery']
    
    # Get Architect's compiler analysis  
    architect_data = shared_memory['analysis_results'].get(2, {})
    
    return {
        'architecture': sentinel_data['binary_info']['architecture'],
        'format': sentinel_data['binary_info']['format'],
        'compiler_info': architect_data.get('compiler_analysis', {}),
        'optimization_level': architect_data.get('optimization_analysis', {}).get('level', 'unknown'),
        'calling_convention': architect_data.get('abi_analysis', {}).get('calling_convention', 'unknown')
    }
```

### Advanced Features

#### Ghidra Script Integration
```java
// enhanced_function_analysis.java
public class EnhancedFunctionAnalysis extends GhidraScript {
    @Override
    public void run() throws Exception {
        // Custom analysis to enhance function detection
        // Integrate with compiler-specific patterns from Architect
        // Apply optimization-aware analysis
    }
}
```

#### Code Quality Enhancement
```python
CODE_ENHANCEMENT_RULES = {
    'variable_naming': {
        'patterns': [
            {'pattern': 'iVar', 'replacement': 'index', 'confidence': 0.8},
            {'pattern': 'lparam', 'replacement': 'parameter', 'confidence': 0.9}
        ]
    },
    'function_naming': {
        'api_wrappers': True,
        'algorithm_detection': True,
        'purpose_inference': True
    },
    'comment_generation': {
        'function_purpose': True,
        'parameter_descriptions': True,
        'return_value_meaning': True,
        'algorithm_explanations': True
    }
}
```

### Error Handling Requirements

- **Ghidra Process Failures**: Retry with different configurations
- **Decompilation Timeout**: Fallback to partial decompilation
- **Memory Issues**: Process functions in batches
- **Script Failures**: Continue with core decompilation
- **Invalid Output**: Validate and clean generated code

### Testing Requirements

#### Unit Tests
```python
def test_neo_ghidra_integration():
    """Test Ghidra integration and project setup"""
    
def test_neo_decompilation_quality():
    """Test decompilation accuracy and completeness"""
    
def test_neo_type_recovery():
    """Test type and signature recovery"""
    
def test_neo_code_generation():
    """Test generated code quality and compilability"""
    
def test_neo_integration_with_previous_agents():
    """Test integration with Sentinel and Architect"""
```

#### Integration Tests
```python
def test_neo_full_pipeline():
    """Test complete decompilation pipeline"""
    
def test_neo_code_compilation():
    """Test that generated code compiles successfully"""
```

### Files to Create/Modify

#### New Files
- `/src/core/agents_v2/agent05_neo.py` - Main agent implementation
- `/tests/test_agent05_neo.py` - Unit tests
- `/ghidra/scripts/enhanced_function_analysis.java` - Custom Ghidra script
- `/ghidra/scripts/type_recovery_script.java` - Type recovery script

#### Files to Update  
- `/src/core/agents_v2/__init__.py` - Add Neo import
- `/docs/matrix_agent_implementation_tasks.md` - Mark Task 2.4 complete

### Success Criteria

- [ ] **Functional**: Successfully integrates with Ghidra and produces decompiled code
- [ ] **Quality**: Achieves >85% decompilation accuracy
- [ ] **Integration**: Effectively uses Sentinel and Architect data
- [ ] **Testing**: Achieves >80% test coverage
- [ ] **Performance**: Completes decompilation in <10 minutes
- [ ] **Output**: Generates compilable C code with >70% success rate

### Dependencies for Phase 3

Neo's decompilation will be used by:
- Agent 07 (Trainman) - for assembly analysis validation
- Agent 09 (Commander Locke) - for global code reconstruction
- Agent 10 (Machine) - for compilation testing
- Agent 11 (Oracle) - for final validation

This agent represents the breakthrough moment where binary code transcends back to human-readable form, enabling all subsequent reconstruction and validation phases.