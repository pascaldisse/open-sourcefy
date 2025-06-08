# Task Phase 2.2: Agent 03 - The Merovingian Implementation

## Agent Implementation Task: Agent 03 - The Merovingian

**Phase**: 2 (Core Analysis)
**Priority**: P1 - High
**Dependencies**: Agent 01 (Sentinel)
**Estimated Time**: 2-3 hours

### Character Profile
- **Name**: The Merovingian
- **Role**: Basic decompilation and optimization detection
- **Personality**: Sophisticated, calculating, master of cause and effect
- **Matrix Context**: The Merovingian understands the intricate relationships between code transformations. Agent 03 specializes in basic decompilation, identifying how high-level constructs were transformed into machine code, and detecting the optimization techniques that obscure the original intent.

### Technical Requirements
- **Base Class**: `DecompilerAgent` (from `matrix_agents_v2.py`)
- **Dependencies**: Agent 01 (Sentinel) - requires binary metadata
- **Input Requirements**: 
  - Sentinel's binary format and architecture data
  - Binary file for disassembly and analysis
  - Import/export tables from Sentinel
- **Output Requirements**: 
  - Basic disassembly with function identification
  - Control flow graph construction
  - Loop and conditional structure detection
  - Optimization transformation analysis
  - Function signature inference
  - Basic type inference for variables
- **Quality Metrics**: 
  - Function detection accuracy: >90%
  - Control flow accuracy: >85%
  - Optimization detection: >75%

### Implementation Steps

1. **Initialize Decompilation Engine**
   - Setup disassembly engine (Capstone or similar)
   - Load architecture-specific decompilation rules
   - Initialize optimization pattern database

2. **Basic Disassembly**
   - Disassemble code sections
   - Identify function boundaries using multiple heuristics
   - Separate code from data segments
   - Build initial instruction database

3. **Control Flow Analysis**
   - Construct control flow graphs for each function
   - Identify basic blocks and their relationships
   - Detect loops, conditionals, and function calls
   - Analyze jump tables and indirect calls

4. **Optimization Pattern Detection**
   - Identify common compiler optimizations
   - Detect inlined functions and code duplication
   - Recognize loop optimizations and vectorization
   - Find constant propagation and dead code elimination

5. **Basic Type and Signature Inference**
   - Infer function signatures from calling patterns
   - Detect parameter passing conventions
   - Identify return value types
   - Basic variable type inference from usage patterns

### Detailed Implementation Requirements

#### Disassembly Engine Configuration
```python
DISASSEMBLY_CONFIG = {
    'x86': {
        'capstone_arch': 'CS_ARCH_X86',
        'capstone_mode': 'CS_MODE_32',
        'function_prologues': [b'\x55\x8b\xec', b'\x55\x89\xe5'],  # push ebp; mov ebp, esp
        'function_epilogues': [b'\xc9\xc3', b'\x5d\xc3']  # leave; ret / pop ebp; ret
    },
    'x64': {
        'capstone_arch': 'CS_ARCH_X86',
        'capstone_mode': 'CS_MODE_64',
        'function_prologues': [b'\x48\x89\xe5', b'\x48\x83\xec'],  # mov rbp, rsp; sub rsp, ...
        'function_epilogues': [b'\xc3', b'\x48\x89\xec\xc3']  # ret / mov rsp, rbp; ret
    }
}
```

#### Required Output Structure
```python
merovingian_results = {
    'disassembly_analysis': {
        'total_instructions': 15423,
        'code_sections': [
            {
                'name': '.text',
                'start_address': 0x401000,
                'end_address': 0x405000,
                'instruction_count': 12500
            }
        ],
        'disassembly_quality': 0.92,
        'unresolved_bytes': 245
    },
    'function_analysis': {
        'functions_detected': 127,
        'function_detection_confidence': 0.89,
        'functions': [
            {
                'address': 0x401000,
                'name': 'sub_401000',
                'size': 256,
                'basic_blocks': 8,
                'calls_made': 3,
                'calls_received': 5,
                'signature_confidence': 0.75,
                'inferred_signature': 'int __cdecl (char*, int)',
                'complexity_score': 0.6
            }
        ],
        'entry_points': [0x401000, 0x401500],
        'exported_functions': [...]
    },
    'control_flow_analysis': {
        'total_basic_blocks': 456,
        'control_structures': {
            'loops': [
                {
                    'type': 'for',  # for/while/do-while
                    'address': 0x401100,
                    'loop_body_size': 64,
                    'estimated_iterations': 'variable'
                }
            ],
            'conditionals': [
                {
                    'type': 'if-else',
                    'address': 0x401200,
                    'condition_complexity': 'simple',
                    'branch_probability': 0.7
                }
            ],
            'switches': [
                {
                    'address': 0x401300,
                    'case_count': 8,
                    'jump_table_address': 0x405000
                }
            ]
        },
        'call_graph': {
            'nodes': 127,
            'edges': 234,
            'recursive_functions': [0x401800],
            'orphaned_functions': [0x402000]
        }
    },
    'optimization_analysis': {
        'optimization_level': 'medium',  # none/low/medium/high
        'detected_optimizations': [
            {
                'type': 'inline_expansion',
                'locations': [0x401050, 0x401150],
                'confidence': 0.85,
                'original_function_estimate': 0x403000
            },
            {
                'type': 'constant_folding',
                'locations': [0x401200],
                'confidence': 0.92,
                'folded_expression': '5 * 8 -> 40'
            },
            {
                'type': 'dead_code_elimination',
                'evidence': 'gaps_in_sequence',
                'confidence': 0.78
            }
        ],
        'unoptimized_patterns': [
            {
                'type': 'redundant_loads',
                'count': 12,
                'severity': 'minor'
            }
        ]
    },
    'type_inference': {
        'inferred_types': {
            'global_variables': [
                {
                    'address': 0x406000,
                    'inferred_type': 'int',
                    'confidence': 0.85,
                    'usage_pattern': 'counter'
                }
            ],
            'function_parameters': {
                0x401000: [
                    {'position': 0, 'type': 'char*', 'confidence': 0.9},
                    {'position': 1, 'type': 'int', 'confidence': 0.8}
                ]
            },
            'return_types': {
                0x401000: {'type': 'int', 'confidence': 0.85}
            }
        },
        'string_analysis': {
            'string_constants': 45,
            'format_strings': 12,
            'unicode_strings': 8
        }
    },
    'merovingian_insights': {
        'decompilation_feasibility': 'high',  # low/medium/high
        'complexity_assessment': 'moderate',
        'reverse_engineering_challenges': [
            'optimized_loops',
            'inlined_functions',
            'obfuscated_strings'
        ],
        'recommended_next_steps': [
            'advanced_type_analysis',
            'symbol_recovery',
            'structure_reconstruction'
        ],
        'causality_analysis': {
            'code_relationships': 156,  # Number of identified cause-effect relationships
            'optimization_impact': 'moderate',
            'maintainability_score': 0.72
        }
    }
}
```

### Integration Requirements

#### With Sentinel (Agent 01)
```python
def _integrate_sentinel_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate Sentinel's binary analysis with decompilation process"""
    sentinel_data = context['shared_memory']['binary_metadata']['discovery']
    
    return {
        'architecture': sentinel_data['binary_info']['architecture'],
        'entry_point': sentinel_data['binary_info']['entry_point'],
        'sections': sentinel_data['format_analysis']['sections'],
        'imports': sentinel_data['format_analysis']['imports'],
        'exports': sentinel_data['format_analysis']['exports']
    }
```

### Advanced Features

#### Optimization Pattern Database
```python
OPTIMIZATION_PATTERNS = {
    'function_inlining': {
        'signatures': [
            # Pattern for inlined function calls
            {'pattern': 'direct_code_injection', 'confidence': 0.8},
            {'pattern': 'missing_call_instruction', 'confidence': 0.9}
        ]
    },
    'loop_unrolling': {
        'signatures': [
            {'pattern': 'repeated_instruction_sequences', 'confidence': 0.85},
            {'pattern': 'unnatural_jump_distances', 'confidence': 0.7}
        ]
    },
    'constant_propagation': {
        'signatures': [
            {'pattern': 'immediate_values_in_computations', 'confidence': 0.9},
            {'pattern': 'missing_load_instructions', 'confidence': 0.75}
        ]
    }
}
```

### Error Handling Requirements

- **Disassembly Failures**: Skip problematic sections, log issues
- **Unknown Instructions**: Mark as data, continue processing
- **Control Flow Ambiguity**: Use multiple heuristics, confidence scoring
- **Type Inference Conflicts**: Report multiple possibilities with confidence
- **Memory Access Violations**: Safe boundary checking

### Testing Requirements

#### Unit Tests
```python
def test_merovingian_function_detection():
    """Test function boundary detection accuracy"""
    
def test_merovingian_control_flow():
    """Test control flow graph construction"""
    
def test_merovingian_optimization_detection():
    """Test optimization pattern recognition"""
    
def test_merovingian_type_inference():
    """Test basic type inference capabilities"""
    
def test_merovingian_disassembly_quality():
    """Test disassembly accuracy and completeness"""
```

### Files to Create/Modify

#### New Files
- `/src/core/agents_v2/agent03_merovingian.py` - Main agent implementation
- `/tests/test_agent03_merovingian.py` - Unit tests

#### Files to Update  
- `/src/core/agents_v2/__init__.py` - Add Merovingian import
- `/docs/matrix_agent_implementation_tasks.md` - Mark Task 2.2 complete

### Success Criteria

- [ ] **Functional**: Successfully performs basic decompilation
- [ ] **Quality**: Achieves >90% function detection accuracy
- [ ] **Integration**: Properly utilizes Sentinel's binary data
- [ ] **Testing**: Achieves >80% test coverage
- [ ] **Performance**: Completes analysis in <120 seconds

### Dependencies for Phase 3

The Merovingian's analysis will be used by:
- Agent 05 (Neo) - for advanced decompilation enhancement
- Agent 07 (Trainman) - for detailed assembly analysis
- Agent 09 (Commander Locke) - for global reconstruction

This agent provides the foundational decompilation that enables deeper analysis in subsequent phases.