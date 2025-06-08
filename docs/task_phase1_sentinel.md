# Task Phase 1: Agent 01 - Sentinel Implementation

## Agent Implementation Task: Agent 01 - Sentinel

**Phase**: 1 (Foundation)
**Priority**: P0 - Critical Foundation
**Dependencies**: None (entry point)
**Estimated Time**: 2-3 hours

### Character Profile
- **Name**: Sentinel
- **Role**: Binary discovery and metadata analysis
- **Personality**: Vigilant guardian, meticulous detector, first line of defense
- **Matrix Context**: The Sentinels are the guardians of the Matrix, detecting and analyzing threats. Agent 01 serves as the digital sentinel that scans and catalogs binary structures, identifying formats, architectures, and extracting critical metadata that all other agents depend upon.

### Technical Requirements
- **Base Class**: `AnalysisAgent` (from `matrix_agents_v2.py`)
- **Dependencies**: None (foundation agent)
- **Input Requirements**: 
  - Binary file path (`binary_path` in context)
  - Output directory structure (`output_paths` in context)
- **Output Requirements**: 
  - Binary format detection (PE/ELF/Mach-O)
  - Architecture identification (x86/x64/ARM)
  - File metadata (size, hashes, timestamps)
  - Section/segment analysis
  - Import/export tables
  - Basic security analysis
- **Quality Metrics**: 
  - Format detection accuracy: >95%
  - Architecture detection accuracy: >95%
  - Metadata extraction completeness: >90%

### Implementation Steps

1. **Initialize Sentinel Agent**
   - Setup binary analysis tools (pefile, pyelftools, macholib)
   - Configure logging with Sentinel branding
   - Initialize shared memory structure

2. **Binary Format Detection**
   - Detect PE/ELF/Mach-O/Unknown formats
   - Extract format-specific headers
   - Validate file integrity

3. **Architecture Analysis**
   - Identify target CPU architecture (x86/x64/ARM/etc.)
   - Determine bitness (32-bit/64-bit)
   - Detect endianness and calling conventions

4. **Metadata Extraction**
   - Calculate file hashes (MD5, SHA1, SHA256)
   - Extract compilation timestamps
   - Identify compiler signatures
   - Extract version information

5. **Structure Analysis**
   - Parse sections/segments
   - Extract import/export tables  
   - Identify resources and embedded data
   - Basic entropy analysis for packing detection

### Detailed Implementation Requirements

#### Binary Format Support
```python
# Must support these formats with full analysis:
SUPPORTED_FORMATS = {
    'PE': {'extensions': ['.exe', '.dll', '.sys'], 'priority': 'high'},
    'ELF': {'extensions': ['.elf', '.so'], 'priority': 'medium'}, 
    'Mach-O': {'extensions': ['.dylib', '.bundle'], 'priority': 'low'}
}
```

#### Required Output Structure
```python
sentinel_results = {
    'binary_metadata': {
        'format': 'PE',  # PE/ELF/Mach-O/Unknown
        'architecture': 'x86',  # x86/x64/ARM/ARM64/etc.
        'bitness': 32,  # 32/64
        'endianness': 'little',  # little/big
        'file_size': 5324800,
        'entry_point': 0x401000,
        'base_address': 0x400000,
        'hashes': {
            'md5': '...',
            'sha1': '...',
            'sha256': '...'
        }
    },
    'format_analysis': {
        'pe_header': {...},  # Format-specific data
        'sections': [...],
        'imports': [...],
        'exports': [...],
        'resources': [...]
    },
    'security_analysis': {
        'is_packed': False,
        'has_signature': False,
        'entropy_score': 6.2,
        'suspicious_sections': []
    },
    'sentinel_insights': {
        'threat_level': 'low',  # low/medium/high
        'complexity_score': 0.7,  # 0.0-1.0
        'analysis_confidence': 0.95,  # 0.0-1.0
        'recommendations': [...]
    }
}
```

### Error Handling Requirements

- **File Not Found**: Graceful error with clear message
- **Unsupported Format**: Partial analysis with warnings
- **Corrupted File**: Attempt recovery, log issues
- **Permission Denied**: Clear error message with suggestions
- **Memory Issues**: Chunked processing for large files

### Testing Requirements

#### Unit Tests
```python
def test_sentinel_pe_detection():
    """Test PE format detection and analysis"""
    
def test_sentinel_elf_detection():
    """Test ELF format detection and analysis"""
    
def test_sentinel_architecture_detection():
    """Test CPU architecture identification"""
    
def test_sentinel_metadata_extraction():
    """Test metadata extraction completeness"""
    
def test_sentinel_error_handling():
    """Test error handling for various failure modes"""
```

#### Integration Tests
```python
def test_sentinel_shared_memory_setup():
    """Test that Sentinel properly initializes shared memory"""
    
def test_sentinel_output_structure():
    """Test that output follows required structure"""
    
def test_sentinel_quality_metrics():
    """Test that quality metrics meet thresholds"""
```

### Files to Create/Modify

#### New Files
- `/src/core/agents_v2/agent01_sentinel.py` - Main agent implementation
- `/tests/test_agent01_sentinel.py` - Unit tests
- `/src/core/agents_v2/__init__.py` - Add Sentinel import

#### Files to Update  
- `/src/core/agents_v2/__init__.py` - Add Sentinel to agent registry
- `/docs/matrix_agent_implementation_tasks.md` - Mark Phase 1 complete

### Implementation Template

```python
"""
Agent 01: Sentinel - Binary Discovery & Metadata Analysis
The digital guardian that scans the Matrix for binary anomalies and catalogs their nature.
Serves as the foundation upon which all other Matrix agents build their analysis.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib
import struct
import os

from ..matrix_agents_v2 import AnalysisAgent, MatrixCharacter, AgentStatus
from ..shared_components import (
    MatrixLogger, MatrixFileManager, MatrixValidator, 
    MatrixProgressTracker, MatrixErrorHandler, MatrixMetrics
)

# Import binary analysis libraries
try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False

try:
    from elftools.elf.elffile import ELFFile
    HAS_ELFTOOLS = True
except ImportError:
    HAS_ELFTOOLS = False

try:
    from macholib.MachO import MachO
    HAS_MACHOLIB = True
except ImportError:
    HAS_MACHOLIB = False


class SentinelAgent(AnalysisAgent):
    """
    Agent 01: Sentinel - The Guardian of Binary Space
    
    The Sentinels patrol the digital realm, detecting and cataloging binary entities
    that enter the Matrix. Agent 01 serves as the primary scanner, identifying 
    file formats, architectures, and extracting critical metadata that forms
    the foundation for all subsequent analysis.
    """
    
    def __init__(self):
        super().__init__(
            agent_id=1,
            matrix_character=MatrixCharacter.SENTINEL,
            dependencies=[]  # No dependencies - foundation agent
        )
        
        # Initialize binary analysis capabilities
        self.supported_formats = self._check_format_support()
        self.metrics = MatrixMetrics(self.agent_id, self.matrix_character.value)
        
    def get_matrix_description(self) -> str:
        """The Sentinel's role in the Matrix"""
        return ("The Sentinel stands vigilant at the gates of the Matrix, scanning "
                "and cataloging every binary entity that seeks entry. It identifies "
                "their nature, structure, and potential threats with unwavering precision.")
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Sentinel's binary scanning mission"""
        
        # Implementation continues here...
        # [This is where the actual implementation would go]
        
        pass  # To be implemented
        
    # Additional methods for binary analysis...
```

### Success Criteria

- [ ] **Functional**: Agent successfully analyzes test binaries
- [ ] **Quality**: Meets >95% accuracy on format/architecture detection
- [ ] **Integration**: Properly integrates with Matrix agent framework
- [ ] **Testing**: Achieves >80% test coverage
- [ ] **Documentation**: Complete docstrings and character descriptions
- [ ] **Performance**: Processes typical binaries in <30 seconds

### Next Steps After Completion

1. Mark Task 1.1 complete in implementation tasks
2. Notify Phase 2 developers that foundation is ready
3. Provide integration documentation for Phase 2 agents
4. Conduct integration testing with existing framework

This Sentinel agent will serve as the critical foundation that enables all 15 subsequent Matrix agents to perform their specialized analysis tasks.