# Developer Guide

Comprehensive guide for developers contributing to Open-Sourcefy or extending the Matrix pipeline.

## Development Environment Setup

### Prerequisites

#### Required Software
- **Python 3.9+**: Core runtime environment
- **Git**: Version control and repository management
- **Visual Studio 2022 Preview**: Windows compilation and debugging
- **Java JDK 11+**: Required for Ghidra integration
- **Node.js 16+**: For documentation and build tools (optional)

#### Development Tools
- **IDE**: Visual Studio Code, PyCharm, or similar
- **Debugger**: Python debugger integration
- **Testing**: pytest for unit testing
- **Linting**: pylint, flake8, black for code quality
- **Documentation**: Sphinx for API documentation generation

### Repository Setup

```bash
# Clone repository
git clone https://github.com/pascaldisse/open-sourcefy.git
cd open-sourcefy

# Create development environment
python -m venv venv_dev
source venv_dev/bin/activate  # Linux/macOS
# or
venv_dev\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify environment
python main.py --verify-env
```

### Development Configuration

#### Development Config (`config-dev.yaml`)
```yaml
application:
  debug_mode: true
  log_level: "DEBUG"
  
agents:
  timeout: 600  # Longer timeouts for debugging
  fail_fast: false  # Continue on errors for analysis
  
pipeline:
  execution_mode: "development"
  cache_results: false  # Always fresh results
  validation_level: "comprehensive"
  
logging:
  level: "DEBUG"
  destinations:
    console: true
    file: true
  agents:
    enable_per_agent_logs: true
    debug_level_agents: [1, 5, 9, 15, 16]
```

#### Environment Variables for Development
```bash
# Development settings
export MATRIX_DEBUG=true
export MATRIX_VERBOSE=true
export MATRIX_PROFILE=true
export MATRIX_CONFIG_PATH="config-dev.yaml"

# Test data paths
export MATRIX_TEST_DATA="/path/to/test/binaries"
export MATRIX_TEST_OUTPUT="/path/to/test/output"

# AI integration (optional)
export ANTHROPIC_API_KEY="your_dev_api_key"
```

## Architecture Deep Dive

### Core Framework Components

#### Matrix Pipeline Orchestrator
**File**: `src/core/matrix_pipeline_orchestrator.py`

The orchestrator is the heart of the system, managing agent execution and coordination.

```python
class MatrixPipelineOrchestrator:
    """
    Master coordinator for the Matrix agent pipeline
    
    Responsibilities:
    - Agent dependency resolution
    - Batch execution coordination
    - Error handling and recovery
    - Quality gate enforcement
    """
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config = config_manager or ConfigManager()
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler()
        self.agent_registry = self._initialize_agent_registry()
        
    def execute_pipeline(self, binary_path: str, selected_agents: List[int] = None) -> PipelineResult:
        """Main pipeline execution method"""
        # Implementation details...
        pass
```

Key methods to understand:
- `_resolve_agent_dependencies()`: Dependency graph resolution
- `_create_execution_batches()`: Parallel execution planning
- `_execute_agent_batch()`: Batch execution coordination
- `_enforce_quality_gates()`: Validation checkpoint enforcement

#### Agent Base Framework
**File**: `src/core/matrix_agents.py`

All agents inherit from the base `MatrixAgent` class:

```python
class MatrixAgent:
    """Base class for all Matrix agents"""
    
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter):
        self.agent_id = agent_id
        self.matrix_character = matrix_character
        self.logger = self._setup_logger()
        self.config = ConfigManager()
        
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method - must be overridden"""
        raise NotImplementedError("Agents must implement execute_matrix_task")
        
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate agent prerequisites - must be overridden"""
        raise NotImplementedError("Agents must implement _validate_prerequisites")
```

### Agent Development Patterns

#### Standard Agent Structure
```python
class Agent##_MatrixCharacter(MatrixAgent):
    """Agent ## - Matrix Character Name (Description)"""
    
    def __init__(self):
        super().__init__(
            agent_id=##,
            matrix_character=MatrixCharacter.CHARACTER_NAME
        )
        self.dependencies = [list_of_dependent_agent_ids]
        self.capabilities = self._initialize_capabilities()
        
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution logic"""
        start_time = time.time()
        
        try:
            # 1. Validate prerequisites
            self._validate_prerequisites(context)
            
            # 2. Extract required data from context
            required_data = self._extract_context_data(context)
            
            # 3. Perform agent-specific analysis
            analysis_result = self._perform_analysis(required_data)
            
            # 4. Process and validate results
            processed_result = self._process_results(analysis_result)
            
            # 5. Calculate quality metrics
            quality_score = self._calculate_quality_score(processed_result)
            
            execution_time = time.time() - start_time
            
            # 6. Return standardized result
            return {
                'agent_id': self.agent_id,
                'status': 'SUCCESS',
                'data': processed_result,
                'execution_time': execution_time,
                'quality_score': quality_score,
                'capabilities_used': self.capabilities
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Agent {self.agent_id} execution failed: {str(e)}")
            
            return {
                'agent_id': self.agent_id,
                'status': 'FAILED',
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate required context and dependencies"""
        # Check required context keys
        required_keys = self._get_required_context_keys()
        missing_keys = [k for k in required_keys if k not in context]
        if missing_keys:
            raise ValueError(f"Missing required context keys: {missing_keys}")
        
        # Validate agent dependencies
        agent_results = context.get('agent_results', {})
        for dep_id in self.dependencies:
            if dep_id not in agent_results:
                raise ValueError(f"Required dependency Agent {dep_id} not satisfied")
            if agent_results[dep_id].status != AgentStatus.SUCCESS:
                raise ValueError(f"Dependency Agent {dep_id} failed")
    
    def _get_required_context_keys(self) -> List[str]:
        """Define required context keys"""
        return ['binary_path', 'shared_memory', 'output_paths']
    
    def get_matrix_description(self) -> str:
        """Agent description for documentation"""
        return f"Agent {self.agent_id}: Matrix Character - Description of capabilities"
```

### Creating New Agents

#### Step 1: Define Agent Class
```python
# File: src/core/agents/agent17_new_character.py

from ..matrix_agents import MatrixAgent
from ..matrix_agents import MatrixCharacter, AgentStatus
from typing import Dict, Any, List
import time

class Agent17_NewCharacter(MatrixAgent):
    """Agent 17: New Character - Custom functionality description"""
    
    def __init__(self):
        super().__init__(
            agent_id=17,
            matrix_character=MatrixCharacter.NEW_CHARACTER  # Add to enum
        )
        self.dependencies = [1, 5]  # Define dependencies
        self.capabilities = {
            'custom_analysis': True,
            'specialized_processing': True,
            'advanced_features': True
        }
```

#### Step 2: Update Matrix Character Enum
```python
# File: src/core/matrix_agents.py

class MatrixCharacter(Enum):
    # Existing characters...
    NEW_CHARACTER = "new_character"
```

#### Step 3: Register Agent
```python
# File: src/core/matrix_pipeline_orchestrator.py

def _initialize_agent_registry(self) -> Dict[int, MatrixAgent]:
    """Initialize all available agents"""
    agents = {}
    
    # Existing agents...
    
    # Add new agent
    try:
        from .agents.agent17_new_character import Agent17_NewCharacter
        agents[17] = Agent17_NewCharacter()
    except ImportError:
        self.logger.warning("Agent 17 not available")
    
    return agents
```

#### Step 4: Add Tests
```python
# File: tests/test_agent17_new_character.py

import unittest
from src.core.agents.agent17_new_character import Agent17_NewCharacter
from src.core.matrix_agents import AgentStatus, AgentResult

class TestAgent17NewCharacter(unittest.TestCase):
    def setUp(self):
        self.agent = Agent17_NewCharacter()
        self.test_context = {
            'binary_path': 'test/binary.exe',
            'shared_memory': {},
            'output_paths': {},
            'agent_results': {
                1: AgentResult(
                    agent_id=1,
                    status=AgentStatus.SUCCESS,
                    data={'binary_info': {}},
                    agent_name="Sentinel",
                    matrix_character="sentinel"
                ),
                5: AgentResult(
                    agent_id=5,
                    status=AgentStatus.SUCCESS,
                    data={'decompilation': {}},
                    agent_name="Neo",
                    matrix_character="neo"
                )
            }
        }
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        self.assertEqual(self.agent.agent_id, 17)
        self.assertEqual(self.agent.dependencies, [1, 5])
    
    def test_agent_execution(self):
        """Test agent executes successfully"""
        result = self.agent.execute_matrix_task(self.test_context)
        self.assertEqual(result['agent_id'], 17)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertIn('data', result)
```

## Testing Framework

### Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_agents/        # Individual agent tests
│   ├── test_core/          # Core framework tests
│   └── test_utils/         # Utility function tests
├── integration/            # Integration tests
│   ├── test_pipeline/      # Pipeline integration tests
│   └── test_agent_chains/  # Agent dependency tests
├── e2e/                   # End-to-end tests
│   └── test_full_pipeline/ # Complete pipeline tests
└── fixtures/              # Test data and fixtures
    ├── binaries/          # Test binary files
    └── expected_outputs/  # Expected test results
```

### Test Base Classes

```python
# tests/base.py

import unittest
import tempfile
import shutil
from pathlib import Path

class MatrixTestCase(unittest.TestCase):
    """Base test case for Matrix pipeline tests"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_binary = self.temp_dir / "test_binary.exe"
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def create_test_context(self, agent_results: Dict[int, AgentResult] = None) -> Dict[str, Any]:
        """Create standard test context"""
        return {
            'binary_path': str(self.test_binary),
            'output_paths': {
                'base': self.output_dir,
                'agents': self.output_dir / 'agents',
                'reports': self.output_dir / 'reports'
            },
            'shared_memory': {
                'analysis_results': {},
                'binary_metadata': {}
            },
            'agent_results': agent_results or {}
        }
    
    def assert_agent_success(self, result: Dict[str, Any]):
        """Assert agent execution was successful"""
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertIn('data', result)
        self.assertIn('execution_time', result)
        self.assertGreater(result['execution_time'], 0)
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/e2e/ -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific agent tests
python -m pytest tests/unit/test_agents/test_agent01_sentinel.py -v

# Run tests with debugging
python -m pytest tests/ -v -s --tb=long
```

### Test Data Management

```python
# tests/fixtures/binary_samples.py

class BinarySamples:
    """Test binary samples for consistent testing"""
    
    LAUNCHER_EXE = {
        'path': 'tests/fixtures/binaries/launcher.exe',
        'size': 5369856,
        'format': 'PE32',
        'architecture': 'x86',
        'expected_functions': 156
    }
    
    SIMPLE_EXE = {
        'path': 'tests/fixtures/binaries/simple.exe',
        'size': 12288,
        'format': 'PE32',
        'architecture': 'x86',
        'expected_functions': 5
    }
    
    @staticmethod
    def get_sample(name: str) -> Dict[str, Any]:
        """Get test sample by name"""
        return getattr(BinarySamples, name.upper())
```

## Code Quality Standards

### Code Style Guidelines

#### Python Style (PEP 8 + Project Standards)
```python
# Good: Clear, descriptive names
class BinaryAnalysisEngine:
    def extract_import_table(self, binary_path: str) -> ImportAnalysis:
        """Extract import table from PE binary"""
        pass

# Good: Proper type hints
def process_agent_results(
    results: Dict[int, AgentResult], 
    quality_threshold: float = 0.75
) -> ValidationResult:
    """Process agent results with quality validation"""
    pass

# Good: Comprehensive docstrings
class Agent01_Sentinel(MatrixAgent):
    """
    Agent 1: Sentinel - Binary Discovery and Security Scanning
    
    The Sentinel serves as the foundation agent, performing initial binary
    analysis, format detection, import table extraction, and security assessment.
    
    Capabilities:
        - Binary format detection (PE, ELF, Mach-O)
        - Import table analysis (538+ functions from 14+ DLLs)
        - Security scanning and threat assessment
        - Metadata extraction and digital signature validation
        
    Dependencies: None (Foundation agent)
    """
    pass
```

#### Documentation Standards
```python
def complex_analysis_function(
    binary_data: bytes,
    analysis_depth: str = "comprehensive",
    timeout: int = 300
) -> AnalysisResult:
    """
    Perform comprehensive binary analysis with configurable depth.
    
    This function analyzes binary data using multiple techniques including
    static analysis, pattern recognition, and metadata extraction.
    
    Args:
        binary_data: Raw binary data to analyze
        analysis_depth: Analysis depth level ("basic", "standard", "comprehensive")
        timeout: Maximum analysis time in seconds
        
    Returns:
        AnalysisResult containing:
            - format_info: Binary format details
            - function_list: Identified functions
            - security_assessment: Security analysis results
            - quality_score: Analysis confidence (0.0-1.0)
            
    Raises:
        ValueError: If binary_data is empty or invalid
        TimeoutError: If analysis exceeds timeout
        AnalysisError: If analysis fails due to unsupported format
        
    Example:
        >>> with open("binary.exe", "rb") as f:
        ...     data = f.read()
        >>> result = complex_analysis_function(data, "comprehensive")
        >>> print(f"Quality: {result.quality_score:.2f}")
        Quality: 0.85
    """
    pass
```

### Linting and Formatting

#### Configuration Files

**`.pylintrc`**:
```ini
[MASTER]
load-plugins=pylint.extensions.docparams

[MESSAGES CONTROL]
disable=missing-module-docstring,
        too-few-public-methods,
        too-many-arguments,
        too-many-locals

[FORMAT]
max-line-length=100
good-names=i,j,k,ex,Run,_,id

[DESIGN]
max-args=7
max-locals=15
max-returns=6
max-branches=12
```

**`pyproject.toml`** (for Black formatting):
```toml
[tool.black]
line-length = 100
target-version = ['py39']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.venv
  | build
  | dist
)/
'''
```

#### Pre-commit Hooks

**`.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=100]

  - repo: https://github.com/pycqa/pylint
    rev: v2.13.7
    hooks:
      - id: pylint
        args: [--rcfile=.pylintrc]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

## Debugging and Profiling

### Debug Configuration

```python
# Enable comprehensive debugging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Agent-specific debugging
from src.core.agents.agent01_sentinel import SentinelAgent

agent = SentinelAgent()
agent.logger.setLevel(logging.DEBUG)

# Context debugging
context = create_test_context()
print(f"Context keys: {list(context.keys())}")
print(f"Agent results: {len(context.get('agent_results', {}))}")
```

### Performance Profiling

```python
# Profile agent execution
import cProfile
import pstats

def profile_agent_execution():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute agent
    agent = SentinelAgent()
    result = agent.execute_matrix_task(context)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

### Debug Utilities

```python
# Debug context helper
def debug_context(context: Dict[str, Any], agent_id: int = None):
    """Print context information for debugging"""
    print(f"=== Context Debug {f'(Agent {agent_id})' if agent_id else ''} ===")
    print(f"Binary path: {context.get('binary_path', 'NOT SET')}")
    print(f"Output paths: {list(context.get('output_paths', {}).keys())}")
    print(f"Agent results: {list(context.get('agent_results', {}).keys())}")
    print(f"Shared memory keys: {list(context.get('shared_memory', {}).keys())}")
    print("=" * 50)

# Agent result validator
def validate_agent_result(result: Dict[str, Any], expected_keys: List[str] = None):
    """Validate agent result structure"""
    required_keys = ['agent_id', 'status', 'data', 'execution_time']
    if expected_keys:
        required_keys.extend(expected_keys)
    
    missing_keys = [k for k in required_keys if k not in result]
    if missing_keys:
        raise ValueError(f"Agent result missing keys: {missing_keys}")
    
    if result['status'] not in ['SUCCESS', 'FAILED', 'WARNING']:
        raise ValueError(f"Invalid status: {result['status']}")
```

## Contribution Guidelines

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/agent-enhancement

# Make changes and commit
git add .
git commit -m "feat: enhance agent 5 decompilation accuracy

- Improve Ghidra integration with better type inference
- Add function signature validation
- Increase decompilation quality threshold to 0.85

Closes #123"

# Push and create pull request
git push origin feature/agent-enhancement
```

### Commit Message Format

```
<type>(<scope>): <description>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples**:
```
feat(agent01): add MFC 7.1 import table support

Enhance Sentinel agent to properly detect and resolve MFC 7.1 
import tables with ordinal-to-function name mapping.

- Add MFC version detection
- Implement ordinal resolution
- Update import analysis to handle 538+ functions
- Add comprehensive test coverage

Fixes #145
```

### Pull Request Guidelines

1. **Branch Naming**: `feature/description`, `fix/issue-number`, `docs/update-type`
2. **Testing**: All tests must pass, new tests for new functionality
3. **Documentation**: Update relevant documentation
4. **Code Review**: At least one approval required
5. **Quality Gates**: Linting, formatting, and coverage requirements

### Release Process

```bash
# Create release branch
git checkout -b release/v2.1.0

# Update version numbers
# Update CHANGELOG.md
# Run full test suite
python -m pytest tests/ --cov=src

# Create release
git tag v2.1.0
git push origin v2.1.0
```

---

**Related**: [[API Reference|API-Reference]] - Programming interfaces  
**Next**: [[Agent Documentation|Agent-Documentation]] - Individual agent details