# API Reference

Complete API reference for Open-Sourcefy Matrix pipeline components.

## Core Framework APIs

### MatrixPipelineOrchestrator

**File**: `src/core/matrix_pipeline_orchestrator.py`

The master coordinator that orchestrates all agent execution.

#### Class Definition
```python
class MatrixPipelineOrchestrator:
    def __init__(self, config_manager: ConfigManager = None)
```

#### Methods

##### execute_pipeline()
```python
def execute_pipeline(
    self, 
    binary_path: str, 
    selected_agents: List[int] = None,
    output_base_dir: str = None
) -> PipelineResult
```

Executes the complete Matrix pipeline.

**Parameters**:
- `binary_path` (str): Path to target binary file
- `selected_agents` (List[int], optional): Specific agents to run
- `output_base_dir` (str, optional): Custom output directory

**Returns**: `PipelineResult` object with execution summary

**Example**:
```python
from src.core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator

orchestrator = MatrixPipelineOrchestrator()
result = orchestrator.execute_pipeline("input/launcher.exe")
print(f"Pipeline success: {result.success}")
print(f"Agents completed: {result.agents_completed}")
```

##### validate_environment()
```python
def validate_environment(self) -> EnvironmentValidation
```

Validates system environment and dependencies.

**Returns**: `EnvironmentValidation` with validation results

##### get_agent_dependencies()
```python
def get_agent_dependencies(self, agent_ids: List[int]) -> Dict[int, List[int]]
```

Returns dependency mapping for specified agents.

**Parameters**:
- `agent_ids` (List[int]): Agent IDs to analyze

**Returns**: Dictionary mapping agent ID to its dependencies

### Agent Base Classes

#### ReconstructionAgent

**File**: `src/core/shared_components.py`

Base class for all Matrix agents.

```python
class ReconstructionAgent:
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter)
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None
    def get_matrix_description(self) -> str
    def _get_required_context_keys(self) -> List[str]
```

#### Agent Implementation Example
```python
from src.core.shared_components import ReconstructionAgent
from src.core.matrix_agents import MatrixCharacter, AgentStatus

class MyCustomAgent(ReconstructionAgent):
    def __init__(self):
        super().__init__(
            agent_id=99,
            matrix_character=MatrixCharacter.CUSTOM
        )
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Validate prerequisites
        self._validate_prerequisites(context)
        
        # Perform agent-specific work
        result_data = self._perform_analysis(context)
        
        # Return structured result
        return {
            'agent_id': self.agent_id,
            'status': 'SUCCESS',
            'data': result_data,
            'execution_time': time.time() - start_time
        }
    
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        required_keys = ['binary_path', 'shared_memory']
        missing_keys = [k for k in required_keys if k not in context]
        if missing_keys:
            raise ValueError(f"Missing required context keys: {missing_keys}")
```

### Data Structures

#### AgentResult

Standardized result object for agent execution.

```python
@dataclass
class AgentResult:
    agent_id: int
    status: AgentStatus
    data: Dict[str, Any]
    agent_name: str
    matrix_character: str
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    quality_score: Optional[float] = None
```

**Usage**:
```python
from src.core.matrix_agents import AgentResult, AgentStatus

result = AgentResult(
    agent_id=1,
    status=AgentStatus.SUCCESS,
    data={'binary_info': {'format': 'PE32'}},
    agent_name="Sentinel",
    matrix_character="sentinel"
)
```

#### PipelineResult

Overall pipeline execution result.

```python
@dataclass
class PipelineResult:
    success: bool
    agents_completed: int
    agents_failed: int
    total_execution_time: float
    output_directory: str
    agent_results: Dict[int, AgentResult]
    pipeline_metrics: Dict[str, Any]
    error_summary: Optional[str] = None
```

#### EnvironmentValidation

Environment validation results.

```python
@dataclass
class EnvironmentValidation:
    overall_status: bool
    python_version: str
    dependencies_satisfied: bool
    build_tools_available: bool
    ghidra_available: bool
    vs2022_available: bool
    validation_details: Dict[str, Any]
```

## Agent-Specific APIs

### Agent 1: Sentinel API

```python
class SentinelAgent(ReconstructionAgent):
    def analyze_binary_format(self, binary_path: str) -> BinaryInfo
    def extract_import_table(self, binary_path: str) -> ImportAnalysis
    def perform_security_scan(self, binary_path: str) -> SecurityAssessment
    def extract_metadata(self, binary_path: str) -> BinaryMetadata
```

**Data Structures**:
```python
@dataclass
class BinaryInfo:
    format_type: str  # "PE32", "PE32+", "ELF", "Mach-O"
    architecture: str  # "x86", "x64", "ARM"
    file_size: int
    compilation_timestamp: Optional[datetime]
    digital_signature: Optional[str]

@dataclass
class ImportAnalysis:
    total_functions: int
    dll_dependencies: List[str]
    function_list: List[FunctionImport]
    ordinal_imports: List[OrdinalImport]
    mfc_version: Optional[str]
```

### Agent 5: Neo API (Ghidra Integration)

```python
class NeoAgent(ReconstructionAgent):
    def initialize_ghidra_project(self, binary_path: str) -> GhidraProject
    def perform_decompilation(self, project: GhidraProject) -> DecompilationResult
    def extract_function_signatures(self, project: GhidraProject) -> List[FunctionSignature]
    def analyze_data_types(self, project: GhidraProject) -> TypeAnalysis
```

**Data Structures**:
```python
@dataclass
class GhidraProject:
    project_path: str
    binary_loaded: bool
    analysis_complete: bool
    function_count: int

@dataclass
class DecompilationResult:
    functions_decompiled: int
    decompilation_quality: float
    source_code: Dict[str, str]  # function_name -> decompiled_code
    analysis_time: float
```

### Agent 9: Commander Locke API (Compilation)

```python
class CommanderLockeAgent(ReconstructionAgent):
    def generate_source_code(self, context: Dict[str, Any]) -> SourceGeneration
    def create_build_system(self, output_dir: str) -> BuildSystem
    def compile_binary(self, build_dir: str) -> CompilationResult
    def integrate_resources(self, resources: Dict[str, Any]) -> ResourceIntegration
```

**Data Structures**:
```python
@dataclass
class SourceGeneration:
    main_file: str
    header_files: List[str]
    source_files: List[str]
    total_lines: int
    generation_quality: float

@dataclass
class CompilationResult:
    success: bool
    output_binary: str
    output_size: int
    compilation_time: float
    warnings: List[str]
    errors: List[str]
```

## Configuration APIs

### ConfigManager

**File**: `src/core/config_manager.py`

```python
class ConfigManager:
    def __init__(self, config_path: str = "config.yaml")
    
    def get_value(self, key_path: str, default: Any = None) -> Any
    def set_value(self, key_path: str, value: Any) -> None
    def validate_configuration(self) -> ValidationResult
    def get_build_config(self) -> BuildConfiguration
    def get_agent_config(self, agent_id: int) -> AgentConfiguration
```

**Usage**:
```python
from src.core.config_manager import ConfigManager

config = ConfigManager("build_config.yaml")

# Get specific configuration values
vs_path = config.get_value("build_system.visual_studio.installation_path")
timeout = config.get_value("agents.timeout", 300)

# Validate configuration
validation = config.validate_configuration()
if not validation.is_valid:
    print(f"Configuration errors: {validation.errors}")
```

### BuildConfiguration

```python
@dataclass
class BuildConfiguration:
    visual_studio_path: str
    msbuild_path: str
    cl_exe_path: str
    rc_exe_path: str
    lib_exe_path: str
    sdk_version: str
    target_platform: str
```

## Utility APIs

### PerformanceMonitor

**File**: `src/core/shared_utils.py`

```python
class PerformanceMonitor:
    def start_operation(self, operation_name: str) -> None
    def end_operation(self, operation_name: str) -> float
    def get_metrics(self) -> Dict[str, Any]
    def generate_report(self) -> PerformanceReport
```

**Usage**:
```python
from src.core.shared_utils import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_operation("binary_analysis")
# ... perform analysis ...
duration = monitor.end_operation("binary_analysis")
print(f"Analysis took {duration:.2f} seconds")
```

### ErrorHandler

```python
class ErrorHandler:
    def handle_agent_error(self, agent_id: int, error: Exception) -> ErrorResponse
    def log_pipeline_error(self, context: str, error: Exception) -> None
    def get_error_summary(self) -> List[ErrorSummary]
```

## CLI Interface API

### Command Line Arguments

```python
# Main CLI entry point
python main.py [options] [binary_path]

# Core options
--agents AGENT_LIST          # Specific agents to run (e.g., "1,3,7" or "1-5")
--output-dir OUTPUT_DIR       # Custom output directory
--debug                       # Enable debug logging
--profile                     # Enable performance profiling

# Pipeline modes
--full-pipeline              # Run all agents (0-16)
--decompile-only            # Decompilation-focused agents
--analyze-only              # Analysis without compilation
--compile-only              # Compilation-focused agents

# Validation options
--verify-env                # Validate environment setup
--validate-pipeline MODE    # Validate pipeline (basic/comprehensive)
--config-summary           # Show configuration summary

# Development options
--dry-run                  # Show execution plan without running
--force-reprocess         # Ignore cached results
--clean-temp             # Clean temporary files
```

### Programmatic CLI Usage

```python
from src.main import main
import sys

# Set command line arguments programmatically
sys.argv = [
    'main.py',
    'input/launcher.exe',
    '--agents', '1,2,5,9',
    '--output-dir', 'custom_output',
    '--debug'
]

# Run main function
result = main()
print(f"Exit code: {result}")
```

## Error Handling

### Exception Types

```python
class MatrixPipelineError(Exception):
    """Base exception for pipeline errors"""
    pass

class AgentExecutionError(MatrixPipelineError):
    """Agent execution failure"""
    def __init__(self, agent_id: int, message: str):
        self.agent_id = agent_id
        super().__init__(f"Agent {agent_id}: {message}")

class ConfigurationError(MatrixPipelineError):
    """Configuration validation error"""
    pass

class EnvironmentError(MatrixPipelineError):
    """Environment setup error"""
    pass
```

### Error Response Structure

```python
@dataclass
class ErrorResponse:
    error_code: str
    error_message: str
    agent_id: Optional[int]
    context: Dict[str, Any]
    recovery_suggestion: Optional[str]
    timestamp: datetime
```

## Testing APIs

### Test Framework Integration

```python
import unittest
from src.tests.test_framework import MatrixTestCase

class TestCustomAgent(MatrixTestCase):
    def setUp(self):
        super().setUp()
        self.agent = MyCustomAgent()
    
    def test_agent_execution(self):
        context = self.create_test_context()
        result = self.agent.execute_matrix_task(context)
        self.assert_agent_success(result)
    
    def test_agent_validation(self):
        invalid_context = {}
        with self.assertRaises(ValueError):
            self.agent._validate_prerequisites(invalid_context)
```

## Integration Examples

### Custom Pipeline Integration

```python
from src.core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator
from src.core.config_manager import ConfigManager

def custom_pipeline_execution():
    # Initialize components
    config = ConfigManager("custom_config.yaml")
    orchestrator = MatrixPipelineOrchestrator(config)
    
    # Validate environment
    env_validation = orchestrator.validate_environment()
    if not env_validation.overall_status:
        raise RuntimeError("Environment validation failed")
    
    # Execute pipeline
    result = orchestrator.execute_pipeline(
        binary_path="input/target.exe",
        selected_agents=[1, 2, 5, 9, 15, 16],
        output_base_dir="custom_output"
    )
    
    # Process results
    if result.success:
        print(f"Pipeline completed successfully in {result.total_execution_time:.2f}s")
        print(f"Output available in: {result.output_directory}")
    else:
        print(f"Pipeline failed: {result.error_summary}")
    
    return result
```

### Agent Development Template

```python
from src.core.shared_components import ReconstructionAgent
from src.core.matrix_agents import MatrixCharacter, AgentStatus
from typing import Dict, Any, List
import time

class Agent99_CustomAgent(ReconstructionAgent):
    """Custom agent implementation template"""
    
    def __init__(self):
        super().__init__(
            agent_id=99,
            matrix_character=MatrixCharacter.CUSTOM  # Define in MatrixCharacter enum
        )
        self.capabilities = {
            'custom_analysis': True,
            'specialized_processing': True
        }
    
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method - required override"""
        start_time = time.time()
        
        try:
            # Validate prerequisites
            self._validate_prerequisites(context)
            
            # Perform custom analysis
            analysis_result = self._perform_custom_analysis(context)
            
            # Process results
            processed_data = self._process_results(analysis_result)
            
            execution_time = time.time() - start_time
            
            return {
                'agent_id': self.agent_id,
                'status': 'SUCCESS',
                'data': processed_data,
                'execution_time': execution_time,
                'quality_score': self._calculate_quality_score(processed_data),
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
        """Validate required context - required override"""
        required_keys = self._get_required_context_keys()
        missing_keys = [k for k in required_keys if k not in context]
        
        if missing_keys:
            raise ValueError(f"Missing required context keys: {missing_keys}")
        
        # Add custom validation logic here
        if 'binary_path' in context:
            binary_path = Path(context['binary_path'])
            if not binary_path.exists():
                raise FileNotFoundError(f"Binary file not found: {binary_path}")
    
    def _get_required_context_keys(self) -> List[str]:
        """Define required context keys"""
        return ['binary_path', 'shared_memory', 'output_paths']
    
    def get_matrix_description(self) -> str:
        """Agent description for logging and documentation"""
        return (
            f"Agent {self.agent_id}: Custom Agent - "
            f"Specialized processing with custom analysis capabilities"
        )
    
    def _perform_custom_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Custom analysis implementation"""
        # Implement your custom analysis logic here
        return {
            'analysis_type': 'custom',
            'results': {},
            'confidence': 0.85
        }
    
    def _process_results(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format analysis results"""
        return {
            'processed_data': analysis_result,
            'metadata': {
                'agent_version': '1.0',
                'processing_method': 'custom'
            }
        }
    
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate quality score for results"""
        # Implement quality scoring logic
        return 0.85  # Example score
```

---

**Related**: [[Agent Documentation|Agent-Documentation]] - Individual agent details  
**Examples**: [[Developer Guide|Developer-Guide]] - Development examples