# API Reference

Complete programming interface documentation for Open-Sourcefy Matrix pipeline.

## Core API Components

### Matrix Pipeline Orchestrator

#### MatrixPipelineOrchestrator Class

**File**: `src/core/matrix_pipeline_orchestrator.py`

```python
class MatrixPipelineOrchestrator:
    """Master coordinator for the Matrix agent pipeline"""
    
    def __init__(self, config_manager: ConfigManager = None):
        """Initialize the orchestrator with optional configuration manager"""
        pass
        
    def execute_pipeline(self, binary_path: str, selected_agents: List[int] = None) -> PipelineResult:
        """Execute the Matrix pipeline on a binary file"""
        pass
        
    def get_agent_dependencies(self, agent_id: int) -> List[int]:
        """Get dependencies for a specific agent"""
        pass
        
    def validate_prerequisites(self) -> ValidationResult:
        """Validate all system prerequisites"""
        pass
```

#### Pipeline Execution Methods

```python
def execute_full_pipeline(binary_path: str, **kwargs) -> Dict[str, Any]:
    """Execute complete 17-agent pipeline"""
    pass

def execute_agent_batch(agent_ids: List[int], context: Dict[str, Any]) -> Dict[int, AgentResult]:
    """Execute a batch of agents in parallel"""
    pass

def execute_single_agent(agent_id: int, context: Dict[str, Any]) -> AgentResult:
    """Execute a single agent"""
    pass
```

### Agent Base Framework

#### MatrixAgent Base Class

**File**: `src/core/matrix_agents.py`

```python
class MatrixAgent(abc.ABC):
    """Enhanced base class for all Matrix agents with LangChain integration"""
    
    def __init__(self, agent_id: int, matrix_character: MatrixCharacter):
        """Initialize agent with ID and Matrix character"""
        # Source: src/core/matrix_agents.py:93
        
    @abc.abstractmethod
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method - must be overridden by subclasses"""
        # Abstract method defined in matrix_agents.py
        
    def get_dependencies(self) -> List[int]:
        """Get agent dependencies"""
        # Implemented in base class
        
    def get_matrix_description(self) -> str:
        """Get agent description for documentation"""
        # Optional override method
```

**Note**: The actual base class is `MatrixAgent`, not `ReconstructionAgent`. There are also specialized subclasses like `AnalysisAgent`, `ReconstructionAgent`, etc. in the same file.

#### Agent Result Structures

```python
@dataclass
class AgentResult:
    """Standard agent execution result"""
    agent_id: int
    status: AgentStatus
    data: Dict[str, Any]
    agent_name: str
    matrix_character: str
    execution_time: float = 0.0
    quality_score: float = 0.0
    error_message: str = None

@dataclass
class PipelineResult:
    """Complete pipeline execution result"""
    status: PipelineStatus
    agent_results: Dict[int, AgentResult]
    execution_time: float
    quality_metrics: Dict[str, float]
    output_paths: Dict[str, str]
    error_summary: List[str] = None
```

### Configuration Management

#### ConfigManager Class

**File**: `src/core/config_manager.py`

```python
class ConfigManager:
    """System configuration management"""
    
    def __init__(self, config_path: str = None):
        """Initialize with optional config file path"""
        pass
        
    def get_agent_config(self, agent_id: int) -> Dict[str, Any]:
        """Get configuration for specific agent"""
        pass
        
    def get_build_config(self) -> BuildConfig:
        """Get build system configuration"""
        pass
        
    def validate_configuration(self) -> ValidationResult:
        """Validate all configuration settings"""
        pass
        
    def update_config(self, section: str, values: Dict[str, Any]) -> None:
        """Update configuration section with new values"""
        pass
```

#### Build Configuration

```python
@dataclass
class BuildConfig:
    """Build system configuration"""
    visual_studio_path: str
    msbuild_path: str
    cl_exe_path: str
    rc_exe_path: str
    lib_exe_path: str
    target_platform: str = "x64"
    configuration: str = "Release"
    
    def validate_paths(self) -> bool:
        """Validate all configured paths exist"""
        pass
        
    def get_compiler_flags(self) -> List[str]:
        """Get configured compiler flags"""
        pass
```

## Agent-Specific APIs

### Agent 1: Sentinel

```python
class Agent01_Sentinel(MatrixAgent):
    """Binary Discovery and Security Scanning"""
    
    def analyze_binary_format(self, binary_path: str) -> BinaryAnalysis:
        """Analyze binary format and structure"""
        pass
        
    def extract_import_table(self, binary_path: str) -> ImportAnalysis:
        """Extract complete import table with 538+ functions"""
        pass
        
    def perform_security_scan(self, binary_path: str) -> SecurityAssessment:
        """Comprehensive security analysis"""
        pass
        
    def extract_metadata(self, binary_path: str) -> BinaryMetadata:
        """Extract file metadata and properties"""
        pass
```

### Agent 5: Neo

```python
class Agent05_Neo(MatrixAgent):
    """Advanced Decompilation with Ghidra Integration"""
    
    def initialize_ghidra_project(self, binary_path: str) -> GhidraProject:
        """Initialize Ghidra project for decompilation"""
        pass
        
    def perform_decompilation(self, project: GhidraProject) -> DecompilationResult:
        """Execute Ghidra decompilation"""
        pass
        
    def infer_types(self, decompilation: DecompilationResult) -> TypeInference:
        """Perform type inference on decompiled code"""
        pass
        
    def analyze_functions(self, decompilation: DecompilationResult) -> FunctionAnalysis:
        """Analyze function signatures and relationships"""
        pass
```

### Agent 9: Commander Locke

```python
class Agent09_TheMachine(MatrixAgent):
    """Global Source Reconstruction and Compilation"""
    
    def generate_source_code(self, analysis_data: Dict[str, Any]) -> SourceGeneration:
        """Generate complete C source code"""
        pass
        
    def create_build_system(self, source_path: str) -> BuildSystemGeneration:
        """Generate MSBuild/CMake build files"""
        pass
        
    def compile_binary(self, build_config: BuildConfig) -> CompilationResult:
        """Compile generated source using VS2022"""
        pass
        
    def validate_output(self, original_path: str, compiled_path: str) -> ValidationResult:
        """Validate compiled output against original"""
        pass
```

## Data Structures

### Core Data Types

```python
@dataclass
class BinaryAnalysis:
    """Binary format analysis result"""
    format: str  # PE32, PE64, ELF, etc.
    architecture: str  # x86, x64, ARM, etc.
    file_size: int
    entropy: float
    sections: List[SectionInfo]
    imports: List[ImportInfo]
    exports: List[ExportInfo]
    resources: List[ResourceInfo]

@dataclass
class ImportAnalysis:
    """Import table analysis result"""
    total_functions: int
    dll_count: int
    resolved_functions: int
    ordinal_imports: int
    mfc_detected: bool
    import_details: List[ImportDetail]

@dataclass
class DecompilationResult:
    """Ghidra decompilation result"""
    success: bool
    functions: List[FunctionInfo]
    global_variables: List[VariableInfo]
    data_types: List[DataTypeInfo]
    quality_score: float
    analysis_time: float
```

### Quality Metrics

```python
@dataclass
class QualityMetrics:
    """Pipeline quality assessment"""
    overall_quality: float
    agent_scores: Dict[int, float]
    compilation_success: bool
    binary_accuracy: float  # Size comparison
    function_recovery_rate: float
    import_resolution_rate: float
    resource_extraction_rate: float
```

## Utility Functions

### File Operations

```python
def validate_binary_file(file_path: str) -> bool:
    """Validate that file is a supported binary format"""
    pass

def create_output_structure(base_path: str, binary_name: str) -> Dict[str, str]:
    """Create structured output directory"""
    pass

def cleanup_temporary_files(output_path: str) -> None:
    """Clean up temporary files and directories"""
    pass
```

### Validation Utilities

```python
def validate_agent_result(result: Dict[str, Any]) -> ValidationResult:
    """Validate agent execution result structure"""
    pass

def calculate_quality_score(metrics: Dict[str, float]) -> float:
    """Calculate overall quality score from metrics"""
    pass

def verify_prerequisites() -> List[PrerequisiteCheck]:
    """Check all system prerequisites"""
    pass
```

## Error Handling

### Exception Classes

```python
class MatrixError(Exception):
    """Base exception for Matrix pipeline errors"""
    pass

class AgentExecutionError(MatrixError):
    """Agent execution failure"""
    def __init__(self, agent_id: int, message: str):
        self.agent_id = agent_id
        super().__init__(f"Agent {agent_id}: {message}")

class PrerequisiteError(MatrixError):
    """Missing prerequisite error"""
    pass

class ConfigurationError(MatrixError):
    """Configuration validation error"""
    pass

class CompilationError(MatrixError):
    """Build system compilation error"""
    pass
```

### Error Codes

```python
class ErrorCodes:
    """Standard error codes"""
    E001_MISSING_VS2022 = "E001: Missing VS2022 Preview installation"
    E002_INVALID_CONFIG = "E002: Invalid build_config.yaml configuration"
    E003_INSUFFICIENT_RESOURCES = "E003: Insufficient system resources"
    E004_AGENT_PREREQUISITES = "E004: Agent prerequisite validation failure"
    E005_IMPORT_TABLE_FAILURE = "E005: Import table reconstruction failure"
```

## Usage Examples

### Basic Pipeline Execution

```python
from src.core.matrix_pipeline_orchestrator import MatrixPipelineOrchestrator
from src.core.config_manager import ConfigManager

# Initialize orchestrator
config = ConfigManager()
orchestrator = MatrixPipelineOrchestrator(config)

# Execute full pipeline
result = orchestrator.execute_pipeline("input/binary.exe")

if result.status == PipelineStatus.SUCCESS:
    print(f"Pipeline completed successfully in {result.execution_time:.2f}s")
    print(f"Overall quality: {result.quality_metrics['overall_quality']:.2f}")
else:
    print("Pipeline failed:")
    for error in result.error_summary:
        print(f"  - {error}")
```

### Custom Agent Execution

```python
# Execute specific agents
selected_agents = [1, 2, 5, 9]  # Foundation + Advanced + Compilation
result = orchestrator.execute_pipeline(
    binary_path="input/custom.exe",
    selected_agents=selected_agents
)

# Access individual agent results
sentinel_result = result.agent_results[1]
neo_result = result.agent_results[5]
commander_result = result.agent_results[9]
```

### Configuration Management

```python
# Load and validate configuration
config = ConfigManager("custom_config.yaml")
validation = config.validate_configuration()

if not validation.is_valid:
    for error in validation.errors:
        print(f"Config error: {error}")

# Update build configuration
build_config = config.get_build_config()
build_config.target_platform = "x64"
build_config.configuration = "Debug"
```

---

**Related**: [[Architecture Overview|Architecture-Overview]] - System design details  
**Next**: [[Developer Guide|Developer-Guide]] - Development and contribution guide