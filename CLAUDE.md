# CLAUDE.md

🚨 **OBLIGATORY READING OF rules.md FILE OR DEATH SENTENCE** 🚨
READ /mnt/c/Users/pascaldisse/Downloads/open-sourcefy/rules.md IMMEDIATELY BEFORE ANY WORK
ALL RULES IN rules.md ARE ABSOLUTE AND NON-NEGOTIABLE

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CRITICAL: NO FALLBACKS, NO ALTERNATIVES, NO DEGRADATION
NEVER USE FALLBACK PATHS, MOCK IMPLEMENTATIONS, OR WORKAROUNDS
STRICT MODE ONLY - FAIL FAST WHEN TOOLS ARE MISSING

## Project Overview

Open-Sourcefy is an AI-powered binary decompilation system that reconstructs compilable C source code from Windows PE executables using a 17-agent Matrix pipeline with Ghidra integration. 

**WINDOWS ONLY SYSTEM**: This system exclusively supports Windows PE executables and requires Visual Studio/MSBuild for compilation. Linux/macOS platforms and other binary formats (ELF/Mach-O) are not supported.

The primary test target is the Matrix Online launcher.exe binary.

### Latest Enhancements (Phase 4 - June 2025)

**Phase 4: Advanced Validation & Update Systems** ✅ COMPLETE
- **P4.1**: Pipeline update mode with `--update` flag for incremental development workflow
- **P4.2**: Advanced binary comparison engine with Shannon entropy calculation and semantic validation
- **P4.3**: Multi-dimensional quality scoring system with 7 quality dimensions and weighted analysis
- **P4.4**: Integrated validation reporting system (JSON/HTML/Markdown formats) with executive summaries
- **P4.5**: True semantic decompilation engine vs intelligent scaffolding with advanced type inference
- **P4.6**: Git repository integration with comprehensive .gitignore and professional development workflow

**Phase 3: Semantic Decompilation Engine** ✅ COMPLETE
- **P3.1**: Advanced function signature recovery with Windows API analysis and calling convention detection
- **P3.2**: Data type inference and reconstruction with constraint-based solving and data flow analysis
- **P3.3**: Data structure recovery for complex types (linked lists, trees, nested structures)
- **P3.4**: Cross-validation between multiple analysis engines for enhanced accuracy

*Combined Phase 3 & 4: ~6,850 additional lines of production code enhancing the core Matrix pipeline*

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment and dependencies
python3 main.py --verify-env

# List available agents
python3 main.py --list-agents
```

### Running the Pipeline
```bash
# Full pipeline (auto-detects binary from input/ directory)
python3 main.py

# Specific binary
python3 main.py launcher.exe

# Pipeline modes
python3 main.py --full-pipeline              # All agents (0-16)
python3 main.py --decompile-only             # Agents 1,2,5,7,14
python3 main.py --analyze-only               # Agents 1,2,3,4,5,6,7,8,9,14,15
python3 main.py --compile-only               # Agents 1,2,4,5,6,7,8,9,10,11,12
python3 main.py --validate-only              # Agents 1,2,4,5,6,7,8,9,10,11,12,13,16

# Specific agents
python3 main.py --agents 1                   # Single agent (Sentinel)
python3 main.py --agents 1,3,7               # Multiple agents (Sentinel, Merovingian, Trainman)
python3 main.py --agents 1-5                 # Agent ranges (Foundation + Core Analysis)
python3 main.py --agents 0                   # Master orchestrator only

# Execution modes
python3 main.py --execution-mode master_first_parallel    # Default
python3 main.py --execution-mode pure_parallel           # Pure parallel
python3 main.py --execution-mode sequential              # Sequential

# Resource profiles
python3 main.py --resource-profile standard              # Default
python3 main.py --resource-profile high_performance      # High resource usage
python3 main.py --resource-profile conservative         # Conservative usage

# Development options
python3 main.py --dry-run                    # Show execution plan
python3 main.py --debug --profile            # Debug with profiling
```

### Testing and Validation
```bash
# Environment validation
python3 main.py --verify-env

# Configuration summary
python3 main.py --config-summary

# List available agents and modes
python3 main.py --list-agents

# Run test suites (using unittest framework)
python3 -m unittest discover tests -v              # Run all tests
python3 -m unittest discover -s tests -p "test_*.py" -v  # Alternative with pattern
python3 tests/test_full_pipeline.py                # Run specific test file
python3 tests/test_agent_individual.py             # Individual agent tests
python3 tests/test_week4_validation.py             # Validation framework tests

# Built-in testing commands in main.py
python3 main.py --run-regression-tests             # Regression testing
python3 main.py --validate-pipeline basic          # Pipeline validation
python3 main.py --validate-pipeline comprehensive  # Comprehensive validation
python3 main.py --validate-binary file1.exe        # Binary validation
python3 main.py --benchmark-performance             # Performance benchmarks
```

## Architecture Overview

### Matrix Agent Pipeline System

The system implements a **17-agent Matrix pipeline** with master-first execution and production-ready architecture:

**Agent 0 - Master Orchestrator**:
- **Deus Ex Machina**: Supreme orchestrator that coordinates the entire pipeline
- Creates execution plans, manages agent batches, validates prerequisites
- Generates comprehensive reports and performance metrics

**Phase 1 - Foundation** (Agent 1):
- **Sentinel**: Binary discovery, metadata analysis, and security scanning
- Multi-format support (PE/ELF/Mach-O), hash calculation, entropy analysis
- LangChain AI integration for enhanced threat detection

**Phase 2 - Core Analysis** (Agents 2-4):
- **The Architect**: Architecture analysis, compiler detection, optimization patterns
- **The Merovingian**: Basic decompilation, function detection, control flow analysis
- **Agent Smith**: Binary structure analysis, data extraction, dynamic bridge preparation

**Phase 3 - Advanced Analysis** (Agents 5-12):
- **Neo**: Advanced decompilation with comprehensive Ghidra integration
- **The Twins**: Binary differential analysis and comparison engine
- **The Trainman**: Advanced assembly analysis and instruction flow transportation
- **The Keymaker**: Resource reconstruction and dependency access management
- **Commander Locke**: Global reconstruction orchestration and project coordination
- **The Machine**: Compilation orchestration and MSBuild integration
- **The Oracle**: Final validation, truth verification, and quality assessment
- **Link**: Cross-reference analysis and symbol linking

**Phase 4 - Final Processing** (Agents 13-16):
- **Agent Johnson**: Security analysis and vulnerability detection
- **The Cleaner**: Code cleanup, optimization, and formatting
- **The Analyst**: Advanced metadata analysis and intelligence synthesis
- **Agent Brown**: Final quality assurance and automated testing

### Execution Model

**Master-First Parallel Execution**:
1. **Master Agent (Agent 0)** coordinates the entire pipeline
2. **Dependency-Based Batching**: Agents organized into execution batches based on dependencies
3. **Parallel Execution**: Agents within batches execute in parallel with timeout management
4. **Context Sharing**: Global execution context passed between agents with shared memory
5. **Fail-Fast Validation**: Quality thresholds enforced at each stage

**Dependency Structure**:
```
Agent 1 (Sentinel) → No dependencies
Agents 2,3,4 → Depend on Agent 1
Agents 5,6,7,8 → Depend on Agents 1,2
Agents 9,12,13 → Depend on Agents 5,6,7,8
Agent 10 → Depends on Agent 9
Agent 11 → Depends on Agent 10
Agents 14,15 → Depend on Agents 9,10,11
Agent 16 → Depends on Agents 14,15
```

### Core System Components

**Pipeline Orchestrator** (`core/matrix_pipeline_orchestrator.py`):
- Master-first execution with parallel agent coordination
- Comprehensive configuration management and resource limits
- Async execution with timeout and error handling
- Report generation and performance metrics

**Matrix Agent Framework** (`core/matrix_agents_v2.py`):
- Production-ready base classes with Matrix-themed architecture
- Standardized agent result structures and status management
- Specialized base classes: AnalysisAgent, DecompilerAgent, ReconstructionAgent, ValidationAgent
- Comprehensive dependency mapping for all 17 agents

**Shared Components** (`core/shared_components.py`):
- MatrixLogger: Enhanced logging with Matrix-themed formatting
- MatrixFileManager: Standardized file operations
- MatrixValidator: Common validation functions
- MatrixProgressTracker: Progress tracking with ETA calculation
- MatrixErrorHandler: Standardized error handling with retry logic
- SharedAnalysisTools: Entropy calculation and pattern detection

**Configuration Manager** (`core/config_manager.py`):
- Hierarchical configuration with environment variables, YAML/JSON config files
- Auto-detection of tools (Ghidra, Java, Visual Studio)
- Agent-specific settings and resource limits
- No hardcoded values - fully configurable system

**CLI Interface** (`main.py`):
- Advanced CLI with comprehensive argument parsing
- Multiple execution modes and resource profiles
- Async pipeline execution with performance profiling
- Dry-run mode for execution planning

### Output Organization

**Standard Mode**: All output is organized under `output/{binary_name}/{yyyymmdd-hhmmss}/`:
```
output/launcher/20250609-143022/
├── agents/          # Agent-specific analysis outputs
│   ├── agent_01_sentinel/
│   ├── agent_02_architect/
│   └── ...
├── ghidra/          # Ghidra decompilation results
├── compilation/     # MSBuild artifacts and generated source
├── reports/         # Pipeline execution reports
│   └── matrix_pipeline_report.json
├── logs/            # Execution logs and debug information
├── temp/            # Temporary files (auto-cleaned)
├── tests/           # Generated test files
└── docs/            # General source code documentation
```

**Update Mode** (`--update` flag): Output saved to `output/{binary_name}/latest/`:
```
output/launcher/latest/
├── agents/          # Agent-specific analysis outputs (updated incrementally)
│   ├── agent_01_sentinel/
│   ├── agent_02_architect/
│   └── ...
├── ghidra/          # Ghidra decompilation results (updated)
├── compilation/     # MSBuild artifacts and generated source (updated)
├── reports/         # Pipeline execution reports (updated)
│   ├── matrix_pipeline_report.json
│   ├── binary_comparison_report.json
│   ├── quality_metrics_report.json
│   └── validation_report.html
├── logs/            # Execution logs (appended)
├── temp/            # Temporary files (auto-cleaned)
└── tests/           # Generated test files (updated)
```

**Path Configuration**: All paths are configurable via the config manager. The timestamp format can be customized using the `paths.timestamp_format` configuration (default: `%Y%m%d-%H%M%S`). Update mode uses a fixed `latest` directory for incremental development.

## Development Guidelines

### Matrix Agent Development

When creating or modifying Matrix agents:

1. **Inherit from appropriate base class**:
   - `AnalysisAgent` for analysis-focused agents (Agents 1-8)
   - `DecompilerAgent` for decompilation agents (Agent 3, 5, 7)
   - `ReconstructionAgent` for reconstruction agents (Agents 9-16)
   - `ValidationAgent` for validation agents (Agents 11, 13, 16)

2. **Follow Matrix naming conventions**:
   - Use Matrix character themes for agent names
   - Implement `get_matrix_description()` with character-appropriate descriptions

3. **Implement required methods**:
   - `execute_matrix_task(context)`: Main agent logic
   - `_validate_prerequisites(context)`: Input validation
   - `_get_required_context_keys()`: Required context dependencies

4. **Use shared components**:
   - Import from `shared_components` for common functionality
   - Use `MatrixLogger`, `MatrixFileManager`, `MatrixValidator`
   - Leverage `SharedAnalysisTools` and `SharedValidationTools`

5. **Follow SOLID principles**:
   - No hardcoded values or absolute paths
   - Use configuration manager for all settings
   - Implement comprehensive error handling
   - Use dependency injection for external tools

6. **Quality validation**:
   - Implement fail-fast validation with quality thresholds
   - Use `ValidationError` for prerequisite failures
   - Return structured data with confidence scores

### Configuration

**Environment Variables**:
- `GHIDRA_HOME`: Ghidra installation directory
- `JAVA_HOME`: Java installation directory
- `MATRIX_DEBUG`: Enable debug logging
- `MATRIX_AI_ENABLED`: Enable LangChain AI features
- `MATRIX_OUTPUT_DIR`: Default output directory (default: `output`)
- `MATRIX_TEMP_DIR`: Temporary files directory (default: `temp`)
- `MATRIX_LOG_DIR`: Log files directory (default: `logs`)

**Configuration Files**:
- Support for YAML and JSON configuration files
- Hierarchical configuration: env vars > config files > defaults
- Agent-specific timeouts, retries, and resource limits

**Tool Detection**:
- Automatic detection of Ghidra, Java, Visual Studio
- Relative path resolution from project root
- Graceful degradation when tools are unavailable

### Quality Validation

The system implements strict validation thresholds:
- **Code Quality**: 75% threshold for meaningful code structure
- **Implementation Score**: 75% threshold for real vs placeholder code
- **Completeness**: 70% threshold for project completeness
- **Binary Analysis Confidence**: Minimum confidence scores for format detection

### Dependencies

**Required**:
- Python 3.8+ (async/await support required)
- Java 17+ (for Ghidra integration)
- packaging>=21.0
- setuptools>=60.0
- psutil>=5.9.0 (for system monitoring)

**Included**:
- Ghidra 11.0.3 (in ghidra/ directory)
- Matrix agent implementations (ALL 17 agents substantially implemented)

**Optional** (commented in requirements.txt):
- Microsoft Visual C++ Build Tools (for compilation testing)
- pefile>=2022.5.30 (for PE analysis)
- capstone>=5.0.0 (for disassembly)
- numpy, scikit-learn (for ML features)
- AI providers: anthropic, openai, langchain packages
- Development: pytest, pytest-cov

### Current Implementation Status

**✅ Production-Ready Infrastructure**:
- Complete Matrix agent framework with 17 agents fully implemented
- Master-first parallel execution orchestrator operational  
- Configuration management system with hierarchical config loading
- Advanced CLI interface with comprehensive argument parsing and update mode
- Shared components and utilities complete with Matrix theming
- Git repository integration with comprehensive .gitignore and development workflow

**✅ Phase 4 Advanced Systems** (Recently Completed June 2025):
- **Semantic Decompilation Engine**: True semantic analysis vs scaffolding with advanced type inference (1,100+ lines)
- **Function Signature Recovery**: Windows API analysis and calling convention detection (900+ lines)
- **Data Type Inference**: Constraint-based solving with data flow analysis (1,000+ lines)
- **Data Structure Recovery**: Complex type reconstruction for linked lists, trees, nested structures (1,200+ lines)
- **Binary Comparison Engine**: Semantic validation with Shannon entropy calculation (enhanced existing module)
- **Quality Scoring System**: Multi-dimensional scoring with 7 quality dimensions (1,400+ lines)
- **Validation Reporting**: JSON/HTML/Markdown reports with executive summaries (1,600+ lines)
- **Update Mode Pipeline**: Incremental development workflow with `--update` flag
- **Git Integration**: Professional development workflow with comprehensive .gitignore

**✅ Complete Agent Implementation** (All 17 Agents Implemented):
- Agent 0: Deus Ex Machina (Master Orchestrator) - Production coordination and pipeline management
- Agent 1: Sentinel (Binary Discovery) - Binary analysis, metadata extraction, security scanning  
- Agent 2: The Architect (Architecture Analysis) - Compiler detection, optimization patterns
- Agent 3: The Merovingian (Basic Decompilation) - Function detection, control flow analysis
- Agent 4: Agent Smith (Binary Structure) - Structure analysis, data extraction, dynamic bridge
- Agent 5: Neo (Advanced Decompiler) - Advanced decompilation with Ghidra integration, semantic engine integration
- Agent 6: The Twins (Binary Differential) - Binary comparison and differential analysis with advanced validation
- Agent 7: The Trainman (Assembly Analysis) - Advanced assembly analysis and instruction flow
- Agent 8: The Keymaker (Resource Reconstruction) - Resource extraction and dependency analysis
- Agent 9: Commander Locke (Global Reconstruction) - Project structure and global coordination
- Agent 10: The Machine (Compilation Orchestrator) - MSBuild integration and compilation management
- Agent 11: The Oracle (Final Validation) - Truth verification and final validation
- Agent 12: Link (Cross-Reference Analysis) - Symbol resolution and cross-referencing
- Agent 13: Agent Johnson (Security Analysis) - Vulnerability detection and security scanning
- Agent 14: The Cleaner (Code Cleanup) - Code optimization and cleanup routines
- Agent 15: The Analyst (Metadata Analysis) - Intelligence synthesis and metadata analysis
- Agent 16: Agent Brown (Final QA) - Quality assurance and automated testing

**📊 System Status**:
- **Architecture**: Production-ready with comprehensive error handling and SOLID principles
- **Total Codebase**: ~25,850+ lines (19,000 base + 6,850 Phase 3/4 enhancements)
- **Implementation Quality**: Advanced semantic analysis and validation systems integrated ✅
- **Primary Target**: Matrix Online launcher.exe (5.3MB, x86 PE32, MSVC .NET 2003)  
- **Execution Model**: Master-first parallel with dependency batching and update mode support
- **AI Integration**: Claude Code CLI integration operational throughout the agent framework
- **Quality Assurance**: NSA-level standards with multi-dimensional quality scoring
- **Pipeline Features**: Update mode, advanced validation, semantic decompilation, git integration
- **Repository Status**: Professional git workflow with comprehensive .gitignore and commit standards

### Testing Approach

**Test Framework**: Uses Python's built-in `unittest` framework (not pytest). Test files are located in `/tests/` directory.

**Test Categories**:
- **Integration Tests**: `test_full_pipeline.py`, `test_integration_*.py`
- **Unit Tests**: `test_agent_individual.py`, `test_context_propagation.py`
- **Validation Tests**: `test_pipeline_validation.py`, `test_week4_validation.py`
- **Quality Assurance**: `test_regression.py`, `test_phase4_comprehensive.py`

**Built-in Validation**:
- Agent results validated for quality and completeness
- Pipeline can terminate early if validation thresholds not met
- Comprehensive error handling and retry logic
- Performance metrics and execution reports

**Environment Verification**:
```bash
python3 main.py --verify-env    # Check all dependencies
python3 main.py --dry-run       # Preview execution plan
python3 main.py --debug         # Detailed logging
```

**Testing Commands**:
```bash
# Single agent testing
python3 main.py --agents 1

# Core analysis testing  
python3 main.py --agents 1-4

# Decompilation pipeline testing
python3 main.py --decompile-only

# Update mode testing
python3 main.py --agents 1 --update
python3 main.py launcher.exe --update
```

### Git Workflow and Repository Management

**Repository Status**: Fully initialized Git repository with professional development workflow.

**Git Commands**:
```bash
# Check repository status
git status

# View commit history
git log --oneline

# Stage and commit changes
git add .
git commit -m "Your commit message

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Create feature branch
git checkout -b feature/new-enhancement

# View repository information
git remote -v
git branch -a
```

**Comprehensive .gitignore**: The repository includes a comprehensive .gitignore covering:
- Python artifacts (*.pyc, __pycache__, *.egg-info)
- Build systems (MSBuild, Visual Studio, compilation outputs)
- IDEs (VS Code, PyCharm, Vim, Emacs)
- Operating systems (Windows, macOS, Linux)
- Ghidra files (*.gpr, *.rep, *.lock)
- Temporary files (temp/, *.tmp, *.bak)
- Project-specific patterns with selective output directory management

## System Requirements

### **Windows Requirements (MANDATORY)**
- **Operating System**: Windows 10/11 (64-bit)
- **Visual Studio**: 2022 Preview (FIXED REQUIREMENT - configured in build_config.yaml)
- **MSBuild**: VS2022 Preview MSBuild (centralized path configuration)
- **Architecture**: x86/x64 Windows executables only
- **⚠️ CRITICAL**: Must have VS2022 Preview installed at the EXACT paths specified in build_config.yaml

### **Core Dependencies**
- **Python**: 3.8+ (Windows version)
- **Java**: 17+ (for Ghidra integration)
- **Ghidra**: 11.0.3 (included in project)
- **MSVC Compiler**: VS2022 Preview cl.exe (configured in build_config.yaml - NOT in PATH)
- **Build System Manager**: Centralized tool access (src/core/build_system_manager.py)

### **Unsupported Platforms & Build Systems**
❌ **Linux/Unix**: Not supported
❌ **macOS**: Not supported  
❌ **ELF binaries**: Not supported
❌ **Mach-O binaries**: Not supported
❌ **GCC/Clang**: Not supported
❌ **Make/CMake**: Not supported
❌ **Alternative Visual Studio versions**: Not supported (only VS2022 Preview)
❌ **Build system fallbacks**: DISABLED - only centralized configuration used
❌ **Auto-detection of build tools**: DISABLED - only configured paths used

### Ghidra Integration

**Installation**: Ghidra 11.0.3 included in `ghidra/` directory
**Detection**: Automatic path resolution from project root
**Usage**: Headless analysis and decompilation
**Quality**: Assessment and confidence scoring
**Management**: Temporary project creation and cleanup

**Custom Scripts**: Located in ghidra/ directory (e.g., CompleteDecompiler.java)
**Integration Points**: Agents 3, 5, 7, 14 leverage Ghidra for decompilation tasks

## Code Quality Standards & Style Guidelines

### NSA-Level Production Code Quality

This project enforces **NSA-level security and production standards** with zero tolerance for security vulnerabilities, hardcoded values, or poor code practices.

#### Core Quality Principles

1. **SOLID Principles Mandatory**:
   - Single Responsibility: Each class/function has ONE clear purpose
   - Open/Closed: Extensible without modification
   - Liskov Substitution: Proper inheritance hierarchies
   - Interface Segregation: Minimal, focused interfaces
   - Dependency Inversion: Depend on abstractions, not concretions

2. **Security-First Development**:
   - NO hardcoded secrets, API keys, or sensitive data
   - ALL file paths must be configurable and validated
   - Input validation and sanitization mandatory
   - Error messages must not leak sensitive information
   - Proper exception handling with fail-safe defaults

3. **Configuration-Driven Architecture**:
   - ZERO hardcoded values in production code
   - All settings via environment variables or config files
   - Hierarchical configuration: env vars > config files > defaults
   - Runtime configuration validation and graceful degradation

#### Code Organization Standards

**Project Structure**:
```
open-sourcefy/
├── input/                   # Input binary files for analysis
├── output/                  # Pipeline execution results and artifacts
├── src/                     # Source code and core system
│   ├── core/               # Core framework components
│   │   ├── agents/         # Matrix agent implementations (0-16)
│   │   ├── config_manager.py # Configuration management
│   │   ├── agent_base.py   # Base classes and interfaces
│   │   └── shared_utils.py # Shared utilities and components
│   ├── ml/                 # Machine learning components
│   └── utils/              # Pure utility functions
├── tests/                  # Test suites and validation
├── docs/                   # Project documentation and analysis reports
├── ghidra/                 # Ghidra installation and analysis tools
├── temp/                   # Temporary files and development artifacts
├── prompts/                # AI prompts and pipeline instructions
├── venv/                   # Python virtual environment
├── main.py                 # Primary CLI entry point
├── requirements.txt        # Python dependencies
└── config.yaml             # Main configuration file
```

**Separation of Concerns**:
- **UI Logic**: Separated from business logic (CLI in main.py)
- **Business Logic**: Core functionality in src/core/
- **Data Access**: Configuration and file operations isolated
- **External Services**: Ghidra, AI engines in separate modules
- **Test Infrastructure**: Comprehensive testing in tests/
- **Documentation**: Project docs and analysis reports in docs/

#### Coding Standards

**Python Code Quality**:
```python
# ✅ CORRECT: Production-ready class
class MatrixAgent(ABC):
    """Production-ready agent base class with full documentation."""
    
    def __init__(self, agent_id: int, config: Optional[ConfigManager] = None):
        self._agent_id = self._validate_agent_id(agent_id)
        self._config = config or get_config_manager()
        self._logger = self._setup_logging()
        
    def _validate_agent_id(self, agent_id: int) -> int:
        """Validate agent ID is within acceptable range."""
        if not isinstance(agent_id, int) or agent_id < 0:
            raise ValueError(f"Invalid agent ID: {agent_id}")
        return agent_id
        
    @abstractmethod
    def execute(self, context: ExecutionContext) -> AgentResult:
        """Execute agent task with comprehensive error handling."""
        pass

# ❌ INCORRECT: Poor quality code
class Agent:
    def __init__(self, id):
        self.id = id
        self.config = "/hardcoded/path"  # NEVER do this
        
    def run(self, data):  # No type hints, no validation
        return data  # No error handling
```

**Function Design**:
```python
# ✅ CORRECT: Small, focused, well-documented
def validate_binary_format(
    binary_path: Path, 
    expected_formats: List[str],
    config: ConfigManager
) -> BinaryValidationResult:
    """
    Validate binary file format against expected formats.
    
    Args:
        binary_path: Path to binary file to validate
        expected_formats: List of acceptable formats (PE, ELF, etc.)
        config: Configuration manager for validation settings
        
    Returns:
        BinaryValidationResult with validation status and metadata
        
    Raises:
        ValidationError: If binary format is invalid or unsupported
        FileNotFoundError: If binary file doesn't exist
    """
    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")
        
    # Implementation with proper error handling
    # ...

# ❌ INCORRECT: Large, complex, poorly documented
def process_stuff(data, flag=True):  # Vague naming, no types
    # No docstring, unclear purpose
    if flag:
        # Complex logic without error handling
        return data * 2
    else:
        return None  # Inconsistent return types
```

**Error Handling**:
```python
# ✅ CORRECT: Comprehensive error handling
class MatrixPipelineError(Exception):
    """Base exception for Matrix pipeline errors."""
    pass

class ValidationError(MatrixPipelineError):
    """Raised when validation fails."""
    pass

def execute_agent(agent: MatrixAgent, context: ExecutionContext) -> AgentResult:
    """Execute agent with full error handling and logging."""
    try:
        # Validate inputs
        if not isinstance(agent, MatrixAgent):
            raise TypeError(f"Expected MatrixAgent, got {type(agent)}")
            
        # Execute with timeout
        with timeout_context(agent.timeout_seconds):
            result = agent.execute(context)
            
        return result
        
    except ValidationError as e:
        logger.error(f"Validation failed for {agent.agent_name}: {e}")
        raise
    except TimeoutError as e:
        logger.error(f"Agent {agent.agent_name} timed out: {e}")
        return AgentResult.failed(agent.agent_id, f"Timeout: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in {agent.agent_name}: {e}", exc_info=True)
        return AgentResult.failed(agent.agent_id, f"Execution error: {e}")

# ❌ INCORRECT: Poor error handling
def run_agent(agent):
    try:
        return agent.run()  # What if this fails?
    except:  # Too broad exception handling
        print("Error")  # No logging, no context
        return None  # Unclear failure mode
```

#### Documentation Standards

**Class Documentation**:
```python
class BinaryAnalyzer:
    """
    Advanced binary analysis engine for reverse engineering.
    
    Provides comprehensive binary analysis including:
    - Format detection (PE, ELF, Mach-O)
    - Architecture identification (x86, x64, ARM)
    - Compiler detection and optimization analysis
    - Security feature analysis (ASLR, DEP, etc.)
    
    Thread-safe and production-ready with extensive error handling.
    
    Example:
        analyzer = BinaryAnalyzer(config_manager)
        result = analyzer.analyze_binary(Path("target.exe"))
        print(f"Format: {result.format}, Arch: {result.architecture}")
    
    Attributes:
        config: Configuration manager for analysis settings
        supported_formats: List of supported binary formats
        
    Note:
        Requires appropriate permissions for binary file access.
        Some analysis features may require additional tools (objdump, etc.).
    """
```

**Function Documentation**:
```python
def calculate_entropy(data: bytes, block_size: int = 256) -> float:
    """
    Calculate Shannon entropy of binary data for analysis.
    
    Entropy calculation helps identify:
    - Packed/compressed sections (high entropy)
    - Code sections (medium entropy) 
    - Data sections (low entropy)
    - Encrypted content (high entropy)
    
    Args:
        data: Binary data to analyze
        block_size: Size of analysis blocks in bytes (default: 256)
        
    Returns:
        Float between 0.0 (no entropy) and 8.0 (maximum entropy)
        
    Raises:
        ValueError: If data is empty or block_size is invalid
        
    Example:
        entropy = calculate_entropy(binary_data)
        if entropy > 7.5:
            print("Likely packed or encrypted")
    """
```

#### Configuration Management

**Environment Variables**:
```bash
# Required environment variables
GHIDRA_HOME=/path/to/ghidra
JAVA_HOME=/path/to/java
MATRIX_AI_API_KEY=your_api_key_here

# Optional configuration
MATRIX_DEBUG=true
MATRIX_LOG_LEVEL=INFO
MATRIX_PARALLEL_AGENTS=8
```

**PERMANENT BUILD SYSTEM CONFIGURATION**:

🔒 **CRITICAL**: This system uses ONLY Visual Studio 2022 Preview with MSBuild. All build operations are centralized and configured in `build_config.yaml`. NO FALLBACKS are supported.

**Centralized Build System**:
- **Build System Manager**: `src/core/build_system_manager.py` - Single source of truth for all build operations
- **Configuration File**: `build_config.yaml` - Contains all VS2022 Preview paths (NEVER modify at runtime)
- **Enforcement**: All agents use centralized build manager - no direct tool access
- **Tool Detection**: DISABLED - only configured paths in `build_config.yaml` are used

**Fixed Tool Paths** (DO NOT CHANGE):
```bash
# Visual Studio 2022 Preview Compiler (x64)
COMPILER_X64="/mnt/c/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe"

# MSBuild 2022 Preview
MSBUILD_PATH="/mnt/c/Program Files/Microsoft Visual Studio/2022/Preview/MSBuild/Current/Bin/MSBuild.exe"

# VS2022 Preview VC Tools Base
VC_TOOLS_PATH="/mnt/c/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.44.35207"
```

**Agent Build Integration**:
- **Agent 10 (The Machine)**: Primary compilation orchestrator using centralized build system
- **Agent 09 (Commander Locke)**: Generates VS2022-compatible project files with centralized paths
- **All Agents**: Must use `get_build_manager()` for any build operations
- **Validation**: Build system manager validates all tool paths on startup

**ALWAYS REMEMBER**: This location and configuration are PERMANENT. All pipeline and agent operations default to this centralized VS2022 Preview configuration.

**Configuration Files** (YAML preferred):
```yaml
# matrix_config.yaml
pipeline:
  parallel_mode: thread
  batch_size: 4
  timeout_agent: 300
  max_retries: 3

paths:
  default_output_dir: output
  temp_dir: temp
  log_dir: logs
  timestamp_format: "%Y%m%d-%H%M%S"
  
ghidra:
  max_memory: "4G"
  timeout: 600
  
ai_engine:
  provider: langchain
  model: gpt-4
  temperature: 0.1
  max_tokens: 2048
```

#### Performance & Resource Management

**Memory Management**:
```python
# ✅ CORRECT: Proper resource management
class GhidraProcessor:
    def __init__(self, config: ConfigManager):
        self._config = config
        self._process: Optional[subprocess.Popen] = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up Ghidra processes and temporary files."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            finally:
                self._process = None

# Usage
with GhidraProcessor(config) as processor:
    result = processor.analyze_binary(binary_path)
# Automatic cleanup on exit
```

**Async Programming**:
```python
# ✅ CORRECT: Proper async implementation
class AsyncAgentExecutor:
    async def execute_agent_batch(
        self, 
        agents: List[MatrixAgent], 
        context: ExecutionContext
    ) -> Dict[int, AgentResult]:
        """Execute agents concurrently with proper error handling."""
        tasks = []
        
        for agent in agents:
            task = asyncio.create_task(
                self._execute_single_agent(agent, context),
                name=f"Agent_{agent.agent_id}"
            )
            tasks.append(task)
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._config.get_timeout('batch', 600)
            )
            
            return self._process_batch_results(agents, results)
            
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise TimeoutError("Batch execution timed out")
```

#### Testing Standards

**Unit Tests**:
```python
class TestBinaryAnalyzer:
    """Comprehensive test suite for BinaryAnalyzer."""
    
    @pytest.fixture
    def analyzer(self, mock_config):
        """Create analyzer instance with mocked configuration."""
        return BinaryAnalyzer(mock_config)
        
    @pytest.fixture
    def sample_pe_binary(self, tmp_path):
        """Create sample PE binary for testing."""
        binary_path = tmp_path / "test.exe"
        # Create minimal valid PE structure
        binary_path.write_bytes(PE_HEADER + b"test_data")
        return binary_path
        
    def test_analyze_valid_pe_binary(self, analyzer, sample_pe_binary):
        """Test analysis of valid PE binary."""
        result = analyzer.analyze_binary(sample_pe_binary)
        
        assert result.format == BinaryFormat.PE
        assert result.architecture == Architecture.X86_64
        assert result.confidence > 0.9
        assert len(result.sections) > 0
        
    def test_analyze_nonexistent_file(self, analyzer):
        """Test proper error handling for missing files."""
        with pytest.raises(FileNotFoundError, match="Binary not found"):
            analyzer.analyze_binary(Path("nonexistent.exe"))
            
    def test_analyze_invalid_binary(self, analyzer, tmp_path):
        """Test handling of invalid binary format."""
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("This is not a binary")
        
        with pytest.raises(ValidationError, match="Invalid binary format"):
            analyzer.analyze_binary(invalid_file)
```

#### Security Requirements

**Input Validation**:
```python
def validate_file_path(path: Union[str, Path], base_dir: Path) -> Path:
    """
    Validate file path to prevent directory traversal attacks.
    
    Args:
        path: User-provided file path
        base_dir: Base directory to restrict access
        
    Returns:
        Validated Path object within base_dir
        
    Raises:
        SecurityError: If path attempts directory traversal
        ValidationError: If path is invalid
    """
    try:
        path_obj = Path(path).resolve()
        base_dir_resolved = base_dir.resolve()
        
        # Ensure path is within base directory
        if not str(path_obj).startswith(str(base_dir_resolved)):
            raise SecurityError(f"Path traversal detected: {path}")
            
        return path_obj
        
    except (OSError, ValueError) as e:
        raise ValidationError(f"Invalid path: {path}") from e
```

**Logging Security**:
```python
class SecurityAwareLogger:
    """Logger that sanitizes sensitive information."""
    
    SENSITIVE_PATTERNS = [
        r'\b[A-Za-z0-9+/]{40,}\b',  # API keys
        r'\b(?:password|pwd|token)=\S+',  # Credentials
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit cards
    ]
    
    def sanitize_message(self, message: str) -> str:
        """Remove sensitive information from log messages."""
        for pattern in self.SENSITIVE_PATTERNS:
            message = re.sub(pattern, '[REDACTED]', message, flags=re.IGNORECASE)
        return message
        
    def info(self, message: str, *args, **kwargs):
        """Log info message with sanitization."""
        sanitized = self.sanitize_message(message)
        self._logger.info(sanitized, *args, **kwargs)
```

#### Code Review Checklist

**Pre-Commit Requirements**:
- [ ] No hardcoded values or secrets
- [ ] All functions have type hints and docstrings
- [ ] Comprehensive error handling with specific exceptions
- [ ] Input validation for all external data
- [ ] Resource cleanup (files, processes, connections)
- [ ] Unit tests with >90% coverage
- [ ] Security review for input validation
- [ ] Performance analysis for critical paths
- [ ] Memory leak analysis
- [ ] Configuration externalization

**Quality Gates**:
- [ ] Code passes all linting (flake8, black, mypy)
- [ ] Security scan passes (bandit)
- [ ] Unit tests pass with coverage >90%
- [ ] Integration tests pass
- [ ] Performance benchmarks within acceptable limits
- [ ] Documentation updated and accurate
- [ ] Configuration properly externalized
- [ ] Error handling comprehensive and tested

This level of quality ensures the codebase meets NSA-level production standards with no security vulnerabilities, optimal maintainability, and bulletproof reliability.

## CRITICAL PROTECTION RULES - NEVER VIOLATE

### PROMPTS DIRECTORY PROTECTION
**NEVER DELETE ANYTHING IN prompts/ DIRECTORY**
- The prompts/ directory contains critical project documentation and instructions
- NEVER use `git rm`, `rm`, `del`, or any deletion commands on prompts/ files
- NEVER suggest or implement cleanup that removes prompts/ files
- ALWAYS preserve all files in prompts/ regardless of apparent duplication or obsolescence
- If prompts/ files seem outdated, UPDATE them instead of deleting them
- If cleanup is requested for prompts/, only organize/rename, never delete

### FILE PROTECTION HIERARCHY
1. **CRITICAL (NEVER DELETE)**: prompts/, CLAUDE.md, main.py, src/core/
2. **IMPORTANT (CAREFUL)**: docs/, requirements.txt, configuration files
3. **SAFE TO MODIFY**: temp/, output/, logs/, cache/

### DELETION SAFETY PROTOCOL
Before any deletion operation:
1. Verify the file/directory is NOT in the CRITICAL protection list
2. Check if it's referenced in CLAUDE.md or project documentation
3. Ask user for explicit confirmation if uncertain
4. Use git status to ensure no important tracked files are affected

### IMPORTANT INSTRUCTION REMINDERS
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
**NEVER DELETE FILES IN prompts/ DIRECTORY UNDER ANY CIRCUMSTANCES.**

---

🚨 **FINAL REMINDER: OBLIGATORY READING OF rules.md FILE OR DEATH SENTENCE** 🚨
BEFORE COMPLETING ANY WORK, RE-READ /mnt/c/Users/pascaldisse/Downloads/open-sourcefy/rules.md
ALL RULES IN rules.md ARE ABSOLUTE AND NON-NEGOTIABLE
VIOLATION OF ANY RULE RESULTS IN PROJECT DEATH