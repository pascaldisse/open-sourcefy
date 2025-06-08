# Pipeline Compilation and Error Fixing Prompt

## Objective
Fix compilation issues, resolve pipeline execution errors, and ensure the entire open-sourcefy system runs smoothly from binary input to compilable source code output.

## CRITICAL OUTPUT REQUIREMENT
⚠️ **ALL OUTPUT MUST GO TO `/output` DIRECTORY ONLY** ⚠️
- NO files should be created outside of `/output/` or its subdirectories
- NO build artifacts, logs, or temp files in project root or system directories
- ALL compilation testing, build files, and results MUST be in `/output/`
- Use structured paths: `/output/compilation/`, `/output/logs/`, `/output/temp/`
- Validate all build processes respect the `/output/` boundary

## Problem Categories

### 1. Pipeline Execution Errors
Issues that prevent the agent pipeline from completing successfully.

### 2. Agent Integration Failures  
Problems with agent dependencies, context passing, and result coordination.

### 3. Build System Failures
Issues with generating and executing build files (CMake, MSBuild, Makefile).

### 4. Cross-Platform Compatibility
Platform-specific issues that break functionality on different operating systems.

### 5. Performance and Memory Issues
Problems that cause timeouts, memory exhaustion, or excessive resource usage.

## Systematic Debugging Approach

### Phase 1: Pipeline Health Check

#### 1.1 Agent Dependency Validation
```python
# Verify agent dependency graph is acyclic and complete
def validate_agent_dependencies():
    agents = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    dependency_map = {
        1: [],           # BinaryDiscovery
        2: [1],          # ArchitectureAnalysis  
        3: [1, 2],       # SmartErrorPatternMatching
        4: [1, 2, 3],    # BasicDecompiler
        5: [1],          # BinaryStructureAnalyzer
        6: [4, 5],       # OptimizationMatcher
        7: [4, 5],       # AdvancedDecompiler
        8: [4, 7],       # BinaryDiffAnalyzer
        9: [4, 7],       # AdvancedAssemblyAnalyzer
        10: [5],         # ResourceReconstructor
        11: [7, 8, 9, 10], # GlobalReconstructor
        12: [11],        # CompilationOrchestrator
        13: [12],        # FinalValidator
        14: [7],         # AdvancedGhidra
        15: [1, 5]       # MetadataAnalysis
    }
    # Check for circular dependencies
    # Validate all dependencies exist
    # Ensure topological ordering is possible
```

#### 1.2 Context Validation
```python
# Ensure context data flows correctly between agents
def validate_context_flow():
    required_context_keys = {
        'binary_path': 'Path to target binary',
        'output_dir': 'Output directory path', 
        'output_paths': 'Structured output paths',
        'agent_results': 'Previous agent results',
        'global_data': 'Shared analysis data'
    }
    # Verify context structure
    # Check data consistency
    # Validate type annotations
```

### Phase 2: Common Error Patterns

#### 2.1 Import and Module Issues
**Symptoms**: `ModuleNotFoundError`, `ImportError`

**Common Causes**:
```python
# Relative import issues
from ..agent_base import BaseAgent  # May fail in certain contexts

# Missing dependencies
import pefile  # Library not installed

# Path resolution issues  
sys.path.append(os.path.dirname(__file__))
```

**Solutions**:
```python
# Use absolute imports where possible
from src.core.agent_base import BaseAgent

# Add proper error handling for optional dependencies
try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False
    print("Warning: pefile not available, PE analysis disabled")

# Fix path resolution
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ENSURE OUTPUT DIRECTORY COMPLIANCE
def validate_output_path(path):
    """Ensure path is under /output/ directory"""
    output_root = Path("output").resolve()
    path_resolved = Path(path).resolve()
    if not str(path_resolved).startswith(str(output_root)):
        raise ValueError(f"Path {path} is not under /output/ directory")
    return path_resolved
```

#### 2.2 File Path and Directory Issues
**Symptoms**: `FileNotFoundError`, `PermissionError`, path-related crashes

**Common Causes**:
```python
# Hardcoded paths
ghidra_path = "/mnt/c/Users/..."  # Breaks on other systems

# Relative path assumptions
output_file = "results/analysis.json"  # May not exist

# Cross-platform path issues
path = "output\\ghidra\\analysis"  # Windows-only
```

**Solutions**:
```python
# Use Path objects for cross-platform compatibility
from pathlib import Path

# Relative to project root
project_root = Path(__file__).parent.parent.parent
ghidra_path = project_root / "ghidra"

# ENSURE OUTPUT DIRECTORIES ONLY - All output under /output/
output_root = Path("output").resolve()
output_dir = output_root / "agents" / "current_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

# VALIDATE: Ensure output_dir is under /output/
if not str(output_dir.resolve()).startswith(str(output_root)):
    raise ValueError(f"Output directory {output_dir} is not under /output/")

# Use forward slashes for cross-platform paths
path_str = str(path).replace('\\', '/')
```

#### 2.3 Ghidra Integration Issues
**Symptoms**: Ghidra scripts fail, headless mode crashes, timeout errors

**Common Causes**:
```python
# GHIDRA_HOME not set or incorrect
ghidra_home = os.environ.get('GHIDRA_HOME')  # May be None

# Script generation errors
script_content = f"// Script with {unclosed_brace"  # Syntax error

# Timeout issues  
subprocess.run(cmd, timeout=30)  # Too short for large binaries
```

**Solutions**:
```python
# Robust Ghidra path detection
def find_ghidra_installation():
    # Check environment variable
    if 'GHIDRA_HOME' in os.environ:
        return os.environ['GHIDRA_HOME']
    
    # Check project directory
    project_ghidra = Path(__file__).parent.parent.parent / "ghidra"
    if project_ghidra.exists():
        return str(project_ghidra)
    
    # Check common installation paths
    common_paths = [
        "/opt/ghidra",
        "C:\\ghidra", 
        "/Applications/ghidra"
    ]
    for path in common_paths:
        if Path(path).exists():
            return path
    
    raise FileNotFoundError("Ghidra installation not found")

# Proper script generation with escaping
def generate_ghidra_script(template, **kwargs):
    # Escape special characters
    for key, value in kwargs.items():
        if isinstance(value, str):
            kwargs[key] = value.replace('\\', '\\\\').replace('"', '\\"')
    
    return template.format(**kwargs)

# Adaptive timeouts based on binary size
def calculate_timeout(binary_path):
    size_mb = Path(binary_path).stat().st_size / (1024 * 1024)
    return max(60, int(size_mb * 10))  # 10 seconds per MB, minimum 60s
```

#### 2.4 Agent Result Handling
**Symptoms**: `KeyError`, `AttributeError`, inconsistent agent states

**Common Causes**:
```python
# Missing result validation
agent_result = context['agent_results'][7]  # May not exist
data = agent_result.data['functions']  # May be None

# Status checking inconsistency
if agent_result.status == "completed":  # String vs Enum
    
# Result format assumptions
functions = result.data.get('decompiled_functions', {})  # May be list
```

**Solutions**:
```python
# Robust result access
def get_agent_result(context, agent_id, required=True):
    agent_results = context.get('agent_results', {})
    result = agent_results.get(agent_id)
    
    if result is None:
        if required:
            raise ValueError(f"Agent {agent_id} result not found")
        return None
    
    if result.status != AgentStatus.COMPLETED:
        if required:
            raise ValueError(f"Agent {agent_id} did not complete successfully")
        return None
    
    return result

# Type checking and validation
def validate_result_data(result, expected_keys):
    if not isinstance(result.data, dict):
        raise TypeError(f"Expected dict, got {type(result.data)}")
    
    missing_keys = set(expected_keys) - set(result.data.keys())
    if missing_keys:
        raise KeyError(f"Missing required keys: {missing_keys}")
    
    return result.data
```

### Phase 3: Build System Fixes

#### 3.1 CMake Generation Issues
**Common Problems**:
```cmake
# Missing CMakeLists.txt structure
cmake_minimum_required(VERSION 3.0)  # Too old
project(reconstructed)                # Missing language

# Missing source files
add_executable(reconstructed)         # No sources specified

# Missing libraries
target_link_libraries(reconstructed)  # No libraries specified
```

**Solutions**:
```cmake
# Modern CMake with proper structure
cmake_minimum_required(VERSION 3.16)
project(ReconstructedBinary LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(Threads REQUIRED)

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.c" "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h" "include/*.hpp")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Link libraries
target_link_libraries(${PROJECT_NAME} Threads::Threads)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(${PROJECT_NAME} kernel32 user32 gdi32)
elseif(UNIX)
    target_link_libraries(${PROJECT_NAME} m dl)
endif()
```

#### 3.2 MSBuild Integration
**Common Issues**:
```xml
<!-- Missing project structure -->
<Project>
  <ItemGroup>
    <ClCompile Include="main.c" />  <!-- File may not exist -->
  </ItemGroup>
</Project>

<!-- Missing toolset specification -->
<PropertyGroup>
  <TargetFramework>net6.0</TargetFramework>  <!-- Wrong for C/C++ -->
</PropertyGroup>
```

**Solutions**:
```xml
<?xml version="1.0" encoding="utf-8"?>
<Project DefaultTargets="Build" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  
  <!-- Configuration -->
  <ItemGroup Label="ProjectConfigurations">
    <ProjectConfiguration Include="Debug|x64">
      <Configuration>Debug</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
    <ProjectConfiguration Include="Release|x64">
      <Configuration>Release</Configuration>
      <Platform>x64</Platform>
    </ProjectConfiguration>
  </ItemGroup>
  
  <!-- Global Properties -->
  <PropertyGroup Label="Globals">
    <ProjectGuid>{GENERATED-GUID}</ProjectGuid>
    <Keyword>Win32Proj</Keyword>
    <RootNamespace>ReconstructedBinary</RootNamespace>
    <WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>
  </PropertyGroup>
  
  <!-- Import Default Props -->
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.Default.props" />
  
  <!-- Configuration Properties -->
  <PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Debug|x64'" Label="Configuration">
    <ConfigurationType>Application</ConfigurationType>
    <UseDebugLibraries>true</UseDebugLibraries>
    <PlatformToolset>v143</PlatformToolset>
    <CharacterSet>Unicode</CharacterSet>
  </PropertyGroup>
  
  <!-- Import Cpp Props -->
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.props" />
  
  <!-- Source Files -->
  <ItemGroup>
    <!-- Dynamically generated from analysis -->
  </ItemGroup>
  
  <!-- Import Targets -->
  <Import Project="$(VCTargetsPath)\Microsoft.Cpp.targets" />
  
</Project>
```

### Phase 4: Error Recovery and Resilience

#### 4.1 Graceful Degradation
```python
def execute_agent_with_fallback(agent, context):
    """Execute agent with fallback strategies"""
    try:
        # Primary execution
        return agent.execute(context)
    except NotImplementedError as e:
        # Expected for incomplete features
        return AgentResult(
            agent_id=agent.agent_id,
            status=AgentStatus.SKIPPED,
            data={},
            error_message=f"Feature not implemented: {str(e)}"
        )
    except Exception as e:
        # Unexpected errors - try recovery
        try:
            # Attempt minimal functionality
            return agent.execute_minimal(context)
        except:
            # Complete failure
            return AgentResult(
                agent_id=agent.agent_id, 
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Agent failed: {str(e)}"
            )
```

#### 4.2 Resource Management
```python
import resource
import psutil
from contextlib import contextmanager

@contextmanager
def resource_monitor(max_memory_mb=2048, max_time_seconds=300):
    """Monitor resource usage and enforce limits"""
    start_time = time.time()
    process = psutil.Process()
    
    try:
        yield
    finally:
        # Check memory usage
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > max_memory_mb:
            warnings.warn(f"High memory usage: {memory_mb:.1f}MB")
        
        # Check execution time  
        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            warnings.warn(f"Long execution time: {elapsed:.1f}s")
```

#### 4.3 Comprehensive Testing Framework
```python
def run_pipeline_health_check(binary_path, output_dir):
    """Comprehensive pipeline testing"""
    
    # Test 1: Basic pipeline execution
    try:
        result = run_pipeline(binary_path, output_dir)
        assert result.overall_status == PipelineStatus.COMPLETED
        print("✅ Pipeline execution successful")
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False
    
    # Test 2: Output validation
    try:
        validate_output_structure(output_dir)
        print("✅ Output structure valid")
    except Exception as e:
        print(f"❌ Output validation failed: {e}")
        return False
    
    # Test 3: Generated code compilation
    try:
        compile_result = test_compilation(output_dir)
        if compile_result.success:
            print("✅ Generated code compiles")
        else:
            print(f"⚠️ Compilation issues: {compile_result.errors}")
    except Exception as e:
        print(f"❌ Compilation testing failed: {e}")
    
    # Test 4: Cross-platform compatibility
    if test_cross_platform_paths(output_dir):
        print("✅ Cross-platform paths valid")
    else:
        print("⚠️ Cross-platform path issues detected")
    
    return True
```

## Debugging Tools and Utilities

### 1. Pipeline Visualization
```python
def visualize_pipeline_execution(results):
    """Create visual representation of pipeline execution"""
    import matplotlib.pyplot as plt
    
    agents = list(range(1, 16))
    statuses = [results.get(i, {}).get('status', 'not_run') for i in agents]
    
    colors = {
        'completed': 'green',
        'failed': 'red', 
        'skipped': 'yellow',
        'not_run': 'gray'
    }
    
    plt.figure(figsize=(15, 6))
    plt.bar(agents, [1]*15, color=[colors[s] for s in statuses])
    plt.xlabel('Agent ID')
    plt.ylabel('Status')
    plt.title('Pipeline Execution Status')
    plt.xticks(agents)
    plt.show()
```

### 2. Error Analysis Dashboard
```python
def analyze_error_patterns(log_files):
    """Analyze common error patterns across runs"""
    error_counts = {}
    
    for log_file in log_files:
        with open(log_file) as f:
            for line in f:
                if 'ERROR' in line or 'Exception' in line:
                    # Extract error type
                    error_type = extract_error_type(line)
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
    
    # Sort by frequency
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Most Common Errors:")
    for error, count in sorted_errors[:10]:
        print(f"  {error}: {count} occurrences")
    
    return sorted_errors
```

### 3. Performance Profiling
```python
import cProfile
import pstats

def profile_pipeline_execution(binary_path, output_dir):
    """Profile pipeline execution for performance bottlenecks"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        run_pipeline(binary_path, output_dir)
    finally:
        profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    print("Top 20 Time-Consuming Functions:")
    stats.print_stats(20)
    
    # Save detailed report
    stats.dump_stats('pipeline_profile.prof')
```

## Success Criteria

### Pipeline Reliability
- [ ] 95%+ successful pipeline completion rate
- [ ] <5% agent failure rate for common binaries
- [ ] Graceful degradation when agents fail
- [ ] Comprehensive error logging and reporting

### Build System Integration
- [ ] Generated CMake files compile successfully (>80% success rate)
- [ ] MSBuild integration works on Windows
- [ ] Cross-platform build compatibility
- [ ] Automatic dependency detection and linking

### Performance Standards
- [ ] <5 minutes total analysis time for typical binaries (<10MB)
- [ ] <2GB memory usage for large binaries
- [ ] Parallel agent execution where possible
- [ ] Efficient resource cleanup

### Code Quality
- [ ] All import errors resolved
- [ ] No hardcoded paths or platform assumptions
- [ ] Proper exception handling throughout
- [ ] Comprehensive test coverage (>80%)

## Troubleshooting Quick Reference

### Common Error Solutions
```bash
# ModuleNotFoundError: Install missing dependencies
pip install -r requirements.txt

# GHIDRA_HOME not found: Set environment variable  
export GHIDRA_HOME=/path/to/ghidra

# Permission errors: Fix directory permissions
chmod -R 755 output/
chown -R $USER:$USER output/

# Memory issues: Increase system limits
ulimit -m 4194304  # 4GB memory limit

# Cross-platform path issues: Use pathlib
python -c "from pathlib import Path; print(Path('output/test').resolve())"
```

### Debug Mode Execution
```bash
# Run with verbose logging (OUTPUT TO /output/ ONLY)
python main.py target.exe --output-dir output/debug_test --debug --log-level DEBUG

# Single agent execution for testing (OUTPUT TO /output/ ONLY)
python -m src.core.agents.agent01_binary_discovery target.exe --output-dir output/agent_test --debug

# Pipeline health check (OUTPUT TO /output/ ONLY)
python -m src.core.testing.pipeline_health_check target.exe --output-dir output/health_check

# VERIFY: No files created outside /output/
find . -maxdepth 2 -name "*.log" -o -name "*.tmp" -o -name "*.json" | grep -v "^\./output/" || echo "✅ No files outside /output/"
```