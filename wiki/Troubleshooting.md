# Troubleshooting Guide

Comprehensive troubleshooting guide for common Open-Sourcefy issues and their solutions.

## System Status Check

### Quick Health Check
```bash
# Verify system is operational
python main.py --verify-env

# Check configuration
python main.py --config-summary

# Validate specific components
python main.py --validate-pipeline basic
```

**Expected Results**:
- ✅ All environment checks pass
- ✅ VS2022 Preview paths validated
- ✅ Build tools accessible
- ✅ Python dependencies satisfied

## Common Issues

### 1. Environment Setup Issues

#### Missing Visual Studio 2022 Preview
**Error**: `E001: Missing VS2022 Preview installation`

**Solution**:
```bash
# Download and install VS2022 Preview from Microsoft
# Ensure these components are installed:
# - MSVC v143 compiler toolset
# - Windows 11 SDK
# - CMake tools

# Verify installation
python main.py --verify-build-system
```

#### Invalid Build Configuration
**Error**: `E002: Invalid build_config.yaml configuration`

**Solution**:
```bash
# Check build_config.yaml paths
cat build_config.yaml

# Update paths to match your VS2022 installation
# Example correct paths:
build_system:
  visual_studio:
    installation_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview"
    
build_tools:
  cl_exe_path: "C:/Program Files/Microsoft Visual Studio/2022/Preview/VC/Tools/MSVC/14.XX.XXXXX/bin/Hostx64/x64/cl.exe"
```

#### Python Dependencies
**Error**: Import errors or missing modules

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify Python version
python --version  # Should be 3.9+

# Check virtual environment
which python
```

### 2. Pipeline Execution Issues

#### Agent Prerequisite Failures
**Error**: `E004: Agent prerequisite validation failure`

**Solution**:
```bash
# Check agent dependencies
python main.py --show-dependencies

# Run agents in dependency order
python main.py --agents 1    # Foundation first
python main.py --agents 2,3,4  # Then dependent agents

# Debug specific agent
python main.py --agents X --debug
```

#### Memory Issues
**Error**: `E003: Insufficient system resources`

**Solution**:
```bash
# Check available memory
free -h  # Linux
wmic OS get TotalVisibleMemorySize /value  # Windows

# Reduce parallel execution
python main.py --max-parallel 2

# Use memory-optimized mode
python main.py --optimize-memory
```

#### Import Table Issues
**Error**: `E005: Import table reconstruction failure`

**Solution**: This is now resolved in current version, but if encountered:
```bash
# Verify Agent 1 (Sentinel) execution
python main.py --agents 1 --debug

# Check binary format support
file input/binary.exe

# Ensure binary is valid PE executable
python main.py --validate-binary input/binary.exe
```

### 3. Compilation Issues

#### Build System Failures
**Symptoms**: Generated code doesn't compile

**Diagnosis**:
```bash
# Check generated source
ls -la output/binary/timestamp/compilation/src/

# Test manual compilation
cd output/binary/timestamp/compilation/
cmake .
make  # or msbuild on Windows
```

**Solutions**:
```bash
# Verify VS2022 configuration
python main.py --verify-build-system

# Check for missing headers
find /usr/include -name "*.h" | grep windows  # Linux
dir "C:\Program Files (x86)\Windows Kits\10\Include" # Windows

# Regenerate build files
python main.py --force-reprocess
```

#### Resource Compilation Failures
**Symptoms**: RC.EXE errors or missing resources

**Solution**:
```bash
# Check RC.EXE path in build_config.yaml
cat build_config.yaml | grep rc_exe_path

# Verify Windows SDK installation
ls "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\rc.exe"

# Test resource compilation manually
rc.exe /r resources/app.rc
```

### 4. Performance Issues

#### Slow Execution
**Symptoms**: Pipeline takes >1 hour to complete

**Diagnosis**:
```bash
# Profile execution
python main.py --profile --benchmark

# Check system resources
top  # Linux
taskmgr  # Windows
```

**Solutions**:
```bash
# Use faster mode (fewer quality checks)
python main.py --fast

# Reduce agent scope
python main.py --agents 1,2,5,9  # Core agents only

# Optimize for available resources
python main.py --optimize-cpu --max-memory 8G
```

#### High Memory Usage
**Symptoms**: System becomes unresponsive, swap usage high

**Solutions**:
```bash
# Monitor memory usage
python main.py --profile --debug

# Use memory-optimized settings
export MATRIX_MEMORY_LIMIT=8G
python main.py --optimize-memory

# Process in smaller chunks
python main.py --chunk-size 1000
```

### 5. Output Quality Issues

#### Poor Decompilation Quality
**Symptoms**: Generated code is incomplete or incorrect

**Diagnosis**:
```bash
# Check quality metrics
cat output/binary/timestamp/reports/quality_report.json

# Review agent performance
cat output/binary/timestamp/reports/agent_summary.json
```

**Solutions**:
```bash
# Use thorough analysis mode
python main.py --thorough

# Enable all quality agents
python main.py --agents 1-16

# Check Ghidra integration
python main.py --verify-ghidra
export GHIDRA_HOME=/path/to/ghidra
```

#### Binary Size Mismatch
**Symptoms**: Generated binary is much smaller/larger than original

**Expected**: ~83% of original size (4.3MB for 5.1MB input)

**Solutions**:
```bash
# Check resource extraction
ls -la output/binary/timestamp/compilation/resources/

# Verify import table reconstruction
grep -r "import" output/binary/timestamp/compilation/

# Review Agent 9 output
cat output/binary/timestamp/logs/agents/agent_09.log
```

### 6. AI Integration Issues

#### Claude API Timeouts
**Symptoms**: "Claude CLI timeout after 10s" errors

**Solutions**:
```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" https://api.anthropic.com/

# Use AI fallback mode
export MATRIX_AI_FALLBACK=true
python main.py --no-ai
```

#### AI Service Unavailable
**Symptoms**: AI-enhanced features fail

**Solutions**:
```bash
# Disable AI features temporarily
python main.py --disable-ai

# Use basic validation only
python main.py --basic-validation

# Check service status
curl -I https://api.anthropic.com/v1/health
```

## Advanced Debugging

### Log Analysis

#### Pipeline Logs
```bash
# Main pipeline log
tail -f output/binary/timestamp/logs/pipeline.log

# Agent-specific logs
ls output/binary/timestamp/logs/agents/
cat output/binary/timestamp/logs/agents/agent_XX.log
```

#### Debug Information
```bash
# Enable comprehensive debug logging
export MATRIX_DEBUG=true
python main.py --debug --verbose

# Check debug output
cat output/binary/timestamp/logs/debug/detailed.log
```

### System Validation

#### Component Testing
```bash
# Test individual components
python main.py --test-ghidra
python main.py --test-build-system
python main.py --test-ai-integration

# Validate installation
python main.py --validate-installation
```

#### Performance Profiling
```bash
# Generate performance report
python main.py --profile --generate-report

# CPU profiling
python -m cProfile main.py > profile.txt

# Memory profiling
python -m memory_profiler main.py
```

## Error Code Reference

### Critical Errors (E001-E005)
- **E001**: ✅ **RESOLVED** - VS2022 Preview installation validated
- **E002**: ✅ **RESOLVED** - build_config.yaml configuration validated
- **E003**: ⚠️ **MONITOR** - System resources (16GB+ RAM recommended)
- **E004**: ✅ **RESOLVED** - Agent prerequisite validation operational
- **E005**: ✅ **RESOLVED** - Import table reconstruction operational

### Warning Codes (W001-W010)
- **W001**: AI service timeout (gracefully handled)
- **W002**: Ghidra integration unavailable (basic decompilation used)
- **W003**: Resource extraction incomplete (partial success)
- **W004**: Compilation warnings (code compiles with warnings)
- **W005**: Performance degradation (slower than expected)

### Info Codes (I001-I020)
- **I001**: Pipeline execution started
- **I002**: Agent batch completed successfully
- **I003**: Quality threshold exceeded
- **I004**: Compilation successful
- **I005**: All agents completed successfully

## Platform-Specific Issues

### Windows Issues
- **Path Length Limits**: Use shorter output paths
- **Permission Issues**: Run as Administrator if needed
- **Antivirus Interference**: Add exception for Open-Sourcefy directory

### Linux/WSL Issues
- **Windows Tool Emulation**: Some tools run through Wine
- **Path Translation**: Windows paths may need adjustment
- **Limited Compilation**: Some Windows-specific features unavailable

### macOS Issues
- **Tool Availability**: Some Windows tools not available
- **Case Sensitivity**: File system case sensitivity differences
- **Permission Model**: Different permission requirements

## Getting Additional Help

### Documentation Resources
- **[[User Guide|User-Guide]]**: Complete usage documentation
- **[[Architecture Overview|Architecture-Overview]]**: System design details
- **[[Agent Documentation|Agent-Documentation]]**: Individual agent specifications

### Support Channels
- **GitHub Issues**: [Report bugs and request features](https://github.com/pascaldisse/open-sourcefy/issues)
- **Documentation**: Search this wiki for specific topics
- **Log Analysis**: Use log files for detailed debugging information

### Diagnostic Data Collection

When reporting issues, include:
```bash
# System information
python main.py --system-info

# Configuration summary
python main.py --config-summary

# Recent logs
tar -czf debug-info.tar.gz output/*/logs/

# Version information
python main.py --version
git log --oneline -5
```

---

**Related**: [[User Guide|User-Guide]] - Usage documentation  
**Support**: [GitHub Issues](https://github.com/pascaldisse/open-sourcefy/issues) - Report problems