# Open-Sourcefy Complete User Guide

**Advanced AI-Powered Binary Decompilation & Reconstruction System**

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Usage Guide](#usage-guide)
4. [Agent System](#agent-system)
5. [Pipeline Architecture](#pipeline-architecture)
6. [Configuration](#configuration)
7. [Output Structure](#output-structure)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Features](#advanced-features)
10. [Performance Tuning](#performance-tuning)

## Quick Start

### Basic Usage
```bash
# Analyze a binary with all agents
python main.py launcher.exe

# Analyze with custom output directory
python main.py launcher.exe --output-dir my_analysis

# Verify environment before analysis
python main.py --verify-env
```

### Pipeline Stages
```bash
# Decompilation only (Agents 1,2,5,7)
python main.py launcher.exe --decompile-only

# Analysis only (Agents 1,2,3,4,5,8,9)
python main.py launcher.exe --analyze-only

# Compilation only (Agents 6,11,12)
python main.py launcher.exe --compile-only
```

## Installation

### Prerequisites

- **Python 3.8+** with pip package manager
- **Java 17+** (for Ghidra integration)
- **Microsoft Visual C++ Build Tools** (for compilation testing)
- **8GB+ RAM** (recommended for AI-enhanced analysis)
- **WSL/Linux environment** (for optimal performance)

### Setup Steps

1. **Clone and setup:**
   ```bash
   git clone https://github.com/yourusername/open-sourcefy.git
   cd open-sourcefy
   pip install -r requirements.txt
   ```

2. **Environment verification:**
   ```bash
   python main.py --verify-env
   ```

3. **Test with sample binary:**
   ```bash
   python main.py launcher.exe --timeout 300
   ```

## Usage Guide

### Command Line Interface

#### Basic Commands
```bash
# Full pipeline execution
python main.py <binary_path> [options]

# Environment verification
python main.py --verify-env [--detailed]

# Help and usage
python main.py --help
```

#### Agent Selection
```bash
# Run specific agents
python main.py launcher.exe --agents 1,3,5,7

# Run agent range
python main.py launcher.exe --agents 1-5

# Run single agent
python main.py launcher.exe --agent 7
```

#### Pipeline Stages
```bash
# Decompilation pipeline
python main.py launcher.exe --decompile-only

# Analysis pipeline
python main.py launcher.exe --analyze-only

# Compilation pipeline
python main.py launcher.exe --compile-only

# Validation pipeline
python main.py launcher.exe --validate-only
```

#### Parallel Processing
```bash
# Enable parallel execution
python main.py launcher.exe --parallel-mode process --batch-size 4

# Thread-based parallelism
python main.py launcher.exe --parallel-mode thread --batch-size 6

# Sequential execution (default)
python main.py launcher.exe --parallel-mode sequential
```

### Advanced Options

#### Timeout and Retry Configuration
```bash
# Custom timeout (seconds)
python main.py launcher.exe --timeout 1800

# Maximum retries
python main.py launcher.exe --max-retries 5

# Combined configuration
python main.py launcher.exe --timeout 1800 --max-retries 3
```

#### AI Enhancement Features
```bash
# Enable AI naming
python main.py launcher.exe --enable-ai-naming

# Enable quality assessment
python main.py launcher.exe --enable-quality-assessment

# Full AI enhancement
python main.py launcher.exe --enable-ai-naming --enable-quality-assessment
```

## Agent System

### Agent Overview

Open-Sourcefy uses 15+ specialized agents organized into functional groups:

#### Discovery & Analysis (Agents 1-5)
- **Agent 1**: Binary Discovery - Initial binary analysis and metadata extraction
- **Agent 2**: Architecture Analysis - x86 architecture and calling convention analysis
- **Agent 3**: Smart Error Pattern Matching - AI-powered error detection
- **Agent 4**: Optimization Detection - Compiler optimization identification
- **Agent 5**: Binary Structure Analyzer - PE32 structure parsing

#### Processing & Decompilation (Agents 6-10)
- **Agent 6**: Optimization Matcher - Pattern matching and reconstruction
- **Agent 7**: Advanced Decompiler - Ghidra integration and function decompilation
- **Agent 8**: Binary Diff Analyzer - Binary difference analysis
- **Agent 9**: Advanced Assembly Analyzer - Deep assembly analysis
- **Agent 10**: Resource Reconstructor - Resource section reconstruction

#### Enhancement & Integration (Agents 11-15)
- **Agent 11**: Global Reconstructor - AI-powered code enhancement
- **Agent 12**: Compilation Orchestrator - Build system integration
- **Agent 13**: Final Validator - Binary reproduction validation
- **Agent 14**: Advanced Ghidra - Enhanced Ghidra capabilities
- **Agent 15**: Metadata Analysis - Advanced metadata extraction

#### Extension Agents (Agents 16-20)
- **Agent 16**: Dynamic Bridge - Runtime analysis bridge
- **Agent 18**: Advanced Build Systems - Complex build environment support
- **Agent 19**: Binary Comparison - Advanced binary comparison
- **Agent 20**: Auto Testing - Automated testing frameworks

### Agent Execution Modes

#### Individual Agent Execution
```bash
# Run specific agent
python main.py launcher.exe --agent 7

# Agent with custom parameters
python main.py launcher.exe --agent 12 --timeout 600
```

#### Agent Groups
```bash
# Discovery agents
python main.py launcher.exe --agents 1-5

# Decompilation agents
python main.py launcher.exe --agents 6,7,9

# Validation agents
python main.py launcher.exe --agents 8,12,13
```

## Pipeline Architecture

### Four-Stage Pipeline

1. **Decompile Stage**
   - Agents: 1, 2, 5, 7
   - Purpose: Binary discovery, architecture analysis, structure parsing, Ghidra decompilation
   - Output: Decompiled C code, binary metadata, architecture analysis

2. **Analyze Stage**
   - Agents: 1, 2, 3, 4, 5, 8, 9
   - Purpose: Comprehensive analysis, pattern detection, optimization identification
   - Output: Analysis reports, pattern matches, optimization metadata

3. **Compile Stage**
   - Agents: 6, 11, 12
   - Purpose: Code enhancement, build integration, compilation testing
   - Output: Enhanced source code, build files, compilation reports

4. **Validate Stage**
   - Agents: 8, 12, 13
   - Purpose: Binary comparison, testing, final validation
   - Output: Validation reports, binary diffs, quality metrics

### Execution Flow

```
Binary Input → Decompile → Analyze → Compile → Validate → Source Output
     ↓           ↓          ↓         ↓         ↓
   Metadata → Functions → Patterns → Build → Validation
```

## Configuration

### Environment Variables

```bash
# Development mode
export OPEN_SOURCEFY_DEV=1

# Custom Ghidra path
export GHIDRA_INSTALL_DIR=/path/to/ghidra

# Performance tuning
export OPEN_SOURCEFY_MAX_MEMORY=8G
```

### Configuration Files

#### Agent Configuration
```json
{
  "agents": {
    "timeout": 600,
    "max_retries": 3,
    "parallel_mode": "process",
    "batch_size": 4
  }
}
```

#### AI Enhancement Configuration
```json
{
  "ai_enhancement": {
    "enable_naming": true,
    "enable_quality_assessment": true,
    "confidence_threshold": 0.8
  }
}
```

## Output Structure

### Directory Organization

```
output/
├── agents/          # Agent-specific analysis outputs
├── ghidra/          # Ghidra decompilation results and projects
├── compilation/     # Compilation artifacts and generated source
├── reports/         # Pipeline execution reports and summaries
├── logs/            # Execution logs and debug information
└── temp/            # Temporary files (auto-cleaned)
```

### Key Output Files

- **`reports/pipeline_report.json`** - Complete pipeline execution report
- **`compilation/src/main.c`** - Primary reconstructed source file
- **`compilation/build.ps1`** - Build script for compilation
- **`ghidra/analysis.log`** - Ghidra analysis log
- **`agents/agent_N_output.json`** - Individual agent results

## Troubleshooting

### Common Issues

#### Environment Issues
```bash
# Java not found
sudo apt install openjdk-17-jdk

# Ghidra path issues
python main.py --verify-env --detailed

# Permission issues
chmod +x ghidra/ghidraRun
```

#### Memory Issues
```bash
# Reduce batch size
python main.py launcher.exe --batch-size 2

# Increase timeout
python main.py launcher.exe --timeout 3600

# Sequential processing
python main.py launcher.exe --parallel-mode sequential
```

#### Agent Failures
```bash
# Retry with increased timeout
python main.py launcher.exe --timeout 1800 --max-retries 5

# Run specific failing agent
python main.py launcher.exe --agent 7 --timeout 1200

# Skip problematic agents
python main.py launcher.exe --agents 1,2,3,5,6
```

### Debug Mode

```bash
# Enable verbose logging
python main.py launcher.exe --debug

# Save debug information
python main.py launcher.exe --debug --output-dir debug_analysis
```

## Advanced Features

### AI Enhancement

#### Pattern Recognition
- Multi-feature extraction with 80%+ confidence scores
- Advanced assembly pattern matching
- Optimization pattern detection

#### Intelligent Naming
- Semantic-based function naming
- Variable name inference
- Type inference and annotation

#### Quality Assessment
- Code complexity analysis
- Maintainability scoring
- Performance impact assessment

### Parallel Processing

#### Process-based Parallelism
```bash
# Multi-process execution
python main.py launcher.exe --parallel-mode process --batch-size 4
```

#### Thread-based Parallelism
```bash
# Multi-threaded execution
python main.py launcher.exe --parallel-mode thread --batch-size 6
```

#### Adaptive Batch Sizing
- Automatic batch size optimization
- Memory-aware processing
- Load balancing across agents

### Ghidra Integration

#### Headless Analysis
- Automated Ghidra project creation
- Script-based decompilation
- Function signature analysis

#### Advanced Decompilation
- Control flow analysis
- Data flow analysis
- Cross-reference generation

## Performance Tuning

### Memory Optimization

```bash
# Conservative memory usage
python main.py launcher.exe --batch-size 2 --parallel-mode sequential

# Optimized for 8GB+ systems
python main.py launcher.exe --batch-size 4 --parallel-mode process

# High-memory systems (16GB+)
python main.py launcher.exe --batch-size 8 --parallel-mode process
```

### Execution Time Optimization

```bash
# Quick analysis (reduced accuracy)
python main.py launcher.exe --agents 1,2,5,7 --timeout 300

# Balanced analysis
python main.py launcher.exe --timeout 900

# Comprehensive analysis
python main.py launcher.exe --timeout 1800 --max-retries 3
```

### Quality vs Speed Trade-offs

| Mode | Agents | Time | Accuracy | Use Case |
|------|--------|------|----------|----------|
| Quick | 1,2,5,7 | 5-15 min | 70-80% | Initial assessment |
| Standard | All | 30-60 min | 90-95% | Development |
| Comprehensive | All + Retries | 60-120 min | 95-99% | Production |

## Target: Matrix Online Launcher

### Specifications
- **File Size**: 5.3MB
- **Architecture**: x86 PE32
- **Compiler**: Microsoft Visual C++ 7.1 (MSVC .NET 2003)
- **Functions**: 2,099+ identified functions
- **Target Accuracy**: 99%+ binary reproduction

### Optimized Command
```bash
# Optimized for Matrix Online launcher
python main.py launcher.exe --timeout 1800 --max-retries 3 --batch-size 4
```

### Current Development Status
- **Overall Quality**: 66.5% (Phase 3 - Active Development)
- **Processing Time**: Variable (optimization in progress)
- **Memory Usage**: 4-8GB peak (monitoring and optimization ongoing)
- **Agent Success Rate**: 16/16 agents functional

---

**Open-Sourcefy Complete User Guide v1.0**  
**Last Updated**: 2025-06-08  
**System Version**: 1.0.0