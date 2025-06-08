# Automation Scripts

This directory contains automation scripts that handle tasks that don't require LLM intervention, as referenced in the project prompts.

## Available Scripts

### 🔧 `environment_validator.py`
**Purpose**: Validates the development environment setup for open-sourcefy project.

**Features**:
- Python version checking (3.8+ required)
- Required and optional package validation  
- Ghidra installation detection and validation
- Java installation checking for Ghidra
- Build tools detection (CMake, MSBuild, GCC, etc.)
- Project structure validation
- Output directory compliance checking

**Usage**:
```bash
./environment_validator.py                    # Basic validation
./environment_validator.py --json             # JSON output for scripting
./environment_validator.py --setup-help       # Get setup instructions
./environment_validator.py --quiet            # Suppress verbose output
```

**Examples**:
```bash
# Check if environment is ready for development
./environment_validator.py

# Get specific setup instructions based on what's missing
./environment_validator.py --setup-help

# Use in scripts with JSON output
./environment_validator.py --json | jq '.overall.validation_passed'
```

---

### 📁 `file_operations.py`
**Purpose**: Automates common file operations and directory management tasks.

**Features**:
- Output directory structure creation (agents/, ghidra/, compilation/, etc.)
- File metadata extraction (hash, size, timestamps)
- Temporary file cleanup with age-based filtering
- File pattern searching and extension filtering
- Large file detection and reporting
- Directory analysis and reporting
- Output path validation (ensures all files stay under `/output/`)

**Usage**:
```bash
./file_operations.py create-structure <output_dir>           # Create directory structure
./file_operations.py clean-temp <temp_dir> [--max-age 24]   # Clean old temp files
./file_operations.py directory-report <dir> <output_file>   # Generate directory report
./file_operations.py find-files <directory> <pattern>       # Find files by pattern
```

**Examples**:
```bash
# Create standard output structure for a new analysis session
./file_operations.py create-structure output/20250608_analysis

# Clean up temporary files older than 24 hours
./file_operations.py clean-temp output/temp --max-age 24

# Generate comprehensive directory analysis report
./file_operations.py directory-report src/ src_analysis.json

# Find all Python files in src directory
./file_operations.py find-files src "*.py"
```

---

### 🏗️ `build_system_automation.py`
**Purpose**: Automates build system generation and compilation testing.

**Features**:
- CMake project file generation with modern CMake practices
- MSBuild project (.vcxproj) and solution (.sln) file generation
- Makefile generation for Unix systems
- Cross-platform build system support
- Automated compilation testing with timeout handling
- Build result analysis and reporting
- Output directory validation (all builds under `/output/`)

**Usage**:
```bash
./build_system_automation.py --output-dir <dir> cmake --project-name <name> --sources <files> [--includes <dirs>] [--libraries <libs>]
./build_system_automation.py --output-dir <dir> msbuild --project-name <name> --sources <files> [--includes <dirs>]
./build_system_automation.py --output-dir <dir> makefile --project-name <name> --sources <files> [--includes <dirs>] [--libraries <libs>]
./build_system_automation.py --output-dir <dir> test [--build-system auto|cmake|msbuild|make]
```

**Examples**:
```bash
# Generate CMake project for reconstructed code
./build_system_automation.py --output-dir output/test cmake \
    --project-name ReconstructedBinary \
    --sources src/main.c src/utils.c \
    --includes include/ \
    --libraries pthread

# Generate MSBuild project for Windows
./build_system_automation.py --output-dir output/test msbuild \
    --project-name ReconstructedBinary \
    --sources src/main.c src/utils.c \
    --includes include/

# Test compilation using auto-detected build system
./build_system_automation.py --output-dir output/test test --build-system auto
```

---

### 🚀 `pipeline_helper.py`
**Purpose**: Provides high-level pipeline execution and management utilities.

**Features**:
- Environment validation using environment_validator.py
- Automated output directory structure creation
- Pipeline execution with configurable modes and agents
- Result analysis and reporting
- Compilation testing integration
- Old output cleanup
- Comprehensive pipeline reporting

**Usage**:
```bash
./pipeline_helper.py validate-env                                        # Validate environment
./pipeline_helper.py run <binary> [--output-dir <dir>] [--agents <list>] [--mode <mode>] [--debug]
./pipeline_helper.py analyze <output_dir>                               # Analyze pipeline results  
./pipeline_helper.py test-compile <output_dir> [--build-system <system>] # Test compilation
./pipeline_helper.py cleanup [--max-age <days>]                         # Clean old outputs
./pipeline_helper.py report <output_dir> [--report-file <file>]         # Generate report
```

**Examples**:
```bash
# Complete automated workflow
./pipeline_helper.py validate-env
./pipeline_helper.py run launcher.exe --mode full --debug
./pipeline_helper.py analyze output/20250608_143052
./pipeline_helper.py test-compile output/20250608_143052
./pipeline_helper.py report output/20250608_143052

# Run specific agents only
./pipeline_helper.py run launcher.exe --agents 1,2,5,7 --mode analyze

# Clean up outputs older than 3 days
./pipeline_helper.py cleanup --max-age 3
```

---

### 🔧 `make_executable.sh`
**Purpose**: Makes all Python scripts in the scripts directory executable.

**Usage**:
```bash
./make_executable.sh
```

This script sets executable permissions on all `.py` files in the scripts directory and displays usage information for each script.

## Integration with Prompts

These automation scripts are referenced throughout the project prompts to replace manual tasks:

- **agent_cleanup.md**: Uses `environment_validator.py` and `file_operations.py` for environment validation and file management
- **full_pipeline_compilation.md**: Uses `pipeline_helper.py` for automated pipeline execution and `build_system_automation.py` for compilation testing
- **implementation_fixing.md**: Uses all scripts for comprehensive automation during implementation
- **pipeline_compilation.md**: Uses `pipeline_helper.py` and `environment_validator.py` for debugging and validation

## Output Directory Compliance

All scripts strictly enforce the `/output/` directory requirement:

- ✅ **All file operations are validated** to ensure they occur under `/output/` or its subdirectories
- ✅ **Build systems are configured** to generate files only in `/output/compilation/`
- ✅ **Temporary files are managed** in `/output/temp/` with automatic cleanup
- ✅ **Reports and logs** are saved to appropriate `/output/` subdirectories

## Error Handling

All scripts include comprehensive error handling:

- **Timeout protection** for long-running operations
- **Graceful degradation** when optional dependencies are missing
- **Clear error messages** with actionable guidance
- **Proper exit codes** for script integration

## Cross-Platform Support

The scripts are designed to work across platforms:

- **Path handling** uses `pathlib` for cross-platform compatibility
- **Build system detection** automatically selects appropriate tools (MSBuild on Windows, Make on Unix)
- **Environment detection** adapts to different operating systems
- **Tool detection** searches common installation paths on each platform

## Getting Started

1. **Make scripts executable**:
   ```bash
   cd scripts/
   chmod +x make_executable.sh
   ./make_executable.sh
   ```

2. **Validate your environment**:
   ```bash
   ./environment_validator.py
   ```

3. **Run a complete automated workflow**:
   ```bash
   ./pipeline_helper.py validate-env
   ./pipeline_helper.py run /path/to/binary.exe --debug
   ```

The scripts will guide you through any missing dependencies or setup issues.