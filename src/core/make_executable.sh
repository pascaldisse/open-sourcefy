#!/bin/bash
# Make all Python scripts in the scripts directory executable

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Making scripts executable in: $SCRIPT_DIR"

# Make all Python scripts executable
for script in "$SCRIPT_DIR"/*.py; do
    if [[ -f "$script" ]]; then
        chmod +x "$script"
        echo "Made executable: $(basename "$script")"
    fi
done

# Make this script executable too
chmod +x "$0"

echo "All scripts are now executable"
echo ""
echo "Available automation scripts:"
echo "  file_operations.py - File operations and directory management"
echo "  environment_validator.py - Environment validation and setup checking"
echo "  build_system_automation.py - Build system generation and compilation testing"
echo "  pipeline_helper.py - Pipeline execution and management utilities"
echo ""
echo "Usage examples:"
echo "  ./environment_validator.py"
echo "  ./file_operations.py create-structure /path/to/output"
echo "  ./pipeline_helper.py validate-env"
echo "  ./build_system_automation.py --output-dir output/test cmake --project-name test --sources src/*.c"