#!/bin/bash
# Automated Pipeline Fixer Launcher
# Runs the continuous pipeline fixer with proper environment setup

set -e

echo "========================================================================"
echo "OPEN-SOURCEFY AUTOMATED PIPELINE FIXER"
echo "ZERO TOLERANCE MODE - WILL NOT STOP UNTIL PIPELINE IS FIXED"
echo "========================================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we're in the right directory
if [[ ! -f "main.py" ]] || [[ ! -f "rules.md" ]] || [[ ! -f "CLAUDE.md" ]]; then
    echo "❌ ERROR: Required files not found in current directory"
    echo "   Make sure you're running this from the open-sourcefy project root"
    exit 1
fi

# Check for virtual environment
VENV_PATH=""
if [[ -d "matrix_venv" ]]; then
    VENV_PATH="matrix_venv"
elif [[ -d "../../venv" ]]; then
    VENV_PATH="../../venv"
elif [[ -d "../venv" ]]; then
    VENV_PATH="../venv"
elif [[ -d "venv" ]]; then
    VENV_PATH="venv"
else
    echo "❌ ERROR: Virtual environment not found"
    echo "   Expected locations: matrix_venv, ../../venv, ../venv, or ./venv"
    exit 1
fi

echo "✅ Found virtual environment at: $VENV_PATH"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Verify claude-code-sdk is installed
if ! python -c "import claude_code_sdk" 2>/dev/null; then
    echo "❌ ERROR: claude-code-sdk not found in virtual environment"
    echo "   Please install it with: pip install claude-code-sdk"
    exit 1
fi

echo "✅ Claude Code SDK found"

# Initialize git repository if needed
if [[ ! -d ".git" ]]; then
    echo "🔧 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for automated pipeline fixing"
fi

echo "✅ Git repository ready"

# Set environment variables - inherit from parent shell
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "⚠️  WARNING: ANTHROPIC_API_KEY not set"
    echo "   The Claude Code SDK may not function without an API key"
    echo "   Set it with: export ANTHROPIC_API_KEY=your_key_here"
    echo "   Continuing anyway..."
else
    echo "✅ ANTHROPIC_API_KEY is set"
fi

# Create logs directory
mkdir -p logs

echo "========================================================================"
echo "🚀 STARTING AUTOMATED PIPELINE FIXER"
echo "   Working directory: $PWD"
echo "   Virtual environment: $VENV_PATH"
echo "   Git worktree support: ENABLED"
echo "   Zero tolerance mode: ACTIVE"
echo "========================================================================"

# Run the automated pipeline fixer
echo "⚡ Launching auto_pipeline_fixer.py..."
echo ""

# Run with python buffering disabled for real-time logs
PYTHONUNBUFFERED=1 python auto_pipeline_fixer.py "$PWD" 2>&1 | tee "logs/auto_fixer_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "========================================================================"
echo "AUTOMATED PIPELINE FIXER COMPLETED"
echo "Check the logs above for results"
echo "========================================================================"