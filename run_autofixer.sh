#!/bin/bash
# Automated Pipeline Fixer - Complete Setup and Execution
# Prepares environment and launches continuous pipeline fixing

set -e

echo "========================================================================"
echo "OPEN-SOURCEFY AUTOMATED PIPELINE FIXER SETUP"
echo "PREPARING ENVIRONMENT FOR CONTINUOUS PIPELINE EXECUTION"
echo "========================================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔧 Working directory: $PWD"

# Validate we're in the right project
if [[ ! -f "main.py" ]] || [[ ! -f "rules.md" ]] || [[ ! -f "CLAUDE.md" ]]; then
    echo "❌ ERROR: Not in open-sourcefy project root"
    echo "   Required files: main.py, rules.md, CLAUDE.md"
    exit 1
fi

echo "✅ Confirmed open-sourcefy project root"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 not found"
    echo "   Please install Python 3.8+ to continue"
    exit 1
fi

echo "✅ Python 3 available: $(python3 --version)"

# Setup virtual environment
VENV_PATH="matrix_venv"
if [[ ! -d "$VENV_PATH" ]]; then
    echo "🔧 Creating virtual environment: $VENV_PATH"
    python3 -m venv "$VENV_PATH"
else
    echo "✅ Virtual environment exists: $VENV_PATH"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "🔧 Installing required packages..."
pip install claude-code-sdk

# Verify claude-code-sdk installation
if python -c "import claude_code_sdk" 2>/dev/null; then
    echo "✅ Claude Code SDK installed successfully"
else
    echo "❌ ERROR: Failed to install Claude Code SDK"
    exit 1
fi

# Check for git repository
if [[ ! -d ".git" ]]; then
    echo "🔧 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit for automated pipeline fixing"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository exists"
fi

# Make scripts executable
chmod +x auto_pipeline_fixer.py
echo "✅ Scripts made executable"

# Create logs directory
mkdir -p logs
echo "✅ Logs directory created"

# Check for ANTHROPIC_API_KEY
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "⚠️  WARNING: ANTHROPIC_API_KEY not set"
    echo "   The Claude Code SDK requires an API key to function"
    echo "   Set it with: export ANTHROPIC_API_KEY=your_key_here"
    echo "   Continuing setup anyway..."
else
    echo "✅ ANTHROPIC_API_KEY is configured"
fi

# Test pipeline script exists
if [[ ! -f "main.py" ]]; then
    echo "❌ ERROR: main.py pipeline script not found"
    exit 1
fi

echo "✅ Pipeline script found: main.py"

# Test input binary exists
if [[ ! -f "input/launcher.exe" ]]; then
    echo "⚠️  WARNING: input/launcher.exe not found"
    echo "   Place your target binary at input/launcher.exe"
else
    echo "✅ Target binary found: input/launcher.exe"
fi

# Create input directory if needed
mkdir -p input
echo "✅ Input directory ready"

# Display setup summary
echo "========================================================================"
echo "🎉 AUTOMATED PIPELINE FIXER SETUP COMPLETE"
echo ""
echo "Environment Summary:"
echo "   - Working Directory: $PWD"
echo "   - Virtual Environment: $VENV_PATH"
echo "   - Claude Code SDK: Installed"
echo "   - Git Repository: Ready"
echo "   - Target Binary: $([ -f "input/launcher.exe" ] && echo 'Found' || echo 'Missing')"
echo "   - API Key: $([ -n "$ANTHROPIC_API_KEY" ] && echo 'Configured' || echo 'Missing')"
echo ""
echo "Ready to run automated pipeline fixer!"
echo "========================================================================"
echo ""
# Test individual agents starting with Agent 0
echo "========================================================================"
echo "🧪 TESTING INDIVIDUAL AGENTS (STARTING WITH AGENT 0)"
echo "Validating each Matrix agent before full pipeline execution"
echo "========================================================================"

# Test Agent 0 (Deus Ex Machina) - Master Orchestrator
echo "🤖 Testing Agent 0 (Deus Ex Machina) - Master Orchestrator..."
if python main.py input/launcher.exe --agents 0 --debug 2>&1 | tee "logs/agent_0_test_$(date +%Y%m%d_%H%M%S).log"; then
    echo "✅ Agent 0 (Deus Ex Machina) test PASSED"
else
    echo "❌ Agent 0 (Deus Ex Machina) test FAILED"
    echo "   Check logs/agent_0_test_*.log for details"
    echo "   Continuing with individual agent tests..."
fi
echo ""

# Test Agents 1-16 individually
for agent_id in {1..16}; do
    echo "🤖 Testing Agent $agent_id..."
    if python main.py input/launcher.exe --agents $agent_id --debug 2>&1 | tee "logs/agent_${agent_id}_test_$(date +%Y%m%d_%H%M%S).log"; then
        echo "✅ Agent $agent_id test PASSED"
    else
        echo "❌ Agent $agent_id test FAILED"
        echo "   Check logs/agent_${agent_id}_test_*.log for details"
    fi
    echo ""
    
    # Brief pause between agent tests
    sleep 2
done

echo "========================================================================"
echo "🎯 INDIVIDUAL AGENT TESTING COMPLETED"
echo "All agent test logs saved in logs/ directory"
echo "========================================================================"
echo ""

# Ask user if they want to proceed with full pipeline fixer
echo "Individual agent testing complete! Would you like to run the full automated pipeline fixer now? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
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
else
    echo ""
    echo "Setup and testing complete! To run the automated fixer later:"
    echo "1. Ensure ANTHROPIC_API_KEY is set: export ANTHROPIC_API_KEY=your_key_here"
    echo "2. Place target binary at: input/launcher.exe"
    echo "3. Run: source matrix_venv/bin/activate && python auto_pipeline_fixer.py"
    echo ""
    echo "Individual agent test logs are available in logs/ directory"
    echo ""
    echo "The automated fixer will:"
    echo "   - Use single worktree for entire execution"
    echo "   - Continuously attempt pipeline until ~5MB executable success"
    echo "   - Apply Claude Code SDK fixes automatically"
    echo "   - Create new branch from master with all fixes"
    echo "   - Never stop until pipeline is completely functional"
    echo ""
    echo "Zero tolerance mode: Will not stop until success!"
fi