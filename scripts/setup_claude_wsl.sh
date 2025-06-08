#!/bin/bash
# Setup Claude Code CLI for WSL environment
# Fixes the issue where claude-code does not work on Windows but claude-skip works from WSL

echo "Setting up Claude Code CLI for WSL..."

# Check if we're in WSL
if grep -q microsoft /proc/version; then
    echo "✅ WSL environment detected"
else
    echo "⚠️  This script is designed for WSL. You may want to install claude-code directly."
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install Node.js first:"
    echo "   curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -"
    echo "   sudo apt-get install -y nodejs"
    exit 1
fi

echo "📦 Installing Claude Code CLI packages..."

# Try to install claude-code-skip first (WSL-specific)
if npm install -g @anthropic-ai/claude-code-skip; then
    echo "✅ claude-skip installed successfully"
    echo "Command available: claude-skip"
else
    echo "⚠️  claude-skip installation failed, trying claude-code..."
    
    # Fallback to regular claude-code
    if npm install -g @anthropic-ai/claude-code; then
        echo "✅ claude-code installed successfully"
        echo "Command available: claude-code"
    else
        echo "❌ Both claude-skip and claude-code installation failed"
        echo "Please check your npm permissions and internet connection"
        exit 1
    fi
fi

# Test the installation
echo "🧪 Testing Claude CLI installation..."

for cmd in claude-skip claude-code claude; do
    if command -v $cmd &> /dev/null; then
        echo "✅ Found command: $cmd"
        
        # Test version (with timeout to avoid hanging)
        if timeout 10s $cmd --version >/dev/null 2>&1; then
            echo "✅ $cmd is working"
        else
            echo "⚠️  $cmd found but version check failed (may need authentication)"
        fi
    fi
done

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To enable AI features in open-sourcefy:"
echo "1. Edit config.yaml and set ai.enabled: true"
echo "2. Ensure you have Claude Pro/subscription for API access"
echo "3. Run: python3 main.py --verify-env to test the setup"
echo ""
echo "The open-sourcefy system will automatically detect and use the appropriate Claude CLI command."