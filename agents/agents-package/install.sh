#!/bin/bash

# Multi-Agent System Installation Script

echo "🤖 Installing Multi-Agent System..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    echo "Please install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is required but not installed"
    exit 1
fi

echo "✓ Node.js found: $(node --version)"
echo "✓ npm found: $(npm --version)"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Make CLI executable
chmod +x cli.js

echo "✅ Installation complete!"
echo ""
echo "🚀 Usage:"
echo "  ./cli.js start                    # Start in current directory"
echo "  ./cli.js config generate my.json # Generate config"
echo "  ./cli.js execute 'Task here'     # Execute single task"
echo "  ./cli.js status                  # Check status"
echo ""
echo "📝 The system will auto-detect your project type and configure accordingly."
echo "You can also create custom configurations for specific workflows."