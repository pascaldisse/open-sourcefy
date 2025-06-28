#!/bin/bash

# Example script to run the multi-agent system for GaiaScript compiler development

echo "🚀 Starting Multi-Agent System for GaiaScript Compiler Development"
echo "=================================================="

# Path to the GaiaScript compiler project
COMPILER_PATH="../.gaia"
CONFIG_FILE="./gaiascript-compiler-config.json"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Configuration file not found: $CONFIG_FILE"
    echo "📝 Generating example configuration..."
    gaia-mas config generate "$CONFIG_FILE"
    exit 1
fi

# Validate configuration
echo "✓ Validating configuration..."
gaia-mas config validate "$CONFIG_FILE"

# Start the multi-agent system
echo ""
echo "🤖 Starting agents with GaiaScript compiler configuration..."
echo ""

# Run with all features enabled
gaia-mas start \
    --config "$CONFIG_FILE" \
    --project "$COMPILER_PATH" \
    --dashboard \
    --learning \
    --port 3001

# The system will now:
# 1. Load the GaiaScript compiler configuration
# 2. Initialize all 17 agents with specialized roles
# 3. Agent 0 will coordinate the compiler development
# 4. Dashboard will be available at http://localhost:3001
# 5. Learning system will optimize agent performance