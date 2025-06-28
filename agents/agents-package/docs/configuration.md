# Multi-Agent System Configuration Guide

The Gaia Multi-Agent System supports flexible configuration to adapt to any project type and custom requirements.

## Quick Start

### 1. Generate Example Configuration
```bash
gaia-mas config generate my-config.json
```

### 2. Auto-Detect Project Configuration
```bash
gaia-mas config detect /path/to/project
```

### 3. Start with Custom Configuration
```bash
gaia-mas start --config my-config.json --project /path/to/project
```

## Configuration Options

### Basic Usage

Start the system with custom options:
```bash
# With custom system prompt
gaia-mas start --prompt "Focus on security and performance optimization"

# With specific project path
gaia-mas start --project ../my-project

# With configuration file
gaia-mas start --config ./gaia-config.json
```

### Configuration File Structure

```json
{
  "projectName": "my-project",
  "projectPath": "./",
  "taskMapping": {
    "custom_task": [1, 2],
    "analysis": [11, 16],
    "compilation": [7, 8, 9]
  },
  "agentOverrides": {
    "1": {
      "systemPrompt": "You are a specialized testing agent...",
      "capabilities": ["test_driven_development", "mocking"],
      "allowedTools": ["Read", "Write", "Bash"]
    }
  },
  "conditions": [
    {
      "id": "all_tests_pass",
      "type": "boolean",
      "check": "all_tests_pass",
      "description": "All unit tests must pass"
    }
  ],
  "systemPrompt": "Custom prompt for Agent 0 (coordinator)"
}
```

## Project Type Detection

The system automatically detects project types and suggests appropriate configurations:

### GaiaScript Projects
- Detected by `.gaia` files
- Configures agents for:
  - Lexer development (Agent 1)
  - Parser implementation (Agent 2)
  - AST transformation (Agent 3)
  - Code generation (Agents 4-5)
  - Testing (Agent 6)

### Node.js/TypeScript Projects
- Detected by `package.json`
- Configures agents for:
  - Unit testing with Jest/Mocha
  - API documentation
  - Dependency management
  - Build automation
  - TypeScript compilation

### Python Projects
- Detected by `requirements.txt`, `setup.py`, or `pyproject.toml`
- Configures agents for:
  - Pytest testing
  - Sphinx documentation
  - Pylint analysis
  - Virtual environment management

## Task Mapping

Define which agents handle specific task types:

```json
{
  "taskMapping": {
    "lexer": [1],              // Agent 1 handles lexer tasks
    "parser": [2],             // Agent 2 handles parser tasks
    "testing": [1, 6],         // Agents 1 and 6 handle testing
    "documentation": [2, 7],   // Agents 2 and 7 handle docs
    "custom_analysis": [11, 12, 16]  // Multiple agents for analysis
  }
}
```

## Agent Overrides

Customize individual agent behavior:

```json
{
  "agentOverrides": {
    "1": {
      "systemPrompt": "You are an expert in Test-Driven Development...",
      "capabilities": [
        "unit_testing",
        "integration_testing",
        "test_coverage_analysis"
      ],
      "allowedTools": ["Read", "Write", "Bash", "Grep"]
    },
    "2": {
      "systemPrompt": "You specialize in API documentation...",
      "allowedTools": ["Read", "Write", "WebFetch"]
    }
  }
}
```

## Conditions

Define completion conditions for the agent system:

```json
{
  "conditions": [
    {
      "id": "tests_pass",
      "type": "boolean",
      "check": "all_tests_pass",
      "description": "All tests must pass"
    },
    {
      "id": "docs_complete",
      "type": "llm_validated",
      "description": "Documentation is comprehensive and accurate"
    }
  ]
}
```

## Examples

### GaiaScript Compiler Configuration
```json
{
  "projectName": "gaia-compiler",
  "taskMapping": {
    "lexer": [1],
    "parser": [2],
    "ast_transformer": [3],
    "js_codegen": [4],
    "go_codegen": [5],
    "compiler_testing": [6],
    "documentation": [7],
    "optimization": [8]
  },
  "agentOverrides": {
    "1": {
      "systemPrompt": "You specialize in tokenizing Chinese characters for GaiaScript..."
    }
  }
}
```

### Web Application Configuration
```json
{
  "projectName": "web-app",
  "taskMapping": {
    "frontend_testing": [1],
    "api_testing": [1, 6],
    "documentation": [2],
    "security_audit": [10],
    "deployment": [8],
    "performance": [9]
  },
  "conditions": [
    {
      "id": "security_check",
      "type": "boolean",
      "check": "security_audit_pass"
    }
  ]
}
```

## CLI Commands

### Configuration Management
```bash
# Generate example configuration
gaia-mas config generate config.json

# Validate configuration file
gaia-mas config validate config.json

# Auto-detect project type
gaia-mas config detect /path/to/project
```

### Running with Configuration
```bash
# Start with all options
gaia-mas start \
  --config ./my-config.json \
  --project /path/to/project \
  --dashboard \
  --learning \
  --prompt "Additional instructions for Agent 0"
```

## Best Practices

1. **Project-Specific Configs**: Create separate configuration files for different projects
2. **Version Control**: Commit configuration files to track changes
3. **Incremental Overrides**: Start with auto-detection, then customize as needed
4. **Clear Task Names**: Use descriptive task type names for better agent selection
5. **Test Configurations**: Validate configs before production use

## Troubleshooting

### Configuration Not Loading
- Ensure the path is correct and file exists
- Check JSON syntax with `gaia-mas config validate`
- Verify agent IDs are between 0-16

### Agents Not Responding to Custom Tasks
- Check task mapping includes appropriate agent IDs
- Ensure agents have required capabilities
- Verify agent tools match task requirements

### Project Detection Failed
- Manually specify project type in configuration
- Ensure project has standard structure files
- Use explicit task mappings instead of auto-detection