# Multi-Agent System - Portable Version

A complete 17-agent coordination system that can be run in any directory with any project. Agent 0 acts as the master coordinator, delegating tasks to 16 specialized agents for testing, documentation, bug fixing, git operations, and more.

## 🚀 Quick Start

```bash
# Extract agents.zip to your desired location
unzip agents.zip
cd agents-package

# Install dependencies and make CLI executable
./install.sh

# Test the system
./test-system.js

# Start in current directory (auto-detects project type)
./cli.js start
```

## 📋 Features

- **17 Specialized Agents** coordinated by Agent 0 (master coordinator)
- **Auto-Detection** of project types (Node.js, Python, GaiaScript, Generic)
- **Custom Configuration** for any workflow with task-to-agent mapping
- **CLI Interface** for easy usage with multiple commands
- **Programmatic API** for integration into existing tools
- **Real-time Monitoring** with optional dashboard
- **Learning System** for performance optimization
- **Portable** - works in any directory without global installation

## 🛠 Usage

### Basic Commands
```bash
# Start system in current directory (auto-detects project)
./cli.js start

# With custom configuration file
./cli.js start --config my-config.json

# With custom project path
./cli.js start --project /path/to/project

# With custom system prompt for Agent 0
./cli.js start --prompt "You are a specialized coordinator for..."

# With dashboard monitoring
./cli.js start --dashboard --port 3001

# Execute single task
./cli.js execute "Implement feature X" --type development --priority high

# Check system status
./cli.js status
```

### Configuration Management
```bash
# Generate example configuration
./cli.js config generate my-config.json

# Auto-detect project configuration
./cli.js config detect /path/to/project

# Validate configuration file
./cli.js config validate my-config.json
```

## 🎯 Agent Specializations

The system includes 17 agents with specific roles:

- **Agent 0**: Master Coordinator - delegates tasks, monitors progress, ensures compliance
- **Agent 1**: Testing and Quality Assurance - unit tests, integration tests, test automation
- **Agent 2**: Documentation and Technical Writing - README, API docs, code comments
- **Agent 3**: Bug Hunting and Debugging - finding and fixing issues
- **Agent 4**: Code Commenting and Annotation - improving code readability
- **Agent 5**: Git Operations and Version Control - commits, branches, merges
- **Agent 6**: Task Scheduling and Workflow - project management and coordination
- **Agent 7**: Code Review and Analysis - static analysis and code quality
- **Agent 8**: Deployment and DevOps - CI/CD, build automation
- **Agent 9**: Performance Optimization - profiling and optimization
- **Agent 10**: Security Analysis - vulnerability scanning and fixes
- **Agent 11**: Database Operations - schema, queries, migrations
- **Agent 12**: API Development - REST/GraphQL endpoint creation
- **Agent 13**: Frontend Development - UI components and styling
- **Agent 14**: Backend Development - server logic and architecture
- **Agent 15**: Monitoring and Logging - observability and debugging
- **Agent 16**: Research and Learning - staying updated with best practices

## 📊 Supported Project Types

### Auto-Detection Features:
- **Node.js/TypeScript**: Detects package.json, runs npm/yarn commands, Jest/Mocha testing
- **Python**: Detects requirements.txt/pyproject.toml, pytest, virtual environments
- **GaiaScript**: Mathematical symbol compilation, recognizes .gaia files
- **Generic**: Customizable configuration for any project type

### Project-Specific Configurations:
Each project type gets optimized task mappings and agent specializations.

## 🔧 Custom Configuration

Create a JSON configuration file to customize behavior:

```json
{
  "projectName": "my-project",
  "projectType": "node",
  "projectPath": "/path/to/project",
  "systemPrompt": "Custom prompt for Agent 0...",
  "taskMapping": {
    "testing": [1, 7],
    "documentation": [2, 4],
    "bug_fixing": [3, 9, 10],
    "git_operations": [5],
    "custom_task": [11, 16],
    "analysis": [7, 10, 15],
    "deployment": [8]
  },
  "agentOverrides": {
    "1": {
      "systemPrompt": "You specialize in my project's testing needs...",
      "specialization": "custom testing framework"
    },
    "2": {
      "systemPrompt": "Focus on API documentation...",
      "tools": ["swagger", "jsdoc"]
    }
  },
  "conditions": [
    {
      "type": "boolean",
      "description": "All tests pass",
      "check": "test_results.success === true"
    },
    {
      "type": "llm_validated",
      "description": "Documentation is comprehensive",
      "validation_prompt": "Assess if the documentation covers all major features"
    }
  ]
}
```

## 🎮 Programmatic Usage

Integrate the system into your own tools:

```javascript
import { MultiAgentSystem } from './src/index.js';
import { agentConfig } from './src/config/agent-config.js';

// Initialize system
const system = new MultiAgentSystem();
await system.initialize();

// Apply configuration
const config = await agentConfig.createProjectConfig('/path/to/project');
agentConfig.applyToCoordinator(system.coordinator, config.config);
agentConfig.applyAgentOverrides(system.agents, config.config);

// Start coordination
await system.start();

// Execute specific tasks
await system.executeTask({
  description: "Analyze codebase for security vulnerabilities",
  type: "security_analysis",
  priority: "high"
});

// Execute multiple tasks
await system.executeTasks([
  { description: "Run all tests", type: "testing", priority: "high" },
  { description: "Update documentation", type: "documentation", priority: "medium" },
  { description: "Deploy to staging", type: "deployment", priority: "low" }
]);

// Monitor progress
system.on('task_completed', (task, agent) => {
  console.log(`Task "${task.description}" completed by Agent ${agent.id}`);
});

// Shutdown when done
await system.shutdown();
```

## 🎯 Real-World Examples

### Completing the GaiaScript Compiler
The system was successfully used to complete a mathematical symbol programming compiler:

```bash
# Used specialized agents to build:
# - Lexer for mathematical symbols (λ, Σ, ∆, Ω)
# - Parser for Chinese character tokens
# - AST generator
# - JavaScript and Go code generators
# Result: 70-90% token reduction vs traditional languages
```

### Full-Stack Web Application
```bash
# Agent 12: Created REST API endpoints
# Agent 13: Built React frontend components  
# Agent 1: Implemented comprehensive test suite
# Agent 2: Generated API documentation
# Agent 5: Set up git workflow with proper branching
# Agent 8: Configured CI/CD pipeline
```

### DevOps Automation
```bash
# Agent 8: Set up Docker containers and Kubernetes configs
# Agent 15: Implemented monitoring and alerting
# Agent 10: Performed security scanning
# Agent 6: Coordinated deployment workflow
```

## 📝 Requirements

- **Node.js 18+** (uses ES modules)
- **npm or yarn** for dependency management
- **Git** (optional, for git operations)
- **Docker** (optional, for containerized deployments)

## 🏗️ Directory Structure After Installation

```
agents-package/
├── cli.js                    # Main CLI entry point
├── install.sh               # Installation script
├── test-system.js           # System test suite
├── package.json             # Dependencies and scripts
├── README.md                # This file
├── src/
│   ├── index.js            # MultiAgentSystem main class
│   ├── agents/
│   │   ├── agent-0.js      # Master Coordinator
│   │   ├── agent-1.js      # Testing Agent
│   │   ├── agent-2.js      # Documentation Agent
│   │   └── ...             # Agents 3-16
│   ├── config/
│   │   └── agent-config.js # Configuration management
│   ├── cli/
│   │   └── cli.js          # CLI implementation
│   └── utils/
│       └── common.js       # Shared utilities
└── examples/
    ├── gaiascript-compiler-config.json
    ├── node-project-config.json
    └── python-project-config.json
```

## 🆘 Troubleshooting

### Common Issues:

1. **"Module not found" errors**
   ```bash
   # Solution: Install dependencies
   npm install
   ```

2. **"Permission denied" when running CLI**
   ```bash
   # Solution: Make CLI executable
   chmod +x cli.js
   ```

3. **Configuration errors**
   ```bash
   # Solution: Validate your config file
   ./cli.js config validate my-config.json
   ```

4. **Agent not responding**
   ```bash
   # Solution: Check system status and restart
   ./cli.js status
   ./cli.js start
   ```

5. **Memory issues with large projects**
   ```bash
   # Solution: Increase Node.js memory limit
   export NODE_OPTIONS="--max-old-space-size=4096"
   ./cli.js start
   ```

### Debug Mode:
```bash
# Enable detailed logging
DEBUG=* ./cli.js start

# Or specific debug categories
DEBUG=agent:*,coordination:* ./cli.js start
```

## 🤝 Contributing

The system is designed to be extensible:

1. **Add new agent specializations** by creating agent-N.js files
2. **Extend project type detection** in agent-config.js
3. **Add new task types** and mappings
4. **Contribute example configurations** for different project types

## 📖 Advanced Usage

### Custom Agent Development
```javascript
// Create custom agent with specific tools
class CustomAgent extends BaseAgent {
  constructor(id, specialization) {
    super(id, specialization);
    this.tools = ['custom-tool-1', 'custom-tool-2'];
  }
  
  async executeTask(task) {
    // Custom task execution logic
  }
}
```

### Integration with CI/CD
```yaml
# GitHub Actions example
- name: Run Multi-Agent System
  run: |
    unzip agents.zip
    cd agents-package
    ./install.sh
    ./cli.js start --config .github/agents-config.json
```

## 🚀 Ready to Transform Your Development Workflow!

The Multi-Agent System revolutionizes software development by coordinating specialized AI agents to handle every aspect of your project. From initial coding to testing, documentation, deployment, and maintenance - let the agents handle the complexity while you focus on innovation.

**Start your journey:**
1. Extract agents.zip
2. Run ./install.sh  
3. Execute ./cli.js start
4. Watch as 17 specialized agents transform your development process!

*Developed with ❤️ using the power of AI coordination and specialized agent systems.*