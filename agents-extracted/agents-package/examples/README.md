# Multi-Agent System Examples

This directory contains example workflows and use cases for the Multi-Agent System.

## Examples Overview

1. **[Basic Task Execution](./01-basic-task.js)** - Simple single-agent task
2. **[Multi-Agent Collaboration](./02-collaboration.js)** - Multiple agents working together
3. **[Code Review Workflow](./03-code-review.js)** - Complete code review process
4. **[Bug Fix Pipeline](./04-bug-fix.js)** - Finding and fixing bugs
5. **[Documentation Generation](./05-documentation.js)** - Auto-generating docs
6. **[Deployment Pipeline](./06-deployment.js)** - Full deployment workflow
7. **[Quality Assurance](./07-quality-assurance.js)** - QA validation process
8. **[Performance Optimization](./08-performance.js)** - Performance analysis and optimization
9. **[Security Audit](./09-security-audit.js)** - Security scanning and fixes
10. **[Custom Conditions](./10-custom-conditions.js)** - Advanced condition management

## Running Examples

```bash
# Run a specific example
node examples/01-basic-task.js

# Run with debug output
DEBUG=* node examples/02-collaboration.js

# Run all examples
npm run examples
```

## Prerequisites

1. Set up your `.env` file with your Anthropic API key
2. Install dependencies: `npm install`
3. Ensure the system is properly configured

## Example Structure

Each example follows this pattern:

```javascript
import { MultiAgentSystem } from '../src/index.js';

async function runExample() {
  // Initialize system
  const system = new MultiAgentSystem();
  await system.initialize();
  
  // Start system with conditions
  await system.start({ conditions: [...] });
  
  // Execute tasks
  const result = await system.executeTask({...});
  
  // Process results
  console.log('Result:', result);
  
  // Cleanup
  await system.shutdown();
}

runExample().catch(console.error);
```