import { MultiAgentSystem } from '../src/index.js';

/**
 * Example 1: Basic Task Execution
 * 
 * This example demonstrates:
 * - System initialization
 * - Simple task execution
 * - Result handling
 * - Proper shutdown
 */

async function runBasicTaskExample() {
  console.log('=== Basic Task Execution Example ===\n');

  // Initialize the system
  const system = new MultiAgentSystem();
  
  try {
    console.log('1. Initializing Multi-Agent System...');
    await system.initialize();
    console.log('✓ System initialized successfully\n');

    console.log('2. Starting system...');
    await system.start();
    console.log('✓ System started\n');

    console.log('3. Executing a simple test task...');
    const testTask = {
      description: 'Write a simple unit test for a calculator add function',
      type: 'testing',
      priority: 'medium',
      context: {
        functionSignature: 'function add(a, b) { return a + b; }',
        testFramework: 'jest'
      },
      requirements: 'Create a comprehensive test suite with edge cases'
    };

    const result = await system.executeTask(testTask);

    console.log('✓ Task completed!\n');
    console.log('Task Result:');
    console.log('- Task ID:', result.taskId);
    console.log('- Success:', result.success);
    console.log('- Agent ID:', result.agentId);
    console.log('- Duration:', result.duration, 'ms');
    console.log('- Cost:', `$${result.cost || 0}`);
    console.log('\nGenerated Test:');
    console.log(result.result);

    // Get agent status
    console.log('\n4. Checking agent status...');
    const agentStatus = system.agents.get(result.agentId).getStatus();
    console.log('Agent Performance:');
    console.log('- Tasks Completed:', agentStatus.performance.tasksCompleted);
    console.log('- Success Rate:', 
      agentStatus.performance.tasksCompleted > 0 
        ? `${(agentStatus.performance.tasksSuccess / agentStatus.performance.tasksCompleted * 100).toFixed(1)}%`
        : 'N/A'
    );
    console.log('- Average Response Time:', agentStatus.performance.averageResponseTime, 'ms');

  } catch (error) {
    console.error('Error during example execution:', error);
  } finally {
    console.log('\n5. Shutting down system...');
    await system.shutdown();
    console.log('✓ System shut down successfully');
  }
}

// Run the example
runBasicTaskExample().catch(console.error);