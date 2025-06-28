/**
 * Example: Programmatic usage of the Multi-Agent System
 * 
 * This shows how to use the system from your own code
 */

import { MultiAgentSystem } from '../src/index.js';
import { agentConfig } from '../src/config/agent-config.js';
import { logger } from '../src/utils/logger.js';

async function runCompilerDevelopment() {
  console.log('🚀 Starting GaiaScript Compiler Development with Multi-Agent System\n');
  
  // Initialize the system
  const system = new MultiAgentSystem();
  
  try {
    // Load configuration for GaiaScript compiler
    const config = await agentConfig.loadConfigFile('./gaiascript-compiler-config.json');
    
    // Initialize system
    await system.initialize();
    
    // Apply configuration
    agentConfig.applyToCoordinator(system.coordinator, config);
    agentConfig.applyAgentOverrides(system.agents, config);
    
    // Start the system with conditions
    await system.start({ conditions: config.conditions });
    
    console.log('✓ System initialized and running\n');
    
    // Execute compiler development tasks
    const tasks = [
      {
        description: 'Implement lexer for Chinese character tokenization',
        type: 'lexer',
        priority: 'high',
        context: {
          requirements: 'Support all characters from CLAUDE.md character maps',
          outputFormat: 'TypeScript modules in TypeScript/src/compiler/lexer/'
        }
      },
      {
        description: 'Implement parser for GaiaScript syntax',
        type: 'parser',
        priority: 'high',
        context: {
          requirements: 'Parse all GaiaScript constructs: 文⟨⟩, 列⟨⟩, 物⟨⟩, etc.',
          dependencies: ['lexer']
        }
      },
      {
        description: 'Create comprehensive test suite',
        type: 'compiler_testing',
        priority: 'high',
        context: {
          coverage: 'All language features',
          framework: 'Jest or Vitest'
        }
      }
    ];
    
    // Submit tasks to the coordinator
    console.log('📋 Submitting compiler development tasks...\n');
    
    for (const task of tasks) {
      const result = await system.executeTask(task);
      
      if (result.success) {
        console.log(`✓ ${task.description}`);
        console.log(`  Completed by Agent ${result.agentId} in ${result.duration}ms\n`);
      } else {
        console.log(`✗ ${task.description}`);
        console.log(`  Error: ${result.error}\n`);
      }
    }
    
    // Monitor progress
    const interval = setInterval(async () => {
      const status = system.getSystemStatus();
      const progress = (status.systemState.completedTasks / status.systemState.totalTasks) * 100;
      
      console.log(`Progress: ${progress.toFixed(1)}% (${status.systemState.completedTasks}/${status.systemState.totalTasks} tasks)`);
      
      // Check if all conditions are met
      const conditionsMet = await system.coordinator.checkConditions();
      if (conditionsMet) {
        console.log('\n🎉 All conditions met! Compiler development complete.');
        clearInterval(interval);
        await system.shutdown();
      }
    }, 5000);
    
  } catch (error) {
    console.error('Error:', error);
    await system.shutdown();
  }
}

// Custom usage for any project
async function runCustomProject(projectPath, customConfig) {
  const system = new MultiAgentSystem();
  
  try {
    // Create project-specific configuration
    const { configId, config } = await agentConfig.createProjectConfig(projectPath, customConfig);
    
    await system.initialize();
    
    // Apply configuration
    agentConfig.applyToCoordinator(system.coordinator, config);
    agentConfig.applyAgentOverrides(system.agents, config);
    
    await system.start({ conditions: config.conditions });
    
    // Your custom logic here
    console.log(`System running for project: ${config.projectName}`);
    
    return system;
    
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}

// Example: Running on any project with custom configuration
async function example() {
  // Example 1: GaiaScript Compiler
  await runCompilerDevelopment();
  
  // Example 2: Custom project
  const customSystem = await runCustomProject('/path/to/my/project', {
    taskMapping: {
      'my_custom_task': [1, 2, 3],
      'analysis': [11, 12],
      'deployment': [8]
    },
    agentOverrides: {
      '1': {
        systemPrompt: 'You are specialized for my project needs...'
      }
    },
    systemPrompt: 'Coordinate agents for my custom project requirements'
  });
  
  // Execute custom tasks
  await customSystem.executeTask({
    description: 'Analyze codebase for optimization opportunities',
    type: 'analysis',
    priority: 'medium'
  });
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  example().catch(console.error);
}

export { runCompilerDevelopment, runCustomProject };