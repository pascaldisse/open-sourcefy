#!/usr/bin/env node

/**
 * Test the multi-agent system installation
 */

import { MultiAgentSystem } from './src/index.js';
import { agentConfig } from './src/config/agent-config.js';

async function testSystem() {
  console.log('🧪 Testing Multi-Agent System...');
  
  try {
    // Test system initialization
    const system = new MultiAgentSystem();
    await system.initialize();
    console.log('✅ System initialization: PASS');
    
    // Test configuration
    const config = await agentConfig.createProjectConfig(process.cwd());
    agentConfig.applyToCoordinator(system.coordinator, config.config);
    console.log('✅ Configuration system: PASS');
    
    // Test basic task execution
    await system.start();
    const result = await system.executeTask({
      description: 'Test task',
      type: 'testing',
      priority: 'low'
    });
    console.log('✅ Task execution: PASS');
    
    await system.shutdown();
    console.log('✅ System shutdown: PASS');
    
    console.log('\n🎉 All tests passed! Multi-Agent System is ready.');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  }
}

testSystem();