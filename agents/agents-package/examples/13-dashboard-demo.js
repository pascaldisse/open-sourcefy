import { MultiAgentSystem } from '../src/index-optimized.js';
import { MonitoringDashboard } from '../src/monitoring/dashboard.js';

/**
 * Example 13: Monitoring Dashboard Demo
 * 
 * This example demonstrates:
 * - Real-time monitoring dashboard
 * - Live metrics visualization
 * - Agent status tracking
 * - Performance charts
 * - Alert management
 */

async function runDashboardDemo() {
  console.log('=== Monitoring Dashboard Demo ===\n');

  const system = new MultiAgentSystem();
  let dashboard = null;
  
  try {
    // Initialize system
    await system.initialize();
    await system.start({
      conditions: [
        {
          id: 'system_health',
          type: 'boolean',
          check: 'all_agents_healthy',
          description: 'All agents are healthy'
        },
        {
          id: 'performance_target',
          type: 'boolean',
          check: 'response_time_under_500ms',
          description: 'Response time under 500ms'
        }
      ]
    });
    
    console.log('✓ System initialized\n');

    // Start monitoring dashboard
    dashboard = new MonitoringDashboard(system, 3001);
    await dashboard.start();
    
    console.log('✓ Dashboard running at http://localhost:3001\n');
    console.log('Open your browser to view the real-time dashboard\n');

    // Simulate various activities for dashboard visualization
    console.log('Starting simulated workload...\n');

    // 1. Generate initial tasks
    console.log('Phase 1: Initial task generation');
    const initialTasks = [];
    
    for (let i = 0; i < 10; i++) {
      initialTasks.push(
        system.executeTask({
          description: `Initial task ${i}`,
          type: ['testing', 'documentation', 'code_review'][i % 3],
          priority: ['high', 'medium', 'low'][i % 3],
          context: {
            complexity: 'medium',
            iteration: i
          }
        })
      );
    }
    
    await Promise.all(initialTasks);
    console.log('✓ Initial tasks completed\n');

    // 2. Continuous workload simulation
    console.log('Phase 2: Continuous workload (press Ctrl+C to stop)\n');
    
    let taskCounter = 0;
    const workloadInterval = setInterval(async () => {
      // Generate varied tasks
      const taskTypes = [
        { type: 'testing', description: 'Write unit tests for module' },
        { type: 'documentation', description: 'Update API documentation' },
        { type: 'code_review', description: 'Review pull request changes' },
        { type: 'bug_fixing', description: 'Fix reported bug' },
        { type: 'git_operations', description: 'Prepare release branch' },
        { type: 'performance_optimization', description: 'Optimize query performance' }
      ];
      
      const selectedTask = taskTypes[taskCounter % taskTypes.length];
      const priority = taskCounter % 10 === 0 ? 'high' : 
                      taskCounter % 5 === 0 ? 'medium' : 'low';
      
      // Execute task
      system.executeTask({
        description: `${selectedTask.description} #${taskCounter}`,
        type: selectedTask.type,
        priority,
        context: {
          source: 'dashboard-demo',
          timestamp: Date.now()
        },
        cacheable: taskCounter % 3 === 0 // Some tasks use cache
      }).catch(error => {
        console.error('Task error:', error.message);
      });
      
      taskCounter++;
      
      // Occasionally trigger alerts
      if (taskCounter % 20 === 0) {
        system.systemMonitor.emit('alert', {
          severity: 'warning',
          category: 'performance',
          message: 'High task queue detected',
          metric: 'queueSize',
          value: taskCounter
        });
      }
      
      // Log progress
      if (taskCounter % 10 === 0) {
        const status = system.getSystemStatus();
        const activeAgents = Object.values(status.agents)
          .filter(a => a.status === 'working').length;
        
        console.log(`Tasks submitted: ${taskCounter}, Active agents: ${activeAgents}`);
      }
    }, 1000); // Submit task every second

    // 3. Performance stress test
    setTimeout(async () => {
      console.log('\nPhase 3: Performance stress test');
      
      const stressTasks = [];
      for (let i = 0; i < 50; i++) {
        stressTasks.push(
          system.executeTask({
            description: `Stress test task ${i}`,
            type: 'testing',
            priority: 'high',
            context: {
              complexity: 'high',
              stressTest: true
            }
          })
        );
      }
      
      const startTime = Date.now();
      await Promise.all(stressTasks);
      const duration = Date.now() - startTime;
      
      console.log(`✓ Stress test completed: 50 tasks in ${duration}ms`);
      console.log(`  Average: ${(duration / 50).toFixed(2)}ms per task\n`);
    }, 10000); // Start stress test after 10 seconds

    // 4. Cache effectiveness demonstration
    setTimeout(async () => {
      console.log('Phase 4: Cache effectiveness test');
      
      // Submit identical tasks to test cache
      const cacheTestTasks = [];
      const testDescription = 'Cache test: Generate comprehensive test suite';
      
      for (let i = 0; i < 5; i++) {
        cacheTestTasks.push(
          system.executeTask({
            description: testDescription,
            type: 'testing',
            priority: 'medium',
            context: {
              module: 'cacheTest',
              version: '1.0.0'
            },
            cacheable: true
          })
        );
      }
      
      const cacheResults = await Promise.all(cacheTestTasks);
      const cachedCount = cacheResults.filter(r => r.fromCache).length;
      
      console.log(`✓ Cache test completed: ${cachedCount}/5 served from cache\n`);
    }, 20000); // Start cache test after 20 seconds

    // 5. System health monitoring
    const healthInterval = setInterval(() => {
      const report = system.getPerformanceReport();
      const cacheStats = system.cacheLayer.getStats();
      const poolStats = system.claudePool.getStats();
      
      console.log('\n--- System Health Report ---');
      console.log(`Uptime: ${Math.floor(report.uptime / 1000)}s`);
      console.log(`Cache Hit Rate: ${cacheStats.hitRate}`);
      console.log(`Pool Efficiency: ${
        poolStats.metrics.acquired > 0 
          ? ((poolStats.metrics.acquired - poolStats.metrics.created) / poolStats.metrics.acquired * 100).toFixed(2) 
          : 0
      }%`);
      console.log(`Active Alerts: ${system.systemMonitor.getActiveAlerts().length}`);
      console.log('---------------------------\n');
    }, 30000); // Report every 30 seconds

    // Keep the demo running
    console.log('Dashboard demo is running. Visit http://localhost:3001');
    console.log('Press Ctrl+C to stop\n');

    // Handle graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n\nShutting down demo...');
      
      clearInterval(workloadInterval);
      clearInterval(healthInterval);
      
      if (dashboard) {
        await dashboard.stop();
      }
      
      await system.shutdown();
      
      console.log('Demo shutdown complete');
      process.exit(0);
    });

    // Keep process alive
    await new Promise(() => {});

  } catch (error) {
    console.error('Error in dashboard demo:', error);
    
    if (dashboard) {
      await dashboard.stop();
    }
    
    await system.shutdown();
  }
}

// Run the demo
runDashboardDemo().catch(console.error);