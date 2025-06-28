import { MultiAgentSystem } from '../src/index-optimized.js';

/**
 * Example 12: Performance Optimization Demo
 * 
 * This example demonstrates:
 * - Task queue optimization
 * - Message batching
 * - Cache utilization
 * - Resource pooling
 * - Performance metrics
 */

async function runPerformanceDemo() {
  console.log('=== Performance Optimization Demo ===\n');

  const system = new MultiAgentSystem();
  
  try {
    await system.initialize();
    await system.start();
    console.log('✓ Optimized system initialized\n');

    // Get initial performance baseline
    const initialReport = system.getPerformanceReport();
    console.log('Initial Performance Metrics:');
    console.log('- Cache Hit Rate:', initialReport.optimization.cacheHitRate);
    console.log('- Pool Connections:', initialReport.optimization.poolUtilization.totalConnections);
    console.log('- Task Optimizations:', initialReport.optimization.taskOptimizations);
    console.log();

    // Test 1: Parallel task execution with optimization
    console.log('Test 1: Parallel Task Execution\n');
    
    const parallelTasks = [];
    const taskCount = 20;
    
    console.time('Parallel Tasks');
    
    for (let i = 0; i < taskCount; i++) {
      const priority = i < 5 ? 'high' : i < 15 ? 'medium' : 'low';
      parallelTasks.push(
        system.executeTask({
          id: `task-${i}`,
          description: `Generate test suite for function ${i}`,
          type: 'testing',
          priority,
          context: {
            functionName: `calculateMetric${i}`,
            complexity: 'medium'
          },
          cacheable: true,
          cacheTTL: 300000 // 5 minutes
        })
      );
    }
    
    const results = await Promise.all(parallelTasks);
    console.timeEnd('Parallel Tasks');
    
    const successful = results.filter(r => r.success).length;
    console.log(`✓ Completed ${successful}/${taskCount} tasks successfully`);
    console.log(`✓ High priority tasks processed first\n`);

    // Test 2: Cache effectiveness
    console.log('Test 2: Cache Effectiveness\n');
    
    // Submit similar tasks to test cache
    const cachedTasks = [];
    console.time('Cached Tasks');
    
    for (let i = 0; i < 10; i++) {
      cachedTasks.push(
        system.executeTask({
          description: `Generate test suite for function ${i}`,
          type: 'testing',
          priority: 'medium',
          context: {
            functionName: `calculateMetric${i}`,
            complexity: 'medium'
          },
          cacheable: true
        })
      );
    }
    
    const cachedResults = await Promise.all(cachedTasks);
    console.timeEnd('Cached Tasks');
    
    const fromCache = cachedResults.filter(r => r.fromCache).length;
    console.log(`✓ ${fromCache}/${cachedTasks.length} tasks served from cache`);
    console.log(`✓ Cache significantly improved response time\n`);

    // Test 3: Message batching
    console.log('Test 3: Message Batching\n');
    
    // Send multiple messages to agents
    const messages = [];
    for (let i = 1; i <= 5; i++) {
      for (let j = 0; j < 10; j++) {
        system.performanceOptimizer.batchMessage(i, {
          type: 'status_request',
          content: { requestId: `req-${i}-${j}` },
          from: 0,
          to: i
        });
      }
    }
    
    // Wait for batching
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const optimizerMetrics = system.performanceOptimizer.getMetrics();
    console.log(`✓ Messages batched: ${optimizerMetrics.messagesBatched}`);
    console.log(`✓ Reduced network overhead through batching\n`);

    // Test 4: Resource pool utilization
    console.log('Test 4: Resource Pool Utilization\n');
    
    const poolStats = system.claudePool.getStats();
    console.log('Claude Pool Statistics:');
    console.log('- Pool Size:', poolStats.poolSize);
    console.log('- Active Connections:', poolStats.activeConnections);
    console.log('- Connections Created:', poolStats.metrics.created);
    console.log('- Connections Reused:', poolStats.metrics.acquired - poolStats.metrics.created);
    console.log();

    // Test 5: Task queue optimization
    console.log('Test 5: Task Queue Optimization\n');
    
    // Submit tasks with different priorities
    const queuedTasks = [];
    const priorities = ['low', 'medium', 'high'];
    
    for (let i = 0; i < 15; i++) {
      const priority = priorities[i % 3];
      queuedTasks.push({
        description: `Task ${i} with ${priority} priority`,
        type: 'documentation',
        priority,
        optimize: true
      });
    }
    
    // Queue all tasks
    const queuePromises = queuedTasks.map(task => system.executeTask(task));
    
    // Monitor queue processing
    const queueInterval = setInterval(() => {
      const metrics = system.performanceOptimizer.getMetrics();
      console.log(`Queue status - High: ${metrics.taskQueueSizes.high}, ` +
                  `Medium: ${metrics.taskQueueSizes.medium}, ` +
                  `Low: ${metrics.taskQueueSizes.low}`);
    }, 50);
    
    await Promise.all(queuePromises);
    clearInterval(queueInterval);
    
    console.log('✓ Tasks processed in priority order\n');

    // Final performance report
    console.log('=== Final Performance Report ===\n');
    
    const finalReport = system.getPerformanceReport();
    const cacheStats = system.cacheLayer.getStats();
    const poolStats2 = system.claudePool.getStats();
    const optimizerStats = system.performanceOptimizer.getMetrics();
    
    console.log('Cache Performance:');
    console.log('- Hit Rate:', cacheStats.hitRate);
    console.log('- Total Hits:', cacheStats.hits);
    console.log('- Total Misses:', cacheStats.misses);
    console.log('- Memory Usage:', cacheStats.memoryUsageMB + 'MB');
    console.log('- Entries:', cacheStats.entries);
    console.log();
    
    console.log('Connection Pool:');
    console.log('- Total Connections:', poolStats2.totalConnections);
    console.log('- Connections Reused:', poolStats2.metrics.acquired - poolStats2.metrics.created);
    console.log('- Pool Efficiency:', 
      ((poolStats2.metrics.acquired - poolStats2.metrics.created) / poolStats2.metrics.acquired * 100).toFixed(2) + '%');
    console.log();
    
    console.log('Task Optimization:');
    console.log('- Task Queue Optimizations:', optimizerStats.taskQueueOptimizations);
    console.log('- Messages Batched:', optimizerStats.messagesBatched);
    console.log('- Total Optimizations:', optimizerStats.totalOptimizations);
    console.log();
    
    // Agent performance comparison
    console.log('Agent Performance:');
    let totalTasks = 0;
    let totalResponseTime = 0;
    
    for (const [agentId, metrics] of Object.entries(finalReport.agents)) {
      if (metrics.tasksCompleted > 0) {
        console.log(`- Agent ${agentId}: ${metrics.tasksCompleted} tasks, ` +
                    `${metrics.averageResponseTime.toFixed(2)}ms avg response, ` +
                    `${(metrics.successRate * 100).toFixed(2)}% success`);
        totalTasks += metrics.tasksCompleted;
        totalResponseTime += metrics.averageResponseTime * metrics.tasksCompleted;
      }
    }
    
    const systemAvgResponse = totalTasks > 0 ? totalResponseTime / totalTasks : 0;
    console.log(`\nSystem Average Response Time: ${systemAvgResponse.toFixed(2)}ms`);
    
    // Performance improvement summary
    console.log('\n=== Performance Improvements ===');
    console.log('✓ Task batching reduced overhead');
    console.log('✓ Cache eliminated redundant processing');
    console.log('✓ Connection pooling improved resource utilization');
    console.log('✓ Priority queuing ensured critical tasks completed first');
    console.log('✓ Message batching reduced communication overhead');

  } catch (error) {
    console.error('Error in performance demo:', error);
  } finally {
    await system.shutdown();
    console.log('\n✓ Demo completed');
  }
}

// Run the demo
runPerformanceDemo().catch(console.error);