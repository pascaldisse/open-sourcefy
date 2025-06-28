import { MultiAgentSystem } from '../src/index-optimized.js';
import { AgentLearning } from '../src/learning/agent-learning.js';

/**
 * Example 14: Agent Learning Demo
 * 
 * This example demonstrates:
 * - Agent learning from task execution
 * - Pattern recognition
 * - Performance adaptation
 * - Knowledge sharing between agents
 * - Learning insights and recommendations
 */

async function runLearningDemo() {
  console.log('=== Agent Learning Demo ===\n');

  const system = new MultiAgentSystem();
  const learningSystem = new AgentLearning({
    learningRate: 0.2,
    minConfidence: 0.6,
    persistencePath: './learning-data-demo'
  });
  
  try {
    // Initialize systems
    await system.initialize();
    await system.start();
    await learningSystem.initialize();
    
    console.log('✓ Systems initialized\n');

    // Connect learning system to agents
    for (const [agentId, agent] of system.agents) {
      agent.on('taskComplete', async (data) => {
        await learningSystem.recordTaskExecution(agentId, data.task, data.result);
      });
    }

    // Listen for learning events
    learningSystem.on('adaptation-suggested', (data) => {
      console.log(`\n💡 Adaptation suggested for Agent ${data.agentId}:`);
      console.log(`   Strategy: ${data.adaptation.strategy}`);
      console.log(`   Confidence: ${(data.confidence * 100).toFixed(1)}%`);
    });

    learningSystem.on('knowledge-shared', (data) => {
      console.log(`\n🤝 Knowledge shared from Agent ${data.from} to Agent ${data.to}`);
    });

    console.log('Phase 1: Initial Learning Phase\n');

    // Submit various tasks to build learning data
    const learningTasks = [
      // Testing tasks with varying complexity
      ...Array(15).fill(null).map((_, i) => ({
        description: `Write unit tests for module ${i}`,
        type: 'testing',
        priority: i < 5 ? 'high' : 'medium',
        context: {
          complexity: i % 3 === 0 ? 'high' : i % 2 === 0 ? 'medium' : 'low',
          module: `module-${i % 5}`
        }
      })),
      
      // Documentation tasks
      ...Array(10).fill(null).map((_, i) => ({
        description: `Update documentation for API ${i}`,
        type: 'documentation',
        priority: 'medium',
        context: {
          apiVersion: `v${i % 3 + 1}`,
          scope: i % 2 === 0 ? 'full' : 'partial'
        }
      })),
      
      // Bug fixing tasks
      ...Array(8).fill(null).map((_, i) => ({
        description: `Fix bug in component ${i}`,
        type: 'bug_fixing',
        priority: 'high',
        context: {
          severity: i % 3 === 0 ? 'critical' : 'medium',
          component: `component-${i % 4}`
        }
      }))
    ];

    // Execute learning tasks
    console.log('Executing learning tasks...');
    const results = await Promise.all(
      learningTasks.map(task => system.executeTask(task))
    );

    const successCount = results.filter(r => r.success).length;
    console.log(`✓ Completed ${successCount}/${learningTasks.length} tasks successfully\n`);

    // Display initial learning insights
    console.log('Initial Learning Insights:');
    for (let i = 1; i <= 6; i++) {
      const insights = learningSystem.getLearningInsights(i);
      if (insights && insights.performance.totalTasks > 0) {
        console.log(`\nAgent ${i} (${system.agents.get(i).name}):`);
        console.log(`  - Success Rate: ${(insights.performance.successRate * 100).toFixed(1)}%`);
        console.log(`  - Avg Duration: ${Math.round(insights.performance.avgDuration)}ms`);
        console.log(`  - Patterns Detected: ${insights.patterns.total}`);
      }
    }

    console.log('\n\nPhase 2: Pattern Recognition\n');

    // Submit similar tasks to test pattern recognition
    const patternTasks = [
      // Repeated testing patterns
      ...Array(10).fill(null).map((_, i) => ({
        description: `Write unit tests for module ${i % 3}`,
        type: 'testing',
        priority: 'medium',
        context: {
          complexity: 'medium',
          module: `module-${i % 3}`
        }
      })),
      
      // Repeated error patterns
      ...Array(5).fill(null).map(() => ({
        description: 'Process invalid data format',
        type: 'bug_fixing',
        priority: 'high',
        context: {
          error: 'invalid_format',
          source: 'api'
        }
      }))
    ];

    console.log('Testing pattern recognition...');
    await Promise.all(patternTasks.map(task => system.executeTask(task)));

    // Check pattern detection
    const agent1Insights = learningSystem.getLearningInsights(1);
    console.log('\nPattern Recognition Results:');
    console.log(`- Successful patterns: ${agent1Insights.patterns.successful}`);
    console.log(`- Problematic patterns: ${agent1Insights.patterns.problematic}`);

    console.log('\n\nPhase 3: Performance Adaptation\n');

    // Simulate performance degradation and adaptation
    const challengingTasks = Array(20).fill(null).map((_, i) => ({
      description: `Complex integration test ${i}`,
      type: 'testing',
      priority: 'high',
      context: {
        complexity: 'very_high',
        timeout: 5000,
        retries: 0
      }
    }));

    console.log('Submitting challenging tasks to trigger adaptation...');
    
    // Execute in batches to observe adaptation
    for (let batch = 0; batch < 4; batch++) {
      const batchTasks = challengingTasks.slice(batch * 5, (batch + 1) * 5);
      await Promise.all(batchTasks.map(task => system.executeTask(task)));
      
      // Check for adaptations
      const metrics = learningSystem.getMetrics();
      console.log(`Batch ${batch + 1}: Adaptations made: ${metrics.adaptationsMade}`);
    }

    console.log('\n\nPhase 4: Knowledge Sharing\n');

    // Find best and worst performing agents
    let bestAgent = null;
    let worstAgent = null;
    let bestPerformance = 0;
    let worstPerformance = 1;

    for (let i = 1; i <= 6; i++) {
      const insights = learningSystem.getLearningInsights(i);
      if (insights && insights.performance.totalTasks > 10) {
        if (insights.performance.successRate > bestPerformance) {
          bestPerformance = insights.performance.successRate;
          bestAgent = i;
        }
        if (insights.performance.successRate < worstPerformance) {
          worstPerformance = insights.performance.successRate;
          worstAgent = i;
        }
      }
    }

    if (bestAgent && worstAgent && bestAgent !== worstAgent) {
      console.log(`Sharing knowledge from Agent ${bestAgent} (${(bestPerformance * 100).toFixed(1)}% success)`);
      console.log(`to Agent ${worstAgent} (${(worstPerformance * 100).toFixed(1)}% success)`);
      
      await learningSystem.shareKnowledge(bestAgent, worstAgent, 'patterns');
      await learningSystem.shareKnowledge(bestAgent, worstAgent, 'strategies');
      
      console.log('✓ Knowledge transfer completed');
    }

    console.log('\n\nPhase 5: Learning Recommendations\n');

    // Get recommendations for each agent
    for (let i = 1; i <= 6; i++) {
      const insights = learningSystem.getLearningInsights(i);
      if (insights && insights.recommendations.length > 0) {
        console.log(`\nRecommendations for Agent ${i}:`);
        insights.recommendations.forEach(rec => {
          console.log(`  - [${rec.type}] ${rec.suggestion} (confidence: ${(rec.confidence * 100).toFixed(0)}%)`);
        });
      }
    }

    // Final learning summary
    console.log('\n\n=== Learning Summary ===\n');
    
    const finalMetrics = learningSystem.getMetrics();
    console.log('Overall Learning Metrics:');
    console.log(`- Patterns Learned: ${finalMetrics.patternsLearned}`);
    console.log(`- Adaptations Made: ${finalMetrics.adaptationsMade}`);
    console.log(`- Performance Improvements: ${finalMetrics.performanceImprovements}`);
    console.log(`- Knowledge Shared: ${finalMetrics.knowledgeShared}`);

    // Performance comparison
    console.log('\nAgent Performance Evolution:');
    for (let i = 1; i <= 6; i++) {
      const insights = learningSystem.getLearningInsights(i);
      if (insights && insights.improvement) {
        const successChange = insights.improvement.successRateChange * 100;
        const durationChange = insights.improvement.durationChange;
        
        console.log(`Agent ${i}:`);
        console.log(`  - Success Rate: ${successChange > 0 ? '+' : ''}${successChange.toFixed(1)}%`);
        console.log(`  - Response Time: ${durationChange > 0 ? '+' : ''}${durationChange.toFixed(0)}ms`);
      }
    }

    // Test improved performance
    console.log('\n\nPhase 6: Performance Validation\n');
    
    const validationTasks = learningTasks.slice(0, 10).map(task => ({
      ...task,
      description: task.description + ' (validation)'
    }));
    
    console.log('Running validation tasks...');
    const validationResults = await Promise.all(
      validationTasks.map(task => system.executeTask(task))
    );
    
    const validationSuccess = validationResults.filter(r => r.success).length;
    const avgDuration = validationResults.reduce((sum, r) => sum + r.duration, 0) / validationResults.length;
    
    console.log(`✓ Validation Success Rate: ${(validationSuccess / validationTasks.length * 100).toFixed(1)}%`);
    console.log(`✓ Average Duration: ${avgDuration.toFixed(0)}ms`);

    console.log('\n=== Key Learning Outcomes ===');
    console.log('✓ Agents successfully learned from task execution patterns');
    console.log('✓ Performance adaptations were triggered based on learning');
    console.log('✓ Knowledge was shared between high and low performing agents');
    console.log('✓ System generated actionable recommendations');
    console.log('✓ Learning data persisted for future sessions');

  } catch (error) {
    console.error('Error in learning demo:', error);
  } finally {
    await learningSystem.shutdown();
    await system.shutdown();
    console.log('\n✓ Demo completed');
  }
}

// Run the demo
runLearningDemo().catch(console.error);