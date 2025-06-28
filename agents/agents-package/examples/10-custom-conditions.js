import { MultiAgentSystem } from '../src/index.js';

/**
 * Example 10: Custom Conditions and Complex Workflows
 * 
 * This example demonstrates:
 * - Custom boolean conditions
 * - LLM-validated conditions
 * - Complex conditional workflows
 * - Dynamic condition management
 * - Intervention and recovery
 */

async function runCustomConditionsExample() {
  console.log('=== Custom Conditions Example ===\n');

  const system = new MultiAgentSystem();
  
  try {
    await system.initialize();

    console.log('1. Setting up custom conditions...\n');

    // Define custom conditions
    const conditions = [
      {
        id: 'min_test_coverage',
        type: 'boolean',
        check: 'test_coverage_80',
        description: 'Test coverage must be at least 80%'
      },
      {
        id: 'performance_threshold',
        type: 'boolean',
        check: 'response_time_under_200ms',
        description: 'All operations must complete under 200ms'
      },
      {
        id: 'code_quality_llm',
        type: 'llm_validated',
        description: 'Code quality must meet senior developer standards',
        criteria: 'Evaluate if the code follows SOLID principles, has proper error handling, and is production-ready'
      }
    ];

    // Start system with conditions
    await system.start({ conditions });
    console.log('✓ System started with custom conditions\n');

    // Project simulation
    console.log('2. Starting project workflow...\n');

    // Step 1: Create initial code
    console.log('Step 1: Generate initial implementation');
    const implementationTask = {
      description: 'Create a user service with CRUD operations',
      type: 'code_generation',
      priority: 'high',
      context: {
        requirements: [
          'User model with id, name, email, createdAt',
          'CRUD operations (create, read, update, delete)',
          'Input validation',
          'Error handling',
          'TypeScript interfaces'
        ]
      }
    };

    // Since we don't have a code generation agent, we'll simulate with Agent 3
    const codeResult = await system.executeTask({
      ...implementationTask,
      type: 'bug_fixing' // Using bug hunter to simulate
    });

    console.log('✓ Initial implementation created\n');

    // Step 2: Write tests
    console.log('Step 2: Generate test suite');
    const testResult = await system.executeTask({
      description: 'Write comprehensive tests for the user service',
      type: 'testing',
      priority: 'high',
      context: {
        code: codeResult.result,
        targetCoverage: 85
      }
    });

    console.log('✓ Test suite generated');

    // Check test coverage condition
    const testMetrics = testResult.testMetrics || { coverage: 75 };
    console.log('Test Coverage:', testMetrics.coverage + '%');

    // Step 3: Quality review
    console.log('\nStep 3: Quality validation');
    const qaResult = await system.qualityAssurance.validateTaskResult({
      taskId: codeResult.taskId,
      agentId: codeResult.agentId,
      success: true,
      result: codeResult.result,
      duration: codeResult.duration,
      testMetrics
    });

    console.log('Quality Score:', qaResult.overallScore.toFixed(2));

    // Step 4: Performance testing
    console.log('\nStep 4: Performance analysis');
    const perfResult = await system.executeTask({
      description: 'Analyze performance characteristics of the code',
      type: 'performance_optimization',
      priority: 'medium',
      context: {
        code: codeResult.result,
        requirements: {
          maxResponseTime: 200,
          targetThroughput: 1000
        }
      }
    });

    console.log('✓ Performance analysis completed');

    // Step 5: Evaluate conditions
    console.log('\n3. Evaluating custom conditions...\n');

    // Simulate context for condition evaluation
    const evaluationContext = {
      systemState: system.getSystemStatus(),
      testResults: [{ success: testMetrics.coverage >= 80 }],
      performanceMetrics: {
        averageResponseTime: 150, // Simulated
        errorRate: 0.5
      },
      documentationMetrics: {
        completeness: 85
      },
      qualityMetrics: {
        overallScore: qaResult.overallScore,
        criticalIssues: qaResult.criticalIssues.length
      },
      // Add LLM agent for LLM-validated conditions
      llmAgent: system.coordinator
    };

    const evaluation = await system.conditionManager.evaluateAllConditions(evaluationContext);

    console.log('Condition Evaluation Results:');
    Object.entries(evaluation.results).forEach(([conditionId, result]) => {
      const condition = conditions.find(c => c.id === conditionId);
      console.log(`- ${condition.description}: ${result.met ? '✓ PASSED' : '✗ FAILED'}`);
      if (result.details) {
        console.log(`  Details:`, result.details);
      }
    });

    console.log(`\nAll conditions met: ${evaluation.allMet ? 'YES' : 'NO'}`);

    // Step 6: Handle failed conditions
    if (!evaluation.allMet) {
      console.log('\n4. Triggering intervention for failed conditions...\n');

      const failedConditions = Object.entries(evaluation.results)
        .filter(([_, result]) => !result.met)
        .map(([id, _]) => id);

      for (const conditionId of failedConditions) {
        const condition = conditions.find(c => c.id === conditionId);
        console.log(`Addressing: ${condition.description}`);

        // Trigger intervention
        const intervention = await system.interventionSystem.triggerIntervention({
          type: 'qualityFailure',
          severity: 'medium',
          agentId: codeResult.agentId,
          description: `Condition failed: ${condition.description}`
        });

        console.log(`✓ Intervention ${intervention.interventionId} executed`);
        console.log(`  Strategy: ${intervention.strategy.action}`);
        console.log(`  Success: ${intervention.success}`);
      }
    }

    // Step 7: Monitor system health
    console.log('\n5. System monitoring and alerts...\n');

    // Start monitoring
    system.systemMonitor.startMonitoring();

    // Wait for monitoring cycle
    await new Promise(resolve => setTimeout(resolve, 1000));

    const snapshot = system.systemMonitor.getCurrentSnapshot();
    const alerts = system.systemMonitor.getActiveAlerts();

    console.log('System Metrics:');
    console.log('- Total Agents:', snapshot?.agents?.summary?.totalAgents || 0);
    console.log('- Active Agents:', snapshot?.agents?.summary?.activeAgents || 0);
    console.log('- Average Response Time:', 
      snapshot?.agents?.summary?.averageResponseTime?.toFixed(2) + 'ms' || 'N/A');
    console.log('- Active Alerts:', alerts.length);

    if (alerts.length > 0) {
      console.log('\nActive Alerts:');
      alerts.forEach(alert => {
        console.log(`- [${alert.severity}] ${alert.category}: ${alert.message}`);
      });
    }

    // Stop monitoring
    system.systemMonitor.stopMonitoring();

    // Final summary
    console.log('\n=== Workflow Summary ===');
    const conditionStatus = system.conditionManager.getConditionStatus();
    console.log('Total Conditions:', conditionStatus.totalConditions);
    console.log('Met Conditions:', conditionStatus.metConditions);
    console.log('Pending Conditions:', conditionStatus.pendingConditions);
    
    const stats = system.messageBus.getStatistics();
    console.log('\nCommunication Stats:');
    console.log('- Total Messages:', stats.totalMessages);
    console.log('- Message Types:', Object.keys(stats.messagesByType).join(', '));

    const interventionStats = system.interventionSystem.getInterventionStatistics();
    console.log('\nIntervention Stats:');
    console.log('- Total Interventions:', interventionStats.totalInterventions);
    console.log('- Success Rate:', interventionStats.successRate.toFixed(2) + '%');

  } catch (error) {
    console.error('Error in custom conditions example:', error);
  } finally {
    await system.shutdown();
    console.log('\n✓ Example completed');
  }
}

// Run the example
runCustomConditionsExample().catch(console.error);