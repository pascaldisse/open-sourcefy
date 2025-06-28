import { MultiAgentSystem } from '../src/index.js';

/**
 * Example 3: Comprehensive Code Review Workflow
 * 
 * This example demonstrates:
 * - Full code review process
 * - Quality assurance integration
 * - Conditional workflow execution
 * - Intervention handling
 */

async function runCodeReviewWorkflow() {
  console.log('=== Comprehensive Code Review Workflow ===\n');

  const system = new MultiAgentSystem();
  
  try {
    await system.initialize();

    // Start with quality conditions
    console.log('1. Starting system with quality conditions...');
    await system.start({
      conditions: [
        {
          id: 'code_quality',
          type: 'boolean',
          check: 'code_review_approved',
          description: 'Code must pass quality review'
        },
        {
          id: 'security_check',
          type: 'boolean', 
          check: 'security_validated',
          description: 'No security vulnerabilities allowed'
        }
      ]
    });
    console.log('✓ System started with quality gates\n');

    // Complex code to review
    const complexCode = `
      const express = require('express');
      const app = express();
      
      // User authentication endpoint
      app.post('/login', (req, res) => {
        const { username, password } = req.body;
        
        // Check credentials directly in code (bad practice)
        if (username === 'admin' && password === 'password123') {
          const token = Math.random().toString(36);
          res.json({ token, user: username });
        } else {
          res.status(401).json({ error: 'Invalid credentials' });
        }
      });
      
      // Get user data
      app.get('/users/:id', async (req, res) => {
        const userId = req.params.id;
        
        // SQL query with potential injection vulnerability
        const query = \`SELECT * FROM users WHERE id = \${userId}\`;
        
        try {
          const result = await db.query(query);
          res.json(result);
        } catch (error) {
          console.log(error); // Logging sensitive errors
          res.status(500).json({ error: error.message });
        }
      });
      
      app.listen(3000);
    `;

    console.log('2. Phase 1: Initial Code Review\n');

    // Step 1: Code Review
    const reviewResult = await system.executeTask({
      description: 'Perform comprehensive code review',
      type: 'code_review',
      priority: 'high',
      context: {
        code: complexCode,
        focusAreas: ['security', 'best-practices', 'performance', 'error-handling']
      }
    });

    console.log('✓ Initial review completed');
    console.log('Issues found:', reviewResult.result.includes('vulnerability') ? 'Yes' : 'No');

    // Step 2: Security Audit
    console.log('\n3. Phase 2: Security Audit\n');
    
    const securityResult = await system.executeTask({
      description: 'Perform security audit on the code',
      type: 'security_audit',
      priority: 'critical',
      context: {
        code: complexCode,
        reviewFindings: reviewResult.result
      }
    });

    console.log('✓ Security audit completed');
    
    // Step 3: Quality Validation
    console.log('\n4. Phase 3: Quality Assurance\n');
    
    const qaValidation = await system.qualityAssurance.validateTaskResult({
      taskId: reviewResult.taskId,
      agentId: reviewResult.agentId,
      success: reviewResult.success,
      result: reviewResult.result,
      duration: reviewResult.duration
    }, 'code');

    console.log('Quality Score:', qaValidation.overallScore.toFixed(2));
    console.log('Quality Level:', qaValidation.qualityLevel);
    console.log('Critical Issues:', qaValidation.criticalIssues.length);

    if (qaValidation.criticalIssues.length > 0) {
      console.log('\n⚠️  Critical issues detected:');
      qaValidation.criticalIssues.forEach(issue => {
        console.log(`  - ${issue.rule}: ${issue.issue}`);
      });
    }

    // Step 4: Generate Recommendations
    console.log('\n5. Phase 4: Improvement Recommendations\n');

    if (qaValidation.recommendations.length > 0) {
      console.log('Recommendations:');
      qaValidation.recommendations.forEach((rec, index) => {
        console.log(`${index + 1}. [${rec.priority}] ${rec.suggestion}`);
      });
    }

    // Step 5: Fix Critical Issues
    console.log('\n6. Phase 5: Automated Fixes\n');

    const fixResult = await system.executeTask({
      description: 'Fix the security vulnerabilities and code issues',
      type: 'bug_fixing',
      priority: 'critical',
      context: {
        originalCode: complexCode,
        securityIssues: securityResult.result,
        reviewIssues: reviewResult.result
      },
      requirements: 'Fix all security vulnerabilities, improve error handling, follow best practices'
    });

    console.log('✓ Fixes applied');

    // Step 6: Re-review Fixed Code
    console.log('\n7. Phase 6: Verification\n');

    const verifyResult = await system.executeTask({
      description: 'Review the fixed code to ensure all issues are resolved',
      type: 'code_review',
      priority: 'high',
      context: {
        code: fixResult.result,
        previousIssues: reviewResult.result
      }
    });

    console.log('✓ Verification completed');

    // Final quality check
    const finalQA = await system.qualityAssurance.validateTaskResult({
      taskId: verifyResult.taskId,
      agentId: verifyResult.agentId,
      success: verifyResult.success,
      result: verifyResult.result,
      duration: verifyResult.duration
    }, 'code');

    console.log('\n=== Final Results ===');
    console.log('Initial Quality Score:', qaValidation.overallScore.toFixed(2));
    console.log('Final Quality Score:', finalQA.overallScore.toFixed(2));
    console.log('Improvement:', `+${(finalQA.overallScore - qaValidation.overallScore).toFixed(2)} points`);
    console.log('Final Quality Level:', finalQA.qualityLevel);
    console.log('All Security Issues Fixed:', finalQA.criticalIssues.length === 0 ? 'Yes' : 'No');

    // Check conditions
    const conditionStatus = system.conditionManager.getConditionStatus();
    console.log('\nCondition Status:');
    Object.entries(conditionStatus.conditions).forEach(([id, condition]) => {
      console.log(`- ${condition.description}: ${condition.status}`);
    });

    // System monitoring snapshot
    const snapshot = system.systemMonitor.getCurrentSnapshot();
    if (snapshot) {
      console.log('\nSystem Health:');
      console.log('- CPU Usage:', snapshot.system?.cpu?.usage?.toFixed(2) + '%' || 'N/A');
      console.log('- Memory Usage:', snapshot.system?.memory?.usage?.toFixed(2) + '%' || 'N/A');
      console.log('- Active Agents:', snapshot.agents?.summary?.activeAgents || 0);
    }

  } catch (error) {
    console.error('Error in code review workflow:', error);
  } finally {
    await system.shutdown();
    console.log('\n✓ Workflow completed');
  }
}

// Run the example
runCodeReviewWorkflow().catch(console.error);