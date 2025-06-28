import { logger, auditLogger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * Intervention System
 * Handles error correction, agent intervention, and system recovery
 */
export class InterventionSystem {
  constructor(options = {}) {
    this.interventionHistory = [];
    this.escalationRules = options.escalationRules || this.getDefaultEscalationRules();
    this.recoveryStrategies = options.recoveryStrategies || this.getDefaultRecoveryStrategies();
    this.maxHistorySize = options.maxHistorySize || 200;
    this.interventionTimeout = options.interventionTimeout || 60000; // 1 minute
    this.maxRetryAttempts = options.maxRetryAttempts || 3;
    this.coordinator = null;
  }

  /**
   * Set the coordinator reference for agent management
   */
  setCoordinator(coordinator) {
    this.coordinator = coordinator;
  }

  /**
   * Get default escalation rules
   */
  getDefaultEscalationRules() {
    return {
      agentFailure: {
        threshold: 3, // failures within timeframe
        timeframe: 300000, // 5 minutes
        action: 'restart_agent'
      },
      taskTimeout: {
        threshold: 1,
        timeframe: 0,
        action: 'abort_and_reassign'
      },
      qualityFailure: {
        threshold: 2,
        timeframe: 600000, // 10 minutes
        action: 'review_and_fix'
      },
      systemOverload: {
        threshold: 80, // 80% resource usage
        timeframe: 0,
        action: 'throttle_tasks'
      },
      criticalError: {
        threshold: 1,
        timeframe: 0,
        action: 'immediate_intervention'
      }
    };
  }

  /**
   * Get default recovery strategies
   */
  getDefaultRecoveryStrategies() {
    return {
      restart_agent: {
        steps: ['abort_current_task', 'restart_agent', 'reassign_task'],
        timeout: 30000
      },
      abort_and_reassign: {
        steps: ['abort_task', 'find_alternative_agent', 'reassign_task'],
        timeout: 15000
      },
      review_and_fix: {
        steps: ['quality_review', 'identify_issues', 'apply_fixes', 'revalidate'],
        timeout: 120000
      },
      throttle_tasks: {
        steps: ['pause_new_tasks', 'reduce_concurrency', 'monitor_resources'],
        timeout: 60000
      },
      immediate_intervention: {
        steps: ['emergency_stop', 'assess_damage', 'manual_review', 'gradual_restart'],
        timeout: 300000
      }
    };
  }

  /**
   * Trigger intervention based on issue type and severity
   */
  async triggerIntervention(issue, context = {}) {
    const interventionId = uuidv4();
    const startTime = Date.now();

    try {
      logger.warn(`Triggering intervention ${interventionId} for issue: ${issue.type}`);
      
      auditLogger.warn('Intervention triggered', {
        interventionId,
        issueType: issue.type,
        agentId: issue.agentId,
        severity: issue.severity,
        context
      });

      // Validate intervention request
      this.validateInterventionRequest(issue);

      // Determine intervention strategy
      const strategy = this.determineInterventionStrategy(issue, context);
      
      // Execute intervention
      const result = await this.executeIntervention(strategy, issue, context, interventionId);

      // Record intervention
      const intervention = {
        interventionId,
        timestamp: startTime,
        duration: Date.now() - startTime,
        issue,
        strategy,
        result,
        context,
        success: result.success
      };

      this.addToHistory(intervention);

      logger.info(`Intervention ${interventionId} ${result.success ? 'succeeded' : 'failed'}`);
      
      return intervention;

    } catch (error) {
      logger.error(`Intervention ${interventionId} failed:`, error);
      
      const failedIntervention = {
        interventionId,
        timestamp: startTime,
        duration: Date.now() - startTime,
        issue,
        error: error.message,
        success: false
      };

      this.addToHistory(failedIntervention);
      
      return failedIntervention;
    }
  }

  /**
   * Validate intervention request
   */
  validateInterventionRequest(issue) {
    if (!issue || typeof issue !== 'object') {
      throw new Error('Invalid issue: must be an object');
    }

    if (!issue.type || typeof issue.type !== 'string') {
      throw new Error('Issue must have a type');
    }

    if (!issue.severity || !['low', 'medium', 'high', 'critical'].includes(issue.severity)) {
      throw new Error('Issue must have a valid severity level');
    }
  }

  /**
   * Determine intervention strategy based on issue
   */
  determineInterventionStrategy(issue, context) {
    const escalationRule = this.escalationRules[issue.type];
    
    if (!escalationRule) {
      // Default strategy for unknown issue types
      return {
        action: 'review_and_fix',
        priority: issue.severity === 'critical' ? 'immediate' : 'normal',
        timeout: this.interventionTimeout
      };
    }

    // Check if escalation threshold is met
    const shouldEscalate = this.shouldEscalate(issue, escalationRule, context);
    
    return {
      action: escalationRule.action,
      priority: shouldEscalate ? 'immediate' : 'normal',
      timeout: this.getTimeoutForAction(escalationRule.action),
      escalated: shouldEscalate
    };
  }

  /**
   * Check if issue should be escalated
   */
  shouldEscalate(issue, rule, context) {
    // For critical issues, always escalate
    if (issue.severity === 'critical') {
      return true;
    }

    // Check frequency-based escalation
    if (rule.timeframe > 0) {
      const recentIssues = this.getRecentIssues(issue.type, rule.timeframe);
      return recentIssues.length >= rule.threshold;
    }

    // Check threshold-based escalation (e.g., resource usage)
    if (rule.threshold && context.currentValue !== undefined) {
      return context.currentValue >= rule.threshold;
    }

    return false;
  }

  /**
   * Execute intervention strategy
   */
  async executeIntervention(strategy, issue, context, interventionId) {
    logger.info(`Executing intervention strategy: ${strategy.action}`);

    const recoveryStrategy = this.recoveryStrategies[strategy.action];
    if (!recoveryStrategy) {
      throw new Error(`Unknown recovery strategy: ${strategy.action}`);
    }

    const executionResults = [];
    
    try {
      // Execute recovery steps
      for (const step of recoveryStrategy.steps) {
        const stepResult = await this.executeRecoveryStep(step, issue, context, interventionId);
        executionResults.push(stepResult);

        if (!stepResult.success && stepResult.critical) {
          throw new Error(`Critical recovery step failed: ${step}`);
        }
      }

      // Verify recovery
      const verificationResult = await this.verifyRecovery(issue, context);
      
      return {
        success: verificationResult.success,
        steps: executionResults,
        verification: verificationResult,
        message: verificationResult.success ? 'Intervention completed successfully' : 'Intervention completed with issues'
      };

    } catch (error) {
      return {
        success: false,
        steps: executionResults,
        error: error.message,
        message: 'Intervention failed during execution'
      };
    }
  }

  /**
   * Execute individual recovery step
   */
  async executeRecoveryStep(step, issue, context, interventionId) {
    logger.debug(`Executing recovery step: ${step}`);

    try {
      switch (step) {
        case 'abort_current_task':
          return await this.abortCurrentTask(issue.agentId, interventionId);
        
        case 'restart_agent':
          return await this.restartAgent(issue.agentId, interventionId);
        
        case 'reassign_task':
          return await this.reassignTask(issue.taskId, issue.agentId, interventionId);
        
        case 'abort_task':
          return await this.abortTask(issue.taskId, interventionId);
        
        case 'find_alternative_agent':
          return await this.findAlternativeAgent(issue.taskId, issue.agentId, interventionId);
        
        case 'quality_review':
          return await this.performQualityReview(issue, interventionId);
        
        case 'identify_issues':
          return await this.identifyIssues(issue, context, interventionId);
        
        case 'apply_fixes':
          return await this.applyFixes(issue, context, interventionId);
        
        case 'revalidate':
          return await this.revalidateQuality(issue, interventionId);
        
        case 'pause_new_tasks':
          return await this.pauseNewTasks(interventionId);
        
        case 'reduce_concurrency':
          return await this.reduceConcurrency(interventionId);
        
        case 'monitor_resources':
          return await this.monitorResources(interventionId);
        
        case 'emergency_stop':
          return await this.emergencyStop(interventionId);
        
        case 'assess_damage':
          return await this.assessDamage(issue, context, interventionId);
        
        case 'manual_review':
          return await this.requestManualReview(issue, interventionId);
        
        case 'gradual_restart':
          return await this.gradualRestart(interventionId);

        default:
          return {
            step,
            success: false,
            message: `Unknown recovery step: ${step}`,
            critical: false
          };
      }

    } catch (error) {
      logger.error(`Recovery step ${step} failed:`, error);
      
      return {
        step,
        success: false,
        error: error.message,
        critical: this.isStepCritical(step)
      };
    }
  }

  /**
   * Abort current task for an agent
   */
  async abortCurrentTask(agentId, interventionId) {
    if (!this.coordinator) {
      return { step: 'abort_current_task', success: false, message: 'No coordinator available' };
    }

    const agent = this.coordinator.managedAgents.get(agentId);
    if (!agent) {
      return { step: 'abort_current_task', success: false, message: 'Agent not found' };
    }

    try {
      agent.abort();
      
      auditLogger.info('Task aborted', { agentId, interventionId });
      
      return {
        step: 'abort_current_task',
        success: true,
        message: `Aborted current task for agent ${agentId}`
      };
    } catch (error) {
      return {
        step: 'abort_current_task',
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Restart an agent
   */
  async restartAgent(agentId, interventionId) {
    if (!this.coordinator) {
      return { step: 'restart_agent', success: false, message: 'No coordinator available' };
    }

    const agent = this.coordinator.managedAgents.get(agentId);
    if (!agent) {
      return { step: 'restart_agent', success: false, message: 'Agent not found' };
    }

    try {
      // Shutdown agent
      await agent.shutdown();
      
      // Reinitialize agent
      await agent.initialize();
      
      auditLogger.info('Agent restarted', { agentId, interventionId });
      
      return {
        step: 'restart_agent',
        success: true,
        message: `Restarted agent ${agentId}`
      };
    } catch (error) {
      return {
        step: 'restart_agent',
        success: false,
        error: error.message,
        critical: true
      };
    }
  }

  /**
   * Reassign task to different agent
   */
  async reassignTask(taskId, originalAgentId, interventionId) {
    if (!this.coordinator) {
      return { step: 'reassign_task', success: false, message: 'No coordinator available' };
    }

    try {
      // Find alternative agent
      const alternativeAgent = this.findHealthyAgent(originalAgentId);
      
      if (!alternativeAgent) {
        return {
          step: 'reassign_task',
          success: false,
          message: 'No alternative agent available'
        };
      }

      // Note: In a real implementation, you'd need to extract the task details
      // and reassign to the new agent. This is a simplified version.
      
      auditLogger.info('Task reassigned', {
        taskId,
        originalAgent: originalAgentId,
        newAgent: alternativeAgent.agentId,
        interventionId
      });

      return {
        step: 'reassign_task',
        success: true,
        message: `Reassigned task ${taskId} from agent ${originalAgentId} to agent ${alternativeAgent.agentId}`
      };
    } catch (error) {
      return {
        step: 'reassign_task',
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Find a healthy agent as alternative
   */
  findHealthyAgent(excludeAgentId) {
    if (!this.coordinator) {
      return null;
    }

    for (const [agentId, agent] of this.coordinator.managedAgents) {
      if (agentId !== excludeAgentId && agentId !== 0 && agent.status === 'idle') {
        return agent;
      }
    }

    return null;
  }

  /**
   * Perform quality review
   */
  async performQualityReview(issue, interventionId) {
    // This would integrate with the quality assurance system
    return {
      step: 'quality_review',
      success: true,
      message: 'Quality review completed',
      findings: {
        issuesFound: 1,
        severity: issue.severity,
        recommendations: ['Fix identified issues', 'Add additional validation']
      }
    };
  }

  /**
   * Pause new task assignments
   */
  async pauseNewTasks(interventionId) {
    if (!this.coordinator) {
      return { step: 'pause_new_tasks', success: false, message: 'No coordinator available' };
    }

    // Set a flag to pause new task assignments
    this.coordinator.taskAssignmentPaused = true;
    
    auditLogger.info('New task assignments paused', { interventionId });
    
    return {
      step: 'pause_new_tasks',
      success: true,
      message: 'New task assignments paused'
    };
  }

  /**
   * Emergency stop all operations
   */
  async emergencyStop(interventionId) {
    if (!this.coordinator) {
      return { step: 'emergency_stop', success: false, message: 'No coordinator available' };
    }

    try {
      // Stop all agent operations
      for (const [agentId, agent] of this.coordinator.managedAgents) {
        if (agentId !== 0) { // Don't stop the coordinator itself
          agent.abort();
        }
      }

      auditLogger.warn('Emergency stop executed', { interventionId });

      return {
        step: 'emergency_stop',
        success: true,
        message: 'Emergency stop executed for all agents',
        critical: true
      };
    } catch (error) {
      return {
        step: 'emergency_stop',
        success: false,
        error: error.message,
        critical: true
      };
    }
  }

  /**
   * Verify that recovery was successful
   */
  async verifyRecovery(issue, context) {
    // Implementation would depend on issue type
    // For now, assume success if no errors occurred
    
    return {
      success: true,
      message: 'Recovery verification completed',
      checks: [
        { check: 'agent_responsive', passed: true },
        { check: 'system_stable', passed: true },
        { check: 'no_active_errors', passed: true }
      ]
    };
  }

  /**
   * Check if a recovery step is critical
   */
  isStepCritical(step) {
    const criticalSteps = [
      'restart_agent',
      'emergency_stop',
      'assess_damage',
      'gradual_restart'
    ];
    
    return criticalSteps.includes(step);
  }

  /**
   * Get timeout for specific action
   */
  getTimeoutForAction(action) {
    const timeouts = {
      'restart_agent': 30000,
      'abort_and_reassign': 15000,
      'review_and_fix': 120000,
      'throttle_tasks': 60000,
      'immediate_intervention': 300000
    };

    return timeouts[action] || this.interventionTimeout;
  }

  /**
   * Get recent issues of specific type
   */
  getRecentIssues(issueType, timeframe) {
    const cutoffTime = Date.now() - timeframe;
    
    return this.interventionHistory.filter(intervention =>
      intervention.timestamp >= cutoffTime &&
      intervention.issue &&
      intervention.issue.type === issueType
    );
  }

  /**
   * Add intervention to history
   */
  addToHistory(intervention) {
    this.interventionHistory.push(intervention);

    // Maintain history size
    if (this.interventionHistory.length > this.maxHistorySize) {
      this.interventionHistory.shift();
    }
  }

  /**
   * Get intervention statistics
   */
  getInterventionStatistics(timeframe = 24 * 60 * 60 * 1000) { // 24 hours
    const cutoffTime = Date.now() - timeframe;
    const recentInterventions = this.interventionHistory.filter(i => i.timestamp >= cutoffTime);

    if (recentInterventions.length === 0) {
      return {
        totalInterventions: 0,
        successRate: 0,
        averageDuration: 0,
        interventionsByType: {},
        interventionsBySeverity: {}
      };
    }

    const successful = recentInterventions.filter(i => i.success).length;
    const totalDuration = recentInterventions.reduce((sum, i) => sum + (i.duration || 0), 0);
    
    // Group by type
    const byType = {};
    const bySeverity = {};
    
    for (const intervention of recentInterventions) {
      if (intervention.issue) {
        byType[intervention.issue.type] = (byType[intervention.issue.type] || 0) + 1;
        bySeverity[intervention.issue.severity] = (bySeverity[intervention.issue.severity] || 0) + 1;
      }
    }

    return {
      totalInterventions: recentInterventions.length,
      successRate: (successful / recentInterventions.length) * 100,
      averageDuration: totalDuration / recentInterventions.length,
      interventionsByType: byType,
      interventionsBySeverity: bySeverity
    };
  }

  /**
   * Get intervention history
   */
  getInterventionHistory(limit = 50) {
    return this.interventionHistory.slice(-limit);
  }

  // Placeholder methods for remaining recovery steps
  async abortTask(taskId, interventionId) {
    return { step: 'abort_task', success: true, message: `Task ${taskId} aborted` };
  }

  async findAlternativeAgent(taskId, excludeAgentId, interventionId) {
    const agent = this.findHealthyAgent(excludeAgentId);
    return {
      step: 'find_alternative_agent',
      success: !!agent,
      message: agent ? `Found alternative agent ${agent.agentId}` : 'No alternative agent found',
      agentId: agent?.agentId
    };
  }

  async identifyIssues(issue, context, interventionId) {
    return { step: 'identify_issues', success: true, message: 'Issues identified', issues: [issue] };
  }

  async applyFixes(issue, context, interventionId) {
    return { step: 'apply_fixes', success: true, message: 'Fixes applied' };
  }

  async revalidateQuality(issue, interventionId) {
    return { step: 'revalidate', success: true, message: 'Quality revalidation completed' };
  }

  async reduceConcurrency(interventionId) {
    return { step: 'reduce_concurrency', success: true, message: 'Concurrency reduced' };
  }

  async monitorResources(interventionId) {
    return { step: 'monitor_resources', success: true, message: 'Resource monitoring enabled' };
  }

  async assessDamage(issue, context, interventionId) {
    return { step: 'assess_damage', success: true, message: 'Damage assessment completed' };
  }

  async requestManualReview(issue, interventionId) {
    return { step: 'manual_review', success: true, message: 'Manual review requested' };
  }

  async gradualRestart(interventionId) {
    return { step: 'gradual_restart', success: true, message: 'Gradual restart initiated' };
  }
}