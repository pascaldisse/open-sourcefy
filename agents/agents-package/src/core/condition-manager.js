import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * Condition Manager
 * Handles boolean and LLM-validated conditions for system loops
 */
export class ConditionManager {
  constructor(options = {}) {
    this.conditions = new Map();
    this.evaluationHistory = [];
    this.maxHistorySize = options.maxHistorySize || 200;
    this.evaluationInterval = options.evaluationInterval || 5000; // 5 seconds
    this.llmValidationTimeout = options.llmValidationTimeout || 30000; // 30 seconds
  }

  /**
   * Add a condition to be monitored
   */
  addCondition(condition) {
    if (!this.validateCondition(condition)) {
      throw new Error('Invalid condition format');
    }

    const conditionId = condition.id || uuidv4();
    const enhancedCondition = {
      ...condition,
      id: conditionId,
      addedAt: Date.now(),
      lastEvaluated: null,
      lastResult: null,
      evaluationCount: 0,
      status: 'pending'
    };

    this.conditions.set(conditionId, enhancedCondition);
    logger.info(`Condition added: ${conditionId} - ${condition.description}`);

    return conditionId;
  }

  /**
   * Remove a condition from monitoring
   */
  removeCondition(conditionId) {
    const removed = this.conditions.delete(conditionId);
    if (removed) {
      logger.info(`Condition removed: ${conditionId}`);
    }
    return removed;
  }

  /**
   * Validate condition format
   */
  validateCondition(condition) {
    if (!condition || typeof condition !== 'object') {
      return false;
    }

    if (!condition.type || !['boolean', 'llm_validated'].includes(condition.type)) {
      return false;
    }

    if (!condition.description || typeof condition.description !== 'string') {
      return false;
    }

    if (condition.type === 'boolean' && !condition.check) {
      return false;
    }

    return true;
  }

  /**
   * Evaluate all conditions
   */
  async evaluateAllConditions(context = {}) {
    const evaluationId = uuidv4();
    const startTime = Date.now();
    const results = {};

    logger.debug(`Starting condition evaluation ${evaluationId}`);

    for (const [conditionId, condition] of this.conditions) {
      try {
        const result = await this.evaluateCondition(condition, context, evaluationId);
        results[conditionId] = result;

        // Update condition state
        condition.lastEvaluated = Date.now();
        condition.lastResult = result.met;
        condition.evaluationCount++;
        condition.status = result.met ? 'met' : 'not_met';

      } catch (error) {
        logger.error(`Failed to evaluate condition ${conditionId}:`, error);
        results[conditionId] = {
          met: false,
          error: error.message,
          timestamp: Date.now()
        };
        condition.status = 'error';
      }
    }

    const evaluation = {
      evaluationId,
      timestamp: startTime,
      duration: Date.now() - startTime,
      results,
      allMet: Object.values(results).every(r => r.met),
      context
    };

    this.addToHistory(evaluation);
    
    logger.debug(`Condition evaluation ${evaluationId} completed: ${evaluation.allMet ? 'ALL MET' : 'PENDING'}`);

    return evaluation;
  }

  /**
   * Evaluate a single condition
   */
  async evaluateCondition(condition, context, evaluationId) {
    logger.debug(`Evaluating condition ${condition.id}: ${condition.description}`);

    const startTime = Date.now();

    try {
      let result;

      if (condition.type === 'boolean') {
        result = await this.evaluateBooleanCondition(condition, context);
      } else if (condition.type === 'llm_validated') {
        result = await this.evaluateLLMCondition(condition, context);
      } else {
        throw new Error(`Unknown condition type: ${condition.type}`);
      }

      return {
        met: result.met,
        details: result.details || {},
        duration: Date.now() - startTime,
        timestamp: Date.now(),
        evaluationId
      };

    } catch (error) {
      return {
        met: false,
        error: error.message,
        duration: Date.now() - startTime,
        timestamp: Date.now(),
        evaluationId
      };
    }
  }

  /**
   * Evaluate boolean condition
   */
  async evaluateBooleanCondition(condition, context) {
    const check = condition.check;
    
    switch (check) {
      case 'all_tests_pass':
        return this.checkAllTestsPass(context);
      
      case 'all_agents_healthy':
        return this.checkAllAgentsHealthy(context);
      
      case 'documentation_complete':
        return this.checkDocumentationComplete(context);
      
      case 'no_critical_issues':
        return this.checkNoCriticalIssues(context);
      
      case 'deployment_successful':
        return this.checkDeploymentSuccessful(context);
      
      case 'performance_acceptable':
        return this.checkPerformanceAcceptable(context);
      
      case 'security_validated':
        return this.checkSecurityValidated(context);
      
      case 'code_review_approved':
        return this.checkCodeReviewApproved(context);

      default:
        // Try to evaluate as a custom expression
        return this.evaluateCustomBooleanCondition(condition, context);
    }
  }

  /**
   * Check if all tests pass
   */
  async checkAllTestsPass(context) {
    const systemState = context.systemState || {};
    const failedTasks = systemState.failedTasks || 0;
    
    // If we have test results in context
    if (context.testResults) {
      const allPassed = context.testResults.every(result => result.success);
      return {
        met: allPassed,
        details: {
          totalTests: context.testResults.length,
          passed: context.testResults.filter(r => r.success).length,
          failed: context.testResults.filter(r => !r.success).length
        }
      };
    }

    // Fallback to system state
    return {
      met: failedTasks === 0,
      details: { failedTasks }
    };
  }

  /**
   * Check if all agents are healthy
   */
  async checkAllAgentsHealthy(context) {
    const agentStatuses = context.agentStatuses || {};
    const unhealthyAgents = [];

    for (const [agentId, status] of Object.entries(agentStatuses)) {
      if (status.status === 'error' || status.status === 'shutdown') {
        unhealthyAgents.push(agentId);
      }
    }

    return {
      met: unhealthyAgents.length === 0,
      details: {
        totalAgents: Object.keys(agentStatuses).length,
        unhealthyAgents
      }
    };
  }

  /**
   * Check if documentation is complete
   */
  async checkDocumentationComplete(context) {
    const documentationMetrics = context.documentationMetrics || {};
    const completeness = documentationMetrics.completeness || 0;
    const target = 90; // 90% completeness required

    return {
      met: completeness >= target,
      details: {
        completeness,
        target,
        gap: Math.max(0, target - completeness)
      }
    };
  }

  /**
   * Check if there are no critical issues
   */
  async checkNoCriticalIssues(context) {
    const qualityMetrics = context.qualityMetrics || {};
    const criticalIssues = qualityMetrics.criticalIssues || 0;

    return {
      met: criticalIssues === 0,
      details: { criticalIssues }
    };
  }

  /**
   * Check if deployment was successful
   */
  async checkDeploymentSuccessful(context) {
    const deploymentStatus = context.deploymentStatus || {};
    const success = deploymentStatus.success === true;
    const healthy = deploymentStatus.healthy !== false;

    return {
      met: success && healthy,
      details: deploymentStatus
    };
  }

  /**
   * Check if performance is acceptable
   */
  async checkPerformanceAcceptable(context) {
    const performanceMetrics = context.performanceMetrics || {};
    const responseTime = performanceMetrics.averageResponseTime || 0;
    const errorRate = performanceMetrics.errorRate || 0;
    
    const responseTimeOk = responseTime <= 200; // 200ms threshold
    const errorRateOk = errorRate <= 1; // 1% error rate threshold

    return {
      met: responseTimeOk && errorRateOk,
      details: {
        responseTime,
        errorRate,
        responseTimeOk,
        errorRateOk
      }
    };
  }

  /**
   * Check if security validation passed
   */
  async checkSecurityValidated(context) {
    const securityMetrics = context.securityMetrics || {};
    const vulnerabilities = securityMetrics.vulnerabilities || 0;
    const complianceScore = securityMetrics.complianceScore || 0;

    const noVulnerabilities = vulnerabilities === 0;
    const compliant = complianceScore >= 95; // 95% compliance required

    return {
      met: noVulnerabilities && compliant,
      details: {
        vulnerabilities,
        complianceScore,
        noVulnerabilities,
        compliant
      }
    };
  }

  /**
   * Check if code review is approved
   */
  async checkCodeReviewApproved(context) {
    const reviewStatus = context.codeReviewStatus || {};
    const approved = reviewStatus.approved === true;
    const qualityScore = reviewStatus.qualityScore || 0;

    return {
      met: approved && qualityScore >= 80,
      details: reviewStatus
    };
  }

  /**
   * Evaluate custom boolean condition
   */
  async evaluateCustomBooleanCondition(condition, context) {
    // For custom conditions, look for the value in context
    const value = this.getNestedValue(context, condition.check);
    
    return {
      met: Boolean(value),
      details: { value, path: condition.check }
    };
  }

  /**
   * Evaluate LLM-validated condition
   */
  async evaluateLLMCondition(condition, context) {
    if (!context.llmAgent) {
      throw new Error('LLM agent not available for validation');
    }

    const prompt = this.buildLLMValidationPrompt(condition, context);
    
    try {
      const result = await Promise.race([
        context.llmAgent.processTask({
          description: `Evaluate condition: ${condition.description}`,
          type: 'condition_evaluation',
          context: { condition, systemContext: context },
          requirements: prompt
        }),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('LLM validation timeout')), this.llmValidationTimeout)
        )
      ]);

      if (!result.success) {
        throw new Error(`LLM validation failed: ${result.error}`);
      }

      // Parse LLM response for boolean result
      const response = result.result.toLowerCase();
      const met = response.includes('true') || response.includes('met') || response.includes('satisfied');

      return {
        met,
        details: {
          llmResponse: result.result,
          confidence: this.extractConfidence(result.result)
        }
      };

    } catch (error) {
      logger.error('LLM condition evaluation failed:', error);
      throw error;
    }
  }

  /**
   * Build prompt for LLM validation
   */
  buildLLMValidationPrompt(condition, context) {
    return `Please evaluate whether the following condition is met:

CONDITION: ${condition.description}

CONTEXT:
${JSON.stringify(context, null, 2)}

EVALUATION CRITERIA:
${condition.criteria || 'Use your best judgment based on the available context and system state.'}

Please respond with:
1. True/False - whether the condition is met
2. Confidence level (0-100%)
3. Brief explanation of your reasoning

Format your response clearly indicating whether the condition is MET or NOT MET.`;
  }

  /**
   * Extract confidence level from LLM response
   */
  extractConfidence(response) {
    const confidenceMatch = response.match(/confidence[:\s]+(\d+)%?/i);
    return confidenceMatch ? parseInt(confidenceMatch[1]) : 50;
  }

  /**
   * Get nested value from object using dot notation
   */
  getNestedValue(obj, path) {
    return path.split('.').reduce((current, key) => {
      return current && current[key] !== undefined ? current[key] : undefined;
    }, obj);
  }

  /**
   * Add evaluation to history
   */
  addToHistory(evaluation) {
    this.evaluationHistory.push(evaluation);

    // Maintain history size
    if (this.evaluationHistory.length > this.maxHistorySize) {
      this.evaluationHistory.shift();
    }
  }

  /**
   * Get condition status summary
   */
  getConditionStatus() {
    const status = {
      totalConditions: this.conditions.size,
      metConditions: 0,
      pendingConditions: 0,
      errorConditions: 0,
      conditions: {}
    };

    for (const [conditionId, condition] of this.conditions) {
      status.conditions[conditionId] = {
        description: condition.description,
        type: condition.type,
        status: condition.status,
        lastEvaluated: condition.lastEvaluated,
        lastResult: condition.lastResult,
        evaluationCount: condition.evaluationCount
      };

      switch (condition.status) {
        case 'met':
          status.metConditions++;
          break;
        case 'not_met':
        case 'pending':
          status.pendingConditions++;
          break;
        case 'error':
          status.errorConditions++;
          break;
      }
    }

    return status;
  }

  /**
   * Get evaluation history
   */
  getEvaluationHistory(limit = 50) {
    return this.evaluationHistory.slice(-limit);
  }

  /**
   * Clear all conditions
   */
  clearConditions() {
    this.conditions.clear();
    logger.info('All conditions cleared');
  }

  /**
   * Get conditions that are currently met
   */
  getMetConditions() {
    const metConditions = [];
    for (const [conditionId, condition] of this.conditions) {
      if (condition.status === 'met') {
        metConditions.push(conditionId);
      }
    }
    return metConditions;
  }

  /**
   * Get conditions that are not met
   */
  getPendingConditions() {
    const pendingConditions = [];
    for (const [conditionId, condition] of this.conditions) {
      if (condition.status === 'not_met' || condition.status === 'pending') {
        pendingConditions.push(conditionId);
      }
    }
    return pendingConditions;
  }

  /**
   * Check if all conditions are met
   */
  areAllConditionsMet() {
    if (this.conditions.size === 0) {
      return false; // No conditions to evaluate
    }

    for (const condition of this.conditions.values()) {
      if (condition.status !== 'met') {
        return false;
      }
    }

    return true;
  }
}