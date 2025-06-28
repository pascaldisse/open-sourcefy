import { logger, auditLogger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * Quality Assurance System
 * Validates task results, code quality, and system compliance
 */
export class QualityAssurance {
  constructor(options = {}) {
    this.qualityStandards = options.qualityStandards || this.getDefaultStandards();
    this.validationRules = options.validationRules || this.getDefaultValidationRules();
    this.thresholds = options.thresholds || this.getDefaultThresholds();
    this.validationHistory = [];
    this.maxHistorySize = options.maxHistorySize || 500;
  }

  /**
   * Get default quality standards
   */
  getDefaultStandards() {
    return {
      code: {
        complexity: 'low',
        coverage: 80,
        duplication: 5,
        maintainability: 'A',
        security: 'high'
      },
      documentation: {
        completeness: 90,
        accuracy: 95,
        readability: 'high',
        examples: true
      },
      testing: {
        coverage: 80,
        passRate: 100,
        performance: 'acceptable',
        reliability: 'high'
      },
      deployment: {
        reliability: 99.9,
        rollbackTime: 300,
        healthChecks: true,
        monitoring: true
      },
      security: {
        vulnerabilities: 0,
        compliance: 100,
        authentication: 'strong',
        encryption: 'required'
      }
    };
  }

  /**
   * Get default validation rules
   */
  getDefaultValidationRules() {
    return {
      codeQuality: [
        { rule: 'noSyntaxErrors', weight: 1.0, critical: true },
        { rule: 'testCoverage', weight: 0.8, critical: false },
        { rule: 'noSecurityVulnerabilities', weight: 1.0, critical: true },
        { rule: 'codeComplexity', weight: 0.6, critical: false },
        { rule: 'documentation', weight: 0.7, critical: false }
      ],
      functionality: [
        { rule: 'requirementsMet', weight: 1.0, critical: true },
        { rule: 'performanceAcceptable', weight: 0.8, critical: false },
        { rule: 'errorHandling', weight: 0.9, critical: true },
        { rule: 'userExperience', weight: 0.7, critical: false }
      ],
      compliance: [
        { rule: 'codingStandards', weight: 0.8, critical: false },
        { rule: 'securityStandards', weight: 1.0, critical: true },
        { rule: 'documentationStandards', weight: 0.6, critical: false },
        { rule: 'accessibilityStandards', weight: 0.7, critical: false }
      ]
    };
  }

  /**
   * Get default quality thresholds
   */
  getDefaultThresholds() {
    return {
      overall: {
        minimum: 70,
        good: 80,
        excellent: 90
      },
      critical: {
        tolerance: 0 // No critical issues allowed
      },
      performance: {
        responseTime: 200, // ms
        throughput: 1000,  // requests/sec
        errorRate: 0.1     // %
      }
    };
  }

  /**
   * Validate task result against quality standards
   */
  async validateTaskResult(taskResult, validationType = 'general') {
    const validationId = uuidv4();
    const startTime = Date.now();

    try {
      logger.info(`Starting quality validation ${validationId} for task ${taskResult.taskId}`);

      // Choose appropriate validation rules
      const rules = this.getValidationRulesForType(validationType);
      
      // Perform validation
      const validation = await this.performValidation(taskResult, rules, validationId);
      
      // Calculate overall score
      const overallScore = this.calculateOverallScore(validation.results);
      
      // Determine quality level
      const qualityLevel = this.determineQualityLevel(overallScore);
      
      // Generate recommendations
      const recommendations = this.generateRecommendations(validation.results);

      const validationResult = {
        validationId,
        taskId: taskResult.taskId,
        agentId: taskResult.agentId,
        validationType,
        overallScore,
        qualityLevel,
        passed: overallScore >= this.thresholds.overall.minimum,
        criticalIssues: validation.criticalIssues,
        results: validation.results,
        recommendations,
        duration: Date.now() - startTime,
        timestamp: Date.now()
      };

      // Store validation result
      this.addToHistory(validationResult);

      // Log validation completion
      auditLogger.info('Quality validation completed', {
        validationId,
        taskId: taskResult.taskId,
        agentId: taskResult.agentId,
        score: overallScore,
        passed: validationResult.passed,
        criticalIssues: validation.criticalIssues.length
      });

      return validationResult;

    } catch (error) {
      logger.error(`Quality validation ${validationId} failed:`, error);
      
      return {
        validationId,
        taskId: taskResult.taskId,
        agentId: taskResult.agentId,
        error: error.message,
        passed: false,
        timestamp: Date.now()
      };
    }
  }

  /**
   * Get validation rules for specific type
   */
  getValidationRulesForType(validationType) {
    const typeMapping = {
      'code': 'codeQuality',
      'functionality': 'functionality',
      'compliance': 'compliance',
      'general': 'codeQuality' // default
    };

    const ruleCategory = typeMapping[validationType] || 'codeQuality';
    return this.validationRules[ruleCategory] || this.validationRules.codeQuality;
  }

  /**
   * Perform validation using specified rules
   */
  async performValidation(taskResult, rules, validationId) {
    const results = [];
    const criticalIssues = [];

    for (const rule of rules) {
      try {
        const ruleResult = await this.validateRule(taskResult, rule, validationId);
        results.push(ruleResult);

        if (!ruleResult.passed && rule.critical) {
          criticalIssues.push({
            rule: rule.rule,
            issue: ruleResult.message,
            severity: 'critical'
          });
        }

      } catch (error) {
        logger.error(`Validation rule ${rule.rule} failed:`, error);
        results.push({
          rule: rule.rule,
          passed: false,
          score: 0,
          weight: rule.weight,
          message: `Validation error: ${error.message}`,
          critical: rule.critical
        });
      }
    }

    return { results, criticalIssues };
  }

  /**
   * Validate individual rule
   */
  async validateRule(taskResult, rule, validationId) {
    logger.debug(`Validating rule ${rule.rule} for validation ${validationId}`);

    switch (rule.rule) {
      case 'noSyntaxErrors':
        return this.validateSyntax(taskResult, rule);
      
      case 'testCoverage':
        return this.validateTestCoverage(taskResult, rule);
      
      case 'noSecurityVulnerabilities':
        return this.validateSecurity(taskResult, rule);
      
      case 'codeComplexity':
        return this.validateComplexity(taskResult, rule);
      
      case 'documentation':
        return this.validateDocumentation(taskResult, rule);
      
      case 'requirementsMet':
        return this.validateRequirements(taskResult, rule);
      
      case 'performanceAcceptable':
        return this.validatePerformance(taskResult, rule);
      
      case 'errorHandling':
        return this.validateErrorHandling(taskResult, rule);
      
      case 'codingStandards':
        return this.validateCodingStandards(taskResult, rule);
      
      default:
        return {
          rule: rule.rule,
          passed: true,
          score: 100,
          weight: rule.weight,
          message: `Rule ${rule.rule} not implemented, assuming pass`,
          critical: rule.critical
        };
    }
  }

  /**
   * Validate syntax and compilation
   */
  async validateSyntax(taskResult, rule) {
    const result = taskResult.result || '';
    
    // Check for syntax error indicators in the result
    const syntaxErrorPatterns = [
      /SyntaxError/i,
      /compilation error/i,
      /parse error/i,
      /unexpected token/i,
      /missing semicolon/i
    ];

    let hasErrors = false;
    let errorMessages = [];

    for (const pattern of syntaxErrorPatterns) {
      if (pattern.test(result)) {
        hasErrors = true;
        errorMessages.push(`Syntax issue detected: ${pattern.source}`);
      }
    }

    return {
      rule: rule.rule,
      passed: !hasErrors,
      score: hasErrors ? 0 : 100,
      weight: rule.weight,
      message: hasErrors ? errorMessages.join('; ') : 'No syntax errors detected',
      critical: rule.critical
    };
  }

  /**
   * Validate test coverage
   */
  async validateTestCoverage(taskResult, rule) {
    const targetCoverage = this.qualityStandards.testing.coverage;
    
    // Extract coverage from task metrics if available
    let coverage = 0;
    if (taskResult.testMetrics && taskResult.testMetrics.coverage) {
      coverage = taskResult.testMetrics.coverage;
    } else {
      // Try to parse from result text
      const coverageMatch = taskResult.result.match(/coverage[:\s]+(\d+(?:\.\d+)?)%/i);
      if (coverageMatch) {
        coverage = parseFloat(coverageMatch[1]);
      }
    }

    const passed = coverage >= targetCoverage;
    const score = Math.min(100, (coverage / targetCoverage) * 100);

    return {
      rule: rule.rule,
      passed,
      score,
      weight: rule.weight,
      message: `Test coverage: ${coverage}% (target: ${targetCoverage}%)`,
      critical: rule.critical,
      metrics: { coverage, target: targetCoverage }
    };
  }

  /**
   * Validate security vulnerabilities
   */
  async validateSecurity(taskResult, rule) {
    const result = taskResult.result || '';
    
    // Check for security vulnerability indicators
    const securityIssuePatterns = [
      /vulnerability/i,
      /security issue/i,
      /injection/i,
      /xss/i,
      /csrf/i,
      /sql injection/i,
      /insecure/i
    ];

    let hasVulnerabilities = false;
    let vulnerabilities = [];

    for (const pattern of securityIssuePatterns) {
      if (pattern.test(result)) {
        hasVulnerabilities = true;
        vulnerabilities.push(`Potential security issue: ${pattern.source}`);
      }
    }

    return {
      rule: rule.rule,
      passed: !hasVulnerabilities,
      score: hasVulnerabilities ? 0 : 100,
      weight: rule.weight,
      message: hasVulnerabilities ? vulnerabilities.join('; ') : 'No security vulnerabilities detected',
      critical: rule.critical
    };
  }

  /**
   * Validate code complexity
   */
  async validateComplexity(taskResult, rule) {
    // Simplified complexity check based on result content
    const result = taskResult.result || '';
    const lines = result.split('\n').length;
    
    // Simple heuristic: longer results might indicate higher complexity
    let complexityScore = 100;
    if (lines > 500) {
      complexityScore = 60;
    } else if (lines > 200) {
      complexityScore = 80;
    }

    return {
      rule: rule.rule,
      passed: complexityScore >= 70,
      score: complexityScore,
      weight: rule.weight,
      message: `Code complexity score: ${complexityScore} (${lines} lines)`,
      critical: rule.critical
    };
  }

  /**
   * Validate documentation quality
   */
  async validateDocumentation(taskResult, rule) {
    const result = taskResult.result || '';
    
    // Check for documentation indicators
    const docIndicators = [
      /\/\*\*/,           // JSDoc comments
      /@param/,           // Parameter documentation
      /@returns?/,        // Return documentation
      /README/i,          // README files
      /documentation/i,   // Documentation mentions
      /example/i          // Examples
    ];

    let docScore = 0;
    const foundIndicators = [];

    for (const indicator of docIndicators) {
      if (indicator.test(result)) {
        docScore += 20;
        foundIndicators.push(indicator.source);
      }
    }

    docScore = Math.min(100, docScore);
    const passed = docScore >= this.qualityStandards.documentation.completeness;

    return {
      rule: rule.rule,
      passed,
      score: docScore,
      weight: rule.weight,
      message: `Documentation score: ${docScore}% (found: ${foundIndicators.length} indicators)`,
      critical: rule.critical
    };
  }

  /**
   * Validate requirements fulfillment
   */
  async validateRequirements(taskResult, rule) {
    // Check if task was successful and meets basic requirements
    const passed = taskResult.success && taskResult.result && taskResult.result.length > 0;
    const score = passed ? 100 : 0;

    return {
      rule: rule.rule,
      passed,
      score,
      weight: rule.weight,
      message: passed ? 'Requirements appear to be met' : 'Requirements not met or task failed',
      critical: rule.critical
    };
  }

  /**
   * Validate performance metrics
   */
  async validatePerformance(taskResult, rule) {
    const duration = taskResult.duration || 0;
    const threshold = this.thresholds.performance.responseTime;
    
    const passed = duration <= threshold;
    const score = passed ? 100 : Math.max(0, 100 - ((duration - threshold) / threshold) * 50);

    return {
      rule: rule.rule,
      passed,
      score,
      weight: rule.weight,
      message: `Performance: ${duration}ms (threshold: ${threshold}ms)`,
      critical: rule.critical,
      metrics: { duration, threshold }
    };
  }

  /**
   * Validate error handling
   */
  async validateErrorHandling(taskResult, rule) {
    const result = taskResult.result || '';
    
    // Check for error handling patterns
    const errorHandlingPatterns = [
      /try\s*{/,
      /catch\s*\(/,
      /error handling/i,
      /exception/i,
      /\.catch\(/,
      /throw\s+/
    ];

    let hasErrorHandling = false;
    for (const pattern of errorHandlingPatterns) {
      if (pattern.test(result)) {
        hasErrorHandling = true;
        break;
      }
    }

    const score = hasErrorHandling ? 100 : 60; // Partial score if no explicit error handling

    return {
      rule: rule.rule,
      passed: score >= 70,
      score,
      weight: rule.weight,
      message: hasErrorHandling ? 'Error handling detected' : 'No explicit error handling found',
      critical: rule.critical
    };
  }

  /**
   * Validate coding standards compliance
   */
  async validateCodingStandards(taskResult, rule) {
    const result = taskResult.result || '';
    
    // Simple coding standards checks
    let violations = 0;
    const checks = [
      { pattern: /\t/, violation: 'Uses tabs instead of spaces' },
      { pattern: /\s+$/, violation: 'Trailing whitespace' },
      { pattern: /var\s+/, violation: 'Uses var instead of let/const' }
    ];

    const foundViolations = [];
    for (const check of checks) {
      if (check.pattern.test(result)) {
        violations++;
        foundViolations.push(check.violation);
      }
    }

    const score = Math.max(0, 100 - (violations * 20));
    const passed = score >= 80;

    return {
      rule: rule.rule,
      passed,
      score,
      weight: rule.weight,
      message: violations > 0 ? `Coding standard violations: ${foundViolations.join(', ')}` : 'No coding standard violations',
      critical: rule.critical
    };
  }

  /**
   * Calculate overall validation score
   */
  calculateOverallScore(results) {
    let totalWeightedScore = 0;
    let totalWeight = 0;

    for (const result of results) {
      totalWeightedScore += result.score * result.weight;
      totalWeight += result.weight;
    }

    return totalWeight > 0 ? totalWeightedScore / totalWeight : 0;
  }

  /**
   * Determine quality level from score
   */
  determineQualityLevel(score) {
    if (score >= this.thresholds.overall.excellent) {
      return 'excellent';
    } else if (score >= this.thresholds.overall.good) {
      return 'good';
    } else if (score >= this.thresholds.overall.minimum) {
      return 'acceptable';
    } else {
      return 'poor';
    }
  }

  /**
   * Generate improvement recommendations
   */
  generateRecommendations(results) {
    const recommendations = [];

    for (const result of results) {
      if (!result.passed) {
        const priority = result.critical ? 'high' : 'medium';
        recommendations.push({
          rule: result.rule,
          priority,
          issue: result.message,
          suggestion: this.getSuggestionForRule(result.rule, result)
        });
      }
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  /**
   * Get improvement suggestion for specific rule
   */
  getSuggestionForRule(ruleName, result) {
    const suggestions = {
      'noSyntaxErrors': 'Fix syntax errors and ensure code compiles correctly',
      'testCoverage': 'Add more unit tests to increase coverage',
      'noSecurityVulnerabilities': 'Review code for security issues and apply fixes',
      'codeComplexity': 'Refactor complex code into smaller, simpler functions',
      'documentation': 'Add JSDoc comments and improve code documentation',
      'requirementsMet': 'Ensure all requirements are implemented and tested',
      'performanceAcceptable': 'Optimize code performance to meet response time requirements',
      'errorHandling': 'Add proper error handling with try-catch blocks',
      'codingStandards': 'Follow established coding standards and style guidelines'
    };

    return suggestions[ruleName] || 'Review and improve implementation';
  }

  /**
   * Add validation result to history
   */
  addToHistory(validationResult) {
    this.validationHistory.push(validationResult);

    // Maintain history size limit
    if (this.validationHistory.length > this.maxHistorySize) {
      this.validationHistory.shift();
    }
  }

  /**
   * Get validation statistics
   */
  getValidationStatistics(timeframe = 24 * 60 * 60 * 1000) { // 24 hours default
    const cutoffTime = Date.now() - timeframe;
    const recentValidations = this.validationHistory.filter(v => v.timestamp >= cutoffTime);

    if (recentValidations.length === 0) {
      return {
        totalValidations: 0,
        averageScore: 0,
        passRate: 0,
        criticalIssueRate: 0,
        qualityDistribution: {}
      };
    }

    const totalValidations = recentValidations.length;
    const passedValidations = recentValidations.filter(v => v.passed).length;
    const validationsWithCriticalIssues = recentValidations.filter(v => v.criticalIssues && v.criticalIssues.length > 0).length;
    
    const averageScore = recentValidations.reduce((sum, v) => sum + (v.overallScore || 0), 0) / totalValidations;
    const passRate = (passedValidations / totalValidations) * 100;
    const criticalIssueRate = (validationsWithCriticalIssues / totalValidations) * 100;

    // Quality level distribution
    const qualityDistribution = {};
    for (const validation of recentValidations) {
      const level = validation.qualityLevel || 'unknown';
      qualityDistribution[level] = (qualityDistribution[level] || 0) + 1;
    }

    return {
      totalValidations,
      averageScore: Math.round(averageScore * 100) / 100,
      passRate: Math.round(passRate * 100) / 100,
      criticalIssueRate: Math.round(criticalIssueRate * 100) / 100,
      qualityDistribution
    };
  }

  /**
   * Get validation history with filtering
   */
  getValidationHistory(filter = {}) {
    let filtered = [...this.validationHistory];

    if (filter.agentId !== undefined) {
      filtered = filtered.filter(v => v.agentId === filter.agentId);
    }

    if (filter.passed !== undefined) {
      filtered = filtered.filter(v => v.passed === filter.passed);
    }

    if (filter.qualityLevel) {
      filtered = filtered.filter(v => v.qualityLevel === filter.qualityLevel);
    }

    if (filter.since) {
      filtered = filtered.filter(v => v.timestamp >= filter.since);
    }

    if (filter.limit) {
      filtered = filtered.slice(-filter.limit);
    }

    return filtered;
  }
}