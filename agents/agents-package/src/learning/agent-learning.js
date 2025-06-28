/**
 * Agent Learning System
 * 
 * Implements learning capabilities for agents to improve over time
 */

import { EventEmitter } from 'events';
import fs from 'fs/promises';
import path from 'path';

export class AgentLearning extends EventEmitter {
  constructor(config = {}) {
    super();
    
    this.config = {
      learningRate: 0.1,
      memorySize: 1000,
      minConfidence: 0.7,
      adaptationThreshold: 0.8,
      persistencePath: './learning-data',
      ...config
    };
    
    // Learning memory for each agent
    this.agentMemory = new Map();
    
    // Pattern recognition
    this.taskPatterns = new Map();
    this.successPatterns = new Map();
    this.failurePatterns = new Map();
    
    // Performance baselines
    this.performanceBaselines = new Map();
    
    // Adaptation strategies
    this.adaptationStrategies = new Map();
    
    // Learning metrics
    this.metrics = {
      patternsLearned: 0,
      adaptationsMade: 0,
      performanceImprovements: 0,
      knowledgeShared: 0
    };
  }
  
  /**
   * Initialize learning system
   */
  async initialize() {
    try {
      // Create persistence directory
      await fs.mkdir(this.config.persistencePath, { recursive: true });
      
      // Load existing knowledge
      await this.loadKnowledgeBase();
      
      // Initialize default strategies
      this.initializeDefaultStrategies();
      
      this.emit('initialized');
    } catch (error) {
      console.error('Failed to initialize learning system:', error);
    }
  }
  
  /**
   * Initialize default adaptation strategies
   */
  initializeDefaultStrategies() {
    // Task optimization strategy
    this.adaptationStrategies.set('task-optimization', {
      name: 'Task Optimization',
      conditions: {
        minSamples: 10,
        successRateThreshold: 0.9
      },
      apply: (agent, pattern) => {
        return {
          adjustments: {
            priority: pattern.avgPriority,
            timeout: pattern.avgDuration * 1.2,
            retryAttempts: pattern.successRate > 0.95 ? 1 : 2
          }
        };
      }
    });
    
    // Resource allocation strategy
    this.adaptationStrategies.set('resource-allocation', {
      name: 'Resource Allocation',
      conditions: {
        minSamples: 20,
        performanceThreshold: 0.8
      },
      apply: (agent, metrics) => {
        return {
          adjustments: {
            maxConcurrentTasks: metrics.avgConcurrency,
            taskTimeout: metrics.p95Duration,
            cacheStrategy: metrics.cacheHitRate > 0.7 ? 'aggressive' : 'normal'
          }
        };
      }
    });
    
    // Error recovery strategy
    this.adaptationStrategies.set('error-recovery', {
      name: 'Error Recovery',
      conditions: {
        minFailures: 5,
        errorPatternThreshold: 0.6
      },
      apply: (agent, errorPatterns) => {
        return {
          adjustments: {
            errorHandlers: errorPatterns.commonErrors,
            fallbackStrategies: errorPatterns.recoveryMethods,
            preventiveMeasures: errorPatterns.avoidancePatterns
          }
        };
      }
    });
  }
  
  /**
   * Record task execution for learning
   */
  async recordTaskExecution(agentId, task, result) {
    // Get or create agent memory
    if (!this.agentMemory.has(agentId)) {
      this.agentMemory.set(agentId, {
        taskHistory: [],
        patterns: new Map(),
        performance: {
          successRate: 0,
          avgDuration: 0,
          totalTasks: 0
        }
      });
    }
    
    const memory = this.agentMemory.get(agentId);
    
    // Record task execution
    const execution = {
      taskId: task.id,
      type: task.type,
      description: task.description,
      context: task.context,
      priority: task.priority,
      success: result.success,
      duration: result.duration,
      error: result.error,
      timestamp: Date.now()
    };
    
    memory.taskHistory.push(execution);
    
    // Maintain memory size limit
    if (memory.taskHistory.length > this.config.memorySize) {
      memory.taskHistory.shift();
    }
    
    // Update performance metrics
    this.updatePerformanceMetrics(agentId, execution);
    
    // Detect patterns
    await this.detectPatterns(agentId, execution);
    
    // Check for adaptation opportunities
    await this.checkAdaptation(agentId);
    
    // Persist learning
    await this.persistLearning(agentId);
  }
  
  /**
   * Update performance metrics
   */
  updatePerformanceMetrics(agentId, execution) {
    const memory = this.agentMemory.get(agentId);
    const perf = memory.performance;
    
    perf.totalTasks++;
    
    // Update success rate
    if (execution.success) {
      perf.successRate = ((perf.successRate * (perf.totalTasks - 1)) + 1) / perf.totalTasks;
    } else {
      perf.successRate = (perf.successRate * (perf.totalTasks - 1)) / perf.totalTasks;
    }
    
    // Update average duration
    perf.avgDuration = ((perf.avgDuration * (perf.totalTasks - 1)) + execution.duration) / perf.totalTasks;
    
    // Track performance baseline
    if (!this.performanceBaselines.has(agentId)) {
      this.performanceBaselines.set(agentId, {
        initial: { ...perf },
        current: { ...perf },
        improvements: 0
      });
    } else {
      const baseline = this.performanceBaselines.get(agentId);
      baseline.current = { ...perf };
      
      // Check for improvement
      if (perf.successRate > baseline.initial.successRate * 1.1) {
        baseline.improvements++;
        this.metrics.performanceImprovements++;
      }
    }
  }
  
  /**
   * Detect patterns in task execution
   */
  async detectPatterns(agentId, execution) {
    const memory = this.agentMemory.get(agentId);
    
    // Pattern key based on task type and context
    const patternKey = `${execution.type}:${JSON.stringify(execution.context || {})}`;
    
    if (!memory.patterns.has(patternKey)) {
      memory.patterns.set(patternKey, {
        occurrences: 0,
        successes: 0,
        failures: 0,
        avgDuration: 0,
        contexts: [],
        errors: []
      });
    }
    
    const pattern = memory.patterns.get(patternKey);
    pattern.occurrences++;
    
    if (execution.success) {
      pattern.successes++;
      this.updateSuccessPattern(execution);
    } else {
      pattern.failures++;
      pattern.errors.push(execution.error);
      this.updateFailurePattern(execution);
    }
    
    pattern.avgDuration = ((pattern.avgDuration * (pattern.occurrences - 1)) + execution.duration) / pattern.occurrences;
    pattern.contexts.push(execution.context);
    
    // Identify significant patterns
    if (pattern.occurrences >= 5) {
      const successRate = pattern.successes / pattern.occurrences;
      
      if (successRate > 0.8) {
        this.taskPatterns.set(patternKey, {
          type: 'success',
          confidence: successRate,
          pattern
        });
      } else if (successRate < 0.3) {
        this.taskPatterns.set(patternKey, {
          type: 'failure',
          confidence: 1 - successRate,
          pattern
        });
      }
      
      this.metrics.patternsLearned++;
    }
  }
  
  /**
   * Update success patterns
   */
  updateSuccessPattern(execution) {
    const key = execution.type;
    
    if (!this.successPatterns.has(key)) {
      this.successPatterns.set(key, {
        count: 0,
        avgDuration: 0,
        contexts: [],
        strategies: []
      });
    }
    
    const pattern = this.successPatterns.get(key);
    pattern.count++;
    pattern.avgDuration = ((pattern.avgDuration * (pattern.count - 1)) + execution.duration) / pattern.count;
    pattern.contexts.push(execution.context);
  }
  
  /**
   * Update failure patterns
   */
  updateFailurePattern(execution) {
    const key = `${execution.type}:${execution.error}`;
    
    if (!this.failurePatterns.has(key)) {
      this.failurePatterns.set(key, {
        count: 0,
        errors: [],
        contexts: [],
        recoveries: []
      });
    }
    
    const pattern = this.failurePatterns.get(key);
    pattern.count++;
    pattern.errors.push(execution.error);
    pattern.contexts.push(execution.context);
  }
  
  /**
   * Check if agent should adapt based on learning
   */
  async checkAdaptation(agentId) {
    const memory = this.agentMemory.get(agentId);
    const baseline = this.performanceBaselines.get(agentId);
    
    if (!memory || !baseline) return;
    
    // Check each adaptation strategy
    for (const [strategyId, strategy] of this.adaptationStrategies) {
      const shouldAdapt = this.evaluateAdaptationNeed(
        agentId,
        memory,
        baseline,
        strategy
      );
      
      if (shouldAdapt) {
        const adaptation = await this.generateAdaptation(
          agentId,
          strategy,
          memory
        );
        
        if (adaptation) {
          this.emit('adaptation-suggested', {
            agentId,
            strategyId,
            adaptation,
            confidence: adaptation.confidence
          });
          
          this.metrics.adaptationsMade++;
        }
      }
    }
  }
  
  /**
   * Evaluate if adaptation is needed
   */
  evaluateAdaptationNeed(agentId, memory, baseline, strategy) {
    const conditions = strategy.conditions;
    
    // Check minimum samples
    if (memory.taskHistory.length < (conditions.minSamples || 10)) {
      return false;
    }
    
    // Check performance thresholds
    if (conditions.successRateThreshold && 
        memory.performance.successRate < conditions.successRateThreshold) {
      return true;
    }
    
    if (conditions.performanceThreshold) {
      const currentPerf = memory.performance.successRate;
      const baselinePerf = baseline.initial.successRate;
      
      if (currentPerf < baselinePerf * conditions.performanceThreshold) {
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Generate adaptation based on learning
   */
  async generateAdaptation(agentId, strategy, memory) {
    try {
      // Analyze patterns
      const patterns = this.analyzeAgentPatterns(agentId);
      
      // Apply strategy
      const adaptation = strategy.apply({
        agentId,
        memory,
        patterns
      });
      
      // Calculate confidence
      const confidence = this.calculateAdaptationConfidence(
        memory,
        patterns,
        adaptation
      );
      
      if (confidence >= this.config.minConfidence) {
        return {
          ...adaptation,
          confidence,
          strategy: strategy.name,
          timestamp: Date.now()
        };
      }
      
      return null;
    } catch (error) {
      console.error('Error generating adaptation:', error);
      return null;
    }
  }
  
  /**
   * Analyze agent patterns
   */
  analyzeAgentPatterns(agentId) {
    const memory = this.agentMemory.get(agentId);
    if (!memory) return {};
    
    const patterns = {
      taskTypes: new Map(),
      errorTypes: new Map(),
      performanceTrends: [],
      contextCorrelations: new Map()
    };
    
    // Analyze task types
    for (const task of memory.taskHistory) {
      if (!patterns.taskTypes.has(task.type)) {
        patterns.taskTypes.set(task.type, {
          count: 0,
          successes: 0,
          avgDuration: 0
        });
      }
      
      const typePattern = patterns.taskTypes.get(task.type);
      typePattern.count++;
      if (task.success) typePattern.successes++;
      typePattern.avgDuration = ((typePattern.avgDuration * (typePattern.count - 1)) + task.duration) / typePattern.count;
    }
    
    // Analyze error patterns
    for (const task of memory.taskHistory.filter(t => !t.success)) {
      const errorKey = task.error || 'unknown';
      patterns.errorTypes.set(errorKey, (patterns.errorTypes.get(errorKey) || 0) + 1);
    }
    
    return patterns;
  }
  
  /**
   * Calculate adaptation confidence
   */
  calculateAdaptationConfidence(memory, patterns, adaptation) {
    let confidence = 0;
    let factors = 0;
    
    // Factor 1: Sample size
    const sampleSizeFactor = Math.min(memory.taskHistory.length / 100, 1);
    confidence += sampleSizeFactor * 0.3;
    factors++;
    
    // Factor 2: Pattern consistency
    let patternConsistency = 0;
    for (const [_, pattern] of memory.patterns) {
      if (pattern.occurrences > 5) {
        const consistency = Math.abs(pattern.successes / pattern.occurrences - 0.5) * 2;
        patternConsistency += consistency;
      }
    }
    patternConsistency = patternConsistency / Math.max(memory.patterns.size, 1);
    confidence += patternConsistency * 0.4;
    factors++;
    
    // Factor 3: Performance stability
    const recentTasks = memory.taskHistory.slice(-20);
    const recentSuccessRate = recentTasks.filter(t => t.success).length / recentTasks.length;
    const stabilityFactor = 1 - Math.abs(recentSuccessRate - memory.performance.successRate);
    confidence += stabilityFactor * 0.3;
    factors++;
    
    return confidence / factors;
  }
  
  /**
   * Share learning between agents
   */
  async shareKnowledge(fromAgentId, toAgentId, knowledgeType = 'patterns') {
    const fromMemory = this.agentMemory.get(fromAgentId);
    const toMemory = this.agentMemory.get(toAgentId);
    
    if (!fromMemory || !toMemory) return false;
    
    try {
      switch (knowledgeType) {
        case 'patterns':
          // Share successful patterns
          for (const [key, pattern] of fromMemory.patterns) {
            if (pattern.successes / pattern.occurrences > 0.8) {
              toMemory.patterns.set(`shared:${key}`, {
                ...pattern,
                sharedFrom: fromAgentId,
                sharedAt: Date.now()
              });
            }
          }
          break;
          
        case 'strategies':
          // Share successful strategies
          const successfulTasks = fromMemory.taskHistory
            .filter(t => t.success)
            .slice(-50);
          
          toMemory.sharedStrategies = successfulTasks.map(task => ({
            type: task.type,
            context: task.context,
            duration: task.duration,
            sharedFrom: fromAgentId
          }));
          break;
      }
      
      this.metrics.knowledgeShared++;
      this.emit('knowledge-shared', {
        from: fromAgentId,
        to: toAgentId,
        type: knowledgeType,
        timestamp: Date.now()
      });
      
      return true;
    } catch (error) {
      console.error('Error sharing knowledge:', error);
      return false;
    }
  }
  
  /**
   * Get learning insights for an agent
   */
  getLearningInsights(agentId) {
    const memory = this.agentMemory.get(agentId);
    const baseline = this.performanceBaselines.get(agentId);
    
    if (!memory) return null;
    
    const insights = {
      agentId,
      performance: memory.performance,
      improvement: baseline ? {
        successRateChange: memory.performance.successRate - baseline.initial.successRate,
        durationChange: memory.performance.avgDuration - baseline.initial.avgDuration,
        improvementCount: baseline.improvements
      } : null,
      patterns: {
        total: memory.patterns.size,
        successful: Array.from(memory.patterns.values())
          .filter(p => p.successes / p.occurrences > 0.8).length,
        problematic: Array.from(memory.patterns.values())
          .filter(p => p.successes / p.occurrences < 0.3).length
      },
      recommendations: this.generateRecommendations(agentId)
    };
    
    return insights;
  }
  
  /**
   * Generate recommendations based on learning
   */
  generateRecommendations(agentId) {
    const memory = this.agentMemory.get(agentId);
    if (!memory) return [];
    
    const recommendations = [];
    
    // Analyze patterns for recommendations
    for (const [key, pattern] of memory.patterns) {
      const successRate = pattern.successes / pattern.occurrences;
      
      if (successRate < 0.5 && pattern.occurrences > 10) {
        recommendations.push({
          type: 'improvement',
          area: key.split(':')[0],
          suggestion: `Consider reviewing approach for ${key.split(':')[0]} tasks`,
          confidence: pattern.occurrences / 100
        });
      }
      
      if (pattern.avgDuration > memory.performance.avgDuration * 2) {
        recommendations.push({
          type: 'optimization',
          area: key.split(':')[0],
          suggestion: `Optimize performance for ${key.split(':')[0]} tasks`,
          confidence: 0.8
        });
      }
    }
    
    return recommendations;
  }
  
  /**
   * Load knowledge base from disk
   */
  async loadKnowledgeBase() {
    try {
      const files = await fs.readdir(this.config.persistencePath);
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          const agentId = parseInt(file.replace('.json', ''));
          const data = await fs.readFile(
            path.join(this.config.persistencePath, file),
            'utf8'
          );
          
          const knowledge = JSON.parse(data);
          this.agentMemory.set(agentId, knowledge);
        }
      }
    } catch (error) {
      console.error('Error loading knowledge base:', error);
    }
  }
  
  /**
   * Persist learning data
   */
  async persistLearning(agentId) {
    try {
      const memory = this.agentMemory.get(agentId);
      if (!memory) return;
      
      const filePath = path.join(
        this.config.persistencePath,
        `${agentId}.json`
      );
      
      await fs.writeFile(
        filePath,
        JSON.stringify(memory, null, 2),
        'utf8'
      );
    } catch (error) {
      console.error('Error persisting learning data:', error);
    }
  }
  
  /**
   * Get learning metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      agentsLearning: this.agentMemory.size,
      totalPatterns: this.taskPatterns.size,
      successPatterns: this.successPatterns.size,
      failurePatterns: this.failurePatterns.size
    };
  }
  
  /**
   * Reset learning for an agent
   */
  resetAgentLearning(agentId) {
    this.agentMemory.delete(agentId);
    this.performanceBaselines.delete(agentId);
    
    // Remove persisted data
    const filePath = path.join(
      this.config.persistencePath,
      `${agentId}.json`
    );
    
    fs.unlink(filePath).catch(() => {});
  }
  
  /**
   * Shutdown learning system
   */
  async shutdown() {
    // Persist all learning data
    for (const [agentId] of this.agentMemory) {
      await this.persistLearning(agentId);
    }
    
    this.emit('shutdown');
  }
}