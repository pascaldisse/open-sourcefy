import { logger, performanceLogger, healthLogger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';
import os from 'os';

/**
 * System Monitor
 * Real-time monitoring of system state, agent performance, and resource usage
 */
export class SystemMonitor {
  constructor(options = {}) {
    this.monitoringInterval = options.monitoringInterval || 10000; // 10 seconds
    this.metricsRetention = options.metricsRetention || 24 * 60 * 60 * 1000; // 24 hours
    this.alertThresholds = options.alertThresholds || this.getDefaultThresholds();
    
    this.metrics = new Map();
    this.alerts = [];
    this.isMonitoring = false;
    this.monitoringTimer = null;
    this.coordinator = null;
    this.messageBus = null;
    this.qualityAssurance = null;
    
    this.lastSystemSnapshot = null;
    this.performanceBaseline = null;
  }

  /**
   * Set system components for monitoring
   */
  setComponents(coordinator, messageBus, qualityAssurance) {
    this.coordinator = coordinator;
    this.messageBus = messageBus;
    this.qualityAssurance = qualityAssurance;
  }

  /**
   * Get default alert thresholds
   */
  getDefaultThresholds() {
    return {
      system: {
        cpuUsage: 80,        // %
        memoryUsage: 85,     // %
        diskUsage: 90,       // %
        loadAverage: 8       // 1-minute load average
      },
      agents: {
        responseTime: 5000,  // ms
        errorRate: 10,       // %
        taskFailureRate: 20, // %
        inactivityTime: 300000 // 5 minutes
      },
      communication: {
        messageLatency: 1000,    // ms
        messageFailureRate: 5,   // %
        queueBacklog: 100        // messages
      },
      quality: {
        averageScore: 70,        // minimum quality score
        criticalIssueRate: 1,    // %
        regressionRate: 5        // %
      }
    };
  }

  /**
   * Start monitoring
   */
  startMonitoring() {
    if (this.isMonitoring) {
      logger.warn('System monitoring is already running');
      return;
    }

    logger.info('Starting system monitoring');
    this.isMonitoring = true;
    
    // Take initial baseline
    this.capturePerformanceBaseline();
    
    // Start monitoring loop
    this.monitoringTimer = setInterval(() => {
      this.performMonitoringCycle();
    }, this.monitoringInterval);
    
    // Perform initial monitoring cycle
    this.performMonitoringCycle();
  }

  /**
   * Stop monitoring
   */
  stopMonitoring() {
    if (!this.isMonitoring) {
      return;
    }

    logger.info('Stopping system monitoring');
    this.isMonitoring = false;
    
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = null;
    }
  }

  /**
   * Perform a complete monitoring cycle
   */
  async performMonitoringCycle() {
    const cycleId = uuidv4();
    const startTime = Date.now();

    try {
      logger.debug(`Starting monitoring cycle ${cycleId}`);

      // Collect all metrics
      const systemMetrics = await this.collectSystemMetrics();
      const agentMetrics = await this.collectAgentMetrics();
      const communicationMetrics = await this.collectCommunicationMetrics();
      const qualityMetrics = await this.collectQualityMetrics();

      // Create system snapshot
      const snapshot = {
        cycleId,
        timestamp: startTime,
        system: systemMetrics,
        agents: agentMetrics,
        communication: communicationMetrics,
        quality: qualityMetrics,
        duration: Date.now() - startTime
      };

      // Store metrics
      this.storeMetrics(snapshot);

      // Check for alerts
      await this.checkAlertConditions(snapshot);

      // Update last snapshot
      this.lastSystemSnapshot = snapshot;

      // Log performance metrics
      performanceLogger.info('Monitoring cycle completed', {
        cycleId,
        duration: snapshot.duration,
        metricsCollected: Object.keys(snapshot).length - 3 // exclude metadata
      });

      healthLogger.info('System health snapshot', snapshot);

    } catch (error) {
      logger.error(`Monitoring cycle ${cycleId} failed:`, error);
    }
  }

  /**
   * Collect system resource metrics
   */
  async collectSystemMetrics() {
    const cpus = os.cpus();
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const loadAvg = os.loadavg();

    // Calculate CPU usage (simplified)
    const cpuUsage = await this.calculateCpuUsage();

    return {
      cpu: {
        usage: cpuUsage,
        cores: cpus.length,
        model: cpus[0]?.model || 'Unknown'
      },
      memory: {
        total: totalMem,
        free: freeMem,
        used: totalMem - freeMem,
        usage: ((totalMem - freeMem) / totalMem) * 100
      },
      load: {
        average1m: loadAvg[0],
        average5m: loadAvg[1],
        average15m: loadAvg[2]
      },
      uptime: os.uptime(),
      platform: os.platform(),
      arch: os.arch()
    };
  }

  /**
   * Calculate CPU usage percentage
   */
  async calculateCpuUsage() {
    return new Promise((resolve) => {
      const startUsage = process.cpuUsage();
      
      setTimeout(() => {
        const endUsage = process.cpuUsage(startUsage);
        const totalUsage = endUsage.user + endUsage.system;
        const percentage = (totalUsage / 1000000) * 100; // Convert to percentage
        resolve(Math.min(100, percentage));
      }, 100);
    });
  }

  /**
   * Collect agent performance metrics
   */
  async collectAgentMetrics() {
    if (!this.coordinator) {
      return {};
    }

    const agentMetrics = {};
    const agentSummary = {
      totalAgents: 0,
      activeAgents: 0,
      idleAgents: 0,
      errorAgents: 0,
      averageResponseTime: 0,
      totalTasksCompleted: 0,
      totalTasksFailed: 0
    };

    let totalResponseTime = 0;
    let responseTimeCount = 0;

    for (const [agentId, agent] of this.coordinator.managedAgents) {
      try {
        const status = agent.getStatus();
        
        agentMetrics[agentId] = {
          status: status.status,
          currentTask: status.currentTask,
          performance: status.performance,
          uptime: status.uptime,
          lastActivity: Date.now() - (status.uptime || 0)
        };

        // Update summary
        agentSummary.totalAgents++;
        
        switch (status.status) {
          case 'working':
            agentSummary.activeAgents++;
            break;
          case 'idle':
            agentSummary.idleAgents++;
            break;
          case 'error':
          case 'shutdown':
            agentSummary.errorAgents++;
            break;
        }

        if (status.performance) {
          agentSummary.totalTasksCompleted += status.performance.tasksCompleted || 0;
          agentSummary.totalTasksFailed += status.performance.tasksFailed || 0;
          
          if (status.performance.averageResponseTime) {
            totalResponseTime += status.performance.averageResponseTime;
            responseTimeCount++;
          }
        }

      } catch (error) {
        logger.error(`Failed to collect metrics for agent ${agentId}:`, error);
        agentMetrics[agentId] = { error: error.message };
        agentSummary.errorAgents++;
      }
    }

    if (responseTimeCount > 0) {
      agentSummary.averageResponseTime = totalResponseTime / responseTimeCount;
    }

    return {
      summary: agentSummary,
      individual: agentMetrics
    };
  }

  /**
   * Collect communication metrics
   */
  async collectCommunicationMetrics() {
    if (!this.messageBus) {
      return {};
    }

    try {
      const stats = this.messageBus.getStatistics();
      const health = await this.messageBus.performHealthCheck();
      
      return {
        statistics: stats,
        health,
        timestamp: Date.now()
      };
    } catch (error) {
      logger.error('Failed to collect communication metrics:', error);
      return { error: error.message };
    }
  }

  /**
   * Collect quality assurance metrics
   */
  async collectQualityMetrics() {
    if (!this.qualityAssurance) {
      return {};
    }

    try {
      const stats = this.qualityAssurance.getValidationStatistics();
      const recentValidations = this.qualityAssurance.getValidationHistory({ limit: 10 });
      
      return {
        statistics: stats,
        recentValidations: recentValidations.map(v => ({
          validationId: v.validationId,
          taskId: v.taskId,
          agentId: v.agentId,
          overallScore: v.overallScore,
          passed: v.passed,
          timestamp: v.timestamp
        })),
        timestamp: Date.now()
      };
    } catch (error) {
      logger.error('Failed to collect quality metrics:', error);
      return { error: error.message };
    }
  }

  /**
   * Store metrics with cleanup
   */
  storeMetrics(snapshot) {
    const metricKey = `snapshot_${snapshot.timestamp}`;
    this.metrics.set(metricKey, snapshot);

    // Clean up old metrics
    this.cleanupOldMetrics();
  }

  /**
   * Clean up old metrics beyond retention period
   */
  cleanupOldMetrics() {
    const cutoffTime = Date.now() - this.metricsRetention;
    
    for (const [key, metric] of this.metrics) {
      if (metric.timestamp < cutoffTime) {
        this.metrics.delete(key);
      }
    }
  }

  /**
   * Check alert conditions
   */
  async checkAlertConditions(snapshot) {
    const alerts = [];

    // System resource alerts
    alerts.push(...this.checkSystemAlerts(snapshot.system));
    
    // Agent performance alerts
    alerts.push(...this.checkAgentAlerts(snapshot.agents));
    
    // Communication alerts
    alerts.push(...this.checkCommunicationAlerts(snapshot.communication));
    
    // Quality alerts
    alerts.push(...this.checkQualityAlerts(snapshot.quality));

    // Process new alerts
    for (const alert of alerts) {
      await this.processAlert(alert, snapshot);
    }
  }

  /**
   * Check system resource alerts
   */
  checkSystemAlerts(systemMetrics) {
    const alerts = [];
    const thresholds = this.alertThresholds.system;

    if (systemMetrics.cpu?.usage > thresholds.cpuUsage) {
      alerts.push({
        type: 'system_resource',
        category: 'cpu_high',
        severity: 'high',
        message: `CPU usage is ${systemMetrics.cpu.usage.toFixed(1)}% (threshold: ${thresholds.cpuUsage}%)`,
        value: systemMetrics.cpu.usage,
        threshold: thresholds.cpuUsage
      });
    }

    if (systemMetrics.memory?.usage > thresholds.memoryUsage) {
      alerts.push({
        type: 'system_resource',
        category: 'memory_high',
        severity: 'high',
        message: `Memory usage is ${systemMetrics.memory.usage.toFixed(1)}% (threshold: ${thresholds.memoryUsage}%)`,
        value: systemMetrics.memory.usage,
        threshold: thresholds.memoryUsage
      });
    }

    if (systemMetrics.load?.average1m > thresholds.loadAverage) {
      alerts.push({
        type: 'system_resource',
        category: 'load_high',
        severity: 'medium',
        message: `Load average is ${systemMetrics.load.average1m.toFixed(2)} (threshold: ${thresholds.loadAverage})`,
        value: systemMetrics.load.average1m,
        threshold: thresholds.loadAverage
      });
    }

    return alerts;
  }

  /**
   * Check agent performance alerts
   */
  checkAgentAlerts(agentMetrics) {
    const alerts = [];
    const thresholds = this.alertThresholds.agents;

    if (!agentMetrics.summary) {
      return alerts;
    }

    const summary = agentMetrics.summary;

    // Check error agents
    if (summary.errorAgents > 0) {
      alerts.push({
        type: 'agent_performance',
        category: 'agents_error',
        severity: 'high',
        message: `${summary.errorAgents} agent(s) in error state`,
        value: summary.errorAgents,
        agentIds: Object.keys(agentMetrics.individual).filter(id => 
          agentMetrics.individual[id].status === 'error'
        )
      });
    }

    // Check response time
    if (summary.averageResponseTime > thresholds.responseTime) {
      alerts.push({
        type: 'agent_performance',
        category: 'response_time_high',
        severity: 'medium',
        message: `Average response time is ${summary.averageResponseTime.toFixed(0)}ms (threshold: ${thresholds.responseTime}ms)`,
        value: summary.averageResponseTime,
        threshold: thresholds.responseTime
      });
    }

    // Check task failure rate
    const totalTasks = summary.totalTasksCompleted + summary.totalTasksFailed;
    if (totalTasks > 0) {
      const failureRate = (summary.totalTasksFailed / totalTasks) * 100;
      if (failureRate > thresholds.taskFailureRate) {
        alerts.push({
          type: 'agent_performance',
          category: 'task_failure_rate_high',
          severity: 'medium',
          message: `Task failure rate is ${failureRate.toFixed(1)}% (threshold: ${thresholds.taskFailureRate}%)`,
          value: failureRate,
          threshold: thresholds.taskFailureRate
        });
      }
    }

    return alerts;
  }

  /**
   * Check communication alerts
   */
  checkCommunicationAlerts(communicationMetrics) {
    const alerts = [];
    const thresholds = this.alertThresholds.communication;

    if (!communicationMetrics.statistics) {
      return alerts;
    }

    const stats = communicationMetrics.statistics;

    // Check message latency (simplified)
    if (stats.averageResponseTime > thresholds.messageLatency) {
      alerts.push({
        type: 'communication',
        category: 'message_latency_high',
        severity: 'medium',
        message: `Message latency is ${stats.averageResponseTime.toFixed(0)}ms (threshold: ${thresholds.messageLatency}ms)`,
        value: stats.averageResponseTime,
        threshold: thresholds.messageLatency
      });
    }

    return alerts;
  }

  /**
   * Check quality alerts
   */
  checkQualityAlerts(qualityMetrics) {
    const alerts = [];
    const thresholds = this.alertThresholds.quality;

    if (!qualityMetrics.statistics) {
      return alerts;
    }

    const stats = qualityMetrics.statistics;

    // Check average quality score
    if (stats.averageScore < thresholds.averageScore) {
      alerts.push({
        type: 'quality',
        category: 'quality_score_low',
        severity: 'medium',
        message: `Average quality score is ${stats.averageScore.toFixed(1)} (threshold: ${thresholds.averageScore})`,
        value: stats.averageScore,
        threshold: thresholds.averageScore
      });
    }

    // Check critical issue rate
    if (stats.criticalIssueRate > thresholds.criticalIssueRate) {
      alerts.push({
        type: 'quality',
        category: 'critical_issues_high',
        severity: 'high',
        message: `Critical issue rate is ${stats.criticalIssueRate.toFixed(1)}% (threshold: ${thresholds.criticalIssueRate}%)`,
        value: stats.criticalIssueRate,
        threshold: thresholds.criticalIssueRate
      });
    }

    return alerts;
  }

  /**
   * Process and handle alerts
   */
  async processAlert(alert, snapshot) {
    const alertId = uuidv4();
    const enrichedAlert = {
      ...alert,
      alertId,
      timestamp: Date.now(),
      snapshotId: snapshot.cycleId,
      resolved: false
    };

    // Add to alerts list
    this.alerts.push(enrichedAlert);

    // Log alert
    logger.warn(`Alert triggered: ${alert.category}`, {
      alertId,
      type: alert.type,
      category: alert.category,
      severity: alert.severity,
      message: alert.message
    });

    // Clean up old alerts
    this.cleanupOldAlerts();

    return enrichedAlert;
  }

  /**
   * Clean up old alerts
   */
  cleanupOldAlerts() {
    const maxAlerts = 100;
    if (this.alerts.length > maxAlerts) {
      this.alerts = this.alerts.slice(-maxAlerts);
    }
  }

  /**
   * Capture performance baseline
   */
  async capturePerformanceBaseline() {
    try {
      const baseline = await this.collectSystemMetrics();
      this.performanceBaseline = {
        ...baseline,
        timestamp: Date.now()
      };
      
      logger.info('Performance baseline captured');
    } catch (error) {
      logger.error('Failed to capture performance baseline:', error);
    }
  }

  /**
   * Get current system snapshot
   */
  getCurrentSnapshot() {
    return this.lastSystemSnapshot;
  }

  /**
   * Get metrics history
   */
  getMetricsHistory(timeframe = 60 * 60 * 1000) { // 1 hour default
    const cutoffTime = Date.now() - timeframe;
    const history = [];

    for (const metric of this.metrics.values()) {
      if (metric.timestamp >= cutoffTime) {
        history.push(metric);
      }
    }

    return history.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Get active alerts
   */
  getActiveAlerts() {
    return this.alerts.filter(alert => !alert.resolved);
  }

  /**
   * Get alert history
   */
  getAlertHistory(limit = 50) {
    return this.alerts.slice(-limit);
  }

  /**
   * Get monitoring statistics
   */
  getMonitoringStatistics() {
    const activeAlerts = this.getActiveAlerts();
    const alertsBySeverity = {};
    const alertsByType = {};

    for (const alert of activeAlerts) {
      alertsBySeverity[alert.severity] = (alertsBySeverity[alert.severity] || 0) + 1;
      alertsByType[alert.type] = (alertsByType[alert.type] || 0) + 1;
    }

    return {
      isMonitoring: this.isMonitoring,
      metricsCollected: this.metrics.size,
      totalAlerts: this.alerts.length,
      activeAlerts: activeAlerts.length,
      alertsBySeverity,
      alertsByType,
      lastSnapshot: this.lastSystemSnapshot?.timestamp,
      baseline: this.performanceBaseline?.timestamp
    };
  }

  /**
   * Resolve an alert
   */
  resolveAlert(alertId) {
    const alert = this.alerts.find(a => a.alertId === alertId);
    if (alert) {
      alert.resolved = true;
      alert.resolvedAt = Date.now();
      logger.info(`Alert resolved: ${alertId}`);
      return true;
    }
    return false;
  }
}