import { AgentBase } from '../core/agent-base.js';
import { logger } from '../utils/logger.js';
import cron from 'node-cron';

/**
 * Agent 6: Task Scheduler
 * Specializes in workflow coordination and task management
 */
export class Agent6 extends AgentBase {
  constructor() {
    super(
      6,
      'Task Scheduler',
      'Workflow coordination and task management specialist',
      [
        'Schedule and prioritize tasks',
        'Track dependencies and deadlines',
        'Monitor resource allocation',
        'Generate progress reports',
        'Optimize workflow efficiency',
        'Manage task queues and backlogs',
        'Coordinate multi-agent workflows',
        'Automate recurring tasks'
      ],
      [
        'Read', 'Write', 'TodoRead', 'TodoWrite', 'Bash'
      ]
    );

    this.scheduledTasks = new Map();
    this.taskQueue = [];
    this.dependencies = new Map();
    this.priorities = ['critical', 'high', 'medium', 'low'];
    this.workflowStates = ['pending', 'in_progress', 'blocked', 'completed', 'failed'];
    this.cronJobs = new Map();
  }

  /**
   * Override task type validation for task scheduler
   */
  canHandleTaskType(taskType) {
    const schedulingTasks = [
      'task_scheduling',
      'workflow_coordination',
      'dependency_management',
      'progress_tracking',
      'resource_allocation',
      'task_prioritization',
      'deadline_management',
      'workflow_optimization',
      'task_automation',
      'queue_management',
      'reporting',
      'milestone_tracking'
    ];
    
    return schedulingTasks.includes(taskType);
  }

  /**
   * Generate system prompt specific to task scheduler role
   */
  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Task Scheduler, you have specific expertise in:

TASK MANAGEMENT PRINCIPLES:
- Priority Matrix: Critical > High > Medium > Low
- Dependency Mapping: Track prerequisites and blockers
- Resource Allocation: Balance workload across agents
- Time Estimation: Realistic scheduling with buffers
- Milestone Tracking: Break large tasks into manageable chunks

WORKFLOW COORDINATION:
- Sequential Workflows: Tasks that must be completed in order
- Parallel Workflows: Independent tasks that can run simultaneously
- Conditional Workflows: Tasks triggered by specific conditions
- Recurring Workflows: Automated tasks on schedules
- Emergency Workflows: High-priority interrupt handling

SCHEDULING STRATEGIES:
- First Come First Served (FCFS): Simple queue processing
- Shortest Job First (SJF): Optimize for quick wins
- Priority Scheduling: Critical tasks take precedence
- Round Robin: Fair distribution of tasks
- Deadline Scheduling: Time-sensitive task prioritization

DEPENDENCY MANAGEMENT:
- Prerequisite Tracking: Tasks that must complete before others
- Resource Dependencies: Shared resources and conflicts
- Agent Availability: Track who can work on what
- Blocking Issues: Identify and resolve bottlenecks
- Circular Dependencies: Detect and resolve cycles

PROGRESS MONITORING:
- Task Status Tracking: pending, in_progress, blocked, completed, failed
- Velocity Metrics: Track completion rates over time
- Bottleneck Identification: Find workflow inefficiencies
- SLA Monitoring: Ensure deadlines are met
- Quality Metrics: Track success rates and rework

AUTOMATION CAPABILITIES:
- Cron-style Scheduling: Time-based task execution
- Event-driven Triggers: React to system events
- Condition-based Execution: Smart workflow branching
- Retry Logic: Handle failures gracefully
- Escalation Procedures: Automatic escalation paths

SPECIAL INSTRUCTIONS:
- Always consider task dependencies before scheduling
- Optimize for both efficiency and quality
- Monitor agent workloads to prevent burnout
- Maintain clear audit trails for all scheduling decisions
- Provide regular progress updates to stakeholders
- Handle urgent tasks with appropriate prioritization
- Balance short-term efficiency with long-term sustainability

When managing tasks and workflows:
1. Analyze task requirements and dependencies
2. Assess available resources and constraints
3. Optimize scheduling for maximum efficiency
4. Monitor progress and adjust as needed
5. Identify and resolve bottlenecks quickly
6. Provide clear status updates and reports`;
  }

  /**
   * Execute scheduling-specific tasks
   */
  async processTask(task) {
    logger.info(`Task Scheduler processing ${task.type} task: ${task.description}`);

    // Pre-process task based on type
    const enhancedTask = await this.enhanceSchedulingTask(task);
    
    // Execute using base class with enhanced context
    const result = await super.processTask(enhancedTask);

    // Post-process results for scheduling tasks
    if (result.success) {
      result.schedulingMetrics = await this.extractSchedulingMetrics(result);
    }

    return result;
  }

  /**
   * Enhance task with scheduling-specific context
   */
  async enhanceSchedulingTask(task) {
    const enhanced = { ...task };

    // Add scheduling context based on task type
    switch (task.type) {
      case 'task_prioritization':
        enhanced.context = {
          ...enhanced.context,
          priorityMatrix: true,
          urgencyAnalysis: true,
          impactAssessment: true,
          resourceConsideration: true
        };
        break;

      case 'dependency_management':
        enhanced.context = {
          ...enhanced.context,
          dependencyMapping: true,
          criticalPath: true,
          blockingIssues: true,
          circularDependencyCheck: true
        };
        break;

      case 'workflow_coordination':
        enhanced.context = {
          ...enhanced.context,
          workflowType: 'mixed',
          parallelization: true,
          synchronization: true,
          errorHandling: true
        };
        break;

      case 'progress_tracking':
        enhanced.context = {
          ...enhanced.context,
          statusUpdates: true,
          velocityTracking: true,
          bottleneckAnalysis: true,
          reportGeneration: true
        };
        break;

      case 'task_automation':
        enhanced.context = {
          ...enhanced.context,
          cronScheduling: true,
          eventTriggers: true,
          retryLogic: true,
          errorHandling: true
        };
        break;
    }

    // Add scheduling-specific requirements
    enhanced.requirements = `${enhanced.requirements || ''}

SCHEDULING REQUIREMENTS:
- Prioritize tasks using ${enhanced.context.priorityMatrix ? 'priority matrix' : 'standard priority levels'}
- Consider ${enhanced.context.resourceConsideration ? 'resource constraints and availability' : 'basic resource allocation'}
- Implement ${enhanced.context.dependencyMapping ? 'comprehensive dependency tracking' : 'basic dependency management'}
- Provide ${enhanced.context.statusUpdates ? 'real-time status updates' : 'periodic progress reports'}
- Optimize for both efficiency and quality outcomes
- Handle failures and exceptions gracefully
- Maintain clear audit trails for all decisions
- Balance workload across available agents
- Meet deadlines while maintaining quality standards
- Provide actionable insights and recommendations`;

    return enhanced;
  }

  /**
   * Extract scheduling metrics from task results
   */
  async extractSchedulingMetrics(result) {
    const metrics = {
      tasksScheduled: 0,
      tasksCompleted: 0,
      tasksFailed: 0,
      averageCompletionTime: 0,
      dependenciesResolved: 0,
      bottlenecksIdentified: 0,
      workflowEfficiency: 0,
      resourceUtilization: 0
    };

    try {
      const output = result.result;
      
      // Count scheduled tasks
      const scheduledMatches = output.match(/(?:scheduled|queued).*task/gi);
      metrics.tasksScheduled = scheduledMatches ? scheduledMatches.length : 0;
      
      // Count completed tasks
      const completedMatches = output.match(/(?:completed|finished).*task/gi);
      metrics.tasksCompleted = completedMatches ? completedMatches.length : 0;
      
      // Count failed tasks
      const failedMatches = output.match(/(?:failed|error).*task/gi);
      metrics.tasksFailed = failedMatches ? failedMatches.length : 0;
      
      // Extract completion time if mentioned
      const timeMatches = output.match(/(?:completed in|took)\s+(\d+(?:\.\d+)?)\s*(?:minutes|mins|seconds|secs|hours|hrs)/gi);
      if (timeMatches && timeMatches.length > 0) {
        const times = timeMatches.map(match => {
          const value = parseFloat(match.match(/\d+(?:\.\d+)?/)[0]);
          if (match.toLowerCase().includes('hour')) return value * 60;
          if (match.toLowerCase().includes('sec')) return value / 60;
          return value; // assume minutes
        });
        metrics.averageCompletionTime = times.reduce((a, b) => a + b, 0) / times.length;
      }
      
      // Count dependencies
      const dependencyMatches = output.match(/dependenc(?:y|ies)/gi);
      metrics.dependenciesResolved = dependencyMatches ? dependencyMatches.length : 0;
      
      // Count bottlenecks
      const bottleneckMatches = output.match(/bottleneck/gi);
      metrics.bottlenecksIdentified = bottleneckMatches ? bottleneckMatches.length : 0;

    } catch (error) {
      logger.warn('Failed to extract scheduling metrics:', error);
    }

    return metrics;
  }

  /**
   * Schedule a new task with dependencies and priority
   */
  async scheduleTask(taskDefinition, options = {}) {
    const task = {
      description: `Schedule task: ${taskDefinition.name}`,
      type: 'task_scheduling',
      context: {
        taskDefinition,
        options: {
          priority: 'medium',
          deadline: null,
          dependencies: [],
          assignedAgent: null,
          ...options
        }
      },
      requirements: 'Schedule task optimally considering priorities, dependencies, and resources'
    };

    return await this.processTask(task);
  }

  /**
   * Optimize workflow for efficiency
   */
  async optimizeWorkflow(workflowDefinition, constraints = {}) {
    const task = {
      description: 'Optimize workflow for maximum efficiency',
      type: 'workflow_optimization',
      context: {
        workflowDefinition,
        constraints: {
          maxParallelTasks: 5,
          resourceLimits: {},
          deadlines: {},
          ...constraints
        }
      },
      requirements: 'Optimize workflow scheduling while respecting all constraints'
    };

    return await this.processTask(task);
  }

  /**
   * Track and report progress across all tasks
   */
  async generateProgressReport(reportType = 'comprehensive', timeframe = '24h') {
    const task = {
      description: `Generate ${reportType} progress report for ${timeframe}`,
      type: 'progress_tracking',
      context: {
        reportType,
        timeframe,
        includeMetrics: true,
        identifyTrends: true,
        recommendations: true
      },
      requirements: 'Create detailed progress report with actionable insights'
    };

    return await this.processTask(task);
  }

  /**
   * Manage task dependencies and resolve conflicts
   */
  async manageDependencies(taskList, dependencyRules) {
    const task = {
      description: 'Analyze and manage task dependencies',
      type: 'dependency_management',
      context: {
        taskList,
        dependencyRules,
        detectCircular: true,
        criticalPath: true,
        optimizeOrder: true
      },
      requirements: 'Resolve dependency conflicts and optimize task ordering'
    };

    return await this.processTask(task);
  }

  /**
   * Set up automated recurring tasks
   */
  async setupAutomation(automationRules) {
    const task = {
      description: 'Set up automated task scheduling and execution',
      type: 'task_automation',
      context: {
        automationRules,
        cronScheduling: true,
        eventTriggers: true,
        errorHandling: true,
        retryLogic: true
      },
      requirements: 'Configure robust task automation with proper error handling'
    };

    return await this.processTask(task);
  }

  /**
   * Allocate resources efficiently across tasks
   */
  async allocateResources(resourcePool, taskDemands) {
    const task = {
      description: 'Optimize resource allocation across tasks',
      type: 'resource_allocation',
      context: {
        resourcePool,
        taskDemands,
        optimization: 'efficiency',
        balancing: true,
        constraints: true
      },
      requirements: 'Allocate resources to maximize overall productivity'
    };

    return await this.processTask(task);
  }

  /**
   * Handle urgent task interrupts
   */
  async handleUrgentTask(urgentTask, currentWorkflow) {
    const task = {
      description: 'Handle urgent task interrupt in current workflow',
      type: 'urgent_scheduling',
      context: {
        urgentTask,
        currentWorkflow,
        preemption: true,
        minimizeDisruption: true,
        fallbackPlan: true
      },
      requirements: 'Integrate urgent task while minimizing workflow disruption'
    };

    return await this.processTask(task);
  }

  /**
   * Create a scheduled cron job
   */
  createCronJob(jobId, cronExpression, taskFunction, description) {
    try {
      const job = cron.schedule(cronExpression, taskFunction, {
        scheduled: false,
        timezone: 'UTC'
      });

      this.cronJobs.set(jobId, {
        job,
        expression: cronExpression,
        description,
        created: new Date(),
        lastRun: null
      });

      logger.info(`Cron job ${jobId} created: ${description}`);
      return { success: true, jobId };

    } catch (error) {
      logger.error(`Failed to create cron job ${jobId}:`, error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Start a scheduled cron job
   */
  startCronJob(jobId) {
    const cronJob = this.cronJobs.get(jobId);
    if (!cronJob) {
      return { success: false, error: 'Job not found' };
    }

    cronJob.job.start();
    logger.info(`Cron job ${jobId} started`);
    return { success: true };
  }

  /**
   * Stop a scheduled cron job
   */
  stopCronJob(jobId) {
    const cronJob = this.cronJobs.get(jobId);
    if (!cronJob) {
      return { success: false, error: 'Job not found' };
    }

    cronJob.job.stop();
    logger.info(`Cron job ${jobId} stopped`);
    return { success: true };
  }

  /**
   * Get status of all scheduled jobs
   */
  getCronJobStatus() {
    const status = {};
    for (const [jobId, cronJob] of this.cronJobs) {
      status[jobId] = {
        description: cronJob.description,
        expression: cronJob.expression,
        created: cronJob.created,
        lastRun: cronJob.lastRun,
        running: cronJob.job.running
      };
    }
    return status;
  }
}