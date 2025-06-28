import { AgentBase } from '../core/agent-base.js';
import { logger, auditLogger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * Agent 0: Master Coordinator
 * The central agent that coordinates and oversees all other agents in the system
 */
export class Agent0 extends AgentBase {
  constructor() {
    super(
      0,
      'Master Coordinator',
      'System orchestrator and decision maker',
      [
        'Task delegation and assignment',
        'Progress monitoring across all agents',
        'Quality assurance and validation',
        'Permission control and security oversight',
        'Intervention and error correction',
        'Condition checking and loop management',
        'System health monitoring',
        'Compliance enforcement'
      ],
      [
        'Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 
        'LS', 'TodoRead', 'TodoWrite', 'Task'
      ]
    );

    this.managedAgents = new Map();
    this.taskQueue = [];
    this.activeConditions = new Map();
    this.systemState = {
      totalTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      totalCost: 0,
      systemHealth: 'healthy'
    };
    this.loopActive = false;
    this.maxConcurrentTasks = 5;
    
    // Dynamic task mapping - can be overridden
    this.taskAgentMapping = {
      'testing': [1],
      'documentation': [2],
      'bug_fixing': [3],
      'code_commenting': [4],
      'git_operations': [5],
      'task_scheduling': [6],
      'code_review': [7],
      'deployment': [8],
      'performance_optimization': [9],
      'security_audit': [10],
      'data_analysis': [11],
      'integration': [12],
      'configuration': [13],
      'backup': [14],
      'compliance': [15],
      'research': [16]
    };
    
    // Custom system prompt support
    this.systemPrompt = null;
  }

  /**
   * Register an agent under this coordinator's management
   */
  registerAgent(agent) {
    this.managedAgents.set(agent.agentId, agent);
    auditLogger.info(`Agent ${agent.agentId} registered with coordinator`, {
      agentId: agent.agentId,
      name: agent.name,
      role: agent.role
    });
    logger.info(`Agent 0 registered Agent ${agent.agentId} (${agent.name})`);
  }

  /**
   * Delegate a task to the most appropriate agent
   */
  async delegateTask(task) {
    const taskId = uuidv4();
    const startTime = Date.now();

    try {
      auditLogger.info('Task delegation initiated', {
        taskId,
        task: task.description,
        type: task.type,
        priority: task.priority
      });

      // Find the best agent for this task
      const selectedAgent = this.selectAgentForTask(task);
      
      if (!selectedAgent) {
        throw new Error(`No suitable agent found for task type: ${task.type}`);
      }

      // Check if agent is available
      if (selectedAgent.status !== 'idle') {
        // Queue the task if agent is busy
        this.taskQueue.push({
          ...task,
          taskId,
          assignedAgentId: selectedAgent.agentId,
          queuedAt: Date.now()
        });

        logger.info(`Task ${taskId} queued for Agent ${selectedAgent.agentId}`);
        return {
          success: true,
          taskId,
          status: 'queued',
          assignedAgent: selectedAgent.agentId
        };
      }

      // Assign task directly
      const result = await this.assignTaskToAgent(selectedAgent, { ...task, taskId });
      
      // Update system state
      this.systemState.totalTasks++;
      if (result.success) {
        this.systemState.completedTasks++;
      } else {
        this.systemState.failedTasks++;
      }
      this.systemState.totalCost += result.cost || 0;

      return result;

    } catch (error) {
      logger.error(`Agent 0 failed to delegate task ${taskId}:`, error);
      auditLogger.error('Task delegation failed', {
        taskId,
        error: error.message
      });

      return {
        success: false,
        taskId,
        error: error.message
      };
    }
  }

  /**
   * Select the most appropriate agent for a given task
   */
  selectAgentForTask(task) {
    // Use dynamic task mapping
    const potentialAgents = this.taskAgentMapping[task.type] || [];
    
    // Find the best available agent
    for (const agentId of potentialAgents) {
      const agent = this.managedAgents.get(agentId);
      if (agent && agent.status === 'idle') {
        return agent;
      }
    }

    // If no specialized agent is available, find any idle agent that can handle the task
    for (const [agentId, agent] of this.managedAgents) {
      if (agentId !== 0 && agent.status === 'idle' && agent.canHandleTaskType(task.type)) {
        return agent;
      }
    }

    return null;
  }

  /**
   * Assign a task to a specific agent
   */
  async assignTaskToAgent(agent, task) {
    try {
      logger.info(`Agent 0 assigning task ${task.taskId} to Agent ${agent.agentId}`);
      
      // Send task message to agent
      const result = await agent.processTask(task);
      
      auditLogger.info('Task assignment completed', {
        taskId: task.taskId,
        agentId: agent.agentId,
        success: result.success,
        duration: result.duration,
        cost: result.cost
      });

      return result;

    } catch (error) {
      logger.error(`Failed to assign task to Agent ${agent.agentId}:`, error);
      return {
        success: false,
        taskId: task.taskId,
        error: error.message,
        agentId: agent.agentId
      };
    }
  }

  /**
   * Monitor progress of all managed agents
   */
  async monitorProgress() {
    const statuses = new Map();
    
    for (const [agentId, agent] of this.managedAgents) {
      try {
        const status = agent.getStatus();
        statuses.set(agentId, status);
      } catch (error) {
        logger.error(`Failed to get status from Agent ${agentId}:`, error);
        statuses.set(agentId, { error: error.message, agentId });
      }
    }

    return statuses;
  }

  /**
   * Perform quality assurance validation
   */
  async performQualityAssurance(result, standards) {
    // Use Claude Code SDK to validate result quality
    const qaTask = {
      description: `Validate the quality of this result according to system standards`,
      type: 'quality_assurance',
      context: {
        result: result,
        standards: standards
      },
      requirements: 'Provide detailed assessment of quality, compliance, and recommendations'
    };

    return await this.processTask(qaTask);
  }

  /**
   * Handle permission requests from agents
   */
  async handlePermissionRequest(agentId, action, context) {
    auditLogger.info('Permission request received', {
      requestingAgent: agentId,
      action,
      context
    });

    // Define permission matrix
    const permissions = {
      // Core agents have write access to their domains
      1: ['test_files', 'ci_config'],
      2: ['documentation'],
      3: ['source_code_fixes'],
      4: ['code_comments'],
      5: ['git_operations'],
      6: ['task_management'],
      // Extended agents have more restricted access
      7: ['code_review_comments'],
      8: ['deployment_configs'],
      9: ['performance_configs'],
      10: ['security_reports'],
      11: ['analytics_reports'],
      12: ['integration_configs'],
      13: ['app_configs'],
      14: ['backup_operations'],
      15: ['compliance_reports'],
      16: ['research_documents']
    };

    const agentPermissions = permissions[agentId] || [];
    const isPermitted = agentPermissions.includes(action);

    auditLogger.info('Permission decision', {
      requestingAgent: agentId,
      action,
      granted: isPermitted
    });

    return {
      granted: isPermitted,
      reason: isPermitted ? 'Action permitted for agent role' : 'Action not in agent permission scope'
    };
  }

  /**
   * Intervene when an agent needs correction
   */
  async intervene(agentId, issue, correctionAction) {
    logger.warn(`Agent 0 intervening with Agent ${agentId}: ${issue}`);
    
    auditLogger.warn('Agent intervention triggered', {
      targetAgent: agentId,
      issue,
      correctionAction
    });

    const agent = this.managedAgents.get(agentId);
    if (!agent) {
      throw new Error(`Agent ${agentId} not found`);
    }

    // Abort current agent operations if needed
    if (correctionAction === 'abort') {
      agent.abort();
    }

    // Send correction message
    const correctionResult = await this.assignTaskToAgent(agent, {
      taskId: uuidv4(),
      description: `Correction required: ${issue}`,
      type: 'correction',
      priority: 'high',
      context: { correctionAction }
    });

    return correctionResult;
  }

  /**
   * Start the main coordination loop
   */
  async startLoop(conditions = []) {
    this.loopActive = true;
    logger.info('Agent 0 starting coordination loop');

    // Set up conditions
    for (const condition of conditions) {
      this.activeConditions.set(condition.id, condition);
    }

    while (this.loopActive) {
      try {
        // Process task queue
        await this.processTaskQueue();

        // Monitor agent health
        await this.monitorAgentHealth();

        // Check conditions
        const conditionsMet = await this.checkConditions();
        
        if (conditionsMet) {
          logger.info('All conditions met, stopping loop');
          break;
        }

        // Wait before next iteration
        await new Promise(resolve => setTimeout(resolve, 5000));

      } catch (error) {
        logger.error('Error in coordination loop:', error);
        await new Promise(resolve => setTimeout(resolve, 10000));
      }
    }

    logger.info('Agent 0 coordination loop stopped');
  }

  /**
   * Process queued tasks
   */
  async processTaskQueue() {
    if (this.taskQueue.length === 0) return;

    const activeTasks = Array.from(this.managedAgents.values())
      .filter(agent => agent.status === 'working').length;

    if (activeTasks >= this.maxConcurrentTasks) return;

    // Sort queue by priority and age
    this.taskQueue.sort((a, b) => {
      const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
      const priorityDiff = (priorityOrder[b.priority] || 2) - (priorityOrder[a.priority] || 2);
      if (priorityDiff !== 0) return priorityDiff;
      return a.queuedAt - b.queuedAt;
    });

    // Process tasks
    const tasksToProcess = this.taskQueue.splice(0, this.maxConcurrentTasks - activeTasks);
    
    for (const task of tasksToProcess) {
      const agent = this.managedAgents.get(task.assignedAgentId);
      if (agent && agent.status === 'idle') {
        this.assignTaskToAgent(agent, task);
      } else {
        // Re-queue if agent still not available
        this.taskQueue.push(task);
      }
    }
  }

  /**
   * Monitor health of all agents
   */
  async monitorAgentHealth() {
    for (const [agentId, agent] of this.managedAgents) {
      try {
        const status = agent.getStatus();
        
        // Check for stuck agents
        if (status.currentTask && 
            Date.now() - status.currentTask.startTime > 300000) { // 5 minutes
          logger.warn(`Agent ${agentId} appears stuck on task`);
          await this.intervene(agentId, 'Task timeout', 'abort');
        }

        // Check performance degradation
        if (status.performance.tasksFailed > status.performance.tasksSuccess) {
          logger.warn(`Agent ${agentId} has high failure rate`);
        }

      } catch (error) {
        logger.error(`Health check failed for Agent ${agentId}:`, error);
      }
    }
  }

  /**
   * Check if all conditions are met
   */
  async checkConditions() {
    for (const [conditionId, condition] of this.activeConditions) {
      try {
        let conditionMet = false;

        if (condition.type === 'boolean') {
          conditionMet = await this.evaluateBooleanCondition(condition);
        } else if (condition.type === 'llm_validated') {
          conditionMet = await this.evaluateLLMCondition(condition);
        }

        if (!conditionMet) {
          return false;
        }

      } catch (error) {
        logger.error(`Error checking condition ${conditionId}:`, error);
        return false;
      }
    }

    return this.activeConditions.size > 0;
  }

  /**
   * Evaluate a boolean condition
   */
  async evaluateBooleanCondition(condition) {
    // Example boolean conditions
    switch (condition.check) {
      case 'all_tests_pass':
        return this.systemState.failedTasks === 0;
      case 'documentation_complete':
        // Check if documentation agent has completed all tasks
        const docAgent = this.managedAgents.get(2);
        return docAgent && docAgent.status === 'idle';
      default:
        return false;
    }
  }

  /**
   * Evaluate an LLM-validated condition
   */
  async evaluateLLMCondition(condition) {
    const evaluationTask = {
      description: `Evaluate if this condition is met: ${condition.description}`,
      type: 'condition_evaluation',
      context: {
        condition: condition,
        systemState: this.systemState
      },
      requirements: 'Return true if condition is met, false otherwise'
    };

    const result = await this.processTask(evaluationTask);
    return result.success && result.result.toLowerCase().includes('true');
  }

  /**
   * Stop the coordination loop
   */
  stopLoop() {
    this.loopActive = false;
    logger.info('Agent 0 coordination loop stop requested');
  }

  /**
   * Get comprehensive system status
   */
  getSystemStatus() {
    const agentStatuses = {};
    for (const [agentId, agent] of this.managedAgents) {
      agentStatuses[agentId] = agent.getStatus();
    }

    return {
      coordinator: this.getStatus(),
      managedAgents: agentStatuses,
      systemState: this.systemState,
      taskQueue: this.taskQueue.length,
      activeConditions: Array.from(this.activeConditions.keys()),
      loopActive: this.loopActive
    };
  }

  /**
   * Override task type validation for coordinator
   */
  canHandleTaskType(taskType) {
    // Agent 0 can handle coordination and management tasks
    const coordinatorTasks = [
      'delegation',
      'monitoring',
      'quality_assurance',
      'condition_evaluation',
      'intervention',
      'system_management'
    ];
    
    return coordinatorTasks.includes(taskType);
  }
}