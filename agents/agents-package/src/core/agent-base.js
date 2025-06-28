import { query } from '@anthropic-ai/claude-code';
import { logger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * Base class for all agents in the multi-agent system using Claude Code SDK
 */
export class AgentBase {
  constructor(agentId, name, role, capabilities = [], allowedTools = []) {
    this.agentId = agentId;
    this.name = name;
    this.role = role;
    this.capabilities = capabilities;
    this.allowedTools = allowedTools;
    this.status = 'idle';
    this.currentTask = null;
    this.performance = {
      tasksCompleted: 0,
      tasksSuccess: 0,
      tasksFailed: 0,
      averageResponseTime: 0,
      totalCostUsd: 0
    };
    this.communicationChannel = null;
    this.abortController = new AbortController();
    this.sessionId = null;
    this.startTime = Date.now();
    
    // Performance optimization components
    this.optimizer = null;
    this.cache = null;
    this.claudePool = null;
    
    this.initialize();
  }

  /**
   * Initialize the agent
   */
  async initialize() {
    try {
      logger.info(`Agent ${this.agentId} (${this.name}) initialized successfully`);
    } catch (error) {
      logger.error(`Failed to initialize agent ${this.agentId}:`, error);
      throw error;
    }
  }

  /**
   * Generate system prompt based on agent role and capabilities
   */
  generateSystemPrompt() {
    return `You are ${this.name} (Agent ${this.agentId}), specializing in ${this.role}.

Your capabilities include:
${this.capabilities.map(cap => `- ${cap}`).join('\n')}

Your allowed tools: ${this.allowedTools.join(', ')}

Your responsibilities:
- Follow the chain of command with Agent 0 as the master coordinator
- Communicate status updates regularly
- Validate all inputs before processing
- Handle errors gracefully with proper fallbacks
- Log all significant actions for audit purposes
- Respect permission boundaries defined in your profile

Communication Protocol:
- Use structured JSON for inter-agent messages
- Include timestamps and agent identifiers
- Acknowledge received messages
- Escalate issues to Agent 0 when needed

Remember: You are part of a coordinated system. Your effectiveness depends on clear communication and adherence to established protocols.`;
  }

  /**
   * Process a task assigned by Agent 0 or another authorized agent using Claude Code SDK
   */
  async processTask(task) {
    const taskId = uuidv4();
    const startTime = Date.now();

    // Check cache first if available
    if (this.cache && task.cacheable !== false) {
      const cacheKey = this.generateCacheKey(task);
      const cached = await this.cache.get(cacheKey);
      
      if (cached) {
        logger.debug(`Agent ${this.agentId} cache hit for task: ${task.description}`);
        return {
          success: true,
          taskId,
          result: cached,
          duration: 0,
          cost: 0,
          agentId: this.agentId,
          fromCache: true
        };
      }
    }

    try {
      this.status = 'working';
      this.currentTask = { ...task, taskId, startTime };

      logger.info(`Agent ${this.agentId} starting task: ${task.description}`);

      // Validate task
      this.validateTask(task);

      // Get pooled connection if available
      let connection = null;
      if (this.claudePool) {
        connection = await this.claudePool.acquire();
      }

      // Create abort controller for this task
      const taskAbortController = connection?.abortController || new AbortController();
      
      // Set timeout if specified
      if (task.timeout) {
        setTimeout(() => taskAbortController.abort(), task.timeout);
      }

      const messages = [];
      
      // Execute task using Claude Code SDK
      for await (const message of query({
        prompt: this.formatTaskInput(task),
        abortController: taskAbortController,
        options: {
          maxTurns: task.maxTurns || 3,
          systemPrompt: this.generateSystemPrompt(),
          allowedTools: this.allowedTools,
          outputFormat: 'json',
          cwd: process.cwd()
        }
      })) {
        messages.push(message);
        
        // Log progress for debugging
        if (message.type === 'assistant') {
          logger.debug(`Agent ${this.agentId} received assistant message`);
        }
      }

      // Release connection back to pool
      if (connection && this.claudePool) {
        this.claudePool.release(connection);
      }

      // Extract result from final message
      const finalMessage = messages[messages.length - 1];
      let result = '';
      let cost = 0;

      if (finalMessage?.type === 'result') {
        result = finalMessage.result || '';
        cost = finalMessage.total_cost_usd || 0;
        this.sessionId = finalMessage.session_id;
      }

      // Cache result if appropriate
      if (this.cache && task.cacheable !== false && result) {
        const cacheKey = this.generateCacheKey(task);
        await this.cache.set(cacheKey, result, {
          ttl: task.cacheTTL,
          pattern: 'task-result'
        });
      }

      // Update performance metrics
      const duration = Date.now() - startTime;
      this.updatePerformanceMetrics(true, duration, cost);

      this.status = 'idle';
      this.currentTask = null;

      logger.info(`Agent ${this.agentId} completed task ${taskId} in ${duration}ms (cost: $${cost})`);

      return {
        success: true,
        taskId,
        result,
        duration,
        cost,
        agentId: this.agentId,
        sessionId: this.sessionId,
        messages
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      this.updatePerformanceMetrics(false, duration, 0);
      
      this.status = 'error';
      logger.error(`Agent ${this.agentId} failed task ${taskId}:`, error);

      return {
        success: false,
        taskId,
        error: error.message,
        duration,
        agentId: this.agentId
      };
    }
  }

  /**
   * Continue a conversation from a previous session
   */
  async continueTask(additionalPrompt, sessionId = null) {
    const taskId = uuidv4();
    const startTime = Date.now();

    try {
      this.status = 'working';

      const messages = [];
      const useSessionId = sessionId || this.sessionId;

      for await (const message of query({
        prompt: additionalPrompt,
        abortController: new AbortController(),
        options: {
          maxTurns: 3,
          systemPrompt: this.generateSystemPrompt(),
          allowedTools: this.allowedTools,
          outputFormat: 'json',
          resume: useSessionId,
          cwd: process.cwd()
        }
      })) {
        messages.push(message);
      }

      const finalMessage = messages[messages.length - 1];
      let result = '';
      let cost = 0;

      if (finalMessage?.type === 'result') {
        result = finalMessage.result || '';
        cost = finalMessage.total_cost_usd || 0;
      }

      const duration = Date.now() - startTime;
      this.updatePerformanceMetrics(true, duration, cost);
      this.status = 'idle';

      return {
        success: true,
        taskId,
        result,
        duration,
        cost,
        agentId: this.agentId,
        messages
      };

    } catch (error) {
      const duration = Date.now() - startTime;
      this.updatePerformanceMetrics(false, duration, 0);
      this.status = 'error';

      return {
        success: false,
        taskId,
        error: error.message,
        duration,
        agentId: this.agentId
      };
    }
  }

  /**
   * Validate task before processing
   */
  validateTask(task) {
    if (!task || typeof task !== 'object') {
      throw new Error('Invalid task: must be an object');
    }

    if (!task.description || typeof task.description !== 'string') {
      throw new Error('Invalid task: description is required');
    }

    if (!task.type || typeof task.type !== 'string') {
      throw new Error('Invalid task: type is required');
    }

    // Check if agent has capability for this task type
    if (!this.canHandleTaskType(task.type)) {
      throw new Error(`Agent ${this.agentId} cannot handle task type: ${task.type}`);
    }
  }

  /**
   * Check if agent can handle a specific task type
   */
  canHandleTaskType(taskType) {
    // This should be overridden by specific agent implementations
    return true;
  }

  /**
   * Format task input for Claude Code SDK
   */
  formatTaskInput(task) {
    return `Task: ${task.description}

Type: ${task.type}
Priority: ${task.priority || 'medium'}
Context: ${JSON.stringify(task.context || {})}
Requirements: ${task.requirements || 'Complete the task according to system guidelines'}

Please execute this task using your specialized capabilities as ${this.role}. Follow all system protocols and provide detailed feedback on your progress.`;
  }

  /**
   * Update performance metrics
   */
  updatePerformanceMetrics(success, duration, cost = 0) {
    this.performance.tasksCompleted++;
    this.performance.totalCostUsd += cost;
    
    if (success) {
      this.performance.tasksSuccess++;
    } else {
      this.performance.tasksFailed++;
    }

    // Update average response time
    const totalTasks = this.performance.tasksCompleted;
    this.performance.averageResponseTime = 
      ((this.performance.averageResponseTime * (totalTasks - 1)) + duration) / totalTasks;
  }

  /**
   * Send message to another agent
   */
  async sendMessage(targetAgentId, message) {
    if (!this.communicationChannel) {
      throw new Error('Communication channel not established');
    }

    const formattedMessage = {
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      from: this.agentId,
      to: targetAgentId,
      type: message.type || 'general',
      content: message.content,
      requiresResponse: message.requiresResponse || false
    };

    await this.communicationChannel.send(formattedMessage);
    logger.debug(`Agent ${this.agentId} sent message to Agent ${targetAgentId}`);

    return formattedMessage.id;
  }

  /**
   * Handle incoming messages
   */
  async handleMessage(message) {
    logger.debug(`Agent ${this.agentId} received message from Agent ${message.from}`);

    try {
      // Process message based on type
      switch (message.type) {
        case 'task':
          return await this.processTask(message.content);
        case 'continue_task':
          return await this.continueTask(message.content.prompt, message.content.sessionId);
        case 'status_request':
          return this.getStatus();
        case 'health_check':
          return { status: 'healthy', agentId: this.agentId };
        case 'shutdown':
          await this.shutdown();
          return { status: 'shutdown', agentId: this.agentId };
        default:
          logger.warn(`Agent ${this.agentId} received unknown message type: ${message.type}`);
          return { error: 'Unknown message type' };
      }
    } catch (error) {
      logger.error(`Agent ${this.agentId} error handling message:`, error);
      return { error: error.message };
    }
  }

  /**
   * Set optimization components
   */
  setOptimizationComponents(optimizer, cache, claudePool) {
    this.optimizer = optimizer;
    this.cache = cache;
    this.claudePool = claudePool;
  }
  
  /**
   * Generate cache key for task
   */
  generateCacheKey(task) {
    return `agent-${this.agentId}:${task.type}:${JSON.stringify({
      description: task.description,
      context: task.context,
      requirements: task.requirements
    })}`;
  }
  
  /**
   * Get current agent status
   */
  getStatus() {
    return {
      agentId: this.agentId,
      name: this.name,
      role: this.role,
      status: this.status,
      currentTask: this.currentTask,
      performance: this.performance,
      capabilities: this.capabilities,
      allowedTools: this.allowedTools,
      sessionId: this.sessionId,
      uptime: Date.now() - this.startTime,
      optimization: {
        hasCache: !!this.cache,
        hasPool: !!this.claudePool,
        hasOptimizer: !!this.optimizer
      }
    };
  }

  /**
   * Set communication channel
   */
  setCommunicationChannel(channel) {
    this.communicationChannel = channel;
  }

  /**
   * Abort current operations
   */
  abort() {
    this.abortController.abort();
    this.status = 'aborted';
    logger.info(`Agent ${this.agentId} operations aborted`);
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    logger.info(`Shutting down agent ${this.agentId} (${this.name})`);
    
    if (this.currentTask) {
      logger.warn(`Agent ${this.agentId} shutting down with active task`);
      this.abort();
    }

    this.status = 'shutdown';
    
    if (this.communicationChannel) {
      await this.communicationChannel.disconnect();
    }
  }
}