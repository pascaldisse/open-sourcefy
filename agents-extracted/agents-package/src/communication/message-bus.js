import EventEmitter from 'events';
import { logger, auditLogger } from '../utils/logger.js';
import { v4 as uuidv4 } from 'uuid';

/**
 * Message Bus for inter-agent communication
 * Handles message routing, delivery, and acknowledgment
 */
export class MessageBus extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.agents = new Map();
    this.messageHistory = [];
    this.pendingMessages = new Map();
    this.subscriptions = new Map();
    this.messageTimeout = options.messageTimeout || 30000; // 30 seconds
    this.maxHistorySize = options.maxHistorySize || 1000;
    this.retryAttempts = options.retryAttempts || 3;
    
    this.setupEventHandlers();
  }

  /**
   * Set up event handlers for the message bus
   */
  setupEventHandlers() {
    this.on('error', (error) => {
      logger.error('Message Bus error:', error);
    });

    this.on('message_sent', (message) => {
      auditLogger.info('Message sent', {
        messageId: message.id,
        from: message.from,
        to: message.to,
        type: message.type
      });
    });

    this.on('message_delivered', (message) => {
      auditLogger.info('Message delivered', {
        messageId: message.id,
        from: message.from,
        to: message.to,
        deliveryTime: Date.now() - message.timestamp
      });
    });
  }

  /**
   * Register an agent with the message bus
   */
  registerAgent(agent) {
    if (!agent || !agent.agentId) {
      throw new Error('Invalid agent: must have agentId');
    }

    this.agents.set(agent.agentId, {
      agent,
      lastSeen: Date.now(),
      messageCount: 0,
      status: 'online'
    });

    // Set the communication channel for the agent
    agent.setCommunicationChannel(this);

    logger.info(`Agent ${agent.agentId} registered with message bus`);
    this.emit('agent_registered', agent.agentId);

    return true;
  }

  /**
   * Unregister an agent from the message bus
   */
  unregisterAgent(agentId) {
    const agentInfo = this.agents.get(agentId);
    if (!agentInfo) {
      return false;
    }

    this.agents.delete(agentId);
    
    // Clean up any pending messages for this agent
    this.cleanupAgentMessages(agentId);

    logger.info(`Agent ${agentId} unregistered from message bus`);
    this.emit('agent_unregistered', agentId);

    return true;
  }

  /**
   * Send a message to a specific agent
   */
  async send(message) {
    try {
      // Validate message format
      this.validateMessage(message);

      // Add metadata
      const enrichedMessage = {
        ...message,
        id: message.id || uuidv4(),
        timestamp: message.timestamp || Date.now(),
        attempts: 0,
        status: 'pending'
      };

      // Check if target agent exists
      const targetAgent = this.agents.get(message.to);
      if (!targetAgent) {
        throw new Error(`Target agent ${message.to} not found`);
      }

      // Store message in history
      this.addToHistory(enrichedMessage);

      // Emit message sent event
      this.emit('message_sent', enrichedMessage);

      // Deliver message
      const deliveryResult = await this.deliverMessage(enrichedMessage);

      return {
        messageId: enrichedMessage.id,
        delivered: deliveryResult.success,
        response: deliveryResult.response,
        error: deliveryResult.error
      };

    } catch (error) {
      logger.error('Failed to send message:', error);
      throw error;
    }
  }

  /**
   * Broadcast a message to all agents (except sender)
   */
  async broadcast(message, excludeAgent = null) {
    const results = [];
    
    for (const [agentId] of this.agents) {
      if (agentId !== excludeAgent && agentId !== message.from) {
        try {
          const broadcastMessage = {
            ...message,
            to: agentId,
            type: 'broadcast'
          };
          
          const result = await this.send(broadcastMessage);
          results.push({ agentId, result });
        } catch (error) {
          results.push({ agentId, error: error.message });
        }
      }
    }

    return results;
  }

  /**
   * Subscribe to messages of a specific type
   */
  subscribe(agentId, messageType, handler) {
    if (!this.subscriptions.has(agentId)) {
      this.subscriptions.set(agentId, new Map());
    }

    const agentSubscriptions = this.subscriptions.get(agentId);
    agentSubscriptions.set(messageType, handler);

    logger.debug(`Agent ${agentId} subscribed to ${messageType} messages`);
  }

  /**
   * Unsubscribe from messages of a specific type
   */
  unsubscribe(agentId, messageType) {
    const agentSubscriptions = this.subscriptions.get(agentId);
    if (agentSubscriptions) {
      agentSubscriptions.delete(messageType);
      logger.debug(`Agent ${agentId} unsubscribed from ${messageType} messages`);
    }
  }

  /**
   * Validate message format
   */
  validateMessage(message) {
    if (!message || typeof message !== 'object') {
      throw new Error('Message must be an object');
    }

    if (!message.from || !message.to) {
      throw new Error('Message must have from and to fields');
    }

    if (!message.type) {
      throw new Error('Message must have a type');
    }

    if (!message.content) {
      throw new Error('Message must have content');
    }
  }

  /**
   * Deliver message to target agent
   */
  async deliverMessage(message) {
    const targetAgentInfo = this.agents.get(message.to);
    if (!targetAgentInfo) {
      return { success: false, error: 'Target agent not found' };
    }

    const { agent } = targetAgentInfo;

    try {
      // Check if agent has a subscription handler for this message type
      const agentSubscriptions = this.subscriptions.get(message.to);
      if (agentSubscriptions && agentSubscriptions.has(message.type)) {
        const handler = agentSubscriptions.get(message.type);
        const response = await handler(message);
        
        this.emit('message_delivered', message);
        return { success: true, response };
      }

      // Use agent's default message handler
      const response = await agent.handleMessage(message);
      
      // Update agent's last seen time and message count
      targetAgentInfo.lastSeen = Date.now();
      targetAgentInfo.messageCount++;

      this.emit('message_delivered', message);
      
      return { success: true, response };

    } catch (error) {
      logger.error(`Failed to deliver message ${message.id} to agent ${message.to}:`, error);
      
      // Retry logic
      if (message.attempts < this.retryAttempts) {
        message.attempts++;
        logger.info(`Retrying message ${message.id} (attempt ${message.attempts})`);
        
        // Schedule retry after delay
        setTimeout(() => {
          this.deliverMessage(message);
        }, 1000 * message.attempts); // Exponential backoff
        
        return { success: false, error: 'Delivery failed, retrying' };
      }

      return { success: false, error: error.message };
    }
  }

  /**
   * Add message to history with size management
   */
  addToHistory(message) {
    this.messageHistory.push({
      ...message,
      processedAt: Date.now()
    });

    // Maintain history size limit
    if (this.messageHistory.length > this.maxHistorySize) {
      this.messageHistory.shift();
    }
  }

  /**
   * Get message history with optional filtering
   */
  getMessageHistory(filter = {}) {
    let filteredHistory = [...this.messageHistory];

    if (filter.agentId) {
      filteredHistory = filteredHistory.filter(
        msg => msg.from === filter.agentId || msg.to === filter.agentId
      );
    }

    if (filter.type) {
      filteredHistory = filteredHistory.filter(msg => msg.type === filter.type);
    }

    if (filter.since) {
      filteredHistory = filteredHistory.filter(msg => msg.timestamp >= filter.since);
    }

    if (filter.limit) {
      filteredHistory = filteredHistory.slice(-filter.limit);
    }

    return filteredHistory;
  }

  /**
   * Get communication statistics
   */
  getStatistics() {
    const stats = {
      totalAgents: this.agents.size,
      totalMessages: this.messageHistory.length,
      messagesByType: {},
      messagesByAgent: {},
      averageResponseTime: 0,
      onlineAgents: 0
    };

    // Count online agents
    for (const [agentId, agentInfo] of this.agents) {
      if (agentInfo.status === 'online') {
        stats.onlineAgents++;
      }
    }

    // Analyze message history
    let totalResponseTime = 0;
    let responseCount = 0;

    for (const message of this.messageHistory) {
      // Count by type
      stats.messagesByType[message.type] = (stats.messagesByType[message.type] || 0) + 1;

      // Count by agent
      stats.messagesByAgent[message.from] = (stats.messagesByAgent[message.from] || 0) + 1;

      // Calculate response times
      if (message.processedAt && message.timestamp) {
        totalResponseTime += message.processedAt - message.timestamp;
        responseCount++;
      }
    }

    if (responseCount > 0) {
      stats.averageResponseTime = totalResponseTime / responseCount;
    }

    return stats;
  }

  /**
   * Clean up messages for a specific agent
   */
  cleanupAgentMessages(agentId) {
    // Remove pending messages
    for (const [messageId, message] of this.pendingMessages) {
      if (message.to === agentId || message.from === agentId) {
        this.pendingMessages.delete(messageId);
      }
    }

    // Remove subscriptions
    this.subscriptions.delete(agentId);
  }

  /**
   * Health check for all registered agents
   */
  async performHealthCheck() {
    const healthResults = {};
    const healthCheckMessage = {
      type: 'health_check',
      content: { timestamp: Date.now() },
      from: 'message_bus',
      requiresResponse: true
    };

    for (const [agentId] of this.agents) {
      try {
        const result = await this.send({
          ...healthCheckMessage,
          to: agentId
        });

        healthResults[agentId] = {
          status: result.delivered ? 'healthy' : 'unhealthy',
          responseTime: result.response?.duration || 0,
          lastCheck: Date.now()
        };

      } catch (error) {
        healthResults[agentId] = {
          status: 'error',
          error: error.message,
          lastCheck: Date.now()
        };
      }
    }

    return healthResults;
  }

  /**
   * Disconnect the message bus and clean up
   */
  async disconnect() {
    logger.info('Disconnecting message bus');

    // Notify all agents of shutdown
    const shutdownMessage = {
      type: 'system_shutdown',
      content: { reason: 'Message bus disconnecting' },
      from: 'message_bus'
    };

    await this.broadcast(shutdownMessage);

    // Clear all data
    this.agents.clear();
    this.messageHistory.length = 0;
    this.pendingMessages.clear();
    this.subscriptions.clear();

    // Remove all event listeners
    this.removeAllListeners();

    logger.info('Message bus disconnected');
  }
}