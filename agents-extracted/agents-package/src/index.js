import dotenv from 'dotenv';
import { logger } from './utils/logger.js';
import { MessageBus } from './communication/message-bus.js';
import { QualityAssurance } from './core/quality-assurance.js';
import { ConditionManager } from './core/condition-manager.js';
import { InterventionSystem } from './core/intervention-system.js';
import { SystemMonitor } from './monitoring/system-monitor.js';
import { Agent0 } from './agents/agent-0.js';
import { Agent1 } from './agents/agent-1.js';
import { Agent2 } from './agents/agent-2.js';
import { Agent3 } from './agents/agent-3.js';
import { Agent4 } from './agents/agent-4.js';
import { Agent5 } from './agents/agent-5.js';
import { Agent6 } from './agents/agent-6.js';
import { 
  Agent7, Agent8, Agent9, Agent10, Agent11, Agent12,
  Agent13, Agent14, Agent15
} from './agents/extended-agents.js';

// Load environment variables
dotenv.config();

/**
 * Multi-Agent System Main Entry Point
 */
class MultiAgentSystem {
  constructor() {
    this.messageBus = null;
    this.agents = new Map();
    this.coordinator = null;
    this.qualityAssurance = null;
    this.conditionManager = null;
    this.interventionSystem = null;
    this.systemMonitor = null;
    this.isRunning = false;
  }

  /**
   * Initialize the multi-agent system
   */
  async initialize() {
    try {
      logger.info('Initializing Multi-Agent System');

      // Create message bus
      this.messageBus = new MessageBus({
        messageTimeout: 30000,
        maxHistorySize: 1000,
        retryAttempts: 3
      });

      // Initialize core systems
      this.qualityAssurance = new QualityAssurance();
      this.conditionManager = new ConditionManager();
      this.interventionSystem = new InterventionSystem();
      this.systemMonitor = new SystemMonitor();

      // Initialize agents
      await this.initializeAgents();

      // Register agents with message bus
      this.registerAgents();

      // Connect systems
      this.connectSystems();

      // Set up system event handlers
      this.setupEventHandlers();

      logger.info('Multi-Agent System initialized successfully');
      return true;

    } catch (error) {
      logger.error('Failed to initialize Multi-Agent System:', error);
      throw error;
    }
  }

  /**
   * Initialize all agents
   */
  async initializeAgents() {
    try {
      // Initialize coordinator (Agent 0)
      this.coordinator = new Agent0();
      this.agents.set(0, this.coordinator);

      // Initialize all specialized agents (1-16)
      const agentClasses = [
        Agent1, // Test Engineer
        Agent2, // Documentation Specialist
        Agent3, // Bug Hunter
        Agent4, // Code Commentator
        Agent5, // Git Operations Manager
        Agent6, // Task Scheduler
        Agent7, // Code Reviewer
        Agent8, // Deployment Manager
        Agent9, // Performance Optimizer
        Agent10, // Security Auditor
        Agent11, // Data Analyst
        Agent12, // Integration Specialist
        Agent13, // Configuration Manager
        Agent14, // Backup Coordinator
        Agent15, // Compliance Monitor
      ];

      for (let i = 0; i < agentClasses.length; i++) {
        const AgentClass = agentClasses[i];
        const agent = new AgentClass();
        this.agents.set(i + 1, agent);
      }

      logger.info(`Initialized ${this.agents.size} agents`);

    } catch (error) {
      logger.error('Failed to initialize agents:', error);
      throw error;
    }
  }

  /**
   * Register all agents with the message bus
   */
  registerAgents() {
    for (const [agentId, agent] of this.agents) {
      this.messageBus.registerAgent(agent);
      
      // Register specialized agents with the coordinator
      if (agentId !== 0) {
        this.coordinator.registerAgent(agent);
      }
    }

    logger.info('All agents registered with message bus and coordinator');
  }

  /**
   * Connect all systems together
   */
  connectSystems() {
    // Connect intervention system with coordinator
    this.interventionSystem.setCoordinator(this.coordinator);

    // Connect system monitor with all components
    this.systemMonitor.setComponents(
      this.coordinator,
      this.messageBus,
      this.qualityAssurance
    );

    logger.info('All systems connected');
  }

  /**
   * Set up system event handlers
   */
  setupEventHandlers() {
    // Handle system shutdown gracefully
    process.on('SIGINT', async () => {
      logger.info('Received SIGINT, shutting down gracefully...');
      await this.shutdown();
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      logger.info('Received SIGTERM, shutting down gracefully...');
      await this.shutdown();
      process.exit(0);
    });

    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('Uncaught exception:', error);
      // Don't exit immediately, try to clean up
    });

    process.on('unhandledRejection', (reason, promise) => {
      logger.error('Unhandled rejection at:', promise, 'reason:', reason);
    });
  }

  /**
   * Start the multi-agent system
   */
  async start(options = {}) {
    try {
      if (this.isRunning) {
        logger.warn('Multi-Agent System is already running');
        return;
      }

      logger.info('Starting Multi-Agent System');
      this.isRunning = true;

      // Start system monitoring
      this.systemMonitor.startMonitoring();

      // Set up conditions if provided
      const conditions = options.conditions || [];
      for (const condition of conditions) {
        this.conditionManager.addCondition(condition);
      }

      // Start the coordinator loop
      await this.coordinator.startLoop(conditions);

      logger.info('Multi-Agent System started');

    } catch (error) {
      logger.error('Failed to start Multi-Agent System:', error);
      this.isRunning = false;
      throw error;
    }
  }

  /**
   * Stop the multi-agent system
   */
  async stop() {
    try {
      if (!this.isRunning) {
        logger.warn('Multi-Agent System is not running');
        return;
      }

      logger.info('Stopping Multi-Agent System');

      // Stop system monitoring
      if (this.systemMonitor) {
        this.systemMonitor.stopMonitoring();
      }

      // Stop the coordinator loop
      if (this.coordinator) {
        this.coordinator.stopLoop();
      }

      this.isRunning = false;
      logger.info('Multi-Agent System stopped');

    } catch (error) {
      logger.error('Error stopping Multi-Agent System:', error);
      throw error;
    }
  }

  /**
   * Shutdown the multi-agent system
   */
  async shutdown() {
    try {
      logger.info('Shutting down Multi-Agent System');

      // Stop the system first
      await this.stop();

      // Shutdown all agents
      for (const [agentId, agent] of this.agents) {
        try {
          await agent.shutdown();
        } catch (error) {
          logger.error(`Error shutting down agent ${agentId}:`, error);
        }
      }

      // Disconnect message bus
      if (this.messageBus) {
        await this.messageBus.disconnect();
      }

      logger.info('Multi-Agent System shutdown complete');

    } catch (error) {
      logger.error('Error during shutdown:', error);
      throw error;
    }
  }

  /**
   * Get system status
   */
  getSystemStatus() {
    const status = {
      isRunning: this.isRunning,
      agents: {},
      messageBus: null,
      coordinator: null
    };

    // Get agent statuses
    for (const [agentId, agent] of this.agents) {
      try {
        status.agents[agentId] = agent.getStatus();
      } catch (error) {
        status.agents[agentId] = { error: error.message };
      }
    }

    // Get message bus statistics
    if (this.messageBus) {
      status.messageBus = this.messageBus.getStatistics();
    }

    // Get coordinator system status
    if (this.coordinator && agentId === 0) {
      status.coordinator = this.coordinator.getSystemStatus();
    }

    return status;
  }

  /**
   * Execute a task through the coordinator
   */
  async executeTask(task) {
    if (!this.isRunning) {
      throw new Error('Multi-Agent System is not running');
    }

    return await this.coordinator.delegateTask(task);
  }

  /**
   * Perform system health check
   */
  async performHealthCheck() {
    const healthStatus = {
      overall: 'healthy',
      agents: {},
      messageBus: null,
      timestamp: Date.now()
    };

    // Check agent health
    for (const [agentId, agent] of this.agents) {
      try {
        const status = agent.getStatus();
        healthStatus.agents[agentId] = {
          status: status.status,
          uptime: status.uptime,
          performance: status.performance
        };

        if (status.status === 'error' || status.status === 'shutdown') {
          healthStatus.overall = 'degraded';
        }
      } catch (error) {
        healthStatus.agents[agentId] = { status: 'error', error: error.message };
        healthStatus.overall = 'degraded';
      }
    }

    // Check message bus health
    if (this.messageBus) {
      try {
        healthStatus.messageBus = await this.messageBus.performHealthCheck();
      } catch (error) {
        healthStatus.messageBus = { error: error.message };
        healthStatus.overall = 'unhealthy';
      }
    }

    return healthStatus;
  }
}

// Create and export system instance
const system = new MultiAgentSystem();

// Auto-initialize if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  (async () => {
    try {
      await system.initialize();
      
      // Example: Start with basic conditions
      const conditions = [
        {
          id: 'system_health',
          type: 'boolean',
          check: 'all_agents_healthy',
          description: 'All agents are healthy and responsive'
        }
      ];

      await system.start({ conditions });

      // Log system status periodically
      setInterval(() => {
        const status = system.getSystemStatus();
        logger.info('System Status:', {
          running: status.isRunning,
          agentCount: Object.keys(status.agents).length,
          messageBusStats: status.messageBus
        });
      }, 60000); // Every minute

    } catch (error) {
      logger.error('Failed to start Multi-Agent System:', error);
      process.exit(1);
    }
  })();
}

export default system;
export { MultiAgentSystem };