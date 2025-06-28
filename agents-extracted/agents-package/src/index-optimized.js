import dotenv from 'dotenv';
import { logger } from './utils/logger.js';
import { MessageBus } from './communication/message-bus.js';
import { QualityAssurance } from './core/quality-assurance.js';
import { ConditionManager } from './core/condition-manager.js';
import { InterventionSystem } from './core/intervention-system.js';
import { SystemMonitor } from './monitoring/system-monitor.js';
import { PerformanceOptimizer } from './performance/performance-optimizer.js';
import { CacheLayer } from './performance/cache-layer.js';
import { ClaudePool } from './performance/claude-pool.js';
import { Agent0 } from './agents/agent-0.js';
import { Agent1 } from './agents/agent-1.js';
import { Agent2 } from './agents/agent-2.js';
import { Agent3 } from './agents/agent-3.js';
import { Agent4 } from './agents/agent-4.js';
import { Agent5 } from './agents/agent-5.js';
import { Agent6 } from './agents/agent-6.js';
import { 
  Agent7, Agent8, Agent9, Agent10, Agent11, Agent12,
  Agent13, Agent14, Agent15, Agent16 
} from './agents/extended-agents.js';

// Load environment variables
dotenv.config();

/**
 * Optimized Multi-Agent System with Performance Enhancements
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
    
    // Performance optimization components
    this.performanceOptimizer = null;
    this.cacheLayer = null;
    this.claudePool = null;
    
    this.isRunning = false;
    this.taskPromises = new Map();
  }

  /**
   * Initialize the multi-agent system
   */
  async initialize() {
    try {
      logger.info('Initializing Optimized Multi-Agent System');

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
      
      // Initialize performance optimization components
      this.performanceOptimizer = new PerformanceOptimizer();
      this.cacheLayer = new CacheLayer({
        maxMemorySize: 200, // 200MB cache
        maxEntries: 20000,
        defaultTTL: 600000, // 10 minutes
        enablePatternCaching: true
      });
      this.claudePool = new ClaudePool({
        minSize: 3,
        maxSize: 15,
        acquireTimeout: 30000,
        idleTimeout: 300000
      });

      // Initialize Claude pool
      await this.claudePool.initialize();

      // Initialize agents
      await this.initializeAgents();

      // Register agents with message bus
      this.registerAgents();

      // Connect systems
      this.connectSystems();

      // Set up system event handlers
      this.setupEventHandlers();

      // Set up performance optimization handlers
      this.setupOptimizationHandlers();

      logger.info('Optimized Multi-Agent System initialized successfully');
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
      this.coordinator.setOptimizationComponents(
        this.performanceOptimizer,
        this.cacheLayer,
        this.claudePool
      );
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
        Agent16  // Research Assistant
      ];

      for (let i = 0; i < agentClasses.length; i++) {
        const AgentClass = agentClasses[i];
        const agent = new AgentClass();
        
        // Set optimization components for each agent
        agent.setOptimizationComponents(
          this.performanceOptimizer,
          this.cacheLayer,
          this.claudePool
        );
        
        this.agents.set(i + 1, agent);
      }

      // Warm up cache with agent profiles
      await this.warmupCache();

      logger.info(`Initialized ${this.agents.size} agents with optimization`);

    } catch (error) {
      logger.error('Failed to initialize agents:', error);
      throw error;
    }
  }

  /**
   * Warm up cache with common data
   */
  async warmupCache() {
    const warmupTasks = [
      {
        generator: async () => ({
          key: 'agent-profiles',
          value: this.getAgentProfiles()
        }),
        options: { ttl: 3600000, pattern: 'agent-profile' } // 1 hour
      },
      {
        generator: async () => ({
          key: 'system-config',
          value: this.getSystemConfig()
        }),
        options: { ttl: 3600000, pattern: 'agent-profile' } // 1 hour
      }
    ];

    await this.cacheLayer.warmup(warmupTasks);
  }

  /**
   * Get agent profiles for caching
   */
  getAgentProfiles() {
    const profiles = {};
    for (const [id, agent] of this.agents) {
      profiles[id] = {
        name: agent.name,
        role: agent.role,
        capabilities: agent.capabilities,
        allowedTools: agent.allowedTools
      };
    }
    return profiles;
  }

  /**
   * Get system configuration
   */
  getSystemConfig() {
    return {
      maxAgents: this.agents.size,
      optimizationEnabled: true,
      cacheConfig: this.cacheLayer.config,
      poolConfig: this.claudePool.config,
      optimizerConfig: {
        taskBatchSize: this.performanceOptimizer.taskBatchSize,
        messageBatchSize: this.performanceOptimizer.messageBatchSize
      }
    };
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
   * Set up performance optimization handlers
   */
  setupOptimizationHandlers() {
    // Handle optimized task batches
    this.performanceOptimizer.on('tasksOptimized', async (tasks) => {
      await this.processBatchedTasks(tasks);
    });

    // Handle batched messages
    this.performanceOptimizer.on('messagesBatched', async (batch) => {
      await this.deliverBatchedMessages(batch);
    });

    // Handle cache cleanup
    this.cacheLayer.on('cacheCleanup', (stats) => {
      logger.debug('Cache cleanup:', stats);
    });

    // Monitor optimizer metrics
    setInterval(() => {
      const metrics = this.performanceOptimizer.getMetrics();
      if (metrics.totalOptimizations > 0) {
        logger.info('Performance metrics:', metrics);
      }
    }, 300000); // Every 5 minutes
  }

  /**
   * Process batched tasks from optimizer
   */
  async processBatchedTasks(tasks) {
    const results = await Promise.all(
      tasks.map(task => this.coordinator.delegateTask(task))
    );
    
    // Resolve waiting promises
    results.forEach((result, index) => {
      const task = tasks[index];
      const resolver = this.taskPromises.get(task.id);
      if (resolver) {
        resolver(result);
        this.taskPromises.delete(task.id);
      }
    });
  }

  /**
   * Deliver batched messages
   */
  async deliverBatchedMessages(batch) {
    const { agentId, messages } = batch;
    const agent = this.agents.get(parseInt(agentId));
    
    if (agent) {
      // Process messages in sequence
      for (const message of messages) {
        await agent.handleMessage(message);
      }
    }
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

      logger.info('Starting Optimized Multi-Agent System');
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

      logger.info('Optimized Multi-Agent System started');

    } catch (error) {
      logger.error('Failed to start Multi-Agent System:', error);
      this.isRunning = false;
      throw error;
    }
  }

  /**
   * Execute a task through the coordinator
   */
  async executeTask(task) {
    if (!this.isRunning) {
      throw new Error('Multi-Agent System is not running');
    }

    // Use optimizer for task queuing if configured
    if (this.performanceOptimizer && task.optimize !== false) {
      const priority = task.priority || 'medium';
      const optimizedTask = this.performanceOptimizer.queueTask(task, priority);
      
      // High priority tasks are processed immediately
      if (priority !== 'high') {
        return new Promise((resolve) => {
          this.taskPromises.set(optimizedTask.id || task.id, resolve);
        });
      }
    }

    return await this.coordinator.delegateTask(task);
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

      logger.info('Stopping Optimized Multi-Agent System');

      // Stop system monitoring
      if (this.systemMonitor) {
        this.systemMonitor.stopMonitoring();
      }

      // Stop the coordinator loop
      if (this.coordinator) {
        this.coordinator.stopLoop();
      }

      // Stop optimizer
      this.performanceOptimizer.stopOptimizationLoops();

      this.isRunning = false;
      logger.info('Optimized Multi-Agent System stopped');

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
      logger.info('Shutting down Optimized Multi-Agent System');

      // Stop the system first
      await this.stop();

      // Shutdown performance components
      this.performanceOptimizer.shutdown();
      this.cacheLayer.shutdown();
      await this.claudePool.shutdown();

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

      logger.info('Optimized Multi-Agent System shutdown complete');

    } catch (error) {
      logger.error('Error during shutdown:', error);
      throw error;
    }
  }

  /**
   * Get system status with optimization metrics
   */
  getSystemStatus() {
    const status = {
      isRunning: this.isRunning,
      agents: {},
      messageBus: null,
      coordinator: null,
      optimization: null
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
    if (this.coordinator) {
      status.coordinator = this.coordinator.getSystemStatus();
    }

    // Get optimization metrics
    status.optimization = {
      optimizer: this.performanceOptimizer.getMetrics(),
      cache: this.cacheLayer.getStats(),
      pool: this.claudePool.getStats()
    };

    return status;
  }

  /**
   * Get performance report
   */
  getPerformanceReport() {
    const report = {
      timestamp: new Date().toISOString(),
      uptime: Date.now() - (this.coordinator?.startTime || Date.now()),
      optimization: {
        taskOptimizations: this.performanceOptimizer.metrics.taskQueueOptimizations,
        messagesBatched: this.performanceOptimizer.metrics.messagesBatched,
        cacheHitRate: this.cacheLayer.getStats().hitRate,
        poolUtilization: this.claudePool.getStats()
      },
      agents: {}
    };

    // Collect agent performance metrics
    for (const [agentId, agent] of this.agents) {
      const status = agent.getStatus();
      report.agents[agentId] = {
        tasksCompleted: status.performance.tasksCompleted,
        successRate: status.performance.successRate,
        averageResponseTime: status.performance.averageResponseTime,
        optimization: status.optimization
      };
    }

    return report;
  }
}

// Create and export system instance
const system = new MultiAgentSystem();

// Auto-initialize if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  (async () => {
    try {
      await system.initialize();
      
      // Example: Start with performance monitoring conditions
      const conditions = [
        {
          id: 'system_health',
          type: 'boolean',
          check: 'all_agents_healthy',
          description: 'All agents are healthy and responsive'
        },
        {
          id: 'performance_threshold',
          type: 'boolean',
          check: 'response_time_under_500ms',
          description: 'Average response time is under 500ms'
        }
      ];

      await system.start({ conditions });

      // Log system status and performance periodically
      setInterval(() => {
        const status = system.getSystemStatus();
        const perf = system.getPerformanceReport();
        
        logger.info('System Status:', {
          running: status.isRunning,
          agentCount: Object.keys(status.agents).length,
          cacheHitRate: perf.optimization.cacheHitRate,
          taskOptimizations: perf.optimization.taskOptimizations
        });
      }, 60000); // Every minute

    } catch (error) {
      logger.error('Failed to start Optimized Multi-Agent System:', error);
      process.exit(1);
    }
  })();
}

export default system;
export { MultiAgentSystem };