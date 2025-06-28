/**
 * Performance Optimizer for Multi-Agent System
 * 
 * Provides optimization strategies for:
 * - Task queue management
 * - Message batching
 * - Resource pooling
 * - Request deduplication
 */

import { EventEmitter } from 'events';

export class PerformanceOptimizer extends EventEmitter {
  constructor() {
    super();
    
    // Task queue optimization
    this.taskQueue = {
      high: [],
      medium: [],
      low: []
    };
    this.taskBatchSize = 5;
    this.taskBatchDelay = 100; // ms
    
    // Message batching
    this.messageBatch = new Map();
    this.messageBatchSize = 10;
    this.messageBatchTimeout = 50; // ms
    this.batchTimers = new Map();
    
    // Resource pooling
    this.resourcePool = new Map();
    this.poolConfig = {
      maxPoolSize: 10,
      idleTimeout: 300000, // 5 minutes
      warmupSize: 3
    };
    
    // Request deduplication
    this.requestCache = new Map();
    this.cacheExpiry = 60000; // 1 minute
    
    // Performance metrics
    this.metrics = {
      taskQueueOptimizations: 0,
      messagesBatched: 0,
      resourcePoolHits: 0,
      cacheHits: 0,
      totalOptimizations: 0
    };
    
    // Start optimization loops
    this.startOptimizationLoops();
  }
  
  /**
   * Start continuous optimization loops
   */
  startOptimizationLoops() {
    // Task queue optimization loop
    this.taskQueueInterval = setInterval(() => {
      this.optimizeTaskQueue();
    }, this.taskBatchDelay);
    
    // Cache cleanup loop
    this.cacheCleanupInterval = setInterval(() => {
      this.cleanupCache();
    }, this.cacheExpiry);
    
    // Resource pool maintenance
    this.poolMaintenanceInterval = setInterval(() => {
      this.maintainResourcePool();
    }, 60000); // Every minute
  }
  
  /**
   * Stop optimization loops
   */
  stopOptimizationLoops() {
    clearInterval(this.taskQueueInterval);
    clearInterval(this.cacheCleanupInterval);
    clearInterval(this.poolMaintenanceInterval);
    
    // Clear batch timers
    for (const timer of this.batchTimers.values()) {
      clearTimeout(timer);
    }
  }
  
  /**
   * Optimize task queue processing
   */
  optimizeTaskQueue() {
    const optimizedTasks = [];
    
    // Process high priority first
    if (this.taskQueue.high.length > 0) {
      optimizedTasks.push(...this.taskQueue.high.splice(0, this.taskBatchSize));
    }
    
    // Then medium priority
    const remainingSlots = this.taskBatchSize - optimizedTasks.length;
    if (remainingSlots > 0 && this.taskQueue.medium.length > 0) {
      optimizedTasks.push(...this.taskQueue.medium.splice(0, remainingSlots));
    }
    
    // Finally low priority
    const finalSlots = this.taskBatchSize - optimizedTasks.length;
    if (finalSlots > 0 && this.taskQueue.low.length > 0) {
      optimizedTasks.push(...this.taskQueue.low.splice(0, finalSlots));
    }
    
    if (optimizedTasks.length > 0) {
      this.metrics.taskQueueOptimizations++;
      this.metrics.totalOptimizations++;
      this.emit('tasksOptimized', optimizedTasks);
    }
  }
  
  /**
   * Add task to optimized queue
   */
  queueTask(task, priority = 'medium') {
    const queuedTask = {
      ...task,
      queuedAt: Date.now(),
      priority
    };
    
    this.taskQueue[priority].push(queuedTask);
    
    // Emit immediately if high priority and queue was empty
    if (priority === 'high' && this.taskQueue.high.length === 1) {
      this.optimizeTaskQueue();
    }
    
    return queuedTask;
  }
  
  /**
   * Batch messages for efficient delivery
   */
  batchMessage(agentId, message) {
    if (!this.messageBatch.has(agentId)) {
      this.messageBatch.set(agentId, []);
    }
    
    const batch = this.messageBatch.get(agentId);
    batch.push(message);
    
    // Clear existing timer
    if (this.batchTimers.has(agentId)) {
      clearTimeout(this.batchTimers.get(agentId));
    }
    
    // Send immediately if batch is full
    if (batch.length >= this.messageBatchSize) {
      this.flushMessageBatch(agentId);
    } else {
      // Set timeout for batch
      const timer = setTimeout(() => {
        this.flushMessageBatch(agentId);
      }, this.messageBatchTimeout);
      this.batchTimers.set(agentId, timer);
    }
  }
  
  /**
   * Flush message batch
   */
  flushMessageBatch(agentId) {
    const batch = this.messageBatch.get(agentId);
    if (!batch || batch.length === 0) return;
    
    this.metrics.messagesBatched += batch.length;
    this.metrics.totalOptimizations++;
    
    this.emit('messagesBatched', {
      agentId,
      messages: [...batch],
      count: batch.length
    });
    
    // Clear batch
    this.messageBatch.set(agentId, []);
    this.batchTimers.delete(agentId);
  }
  
  /**
   * Get or create pooled resource
   */
  async getPooledResource(type, factory) {
    const poolKey = `${type}`;
    
    if (!this.resourcePool.has(poolKey)) {
      this.resourcePool.set(poolKey, {
        available: [],
        inUse: new Set(),
        factory,
        created: 0
      });
      
      // Warmup pool
      await this.warmupPool(poolKey);
    }
    
    const pool = this.resourcePool.get(poolKey);
    
    // Try to get from available pool
    if (pool.available.length > 0) {
      const resource = pool.available.pop();
      resource.lastUsed = Date.now();
      pool.inUse.add(resource);
      this.metrics.resourcePoolHits++;
      this.metrics.totalOptimizations++;
      return resource.instance;
    }
    
    // Create new resource if under limit
    if (pool.created < this.poolConfig.maxPoolSize) {
      const instance = await factory();
      const resource = {
        instance,
        created: Date.now(),
        lastUsed: Date.now()
      };
      pool.inUse.add(resource);
      pool.created++;
      return instance;
    }
    
    // Wait for available resource
    return new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (pool.available.length > 0) {
          clearInterval(checkInterval);
          const resource = pool.available.pop();
          resource.lastUsed = Date.now();
          pool.inUse.add(resource);
          this.metrics.resourcePoolHits++;
          resolve(resource.instance);
        }
      }, 100);
    });
  }
  
  /**
   * Return resource to pool
   */
  releasePooledResource(type, instance) {
    const poolKey = `${type}`;
    const pool = this.resourcePool.get(poolKey);
    
    if (!pool) return;
    
    // Find resource in use
    let resourceToRelease = null;
    for (const resource of pool.inUse) {
      if (resource.instance === instance) {
        resourceToRelease = resource;
        break;
      }
    }
    
    if (resourceToRelease) {
      pool.inUse.delete(resourceToRelease);
      pool.available.push(resourceToRelease);
    }
  }
  
  /**
   * Warmup resource pool
   */
  async warmupPool(poolKey) {
    const pool = this.resourcePool.get(poolKey);
    if (!pool) return;
    
    const warmupPromises = [];
    for (let i = 0; i < this.poolConfig.warmupSize; i++) {
      warmupPromises.push(pool.factory());
    }
    
    const instances = await Promise.all(warmupPromises);
    instances.forEach(instance => {
      pool.available.push({
        instance,
        created: Date.now(),
        lastUsed: null
      });
      pool.created++;
    });
  }
  
  /**
   * Maintain resource pool health
   */
  maintainResourcePool() {
    const now = Date.now();
    
    for (const [poolKey, pool] of this.resourcePool) {
      // Remove idle resources
      pool.available = pool.available.filter(resource => {
        const idleTime = now - (resource.lastUsed || resource.created);
        if (idleTime > this.poolConfig.idleTimeout) {
          pool.created--;
          if (resource.instance.destroy) {
            resource.instance.destroy();
          }
          return false;
        }
        return true;
      });
    }
  }
  
  /**
   * Cache request results
   */
  cacheRequest(key, result, ttl = this.cacheExpiry) {
    const cacheEntry = {
      result,
      timestamp: Date.now(),
      ttl,
      hits: 0
    };
    
    this.requestCache.set(key, cacheEntry);
    return result;
  }
  
  /**
   * Get cached request
   */
  getCachedRequest(key) {
    const entry = this.requestCache.get(key);
    
    if (!entry) return null;
    
    const now = Date.now();
    if (now - entry.timestamp > entry.ttl) {
      this.requestCache.delete(key);
      return null;
    }
    
    entry.hits++;
    this.metrics.cacheHits++;
    this.metrics.totalOptimizations++;
    
    return entry.result;
  }
  
  /**
   * Generate cache key
   */
  generateCacheKey(type, params) {
    return `${type}:${JSON.stringify(params)}`;
  }
  
  /**
   * Cleanup expired cache entries
   */
  cleanupCache() {
    const now = Date.now();
    let cleaned = 0;
    
    for (const [key, entry] of this.requestCache) {
      if (now - entry.timestamp > entry.ttl) {
        this.requestCache.delete(key);
        cleaned++;
      }
    }
    
    if (cleaned > 0) {
      this.emit('cacheCleanup', { entriesCleaned: cleaned });
    }
  }
  
  /**
   * Get optimization metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      taskQueueSizes: {
        high: this.taskQueue.high.length,
        medium: this.taskQueue.medium.length,
        low: this.taskQueue.low.length
      },
      messageBatchSizes: Array.from(this.messageBatch.entries()).map(
        ([agentId, batch]) => ({ agentId, size: batch.length })
      ),
      resourcePoolStats: Array.from(this.resourcePool.entries()).map(
        ([type, pool]) => ({
          type,
          available: pool.available.length,
          inUse: pool.inUse.size,
          total: pool.created
        })
      ),
      cacheSize: this.requestCache.size
    };
  }
  
  /**
   * Reset metrics
   */
  resetMetrics() {
    this.metrics = {
      taskQueueOptimizations: 0,
      messagesBatched: 0,
      resourcePoolHits: 0,
      cacheHits: 0,
      totalOptimizations: 0
    };
  }
  
  /**
   * Configure optimization parameters
   */
  configure(config) {
    if (config.taskBatchSize) this.taskBatchSize = config.taskBatchSize;
    if (config.taskBatchDelay) this.taskBatchDelay = config.taskBatchDelay;
    if (config.messageBatchSize) this.messageBatchSize = config.messageBatchSize;
    if (config.messageBatchTimeout) this.messageBatchTimeout = config.messageBatchTimeout;
    if (config.cacheExpiry) this.cacheExpiry = config.cacheExpiry;
    if (config.poolConfig) {
      this.poolConfig = { ...this.poolConfig, ...config.poolConfig };
    }
    
    // Restart optimization loops with new config
    this.stopOptimizationLoops();
    this.startOptimizationLoops();
  }
  
  /**
   * Shutdown optimizer
   */
  shutdown() {
    this.stopOptimizationLoops();
    
    // Clear all caches and pools
    this.taskQueue = { high: [], medium: [], low: [] };
    this.messageBatch.clear();
    this.requestCache.clear();
    
    // Destroy pooled resources
    for (const pool of this.resourcePool.values()) {
      for (const resource of [...pool.available, ...pool.inUse]) {
        if (resource.instance.destroy) {
          resource.instance.destroy();
        }
      }
    }
    this.resourcePool.clear();
    
    this.emit('shutdown');
  }
}