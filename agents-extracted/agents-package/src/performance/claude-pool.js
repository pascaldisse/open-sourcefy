/**
 * Claude Code SDK Connection Pool
 * 
 * Manages pooled connections to Claude Code SDK
 * for improved performance and resource utilization
 */

export class ClaudePool {
  constructor(config = {}) {
    this.config = {
      minSize: 2,
      maxSize: 10,
      acquireTimeout: 30000,
      idleTimeout: 300000,
      ...config
    };
    
    // Pool state
    this.pool = [];
    this.activeConnections = new Set();
    this.waitingQueue = [];
    this.isShuttingDown = false;
    
    // Metrics
    this.metrics = {
      created: 0,
      destroyed: 0,
      acquired: 0,
      released: 0,
      timeouts: 0,
      errors: 0
    };
    
    // Initialize pool
    this.initialize();
  }
  
  /**
   * Initialize connection pool
   */
  async initialize() {
    const promises = [];
    for (let i = 0; i < this.config.minSize; i++) {
      promises.push(this.createConnection());
    }
    
    const connections = await Promise.all(promises);
    connections.forEach(conn => {
      if (conn) this.pool.push(conn);
    });
  }
  
  /**
   * Create a new connection
   */
  async createConnection() {
    try {
      // Create a connection wrapper
      const connection = {
        id: `claude-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        created: Date.now(),
        lastUsed: null,
        useCount: 0,
        abortController: new AbortController()
      };
      
      this.metrics.created++;
      return connection;
    } catch (error) {
      this.metrics.errors++;
      console.error('Failed to create Claude connection:', error);
      return null;
    }
  }
  
  /**
   * Acquire a connection from the pool
   */
  async acquire() {
    if (this.isShuttingDown) {
      throw new Error('Pool is shutting down');
    }
    
    // Try to get from pool
    let connection = this.pool.shift();
    
    if (!connection && this.activeConnections.size < this.config.maxSize) {
      // Create new connection if under limit
      connection = await this.createConnection();
    }
    
    if (connection) {
      connection.lastUsed = Date.now();
      connection.useCount++;
      this.activeConnections.add(connection);
      this.metrics.acquired++;
      return connection;
    }
    
    // Wait for available connection
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        const index = this.waitingQueue.indexOf(entry);
        if (index > -1) {
          this.waitingQueue.splice(index, 1);
          this.metrics.timeouts++;
          reject(new Error('Acquire timeout'));
        }
      }, this.config.acquireTimeout);
      
      const entry = { resolve, reject, timeout };
      this.waitingQueue.push(entry);
    });
  }
  
  /**
   * Release a connection back to the pool
   */
  release(connection) {
    if (!connection || !this.activeConnections.has(connection)) {
      return;
    }
    
    this.activeConnections.delete(connection);
    this.metrics.released++;
    
    // Check if connection should be destroyed
    const idleTime = Date.now() - connection.lastUsed;
    if (idleTime > this.config.idleTimeout || this.isShuttingDown) {
      this.destroyConnection(connection);
      return;
    }
    
    // Return to pool or give to waiting request
    if (this.waitingQueue.length > 0) {
      const entry = this.waitingQueue.shift();
      clearTimeout(entry.timeout);
      connection.lastUsed = Date.now();
      connection.useCount++;
      this.activeConnections.add(connection);
      entry.resolve(connection);
    } else {
      this.pool.push(connection);
    }
    
    // Maintain minimum pool size
    this.maintainPoolSize();
  }
  
  /**
   * Destroy a connection
   */
  destroyConnection(connection) {
    if (connection.abortController) {
      connection.abortController.abort();
    }
    this.metrics.destroyed++;
  }
  
  /**
   * Maintain minimum pool size
   */
  async maintainPoolSize() {
    const totalSize = this.pool.length + this.activeConnections.size;
    
    if (totalSize < this.config.minSize && !this.isShuttingDown) {
      const needed = this.config.minSize - totalSize;
      const promises = [];
      
      for (let i = 0; i < needed; i++) {
        promises.push(this.createConnection());
      }
      
      const connections = await Promise.all(promises);
      connections.forEach(conn => {
        if (conn) this.pool.push(conn);
      });
    }
  }
  
  /**
   * Get pool statistics
   */
  getStats() {
    return {
      poolSize: this.pool.length,
      activeConnections: this.activeConnections.size,
      waitingRequests: this.waitingQueue.length,
      totalConnections: this.pool.length + this.activeConnections.size,
      metrics: { ...this.metrics }
    };
  }
  
  /**
   * Shutdown the pool
   */
  async shutdown() {
    this.isShuttingDown = true;
    
    // Reject all waiting requests
    while (this.waitingQueue.length > 0) {
      const entry = this.waitingQueue.shift();
      clearTimeout(entry.timeout);
      entry.reject(new Error('Pool shutting down'));
    }
    
    // Wait for active connections to be released
    const timeout = Date.now() + 30000; // 30 second timeout
    while (this.activeConnections.size > 0 && Date.now() < timeout) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Destroy all connections
    [...this.pool, ...this.activeConnections].forEach(conn => {
      this.destroyConnection(conn);
    });
    
    this.pool = [];
    this.activeConnections.clear();
  }
}