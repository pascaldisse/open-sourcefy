/**
 * Intelligent Caching Layer
 * 
 * Provides multi-level caching with:
 * - Memory cache (LRU)
 * - Request deduplication
 * - Pattern-based caching
 * - Cache warming
 */

export class CacheLayer {
  constructor(config = {}) {
    this.config = {
      maxMemorySize: 100, // MB
      maxEntries: 10000,
      defaultTTL: 300000, // 5 minutes
      enablePatternCaching: true,
      warmupPatterns: [],
      ...config
    };
    
    // Memory cache (LRU)
    this.memoryCache = new Map();
    this.cacheOrder = [];
    this.currentMemoryUsage = 0;
    
    // Request deduplication
    this.pendingRequests = new Map();
    
    // Pattern-based caching rules
    this.cachePatterns = new Map([
      ['agent-profile', { ttl: 3600000, priority: 'high' }], // 1 hour
      ['task-result', { ttl: 600000, priority: 'medium' }],  // 10 minutes
      ['validation', { ttl: 300000, priority: 'low' }],       // 5 minutes
      ['llm-response', { ttl: 1800000, priority: 'high' }]    // 30 minutes
    ]);
    
    // Cache statistics
    this.stats = {
      hits: 0,
      misses: 0,
      evictions: 0,
      deduplicationHits: 0
    };
    
    // Start maintenance
    this.startMaintenance();
  }
  
  /**
   * Start cache maintenance
   */
  startMaintenance() {
    // Cleanup expired entries
    this.cleanupInterval = setInterval(() => {
      this.cleanupExpired();
    }, 60000); // Every minute
    
    // Memory pressure check
    this.memoryCheckInterval = setInterval(() => {
      this.checkMemoryPressure();
    }, 30000); // Every 30 seconds
  }
  
  /**
   * Get value from cache
   */
  async get(key, options = {}) {
    // Check memory cache
    const cached = this.getFromMemory(key);
    if (cached !== null) {
      this.stats.hits++;
      return cached;
    }
    
    // Check if request is already pending (deduplication)
    if (this.pendingRequests.has(key)) {
      this.stats.deduplicationHits++;
      return this.pendingRequests.get(key);
    }
    
    this.stats.misses++;
    return null;
  }
  
  /**
   * Set value in cache
   */
  async set(key, value, options = {}) {
    const pattern = this.detectPattern(key);
    const config = pattern ? this.cachePatterns.get(pattern) : {};
    
    const ttl = options.ttl || config.ttl || this.config.defaultTTL;
    const priority = options.priority || config.priority || 'medium';
    
    // Calculate size
    const size = this.calculateSize(value);
    
    // Check if we need to evict
    while (this.shouldEvict(size)) {
      this.evictLRU();
    }
    
    // Store in memory cache
    this.setInMemory(key, value, {
      ttl,
      priority,
      size,
      pattern
    });
    
    return value;
  }
  
  /**
   * Get from memory cache
   */
  getFromMemory(key) {
    const entry = this.memoryCache.get(key);
    
    if (!entry) return null;
    
    // Check expiration
    if (Date.now() > entry.expiresAt) {
      this.memoryCache.delete(key);
      this.removeFromOrder(key);
      return null;
    }
    
    // Update LRU order
    this.updateLRU(key);
    entry.hits++;
    
    return entry.value;
  }
  
  /**
   * Set in memory cache
   */
  setInMemory(key, value, options) {
    const entry = {
      value,
      size: options.size,
      priority: options.priority,
      pattern: options.pattern,
      createdAt: Date.now(),
      expiresAt: Date.now() + options.ttl,
      hits: 0
    };
    
    // Remove old entry if exists
    if (this.memoryCache.has(key)) {
      const oldEntry = this.memoryCache.get(key);
      this.currentMemoryUsage -= oldEntry.size;
      this.removeFromOrder(key);
    }
    
    this.memoryCache.set(key, entry);
    this.cacheOrder.push(key);
    this.currentMemoryUsage += entry.size;
  }
  
  /**
   * Update LRU order
   */
  updateLRU(key) {
    const index = this.cacheOrder.indexOf(key);
    if (index > -1) {
      this.cacheOrder.splice(index, 1);
      this.cacheOrder.push(key);
    }
  }
  
  /**
   * Remove from order tracking
   */
  removeFromOrder(key) {
    const index = this.cacheOrder.indexOf(key);
    if (index > -1) {
      this.cacheOrder.splice(index, 1);
    }
  }
  
  /**
   * Check if eviction is needed
   */
  shouldEvict(newSize) {
    const memoryLimitBytes = this.config.maxMemorySize * 1024 * 1024;
    return (
      this.currentMemoryUsage + newSize > memoryLimitBytes ||
      this.memoryCache.size >= this.config.maxEntries
    );
  }
  
  /**
   * Evict least recently used entry
   */
  evictLRU() {
    if (this.cacheOrder.length === 0) return;
    
    // Find entry with lowest priority to evict
    let keyToEvict = null;
    let lowestPriority = null;
    
    // Check first 10% of LRU entries
    const checkCount = Math.max(1, Math.floor(this.cacheOrder.length * 0.1));
    for (let i = 0; i < checkCount; i++) {
      const key = this.cacheOrder[i];
      const entry = this.memoryCache.get(key);
      
      if (!entry) continue;
      
      const priorityScore = this.getPriorityScore(entry.priority);
      if (lowestPriority === null || priorityScore < lowestPriority) {
        lowestPriority = priorityScore;
        keyToEvict = key;
      }
    }
    
    if (keyToEvict) {
      const entry = this.memoryCache.get(keyToEvict);
      this.memoryCache.delete(keyToEvict);
      this.removeFromOrder(keyToEvict);
      this.currentMemoryUsage -= entry.size;
      this.stats.evictions++;
    }
  }
  
  /**
   * Get priority score
   */
  getPriorityScore(priority) {
    const scores = {
      low: 1,
      medium: 2,
      high: 3
    };
    return scores[priority] || 1;
  }
  
  /**
   * Detect cache pattern
   */
  detectPattern(key) {
    for (const [pattern] of this.cachePatterns) {
      if (key.includes(pattern)) {
        return pattern;
      }
    }
    return null;
  }
  
  /**
   * Calculate size of value
   */
  calculateSize(value) {
    if (typeof value === 'string') {
      return value.length * 2; // 2 bytes per character
    }
    
    try {
      const json = JSON.stringify(value);
      return json.length * 2;
    } catch {
      return 1024; // Default 1KB
    }
  }
  
  /**
   * Register pending request
   */
  registerPending(key, promise) {
    this.pendingRequests.set(key, promise);
    
    // Remove from pending when resolved
    promise.finally(() => {
      this.pendingRequests.delete(key);
    });
    
    return promise;
  }
  
  /**
   * Warm up cache with common patterns
   */
  async warmup(patterns = []) {
    const warmupTasks = [...this.config.warmupPatterns, ...patterns];
    
    for (const task of warmupTasks) {
      try {
        if (task.generator) {
          const { key, value } = await task.generator();
          await this.set(key, value, task.options);
        }
      } catch (error) {
        console.error('Cache warmup error:', error);
      }
    }
  }
  
  /**
   * Clean up expired entries
   */
  cleanupExpired() {
    const now = Date.now();
    let cleaned = 0;
    
    for (const [key, entry] of this.memoryCache) {
      if (now > entry.expiresAt) {
        this.memoryCache.delete(key);
        this.removeFromOrder(key);
        this.currentMemoryUsage -= entry.size;
        cleaned++;
      }
    }
    
    if (cleaned > 0) {
      console.log(`Cache cleanup: removed ${cleaned} expired entries`);
    }
  }
  
  /**
   * Check memory pressure
   */
  checkMemoryPressure() {
    const memoryLimitBytes = this.config.maxMemorySize * 1024 * 1024;
    const usagePercent = (this.currentMemoryUsage / memoryLimitBytes) * 100;
    
    if (usagePercent > 90) {
      // Aggressive eviction
      const targetSize = memoryLimitBytes * 0.7; // Free up to 70%
      while (this.currentMemoryUsage > targetSize && this.cacheOrder.length > 0) {
        this.evictLRU();
      }
    }
  }
  
  /**
   * Get cache statistics
   */
  getStats() {
    const hitRate = this.stats.hits + this.stats.misses > 0
      ? (this.stats.hits / (this.stats.hits + this.stats.misses)) * 100
      : 0;
      
    return {
      ...this.stats,
      hitRate: hitRate.toFixed(2) + '%',
      entries: this.memoryCache.size,
      memoryUsageMB: (this.currentMemoryUsage / 1024 / 1024).toFixed(2),
      pendingRequests: this.pendingRequests.size
    };
  }
  
  /**
   * Clear cache
   */
  clear(pattern = null) {
    if (pattern) {
      // Clear entries matching pattern
      for (const [key, entry] of this.memoryCache) {
        if (entry.pattern === pattern || key.includes(pattern)) {
          this.memoryCache.delete(key);
          this.removeFromOrder(key);
          this.currentMemoryUsage -= entry.size;
        }
      }
    } else {
      // Clear all
      this.memoryCache.clear();
      this.cacheOrder = [];
      this.currentMemoryUsage = 0;
    }
  }
  
  /**
   * Shutdown cache
   */
  shutdown() {
    clearInterval(this.cleanupInterval);
    clearInterval(this.memoryCheckInterval);
    this.clear();
    this.pendingRequests.clear();
  }
}