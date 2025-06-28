/**
 * Real-time Monitoring Dashboard
 * 
 * Provides web-based dashboard for system monitoring
 */

import express from 'express';
import http from 'http';
import { Server } from 'socket.io';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class MonitoringDashboard {
  constructor(system, port = 3000) {
    this.system = system;
    this.port = port;
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = new Server(this.server, {
      cors: {
        origin: "*",
        methods: ["GET", "POST"]
      }
    });
    
    this.updateInterval = null;
    this.connectedClients = new Set();
    
    this.setupRoutes();
    this.setupSocketHandlers();
  }
  
  /**
   * Set up Express routes
   */
  setupRoutes() {
    // Serve static files
    this.app.use(express.static(path.join(__dirname, '../../public')));
    
    // API endpoints
    this.app.get('/api/status', (req, res) => {
      res.json(this.system.getSystemStatus());
    });
    
    this.app.get('/api/performance', (req, res) => {
      res.json(this.system.getPerformanceReport());
    });
    
    this.app.get('/api/agents', (req, res) => {
      const agents = {};
      for (const [id, agent] of this.system.agents) {
        agents[id] = agent.getStatus();
      }
      res.json(agents);
    });
    
    this.app.get('/api/metrics/optimizer', (req, res) => {
      res.json(this.system.performanceOptimizer.getMetrics());
    });
    
    this.app.get('/api/metrics/cache', (req, res) => {
      res.json(this.system.cacheLayer.getStats());
    });
    
    this.app.get('/api/metrics/pool', (req, res) => {
      res.json(this.system.claudePool.getStats());
    });
    
    this.app.get('/api/alerts', (req, res) => {
      res.json(this.system.systemMonitor.getActiveAlerts());
    });
    
    this.app.get('/api/conditions', (req, res) => {
      res.json(this.system.conditionManager.getConditionStatus());
    });
    
    this.app.get('/api/interventions', (req, res) => {
      res.json(this.system.interventionSystem.getInterventionStatistics());
    });
    
    // Serve dashboard HTML
    this.app.get('/', (req, res) => {
      res.sendFile(path.join(__dirname, '../../public/dashboard.html'));
    });
  }
  
  /**
   * Set up Socket.IO handlers
   */
  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      console.log('Dashboard client connected:', socket.id);
      this.connectedClients.add(socket.id);
      
      // Send initial data
      socket.emit('initial-data', {
        status: this.system.getSystemStatus(),
        performance: this.system.getPerformanceReport(),
        alerts: this.system.systemMonitor.getActiveAlerts()
      });
      
      // Handle disconnection
      socket.on('disconnect', () => {
        console.log('Dashboard client disconnected:', socket.id);
        this.connectedClients.delete(socket.id);
      });
      
      // Handle client requests
      socket.on('request-update', () => {
        this.sendUpdate(socket);
      });
      
      socket.on('execute-task', async (task) => {
        try {
          const result = await this.system.executeTask(task);
          socket.emit('task-result', { success: true, result });
        } catch (error) {
          socket.emit('task-result', { success: false, error: error.message });
        }
      });
      
      socket.on('get-agent-details', (agentId) => {
        const agent = this.system.agents.get(parseInt(agentId));
        if (agent) {
          socket.emit('agent-details', {
            agentId,
            status: agent.getStatus(),
            history: agent.taskHistory || []
          });
        }
      });
    });
  }
  
  /**
   * Send update to specific client
   */
  sendUpdate(socket) {
    const update = this.collectMetrics();
    socket.emit('metrics-update', update);
  }
  
  /**
   * Collect current metrics
   */
  collectMetrics() {
    const systemStatus = this.system.getSystemStatus();
    const performanceReport = this.system.getPerformanceReport();
    const systemMonitor = this.system.systemMonitor.getCurrentSnapshot();
    
    return {
      timestamp: Date.now(),
      system: {
        isRunning: systemStatus.isRunning,
        uptime: performanceReport.uptime,
        totalAgents: Object.keys(systemStatus.agents).length,
        activeAgents: Object.values(systemStatus.agents).filter(a => a.status === 'working').length
      },
      performance: {
        taskOptimizations: performanceReport.optimization.taskOptimizations,
        messagesBatched: performanceReport.optimization.messagesBatched,
        cacheHitRate: performanceReport.optimization.cacheHitRate,
        averageResponseTime: this.calculateAverageResponseTime(performanceReport.agents)
      },
      optimization: systemStatus.optimization,
      agents: systemStatus.agents,
      alerts: this.system.systemMonitor.getActiveAlerts(),
      conditions: this.system.conditionManager.getConditionStatus(),
      resources: {
        cpu: systemMonitor?.system?.cpu?.usage || 0,
        memory: systemMonitor?.system?.memory?.usage || 0,
        cacheMemory: parseFloat(systemStatus.optimization?.cache?.memoryUsageMB || 0),
        poolConnections: systemStatus.optimization?.pool?.totalConnections || 0
      }
    };
  }
  
  /**
   * Calculate average response time across all agents
   */
  calculateAverageResponseTime(agents) {
    let totalTime = 0;
    let totalTasks = 0;
    
    for (const metrics of Object.values(agents)) {
      if (metrics.tasksCompleted > 0) {
        totalTime += metrics.averageResponseTime * metrics.tasksCompleted;
        totalTasks += metrics.tasksCompleted;
      }
    }
    
    return totalTasks > 0 ? totalTime / totalTasks : 0;
  }
  
  /**
   * Start the dashboard server
   */
  async start() {
    return new Promise((resolve) => {
      this.server.listen(this.port, () => {
        console.log(`Monitoring dashboard running at http://localhost:${this.port}`);
        
        // Start periodic updates
        this.updateInterval = setInterval(() => {
          if (this.connectedClients.size > 0) {
            const update = this.collectMetrics();
            this.io.emit('metrics-update', update);
          }
        }, 2000); // Update every 2 seconds
        
        resolve();
      });
    });
  }
  
  /**
   * Stop the dashboard server
   */
  async stop() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    
    this.io.close();
    
    return new Promise((resolve) => {
      this.server.close(() => {
        console.log('Monitoring dashboard stopped');
        resolve();
      });
    });
  }
}