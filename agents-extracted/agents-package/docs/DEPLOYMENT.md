# Deployment and Configuration Guide

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment Options](#deployment-options)
- [Production Setup](#production-setup)
- [Monitoring Setup](#monitoring-setup)
- [Scaling Considerations](#scaling-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Node.js**: v18.0.0 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for production)
- **CPU**: 2+ cores recommended
- **Storage**: 10GB minimum for logs and data
- **OS**: Linux, macOS, or Windows with WSL2

### Required Services

- **Anthropic API Access**: Valid API key with sufficient quota
- **Optional**: 
  - Redis for distributed caching
  - PostgreSQL for persistent storage
  - Elasticsearch for log aggregation

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/multi-agent-system.git
cd multi-agent-system
```

### 2. Install Dependencies

```bash
# Using npm
npm install

# Using yarn
yarn install

# For production (skip dev dependencies)
npm install --production
```

### 3. Environment Setup

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx

# Environment
NODE_ENV=production
LOG_LEVEL=info

# System Configuration
PORT=3000
MAX_AGENTS=17
AGENT_TIMEOUT=120000
COMMUNICATION_PROTOCOL=websocket

# Monitoring
ENABLE_METRICS=true
LOG_RETENTION_DAYS=30
HEALTH_CHECK_INTERVAL=60000

# Security
SECRET_KEY=generate-a-secure-random-key
ENABLE_AUDIT_LOGS=true

# Optional External Services
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost:5432/multiagent
ELASTICSEARCH_URL=http://localhost:9200
```

### 4. Verify Installation

```bash
# Run tests
npm test

# Check system
npm run system:health
```

## Configuration

### Agent Configuration

Each agent can be configured through environment variables or configuration files.

#### Environment Variables per Agent

```env
# Agent 1 - Test Engineer
AGENT_1_ENABLED=true
AGENT_1_MAX_CONCURRENT_TASKS=3
AGENT_1_TOOLS=Read,Write,Edit,Bash,Glob,Grep,LS

# Agent 7 - Code Reviewer
AGENT_7_ENABLED=true
AGENT_7_REVIEW_DEPTH=comprehensive
AGENT_7_SECURITY_CHECK=true
```

#### Configuration File (`config/agents.json`)

```json
{
  "agents": {
    "1": {
      "enabled": true,
      "maxConcurrentTasks": 3,
      "tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep", "LS"],
      "specialConfig": {
        "testFrameworks": ["jest", "mocha", "cypress"],
        "coverageThreshold": 80
      }
    },
    "7": {
      "enabled": true,
      "reviewDepth": "comprehensive",
      "securityCheck": true,
      "codeStandards": ["eslint", "prettier"]
    }
  }
}
```

### Quality Assurance Configuration

Configure quality standards in `config/quality.json`:

```json
{
  "standards": {
    "code": {
      "complexity": "low",
      "coverage": 80,
      "duplication": 5
    },
    "documentation": {
      "completeness": 90,
      "readability": "high"
    },
    "security": {
      "vulnerabilities": 0,
      "compliance": 100
    }
  },
  "thresholds": {
    "overall": {
      "minimum": 70,
      "good": 80,
      "excellent": 90
    }
  }
}
```

### Monitoring Configuration

Configure monitoring in `config/monitoring.json`:

```json
{
  "alerts": {
    "system": {
      "cpuUsage": 80,
      "memoryUsage": 85,
      "diskUsage": 90
    },
    "agents": {
      "responseTime": 5000,
      "errorRate": 10,
      "taskFailureRate": 20
    },
    "quality": {
      "averageScore": 70,
      "criticalIssueRate": 1
    }
  },
  "metrics": {
    "interval": 10000,
    "retention": 604800000
  }
}
```

## Deployment Options

### 1. Local Development

```bash
# Start in development mode
npm run dev

# With debugging
NODE_ENV=development DEBUG=* npm start
```

### 2. Docker Deployment

#### Dockerfile

```dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --production

# Copy application
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nodejs -u 1001
USER nodejs

EXPOSE 3000

CMD ["node", "src/index.js"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  multi-agent-system:
    build: .
    environment:
      - NODE_ENV=production
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    ports:
      - "3000:3000"
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### 3. Kubernetes Deployment

#### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: multi-agent-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: multi-agent-system
  template:
    metadata:
      labels:
        app: multi-agent-system
    spec:
      containers:
      - name: multi-agent-system
        image: your-registry/multi-agent-system:latest
        ports:
        - containerPort: 3000
        env:
        - name: NODE_ENV
          value: "production"
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: multi-agent-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 4. Cloud Deployment

#### AWS ECS Task Definition

```json
{
  "family": "multi-agent-system",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsExecutionRole",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "multi-agent-system",
      "image": "your-ecr-repo/multi-agent-system:latest",
      "cpu": 2048,
      "memory": 4096,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NODE_ENV",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:123456789012:secret:anthropic-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/multi-agent-system",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## Production Setup

### 1. Process Management with PM2

```javascript
// ecosystem.config.js
module.exports = {
  apps: [{
    name: 'multi-agent-system',
    script: './src/index.js',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production'
    },
    error_file: './logs/pm2-error.log',
    out_file: './logs/pm2-out.log',
    log_file: './logs/pm2-combined.log',
    time: true,
    max_memory_restart: '2G',
    restart_delay: 4000,
    autorestart: true,
    watch: false
  }]
};
```

Start with PM2:

```bash
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### 2. Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /ws {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. SSL/TLS Setup

Use Let's Encrypt for free SSL certificates:

```bash
sudo certbot --nginx -d your-domain.com
```

### 4. Security Hardening

```bash
# Set up firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Create dedicated user
sudo useradd -m -s /bin/bash multiagent
sudo chown -R multiagent:multiagent /opt/multi-agent-system

# Set appropriate permissions
chmod 750 /opt/multi-agent-system
chmod 640 /opt/multi-agent-system/.env
```

## Monitoring Setup

### 1. Health Check Endpoint

The system provides health check endpoints:

```bash
# Basic health check
curl http://localhost:3000/health

# Detailed health check
curl http://localhost:3000/health/detailed
```

### 2. Prometheus Metrics

Configure Prometheus to scrape metrics:

```yaml
scrape_configs:
  - job_name: 'multi-agent-system'
    static_configs:
      - targets: ['localhost:3000']
    metrics_path: '/metrics'
```

### 3. Grafana Dashboard

Import the provided dashboard from `monitoring/grafana-dashboard.json` for:

- System resource usage
- Agent performance metrics
- Task completion rates
- Error rates and alerts
- Quality assurance trends

### 4. Log Aggregation

Configure log shipping to Elasticsearch:

```javascript
// config/logging.js
module.exports = {
  elasticsearch: {
    node: process.env.ELASTICSEARCH_URL,
    index: 'multi-agent-system',
    type: 'logs'
  },
  fields: {
    application: 'multi-agent-system',
    environment: process.env.NODE_ENV
  }
};
```

## Scaling Considerations

### Horizontal Scaling

The system is designed to run as a single instance due to the coordinator pattern. For scaling:

1. **Task Queue**: Implement external task queue (Redis/RabbitMQ)
2. **Distributed Agents**: Run specialized agents on separate instances
3. **Load Balancing**: Use a load balancer for API endpoints

### Performance Optimization

```javascript
// config/performance.js
module.exports = {
  // Limit concurrent tasks
  maxConcurrentTasks: 10,
  
  // Agent task queues
  taskQueueSize: 100,
  
  // Message bus optimization
  messageBatchSize: 50,
  messageCompressionThreshold: 1024,
  
  // Cache configuration
  cache: {
    enabled: true,
    ttl: 3600,
    maxSize: 1000
  }
};
```

### Resource Limits

Set appropriate resource limits:

```bash
# System limits
ulimit -n 65536  # File descriptors
ulimit -u 32768  # Processes

# Node.js memory
NODE_OPTIONS="--max-old-space-size=4096"
```

## Troubleshooting

### Common Issues

#### 1. Agent Communication Failures

```bash
# Check message bus status
curl http://localhost:3000/api/message-bus/status

# View message history
curl http://localhost:3000/api/message-bus/history?limit=100
```

#### 2. High Memory Usage

```bash
# Generate heap snapshot
kill -USR2 $(pgrep node)

# Analyze with Chrome DevTools
```

#### 3. Task Timeouts

Check agent logs:

```bash
tail -f logs/agents.log | grep "timeout"
```

Adjust timeout settings:

```env
AGENT_TIMEOUT=300000  # 5 minutes
TASK_TIMEOUT=600000   # 10 minutes
```

#### 4. API Rate Limiting

Monitor API usage:

```bash
curl http://localhost:3000/api/metrics/anthropic
```

Implement rate limiting:

```javascript
// config/rateLimits.js
module.exports = {
  anthropic: {
    requestsPerMinute: 50,
    tokensPerMinute: 100000
  }
};
```

### Debug Mode

Enable debug logging:

```bash
DEBUG=* NODE_ENV=development npm start
```

### Health Checks

```bash
# System health
curl http://localhost:3000/health

# Agent health
curl http://localhost:3000/api/agents/health

# Individual agent
curl http://localhost:3000/api/agents/1/health
```

### Recovery Procedures

#### 1. System Recovery

```bash
# Stop system
pm2 stop multi-agent-system

# Clear corrupted state
rm -rf data/state/*

# Restart
pm2 start multi-agent-system
```

#### 2. Agent Recovery

```javascript
// Manual agent restart
const agent = system.agents.get(3);
await agent.shutdown();
await agent.initialize();
```

#### 3. Message Bus Recovery

```javascript
// Clear message queue
system.messageBus.clearPendingMessages();

// Reconnect all agents
for (const [id, agent] of system.agents) {
  system.messageBus.registerAgent(agent);
}
```