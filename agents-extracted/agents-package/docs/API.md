# Multi-Agent System API Documentation

## Table of Contents

- [Core Classes](#core-classes)
  - [MultiAgentSystem](#multiagentsystem)
  - [AgentBase](#agentbase)
  - [Agent0 (Master Coordinator)](#agent0-master-coordinator)
- [Communication](#communication)
  - [MessageBus](#messagebus)
- [Quality & Monitoring](#quality--monitoring)
  - [QualityAssurance](#qualityassurance)
  - [ConditionManager](#conditionmanager)
  - [InterventionSystem](#interventionsystem)
  - [SystemMonitor](#systemmonitor)
- [Message Formats](#message-formats)
- [Event System](#event-system)

## Core Classes

### MultiAgentSystem

The main orchestrator class that manages the entire multi-agent system.

#### Constructor
```javascript
new MultiAgentSystem()
```

#### Methods

##### initialize()
Initialize the system and all components.

```javascript
await system.initialize();
```

**Returns**: `Promise<boolean>` - Success status

##### start(options)
Start the system with optional conditions.

```javascript
await system.start({
  conditions: [
    {
      id: 'condition-1',
      type: 'boolean',
      check: 'all_tests_pass',
      description: 'All tests must pass'
    }
  ]
});
```

**Parameters**:
- `options.conditions` (Array): Array of condition objects

##### stop()
Stop the system gracefully.

```javascript
await system.stop();
```

##### shutdown()
Complete system shutdown with cleanup.

```javascript
await system.shutdown();
```

##### executeTask(task)
Execute a task through the coordinator.

```javascript
const result = await system.executeTask({
  description: 'Write unit tests',
  type: 'testing',
  priority: 'high',
  context: { /* ... */ }
});
```

**Parameters**:
- `task.description` (string, required): Task description
- `task.type` (string, required): Task type
- `task.priority` (string): Priority level (low/medium/high/critical)
- `task.context` (object): Additional context
- `task.requirements` (string): Specific requirements
- `task.timeout` (number): Timeout in milliseconds
- `task.maxTurns` (number): Max Claude turns

**Returns**: 
```javascript
{
  success: boolean,
  taskId: string,
  result: string,
  duration: number,
  cost: number,
  agentId: number,
  sessionId: string
}
```

##### getSystemStatus()
Get comprehensive system status.

```javascript
const status = system.getSystemStatus();
```

**Returns**:
```javascript
{
  isRunning: boolean,
  agents: { [agentId]: AgentStatus },
  messageBus: MessageBusStats,
  coordinator: CoordinatorStatus
}
```

##### performHealthCheck()
Perform system health check.

```javascript
const health = await system.performHealthCheck();
```

**Returns**:
```javascript
{
  overall: 'healthy' | 'degraded' | 'unhealthy',
  agents: { [agentId]: HealthStatus },
  messageBus: HealthCheckResults,
  timestamp: number
}
```

### AgentBase

Base class for all agents in the system.

#### Constructor
```javascript
new AgentBase(agentId, name, role, capabilities, allowedTools)
```

**Parameters**:
- `agentId` (number): Unique agent identifier
- `name` (string): Agent name
- `role` (string): Agent role description
- `capabilities` (string[]): List of capabilities
- `allowedTools` (string[]): Allowed Claude Code SDK tools

#### Methods

##### processTask(task)
Process a task assigned by the coordinator.

```javascript
const result = await agent.processTask({
  description: 'Task description',
  type: 'task_type',
  context: {}
});
```

##### sendMessage(targetAgentId, message)
Send message to another agent.

```javascript
const messageId = await agent.sendMessage(2, {
  type: 'collaboration',
  content: 'Message content',
  requiresResponse: true
});
```

##### handleMessage(message)
Handle incoming message (usually called by MessageBus).

```javascript
const response = await agent.handleMessage({
  from: 0,
  type: 'task',
  content: taskData
});
```

##### getStatus()
Get agent status.

```javascript
const status = agent.getStatus();
```

**Returns**:
```javascript
{
  agentId: number,
  name: string,
  role: string,
  status: string,
  currentTask: object | null,
  performance: PerformanceMetrics,
  capabilities: string[],
  allowedTools: string[],
  uptime: number
}
```

### Agent0 (Master Coordinator)

Master coordinator agent that manages all other agents.

#### Additional Methods

##### registerAgent(agent)
Register an agent under coordinator management.

```javascript
coordinator.registerAgent(agent);
```

##### delegateTask(task)
Delegate task to appropriate agent.

```javascript
const result = await coordinator.delegateTask({
  description: 'Task description',
  type: 'testing',
  priority: 'high'
});
```

##### monitorProgress()
Get progress status of all managed agents.

```javascript
const progress = await coordinator.monitorProgress();
```

**Returns**: `Map<agentId, AgentStatus>`

##### handlePermissionRequest(agentId, action, context)
Handle permission request from an agent.

```javascript
const permission = await coordinator.handlePermissionRequest(
  5,
  'git_push',
  { branch: 'main' }
);
```

##### intervene(agentId, issue, correctionAction)
Intervene when agent needs correction.

```javascript
const result = await coordinator.intervene(
  3,
  'Task timeout',
  'abort'
);
```

## Communication

### MessageBus

Event-driven message bus for inter-agent communication.

#### Constructor
```javascript
new MessageBus(options)
```

**Options**:
- `messageTimeout` (number): Message delivery timeout (ms)
- `maxHistorySize` (number): Max messages to keep in history
- `retryAttempts` (number): Number of retry attempts

#### Methods

##### registerAgent(agent)
Register agent with message bus.

```javascript
messageBus.registerAgent(agent);
```

##### send(message)
Send message to specific agent.

```javascript
const result = await messageBus.send({
  from: 1,
  to: 2,
  type: 'collaboration',
  content: 'Message content'
});
```

**Returns**:
```javascript
{
  messageId: string,
  delivered: boolean,
  response: any,
  error?: string
}
```

##### broadcast(message, excludeAgent)
Broadcast message to all agents.

```javascript
const results = await messageBus.broadcast({
  from: 0,
  type: 'announcement',
  content: 'System update'
}, 0);
```

##### subscribe(agentId, messageType, handler)
Subscribe to specific message types.

```javascript
messageBus.subscribe(1, 'custom_type', async (message) => {
  // Handle message
  return { processed: true };
});
```

##### getMessageHistory(filter)
Get message history with optional filtering.

```javascript
const history = messageBus.getMessageHistory({
  agentId: 1,
  type: 'task',
  since: Date.now() - 3600000,
  limit: 100
});
```

##### getStatistics()
Get message bus statistics.

```javascript
const stats = messageBus.getStatistics();
```

**Returns**:
```javascript
{
  totalAgents: number,
  totalMessages: number,
  messagesByType: object,
  messagesByAgent: object,
  averageResponseTime: number,
  onlineAgents: number
}
```

## Quality & Monitoring

### QualityAssurance

System for validating task results and maintaining quality standards.

#### Methods

##### validateTaskResult(taskResult, validationType)
Validate task result against quality standards.

```javascript
const validation = await qa.validateTaskResult({
  taskId: 'task-123',
  agentId: 1,
  success: true,
  result: 'task output',
  duration: 1000
}, 'code');
```

**Returns**:
```javascript
{
  validationId: string,
  taskId: string,
  agentId: number,
  overallScore: number,
  qualityLevel: string,
  passed: boolean,
  criticalIssues: array,
  recommendations: array,
  timestamp: number
}
```

##### getValidationStatistics(timeframe)
Get validation statistics.

```javascript
const stats = qa.getValidationStatistics(86400000); // 24 hours
```

### ConditionManager

Manages boolean and LLM-validated conditions for system control.

#### Methods

##### addCondition(condition)
Add condition to monitor.

```javascript
const conditionId = conditionManager.addCondition({
  id: 'test-condition',
  type: 'boolean',
  check: 'all_tests_pass',
  description: 'All tests must pass'
});
```

##### evaluateAllConditions(context)
Evaluate all registered conditions.

```javascript
const evaluation = await conditionManager.evaluateAllConditions({
  systemState: { /* ... */ },
  agentStatuses: { /* ... */ }
});
```

**Returns**:
```javascript
{
  evaluationId: string,
  timestamp: number,
  results: object,
  allMet: boolean,
  context: object
}
```

### InterventionSystem

Handles error recovery and system interventions.

#### Methods

##### triggerIntervention(issue, context)
Trigger intervention for an issue.

```javascript
const intervention = await interventionSystem.triggerIntervention({
  type: 'agentFailure',
  agentId: 3,
  severity: 'high',
  description: 'Agent not responding'
}, { attempts: 3 });
```

**Returns**:
```javascript
{
  interventionId: string,
  timestamp: number,
  issue: object,
  strategy: object,
  result: object,
  success: boolean
}
```

### SystemMonitor

Real-time system monitoring and alerting.

#### Methods

##### startMonitoring()
Start system monitoring.

```javascript
systemMonitor.startMonitoring();
```

##### stopMonitoring()
Stop system monitoring.

```javascript
systemMonitor.stopMonitoring();
```

##### getCurrentSnapshot()
Get current system snapshot.

```javascript
const snapshot = systemMonitor.getCurrentSnapshot();
```

**Returns**:
```javascript
{
  cycleId: string,
  timestamp: number,
  system: SystemMetrics,
  agents: AgentMetrics,
  communication: CommunicationMetrics,
  quality: QualityMetrics
}
```

##### getActiveAlerts()
Get active system alerts.

```javascript
const alerts = systemMonitor.getActiveAlerts();
```

## Message Formats

### Agent Message
```javascript
{
  id: string,
  timestamp: number,
  from: number,
  to: number,
  type: string,
  content: any,
  requiresResponse?: boolean
}
```

### Task Message
```javascript
{
  taskId: string,
  description: string,
  type: string,
  priority: 'low' | 'medium' | 'high' | 'critical',
  context: object,
  requirements?: string,
  timeout?: number,
  maxTurns?: number
}
```

### System Alert
```javascript
{
  alertId: string,
  timestamp: number,
  type: string,
  category: string,
  severity: 'low' | 'medium' | 'high' | 'critical',
  message: string,
  value: number,
  threshold: number,
  resolved: boolean
}
```

## Event System

The system uses Node.js EventEmitter for various events:

### MessageBus Events
- `agent_registered`: Agent registered with message bus
- `agent_unregistered`: Agent unregistered
- `message_sent`: Message sent
- `message_delivered`: Message delivered successfully
- `error`: Error occurred

### System Events
- `task_started`: Task execution started
- `task_completed`: Task completed successfully
- `task_failed`: Task execution failed
- `alert_triggered`: System alert triggered
- `intervention_triggered`: Intervention triggered
- `condition_met`: Condition satisfied
- `system_shutdown`: System shutting down

### Usage Example
```javascript
system.messageBus.on('message_delivered', (message) => {
  console.log(`Message ${message.id} delivered to agent ${message.to}`);
});

system.coordinator.on('task_completed', (result) => {
  console.log(`Task ${result.taskId} completed by agent ${result.agentId}`);
});
```