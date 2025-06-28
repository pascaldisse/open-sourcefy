import winston from 'winston';
import path from 'path';

// Define log format
const logFormat = winston.format.combine(
  winston.format.timestamp(),
  winston.format.errors({ stack: true }),
  winston.format.json(),
  winston.format.printf(({ timestamp, level, message, agentId, taskId, ...meta }) => {
    const logObject = {
      timestamp,
      level,
      message,
      ...(agentId && { agentId }),
      ...(taskId && { taskId }),
      ...meta
    };
    return JSON.stringify(logObject);
  })
);

// Create logger instance
export const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: logFormat,
  transports: [
    // Console logging
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple(),
        winston.format.printf(({ timestamp, level, message, agentId, taskId }) => {
          const prefix = agentId ? `[Agent ${agentId}]` : '';
          const taskPrefix = taskId ? `[Task ${taskId}]` : '';
          return `${timestamp} ${level}: ${prefix}${taskPrefix} ${message}`;
        })
      )
    }),
    
    // File logging for all logs
    new winston.transports.File({
      filename: path.join(process.cwd(), 'logs', 'combined.log'),
      maxsize: 5242880, // 5MB
      maxFiles: 10
    }),
    
    // Error logs
    new winston.transports.File({
      filename: path.join(process.cwd(), 'logs', 'error.log'),
      level: 'error',
      maxsize: 5242880,
      maxFiles: 5
    }),
    
    // Agent-specific logs
    new winston.transports.File({
      filename: path.join(process.cwd(), 'logs', 'agents.log'),
      format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
      ),
      maxsize: 5242880,
      maxFiles: 10
    })
  ]
});

// Agent-specific logger
export const createAgentLogger = (agentId) => {
  return logger.child({ agentId });
};

// Task-specific logger
export const createTaskLogger = (agentId, taskId) => {
  return logger.child({ agentId, taskId });
};

// Performance logger
export const performanceLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({
      filename: path.join(process.cwd(), 'logs', 'performance.log'),
      maxsize: 5242880,
      maxFiles: 5
    })
  ]
});

// Audit logger for compliance and security
export const auditLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({
      filename: path.join(process.cwd(), 'logs', 'audit.log'),
      maxsize: 10485760, // 10MB
      maxFiles: 20
    })
  ]
});

// System health logger
export const healthLogger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({
      filename: path.join(process.cwd(), 'logs', 'health.log'),
      maxsize: 5242880,
      maxFiles: 10
    })
  ]
});

// Create logs directory if it doesn't exist
import fs from 'fs';
const logsDir = path.join(process.cwd(), 'logs');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

export default logger;