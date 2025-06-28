#!/usr/bin/env node

/**
 * Multi-Agent System CLI
 * 
 * Command-line interface for managing the multi-agent system
 */

import { Command } from 'commander';
import { MultiAgentSystem } from '../index-optimized.js';
import { MonitoringDashboard } from '../monitoring/dashboard.js';
import { AgentLearning } from '../learning/agent-learning.js';
import { agentConfig } from '../config/agent-config.js';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';
import Table from 'cli-table3';
import fs from 'fs/promises';
import path from 'path';

const program = new Command();
let system = null;
let dashboard = null;
let learningSystem = null;

// CLI configuration
program
  .name('gaia-mas')
  .description('CLI for Gaia Multi-Agent System')
  .version('1.0.0');

// Start command
program
  .command('start')
  .description('Start the multi-agent system')
  .option('-d, --dashboard', 'Start with monitoring dashboard')
  .option('-p, --port <port>', 'Dashboard port', '3001')
  .option('-l, --learning', 'Enable learning system')
  .option('-c, --config <path>', 'Path to configuration file')
  .option('--project <path>', 'Project path for context', process.cwd())
  .option('--prompt <prompt>', 'Custom system prompt for Agent 0')
  .action(async (options) => {
    const spinner = ora('Starting Multi-Agent System...').start();
    
    try {
      // Load configuration if provided
      let config = null;
      if (options.config) {
        config = await agentConfig.loadConfigFile(options.config);
      } else {
        // Create project-specific config
        const projectConfig = await agentConfig.createProjectConfig(options.project, {
          systemPrompt: options.prompt
        });
        config = projectConfig.config;
      }
      
      system = new MultiAgentSystem();
      await system.initialize();
      
      // Apply configuration
      if (config) {
        agentConfig.applyToCoordinator(system.coordinator, config);
        agentConfig.applyAgentOverrides(system.agents, config);
      }
      
      await system.start({ conditions: config?.conditions || [] });
      
      if (options.learning) {
        learningSystem = new AgentLearning();
        await learningSystem.initialize();
        connectLearningSystem();
      }
      
      spinner.succeed('Multi-Agent System started successfully');
      
      if (options.dashboard) {
        dashboard = new MonitoringDashboard(system, parseInt(options.port));
        await dashboard.start();
        console.log(chalk.blue(`\n📊 Dashboard running at http://localhost:${options.port}`));
      }
      
      console.log(chalk.green('\n✓ System is running. Use "gaia-mas status" to check status.'));
      
      // Keep process alive
      process.stdin.resume();
      
    } catch (error) {
      spinner.fail('Failed to start system');
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Stop command
program
  .command('stop')
  .description('Stop the multi-agent system')
  .action(async () => {
    const spinner = ora('Stopping Multi-Agent System...').start();
    
    try {
      if (dashboard) {
        await dashboard.stop();
      }
      
      if (learningSystem) {
        await learningSystem.shutdown();
      }
      
      if (system) {
        await system.shutdown();
      }
      
      spinner.succeed('Multi-Agent System stopped');
      process.exit(0);
      
    } catch (error) {
      spinner.fail('Error stopping system');
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Status command
program
  .command('status')
  .description('Check system status')
  .option('-a, --agents', 'Show detailed agent status')
  .option('-p, --performance', 'Show performance metrics')
  .action(async (options) => {
    try {
      if (!system || !system.isRunning) {
        console.log(chalk.yellow('System is not running'));
        return;
      }
      
      const status = system.getSystemStatus();
      
      // System overview
      console.log(chalk.bold('\n🌐 System Status\n'));
      console.log(`Status: ${status.isRunning ? chalk.green('Running') : chalk.red('Stopped')}`);
      console.log(`Total Agents: ${Object.keys(status.agents).length}`);
      console.log(`Active Agents: ${Object.values(status.agents).filter(a => a.status === 'working').length}`);
      
      if (options.agents) {
        // Agent details table
        const table = new Table({
          head: ['ID', 'Name', 'Status', 'Tasks', 'Success Rate', 'Avg Response'],
          style: { head: ['cyan'] }
        });
        
        for (const [id, agent] of Object.entries(status.agents)) {
          table.push([
            id,
            agent.name,
            agent.status === 'idle' ? chalk.green(agent.status) : 
            agent.status === 'working' ? chalk.yellow(agent.status) : chalk.red(agent.status),
            agent.performance.tasksCompleted,
            (agent.performance.successRate * 100).toFixed(1) + '%',
            agent.performance.averageResponseTime.toFixed(0) + 'ms'
          ]);
        }
        
        console.log('\n' + table.toString());
      }
      
      if (options.performance) {
        const perf = system.getPerformanceReport();
        
        console.log(chalk.bold('\n📈 Performance Metrics\n'));
        console.log(`Cache Hit Rate: ${perf.optimization.cacheHitRate}`);
        console.log(`Task Optimizations: ${perf.optimization.taskOptimizations}`);
        console.log(`Messages Batched: ${perf.optimization.messagesBatched}`);
        console.log(`Pool Utilization: ${perf.optimization.poolUtilization.activeConnections}/${perf.optimization.poolUtilization.totalConnections}`);
      }
      
    } catch (error) {
      console.error(chalk.red('Error getting status:', error.message));
    }
  });

// Execute command
program
  .command('execute <task>')
  .description('Execute a task')
  .option('-t, --type <type>', 'Task type', 'testing')
  .option('-p, --priority <priority>', 'Task priority', 'medium')
  .option('-c, --context <context>', 'Task context (JSON)')
  .action(async (taskDescription, options) => {
    try {
      if (!system || !system.isRunning) {
        console.log(chalk.yellow('System is not running'));
        return;
      }
      
      const spinner = ora('Executing task...').start();
      
      const task = {
        description: taskDescription,
        type: options.type,
        priority: options.priority,
        context: options.context ? JSON.parse(options.context) : {}
      };
      
      const result = await system.executeTask(task);
      
      if (result.success) {
        spinner.succeed('Task completed successfully');
        console.log(chalk.green(`\n✓ Task ID: ${result.taskId}`));
        console.log(`Agent: ${result.agentId}`);
        console.log(`Duration: ${result.duration}ms`);
        console.log(`Cost: $${result.cost || 0}`);
        
        if (result.fromCache) {
          console.log(chalk.blue('(Result from cache)'));
        }
      } else {
        spinner.fail('Task failed');
        console.log(chalk.red(`\nError: ${result.error}`));
      }
      
    } catch (error) {
      console.error(chalk.red('Error executing task:', error.message));
    }
  });

// Agent command group
const agentCmd = program
  .command('agent')
  .description('Agent management commands');

agentCmd
  .command('list')
  .description('List all agents')
  .action(async () => {
    try {
      if (!system) {
        console.log(chalk.yellow('System is not running'));
        return;
      }
      
      const agents = system.agents;
      
      console.log(chalk.bold('\n👥 Agents\n'));
      
      for (const [id, agent] of agents) {
        const status = agent.getStatus();
        console.log(`${chalk.cyan(`Agent ${id}`)}: ${status.name}`);
        console.log(`  Role: ${status.role}`);
        console.log(`  Status: ${status.status}`);
        console.log(`  Capabilities: ${status.capabilities.join(', ')}`);
        console.log();
      }
      
    } catch (error) {
      console.error(chalk.red('Error listing agents:', error.message));
    }
  });

agentCmd
  .command('details <id>')
  .description('Get detailed agent information')
  .action(async (agentId) => {
    try {
      if (!system) {
        console.log(chalk.yellow('System is not running'));
        return;
      }
      
      const agent = system.agents.get(parseInt(agentId));
      if (!agent) {
        console.log(chalk.red(`Agent ${agentId} not found`));
        return;
      }
      
      const status = agent.getStatus();
      
      console.log(chalk.bold(`\n🤖 Agent ${agentId}: ${status.name}\n`));
      console.log(`Role: ${status.role}`);
      console.log(`Status: ${status.status}`);
      console.log(`Uptime: ${Math.floor(status.uptime / 1000)}s`);
      console.log(`\nCapabilities:`);
      status.capabilities.forEach(cap => console.log(`  - ${cap}`));
      console.log(`\nAllowed Tools:`);
      status.allowedTools.forEach(tool => console.log(`  - ${tool}`));
      console.log(`\nPerformance:`);
      console.log(`  Tasks Completed: ${status.performance.tasksCompleted}`);
      console.log(`  Success Rate: ${(status.performance.successRate * 100).toFixed(1)}%`);
      console.log(`  Average Response: ${status.performance.averageResponseTime.toFixed(0)}ms`);
      console.log(`  Total Cost: $${status.performance.totalCostUsd.toFixed(4)}`);
      
      if (status.optimization) {
        console.log(`\nOptimization:`);
        console.log(`  Cache: ${status.optimization.hasCache ? '✓' : '✗'}`);
        console.log(`  Pool: ${status.optimization.hasPool ? '✓' : '✗'}`);
        console.log(`  Optimizer: ${status.optimization.hasOptimizer ? '✓' : '✗'}`);
      }
      
    } catch (error) {
      console.error(chalk.red('Error getting agent details:', error.message));
    }
  });

// Learning command group
const learningCmd = program
  .command('learning')
  .description('Learning system commands');

learningCmd
  .command('insights <agentId>')
  .description('Get learning insights for an agent')
  .action(async (agentId) => {
    try {
      if (!learningSystem) {
        console.log(chalk.yellow('Learning system is not enabled'));
        return;
      }
      
      const insights = learningSystem.getLearningInsights(parseInt(agentId));
      
      if (!insights) {
        console.log(chalk.yellow(`No learning data for Agent ${agentId}`));
        return;
      }
      
      console.log(chalk.bold(`\n📊 Learning Insights for Agent ${agentId}\n`));
      
      console.log('Performance:');
      console.log(`  Success Rate: ${(insights.performance.successRate * 100).toFixed(1)}%`);
      console.log(`  Avg Duration: ${Math.round(insights.performance.avgDuration)}ms`);
      console.log(`  Total Tasks: ${insights.performance.totalTasks}`);
      
      if (insights.improvement) {
        console.log('\nImprovement:');
        console.log(`  Success Rate Change: ${(insights.improvement.successRateChange * 100).toFixed(1)}%`);
        console.log(`  Duration Change: ${insights.improvement.durationChange.toFixed(0)}ms`);
        console.log(`  Improvements: ${insights.improvement.improvementCount}`);
      }
      
      console.log('\nPatterns:');
      console.log(`  Total: ${insights.patterns.total}`);
      console.log(`  Successful: ${insights.patterns.successful}`);
      console.log(`  Problematic: ${insights.patterns.problematic}`);
      
      if (insights.recommendations.length > 0) {
        console.log('\nRecommendations:');
        insights.recommendations.forEach(rec => {
          console.log(`  - [${rec.type}] ${rec.suggestion}`);
        });
      }
      
    } catch (error) {
      console.error(chalk.red('Error getting insights:', error.message));
    }
  });

learningCmd
  .command('share <fromAgent> <toAgent>')
  .description('Share knowledge between agents')
  .action(async (fromAgent, toAgent) => {
    try {
      if (!learningSystem) {
        console.log(chalk.yellow('Learning system is not enabled'));
        return;
      }
      
      const spinner = ora('Sharing knowledge...').start();
      
      const success = await learningSystem.shareKnowledge(
        parseInt(fromAgent),
        parseInt(toAgent),
        'patterns'
      );
      
      if (success) {
        spinner.succeed(`Knowledge shared from Agent ${fromAgent} to Agent ${toAgent}`);
      } else {
        spinner.fail('Failed to share knowledge');
      }
      
    } catch (error) {
      console.error(chalk.red('Error sharing knowledge:', error.message));
    }
  });

// Cache command group
const cacheCmd = program
  .command('cache')
  .description('Cache management commands');

cacheCmd
  .command('stats')
  .description('Show cache statistics')
  .action(async () => {
    try {
      if (!system) {
        console.log(chalk.yellow('System is not running'));
        return;
      }
      
      const stats = system.cacheLayer.getStats();
      
      console.log(chalk.bold('\n💾 Cache Statistics\n'));
      console.log(`Hit Rate: ${stats.hitRate}`);
      console.log(`Hits: ${stats.hits}`);
      console.log(`Misses: ${stats.misses}`);
      console.log(`Evictions: ${stats.evictions}`);
      console.log(`Entries: ${stats.entries}`);
      console.log(`Memory Usage: ${stats.memoryUsageMB}MB`);
      console.log(`Pending Requests: ${stats.pendingRequests}`);
      
    } catch (error) {
      console.error(chalk.red('Error getting cache stats:', error.message));
    }
  });

cacheCmd
  .command('clear [pattern]')
  .description('Clear cache (optionally by pattern)')
  .action(async (pattern) => {
    try {
      if (!system) {
        console.log(chalk.yellow('System is not running'));
        return;
      }
      
      const { confirm } = await inquirer.prompt([{
        type: 'confirm',
        name: 'confirm',
        message: pattern ? 
          `Clear cache entries matching "${pattern}"?` : 
          'Clear entire cache?',
        default: false
      }]);
      
      if (confirm) {
        system.cacheLayer.clear(pattern);
        console.log(chalk.green('✓ Cache cleared'));
      }
      
    } catch (error) {
      console.error(chalk.red('Error clearing cache:', error.message));
    }
  });

// Config command group
const configCmd = program
  .command('config')
  .description('Configuration management commands');

configCmd
  .command('generate <output>')
  .description('Generate example configuration file')
  .action(async (outputPath) => {
    try {
      const spinner = ora('Generating configuration...').start();
      await agentConfig.generateExampleConfig(outputPath);
      spinner.succeed(`Configuration file created at ${outputPath}`);
      console.log(chalk.green('\nEdit this file to customize agent behavior and task mappings.'));
    } catch (error) {
      console.error(chalk.red('Error generating config:', error.message));
    }
  });

configCmd
  .command('validate <path>')
  .description('Validate a configuration file')
  .action(async (configPath) => {
    try {
      const spinner = ora('Validating configuration...').start();
      const config = await agentConfig.loadConfigFile(configPath);
      spinner.succeed('Configuration is valid');
      
      console.log(chalk.bold('\n📋 Configuration Summary\n'));
      console.log(`Project: ${config.projectName || 'N/A'}`);
      console.log(`Task Types: ${Object.keys(config.taskMapping || {}).length}`);
      console.log(`Agent Overrides: ${Object.keys(config.agentOverrides || {}).length}`);
      console.log(`Conditions: ${(config.conditions || []).length}`);
      
    } catch (error) {
      console.error(chalk.red('Configuration validation failed:', error.message));
    }
  });

configCmd
  .command('detect [path]')
  .description('Detect project type and suggest configuration')
  .action(async (projectPath = process.cwd()) => {
    try {
      const spinner = ora('Analyzing project...').start();
      const suggestions = await agentConfig.detectProjectType(projectPath);
      spinner.stop();
      
      if (suggestions) {
        console.log(chalk.bold(`\n🔍 Detected Project Type: ${suggestions.projectType}\n`));
        console.log('Suggested task mappings:');
        
        const table = new Table({
          head: ['Task Type', 'Assigned Agents'],
          style: { head: ['cyan'] }
        });
        
        for (const [task, agents] of Object.entries(suggestions.taskMapping)) {
          table.push([task, agents.join(', ')]);
        }
        
        console.log(table.toString());
        
        const { save } = await inquirer.prompt([{
          type: 'confirm',
          name: 'save',
          message: 'Save this configuration?',
          default: true
        }]);
        
        if (save) {
          const { filename } = await inquirer.prompt([{
            type: 'input',
            name: 'filename',
            message: 'Configuration filename:',
            default: 'agent-config.json'
          }]);
          
          const config = {
            projectType: suggestions.projectType,
            projectPath,
            ...suggestions
          };
          
          await agentConfig.saveConfig(config, filename);
          console.log(chalk.green(`\n✓ Configuration saved to ${filename}`));
        }
        
      } else {
        console.log(chalk.yellow('\nCould not detect specific project type. Using default configuration.'));
      }
      
    } catch (error) {
      console.error(chalk.red('Error detecting project:', error.message));
    }
  });

// Interactive mode
program
  .command('interactive')
  .description('Start interactive mode')
  .action(async () => {
    console.log(chalk.bold('\n🎮 Interactive Mode\n'));
    console.log('Type "help" for available commands or "exit" to quit.\n');
    
    const commands = {
      help: () => {
        console.log('\nAvailable commands:');
        console.log('  status    - Show system status');
        console.log('  agents    - List all agents');
        console.log('  execute   - Execute a task');
        console.log('  cache     - Show cache stats');
        console.log('  perf      - Show performance metrics');
        console.log('  exit      - Exit interactive mode');
      },
      status: async () => {
        if (!system || !system.isRunning) {
          console.log(chalk.yellow('System is not running'));
          return;
        }
        const status = system.getSystemStatus();
        console.log(`\nSystem: ${status.isRunning ? chalk.green('Running') : chalk.red('Stopped')}`);
        console.log(`Agents: ${Object.keys(status.agents).length} total, ${Object.values(status.agents).filter(a => a.status === 'working').length} active`);
      },
      agents: async () => {
        if (!system) return;
        for (const [id, agent] of system.agents) {
          const status = agent.getStatus();
          console.log(`${chalk.cyan(`Agent ${id}`)}: ${status.name} (${status.status})`);
        }
      },
      execute: async () => {
        if (!system || !system.isRunning) {
          console.log(chalk.yellow('System is not running'));
          return;
        }
        
        const answers = await inquirer.prompt([
          {
            type: 'input',
            name: 'description',
            message: 'Task description:'
          },
          {
            type: 'list',
            name: 'type',
            message: 'Task type:',
            choices: ['testing', 'documentation', 'bug_fixing', 'code_review', 'git_operations']
          },
          {
            type: 'list',
            name: 'priority',
            message: 'Priority:',
            choices: ['high', 'medium', 'low']
          }
        ]);
        
        const spinner = ora('Executing task...').start();
        const result = await system.executeTask(answers);
        
        if (result.success) {
          spinner.succeed(`Task completed by Agent ${result.agentId} in ${result.duration}ms`);
        } else {
          spinner.fail(`Task failed: ${result.error}`);
        }
      },
      cache: async () => {
        if (!system) return;
        const stats = system.cacheLayer.getStats();
        console.log(`\nCache: ${stats.hitRate} hit rate, ${stats.entries} entries, ${stats.memoryUsageMB}MB used`);
      },
      perf: async () => {
        if (!system) return;
        const perf = system.getPerformanceReport();
        console.log(`\nPerformance:`);
        console.log(`  Cache Hit Rate: ${perf.optimization.cacheHitRate}`);
        console.log(`  Task Optimizations: ${perf.optimization.taskOptimizations}`);
        console.log(`  Pool Connections: ${perf.optimization.poolUtilization.activeConnections}/${perf.optimization.poolUtilization.totalConnections}`);
      },
      exit: () => {
        console.log(chalk.green('\nGoodbye!'));
        process.exit(0);
      }
    };
    
    while (true) {
      const { command } = await inquirer.prompt([{
        type: 'input',
        name: 'command',
        message: '>'
      }]);
      
      const [cmd, ...args] = command.trim().toLowerCase().split(' ');
      
      if (commands[cmd]) {
        await commands[cmd](args);
      } else if (cmd) {
        console.log(chalk.red(`Unknown command: ${cmd}. Type "help" for available commands.`));
      }
    }
  });

// Connect learning system to agents
function connectLearningSystem() {
  for (const [agentId, agent] of system.agents) {
    agent.on('taskComplete', async (data) => {
      await learningSystem.recordTaskExecution(agentId, data.task, data.result);
    });
  }
}

// Handle process termination
process.on('SIGINT', async () => {
  console.log(chalk.yellow('\n\nReceived interrupt signal...'));
  
  if (dashboard) await dashboard.stop();
  if (learningSystem) await learningSystem.shutdown();
  if (system) await system.shutdown();
  
  console.log(chalk.green('Shutdown complete'));
  process.exit(0);
});

// Parse and execute
program.parse(process.argv);