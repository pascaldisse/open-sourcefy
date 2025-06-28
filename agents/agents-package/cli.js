#!/usr/bin/env node

/**
 * Multi-Agent System Standalone CLI
 * Can be run in any directory with any project
 */

import { Command } from 'commander';
import { MultiAgentSystem } from './src/index.js';
import { agentConfig } from './src/config/agent-config.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const program = new Command();

program
  .name('gaia-mas')
  .description('Multi-Agent System - Run in any directory')
  .version('1.0.0');

// Start command
program
  .command('start')
  .description('Start the multi-agent system')
  .option('-c, --config <path>', 'Configuration file')
  .option('--project <path>', 'Project path', process.cwd())
  .option('--prompt <prompt>', 'Custom system prompt')
  .option('-d, --dashboard', 'Start dashboard')
  .option('-p, --port <port>', 'Dashboard port', '3001')
  .action(async (options) => {
    try {
      console.log('🚀 Starting Multi-Agent System...');
      console.log('Working directory:', options.project);
      
      const system = new MultiAgentSystem();
      
      // Load or create configuration
      let config = null;
      if (options.config) {
        const configPath = path.resolve(options.config);
        config = await agentConfig.loadConfigFile(configPath);
      } else {
        console.log('Auto-detecting project configuration...');
        const projectConfig = await agentConfig.createProjectConfig(options.project, {
          systemPrompt: options.prompt
        });
        config = projectConfig.config;
      }
      
      await system.initialize();
      
      if (config) {
        agentConfig.applyToCoordinator(system.coordinator, config);
        agentConfig.applyAgentOverrides(system.agents, config);
      }
      
      await system.start({ conditions: config?.conditions || [] });
      
      console.log('✅ Multi-Agent System running!');
      console.log('System ready for task execution.');
      
      // Keep running
      process.stdin.resume();
      
    } catch (error) {
      console.error('❌ Error:', error.message);
      process.exit(1);
    }
  });

// Config commands
program
  .command('config')
  .description('Configuration management')
  .addCommand(
    new Command('generate')
      .argument('<output>', 'Output file')
      .description('Generate example configuration')
      .action(async (output) => {
        await agentConfig.generateExampleConfig(output);
        console.log('✅ Configuration generated:', output);
      })
  )
  .addCommand(
    new Command('detect')
      .argument('[path]', 'Project path', process.cwd())
      .description('Detect project type')
      .action(async (projectPath) => {
        const suggestions = await agentConfig.detectProjectType(projectPath);
        if (suggestions) {
          console.log('Detected project type:', suggestions.projectType);
          console.log('Suggested configuration available');
        } else {
          console.log('Could not detect specific project type');
        }
      })
  );

// Execute command
program
  .command('execute')
  .argument('<description>', 'Task description')
  .option('-t, --type <type>', 'Task type', 'testing')
  .option('-p, --priority <priority>', 'Priority', 'medium')
  .description('Execute a single task')
  .action(async (description, options) => {
    try {
      const system = new MultiAgentSystem();
      await system.initialize();
      await system.start();
      
      const result = await system.executeTask({
        description,
        type: options.type,
        priority: options.priority
      });
      
      if (result.success) {
        console.log('✅ Task completed successfully');
      } else {
        console.log('❌ Task failed:', result.error);
      }
      
      await system.shutdown();
      
    } catch (error) {
      console.error('❌ Error:', error.message);
      process.exit(1);
    }
  });

// Status command
program
  .command('status')
  .description('Check system status')
  .action(() => {
    console.log('Multi-Agent System Status');
    console.log('Ready to start in any directory');
    console.log('Use "gaia-mas start" to begin');
  });

program.parse();