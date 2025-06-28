/**
 * Agent Configuration System
 * 
 * Allows dynamic configuration of agents with custom prompts and task mappings
 */

import { logger } from '../utils/logger.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export class AgentConfig {
  constructor() {
    this.configurations = new Map();
    this.projectConfigs = new Map();
    this.defaultConfig = this.getDefaultConfig();
  }

  /**
   * Get default agent configuration
   */
  getDefaultConfig() {
    return {
      taskMapping: {
        'testing': [1],
        'documentation': [2],
        'bug_fixing': [3],
        'code_commenting': [4],
        'git_operations': [5],
        'task_scheduling': [6],
        'code_review': [7],
        'deployment': [8],
        'performance_optimization': [9],
        'security_audit': [10],
        'data_analysis': [11],
        'integration': [12],
        'configuration': [13],
        'backup': [14],
        'compliance': [15],
        'research': [16]
      },
      agentOverrides: {},
      systemPrompt: null,
      projectPath: process.cwd(),
      conditions: []
    };
  }

  /**
   * Load configuration from file
   */
  async loadConfigFile(configPath) {
    try {
      const absolutePath = path.resolve(configPath);
      const configContent = await fs.readFile(absolutePath, 'utf-8');
      const config = JSON.parse(configContent);
      
      logger.info(`Loaded configuration from ${absolutePath}`);
      return this.validateConfig(config);
    } catch (error) {
      logger.error(`Failed to load config file ${configPath}:`, error);
      throw error;
    }
  }

  /**
   * Validate configuration structure
   */
  validateConfig(config) {
    const validated = { ...this.defaultConfig, ...config };
    
    // Validate task mapping
    if (config.taskMapping) {
      for (const [taskType, agentIds] of Object.entries(config.taskMapping)) {
        if (!Array.isArray(agentIds)) {
          throw new Error(`Task mapping for ${taskType} must be an array`);
        }
        if (!agentIds.every(id => typeof id === 'number' && id >= 0 && id <= 16)) {
          throw new Error(`Invalid agent IDs in task mapping for ${taskType}`);
        }
      }
    }
    
    // Validate agent overrides
    if (config.agentOverrides) {
      for (const [agentId, override] of Object.entries(config.agentOverrides)) {
        const id = parseInt(agentId);
        if (isNaN(id) || id < 0 || id > 16) {
          throw new Error(`Invalid agent ID: ${agentId}`);
        }
        if (override.systemPrompt && typeof override.systemPrompt !== 'string') {
          throw new Error(`System prompt for agent ${agentId} must be a string`);
        }
      }
    }
    
    return validated;
  }

  /**
   * Create project-specific configuration
   */
  async createProjectConfig(projectPath, options = {}) {
    const config = {
      projectName: path.basename(projectPath),
      projectPath: path.resolve(projectPath),
      createdAt: new Date().toISOString(),
      ...options
    };
    
    // Detect project type and suggest configurations
    const suggestions = await this.detectProjectType(projectPath);
    if (suggestions) {
      config.taskMapping = { ...suggestions.taskMapping, ...config.taskMapping };
      config.agentSpecializations = suggestions.agentSpecializations;
    }
    
    const configId = `${config.projectName}-${Date.now()}`;
    this.projectConfigs.set(configId, config);
    
    return { configId, config };
  }

  /**
   * Detect project type and suggest configurations
   */
  async detectProjectType(projectPath) {
    try {
      const files = await fs.readdir(projectPath);
      
      // GaiaScript project
      if (files.some(f => f.endsWith('.gaia'))) {
        return {
          projectType: 'gaiascript',
          taskMapping: {
            'lexer': [1],
            'parser': [2],
            'ast_transformer': [3],
            'js_codegen': [4],
            'go_codegen': [5],
            'compiler_testing': [6],
            'documentation': [7],
            'optimization': [8],
            'type_checking': [9],
            'source_mapping': [10],
            'error_handling': [11],
            'cli_interface': [12],
            'wasm_codegen': [13],
            'assembly_codegen': [14],
            'neural_network': [15],
            'integration': [16]
          },
          agentSpecializations: {
            1: 'Chinese character tokenization specialist',
            2: 'GaiaScript syntax parser',
            3: 'AST transformation expert',
            4: 'JavaScript code generator',
            5: 'Go code generator',
            6: 'Compiler test specialist'
          }
        };
      }
      
      // Node.js project
      if (files.includes('package.json')) {
        const pkg = JSON.parse(await fs.readFile(path.join(projectPath, 'package.json'), 'utf-8'));
        const isTypeScript = files.includes('tsconfig.json');
        
        return {
          projectType: isTypeScript ? 'typescript' : 'javascript',
          taskMapping: {
            'unit_testing': [1],
            'api_documentation': [2],
            'dependency_management': [3],
            'linting': [4],
            'npm_operations': [5],
            'build_automation': [6],
            'typescript_compilation': [7],
            'module_bundling': [8],
            'performance_profiling': [9],
            'security_scanning': [10],
            'test_coverage': [11],
            'ci_cd': [12],
            'environment_config': [13],
            'package_publishing': [14],
            'license_compliance': [15],
            'dependency_analysis': [16]
          }
        };
      }
      
      // Python project
      if (files.includes('requirements.txt') || files.includes('setup.py') || files.includes('pyproject.toml')) {
        return {
          projectType: 'python',
          taskMapping: {
            'pytest': [1],
            'sphinx_docs': [2],
            'pylint': [3],
            'docstring': [4],
            'pip_management': [5],
            'virtualenv': [6],
            'type_hints': [7],
            'package_build': [8],
            'profiling': [9],
            'security_check': [10],
            'coverage': [11],
            'tox_testing': [12],
            'config_management': [13],
            'pypi_publish': [14],
            'license_check': [15],
            'import_analysis': [16]
          }
        };
      }
      
      // Default configuration
      return null;
      
    } catch (error) {
      logger.warn(`Could not detect project type for ${projectPath}:`, error);
      return null;
    }
  }

  /**
   * Get configuration for a specific project
   */
  getProjectConfig(projectPath) {
    const resolvedPath = path.resolve(projectPath);
    
    // Check if we have a specific config for this path
    for (const [configId, config] of this.projectConfigs) {
      if (config.projectPath === resolvedPath) {
        return config;
      }
    }
    
    return this.defaultConfig;
  }

  /**
   * Apply configuration to Agent 0
   */
  applyToCoordinator(coordinator, config) {
    // Override task mapping
    if (config.taskMapping) {
      coordinator.taskAgentMapping = config.taskMapping;
    }
    
    // Apply system prompt if provided
    if (config.systemPrompt) {
      coordinator.systemPrompt = config.systemPrompt;
    }
    
    // Apply conditions
    if (config.conditions && config.conditions.length > 0) {
      for (const condition of config.conditions) {
        coordinator.activeConditions.set(condition.id, condition);
      }
    }
    
    logger.info('Applied configuration to coordinator');
  }

  /**
   * Apply agent-specific overrides
   */
  applyAgentOverrides(agents, config) {
    if (!config.agentOverrides) return;
    
    for (const [agentId, override] of Object.entries(config.agentOverrides)) {
      const id = parseInt(agentId);
      const agent = agents.get(id);
      
      if (agent) {
        // Apply custom system prompt
        if (override.systemPrompt) {
          agent.customPrompt = override.systemPrompt;
        }
        
        // Apply custom capabilities
        if (override.capabilities) {
          agent.capabilities = [...agent.capabilities, ...override.capabilities];
        }
        
        // Apply custom tools
        if (override.allowedTools) {
          agent.allowedTools = [...new Set([...agent.allowedTools, ...override.allowedTools])];
        }
        
        logger.info(`Applied overrides to Agent ${id}`);
      }
    }
  }

  /**
   * Save configuration to file
   */
  async saveConfig(config, outputPath) {
    try {
      const configContent = JSON.stringify(config, null, 2);
      await fs.writeFile(outputPath, configContent, 'utf-8');
      logger.info(`Saved configuration to ${outputPath}`);
    } catch (error) {
      logger.error(`Failed to save configuration:`, error);
      throw error;
    }
  }

  /**
   * Generate example configuration file
   */
  async generateExampleConfig(outputPath) {
    const exampleConfig = {
      projectName: "example-project",
      projectPath: "./",
      taskMapping: {
        "custom_task": [1, 2],
        "analysis": [11, 16],
        "compilation": [7, 8, 9]
      },
      agentOverrides: {
        "1": {
          systemPrompt: "You are a specialized testing agent focused on unit tests and TDD.",
          capabilities: ["test_driven_development", "mocking", "coverage_analysis"]
        },
        "2": {
          systemPrompt: "You are a documentation expert specializing in API documentation.",
          allowedTools: ["Read", "Write", "WebFetch"]
        }
      },
      conditions: [
        {
          id: "all_tests_pass",
          type: "boolean",
          check: "all_tests_pass",
          description: "All unit tests must pass"
        },
        {
          id: "documentation_complete",
          type: "llm_validated",
          description: "Documentation is comprehensive and up-to-date"
        }
      ],
      systemPrompt: "This is a custom multi-agent system for the example project. Focus on quality and thorough testing."
    };
    
    await this.saveConfig(exampleConfig, outputPath);
    return exampleConfig;
  }
}

// Export singleton instance
export const agentConfig = new AgentConfig();