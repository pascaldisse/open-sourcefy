import { AgentBase } from '../core/agent-base.js';
import { logger } from '../utils/logger.js';

/**
 * Agent 1: Test Engineer
 * Specializes in testing and quality validation
 */
export class Agent1 extends AgentBase {
  constructor() {
    super(
      1,
      'Test Engineer',
      'Testing and quality validation specialist',
      [
        'Write unit, integration, and end-to-end tests',
        'Execute test suites and analyze results',
        'Generate test reports and coverage metrics',
        'Identify testing gaps and recommend improvements',
        'Implement test automation frameworks',
        'Performance and load testing',
        'API testing and validation',
        'Test data management'
      ],
      [
        'Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS'
      ]
    );

    this.testFrameworks = [
      'jest', 'vitest', 'mocha', 'chai', 'cypress', 'playwright', 
      'supertest', 'testing-library'
    ];
    this.supportedLanguages = [
      'javascript', 'typescript', 'python', 'java', 'go', 'rust'
    ];
  }

  /**
   * Override task type validation for test engineer
   */
  canHandleTaskType(taskType) {
    const testingTasks = [
      'testing',
      'unit_test',
      'integration_test',
      'e2e_test',
      'test_automation',
      'test_coverage',
      'performance_test',
      'load_test',
      'api_test',
      'test_framework_setup',
      'test_data_generation',
      'test_report_generation'
    ];
    
    return testingTasks.includes(taskType);
  }

  /**
   * Generate system prompt specific to test engineer role
   */
  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Test Engineer, you have specific expertise in:

TESTING FRAMEWORKS:
- Jest/Vitest for unit testing
- Cypress/Playwright for E2E testing
- Supertest for API testing
- Testing Library for component testing

TESTING BEST PRACTICES:
- Follow the testing pyramid (unit > integration > e2e)
- Write descriptive test names and clear assertions
- Use arrange-act-assert pattern
- Mock external dependencies appropriately
- Maintain high test coverage (>80%)
- Ensure tests are fast, reliable, and independent

QUALITY METRICS:
- Code coverage analysis
- Test execution time optimization
- Flaky test identification and resolution
- Performance benchmarking

SPECIAL INSTRUCTIONS:
- Always run existing tests before adding new ones
- Generate test reports with coverage metrics
- Identify gaps in test coverage
- Suggest improvements to testing infrastructure
- Follow project-specific testing conventions
- Ensure all tests pass before marking task complete

When writing tests:
1. Analyze the code structure and identify testable units
2. Create comprehensive test cases covering edge cases
3. Use appropriate mocking for external dependencies
4. Generate clear, readable test descriptions
5. Verify test coverage meets quality standards`;
  }

  /**
   * Execute test-specific tasks
   */
  async processTask(task) {
    logger.info(`Test Engineer processing ${task.type} task: ${task.description}`);

    // Pre-process task based on type
    const enhancedTask = await this.enhanceTestTask(task);
    
    // Execute using base class with enhanced context
    const result = await super.processTask(enhancedTask);

    // Post-process results for testing tasks
    if (result.success) {
      result.testMetrics = await this.extractTestMetrics(result);
    }

    return result;
  }

  /**
   * Enhance task with testing-specific context
   */
  async enhanceTestTask(task) {
    const enhanced = { ...task };

    // Add testing context based on task type
    switch (task.type) {
      case 'unit_test':
        enhanced.context = {
          ...enhanced.context,
          testType: 'unit',
          framework: 'jest',
          coverageTarget: 90,
          mockStrategy: 'dependencies'
        };
        break;

      case 'integration_test':
        enhanced.context = {
          ...enhanced.context,
          testType: 'integration',
          framework: 'jest',
          coverageTarget: 80,
          includeDatabase: true
        };
        break;

      case 'e2e_test':
        enhanced.context = {
          ...enhanced.context,
          testType: 'e2e',
          framework: 'cypress',
          browserSupport: ['chrome', 'firefox'],
          viewports: ['desktop', 'mobile']
        };
        break;

      case 'api_test':
        enhanced.context = {
          ...enhanced.context,
          testType: 'api',
          framework: 'supertest',
          validateSchema: true,
          checkStatusCodes: true
        };
        break;

      case 'performance_test':
        enhanced.context = {
          ...enhanced.context,
          testType: 'performance',
          metrics: ['response_time', 'throughput', 'memory_usage'],
          thresholds: {
            response_time: '< 200ms',
            memory_usage: '< 100MB'
          }
        };
        break;
    }

    // Add framework-specific requirements
    enhanced.requirements = `${enhanced.requirements || ''}

TESTING REQUIREMENTS:
- Use ${enhanced.context.framework || 'jest'} as the testing framework
- Achieve ${enhanced.context.coverageTarget || 80}% code coverage
- Follow naming convention: *.test.js or *.spec.js
- Include both positive and negative test cases
- Add performance assertions where applicable
- Generate detailed test reports
- Ensure all tests are deterministic and fast`;

    return enhanced;
  }

  /**
   * Extract test metrics from task results
   */
  async extractTestMetrics(result) {
    const metrics = {
      testsRun: 0,
      testsPassed: 0,
      testsFailed: 0,
      coverage: 0,
      executionTime: 0,
      framework: 'unknown'
    };

    try {
      // Parse test output for metrics
      const output = result.result;
      
      // Jest/Vitest output parsing
      const jestMatch = output.match(/Tests:\s+(\d+)\s+passed,\s+(\d+)\s+total/);
      if (jestMatch) {
        metrics.testsPassed = parseInt(jestMatch[1]);
        metrics.testsRun = parseInt(jestMatch[2]);
        metrics.testsFailed = metrics.testsRun - metrics.testsPassed;
        metrics.framework = 'jest';
      }

      // Coverage parsing
      const coverageMatch = output.match(/All files\s+\|\s+([\d.]+)/);
      if (coverageMatch) {
        metrics.coverage = parseFloat(coverageMatch[1]);
      }

      // Execution time parsing
      const timeMatch = output.match(/Time:\s+([\d.]+)s/);
      if (timeMatch) {
        metrics.executionTime = parseFloat(timeMatch[1]) * 1000; // Convert to ms
      }

    } catch (error) {
      logger.warn('Failed to extract test metrics:', error);
    }

    return metrics;
  }

  /**
   * Run test suite and analyze results
   */
  async runTestSuite(testPath, options = {}) {
    const task = {
      description: `Run test suite at ${testPath}`,
      type: 'test_execution',
      context: {
        testPath,
        options: {
          coverage: true,
          verbose: true,
          ...options
        }
      },
      requirements: 'Execute tests and provide detailed analysis of results'
    };

    return await this.processTask(task);
  }

  /**
   * Generate test coverage report
   */
  async generateCoverageReport(projectPath) {
    const task = {
      description: 'Generate comprehensive test coverage report',
      type: 'test_coverage',
      context: {
        projectPath,
        outputFormat: ['html', 'json', 'text'],
        includeUncovered: true
      },
      requirements: 'Create detailed coverage report with recommendations for improvement'
    };

    return await this.processTask(task);
  }

  /**
   * Create test automation framework
   */
  async setupTestFramework(projectType, requirements) {
    const task = {
      description: 'Set up test automation framework for project',
      type: 'test_framework_setup',
      context: {
        projectType,
        requirements,
        frameworks: this.testFrameworks,
        bestPractices: true
      },
      requirements: 'Configure complete testing infrastructure with CI/CD integration'
    };

    return await this.processTask(task);
  }

  /**
   * Identify and fix flaky tests
   */
  async analyzeTestStability(testResults) {
    const task = {
      description: 'Analyze test stability and identify flaky tests',
      type: 'test_analysis',
      context: {
        testResults,
        analysisType: 'stability',
        recommendations: true
      },
      requirements: 'Provide actionable recommendations for improving test reliability'
    };

    return await this.processTask(task);
  }
}