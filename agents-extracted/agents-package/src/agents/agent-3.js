import { AgentBase } from '../core/agent-base.js';
import { logger } from '../utils/logger.js';

/**
 * Agent 3: Bug Hunter
 * Specializes in error detection and resolution
 */
export class Agent3 extends AgentBase {
  constructor() {
    super(
      3,
      'Bug Hunter',
      'Error detection and resolution specialist',
      [
        'Analyze error logs and stack traces',
        'Identify root causes of issues',
        'Implement bug fixes and patches',
        'Monitor system health and performance',
        'Prevent regression issues',
        'Debug complex application issues',
        'Performance bottleneck identification',
        'Memory leak detection and resolution'
      ],
      [
        'Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS'
      ]
    );

    this.debuggingTools = [
      'chrome-devtools', 'node-inspector', 'gdb', 'lldb',
      'strace', 'perf', 'valgrind', 'memory-profiler'
    ];
    this.analysisTypes = [
      'stack_trace', 'memory_leak', 'performance', 'concurrency',
      'logic_error', 'runtime_error', 'compilation_error'
    ];
    this.bugCategories = [
      'critical', 'high', 'medium', 'low', 'enhancement'
    ];
  }

  /**
   * Override task type validation for bug hunter
   */
  canHandleTaskType(taskType) {
    const debuggingTasks = [
      'bug_fixing',
      'debugging',
      'error_analysis',
      'stack_trace_analysis',
      'performance_debugging',
      'memory_leak_fix',
      'crash_investigation',
      'regression_fix',
      'error_monitoring',
      'issue_triage',
      'root_cause_analysis',
      'system_diagnostics'
    ];
    
    return debuggingTasks.includes(taskType);
  }

  /**
   * Generate system prompt specific to bug hunter role
   */
  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Bug Hunter, you have specific expertise in:

DEBUGGING METHODOLOGIES:
- Systematic problem isolation and reproduction
- Root cause analysis using the 5 Whys technique
- Binary search debugging for complex issues
- Rubber duck debugging for logic problems
- Time travel debugging with git bisect

ERROR ANALYSIS TECHNIQUES:
- Stack trace interpretation and navigation
- Log file analysis and pattern recognition
- Memory dump analysis and heap inspection
- Network traffic analysis for distributed issues
- Database query optimization and debugging

DEBUGGING TOOLS:
- Browser DevTools for frontend issues
- Node.js inspector for backend debugging
- Performance profilers (Chrome DevTools, Node.js profiler)
- Memory profilers (heap snapshots, memory usage)
- Static analysis tools (ESLint, SonarQube)

BUG CLASSIFICATION:
- Critical: System crashes, data loss, security vulnerabilities
- High: Major functionality broken, performance severely degraded
- Medium: Minor functionality issues, moderate performance impact
- Low: Cosmetic issues, minor inconveniences
- Enhancement: Feature requests and improvements

DEBUGGING BEST PRACTICES:
- Reproduce the issue consistently before fixing
- Create minimal test cases that demonstrate the bug
- Document the debugging process and findings
- Write regression tests to prevent reoccurrence
- Consider edge cases and boundary conditions
- Use version control to track debugging changes

SPECIAL INSTRUCTIONS:
- Always backup code before applying fixes
- Test fixes thoroughly in isolated environment
- Verify the fix doesn't introduce new issues
- Document the root cause and solution
- Add monitoring to prevent similar issues
- Consider performance impact of fixes

When debugging issues:
1. Gather all available information (logs, error messages, reproduction steps)
2. Form hypotheses about potential causes
3. Test hypotheses systematically
4. Implement minimal fixes that address root causes
5. Verify fixes work across different scenarios
6. Add preventive measures (tests, monitoring, validation)`;
  }

  /**
   * Execute debugging-specific tasks
   */
  async processTask(task) {
    logger.info(`Bug Hunter processing ${task.type} task: ${task.description}`);

    // Pre-process task based on type
    const enhancedTask = await this.enhanceDebuggingTask(task);
    
    // Execute using base class with enhanced context
    const result = await super.processTask(enhancedTask);

    // Post-process results for debugging tasks
    if (result.success) {
      result.debugMetrics = await this.extractDebuggingMetrics(result);
    }

    return result;
  }

  /**
   * Enhance task with debugging-specific context
   */
  async enhanceDebuggingTask(task) {
    const enhanced = { ...task };

    // Add debugging context based on task type
    switch (task.type) {
      case 'stack_trace_analysis':
        enhanced.context = {
          ...enhanced.context,
          analysisType: 'stack_trace',
          includeSourceMaps: true,
          identifyRootCause: true,
          suggestFixes: true
        };
        break;

      case 'memory_leak_fix':
        enhanced.context = {
          ...enhanced.context,
          analysisType: 'memory_leak',
          monitoringPeriod: '24h',
          heapAnalysis: true,
          garbageCollection: true
        };
        break;

      case 'performance_debugging':
        enhanced.context = {
          ...enhanced.context,
          analysisType: 'performance',
          metrics: ['cpu', 'memory', 'io', 'network'],
          profiling: true,
          benchmarking: true
        };
        break;

      case 'crash_investigation':
        enhanced.context = {
          ...enhanced.context,
          analysisType: 'crash',
          coreDumpAnalysis: true,
          systemLogs: true,
          reproduction: true
        };
        break;

      case 'regression_fix':
        enhanced.context = {
          ...enhanced.context,
          analysisType: 'regression',
          gitBisect: true,
          changesetAnalysis: true,
          testIsolation: true
        };
        break;
    }

    // Add debugging-specific requirements
    enhanced.requirements = `${enhanced.requirements || ''}

DEBUGGING REQUIREMENTS:
- Reproduce the issue before attempting fixes
- Analyze ${enhanced.context.analysisType || 'general'} thoroughly
- Document all findings and debugging steps
- Create minimal test cases that demonstrate the issue
- Implement targeted fixes that address root causes
- Add regression tests to prevent reoccurrence
- Verify fixes don't introduce new problems
- Consider performance and security implications
- Add appropriate logging and monitoring
- Update documentation with debugging insights`;

    return enhanced;
  }

  /**
   * Extract debugging metrics from task results
   */
  async extractDebuggingMetrics(result) {
    const metrics = {
      issuesFound: 0,
      issuesFixed: 0,
      severity: 'unknown',
      rootCausesIdentified: 0,
      testsAdded: 0,
      timeToResolution: 0,
      regressionRisk: 'low'
    };

    try {
      const output = result.result;
      
      // Count issues found
      const issueMatches = output.match(/(?:bug|issue|error|problem)(?:s)?\s+(?:found|identified)/gi);
      metrics.issuesFound = issueMatches ? issueMatches.length : 0;
      
      // Count fixes applied
      const fixMatches = output.match(/(?:fix|resolve|patch)(?:ed)?/gi);
      metrics.issuesFixed = fixMatches ? fixMatches.length : 0;
      
      // Detect severity
      if (output.toLowerCase().includes('critical') || output.toLowerCase().includes('crash')) {
        metrics.severity = 'critical';
      } else if (output.toLowerCase().includes('high') || output.toLowerCase().includes('major')) {
        metrics.severity = 'high';
      } else if (output.toLowerCase().includes('medium') || output.toLowerCase().includes('moderate')) {
        metrics.severity = 'medium';
      } else if (output.toLowerCase().includes('low') || output.toLowerCase().includes('minor')) {
        metrics.severity = 'low';
      }
      
      // Count root causes
      const rootCauseMatches = output.match(/root\s+cause/gi);
      metrics.rootCausesIdentified = rootCauseMatches ? rootCauseMatches.length : 0;
      
      // Count tests added
      const testMatches = output.match(/test(?:s)?\s+(?:added|created)/gi);
      metrics.testsAdded = testMatches ? testMatches.length : 0;

    } catch (error) {
      logger.warn('Failed to extract debugging metrics:', error);
    }

    return metrics;
  }

  /**
   * Analyze error logs and stack traces
   */
  async analyzeError(errorData, context = {}) {
    const task = {
      description: 'Analyze error logs and identify root cause',
      type: 'error_analysis',
      context: {
        errorData,
        includeStackTrace: true,
        suggestFixes: true,
        ...context
      },
      requirements: 'Provide comprehensive error analysis with actionable solutions'
    };

    return await this.processTask(task);
  }

  /**
   * Debug performance issues
   */
  async debugPerformance(performanceData, thresholds) {
    const task = {
      description: 'Debug performance issues and identify bottlenecks',
      type: 'performance_debugging',
      context: {
        performanceData,
        thresholds,
        profiling: true,
        optimization: true
      },
      requirements: 'Identify performance bottlenecks and provide optimization recommendations'
    };

    return await this.processTask(task);
  }

  /**
   * Investigate memory leaks
   */
  async investigateMemoryLeak(memoryProfile, applicationLogs) {
    const task = {
      description: 'Investigate and fix memory leaks',
      type: 'memory_leak_fix',
      context: {
        memoryProfile,
        applicationLogs,
        heapAnalysis: true,
        leakDetection: true
      },
      requirements: 'Identify memory leak sources and implement fixes'
    };

    return await this.processTask(task);
  }

  /**
   * Investigate system crashes
   */
  async investigateCrash(crashReport, systemLogs) {
    const task = {
      description: 'Investigate system crash and identify cause',
      type: 'crash_investigation',
      context: {
        crashReport,
        systemLogs,
        coreDumpAnalysis: true,
        reproduction: true
      },
      requirements: 'Determine crash cause and implement preventive measures'
    };

    return await this.processTask(task);
  }

  /**
   * Fix regression issues
   */
  async fixRegression(regressionData, lastWorkingVersion) {
    const task = {
      description: 'Fix regression issue by identifying problematic changes',
      type: 'regression_fix',
      context: {
        regressionData,
        lastWorkingVersion,
        gitBisect: true,
        changesetAnalysis: true
      },
      requirements: 'Identify and fix the specific change that caused the regression'
    };

    return await this.processTask(task);
  }

  /**
   * Triage and classify bugs
   */
  async triageBugs(bugReports) {
    const task = {
      description: 'Triage and classify bug reports by severity and priority',
      type: 'issue_triage',
      context: {
        bugReports,
        classification: this.bugCategories,
        prioritization: true
      },
      requirements: 'Classify bugs by severity and provide priority recommendations'
    };

    return await this.processTask(task);
  }

  /**
   * Perform system health check
   */
  async performHealthCheck(systemComponents) {
    const task = {
      description: 'Perform comprehensive system health check',
      type: 'system_diagnostics',
      context: {
        systemComponents,
        healthMetrics: ['cpu', 'memory', 'disk', 'network'],
        alertThresholds: true
      },
      requirements: 'Assess system health and identify potential issues'
    };

    return await this.processTask(task);
  }
}