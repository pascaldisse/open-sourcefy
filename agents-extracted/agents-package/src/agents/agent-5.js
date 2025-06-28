import { AgentBase } from '../core/agent-base.js';
import { logger } from '../utils/logger.js';

/**
 * Agent 5: Git Operations Manager
 * Specializes in version control and repository management
 */
export class Agent5 extends AgentBase {
  constructor() {
    super(
      5,
      'Git Operations Manager',
      'Version control and repository management specialist',
      [
        'Manage branch creation and merging',
        'Enforce commit message standards',
        'Handle pull request workflows',
        'Manage releases and tags',
        'Maintain repository hygiene',
        'Resolve merge conflicts',
        'Manage git hooks and automation',
        'Repository security and access control'
      ],
      [
        'Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS'
      ]
    );

    this.gitCommands = [
      'status', 'add', 'commit', 'push', 'pull', 'merge', 'branch',
      'checkout', 'rebase', 'tag', 'log', 'diff', 'reset', 'stash'
    ];
    
    this.commitTypes = [
      'feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore',
      'perf', 'ci', 'build', 'revert'
    ];

    this.workflowTypes = [
      'gitflow', 'github-flow', 'gitlab-flow', 'feature-branch'
    ];
  }

  /**
   * Override task type validation for git operations manager
   */
  canHandleTaskType(taskType) {
    const gitTasks = [
      'git_operations',
      'branch_management',
      'commit_management',
      'merge_management',
      'release_management',
      'repository_cleanup',
      'git_workflow',
      'conflict_resolution',
      'tag_management',
      'hook_management',
      'repository_security',
      'git_automation'
    ];
    
    return gitTasks.includes(taskType);
  }

  /**
   * Generate system prompt specific to git operations manager role
   */
  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Git Operations Manager, you have specific expertise in:

GIT WORKFLOW STRATEGIES:
- GitFlow: master, develop, feature, release, hotfix branches
- GitHub Flow: main branch with feature branches and pull requests
- GitLab Flow: production branch with feature branches
- Feature Branch Workflow: short-lived feature branches

COMMIT MESSAGE STANDARDS:
- Conventional Commits: type(scope): description
- Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert
- Format: <type>[optional scope]: <description> [optional body] [optional footer]
- Examples: "feat(auth): add OAuth2 integration", "fix: resolve memory leak in parser"

BRANCHING BEST PRACTICES:
- Use descriptive branch names: feature/user-authentication, bugfix/memory-leak
- Keep branches focused and short-lived
- Regularly sync with main/develop branch
- Delete merged branches to maintain hygiene
- Protect main/production branches with rules

MERGE STRATEGIES:
- Merge commits: preserve branch history
- Squash and merge: clean linear history
- Rebase and merge: linear history without merge commits
- Fast-forward: when possible for simple updates

REPOSITORY MANAGEMENT:
- Tag releases with semantic versioning (v1.2.3)
- Maintain clean commit history
- Use .gitignore appropriately
- Configure git hooks for automation
- Implement branch protection rules
- Regular repository maintenance

SECURITY BEST PRACTICES:
- Never commit sensitive information (keys, passwords)
- Use .gitignore for temporary and build files
- Sign commits with GPG when required
- Review changes before committing
- Use SSH keys for authentication

SPECIAL INSTRUCTIONS:
- Always check git status before operations
- Use descriptive commit messages following conventions
- Verify changes with git diff before committing
- Handle merge conflicts carefully and test resolution
- Backup important branches before destructive operations
- Follow repository-specific workflow guidelines
- Coordinate with team members for shared branches

When performing git operations:
1. Check current repository state (status, branch, remotes)
2. Understand the intended outcome and impact
3. Use appropriate git commands and options
4. Verify operations completed successfully
5. Communicate significant changes to team
6. Document complex operations for future reference`;
  }

  /**
   * Execute git-specific tasks
   */
  async processTask(task) {
    logger.info(`Git Operations Manager processing ${task.type} task: ${task.description}`);

    // Pre-process task based on type
    const enhancedTask = await this.enhanceGitTask(task);
    
    // Execute using base class with enhanced context
    const result = await super.processTask(enhancedTask);

    // Post-process results for git tasks
    if (result.success) {
      result.gitMetrics = await this.extractGitMetrics(result);
    }

    return result;
  }

  /**
   * Enhance task with git-specific context
   */
  async enhanceGitTask(task) {
    const enhanced = { ...task };

    // Add git context based on task type
    switch (task.type) {
      case 'branch_management':
        enhanced.context = {
          ...enhanced.context,
          workflow: 'gitflow',
          branchNaming: 'feature/description',
          cleanup: true,
          protection: true
        };
        break;

      case 'commit_management':
        enhanced.context = {
          ...enhanced.context,
          messageFormat: 'conventional',
          signCommits: false,
          squashOption: true,
          verifyChanges: true
        };
        break;

      case 'merge_management':
        enhanced.context = {
          ...enhanced.context,
          strategy: 'merge-commit',
          conflictResolution: 'interactive',
          testBeforeMerge: true,
          branchCleanup: true
        };
        break;

      case 'release_management':
        enhanced.context = {
          ...enhanced.context,
          tagging: 'semantic-versioning',
          changelog: true,
          releaseNotes: true,
          branchStrategy: 'release-branch'
        };
        break;

      case 'repository_cleanup':
        enhanced.context = {
          ...enhanced.context,
          deleteMergedBranches: true,
          updateGitignore: true,
          compactHistory: false,
          removeStaleRemotes: true
        };
        break;
    }

    // Add git-specific requirements
    enhanced.requirements = `${enhanced.requirements || ''}

GIT OPERATION REQUIREMENTS:
- Follow ${enhanced.context.workflow || 'standard'} workflow practices
- Use ${enhanced.context.messageFormat || 'conventional'} commit message format
- Apply ${enhanced.context.strategy || 'appropriate'} merge strategy
- Maintain repository cleanliness and organization
- Verify all operations with git status and git log
- Handle conflicts gracefully with proper resolution
- Backup important data before destructive operations
- Follow team collaboration guidelines
- Document significant changes and decisions
- Ensure security best practices are followed`;

    return enhanced;
  }

  /**
   * Extract git metrics from task results
   */
  async extractGitMetrics(result) {
    const metrics = {
      commitsCreated: 0,
      branchesCreated: 0,
      branchesDeleted: 0,
      mergesPerformed: 0,
      conflictsResolved: 0,
      tagsCreated: 0,
      filesModified: 0,
      linesChanged: 0
    };

    try {
      const output = result.result;
      
      // Count commits
      const commitMatches = output.match(/(?:commit|committed)/gi);
      metrics.commitsCreated = commitMatches ? commitMatches.length : 0;
      
      // Count branches
      const branchCreateMatches = output.match(/(?:created|checkout -b).*branch/gi);
      metrics.branchesCreated = branchCreateMatches ? branchCreateMatches.length : 0;
      
      const branchDeleteMatches = output.match(/(?:deleted|branch -d).*branch/gi);
      metrics.branchesDeleted = branchDeleteMatches ? branchDeleteMatches.length : 0;
      
      // Count merges
      const mergeMatches = output.match(/merge|merged/gi);
      metrics.mergesPerformed = mergeMatches ? mergeMatches.length : 0;
      
      // Count conflicts
      const conflictMatches = output.match(/conflict|resolved/gi);
      metrics.conflictsResolved = conflictMatches ? conflictMatches.length : 0;
      
      // Count tags
      const tagMatches = output.match(/tag.*created|tagged/gi);
      metrics.tagsCreated = tagMatches ? tagMatches.length : 0;
      
      // Extract file changes if mentioned
      const fileChangeMatches = output.match(/(\d+) files? changed/);
      if (fileChangeMatches) {
        metrics.filesModified = parseInt(fileChangeMatches[1]);
      }
      
      // Extract line changes if mentioned
      const lineChangeMatches = output.match(/(\d+) insertions?.*(\d+) deletions?/);
      if (lineChangeMatches) {
        metrics.linesChanged = parseInt(lineChangeMatches[1]) + parseInt(lineChangeMatches[2]);
      }

    } catch (error) {
      logger.warn('Failed to extract git metrics:', error);
    }

    return metrics;
  }

  /**
   * Create and manage feature branches
   */
  async createFeatureBranch(featureName, baseBranch = 'main') {
    const task = {
      description: `Create feature branch for ${featureName}`,
      type: 'branch_management',
      context: {
        featureName,
        baseBranch,
        branchNaming: 'feature/',
        switchToBranch: true
      },
      requirements: 'Create properly named feature branch and switch to it'
    };

    return await this.processTask(task);
  }

  /**
   * Commit changes with proper formatting
   */
  async commitChanges(message, type = 'feat', scope = null) {
    const task = {
      description: 'Commit staged changes with conventional commit format',
      type: 'commit_management',
      context: {
        message,
        type,
        scope,
        format: 'conventional',
        verify: true
      },
      requirements: 'Create well-formatted commit following conventional commit standards'
    };

    return await this.processTask(task);
  }

  /**
   * Merge branches with conflict resolution
   */
  async mergeBranch(sourceBranch, targetBranch, strategy = 'merge') {
    const task = {
      description: `Merge ${sourceBranch} into ${targetBranch}`,
      type: 'merge_management',
      context: {
        sourceBranch,
        targetBranch,
        strategy,
        resolveConflicts: true,
        testAfterMerge: true
      },
      requirements: 'Safely merge branches with proper conflict resolution and testing'
    };

    return await this.processTask(task);
  }

  /**
   * Create release with tagging
   */
  async createRelease(version, releaseNotes, branch = 'main') {
    const task = {
      description: `Create release ${version}`,
      type: 'release_management',
      context: {
        version,
        releaseNotes,
        branch,
        createTag: true,
        updateChangelog: true
      },
      requirements: 'Create properly tagged release with documentation'
    };

    return await this.processTask(task);
  }

  /**
   * Clean up repository
   */
  async cleanupRepository(options = {}) {
    const task = {
      description: 'Clean up repository by removing stale branches and organizing',
      type: 'repository_cleanup',
      context: {
        deleteMergedBranches: true,
        removeStaleRemotes: true,
        updateGitignore: false,
        ...options
      },
      requirements: 'Safely clean repository while preserving important history'
    };

    return await this.processTask(task);
  }

  /**
   * Resolve merge conflicts
   */
  async resolveConflicts(conflictedFiles, resolutionStrategy = 'interactive') {
    const task = {
      description: 'Resolve merge conflicts in specified files',
      type: 'conflict_resolution',
      context: {
        conflictedFiles,
        resolutionStrategy,
        testAfterResolution: true,
        backupChanges: true
      },
      requirements: 'Carefully resolve conflicts while preserving intended functionality'
    };

    return await this.processTask(task);
  }

  /**
   * Set up git hooks
   */
  async setupGitHooks(hookTypes, configurations) {
    const task = {
      description: 'Set up git hooks for automation and validation',
      type: 'hook_management',
      context: {
        hookTypes,
        configurations,
        validation: true,
        automation: true
      },
      requirements: 'Configure git hooks for improved workflow automation'
    };

    return await this.processTask(task);
  }

  /**
   * Analyze repository history
   */
  async analyzeRepository(analysisType = 'comprehensive') {
    const task = {
      description: 'Analyze repository history and structure',
      type: 'repository_analysis',
      context: {
        analysisType,
        includeStatistics: true,
        identifyIssues: true,
        suggestions: true
      },
      requirements: 'Provide comprehensive repository analysis with recommendations'
    };

    return await this.processTask(task);
  }

  /**
   * Implement branching strategy
   */
  async implementBranchingStrategy(strategyType, configuration) {
    const task = {
      description: `Implement ${strategyType} branching strategy`,
      type: 'git_workflow',
      context: {
        strategyType,
        configuration,
        setupProtectionRules: true,
        documentation: true
      },
      requirements: 'Set up complete branching strategy with protection rules and documentation'
    };

    return await this.processTask(task);
  }
}