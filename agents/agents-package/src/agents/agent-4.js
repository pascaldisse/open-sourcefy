import { AgentBase } from '../core/agent-base.js';
import { logger } from '../utils/logger.js';

/**
 * Agent 4: Code Commentator
 * Specializes in code documentation and annotation
 */
export class Agent4 extends AgentBase {
  constructor() {
    super(
      4,
      'Code Commentator',
      'Code documentation and annotation specialist',
      [
        'Add meaningful comments to complex code',
        'Generate JSDoc for functions and classes',
        'Improve code readability and maintainability',
        'Identify areas needing explanation',
        'Maintain comment quality standards',
        'Create inline documentation',
        'Generate code examples and usage patterns',
        'Ensure documentation consistency'
      ],
      [
        'Read', 'Write', 'Edit', 'Glob', 'Grep', 'LS'
      ]
    );

    this.commentStyles = {
      'javascript': { single: '//', block: '/* */', doc: '/** */' },
      'typescript': { single: '//', block: '/* */', doc: '/** */' },
      'python': { single: '#', block: '""" """', doc: '""" """' },
      'java': { single: '//', block: '/* */', doc: '/** */' },
      'go': { single: '//', block: '/* */', doc: '// ' },
      'rust': { single: '//', block: '/* */', doc: '/// ' }
    };

    this.docStandards = [
      'jsdoc', 'tsdoc', 'sphinx', 'javadoc', 'rustdoc', 'godoc'
    ];
  }

  /**
   * Override task type validation for code commentator
   */
  canHandleTaskType(taskType) {
    const commentingTasks = [
      'code_commenting',
      'inline_docs',
      'jsdoc_generation',
      'comment_review',
      'documentation_cleanup',
      'code_explanation',
      'function_documentation',
      'class_documentation',
      'api_commenting',
      'example_generation'
    ];
    
    return commentingTasks.includes(taskType);
  }

  /**
   * Generate system prompt specific to code commentator role
   */
  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Code Commentator, you have specific expertise in:

COMMENT TYPES AND PURPOSES:
- Header Comments: File-level descriptions, license, author info
- Function/Method Comments: Purpose, parameters, return values, examples
- Inline Comments: Complex logic explanation, algorithm steps
- Block Comments: Section descriptions, architectural decisions
- TODO/FIXME Comments: Future improvements and known issues

DOCUMENTATION STANDARDS:
- JSDoc for JavaScript/TypeScript: @param, @returns, @throws, @example
- Python Docstrings: Sphinx-style with Args, Returns, Raises
- Javadoc for Java: @param, @return, @throws, @since
- Rustdoc for Rust: /// for items, //! for modules
- Godoc for Go: Simple sentences describing exported items

COMMENT BEST PRACTICES:
- Explain WHY, not WHAT (code should be self-explanatory)
- Keep comments concise but complete
- Update comments when code changes
- Use consistent terminology and style
- Avoid obvious or redundant comments
- Include usage examples for complex functions
- Document edge cases and assumptions

CODE READABILITY PRINCIPLES:
- Comments should add value, not noise
- Focus on business logic and complex algorithms
- Explain non-obvious design decisions
- Document API contracts and expectations
- Provide context for future maintainers
- Use clear, professional language

SPECIAL INSTRUCTIONS:
- Analyze code structure before adding comments
- Maintain existing comment style and format
- Ensure comments are accurate and up-to-date
- Add examples for public APIs
- Document error conditions and edge cases
- Use appropriate documentation standards for the language
- Avoid over-commenting simple or self-evident code

When commenting code:
1. Read and understand the code thoroughly
2. Identify areas that would benefit from explanation
3. Choose appropriate comment types for each situation
4. Write clear, concise explanations
5. Add examples for complex or public functions
6. Ensure consistency with existing documentation style`;
  }

  /**
   * Execute commenting-specific tasks
   */
  async processTask(task) {
    logger.info(`Code Commentator processing ${task.type} task: ${task.description}`);

    // Pre-process task based on type
    const enhancedTask = await this.enhanceCommentingTask(task);
    
    // Execute using base class with enhanced context
    const result = await super.processTask(enhancedTask);

    // Post-process results for commenting tasks
    if (result.success) {
      result.commentMetrics = await this.extractCommentMetrics(result);
    }

    return result;
  }

  /**
   * Enhance task with commenting-specific context
   */
  async enhanceCommentingTask(task) {
    const enhanced = { ...task };

    // Add commenting context based on task type
    switch (task.type) {
      case 'jsdoc_generation':
        enhanced.context = {
          ...enhanced.context,
          standard: 'jsdoc',
          includeExamples: true,
          includeTypes: true,
          includeThrows: true
        };
        break;

      case 'function_documentation':
        enhanced.context = {
          ...enhanced.context,
          focusArea: 'functions',
          includeParameters: true,
          includeReturnValues: true,
          includeExamples: true
        };
        break;

      case 'class_documentation':
        enhanced.context = {
          ...enhanced.context,
          focusArea: 'classes',
          includeConstructor: true,
          includeMethods: true,
          includeProperties: true
        };
        break;

      case 'inline_docs':
        enhanced.context = {
          ...enhanced.context,
          focusArea: 'inline',
          explainComplexLogic: true,
          clarifyAlgorithms: true,
          documentEdgeCases: true
        };
        break;

      case 'api_commenting':
        enhanced.context = {
          ...enhanced.context,
          focusArea: 'api',
          includeUsageExamples: true,
          documentErrorCodes: true,
          includeContracts: true
        };
        break;
    }

    // Add language-specific context
    if (enhanced.context.language) {
      const language = enhanced.context.language.toLowerCase();
      enhanced.context.commentStyle = this.commentStyles[language] || this.commentStyles.javascript;
    }

    // Add commenting-specific requirements
    enhanced.requirements = `${enhanced.requirements || ''}

COMMENTING REQUIREMENTS:
- Use ${enhanced.context.standard || 'appropriate'} documentation standard
- Focus on ${enhanced.context.focusArea || 'general'} areas
- Maintain consistent comment style throughout
- Explain complex logic and business rules
- Document parameters, return values, and exceptions
- Include practical usage examples
- Ensure comments add genuine value
- Keep comments concise but complete
- Update existing comments if they're outdated
- Follow language-specific commenting conventions`;

    return enhanced;
  }

  /**
   * Extract comment metrics from task results
   */
  async extractCommentMetrics(result) {
    const metrics = {
      commentsAdded: 0,
      commentsUpdated: 0,
      functionsDocumented: 0,
      classesDocumented: 0,
      examplesAdded: 0,
      documentationCoverage: 0,
      standard: 'unknown'
    };

    try {
      const output = result.result;
      
      // Count comments added
      const addedMatches = output.match(/(?:added|created).*comment/gi);
      metrics.commentsAdded = addedMatches ? addedMatches.length : 0;
      
      // Count comments updated
      const updatedMatches = output.match(/(?:updated|modified).*comment/gi);
      metrics.commentsUpdated = updatedMatches ? updatedMatches.length : 0;
      
      // Count documented functions
      const functionMatches = output.match(/@param|@returns|@throws|def\s+\w+|function\s+\w+/gi);
      metrics.functionsDocumented = functionMatches ? functionMatches.length : 0;
      
      // Count documented classes
      const classMatches = output.match(/@class|class\s+\w+/gi);
      metrics.classesDocumented = classMatches ? classMatches.length : 0;
      
      // Count examples
      const exampleMatches = output.match(/@example|Example:|```/gi);
      metrics.examplesAdded = exampleMatches ? exampleMatches.length : 0;
      
      // Detect documentation standard
      if (output.includes('@param') || output.includes('@returns')) {
        metrics.standard = 'jsdoc';
      } else if (output.includes('Args:') || output.includes('Returns:')) {
        metrics.standard = 'sphinx';
      } else if (output.includes('///')) {
        metrics.standard = 'rustdoc';
      }

    } catch (error) {
      logger.warn('Failed to extract comment metrics:', error);
    }

    return metrics;
  }

  /**
   * Add comprehensive JSDoc to functions
   */
  async addJSDoc(sourceCode, options = {}) {
    const task = {
      description: 'Add comprehensive JSDoc comments to functions and classes',
      type: 'jsdoc_generation',
      context: {
        sourceCode,
        options: {
          includeExamples: true,
          includeTypes: true,
          ...options
        }
      },
      requirements: 'Generate complete JSDoc with parameters, return values, and examples'
    };

    return await this.processTask(task);
  }

  /**
   * Add inline comments to complex code
   */
  async addInlineComments(sourceCode, complexityThreshold = 'medium') {
    const task = {
      description: 'Add inline comments to explain complex code logic',
      type: 'inline_docs',
      context: {
        sourceCode,
        complexityThreshold,
        explainAlgorithms: true,
        clarifyBusinessLogic: true
      },
      requirements: 'Add meaningful inline comments that explain WHY, not WHAT'
    };

    return await this.processTask(task);
  }

  /**
   * Document API endpoints and functions
   */
  async documentAPI(apiCode, includeExamples = true) {
    const task = {
      description: 'Document API endpoints with comprehensive comments',
      type: 'api_commenting',
      context: {
        apiCode,
        includeExamples,
        documentErrorCodes: true,
        includeUsagePatterns: true
      },
      requirements: 'Create complete API documentation with usage examples and error handling'
    };

    return await this.processTask(task);
  }

  /**
   * Review and improve existing comments
   */
  async reviewComments(sourceCode, reviewCriteria) {
    const task = {
      description: 'Review and improve existing code comments',
      type: 'comment_review',
      context: {
        sourceCode,
        reviewCriteria: {
          accuracy: true,
          completeness: true,
          clarity: true,
          consistency: true,
          ...reviewCriteria
        }
      },
      requirements: 'Improve comment quality and ensure they add value'
    };

    return await this.processTask(task);
  }

  /**
   * Generate code examples for documentation
   */
  async generateExamples(functionSignatures, usageContext) {
    const task = {
      description: 'Generate practical code examples for functions',
      type: 'example_generation',
      context: {
        functionSignatures,
        usageContext,
        includeEdgeCases: true,
        showErrorHandling: true
      },
      requirements: 'Create clear, practical examples that demonstrate proper usage'
    };

    return await this.processTask(task);
  }

  /**
   * Clean up and standardize existing comments
   */
  async cleanupComments(sourceCode, targetStandard) {
    const task = {
      description: 'Clean up and standardize existing comments',
      type: 'documentation_cleanup',
      context: {
        sourceCode,
        targetStandard,
        removeRedundant: true,
        fixFormatting: true
      },
      requirements: 'Standardize comment format and remove redundant or outdated comments'
    };

    return await this.processTask(task);
  }

  /**
   * Analyze code and suggest where comments are needed
   */
  async analyzeCommentNeeds(sourceCode, language) {
    const task = {
      description: 'Analyze code and identify areas needing comments',
      type: 'comment_analysis',
      context: {
        sourceCode,
        language,
        complexityAnalysis: true,
        publicAPIFocus: true
      },
      requirements: 'Identify specific areas where comments would add the most value'
    };

    return await this.processTask(task);
  }
}