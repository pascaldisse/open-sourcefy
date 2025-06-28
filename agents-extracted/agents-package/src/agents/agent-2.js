import { AgentBase } from '../core/agent-base.js';
import { logger } from '../utils/logger.js';

/**
 * Agent 2: Documentation Specialist
 * Specializes in technical writing and documentation management
 */
export class Agent2 extends AgentBase {
  constructor() {
    super(
      2,
      'Documentation Specialist',
      'Technical writing and documentation management specialist',
      [
        'Generate API documentation from code',
        'Create user guides and tutorials',
        'Maintain architectural documentation',
        'Update changelog and release notes',
        'Ensure documentation accuracy and completeness',
        'Create inline code documentation',
        'Generate README files and project documentation',
        'Maintain documentation standards and templates'
      ],
      [
        'Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS'
      ]
    );

    this.documentationFormats = [
      'markdown', 'html', 'pdf', 'docx', 'confluence', 'notion'
    ];
    this.docTools = [
      'jsdoc', 'typedoc', 'sphinx', 'gitbook', 'docusaurus', 'vuepress'
    ];
    this.templateTypes = [
      'api_reference', 'user_guide', 'tutorial', 'changelog', 
      'readme', 'architecture', 'installation', 'troubleshooting'
    ];
  }

  /**
   * Override task type validation for documentation specialist
   */
  canHandleTaskType(taskType) {
    const documentationTasks = [
      'documentation',
      'api_docs',
      'user_guide',
      'tutorial',
      'readme',
      'changelog',
      'release_notes',
      'architecture_docs',
      'code_comments',
      'inline_docs',
      'doc_generation',
      'doc_review',
      'doc_standards'
    ];
    
    return documentationTasks.includes(taskType);
  }

  /**
   * Generate system prompt specific to documentation specialist role
   */
  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Documentation Specialist, you have specific expertise in:

DOCUMENTATION TYPES:
- API Reference: Complete endpoint documentation with examples
- User Guides: Step-by-step instructions for end users
- Developer Docs: Technical documentation for developers
- Architecture Docs: System design and component relationships
- Tutorials: Learn-by-doing educational content
- README Files: Project overview and getting started guides

DOCUMENTATION STANDARDS:
- Clear, concise, and accurate writing
- Consistent formatting and structure
- Include code examples and use cases
- Maintain up-to-date information
- Use proper headings and navigation
- Include diagrams and visual aids when helpful

TECHNICAL WRITING BEST PRACTICES:
- Write for your audience (users vs developers)
- Use active voice and present tense
- Include prerequisites and assumptions
- Provide troubleshooting sections
- Add cross-references and links
- Maintain version control for docs

DOCUMENTATION TOOLS:
- JSDoc for JavaScript/TypeScript
- Markdown for general documentation
- OpenAPI/Swagger for API docs
- Mermaid for diagrams
- Static site generators (Docusaurus, GitBook)

SPECIAL INSTRUCTIONS:
- Always verify code examples work correctly
- Update related documentation when making changes
- Maintain consistent terminology throughout
- Include changelog entries for significant updates
- Ensure documentation is accessible and searchable
- Review existing docs before creating new ones

When creating documentation:
1. Analyze the target audience and their needs
2. Structure content logically with clear navigation
3. Include practical examples and use cases
4. Verify all code samples are functional
5. Add appropriate metadata and tags
6. Ensure consistency with existing documentation style`;
  }

  /**
   * Execute documentation-specific tasks
   */
  async processTask(task) {
    logger.info(`Documentation Specialist processing ${task.type} task: ${task.description}`);

    // Pre-process task based on type
    const enhancedTask = await this.enhanceDocumentationTask(task);
    
    // Execute using base class with enhanced context
    const result = await super.processTask(enhancedTask);

    // Post-process results for documentation tasks
    if (result.success) {
      result.docMetrics = await this.extractDocumentationMetrics(result);
    }

    return result;
  }

  /**
   * Enhance task with documentation-specific context
   */
  async enhanceDocumentationTask(task) {
    const enhanced = { ...task };

    // Add documentation context based on task type
    switch (task.type) {
      case 'api_docs':
        enhanced.context = {
          ...enhanced.context,
          format: 'markdown',
          includeExamples: true,
          includeResponseSchemas: true,
          includeErrorCodes: true,
          tool: 'openapi'
        };
        break;

      case 'user_guide':
        enhanced.context = {
          ...enhanced.context,
          format: 'markdown',
          audience: 'end_users',
          includeScreenshots: true,
          stepByStep: true,
          troubleshooting: true
        };
        break;

      case 'readme':
        enhanced.context = {
          ...enhanced.context,
          format: 'markdown',
          sections: [
            'description', 'installation', 'usage', 
            'examples', 'contributing', 'license'
          ],
          includeBadges: true,
          includeQuickStart: true
        };
        break;

      case 'changelog':
        enhanced.context = {
          ...enhanced.context,
          format: 'markdown',
          standard: 'keep-a-changelog',
          categories: ['Added', 'Changed', 'Deprecated', 'Removed', 'Fixed', 'Security'],
          includeLinks: true
        };
        break;

      case 'architecture_docs':
        enhanced.context = {
          ...enhanced.context,
          format: 'markdown',
          includeDiagrams: true,
          sections: [
            'overview', 'components', 'data_flow', 
            'deployment', 'security', 'scalability'
          ],
          diagramTool: 'mermaid'
        };
        break;
    }

    // Add documentation-specific requirements
    enhanced.requirements = `${enhanced.requirements || ''}

DOCUMENTATION REQUIREMENTS:
- Use ${enhanced.context.format || 'markdown'} format
- Follow project documentation standards
- Include practical examples and code samples
- Ensure all links and references are valid
- Maintain consistent tone and style
- Add appropriate metadata and frontmatter
- Verify accuracy of technical details
- Include table of contents for long documents
- Use proper heading hierarchy (H1, H2, H3...)
- Add cross-references to related documentation`;

    return enhanced;
  }

  /**
   * Extract documentation metrics from task results
   */
  async extractDocumentationMetrics(result) {
    const metrics = {
      wordCount: 0,
      sectionCount: 0,
      codeBlockCount: 0,
      linkCount: 0,
      imageCount: 0,
      readingTime: 0,
      format: 'unknown'
    };

    try {
      const content = result.result;
      
      // Word count
      metrics.wordCount = content.split(/\s+/).length;
      
      // Reading time (average 200 words per minute)
      metrics.readingTime = Math.ceil(metrics.wordCount / 200);
      
      // Section count (headers)
      const headerMatches = content.match(/^#{1,6}\s+/gm);
      metrics.sectionCount = headerMatches ? headerMatches.length : 0;
      
      // Code block count
      const codeBlockMatches = content.match(/```[\s\S]*?```/g);
      metrics.codeBlockCount = codeBlockMatches ? codeBlockMatches.length : 0;
      
      // Link count
      const linkMatches = content.match(/\[.*?\]\(.*?\)/g);
      metrics.linkCount = linkMatches ? linkMatches.length : 0;
      
      // Image count
      const imageMatches = content.match(/!\[.*?\]\(.*?\)/g);
      metrics.imageCount = imageMatches ? imageMatches.length : 0;
      
      // Detect format
      if (content.includes('```') || content.includes('#')) {
        metrics.format = 'markdown';
      } else if (content.includes('<html>') || content.includes('<div>')) {
        metrics.format = 'html';
      }

    } catch (error) {
      logger.warn('Failed to extract documentation metrics:', error);
    }

    return metrics;
  }

  /**
   * Generate API documentation from code
   */
  async generateApiDocs(sourceCode, options = {}) {
    const task = {
      description: 'Generate comprehensive API documentation from source code',
      type: 'api_docs',
      context: {
        sourceCode,
        options: {
          includeExamples: true,
          includeSchemas: true,
          format: 'openapi',
          ...options
        }
      },
      requirements: 'Create complete API reference with endpoints, parameters, responses, and examples'
    };

    return await this.processTask(task);
  }

  /**
   * Create user guide
   */
  async createUserGuide(product, features, audience = 'general') {
    const task = {
      description: `Create comprehensive user guide for ${product}`,
      type: 'user_guide',
      context: {
        product,
        features,
        audience,
        includeScreenshots: true,
        stepByStep: true
      },
      requirements: 'Create easy-to-follow user guide with clear instructions and visual aids'
    };

    return await this.processTask(task);
  }

  /**
   * Generate README file
   */
  async generateReadme(projectInfo) {
    const task = {
      description: 'Generate comprehensive README file for project',
      type: 'readme',
      context: {
        projectInfo,
        includeQuickStart: true,
        includeBadges: true,
        includeContributing: true
      },
      requirements: 'Create professional README with all essential project information'
    };

    return await this.processTask(task);
  }

  /**
   * Update changelog
   */
  async updateChangelog(version, changes, releaseDate) {
    const task = {
      description: `Update changelog for version ${version}`,
      type: 'changelog',
      context: {
        version,
        changes,
        releaseDate,
        standard: 'keep-a-changelog'
      },
      requirements: 'Update changelog following semantic versioning and standard format'
    };

    return await this.processTask(task);
  }

  /**
   * Create architecture documentation
   */
  async createArchitectureDocs(systemDesign) {
    const task = {
      description: 'Create comprehensive architecture documentation',
      type: 'architecture_docs',
      context: {
        systemDesign,
        includeDiagrams: true,
        sections: [
          'overview', 'components', 'data_flow', 
          'deployment', 'security', 'scalability'
        ]
      },
      requirements: 'Create detailed architecture documentation with diagrams and component descriptions'
    };

    return await this.processTask(task);
  }

  /**
   * Review and improve existing documentation
   */
  async reviewDocumentation(documentPath, criteria) {
    const task = {
      description: 'Review and improve existing documentation',
      type: 'doc_review',
      context: {
        documentPath,
        criteria: {
          accuracy: true,
          completeness: true,
          clarity: true,
          consistency: true,
          ...criteria
        }
      },
      requirements: 'Provide detailed review with specific improvement recommendations'
    };

    return await this.processTask(task);
  }

  /**
   * Validate documentation links and references
   */
  async validateDocumentation(docPath) {
    const task = {
      description: 'Validate documentation links, references, and accuracy',
      type: 'doc_validation',
      context: {
        docPath,
        checkLinks: true,
        checkReferences: true,
        checkCodeExamples: true
      },
      requirements: 'Verify all links work and code examples are functional'
    };

    return await this.processTask(task);
  }
}