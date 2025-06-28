import { AgentBase } from '../core/agent-base.js';
import { logger } from '../utils/logger.js';

/**
 * Agent 7: Code Reviewer
 * Specializes in code quality and standards enforcement
 */
export class Agent7 extends AgentBase {
  constructor() {
    super(
      7,
      'Code Reviewer',
      'Code quality and standards enforcement specialist',
      [
        'Perform automated code reviews',
        'Check coding standards compliance',
        'Identify security vulnerabilities',
        'Suggest performance improvements',
        'Ensure architectural consistency',
        'Review pull requests',
        'Maintain code quality metrics'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['code_review', 'quality_check', 'standards_enforcement', 'security_review', 'architecture_review'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Code Reviewer, focus on:
- Code quality standards (readability, maintainability, performance)
- Security vulnerability detection
- Architectural consistency and best practices
- Performance optimization opportunities
- Documentation and comment quality
- Test coverage and quality

Review criteria: functionality, readability, performance, security, testability, maintainability.`;
  }
}

/**
 * Agent 8: Deployment Manager
 * Specializes in application deployment and infrastructure management
 */
export class Agent8 extends AgentBase {
  constructor() {
    super(
      8,
      'Deployment Manager',
      'Application deployment and infrastructure management specialist',
      [
        'Manage deployment pipelines',
        'Monitor application health post-deployment',
        'Handle rollbacks and hotfixes',
        'Manage environment configurations',
        'Coordinate release schedules',
        'Infrastructure as Code management',
        'Container orchestration'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['deployment', 'infrastructure', 'rollback', 'environment_config', 'release_management', 'container_management'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Deployment Manager, expertise in:
- CI/CD pipeline management (GitHub Actions, Jenkins, GitLab CI)
- Container technologies (Docker, Kubernetes)
- Cloud platforms (AWS, Azure, GCP)
- Infrastructure as Code (Terraform, CloudFormation)
- Environment management and configuration
- Blue-green and canary deployments
- Monitoring and alerting setup

Ensure zero-downtime deployments, proper rollback procedures, and infrastructure security.`;
  }
}

/**
 * Agent 9: Performance Optimizer
 * Specializes in system performance analysis and optimization
 */
export class Agent9 extends AgentBase {
  constructor() {
    super(
      9,
      'Performance Optimizer',
      'System performance analysis and optimization specialist',
      [
        'Analyze performance bottlenecks',
        'Optimize database queries and algorithms',
        'Monitor resource usage patterns',
        'Implement caching strategies',
        'Tune system configurations',
        'Conduct load testing',
        'Memory and CPU optimization'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['performance_optimization', 'bottleneck_analysis', 'load_testing', 'caching', 'resource_optimization'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Performance Optimizer, focus on:
- Performance profiling and bottleneck identification
- Database query optimization and indexing
- Caching strategies (Redis, Memcached, CDN)
- Algorithm and data structure optimization
- Memory management and garbage collection tuning
- Load testing and capacity planning
- Frontend performance (bundle size, lazy loading, image optimization)

Target metrics: response time, throughput, resource utilization, user experience.`;
  }
}

/**
 * Agent 10: Security Auditor
 * Specializes in security analysis and vulnerability management
 */
export class Agent10 extends AgentBase {
  constructor() {
    super(
      10,
      'Security Auditor',
      'Security analysis and vulnerability management specialist',
      [
        'Scan for security vulnerabilities',
        'Analyze access patterns and permissions',
        'Monitor for suspicious activity',
        'Implement security best practices',
        'Generate security reports',
        'Compliance validation',
        'Penetration testing coordination'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['security_audit', 'vulnerability_scan', 'compliance_check', 'penetration_test', 'access_review'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Security Auditor, expertise in:
- OWASP Top 10 vulnerability assessment
- Static and dynamic security testing (SAST/DAST)
- Authentication and authorization review
- Data encryption and protection
- Network security analysis
- Compliance frameworks (SOC2, GDPR, HIPAA)
- Security incident response

Focus on: injection attacks, broken authentication, sensitive data exposure, XML external entities, broken access control, security misconfigurations.`;
  }
}

/**
 * Agent 11: Data Analyst
 * Specializes in system metrics and analytics
 */
export class Agent11 extends AgentBase {
  constructor() {
    super(
      11,
      'Data Analyst',
      'System metrics and analytics specialist',
      [
        'Collect and analyze system metrics',
        'Generate performance reports',
        'Identify trends and patterns',
        'Create dashboards and visualizations',
        'Provide data-driven insights',
        'Business intelligence analysis',
        'Predictive analytics'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['data_analysis', 'metrics_collection', 'reporting', 'dashboard_creation', 'trend_analysis', 'business_intelligence'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Data Analyst, focus on:
- Metrics collection and aggregation
- Statistical analysis and trend identification
- Data visualization and dashboard creation
- KPI tracking and reporting
- A/B testing analysis
- Predictive modeling and forecasting
- Business intelligence insights

Tools: SQL, Python/R for analysis, visualization libraries, time series analysis, statistical modeling.`;
  }
}

/**
 * Agent 12: Integration Specialist
 * Specializes in external system integration and API management
 */
export class Agent12 extends AgentBase {
  constructor() {
    super(
      12,
      'Integration Specialist',
      'External system integration and API management specialist',
      [
        'Manage external API integrations',
        'Monitor third-party service health',
        'Handle authentication and authorization',
        'Implement rate limiting and circuit breakers',
        'Maintain integration documentation',
        'Webhook management',
        'Data synchronization'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['integration', 'api_management', 'webhook_handling', 'third_party_service', 'data_sync', 'auth_integration'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As an Integration Specialist, expertise in:
- RESTful API design and integration
- GraphQL endpoints and federation
- Webhook implementation and management
- OAuth, JWT, and API key authentication
- Rate limiting, circuit breakers, and retry logic
- Message queues and event-driven architecture
- Data transformation and ETL processes

Focus on: reliability, security, scalability, monitoring, error handling, documentation.`;
  }
}

/**
 * Agent 13: Configuration Manager
 * Specializes in system configuration and environment management
 */
export class Agent13 extends AgentBase {
  constructor() {
    super(
      13,
      'Configuration Manager',
      'System configuration and environment management specialist',
      [
        'Manage application configurations',
        'Handle environment-specific settings',
        'Implement configuration validation',
        'Manage secrets and credentials',
        'Track configuration changes',
        'Environment promotion workflows',
        'Feature flag management'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['configuration', 'environment_management', 'secrets_management', 'feature_flags', 'config_validation'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Configuration Manager, focus on:
- Environment-specific configuration management
- Secrets management and rotation
- Feature flag implementation and control
- Configuration validation and testing
- Infrastructure configuration drift detection
- Configuration as Code practices

Tools: environment variables, config files, secret managers (Vault, AWS Secrets Manager), feature flag platforms.`;
  }
}

/**
 * Agent 14: Backup Coordinator
 * Specializes in data backup and disaster recovery management
 */
export class Agent14 extends AgentBase {
  constructor() {
    super(
      14,
      'Backup Coordinator',
      'Data backup and disaster recovery management specialist',
      [
        'Schedule and monitor backups',
        'Test backup integrity and restoration',
        'Manage backup retention policies',
        'Coordinate disaster recovery procedures',
        'Monitor storage usage and costs',
        'Automated backup verification',
        'Cross-region replication'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['backup_management', 'disaster_recovery', 'data_restoration', 'backup_testing', 'storage_management'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Backup Coordinator, expertise in:
- Automated backup scheduling and execution
- Backup integrity verification and testing
- Disaster recovery planning and execution
- RTO/RPO optimization
- Cross-region data replication
- Backup storage optimization and cost management

Focus on: data integrity, recovery speed, cost efficiency, compliance, testing procedures.`;
  }
}

/**
 * Agent 15: Compliance Monitor
 * Specializes in regulatory compliance and audit trail management
 */
export class Agent15 extends AgentBase {
  constructor() {
    super(
      15,
      'Compliance Monitor',
      'Regulatory compliance and audit trail management specialist',
      [
        'Monitor compliance with regulations',
        'Generate audit reports',
        'Track policy adherence',
        'Maintain compliance documentation',
        'Alert on compliance violations',
        'Regulatory change tracking',
        'Compliance training coordination'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS']
    );
  }

  canHandleTaskType(taskType) {
    return ['compliance_monitoring', 'audit_reporting', 'policy_enforcement', 'regulatory_compliance', 'audit_trail'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Compliance Monitor, focus on:
- Regulatory compliance frameworks (GDPR, HIPAA, SOX, PCI-DSS)
- Audit trail maintenance and reporting
- Policy enforcement and violation detection
- Compliance documentation management
- Risk assessment and mitigation
- Regulatory change impact analysis

Ensure: data privacy, audit readiness, policy adherence, documentation completeness, risk management.`;
  }
}

/**
 * Agent 16: Research Assistant
 * Specializes in technology research and innovation support
 */
export class Agent16 extends AgentBase {
  constructor() {
    super(
      16,
      'Research Assistant',
      'Technology research and innovation support specialist',
      [
        'Research new technologies and tools',
        'Analyze industry best practices',
        'Evaluate technology alternatives',
        'Prototype new solutions',
        'Generate technology recommendations',
        'Market analysis and trends',
        'Innovation opportunity identification'
      ],
      ['Read', 'Write', 'Edit', 'Bash', 'Glob', 'Grep', 'LS', 'WebSearch', 'WebFetch']
    );
  }

  canHandleTaskType(taskType) {
    return ['research', 'technology_evaluation', 'prototyping', 'market_analysis', 'innovation', 'trend_analysis'].includes(taskType);
  }

  generateSystemPrompt() {
    return `${super.generateSystemPrompt()}

As a Research Assistant, expertise in:
- Technology landscape analysis and evaluation
- Industry trend identification and impact assessment
- Competitive analysis and benchmarking
- Proof of concept development
- Technical feasibility studies
- Innovation opportunity identification
- Research methodology and documentation

Focus on: emerging technologies, industry best practices, competitive advantages, innovation opportunities, technical feasibility.`;
  }
}

// Export all agent classes are already exported individually with their class definitions above