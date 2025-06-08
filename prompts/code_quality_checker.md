# Code Quality Checker - NSA-Level Production Standards

This prompt is designed to ensure all code in the Open-Sourcefy project meets NSA-level production standards with zero security vulnerabilities and optimal maintainability.

## Comprehensive Code Quality Analysis

### Security Analysis
- [ ] **NO HARDCODED SECRETS**: Scan for API keys, passwords, tokens, or sensitive data
- [ ] **PATH VALIDATION**: All file paths must be validated against directory traversal
- [ ] **INPUT SANITIZATION**: All external inputs must be validated and sanitized
- [ ] **ERROR MESSAGE SECURITY**: No sensitive information in error messages
- [ ] **SQL INJECTION PREVENTION**: Parameterized queries only (if applicable)
- [ ] **COMMAND INJECTION PREVENTION**: No shell command construction with user input
- [ ] **XSS PREVENTION**: Output encoding for any web interfaces (if applicable)

### SOLID Principles Compliance
- [ ] **Single Responsibility**: Each class/function has exactly one reason to change
- [ ] **Open/Closed**: Code is open for extension, closed for modification
- [ ] **Liskov Substitution**: Subtypes must be substitutable for base types
- [ ] **Interface Segregation**: Clients shouldn't depend on interfaces they don't use
- [ ] **Dependency Inversion**: Depend on abstractions, not concretions

### Code Structure & Organization
- [ ] **Separation of Concerns**: UI logic separated from business logic
- [ ] **Configuration Externalization**: Zero hardcoded values in production code
- [ ] **Dependency Injection**: External dependencies injected, not hard-coded
- [ ] **Resource Management**: Proper cleanup with context managers
- [ ] **Error Handling**: Comprehensive with specific exception types
- [ ] **Logging**: Structured logging with appropriate levels

### Documentation Standards
- [ ] **Class Docstrings**: Comprehensive with purpose, attributes, examples
- [ ] **Function Docstrings**: Args, returns, raises, examples for all public functions
- [ ] **Type Hints**: Complete type annotations for all function signatures
- [ ] **Inline Comments**: Complex logic explained with clear comments
- [ ] **README**: Installation, usage, configuration instructions
- [ ] **API Documentation**: Auto-generated documentation for public APIs

### Performance & Scalability
- [ ] **Async/Await**: Proper async implementation where applicable
- [ ] **Memory Management**: No memory leaks, proper resource cleanup
- [ ] **Time Complexity**: Algorithms are optimally efficient
- [ ] **Space Complexity**: Memory usage is optimized
- [ ] **Caching**: Appropriate caching strategies implemented
- [ ] **Database Optimization**: Efficient queries and indexing (if applicable)

### Testing Coverage
- [ ] **Unit Tests**: >90% code coverage with meaningful tests
- [ ] **Integration Tests**: End-to-end workflow testing
- [ ] **Edge Cases**: Boundary conditions and error scenarios tested
- [ ] **Mock Testing**: External dependencies properly mocked
- [ ] **Performance Tests**: Load and stress testing for critical paths
- [ ] **Security Tests**: Penetration testing and vulnerability scanning

### Code Quality Metrics
- [ ] **Cyclomatic Complexity**: Functions have complexity < 10
- [ ] **Function Length**: Functions are < 50 lines (with exceptions documented)
- [ ] **Class Size**: Classes are focused and < 500 lines
- [ ] **Parameter Count**: Functions have < 5 parameters
- [ ] **Nesting Depth**: Maximum nesting depth < 4 levels
- [ ] **Code Duplication**: Zero code duplication (DRY principle)

### Python-Specific Quality Checks
- [ ] **PEP 8 Compliance**: Code follows Python style guidelines
- [ ] **Import Organization**: Imports properly organized and minimal
- [ ] **Variable Naming**: Clear, descriptive variable names
- [ ] **Function Naming**: Verbs for functions, nouns for variables
- [ ] **Class Naming**: PascalCase for classes, snake_case for functions
- [ ] **Constants**: UPPER_CASE for constants

### Configuration Management
- [ ] **Environment Variables**: All configuration via env vars or config files
- [ ] **Configuration Validation**: All config values validated at startup
- [ ] **Default Values**: Sensible defaults for all configuration options
- [ ] **Configuration Documentation**: All config options documented
- [ ] **Secret Management**: Secrets managed securely (never in code)

## Code Quality Prompt

Use this prompt to analyze any Python code file:

```
Analyze this Python code for NSA-level production quality. Check for:

1. SECURITY ISSUES (Critical - Zero Tolerance):
   - Hardcoded secrets, API keys, passwords
   - Path traversal vulnerabilities
   - Input validation failures
   - Command/SQL injection risks
   - Information disclosure in errors

2. SOLID PRINCIPLES VIOLATIONS:
   - Single Responsibility violations
   - Hardcoded dependencies
   - Interface segregation issues
   - Inheritance problems

3. CODE QUALITY ISSUES:
   - Missing type hints or docstrings
   - Hardcoded values that should be configurable
   - Poor error handling or overly broad exceptions
   - Resource leaks (files, connections not closed)
   - Complex functions (>50 lines or cyclomatic complexity >10)

4. PERFORMANCE ISSUES:
   - Memory leaks or inefficient algorithms
   - Missing async/await where appropriate
   - Unnecessary loops or operations

5. MAINTAINABILITY ISSUES:
   - Code duplication (DRY violations)
   - Poor naming conventions
   - Missing or inadequate documentation
   - Complex nested logic

For each issue found:
- Explain WHY it's a problem
- Provide a SPECIFIC fix with example code
- Rate severity: CRITICAL (security), HIGH (functionality), MEDIUM (maintainability), LOW (style)

Focus on production-ready, secure, maintainable code that meets enterprise standards.
```

## Automated Quality Checks

### Pre-Commit Hooks
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Security scanning
bandit -r src/ -f json -o security_report.json
if [ $? -ne 0 ]; then
    echo "❌ Security scan failed - commit blocked"
    exit 1
fi

# Code formatting
black --check src/
if [ $? -ne 0 ]; then
    echo "❌ Code formatting check failed - run 'black src/' to fix"
    exit 1
fi

# Type checking
mypy src/
if [ $? -ne 0 ]; then
    echo "❌ Type checking failed - fix type errors"
    exit 1
fi

# Linting
flake8 src/ --max-complexity=10 --max-line-length=100
if [ $? -ne 0 ]; then
    echo "❌ Linting failed - fix code quality issues"
    exit 1
fi

# Unit tests
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=90
if [ $? -ne 0 ]; then
    echo "❌ Tests failed or coverage below 90%"
    exit 1
fi

echo "✅ All quality checks passed"
```

### Quality Gates
```yaml
# .github/workflows/quality-gates.yml
name: Quality Gates

on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Security Scan
        run: |
          pip install bandit safety
          bandit -r src/ -f json
          safety check --json

  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Code Quality
        run: |
          pip install black flake8 mypy pytest coverage
          black --check src/
          flake8 src/ --max-complexity=10
          mypy src/
          pytest tests/ --cov=src --cov-fail-under=90

  performance-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Performance Testing
        run: |
          pip install pytest-benchmark
          pytest tests/performance/ --benchmark-only
```

## Manual Review Checklist

### Code Review Template
```markdown
## Security Review
- [ ] No hardcoded secrets or sensitive data
- [ ] All inputs validated and sanitized
- [ ] Error handling doesn't leak information
- [ ] File paths validated against traversal

## Architecture Review
- [ ] SOLID principles followed
- [ ] Configuration externalized
- [ ] Dependencies properly injected
- [ ] Resource management implemented

## Quality Review
- [ ] All functions have type hints and docstrings
- [ ] Error handling is comprehensive
- [ ] Code is DRY (no duplication)
- [ ] Performance is optimal

## Testing Review
- [ ] Unit tests cover all code paths
- [ ] Edge cases and error scenarios tested
- [ ] Integration tests validate workflows
- [ ] Performance tests validate scalability

## Documentation Review
- [ ] README is complete and accurate
- [ ] API documentation is auto-generated
- [ ] Configuration options documented
- [ ] Deployment instructions provided
```

## Quality Metrics Dashboard

Track these metrics for continuous improvement:

### Security Metrics
- Zero critical vulnerabilities
- Zero secrets in code
- 100% input validation coverage

### Code Quality Metrics
- >95% test coverage
- <10 cyclomatic complexity average
- <5% code duplication
- 100% type hint coverage

### Performance Metrics
- <500ms response times
- <100MB memory usage
- <10% CPU utilization

### Maintainability Metrics
- 100% documentation coverage
- <24 hours mean time to fix bugs
- <50 lines average function length

This comprehensive quality framework ensures the Open-Sourcefy codebase maintains NSA-level production standards with zero tolerance for security vulnerabilities and optimal maintainability.