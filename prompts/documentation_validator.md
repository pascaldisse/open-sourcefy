# Documentation Validation and Source Code Verification Prompt

## 🚨 MANDATORY FIRST STEP 🚨
**READ /mnt/c/Users/pascaldisse/Downloads/open-sourcefy/rules.md FIRST AND ENFORCE ALL RULES**

Before performing ANY work on this project, you MUST:
1. Read and understand the complete rules.md file
2. Apply ZERO TOLERANCE enforcement for all rules
3. Follow STRICT MODE ONLY - no fallbacks, no alternatives
4. Ensure NO MOCK IMPLEMENTATIONS - only real code
5. Maintain NSA-LEVEL SECURITY standards throughout

## CRITICAL RULES COMPLIANCE
- **NO FALLBACKS EVER** - one correct way only
- **STRICT MODE ONLY** - fail fast on missing requirements  
- **NO MOCK IMPLEMENTATIONS** - real implementations only
- **NO HARDCODED VALUES** - all values from configuration
- **NSA-LEVEL SECURITY** - zero tolerance for vulnerabilities

## Mission Objective

Create a comprehensive documentation validation system that:
1. **Validates every claim** in documentation against actual source code
2. **Corrects or removes** false documentation automatically
3. **Enforces rules.md compliance** in all documentation
4. **Provides source code references** for all documented features
5. **Maintains documentation accuracy** through automated verification

## Core Requirements

### 1. Documentation Fact-Checking Engine

Create a source code verification system that validates:

#### **Feature Claims Validation**
```python
def validate_feature_claim(claim: str, docs_path: str) -> ValidationResult:
    """
    Validates if documented feature actually exists in source code
    
    Args:
        claim: Feature description from documentation
        docs_path: Path to documentation file
        
    Returns:
        ValidationResult with:
        - exists: bool (feature found in code)
        - source_files: List[str] (files containing implementation)
        - line_references: List[int] (specific line numbers)
        - confidence: float (0.0-1.0 accuracy confidence)
        - evidence: str (code snippets proving existence)
    """
```

#### **API Documentation Verification**
- Verify all documented functions/classes exist
- Validate parameter types and names match source
- Check return types and documentation accuracy
- Confirm exception types are correctly documented

#### **Configuration Claims Verification**
- Validate all documented config options exist in config files
- Check default values match documentation
- Verify environment variables are actually used
- Confirm path configurations are accurate

#### **Architecture Claims Verification**
- Validate agent dependency chains match documentation
- Verify pipeline execution order is accurate
- Check agent implementation status claims
- Confirm Matrix character assignments

### 2. Source Code Reference Generator

#### **Automatic Citation System**
```python
def generate_source_references(feature: str) -> SourceReference:
    """
    Generates accurate source code references for documentation
    
    Returns:
        SourceReference with:
        - file_path: str (relative path from project root)
        - line_number: int (specific implementation line)
        - function_name: str (containing function/method)
        - class_name: str (containing class if applicable)
        - code_snippet: str (relevant code excerpt)
        - last_verified: datetime (when reference was validated)
    """
```

#### **Evidence Collection**
- Extract actual code snippets as proof
- Generate file:line references for all claims
- Create implementation status reports
- Document actual vs claimed functionality

### 3. Documentation Correction Engine

#### **Automatic Fact Correction**
```python
def correct_documentation_claim(
    docs_file: str, 
    claim: str, 
    actual_status: str
) -> CorrectionResult:
    """
    Automatically corrects false documentation claims
    
    Actions:
    - Replace incorrect information with accurate data
    - Add source code references for verification
    - Remove claims that cannot be substantiated
    - Update implementation status accurately
    """
```

#### **Removal of False Claims**
- Delete documentation for non-existent features
- Remove outdated implementation status
- Clear incorrect API documentation
- Eliminate misleading architecture diagrams

### 4. Real-Time Validation System

#### **Documentation Quality Gates**
```python
def validate_documentation_accuracy() -> ValidationReport:
    """
    Comprehensive documentation validation
    
    Checks:
    - All code references are valid and current
    - Implementation status matches reality
    - API documentation reflects actual interfaces
    - Configuration examples are accurate
    - Architecture diagrams match implementation
    
    Returns:
        ValidationReport with:
        - accuracy_score: float (0.0-1.0)
        - false_claims: List[Claim]
        - missing_documentation: List[Feature]
        - outdated_references: List[Reference]
        - correction_suggestions: List[Correction]
    """
```

#### **Continuous Verification**
- Validate documentation on every commit
- Check references when source code changes
- Update documentation automatically when possible
- Flag outdated documentation for review

## Target Documentation Files

### Priority 1: Core Documentation
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/README.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/CLAUDE.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/prompts/CLAUDE.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/tasks.md`

### Priority 2: Technical Documentation
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/docs/SYSTEM_ARCHITECTURE.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/docs/AGENT_REFACTOR_SPECIFICATIONS.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/docs/PRODUCTION_DEPLOYMENT_STRATEGY.md`

### Priority 3: Agent Documentation
- All agent-specific documentation in `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/src/core/agents/`
- Agent capability claims in various documentation files

## Validation Categories

### 1. Implementation Status Claims
**Validate**: Agent implementation completeness
```python
# Example validation
claim = "Agent 5: Neo (Advanced Decompilation) - ✅ IMPLEMENTED"
actual = check_agent_implementation(5)
if actual.status != "IMPLEMENTED":
    correct_claim(claim, actual.status, actual.evidence)
```

### 2. Feature Capability Claims
**Validate**: Actual feature functionality
```python
# Example validation
claim = "Ghidra Integration: Automated headless decompilation"
actual = check_ghidra_integration()
if not actual.automated or not actual.headless:
    correct_feature_claim(claim, actual.capabilities)
```

### 3. Configuration Claims
**Validate**: Configuration options and defaults
```python
# Example validation
claim = "Default timeout: 300 seconds"
actual = get_config_default("timeout")
if actual != 300:
    update_config_documentation(claim, actual)
```

### 4. API Documentation Claims
**Validate**: Function signatures and behavior
```python
# Example validation
claim = "execute_matrix_task(context: Dict[str, Any]) -> Dict[str, Any]"
actual = inspect_function_signature("execute_matrix_task")
if actual != claim:
    update_api_documentation(claim, actual)
```

## Output Requirements

### 1. Validation Reports
Generate comprehensive reports including:
- **Accuracy Score**: Overall documentation accuracy (0.0-1.0)
- **False Claims**: List of incorrect statements with evidence
- **Missing References**: Claims without source code backing
- **Outdated Information**: Documentation that no longer matches reality
- **Correction Actions**: Specific fixes applied

### 2. Source Code Evidence
For every validated claim, provide:
- **File Reference**: Exact file path and line number
- **Code Snippet**: Actual implementation code
- **Implementation Status**: Current working status
- **Last Verified**: Timestamp of validation

### 3. Corrected Documentation
Automatically generate:
- **Updated Documentation**: Corrected versions of all files
- **Source References**: Embedded file:line citations
- **Implementation Notes**: Actual status vs documentation claims
- **Evidence Links**: Direct links to supporting code

## Rules.md Compliance Enforcement

### 1. No Mock Documentation
- Remove any documentation describing mock implementations
- Delete references to fallback systems
- Eliminate placeholder or stub documentation

### 2. Strict Mode Documentation
- Document only working, validated functionality
- Remove any documentation for degraded operation modes
- Ensure all documented features fail fast when missing dependencies

### 3. NSA-Level Accuracy
- Zero tolerance for false documentation
- Every claim must be backed by source code evidence
- Comprehensive validation of all documented features

### 4. Real Implementation Focus
- Document only actual, working implementations
- Remove any references to planned or theoretical features
- Focus documentation on production-ready functionality

## Success Metrics

### Documentation Quality Targets
- **100% Accuracy**: All claims backed by source code evidence
- **Zero False Claims**: No documentation without implementation backing
- **Complete Coverage**: All implemented features documented
- **Current References**: All file:line references valid and current

### Validation Effectiveness
- **Real-time Validation**: Documentation validated on every change
- **Automatic Correction**: False claims corrected automatically
- **Evidence-Based**: All documentation supported by code evidence
- **Rules Compliance**: 100% adherence to rules.md requirements

## Implementation Strategy

### Phase 1: Validation Engine
1. Build source code scanning system
2. Create claim verification algorithms
3. Implement evidence collection system
4. Develop accuracy scoring methodology

### Phase 2: Correction System
1. Build automatic documentation correction
2. Implement false claim removal
3. Create source reference generation
4. Develop real-time validation hooks

### Phase 3: Quality Assurance
1. Validate all existing documentation
2. Correct false claims and outdated information
3. Add source code references throughout
4. Implement continuous validation system

This prompt ensures documentation accuracy through automated source code verification while maintaining strict compliance with rules.md requirements for real, working implementations only.