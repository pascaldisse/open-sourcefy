# Link Validation and Dead Link Elimination Prompt

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

Create a comprehensive link validation system that:
1. **Scans all documentation** for internal and external links
2. **Validates link accessibility** and correctness
3. **Fixes broken links** automatically where possible
4. **Removes dead links** that cannot be repaired
5. **Maintains link health** through automated monitoring

## Core Requirements

### 1. Comprehensive Link Discovery Engine

#### **Multi-Format Link Detection**
```python
def discover_all_links(file_path: str) -> List[LinkReference]:
    """
    Discovers all types of links in documentation files
    
    Detects:
    - Markdown links: [text](url)
    - HTML links: <a href="url">text</a>
    - Direct URLs: http://example.com
    - File references: ./path/to/file.ext
    - Internal anchors: #section-name
    - Relative paths: ../other/file.md
    
    Returns:
        List of LinkReference objects with:
        - url: str (the actual link)
        - text: str (display text)
        - file_path: str (containing file)
        - line_number: int (location in file)
        - link_type: LinkType (internal/external/file/anchor)
    """
```

#### **Link Classification System**
```python
class LinkType(Enum):
    EXTERNAL_HTTP = "external_http"      # https://example.com
    EXTERNAL_FTP = "external_ftp"        # ftp://files.example.com
    INTERNAL_FILE = "internal_file"      # ./src/main.py
    INTERNAL_ANCHOR = "internal_anchor"  # #section-header
    RELATIVE_PATH = "relative_path"      # ../docs/file.md
    ABSOLUTE_PATH = "absolute_path"      # /full/path/to/file
    EMAIL_LINK = "email"                 # mailto:user@domain.com
    PROTOCOL_OTHER = "protocol_other"    # git://, ssh://, etc.
```

### 2. Link Validation Engine

#### **External Link Validation**
```python
def validate_external_link(url: str) -> ValidationResult:
    """
    Validates external links with comprehensive checking
    
    Validation includes:
    - HTTP/HTTPS accessibility (status codes)
    - SSL certificate validity
    - Response time measurement
    - Content type verification
    - Redirect chain analysis
    - Domain reputation checking
    
    Returns:
        ValidationResult with:
        - is_valid: bool
        - status_code: int
        - response_time: float
        - final_url: str (after redirects)
        - error_message: str
        - security_warnings: List[str]
    """
```

#### **Internal Link Validation**
```python
def validate_internal_link(
    link_path: str, 
    source_file: str
) -> InternalValidationResult:
    """
    Validates internal file references and anchors
    
    Validation includes:
    - File existence verification
    - Path resolution accuracy
    - Anchor/section existence in target files
    - Permission accessibility
    - Case sensitivity issues
    
    Returns:
        InternalValidationResult with:
        - exists: bool
        - resolved_path: str
        - anchor_found: bool (for anchor links)
        - permissions_ok: bool
        - suggestions: List[str] (possible corrections)
    """
```

### 3. Automated Link Repair System

#### **Intelligent Link Correction**
```python
def attempt_link_repair(
    broken_link: LinkReference
) -> RepairResult:
    """
    Attempts to automatically repair broken links
    
    Repair strategies:
    - Archive.org Wayback Machine lookup
    - Alternative URL pattern matching
    - File path fuzzy matching for moved files
    - Anchor name similarity matching
    - Domain redirection following
    
    Returns:
        RepairResult with:
        - repair_possible: bool
        - suggested_url: str
        - confidence: float (0.0-1.0)
        - repair_method: str
        - verification_needed: bool
    """
```

#### **Smart Link Replacement**
```python
def replace_broken_link(
    file_path: str,
    broken_link: LinkReference,
    replacement: str
) -> ReplacementResult:
    """
    Safely replaces broken links in documentation
    
    Features:
    - Preserves original display text
    - Maintains markdown/HTML formatting
    - Creates backup before modification
    - Validates replacement link
    - Logs all changes for review
    """
```

### 4. Dead Link Removal System

#### **Safe Link Removal**
```python
def remove_dead_link(
    file_path: str,
    dead_link: LinkReference
) -> RemovalResult:
    """
    Safely removes dead links from documentation
    
    Removal strategies:
    - Convert link to plain text (preserve information)
    - Remove entire sentence if link-dependent
    - Replace with archive.org snapshot if available
    - Add deprecation note explaining removal
    
    Returns:
        RemovalResult with:
        - removal_method: str
        - preserved_text: str
        - backup_created: bool
        - manual_review_needed: bool
    """
```

#### **Context-Aware Removal**
```python
def analyze_link_context(
    file_content: str,
    link: LinkReference
) -> ContextAnalysis:
    """
    Analyzes context around dead links for intelligent removal
    
    Analysis includes:
    - Sentence dependency on link
    - Information value without link
    - Alternative information sources
    - Impact on document flow
    
    Returns decision on removal strategy
    """
```

## Target Documentation Scan Areas

### Priority 1: Core Project Documentation
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/README.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/CLAUDE.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/rules.md`

### Priority 2: Technical Documentation
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/docs/*.md`
- `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/prompts/*.md`

### Priority 3: Agent Documentation
- All documentation files in `/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/src/core/agents/`
- Any embedded documentation in Python files

### Priority 4: Configuration and Setup
- All configuration files with external references
- Setup and installation documentation
- Environment configuration guides

## Validation Categories

### 1. External Resource Links
**Validate**: All HTTP/HTTPS links to external resources
```python
# Example validation targets
external_links = [
    "https://github.com/NationalSecurityAgency/ghidra",
    "https://docs.microsoft.com/en-us/windows/",
    "https://python.org",
    "https://cmake.org"
]
```

### 2. Internal File References
**Validate**: All internal project file links
```python
# Example validation targets
internal_links = [
    "./src/core/agents/agent01_sentinel.py",
    "../docs/SYSTEM_ARCHITECTURE.md",
    "/output/reports/analysis.json"
]
```

### 3. Documentation Cross-References
**Validate**: Links between documentation files
```python
# Example validation targets
doc_cross_refs = [
    "[Architecture](docs/SYSTEM_ARCHITECTURE.md)",
    "[Agent Specifications](docs/AGENT_REFACTOR_SPECIFICATIONS.md)",
    "[Installation Guide](README.md#installation)"
]
```

### 4. Tool and Dependency Links
**Validate**: Links to required tools and dependencies
```python
# Example validation targets
tool_links = [
    "https://visualstudio.microsoft.com/downloads/",
    "https://www.python.org/downloads/",
    "https://git-scm.com/downloads"
]
```

## Security and Safety Requirements

### 1. NSA-Level Security Validation
```python
def validate_link_security(url: str) -> SecurityResult:
    """
    Performs comprehensive security validation of links
    
    Security checks:
    - SSL certificate validation
    - Domain reputation analysis
    - Malware scanning integration
    - Phishing detection
    - Certificate transparency logging
    
    Returns:
        SecurityResult with:
        - is_secure: bool
        - security_score: float (0.0-1.0)
        - security_warnings: List[str]
        - recommended_action: str
    """
```

### 2. Safe Link Processing
- **Never follow redirects** to untrusted domains
- **Validate SSL certificates** for all HTTPS links
- **Check domain reputation** before validation
- **Sandbox external requests** for security
- **Log all external communications** for audit

### 3. Privacy Protection
- **No tracking parameters** in saved links
- **Remove analytics tokens** from URLs
- **Prefer canonical URLs** over shortened links
- **Avoid privacy-invasive services** in link checking

## Output Requirements

### 1. Comprehensive Link Health Report
```python
class LinkHealthReport:
    total_links_found: int
    valid_links: int
    broken_links: int
    repaired_links: int
    removed_links: int
    security_warnings: List[str]
    manual_review_needed: List[LinkReference]
    processing_time: float
    accuracy_confidence: float
```

### 2. Detailed Validation Results
For each link, provide:
- **Validation Status**: Valid/Broken/Repaired/Removed
- **Location**: File path and line number
- **Error Details**: Specific reason for failure
- **Repair Action**: Action taken or recommended
- **Security Assessment**: Security score and warnings

### 3. Change Documentation
Generate comprehensive change logs:
- **Links Repaired**: Old URL → New URL with reasoning
- **Links Removed**: Removed links with context explanation
- **Security Issues**: Any security concerns found
- **Manual Review Items**: Links requiring human attention

## Rules.md Compliance Enforcement

### 1. Strict Validation Only
- **NO FALLBACK** link checking methods
- **FAIL FAST** on validation tool failures
- **NO MOCK** link validation - real HTTP requests only
- **REAL TOOLS ONLY** - authentic link validation

### 2. NSA-Level Security
- **ZERO TOLERANCE** for insecure links
- **COMPREHENSIVE** security validation for all external links
- **IMMEDIATE REMOVAL** of security-risk links
- **AUDIT TRAIL** for all link modifications

### 3. Configuration-Based Operation
- **NO HARDCODED** timeout values - use configuration
- **NO HARDCODED** user agents or request headers
- **CONFIGURABLE** validation rules and security policies
- **EXTERNAL CONFIG** for all validation parameters

## Success Metrics

### Link Quality Targets
- **100% Link Validation**: All links checked and verified
- **Zero Dead Links**: No broken or inaccessible links
- **Complete Security**: All links meet NSA security standards
- **Audit Compliance**: Full documentation of all changes

### Performance Requirements
- **Fast Validation**: Complete project scan in under 5 minutes
- **Reliable Detection**: 100% link discovery accuracy
- **Smart Repair**: >80% automatic repair success rate
- **Safe Operation**: Zero false positive removals

## Implementation Strategy

### Phase 1: Discovery and Classification
1. Build comprehensive link discovery engine
2. Implement link type classification system
3. Create security-aware validation framework
4. Develop intelligent context analysis

### Phase 2: Validation and Repair
1. Implement external link validation with security checks
2. Build internal link validation system
3. Create automated repair algorithms
4. Develop safe link replacement mechanisms

### Phase 3: Dead Link Elimination
1. Implement context-aware link removal
2. Build backup and change documentation systems
3. Create manual review flagging system
4. Develop continuous monitoring capabilities

This prompt ensures comprehensive link health management while maintaining strict compliance with rules.md requirements for security, reliability, and real implementation standards.