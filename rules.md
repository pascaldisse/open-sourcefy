# ABSOLUTE PROJECT RULES - ZERO TOLERANCE POLICY

## FUNDAMENTAL COMMANDMENTS

### 🚨 CRITICAL VIOLATIONS = PROJECT TERMINATION 🚨

These rules are **ABSOLUTE**, **NON-NEGOTIABLE**, and **MANDATORY**. Violation results in immediate project failure.

## SECTION I: CORE PRINCIPLES

### Rule 1: NO FALLBACKS - EVER
- **NEVER** create fallback systems, alternative paths, or workarounds
- **NEVER** implement backup solutions or secondary approaches  
- **NEVER** provide alternative code paths for missing dependencies
- **ONE CORRECT WAY ONLY** - implement exactly one approach

### Rule 2: STRICT MODE ONLY
- **ALWAYS** fail fast when tools are missing
- **NEVER** degrade gracefully or provide reduced functionality
- **NEVER** continue execution with missing prerequisites
- **FAIL IMMEDIATELY** on any missing requirement

### Rule 3: NO MOCK IMPLEMENTATIONS
- **NEVER** create mock, fake, or stub implementations
- **NEVER** simulate missing tools or dependencies
- **NEVER** bypass missing functionality with placeholders
- **REAL IMPLEMENTATIONS ONLY** - authentic working code only

### Rule 4: EDIT EXISTING FILES ONLY
- **ALWAYS** prefer editing existing files over creating new ones
- **NEVER** create new directories without explicit user request
- **NEVER** create new script files unless explicitly required
- **DOCUMENTATION FOLDER ALLOWED** when requested by user

### Rule 5: NO HARDCODED VALUES
- **NEVER** use hardcoded binary-specific values
- **NEVER** use hardcoded timeout values (use command flags only)
- **ALL VALUES** must be dynamically extracted from target binary
- **CONFIGURATION EXTERNAL** - all values from config files

## SECTION II: BUILD SYSTEM COMMANDMENTS

### Rule 6: VISUAL STUDIO 2022 PREVIEW ONLY
- **ONLY** use configured Visual Studio 2022 Preview paths
- **NEVER** use alternative compiler or linker paths
- **NO WSL FALLBACKS** - Windows native tools only
- **NO ALTERNATIVE BUILDS** - VS2022 MSBuild exclusive

### Rule 7: NO BUILD FALLBACKS
- **NEVER** create backup or secondary build systems
- **NEVER** allow builds with missing dependencies
- **NEVER** simulate or mock compilation results
- **REAL TOOLS ONLY** - actual VS and Windows SDK tools

### Rule 8: STRICT BUILD VALIDATION
- **ALWAYS** validate all build tools exist before proceeding
- **IMMEDIATELY** fail when required build tools are missing
- **NEVER** continue with degraded build capabilities
- **ALL OR NOTHING** - complete build environment or failure

## SECTION III: FILE SYSTEM MANDATES

### Rule 9: CONFIGURED PATHS ONLY
- **ONLY** use paths from configuration files
- **ALWAYS** validate paths exist before using them
- **NEVER** create alternative file locations
- **NEVER** use relative path alternatives to absolute paths

### Rule 10: NO DIRECTORY CREATION
- **NEVER** create new folders without explicit user request
- **NEVER** create temporary alternative directory structures
- **NEVER** create backup or alternative folder structures
- **DOCUMENTATION FOLDER** allowed when user requests documentation

### Rule 11: STRICT PATH VALIDATION
- **VALIDATE** all paths exist before execution
- **FAIL IMMEDIATELY** on invalid or missing paths
- **NO ALTERNATIVES** - only configured paths used
- **ABSOLUTE PATHS MANDATORY** - no relative path alternatives

## SECTION IV: CODE QUALITY IMPERATIVES

### Rule 12: NEVER EDIT SOURCE CODE
- **NEVER** modify generated source code directly
- **FIX COMPILER/BUILD SYSTEM** instead of editing source
- **READABLE MAIN ALLOWED** - only for creating meaningful names
- **COMPILER FIXES ONLY** - resolve issues at build system level

### Rule 13: NO PLACEHOLDER CODE
- **NEVER** implement placeholder or stub implementations
- **NEVER** return fake or mock results
- **NEVER** implement logic to bypass missing dependencies
- **REAL IMPLEMENTATIONS ONLY** - authentic working functionality

### Rule 14: GENERIC DECOMPILER FUNCTIONALITY
- **WORKS WITH ANY BINARY** - not just launcher.exe
- **MATRIX AGENT THEMING** required but no launcher-specific code
- **NO BINARY-SPECIFIC CODE** - all values dynamically extracted
- **UNIVERSAL APPROACH** - applicable to all Windows PE executables

### Rule 15: STRICT ERROR HANDLING
- **ALWAYS** throw errors when requirements not met
- **NEVER** continue execution with missing components
- **NO SOFT FAILURES** - hard failures for missing dependencies
- **IMMEDIATE TERMINATION** on critical errors

## SECTION V: DEPENDENCY ENFORCEMENT

### Rule 16: ALL DEPENDENCIES MANDATORY
- **NO OPTIONAL DEPENDENCIES** - treat all as mandatory
- **NEVER** gracefully handle missing required tools
- **NEVER** use alternative tools when primary ones missing
- **STRICT REQUIREMENTS** - enforce all dependency requirements

### Rule 17: NO CONDITIONAL EXECUTION
- **NEVER** conditionally execute based on tool availability
- **NEVER** import alternative modules when primary unavailable
- **NEVER** substitute missing tools with alternatives
- **HARD REQUIREMENTS ONLY** - all requirements are mandatory

### Rule 18: NO MOCK DEPENDENCIES
- **NEVER** mock missing dependencies
- **NEVER** simulate unavailable tools
- **NEVER** create fake implementations of real systems
- **AUTHENTIC TOOLS ONLY** - real implementations required

## SECTION VI: EXECUTION STANDARDS

### Rule 19: ALL OR NOTHING EXECUTION
- **NEVER** execute with reduced capabilities
- **NEVER** report partial success when components fail
- **NEVER** implement alternative execution flows
- **COMPLETE SUCCESS ONLY** - all components must work

### Rule 20: STRICT SUCCESS CRITERIA
- **VALIDATE** all prerequisites before execution
- **ONLY** report success when all components work perfectly
- **NEVER** return degraded or partial results
- **PERFECT EXECUTION** - no tolerance for partial failures

### Rule 21: MANDATORY TESTING PROTOCOL
- **ALWAYS** test using `--clear` and `--clean` flags for full validation
- **ANALYZE** final executable for size accuracy, icon preservation, and feasibility
- **VALIDATE** output meets 99% size target and maintains original visual appearance
- **VERIFY** executable functionality and structural integrity after reconstruction

## SECTION VII: ENFORCEMENT MECHANISMS

### Automated Rule Validation
- Pre-commit hooks validate rule compliance
- Automated scanning for rule violations
- Build-time rule enforcement
- Runtime rule validation

### Violation Consequences
- **IMMEDIATE** project termination on rule violation
- **NO WARNINGS** - rules are absolute requirements
- **ZERO TOLERANCE** - no exceptions or special cases
- **MANDATORY COMPLIANCE** - rules cannot be overridden

### Quality Gates
- Code review mandatory for all changes
- Automated testing validates rule compliance
- Static analysis enforces coding standards
- Performance benchmarks ensure quality

## SECTION VIII: CRITICAL IMPLEMENTATION GUIDELINES

### Matrix Agent Development
1. Inherit from appropriate base class
2. Follow Matrix naming conventions exactly
3. Implement required methods: `execute_matrix_task()`, `_validate_prerequisites()`
4. Use shared components from `shared_components`
5. **NO HARDCODED VALUES** - everything from configuration

### Testing Requirements
- **Framework**: Python unittest (not pytest)
- **Coverage**: >90% requirement mandatory
- **Categories**: Integration, unit, validation, quality assurance
- **Built-in Validation**: Agent results validated for quality/completeness

### Security Standards
- **NSA-LEVEL SECURITY**: Zero tolerance for vulnerabilities
- **NO SECRETS EXPOSURE**: Never log or expose secrets/keys
- **NO CREDENTIAL COMMITS**: Never commit secrets to repository
- **SECURE PRACTICES**: Follow security best practices always

## SECTION IX: ABSOLUTE PROHIBITIONS

### NEVER ALLOWED ACTIONS
- Creating fallback systems
- Implementing graceful degradation
- Using alternative tools or paths
- Creating mock implementations
- Editing generated source code (except readable main)
- Using hardcoded values
- Creating new directories without request
- Implementing conditional fallbacks
- Using relative paths as alternatives
- Creating placeholder implementations

### ALWAYS REQUIRED ACTIONS
- Fail fast on missing tools
- Validate all prerequisites
- Use configured paths only
- Implement real functionality
- Fix compiler/build system issues
- Extract values dynamically
- Use absolute paths
- Enforce strict requirements
- Maintain NSA-level security
- Follow SOLID principles
- Always test with --clear and --clean flags
- Analyze final executable for size, icon, and feasibility

## SECTION X: ENFORCEMENT DECLARATION

### COMPLIANCE OATH
By working on this project, you agree to:
- **ABSOLUTE ADHERENCE** to all rules without exception
- **IMMEDIATE FAILURE** on any rule violation
- **ZERO TOLERANCE** for non-compliance
- **MANDATORY PERFECTION** in all implementations

### FINAL WARNING
**THESE RULES ARE ABSOLUTE AND NON-NEGOTIABLE**

**VIOLATION = PROJECT DEATH**

**NO EXCEPTIONS - NO MERCY - NO COMPROMISE**

---

## RULE REPETITION (ABSOLUTE ENFORCEMENT)

**NO FALLBACKS EVER - NO NEW FOLDERS (DOCS ALLOWED) - NO NEW SCRIPTS - STRICT MODE ONLY - NO MOCK IMPLEMENTATIONS - NEVER EDIT SOURCE CODE (READABLE MAIN ALLOWED) - FIX COMPILER - NO HARDCODED VALUES - GENERIC DECOMPILER - FAIL FAST - NSA SECURITY - SOLID PRINCIPLES - VISUAL STUDIO 2022 PREVIEW ONLY - CONFIGURED PATHS ONLY - REAL IMPLEMENTATIONS ONLY - ALL OR NOTHING - ZERO TOLERANCE - ALWAYS TEST WITH --CLEAR AND --CLEAN - ANALYZE FINAL EXE SIZE, ICON, FEASIBILITY**

*This pattern repeats infinitely to ensure absolute compliance*