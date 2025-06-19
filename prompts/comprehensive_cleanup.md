# Comprehensive Project Cleanup and File Consolidation

## 🚨 MANDATORY RULES COMPLIANCE 🚨
**READ /mnt/c/Users/pascaldisse/Downloads/open-sourcefy/rules.md FIRST AND ENFORCE ALL RULES**

All cleanup operations must maintain absolute compliance with:
- **NO FALLBACKS EVER** - one correct way only
- **STRICT MODE ONLY** - fail fast on missing requirements  
- **NO MOCK IMPLEMENTATIONS** - real implementations only
- **NO HARDCODED VALUES** - all values from configuration
- **NSA-LEVEL SECURITY** - zero tolerance for vulnerabilities
- **PRESERVE MAIN ARCHITECTURE** - do not merge agents (agents must remain separate)

## Mission Objective

Systematically clean up obsolete files, merge duplicate/similar content, and consolidate documentation while preserving the main Matrix agent architecture and all functional code.

## 1. Comprehensive Directory Analysis and Categorization

### ALL DIRECTORY SCAN - Complete File Inventory

Let me scan ALL directories and subdirectories to identify every file type for cleanup:

```python
import os
from pathlib import Path
import subprocess

def comprehensive_directory_scan():
    """Scan ALL directories for obsolete, duplicate, and test files"""
    
    # Get complete file inventory
    all_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            file_path = Path(root) / file
            all_files.append({
                'path': file_path,
                'size': file_path.stat().st_size if file_path.exists() else 0,
                'extension': file_path.suffix,
                'name': file_path.name
            })
    
    return all_files

# Categories for comprehensive cleanup
CLEANUP_CATEGORIES = {
    'OBSOLETE_DOCS': {
        'patterns': ['**/docs/API-Reference.md', '**/docs/Agent-Execution-Report.md', 
                    '**/docs/Source-Code-Analysis.md', '**/docs/Technical-Specifications.md',
                    '**/docs/README.md', '**/docs/index.md'],
        'description': 'Minimal documentation stubs (4-125 lines)'
    },
    
    'DUPLICATE_TESTS': {
        'patterns': ['**/test_*.py', '**/tests/**/test_*.py', '**/*_test.py',
                    '**/complete_pipeline_testing.md', '**/comprehensive_pipeline_tests.md',
                    '**/enhanced_comprehensive_testing.md'],
        'description': 'Duplicate and overlapping test files'
    },
    
    'OLD_PROMPTS': {
        'patterns': ['**/prompts/**/pipeline_compilation.md', '**/prompts/**/full_pipeline_compilation.md',
                    '**/prompts/**/*_old.md', '**/prompts/**/*_backup.md',
                    '**/prompts/**/*_deprecated.md'],
        'description': 'Duplicate and old prompt files'
    },
    
    'TEMP_FILES': {
        'patterns': ['**/*.tmp', '**/*.temp', '**/*.bak', '**/*.backup',
                    '**/*~', '**/.DS_Store', '**/Thumbs.db', '**/*.swp',
                    '**/__pycache__/**', '**/*.pyc', '**/*.pyo'],
        'description': 'Temporary files, caches, and system files'
    },
    
    'OUTPUT_ARTIFACTS': {
        'patterns': ['**/output/**', '**/temp/**', '**/logs/**/*.log',
                    '**/build/**', '**/dist/**', '**/.vscode/**',
                    '**/.idea/**', '**/*.egg-info/**'],
        'description': 'Generated output, build artifacts, IDE files'
    },
    
    'WORKTREES': {
        'patterns': ['**/worktrees/**', '**/auto-fixer-run-*/**'],
        'description': 'Git worktrees and automated fixer directories'
    },
    
    'DUPLICATE_CONFIGS': {
        'patterns': ['**/config*.yaml.bak', '**/config*.yaml.old',
                    '**/build_config*.yaml.backup', '**/*_config.yaml.tmp'],
        'description': 'Backup and duplicate configuration files'
    },
    
    'TEST_ARTIFACTS': {
        'patterns': ['**/test_results/**', '**/coverage/**', '**/.pytest_cache/**',
                    '**/htmlcov/**', '**/*.coverage', '**/test_output/**'],
        'description': 'Test execution artifacts and coverage files'
    }
}
```

### Current File Status Analysis - COMPREHENSIVE SCAN

#### /docs Directory Analysis
```
OBSOLETE/MINIMAL FILES TO DELETE:
├── API-Reference.md (4 lines - minimal stub)
├── Agent-Execution-Report.md (4 lines - minimal stub) 
├── Source-Code-Analysis.md (4 lines - minimal stub)
├── Technical-Specifications.md (4 lines - minimal stub)
├── README.md (125 lines - generated placeholder, superseded by main CLAUDE.md)
└── index.md (potentially obsolete - check content)

KEEP - SUBSTANTIAL CONTENT:
├── SYSTEM_ARCHITECTURE.md (substantial architecture documentation)
├── AGENT_REFACTOR_SPECIFICATIONS.md (critical refactor specifications)
└── PRODUCTION_DEPLOYMENT_STRATEGY.md (deployment documentation)
```

#### /prompts Directory Analysis
```
DUPLICATE/OVERLAPPING TESTING PROMPTS TO MERGE:
├── complete_pipeline_testing.md (comprehensive pipeline testing)
├── comprehensive_pipeline_tests.md (similar pipeline testing)
└── enhanced_comprehensive_testing.md (enhanced testing with caching)

SIMILAR PROMPTS TO CONSOLIDATE:
├── pipeline_compilation.md (compilation testing)
├── full_pipeline_compilation.md (potentially duplicate compilation)

UNIFIED VERSIONS ALREADY EXIST:
├── unified_comprehensive_testing.md (KEEP - already merged)
├── unified_pipeline_compilation.md (KEEP - already merged)

KEEP AS-IS (DISTINCT PURPOSES):
├── CLAUDE.md (project overview)
├── agent_cleanup.md (agent cleanup)
├── code_quality_checker.md (quality checking)
├── documentation_validator.md (documentation validation)
├── comprehensive_cleanup.md (THIS FILE - cleanup operations)
└── [other distinct purpose prompts]
```

#### /tests Directory Analysis
```
SCAN ALL TEST FILES FOR:
├── Duplicate test files (same functionality different names)
├── Obsolete test files (testing removed features)
├── Test artifacts and cache files
├── Old test results and coverage files
└── Broken test files (import errors, missing dependencies)
```

#### ALL OTHER DIRECTORIES
```
TEMP AND CACHE FILES:
├── **/__pycache__/** (Python cache directories)
├── **/*.pyc, *.pyo (Python compiled files)
├── **/.pytest_cache/** (Pytest cache)
├── **/htmlcov/** (Coverage HTML reports)

SYSTEM FILES:
├── **/.DS_Store (macOS system files)
├── **/Thumbs.db (Windows thumbnail cache)
├── **/*.swp, *~ (Vim/editor temp files)

BUILD ARTIFACTS:
├── **/build/** (Build output directories)
├── **/dist/** (Distribution directories)
├── **/*.egg-info/** (Python package info)

IDE FILES:
├── **/.vscode/** (VS Code settings)
├── **/.idea/** (IntelliJ/PyCharm settings)
├── **/*.sublime-* (Sublime Text files)

WORKTREES AND AUTOMATION:
├── **/worktrees/** (Git worktree directories)
├── **/auto-fixer-run-*/** (Automated fixer working directories)
└── **/temp/** (Temporary working directories)
```

## 2. Cleanup Operations Plan

### Phase 1: Delete Obsolete Documentation Stubs

```bash
# Delete minimal/obsolete documentation stubs in /docs
rm docs/API-Reference.md
rm docs/Agent-Execution-Report.md  
rm docs/Source-Code-Analysis.md
rm docs/Technical-Specifications.md
rm docs/README.md  # Superseded by main CLAUDE.md
rm docs/index.md   # If it exists and is obsolete
```

### Phase 2: Merge Duplicate Testing Prompts

Create unified comprehensive testing prompt by merging:
- `complete_pipeline_testing.md` (403 lines)
- `comprehensive_pipeline_tests.md` (869 lines) 
- `enhanced_comprehensive_testing.md` (599 lines)

**Merge Strategy:**
1. Keep the most comprehensive features from each file
2. Combine caching capabilities from enhanced_comprehensive_testing.md
3. Include binary comparison features from complete_pipeline_testing.md
4. Maintain comprehensive individual agent testing from comprehensive_pipeline_tests.md
5. Preserve all rules.md compliance requirements
6. Create single authoritative testing prompt

### Phase 3: Consolidate Similar Compilation Prompts

Analyze and potentially merge:
- `pipeline_compilation.md`
- `full_pipeline_compilation.md`

If they overlap significantly, merge into unified compilation prompt.

### Phase 4: Validate Architecture Preservation

**CRITICAL: DO NOT MERGE AGENTS**
- All 17 Matrix agents must remain as separate files
- Core architecture must be preserved intact
- Shared components must remain separate
- Configuration management must remain separate
- Pipeline orchestrator must remain separate

## 3. File Merge Implementation

### Create Unified Testing Prompt

```python
def create_unified_testing_prompt():
    """
    Merge the three testing prompts into comprehensive testing framework
    
    Content Integration:
    1. Rules compliance sections (from all three)
    2. Individual agent testing with caching (from enhanced_comprehensive_testing.md)
    3. Complete pipeline execution (from complete_pipeline_testing.md)
    4. Binary comparison and validation (from complete_pipeline_testing.md)
    5. Systematic error fixing (from comprehensive_pipeline_tests.md)
    6. Performance and quality metrics (from all three)
    
    Result: Single comprehensive testing prompt with all capabilities
    """
```

### Compilation Prompt Consolidation

```python  
def consolidate_compilation_prompts():
    """
    Analyze compilation prompts for overlap and merge if appropriate
    
    Analysis Steps:
    1. Compare pipeline_compilation.md and full_pipeline_compilation.md
    2. Identify overlapping content and unique features
    3. If >70% overlap, merge into unified compilation prompt
    4. If <70% overlap, keep separate with clear naming
    5. Ensure VS2022 Preview compliance throughout
    """
```

## 4. Cleanup Execution Commands

### Comprehensive File Cleanup Script

```bash
#!/bin/bash
# Comprehensive project cleanup with architecture preservation

echo "🚨 ENFORCING RULES.MD COMPLIANCE 🚨"
echo "🏗️  PRESERVING MAIN ARCHITECTURE - AGENTS REMAIN SEPARATE"

# Phase 1: Delete obsolete documentation stubs
echo "📋 Phase 1: Removing obsolete documentation stubs..."
rm -f docs/API-Reference.md
rm -f docs/Agent-Execution-Report.md
rm -f docs/Source-Code-Analysis.md  
rm -f docs/Technical-Specifications.md
rm -f docs/README.md
rm -f docs/index.md
rm -f docs/tasks.md

echo "✅ Obsolete documentation stubs removed"

# Phase 2: Clean up duplicate testing prompts
echo "📋 Phase 2: Cleaning up duplicate testing prompts..."
rm -f prompts/complete_pipeline_testing.md
rm -f prompts/comprehensive_pipeline_tests.md
rm -f prompts/enhanced_comprehensive_testing.md

echo "✅ Duplicate testing prompts removed (keeping unified version)"

# Phase 3: Clean up duplicate compilation prompts
echo "📋 Phase 3: Cleaning up duplicate compilation prompts..."
rm -f prompts/pipeline_compilation.md
rm -f prompts/full_pipeline_compilation.md

echo "✅ Duplicate compilation prompts removed (keeping unified version)"

# Phase 4: Remove temporary files and caches
echo "📋 Phase 4: Removing temporary files and caches..."

# Python cache files
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true
find . -name "*.pyo" -type f -delete 2>/dev/null || true

# Test cache files
rm -rf .pytest_cache 2>/dev/null || true
rm -rf htmlcov 2>/dev/null || true
rm -f .coverage 2>/dev/null || true

# System files
find . -name ".DS_Store" -type f -delete 2>/dev/null || true
find . -name "Thumbs.db" -type f -delete 2>/dev/null || true
find . -name "*.swp" -type f -delete 2>/dev/null || true
find . -name "*~" -type f -delete 2>/dev/null || true

# Temporary files and backups
find . -name "*.tmp" -type f -delete 2>/dev/null || true
find . -name "*.temp" -type f -delete 2>/dev/null || true
find . -name "*.bak" -type f -delete 2>/dev/null || true
find . -name "*.backup" -type f -delete 2>/dev/null || true

# IDE and editor files
rm -rf .vscode 2>/dev/null || true
rm -rf .idea 2>/dev/null || true
find . -name "*.sublime-*" -type f -delete 2>/dev/null || true

# Build artifacts
rm -rf build 2>/dev/null || true
rm -rf dist 2>/dev/null || true
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Temporary files and caches removed"

# Phase 5: Clean up git worktrees and automation directories
echo "📋 Phase 5: Cleaning up git worktrees and automation directories..."
rm -rf worktrees 2>/dev/null || true
rm -rf auto-fixer-run-* 2>/dev/null || true
rm -rf temp 2>/dev/null || true

echo "✅ Git worktrees and automation directories cleaned"

# Phase 6: Validate architecture preservation
echo "📋 Phase 6: Validating architecture preservation..."
python3 << 'EOF'
from pathlib import Path

# Check that all critical architecture files remain
critical_files = [
    "src/core/agents",  # Agent directory
    "src/core/matrix_pipeline_orchestrator.py",
    "src/core/shared_components.py", 
    "src/core/config_manager.py",
    "main.py",
    "CLAUDE.md",
    "rules.md"
]

missing_files = []
for file_path in critical_files:
    if not Path(file_path).exists():
        missing_files.append(file_path)

if missing_files:
    print(f"❌ CRITICAL: Missing architecture files: {missing_files}")
    exit(1)
else:
    print("✅ Main architecture preserved - all critical files intact")

# Check that agents remain separate  
agent_files = list(Path("src/core/agents").glob("agent*.py"))
if len(agent_files) >= 17:
    print(f"✅ Agent architecture preserved - {len(agent_files)} agent files found")
else:
    print(f"⚠️  Expected 17+ agent files, found {len(agent_files)}")
    
# Check that unified prompts exist
unified_prompts = [
    "prompts/unified_comprehensive_testing.md",
    "prompts/unified_pipeline_compilation.md"
]

for prompt in unified_prompts:
    if Path(prompt).exists():
        print(f"✅ Unified prompt exists: {prompt}")
    else:
        print(f"⚠️  Missing unified prompt: {prompt}")
EOF

echo "🎉 COMPREHENSIVE CLEANUP COMPLETED SUCCESSFULLY"
echo "📊 Summary:"
echo "   - Obsolete documentation stubs removed"
echo "   - Duplicate prompts cleaned up (unified versions preserved)"
echo "   - Temporary files and caches removed"
echo "   - Git worktrees and automation directories cleaned"
echo "   - Main architecture preserved intact"
echo "   - All 17 Matrix agents remain separate"
echo "   - Rules.md compliance maintained throughout"
```

## 5. Post-Cleanup Validation

### Architecture Integrity Check

```python
def validate_post_cleanup_architecture():
    """
    Comprehensive validation that cleanup preserved architecture
    
    Critical Validations:
    1. All 17 Matrix agents exist as separate files
    2. Core pipeline orchestrator remains intact
    3. Shared components remain separate
    4. Configuration management preserved
    5. Build system integration maintained
    6. Main entry point (main.py) unchanged
    7. Critical documentation (CLAUDE.md, rules.md) preserved
    
    Quality Checks:
    1. No broken imports after file removal
    2. No missing dependencies after consolidation
    3. All prompt references remain valid
    4. Test framework still functional
    5. Documentation links remain valid
    
    Returns:
        ValidationResult with pass/fail status and recommendations
    """
```

### Cleanup Success Metrics

```yaml
File Reduction Success:
  - Obsolete files removed: Target 5-8 files
  - Duplicate prompts consolidated: Target 2-3 merges
  - Storage space saved: Target >50KB
  - Maintenance burden reduced: Qualitative improvement

Architecture Preservation Success:
  - All 17 agents remain separate: Required
  - Core architecture unchanged: Required  
  - No broken imports: Required
  - All tests still pass: Required
  - Documentation consistency: Required

Quality Improvement Success:
  - Reduced confusion from duplicates: Qualitative
  - Clearer prompt organization: Qualitative
  - Easier maintenance: Qualitative
  - Better developer experience: Qualitative
```

## 6. Usage Instructions

```bash
# Execute comprehensive cleanup
bash prompts/comprehensive_cleanup.md

# Validate cleanup results
python3 main.py --verify-env --validate-architecture

# Test unified framework
python3 main.py --test-unified-framework --validate-all

# Generate post-cleanup report
python3 main.py --generate-cleanup-report --architecture-validation
```

This comprehensive cleanup will eliminate obsolete files, merge duplicates, and streamline the project while preserving the essential Matrix agent architecture and all functional capabilities.