# Fallback Code Removal Automation Prompt

## CRITICAL OBJECTIVE
Remove ALL fallback code from the entire open-sourcefy project and replace with proper error handling. The system should fail fast and fail hard when required components are unavailable, rather than proceeding with degraded functionality.

## CORE PRINCIPLE
**No Fallbacks, Only Excellence**: When Ghidra fails, when AI is unavailable, when dependencies are missing - the pipeline should throw clear errors instead of continuing with placeholder/dummy code.

## CRITICAL OUTPUT REQUIREMENT
⚠️ **ALL OUTPUT MUST GO TO `/output` DIRECTORY ONLY** ⚠️
- NO files should be created outside of `/output/` or its subdirectories
- ALL agent results, logs, reports, and analysis data MUST be in `/output/`

## Fallback Patterns to Remove Project-Wide

### 1. Ghidra Fallback Patterns
**Search Patterns**:
```python
# Pattern 1: Ghidra timeout fallbacks
"Ghidra.*failed.*fallback"
"timeout.*fallback"
"proceeding.*fallback"

# Pattern 2: Fallback function generation
"_generate_fallback_functions"
"fallback_static_reconstruction"
"fallback_analysis"

# Pattern 3: Fallback result parsing
"Fallback analysis used"
"fallback.*results"
"using.*fallback"
```

**Replacement Strategy**:
```python
# BEFORE (fallback code)
except TimeoutError:
    self.logger.warning(f"Ghidra analysis timed out - proceeding with fallback")
    return self._generate_fallback_functions()

# AFTER (fail-fast)
except TimeoutError:
    self.logger.error(f"Ghidra analysis timed out after {timeout}s")
    raise RuntimeError(f"Ghidra analysis timed out - pipeline requires successful Ghidra execution")
```

### 2. AI System Fallback Patterns
**Search Patterns**:
```python
# Pattern 1: AI unavailable fallbacks
"AI.*not.*available.*fallback"
"ai_enabled.*False.*fallback"
"claude.*not.*found.*fallback"

# Pattern 2: AI timeout fallbacks  
"ai.*timeout.*fallback"
"AI.*failed.*proceeding"
"AI.*enhancement.*failed.*fallback"
```

**Replacement Strategy**:
```python
# BEFORE (AI fallback)
if not self.ai_enabled:
    return self._generate_basic_analysis()

# AFTER (fail-fast)
if not self.ai_enabled:
    raise RuntimeError("AI system required for advanced analysis - configure Claude CLI or disable AI-dependent agents")
```

### 3. Dependency Fallback Patterns
**Search Patterns**:
```python
# Pattern 1: Missing dependency fallbacks
"dependency.*not.*found.*fallback"
"prerequisite.*missing.*proceeding"
"agent.*failed.*fallback"

# Pattern 2: Partial result fallbacks
"incomplete.*data.*fallback"
"missing.*results.*proceeding"
"partial.*analysis.*fallback"
```

**Replacement Strategy**:
```python
# BEFORE (dependency fallback)
if not dependency_available:
    self.logger.warning("Dependency missing - using basic analysis")
    return self._basic_analysis_fallback()

# AFTER (fail-fast)
if not dependency_available:
    raise ValueError(f"Required dependency Agent{dep_id} not satisfied - cannot proceed")
```

### 4. Quality Threshold Fallback Patterns
**Search Patterns**:
```python
# Pattern 1: Quality threshold fallbacks
"quality.*below.*threshold.*proceeding"
"confidence.*low.*fallback"
"validation.*failed.*proceeding"

# Pattern 2: Retry failure fallbacks
"max.*retries.*exceeded.*fallback"
"analysis.*failed.*using.*fallback"
"threshold.*not.*met.*proceeding"
```

**Replacement Strategy**:
```python
# BEFORE (quality fallback)
if quality_score < threshold:
    self.logger.warning("Quality below threshold - proceeding with current results")
    return current_results

# AFTER (fail-fast)
if quality_score < threshold:
    raise ValidationError(f"Quality score {quality_score} below required threshold {threshold}")
```

### 5. Build System Fallback Patterns
**Search Patterns**:
```python
# Pattern 1: Compiler fallbacks
"msvc.*not.*found.*fallback"
"visual.*studio.*missing.*fallback"
"build.*tools.*unavailable.*fallback"

# Pattern 2: Platform fallbacks
"windows.*sdk.*missing.*fallback"
"path.*not.*found.*fallback"
"compilation.*failed.*fallback"
```

**Replacement Strategy**:
```python
# BEFORE (build fallback)
if not msvc_available:
    self.logger.warning("MSVC not found - using basic compilation")
    return self._generate_basic_makefile()

# AFTER (fail-fast)
if not msvc_available:
    raise EnvironmentError("MSVC compiler required for Windows PE compilation - install Visual Studio")
```

## Automation Script Requirements

### Script Functionality
The automation script should:

1. **Scan all Python files** in the project for fallback patterns
2. **Identify fallback code blocks** using regex patterns
3. **Generate replacement code** with proper error handling
4. **Validate changes** don't break imports or syntax
5. **Create backup** of original files
6. **Apply changes systematically** across the project
7. **Generate report** of all changes made

### Search and Replace Patterns

```python
FALLBACK_PATTERNS = {
    'ghidra_timeout_fallback': {
        'search': r'except\s+TimeoutError:.*?fallback.*?\n.*?return.*?fallback',
        'replace': '''except TimeoutError:
    self.logger.error(f"Ghidra analysis timed out after {self.timeout}s")
    raise RuntimeError("Ghidra analysis timed out - pipeline requires successful Ghidra execution")'''
    },
    
    'ai_unavailable_fallback': {
        'search': r'if\s+not\s+.*ai_enabled.*:\s*\n.*?return.*?fallback',
        'replace': '''if not self.ai_enabled:
    raise RuntimeError("AI system required for this agent - configure Claude CLI or use non-AI agents")'''
    },
    
    'dependency_missing_fallback': {
        'search': r'if\s+not\s+.*dependency.*:\s*\n.*?warning.*fallback.*\n.*?return.*?fallback',
        'replace': '''if not dependency_available:
    raise ValueError(f"Required dependency not satisfied - cannot proceed")'''
    },
    
    'quality_threshold_fallback': {
        'search': r'if\s+.*quality.*<.*threshold.*:\s*\n.*?warning.*proceeding.*\n.*?return',
        'replace': '''if quality_score < threshold:
    raise ValidationError(f"Quality score {quality_score} below required threshold {threshold}")'''
    },
    
    'ghidra_failure_fallback': {
        'search': r'if\s+not\s+success.*:\s*\n.*?warning.*fallback.*\n.*?return.*?fallback',
        'replace': '''if not success:
    self.logger.error(f"Ghidra analysis failed: {output}")
    raise RuntimeError(f"Ghidra analysis failed: {output}")'''
    }
}

FALLBACK_METHODS_TO_REMOVE = [
    '_generate_fallback_functions',
    '_create_fallback_analysis',
    '_basic_analysis_fallback', 
    '_fallback_static_reconstruction',
    '_generate_basic_analysis',
    '_simple_fallback_analysis',
    '_minimal_fallback_result'
]

FALLBACK_KEYWORDS = [
    'fallback',
    'Fallback', 
    'proceeding with',
    'using basic',
    'degraded mode',
    'simplified analysis',
    'minimal functionality'
]
```

### File Processing Logic

```python
def remove_fallbacks_from_file(file_path: str) -> Dict[str, Any]:
    """Remove all fallback code from a single Python file"""
    
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    modified_content = original_content
    changes_made = []
    
    # Remove fallback methods entirely
    for method_name in FALLBACK_METHODS_TO_REMOVE:
        pattern = rf'def {method_name}\(.*?\):.*?(?=def|\Z)'
        if re.search(pattern, modified_content, re.DOTALL):
            modified_content = re.sub(pattern, '', modified_content, flags=re.DOTALL)
            changes_made.append(f"Removed method: {method_name}")
    
    # Replace fallback patterns with error handling
    for pattern_name, pattern_info in FALLBACK_PATTERNS.items():
        if re.search(pattern_info['search'], modified_content, re.DOTALL | re.MULTILINE):
            modified_content = re.sub(
                pattern_info['search'], 
                pattern_info['replace'], 
                modified_content, 
                flags=re.DOTALL | re.MULTILINE
            )
            changes_made.append(f"Replaced pattern: {pattern_name}")
    
    # Remove fallback comments and documentation
    fallback_comment_patterns = [
        r'#.*fallback.*\n',
        r'""".*fallback.*?"""',
        r"'''.*fallback.*?'''"
    ]
    
    for comment_pattern in fallback_comment_patterns:
        if re.search(comment_pattern, modified_content, re.DOTALL | re.IGNORECASE):
            modified_content = re.sub(comment_pattern, '', modified_content, flags=re.DOTALL | re.IGNORECASE)
            changes_made.append("Removed fallback comments")
    
    return {
        'file_path': file_path,
        'original_content': original_content,
        'modified_content': modified_content,
        'changes_made': changes_made,
        'has_changes': len(changes_made) > 0
    }
```

## Target Files for Fallback Removal

### Priority 1: Core Agents
- `src/core/agents/agent05_neo_advanced_decompiler.py` ✅ (Already cleaned)
- `src/core/agents/agent03_merovingian.py` ✅ (Already cleaned)
- `src/core/agents/agent01_sentinel.py`
- `src/core/agents/agent02_architect.py`
- `src/core/agents/agent04_agent_smith.py`
- `src/core/agents/agent06_twins_binary_diff.py`
- `src/core/agents/agent07_trainman_assembly_analysis.py`
- `src/core/agents/agent08_keymaker_resource_reconstruction.py`
- `src/core/agents/agent09_commander_locke.py`
- `src/core/agents/agent10_the_machine.py`
- `src/core/agents/agent11_the_oracle.py`
- `src/core/agents/agent12_link.py`
- `src/core/agents/agent13_agent_johnson.py`
- `src/core/agents/agent14_the_cleaner.py`
- `src/core/agents/agent15_analyst.py`
- `src/core/agents/agent16_agent_brown.py`

### Priority 2: Core Infrastructure
- `src/core/ghidra_headless.py`
- `src/core/ghidra_processor.py`
- `src/core/ai_system.py`
- `src/core/build_system_manager.py`
- `src/core/semantic_decompiler.py`
- `src/core/advanced_data_type_inference.py`

### Priority 3: Utility and Support Files
- `src/core/shared_utils.py`
- `src/core/file_utils.py`
- `src/core/binary_utils.py`
- `src/core/error_handler.py`
- `src/core/config_manager.py`

## Success Criteria

### Code Quality Metrics
- [ ] **Zero Fallback Keywords**: No instances of 'fallback', 'proceeding with', 'degraded mode' in codebase
- [ ] **No Fallback Methods**: All `_generate_fallback_*` and `_*_fallback` methods removed
- [ ] **Fail-Fast Error Handling**: All conditional fallbacks replaced with RuntimeError/ValidationError
- [ ] **No Quality Degradation**: No code that proceeds with substandard results
- [ ] **No Timeout Fallbacks**: All timeout scenarios result in pipeline failure

### Behavioral Changes
- [ ] **Ghidra Required**: Pipeline fails immediately if Ghidra is unavailable or times out
- [ ] **AI Required**: Agents requiring AI fail immediately if Claude CLI is unavailable
- [ ] **Dependencies Enforced**: All agent dependencies strictly enforced with no workarounds
- [ ] **Quality Thresholds**: Analysis results below quality thresholds cause pipeline failure
- [ ] **Build Requirements**: Compilation requires proper build tools or fails immediately

### Pipeline Integrity
- [ ] **No Silent Failures**: All failures are loud and explicit
- [ ] **Clear Error Messages**: All errors explain exactly what's missing and how to fix it
- [ ] **Fast Failure**: Pipeline fails as early as possible when requirements aren't met
- [ ] **No Degraded Results**: Pipeline either produces high-quality results or fails completely

## Implementation Command

To execute this fallback removal across the entire project:

```bash
python3 prompts/fallback_removal_automation.py --scan-all --apply-changes --backup-original --generate-report
```

The automation script will:
1. Scan all Python files for fallback patterns
2. Generate replacement code with proper error handling  
3. Create backups of original files
4. Apply changes systematically
5. Validate syntax and imports still work
6. Generate comprehensive report of all changes made
7. Ensure all output goes to `/output/` directory only

## Expected Outcome

After running this automation:
- **Zero tolerance for fallbacks**: System fails fast when requirements aren't met
- **Clear error messages**: Users know exactly what needs to be fixed
- **Quality assurance**: No degraded or placeholder results ever produced
- **Reliable pipeline**: System either works correctly or fails with clear diagnostics
- **Professional behavior**: No "try this backup approach" mentality

The result will be a robust, professional decompilation system that maintains high standards and fails gracefully with clear guidance rather than producing substandard results.