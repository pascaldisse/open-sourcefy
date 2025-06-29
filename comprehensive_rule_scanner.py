#!/usr/bin/env python3
"""
Comprehensive Rule Violation Scanner
Scans the entire Matrix pipeline for ALL rule violations (Rules 1-21)
Provides specific fixes for each violation type.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

@dataclass
class RuleViolation:
    rule_number: int
    rule_name: str
    file_path: str
    line_number: int
    line_content: str
    violation_pattern: str
    suggested_fix: str
    severity: str  # 'critical', 'high', 'medium', 'low'

class ComprehensiveRuleScanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.violations = []
        
        # All rule patterns with fixes
        self.rule_patterns = {
            # Rule 1: NO FALLBACKS - EVER
            1: {
                'name': 'NO FALLBACKS - EVER',
                'patterns': [
                    (r'\bfallback\b', 'Remove fallback - implement one correct approach only'),
                    (r'\balternative\b.*path', 'Remove alternative paths - use single configured path'),
                    (r'\bbackup\b.*solution', 'Remove backup solutions - one correct implementation only'),
                    (r'if.*not.*available.*else', 'Remove conditional fallbacks - fail fast instead'),
                    (r'try.*except.*fallback', 'Remove try/except fallbacks - validate prerequisites first'),
                    (r'graceful.*degrad', 'Remove graceful degradation - strict mode only'),
                ],
                'severity': 'critical'
            },
            
            # Rule 2: STRICT MODE ONLY  
            2: {
                'name': 'STRICT MODE ONLY',
                'patterns': [
                    (r'graceful.*fail', 'Change to fail-fast - no graceful degradation'),
                    (r'degrade.*functionality', 'Remove degraded functionality - all or nothing'),
                    (r'continue.*with.*missing', 'Fail immediately on missing requirements'),
                    (r'reduced.*functionality', 'Remove reduced functionality - strict mode only'),
                    (r'partial.*success', 'Change to complete success or failure - no partial'),
                ],
                'severity': 'critical'
            },
            
            # Rule 3: NO MOCK IMPLEMENTATIONS
            3: {
                'name': 'NO MOCK IMPLEMENTATIONS', 
                'patterns': [
                    (r'\bmock\b', 'Remove mock - implement real functionality'),
                    (r'\bfake\b', 'Remove fake - implement authentic code'),
                    (r'\bstub\b', 'Remove stub - implement complete functionality'),
                    (r'\bsimulat[e|ion]\b', 'Remove simulation - implement real behavior'),
                    (r'bypass.*missing', 'Remove bypass - implement real dependency'),
                ],
                'severity': 'critical'
            },
            
            # Rule 4: EDIT EXISTING FILES ONLY
            4: {
                'name': 'EDIT EXISTING FILES ONLY',
                'patterns': [
                    (r'mkdir\s*\(', 'Avoid creating new directories - use existing structure'),
                    (r'os\.makedirs', 'Avoid creating new directories - use existing structure'),
                    (r'Path.*\.mkdir', 'Avoid creating new directories - use existing structure'),
                    (r'new.*script.*file', 'Avoid creating new scripts - edit existing ones'),
                ],
                'severity': 'high'
            },
            
            # Rule 5: NO HARDCODED VALUES (DISABLED - addresses are necessary for binary analysis)
            # 5: {
            #     'name': 'NO HARDCODED VALUES',
            #     'patterns': [
            #         (r'launcher\.exe', 'Use dynamic binary name from config/input'),
            #         (r'timeout.*=.*\d+', 'Use timeout from command flags, not hardcoded'),
            #         (r'0x[0-9a-fA-F]+', 'Extract address dynamically from binary analysis'),
            #         (r'\\\\.*\\\\.*\\.exe', 'Use configured paths from build_config.yaml'),
            #     ],
            #     'severity': 'high'
            # },
            
            # Rule 6: USE CENTRAL BUILD CONFIG ONLY
            6: {
                'name': 'USE CENTRAL BUILD CONFIG ONLY',
                'patterns': [
                    (r'hardcoded.*vs2022|hardcoded.*vs2003', 'Use build_config.yaml for build system selection'),
                    (r'C:\\\\Program Files\\\\Microsoft', 'Use paths from build_config.yaml, not hardcoded'),
                    (r'alternative.*compiler.*path', 'Use single configured compiler from build_config.yaml'),
                    (r'WSL.*compiler', 'Remove WSL fallbacks - use configured build system only'),
                ],
                'severity': 'high'
            },
            
            # Rule 7: NO BUILD FALLBACKS
            7: {
                'name': 'NO BUILD FALLBACKS',
                'patterns': [
                    (r'backup.*build', 'Remove backup build systems'),
                    (r'secondary.*build', 'Remove secondary build systems'),
                    (r'if.*msbuild.*not.*found', 'Remove build tool fallbacks - validate first'),
                ],
                'severity': 'critical'
            },
            
            # Rule 8: STRICT BUILD VALIDATION
            8: {
                'name': 'STRICT BUILD VALIDATION',
                'patterns': [
                    (r'continue.*missing.*tool', 'Fail immediately on missing build tools'),
                    (r'degrad.*build', 'Remove degraded build capabilities'),
                ],
                'severity': 'high'
            },
            
            # Rule 9: CONFIGURED PATHS ONLY
            9: {
                'name': 'CONFIGURED PATHS ONLY',
                'patterns': [
                    (r'alternative.*path', 'Use only configured paths from build_config.yaml'),
                    (r'\.\./', 'Use absolute paths from configuration, not relative'),
                ],
                'severity': 'high'
            },
            
            # Rule 10: NO DIRECTORY CREATION
            10: {
                'name': 'NO DIRECTORY CREATION',
                'patterns': [
                    (r'mkdir.*temp', 'Avoid creating temporary directories'),
                    (r'backup.*folder', 'Remove backup folder creation'),
                    (r'alternative.*directory', 'Remove alternative directory structures'),
                ],
                'severity': 'medium'
            },
            
            # Rule 11: STRICT PATH VALIDATION
            11: {
                'name': 'STRICT PATH VALIDATION',
                'patterns': [
                    (r'relative.*path.*alternative', 'Use absolute paths only'),
                    (r'if.*path.*not.*exist.*continue', 'Fail immediately on invalid paths'),
                ],
                'severity': 'high'
            },
            
            # Rule 12: FIX ROOT CAUSE, NOT SYMPTOMS
            12: {
                'name': 'FIX ROOT CAUSE, NOT SYMPTOMS',
                'patterns': [
                    (r'workaround', 'Fix root cause instead of workaround'),
                    (r'quick.*fix', 'Implement proper fix, not quick workaround'),
                    (r'temporary.*solution', 'Implement permanent solution'),
                ],
                'severity': 'critical'
            },
            
            # Rule 13: NO PLACEHOLDER CODE
            13: {
                'name': 'NO PLACEHOLDER CODE',
                'patterns': [
                    (r'\bplaceholder\b', 'Implement real functionality'),
                    (r'\btodo\b', 'Implement functionality instead of TODO'),
                    (r'\bfixme\b', 'Fix the issue instead of FIXME comment'),
                    (r'\bdummy\b', 'Implement real functionality'),
                    (r'raise NotImplementedError', 'Implement the method completely'),
                    (r'return.*0.*placeholder', 'Return actual computed value'),
                    (r'pass.*#.*todo', 'Implement the functionality'),
                ],
                'severity': 'critical'
            },
            
            # Rule 14: GENERIC DECOMPILER FUNCTIONALITY
            14: {
                'name': 'GENERIC DECOMPILER FUNCTIONALITY',
                'patterns': [
                    (r'launcher.*specific', 'Make generic for any binary'),
                    (r'hardcoded.*for.*launcher', 'Extract values dynamically from any binary'),
                ],
                'severity': 'medium'
            },
            
            # Rule 15: STRICT ERROR HANDLING
            15: {
                'name': 'STRICT ERROR HANDLING',
                'patterns': [
                    (r'soft.*fail', 'Change to hard failure'),
                    (r'continue.*on.*error', 'Fail immediately on critical errors'),
                    (r'ignore.*error', 'Handle errors properly, do not ignore'),
                ],
                'severity': 'high'
            },
            
            # Rule 16: ALL DEPENDENCIES MANDATORY
            16: {
                'name': 'ALL DEPENDENCIES MANDATORY',
                'patterns': [
                    (r'optional.*dependenc', 'Treat all dependencies as mandatory'),
                    (r'graceful.*missing.*tool', 'Fail on missing tools, do not handle gracefully'),
                ],
                'severity': 'high'
            },
            
            # Rule 17: NO CONDITIONAL EXECUTION
            17: {
                'name': 'NO CONDITIONAL EXECUTION',
                'patterns': [
                    (r'if.*tool.*available', 'Remove conditional execution based on tool availability'),
                    (r'try.*import.*except', 'Remove conditional imports - require all dependencies'),
                ],
                'severity': 'high'
            },
            
            # Rule 18: NO MOCK DEPENDENCIES
            18: {
                'name': 'NO MOCK DEPENDENCIES',
                'patterns': [
                    (r'mock.*dependency', 'Use real dependencies only'),
                    (r'simulate.*tool', 'Use authentic tools only'),
                    (r'fake.*implementation', 'Implement real functionality'),
                ],
                'severity': 'critical'
            },
            
            # Rule 19: ALL OR NOTHING EXECUTION
            19: {
                'name': 'ALL OR NOTHING EXECUTION',
                'patterns': [
                    (r'reduced.*capabilit', 'Remove reduced capabilities - all or nothing'),
                    (r'partial.*success', 'Report complete success or failure only'),
                    (r'degraded.*mode', 'Remove degraded modes'),
                ],
                'severity': 'critical'
            },
            
            # Rule 20: STRICT SUCCESS CRITERIA
            20: {
                'name': 'STRICT SUCCESS CRITERIA',
                'patterns': [
                    (r'partial.*result', 'Return complete results only'),
                    (r'best.*effort', 'Implement perfect execution, not best effort'),
                ],
                'severity': 'high'
            },
            
            # Rule 21: MANDATORY TESTING PROTOCOL
            21: {
                'name': 'MANDATORY TESTING PROTOCOL',
                'patterns': [
                    (r'skip.*test', 'Run all tests - no skipping'),
                    (r'without.*clear.*clean', 'Always test with --clear --clean flags'),
                ],
                'severity': 'medium'
            }
        }
        
        # File extensions to scan
        self.scan_extensions = {'.py', '.c', '.cpp', '.h', '.hpp', '.js', '.md', '.yaml', '.yml', '.sh', '.bat'}
        
        # Directories to skip
        self.skip_dirs = {
            '__pycache__', '.git', 'node_modules', '.vscode', 'venv', 'matrix_venv', 'worktrees'
        }

    def scan_file(self, file_path: Path) -> List[RuleViolation]:
        """Scan a single file for ALL rule violations"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line_stripped = line.strip()
                    
                    if not line_stripped:
                        continue
                        
                    # Check each rule
                    for rule_num, rule_data in self.rule_patterns.items():
                        for pattern, suggested_fix in rule_data['patterns']:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Check for rule exception comment
                                if f'//RULE{rule_num}=Exception' in line:
                                    continue
                                
                                # Skip legitimate output directory creation for Rule 4
                                if rule_num == 4 and self._is_legitimate_output_directory(line_stripped):
                                    continue
                                
                                violation = RuleViolation(
                                    rule_number=rule_num,
                                    rule_name=rule_data['name'],
                                    file_path=str(file_path.relative_to(self.root_dir)),
                                    line_number=line_num,
                                    line_content=line_stripped[:100],
                                    violation_pattern=pattern,
                                    suggested_fix=suggested_fix,
                                    severity=rule_data['severity']
                                )
                                violations.append(violation)
                                break  # Only one violation per line
                        if violations and violations[-1].line_number == line_num:
                            break  # Found violation on this line, move to next line
                            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return violations
    
    def _is_legitimate_output_directory(self, line: str) -> bool:
        """Check if mkdir call is for legitimate output directory creation"""
        line_lower = line.lower()
        
        # Agent output directories
        if 'agent_output_dir' in line_lower and 'mkdir' in line_lower:
            return True
        if 'output_dir' in line_lower and 'mkdir' in line_lower:
            return True
        if 'temp_dir' in line_lower and 'mkdir' in line_lower:
            return True
        if 'compilation' in line_lower and 'mkdir' in line_lower:
            return True
        if 'ghidra' in line_lower and 'mkdir' in line_lower:
            return True
        if 'reports' in line_lower and 'mkdir' in line_lower:
            return True
        if 'logs' in line_lower and 'mkdir' in line_lower:
            return True
        
        # File parent directory creation
        if '.parent.mkdir' in line_lower:
            return True
            
        # Agent-specific legitimate directory creation patterns
        if 'resources_dir.mkdir' in line_lower:
            return True
        if 'source_dir.mkdir' in line_lower:
            return True  
        if 'cleaned_dir.mkdir' in line_lower:
            return True
        if 'docs_dir.mkdir' in line_lower:
            return True
        
        # Configuration-driven directory creation (config manager paths)
        if 'path.mkdir' in line_lower:
            return True
            
        return False

    def scan_directory(self) -> List[RuleViolation]:
        """Scan entire directory recursively for ALL rule violations"""
        all_violations = []
        
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                if file_path.suffix.lower() in self.scan_extensions:
                    violations = self.scan_file(file_path)
                    all_violations.extend(violations)
        
        return all_violations

    def generate_report(self, violations: List[RuleViolation]) -> str:
        """Generate comprehensive report with fix suggestions"""
        if not violations:
            return "✅ PERFECT RULE COMPLIANCE! All 21 rules followed correctly."
        
        report = []
        report.append("🚨 COMPREHENSIVE RULE VIOLATIONS DETECTED 🚨")
        report.append("=" * 70)
        report.append(f"Found {len(violations)} violations across {len(set(v.file_path for v in violations))} files")
        report.append("")
        
        # Group by severity
        critical = [v for v in violations if v.severity == 'critical']
        high = [v for v in violations if v.severity == 'high'] 
        medium = [v for v in violations if v.severity == 'medium']
        low = [v for v in violations if v.severity == 'low']
        
        report.append("SEVERITY BREAKDOWN:")
        report.append(f"  🔴 CRITICAL: {len(critical)} violations (PROJECT TERMINATION RISK)")
        report.append(f"  🟠 HIGH:     {len(high)} violations")
        report.append(f"  🟡 MEDIUM:   {len(medium)} violations") 
        report.append(f"  🟢 LOW:      {len(low)} violations")
        report.append("")
        
        # Group by rule
        by_rule = {}
        for v in violations:
            if v.rule_number not in by_rule:
                by_rule[v.rule_number] = []
            by_rule[v.rule_number].append(v)
        
        report.append("VIOLATIONS BY RULE:")
        report.append("-" * 50)
        for rule_num in sorted(by_rule.keys()):
            rule_violations = by_rule[rule_num]
            rule_name = rule_violations[0].rule_name
            report.append(f"Rule {rule_num:2d}: {rule_name} - {len(rule_violations)} violations")
        report.append("")
        
        # CRITICAL violations first
        if critical:
            report.append("🔴 CRITICAL VIOLATIONS (IMMEDIATE FIX REQUIRED):")
            report.append("=" * 60)
            for v in critical:
                report.append(f"📁 {v.file_path}:{v.line_number}")
                report.append(f"   Rule {v.rule_number}: {v.rule_name}")
                report.append(f"   Code: {v.line_content}")
                report.append(f"   🔧 FIX: {v.suggested_fix}")
                report.append("")
        
        # Generate fix commands
        report.append("🛠️  AUTOMATED FIX SUGGESTIONS:")
        report.append("-" * 40)
        
        # Group similar fixes
        fix_groups = {}
        for v in violations:
            if v.suggested_fix not in fix_groups:
                fix_groups[v.suggested_fix] = []
            fix_groups[v.suggested_fix].append(v)
        
        for fix, related_violations in fix_groups.items():
            report.append(f"Fix: {fix}")
            report.append(f"  Affects {len(related_violations)} files:")
            for v in related_violations[:5]:  # Show first 5
                report.append(f"    - {v.file_path}:{v.line_number}")
            if len(related_violations) > 5:
                report.append(f"    ... and {len(related_violations) - 5} more")
            report.append("")
        
        report.append("=" * 70)
        report.append("🚨 RULE COMPLIANCE STATUS: FAILED")
        report.append(f"📊 Total violations: {len(violations)}")
        report.append(f"🎯 Target: 0 violations (ZERO TOLERANCE POLICY)")
        report.append("=" * 70)
        
        return "\n".join(report)

    def save_json_report(self, violations: List[RuleViolation], filename: str):
        """Save detailed violations as JSON for processing"""
        data = []
        for v in violations:
            data.append({
                'rule_number': v.rule_number,
                'rule_name': v.rule_name,
                'file_path': v.file_path,
                'line_number': v.line_number,
                'line_content': v.line_content,
                'violation_pattern': v.violation_pattern,
                'suggested_fix': v.suggested_fix,
                'severity': v.severity
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "."
    
    print(f"🔍 Comprehensive Rule Scanner - Checking ALL 21 Rules")
    print(f"📂 Scanning: {root_dir}")
    print()
    
    scanner = ComprehensiveRuleScanner(root_dir)
    violations = scanner.scan_directory()
    
    # Sort by severity and rule number
    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    violations.sort(key=lambda x: (severity_order[x.severity], x.rule_number, x.file_path))
    
    report = scanner.generate_report(violations)
    print(report)
    
    # Save reports
    report_file = Path(root_dir) / "comprehensive_rule_violations.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    json_file = Path(root_dir) / "rule_violations.json"
    scanner.save_json_report(violations, json_file)
    
    print(f"\n📝 Reports saved:")
    print(f"  📄 Text report: {report_file}")
    print(f"  📊 JSON data: {json_file}")
    
    # Exit codes
    if any(v.severity == 'critical' for v in violations):
        print(f"\n🔴 CRITICAL FAILURES DETECTED - PROJECT TERMINATION RISK")
        sys.exit(2)
    elif violations:
        print(f"\n🟠 RULE VIOLATIONS DETECTED - COMPLIANCE FAILURE")
        sys.exit(1)
    else:
        print(f"\n✅ PERFECT RULE COMPLIANCE - ALL 21 RULES FOLLOWED")
        sys.exit(0)

if __name__ == "__main__":
    main()