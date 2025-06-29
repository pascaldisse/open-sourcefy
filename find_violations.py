#!/usr/bin/env python3
"""
Rule Violation Scanner
Searches the entire Matrix pipeline for violations of Rule 13 (No Placeholder Code)
and other mock/fake/placeholder implementations.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

class ViolationScanner:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.violations = []
        
        # Patterns that indicate Rule 13 violations
        self.violation_patterns = [
            # Direct violations
            r'\bmock\b',
            r'\bfake\b', 
            r'\bplaceholder\b',
            r'\bstub\b',
            r'\bdummy\b',
            r'\btodo\b',
            r'\bfixme\b',
            
            # Simulation/bypass patterns
            r'\bsimulat[e|ion]\b',
            r'\bassume\b.*works',
            r'\bassume\b.*succeed',
            r'bypass',
            r'workaround',
            
            # Return value patterns that suggest placeholders
            r'return 0;?\s*//.*placeholder',
            r'return 1;?\s*//.*placeholder', 
            r'return.*success.*placeholder',
            r'return.*fake',
            r'return.*mock',
            
            # Comment patterns
            r'//.*placeholder',
            r'//.*mock',
            r'//.*fake',
            r'//.*stub',
            r'//.*todo',
            r'//.*fixme',
            r'#.*placeholder',
            r'#.*mock',
            r'#.*fake',
            
            # Function/variable naming
            r'\bfake_\w+',
            r'\bmock_\w+',
            r'\bplaceholder_\w+',
            r'\bstub_\w+',
            r'\bdummy_\w+',
            
            # Implementation patterns
            r'raise NotImplementedError',
            r'pass\s*#.*placeholder',
            r'pass\s*#.*todo',
            r'NotImplemented',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.violation_patterns]
        
        # File extensions to scan
        self.scan_extensions = {'.py', '.c', '.cpp', '.h', '.hpp', '.js', '.md', '.yaml', '.yml'}
        
        # Directories to skip
        self.skip_dirs = {
            '__pycache__', 
            '.git', 
            'node_modules', 
            '.vscode', 
            'venv',
            'matrix_venv',
            'worktrees'
        }

    def scan_file(self, file_path: Path) -> List[Tuple[int, str, str]]:
        """Scan a single file for violations. Returns list of (line_num, line_content, pattern)"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line_stripped = line.strip()
                    
                    # Skip empty lines
                    if not line_stripped:
                        continue
                        
                    # Check each pattern
                    for pattern in self.compiled_patterns:
                        if pattern.search(line):
                            violations.append((line_num, line_stripped, pattern.pattern))
                            break  # Only report first match per line
                            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return violations

    def scan_directory(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """Scan entire directory recursively"""
        all_violations = {}
        
        for root, dirs, files in os.walk(self.root_dir):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in self.skip_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                # Only scan relevant file types
                if file_path.suffix.lower() in self.scan_extensions:
                    violations = self.scan_file(file_path)
                    if violations:
                        # Store relative path for cleaner output
                        rel_path = file_path.relative_to(self.root_dir)
                        all_violations[str(rel_path)] = violations
        
        return all_violations

    def generate_report(self, violations: Dict[str, List[Tuple[int, str, str]]]) -> str:
        """Generate a detailed report of all violations"""
        if not violations:
            return "✅ No Rule 13 violations found! All placeholder/mock/fake code has been eliminated."
        
        report = []
        report.append("🚨 RULE 13 VIOLATIONS DETECTED 🚨")
        report.append("=" * 60)
        report.append(f"Found violations in {len(violations)} files")
        report.append("")
        
        total_violations = sum(len(v) for v in violations.values())
        report.append(f"Total violations: {total_violations}")
        report.append("")
        
        # Group by violation type
        pattern_counts = {}
        for file_violations in violations.values():
            for _, _, pattern in file_violations:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        report.append("Violation Summary by Type:")
        report.append("-" * 30)
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {pattern}: {count} occurrences")
        report.append("")
        
        # Detailed file-by-file report
        report.append("Detailed Violations:")
        report.append("-" * 30)
        
        for file_path in sorted(violations.keys()):
            file_violations = violations[file_path]
            report.append(f"\n📁 {file_path} ({len(file_violations)} violations)")
            
            for line_num, line_content, pattern in file_violations:
                report.append(f"  Line {line_num:4d}: {line_content[:100]}")
                report.append(f"             Pattern: {pattern}")
                report.append("")
        
        report.append("=" * 60)
        report.append("ACTION REQUIRED: Fix all violations to comply with Rule 13")
        report.append("Rule 13: NO PLACEHOLDER CODE - no mock/fake/stub implementations")
        report.append("=" * 60)
        
        return "\n".join(report)

def main():
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "."
    
    print(f"🔍 Scanning Matrix pipeline for Rule 13 violations in: {root_dir}")
    print("Searching for: mock, fake, placeholder, stub, simulation patterns...")
    print()
    
    scanner = ViolationScanner(root_dir)
    violations = scanner.scan_directory()
    
    report = scanner.generate_report(violations)
    print(report)
    
    # Save report to file
    report_file = Path(root_dir) / "rule_violations_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📝 Report saved to: {report_file}")
    
    # Exit with error code if violations found
    if violations:
        print(f"\n❌ Found {len(violations)} files with violations - RULE 13 COMPLIANCE FAILURE")
        sys.exit(1)
    else:
        print(f"\n✅ RULE 13 COMPLIANCE SUCCESS - No violations found")
        sys.exit(0)

if __name__ == "__main__":
    main()