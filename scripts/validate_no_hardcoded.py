#!/usr/bin/env python3
"""
Validation script to check for hardcoded values in Matrix agents
Part of the production-ready quality assurance process
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class HardcodedValueValidator:
    """Validator to detect hardcoded values in Python source code"""
    
    # Patterns that indicate hardcoded values (forbidden in production code)
    FORBIDDEN_PATTERNS = {
        'absolute_paths_windows': {
            'pattern': r'["\']C:\\[^"\']*["\']',
            'description': 'Windows absolute paths',
            'examples': ['"C:\\Program Files\\app"', "'C:\\Windows\\System32'"]
        },
        'absolute_paths_unix': {
            'pattern': r'["\'](?:/home/|/usr/|/opt/|/etc/)[^"\']*["\']',
            'description': 'Unix absolute paths',
            'examples': ['/home/user', '/usr/bin/python', '/etc/config']
        },
        'hardcoded_hosts': {
            'pattern': r'(?:localhost|127\.0\.0\.1|192\.168\.|10\.0\.)',
            'description': 'Hardcoded hostnames/IPs',
            'examples': ['localhost', '127.0.0.1', '192.168.1.1']
        },
        'hardcoded_timeouts': {
            'pattern': r'timeout\s*=\s*\d+(?!\s*[,)\]])',
            'description': 'Hardcoded timeout values',
            'examples': ['timeout = 300', 'timeout=60']
        },
        'hardcoded_retries': {
            'pattern': r'(?:max_retries?|retry_count)\s*=\s*\d+',
            'description': 'Hardcoded retry counts',
            'examples': ['max_retries = 3', 'retry_count=5']
        },
        'hardcoded_ports': {
            'pattern': r'(?:port|PORT)\s*=\s*\d{4,5}',
            'description': 'Hardcoded port numbers',
            'examples': ['port = 8080', 'PORT=3000']
        },
        'magic_numbers': {
            'pattern': r'(?<![\w.])\d{4,}(?![\w.])',
            'description': 'Magic numbers (4+ digits)',
            'examples': ['1024', '65536']
        },
        'hardcoded_credentials': {
            'pattern': r'(?:password|passwd|secret|key|token)\s*=\s*["\'][^"\']+["\']',
            'description': 'Hardcoded credentials',
            'examples': ['password = "secret"', 'api_key="12345"'],
            'severity': 'CRITICAL'
        },
        'print_statements': {
            'pattern': r'(?:^|\s)print\s*\(',
            'description': 'Print statements (use logging instead)',
            'examples': ['print("debug")', 'print(result)']
        },
        'bare_except': {
            'pattern': r'except\s*:',
            'description': 'Bare except clauses',
            'examples': ['except:', 'except: pass']
        }
    }
    
    # Exceptions - patterns that are allowed in specific contexts
    ALLOWED_EXCEPTIONS = {
        'test_files': {
            'patterns': ['test_', '_test.py', 'conftest.py'],
            'allowed_violations': ['hardcoded_timeouts', 'magic_numbers', 'print_statements']
        },
        'constants_files': {
            'patterns': ['constants.py', 'config.py', 'settings.py'],
            'allowed_violations': ['magic_numbers']
        },
        'documentation': {
            'patterns': ['"""', "'''", '#'],
            'allowed_violations': ['absolute_paths_windows', 'absolute_paths_unix', 'hardcoded_hosts']
        }
    }
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.violations = []
    
    def validate_file(self, file_path: Path) -> List[Dict]:
        """Validate a single Python file for hardcoded values"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return [{'error': f"Failed to read file: {e}"}]
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and docstrings for some patterns
            stripped_line = line.strip()
            if self._is_comment_or_docstring(stripped_line, lines, line_num - 1):
                continue
            
            # Check each forbidden pattern
            for pattern_name, pattern_info in self.FORBIDDEN_PATTERNS.items():
                if self._check_pattern_violation(
                    line, pattern_info['pattern'], file_path, pattern_name
                ):
                    severity = pattern_info.get('severity', 'ERROR')
                    violations.append({
                        'file': str(file_path),
                        'line': line_num,
                        'pattern': pattern_name,
                        'description': pattern_info['description'],
                        'content': line.strip(),
                        'severity': severity
                    })
        
        return violations
    
    def _is_comment_or_docstring(self, line: str, all_lines: List[str], line_index: int) -> bool:
        """Check if line is part of comment or docstring"""
        # Single line comments
        if line.startswith('#'):
            return True
        
        # Check if we're inside a multiline string/docstring
        # This is a simplified check - a full parser would be more accurate
        if '"""' in line or "'''" in line:
            return True
        
        return False
    
    def _check_pattern_violation(self, line: str, pattern: str, file_path: Path, pattern_name: str) -> bool:
        """Check if line violates a specific pattern"""
        # Check if this file type allows this violation
        if self._is_violation_allowed(file_path, pattern_name):
            return False
        
        return bool(re.search(pattern, line))
    
    def _is_violation_allowed(self, file_path: Path, pattern_name: str) -> bool:
        """Check if violation is allowed for this file type"""
        file_name = file_path.name
        file_str = str(file_path)
        
        for exception_type, exception_info in self.ALLOWED_EXCEPTIONS.items():
            # Check if file matches exception pattern
            for allowed_pattern in exception_info['patterns']:
                if allowed_pattern in file_name or allowed_pattern in file_str:
                    # Check if this specific violation is allowed
                    if pattern_name in exception_info.get('allowed_violations', []):
                        return True
        
        return False
    
    def validate_directory(self, directory: Path, pattern: str = "**/*.py") -> List[Dict]:
        """Validate all Python files in directory"""
        all_violations = []
        
        python_files = list(directory.glob(pattern))
        
        if not python_files:
            print(f"Warning: No Python files found in {directory}")
            return []
        
        for file_path in python_files:
            # Skip __pycache__ and other generated directories
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', '.pytest_cache']):
                continue
            
            file_violations = self.validate_file(file_path)
            all_violations.extend(file_violations)
        
        return all_violations
    
    def print_violations(self, violations: List[Dict]) -> None:
        """Print violations in a readable format"""
        if not violations:
            print("✅ No hardcoded values detected!")
            return
        
        # Group by severity
        critical_violations = [v for v in violations if v.get('severity') == 'CRITICAL']
        error_violations = [v for v in violations if v.get('severity', 'ERROR') == 'ERROR']
        
        if critical_violations:
            print("🚨 CRITICAL VIOLATIONS (Security Risk):")
            for violation in critical_violations:
                self._print_violation(violation)
            print()
        
        if error_violations:
            print("❌ ERROR VIOLATIONS:")
            for violation in error_violations:
                self._print_violation(violation)
            print()
        
        # Summary
        total = len(violations)
        critical_count = len(critical_violations)
        error_count = len(error_violations)
        
        print(f"📊 SUMMARY: {total} violations found")
        if critical_count > 0:
            print(f"   🚨 Critical: {critical_count}")
        if error_count > 0:
            print(f"   ❌ Errors: {error_count}")
    
    def _print_violation(self, violation: Dict) -> None:
        """Print a single violation"""
        print(f"  {violation['file']}:{violation['line']}")
        print(f"    Pattern: {violation['pattern']} ({violation['description']})")
        print(f"    Content: {violation['content']}")
        print()
    
    def get_exit_code(self, violations: List[Dict]) -> int:
        """Get appropriate exit code based on violations"""
        if not violations:
            return 0
        
        # Critical violations always cause failure
        critical_violations = [v for v in violations if v.get('severity') == 'CRITICAL']
        if critical_violations:
            return 2  # Critical failure
        
        # In strict mode, any violation causes failure
        if self.strict_mode:
            return 1  # General failure
        
        return 0  # Pass with warnings


def main():
    """Main entry point for validation script"""
    parser = argparse.ArgumentParser(
        description="Validate Python code for hardcoded values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate entire source directory
  python validate_no_hardcoded.py src/
  
  # Validate specific file
  python validate_no_hardcoded.py src/core/agents_v2/agent01_sentinel.py
  
  # Non-strict mode (warnings only)
  python validate_no_hardcoded.py --no-strict src/
        """
    )
    
    parser.add_argument(
        'path',
        type=Path,
        help='Path to file or directory to validate'
    )
    
    parser.add_argument(
        '--no-strict',
        action='store_true',
        help='Non-strict mode: violations cause warnings, not errors'
    )
    
    parser.add_argument(
        '--pattern',
        default='**/*.py',
        help='File pattern for directory validation (default: **/*.py)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist")
        return 1
    
    # Create validator
    validator = HardcodedValueValidator(strict_mode=not args.no_strict)
    
    # Validate files
    if args.path.is_file():
        violations = validator.validate_file(args.path)
    else:
        violations = validator.validate_directory(args.path, args.pattern)
    
    # Print results
    validator.print_violations(violations)
    
    # Return appropriate exit code
    return validator.get_exit_code(violations)


if __name__ == '__main__':
    sys.exit(main())