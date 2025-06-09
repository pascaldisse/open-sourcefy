#!/usr/bin/env python3
"""
Fallback Code Removal Automation Script
Removes all fallback code from the open-sourcefy project and replaces with proper error handling.
"""

import os
import re
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Fallback patterns to search and replace
FALLBACK_PATTERNS = {
    'ghidra_timeout_fallback': {
        'search': r'except\s+TimeoutError:.*?fallback.*?\n.*?return.*?fallback',
        'replace': '''except TimeoutError:
    self.logger.error(f"Ghidra analysis timed out after {self.timeout}s")
    raise RuntimeError("Ghidra analysis timed out - pipeline requires successful Ghidra execution")''',
        'flags': re.DOTALL | re.MULTILINE
    },
    
    'ghidra_failure_fallback': {
        'search': r'if\s+not\s+success.*?:\s*\n.*?warning.*fallback.*\n.*?return.*?fallback',
        'replace': '''if not success:
    self.logger.error(f"Ghidra analysis failed: {output}")
    raise RuntimeError(f"Ghidra analysis failed: {output}")''',
        'flags': re.DOTALL | re.MULTILINE
    },
    
    'ai_unavailable_fallback': {
        'search': r'if\s+not\s+.*ai_enabled.*:\s*\n.*?return.*?fallback',
        'replace': '''if not self.ai_enabled:
    raise RuntimeError("AI system required for this agent - configure Claude CLI or use non-AI agents")''',
        'flags': re.DOTALL | re.MULTILINE
    },
    
    'dependency_missing_fallback': {
        'search': r'if\s+not\s+.*dependency.*:\s*\n.*?warning.*fallback.*\n.*?return.*?fallback',
        'replace': '''if not dependency_available:
    raise ValueError(f"Required dependency not satisfied - cannot proceed")''',
        'flags': re.DOTALL | re.MULTILINE
    },
    
    'quality_threshold_fallback': {
        'search': r'if\s+.*quality.*<.*threshold.*:\s*\n.*?warning.*proceeding.*\n.*?return',
        'replace': '''if quality_score < threshold:
    raise ValidationError(f"Quality score {quality_score} below required threshold {threshold}")''',
        'flags': re.DOTALL | re.MULTILINE
    },
    
    'semantic_decompilation_fallback': {
        'search': r'except\s+Exception.*?:\s*\n.*?warning.*fallback.*\n.*?return.*?fallback',
        'replace': '''except Exception as e:
    self.logger.error(f"Semantic decompilation failed: {e}")
    raise RuntimeError(f"Semantic decompilation failed: {e}")''',
        'flags': re.DOTALL | re.MULTILINE
    },
    
    'fallback_usage_warning': {
        'search': r'self\.logger\.warning\(.*?fallback.*?\)',
        'replace': 'self.logger.error("Pipeline failure - no fallback available")',
        'flags': re.DOTALL
    },
    
    'proceeding_with_fallback': {
        'search': r'proceeding\s+with\s+fallback',
        'replace': 'pipeline failure - requirement not met',
        'flags': re.IGNORECASE
    },
    
    'using_fallback': {
        'search': r'using\s+fallback',
        'replace': 'requirement not satisfied',
        'flags': re.IGNORECASE
    }
}

# Methods to remove entirely
FALLBACK_METHODS_TO_REMOVE = [
    '_generate_fallback_functions',
    '_create_fallback_analysis', 
    '_basic_analysis_fallback',
    '_fallback_static_reconstruction',
    '_generate_basic_analysis',
    '_simple_fallback_analysis',
    '_minimal_fallback_result',
    '_parse_ghidra_fallback',
    '_fallback_function_analysis',
    '_create_minimal_fallback'
]

# Fallback keywords that should trigger investigation
FALLBACK_KEYWORDS = [
    'fallback',
    'Fallback', 
    'proceeding with',
    'using basic',
    'degraded mode',
    'simplified analysis',
    'minimal functionality',
    'backup analysis',
    'alternative approach'
]

class FallbackRemover:
    """Main class for removing fallback code from the project"""
    
    def __init__(self, project_root: Path, output_dir: Path):
        self.project_root = project_root
        self.output_dir = output_dir
        self.backup_dir = output_dir / "fallback_removal_backup"
        self.report_file = output_dir / "fallback_removal_report.json"
        
        # Ensure output directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.scan_results = {}
        self.changes_made = {}
        
    def scan_project(self) -> Dict[str, Any]:
        """Scan entire project for fallback patterns"""
        print("🔍 Scanning project for fallback patterns...")
        
        python_files = list(self.project_root.rglob("*.py"))
        
        scan_summary = {
            'total_files': len(python_files),
            'files_with_fallbacks': 0,
            'total_fallback_instances': 0,
            'fallback_patterns_found': {},
            'files_details': {}
        }
        
        for file_path in python_files:
            # Skip virtual environment and __pycache__ directories
            if any(skip in str(file_path) for skip in ['venv', '__pycache__', '.git']):
                continue
                
            file_results = self.scan_file(file_path)
            if file_results['has_fallbacks']:
                scan_summary['files_with_fallbacks'] += 1
                scan_summary['total_fallback_instances'] += file_results['fallback_count']
                scan_summary['files_details'][str(file_path)] = file_results
                
                # Track pattern frequency
                for pattern in file_results['patterns_found']:
                    if pattern not in scan_summary['fallback_patterns_found']:
                        scan_summary['fallback_patterns_found'][pattern] = 0
                    scan_summary['fallback_patterns_found'][pattern] += 1
        
        self.scan_results = scan_summary
        
        print(f"📊 Scan complete:")
        print(f"   - Files scanned: {scan_summary['total_files']}")
        print(f"   - Files with fallbacks: {scan_summary['files_with_fallbacks']}")
        print(f"   - Total fallback instances: {scan_summary['total_fallback_instances']}")
        
        return scan_summary
    
    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single file for fallback patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {
                'file_path': str(file_path),
                'has_fallbacks': False,
                'fallback_count': 0,
                'patterns_found': [],
                'methods_to_remove': [],
                'keywords_found': [],
                'error': str(e)
            }
        
        patterns_found = []
        methods_to_remove = []
        keywords_found = []
        
        # Check for fallback patterns
        for pattern_name, pattern_info in FALLBACK_PATTERNS.items():
            if re.search(pattern_info['search'], content, pattern_info['flags']):
                patterns_found.append(pattern_name)
        
        # Check for methods to remove
        for method_name in FALLBACK_METHODS_TO_REMOVE:
            method_pattern = rf'def\s+{re.escape(method_name)}\s*\('
            if re.search(method_pattern, content):
                methods_to_remove.append(method_name)
        
        # Check for fallback keywords
        for keyword in FALLBACK_KEYWORDS:
            if keyword.lower() in content.lower():
                keywords_found.append(keyword)
        
        fallback_count = len(patterns_found) + len(methods_to_remove) + len(keywords_found)
        
        return {
            'file_path': str(file_path),
            'has_fallbacks': fallback_count > 0,
            'fallback_count': fallback_count,
            'patterns_found': patterns_found,
            'methods_to_remove': methods_to_remove,
            'keywords_found': keywords_found
        }
    
    def remove_fallbacks_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Remove all fallback code from a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return {
                'file_path': str(file_path),
                'success': False,
                'error': f"Failed to read file: {e}",
                'changes_made': []
            }
        
        modified_content = original_content
        changes_made = []
        
        # Remove fallback methods entirely
        for method_name in FALLBACK_METHODS_TO_REMOVE:
            # More comprehensive pattern to match entire method
            method_pattern = rf'(\s*)def\s+{re.escape(method_name)}\s*\([^)]*\):[^\\n]*\n((?:\1    .*\n)*)'
            if re.search(method_pattern, modified_content, re.MULTILINE):
                modified_content = re.sub(method_pattern, '', modified_content, flags=re.MULTILINE)
                changes_made.append(f"Removed method: {method_name}")
        
        # Replace fallback patterns with error handling
        for pattern_name, pattern_info in FALLBACK_PATTERNS.items():
            if re.search(pattern_info['search'], modified_content, pattern_info['flags']):
                old_content = modified_content
                modified_content = re.sub(
                    pattern_info['search'], 
                    pattern_info['replace'], 
                    modified_content, 
                    flags=pattern_info['flags']
                )
                if modified_content != old_content:
                    changes_made.append(f"Replaced pattern: {pattern_name}")
        
        # Remove fallback comments and documentation
        fallback_comment_patterns = [
            (r'#.*fallback.*\n', re.IGNORECASE),
            (r'""".*?fallback.*?"""', re.DOTALL | re.IGNORECASE),
            (r"'''.*?fallback.*?'''", re.DOTALL | re.IGNORECASE)
        ]
        
        for comment_pattern, flags in fallback_comment_patterns:
            if re.search(comment_pattern, modified_content, flags):
                old_content = modified_content
                modified_content = re.sub(comment_pattern, '', modified_content, flags=flags)
                if modified_content != old_content:
                    changes_made.append("Removed fallback comments")
        
        # Clean up extra whitespace created by removals
        modified_content = re.sub(r'\n\s*\n\s*\n', '\n\n', modified_content)
        
        return {
            'file_path': str(file_path),
            'success': True,
            'original_content': original_content,
            'modified_content': modified_content,
            'changes_made': changes_made,
            'has_changes': len(changes_made) > 0
        }
    
    def backup_file(self, file_path: Path) -> Path:
        """Create backup of original file"""
        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def apply_changes(self, create_backup: bool = True) -> Dict[str, Any]:
        """Apply fallback removal changes to all files"""
        print("🔧 Applying fallback removal changes...")
        
        files_changed = 0
        total_changes = 0
        errors = []
        
        for file_path_str, file_details in self.scan_results.get('files_details', {}).items():
            file_path = Path(file_path_str)
            
            if not file_details['has_fallbacks']:
                continue
            
            # Create backup if requested
            if create_backup:
                backup_path = self.backup_file(file_path)
                print(f"📋 Backed up: {file_path} -> {backup_path}")
            
            # Apply changes
            result = self.remove_fallbacks_from_file(file_path)
            
            if not result['success']:
                errors.append(result)
                continue
            
            if result['has_changes']:
                try:
                    # Write modified content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(result['modified_content'])
                    
                    files_changed += 1
                    total_changes += len(result['changes_made'])
                    self.changes_made[str(file_path)] = result
                    
                    print(f"✅ Modified: {file_path}")
                    for change in result['changes_made']:
                        print(f"    - {change}")
                        
                except Exception as e:
                    errors.append({
                        'file_path': str(file_path),
                        'error': f"Failed to write file: {e}"
                    })
        
        print(f"🎉 Changes applied:")
        print(f"   - Files modified: {files_changed}")
        print(f"   - Total changes: {total_changes}")
        print(f"   - Errors: {len(errors)}")
        
        return {
            'files_changed': files_changed,
            'total_changes': total_changes,
            'errors': errors
        }
    
    def validate_syntax(self) -> Dict[str, Any]:
        """Validate that all modified files still have valid Python syntax"""
        print("🔍 Validating Python syntax...")
        
        syntax_errors = []
        files_validated = 0
        
        for file_path_str in self.changes_made.keys():
            file_path = Path(file_path_str)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to compile the Python code
                compile(content, str(file_path), 'exec')
                files_validated += 1
                
            except SyntaxError as e:
                syntax_errors.append({
                    'file_path': str(file_path),
                    'error': str(e),
                    'line': e.lineno,
                    'column': e.offset
                })
            except Exception as e:
                syntax_errors.append({
                    'file_path': str(file_path),
                    'error': str(e)
                })
        
        if syntax_errors:
            print(f"❌ Syntax validation failed for {len(syntax_errors)} files:")
            for error in syntax_errors:
                print(f"    - {error['file_path']}: {error['error']}")
        else:
            print(f"✅ Syntax validation passed for {files_validated} files")
        
        return {
            'files_validated': files_validated,
            'syntax_errors': syntax_errors,
            'validation_passed': len(syntax_errors) == 0
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of fallback removal"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'scan_results': self.scan_results,
            'changes_applied': len(self.changes_made),
            'changes_details': self.changes_made,
            'validation_results': self.validate_syntax()
        }
        
        # Write report to output directory
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report generated: {self.report_file}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Remove fallback code from open-sourcefy project')
    parser.add_argument('--project-root', type=Path, default=Path('.'), 
                       help='Root directory of the project')
    parser.add_argument('--output-dir', type=Path, default=Path('./output'),
                       help='Output directory for reports and backups')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan for fallbacks, do not apply changes')
    parser.add_argument('--apply-changes', action='store_true',
                       help='Apply fallback removal changes')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate detailed report')
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    
    print(f"🚀 Fallback Removal Automation")
    print(f"   Project root: {project_root}")
    print(f"   Output directory: {output_dir}")
    print()
    
    # Initialize remover
    remover = FallbackRemover(project_root, output_dir)
    
    # Scan project
    scan_results = remover.scan_project()
    
    if scan_results['files_with_fallbacks'] == 0:
        print("✅ No fallback code found in project!")
        return 0
    
    # Apply changes if requested
    if args.apply_changes and not args.scan_only:
        apply_results = remover.apply_changes(create_backup=not args.no_backup)
        
        if apply_results['errors']:
            print(f"❌ {len(apply_results['errors'])} errors occurred during changes")
            for error in apply_results['errors']:
                print(f"    - {error.get('file_path', 'Unknown')}: {error.get('error', 'Unknown error')}")
    
    # Generate report if requested
    if args.generate_report:
        report = remover.generate_report()
        
        print(f"\n📋 Summary:")
        print(f"   - Files scanned: {report['scan_results']['total_files']}")
        print(f"   - Files with fallbacks: {report['scan_results']['files_with_fallbacks']}")
        print(f"   - Changes applied: {report['changes_applied']}")
        print(f"   - Syntax validation: {'✅ PASSED' if report['validation_results']['validation_passed'] else '❌ FAILED'}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())