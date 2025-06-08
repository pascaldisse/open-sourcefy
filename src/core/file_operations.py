#!/usr/bin/env python3
"""
File Operations Automation Script
Automates common file operations identified in prompts that don't require LLM.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


def setup_logging():
    """Setup logging for file operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def validate_output_path(path: Path, output_root: Path) -> Path:
    """Validate that path is under /output/ directory."""
    path_resolved = path.resolve()
    output_root_resolved = output_root.resolve()
    
    if not str(path_resolved).startswith(str(output_root_resolved)):
        raise ValueError(f"Path {path} is not under /output/ directory")
    
    return path_resolved


def create_output_structure(base_output_dir: str) -> Dict[str, Path]:
    """Create standardized output directory structure."""
    logger = setup_logging()
    
    output_root = Path(base_output_dir).resolve()
    
    # Define directory structure from CLAUDE.md
    directories = {
        'root': output_root,
        'agents': output_root / 'agents',
        'ghidra': output_root / 'ghidra',
        'compilation': output_root / 'compilation',
        'reports': output_root / 'reports',
        'logs': output_root / 'logs',
        'temp': output_root / 'temp',
        'tests': output_root / 'tests'
    }
    
    # Create directories
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    return directories


def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """Calculate hash of a file."""
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def get_file_metadata(file_path: Path) -> Dict:
    """Get comprehensive file metadata."""
    stat = file_path.stat()
    
    metadata = {
        'path': str(file_path),
        'name': file_path.name,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'created': stat.st_ctime,
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir(),
        'exists': file_path.exists()
    }
    
    if file_path.is_file():
        metadata['hash_sha256'] = calculate_file_hash(file_path, 'sha256')
        metadata['hash_md5'] = calculate_file_hash(file_path, 'md5')
    
    return metadata


def cleanup_temp_files(temp_dir: Path, max_age_hours: int = 24) -> List[str]:
    """Clean up old temporary files."""
    logger = setup_logging()
    import time
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_files = []
    
    if not temp_dir.exists():
        return cleaned_files
    
    for file_path in temp_dir.rglob('*'):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                try:
                    file_path.unlink()
                    cleaned_files.append(str(file_path))
                    logger.info(f"Cleaned up old temp file: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to clean up {file_path}: {e}")
    
    return cleaned_files


def copy_with_validation(src: Path, dst: Path, output_root: Path) -> bool:
    """Copy file with output directory validation."""
    logger = setup_logging()
    
    try:
        # Validate destination is under output directory
        validate_output_path(dst, output_root)
        
        # Create parent directories if needed
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(src, dst)
        logger.info(f"Copied {src} to {dst}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def find_files_by_pattern(directory: Path, pattern: str) -> List[Path]:
    """Find files matching a pattern."""
    return list(directory.rglob(pattern))


def find_files_by_extension(directory: Path, extensions: List[str]) -> List[Path]:
    """Find files with specific extensions."""
    files = []
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        files.extend(directory.rglob(f'*{ext}'))
    return files


def find_large_files(directory: Path, min_size_mb: float = 10) -> List[Tuple[Path, float]]:
    """Find files larger than specified size."""
    min_size_bytes = min_size_mb * 1024 * 1024
    large_files = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            size = file_path.stat().st_size
            if size > min_size_bytes:
                size_mb = size / (1024 * 1024)
                large_files.append((file_path, size_mb))
    
    return sorted(large_files, key=lambda x: x[1], reverse=True)


def validate_file_permissions(file_path: Path) -> Dict[str, bool]:
    """Check file permissions."""
    return {
        'readable': os.access(file_path, os.R_OK),
        'writable': os.access(file_path, os.W_OK),
        'executable': os.access(file_path, os.X_OK)
    }


def create_directory_report(directory: Path, output_file: Path) -> Dict:
    """Create comprehensive directory analysis report."""
    logger = setup_logging()
    
    report = {
        'directory': str(directory),
        'total_files': 0,
        'total_directories': 0,
        'total_size_bytes': 0,
        'file_types': {},
        'large_files': [],
        'recent_files': [],
        'metadata': {}
    }
    
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                report['total_files'] += 1
                size = item.stat().st_size
                report['total_size_bytes'] += size
                
                # Track file types
                suffix = item.suffix.lower()
                if suffix:
                    report['file_types'][suffix] = report['file_types'].get(suffix, 0) + 1
                
                # Track large files (>10MB)
                if size > 10 * 1024 * 1024:
                    report['large_files'].append({
                        'path': str(item),
                        'size_mb': size / (1024 * 1024)
                    })
                
                # Track recent files (modified in last 24 hours)
                import time
                if time.time() - item.stat().st_mtime < 86400:
                    report['recent_files'].append(str(item))
                    
            elif item.is_dir():
                report['total_directories'] += 1
        
        report['total_size_mb'] = report['total_size_bytes'] / (1024 * 1024)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Directory report saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to create directory report: {e}")
        raise
    
    return report


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='File Operations Automation')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create output structure
    create_parser = subparsers.add_parser('create-structure', help='Create output directory structure')
    create_parser.add_argument('output_dir', help='Base output directory')
    
    # Clean temp files
    clean_parser = subparsers.add_parser('clean-temp', help='Clean temporary files')
    clean_parser.add_argument('temp_dir', help='Temporary directory to clean')
    clean_parser.add_argument('--max-age', type=int, default=24, help='Max age in hours')
    
    # Directory report
    report_parser = subparsers.add_parser('directory-report', help='Create directory report')
    report_parser.add_argument('directory', help='Directory to analyze')
    report_parser.add_argument('output_file', help='Output report file')
    
    # Find files
    find_parser = subparsers.add_parser('find-files', help='Find files by pattern')
    find_parser.add_argument('directory', help='Directory to search')
    find_parser.add_argument('pattern', help='File pattern to search for')
    
    args = parser.parse_args()
    
    if args.command == 'create-structure':
        directories = create_output_structure(args.output_dir)
        print(f"Created directory structure in {args.output_dir}")
        for name, path in directories.items():
            print(f"  {name}: {path}")
            
    elif args.command == 'clean-temp':
        cleaned = cleanup_temp_files(Path(args.temp_dir), args.max_age)
        print(f"Cleaned {len(cleaned)} temporary files")
        
    elif args.command == 'directory-report':
        report = create_directory_report(Path(args.directory), Path(args.output_file))
        print(f"Directory report created: {args.output_file}")
        print(f"  Total files: {report['total_files']}")
        print(f"  Total size: {report['total_size_mb']:.2f} MB")
        
    elif args.command == 'find-files':
        files = find_files_by_pattern(Path(args.directory), args.pattern)
        print(f"Found {len(files)} files matching pattern '{args.pattern}':")
        for file_path in files:
            print(f"  {file_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()