"""
Utility Functions for Open-Sourcefy Matrix Pipeline

This module provides common utility functions used across the Matrix agent system
for file operations, validation, formatting, and other shared functionality.
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, IO
from datetime import datetime, timezone

__version__ = "2.0.0"
__all__ = [
    'ensure_directory', 'safe_file_read', 'safe_file_write', 'calculate_file_hash',
    'sanitize_filename', 'format_file_size', 'format_duration', 'validate_path',
    'extract_filename_without_extension', 'get_file_extension', 'is_binary_file',
    'create_backup_filename', 'cleanup_temporary_files', 'normalize_path',
    'get_timestamp_string', 'parse_version_string', 'compare_versions',
    'deep_merge_dicts', 'flatten_dict', 'safe_json_load', 'safe_json_save'
]


def ensure_directory(path: Union[str, Path], create_parents: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure exists
        create_parents: Whether to create parent directories
        
    Returns:
        Path object for the directory
        
    Raises:
        OSError: If directory creation fails
    """
    path_obj = Path(path)
    try:
        path_obj.mkdir(parents=create_parents, exist_ok=True)
        return path_obj
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {e}")


def safe_file_read(file_path: Union[str, Path], encoding: str = 'utf-8', 
                   fallback_encoding: str = 'latin-1') -> str:
    """
    Safely read a file with encoding fallback.
    
    Args:
        file_path: Path to file to read
        encoding: Primary encoding to try
        fallback_encoding: Fallback encoding if primary fails
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If both encodings fail
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try primary encoding first
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to alternative encoding
        try:
            with open(file_path, 'r', encoding=fallback_encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Last resort: read as binary and handle errors
            with open(file_path, 'rb') as f:
                content = f.read()
                return content.decode(encoding, errors='replace')


def safe_file_write(file_path: Union[str, Path], content: str, 
                    encoding: str = 'utf-8', backup: bool = True) -> None:
    """
    Safely write content to a file with optional backup.
    
    Args:
        file_path: Path to file to write
        content: Content to write
        encoding: Text encoding to use
        backup: Whether to create backup of existing file
        
    Raises:
        OSError: If write operation fails
    """
    file_path = Path(file_path)
    
    # Create backup if file exists and backup is requested
    if backup and file_path.exists():
        backup_path = create_backup_filename(file_path)
        file_path.rename(backup_path)
    
    # Ensure parent directory exists
    ensure_directory(file_path.parent)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    except OSError as e:
        raise OSError(f"Failed to write file {file_path}: {e}")


def calculate_file_hash(file_path: Union[str, Path], 
                       algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        
    Returns:
        Hexadecimal hash string
        
    Raises:
        ValueError: If algorithm is not supported
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        hash_obj = hashlib.new(algorithm)
    except ValueError:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        replacement: Character to replace invalid characters with
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters for Windows/Unix filesystems
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, replacement, filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure filename is not empty
    if not sanitized:
        sanitized = 'unnamed_file'
    
    # Limit length to prevent filesystem issues
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        max_name_len = 255 - len(ext)
        sanitized = name[:max_name_len] + ext
    
    return sanitized


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string (e.g., "2m 30s")
    """
    if seconds < 1:
        return f"{seconds:.2f}s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def validate_path(path: Union[str, Path], must_exist: bool = False,
                 must_be_file: bool = False, must_be_dir: bool = False) -> Path:
    """
    Validate and normalize a path.
    
    Args:
        path: Path to validate
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory
        
    Returns:
        Validated Path object
        
    Raises:
        ValueError: If validation fails
    """
    try:
        path_obj = Path(path).resolve()
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid path: {e}")
    
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    
    if must_be_file and path_obj.exists() and not path_obj.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    if must_be_dir and path_obj.exists() and not path_obj.is_dir():
        raise ValueError(f"Path is not a directory: {path}")
    
    return path_obj


def extract_filename_without_extension(file_path: Union[str, Path]) -> str:
    """Extract filename without extension."""
    return Path(file_path).stem


def get_file_extension(file_path: Union[str, Path]) -> str:
    """Get file extension (including the dot)."""
    return Path(file_path).suffix


def is_binary_file(file_path: Union[str, Path], sample_size: int = 8192) -> bool:
    """
    Check if a file is binary by examining first few bytes.
    
    Args:
        file_path: Path to file to check
        sample_size: Number of bytes to sample
        
    Returns:
        True if file appears to be binary
    """
    try:
        with open(file_path, 'rb') as f:
            sample = f.read(sample_size)
        
        # Check for null bytes (common in binary files)
        if b'\x00' in sample:
            return True
        
        # Check for non-printable characters
        text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
        non_text_count = sum(1 for byte in sample if byte not in text_chars)
        
        # If more than 30% non-text characters, consider it binary
        return (non_text_count / len(sample)) > 0.30 if sample else False
        
    except (IOError, OSError):
        return False


def create_backup_filename(file_path: Union[str, Path]) -> Path:
    """
    Create a backup filename with timestamp.
    
    Args:
        file_path: Original file path
        
    Returns:
        Path for backup file
    """
    file_path = Path(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if file_path.suffix:
        # Insert timestamp before extension
        stem = file_path.stem
        suffix = file_path.suffix
        backup_name = f"{stem}_backup_{timestamp}{suffix}"
    else:
        # Add timestamp to end
        backup_name = f"{file_path.name}_backup_{timestamp}"
    
    return file_path.parent / backup_name


def cleanup_temporary_files(temp_dir: Union[str, Path], 
                           pattern: str = "temp_*", 
                           max_age_hours: float = 24) -> int:
    """
    Clean up temporary files older than specified age.
    
    Args:
        temp_dir: Directory containing temporary files
        pattern: Glob pattern for files to clean
        max_age_hours: Maximum age in hours before cleanup
        
    Returns:
        Number of files cleaned up
    """
    temp_dir = Path(temp_dir)
    if not temp_dir.exists():
        return 0
    
    cleaned_count = 0
    max_age_seconds = max_age_hours * 3600
    current_time = datetime.now().timestamp()
    
    try:
        for file_path in temp_dir.glob(pattern):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                    except OSError:
                        continue  # Skip files that can't be deleted
    except OSError:
        pass  # Handle permission errors gracefully
    
    return cleaned_count


def normalize_path(path: Union[str, Path]) -> str:
    """
    Normalize path for cross-platform compatibility.
    
    Args:
        path: Path to normalize
        
    Returns:
        Normalized path string using forward slashes
    """
    return str(Path(path)).replace('\\', '/')


def get_timestamp_string(dt: Optional[datetime] = None, 
                        format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get formatted timestamp string.
    
    Args:
        dt: Datetime object (defaults to current time)
        format_str: Format string for timestamp
        
    Returns:
        Formatted timestamp string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime(format_str)


def parse_version_string(version: str) -> Tuple[int, ...]:
    """
    Parse version string into tuple of integers.
    
    Args:
        version: Version string (e.g., "2.1.3")
        
    Returns:
        Tuple of version components as integers
        
    Raises:
        ValueError: If version string is invalid
    """
    try:
        parts = version.split('.')
        return tuple(int(part) for part in parts)
    except ValueError:
        raise ValueError(f"Invalid version string: {version}")


def compare_versions(version1: str, version2: str) -> int:
    """
    Compare two version strings.
    
    Args:
        version1: First version string
        version2: Second version string
        
    Returns:
        -1 if version1 < version2, 0 if equal, 1 if version1 > version2
    """
    v1_parts = parse_version_string(version1)
    v2_parts = parse_version_string(version2)
    
    # Pad shorter version with zeros
    max_len = max(len(v1_parts), len(v2_parts))
    v1_padded = v1_parts + (0,) * (max_len - len(v1_parts))
    v2_padded = v2_parts + (0,) * (max_len - len(v2_parts))
    
    if v1_padded < v2_padded:
        return -1
    elif v1_padded > v2_padded:
        return 1
    else:
        return 0


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary using dot notation.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    
    return dict(items)


def safe_json_load(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load JSON file with error handling.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    """
    import json
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")


def safe_json_save(data: Any, file_path: Union[str, Path], 
                   indent: int = 2, ensure_ascii: bool = False) -> None:
    """
    Safely save data to JSON file.
    
    Args:
        data: Data to save as JSON
        file_path: Path to save JSON file
        indent: JSON indentation
        ensure_ascii: Whether to escape non-ASCII characters
        
    Raises:
        OSError: If write operation fails
        TypeError: If data is not JSON serializable
    """
    import json
    
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    except (OSError, TypeError) as e:
        raise type(e)(f"Failed to save JSON to {file_path}: {e}")


# Logging utilities
def setup_file_logger(name: str, log_file: Union[str, Path], 
                     level: int = logging.INFO) -> logging.Logger:
    """
    Set up a file logger with standard formatting.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    ensure_directory(Path(log_file).parent)
    
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger