"""
File System Utilities for Open-Sourcefy Matrix Pipeline
Shared utilities for file operations, path handling, and directory management
"""

import os
import shutil
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import hashlib
import time

# Optional imports with fallbacks
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class FileManager:
    """File system operations manager with safe file handling"""
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        self.logger = logging.getLogger("FileManager")
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.temp_dirs: List[Path] = []
        
    def __del__(self):
        """Cleanup temporary directories on destruction"""
        self.cleanup_temp_dirs()
    
    def create_directory(self, path: Union[str, Path], exist_ok: bool = True) -> Path:
        """Create directory with proper error handling"""
        path = Path(path)
        
        try:
            if not path.is_absolute():
                path = self.base_dir / path
            
            path.mkdir(parents=True, exist_ok=exist_ok)
            self.logger.debug(f"Created directory: {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Failed to create directory {path}: {e}")
            raise
    
    def create_temp_directory(self, prefix: str = "matrix_", suffix: str = "") -> Path:
        """Create temporary directory and track for cleanup"""
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
            self.temp_dirs.append(temp_dir)
            self.logger.debug(f"Created temporary directory: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            self.logger.error(f"Failed to create temporary directory: {e}")
            raise
    
    def cleanup_temp_dirs(self):
        """Clean up all tracked temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")
        
        self.temp_dirs.clear()
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path], 
                  create_dirs: bool = True) -> Path:
        """Copy file with directory creation"""
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        
        if create_dirs:
            dst.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(src, dst)
            self.logger.debug(f"Copied file: {src} -> {dst}")
            return dst
            
        except Exception as e:
            self.logger.error(f"Failed to copy file {src} to {dst}: {e}")
            raise
    
    def move_file(self, src: Union[str, Path], dst: Union[str, Path], 
                  create_dirs: bool = True) -> Path:
        """Move file with directory creation"""
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        
        if create_dirs:
            dst.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.move(str(src), str(dst))
            self.logger.debug(f"Moved file: {src} -> {dst}")
            return dst
            
        except Exception as e:
            self.logger.error(f"Failed to move file {src} to {dst}: {e}")
            raise
    
    def delete_file(self, path: Union[str, Path], missing_ok: bool = True):
        """Delete file with error handling"""
        path = Path(path)
        
        try:
            if path.exists():
                path.unlink()
                self.logger.debug(f"Deleted file: {path}")
            elif not missing_ok:
                raise FileNotFoundError(f"File not found: {path}")
                
        except Exception as e:
            self.logger.error(f"Failed to delete file {path}: {e}")
            if not missing_ok:
                raise
    
    def delete_directory(self, path: Union[str, Path], missing_ok: bool = True):
        """Delete directory and all contents"""
        path = Path(path)
        
        try:
            if path.exists():
                shutil.rmtree(path)
                self.logger.debug(f"Deleted directory: {path}")
            elif not missing_ok:
                raise FileNotFoundError(f"Directory not found: {path}")
                
        except Exception as e:
            self.logger.error(f"Failed to delete directory {path}: {e}")
            if not missing_ok:
                raise
    
    def read_text_file(self, path: Union[str, Path], encoding: str = 'utf-8') -> str:
        """Read text file with encoding handling"""
        path = Path(path)
        
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            # Try different encodings
            for fallback_encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    return path.read_text(encoding=fallback_encoding)
                except UnicodeDecodeError:
                    continue
            raise
        except Exception as e:
            self.logger.error(f"Failed to read file {path}: {e}")
            raise
    
    def write_text_file(self, path: Union[str, Path], content: str, 
                       encoding: str = 'utf-8', create_dirs: bool = True):
        """Write text file with directory creation"""
        path = Path(path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            path.write_text(content, encoding=encoding)
            self.logger.debug(f"Wrote text file: {path}")
        except Exception as e:
            self.logger.error(f"Failed to write file {path}: {e}")
            raise
    
    def read_binary_file(self, path: Union[str, Path]) -> bytes:
        """Read binary file"""
        path = Path(path)
        
        try:
            return path.read_bytes()
        except Exception as e:
            self.logger.error(f"Failed to read binary file {path}: {e}")
            raise
    
    def write_binary_file(self, path: Union[str, Path], content: bytes, 
                         create_dirs: bool = True):
        """Write binary file with directory creation"""
        path = Path(path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            path.write_bytes(content)
            self.logger.debug(f"Wrote binary file: {path}")
        except Exception as e:
            self.logger.error(f"Failed to write binary file {path}: {e}")
            raise
    
    def ensure_output_structure(self, base_output_dir: Union[str, Path], 
                               structure: Optional[Dict[str, str]] = None) -> Dict[str, Path]:
        """Ensure output directory structure exists"""
        base_output_dir = Path(base_output_dir)
        
        if structure is None:
            structure = {
                'agents': 'agents',
                'ghidra': 'ghidra',
                'compilation': 'compilation',
                'reports': 'reports',
                'logs': 'logs',
                'temp': 'temp',
                'tests': 'tests'
            }
        
        output_paths = {}
        
        for key, subdir in structure.items():
            dir_path = base_output_dir / subdir
            self.create_directory(dir_path)
            output_paths[key] = dir_path
        
        return output_paths


class JsonFileHandler:
    """JSON file operations with error handling"""
    
    def __init__(self):
        self.logger = logging.getLogger("JsonFileHandler")
    
    def read_json(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Read JSON file with error handling"""
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {path}: {e}")
            raise
    
    def write_json(self, path: Union[str, Path], data: Dict[str, Any], 
                   indent: int = 2, create_dirs: bool = True):
        """Write JSON file with formatting"""
        path = Path(path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            self.logger.debug(f"Wrote JSON file: {path}")
        except Exception as e:
            self.logger.error(f"Failed to write JSON file {path}: {e}")
            raise
    
    def update_json(self, path: Union[str, Path], updates: Dict[str, Any], 
                   create_if_missing: bool = True):
        """Update JSON file with new data"""
        path = Path(path)
        
        if path.exists():
            try:
                data = self.read_json(path)
                data.update(updates)
                self.write_json(path, data)
            except Exception as e:
                self.logger.error(f"Failed to update JSON file {path}: {e}")
                raise
        elif create_if_missing:
            self.write_json(path, updates)
        else:
            raise FileNotFoundError(f"JSON file not found: {path}")


class YamlFileHandler:
    """YAML file operations with optional dependency"""
    
    def __init__(self):
        self.logger = logging.getLogger("YamlFileHandler")
        
        if not HAS_YAML:
            self.logger.warning("PyYAML not available, YAML support disabled")
    
    def read_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Read YAML file"""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed, cannot read YAML files")
        
        path = Path(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in file {path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to read YAML file {path}: {e}")
            raise
    
    def write_yaml(self, path: Union[str, Path], data: Dict[str, Any], 
                   create_dirs: bool = True):
        """Write YAML file"""
        if not HAS_YAML:
            raise ImportError("PyYAML not installed, cannot write YAML files")
        
        path = Path(path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
            self.logger.debug(f"Wrote YAML file: {path}")
        except Exception as e:
            self.logger.error(f"Failed to write YAML file {path}: {e}")
            raise


class PathUtils:
    """Path utilities and validation"""
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """Normalize path for cross-platform compatibility"""
        return Path(path).resolve()
    
    @staticmethod
    def ensure_absolute(path: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
        """Ensure path is absolute"""
        path = Path(path)
        
        if path.is_absolute():
            return path
        
        if base_dir:
            return (base_dir / path).resolve()
        
        return path.resolve()
    
    @staticmethod
    def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
        """Get relative path from base"""
        try:
            return Path(path).relative_to(Path(base))
        except ValueError:
            # If paths are not related, return absolute path
            return Path(path).resolve()
    
    @staticmethod
    def safe_filename(filename: str, replacement: str = "_") -> str:
        """Create safe filename by replacing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        safe_name = filename
        
        for char in invalid_chars:
            safe_name = safe_name.replace(char, replacement)
        
        # Remove leading/trailing spaces and dots
        safe_name = safe_name.strip(' .')
        
        # Limit length
        if len(safe_name) > 255:
            name_part, ext_part = os.path.splitext(safe_name)
            max_name_len = 255 - len(ext_part)
            safe_name = name_part[:max_name_len] + ext_part
        
        return safe_name or "unnamed"
    
    @staticmethod
    def find_files(directory: Union[str, Path], pattern: str = "*", 
                   recursive: bool = True) -> List[Path]:
        """Find files matching pattern"""
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    @staticmethod
    def get_file_info(path: Union[str, Path]) -> Dict[str, Any]:
        """Get detailed file information"""
        path = Path(path)
        
        if not path.exists():
            return {}
        
        stat = path.stat()
        
        return {
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_file': path.is_file(),
            'is_dir': path.is_dir(),
            'is_symlink': path.is_symlink(),
            'absolute_path': str(path.resolve()),
            'parent': str(path.parent)
        }


class BackupManager:
    """File backup and versioning utilities"""
    
    def __init__(self, backup_dir: Optional[Union[str, Path]] = None):
        self.logger = logging.getLogger("BackupManager")
        self.backup_dir = Path(backup_dir) if backup_dir else Path.cwd() / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, path: Union[str, Path], 
                     backup_name: Optional[str] = None) -> Path:
        """Create backup of file or directory"""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if backup_name is None:
            timestamp = int(time.time())
            backup_name = f"{path.name}_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        
        try:
            if path.is_file():
                shutil.copy2(path, backup_path)
            else:
                shutil.copytree(path, backup_path)
            
            self.logger.info(f"Created backup: {path} -> {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup of {path}: {e}")
            raise
    
    def restore_backup(self, backup_path: Union[str, Path], 
                      restore_path: Union[str, Path]):
        """Restore from backup"""
        backup_path = Path(backup_path)
        restore_path = Path(restore_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        try:
            if restore_path.exists():
                if restore_path.is_file():
                    restore_path.unlink()
                else:
                    shutil.rmtree(restore_path)
            
            if backup_path.is_file():
                shutil.copy2(backup_path, restore_path)
            else:
                shutil.copytree(backup_path, restore_path)
            
            self.logger.info(f"Restored backup: {backup_path} -> {restore_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore backup {backup_path}: {e}")
            raise


# Utility functions
def calculate_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
    """Calculate hash of file"""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    hash_obj = hashlib.new(algorithm)
    
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def compare_files(file1: Union[str, Path], file2: Union[str, Path]) -> bool:
    """Compare two files for equality"""
    path1, path2 = Path(file1), Path(file2)
    
    if not (path1.exists() and path2.exists()):
        return False
    
    # Quick size check
    if path1.stat().st_size != path2.stat().st_size:
        return False
    
    # Hash comparison
    return calculate_file_hash(path1) == calculate_file_hash(path2)


def get_directory_size(directory: Union[str, Path]) -> int:
    """Calculate total size of directory"""
    directory = Path(directory)
    
    if not directory.exists():
        return 0
    
    total_size = 0
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                total_size += file_path.stat().st_size
            except (OSError, FileNotFoundError):
                # Skip files that can't be accessed
                pass
    
    return total_size


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_executable(name: str, paths: Optional[List[str]] = None) -> Optional[Path]:
    """Find executable in system PATH or custom paths"""
    if paths is None:
        paths = os.environ.get('PATH', '').split(os.pathsep)
    
    for path_str in paths:
        path = Path(path_str)
        if path.exists():
            for executable_path in path.glob(f"{name}*"):
                if executable_path.is_file() and os.access(executable_path, os.X_OK):
                    return executable_path
    
    return None


def cleanup_empty_directories(root_dir: Union[str, Path]):
    """Remove empty directories recursively"""
    root_dir = Path(root_dir)
    
    for directory in sorted(root_dir.rglob('*'), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()  # Only removes if empty
            except OSError:
                # Directory not empty, continue
                pass