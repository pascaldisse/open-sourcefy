#!/usr/bin/env python3
"""
Environment Validation Script
Validates development environment setup for open-sourcefy project.
"""

import os
import sys
import shutil
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


def setup_logging():
    """Setup logging for environment validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class EnvironmentValidator:
    """Validates development environment for open-sourcefy."""
    
    def __init__(self):
        self.logger = setup_logging()
        self.validation_results = {}
        self.warnings = []
        self.errors = []
    
    def check_python_version(self) -> bool:
        """Check Python version meets requirements."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        is_valid = current_version >= required_version
        
        self.validation_results['python_version'] = {
            'required': f"{required_version[0]}.{required_version[1]}+",
            'current': f"{current_version[0]}.{current_version[1]}.{sys.version_info[2]}",
            'valid': is_valid
        }
        
        if is_valid:
            self.logger.info(f"✅ Python version: {sys.version}")
        else:
            error_msg = f"❌ Python {required_version[0]}.{required_version[1]}+ required, found {current_version[0]}.{current_version[1]}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)
        
        return is_valid
    
    def check_required_packages(self) -> bool:
        """Check if required Python packages are installed."""
        required_packages = [
            'pathlib',
            'json',
            'subprocess',
            'logging',
            'argparse',
            'hashlib',
            'shutil',
            'time',
            'os',
            'sys'
        ]
        
        optional_packages = [
            'pefile',
            'pyelftools', 
            'macholib',
            'capstone',
            'keystone-engine',
            'scikit-learn',
            'numpy',
            'pandas'
        ]
        
        missing_required = []
        missing_optional = []
        
        # Check required packages
        for package in required_packages:
            try:
                __import__(package)
                self.logger.info(f"✅ Required package available: {package}")
            except ImportError:
                missing_required.append(package)
                self.logger.error(f"❌ Missing required package: {package}")
        
        # Check optional packages
        for package in optional_packages:
            try:
                __import__(package)
                self.logger.info(f"✅ Optional package available: {package}")
            except ImportError:
                missing_optional.append(package)
                self.logger.warning(f"⚠️ Missing optional package: {package}")
        
        self.validation_results['packages'] = {
            'missing_required': missing_required,
            'missing_optional': missing_optional,
            'all_required_available': len(missing_required) == 0
        }
        
        if missing_required:
            self.errors.extend([f"Missing required package: {pkg}" for pkg in missing_required])
        
        if missing_optional:
            self.warnings.extend([f"Missing optional package: {pkg}" for pkg in missing_optional])
        
        return len(missing_required) == 0
    
    def check_ghidra_installation(self) -> bool:
        """Check Ghidra installation and configuration."""
        ghidra_home = os.environ.get('GHIDRA_HOME')
        
        # Check environment variable
        if not ghidra_home:
            # Try to find Ghidra in project directory
            project_root = Path(__file__).parent.parent
            project_ghidra = project_root / "ghidra"
            
            if project_ghidra.exists():
                ghidra_home = str(project_ghidra)
                self.logger.info(f"Found Ghidra in project directory: {ghidra_home}")
            else:
                # Check common installation paths
                common_paths = [
                    "/opt/ghidra",
                    "C:\\ghidra",
                    "/Applications/ghidra"
                ]
                
                for path in common_paths:
                    if Path(path).exists():
                        ghidra_home = path
                        break
        
        ghidra_valid = False
        if ghidra_home:
            ghidra_path = Path(ghidra_home)
            
            # Check if ghidraRun script exists
            ghidra_run = ghidra_path / "ghidraRun"
            ghidra_run_bat = ghidra_path / "ghidraRun.bat"
            
            if ghidra_run.exists() or ghidra_run_bat.exists():
                ghidra_valid = True
                self.logger.info(f"✅ Ghidra installation found: {ghidra_home}")
            else:
                self.logger.error(f"❌ Ghidra installation invalid: {ghidra_home}")
        else:
            self.logger.error("❌ Ghidra installation not found")
        
        self.validation_results['ghidra'] = {
            'ghidra_home': ghidra_home,
            'valid': ghidra_valid
        }
        
        if not ghidra_valid:
            self.errors.append("Ghidra installation not found or invalid")
        
        return ghidra_valid
    
    def check_java_installation(self) -> bool:
        """Check Java installation for Ghidra."""
        try:
            result = subprocess.run(['java', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse Java version from stderr (that's where java -version outputs)
                version_output = result.stderr
                java_valid = True
                self.logger.info(f"✅ Java installation found")
                self.logger.info(f"Java version output: {version_output.split()[0] if version_output else 'Unknown'}")
            else:
                java_valid = False
                self.logger.error("❌ Java installation not working")
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            java_valid = False
            self.logger.error("❌ Java installation not found")
        
        # Check JAVA_HOME if available
        java_home = os.environ.get('JAVA_HOME')
        
        self.validation_results['java'] = {
            'java_home': java_home,
            'valid': java_valid
        }
        
        if not java_valid:
            self.errors.append("Java installation not found or not working")
        
        return java_valid
    
    def check_build_tools(self) -> Dict[str, bool]:
        """Check availability of build tools."""
        build_tools = {
            'cmake': False,
            'make': False,
            'gcc': False,
            'cl': False,  # MSVC compiler
            'msbuild': False
        }
        
        # Check each tool
        for tool in build_tools.keys():
            try:
                result = subprocess.run([tool, '--version'], 
                                      capture_output=True, timeout=10)
                if result.returncode == 0:
                    build_tools[tool] = True
                    self.logger.info(f"✅ Build tool available: {tool}")
                else:
                    self.logger.warning(f"⚠️ Build tool not available: {tool}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.logger.warning(f"⚠️ Build tool not found: {tool}")
        
        self.validation_results['build_tools'] = build_tools
        
        # Check platform-specific requirements
        if platform.system() == 'Windows':
            if not (build_tools['cl'] or build_tools['msbuild']):
                self.warnings.append("No Windows build tools (MSVC/MSBuild) found")
        else:
            if not (build_tools['gcc'] or build_tools['make']):
                self.warnings.append("No Unix build tools (GCC/Make) found")
        
        return build_tools
    
    def check_project_structure(self) -> bool:
        """Check project directory structure."""
        project_root = Path(__file__).parent.parent
        
        required_dirs = [
            'src',
            'src/core',
            'src/core/agents',
            'src/ml',
            'src/utils'
        ]
        
        required_files = [
            'main.py',
            'requirements.txt',
            'CLAUDE.md',
            'src/core/agent_base.py'
        ]
        
        missing_dirs = []
        missing_files = []
        
        # Check directories
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
                self.logger.error(f"❌ Missing directory: {dir_path}")
            else:
                self.logger.info(f"✅ Directory exists: {dir_path}")
        
        # Check files
        for file_path in required_files:
            full_path = project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                self.logger.error(f"❌ Missing file: {file_path}")
            else:
                self.logger.info(f"✅ File exists: {file_path}")
        
        structure_valid = len(missing_dirs) == 0 and len(missing_files) == 0
        
        self.validation_results['project_structure'] = {
            'missing_directories': missing_dirs,
            'missing_files': missing_files,
            'valid': structure_valid
        }
        
        if not structure_valid:
            self.errors.extend([f"Missing directory: {d}" for d in missing_dirs])
            self.errors.extend([f"Missing file: {f}" for f in missing_files])
        
        return structure_valid
    
    def check_output_directory_compliance(self) -> bool:
        """Check that output directory structure is proper."""
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "output"
        
        # Check if output directory exists
        if not output_dir.exists():
            self.logger.warning("⚠️ Output directory doesn't exist - will be created when needed")
            output_valid = True  # Not an error, just doesn't exist yet
        else:
            # Check for files outside output directory that should be inside
            problem_files = []
            
            # Common file patterns that should be in output/
            check_patterns = ['*.log', '*.tmp', '*.json', '*.xml']
            
            for pattern in check_patterns:
                for file_path in project_root.glob(pattern):
                    if not str(file_path.resolve()).startswith(str(output_dir.resolve())):
                        problem_files.append(str(file_path))
            
            if problem_files:
                self.logger.warning(f"⚠️ Found {len(problem_files)} files outside output/ directory")
                for file_path in problem_files[:5]:  # Show first 5
                    self.logger.warning(f"  {file_path}")
                if len(problem_files) > 5:
                    self.logger.warning(f"  ... and {len(problem_files) - 5} more")
                output_valid = False
            else:
                self.logger.info("✅ No output files found outside /output/ directory")
                output_valid = True
        
        self.validation_results['output_compliance'] = {
            'valid': output_valid,
            'output_dir_exists': output_dir.exists()
        }
        
        return output_valid
    
    def run_full_validation(self) -> Dict:
        """Run complete environment validation."""
        self.logger.info("Starting environment validation...")
        
        # Run all checks
        python_ok = self.check_python_version()
        packages_ok = self.check_required_packages()
        ghidra_ok = self.check_ghidra_installation()
        java_ok = self.check_java_installation()
        build_tools = self.check_build_tools()
        structure_ok = self.check_project_structure()
        output_ok = self.check_output_directory_compliance()
        
        # Overall validation status
        critical_checks = [python_ok, packages_ok, structure_ok]
        validation_passed = all(critical_checks)
        
        self.validation_results['overall'] = {
            'validation_passed': validation_passed,
            'critical_issues': len(self.errors),
            'warnings': len(self.warnings),
            'ready_for_development': validation_passed and ghidra_ok and java_ok
        }
        
        # Print summary
        self.print_validation_summary()
        
        return self.validation_results
    
    def print_validation_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("ENVIRONMENT VALIDATION SUMMARY")
        print("="*60)
        
        if self.validation_results['overall']['validation_passed']:
            print("🎉 Environment validation PASSED")
        else:
            print("❌ Environment validation FAILED")
        
        print(f"\nCritical Issues: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            print("\n❌ CRITICAL ISSUES:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        print(f"\nReady for development: {'Yes' if self.validation_results['overall']['ready_for_development'] else 'No'}")
        print("="*60)


def generate_setup_instructions(validation_results: Dict) -> str:
    """Generate setup instructions based on validation results."""
    instructions = []
    
    # Python version issues
    if not validation_results.get('python_version', {}).get('valid', False):
        instructions.append("1. Install Python 3.8 or higher")
        instructions.append("   - Download from https://python.org")
        instructions.append("   - Or use package manager: apt install python3.8 / brew install python3")
    
    # Missing packages
    missing_required = validation_results.get('packages', {}).get('missing_required', [])
    missing_optional = validation_results.get('packages', {}).get('missing_optional', [])
    
    if missing_required or missing_optional:
        instructions.append("2. Install missing Python packages:")
        instructions.append("   pip install -r requirements.txt")
        
        if missing_required:
            instructions.append(f"   # Required: {', '.join(missing_required)}")
        if missing_optional:
            instructions.append(f"   # Optional: {', '.join(missing_optional)}")
    
    # Ghidra installation
    if not validation_results.get('ghidra', {}).get('valid', False):
        instructions.append("3. Install Ghidra:")
        instructions.append("   - Download from https://ghidra-sre.org/")
        instructions.append("   - Extract to a directory")
        instructions.append("   - Set GHIDRA_HOME environment variable")
        instructions.append("   - Or place in project/ghidra/ directory")
    
    # Java installation
    if not validation_results.get('java', {}).get('valid', False):
        instructions.append("4. Install Java 17+:")
        instructions.append("   - Download from https://openjdk.org/")
        instructions.append("   - Or use package manager: apt install openjdk-17-jdk")
        instructions.append("   - Set JAVA_HOME environment variable")
    
    # Build tools
    build_tools = validation_results.get('build_tools', {})
    if not any(build_tools.values()):
        if platform.system() == 'Windows':
            instructions.append("5. Install Windows build tools:")
            instructions.append("   - Visual Studio 2019/2022 with MSVC")
            instructions.append("   - Or Build Tools for Visual Studio")
        else:
            instructions.append("5. Install build tools:")
            instructions.append("   - apt install build-essential cmake (Ubuntu/Debian)")
            instructions.append("   - brew install cmake gcc (macOS)")
    
    if instructions:
        return "\nSETUP INSTRUCTIONS:\n" + "\n".join(instructions) + "\n"
    else:
        return "\n✅ Environment is properly configured!\n"


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Environment Validation for open-sourcefy')
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--setup-help', action='store_true', help='Show setup instructions')
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    validator = EnvironmentValidator()
    results = validator.run_full_validation()
    
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    
    if args.setup_help:
        instructions = generate_setup_instructions(results)
        print(instructions)
    
    # Exit with error code if validation failed
    if not results['overall']['validation_passed']:
        sys.exit(1)


if __name__ == '__main__':
    main()