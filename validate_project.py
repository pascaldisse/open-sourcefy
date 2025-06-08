#!/usr/bin/env python3
"""
Comprehensive Open-Sourcefy Project Validation
Validates the entire project structure, components, agents, and system integrity
"""

import sys
import os
import json
import time
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add project src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

@dataclass
class ValidationResult:
    """Result of a validation check"""
    component: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    message: str
    details: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

class ProjectValidator:
    """Comprehensive project validator"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        
    def log_result(self, component: str, status: str, message: str, 
                   details: List[str] = None, metrics: Dict[str, Any] = None,
                   execution_time: float = 0.0):
        """Log a validation result"""
        result = ValidationResult(
            component=component,
            status=status,
            message=message,
            details=details or [],
            metrics=metrics or {},
            execution_time=execution_time
        )
        self.results.append(result)
        
        # Print immediate feedback
        status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}
        print(f"{status_icon.get(status, '❓')} {component}: {message}")
        if details:
            for detail in details[:3]:  # Show first 3 details
                print(f"   • {detail}")
    
    def validate_project_structure(self) -> bool:
        """Validate overall project structure"""
        print("\n🏗️ VALIDATING PROJECT STRUCTURE")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        # Check required directories
        required_dirs = [
            "src/core",
            "src/core/agents",
            "src/core/agents_v2", 
            "src/ml",
            "src/utils",
            "ghidra",
            "input",
            "output",
            "docs"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        if missing_dirs:
            self.log_result("Project Structure", "FAIL", 
                          f"Missing {len(missing_dirs)} required directories",
                          missing_dirs)
            success = False
        else:
            self.log_result("Project Structure", "PASS", 
                          "All required directories present")
        
        # Check required files
        required_files = [
            "main.py",
            "requirements.txt", 
            "CLAUDE.md",
            "README.md",
            "src/core/__init__.py",
            "src/core/agent_base.py",
            "src/core/config_manager.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.log_result("Required Files", "FAIL",
                          f"Missing {len(missing_files)} required files", 
                          missing_files)
            success = False
        else:
            self.log_result("Required Files", "PASS",
                          "All required files present")
        
        execution_time = time.time() - start_time
        return success
    
    def validate_python_syntax(self) -> bool:
        """Validate Python syntax for all Python files"""
        print("\n🐍 VALIDATING PYTHON SYNTAX")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        python_files = list(project_root.rglob("*.py"))
        syntax_errors = []
        
        for py_file in python_files:
            # Skip certain directories
            if any(skip in str(py_file) for skip in ["venv", "__pycache__", ".git"]):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                compile(source, str(py_file), 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{py_file.relative_to(project_root)}: {e}")
                success = False
            except Exception as e:
                # File encoding or other issues
                syntax_errors.append(f"{py_file.relative_to(project_root)}: {type(e).__name__}: {e}")
        
        if syntax_errors:
            self.log_result("Python Syntax", "FAIL",
                          f"Syntax errors in {len(syntax_errors)} files",
                          syntax_errors[:10])  # Show first 10 errors
        else:
            self.log_result("Python Syntax", "PASS",
                          f"All {len(python_files)} Python files have valid syntax")
        
        execution_time = time.time() - start_time
        return success
    
    def validate_imports(self) -> bool:
        """Validate that core modules can be imported"""
        print("\n📦 VALIDATING IMPORTS")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        # Core modules to test
        core_modules = [
            "core.agent_base",
            "core.config_manager", 
            "core.parallel_executor",
            "core.agents",
            "core.agents_v2"
        ]
        
        import_errors = []
        successful_imports = []
        
        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
                successful_imports.append(module_name)
            except ImportError as e:
                import_errors.append(f"{module_name}: {e}")
                success = False
            except Exception as e:
                import_errors.append(f"{module_name}: {type(e).__name__}: {e}")
                success = False
        
        if import_errors:
            self.log_result("Core Imports", "FAIL",
                          f"Failed to import {len(import_errors)} modules",
                          import_errors)
        else:
            self.log_result("Core Imports", "PASS",
                          f"Successfully imported {len(successful_imports)} core modules")
        
        execution_time = time.time() - start_time
        return success
    
    def validate_agents(self) -> bool:
        """Validate all agent implementations"""
        print("\n🤖 VALIDATING AGENTS")
        print("=" * 50)
        
        start_time = time.time()
        overall_success = True
        
        # Test agents directory structure
        agents_dir = project_root / "src" / "core" / "agents"
        agents_v2_dir = project_root / "src" / "core" / "agents_v2"
        
        if not agents_dir.exists():
            self.log_result("Agents Directory", "FAIL", "src/core/agents directory missing")
            return False
        
        if not agents_v2_dir.exists():
            self.log_result("Agents V2 Directory", "FAIL", "src/core/agents_v2 directory missing")
            return False
        
        # Find all agent files
        agent_files = []
        for agents_path in [agents_dir, agents_v2_dir]:
            agent_files.extend(list(agents_path.glob("agent*.py")))
        
        agent_validation_results = {
            "importable": [],
            "instantiable": [],
            "executable": [],
            "errors": []
        }
        
        for agent_file in agent_files:
            agent_name = agent_file.stem
            
            try:
                # Extract agent module path
                if "agents_v2" in str(agent_file):
                    module_path = f"core.agents_v2.{agent_name}"
                else:
                    module_path = f"core.agents.{agent_name}"
                
                # Test import
                try:
                    module = importlib.import_module(module_path)
                    agent_validation_results["importable"].append(agent_name)
                    
                    # Find agent class
                    agent_class = None
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            hasattr(attr, 'execute') and 
                            hasattr(attr, 'agent_id')):
                            agent_class = attr
                            break
                    
                    if agent_class:
                        # Test instantiation
                        try:
                            agent_instance = agent_class()
                            agent_validation_results["instantiable"].append(agent_name)
                            
                            # Test basic execution with mock context
                            try:
                                mock_context = {
                                    'agent_results': {},
                                    'binary_path': 'test.exe',
                                    'output_paths': {'agents': '/tmp'}
                                }
                                result = agent_instance.execute(mock_context)
                                agent_validation_results["executable"].append(agent_name)
                                
                            except Exception as e:
                                agent_validation_results["errors"].append(f"{agent_name} execution: {e}")
                                
                        except Exception as e:
                            agent_validation_results["errors"].append(f"{agent_name} instantiation: {e}")
                    else:
                        agent_validation_results["errors"].append(f"{agent_name}: No agent class found")
                        
                except ImportError as e:
                    agent_validation_results["errors"].append(f"{agent_name} import: {e}")
                    
            except Exception as e:
                agent_validation_results["errors"].append(f"{agent_name}: {e}")
        
        # Report results
        total_agents = len(agent_files)
        
        if agent_validation_results["errors"]:
            self.log_result("Agent Validation", "FAIL",
                          f"{len(agent_validation_results['errors'])} agent issues found",
                          agent_validation_results["errors"][:5])
            overall_success = False
        
        self.log_result("Agent Import Test", "PASS" if len(agent_validation_results["importable"]) > 0 else "FAIL",
                      f"{len(agent_validation_results['importable'])}/{total_agents} agents importable")
        
        self.log_result("Agent Instantiation", "PASS" if len(agent_validation_results["instantiable"]) > 0 else "FAIL", 
                      f"{len(agent_validation_results['instantiable'])}/{total_agents} agents instantiable")
        
        self.log_result("Agent Execution", "PASS" if len(agent_validation_results["executable"]) > 0 else "FAIL",
                      f"{len(agent_validation_results['executable'])}/{total_agents} agents executable")
        
        execution_time = time.time() - start_time
        return overall_success
    
    def validate_matrix_agents(self) -> bool:
        """Specifically validate Matrix agents 10-14"""
        print("\n⚡ VALIDATING MATRIX AGENTS 10-14")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        try:
            from core.agents_v2 import MATRIX_AGENTS, get_implementation_status
            
            matrix_status = get_implementation_status()
            expected_agents = [10, 11, 12, 13, 14]
            
            missing_agents = []
            working_agents = []
            
            for agent_id in expected_agents:
                if agent_id in MATRIX_AGENTS:
                    try:
                        agent = MATRIX_AGENTS[agent_id]()
                        working_agents.append(agent_id)
                    except Exception as e:
                        missing_agents.append(f"Agent {agent_id}: {e}")
                        success = False
                else:
                    missing_agents.append(f"Agent {agent_id}: Not found in MATRIX_AGENTS")
                    success = False
            
            if missing_agents:
                self.log_result("Matrix Agents", "FAIL",
                              f"{len(missing_agents)} Matrix agents have issues",
                              missing_agents)
            else:
                self.log_result("Matrix Agents", "PASS",
                              f"All {len(working_agents)} Matrix agents functional")
                
        except ImportError as e:
            self.log_result("Matrix Agents", "FAIL", f"Cannot import Matrix agents: {e}")
            success = False
        
        execution_time = time.time() - start_time
        return success
    
    def validate_configuration(self) -> bool:
        """Validate configuration system"""
        print("\n⚙️ VALIDATING CONFIGURATION")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        try:
            from core.config_manager import get_config_manager
            
            config_manager = get_config_manager()
            
            # Test basic configuration access
            test_configs = [
                ('agents.timeout', 300),
                ('output.base_dir', 'output'),
                ('ghidra.path', None)
            ]
            
            config_issues = []
            
            for config_key, default_value in test_configs:
                try:
                    value = config_manager.get(config_key, default_value)
                    # Just verify it doesn't crash
                except Exception as e:
                    config_issues.append(f"{config_key}: {e}")
                    success = False
            
            if config_issues:
                self.log_result("Configuration System", "FAIL",
                              f"{len(config_issues)} configuration issues",
                              config_issues)
            else:
                self.log_result("Configuration System", "PASS",
                              "Configuration system functional")
                
        except ImportError as e:
            self.log_result("Configuration System", "FAIL", f"Cannot import config manager: {e}")
            success = False
        
        execution_time = time.time() - start_time
        return success
    
    def validate_dependencies(self) -> bool:
        """Validate Python dependencies"""
        print("\n📋 VALIDATING DEPENDENCIES")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        # Read requirements.txt
        requirements_file = project_root / "requirements.txt"
        if not requirements_file.exists():
            self.log_result("Dependencies", "FAIL", "requirements.txt not found")
            return False
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            missing_packages = []
            available_packages = []
            
            for requirement in requirements:
                # Extract package name (before any version specifiers)
                package_name = requirement.split('>=')[0].split('==')[0].split('<=')[0].split('>')[0].split('<')[0].split('!=')[0]
                
                try:
                    importlib.import_module(package_name)
                    available_packages.append(package_name)
                except ImportError:
                    missing_packages.append(package_name)
                    success = False
            
            if missing_packages:
                self.log_result("Python Dependencies", "FAIL",
                              f"{len(missing_packages)} packages missing",
                              missing_packages)
            else:
                self.log_result("Python Dependencies", "PASS",
                              f"All {len(available_packages)} required packages available")
                
        except Exception as e:
            self.log_result("Dependencies", "FAIL", f"Error reading requirements: {e}")
            success = False
        
        execution_time = time.time() - start_time
        return success
    
    def validate_ghidra_integration(self) -> bool:
        """Validate Ghidra integration"""
        print("\n🔧 VALIDATING GHIDRA INTEGRATION")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        # Check Ghidra directory
        ghidra_dir = project_root / "ghidra"
        if not ghidra_dir.exists():
            self.log_result("Ghidra Directory", "FAIL", "ghidra directory not found")
            return False
        
        # Check for key Ghidra files
        ghidra_files = [
            "ghidra/ghidraRun",
            "ghidra/ghidraRun.bat", 
            "ghidra/support/analyzeHeadless",
            "ghidra/support/analyzeHeadless.bat"
        ]
        
        missing_ghidra = []
        for file_path in ghidra_files:
            if not (project_root / file_path).exists():
                missing_ghidra.append(file_path)
        
        if missing_ghidra:
            self.log_result("Ghidra Files", "WARN",
                          f"{len(missing_ghidra)} Ghidra files missing",
                          missing_ghidra)
        else:
            self.log_result("Ghidra Files", "PASS", "Key Ghidra files present")
        
        # Test Ghidra processor import
        try:
            from core.ghidra_processor import GhidraProcessor
            self.log_result("Ghidra Processor", "PASS", "GhidraProcessor can be imported")
        except ImportError as e:
            self.log_result("Ghidra Processor", "FAIL", f"Cannot import GhidraProcessor: {e}")
            success = False
        
        execution_time = time.time() - start_time
        return success
    
    def validate_main_entry_point(self) -> bool:
        """Validate main.py entry point"""
        print("\n🚀 VALIDATING MAIN ENTRY POINT")
        print("=" * 50)
        
        start_time = time.time()
        success = True
        
        main_file = project_root / "main.py"
        if not main_file.exists():
            self.log_result("Main Entry Point", "FAIL", "main.py not found")
            return False
        
        try:
            # Test if main.py can be imported without executing
            import main
            self.log_result("Main Import", "PASS", "main.py imports successfully")
            
            # Check for key classes/functions
            expected_components = ["MatrixCLI"]
            missing_components = []
            
            for component in expected_components:
                if not hasattr(main, component):
                    missing_components.append(component)
            
            if missing_components:
                self.log_result("Main Components", "WARN",
                              f"Missing components: {missing_components}")
            else:
                self.log_result("Main Components", "PASS", "Key components present")
                
        except Exception as e:
            self.log_result("Main Entry Point", "FAIL", f"Error importing main.py: {e}")
            success = False
        
        execution_time = time.time() - start_time
        return success
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        total_time = time.time() - self.start_time
        
        # Count results by status
        status_counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Calculate success rate
        total_checks = len(self.results)
        success_rate = (status_counts["PASS"] / total_checks * 100) if total_checks > 0 else 0
        
        # Categorize issues
        critical_issues = [r for r in self.results if r.status == "FAIL"]
        warnings = [r for r in self.results if r.status == "WARN"]
        
        summary = {
            "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_execution_time": total_time,
            "total_checks": total_checks,
            "status_counts": status_counts,
            "success_rate": success_rate,
            "overall_status": "PASS" if len(critical_issues) == 0 else "FAIL",
            "critical_issues": len(critical_issues),
            "warnings": len(warnings),
            "project_health": self._assess_project_health(success_rate, critical_issues)
        }
        
        return summary
    
    def _assess_project_health(self, success_rate: float, critical_issues: List[ValidationResult]) -> str:
        """Assess overall project health"""
        if success_rate >= 95 and len(critical_issues) == 0:
            return "EXCELLENT"
        elif success_rate >= 85 and len(critical_issues) <= 2:
            return "GOOD"
        elif success_rate >= 70 and len(critical_issues) <= 5:
            return "FAIR"
        else:
            return "POOR"
    
    def print_final_report(self):
        """Print comprehensive final validation report"""
        summary = self.generate_summary_report()
        
        print(f"\n{'='*60}")
        print("🎯 OPEN-SOURCEFY PROJECT VALIDATION REPORT")
        print(f"{'='*60}")
        
        print(f"📅 Validation Date: {summary['validation_timestamp']}")
        print(f"⏱️  Total Time: {summary['total_execution_time']:.2f}s")
        print(f"🔍 Total Checks: {summary['total_checks']}")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        
        print(f"\n📈 RESULTS BREAKDOWN:")
        for status, count in summary['status_counts'].items():
            icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}[status]
            print(f"   {icon} {status}: {count}")
        
        print(f"\n🏥 PROJECT HEALTH: {summary['project_health']}")
        print(f"🚨 Critical Issues: {summary['critical_issues']}")
        print(f"⚠️  Warnings: {summary['warnings']}")
        
        # Show critical issues
        if summary['critical_issues'] > 0:
            print(f"\n❌ CRITICAL ISSUES TO ADDRESS:")
            critical_issues = [r for r in self.results if r.status == "FAIL"]
            for i, issue in enumerate(critical_issues[:10], 1):  # Show top 10
                print(f"   {i}. {issue.component}: {issue.message}")
        
        # Show warnings
        warnings = [r for r in self.results if r.status == "WARN"]
        if warnings:
            print(f"\n⚠️  WARNINGS:")
            for i, warning in enumerate(warnings[:5], 1):  # Show top 5
                print(f"   {i}. {warning.component}: {warning.message}")
        
        # Final assessment
        if summary['overall_status'] == "PASS":
            print(f"\n🎉 PROJECT VALIDATION PASSED!")
            print("✨ The open-sourcefy project is ready for production use.")
        else:
            print(f"\n❌ PROJECT VALIDATION FAILED!")
            print("🔧 Please address the critical issues before production deployment.")
        
        return summary

def main():
    """Run comprehensive project validation"""
    print("🔍 OPEN-SOURCEFY PROJECT VALIDATION")
    print("🎯 Comprehensive system integrity check")
    print(f"📁 Project Root: {project_root}")
    print()
    
    validator = ProjectValidator()
    
    # Run all validation checks
    validation_checks = [
        validator.validate_project_structure,
        validator.validate_python_syntax,
        validator.validate_imports,
        validator.validate_configuration,
        validator.validate_dependencies,
        validator.validate_agents,
        validator.validate_matrix_agents,
        validator.validate_ghidra_integration,
        validator.validate_main_entry_point
    ]
    
    overall_success = True
    
    for check in validation_checks:
        try:
            success = check()
            if not success:
                overall_success = False
        except Exception as e:
            print(f"❌ Validation check failed: {e}")
            traceback.print_exc()
            overall_success = False
        print()  # Add spacing between checks
    
    # Generate and print final report
    summary = validator.print_final_report()
    
    # Save detailed report
    report_file = project_root / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "summary": summary,
            "detailed_results": [
                {
                    "component": r.component,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "metrics": r.metrics,
                    "execution_time": r.execution_time
                }
                for r in validator.results
            ]
        }, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)