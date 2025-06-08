"""
Agent 19: Binary Comparison Engine
Provides comprehensive binary comparison and validation capabilities.
Phase 4: Build Systems & Production Readiness
"""

import os
import json
import hashlib
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent19_BinaryComparison(BaseAgent):
    """Agent 19: Binary comparison and validation engine"""
    
    def __init__(self):
        super().__init__(
            agent_id=19,
            name="BinaryComparison",
            dependencies=[12, 18]  # Depends on CompilationOrchestrator and AdvancedBuildSystems
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute binary comparison and validation"""
        agent12_result = context['agent_results'].get(12)
        agent18_result = context['agent_results'].get(18)
        
        if not agent12_result or agent12_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 12 (CompilationOrchestrator) did not complete successfully"
            )

        try:
            original_binary = context.get('binary_path', '')
            compilation_data = agent12_result.data
            build_systems_data = agent18_result.data if agent18_result else {}
            
            comparison_result = self._perform_binary_comparison(
                original_binary, compilation_data, build_systems_data, context
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=comparison_result,
                metadata={
                    'depends_on': [12, 18],
                    'analysis_type': 'binary_comparison'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Binary comparison failed: {str(e)}"
            )

    def _perform_binary_comparison(self, original_binary: str, 
                                 compilation_data: Dict[str, Any],
                                 build_systems_data: Dict[str, Any],
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive binary comparison"""
        result = {
            'original_binary': {},
            'generated_binaries': {},
            'comparison_results': {},
            'similarity_scores': {},
            'functional_analysis': {},
            'performance_analysis': {},
            'quality_metrics': {},
            'validation_status': 'unknown',
            'overall_score': 0.0,
            'recommendations': []
        }
        
        try:
            # Analyze original binary
            result['original_binary'] = self._analyze_binary(original_binary)
            
            # Find and analyze generated binaries
            generated_binaries = self._find_generated_binaries(compilation_data, build_systems_data, context)
            result['generated_binaries'] = {
                path: self._analyze_binary(path) for path in generated_binaries
            }
            
            # Perform comparisons
            for binary_path in generated_binaries:
                comparison = self._compare_binaries(original_binary, binary_path)
                result['comparison_results'][binary_path] = comparison
                result['similarity_scores'][binary_path] = self._calculate_similarity_score(comparison)
            
            # Functional analysis
            result['functional_analysis'] = self._analyze_functionality(
                original_binary, generated_binaries
            )
            
            # NEW: Dummy code detection analysis
            result['dummy_code_analysis'] = self._analyze_for_dummy_code(
                generated_binaries, context
            )
            
            # Performance analysis
            result['performance_analysis'] = self._analyze_performance(
                original_binary, generated_binaries
            )
            
            # Calculate quality metrics
            result['quality_metrics'] = self._calculate_quality_metrics(result)
            
            # Determine validation status
            result['validation_status'] = self._determine_validation_status(result)
            
            # Calculate overall score
            result['overall_score'] = self._calculate_overall_score(result)
            
            # Generate recommendations
            result['recommendations'] = self._generate_recommendations(result)
            
        except Exception as e:
            result['error'] = str(e)
            result['validation_status'] = 'failed'
        
        return result

    def _analyze_binary(self, binary_path: str) -> Dict[str, Any]:
        """Analyze a binary file and extract metadata"""
        analysis = {
            'exists': False,
            'file_size': 0,
            'file_hash': '',
            'pe_info': {},
            'imports': [],
            'exports': [],
            'sections': [],
            'entry_point': None,
            'architecture': 'unknown',
            'subsystem': 'unknown',
            'compilation_timestamp': None,
            'is_valid_pe': False
        }
        
        if not os.path.exists(binary_path):
            return analysis
        
        analysis['exists'] = True
        analysis['file_size'] = os.path.getsize(binary_path)
        analysis['file_hash'] = self._calculate_file_hash(binary_path)
        
        # PE analysis
        pe_info = self._analyze_pe_structure(binary_path)
        analysis.update(pe_info)
        
        return analysis

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ''

    def _analyze_pe_structure(self, binary_path: str) -> Dict[str, Any]:
        """Analyze PE structure of Windows executable"""
        pe_info = {
            'is_valid_pe': False,
            'pe_info': {},
            'imports': [],
            'exports': [],
            'sections': [],
            'entry_point': None,
            'architecture': 'unknown',
            'subsystem': 'unknown',
            'compilation_timestamp': None
        }
        
        try:
            # Basic PE header check
            with open(binary_path, 'rb') as f:
                # Check DOS header
                dos_header = f.read(64)
                if len(dos_header) < 64 or dos_header[:2] != b'MZ':
                    return pe_info
                
                # Get PE offset
                pe_offset = int.from_bytes(dos_header[60:64], 'little')
                
                # Check PE signature
                f.seek(pe_offset)
                pe_sig = f.read(4)
                if pe_sig != b'PE\x00\x00':
                    return pe_info
                
                pe_info['is_valid_pe'] = True
                
                # Read COFF header
                coff_header = f.read(20)
                if len(coff_header) >= 20:
                    machine = int.from_bytes(coff_header[0:2], 'little')
                    timestamp = int.from_bytes(coff_header[4:8], 'little')
                    
                    pe_info['architecture'] = self._decode_machine_type(machine)
                    pe_info['compilation_timestamp'] = timestamp
                
                # Read Optional header for more details
                optional_header_size = int.from_bytes(coff_header[16:18], 'little')
                if optional_header_size > 0:
                    optional_header = f.read(min(optional_header_size, 240))
                    if len(optional_header) >= 68:
                        entry_point = int.from_bytes(optional_header[16:20], 'little')
                        subsystem = int.from_bytes(optional_header[68:70], 'little')
                        
                        pe_info['entry_point'] = f"0x{entry_point:08x}"
                        pe_info['subsystem'] = self._decode_subsystem(subsystem)
                
        except Exception as e:
            pe_info['error'] = str(e)
        
        # Try to get more detailed info using external tools if available
        try:
            pe_info.update(self._get_detailed_pe_info(binary_path))
        except Exception:
            pass
        
        return pe_info

    def _decode_machine_type(self, machine: int) -> str:
        """Decode PE machine type"""
        machine_types = {
            0x014c: 'i386',
            0x0200: 'ia64', 
            0x8664: 'x64',
            0x01c0: 'arm',
            0xaa64: 'arm64'
        }
        return machine_types.get(machine, f'unknown(0x{machine:04x})')

    def _decode_subsystem(self, subsystem: int) -> str:
        """Decode PE subsystem"""
        subsystems = {
            1: 'native',
            2: 'windows_gui',
            3: 'windows_cui',
            7: 'posix_cui',
            9: 'windows_ce_gui',
            10: 'efi_application',
            11: 'efi_boot_service_driver',
            12: 'efi_runtime_driver',
            13: 'efi_rom',
            14: 'xbox'
        }
        return subsystems.get(subsystem, f'unknown({subsystem})')

    def _get_detailed_pe_info(self, binary_path: str) -> Dict[str, Any]:
        """Get detailed PE info using external tools"""
        detailed_info = {}
        
        # Try using objdump if available (from MinGW)
        try:
            result = subprocess.run([
                'objdump', '-p', binary_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                detailed_info['objdump_output'] = result.stdout
                detailed_info.update(self._parse_objdump_output(result.stdout))
        except Exception:
            pass
        
        # Try using dumpbin if available (from Visual Studio)
        try:
            result = subprocess.run([
                'dumpbin', '/headers', binary_path
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                detailed_info['dumpbin_output'] = result.stdout
                detailed_info.update(self._parse_dumpbin_output(result.stdout))
        except Exception:
            pass
        
        return detailed_info

    def _parse_objdump_output(self, output: str) -> Dict[str, Any]:
        """Parse objdump output for PE information"""
        info = {}
        imports = []
        exports = []
        sections = []
        
        lines = output.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'DLL Name:' in line:
                dll_name = line.split('DLL Name:')[1].strip()
                imports.append({'dll': dll_name, 'functions': []})
            elif line.startswith('ordinal') and 'hint' in line and 'RVA' in line:
                # Skip header line
                continue
            elif len(line.split()) >= 4 and imports:
                # Import function
                parts = line.split()
                if len(parts) >= 4:
                    function_name = ' '.join(parts[3:])
                    imports[-1]['functions'].append(function_name)
            elif 'Section Table:' in line:
                current_section = 'sections'
            elif current_section == 'sections' and len(line.split()) >= 7:
                parts = line.split()
                if len(parts) >= 7:
                    sections.append({
                        'name': parts[0],
                        'size': parts[1],
                        'vma': parts[2],
                        'lma': parts[3],
                        'file_offset': parts[4],
                        'align': parts[5]
                    })
        
        info['imports'] = imports
        info['exports'] = exports
        info['sections'] = sections
        
        return info

    def _parse_dumpbin_output(self, output: str) -> Dict[str, Any]:
        """Parse dumpbin output for PE information"""
        info = {}
        
        # Extract basic info from dumpbin output
        lines = output.split('\n')
        for line in lines:
            line = line.strip()
            if 'machine' in line.lower():
                info['machine_info'] = line
            elif 'subsystem' in line.lower():
                info['subsystem_info'] = line
        
        return info

    def _find_generated_binaries(self, compilation_data: Dict[str, Any],
                                build_systems_data: Dict[str, Any],
                                context: Dict[str, Any]) -> List[str]:
        """Find all generated binary files"""
        binaries = []
        
        # From compilation data
        if compilation_data.get('generated_binary'):
            binaries.append(compilation_data['generated_binary'])
        
        # From build systems data
        if build_systems_data.get('compilation_attempts'):
            for compiler, attempt in build_systems_data['compilation_attempts'].items():
                if attempt.get('success') and attempt.get('binary_path'):
                    binaries.append(attempt['binary_path'])
        
        # Look in output directory
        output_paths = context.get('output_paths', {})
        compilation_dir = output_paths.get('compilation', context.get('output_dir', 'output'))
        
        if os.path.exists(compilation_dir):
            for root, dirs, files in os.walk(compilation_dir):
                for file in files:
                    if file.endswith('.exe'):
                        full_path = os.path.join(root, file)
                        if full_path not in binaries:
                            binaries.append(full_path)
        
        return [b for b in binaries if os.path.exists(b)]

    def _compare_binaries(self, original: str, generated: str) -> Dict[str, Any]:
        """Compare two binary files"""
        comparison = {
            'size_difference': 0,
            'size_ratio': 0.0,
            'hash_match': False,
            'architecture_match': False,
            'subsystem_match': False,
            'imports_similarity': 0.0,
            'exports_similarity': 0.0,
            'sections_similarity': 0.0,
            'entry_point_match': False,
            'structural_similarity': 0.0
        }
        
        try:
            orig_analysis = self._analyze_binary(original)
            gen_analysis = self._analyze_binary(generated)
            
            if not orig_analysis['exists'] or not gen_analysis['exists']:
                return comparison
            
            # Size comparison
            orig_size = orig_analysis['file_size']
            gen_size = gen_analysis['file_size']
            comparison['size_difference'] = abs(orig_size - gen_size)
            comparison['size_ratio'] = gen_size / orig_size if orig_size > 0 else 0.0
            
            # Hash comparison
            comparison['hash_match'] = orig_analysis['file_hash'] == gen_analysis['file_hash']
            
            # Architecture comparison
            comparison['architecture_match'] = (
                orig_analysis['architecture'] == gen_analysis['architecture']
            )
            
            # Subsystem comparison
            comparison['subsystem_match'] = (
                orig_analysis['subsystem'] == gen_analysis['subsystem']
            )
            
            # Entry point comparison
            comparison['entry_point_match'] = (
                orig_analysis['entry_point'] == gen_analysis['entry_point']
            )
            
            # Imports similarity
            comparison['imports_similarity'] = self._calculate_imports_similarity(
                orig_analysis['imports'], gen_analysis['imports']
            )
            
            # Exports similarity
            comparison['exports_similarity'] = self._calculate_exports_similarity(
                orig_analysis['exports'], gen_analysis['exports']
            )
            
            # Sections similarity
            comparison['sections_similarity'] = self._calculate_sections_similarity(
                orig_analysis['sections'], gen_analysis['sections']
            )
            
            # Overall structural similarity
            comparison['structural_similarity'] = self._calculate_structural_similarity(comparison)
            
        except Exception as e:
            comparison['error'] = str(e)
        
        return comparison

    def _calculate_imports_similarity(self, orig_imports: List[Dict], gen_imports: List[Dict]) -> float:
        """Calculate similarity between import tables"""
        if not orig_imports and not gen_imports:
            return 1.0
        if not orig_imports or not gen_imports:
            return 0.0
        
        # Simple similarity based on common DLLs and functions
        orig_dlls = {imp.get('dll', '') for imp in orig_imports}
        gen_dlls = {imp.get('dll', '') for imp in gen_imports}
        
        common_dlls = orig_dlls.intersection(gen_dlls)
        total_dlls = orig_dlls.union(gen_dlls)
        
        if not total_dlls:
            return 1.0
        
        return len(common_dlls) / len(total_dlls)

    def _calculate_exports_similarity(self, orig_exports: List[Dict], gen_exports: List[Dict]) -> float:
        """Calculate similarity between export tables"""
        if not orig_exports and not gen_exports:
            return 1.0
        if not orig_exports or not gen_exports:
            return 0.0
        
        # Simple comparison of export names
        orig_names = {exp.get('name', '') for exp in orig_exports}
        gen_names = {exp.get('name', '') for exp in gen_exports}
        
        common_names = orig_names.intersection(gen_names)
        total_names = orig_names.union(gen_names)
        
        if not total_names:
            return 1.0
        
        return len(common_names) / len(total_names)

    def _calculate_sections_similarity(self, orig_sections: List[Dict], gen_sections: List[Dict]) -> float:
        """Calculate similarity between section tables"""
        if not orig_sections and not gen_sections:
            return 1.0
        if not orig_sections or not gen_sections:
            return 0.0
        
        # Compare section names
        orig_names = {sec.get('name', '') for sec in orig_sections}
        gen_names = {sec.get('name', '') for sec in gen_sections}
        
        common_names = orig_names.intersection(gen_names)
        total_names = orig_names.union(gen_names)
        
        if not total_names:
            return 1.0
        
        return len(common_names) / len(total_names)

    def _calculate_structural_similarity(self, comparison: Dict[str, Any]) -> float:
        """Calculate overall structural similarity"""
        factors = []
        
        # Architecture match (critical)
        if comparison['architecture_match']:
            factors.append(1.0)
        else:
            factors.append(0.0)
        
        # Subsystem match (important)
        if comparison['subsystem_match']:
            factors.append(1.0)
        else:
            factors.append(0.5)
        
        # Size similarity (reasonable range)
        size_ratio = comparison['size_ratio']
        if 0.5 <= size_ratio <= 2.0:
            size_score = 1.0 - abs(1.0 - size_ratio)
            factors.append(max(0.0, size_score))
        else:
            factors.append(0.0)
        
        # Imports similarity
        factors.append(comparison['imports_similarity'])
        
        # Sections similarity
        factors.append(comparison['sections_similarity'])
        
        return sum(factors) / len(factors) if factors else 0.0

    def _calculate_similarity_score(self, comparison: Dict[str, Any]) -> float:
        """Calculate overall similarity score between binaries"""
        return comparison.get('structural_similarity', 0.0)

    def _analyze_functionality(self, original: str, generated_binaries: List[str]) -> Dict[str, Any]:
        """Analyze functional equivalence"""
        analysis = {
            'execution_tests': {},
            'api_call_analysis': {},
            'behavior_comparison': {},
            'functional_score': 0.0
        }
        
        # Basic execution test
        for binary in generated_binaries:
            analysis['execution_tests'][binary] = self._test_execution(binary)
        
        # Calculate functional score
        successful_executions = sum(
            1 for test in analysis['execution_tests'].values() 
            if test.get('can_execute', False)
        )
        
        if generated_binaries:
            analysis['functional_score'] = successful_executions / len(generated_binaries)
        
        return analysis

    def _test_execution(self, binary_path: str) -> Dict[str, Any]:
        """Test if binary can execute"""
        test_result = {
            'can_execute': False,
            'exit_code': None,
            'execution_time': 0.0,
            'stdout': '',
            'stderr': '',
            'error': None
        }
        
        try:
            import time
            start_time = time.time()
            
            # Run with short timeout to test basic execution
            result = subprocess.run([binary_path], 
                                  capture_output=True, text=True, timeout=5)
            
            test_result['can_execute'] = True
            test_result['exit_code'] = result.returncode
            test_result['execution_time'] = time.time() - start_time
            test_result['stdout'] = result.stdout
            test_result['stderr'] = result.stderr
            
        except subprocess.TimeoutExpired:
            test_result['can_execute'] = True  # It started running
            test_result['error'] = 'timeout'
        except Exception as e:
            test_result['error'] = str(e)
        
        return test_result

    def _analyze_performance(self, original: str, generated_binaries: List[str]) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        analysis = {
            'binary_sizes': {},
            'load_times': {},
            'memory_usage': {},
            'performance_score': 0.0
        }
        
        # Analyze binary sizes
        try:
            original_size = os.path.getsize(original)
            analysis['binary_sizes']['original'] = original_size
            
            for binary in generated_binaries:
                if os.path.exists(binary):
                    analysis['binary_sizes'][binary] = os.path.getsize(binary)
        except Exception:
            pass
        
        # Simple performance score based on size efficiency
        if analysis['binary_sizes']:
            original_size = analysis['binary_sizes'].get('original', 1)
            size_scores = []
            
            for binary, size in analysis['binary_sizes'].items():
                if binary != 'original':
                    ratio = size / original_size if original_size > 0 else 1.0
                    # Prefer smaller binaries but not too much smaller
                    if 0.5 <= ratio <= 1.5:
                        score = 1.0 - abs(1.0 - ratio) * 0.5
                    else:
                        score = 0.5
                    size_scores.append(score)
            
            if size_scores:
                analysis['performance_score'] = sum(size_scores) / len(size_scores)
        
        return analysis

    def _calculate_quality_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics"""
        metrics = {
            'compilation_success_rate': 0.0,
            'binary_similarity_score': 0.0,
            'functional_equivalence_score': 0.0,
            'performance_score': 0.0,
            'overall_quality_score': 0.0
        }
        
        # Compilation success rate
        total_binaries = len(result['generated_binaries'])
        successful_binaries = sum(
            1 for analysis in result['generated_binaries'].values()
            if analysis['exists'] and analysis['is_valid_pe']
        )
        
        if total_binaries > 0:
            metrics['compilation_success_rate'] = successful_binaries / total_binaries
        
        # Binary similarity score
        similarity_scores = list(result['similarity_scores'].values())
        if similarity_scores:
            metrics['binary_similarity_score'] = sum(similarity_scores) / len(similarity_scores)
        
        # Functional equivalence score
        functional_analysis = result.get('functional_analysis', {})
        metrics['functional_equivalence_score'] = functional_analysis.get('functional_score', 0.0)
        
        # Performance score
        performance_analysis = result.get('performance_analysis', {})
        metrics['performance_score'] = performance_analysis.get('performance_score', 0.0)
        
        # Overall quality score (weighted average)
        weights = {
            'compilation_success_rate': 0.3,
            'binary_similarity_score': 0.3,
            'functional_equivalence_score': 0.3,
            'performance_score': 0.1
        }
        
        metrics['overall_quality_score'] = sum(
            metrics[key] * weight for key, weight in weights.items()
        )
        
        return metrics

    def _determine_validation_status(self, result: Dict[str, Any]) -> str:
        """Determine overall validation status"""
        quality_metrics = result.get('quality_metrics', {})
        overall_score = quality_metrics.get('overall_quality_score', 0.0)
        
        if overall_score >= 0.9:
            return 'excellent'
        elif overall_score >= 0.7:
            return 'good'
        elif overall_score >= 0.5:
            return 'fair'
        elif overall_score >= 0.3:
            return 'poor'
        else:
            return 'failed'

    def _calculate_overall_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall comparison score"""
        quality_metrics = result.get('quality_metrics', {})
        return quality_metrics.get('overall_quality_score', 0.0)

    def _generate_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        quality_metrics = result.get('quality_metrics', {})
        
        # Compilation success recommendations
        if quality_metrics.get('compilation_success_rate', 0.0) < 0.8:
            recommendations.append("Improve compilation success rate by fixing build system configuration")
        
        # Binary similarity recommendations
        if quality_metrics.get('binary_similarity_score', 0.0) < 0.7:
            recommendations.append("Enhance decompilation accuracy to improve binary similarity")
        
        # Functional equivalence recommendations
        if quality_metrics.get('functional_equivalence_score', 0.0) < 0.7:
            recommendations.append("Improve function reconstruction to achieve better functional equivalence")
        
        # Performance recommendations
        if quality_metrics.get('performance_score', 0.0) < 0.7:
            recommendations.append("Optimize generated code to improve performance characteristics")
        
        # General recommendations
        overall_score = quality_metrics.get('overall_quality_score', 0.0)
        if overall_score < 0.5:
            recommendations.append("Consider using different compiler or build configuration")
            recommendations.append("Review and improve the decompilation pipeline")
        
        return recommendations

    def _analyze_for_dummy_code(self, generated_binaries: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze generated binaries for dummy code patterns and validate they represent real implementations"""
        analysis = {
            'dummy_code_detected': False,
            'confidence_score': 0.0,
            'binary_analyses': {},
            'source_code_validation': {},
            'overall_authenticity_score': 0.0,
            'authenticity_issues': [],
            'validation_passed': False
        }
        
        try:
            # Analyze each generated binary
            total_authenticity_score = 0.0
            binary_count = 0
            
            for binary_path in generated_binaries:
                if os.path.exists(binary_path):
                    binary_analysis = self._validate_binary_authenticity(binary_path)
                    analysis['binary_analyses'][binary_path] = binary_analysis
                    
                    total_authenticity_score += binary_analysis['authenticity_score']
                    binary_count += 1
                    
                    if binary_analysis['appears_dummy']:
                        analysis['dummy_code_detected'] = True
                        analysis['authenticity_issues'].extend(binary_analysis['dummy_indicators'])
            
            # Validate source code authenticity
            source_validation = self._validate_source_code_authenticity(context)
            analysis['source_code_validation'] = source_validation
            
            if source_validation['appears_dummy']:
                analysis['dummy_code_detected'] = True
                analysis['authenticity_issues'].extend(source_validation['dummy_indicators'])
            
            # Calculate overall authenticity score
            if binary_count > 0:
                binary_score = total_authenticity_score / binary_count
                source_score = source_validation['authenticity_score']
                analysis['overall_authenticity_score'] = (binary_score * 0.6 + source_score * 0.4)
            else:
                analysis['overall_authenticity_score'] = source_validation['authenticity_score']
            
            # Calculate confidence score
            analysis['confidence_score'] = min(1.0, analysis['overall_authenticity_score'] + 0.2)
            
            # Determine if validation passed
            analysis['validation_passed'] = (
                not analysis['dummy_code_detected'] and 
                analysis['overall_authenticity_score'] >= 0.7 and
                len(analysis['authenticity_issues']) == 0
            )
            
        except Exception as e:
            analysis['error'] = str(e)
            analysis['dummy_code_detected'] = True
            analysis['authenticity_issues'].append(f"Analysis failed: {str(e)}")
        
        return analysis

    def _validate_binary_authenticity(self, binary_path: str) -> Dict[str, Any]:
        """Validate that a binary represents real implementation, not dummy code"""
        validation = {
            'appears_dummy': False,
            'authenticity_score': 0.0,
            'dummy_indicators': [],
            'size_analysis': {},
            'content_analysis': {},
            'execution_analysis': {}
        }
        
        try:
            # Size analysis
            file_size = os.path.getsize(binary_path)
            validation['size_analysis'] = {
                'file_size': file_size,
                'size_reasonable': 15000 <= file_size <= 10000000,  # 15KB to 10MB
                'size_category': self._categorize_binary_size(file_size)
            }
            
            if file_size < 15000:
                validation['dummy_indicators'].append(f"Binary too small ({file_size} bytes) - likely simple dummy program")
                validation['appears_dummy'] = True
            
            # Content analysis - check for dummy strings
            content_analysis = self._analyze_binary_content_for_dummy_patterns(binary_path)
            validation['content_analysis'] = content_analysis
            
            if content_analysis['has_dummy_patterns']:
                validation['appears_dummy'] = True
                validation['dummy_indicators'].extend(content_analysis['dummy_patterns_found'])
            
            # Execution analysis
            execution_analysis = self._analyze_binary_execution_authenticity(binary_path)
            validation['execution_analysis'] = execution_analysis
            
            if execution_analysis['appears_trivial']:
                validation['appears_dummy'] = True
                validation['dummy_indicators'].extend(execution_analysis['trivial_indicators'])
            
            # Calculate authenticity score
            size_score = 1.0 if validation['size_analysis']['size_reasonable'] else 0.3
            content_score = 1.0 - content_analysis['dummy_pattern_ratio']
            execution_score = 1.0 if not execution_analysis['appears_trivial'] else 0.2
            
            validation['authenticity_score'] = (size_score * 0.3 + content_score * 0.4 + execution_score * 0.3)
            
        except Exception as e:
            validation['appears_dummy'] = True
            validation['dummy_indicators'].append(f"Binary validation failed: {str(e)}")
            validation['authenticity_score'] = 0.0
        
        return validation

    def _categorize_binary_size(self, size: int) -> str:
        """Categorize binary size"""
        if size < 10000:
            return "very_small"
        elif size < 50000:
            return "small"
        elif size < 500000:
            return "medium"
        elif size < 5000000:
            return "large"
        else:
            return "very_large"

    def _analyze_binary_content_for_dummy_patterns(self, binary_path: str) -> Dict[str, Any]:
        """Analyze binary content for dummy code patterns"""
        analysis = {
            'has_dummy_patterns': False,
            'dummy_patterns_found': [],
            'dummy_pattern_ratio': 0.0,
            'total_strings_analyzed': 0
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read sample of binary for string analysis
                sample_size = min(65536, os.path.getsize(binary_path))
                binary_data = f.read(sample_size)
            
            # Convert to text for analysis
            try:
                text_content = binary_data.decode('utf-8', errors='ignore')
            except:
                text_content = str(binary_data)
            
            # Check for dummy patterns
            dummy_patterns = [
                r'hello\s*world',
                r'Hello\s*World',
                r'HELLO\s*WORLD',
                r'test\s*program',
                r'placeholder',
                r'PLACEHOLDER',
                r'TODO',
                r'FIXME',
                r'Generated\s*by\s*analysis',
                r'Reconstructed\s*binary',
                r'Decompiled\s*program',
                r'Sample\s*application',
                r'Demo\s*program',
                r'Example\s*code'
            ]
            
            found_patterns = []
            for pattern in dummy_patterns:
                import re
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                if matches:
                    found_patterns.append(f"Found dummy pattern: '{pattern}' ({len(matches)} times)")
            
            analysis['dummy_patterns_found'] = found_patterns
            analysis['has_dummy_patterns'] = len(found_patterns) > 0
            
            # Calculate dummy pattern ratio
            if len(text_content) > 0:
                dummy_chars = sum(len(match) for pattern in dummy_patterns 
                                for match in re.findall(pattern, text_content, re.IGNORECASE))
                analysis['dummy_pattern_ratio'] = min(1.0, dummy_chars / len(text_content))
            
            # Count total identifiable strings
            string_patterns = re.findall(r'[a-zA-Z]{3,}', text_content)
            analysis['total_strings_analyzed'] = len(string_patterns)
            
        except Exception as e:
            analysis['has_dummy_patterns'] = True
            analysis['dummy_patterns_found'].append(f"Content analysis failed: {str(e)}")
        
        return analysis

    def _analyze_binary_execution_authenticity(self, binary_path: str) -> Dict[str, Any]:
        """Analyze binary execution to detect trivial/dummy behavior"""
        analysis = {
            'appears_trivial': False,
            'trivial_indicators': [],
            'execution_complexity_score': 0.0,
            'output_analysis': {}
        }
        
        try:
            # Run binary and analyze output
            start_time = time.time()
            
            result = subprocess.run([
                binary_path
            ], capture_output=True, text=True, timeout=10)
            
            execution_time = time.time() - start_time
            
            # Analyze output for dummy patterns
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            analysis['output_analysis'] = {
                'stdout': stdout,
                'stderr': stderr,
                'execution_time': execution_time,
                'exit_code': result.returncode
            }
            
            # Check for trivial output patterns
            trivial_outputs = [
                'hello world',
                'hello, world',
                'test program',
                'generated program',
                'reconstructed program',
                'sample application'
            ]
            
            for trivial in trivial_outputs:
                if trivial.lower() in stdout.lower():
                    analysis['appears_trivial'] = True
                    analysis['trivial_indicators'].append(f"Trivial output detected: '{trivial}'")
            
            # Check execution characteristics
            if execution_time < 0.001 and len(stdout) < 50 and result.returncode == 0:
                analysis['appears_trivial'] = True
                analysis['trivial_indicators'].append("Execution too fast and simple (likely just prints and exits)")
            
            if len(stdout) == 0 and len(stderr) == 0 and result.returncode == 0:
                analysis['trivial_indicators'].append("No output produced - potentially empty implementation")
            
            # Calculate complexity score
            output_complexity = len(stdout) + len(stderr)
            time_complexity = min(1.0, execution_time * 10)  # Normalize to 0-1 scale
            
            analysis['execution_complexity_score'] = min(1.0, (output_complexity / 100) * 0.7 + time_complexity * 0.3)
            
        except subprocess.TimeoutExpired:
            # Long execution might indicate real functionality
            analysis['execution_complexity_score'] = 0.8
            analysis['trivial_indicators'].append("Execution timeout - may indicate complex operation")
        except Exception as e:
            analysis['appears_trivial'] = True
            analysis['trivial_indicators'].append(f"Execution analysis failed: {str(e)}")
        
        return analysis

    def _validate_source_code_authenticity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate source code authenticity from agent results"""
        validation = {
            'appears_dummy': False,
            'authenticity_score': 0.0,
            'dummy_indicators': [],
            'code_complexity_analysis': {},
            'function_analysis': {}
        }
        
        try:
            # Get source code from Agent 11 (Global Reconstructor)
            agent11_result = context.get('agent_results', {}).get(11)
            if not agent11_result or agent11_result.status != AgentStatus.COMPLETED:
                validation['appears_dummy'] = True
                validation['dummy_indicators'].append("No valid source code reconstruction available")
                return validation
            
            global_reconstruction = agent11_result.data
            reconstructed_source = global_reconstruction.get('reconstructed_source', {})
            source_files = reconstructed_source.get('source_files', {})
            
            if not source_files:
                validation['appears_dummy'] = True
                validation['dummy_indicators'].append("No source files generated")
                return validation
            
            # Analyze all source files
            total_complexity = 0.0
            total_authenticity = 0.0
            files_analyzed = 0
            
            for filename, content in source_files.items():
                if isinstance(content, str):
                    file_analysis = self._analyze_source_file_authenticity(filename, content)
                    
                    if file_analysis['appears_dummy']:
                        validation['appears_dummy'] = True
                        validation['dummy_indicators'].extend(file_analysis['dummy_indicators'])
                    
                    total_complexity += file_analysis['complexity_score']
                    total_authenticity += file_analysis['authenticity_score']
                    files_analyzed += 1
            
            if files_analyzed > 0:
                avg_complexity = total_complexity / files_analyzed
                avg_authenticity = total_authenticity / files_analyzed
                
                validation['code_complexity_analysis'] = {
                    'average_complexity': avg_complexity,
                    'files_analyzed': files_analyzed,
                    'complexity_adequate': avg_complexity >= 0.5
                }
                
                validation['authenticity_score'] = avg_authenticity
                
                if avg_complexity < 0.3:
                    validation['appears_dummy'] = True
                    validation['dummy_indicators'].append(f"Average code complexity too low: {avg_complexity:.2f}")
            
        except Exception as e:
            validation['appears_dummy'] = True
            validation['dummy_indicators'].append(f"Source code validation failed: {str(e)}")
        
        return validation

    def _analyze_source_file_authenticity(self, filename: str, content: str) -> Dict[str, Any]:
        """Analyze individual source file for authenticity"""
        analysis = {
            'appears_dummy': False,
            'authenticity_score': 0.0,
            'complexity_score': 0.0,
            'dummy_indicators': []
        }
        
        try:
            lines = content.split('\n')
            code_lines = [line.strip() for line in lines 
                         if line.strip() and not line.strip().startswith('//')]
            
            # Check for dummy patterns in source
            dummy_source_patterns = [
                'printf.*"Hello World"',
                'printf.*"hello world"',
                'TODO.*implementation',
                'FIXME.*placeholder',
                'Generated by.*analysis',
                'Reconstructed.*binary',
                'return 0.*}\s*$'
            ]
            
            dummy_pattern_count = 0
            for pattern in dummy_source_patterns:
                import re
                if re.search(pattern, content, re.IGNORECASE):
                    dummy_pattern_count += 1
                    analysis['dummy_indicators'].append(f"Dummy pattern found: {pattern}")
            
            if dummy_pattern_count >= 2:
                analysis['appears_dummy'] = True
            
            # Calculate complexity score
            function_count = len(re.findall(r'\w+\s*\([^)]*\)\s*\{', content))
            variable_assignments = len(re.findall(r'\w+\s*=\s*[^=]', content))
            control_structures = len(re.findall(r'\b(if|while|for|switch)\s*\(', content))
            
            analysis['complexity_score'] = min(1.0, (
                function_count * 0.2 + 
                variable_assignments * 0.05 + 
                control_structures * 0.15 +
                len(code_lines) * 0.01
            ))
            
            # Calculate authenticity score
            if analysis['appears_dummy']:
                analysis['authenticity_score'] = 0.2
            else:
                # Base score on complexity and absence of dummy patterns
                pattern_penalty = dummy_pattern_count * 0.2
                analysis['authenticity_score'] = max(0.0, analysis['complexity_score'] - pattern_penalty)
            
        except Exception as e:
            analysis['appears_dummy'] = True
            analysis['dummy_indicators'].append(f"File analysis failed: {str(e)}")
        
        return analysis