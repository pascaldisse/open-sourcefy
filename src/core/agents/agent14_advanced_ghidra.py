"""
Agent 14: Advanced Ghidra Integration
Provides enhanced binary analysis using advanced Ghidra features and custom scripts.
"""

import os
import json
import subprocess
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent14_AdvancedGhidra(BaseAgent):
    """Agent 14: Advanced Ghidra analysis with custom scripts and enhanced decompilation"""
    
    def __init__(self):
        super().__init__(
            agent_id=14,
            name="AdvancedGhidra",
            dependencies=[7]  # Depends on basic Ghidra analysis
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute advanced Ghidra analysis"""
        agent7_result = context['agent_results'].get(7)
        if not agent7_result or agent7_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 7 (AdvancedDecompiler) did not complete successfully"
            )

        try:
            binary_path = context.get('binary_path', '')
            output_paths = context.get('output_paths', {})
            ghidra_dir = output_paths.get('ghidra', context.get('output_dir', 'output'))
            
            analysis_result = self._perform_advanced_analysis(
                binary_path, ghidra_dir, agent7_result.data, context
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=analysis_result,
                metadata={
                    'depends_on': [7],
                    'analysis_type': 'advanced_ghidra_analysis'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Advanced Ghidra analysis failed: {str(e)}"
            )

    def _perform_advanced_analysis(self, binary_path: str, output_dir: str, basic_ghidra_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced Ghidra analysis with custom scripts"""
        result = {
            'enhanced_functions': {},
            'control_flow_graphs': {},
            'cross_references': {},
            'data_structures': {},
            'string_analysis': {},
            'import_analysis': {},
            'export_analysis': {},
            'custom_analysis': {},
            'confidence_scores': {},
            'analysis_quality': 'unknown'
        }
        
        # Create advanced analysis directory
        advanced_dir = os.path.join(output_dir, 'advanced')
        os.makedirs(advanced_dir, exist_ok=True)
        
        try:
            # Enhanced function analysis
            result['enhanced_functions'] = self._enhance_function_analysis(
                binary_path, basic_ghidra_data, advanced_dir
            )
            
            # Control flow graph analysis
            result['control_flow_graphs'] = self._analyze_control_flow(
                binary_path, advanced_dir
            )
            
            # Cross-reference analysis
            result['cross_references'] = self._analyze_cross_references(
                binary_path, advanced_dir
            )
            
            # Data structure recovery
            result['data_structures'] = self._recover_data_structures(
                binary_path, advanced_dir
            )
            
            # Advanced string analysis
            result['string_analysis'] = self._advanced_string_analysis(
                binary_path, advanced_dir
            )
            
            # Import/Export analysis
            result['import_analysis'] = self._analyze_imports(binary_path, advanced_dir)
            result['export_analysis'] = self._analyze_exports(binary_path, advanced_dir)
            
            # Calculate confidence scores
            result['confidence_scores'] = self._calculate_confidence_scores(result)
            
            # Determine overall analysis quality
            result['analysis_quality'] = self._assess_analysis_quality(result)
            
        except Exception as e:
            result['error'] = str(e)
            result['analysis_quality'] = 'failed'
        
        return result

    def _enhance_function_analysis(self, binary_path: str, basic_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Enhance function analysis with advanced techniques"""
        enhanced_functions = {}
        
        # Get basic functions from Agent 7
        basic_functions = basic_data.get('decompiled_functions', {})
        
        for func_name, func_data in basic_functions.items():
            if isinstance(func_data, dict):
                enhanced_functions[func_name] = {
                    'basic_info': func_data,
                    'enhanced_signature': self._enhance_function_signature(func_data),
                    'variable_analysis': self._analyze_variables(func_data),
                    'complexity_metrics': self._calculate_complexity(func_data),
                    'purpose_analysis': self._analyze_function_purpose(func_data),
                    'optimization_hints': self._detect_optimizations(func_data),
                    'confidence': self._score_function_confidence(func_data)
                }
        
        return enhanced_functions

    def _enhance_function_signature(self, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance function signature with better type inference"""
        signature = func_data.get('signature', '')
        
        # Basic signature enhancement
        enhanced_sig = {
            'original': signature,
            'enhanced_params': [],
            'return_type': 'int',  # Default
            'calling_convention': 'cdecl',  # Default
            'confidence': 0.5
        }
        
        # Parse and enhance parameters
        if 'parameters' in func_data:
            for param in func_data['parameters']:
                enhanced_param = {
                    'name': param.get('name', 'param'),
                    'type': self._infer_parameter_type(param),
                    'usage': self._analyze_parameter_usage(param),
                    'confidence': 0.6
                }
                enhanced_sig['enhanced_params'].append(enhanced_param)
        
        return enhanced_sig

    def _infer_parameter_type(self, param: Dict[str, Any]) -> str:
        """Infer parameter type based on usage patterns"""
        # Simple type inference based on usage
        usage = param.get('usage', '')
        if 'string' in usage.lower():
            return 'char*'
        elif 'pointer' in usage.lower():
            return 'void*'
        elif 'size' in usage.lower() or 'length' in usage.lower():
            return 'size_t'
        elif 'count' in usage.lower():
            return 'int'
        else:
            return 'int'  # Default

    def _analyze_parameter_usage(self, param: Dict[str, Any]) -> str:
        """Analyze how parameter is used in function"""
        # Placeholder for parameter usage analysis
        return param.get('usage', 'unknown')

    def _analyze_variables(self, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze local variables and their usage"""
        return {
            'local_variables': func_data.get('variables', []),
            'variable_types': {},
            'variable_scopes': {},
            'usage_patterns': {}
        }

    def _calculate_complexity(self, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate function complexity metrics"""
        code = func_data.get('decompiled_code', '')
        
        # Basic complexity metrics
        lines = len(code.split('\n')) if code else 0
        branches = code.count('if') + code.count('while') + code.count('for') if code else 0
        
        return {
            'cyclomatic_complexity': max(1, branches + 1),
            'lines_of_code': lines,
            'branch_count': branches,
            'complexity_rating': 'low' if branches < 3 else 'medium' if branches < 8 else 'high'
        }

    def _analyze_function_purpose(self, func_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function purpose and classify functionality"""
        code = func_data.get('decompiled_code', '')
        name = func_data.get('name', '')
        
        # Simple purpose classification
        purpose_hints = []
        if 'main' in name.lower():
            purpose_hints.append('entry_point')
        if 'init' in name.lower() or 'setup' in name.lower():
            purpose_hints.append('initialization')
        if 'cleanup' in name.lower() or 'destroy' in name.lower():
            purpose_hints.append('cleanup')
        if 'get' in name.lower():
            purpose_hints.append('getter')
        if 'set' in name.lower():
            purpose_hints.append('setter')
        
        return {
            'purpose_hints': purpose_hints,
            'function_category': purpose_hints[0] if purpose_hints else 'utility',
            'confidence': 0.7 if purpose_hints else 0.3
        }

    def _detect_optimizations(self, func_data: Dict[str, Any]) -> List[str]:
        """Detect compiler optimizations in function"""
        optimizations = []
        code = func_data.get('decompiled_code', '')
        
        if 'inline' in code.lower():
            optimizations.append('function_inlining')
        if code and len(code.split('\n')) < 5:
            optimizations.append('code_optimization')
        
        return optimizations

    def _score_function_confidence(self, func_data: Dict[str, Any]) -> float:
        """Score confidence in function analysis"""
        score = 0.5  # Base score
        
        if func_data.get('decompiled_code'):
            score += 0.2
        if func_data.get('signature'):
            score += 0.2
        if func_data.get('parameters'):
            score += 0.1
        
        return min(1.0, score)

    def _analyze_control_flow(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Analyze control flow graphs"""
        # Basic stub implementation
        return {
            'control_flow_graphs': {},
            'basic_blocks': {},
            'edges': [],
            'analysis_status': 'basic_analysis_only'
        }

    def _analyze_cross_references(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Analyze cross-references between functions and data"""
        # Basic stub implementation
        return {
            'function_references': {},
            'data_references': {},
            'call_graph': {},
            'analysis_status': 'basic_analysis_only'
        }

    def _recover_data_structures(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Recover data structures from binary"""
        # Basic stub implementation
        return {
            'structures': {},
            'unions': {},
            'enums': {},
            'typedefs': {},
            'analysis_status': 'basic_analysis_only'
        }

    def _advanced_string_analysis(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Perform advanced string analysis"""
        # Basic stub implementation
        return {
            'strings': [],
            'unicode_strings': [],
            'format_strings': [],
            'encrypted_strings': [],
            'analysis_status': 'basic_analysis_only'
        }

    def _analyze_imports(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Analyze imported functions and libraries"""
        # Basic stub implementation
        return {
            'imported_libraries': [],
            'imported_functions': {},
            'dynamic_imports': {},
            'analysis_status': 'basic_analysis_only'
        }

    def _analyze_exports(self, binary_path: str, output_dir: str) -> Dict[str, Any]:
        """Analyze exported functions"""
        # Basic stub implementation
        return {
            'exported_functions': {},
            'export_ordinals': {},
            'forwarded_exports': {},
            'analysis_status': 'basic_analysis_only'
        }

    def _calculate_confidence_scores(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different analysis components"""
        scores = {}
        
        # Calculate confidence for each analysis component
        for key, value in result.items():
            if isinstance(value, dict) and 'confidence' in value:
                scores[key] = value['confidence']
            else:
                scores[key] = 0.5  # Default confidence
        
        # Overall confidence
        if scores:
            scores['overall'] = sum(scores.values()) / len(scores)
        else:
            scores['overall'] = 0.3
        
        return scores

    def _assess_analysis_quality(self, result: Dict[str, Any]) -> str:
        """Assess overall analysis quality"""
        confidence_scores = result.get('confidence_scores', {})
        overall_confidence = confidence_scores.get('overall', 0.3)
        
        if overall_confidence >= 0.8:
            return 'excellent'
        elif overall_confidence >= 0.6:
            return 'good'
        elif overall_confidence >= 0.4:
            return 'fair'
        else:
            return 'poor'