"""
Agent 3: Smart Error Pattern Matching
Identifies common error patterns and provides intelligent error handling strategies.
"""

import re
from typing import Dict, Any, List, Tuple
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent3_SmartErrorPatternMatching(BaseAgent):
    """Agent 3: Smart error pattern matching and handling"""
    
    def __init__(self):
        super().__init__(
            agent_id=3,
            name="SmartErrorPatternMatching",
            dependencies=[1]
        )
        
        # Initialize error pattern database
        self.error_patterns = self._initialize_error_patterns()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute smart error pattern matching"""
        # Get data from Agent 1
        agent1_result = context['agent_results'].get(1)
        if not agent1_result or agent1_result.status != AgentStatus.COMPLETED:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Agent 1 (BinaryDiscovery) did not complete successfully"
            )

        try:
            binary_info = agent1_result.data
            error_analysis = self._analyze_error_patterns(binary_info, context)
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=error_analysis,
                metadata={
                    'depends_on': [1],
                    'analysis_type': 'error_pattern_matching'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Error pattern matching failed: {str(e)}"
            )

    def _initialize_error_patterns(self) -> Dict[str, Any]:
        """Initialize comprehensive error pattern database"""
        return {
            'compilation_errors': {
                'missing_headers': {
                    'patterns': [
                        r"fatal error: '([^']+)' file not found",
                        r"No such file or directory: ([^\s]+\.h)",
                        r"Cannot open include file: '([^']+)'"
                    ],
                    'solutions': [
                        'Add include directory to compiler flags',
                        'Install missing development packages',
                        'Create stub header file'
                    ]
                },
                'undefined_symbols': {
                    'patterns': [
                        r"undefined reference to `([^']+)'",
                        r"unresolved external symbol ([^\s]+)",
                        r"Undefined symbols for architecture"
                    ],
                    'solutions': [
                        'Link required libraries',
                        'Add function implementation',
                        'Check library compatibility'
                    ]
                },
                'type_errors': {
                    'patterns': [
                        r"error: conflicting types for '([^']+)'",
                        r"error: incompatible types",
                        r"type mismatch in ([^\s]+)"
                    ],
                    'solutions': [
                        'Fix type declarations',
                        'Add explicit type casts',
                        'Update function signatures'
                    ]
                }
            },
            'decompilation_errors': {
                'unsupported_instructions': {
                    'patterns': [
                        r"Unknown instruction: ([^\s]+)",
                        r"Unsupported opcode: (0x[0-9a-fA-F]+)",
                        r"Invalid instruction encoding"
                    ],
                    'solutions': [
                        'Update decompiler database',
                        'Use different decompiler',
                        'Manual analysis required'
                    ]
                },
                'memory_access_errors': {
                    'patterns': [
                        r"Invalid memory access at (0x[0-9a-fA-F]+)",
                        r"Segmentation fault",
                        r"Access violation"
                    ],
                    'solutions': [
                        'Check memory mapping',
                        'Verify address calculations',
                        'Update memory layout'
                    ]
                }
            },
            'analysis_errors': {
                'format_specific': {
                    'PE': {
                        'patterns': [
                            r"Invalid PE header",
                            r"Corrupted import table",
                            r"Invalid section alignment"
                        ],
                        'solutions': [
                            'Verify PE structure',
                            'Repair import table',
                            'Fix section headers'
                        ]
                    },
                    'ELF': {
                        'patterns': [
                            r"Invalid ELF header",
                            r"Corrupted section header",
                            r"Invalid program header"
                        ],
                        'solutions': [
                            'Verify ELF structure',
                            'Repair section headers',
                            'Fix program headers'
                        ]
                    }
                }
            }
        }

    def _analyze_error_patterns(self, binary_info: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential error patterns and provide solutions"""
        analysis = {
            'potential_issues': [],
            'error_predictions': [],
            'recommended_strategies': [],
            'compatibility_warnings': [],
            'format_specific_issues': []
        }
        
        # Analyze based on binary format
        format_info = binary_info.get('format_info', {})
        arch_info = binary_info.get('architecture', {})
        
        # Check for format-specific issues
        analysis['format_specific_issues'] = self._check_format_issues(format_info, binary_info)
        
        # Check for architecture-specific issues
        analysis['potential_issues'].extend(self._check_arch_issues(arch_info))
        
        # Predict common decompilation errors
        analysis['error_predictions'] = self._predict_decompilation_errors(binary_info)
        
        # Generate recommended strategies
        analysis['recommended_strategies'] = self._generate_strategies(binary_info)
        
        # Check compatibility
        analysis['compatibility_warnings'] = self._check_compatibility(binary_info)
        
        return analysis

    def _check_format_issues(self, format_info: Dict[str, Any], binary_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for format-specific issues"""
        issues = []
        binary_format = format_info.get('format', 'Unknown')
        
        if binary_format == 'PE':
            # Check PE-specific issues
            pe_data = binary_info.get('pe_sections', [])
            if not pe_data:
                issues.append({
                    'type': 'warning',
                    'message': 'No PE sections found - may indicate corrupted or packed binary',
                    'severity': 'medium',
                    'solutions': ['Use unpacker', 'Manual section analysis']
                })
                
        elif binary_format == 'ELF':
            # Check ELF-specific issues
            elf_data = binary_info.get('elf_sections', [])
            if not elf_data:
                issues.append({
                    'type': 'warning',
                    'message': 'No ELF sections found - may indicate stripped or corrupted binary',
                    'severity': 'medium',
                    'solutions': ['Check for debug symbols', 'Use different analysis tool']
                })
                
        elif binary_format == 'Unknown':
            issues.append({
                'type': 'error',
                'message': 'Unknown binary format - analysis may be limited',
                'severity': 'high',
                'solutions': ['Use hex editor for manual analysis', 'Try different format detection tools']
            })
        
        return issues

    def _check_arch_issues(self, arch_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for architecture-specific issues"""
        issues = []
        architecture = arch_info.get('architecture', 'Unknown')
        
        if architecture == 'Unknown':
            issues.append({
                'type': 'error',
                'message': 'Unknown architecture - decompilation may fail',
                'severity': 'high',
                'solutions': ['Manual architecture detection', 'Use specialized tools']
            })
        
        # Check for complex architectures
        complex_archs = ['ARM64', 'RISC-V', 'MIPS']
        if architecture in complex_archs:
            issues.append({
                'type': 'warning',
                'message': f'{architecture} is complex - may require specialized handling',
                'severity': 'medium',
                'solutions': [f'Use {architecture}-specific tools', 'Check decompiler support']
            })
        
        return issues

    def _predict_decompilation_errors(self, binary_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict likely decompilation errors"""
        predictions = []
        
        # Check file size
        file_size = binary_info.get('file_info', {}).get('size', 0)
        if file_size > 10 * 1024 * 1024:  # > 10MB
            predictions.append({
                'type': 'performance_warning',
                'message': 'Large binary may cause performance issues during decompilation',
                'probability': 'medium',
                'mitigation': 'Use chunked analysis or increase memory limits'
            })
        
        # Note: Advanced optimization detection would require analysis
        # of instruction patterns and compiler signatures
        
        return predictions

    def _generate_strategies(self, binary_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommended analysis strategies"""
        strategies = []
        
        format_info = binary_info.get('format_info', {})
        arch_info = binary_info.get('architecture', {})
        
        # Basic strategy based on format
        binary_format = format_info.get('format', 'Unknown')
        if binary_format in ['PE', 'ELF', 'Mach-O']:
            strategies.append({
                'strategy': 'multi_tool_approach',
                'description': f'Use multiple tools for {binary_format} analysis',
                'tools': self._get_recommended_tools(binary_format),
                'priority': 'high'
            })
        
        # Architecture-specific strategy
        architecture = arch_info.get('architecture', 'Unknown')
        if architecture != 'Unknown':
            strategies.append({
                'strategy': 'arch_optimized_analysis',
                'description': f'Use {architecture}-optimized analysis techniques',
                'techniques': self._get_arch_techniques(architecture),
                'priority': 'medium'
            })
        
        return strategies

    def _check_compatibility(self, binary_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for compatibility warnings"""
        warnings = []
        
        arch_info = binary_info.get('architecture', {})
        bitness = arch_info.get('bitness', 'Unknown')
        
        # Check for 32-bit vs 64-bit issues
        if bitness == '32-bit':
            warnings.append({
                'type': 'compatibility_warning',
                'message': '32-bit binary may have different calling conventions',
                'impact': 'decompilation_accuracy',
                'recommendation': 'Verify calling convention settings'
            })
        
        return warnings

    def _get_recommended_tools(self, binary_format: str) -> List[str]:
        """Get recommended tools for binary format"""
        tools = {
            'PE': ['IDA Pro', 'Ghidra', 'x64dbg', 'PE-bear'],
            'ELF': ['IDA Pro', 'Ghidra', 'radare2', 'objdump'],
            'Mach-O': ['IDA Pro', 'Ghidra', 'Hopper', 'otool']
        }
        return tools.get(binary_format, ['Ghidra', 'IDA Pro'])

    def _get_arch_techniques(self, architecture: str) -> List[str]:
        """Get architecture-specific analysis techniques"""
        techniques = {
            'x86': ['Stack frame analysis', 'Calling convention detection'],
            'x64': ['RIP-relative addressing', 'Windows x64 ABI analysis'],
            'ARM': ['Thumb mode detection', 'ARM/Thumb interworking'],
            'ARM64': ['AAPCS64 analysis', 'Exception handling analysis']
        }
        return techniques.get(architecture, ['General static analysis'])

    def match_error_pattern(self, error_text: str) -> List[Dict[str, Any]]:
        """Match error text against known patterns"""
        matches = []
        
        for category, subcategories in self.error_patterns.items():
            for subcategory, pattern_data in subcategories.items():
                if isinstance(pattern_data, dict) and 'patterns' in pattern_data:
                    for pattern in pattern_data['patterns']:
                        if re.search(pattern, error_text, re.IGNORECASE):
                            matches.append({
                                'category': category,
                                'subcategory': subcategory,
                                'pattern': pattern,
                                'solutions': pattern_data.get('solutions', [])
                            })
        
        return matches