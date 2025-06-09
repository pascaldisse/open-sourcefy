"""
Agent 13: Agent Johnson - Security Analysis and Vulnerability Detection
Performs comprehensive security analysis and vulnerability detection in reconstructed code.
"""

import os
import re
import json
import hashlib
from typing import Dict, Any, List, Set, Tuple, Optional
from pathlib import Path
from ..matrix_agents import ReconstructionAgent, AgentResult, AgentStatus, MatrixCharacter

class Agent13_AgentJohnson(ReconstructionAgent):
    """Agent 13: Agent Johnson - Security analysis and vulnerability detection"""
    
    def __init__(self):
        super().__init__(
            agent_id=13,
            matrix_character=MatrixCharacter.AGENT_JOHNSON,
            dependencies=[5, 6, 7, 8]  # Depends on Phase B agents
        )

    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate prerequisites with flexible dependency checking"""
        # Initialize shared_memory structure if not present
        shared_memory = context.get('shared_memory', {})
        if 'analysis_results' not in shared_memory:
            shared_memory['analysis_results'] = {}
        if 'binary_metadata' not in shared_memory:
            shared_memory['binary_metadata'] = {}
        
        # Check for dependencies more flexibly - Agent 13 depends on advanced analysis agents
        dependencies_met = False
        agent_results = context.get('agent_results', {})
        
        # Check for any of the Phase B agents (5, 6, 7, 8)
        phase_b_available = any(
            agent_id in agent_results or agent_id in shared_memory['analysis_results']
            for agent_id in [5, 6, 7, 8]
        )
        
        if phase_b_available:
            dependencies_met = True
        
        # Also check for any advanced analysis results from previous agents
        analysis_available = any(
            self._get_agent_data_safely(agent_data, 'security_analysis') or 
            self._get_agent_data_safely(agent_data, 'vulnerability_assessment') or
            self._get_agent_data_safely(agent_data, 'decompiled_code')
            for agent_data in agent_results.values()
            if agent_data
        )
        
        if analysis_available:
            dependencies_met = True
        
        if not dependencies_met:
            self.logger.warning("No advanced analysis dependencies found - proceeding with basic security analysis")

    def _get_agent_data_safely(self, agent_data: Any, key: str) -> Any:
        """Safely get data from agent result, handling both dict and AgentResult objects"""
        if hasattr(agent_data, 'data') and hasattr(agent_data.data, 'get'):
            return agent_data.data.get(key)
        elif hasattr(agent_data, 'get'):
            data = agent_data.get('data', {})
            if hasattr(data, 'get'):
                return data.get(key)
        return None

    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security analysis and vulnerability detection"""
        # Validate prerequisites first
        self._validate_prerequisites(context)
        
        # Gather all available results for comprehensive security analysis
        all_results = context.get('agent_results', {})
        
        try:
            # Perform security vulnerability analysis
            vulnerability_analysis = self._perform_vulnerability_analysis(all_results, context)
            
            # Analyze code security patterns
            security_patterns = self._analyze_security_patterns(all_results)
            
            # Detect potential exploits
            exploit_detection = self._detect_potential_exploits(vulnerability_analysis, security_patterns)
            
            # Analyze authentication and authorization
            auth_analysis = self._analyze_authentication_authorization(all_results)
            
            # Analyze cryptographic usage
            crypto_analysis = self._analyze_cryptographic_usage(all_results)
            
            # Assess input validation
            input_validation = self._assess_input_validation(all_results)
            
            # Generate security recommendations
            security_recommendations = self._generate_security_recommendations(
                vulnerability_analysis, security_patterns, exploit_detection
            )
            
            # Create security report
            security_report = self._create_security_report(
                vulnerability_analysis, security_patterns, exploit_detection,
                auth_analysis, crypto_analysis, input_validation
            )
            
            # Return dict from execute_matrix_task - base class will wrap in AgentResult
            return {
                'vulnerability_analysis': vulnerability_analysis,
                'security_patterns': security_patterns,
                'exploit_detection': exploit_detection,
                'authentication_analysis': auth_analysis,
                'cryptographic_analysis': crypto_analysis,
                'input_validation': input_validation,
                'security_recommendations': security_recommendations,
                'security_report': security_report,
                'johnson_metrics': self._calculate_johnson_metrics(vulnerability_analysis, security_report)
            }
            
        except Exception as e:
            error_msg = f"Agent Johnson security analysis failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Re-raise exception - base class will handle creating AgentResult
            raise Exception(error_msg) from e

    def _perform_vulnerability_analysis(self, all_results: Dict[int, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive vulnerability analysis"""
        analysis = {
            'vulnerabilities': [],
            'buffer_overflows': [],
            'memory_leaks': [],
            'integer_overflows': [],
            'format_string_bugs': [],
            'injection_vulnerabilities': [],
            'race_conditions': [],
            'privilege_escalation': [],
            'information_disclosure': [],
            'denial_of_service': [],
            'vulnerability_statistics': {}
        }
        
        # Get source code for analysis
        source_data = self._get_source_code(all_results)
        
        # Analyze different vulnerability types
        analysis['buffer_overflows'] = self._detect_buffer_overflows(source_data)
        analysis['memory_leaks'] = self._detect_memory_leaks(source_data)
        analysis['integer_overflows'] = self._detect_integer_overflows(source_data)
        analysis['format_string_bugs'] = self._detect_format_string_bugs(source_data)
        analysis['injection_vulnerabilities'] = self._detect_injection_vulnerabilities(source_data)
        analysis['race_conditions'] = self._detect_race_conditions(source_data)
        analysis['privilege_escalation'] = self._detect_privilege_escalation(source_data)
        analysis['information_disclosure'] = self._detect_information_disclosure(source_data)
        analysis['denial_of_service'] = self._detect_denial_of_service(source_data)
        
        # Compile all vulnerabilities
        all_vulns = []
        vuln_categories = [
            ('buffer_overflow', analysis['buffer_overflows']),
            ('memory_leak', analysis['memory_leaks']),
            ('integer_overflow', analysis['integer_overflows']),
            ('format_string', analysis['format_string_bugs']),
            ('injection', analysis['injection_vulnerabilities']),
            ('race_condition', analysis['race_conditions']),
            ('privilege_escalation', analysis['privilege_escalation']),
            ('information_disclosure', analysis['information_disclosure']),
            ('denial_of_service', analysis['denial_of_service'])
        ]
        
        for category, vulns in vuln_categories:
            for vuln in vulns:
                vuln['category'] = category
                all_vulns.append(vuln)
        
        analysis['vulnerabilities'] = all_vulns
        
        # Calculate statistics
        analysis['vulnerability_statistics'] = self._calculate_vulnerability_statistics(analysis)
        
        return analysis

    def _get_source_code(self, all_results: Dict[int, Any]) -> Dict[str, str]:
        """Get source code from reconstruction results"""
        source_code = {}
        
        # From Global Reconstructor (Agent 9)
        if 9 in all_results:
            reconstructed_source = self._get_agent_data_safely(all_results[9], 'reconstructed_source')
            if isinstance(reconstructed_source, dict):
                source_files = reconstructed_source.get('source_files', {})
                header_files = reconstructed_source.get('header_files', {})
                source_code.update(source_files)
                source_code.update(header_files)
        
        return source_code

    def _detect_buffer_overflows(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect potential buffer overflow vulnerabilities"""
        vulnerabilities = []
        
        # Dangerous functions that can cause buffer overflows
        dangerous_functions = {
            'strcpy': 'Use strncpy or strlcpy instead',
            'strcat': 'Use strncat or strlcat instead',
            'sprintf': 'Use snprintf instead',
            'vsprintf': 'Use vsnprintf instead',
            'gets': 'Use fgets instead',
            'scanf': 'Use safer input methods with length limits',
            'fscanf': 'Use safer input methods with length limits'
        }
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                for func_name, recommendation in dangerous_functions.items():
                    pattern = rf'\b{func_name}\s*\('
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        context = self._extract_code_context(content, match.start(), 2)
                        
                        vulnerabilities.append({
                            'type': 'buffer_overflow',
                            'severity': 'high' if func_name in ['strcpy', 'gets', 'sprintf'] else 'medium',
                            'file': filename,
                            'line': line_num,
                            'function': func_name,
                            'description': f'Potentially unsafe use of {func_name}()',
                            'recommendation': recommendation,
                            'context': context,
                            'cwe': 'CWE-120'  # Classic Buffer Overflow
                        })
        
        # Check for array access without bounds checking
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Look for array access patterns
                array_pattern = r'(\w+)\s*\[\s*(\w+)\s*\]'
                for match in re.finditer(array_pattern, content):
                    array_name, index_var = match.groups()
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Check if there's bounds checking nearby
                    context_before = content[max(0, match.start()-200):match.start()]
                    context_after = content[match.end():match.end()+200]
                    
                    # Look for bounds checking patterns
                    bounds_check_patterns = [
                        rf'if\s*\(\s*{index_var}\s*<',
                        rf'if\s*\(\s*{index_var}\s*>=',
                        rf'if\s*\(\s*{index_var}\s*>',
                        rf'if\s*\(\s*{index_var}\s*<=',
                        rf'assert\s*\(\s*{index_var}'
                    ]
                    
                    has_bounds_check = any(re.search(pattern, context_before + context_after) 
                                         for pattern in bounds_check_patterns)
                    
                    if not has_bounds_check and index_var != '0':  # Skip constant array access
                        vulnerabilities.append({
                            'type': 'buffer_overflow',
                            'severity': 'medium',
                            'file': filename,
                            'line': line_num,
                            'description': f'Array access {array_name}[{index_var}] without apparent bounds checking',
                            'recommendation': 'Add bounds checking before array access',
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-129'  # Improper Validation of Array Index
                        })
        
        return vulnerabilities

    def _detect_memory_leaks(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect potential memory leak vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Find malloc/calloc/realloc calls
                alloc_pattern = r'\b(malloc|calloc|realloc)\s*\('
                free_pattern = r'\bfree\s*\('
                
                alloc_calls = []
                free_calls = []
                
                # Find allocation calls
                for match in re.finditer(alloc_pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    func_name = match.group(1)
                    alloc_calls.append({
                        'function': func_name,
                        'line': line_num,
                        'position': match.start()
                    })
                
                # Find free calls
                for match in re.finditer(free_pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    free_calls.append({
                        'line': line_num,
                        'position': match.start()
                    })
                
                # Simple heuristic: more allocations than frees suggests potential memory leaks
                if len(alloc_calls) > len(free_calls):
                    for alloc in alloc_calls[len(free_calls):]:
                        vulnerabilities.append({
                            'type': 'memory_leak',
                            'severity': 'medium',
                            'file': filename,
                            'line': alloc['line'],
                            'description': f'Potential memory leak: {alloc["function"]}() call without corresponding free()',
                            'recommendation': 'Ensure all allocated memory is freed',
                            'context': self._extract_code_context(content, alloc['position'], 2),
                            'cwe': 'CWE-401'  # Memory Leak
                        })
        
        return vulnerabilities

    def _detect_integer_overflows(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect potential integer overflow vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Look for arithmetic operations without overflow checking
                arithmetic_pattern = r'(\w+)\s*([+\-*/])\s*(\w+)'
                for match in re.finditer(arithmetic_pattern, content):
                    var1, operator, var2 = match.groups()
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Skip obvious constants
                    if var1.isdigit() and var2.isdigit():
                        continue
                    
                    # Look for overflow checking
                    context = content[max(0, match.start()-100):match.end()+100]
                    has_overflow_check = any(keyword in context.lower() for keyword in 
                                           ['overflow', 'check', 'assert', 'max', 'limit'])
                    
                    if not has_overflow_check and operator in ['+', '*']:
                        vulnerabilities.append({
                            'type': 'integer_overflow',
                            'severity': 'medium',
                            'file': filename,
                            'line': line_num,
                            'description': f'Potential integer overflow in arithmetic operation: {var1} {operator} {var2}',
                            'recommendation': 'Add overflow checking for arithmetic operations',
                            'context': self._extract_code_context(content, match.start(), 1),
                            'cwe': 'CWE-190'  # Integer Overflow
                        })
        
        return vulnerabilities

    def _detect_format_string_bugs(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect format string vulnerabilities"""
        vulnerabilities = []
        
        format_functions = ['printf', 'fprintf', 'sprintf', 'snprintf', 'scanf', 'fscanf', 'sscanf']
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                for func_name in format_functions:
                    # Look for format string functions with user-controlled input
                    pattern = rf'\b{func_name}\s*\(\s*([^,)]+)(?:,|\))'
                    for match in re.finditer(pattern, content):
                        format_arg = match.group(1).strip()
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Check if format string is a variable (potential vulnerability)
                        if not (format_arg.startswith('"') or format_arg.startswith("'")):
                            vulnerabilities.append({
                                'type': 'format_string',
                                'severity': 'high',
                                'file': filename,
                                'line': line_num,
                                'function': func_name,
                                'description': f'Potential format string vulnerability in {func_name}()',
                                'recommendation': 'Use literal format strings or validate format string input',
                                'context': self._extract_code_context(content, match.start(), 2),
                                'cwe': 'CWE-134'  # Format String Vulnerability
                            })
        
        return vulnerabilities

    def _detect_injection_vulnerabilities(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect injection vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # SQL injection patterns
                sql_patterns = [
                    r'(SELECT|INSERT|UPDATE|DELETE)\s+.*\+\s*\w+',
                    r'(SELECT|INSERT|UPDATE|DELETE)\s+.*%s',
                    r'sprintf\s*\([^,]+,\s*[^,]*SELECT'
                ]
                
                for pattern in sql_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'sql_injection',
                            'severity': 'high',
                            'file': filename,
                            'line': line_num,
                            'description': 'Potential SQL injection vulnerability',
                            'recommendation': 'Use parameterized queries or prepared statements',
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-89'  # SQL Injection
                        })
                
                # Command injection patterns
                command_patterns = [
                    r'system\s*\([^)]*\+',
                    r'exec\w*\s*\([^)]*\+',
                    r'popen\s*\([^)]*\+'
                ]
                
                for pattern in command_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'command_injection',
                            'severity': 'high',
                            'file': filename,
                            'line': line_num,
                            'description': 'Potential command injection vulnerability',
                            'recommendation': 'Validate and sanitize input before executing commands',
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-78'  # Command Injection
                        })
        
        return vulnerabilities

    def _detect_race_conditions(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect potential race conditions"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Look for shared variable access without synchronization
                global_vars = set()
                
                # Find global variable declarations
                global_pattern = r'^(\w+(?:\s*\*)*)\s+(\w+)(?:\s*=\s*[^;]+)?\s*;'
                for match in re.finditer(global_pattern, content, re.MULTILINE):
                    if not self._is_inside_function(content, match.start()):
                        global_vars.add(match.group(2))
                
                # Look for thread-related functions
                thread_functions = ['CreateThread', 'pthread_create', '_beginthread']
                has_threads = any(func in content for func in thread_functions)
                
                if has_threads and global_vars:
                    # Check for unsynchronized access to global variables
                    sync_keywords = ['mutex', 'lock', 'critical', 'atomic', 'volatile']
                    
                    for var in global_vars:
                        var_pattern = rf'\b{var}\b'
                        for match in re.finditer(var_pattern, content):
                            line_num = content[:match.start()].count('\n') + 1
                            context = content[max(0, match.start()-100):match.end()+100]
                            
                            has_sync = any(keyword in context.lower() for keyword in sync_keywords)
                            
                            if not has_sync:
                                vulnerabilities.append({
                                    'type': 'race_condition',
                                    'severity': 'medium',
                                    'file': filename,
                                    'line': line_num,
                                    'description': f'Potential race condition: unsynchronized access to global variable {var}',
                                    'recommendation': 'Use proper synchronization mechanisms (mutexes, locks, etc.)',
                                    'context': self._extract_code_context(content, match.start(), 2),
                                    'cwe': 'CWE-362'  # Race Condition
                                })
                                break  # Only report once per variable
        
        return vulnerabilities

    def _detect_privilege_escalation(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect potential privilege escalation vulnerabilities"""
        vulnerabilities = []
        
        dangerous_functions = {
            'SetThreadToken': 'Be careful with token manipulation',
            'ImpersonateLoggedOnUser': 'Ensure proper privilege validation',
            'SetPrivilege': 'Validate privilege requirements',
            'AdjustTokenPrivileges': 'Use minimal required privileges',
            'LoadLibrary': 'Validate library paths to prevent DLL hijacking',
            'CreateProcess': 'Validate executable paths and parameters'
        }
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                for func_name, recommendation in dangerous_functions.items():
                    pattern = rf'\b{func_name}\s*\('
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'privilege_escalation',
                            'severity': 'high',
                            'file': filename,
                            'line': line_num,
                            'function': func_name,
                            'description': f'Potential privilege escalation risk with {func_name}()',
                            'recommendation': recommendation,
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-269'  # Improper Privilege Management
                        })
        
        return vulnerabilities

    def _detect_information_disclosure(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect information disclosure vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Look for password/key patterns
                sensitive_patterns = [
                    (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
                    (r'api[_\s]*key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
                    (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
                    (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token detected')
                ]
                
                for pattern, description in sensitive_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'information_disclosure',
                            'severity': 'high',
                            'file': filename,
                            'line': line_num,
                            'description': description,
                            'recommendation': 'Use environment variables or secure configuration files',
                            'context': self._extract_code_context(content, match.start(), 1),
                            'cwe': 'CWE-798'  # Hardcoded Credentials
                        })
                
                # Look for debug information leakage
                debug_patterns = [
                    r'printf\s*\([^)]*debug',
                    r'printf\s*\([^)]*error',
                    r'printf\s*\([^)]*exception'
                ]
                
                for pattern in debug_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'information_disclosure',
                            'severity': 'low',
                            'file': filename,
                            'line': line_num,
                            'description': 'Potential information leakage through debug output',
                            'recommendation': 'Remove debug output from production code',
                            'context': self._extract_code_context(content, match.start(), 1),
                            'cwe': 'CWE-209'  # Information Exposure Through Error Messages
                        })
        
        return vulnerabilities

    def _detect_denial_of_service(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect denial of service vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Look for infinite loops without proper termination conditions
                loop_patterns = [
                    r'while\s*\(\s*1\s*\)',
                    r'while\s*\(\s*true\s*\)',
                    r'for\s*\(\s*;\s*;\s*\)'
                ]
                
                for pattern in loop_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Check if there's a break condition within the loop
                        loop_content = self._extract_loop_body(content, match.end())
                        has_break = 'break' in loop_content or 'return' in loop_content
                        
                        if not has_break:
                            vulnerabilities.append({
                                'type': 'denial_of_service',
                                'severity': 'medium',
                                'file': filename,
                                'line': line_num,
                                'description': 'Potential infinite loop without break condition',
                                'recommendation': 'Add proper termination conditions to loops',
                                'context': self._extract_code_context(content, match.start(), 3),
                                'cwe': 'CWE-835'  # Infinite Loop
                            })
                
                # Look for resource exhaustion patterns
                resource_patterns = [
                    (r'malloc\s*\([^)]*\*\s*\w+\)', 'Potential memory exhaustion'),
                    (r'new\s+\w+\[[^]]*\*\s*\w+\]', 'Potential memory exhaustion')
                ]
                
                for pattern, description in resource_patterns:
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'denial_of_service',
                            'severity': 'medium',
                            'file': filename,
                            'line': line_num,
                            'description': description,
                            'recommendation': 'Add input validation and resource limits',
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-400'  # Uncontrolled Resource Consumption
                        })
        
        return vulnerabilities

    def _extract_code_context(self, content: str, position: int, lines_before_after: int = 2) -> str:
        """Extract code context around a position"""
        lines = content.split('\n')
        line_num = content[:position].count('\n')
        
        start_line = max(0, line_num - lines_before_after)
        end_line = min(len(lines), line_num + lines_before_after + 1)
        
        context_lines = []
        for i in range(start_line, end_line):
            prefix = ">>> " if i == line_num else "    "
            context_lines.append(f"{prefix}{i+1:3d}: {lines[i]}")
        
        return '\n'.join(context_lines)

    def _extract_loop_body(self, content: str, loop_start: int) -> str:
        """Extract the body of a loop starting at a given position"""
        # Find the opening brace
        brace_pos = content.find('{', loop_start)
        if brace_pos == -1:
            return ""
        
        # Find the matching closing brace
        brace_count = 1
        pos = brace_pos + 1
        while pos < len(content) and brace_count > 0:
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            return content[brace_pos:pos]
        return ""

    def _is_inside_function(self, content: str, position: int) -> bool:
        """Check if position is inside a function"""
        before_content = content[:position]
        open_braces = before_content.count('{')
        close_braces = before_content.count('}')
        return open_braces > close_braces

    def _calculate_vulnerability_statistics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate vulnerability statistics"""
        stats = {
            'total_vulnerabilities': len(analysis['vulnerabilities']),
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 0,
            'medium_vulnerabilities': 0,
            'low_vulnerabilities': 0,
            'vulnerability_by_category': {},
            'most_common_vulnerability': None,
            'security_score': 0.0
        }
        
        # Count by severity
        for vuln in analysis['vulnerabilities']:
            severity = vuln.get('severity', 'unknown')
            if severity == 'critical':
                stats['critical_vulnerabilities'] += 1
            elif severity == 'high':
                stats['high_vulnerabilities'] += 1
            elif severity == 'medium':
                stats['medium_vulnerabilities'] += 1
            elif severity == 'low':
                stats['low_vulnerabilities'] += 1
        
        # Count by category
        category_counts = {}
        for vuln in analysis['vulnerabilities']:
            category = vuln.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        stats['vulnerability_by_category'] = category_counts
        
        # Find most common vulnerability
        if category_counts:
            stats['most_common_vulnerability'] = max(category_counts, key=category_counts.get)
        
        # Calculate security score (inverse of vulnerability density)
        if stats['total_vulnerabilities'] == 0:
            stats['security_score'] = 1.0
        else:
            # Weighted score based on severity
            weighted_score = (
                stats['critical_vulnerabilities'] * 10 +
                stats['high_vulnerabilities'] * 5 +
                stats['medium_vulnerabilities'] * 2 +
                stats['low_vulnerabilities'] * 1
            )
            # Normalize to 0-1 scale (assuming max of 100 weighted vulnerabilities for scale)
            stats['security_score'] = max(0.0, 1.0 - (weighted_score / 100.0))
        
        return stats

    def _analyze_security_patterns(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze security patterns in the code"""
        patterns = {
            'input_validation_patterns': [],
            'output_encoding_patterns': [],
            'authentication_patterns': [],
            'authorization_patterns': [],
            'encryption_patterns': [],
            'secure_coding_patterns': [],
            'insecure_patterns': [],
            'pattern_analysis': {}
        }
        
        source_data = self._get_source_code(all_results)
        
        # Analyze input validation patterns
        patterns['input_validation_patterns'] = self._find_input_validation_patterns(source_data)
        
        # Analyze authentication patterns
        patterns['authentication_patterns'] = self._find_authentication_patterns(source_data)
        
        # Analyze encryption patterns
        patterns['encryption_patterns'] = self._find_encryption_patterns(source_data)
        
        # Analyze secure coding patterns
        patterns['secure_coding_patterns'] = self._find_secure_coding_patterns(source_data)
        
        # Analyze insecure patterns
        patterns['insecure_patterns'] = self._find_insecure_patterns(source_data)
        
        # Calculate pattern analysis
        patterns['pattern_analysis'] = {
            'total_secure_patterns': (
                len(patterns['input_validation_patterns']) +
                len(patterns['authentication_patterns']) +
                len(patterns['encryption_patterns']) +
                len(patterns['secure_coding_patterns'])
            ),
            'total_insecure_patterns': len(patterns['insecure_patterns']),
            'security_pattern_ratio': 0.0
        }
        
        total_patterns = (patterns['pattern_analysis']['total_secure_patterns'] + 
                         patterns['pattern_analysis']['total_insecure_patterns'])
        if total_patterns > 0:
            patterns['pattern_analysis']['security_pattern_ratio'] = (
                patterns['pattern_analysis']['total_secure_patterns'] / total_patterns
            )
        
        return patterns

    def _find_input_validation_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find input validation patterns"""
        patterns = []
        
        validation_patterns = [
            (r'if\s*\([^)]*null[^)]*\)', 'null_check'),
            (r'if\s*\([^)]*length[^)]*\)', 'length_check'),
            (r'if\s*\([^)]*strlen[^)]*\)', 'string_length_check'),
            (r'assert\s*\([^)]*\)', 'assertion'),
            (r'isdigit\s*\([^)]*\)', 'numeric_validation'),
            (r'isalpha\s*\([^)]*\)', 'alphabetic_validation')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type in validation_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'pattern': match.group(0)
                    })
        
        return patterns

    def _find_authentication_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find authentication patterns"""
        patterns = []
        
        auth_patterns = [
            (r'password', 'password_handling'),
            (r'authenticate', 'authentication_function'),
            (r'login', 'login_function'),
            (r'token', 'token_handling'),
            (r'session', 'session_management')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type in auth_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'context': self._extract_code_context(content, match.start(), 1)
                    })
        
        return patterns

    def _find_encryption_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find encryption patterns"""
        patterns = []
        
        crypto_patterns = [
            (r'encrypt', 'encryption'),
            (r'decrypt', 'decryption'),
            (r'hash', 'hashing'),
            (r'aes', 'aes_encryption'),
            (r'rsa', 'rsa_encryption'),
            (r'sha', 'sha_hashing'),
            (r'md5', 'md5_hashing')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type in crypto_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'context': self._extract_code_context(content, match.start(), 1)
                    })
        
        return patterns

    def _find_secure_coding_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find secure coding patterns"""
        patterns = []
        
        secure_patterns = [
            (r'strncpy', 'safe_string_copy'),
            (r'snprintf', 'safe_string_format'),
            (r'strncat', 'safe_string_concat'),
            (r'memset.*0', 'memory_clearing'),
            (r'free\s*\([^)]+\);\s*\w+\s*=\s*NULL', 'safe_memory_free')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type in secure_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'pattern': match.group(0)
                    })
        
        return patterns

    def _find_insecure_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find insecure coding patterns"""
        patterns = []
        
        insecure_patterns = [
            (r'system\s*\(', 'system_call'),
            (r'eval\s*\(', 'eval_usage'),
            (r'exec\w*\s*\(', 'exec_usage'),
            (r'rand\s*\(\s*\)', 'weak_random'),
            (r'tmp', 'temp_file_usage')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type in insecure_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'pattern': match.group(0),
                        'risk': 'high' if pattern_type in ['system_call', 'eval_usage', 'exec_usage'] else 'medium'
                    })
        
        return patterns

    def _detect_potential_exploits(self, vulnerability_analysis: Dict[str, Any], security_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential exploit scenarios"""
        detection = {
            'exploit_chains': [],
            'attack_vectors': [],
            'exploit_difficulty': {},
            'exploit_impact': {},
            'exploit_risk_score': 0.0,
            'mitigation_priority': []
        }
        
        vulnerabilities = vulnerability_analysis.get('vulnerabilities', [])
        
        # Analyze individual vulnerabilities for exploitability
        for vuln in vulnerabilities:
            exploit_info = self._assess_exploit_potential(vuln)
            if exploit_info['exploitable']:
                detection['attack_vectors'].append({
                    'vulnerability': vuln,
                    'exploit_method': exploit_info['method'],
                    'difficulty': exploit_info['difficulty'],
                    'impact': exploit_info['impact']
                })
        
        # Look for exploit chains (combinations of vulnerabilities)
        detection['exploit_chains'] = self._find_exploit_chains(vulnerabilities)
        
        # Calculate overall exploit risk score
        detection['exploit_risk_score'] = self._calculate_exploit_risk_score(detection)
        
        # Prioritize mitigations
        detection['mitigation_priority'] = self._prioritize_mitigations(detection, vulnerabilities)
        
        return detection

    def _assess_exploit_potential(self, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the exploit potential of a vulnerability"""
        exploit_info = {
            'exploitable': False,
            'method': 'unknown',
            'difficulty': 'unknown',
            'impact': 'unknown'
        }
        
        vuln_type = vulnerability.get('type', '')
        severity = vulnerability.get('severity', 'low')
        
        # High-impact vulnerabilities
        if vuln_type in ['buffer_overflow', 'format_string', 'command_injection', 'sql_injection']:
            exploit_info['exploitable'] = True
            exploit_info['method'] = f'{vuln_type}_exploitation'
            exploit_info['difficulty'] = 'medium' if severity == 'high' else 'hard'
            exploit_info['impact'] = 'high'
        
        # Medium-impact vulnerabilities
        elif vuln_type in ['privilege_escalation', 'race_condition', 'integer_overflow']:
            exploit_info['exploitable'] = True
            exploit_info['method'] = f'{vuln_type}_exploitation'
            exploit_info['difficulty'] = 'hard'
            exploit_info['impact'] = 'medium'
        
        # Low-impact vulnerabilities
        elif vuln_type in ['information_disclosure', 'denial_of_service']:
            exploit_info['exploitable'] = True
            exploit_info['method'] = f'{vuln_type}_exploitation'
            exploit_info['difficulty'] = 'easy'
            exploit_info['impact'] = 'low'
        
        return exploit_info

    def _find_exploit_chains(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find potential exploit chains"""
        chains = []
        
        # Look for combinations that could form exploit chains
        for i, vuln1 in enumerate(vulnerabilities):
            for j, vuln2 in enumerate(vulnerabilities):
                if i >= j:
                    continue
                
                # Check if vulnerabilities can be chained
                if self._can_chain_vulnerabilities(vuln1, vuln2):
                    chains.append({
                        'vulnerabilities': [vuln1, vuln2],
                        'chain_type': f"{vuln1['type']}_to_{vuln2['type']}",
                        'combined_impact': self._calculate_chain_impact(vuln1, vuln2)
                    })
        
        return chains

    def _can_chain_vulnerabilities(self, vuln1: Dict[str, Any], vuln2: Dict[str, Any]) -> bool:
        """Check if two vulnerabilities can be chained together"""
        # Common exploit chains
        chain_patterns = [
            ('information_disclosure', 'buffer_overflow'),
            ('privilege_escalation', 'command_injection'),
            ('race_condition', 'privilege_escalation'),
            ('integer_overflow', 'buffer_overflow')
        ]
        
        vuln1_type = vuln1.get('type', '')
        vuln2_type = vuln2.get('type', '')
        
        return (vuln1_type, vuln2_type) in chain_patterns or (vuln2_type, vuln1_type) in chain_patterns

    def _calculate_chain_impact(self, vuln1: Dict[str, Any], vuln2: Dict[str, Any]) -> str:
        """Calculate the combined impact of chained vulnerabilities"""
        severity_scores = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        score1 = severity_scores.get(vuln1.get('severity', 'low'), 1)
        score2 = severity_scores.get(vuln2.get('severity', 'low'), 1)
        
        combined_score = score1 + score2
        
        if combined_score >= 6:
            return 'critical'
        elif combined_score >= 4:
            return 'high'
        elif combined_score >= 3:
            return 'medium'
        else:
            return 'low'

    def _calculate_exploit_risk_score(self, detection: Dict[str, Any]) -> float:
        """Calculate overall exploit risk score"""
        attack_vectors = detection.get('attack_vectors', [])
        exploit_chains = detection.get('exploit_chains', [])
        
        if not attack_vectors and not exploit_chains:
            return 0.0
        
        # Score individual attack vectors
        vector_score = 0.0
        for vector in attack_vectors:
            difficulty = vector.get('difficulty', 'hard')
            impact = vector.get('impact', 'low')
            
            difficulty_multiplier = {'easy': 1.0, 'medium': 0.7, 'hard': 0.4}.get(difficulty, 0.4)
            impact_multiplier = {'low': 0.3, 'medium': 0.6, 'high': 1.0}.get(impact, 0.3)
            
            vector_score += difficulty_multiplier * impact_multiplier
        
        # Score exploit chains (higher risk)
        chain_score = len(exploit_chains) * 0.8
        
        # Normalize to 0-1 scale
        total_score = (vector_score + chain_score) / max(1, len(attack_vectors) + len(exploit_chains))
        return min(1.0, total_score)

    def _prioritize_mitigations(self, detection: Dict[str, Any], vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize vulnerability mitigations"""
        priorities = []
        
        # High priority: vulnerabilities in exploit chains
        chain_vulns = set()
        for chain in detection.get('exploit_chains', []):
            for vuln in chain['vulnerabilities']:
                chain_vulns.add(id(vuln))
        
        for vuln in vulnerabilities:
            if id(vuln) in chain_vulns:
                priorities.append({
                    'vulnerability': vuln,
                    'priority': 'critical',
                    'reason': 'Part of exploit chain'
                })
        
        # Medium priority: high-severity individual vulnerabilities
        for vuln in vulnerabilities:
            if vuln.get('severity') == 'high' and id(vuln) not in chain_vulns:
                priorities.append({
                    'vulnerability': vuln,
                    'priority': 'high',
                    'reason': 'High severity vulnerability'
                })
        
        # Low priority: other vulnerabilities
        for vuln in vulnerabilities:
            if vuln.get('severity') in ['medium', 'low'] and id(vuln) not in chain_vulns:
                priorities.append({
                    'vulnerability': vuln,
                    'priority': 'medium' if vuln.get('severity') == 'medium' else 'low',
                    'reason': f'{vuln.get("severity", "unknown").title()} severity vulnerability'
                })
        
        return priorities

    def _analyze_authentication_authorization(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze authentication and authorization mechanisms"""
        analysis = {
            'authentication_mechanisms': [],
            'authorization_checks': [],
            'session_management': [],
            'access_control': [],
            'auth_vulnerabilities': [],
            'auth_strength': 'weak'
        }
        
        source_data = self._get_source_code(all_results)
        
        # Find authentication patterns
        auth_patterns = [
            'password', 'authenticate', 'login', 'credential',
            'token', 'session', 'cookie', 'jwt'
        ]
        
        for filename, content in source_data.items():
            for pattern in auth_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    analysis['authentication_mechanisms'].append({
                        'type': pattern,
                        'file': filename,
                        'line': line_num,
                        'context': self._extract_code_context(content, match.start(), 1)
                    })
        
        # Determine authentication strength
        if len(analysis['authentication_mechanisms']) > 5:
            analysis['auth_strength'] = 'strong'
        elif len(analysis['authentication_mechanisms']) > 2:
            analysis['auth_strength'] = 'medium'
        
        return analysis

    def _analyze_cryptographic_usage(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze cryptographic usage"""
        analysis = {
            'encryption_usage': [],
            'hashing_usage': [],
            'random_number_generation': [],
            'crypto_vulnerabilities': [],
            'crypto_strength': 'weak'
        }
        
        source_data = self._get_source_code(all_results)
        
        # Find cryptographic patterns
        crypto_patterns = {
            'encryption': ['encrypt', 'decrypt', 'aes', 'des', 'rsa', 'cipher'],
            'hashing': ['hash', 'sha', 'md5', 'hmac', 'digest'],
            'random': ['rand', 'random', 'entropy', 'nonce']
        }
        
        for filename, content in source_data.items():
            for category, patterns in crypto_patterns.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        
                        if category == 'encryption':
                            analysis['encryption_usage'].append({
                                'algorithm': pattern,
                                'file': filename,
                                'line': line_num
                            })
                        elif category == 'hashing':
                            analysis['hashing_usage'].append({
                                'algorithm': pattern,
                                'file': filename,
                                'line': line_num
                            })
                        elif category == 'random':
                            analysis['random_number_generation'].append({
                                'method': pattern,
                                'file': filename,
                                'line': line_num
                            })
        
        # Check for weak cryptographic practices
        weak_crypto = ['md5', 'sha1', 'des', 'rc4']
        for usage in analysis['encryption_usage'] + analysis['hashing_usage']:
            if usage.get('algorithm', '').lower() in weak_crypto:
                analysis['crypto_vulnerabilities'].append({
                    'type': 'weak_cryptography',
                    'algorithm': usage['algorithm'],
                    'file': usage['file'],
                    'line': usage['line'],
                    'recommendation': f'Replace {usage["algorithm"]} with stronger algorithm'
                })
        
        # Determine crypto strength
        total_crypto = len(analysis['encryption_usage']) + len(analysis['hashing_usage'])
        if total_crypto > 3 and len(analysis['crypto_vulnerabilities']) == 0:
            analysis['crypto_strength'] = 'strong'
        elif total_crypto > 1:
            analysis['crypto_strength'] = 'medium'
        
        return analysis

    def _assess_input_validation(self, all_results: Dict[int, Any]) -> Dict[str, Any]:
        """Assess input validation mechanisms"""
        assessment = {
            'validation_functions': [],
            'sanitization_functions': [],
            'input_sources': [],
            'validation_coverage': 0.0,
            'validation_strength': 'weak'
        }
        
        source_data = self._get_source_code(all_results)
        
        # Find validation patterns
        validation_patterns = [
            'validate', 'sanitize', 'filter', 'escape',
            'check', 'verify', 'assert', 'isdigit', 'isalpha'
        ]
        
        input_patterns = [
            'scanf', 'gets', 'fgets', 'read', 'recv',
            'argv', 'getenv', 'input', 'request'
        ]
        
        for filename, content in source_data.items():
            # Find validation functions
            for pattern in validation_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    assessment['validation_functions'].append({
                        'function': pattern,
                        'file': filename,
                        'line': line_num
                    })
            
            # Find input sources
            for pattern in input_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    assessment['input_sources'].append({
                        'source': pattern,
                        'file': filename,
                        'line': line_num
                    })
        
        # Calculate validation coverage
        if assessment['input_sources']:
            assessment['validation_coverage'] = len(assessment['validation_functions']) / len(assessment['input_sources'])
        
        # Determine validation strength
        if assessment['validation_coverage'] > 0.8:
            assessment['validation_strength'] = 'strong'
        elif assessment['validation_coverage'] > 0.5:
            assessment['validation_strength'] = 'medium'
        
        return assessment

    def _generate_security_recommendations(self, vulnerability_analysis: Dict[str, Any], 
                                         security_patterns: Dict[str, Any], 
                                         exploit_detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security recommendations"""
        recommendations = []
        
        # Recommendations based on vulnerabilities
        vuln_stats = vulnerability_analysis.get('vulnerability_statistics', {})
        
        if vuln_stats.get('high_vulnerabilities', 0) > 0:
            recommendations.append({
                'priority': 'critical',
                'category': 'vulnerability_mitigation',
                'title': 'Address High-Severity Vulnerabilities',
                'description': f'Found {vuln_stats["high_vulnerabilities"]} high-severity vulnerabilities',
                'actions': [
                    'Review and fix all high-severity vulnerabilities immediately',
                    'Implement input validation and bounds checking',
                    'Use secure coding practices and safe functions'
                ]
            })
        
        # Recommendations based on exploit potential
        if exploit_detection.get('exploit_risk_score', 0.0) > 0.7:
            recommendations.append({
                'priority': 'critical',
                'category': 'exploit_prevention',
                'title': 'High Exploit Risk Detected',
                'description': 'Code has high potential for exploitation',
                'actions': [
                    'Implement defense-in-depth security measures',
                    'Add runtime protection mechanisms',
                    'Conduct penetration testing'
                ]
            })
        
        # Recommendations based on security patterns
        pattern_ratio = security_patterns.get('pattern_analysis', {}).get('security_pattern_ratio', 0.0)
        if pattern_ratio < 0.5:
            recommendations.append({
                'priority': 'high',
                'category': 'secure_coding',
                'title': 'Improve Secure Coding Practices',
                'description': f'Security pattern ratio is low: {pattern_ratio:.2f}',
                'actions': [
                    'Implement more input validation',
                    'Add proper error handling',
                    'Use secure string functions'
                ]
            })
        
        return recommendations

    def _create_security_report(self, vulnerability_analysis: Dict[str, Any], 
                              security_patterns: Dict[str, Any], 
                              exploit_detection: Dict[str, Any],
                              auth_analysis: Dict[str, Any], 
                              crypto_analysis: Dict[str, Any], 
                              input_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive security report"""
        report = {
            'executive_summary': {},
            'vulnerability_summary': {},
            'security_posture': {},
            'threat_assessment': {},
            'compliance_assessment': {},
            'security_score': 0.0,
            'risk_level': 'unknown',
            'recommendations_summary': {}
        }
        
        # Executive summary
        vuln_stats = vulnerability_analysis.get('vulnerability_statistics', {})
        report['executive_summary'] = {
            'total_vulnerabilities': vuln_stats.get('total_vulnerabilities', 0),
            'critical_issues': vuln_stats.get('critical_vulnerabilities', 0) + vuln_stats.get('high_vulnerabilities', 0),
            'exploit_risk': exploit_detection.get('exploit_risk_score', 0.0),
            'overall_security_posture': self._assess_security_posture(vulnerability_analysis, security_patterns)
        }
        
        # Vulnerability summary
        report['vulnerability_summary'] = vuln_stats
        
        # Security posture
        report['security_posture'] = {
            'authentication_strength': auth_analysis.get('auth_strength', 'weak'),
            'cryptographic_strength': crypto_analysis.get('crypto_strength', 'weak'),
            'input_validation_coverage': input_validation.get('validation_coverage', 0.0),
            'secure_coding_patterns': len(security_patterns.get('secure_coding_patterns', []))
        }
        
        # Threat assessment
        report['threat_assessment'] = {
            'attack_vectors': len(exploit_detection.get('attack_vectors', [])),
            'exploit_chains': len(exploit_detection.get('exploit_chains', [])),
            'threat_level': self._calculate_threat_level(exploit_detection)
        }
        
        # Calculate overall security score
        report['security_score'] = self._calculate_security_score(
            vulnerability_analysis, security_patterns, exploit_detection,
            auth_analysis, crypto_analysis, input_validation
        )
        
        # Determine risk level
        report['risk_level'] = self._determine_risk_level(report['security_score'], exploit_detection)
        
        return report

    def _assess_security_posture(self, vulnerability_analysis: Dict[str, Any], security_patterns: Dict[str, Any]) -> str:
        """Assess overall security posture"""
        vuln_count = vulnerability_analysis.get('vulnerability_statistics', {}).get('total_vulnerabilities', 0)
        secure_patterns = len(security_patterns.get('secure_coding_patterns', []))
        
        if vuln_count == 0 and secure_patterns > 5:
            return 'excellent'
        elif vuln_count <= 5 and secure_patterns > 3:
            return 'good'
        elif vuln_count <= 10:
            return 'fair'
        else:
            return 'poor'

    def _calculate_threat_level(self, exploit_detection: Dict[str, Any]) -> str:
        """Calculate threat level"""
        risk_score = exploit_detection.get('exploit_risk_score', 0.0)
        
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'

    def _calculate_security_score(self, vulnerability_analysis: Dict[str, Any], 
                                security_patterns: Dict[str, Any], 
                                exploit_detection: Dict[str, Any],
                                auth_analysis: Dict[str, Any], 
                                crypto_analysis: Dict[str, Any], 
                                input_validation: Dict[str, Any]) -> float:
        """Calculate overall security score"""
        # Component scores
        vuln_score = vulnerability_analysis.get('vulnerability_statistics', {}).get('security_score', 0.0)
        pattern_score = security_patterns.get('pattern_analysis', {}).get('security_pattern_ratio', 0.0)
        exploit_score = 1.0 - exploit_detection.get('exploit_risk_score', 0.0)
        
        # Authentication score
        auth_strength = auth_analysis.get('auth_strength', 'weak')
        auth_score = {'weak': 0.2, 'medium': 0.6, 'strong': 1.0}.get(auth_strength, 0.2)
        
        # Crypto score
        crypto_strength = crypto_analysis.get('crypto_strength', 'weak')
        crypto_score = {'weak': 0.2, 'medium': 0.6, 'strong': 1.0}.get(crypto_strength, 0.2)
        
        # Input validation score
        validation_score = input_validation.get('validation_coverage', 0.0)
        
        # Weighted average
        total_score = (
            vuln_score * 0.3 +
            pattern_score * 0.2 +
            exploit_score * 0.2 +
            auth_score * 0.1 +
            crypto_score * 0.1 +
            validation_score * 0.1
        )
        
        return total_score

    def _determine_risk_level(self, security_score: float, exploit_detection: Dict[str, Any]) -> str:
        """Determine overall risk level"""
        exploit_risk = exploit_detection.get('exploit_risk_score', 0.0)
        
        # If exploit risk is high, overall risk is high regardless of security score
        if exploit_risk >= 0.8:
            return 'critical'
        elif exploit_risk >= 0.6:
            return 'high'
        
        # Otherwise, base on security score
        if security_score >= 0.8:
            return 'low'
        elif security_score >= 0.6:
            return 'medium'
        elif security_score >= 0.4:
            return 'high'
        else:
            return 'critical'

    def _calculate_johnson_metrics(self, vulnerability_analysis: Dict[str, Any], security_report: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Agent Johnson specific metrics"""
        metrics = {
            'security_coverage': 0.0,
            'vulnerability_detection_rate': 0.0,
            'false_positive_estimate': 0.0,
            'security_analysis_completeness': 0.0,
            'johnson_efficiency': 0.0
        }
        
        # Security coverage (based on different vulnerability types checked)
        vuln_categories = vulnerability_analysis.get('vulnerability_statistics', {}).get('vulnerability_by_category', {})
        metrics['security_coverage'] = min(1.0, len(vuln_categories) / 9.0)  # 9 main categories
        
        # Vulnerability detection rate (based on vulnerabilities found vs expected)
        total_vulns = vulnerability_analysis.get('vulnerability_statistics', {}).get('total_vulnerabilities', 0)
        # Assume reasonable baseline of 5-10 vulnerabilities in typical code
        metrics['vulnerability_detection_rate'] = min(1.0, total_vulns / 10.0)
        
        # Security analysis completeness
        analysis_components = [
            bool(vulnerability_analysis.get('vulnerabilities')),
            bool(vulnerability_analysis.get('vulnerability_statistics')),
            bool(security_report.get('security_posture')),
            bool(security_report.get('threat_assessment'))
        ]
        metrics['security_analysis_completeness'] = sum(analysis_components) / len(analysis_components)
        
        # Johnson efficiency (overall effectiveness)
        metrics['johnson_efficiency'] = (
            metrics['security_coverage'] * 0.3 +
            metrics['vulnerability_detection_rate'] * 0.3 +
            metrics['security_analysis_completeness'] * 0.4
        )
        
        return metrics

    def get_description(self) -> str:
        """Get description of Agent Johnson"""
        return "Agent Johnson performs comprehensive security analysis and vulnerability detection in reconstructed code"

    def get_dependencies(self) -> List[int]:
        """Get dependencies for Agent Johnson"""
        return [12]  # Depends on Link