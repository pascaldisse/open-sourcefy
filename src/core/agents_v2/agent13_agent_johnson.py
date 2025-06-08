"""
Agent 13: Agent Johnson - Security Analysis and Vulnerability Detection
The relentless security enforcer who meticulously scans for vulnerabilities and security weaknesses.
Performs comprehensive security analysis on reconstructed code with unwavering attention to detail.

Production-ready Matrix v2 implementation following SOLID principles and clean code standards.
Includes comprehensive vulnerability detection, exploit analysis, and security recommendations.
"""

import logging
import time
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Matrix framework imports
try:
    from ..matrix_agents_v2 import AnalysisAgent, MatrixCharacter, AgentStatus
    from ..shared_components import (
        MatrixLogger, MatrixFileManager, MatrixValidator, MatrixProgressTracker, 
        MatrixErrorHandler, MatrixMetrics, SharedAnalysisTools, SharedValidationTools
    )
    from ..exceptions import MatrixAgentError, ValidationError, ConfigurationError
    HAS_MATRIX_FRAMEWORK = True
except ImportError:
    # Fallback for basic execution
    HAS_MATRIX_FRAMEWORK = False


@dataclass
class SecurityAnalysisResult:
    """Result of comprehensive security analysis"""
    success: bool = False
    total_vulnerabilities: int = 0
    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0
    security_score: float = 0.0
    exploit_risk_score: float = 0.0
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    security_patterns: Dict[str, Any] = field(default_factory=dict)
    exploit_chains: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VulnerabilityReport:
    """Individual vulnerability report"""
    vulnerability_type: str
    severity: str
    file: str
    line: int
    description: str
    recommendation: str
    context: str
    cwe: str = ""
    exploitable: bool = False
    exploit_difficulty: str = "unknown"
    impact: str = "unknown"


class Agent13_AgentJohnson:
    """
    Agent 13: Agent Johnson - Security Analysis and Vulnerability Detection
    
    Responsibilities:
    1. Comprehensive vulnerability analysis of reconstructed code
    2. Security pattern detection and analysis
    3. Exploit potential assessment and chain detection
    4. Authentication and authorization analysis
    5. Cryptographic usage evaluation
    6. Input validation assessment
    7. Security recommendations generation
    """
    
    def __init__(self):
        self.agent_id = 13
        self.name = "Agent Johnson"
        self.character = MatrixCharacter.AGENT_JOHNSON if HAS_MATRIX_FRAMEWORK else "agent_johnson"
        
        # Core components
        self.logger = self._setup_logging()
        self.file_manager = MatrixFileManager() if HAS_MATRIX_FRAMEWORK else None
        self.validator = MatrixValidator() if HAS_MATRIX_FRAMEWORK else None
        self.progress_tracker = MatrixProgressTracker() if HAS_MATRIX_FRAMEWORK else None
        self.error_handler = MatrixErrorHandler() if HAS_MATRIX_FRAMEWORK else None
        
        # Security analysis components
        self.vulnerability_detectors = self._initialize_vulnerability_detectors()
        self.security_patterns = self._load_security_patterns()
        self.exploit_analyzers = self._initialize_exploit_analyzers()
        
        # Analysis thresholds
        self.security_thresholds = {
            'minimum_security_score': 0.70,
            'maximum_critical_vulnerabilities': 0,
            'maximum_high_vulnerabilities': 3,
            'maximum_exploit_risk': 0.30
        }
        
        # State tracking
        self.current_phase = "initialization"
        self.analysis_stats = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup agent-specific logging"""
        logger = logging.getLogger(f"Matrix.AgentJohnson")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[Agent Johnson] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_dependencies(self) -> List[int]:
        """Get list of required predecessor agents"""
        return [12]  # Link - Communications Bridge
    
    def get_description(self) -> str:
        """Get agent description"""
        return ("Agent Johnson performs comprehensive security analysis and vulnerability detection, "
                "meticulously scanning reconstructed code for security weaknesses and exploit potential.")

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive security analysis and vulnerability detection"""
        self.logger.info("🕴️ Agent Johnson initiating security enforcement protocol...")
        
        start_time = time.time()
        
        try:
            # Phase 1: Validate dependencies and security environment
            self.current_phase = "security_validation"
            self.logger.info("Phase 1: Validating security analysis prerequisites...")
            validation_result = self._validate_security_dependencies(context)
            
            if not validation_result['valid']:
                return self._create_failure_result(
                    f"Security validation failed: {validation_result['error']}"
                )
            
            # Phase 2: Extract and prepare source code for analysis
            self.current_phase = "source_extraction"
            self.logger.info("Phase 2: Extracting source code for security analysis...")
            source_data = self._extract_source_code(context)
            
            if not source_data:
                return self._create_failure_result("No source code available for security analysis")
            
            # Phase 3: Perform comprehensive vulnerability analysis
            self.current_phase = "vulnerability_analysis"
            self.logger.info("Phase 3: Performing comprehensive vulnerability analysis...")
            vulnerability_analysis = self._perform_comprehensive_vulnerability_analysis(source_data)
            
            # Phase 4: Analyze security patterns and practices
            self.current_phase = "security_patterns"
            self.logger.info("Phase 4: Analyzing security patterns and practices...")
            security_patterns = self._analyze_security_patterns(source_data)
            
            # Phase 5: Detect potential exploits and attack vectors
            self.current_phase = "exploit_detection"
            self.logger.info("Phase 5: Detecting exploit potential and attack vectors...")
            exploit_detection = self._detect_exploit_potential(vulnerability_analysis, security_patterns)
            
            # Phase 6: Analyze authentication and authorization
            self.current_phase = "auth_analysis"
            self.logger.info("Phase 6: Analyzing authentication and authorization...")
            auth_analysis = self._analyze_authentication_authorization(source_data)
            
            # Phase 7: Evaluate cryptographic usage
            self.current_phase = "crypto_analysis"
            self.logger.info("Phase 7: Evaluating cryptographic usage...")
            crypto_analysis = self._analyze_cryptographic_usage(source_data)
            
            # Phase 8: Assess input validation mechanisms
            self.current_phase = "input_validation"
            self.logger.info("Phase 8: Assessing input validation mechanisms...")
            input_validation = self._assess_input_validation(source_data)
            
            # Phase 9: Generate security recommendations
            self.current_phase = "recommendations"
            self.logger.info("Phase 9: Generating security recommendations...")
            recommendations = self._generate_security_recommendations(
                vulnerability_analysis, security_patterns, exploit_detection,
                auth_analysis, crypto_analysis, input_validation
            )
            
            # Phase 10: Create comprehensive security report
            self.current_phase = "report_generation"
            self.logger.info("Phase 10: Creating comprehensive security report...")
            final_result = self._create_security_report(
                vulnerability_analysis, security_patterns, exploit_detection,
                auth_analysis, crypto_analysis, input_validation, recommendations
            )
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"🎯 Agent Johnson security analysis completed in {execution_time:.2f}s")
            self.logger.info(f"🚨 Total Vulnerabilities: {final_result.total_vulnerabilities}")
            self.logger.info(f"⚠️ Critical: {final_result.critical_vulnerabilities}, High: {final_result.high_vulnerabilities}")
            self.logger.info(f"🛡️ Security Score: {final_result.security_score:.2f}")
            self.logger.info(f"💥 Exploit Risk: {final_result.exploit_risk_score:.2f}")
            
            return {
                'agent_id': self.agent_id,
                'agent_name': self.name,
                'status': AgentStatus.SUCCESS if HAS_MATRIX_FRAMEWORK else 'success',
                'execution_time': execution_time,
                'security_analysis_result': final_result,
                'vulnerability_summary': {
                    'total': final_result.total_vulnerabilities,
                    'critical': final_result.critical_vulnerabilities,
                    'high': final_result.high_vulnerabilities,
                    'medium': final_result.medium_vulnerabilities,
                    'low': final_result.low_vulnerabilities
                },
                'security_metrics': {
                    'security_score': final_result.security_score,
                    'exploit_risk_score': final_result.exploit_risk_score,
                    'total_exploit_chains': len(final_result.exploit_chains),
                    'recommendations_count': len(final_result.recommendations)
                },
                'vulnerabilities': final_result.vulnerabilities,
                'security_patterns': final_result.security_patterns,
                'exploit_chains': final_result.exploit_chains,
                'recommendations': final_result.recommendations,
                'analysis_stats': self.analysis_stats,
                'metadata': {
                    'character': self.character,
                    'phase': self.current_phase,
                    'files_analyzed': len(source_data),
                    'warnings': len(final_result.warnings),
                    'errors': len(final_result.error_messages)
                }
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent Johnson security analysis failed in {self.current_phase}: {str(e)}"
            self.logger.error(error_msg)
            
            return self._create_failure_result(error_msg, execution_time)

    def _validate_security_dependencies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security analysis dependencies"""
        agent_results = context.get('agent_results', {})
        
        # Check Link (Agent 12) result
        if 12 not in agent_results:
            return {'valid': False, 'error': 'Link (Agent 12) result not available'}
        
        link_result = agent_results[12]
        status = link_result.get('status', 'unknown')
        
        if status != 'success' and status != AgentStatus.SUCCESS:
            return {'valid': False, 'error': 'Link integration validation failed'}
        
        return {'valid': True, 'error': None}

    def _extract_source_code(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Extract source code from various agent results"""
        agent_results = context.get('agent_results', {})
        source_code = {}
        
        # Priority order for source code extraction
        source_agents = [12, 11, 10, 9, 7]  # Link, Oracle, Machine, Commander Locke, Trainman
        
        for agent_id in source_agents:
            if agent_id in agent_results:
                result = agent_results[agent_id]
                extracted_code = self._extract_code_from_agent_result(agent_id, result)
                source_code.update(extracted_code)
                
                if source_code:  # Found source code
                    self.logger.info(f"Source code extracted from Agent {agent_id}")
                    break
        
        return source_code

    def _extract_code_from_agent_result(self, agent_id: int, result: Dict[str, Any]) -> Dict[str, str]:
        """Extract source code from individual agent result"""
        source_code = {}
        
        try:
            data = result.get('data', {})
            
            # From Link or Oracle (comprehensive results)
            if agent_id in [12, 11]:
                if isinstance(data, dict):
                    # Look for integration results or validation results
                    integration_result = data.get('integration_result', {})
                    if hasattr(integration_result, 'data_flows'):
                        data_flows = integration_result.data_flows
                        if isinstance(data_flows, dict):
                            for flow_name, flow_data in data_flows.items():
                                if isinstance(flow_data, dict):
                                    source_files = flow_data.get('source_files', {})
                                    header_files = flow_data.get('header_files', {})
                                    source_code.update(source_files)
                                    source_code.update(header_files)
            
            # From Machine or Commander Locke (reconstructed source)
            elif agent_id in [10, 9]:
                if isinstance(data, dict):
                    # Look for reconstructed source or build results
                    reconstructed_source = data.get('reconstructed_source', {})
                    if isinstance(reconstructed_source, dict):
                        source_files = reconstructed_source.get('source_files', {})
                        header_files = reconstructed_source.get('header_files', {})
                        source_code.update(source_files)
                        source_code.update(header_files)
                    
                    # Also check for final polish results
                    final_polish = data.get('final_polish', {})
                    if isinstance(final_polish, dict):
                        polished_files = final_polish.get('polished_files', {})
                        source_code.update(polished_files)
            
            # From other agents (decompiled functions)
            else:
                if isinstance(data, dict):
                    functions = data.get('enhanced_functions', {}) or data.get('decompiled_functions', {})
                    for func_name, func_data in functions.items():
                        if isinstance(func_data, dict) and func_data.get('code'):
                            source_code[f"{func_name}.c"] = func_data['code']
        
        except Exception as e:
            self.logger.warning(f"Failed to extract code from agent {agent_id}: {e}")
        
        return source_code

    def _perform_comprehensive_vulnerability_analysis(self, source_data: Dict[str, str]) -> Dict[str, Any]:
        """Perform comprehensive vulnerability analysis"""
        analysis = {
            'vulnerabilities': [],
            'vulnerability_statistics': {},
            'buffer_overflows': [],
            'memory_leaks': [],
            'integer_overflows': [],
            'format_string_bugs': [],
            'injection_vulnerabilities': [],
            'race_conditions': [],
            'privilege_escalation': [],
            'information_disclosure': [],
            'denial_of_service': []
        }
        
        # Execute all vulnerability detection methods
        analysis['buffer_overflows'] = self._detect_buffer_overflows(source_data)
        analysis['memory_leaks'] = self._detect_memory_leaks(source_data)
        analysis['integer_overflows'] = self._detect_integer_overflows(source_data)
        analysis['format_string_bugs'] = self._detect_format_string_vulnerabilities(source_data)
        analysis['injection_vulnerabilities'] = self._detect_injection_vulnerabilities(source_data)
        analysis['race_conditions'] = self._detect_race_conditions(source_data)
        analysis['privilege_escalation'] = self._detect_privilege_escalation(source_data)
        analysis['information_disclosure'] = self._detect_information_disclosure(source_data)
        analysis['denial_of_service'] = self._detect_denial_of_service(source_data)
        
        # Compile all vulnerabilities with categorization
        all_vulnerabilities = []
        categories = [
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
        
        for category, vulnerabilities in categories:
            for vuln in vulnerabilities:
                vuln['category'] = category
                all_vulnerabilities.append(vuln)
        
        analysis['vulnerabilities'] = all_vulnerabilities
        analysis['vulnerability_statistics'] = self._calculate_vulnerability_statistics(analysis)
        
        return analysis

    def _detect_buffer_overflows(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect buffer overflow vulnerabilities"""
        vulnerabilities = []
        
        # Dangerous functions that can cause buffer overflows
        dangerous_functions = {
            'strcpy': {'severity': 'high', 'recommendation': 'Use strncpy or strlcpy instead'},
            'strcat': {'severity': 'high', 'recommendation': 'Use strncat or strlcat instead'},
            'sprintf': {'severity': 'high', 'recommendation': 'Use snprintf instead'},
            'vsprintf': {'severity': 'high', 'recommendation': 'Use vsnprintf instead'},
            'gets': {'severity': 'critical', 'recommendation': 'Use fgets instead'},
            'scanf': {'severity': 'medium', 'recommendation': 'Use safer input methods with length limits'},
            'fscanf': {'severity': 'medium', 'recommendation': 'Use safer input methods with length limits'},
            'strncpy': {'severity': 'low', 'recommendation': 'Ensure proper null termination'}
        }
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Check for dangerous function usage
                for func_name, func_info in dangerous_functions.items():
                    pattern = rf'\b{func_name}\s*\('
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        context = self._extract_code_context(content, match.start(), 2)
                        
                        vulnerabilities.append({
                            'type': 'buffer_overflow',
                            'severity': func_info['severity'],
                            'file': filename,
                            'line': line_num,
                            'function': func_name,
                            'description': f'Potentially unsafe use of {func_name}()',
                            'recommendation': func_info['recommendation'],
                            'context': context,
                            'cwe': 'CWE-120'  # Classic Buffer Overflow
                        })
                
                # Check for array access without bounds checking
                array_pattern = r'(\w+)\s*\[\s*(\w+)\s*\]'
                for match in re.finditer(array_pattern, content):
                    array_name, index_var = match.groups()
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Skip constant array access and simple indices
                    if index_var.isdigit() or index_var in ['0', '1', '2']:
                        continue
                    
                    # Check for bounds checking nearby
                    context_before = content[max(0, match.start()-300):match.start()]
                    context_after = content[match.end():match.end()+300]
                    
                    bounds_check_patterns = [
                        rf'if\s*\(\s*{index_var}\s*<',
                        rf'if\s*\(\s*{index_var}\s*>=',
                        rf'if\s*\(\s*{index_var}\s*>',
                        rf'if\s*\(\s*{index_var}\s*<=',
                        rf'assert\s*\(\s*{index_var}',
                        rf'while\s*\(\s*{index_var}\s*<',
                        rf'for\s*\([^)]*{index_var}\s*<'
                    ]
                    
                    has_bounds_check = any(re.search(pattern, context_before + context_after) 
                                         for pattern in bounds_check_patterns)
                    
                    if not has_bounds_check:
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
        """Detect memory leak vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Track memory allocations and deallocations
                allocations = {}
                deallocations = set()
                
                # Find allocation calls
                alloc_pattern = r'\b(malloc|calloc|realloc)\s*\('
                for match in re.finditer(alloc_pattern, content):
                    line_num = content[:match.start()].count('\n') + 1
                    func_name = match.group(1)
                    
                    # Try to identify the variable being assigned
                    line_start = content.rfind('\n', 0, match.start()) + 1
                    line_end = content.find('\n', match.end())
                    if line_end == -1:
                        line_end = len(content)
                    line_content = content[line_start:line_end]
                    
                    # Look for assignment pattern
                    assign_match = re.search(r'(\w+)\s*=\s*' + func_name, line_content)
                    if assign_match:
                        var_name = assign_match.group(1)
                        allocations[var_name] = {
                            'function': func_name,
                            'line': line_num,
                            'position': match.start()
                        }
                
                # Find deallocation calls
                free_pattern = r'\bfree\s*\(\s*(\w+)\s*\)'
                for match in re.finditer(free_pattern, content):
                    var_name = match.group(1)
                    deallocations.add(var_name)
                
                # Check for potential memory leaks
                for var_name, alloc_info in allocations.items():
                    if var_name not in deallocations:
                        vulnerabilities.append({
                            'type': 'memory_leak',
                            'severity': 'medium',
                            'file': filename,
                            'line': alloc_info['line'],
                            'description': f'Potential memory leak: {alloc_info["function"]}() allocated to {var_name} without corresponding free()',
                            'recommendation': f'Ensure {var_name} is freed before going out of scope',
                            'context': self._extract_code_context(content, alloc_info['position'], 2),
                            'cwe': 'CWE-401'  # Memory Leak
                        })
        
        return vulnerabilities

    def _detect_integer_overflows(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect integer overflow vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Look for arithmetic operations without overflow checking
                arithmetic_pattern = r'(\w+)\s*([+\-*/])\s*(\w+)'
                for match in re.finditer(arithmetic_pattern, content):
                    var1, operator, var2 = match.groups()
                    line_num = content[:match.start()].count('\n') + 1
                    
                    # Skip obvious constants and simple operations
                    if (var1.isdigit() and var2.isdigit()) or var1 in ['1', '0'] or var2 in ['1', '0']:
                        continue
                    
                    # Look for overflow checking patterns
                    context = content[max(0, match.start()-200):match.end()+200]
                    overflow_check_patterns = [
                        r'if\s*\([^)]*overflow',
                        r'if\s*\([^)]*MAX',
                        r'if\s*\([^)]*LIMIT',
                        r'assert\s*\([^)]*MAX',
                        r'check.*overflow',
                        rf'if\s*\(\s*{var1}\s*>\s*\w+\s*-\s*{var2}\s*\)'  # Addition overflow check
                    ]
                    
                    has_overflow_check = any(re.search(pattern, context, re.IGNORECASE) 
                                           for pattern in overflow_check_patterns)
                    
                    # Focus on potentially dangerous operations
                    if not has_overflow_check and operator in ['+', '*']:
                        vulnerabilities.append({
                            'type': 'integer_overflow',
                            'severity': 'medium',
                            'file': filename,
                            'line': line_num,
                            'description': f'Potential integer overflow in arithmetic operation: {var1} {operator} {var2}',
                            'recommendation': 'Add overflow checking for arithmetic operations',
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-190'  # Integer Overflow
                        })
        
        return vulnerabilities

    def _detect_format_string_vulnerabilities(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
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
                            # Additional check for dangerous patterns
                            if not any(safe_pattern in format_arg.lower() for safe_pattern in ['const', 'literal', 'static']):
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
                    (r'(SELECT|INSERT|UPDATE|DELETE)\s+.*\+\s*\w+', 'SQL injection via string concatenation'),
                    (r'(SELECT|INSERT|UPDATE|DELETE)\s+.*%s', 'SQL injection via format string'),
                    (r'sprintf\s*\([^,]+,\s*[^,]*SELECT', 'SQL injection via sprintf'),
                    (r'query.*\+.*user', 'Potential SQL injection with user input')
                ]
                
                for pattern, description in sql_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'sql_injection',
                            'severity': 'high',
                            'file': filename,
                            'line': line_num,
                            'description': description,
                            'recommendation': 'Use parameterized queries or prepared statements',
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-89'  # SQL Injection
                        })
                
                # Command injection patterns
                command_patterns = [
                    (r'system\s*\([^)]*\+', 'Command injection via system()'),
                    (r'exec\w*\s*\([^)]*\+', 'Command injection via exec family'),
                    (r'popen\s*\([^)]*\+', 'Command injection via popen()'),
                    (r'CreateProcess.*\+', 'Command injection via CreateProcess')
                ]
                
                for pattern, description in command_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'command_injection',
                            'severity': 'high',
                            'file': filename,
                            'line': line_num,
                            'description': description,
                            'recommendation': 'Validate and sanitize input before executing commands',
                            'context': self._extract_code_context(content, match.start(), 2),
                            'cwe': 'CWE-78'  # Command Injection
                        })
        
        return vulnerabilities

    def _detect_race_conditions(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect race condition vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Look for threading and shared resource access
                thread_indicators = [
                    'CreateThread', 'pthread_create', '_beginthread', '_beginthreadex',
                    'std::thread', 'boost::thread', 'ThreadPool'
                ]
                
                has_threading = any(indicator in content for indicator in thread_indicators)
                
                if has_threading:
                    # Find global variables
                    global_vars = set()
                    global_pattern = r'^(?:static\s+)?(?:extern\s+)?(\w+(?:\s*\*)*)\s+(\w+)(?:\s*=\s*[^;]+)?\s*;'
                    for match in re.finditer(global_pattern, content, re.MULTILINE):
                        if not self._is_inside_function(content, match.start()):
                            global_vars.add(match.group(2))
                    
                    # Check for unsynchronized access to global variables
                    sync_keywords = ['mutex', 'lock', 'critical', 'atomic', 'volatile', 'synchronized']
                    
                    for var in global_vars:
                        var_pattern = rf'\b{var}\b(?!\s*\()'  # Variable access, not function call
                        accesses = list(re.finditer(var_pattern, content))
                        
                        if len(accesses) > 1:  # Multiple accesses to global variable
                            for match in accesses[:3]:  # Check first few accesses
                                line_num = content[:match.start()].count('\n') + 1
                                context = content[max(0, match.start()-150):match.end()+150]
                                
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
        """Detect privilege escalation vulnerabilities"""
        vulnerabilities = []
        
        dangerous_functions = {
            'SetThreadToken': {'severity': 'high', 'description': 'Token manipulation function'},
            'ImpersonateLoggedOnUser': {'severity': 'high', 'description': 'User impersonation function'},
            'SetPrivilege': {'severity': 'high', 'description': 'Privilege modification function'},
            'AdjustTokenPrivileges': {'severity': 'high', 'description': 'Token privilege adjustment'},
            'LoadLibrary': {'severity': 'medium', 'description': 'Dynamic library loading (DLL hijacking risk)'},
            'LoadLibraryEx': {'severity': 'medium', 'description': 'Extended library loading'},
            'CreateProcess': {'severity': 'medium', 'description': 'Process creation function'},
            'CreateProcessAsUser': {'severity': 'high', 'description': 'Process creation as different user'},
            'LogonUser': {'severity': 'high', 'description': 'User logon function'},
            'setuid': {'severity': 'high', 'description': 'Set user ID function'},
            'seteuid': {'severity': 'high', 'description': 'Set effective user ID'},
            'setgid': {'severity': 'medium', 'description': 'Set group ID function'},
            'chmod': {'severity': 'low', 'description': 'Change file permissions'}
        }
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                for func_name, func_info in dangerous_functions.items():
                    pattern = rf'\b{func_name}\s*\('
                    for match in re.finditer(pattern, content):
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Check for proper validation nearby
                        context = content[max(0, match.start()-200):match.end()+200]
                        has_validation = any(keyword in context.lower() for keyword in 
                                           ['check', 'validate', 'verify', 'assert', 'privilege'])
                        
                        recommendation = 'Validate privileges and add proper access controls'
                        if not has_validation:
                            recommendation = 'Add privilege validation before calling ' + func_name
                        
                        vulnerabilities.append({
                            'type': 'privilege_escalation',
                            'severity': func_info['severity'],
                            'file': filename,
                            'line': line_num,
                            'function': func_name,
                            'description': f'Potential privilege escalation risk: {func_info["description"]}',
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
                # Hardcoded sensitive information
                sensitive_patterns = [
                    (r'password\s*=\s*["\'][^"\']{3,}["\']', 'Hardcoded password', 'high'),
                    (r'api[_\s]*key\s*=\s*["\'][^"\']{8,}["\']', 'Hardcoded API key', 'high'),
                    (r'secret\s*=\s*["\'][^"\']{6,}["\']', 'Hardcoded secret', 'high'),
                    (r'token\s*=\s*["\'][^"\']{16,}["\']', 'Hardcoded token', 'high'),
                    (r'private[_\s]*key\s*=\s*["\'][^"\']{20,}["\']', 'Hardcoded private key', 'critical'),
                    (r'(?:username|user)\s*=\s*["\'][^"\']+["\']', 'Hardcoded username', 'medium'),
                    (r'(?:host|server|url)\s*=\s*["\'][^"\']+["\']', 'Hardcoded server info', 'low')
                ]
                
                for pattern, description, severity in sensitive_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'information_disclosure',
                            'severity': severity,
                            'file': filename,
                            'line': line_num,
                            'description': description,
                            'recommendation': 'Use environment variables or secure configuration files',
                            'context': self._extract_code_context(content, match.start(), 1),
                            'cwe': 'CWE-798'  # Hardcoded Credentials
                        })
                
                # Debug and error information leakage
                debug_patterns = [
                    (r'printf\s*\([^)]*(?:debug|error|exception)', 'Debug information in output', 'low'),
                    (r'cout\s*<<.*(?:debug|error|exception)', 'Debug information in output', 'low'),
                    (r'fprintf\s*\(\s*stderr[^)]*(?:debug|error)', 'Error details in stderr', 'low'),
                    (r'throw.*Exception\s*\([^)]*["\'][^"\']*(?:internal|system)', 'Internal details in exceptions', 'medium')
                ]
                
                for pattern, description, severity in debug_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        vulnerabilities.append({
                            'type': 'information_disclosure',
                            'severity': severity,
                            'file': filename,
                            'line': line_num,
                            'description': description,
                            'recommendation': 'Remove debug output or sanitize error messages in production',
                            'context': self._extract_code_context(content, match.start(), 1),
                            'cwe': 'CWE-209'  # Information Exposure Through Error Messages
                        })
        
        return vulnerabilities

    def _detect_denial_of_service(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Detect denial of service vulnerabilities"""
        vulnerabilities = []
        
        for filename, content in source_data.items():
            if isinstance(content, str):
                # Infinite loops without proper termination
                loop_patterns = [
                    (r'while\s*\(\s*(?:1|true|TRUE)\s*\)', 'Infinite while loop'),
                    (r'for\s*\(\s*;\s*;\s*\)', 'Infinite for loop'),
                    (r'do\s*{[^}]*}\s*while\s*\(\s*(?:1|true|TRUE)\s*\)', 'Infinite do-while loop')
                ]
                
                for pattern, description in loop_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Check if there's a break condition within the loop
                        loop_body = self._extract_loop_body(content, match.end())
                        has_exit = any(keyword in loop_body for keyword in 
                                     ['break', 'return', 'exit', 'goto', 'continue'])
                        
                        if not has_exit:
                            vulnerabilities.append({
                                'type': 'denial_of_service',
                                'severity': 'medium',
                                'file': filename,
                                'line': line_num,
                                'description': f'{description} without apparent exit condition',
                                'recommendation': 'Add proper termination conditions to prevent infinite loops',
                                'context': self._extract_code_context(content, match.start(), 3),
                                'cwe': 'CWE-835'  # Infinite Loop
                            })
                
                # Resource exhaustion patterns
                resource_patterns = [
                    (r'malloc\s*\([^)]*\*\s*(\w+)\)', 'Potential memory exhaustion via malloc'),
                    (r'new\s+\w+\[[^]]*\*\s*(\w+)\]', 'Potential memory exhaustion via new[]'),
                    (r'fopen\s*\([^)]*user', 'File handle exhaustion via user input'),
                    (r'socket\s*\([^)]*\).*loop', 'Socket exhaustion in loop'),
                    (r'CreateThread.*while', 'Thread exhaustion in loop')
                ]
                
                for pattern, description in resource_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # Check for resource limits
                        context = content[max(0, match.start()-300):match.end()+300]
                        has_limits = any(keyword in context.lower() for keyword in 
                                       ['limit', 'max', 'bound', 'check', 'validate'])
                        
                        if not has_limits:
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
            if i < len(lines):
                prefix = ">>> " if i == line_num else "    "
                context_lines.append(f"{prefix}{i+1:3d}: {lines[i]}")
        
        return '\n'.join(context_lines)

    def _extract_loop_body(self, content: str, loop_start: int) -> str:
        """Extract the body of a loop starting after the loop header"""
        brace_pos = content.find('{', loop_start)
        if brace_pos == -1:
            return ""
        
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
        """Calculate comprehensive vulnerability statistics"""
        vulnerabilities = analysis.get('vulnerabilities', [])
        
        stats = {
            'total_vulnerabilities': len(vulnerabilities),
            'critical_vulnerabilities': 0,
            'high_vulnerabilities': 0,
            'medium_vulnerabilities': 0,
            'low_vulnerabilities': 0,
            'vulnerability_by_category': {},
            'vulnerability_by_file': {},
            'most_common_vulnerability': None,
            'security_score': 0.0,
            'risk_assessment': 'unknown'
        }
        
        # Count by severity
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'unknown')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        stats.update({
            'critical_vulnerabilities': severity_counts['critical'],
            'high_vulnerabilities': severity_counts['high'],
            'medium_vulnerabilities': severity_counts['medium'],
            'low_vulnerabilities': severity_counts['low']
        })
        
        # Count by category
        category_counts = {}
        file_counts = {}
        for vuln in vulnerabilities:
            category = vuln.get('category', 'unknown')
            filename = vuln.get('file', 'unknown')
            
            category_counts[category] = category_counts.get(category, 0) + 1
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        stats['vulnerability_by_category'] = category_counts
        stats['vulnerability_by_file'] = file_counts
        
        # Most common vulnerability
        if category_counts:
            stats['most_common_vulnerability'] = max(category_counts, key=category_counts.get)
        
        # Calculate security score (0-1, where 1 is most secure)
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
            # Normalize to 0-1 scale
            max_reasonable_score = 50  # Assume this as a reasonable maximum
            stats['security_score'] = max(0.0, 1.0 - (weighted_score / max_reasonable_score))
        
        # Risk assessment
        if stats['critical_vulnerabilities'] > 0:
            stats['risk_assessment'] = 'critical'
        elif stats['high_vulnerabilities'] > 3:
            stats['risk_assessment'] = 'high'
        elif stats['high_vulnerabilities'] > 0 or stats['medium_vulnerabilities'] > 5:
            stats['risk_assessment'] = 'medium'
        elif stats['medium_vulnerabilities'] > 0 or stats['low_vulnerabilities'] > 10:
            stats['risk_assessment'] = 'low'
        else:
            stats['risk_assessment'] = 'minimal'
        
        return stats

    def _analyze_security_patterns(self, source_data: Dict[str, str]) -> Dict[str, Any]:
        """Analyze security patterns and practices in the code"""
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
        
        patterns['input_validation_patterns'] = self._find_input_validation_patterns(source_data)
        patterns['authentication_patterns'] = self._find_authentication_patterns(source_data)
        patterns['encryption_patterns'] = self._find_encryption_patterns(source_data)
        patterns['secure_coding_patterns'] = self._find_secure_coding_patterns(source_data)
        patterns['insecure_patterns'] = self._find_insecure_patterns(source_data)
        
        # Calculate pattern analysis
        total_secure = (
            len(patterns['input_validation_patterns']) +
            len(patterns['authentication_patterns']) +
            len(patterns['encryption_patterns']) +
            len(patterns['secure_coding_patterns'])
        )
        total_insecure = len(patterns['insecure_patterns'])
        total_patterns = total_secure + total_insecure
        
        patterns['pattern_analysis'] = {
            'total_secure_patterns': total_secure,
            'total_insecure_patterns': total_insecure,
            'security_pattern_ratio': total_secure / max(1, total_patterns),
            'pattern_quality': 'poor'
        }
        
        # Determine pattern quality
        ratio = patterns['pattern_analysis']['security_pattern_ratio']
        if ratio >= 0.8:
            patterns['pattern_analysis']['pattern_quality'] = 'excellent'
        elif ratio >= 0.6:
            patterns['pattern_analysis']['pattern_quality'] = 'good'
        elif ratio >= 0.4:
            patterns['pattern_analysis']['pattern_quality'] = 'fair'
        
        return patterns

    def _find_input_validation_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find input validation patterns"""
        patterns = []
        
        validation_patterns = [
            (r'if\s*\([^)]*(?:!=|==)\s*NULL\s*\)', 'null_check'),
            (r'if\s*\([^)]*length[^)]*\)', 'length_check'),
            (r'if\s*\([^)]*strlen[^)]*\)', 'string_length_check'),
            (r'assert\s*\([^)]*\)', 'assertion'),
            (r'isdigit\s*\([^)]*\)', 'numeric_validation'),
            (r'isalpha\s*\([^)]*\)', 'alphabetic_validation'),
            (r'isalnum\s*\([^)]*\)', 'alphanumeric_validation'),
            (r'if\s*\([^)]*[<>]=?\s*\d+', 'range_check'),
            (r'validate\w*\s*\(', 'validation_function'),
            (r'sanitize\w*\s*\(', 'sanitization_function')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type in validation_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'pattern': match.group(0).strip()
                    })
        
        return patterns

    def _find_authentication_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find authentication patterns"""
        patterns = []
        
        auth_patterns = [
            (r'\bpassword\b', 'password_handling'),
            (r'\bauthenticat\w*', 'authentication_function'),
            (r'\blogin\b', 'login_function'),
            (r'\btoken\b', 'token_handling'),
            (r'\bsession\b', 'session_management'),
            (r'\bcredential\w*', 'credential_handling'),
            (r'\bauthoriz\w*', 'authorization_function'),
            (r'\bjwt\b', 'jwt_token'),
            (r'\boauth\b', 'oauth_authentication'),
            (r'\bverify\w*', 'verification_function')
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
        """Find encryption and cryptographic patterns"""
        patterns = []
        
        crypto_patterns = [
            (r'\bencrypt\w*', 'encryption'),
            (r'\bdecrypt\w*', 'decryption'),
            (r'\bhash\w*', 'hashing'),
            (r'\baes\b', 'aes_encryption'),
            (r'\brsa\b', 'rsa_encryption'),
            (r'\bsha\d*\b', 'sha_hashing'),
            (r'\bmd5\b', 'md5_hashing'),
            (r'\bhmac\b', 'hmac'),
            (r'\bcipher\b', 'cipher'),
            (r'\bcrypto\w*', 'cryptographic_function'),
            (r'\bssl\b', 'ssl_tls'),
            (r'\btls\b', 'ssl_tls')
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
            (r'\bstrncpy\s*\(', 'safe_string_copy'),
            (r'\bsnprintf\s*\(', 'safe_string_format'),
            (r'\bstrncat\s*\(', 'safe_string_concat'),
            (r'\bmemset\s*\([^,]+,\s*0', 'memory_clearing'),
            (r'\bfree\s*\([^)]+\);\s*[^;]*=\s*NULL', 'safe_memory_free'),
            (r'\bstatic_assert\s*\(', 'compile_time_check'),
            (r'\b_Static_assert\s*\(', 'compile_time_check'),
            (r'\bsecure_memcpy\s*\(', 'secure_memory_operation'),
            (r'\bconstexpr\s+', 'constant_expression'),
            (r'\bconst\s+', 'immutable_data')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type in secure_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'pattern': match.group(0).strip()
                    })
        
        return patterns

    def _find_insecure_patterns(self, source_data: Dict[str, str]) -> List[Dict[str, Any]]:
        """Find insecure coding patterns"""
        patterns = []
        
        insecure_patterns = [
            (r'\bsystem\s*\(', 'system_call', 'high'),
            (r'\beval\s*\(', 'eval_usage', 'high'),
            (r'\bexec\w*\s*\(', 'exec_usage', 'high'),
            (r'\brand\s*\(\s*\)', 'weak_random', 'medium'),
            (r'\btmp\w*', 'temp_file_usage', 'medium'),
            (r'\bstrcpy\s*\(', 'unsafe_string_copy', 'high'),
            (r'\bstrcat\s*\(', 'unsafe_string_concat', 'high'),
            (r'\bsprintf\s*\(', 'unsafe_string_format', 'medium'),
            (r'\bgets\s*\(', 'unsafe_input', 'critical'),
            (r'TODO.*security', 'security_todo', 'low'),
            (r'FIXME.*security', 'security_fixme', 'medium')
        ]
        
        for filename, content in source_data.items():
            for pattern, pattern_type, risk_level in insecure_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    patterns.append({
                        'type': pattern_type,
                        'file': filename,
                        'line': line_num,
                        'pattern': match.group(0).strip(),
                        'risk': risk_level
                    })
        
        return patterns

    def _detect_exploit_potential(self, vulnerability_analysis: Dict[str, Any], 
                                security_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Detect potential exploit scenarios and attack vectors"""
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
                    'impact': exploit_info['impact'],
                    'exploitability_score': exploit_info['score']
                })
        
        # Find exploit chains (combinations of vulnerabilities)
        detection['exploit_chains'] = self._find_exploit_chains(vulnerabilities)
        
        # Calculate overall exploit risk score
        detection['exploit_risk_score'] = self._calculate_exploit_risk_score(detection)
        
        # Prioritize mitigations
        detection['mitigation_priority'] = self._prioritize_mitigations(detection, vulnerabilities)
        
        return detection

    def _assess_exploit_potential(self, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the exploit potential of a single vulnerability"""
        exploit_info = {
            'exploitable': False,
            'method': 'unknown',
            'difficulty': 'unknown',
            'impact': 'unknown',
            'score': 0.0
        }
        
        vuln_type = vulnerability.get('type', '')
        severity = vulnerability.get('severity', 'low')
        
        # Define exploit characteristics by vulnerability type
        exploit_characteristics = {
            'buffer_overflow': {'exploitable': True, 'method': 'memory_corruption', 'difficulty': 'medium', 'impact': 'high', 'score': 0.8},
            'format_string': {'exploitable': True, 'method': 'memory_corruption', 'difficulty': 'medium', 'impact': 'high', 'score': 0.8},
            'command_injection': {'exploitable': True, 'method': 'command_execution', 'difficulty': 'easy', 'impact': 'high', 'score': 0.9},
            'sql_injection': {'exploitable': True, 'method': 'data_extraction', 'difficulty': 'easy', 'impact': 'high', 'score': 0.9},
            'privilege_escalation': {'exploitable': True, 'method': 'privilege_abuse', 'difficulty': 'hard', 'impact': 'high', 'score': 0.7},
            'race_condition': {'exploitable': True, 'method': 'timing_attack', 'difficulty': 'hard', 'impact': 'medium', 'score': 0.5},
            'integer_overflow': {'exploitable': True, 'method': 'arithmetic_manipulation', 'difficulty': 'hard', 'impact': 'medium', 'score': 0.6},
            'information_disclosure': {'exploitable': True, 'method': 'information_gathering', 'difficulty': 'easy', 'impact': 'low', 'score': 0.4},
            'denial_of_service': {'exploitable': True, 'method': 'resource_exhaustion', 'difficulty': 'easy', 'impact': 'medium', 'score': 0.6},
            'memory_leak': {'exploitable': False, 'method': 'resource_exhaustion', 'difficulty': 'medium', 'impact': 'low', 'score': 0.3}
        }
        
        if vuln_type in exploit_characteristics:
            characteristics = exploit_characteristics[vuln_type]
            exploit_info.update(characteristics)
            
            # Adjust based on severity
            severity_multiplier = {'critical': 1.2, 'high': 1.0, 'medium': 0.8, 'low': 0.6}.get(severity, 0.6)
            exploit_info['score'] *= severity_multiplier
            exploit_info['score'] = min(1.0, exploit_info['score'])
        
        return exploit_info

    def _find_exploit_chains(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find potential exploit chains by combining vulnerabilities"""
        chains = []
        
        # Define chainable vulnerability combinations
        chain_patterns = [
            ('information_disclosure', 'buffer_overflow', 'Information gathering followed by memory corruption'),
            ('privilege_escalation', 'command_injection', 'Privilege elevation with command execution'),
            ('race_condition', 'privilege_escalation', 'Race condition leading to privilege escalation'),
            ('integer_overflow', 'buffer_overflow', 'Integer overflow causing buffer overflow'),
            ('format_string', 'command_injection', 'Format string leading to command injection'),
            ('sql_injection', 'information_disclosure', 'SQL injection revealing sensitive data'),
            ('denial_of_service', 'race_condition', 'DoS creating race condition opportunity')
        ]
        
        # Group vulnerabilities by type
        vulns_by_type = {}
        for vuln in vulnerabilities:
            vuln_type = vuln.get('type', '')
            if vuln_type not in vulns_by_type:
                vulns_by_type[vuln_type] = []
            vulns_by_type[vuln_type].append(vuln)
        
        # Look for exploit chains
        for type1, type2, description in chain_patterns:
            if type1 in vulns_by_type and type2 in vulns_by_type:
                for vuln1 in vulns_by_type[type1]:
                    for vuln2 in vulns_by_type[type2]:
                        # Check if vulnerabilities are in same file or related files
                        if self._vulnerabilities_can_chain(vuln1, vuln2):
                            chains.append({
                                'vulnerabilities': [vuln1, vuln2],
                                'chain_type': f"{type1}_to_{type2}",
                                'description': description,
                                'combined_impact': self._calculate_chain_impact(vuln1, vuln2),
                                'exploitability': self._calculate_chain_exploitability(vuln1, vuln2)
                            })
        
        return chains

    def _vulnerabilities_can_chain(self, vuln1: Dict[str, Any], vuln2: Dict[str, Any]) -> bool:
        """Check if two vulnerabilities can be chained together"""
        # Same file makes chaining more likely
        if vuln1.get('file') == vuln2.get('file'):
            return True
        
        # Close line numbers in same file
        if vuln1.get('file') == vuln2.get('file'):
            line_diff = abs(vuln1.get('line', 0) - vuln2.get('line', 0))
            if line_diff < 50:  # Within 50 lines
                return True
        
        # Related files (e.g., same module)
        file1 = vuln1.get('file', '')
        file2 = vuln2.get('file', '')
        if file1 and file2:
            # Check if files have common prefix (same module)
            common_prefix = os.path.commonprefix([file1, file2])
            if len(common_prefix) > len(os.path.dirname(file1)) // 2:
                return True
        
        return False

    def _calculate_chain_impact(self, vuln1: Dict[str, Any], vuln2: Dict[str, Any]) -> str:
        """Calculate the combined impact of chained vulnerabilities"""
        severity_scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        score1 = severity_scores.get(vuln1.get('severity', 'low'), 1)
        score2 = severity_scores.get(vuln2.get('severity', 'low'), 1)
        
        # Chain multiplier effect
        combined_score = score1 + score2 + 1  # +1 for chain effect
        
        if combined_score >= 7:
            return 'critical'
        elif combined_score >= 5:
            return 'high'
        elif combined_score >= 3:
            return 'medium'
        else:
            return 'low'

    def _calculate_chain_exploitability(self, vuln1: Dict[str, Any], vuln2: Dict[str, Any]) -> float:
        """Calculate exploitability score for vulnerability chain"""
        # Base scores for individual vulnerabilities
        exploit1 = self._assess_exploit_potential(vuln1)
        exploit2 = self._assess_exploit_potential(vuln2)
        
        # Chain effect: easier if first vuln enables second
        chain_bonus = 0.2 if exploit1['difficulty'] == 'easy' else 0.1
        
        combined_score = (exploit1['score'] + exploit2['score']) / 2 + chain_bonus
        return min(1.0, combined_score)

    def _calculate_exploit_risk_score(self, detection: Dict[str, Any]) -> float:
        """Calculate overall exploit risk score"""
        attack_vectors = detection.get('attack_vectors', [])
        exploit_chains = detection.get('exploit_chains', [])
        
        if not attack_vectors and not exploit_chains:
            return 0.0
        
        # Score individual attack vectors
        vector_scores = []
        for vector in attack_vectors:
            vector_scores.append(vector.get('exploitability_score', 0.0))
        
        # Score exploit chains (higher risk due to combination)
        chain_scores = []
        for chain in exploit_chains:
            chain_scores.append(chain.get('exploitability', 0.0))
        
        # Combine scores with weights
        avg_vector_score = sum(vector_scores) / len(vector_scores) if vector_scores else 0.0
        avg_chain_score = sum(chain_scores) / len(chain_scores) if chain_scores else 0.0
        
        # Chains are weighted higher due to increased risk
        overall_score = (avg_vector_score * 0.6 + avg_chain_score * 0.8)
        
        # Normalize based on number of vulnerabilities
        volume_factor = min(1.0, (len(attack_vectors) + len(exploit_chains)) / 10.0)
        
        return min(1.0, overall_score * (1.0 + volume_factor * 0.3))

    def _prioritize_mitigations(self, detection: Dict[str, Any], 
                              vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize vulnerability mitigations based on risk"""
        priorities = []
        
        # High priority: vulnerabilities in exploit chains
        chain_vulns = set()
        for chain in detection.get('exploit_chains', []):
            for vuln in chain['vulnerabilities']:
                chain_vulns.add(id(vuln))
                priorities.append({
                    'vulnerability': vuln,
                    'priority': 'critical',
                    'reason': f'Part of exploit chain: {chain["description"]}',
                    'urgency_score': 10
                })
        
        # High priority: easily exploitable vulnerabilities
        for vector in detection.get('attack_vectors', []):
            vuln = vector['vulnerability']
            if id(vuln) not in chain_vulns:
                difficulty = vector.get('difficulty', 'unknown')
                impact = vector.get('impact', 'unknown')
                
                if difficulty == 'easy' and impact == 'high':
                    priorities.append({
                        'vulnerability': vuln,
                        'priority': 'critical',
                        'reason': 'Easy to exploit with high impact',
                        'urgency_score': 9
                    })
                elif difficulty == 'easy' or impact == 'high':
                    priorities.append({
                        'vulnerability': vuln,
                        'priority': 'high',
                        'reason': f'Easily exploitable or high impact ({difficulty} difficulty, {impact} impact)',
                        'urgency_score': 7
                    })
        
        # Medium priority: other high-severity vulnerabilities
        for vuln in vulnerabilities:
            if id(vuln) not in {id(p['vulnerability']) for p in priorities}:
                severity = vuln.get('severity', 'low')
                if severity in ['critical', 'high']:
                    urgency = 6 if severity == 'critical' else 5
                    priorities.append({
                        'vulnerability': vuln,
                        'priority': 'high' if severity == 'critical' else 'medium',
                        'reason': f'{severity.title()} severity vulnerability',
                        'urgency_score': urgency
                    })
                elif severity == 'medium':
                    priorities.append({
                        'vulnerability': vuln,
                        'priority': 'medium',
                        'reason': 'Medium severity vulnerability',
                        'urgency_score': 3
                    })
                else:
                    priorities.append({
                        'vulnerability': vuln,
                        'priority': 'low',
                        'reason': 'Low severity vulnerability',
                        'urgency_score': 1
                    })
        
        # Sort by urgency score
        priorities.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        return priorities

    def _analyze_authentication_authorization(self, source_data: Dict[str, str]) -> Dict[str, Any]:
        """Analyze authentication and authorization mechanisms"""
        analysis = {
            'authentication_mechanisms': [],
            'authorization_checks': [],
            'session_management': [],
            'access_control': [],
            'auth_vulnerabilities': [],
            'auth_strength': 'weak',
            'auth_coverage': 0.0
        }
        
        # Authentication patterns
        auth_patterns = [
            'password', 'authenticate', 'login', 'credential',
            'token', 'session', 'cookie', 'jwt', 'oauth',
            'verify', 'validation', 'check_auth'
        ]
        
        # Authorization patterns  
        authz_patterns = [
            'authorize', 'permission', 'role', 'access',
            'privilege', 'check_access', 'allow', 'deny',
            'grant', 'revoke', 'admin', 'user_level'
        ]
        
        total_functions = 0
        auth_protected_functions = 0
        
        for filename, content in source_data.items():
            # Count total functions
            function_count = len(re.findall(r'^\w+\s+\w+\s*\([^)]*\)\s*{', content, re.MULTILINE))
            total_functions += function_count
            
            # Find authentication mechanisms
            for pattern in auth_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    analysis['authentication_mechanisms'].append({
                        'type': pattern,
                        'file': filename,
                        'line': line_num,
                        'context': self._extract_code_context(content, match.start(), 1)
                    })
            
            # Find authorization checks
            for pattern in authz_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    analysis['authorization_checks'].append({
                        'type': pattern,
                        'file': filename,
                        'line': line_num,
                        'context': self._extract_code_context(content, match.start(), 1)
                    })
                    
                    # Count as protected function if authorization check found
                    auth_protected_functions += 1
        
        # Calculate authentication coverage
        if total_functions > 0:
            analysis['auth_coverage'] = min(1.0, auth_protected_functions / total_functions)
        
        # Determine authentication strength
        auth_count = len(analysis['authentication_mechanisms'])
        authz_count = len(analysis['authorization_checks'])
        
        if auth_count >= 5 and authz_count >= 3 and analysis['auth_coverage'] > 0.7:
            analysis['auth_strength'] = 'strong'
        elif auth_count >= 3 and authz_count >= 1 and analysis['auth_coverage'] > 0.3:
            analysis['auth_strength'] = 'medium'
        else:
            analysis['auth_strength'] = 'weak'
        
        return analysis

    def _analyze_cryptographic_usage(self, source_data: Dict[str, str]) -> Dict[str, Any]:
        """Analyze cryptographic usage and implementation"""
        analysis = {
            'encryption_usage': [],
            'hashing_usage': [],
            'random_number_generation': [],
            'crypto_vulnerabilities': [],
            'crypto_strength': 'weak',
            'crypto_coverage': 0.0
        }
        
        # Cryptographic patterns
        crypto_patterns = {
            'encryption': ['encrypt', 'decrypt', 'aes', 'des', 'rsa', 'cipher', 'crypt'],
            'hashing': ['hash', 'sha', 'md5', 'hmac', 'digest', 'checksum'],
            'random': ['rand', 'random', 'entropy', 'nonce', 'uuid', 'guid']
        }
        
        # Weak cryptographic algorithms
        weak_algorithms = ['md5', 'sha1', 'des', 'rc4', 'md4']
        
        total_crypto_usage = 0
        
        for filename, content in source_data.items():
            for category, patterns in crypto_patterns.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        total_crypto_usage += 1
                        
                        usage_info = {
                            'algorithm': pattern,
                            'file': filename,
                            'line': line_num,
                            'context': self._extract_code_context(content, match.start(), 1)
                        }
                        
                        if category == 'encryption':
                            analysis['encryption_usage'].append(usage_info)
                        elif category == 'hashing':
                            analysis['hashing_usage'].append(usage_info)
                        elif category == 'random':
                            analysis['random_number_generation'].append(usage_info)
                        
                        # Check for weak algorithms
                        if pattern.lower() in weak_algorithms:
                            analysis['crypto_vulnerabilities'].append({
                                'type': 'weak_cryptography',
                                'algorithm': pattern,
                                'file': filename,
                                'line': line_num,
                                'severity': 'high' if pattern.lower() in ['md5', 'des'] else 'medium',
                                'recommendation': f'Replace {pattern} with stronger algorithm',
                                'context': usage_info['context']
                            })
        
        # Calculate crypto coverage and strength
        strong_crypto_count = total_crypto_usage - len(analysis['crypto_vulnerabilities'])
        
        if total_crypto_usage > 0:
            analysis['crypto_coverage'] = strong_crypto_count / total_crypto_usage
        
        # Determine crypto strength
        if (len(analysis['encryption_usage']) >= 2 and 
            len(analysis['hashing_usage']) >= 1 and 
            len(analysis['crypto_vulnerabilities']) == 0):
            analysis['crypto_strength'] = 'strong'
        elif total_crypto_usage >= 3 and len(analysis['crypto_vulnerabilities']) <= 1:
            analysis['crypto_strength'] = 'medium'
        
        return analysis

    def _assess_input_validation(self, source_data: Dict[str, str]) -> Dict[str, Any]:
        """Assess input validation mechanisms"""
        assessment = {
            'validation_functions': [],
            'sanitization_functions': [],
            'input_sources': [],
            'validation_coverage': 0.0,
            'validation_strength': 'weak',
            'unvalidated_inputs': []
        }
        
        # Validation patterns
        validation_patterns = [
            'validate', 'sanitize', 'filter', 'escape', 'clean',
            'check', 'verify', 'assert', 'isdigit', 'isalpha',
            'isalnum', 'strlen', 'range_check', 'bounds_check'
        ]
        
        # Input source patterns
        input_patterns = [
            'scanf', 'gets', 'fgets', 'read', 'recv', 'recvfrom',
            'argv', 'getenv', 'input', 'request', 'param',
            'ReadFile', 'GetWindowText', 'GetDlgItemText'
        ]
        
        total_inputs = 0
        validated_inputs = 0
        
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
            
            # Find input sources and check for validation
            for pattern in input_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    line_num = content[:match.start()].count('\n') + 1
                    total_inputs += 1
                    
                    input_info = {
                        'source': pattern,
                        'file': filename,
                        'line': line_num
                    }
                    assessment['input_sources'].append(input_info)
                    
                    # Check for validation near input
                    context = content[max(0, match.start()-200):match.end()+200]
                    has_validation = any(val_pattern in context.lower() 
                                       for val_pattern in validation_patterns)
                    
                    if has_validation:
                        validated_inputs += 1
                    else:
                        assessment['unvalidated_inputs'].append({
                            'input': pattern,
                            'file': filename,
                            'line': line_num,
                            'recommendation': f'Add validation for {pattern} input'
                        })
        
        # Calculate validation coverage
        if total_inputs > 0:
            assessment['validation_coverage'] = validated_inputs / total_inputs
        
        # Determine validation strength
        if assessment['validation_coverage'] >= 0.8:
            assessment['validation_strength'] = 'strong'
        elif assessment['validation_coverage'] >= 0.5:
            assessment['validation_strength'] = 'medium'
        
        return assessment

    def _generate_security_recommendations(self, vulnerability_analysis: Dict[str, Any],
                                         security_patterns: Dict[str, Any],
                                         exploit_detection: Dict[str, Any],
                                         auth_analysis: Dict[str, Any],
                                         crypto_analysis: Dict[str, Any],
                                         input_validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive security recommendations"""
        recommendations = []
        
        vuln_stats = vulnerability_analysis.get('vulnerability_statistics', {})
        
        # Critical vulnerability recommendations
        if vuln_stats.get('critical_vulnerabilities', 0) > 0:
            recommendations.append({
                'priority': 'critical',
                'category': 'vulnerability_mitigation',
                'title': 'Address Critical Security Vulnerabilities',
                'description': f'Found {vuln_stats["critical_vulnerabilities"]} critical vulnerabilities',
                'actions': [
                    'Immediately fix all critical vulnerabilities',
                    'Conduct emergency security review',
                    'Consider taking system offline until fixes are deployed'
                ],
                'timeframe': 'Immediate (0-24 hours)'
            })
        
        # High-severity vulnerability recommendations
        if vuln_stats.get('high_vulnerabilities', 0) > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'vulnerability_mitigation', 
                'title': 'Fix High-Severity Vulnerabilities',
                'description': f'Found {vuln_stats["high_vulnerabilities"]} high-severity vulnerabilities',
                'actions': [
                    'Prioritize fixes for buffer overflows and injection vulnerabilities',
                    'Implement input validation and bounds checking',
                    'Use secure coding practices and safe functions',
                    'Add runtime protection mechanisms'
                ],
                'timeframe': 'Urgent (1-7 days)'
            })
        
        # Exploit chain recommendations
        exploit_chains = exploit_detection.get('exploit_chains', [])
        if exploit_chains:
            recommendations.append({
                'priority': 'critical',
                'category': 'exploit_prevention',
                'title': 'Break Exploit Chains',
                'description': f'Found {len(exploit_chains)} potential exploit chains',
                'actions': [
                    'Fix vulnerabilities that can be chained together',
                    'Implement defense-in-depth security measures',
                    'Add additional validation layers',
                    'Consider architectural changes to isolate components'
                ],
                'timeframe': 'Immediate (0-72 hours)'
            })
        
        # Authentication recommendations
        auth_strength = auth_analysis.get('auth_strength', 'weak')
        if auth_strength == 'weak':
            recommendations.append({
                'priority': 'high',
                'category': 'authentication',
                'title': 'Strengthen Authentication Mechanisms',
                'description': 'Weak authentication implementation detected',
                'actions': [
                    'Implement strong password policies',
                    'Add multi-factor authentication',
                    'Use secure session management',
                    'Implement proper access controls'
                ],
                'timeframe': '1-2 weeks'
            })
        
        # Cryptography recommendations
        crypto_vulnerabilities = crypto_analysis.get('crypto_vulnerabilities', [])
        if crypto_vulnerabilities:
            recommendations.append({
                'priority': 'high',
                'category': 'cryptography',
                'title': 'Update Cryptographic Implementations',
                'description': f'Found {len(crypto_vulnerabilities)} weak cryptographic usages',
                'actions': [
                    'Replace MD5 and SHA1 with SHA-256 or higher',
                    'Replace DES with AES encryption',
                    'Use secure random number generators',
                    'Implement proper key management'
                ],
                'timeframe': '1-2 weeks'
            })
        
        # Input validation recommendations
        validation_coverage = input_validation.get('validation_coverage', 0.0)
        if validation_coverage < 0.6:
            recommendations.append({
                'priority': 'medium',
                'category': 'input_validation',
                'title': 'Improve Input Validation',
                'description': f'Input validation coverage is only {validation_coverage:.1%}',
                'actions': [
                    'Validate all user inputs at entry points',
                    'Implement whitelist-based validation',
                    'Add length and format checks',
                    'Sanitize outputs to prevent XSS'
                ],
                'timeframe': '2-3 weeks'
            })
        
        # Security pattern recommendations
        pattern_ratio = security_patterns.get('pattern_analysis', {}).get('security_pattern_ratio', 0.0)
        if pattern_ratio < 0.5:
            recommendations.append({
                'priority': 'medium',
                'category': 'secure_coding',
                'title': 'Adopt Secure Coding Practices',
                'description': f'Security pattern ratio is low: {pattern_ratio:.1%}',
                'actions': [
                    'Use safe string functions (strncpy, snprintf)',
                    'Implement proper error handling',
                    'Add bounds checking for array access',
                    'Use static analysis tools in development'
                ],
                'timeframe': '2-4 weeks'
            })
        
        # General security improvements
        if vuln_stats.get('total_vulnerabilities', 0) > 10:
            recommendations.append({
                'priority': 'medium',
                'category': 'security_program',
                'title': 'Implement Security Development Lifecycle',
                'description': f'High vulnerability count ({vuln_stats["total_vulnerabilities"]}) indicates systemic issues',
                'actions': [
                    'Implement secure code review process',
                    'Add automated security testing to CI/CD',
                    'Provide security training for developers',
                    'Establish security coding standards'
                ],
                'timeframe': '1-3 months'
            })
        
        return recommendations

    def _create_security_report(self, vulnerability_analysis: Dict[str, Any],
                              security_patterns: Dict[str, Any],
                              exploit_detection: Dict[str, Any],
                              auth_analysis: Dict[str, Any],
                              crypto_analysis: Dict[str, Any],
                              input_validation: Dict[str, Any],
                              recommendations: List[Dict[str, Any]]) -> SecurityAnalysisResult:
        """Create comprehensive security analysis result"""
        result = SecurityAnalysisResult()
        
        try:
            # Extract vulnerability statistics
            vuln_stats = vulnerability_analysis.get('vulnerability_statistics', {})
            result.total_vulnerabilities = vuln_stats.get('total_vulnerabilities', 0)
            result.critical_vulnerabilities = vuln_stats.get('critical_vulnerabilities', 0)
            result.high_vulnerabilities = vuln_stats.get('high_vulnerabilities', 0)
            result.medium_vulnerabilities = vuln_stats.get('medium_vulnerabilities', 0)
            result.low_vulnerabilities = vuln_stats.get('low_vulnerabilities', 0)
            
            # Set security scores
            result.security_score = vuln_stats.get('security_score', 0.0)
            result.exploit_risk_score = exploit_detection.get('exploit_risk_score', 0.0)
            
            # Set analysis data
            result.vulnerabilities = vulnerability_analysis.get('vulnerabilities', [])
            result.security_patterns = security_patterns
            result.exploit_chains = exploit_detection.get('exploit_chains', [])
            result.recommendations = recommendations
            
            # Calculate overall success
            result.success = (
                result.critical_vulnerabilities == 0 and
                result.high_vulnerabilities <= self.security_thresholds['maximum_high_vulnerabilities'] and
                result.security_score >= self.security_thresholds['minimum_security_score'] and
                result.exploit_risk_score <= self.security_thresholds['maximum_exploit_risk']
            )
            
            # Generate warnings for concerning findings
            if result.critical_vulnerabilities > 0:
                result.warnings.append(f"Found {result.critical_vulnerabilities} critical vulnerabilities requiring immediate attention")
            
            if result.exploit_risk_score > 0.7:
                result.warnings.append(f"High exploit risk detected (score: {result.exploit_risk_score:.2f})")
            
            if len(result.exploit_chains) > 0:
                result.warnings.append(f"Found {len(result.exploit_chains)} potential exploit chains")
            
            # Calculate metrics
            result.metrics = {
                'analysis_coverage': {
                    'vulnerability_types_checked': 9,  # Number of vulnerability types we check
                    'security_patterns_analyzed': len(security_patterns.get('pattern_analysis', {})),
                    'authentication_coverage': auth_analysis.get('auth_coverage', 0.0),
                    'crypto_coverage': crypto_analysis.get('crypto_coverage', 0.0),
                    'input_validation_coverage': input_validation.get('validation_coverage', 0.0)
                },
                'risk_assessment': {
                    'overall_risk': vuln_stats.get('risk_assessment', 'unknown'),
                    'exploit_potential': 'high' if result.exploit_risk_score > 0.7 else 'medium' if result.exploit_risk_score > 0.3 else 'low',
                    'attack_surface': len(result.vulnerabilities),
                    'critical_findings': result.critical_vulnerabilities + result.high_vulnerabilities
                },
                'security_posture': {
                    'auth_strength': auth_analysis.get('auth_strength', 'weak'),
                    'crypto_strength': crypto_analysis.get('crypto_strength', 'weak'),
                    'input_validation_strength': input_validation.get('validation_strength', 'weak'),
                    'secure_coding_ratio': security_patterns.get('pattern_analysis', {}).get('security_pattern_ratio', 0.0)
                }
            }
            
        except Exception as e:
            result.error_messages.append(f"Error creating security report: {str(e)}")
            self.logger.error(f"Security report creation failed: {e}")
        
        return result

    def _initialize_vulnerability_detectors(self) -> Dict[str, Any]:
        """Initialize vulnerability detection components"""
        return {
            'buffer_overflow_detector': {
                'enabled': True,
                'dangerous_functions': ['strcpy', 'strcat', 'sprintf', 'gets', 'scanf'],
                'confidence_threshold': 0.7
            },
            'injection_detector': {
                'enabled': True,
                'sql_patterns': ['SELECT.*+', 'INSERT.*+', 'UPDATE.*+', 'DELETE.*+'],
                'command_patterns': ['system(', 'exec(', 'popen(']
            },
            'memory_detector': {
                'enabled': True,
                'track_allocations': True,
                'track_deallocations': True
            }
        }

    def _load_security_patterns(self) -> Dict[str, Any]:
        """Load security pattern definitions"""
        return {
            'secure_functions': [
                'strncpy', 'strncat', 'snprintf', 'fgets',
                'secure_memcpy', 'memset_s'
            ],
            'insecure_functions': [
                'strcpy', 'strcat', 'sprintf', 'gets',
                'system', 'eval', 'exec'
            ],
            'crypto_algorithms': {
                'strong': ['aes', 'sha256', 'sha512', 'rsa2048'],
                'weak': ['md5', 'sha1', 'des', 'rc4']
            }
        }

    def _initialize_exploit_analyzers(self) -> Dict[str, Any]:
        """Initialize exploit analysis components"""
        return {
            'chain_detector': {
                'enabled': True,
                'max_chain_depth': 3,
                'chain_patterns': [
                    'info_disclosure -> buffer_overflow',
                    'privilege_escalation -> command_injection',
                    'race_condition -> privilege_escalation'
                ]
            },
            'impact_assessor': {
                'enabled': True,
                'severity_weights': {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            }
        }

    def _create_failure_result(self, error_message: str, execution_time: float = 0.0) -> Dict[str, Any]:
        """Create failure result"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.name,
            'status': AgentStatus.FAILED if HAS_MATRIX_FRAMEWORK else 'failed',
            'execution_time': execution_time,
            'error_message': error_message,
            'security_analysis_result': SecurityAnalysisResult(),
            'metadata': {
                'character': self.character,
                'phase': self.current_phase,
                'failure_point': self.current_phase
            }
        }


# For backward compatibility
AgentJohnsonAgent = Agent13_AgentJohnson