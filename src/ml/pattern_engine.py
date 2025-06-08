"""
Machine Learning Pattern Recognition Engine
Analyzes assembly patterns and maps them to high-level C code constructs.
"""

import re
import json
import hashlib
import time
import threading
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter, OrderedDict
import math


class LRUCache:
    """Thread-safe LRU cache with size limit and TTL support"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.access_times = {}
        self.lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.access_times[oldest_key]
                
                self.cache[key] = value
            
            self.access_times[key] = time.time()
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds
            }


class PerformanceMonitor:
    """Monitor and track pattern engine performance"""
    
    def __init__(self):
        self.stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_time_per_analysis': 0.0,
            'pattern_type_stats': defaultdict(int),
            'confidence_distribution': defaultdict(int)
        }
        self.lock = threading.Lock()
    
    def record_analysis(self, analysis_time: float, cache_hit: bool, patterns_found: List[str], confidence: float):
        with self.lock:
            self.stats['total_analyses'] += 1
            self.stats['total_time'] += analysis_time
            self.stats['avg_time_per_analysis'] = self.stats['total_time'] / self.stats['total_analyses']
            
            if cache_hit:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
            
            for pattern in patterns_found:
                self.stats['pattern_type_stats'][pattern] += 1
            
            # Bucket confidence scores
            confidence_bucket = f"{int(confidence * 10) * 10}-{int(confidence * 10) * 10 + 10}%"
            self.stats['confidence_distribution'][confidence_bucket] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            cache_hit_rate = 0.0
            if self.stats['total_analyses'] > 0:
                cache_hit_rate = self.stats['cache_hits'] / self.stats['total_analyses']
            
            return {
                **self.stats,
                'cache_hit_rate': cache_hit_rate
            }


class PatternEngine:
    """Enhanced machine learning-based pattern recognition for assembly-to-C mapping"""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        self.patterns = {
            'control_flow': ControlFlowPatterns(),
            'data_access': DataAccessPatterns(),
            'function_prologue': FunctionProloguePatterns(),
            'arithmetic': ArithmeticPatterns(),
            'string_operations': StringOperationPatterns(),
            'system_calls': SystemCallPatterns()
        }
        
        self.confidence_threshold = 0.6
        self.pattern_cache = LRUCache(cache_size, cache_ttl)
        self.performance_monitor = PerformanceMonitor()
        
        # Pre-compiled regex patterns for better performance
        self._compile_regex_patterns()
        
        # Batch processing support
        self.batch_size = 100
        self.enable_parallel_processing = True
        
    def _compile_regex_patterns(self):
        """Pre-compile frequently used regex patterns for better performance"""
        self.compiled_patterns = {
            'instruction': re.compile(r'^([a-zA-Z][a-zA-Z0-9]*)\s+(.*)$'),
            'register': re.compile(r'\b[er]?[abcd]x|[er]?[sd]i|[er]?[sb]p|r[0-9]+[dwb]?\b'),
            'immediate': re.compile(r'#?0x[0-9a-fA-F]+|\b\d+\b'),
            'memory_ref': re.compile(r'\[([^\]]+)\]'),
            'jump_target': re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*:|\bloc_[0-9a-fA-F]+\b'),
            'function_call': re.compile(r'\bcall\s+([a-zA-Z_][a-zA-Z0-9_]*|\[[^\]]+\])')
        }
    
    def analyze_code_block(self, assembly_code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze assembly code block and identify patterns with enhanced caching"""
        start_time = time.time()
        
        if not assembly_code:
            return {'patterns': [], 'confidence': 0.0, 'suggestions': []}
        
        # Create cache key including context for better cache efficiency
        context_hash = hashlib.md5(str(sorted((context or {}).items())).encode()).hexdigest()[:8]
        cache_key = hashlib.md5(f"{assembly_code}:{context_hash}".encode()).hexdigest()
        
        # Check cache first
        cached_result = self.pattern_cache.get(cache_key)
        if cached_result is not None:
            analysis_time = time.time() - start_time
            pattern_names = [p['type'] for p in cached_result.get('patterns', [])]
            self.performance_monitor.record_analysis(
                analysis_time, True, pattern_names, cached_result.get('confidence', 0.0)
            )
            return cached_result
        
        result = {
            'patterns': [],
            'confidence': 0.0,
            'suggestions': [],
            'code_quality_score': 0.0,
            'complexity_metrics': {}
        }
        
        # Normalize assembly code
        normalized_code = self._normalize_assembly(assembly_code)
        
        # Run pattern analysis
        all_patterns = []
        total_confidence = 0.0
        
        for pattern_type, pattern_analyzer in self.patterns.items():
            patterns = pattern_analyzer.detect(normalized_code, context or {})
            for pattern in patterns:
                pattern['type'] = pattern_type
                all_patterns.append(pattern)
                total_confidence += pattern.get('confidence', 0.5)
        
        # Sort patterns by confidence
        all_patterns.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Calculate overall confidence
        if all_patterns:
            result['confidence'] = total_confidence / len(all_patterns)
        
        # Filter patterns by confidence threshold
        result['patterns'] = [p for p in all_patterns if p.get('confidence', 0.0) >= self.confidence_threshold]
        
        # Generate code suggestions
        result['suggestions'] = self._generate_suggestions(result['patterns'], normalized_code)
        
        # Calculate code quality metrics
        result['code_quality_score'] = self._calculate_quality_score(result['patterns'], normalized_code)
        result['complexity_metrics'] = self._calculate_complexity(normalized_code)
        
        # Cache result
        self.pattern_cache.put(cache_key, result)
        
        # Record performance metrics
        analysis_time = time.time() - start_time
        pattern_names = [p['type'] for p in result.get('patterns', [])]
        self.performance_monitor.record_analysis(
            analysis_time, False, pattern_names, result.get('confidence', 0.0)
        )
        
        return result
    
    def analyze_code_blocks_batch(self, code_blocks: List[str], context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Batch analyze multiple code blocks for better performance"""
        if not code_blocks:
            return []
        
        results = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(code_blocks), self.batch_size):
            batch = code_blocks[i:i + self.batch_size]
            
            if self.enable_parallel_processing and len(batch) > 1:
                # Parallel processing for larger batches
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(self.analyze_code_block, code, context) for code in batch]
                    batch_results = [future.result() for future in concurrent.futures.as_completed(futures)]
                results.extend(batch_results)
            else:
                # Sequential processing for smaller batches
                batch_results = [self.analyze_code_block(code, context) for code in batch]
                results.extend(batch_results)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics and cache information"""
        return {
            'performance_stats': self.performance_monitor.get_stats(),
            'cache_stats': self.pattern_cache.stats(),
            'pattern_engine_config': {
                'confidence_threshold': self.confidence_threshold,
                'batch_size': self.batch_size,
                'parallel_processing': self.enable_parallel_processing
            }
        }
    
    def clear_cache(self):
        """Clear pattern cache and reset performance stats"""
        self.pattern_cache.clear()
        self.performance_monitor = PerformanceMonitor()
    
    def _normalize_assembly(self, assembly_code: str) -> List[str]:
        """Normalize assembly code for pattern matching"""
        lines = []
        for line in assembly_code.split('\n'):
            line = line.strip()
            if line and not line.startswith(';') and not line.startswith('//'):
                # Remove comments
                comment_pos = line.find(';')
                if comment_pos != -1:
                    line = line[:comment_pos].strip()
                
                # Normalize whitespace
                line = re.sub(r'\s+', ' ', line)
                lines.append(line.lower())
        
        return lines
    
    def _generate_suggestions(self, patterns: List[Dict[str, Any]], normalized_code: List[str]) -> List[Dict[str, Any]]:
        """Generate C code suggestions based on detected patterns"""
        suggestions = []
        
        for pattern in patterns[:5]:  # Top 5 patterns
            if pattern['type'] == 'control_flow':
                suggestions.extend(self._suggest_control_flow(pattern))
            elif pattern['type'] == 'data_access':
                suggestions.extend(self._suggest_data_access(pattern))
            elif pattern['type'] == 'function_prologue':
                suggestions.extend(self._suggest_function_structure(pattern))
            elif pattern['type'] == 'arithmetic':
                suggestions.extend(self._suggest_arithmetic(pattern))
            elif pattern['type'] == 'string_operations':
                suggestions.extend(self._suggest_string_ops(pattern))
            elif pattern['type'] == 'system_calls':
                suggestions.extend(self._suggest_system_calls(pattern))
        
        # Remove duplicates and sort by confidence
        unique_suggestions = {}
        for suggestion in suggestions:
            key = suggestion['suggestion']
            if key not in unique_suggestions or suggestion['confidence'] > unique_suggestions[key]['confidence']:
                unique_suggestions[key] = suggestion
        
        return sorted(unique_suggestions.values(), key=lambda x: x['confidence'], reverse=True)
    
    def _suggest_control_flow(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate control flow suggestions"""
        suggestions = []
        subtype = pattern.get('subtype', '')
        
        if subtype == 'if_statement':
            suggestions.append({
                'suggestion': f"if ({pattern.get('condition', 'condition')}) {{\n    // code block\n}}",
                'confidence': pattern.get('confidence', 0.7),
                'description': 'Detected conditional branch pattern'
            })
        elif subtype == 'loop':
            loop_type = pattern.get('loop_type', 'while')
            if loop_type == 'for':
                suggestions.append({
                    'suggestion': "for (int i = 0; i < n; i++) {\n    // loop body\n}",
                    'confidence': pattern.get('confidence', 0.7),
                    'description': 'Detected for-loop pattern'
                })
            else:
                suggestions.append({
                    'suggestion': "while (condition) {\n    // loop body\n}",
                    'confidence': pattern.get('confidence', 0.7),
                    'description': 'Detected while-loop pattern'
                })
        
        return suggestions
    
    def _suggest_data_access(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate data access suggestions"""
        suggestions = []
        subtype = pattern.get('subtype', '')
        
        if subtype == 'array_access':
            suggestions.append({
                'suggestion': f"array[{pattern.get('index', 'i')}]",
                'confidence': pattern.get('confidence', 0.7),
                'description': 'Detected array access pattern'
            })
        elif subtype == 'pointer_dereference':
            suggestions.append({
                'suggestion': f"*{pattern.get('pointer', 'ptr')}",
                'confidence': pattern.get('confidence', 0.7),
                'description': 'Detected pointer dereference'
            })
        elif subtype == 'struct_access':
            suggestions.append({
                'suggestion': f"struct_var.{pattern.get('field', 'field')}",
                'confidence': pattern.get('confidence', 0.7),
                'description': 'Detected structure field access'
            })
        
        return suggestions
    
    def _suggest_function_structure(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate function structure suggestions"""
        suggestions = []
        
        if pattern.get('subtype') == 'function_entry':
            param_count = pattern.get('parameter_count', 0)
            if param_count == 0:
                suggestions.append({
                    'suggestion': "int function_name(void) {\n    // function body\n    return 0;\n}",
                    'confidence': pattern.get('confidence', 0.8),
                    'description': 'Function with no parameters'
                })
            else:
                params = ', '.join([f"int param{i}" for i in range(param_count)])
                suggestions.append({
                    'suggestion': f"int function_name({params}) {{\n    // function body\n    return 0;\n}}",
                    'confidence': pattern.get('confidence', 0.8),
                    'description': f'Function with {param_count} parameters'
                })
        
        return suggestions
    
    def _suggest_arithmetic(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate arithmetic operation suggestions"""
        suggestions = []
        operation = pattern.get('operation', '')
        
        if operation == 'addition':
            suggestions.append({
                'suggestion': "a + b",
                'confidence': pattern.get('confidence', 0.9),
                'description': 'Addition operation'
            })
        elif operation == 'multiplication':
            suggestions.append({
                'suggestion': "a * b",
                'confidence': pattern.get('confidence', 0.9),
                'description': 'Multiplication operation'
            })
        elif operation == 'shift':
            direction = pattern.get('direction', 'left')
            if direction == 'left':
                suggestions.append({
                    'suggestion': "value << shift_amount",
                    'confidence': pattern.get('confidence', 0.9),
                    'description': 'Left bit shift'
                })
            else:
                suggestions.append({
                    'suggestion': "value >> shift_amount",
                    'confidence': pattern.get('confidence', 0.9),
                    'description': 'Right bit shift'
                })
        
        return suggestions
    
    def _suggest_string_ops(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate string operation suggestions"""
        suggestions = []
        operation = pattern.get('operation', '')
        
        if operation == 'copy':
            suggestions.append({
                'suggestion': "strcpy(dest, src)",
                'confidence': pattern.get('confidence', 0.8),
                'description': 'String copy operation'
            })
        elif operation == 'compare':
            suggestions.append({
                'suggestion': "strcmp(str1, str2)",
                'confidence': pattern.get('confidence', 0.8),
                'description': 'String comparison'
            })
        elif operation == 'length':
            suggestions.append({
                'suggestion': "strlen(string)",
                'confidence': pattern.get('confidence', 0.8),
                'description': 'String length calculation'
            })
        
        return suggestions
    
    def _suggest_system_calls(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system call suggestions"""
        suggestions = []
        call_type = pattern.get('call_type', '')
        
        if call_type == 'file_io':
            suggestions.append({
                'suggestion': "fopen(filename, mode)",
                'confidence': pattern.get('confidence', 0.7),
                'description': 'File I/O operation'
            })
        elif call_type == 'memory':
            suggestions.append({
                'suggestion': "malloc(size)",
                'confidence': pattern.get('confidence', 0.7),
                'description': 'Memory allocation'
            })
        
        return suggestions
    
    def _calculate_quality_score(self, patterns: List[Dict[str, Any]], normalized_code: List[str]) -> float:
        """Calculate code quality score based on patterns"""
        if not patterns:
            return 0.3  # Low quality if no patterns detected
        
        # Base score from pattern confidence
        confidence_score = sum(p.get('confidence', 0.5) for p in patterns) / len(patterns)
        
        # Adjust for code complexity
        complexity_penalty = min(0.2, len(normalized_code) / 100)  # Penalty for very long functions
        
        # Adjust for pattern diversity
        pattern_types = set(p.get('type', '') for p in patterns)
        diversity_bonus = min(0.2, len(pattern_types) * 0.05)
        
        return min(1.0, confidence_score - complexity_penalty + diversity_bonus)
    
    def _calculate_complexity(self, normalized_code: List[str]) -> Dict[str, Any]:
        """Calculate complexity metrics"""
        return {
            'lines_of_code': len(normalized_code),
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(normalized_code),
            'branching_factor': self._calculate_branching_factor(normalized_code),
            'depth_level': self._estimate_nesting_depth(normalized_code)
        }
    
    def _calculate_cyclomatic_complexity(self, normalized_code: List[str]) -> int:
        """Calculate cyclomatic complexity"""
        branch_keywords = ['jmp', 'je', 'jne', 'jz', 'jnz', 'jl', 'jg', 'jle', 'jge', 'call']
        branches = sum(1 for line in normalized_code if any(keyword in line for keyword in branch_keywords))
        return max(1, branches + 1)  # +1 for the linear path
    
    def _calculate_branching_factor(self, normalized_code: List[str]) -> float:
        """Calculate average branching factor"""
        total_lines = len(normalized_code)
        if total_lines == 0:
            return 0.0
        
        branch_lines = sum(1 for line in normalized_code if any(keyword in line for keyword in ['jmp', 'j', 'call']))
        return branch_lines / total_lines
    
    def _estimate_nesting_depth(self, normalized_code: List[str]) -> int:
        """Estimate nesting depth based on labels and jumps"""
        # Simplified estimation - would need more sophisticated analysis
        labels = sum(1 for line in normalized_code if ':' in line)
        jumps = sum(1 for line in normalized_code if 'jmp' in line or line.startswith('j'))
        
        # Rough estimation of nesting
        if jumps > labels * 2:
            return 3  # High nesting
        elif jumps > labels:
            return 2  # Medium nesting
        else:
            return 1  # Low nesting


class ControlFlowPatterns:
    """Detect control flow patterns in assembly code"""
    
    def detect(self, normalized_code: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        patterns = []
        
        # Detect conditional branches
        patterns.extend(self._detect_conditionals(normalized_code))
        
        # Detect loops
        patterns.extend(self._detect_loops(normalized_code))
        
        return patterns
    
    def _detect_conditionals(self, normalized_code: List[str]) -> List[Dict[str, Any]]:
        """Detect if-statements and conditional branches"""
        patterns = []
        
        for i, line in enumerate(normalized_code):
            if any(cond in line for cond in ['je', 'jne', 'jz', 'jnz', 'jl', 'jg']):
                confidence = 0.8 if 'cmp' in ' '.join(code[max(0, i-2):i]) else 0.6
                patterns.append({
                    'subtype': 'if_statement',
                    'confidence': confidence,
                    'line': i,
                    'condition': 'detected_condition',
                    'details': f'Conditional jump: {line}'
                })
        
        return patterns
    
    def _detect_loops(self, normalized_code: List[str]) -> List[Dict[str, Any]]:
        """Detect loop patterns"""
        patterns = []
        
        # Look for backward jumps (typical loop pattern)
        labels = {}
        for i, line in enumerate(normalized_code):
            if ':' in line:
                label = line.split(':')[0].strip()
                labels[label] = i
        
        for i, line in enumerate(normalized_code):
            if 'jmp' in line or any(cond in line for cond in ['je', 'jne', 'jl', 'jg']):
                # Extract target label
                parts = line.split()
                if len(parts) >= 2:
                    target = parts[-1]
                    if target in labels and labels[target] < i:
                        # Backward jump - likely a loop
                        loop_size = i - labels[target]
                        loop_type = 'for' if loop_size < 10 else 'while'
                        patterns.append({
                            'subtype': 'loop',
                            'loop_type': loop_type,
                            'confidence': 0.7,
                            'line': i,
                            'details': f'Loop from line {labels[target]} to {i}'
                        })
        
        return patterns


class DataAccessPatterns:
    """Detect data access patterns"""
    
    def detect(self, normalized_code: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        patterns = []
        
        for i, line in enumerate(normalized_code):
            # Array access patterns
            if '[' in line and ']' in line:
                patterns.append({
                    'subtype': 'array_access',
                    'confidence': 0.8,
                    'line': i,
                    'details': f'Array access: {line}'
                })
            
            # Pointer dereference
            if 'mov' in line and any(reg in line for reg in ['eax', 'ebx', 'ecx', 'edx']):
                if '[' in line:
                    patterns.append({
                        'subtype': 'pointer_dereference',
                        'confidence': 0.7,
                        'line': i,
                        'details': f'Pointer access: {line}'
                    })
        
        return patterns


class FunctionProloguePatterns:
    """Detect function structure patterns"""
    
    def detect(self, normalized_code: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        patterns = []
        
        # Look for function prologue
        for i, line in enumerate(normalized_code[:5]):  # Check first 5 lines
            if 'push ebp' in line or 'push rbp' in line:
                # Count parameters by looking at stack operations
                param_count = self._count_parameters(normalized_code)
                patterns.append({
                    'subtype': 'function_entry',
                    'confidence': 0.9,
                    'line': i,
                    'parameter_count': param_count,
                    'details': 'Function prologue detected'
                })
                break
        
        return patterns
    
    def _count_parameters(self, code: List[str]) -> int:
        """Estimate parameter count from stack operations"""
        # Simplified parameter counting
        param_accesses = 0
        for line in normalized_code[:20]:  # Check first 20 lines
            if 'ebp+' in line or 'rbp+' in line:
                param_accesses += 1
        
        return min(param_accesses, 8)  # Cap at reasonable number


class ArithmeticPatterns:
    """Detect arithmetic operation patterns"""
    
    def detect(self, normalized_code: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        patterns = []
        
        for i, line in enumerate(normalized_code):
            if 'add' in line:
                patterns.append({
                    'operation': 'addition',
                    'confidence': 0.9,
                    'line': i,
                    'details': f'Addition: {line}'
                })
            elif 'mul' in line or 'imul' in line:
                patterns.append({
                    'operation': 'multiplication',
                    'confidence': 0.9,
                    'line': i,
                    'details': f'Multiplication: {line}'
                })
            elif 'shl' in line:
                patterns.append({
                    'operation': 'shift',
                    'direction': 'left',
                    'confidence': 0.9,
                    'line': i,
                    'details': f'Left shift: {line}'
                })
            elif 'shr' in line:
                patterns.append({
                    'operation': 'shift',
                    'direction': 'right',
                    'confidence': 0.9,
                    'line': i,
                    'details': f'Right shift: {line}'
                })
        
        return patterns


class StringOperationPatterns:
    """Detect string operation patterns"""
    
    def detect(self, normalized_code: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        patterns = []
        
        for i, line in enumerate(normalized_code):
            if 'rep movsb' in line or 'rep movsw' in line:
                patterns.append({
                    'operation': 'copy',
                    'confidence': 0.8,
                    'line': i,
                    'details': 'String copy operation'
                })
            elif 'rep cmpsb' in line:
                patterns.append({
                    'operation': 'compare',
                    'confidence': 0.8,
                    'line': i,
                    'details': 'String comparison'
                })
            elif 'rep scasb' in line:
                patterns.append({
                    'operation': 'length',
                    'confidence': 0.7,
                    'line': i,
                    'details': 'String length calculation'
                })
        
        return patterns


class SystemCallPatterns:
    """Detect system call patterns"""
    
    def detect(self, normalized_code: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        patterns = []
        
        for i, line in enumerate(normalized_code):
            if 'int 80h' in line or 'syscall' in line:
                # Look at preceding instructions to determine call type
                prev_lines = code[max(0, i-3):i]
                
                if any('eax' in l and ('1' in l or '3' in l or '4' in l) for l in prev_lines):
                    patterns.append({
                        'call_type': 'file_io',
                        'confidence': 0.6,
                        'line': i,
                        'details': 'File I/O system call'
                    })
                elif any('eax' in l and ('45' in l or '9' in l) for l in prev_lines):
                    patterns.append({
                        'call_type': 'memory',
                        'confidence': 0.6,
                        'line': i,
                        'details': 'Memory management system call'
                    })
        
        return patterns