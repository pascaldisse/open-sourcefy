"""
Agent 10: Resource Reconstructor
Reconstructs resources, data files, and embedded content from binary analysis.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
import re
import subprocess
import json
import struct
from pathlib import Path
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent10_ResourceReconstructor(BaseAgent):
    """Agent 10: Resource and data reconstruction"""
    
    def __init__(self):
        super().__init__(
            agent_id=10,
            name="ResourceReconstructor",
            dependencies=[8, 9]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute resource reconstruction"""
        agent8_result = context['agent_results'].get(8)
        agent9_result = context['agent_results'].get(9)
        
        if not (agent8_result and agent8_result.status == AgentStatus.COMPLETED and
                agent9_result and agent9_result.status == AgentStatus.COMPLETED):
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Required dependencies not completed successfully"
            )

        try:
            diff_analysis = agent8_result.data
            assembly_analysis = agent9_result.data
            
            resource_reconstruction = self._perform_resource_reconstruction(
                diff_analysis, assembly_analysis, context
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=resource_reconstruction,
                metadata={
                    'depends_on': [8, 9],
                    'analysis_type': 'resource_reconstruction'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Resource reconstruction failed: {str(e)}"
            )

    def _perform_resource_reconstruction(self, diff_analysis: Dict[str, Any], 
                                       assembly_analysis: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform resource reconstruction"""
        # Get binary path for analysis
        binary_path = context.get('global_data', {}).get('binary_path') or context.get('binary_path')
        
        return {
            'extracted_resources': self._extract_resources(diff_analysis, assembly_analysis),
            'reconstructed_data_structures': self._reconstruct_data_structures(assembly_analysis),
            'string_tables': self._reconstruct_string_tables(context),
            'embedded_files': self._identify_embedded_files(context),
            'resource_dependencies': self._analyze_resource_dependencies(diff_analysis),
            'binary_path': binary_path,
            'reconstruction_confidence': self._calculate_reconstruction_confidence(diff_analysis, assembly_analysis),
            'total_resources_found': self._count_total_resources(diff_analysis, assembly_analysis),
            'critical_resources': self._count_critical_resources(diff_analysis, assembly_analysis),
            'resource_integrity': self._calculate_resource_integrity(diff_analysis, assembly_analysis)
        }

    def _extract_resources(self, diff_analysis: Dict[str, Any], assembly_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract resources from binary"""
        try:
            # Extract PE/ELF resources
            pe_resources = self._extract_pe_resources(assembly_analysis)
            
            # Extract string resources
            string_resources = self._extract_string_resources(assembly_analysis)
            
            # Extract icon and GUI resources
            gui_resources = self._extract_gui_resources(assembly_analysis)
            
            # Extract version information
            version_info = self._extract_version_info(assembly_analysis)
            
            return {
                'pe_resources': pe_resources,
                'string_resources': string_resources,
                'gui_resources': gui_resources,
                'version_info': version_info,
                'total_resource_count': len(pe_resources) + len(string_resources) + len(gui_resources),
                'extraction_success_rate': self._calculate_extraction_success_rate(pe_resources, string_resources, gui_resources)
            }
        except Exception as e:
            return {
                'error': f"Resource extraction failed: {str(e)}",
                'pe_resources': {},
                'string_resources': {},
                'gui_resources': {},
                'version_info': {}
            }

    def _reconstruct_data_structures(self, assembly_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct data structures from memory patterns"""
        try:
            # Get memory access patterns from assembly analysis
            memory_patterns = assembly_analysis.get('memory_patterns', {})
            detected_structures = memory_patterns.get('detected_structures', {})
            
            # Reconstruct arrays
            arrays = self._reconstruct_arrays(detected_structures, memory_patterns)
            
            # Reconstruct structures/classes
            structures = self._reconstruct_structures(detected_structures, memory_patterns)
            
            # Reconstruct linked lists and trees
            linked_structures = self._reconstruct_linked_structures(detected_structures, memory_patterns)
            
            # Reconstruct hash tables and maps
            hash_tables = self._reconstruct_hash_tables(detected_structures, memory_patterns)
            
            # Infer data types
            type_inference = self._perform_type_inference(assembly_analysis)
            
            return {
                'arrays': arrays,
                'structures': structures,
                'linked_structures': linked_structures,
                'hash_tables': hash_tables,
                'type_inference': type_inference,
                'total_structures_found': len(arrays) + len(structures) + len(linked_structures) + len(hash_tables),
                'reconstruction_confidence': self._calculate_structure_confidence(arrays, structures, linked_structures, hash_tables)
            }
        except Exception as e:
            return {
                'error': f"Data structure reconstruction failed: {str(e)}",
                'arrays': [],
                'structures': [],
                'linked_structures': [],
                'hash_tables': [],
                'type_inference': {}
            }

    def _reconstruct_string_tables(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct string tables and constants"""
        try:
            binary_path = context.get('global_data', {}).get('binary_path') or context.get('binary_path')
            
            # Extract strings from binary
            extracted_strings = self._extract_strings_from_binary(binary_path)
            
            # Detect encoding
            encoding_analysis = self._detect_string_encoding(extracted_strings)
            
            # Classify strings by usage
            string_classification = self._classify_strings(extracted_strings)
            
            # Detect localization patterns
            localization = self._detect_localization_patterns(extracted_strings)
            
            # Build string tables
            string_tables = self._build_string_tables(string_classification, localization)
            
            return {
                'extracted_strings': extracted_strings,
                'encoding_analysis': encoding_analysis,
                'string_classification': string_classification,
                'localization': localization,
                'string_tables': string_tables,
                'total_strings': len(extracted_strings),
                'unique_strings': len(set(extracted_strings))
            }
        except Exception as e:
            return {
                'error': f"String table reconstruction failed: {str(e)}",
                'extracted_strings': [],
                'encoding_analysis': {},
                'string_classification': {},
                'localization': {},
                'string_tables': {}
            }

    def _identify_embedded_files(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify embedded files and data"""
        try:
            binary_path = context.get('global_data', {}).get('binary_path') or context.get('binary_path')
            
            # Detect file signatures
            file_signatures = self._detect_file_signatures(binary_path)
            
            # Analyze entropy for compression/encryption
            entropy_analysis = self._analyze_entropy_patterns(binary_path)
            
            # Extract embedded files
            embedded_files = self._extract_embedded_files(binary_path, file_signatures)
            
            # Analyze binary sections
            section_analysis = self._analyze_binary_sections(binary_path)
            
            return {
                'file_signatures': file_signatures,
                'entropy_analysis': entropy_analysis,
                'embedded_files': embedded_files,
                'section_analysis': section_analysis,
                'total_embedded_files': len(embedded_files),
                'compression_detected': entropy_analysis.get('high_entropy_regions', 0) > 0
            }
        except Exception as e:
            return {
                'error': f"Embedded file identification failed: {str(e)}",
                'file_signatures': [],
                'entropy_analysis': {},
                'embedded_files': [],
                'section_analysis': {}
            }

    def _analyze_resource_dependencies(self, diff_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependencies between resources"""
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(diff_analysis)
            
            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies(dependency_graph)
            
            # Analyze loading order
            loading_order = self._analyze_loading_order(dependency_graph)
            
            # Analyze resource lifetime
            lifetime_analysis = self._analyze_resource_lifetime(diff_analysis)
            
            return {
                'dependency_graph': dependency_graph,
                'circular_dependencies': circular_deps,
                'loading_order': loading_order,
                'lifetime_analysis': lifetime_analysis,
                'total_dependencies': len(dependency_graph),
                'dependency_complexity': self._calculate_dependency_complexity(dependency_graph)
            }
        except Exception as e:
            return {
                'error': f"Resource dependency analysis failed: {str(e)}",
                'dependency_graph': {},
                'circular_dependencies': [],
                'loading_order': [],
                'lifetime_analysis': {}
            }

    def _calculate_reconstruction_confidence(self, diff_analysis: Dict[str, Any], assembly_analysis: Dict[str, Any]) -> float:
        """Calculate reconstruction confidence score"""
        try:
            confidence_factors = []
            
            # Factor in memory pattern detection success
            memory_patterns = assembly_analysis.get('memory_patterns', {})
            if memory_patterns.get('detected_structures'):
                confidence_factors.append(0.3)
            
            # Factor in string extraction success
            if assembly_analysis.get('string_analysis'):
                confidence_factors.append(0.2)
            
            # Factor in assembly analysis quality
            asm_analysis = assembly_analysis.get('instruction_analysis', {})
            if asm_analysis.get('control_flow_structures'):
                confidence_factors.append(0.2)
            
            # Factor in diff analysis availability
            if diff_analysis and isinstance(diff_analysis, dict):
                confidence_factors.append(0.3)
            
            return min(sum(confidence_factors), 1.0)
        except Exception:
            return 0.5  # Default moderate confidence

    def _count_total_resources(self, diff_analysis: Dict[str, Any], assembly_analysis: Dict[str, Any]) -> int:
        """Count total resources found in binary"""
        try:
            total_count = 0
            
            # Count memory structures
            memory_patterns = assembly_analysis.get('memory_patterns', {})
            detected_structures = memory_patterns.get('detected_structures', {})
            total_count += len(detected_structures.get('arrays', []))
            total_count += len(detected_structures.get('structs', []))
            total_count += len(detected_structures.get('pointers', []))
            
            # Count string resources
            if assembly_analysis.get('string_analysis'):
                total_count += 10  # Estimated string resources
            
            # Count from diff analysis
            if diff_analysis and isinstance(diff_analysis, dict):
                total_count += len(diff_analysis.get('resource_differences', []))
            
            return total_count
        except Exception:
            return 0

    def _count_critical_resources(self, diff_analysis: Dict[str, Any], assembly_analysis: Dict[str, Any]) -> int:
        """Count critical resources required for functionality"""
        try:
            critical_count = 0
            
            # Critical memory structures (main data structures)
            memory_patterns = assembly_analysis.get('memory_patterns', {})
            detected_structures = memory_patterns.get('detected_structures', {})
            
            # Arrays and structs used in main functions are critical
            arrays = detected_structures.get('arrays', [])
            structs = detected_structures.get('structs', [])
            
            for array in arrays:
                if 'main' in array.get('function', '').lower():
                    critical_count += 1
            
            for struct in structs:
                if 'main' in struct.get('function', '').lower():
                    critical_count += 1
            
            # String resources in main functions are critical
            critical_count += min(5, len(arrays) + len(structs))  # Cap at 5
            
            return critical_count
        except Exception:
            return 0

    def _calculate_resource_integrity(self, diff_analysis: Dict[str, Any], assembly_analysis: Dict[str, Any]) -> float:
        """Calculate resource integrity score"""
        try:
            integrity_factors = []
            
            # Check assembly analysis completeness
            asm_analysis = assembly_analysis.get('instruction_analysis', {})
            if asm_analysis.get('control_flow_structures'):
                integrity_factors.append(0.4)
            
            # Check memory pattern detection
            memory_patterns = assembly_analysis.get('memory_patterns', {})
            if memory_patterns.get('detected_structures'):
                integrity_factors.append(0.3)
            
            # Check analysis confidence
            confidence = assembly_analysis.get('analysis_confidence', 0.0)
            if confidence > 0.7:
                integrity_factors.append(0.3)
            elif confidence > 0.4:
                integrity_factors.append(0.15)
            
            return min(sum(integrity_factors), 1.0)
        except Exception:
            return 0.5  # Default moderate integrity
    
    # Helper methods for resource extraction
    def _extract_pe_resources(self, assembly_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract PE/ELF resources from binary analysis"""
        pe_resources = {}
        
        try:
            # Look for resource references in assembly analysis
            if isinstance(assembly_analysis, dict):
                # Check for Ghidra analysis that might contain resource information
                ghidra_data = assembly_analysis.get('ghidra_analysis', {})
                if isinstance(ghidra_data, dict):
                    # Extract resource sections if available
                    functions = ghidra_data.get('functions', {})
                    for func_name, func_data in functions.items():
                        if isinstance(func_data, dict):
                            code = func_data.get('code', '')
                            # Look for resource loading patterns
                            if any(resource_call in code.lower() for resource_call in ['loadresource', 'findresource', 'getmodulehandle']):
                                pe_resources[func_name] = {
                                    'type': 'resource_loader',
                                    'function': func_name,
                                    'code': code[:200],  # First 200 chars
                                    'address': func_data.get('address', '0x0')
                                }
                
                # Look for string references that might be resource IDs
                instruction_analysis = assembly_analysis.get('instruction_analysis', {})
                if isinstance(instruction_analysis, dict):
                    # Check for immediate constants that could be resource IDs
                    optimizations = instruction_analysis.get('optimization_patterns', {})
                    constants = optimizations.get('constant_folding', [])
                    
                    resource_id_candidates = []
                    for constant in constants:
                        if isinstance(constant, dict):
                            instruction = constant.get('instruction', '')
                            # Look for patterns like mov eax, 101 (resource ID)
                            if 'mov' in instruction.lower() and any(char.isdigit() for char in instruction):
                                resource_id_candidates.append({
                                    'instruction': instruction,
                                    'potential_id': self._extract_numeric_constant(instruction)
                                })
                    
                    if resource_id_candidates:
                        pe_resources['resource_ids'] = resource_id_candidates
            
            return pe_resources
        except Exception:
            return {}
    
    def _extract_numeric_constant(self, instruction: str) -> Optional[int]:
        """Extract numeric constant from instruction"""
        # Look for numbers in the instruction
        numbers = re.findall(r'\b\d+\b', instruction)
        if numbers:
            try:
                return int(numbers[0])
            except ValueError:
                pass
        
        # Look for hex numbers
        hex_numbers = re.findall(r'0x[0-9a-fA-F]+', instruction)
        if hex_numbers:
            try:
                return int(hex_numbers[0], 16)
            except ValueError:
                pass
        
        return None
    
    def _extract_string_resources(self, assembly_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract string resources from assembly analysis"""
        string_resources = []
        
        try:
            # Look for string operations in assembly
            instruction_analysis = assembly_analysis.get('instruction_analysis', {})
            if isinstance(instruction_analysis, dict):
                instruction_types = instruction_analysis.get('instruction_classification', {})
                memory_ops = instruction_types.get('memory', [])
                
                for instruction in memory_ops:
                    # Look for string-like patterns
                    if any(string_hint in instruction.lower() for string_hint in ['str', 'char', 'text']):
                        string_resources.append({
                            'instruction': instruction,
                            'type': 'string_reference',
                            'source': 'assembly_analysis'
                        })
            
            # Look for strings in decompiled functions
            if 'ghidra_analysis' in assembly_analysis:
                ghidra_data = assembly_analysis['ghidra_analysis']
                if isinstance(ghidra_data, dict):
                    functions = ghidra_data.get('functions', {})
                    for func_name, func_data in functions.items():
                        if isinstance(func_data, dict):
                            code = func_data.get('code', '')
                            # Extract string literals from code
                            string_literals = re.findall(r'\"([^\"]+)\"', code)
                            for literal in string_literals:
                                if len(literal) > 2:  # Ignore very short strings
                                    string_resources.append({
                                        'value': literal,
                                        'function': func_name,
                                        'type': 'string_literal',
                                        'source': 'decompiled_code'
                                    })
            
            return string_resources
        except Exception:
            return []
    
    def _extract_gui_resources(self, assembly_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract GUI resources (icons, menus, dialogs)"""
        gui_resources = []
        
        try:
            # Look for GUI-related function calls
            if 'ghidra_analysis' in assembly_analysis:
                ghidra_data = assembly_analysis['ghidra_analysis']
                if isinstance(ghidra_data, dict):
                    functions = ghidra_data.get('functions', {})
                    for func_name, func_data in functions.items():
                        if isinstance(func_data, dict):
                            code = func_data.get('code', '')
                            
                            # Look for GUI API calls
                            gui_apis = ['CreateWindow', 'DialogBox', 'LoadIcon', 'LoadMenu', 'MessageBox']
                            for api in gui_apis:
                                if api.lower() in code.lower():
                                    gui_resources.append({
                                        'api_call': api,
                                        'function': func_name,
                                        'type': 'gui_api_reference',
                                        'code_snippet': self._extract_code_snippet(code, api)
                                    })
            
            return gui_resources
        except Exception:
            return []
    
    def _extract_code_snippet(self, code: str, keyword: str) -> str:
        """Extract code snippet around a keyword"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if keyword.lower() in line.lower():
                # Return the line and a few lines around it
                start = max(0, i - 1)
                end = min(len(lines), i + 2)
                return '\n'.join(lines[start:end])
        return ''
    
    def _extract_version_info(self, assembly_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract version information from assembly analysis"""
        version_info = {}
        
        try:
            # Look for version-related strings
            if 'ghidra_analysis' in assembly_analysis:
                ghidra_data = assembly_analysis['ghidra_analysis']
                if isinstance(ghidra_data, dict):
                    functions = ghidra_data.get('functions', {})
                    for func_name, func_data in functions.items():
                        if isinstance(func_data, dict):
                            code = func_data.get('code', '')
                            
                            # Look for version patterns
                            version_patterns = [
                                r'(\d+\.\d+\.\d+\.\d+)',  # 1.0.0.0
                                r'(\d+\.\d+\.\d+)',       # 1.0.0
                                r'(\d+\.\d+)',            # 1.0
                            ]
                            
                            for pattern in version_patterns:
                                matches = re.findall(pattern, code)
                                if matches:
                                    version_info[func_name] = {
                                        'version_candidates': matches,
                                        'function': func_name
                                    }
                                    break
            
            return version_info
        except Exception:
            return {}
    
    def _calculate_extraction_success_rate(self, pe_resources: Dict[str, Any], 
                                         string_resources: List[Dict[str, Any]], 
                                         gui_resources: List[Dict[str, Any]]) -> float:
        """Calculate resource extraction success rate"""
        total_attempted = 3  # PE, strings, GUI
        successful = 0
        
        if pe_resources:
            successful += 1
        if string_resources:
            successful += 1
        if gui_resources:
            successful += 1
        
        return successful / total_attempted
    
    # Helper methods for data structure reconstruction
    def _reconstruct_arrays(self, detected_structures: Dict[str, Any], memory_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reconstruct array data structures"""
        arrays = []
        
        try:
            array_accesses = detected_structures.get('arrays', [])
            for array_access in array_accesses:
                if isinstance(array_access, dict):
                    # Analyze the array access pattern
                    base_register = array_access.get('base_register', '')
                    index_register = array_access.get('index_register', '')
                    
                    array_info = {
                        'type': 'array',
                        'base_register': base_register,
                        'index_register': index_register,
                        'access_pattern': array_access.get('access_instruction', ''),
                        'function': array_access.get('function', ''),
                        'inferred_element_type': self._infer_array_element_type(array_access),
                        'estimated_size': self._estimate_array_size(array_access, memory_patterns),
                        'access_frequency': self._calculate_access_frequency(array_access, memory_patterns)
                    }
                    
                    arrays.append(array_info)
            
            return arrays
        except Exception:
            return []
    
    def _infer_array_element_type(self, array_access: Dict[str, Any]) -> str:
        """Infer array element type from access pattern"""
        instruction = array_access.get('access_instruction', '').lower()
        
        # Look for size indicators in the instruction
        if any(size_hint in instruction for size_hint in ['byte', 'db', 'al', 'bl']):
            return 'char'
        elif any(size_hint in instruction for size_hint in ['word', 'dw', 'ax', 'bx']):
            return 'short'
        elif any(size_hint in instruction for size_hint in ['dword', 'dd', 'eax', 'ebx']):
            return 'int'
        elif any(size_hint in instruction for size_hint in ['qword', 'dq', 'rax', 'rbx']):
            return 'long long'
        else:
            return 'int'  # Default assumption
    
    def _estimate_array_size(self, array_access: Dict[str, Any], memory_patterns: Dict[str, Any]) -> int:
        """Estimate array size from access patterns"""
        # This is a simplified estimation
        # In reality, would need to analyze all accesses to the same array
        function = array_access.get('function', '')
        
        # Look for loop patterns in the same function
        # If array is accessed in a loop, the loop bound gives us size hint
        # For now, return a reasonable default
        return 10  # Default estimated size
    
    def _calculate_access_frequency(self, array_access: Dict[str, Any], memory_patterns: Dict[str, Any]) -> int:
        """Calculate how frequently an array is accessed"""
        # Count accesses to the same base register in the same function
        base_register = array_access.get('base_register', '')
        function = array_access.get('function', '')
        
        # This would need access to all memory accesses
        # For now, return a default
        return 1
    
    def _reconstruct_structures(self, detected_structures: Dict[str, Any], memory_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reconstruct structure/class data types"""
        structures = []
        
        try:
            struct_accesses = detected_structures.get('structs', [])
            
            # Group struct accesses by base register (same struct instance)
            struct_groups = {}
            for struct_access in struct_accesses:
                if isinstance(struct_access, dict):
                    base_register = struct_access.get('base_register', '')
                    function = struct_access.get('function', '')
                    key = f"{function}_{base_register}"
                    
                    if key not in struct_groups:
                        struct_groups[key] = []
                    struct_groups[key].append(struct_access)
            
            # Analyze each struct group
            for group_key, accesses in struct_groups.items():
                struct_info = self._analyze_struct_group(group_key, accesses)
                if struct_info:
                    structures.append(struct_info)
            
            return structures
        except Exception:
            return []
    
    def _analyze_struct_group(self, group_key: str, accesses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze a group of struct accesses to infer structure layout"""
        if not accesses:
            return None
        
        # Extract offsets and infer field types
        fields = []
        offsets = set()
        
        for access in accesses:
            offset_str = access.get('offset', '0')
            try:
                # Parse offset (could be hex or decimal)
                if offset_str.startswith('0x'):
                    offset = int(offset_str, 16)
                else:
                    offset = int(offset_str)
                
                if offset not in offsets:
                    offsets.add(offset)
                    field_type = self._infer_field_type(access)
                    fields.append({
                        'offset': offset,
                        'type': field_type,
                        'access_instruction': access.get('access_instruction', '')
                    })
            except (ValueError, TypeError):
                continue
        
        if not fields:
            return None
        
        # Sort fields by offset
        fields.sort(key=lambda x: x['offset'])
        
        return {
            'type': 'structure',
            'group_key': group_key,
            'fields': fields,
            'estimated_size': max(f['offset'] for f in fields) + 4 if fields else 0,  # Add 4 for last field
            'field_count': len(fields),
            'function': accesses[0].get('function', ''),
            'base_register': accesses[0].get('base_register', '')
        }
    
    def _infer_field_type(self, struct_access: Dict[str, Any]) -> str:
        """Infer field type from struct access pattern"""
        instruction = struct_access.get('access_instruction', '').lower()
        
        # Similar to array element type inference
        if any(size_hint in instruction for size_hint in ['byte', 'db', 'al', 'bl']):
            return 'char'
        elif any(size_hint in instruction for size_hint in ['word', 'dw', 'ax', 'bx']):
            return 'short'
        elif any(size_hint in instruction for size_hint in ['dword', 'dd', 'eax', 'ebx']):
            return 'int'
        elif any(size_hint in instruction for size_hint in ['qword', 'dq', 'rax', 'rbx']):
            return 'long long'
        else:
            return 'int'
    
    def _reconstruct_linked_structures(self, detected_structures: Dict[str, Any], memory_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reconstruct linked lists, trees, and other pointer-based structures"""
        linked_structures = []
        
        try:
            pointer_accesses = detected_structures.get('pointers', [])
            
            # Look for patterns that suggest linked structures
            for pointer_access in pointer_accesses:
                if isinstance(pointer_access, dict):
                    # Check if this looks like a linked structure
                    if self._is_linked_structure_pattern(pointer_access, memory_patterns):
                        linked_info = {
                            'type': 'linked_structure',
                            'pattern_type': self._classify_linked_pattern(pointer_access),
                            'pointer_register': pointer_access.get('pointer_register', ''),
                            'access_instruction': pointer_access.get('access_instruction', ''),
                            'function': pointer_access.get('function', ''),
                            'complexity': self._estimate_linked_complexity(pointer_access)
                        }
                        linked_structures.append(linked_info)
            
            return linked_structures
        except Exception:
            return []
    
    def _is_linked_structure_pattern(self, pointer_access: Dict[str, Any], memory_patterns: Dict[str, Any]) -> bool:
        """Check if pointer access pattern suggests a linked structure"""
        instruction = pointer_access.get('access_instruction', '').lower()
        
        # Look for patterns like: mov eax, [eax] or mov eax, [eax+4]
        # These suggest traversing linked structures
        if 'mov' in instruction and '[' in instruction and ']' in instruction:
            # Check if the same register appears inside and outside brackets
            register = pointer_access.get('pointer_register', '').lower()
            if register and register in instruction:
                # Count occurrences - if register appears both as source and in memory operand
                return instruction.count(register) >= 2
        
        return False
    
    def _classify_linked_pattern(self, pointer_access: Dict[str, Any]) -> str:
        """Classify the type of linked structure pattern"""
        instruction = pointer_access.get('access_instruction', '').lower()
        
        # Simple heuristics for classification
        if '+4' in instruction or '+8' in instruction:
            return 'linked_list'  # Fixed offset suggests next pointer
        elif '+' in instruction and any(reg in instruction for reg in ['ecx', 'edx']):
            return 'tree_structure'  # Multiple registers suggest tree traversal
        else:
            return 'pointer_chain'  # Generic pointer following
    
    def _estimate_linked_complexity(self, pointer_access: Dict[str, Any]) -> str:
        """Estimate complexity of linked structure"""
        instruction = pointer_access.get('access_instruction', '').lower()
        
        # Count the number of memory references and registers
        memory_refs = instruction.count('[')
        register_count = len(set(re.findall(r'\b[er][a-z][x]?\b', instruction)))
        
        if memory_refs > 1 or register_count > 2:
            return 'complex'
        elif memory_refs == 1 and register_count <= 2:
            return 'simple'
        else:
            return 'moderate'
    
    def _reconstruct_hash_tables(self, detected_structures: Dict[str, Any], memory_patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Reconstruct hash tables and associative structures"""
        hash_tables = []
        
        try:
            # Look for patterns that suggest hash table access
            array_accesses = detected_structures.get('arrays', [])
            
            for array_access in array_accesses:
                if isinstance(array_access, dict):
                    # Check if array access pattern suggests hash table
                    if self._is_hash_table_pattern(array_access):
                        hash_info = {
                            'type': 'hash_table',
                            'base_register': array_access.get('base_register', ''),
                            'index_register': array_access.get('index_register', ''),
                            'access_instruction': array_access.get('access_instruction', ''),
                            'function': array_access.get('function', ''),
                            'hash_method': self._infer_hash_method(array_access),
                            'estimated_bucket_count': self._estimate_bucket_count(array_access)
                        }
                        hash_tables.append(hash_info)
            
            return hash_tables
        except Exception:
            return []
    
    def _is_hash_table_pattern(self, array_access: Dict[str, Any]) -> bool:
        """Check if array access pattern suggests hash table usage"""
        instruction = array_access.get('access_instruction', '').lower()
        
        # Look for bit operations or modulo operations that suggest hashing
        hash_indicators = ['and', 'xor', 'shl', 'shr', 'mod']
        return any(indicator in instruction for indicator in hash_indicators)
    
    def _infer_hash_method(self, array_access: Dict[str, Any]) -> str:
        """Infer hash method from access pattern"""
        instruction = array_access.get('access_instruction', '').lower()
        
        if 'and' in instruction:
            return 'bit_mask'  # Common for power-of-2 table sizes
        elif 'xor' in instruction:
            return 'xor_hash'
        elif any(shift in instruction for shift in ['shl', 'shr']):
            return 'shift_hash'
        else:
            return 'unknown_hash'
    
    def _estimate_bucket_count(self, array_access: Dict[str, Any]) -> int:
        """Estimate hash table bucket count"""
        instruction = array_access.get('access_instruction', '').lower()
        
        # Look for mask values that suggest table size
        # For example: and eax, 0x1f suggests 32 buckets
        mask_match = re.search(r'and.*?0x([0-9a-f]+)', instruction)
        if mask_match:
            try:
                mask_value = int(mask_match.group(1), 16)
                return mask_value + 1  # Mask of 0x1f means 32 buckets
            except ValueError:
                pass
        
        return 16  # Default reasonable size
    
    def _perform_type_inference(self, assembly_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced type inference from assembly patterns"""
        type_inference = {
            'inferred_types': {},
            'type_confidence': {},
            'type_relationships': []
        }
        
        try:
            # Analyze instruction patterns for type hints
            instruction_analysis = assembly_analysis.get('instruction_analysis', {})
            if isinstance(instruction_analysis, dict):
                classification = instruction_analysis.get('instruction_classification', {})
                
                # Analyze arithmetic operations for numeric types
                arithmetic_ops = classification.get('arithmetic', [])
                for op in arithmetic_ops:
                    type_hint = self._infer_type_from_arithmetic(op)
                    if type_hint:
                        type_inference['inferred_types'][op] = type_hint
                        type_inference['type_confidence'][op] = 0.7
                
                # Analyze memory operations for pointer types
                memory_ops = classification.get('memory', [])
                for op in memory_ops:
                    type_hint = self._infer_type_from_memory_op(op)
                    if type_hint:
                        type_inference['inferred_types'][op] = type_hint
                        type_inference['type_confidence'][op] = 0.6
            
            return type_inference
        except Exception:
            return {'inferred_types': {}, 'type_confidence': {}, 'type_relationships': []}
    
    def _infer_type_from_arithmetic(self, instruction: str) -> Optional[str]:
        """Infer data type from arithmetic instruction"""
        instruction_lower = instruction.lower()
        
        # Look for floating point operations
        if any(fp_op in instruction_lower for fp_op in ['fadd', 'fsub', 'fmul', 'fdiv', 'fld', 'fst']):
            return 'float'
        
        # Look for 64-bit operations
        if any(reg64 in instruction_lower for reg64 in ['rax', 'rbx', 'rcx', 'rdx']):
            return 'long long'
        
        # Look for 32-bit operations
        if any(reg32 in instruction_lower for reg32 in ['eax', 'ebx', 'ecx', 'edx']):
            return 'int'
        
        # Look for 16-bit operations
        if any(reg16 in instruction_lower for reg16 in ['ax', 'bx', 'cx', 'dx']):
            return 'short'
        
        # Look for 8-bit operations
        if any(reg8 in instruction_lower for reg8 in ['al', 'bl', 'cl', 'dl']):
            return 'char'
        
        return None
    
    def _infer_type_from_memory_op(self, instruction: str) -> Optional[str]:
        """Infer data type from memory operation"""
        instruction_lower = instruction.lower()
        
        # Look for pointer size indicators
        if 'qword ptr' in instruction_lower:
            return 'long long*'
        elif 'dword ptr' in instruction_lower:
            return 'int*'
        elif 'word ptr' in instruction_lower:
            return 'short*'
        elif 'byte ptr' in instruction_lower:
            return 'char*'
        elif '[' in instruction and ']' in instruction:
            return 'void*'  # Generic pointer
        
        return None
    
    def _calculate_structure_confidence(self, arrays: List[Dict[str, Any]], 
                                      structures: List[Dict[str, Any]], 
                                      linked_structures: List[Dict[str, Any]], 
                                      hash_tables: List[Dict[str, Any]]) -> float:
        """Calculate overall structure reconstruction confidence"""
        total_structures = len(arrays) + len(structures) + len(linked_structures) + len(hash_tables)
        
        if total_structures == 0:
            return 0.0
        
        # Weight different structure types by complexity
        confidence_score = 0.0
        confidence_score += len(arrays) * 0.8  # Arrays are easier to detect
        confidence_score += len(structures) * 0.6  # Structures are moderate
        confidence_score += len(linked_structures) * 0.4  # Linked structures are harder
        confidence_score += len(hash_tables) * 0.3  # Hash tables are hardest
        
        max_possible_score = total_structures * 0.8  # If all were arrays
        
        return min(confidence_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0
    
    # Helper methods for string table reconstruction
    def _extract_strings_from_binary(self, binary_path: Optional[str]) -> List[str]:
        """Extract strings from binary file"""
        if not binary_path or not Path(binary_path).exists():
            return []
        
        try:
            # Use strings command to extract printable strings
            result = subprocess.run(['strings', binary_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                strings = result.stdout.strip().split('\n')
                # Filter out very short strings and clean up
                return [s.strip() for s in strings if len(s.strip()) > 3]
            else:
                # Fallback: read binary and extract strings manually
                return self._manual_string_extraction(binary_path)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return self._manual_string_extraction(binary_path)
    
    def _manual_string_extraction(self, binary_path: str) -> List[str]:
        """Manually extract strings from binary file"""
        try:
            with open(binary_path, 'rb') as f:
                binary_data = f.read()
            
            # Extract ASCII strings (minimum length 4)
            strings = []
            current_string = ''
            
            for byte in binary_data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += chr(byte)
                else:
                    if len(current_string) >= 4:
                        strings.append(current_string)
                    current_string = ''
            
            # Add final string if it exists
            if len(current_string) >= 4:
                strings.append(current_string)
            
            return strings[:1000]  # Limit to first 1000 strings
        except Exception:
            return []
    
    def _detect_string_encoding(self, strings: List[str]) -> Dict[str, Any]:
        """Detect string encoding patterns"""
        encoding_analysis = {
            'ascii_count': 0,
            'unicode_count': 0,
            'utf8_count': 0,
            'detected_encoding': 'ascii',
            'confidence': 0.0
        }
        
        if not strings:
            return encoding_analysis
        
        for string in strings:
            try:
                # Check if string is pure ASCII
                string.encode('ascii')
                encoding_analysis['ascii_count'] += 1
            except UnicodeEncodeError:
                # Check if it's UTF-8
                try:
                    string.encode('utf-8')
                    encoding_analysis['utf8_count'] += 1
                except UnicodeEncodeError:
                    encoding_analysis['unicode_count'] += 1
        
        total_strings = len(strings)
        if total_strings > 0:
            ascii_ratio = encoding_analysis['ascii_count'] / total_strings
            utf8_ratio = encoding_analysis['utf8_count'] / total_strings
            
            if ascii_ratio > 0.8:
                encoding_analysis['detected_encoding'] = 'ascii'
                encoding_analysis['confidence'] = ascii_ratio
            elif utf8_ratio > 0.5:
                encoding_analysis['detected_encoding'] = 'utf-8'
                encoding_analysis['confidence'] = utf8_ratio
            else:
                encoding_analysis['detected_encoding'] = 'mixed'
                encoding_analysis['confidence'] = max(ascii_ratio, utf8_ratio)
        
        return encoding_analysis
    
    def _classify_strings(self, strings: List[str]) -> Dict[str, List[str]]:
        """Classify strings by their likely usage"""
        classification = {
            'error_messages': [],
            'ui_strings': [],
            'file_paths': [],
            'urls': [],
            'debug_strings': [],
            'version_strings': [],
            'other': []
        }
        
        for string in strings:
            string_lower = string.lower()
            
            # Classify by patterns
            if any(error_word in string_lower for error_word in ['error', 'failed', 'exception', 'warning']):
                classification['error_messages'].append(string)
            elif any(ui_word in string_lower for ui_word in ['button', 'menu', 'dialog', 'window', 'ok', 'cancel']):
                classification['ui_strings'].append(string)
            elif '\\' in string or '/' in string and ('.' in string or 'program' in string_lower):
                classification['file_paths'].append(string)
            elif string.startswith(('http://', 'https://', 'ftp://', 'www.')):
                classification['urls'].append(string)
            elif any(debug_word in string_lower for debug_word in ['debug', 'trace', 'log', 'printf', 'cout']):
                classification['debug_strings'].append(string)
            elif re.search(r'\d+\.\d+', string):
                classification['version_strings'].append(string)
            else:
                classification['other'].append(string)
        
        return classification
    
    def _detect_localization_patterns(self, strings: List[str]) -> Dict[str, Any]:
        """Detect localization and internationalization patterns"""
        localization = {
            'has_localization': False,
            'language_indicators': [],
            'resource_ids': [],
            'string_tables': {}
        }
        
        # Look for common localization patterns
        for string in strings:
            # Check for language codes
            if re.search(r'\b(en|de|fr|es|it|ja|zh|ko|ru)[-_]([A-Z]{2})\b', string):
                localization['has_localization'] = True
                localization['language_indicators'].append(string)
            
            # Check for resource ID patterns
            if re.search(r'IDS_\w+|STR_\w+|\%\d+', string):
                localization['resource_ids'].append(string)
        
        return localization
    
    def _build_string_tables(self, classification: Dict[str, List[str]], 
                           localization: Dict[str, Any]) -> Dict[str, Any]:
        """Build organized string tables"""
        string_tables = {
            'error_table': classification.get('error_messages', []),
            'ui_table': classification.get('ui_strings', []),
            'debug_table': classification.get('debug_strings', []),
            'version_table': classification.get('version_strings', []),
            'file_path_table': classification.get('file_paths', []),
            'url_table': classification.get('urls', []),
            'total_entries': sum(len(strings) for strings in classification.values())
        }
        
        # Add localization tables if detected
        if localization.get('has_localization'):
            string_tables['localization_table'] = localization.get('language_indicators', [])
            string_tables['resource_id_table'] = localization.get('resource_ids', [])
        
        return string_tables
    
    # Helper methods for embedded file identification
    def _detect_file_signatures(self, binary_path: Optional[str]) -> List[Dict[str, Any]]:
        """Detect file signatures in binary"""
        if not binary_path or not Path(binary_path).exists():
            return []
        
        signatures = []
        file_signatures = {
            b'\\x89PNG\\r\\n\\x1a\\n': 'PNG',
            b'\\xff\\xd8\\xff': 'JPEG',
            b'GIF87a': 'GIF87a',
            b'GIF89a': 'GIF89a',
            b'RIFF': 'RIFF/WAV',
            b'\\x00\\x00\\x01\\x00': 'ICO',
            b'BM': 'BMP',
            b'PK\\x03\\x04': 'ZIP',
            b'\\x1f\\x8b\\x08': 'GZIP',
            b'%PDF': 'PDF',
            b'\\xd0\\xcf\\x11\\xe0': 'MS Office'
        }
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read(1024 * 1024)  # Read first 1MB
            
            for signature, file_type in file_signatures.items():
                offset = 0
                while True:
                    pos = data.find(signature, offset)
                    if pos == -1:
                        break
                    
                    signatures.append({
                        'type': file_type,
                        'offset': pos,
                        'signature': signature.hex(),
                        'size_estimate': self._estimate_embedded_file_size(data, pos)
                    })
                    
                    offset = pos + 1
        
        except Exception:
            pass
        
        return signatures
    
    def _estimate_embedded_file_size(self, data: bytes, start_offset: int) -> int:
        """Estimate size of embedded file"""
        # This is a simplified estimation
        # Look for common file endings or size headers
        max_search = min(len(data) - start_offset, 64 * 1024)  # Search up to 64KB
        
        # For now, return a reasonable default
        return min(max_search, 4096)  # Default 4KB
    
    def _analyze_entropy_patterns(self, binary_path: Optional[str]) -> Dict[str, Any]:
        """Analyze entropy patterns to detect compression/encryption"""
        if not binary_path or not Path(binary_path).exists():
            return {}
        
        entropy_analysis = {
            'high_entropy_regions': 0,
            'low_entropy_regions': 0,
            'compression_indicators': [],
            'encryption_indicators': []
        }
        
        try:
            with open(binary_path, 'rb') as f:
                # Read file in chunks
                chunk_size = 4096
                high_entropy_threshold = 7.0
                low_entropy_threshold = 3.0
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    entropy = self._calculate_entropy(chunk)
                    
                    if entropy > high_entropy_threshold:
                        entropy_analysis['high_entropy_regions'] += 1
                        # High entropy might indicate compression or encryption
                        if self._looks_like_compression(chunk):
                            entropy_analysis['compression_indicators'].append({
                                'offset': f.tell() - len(chunk),
                                'entropy': entropy,
                                'type': 'compression_suspected'
                            })
                        else:
                            entropy_analysis['encryption_indicators'].append({
                                'offset': f.tell() - len(chunk),
                                'entropy': entropy,
                                'type': 'encryption_suspected'
                            })
                    elif entropy < low_entropy_threshold:
                        entropy_analysis['low_entropy_regions'] += 1
        
        except Exception:
            pass
        
        return entropy_analysis
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = [0] * 256
        for byte in data:
            byte_counts[byte] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in byte_counts:
            if count > 0:
                probability = count / data_len
                entropy -= probability * (probability.bit_length() - 1)
        
        return entropy
    
    def _looks_like_compression(self, data: bytes) -> bool:
        """Check if data looks like compressed content"""
        # Look for compression signatures or patterns
        compression_signatures = [
            b'\\x1f\\x8b',  # GZIP
            b'BZ',          # BZIP2
            b'\\x78\\x9c',  # ZLIB default
            b'\\x78\\x01',  # ZLIB best speed
            b'\\x78\\xda',  # ZLIB best compression
        ]
        
        for sig in compression_signatures:
            if data.startswith(sig):
                return True
        
        return False
    
    def _extract_embedded_files(self, binary_path: Optional[str], 
                              signatures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract embedded files based on detected signatures"""
        if not binary_path or not signatures:
            return []
        
        embedded_files = []
        
        try:
            with open(binary_path, 'rb') as f:
                for sig_info in signatures:
                    offset = sig_info['offset']
                    file_type = sig_info['type']
                    estimated_size = sig_info['size_estimate']
                    
                    f.seek(offset)
                    file_data = f.read(estimated_size)
                    
                    if file_data:
                        embedded_files.append({
                            'type': file_type,
                            'offset': offset,
                            'size': len(file_data),
                            'extracted': True,
                            'checksum': self._calculate_simple_checksum(file_data)
                        })
        
        except Exception:
            pass
        
        return embedded_files
    
    def _calculate_simple_checksum(self, data: bytes) -> str:
        """Calculate simple checksum for data"""
        checksum = 0
        for byte in data:
            checksum = (checksum + byte) % 65536
        return f"{checksum:04x}"
    
    def _analyze_binary_sections(self, binary_path: Optional[str]) -> Dict[str, Any]:
        """Analyze binary sections for embedded content"""
        if not binary_path:
            return {}
        
        section_analysis = {
            'pe_sections': [],
            'elf_sections': [],
            'resource_sections': [],
            'data_sections': []
        }
        
        try:
            # This is a simplified analysis
            # In a real implementation, would use libraries like pefile or pyelftools
            with open(binary_path, 'rb') as f:
                header = f.read(64)
                
                # Check for PE signature
                if b'MZ' in header:
                    section_analysis['pe_sections'] = self._analyze_pe_sections(f)
                # Check for ELF signature
                elif header.startswith(b'\\x7fELF'):
                    section_analysis['elf_sections'] = self._analyze_elf_sections(f)
        
        except Exception:
            pass
        
        return section_analysis
    
    def _analyze_pe_sections(self, file_handle) -> List[Dict[str, Any]]:
        """Analyze PE sections (simplified)"""
        # This is a placeholder implementation
        # Real implementation would parse PE headers properly
        return [
            {'name': '.text', 'type': 'code', 'size': 'unknown'},
            {'name': '.data', 'type': 'data', 'size': 'unknown'},
            {'name': '.rsrc', 'type': 'resource', 'size': 'unknown'}
        ]
    
    def _analyze_elf_sections(self, file_handle) -> List[Dict[str, Any]]:
        """Analyze ELF sections (simplified)"""
        # This is a placeholder implementation
        # Real implementation would parse ELF headers properly
        return [
            {'name': '.text', 'type': 'code', 'size': 'unknown'},
            {'name': '.data', 'type': 'data', 'size': 'unknown'},
            {'name': '.rodata', 'type': 'readonly_data', 'size': 'unknown'}
        ]
    
    # Helper methods for resource dependency analysis
    def _build_dependency_graph(self, diff_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build resource dependency graph"""
        dependency_graph = {}
        
        if not isinstance(diff_analysis, dict):
            return dependency_graph
        
        # This is a simplified implementation
        # In reality, would analyze actual resource references
        resource_differences = diff_analysis.get('resource_differences', [])
        
        for i, resource in enumerate(resource_differences):
            resource_id = f"resource_{i}"
            dependency_graph[resource_id] = []
            
            # Look for references to other resources
            if isinstance(resource, dict):
                # Simple heuristic: if resource contains numbers, those might be references
                resource_str = str(resource)
                numbers = re.findall(r'\\b\\d+\\b', resource_str)
                for num in numbers[:3]:  # Limit to first 3 to avoid noise
                    if num != str(i):  # Don't reference self
                        dependency_graph[resource_id].append(f"resource_{num}")
        
        return dependency_graph
    
    def _detect_circular_dependencies(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect circular dependencies in resource graph"""
        circular_deps = []
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                circular_deps.append(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dependency_graph.get(node, []):
                if dfs(neighbor, path):
                    break
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in dependency_graph:
            if node not in visited:
                dfs(node, [])
        
        return circular_deps
    
    def _analyze_loading_order(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Analyze optimal resource loading order"""
        # Topological sort to determine loading order
        in_degree = {node: 0 for node in dependency_graph}
        
        # Calculate in-degrees
        for node in dependency_graph:
            for neighbor in dependency_graph[node]:
                if neighbor in in_degree:
                    in_degree[neighbor] += 1
        
        # Kahn's algorithm for topological sorting
        queue = [node for node in in_degree if in_degree[node] == 0]
        loading_order = []
        
        while queue:
            node = queue.pop(0)
            loading_order.append(node)
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return loading_order
    
    def _analyze_resource_lifetime(self, diff_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource lifetime patterns"""
        lifetime_analysis = {
            'persistent_resources': [],
            'temporary_resources': [],
            'cached_resources': [],
            'average_lifetime': 'unknown'
        }
        
        # This is a simplified analysis
        # In reality, would analyze actual usage patterns
        resource_differences = diff_analysis.get('resource_differences', [])
        
        # Classify resources by estimated lifetime
        for i, resource in enumerate(resource_differences):
            resource_id = f"resource_{i}"
            
            if isinstance(resource, dict):
                # Simple heuristics for lifetime classification
                resource_str = str(resource).lower()
                
                if any(persistent_hint in resource_str for persistent_hint in ['config', 'settings', 'global']):
                    lifetime_analysis['persistent_resources'].append(resource_id)
                elif any(temp_hint in resource_str for temp_hint in ['temp', 'cache', 'buffer']):
                    lifetime_analysis['temporary_resources'].append(resource_id)
                else:
                    lifetime_analysis['cached_resources'].append(resource_id)
        
        return lifetime_analysis
    
    def _calculate_dependency_complexity(self, dependency_graph: Dict[str, List[str]]) -> float:
        """Calculate dependency graph complexity"""
        if not dependency_graph:
            return 0.0
        
        total_nodes = len(dependency_graph)
        total_edges = sum(len(deps) for deps in dependency_graph.values())
        
        # Simple complexity metric: edge-to-node ratio
        if total_nodes == 0:
            return 0.0
        
        complexity = total_edges / total_nodes
        return min(complexity, 10.0)  # Cap at 10.0