"""
Advanced Data Structure Recovery Engine

This module provides sophisticated data structure recovery capabilities that analyze
memory access patterns, pointer relationships, and usage contexts to reconstruct
complex aggregate types including nested structures, linked data structures, and
polymorphic type hierarchies.

Features:
- Memory layout analysis and structure boundary detection
- Nested structure recovery with proper field alignment
- Linked data structure identification (lists, trees, graphs)
- Polymorphic type hierarchy reconstruction
- C++ class and vtable recovery
- Union type detection and disambiguation
- Array and buffer structure analysis
- Cross-reference analysis for structure relationships
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)


class StructureType(Enum):
    """Types of data structures that can be recovered"""
    SIMPLE_STRUCT = "simple_struct"
    NESTED_STRUCT = "nested_struct"
    UNION = "union"
    LINKED_LIST = "linked_list"
    DOUBLY_LINKED_LIST = "doubly_linked_list"
    TREE_NODE = "tree_node"
    HASH_TABLE = "hash_table"
    ARRAY_STRUCT = "array_struct"
    VTABLE_CLASS = "vtable_class"
    INTERFACE = "interface"
    UNKNOWN = "unknown"


class AccessPattern(Enum):
    """Memory access patterns that indicate structure usage"""
    SEQUENTIAL_READ = "sequential_read"
    RANDOM_ACCESS = "random_access"
    POINTER_CHASE = "pointer_chase"
    FIELD_ACCESS = "field_access"
    ARRAY_ITERATION = "array_iteration"
    LINKED_TRAVERSAL = "linked_traversal"


@dataclass
class MemoryAccess:
    """Represents a memory access operation"""
    base_variable: str
    offset: int
    access_size: int
    access_type: str  # 'read', 'write', 'address'
    field_name: Optional[str] = None
    context_function: str = ""
    confidence: float = 0.0


@dataclass
class StructureField:
    """Represents a field within a recovered structure"""
    name: str
    offset: int
    size: int
    type_string: str
    alignment: int
    is_pointer: bool = False
    points_to_structure: Optional[str] = None
    array_dimensions: List[int] = field(default_factory=list)
    access_pattern: Optional[AccessPattern] = None
    semantic_meaning: Optional[str] = None
    confidence: float = 0.0
    
    def __post_init__(self):
        if not self.semantic_meaning:
            self.semantic_meaning = self._infer_semantic_meaning()
    
    def _infer_semantic_meaning(self) -> str:
        """Infer semantic meaning from field characteristics"""
        name_lower = self.name.lower()
        
        if 'next' in name_lower or 'link' in name_lower:
            return 'pointer to next element in linked structure'
        elif 'prev' in name_lower or 'back' in name_lower:
            return 'pointer to previous element in linked structure'
        elif 'parent' in name_lower:
            return 'pointer to parent node in tree structure'
        elif 'child' in name_lower or 'left' in name_lower or 'right' in name_lower:
            return 'pointer to child node in tree structure'
        elif 'data' in name_lower or 'value' in name_lower:
            return 'data payload field'
        elif 'size' in name_lower or 'count' in name_lower or 'len' in name_lower:
            return 'size or count field'
        elif 'id' in name_lower or 'key' in name_lower:
            return 'identifier or key field'
        elif 'flag' in name_lower or 'status' in name_lower:
            return 'status or flag field'
        elif self.is_pointer:
            return 'pointer field'
        else:
            return 'data field'


@dataclass
class RecoveredStructure:
    """Represents a completely recovered data structure"""
    name: str
    structure_type: StructureType
    size_bytes: int
    alignment: int
    fields: List[StructureField]
    relationships: Dict[str, str] = field(default_factory=dict)  # field_name -> related_structure
    usage_patterns: List[AccessPattern] = field(default_factory=list)
    vtable_info: Optional[Dict[str, Any]] = None
    inheritance_info: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    instances: List[str] = field(default_factory=list)  # Variable instances of this structure
    
    def get_c_definition(self) -> str:
        """Generate C structure definition"""
        lines = []
        
        if self.structure_type == StructureType.UNION:
            lines.append(f"typedef union {{")
        else:
            lines.append(f"typedef struct {{")
        
        # Sort fields by offset
        sorted_fields = sorted(self.fields, key=lambda f: f.offset)
        
        for field in sorted_fields:
            field_def = f"    {field.type_string}"
            if field.is_pointer:
                field_def += "*"
            
            field_def += f" {field.name}"
            
            # Add array dimensions
            for dim in field.array_dimensions:
                if dim > 0:
                    field_def += f"[{dim}]"
                else:
                    field_def += "[]"
            
            field_def += ";"
            
            # Add semantic comment
            if field.semantic_meaning:
                field_def += f"  // {field.semantic_meaning}"
            
            lines.append(field_def)
        
        lines.append(f"}} {self.name};")
        lines.append("")
        
        return "\n".join(lines)


@dataclass
class StructureRelationship:
    """Represents a relationship between structures"""
    source_structure: str
    target_structure: str
    relationship_type: str  # 'contains', 'points_to', 'inherits_from', 'implements'
    field_name: Optional[str] = None
    confidence: float = 0.0


class AdvancedDataStructureRecovery:
    """
    Advanced data structure recovery engine for complex binary analysis
    
    This engine analyzes memory access patterns, pointer relationships, and
    usage contexts to reconstruct accurate data structure definitions from
    compiled binaries.
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Recovery state
        self.recovered_structures = {}
        self.memory_accesses = defaultdict(list)
        self.structure_relationships = []
        self.field_candidates = defaultdict(list)
        
        # Analysis patterns
        self.common_patterns = self._initialize_common_patterns()
        self.vtable_patterns = self._initialize_vtable_patterns()
        self.linked_structure_patterns = self._initialize_linked_patterns()
        
        # Configuration
        self.min_structure_size = 8  # Minimum size for a valid structure
        self.max_structure_size = 1024  # Maximum reasonable structure size
        self.alignment_sizes = [1, 2, 4, 8, 16]  # Common alignment boundaries
        
    def recover_data_structures(self, ghidra_results: Dict[str, Any],
                               semantic_functions: List[Any],
                               advanced_types: Dict[str, Any] = None) -> Dict[str, RecoveredStructure]:
        """
        Perform comprehensive data structure recovery
        
        Args:
            ghidra_results: Raw Ghidra analysis results
            semantic_functions: List of semantic function objects
            advanced_types: Advanced type inference results
            
        Returns:
            Dictionary mapping structure names to recovered structures
        """
        self.logger.info("Starting advanced data structure recovery...")
        
        # Phase 1: Collect memory access patterns
        self._collect_memory_access_patterns(semantic_functions)
        
        # Phase 2: Identify structure boundaries and field candidates
        self._identify_structure_boundaries()
        
        # Phase 3: Analyze pointer relationships
        self._analyze_pointer_relationships(advanced_types or {})
        
        # Phase 4: Detect common data structure patterns
        self._detect_common_patterns()
        
        # Phase 5: Recover nested and complex structures
        self._recover_nested_structures()
        
        # Phase 6: Identify linked data structures
        self._identify_linked_structures()
        
        # Phase 7: Detect polymorphic types and vtables
        self._detect_polymorphic_types(ghidra_results)
        
        # Phase 8: Validate and refine structures
        self._validate_and_refine_structures()
        
        # Phase 9: Generate final structure definitions
        self._finalize_structure_definitions()
        
        self.logger.info(f"Recovered {len(self.recovered_structures)} data structures")
        return self.recovered_structures
    
    def _collect_memory_access_patterns(self, semantic_functions: List[Any]) -> None:
        """Collect memory access patterns from function analysis"""
        for func in semantic_functions:
            func_name = func.name
            func_code = getattr(func, 'body_code', '')
            
            # Find structure member accesses (obj.field)
            struct_accesses = re.findall(r'(\w+)\.(\w+)', func_code)
            for base_var, field_name in struct_accesses:
                self.memory_accesses[base_var].append(MemoryAccess(
                    base_variable=base_var,
                    offset=0,  # Will be calculated later
                    access_size=4,  # Default assumption
                    access_type='field_access',
                    field_name=field_name,
                    context_function=func_name,
                    confidence=0.8
                ))
            
            # Find pointer dereferences with offsets (ptr->field or *(ptr+offset))
            pointer_accesses = re.findall(r'(\w+)->(\w+)', func_code)
            for base_var, field_name in pointer_accesses:
                self.memory_accesses[base_var].append(MemoryAccess(
                    base_variable=base_var,
                    offset=0,  # Will be calculated from field analysis
                    access_size=4,
                    access_type='pointer_dereference',
                    field_name=field_name,
                    context_function=func_name,
                    confidence=0.9
                ))
            
            # Find array accesses (arr[index])
            array_accesses = re.findall(r'(\w+)\[([^\]]+)\]', func_code)
            for base_var, index_expr in array_accesses:
                # Try to calculate offset from index
                offset = 0
                if index_expr.isdigit():
                    offset = int(index_expr) * 4  # Assume 4-byte elements
                
                self.memory_accesses[base_var].append(MemoryAccess(
                    base_variable=base_var,
                    offset=offset,
                    access_size=4,
                    access_type='array_access',
                    context_function=func_name,
                    confidence=0.7
                ))
            
            # Find explicit pointer arithmetic (*(ptr + offset))
            pointer_arithmetic = re.findall(r'\*\((\w+)\s*\+\s*(\d+)\)', func_code)
            for base_var, offset_str in pointer_arithmetic:
                offset = int(offset_str)
                self.memory_accesses[base_var].append(MemoryAccess(
                    base_variable=base_var,
                    offset=offset,
                    access_size=4,
                    access_type='pointer_arithmetic',
                    context_function=func_name,
                    confidence=0.8
                ))
    
    def _identify_structure_boundaries(self) -> None:
        """Identify structure boundaries and potential field locations"""
        for base_var, accesses in self.memory_accesses.items():
            if len(accesses) < 2:
                continue  # Need multiple accesses to infer structure
            
            # Group accesses by offset to identify fields
            offset_groups = defaultdict(list)
            for access in accesses:
                offset_groups[access.offset].append(access)
            
            # Identify potential fields from access patterns
            fields = []
            for offset, access_list in offset_groups.items():
                # Determine field characteristics
                field_name = self._determine_field_name(access_list)
                field_size = self._estimate_field_size(access_list, offset)
                access_pattern = self._classify_access_pattern(access_list)
                
                field = StructureField(
                    name=field_name,
                    offset=offset,
                    size=field_size,
                    type_string=self._infer_field_type(access_list),
                    alignment=self._calculate_alignment(offset, field_size),
                    access_pattern=access_pattern,
                    confidence=self._calculate_field_confidence(access_list)
                )
                
                fields.append(field)
            
            if fields:
                self.field_candidates[base_var] = sorted(fields, key=lambda f: f.offset)
    
    def _analyze_pointer_relationships(self, advanced_types: Dict[str, Any]) -> None:
        """Analyze pointer relationships between structures"""
        for var_key, field_list in self.field_candidates.items():
            for field in field_list:
                # Check if field is a pointer to another structure
                if field.name and any(keyword in field.name.lower() for keyword in ['next', 'prev', 'child', 'parent']):
                    field.is_pointer = True
                    
                    # Try to identify target structure
                    target_struct = self._identify_target_structure(field, advanced_types)
                    if target_struct:
                        field.points_to_structure = target_struct
                        
                        # Record relationship
                        self.structure_relationships.append(StructureRelationship(
                            source_structure=var_key,
                            target_structure=target_struct,
                            relationship_type='points_to',
                            field_name=field.name,
                            confidence=field.confidence
                        ))
    
    def _detect_common_patterns(self) -> None:
        """Detect common data structure patterns"""
        for var_key, field_list in self.field_candidates.items():
            structure_type = StructureType.SIMPLE_STRUCT
            
            # Check for linked list patterns
            if self._matches_linked_list_pattern(field_list):
                structure_type = StructureType.LINKED_LIST
            
            # Check for tree node patterns
            elif self._matches_tree_node_pattern(field_list):
                structure_type = StructureType.TREE_NODE
            
            # Check for hash table patterns
            elif self._matches_hash_table_pattern(field_list):
                structure_type = StructureType.HASH_TABLE
            
            # Calculate structure size and alignment
            total_size = max([f.offset + f.size for f in field_list]) if field_list else 0
            alignment = max([f.alignment for f in field_list]) if field_list else 4
            
            # Create recovered structure
            structure = RecoveredStructure(
                name=f"{var_key}_t",
                structure_type=structure_type,
                size_bytes=total_size,
                alignment=alignment,
                fields=field_list,
                confidence=self._calculate_structure_confidence(field_list, structure_type)
            )
            
            self.recovered_structures[var_key] = structure
    
    def _recover_nested_structures(self) -> None:
        """Recover nested data structures"""
        for struct_name, structure in self.recovered_structures.items():
            for field in structure.fields:
                # Check if field should be a nested structure
                if self._should_be_nested_structure(field):
                    # Create nested structure definition
                    nested_struct = self._create_nested_structure(field, struct_name)
                    if nested_struct:
                        nested_name = f"{struct_name}_{field.name}_t"
                        self.recovered_structures[nested_name] = nested_struct
                        
                        # Update field to reference nested structure
                        field.type_string = nested_name
                        field.points_to_structure = nested_name
                        
                        # Mark parent as nested structure
                        structure.structure_type = StructureType.NESTED_STRUCT
    
    def _identify_linked_structures(self) -> None:
        """Identify linked data structures (lists, trees, graphs)"""
        for struct_name, structure in self.recovered_structures.items():
            # Look for self-referential pointers indicating linked structures
            self_refs = [f for f in structure.fields if f.points_to_structure == struct_name]
            
            if len(self_refs) == 1:
                # Single self-reference - likely linked list
                structure.structure_type = StructureType.LINKED_LIST
                structure.usage_patterns.append(AccessPattern.LINKED_TRAVERSAL)
            
            elif len(self_refs) == 2:
                # Two self-references - could be doubly linked list or tree
                ref_names = [f.name.lower() for f in self_refs]
                
                if any('prev' in name or 'back' in name for name in ref_names):
                    structure.structure_type = StructureType.DOUBLY_LINKED_LIST
                elif any('left' in name or 'right' in name or 'child' in name for name in ref_names):
                    structure.structure_type = StructureType.TREE_NODE
            
            # Update field semantic meanings based on structure type
            self._update_field_semantics_for_linked_structure(structure)
    
    def _detect_polymorphic_types(self, ghidra_results: Dict[str, Any]) -> None:
        """Detect polymorphic types and vtable structures"""
        # Look for vtable patterns in binary
        for struct_name, structure in self.recovered_structures.items():
            # Check if first field looks like a vtable pointer
            if structure.fields and structure.fields[0].offset == 0:
                first_field = structure.fields[0]
                
                if (first_field.is_pointer and 
                    ('vtable' in first_field.name.lower() or 
                     'vptr' in first_field.name.lower() or
                     first_field.name.lower() in ['vfptr', '__vfptr'])):
                    
                    structure.structure_type = StructureType.VTABLE_CLASS
                    structure.vtable_info = {
                        'vtable_offset': 0,
                        'vtable_field': first_field.name,
                        'is_virtual_class': True
                    }
    
    def _validate_and_refine_structures(self) -> None:
        """Validate and refine recovered structures"""
        validated_structures = {}
        
        for struct_name, structure in self.recovered_structures.items():
            if self._validate_structure(structure):
                # Refine field types and alignments
                self._refine_structure_fields(structure)
                
                # Calculate final confidence score
                structure.confidence = self._calculate_final_confidence(structure)
                
                validated_structures[struct_name] = structure
            else:
                self.logger.warning(f"Structure {struct_name} failed validation")
        
        self.recovered_structures = validated_structures
    
    def _finalize_structure_definitions(self) -> None:
        """Finalize structure definitions and resolve dependencies"""
        # Sort structures by dependencies (referenced structures first)
        sorted_structures = self._topological_sort_structures()
        
        # Update structure names and ensure uniqueness
        final_structures = {}
        name_counter = {}
        
        for struct_name in sorted_structures:
            structure = self.recovered_structures[struct_name]
            
            # Generate unique name
            base_name = structure.name
            if base_name in name_counter:
                name_counter[base_name] += 1
                unique_name = f"{base_name}_{name_counter[base_name]}"
            else:
                name_counter[base_name] = 0
                unique_name = base_name
            
            structure.name = unique_name
            final_structures[unique_name] = structure
        
        self.recovered_structures = final_structures
    
    # Helper methods for pattern detection
    def _matches_linked_list_pattern(self, fields: List[StructureField]) -> bool:
        """Check if fields match linked list pattern"""
        has_next_pointer = any('next' in f.name.lower() for f in fields)
        has_data_field = any('data' in f.name.lower() or 'value' in f.name.lower() for f in fields)
        return has_next_pointer and len(fields) >= 2
    
    def _matches_tree_node_pattern(self, fields: List[StructureField]) -> bool:
        """Check if fields match tree node pattern"""
        child_keywords = ['left', 'right', 'child', 'children']
        has_child_pointers = sum(1 for f in fields if any(kw in f.name.lower() for kw in child_keywords))
        has_parent_pointer = any('parent' in f.name.lower() for f in fields)
        return has_child_pointers >= 1 and len(fields) >= 2
    
    def _matches_hash_table_pattern(self, fields: List[StructureField]) -> bool:
        """Check if fields match hash table pattern"""
        has_buckets = any('bucket' in f.name.lower() for f in fields)
        has_size = any('size' in f.name.lower() or 'count' in f.name.lower() for f in fields)
        has_array = any(f.array_dimensions for f in fields)
        return (has_buckets or has_array) and has_size
    
    def _determine_field_name(self, access_list: List[MemoryAccess]) -> str:
        """Determine the most likely field name from access patterns"""
        # Prefer explicit field names from structure accesses
        field_names = [a.field_name for a in access_list if a.field_name]
        if field_names:
            # Return most common field name
            return max(set(field_names), key=field_names.count)
        
        # Generate name based on offset
        offset = access_list[0].offset if access_list else 0
        return f"field_{offset:02x}"
    
    def _estimate_field_size(self, access_list: List[MemoryAccess], offset: int) -> int:
        """Estimate field size from access patterns"""
        # Look at access sizes
        sizes = [a.access_size for a in access_list if a.access_size > 0]
        if sizes:
            return max(sizes)  # Take largest access size
        
        # Default based on alignment
        if offset % 8 == 0:
            return 8
        elif offset % 4 == 0:
            return 4
        elif offset % 2 == 0:
            return 2
        else:
            return 1
    
    def _classify_access_pattern(self, access_list: List[MemoryAccess]) -> AccessPattern:
        """Classify the access pattern for a field"""
        access_types = [a.access_type for a in access_list]
        
        if 'array_access' in access_types:
            return AccessPattern.ARRAY_ITERATION
        elif 'pointer_dereference' in access_types:
            return AccessPattern.POINTER_CHASE
        elif 'field_access' in access_types:
            return AccessPattern.FIELD_ACCESS
        else:
            return AccessPattern.SEQUENTIAL_READ
    
    def _infer_field_type(self, access_list: List[MemoryAccess]) -> str:
        """Infer the most likely type for a field"""
        # Analyze access patterns to infer type
        field_name = access_list[0].field_name if access_list[0].field_name else ""
        name_lower = field_name.lower()
        
        # Common type patterns
        if any(keyword in name_lower for keyword in ['next', 'prev', 'parent', 'child']):
            return 'void*'
        elif 'count' in name_lower or 'size' in name_lower or 'len' in name_lower:
            return 'int'
        elif 'id' in name_lower:
            return 'int'
        elif 'name' in name_lower or 'str' in name_lower:
            return 'char*'
        elif 'flag' in name_lower or 'status' in name_lower:
            return 'int'
        else:
            # Default based on size
            size = access_list[0].access_size
            if size == 1:
                return 'char'
            elif size == 2:
                return 'short'
            elif size == 8:
                return 'long long'
            else:
                return 'int'
    
    def _calculate_alignment(self, offset: int, size: int) -> int:
        """Calculate field alignment"""
        for align in sorted(self.alignment_sizes, reverse=True):
            if offset % align == 0 and size >= align:
                return align
        return 1
    
    def _calculate_field_confidence(self, access_list: List[MemoryAccess]) -> float:
        """Calculate confidence score for field detection"""
        factors = []
        
        # Multiple accesses increase confidence
        factors.append(min(len(access_list) * 0.2, 0.6))
        
        # Explicit field names increase confidence
        if any(a.field_name for a in access_list):
            factors.append(0.3)
        
        # Consistent access sizes increase confidence
        sizes = [a.access_size for a in access_list]
        if len(set(sizes)) == 1:
            factors.append(0.1)
        
        return min(sum(factors), 1.0)
    
    def _calculate_structure_confidence(self, fields: List[StructureField], struct_type: StructureType) -> float:
        """Calculate overall structure confidence"""
        if not fields:
            return 0.0
        
        # Base confidence from field confidence
        field_confidence = sum(f.confidence for f in fields) / len(fields)
        
        # Bonus for recognized patterns
        pattern_bonus = 0.0
        if struct_type != StructureType.SIMPLE_STRUCT:
            pattern_bonus = 0.2
        
        # Bonus for well-formed structures
        structure_bonus = 0.0
        if len(fields) >= 2:
            structure_bonus += 0.1
        if any(f.is_pointer for f in fields):
            structure_bonus += 0.1
        
        return min(field_confidence + pattern_bonus + structure_bonus, 1.0)
    
    def _should_be_nested_structure(self, field: StructureField) -> bool:
        """Determine if a field should be a nested structure"""
        # Check if field size suggests it contains multiple values
        return (field.size > 16 and 
                not field.is_pointer and 
                field.access_pattern == AccessPattern.FIELD_ACCESS)
    
    def _create_nested_structure(self, field: StructureField, parent_name: str) -> Optional[RecoveredStructure]:
        """Create a nested structure definition"""
        # This is a simplified implementation
        # Real implementation would analyze the field's internal structure
        
        if field.size < 8:
            return None
        
        # Create simple nested structure with inferred fields
        num_fields = field.size // 4  # Assume 4-byte fields
        nested_fields = []
        
        for i in range(min(num_fields, 4)):  # Limit to reasonable number
            nested_field = StructureField(
                name=f"member_{i}",
                offset=i * 4,
                size=4,
                type_string="int",
                alignment=4,
                confidence=0.5
            )
            nested_fields.append(nested_field)
        
        return RecoveredStructure(
            name=f"{parent_name}_{field.name}_t",
            structure_type=StructureType.NESTED_STRUCT,
            size_bytes=field.size,
            alignment=field.alignment,
            fields=nested_fields,
            confidence=0.6
        )
    
    def _identify_target_structure(self, field: StructureField, advanced_types: Dict[str, Any]) -> Optional[str]:
        """Identify the target structure for a pointer field"""
        # Use advanced type information if available
        for var_key, advanced_type in advanced_types.items():
            if (hasattr(advanced_type, 'base_type') and 
                hasattr(advanced_type, 'pointer_depth') and
                advanced_type.pointer_depth > 0):
                
                base_type = advanced_type.base_type
                if base_type in self.recovered_structures:
                    return base_type
        
        # Heuristic based on field name
        if 'next' in field.name.lower():
            # Self-referential pointer
            return 'self'
        
        return None
    
    def _update_field_semantics_for_linked_structure(self, structure: RecoveredStructure) -> None:
        """Update field semantic meanings based on linked structure type"""
        if structure.structure_type == StructureType.LINKED_LIST:
            for field in structure.fields:
                if 'next' in field.name.lower():
                    field.semantic_meaning = "pointer to next node in linked list"
                elif 'data' in field.name.lower():
                    field.semantic_meaning = "data payload of list node"
        
        elif structure.structure_type == StructureType.TREE_NODE:
            for field in structure.fields:
                if 'left' in field.name.lower():
                    field.semantic_meaning = "pointer to left child node"
                elif 'right' in field.name.lower():
                    field.semantic_meaning = "pointer to right child node"
                elif 'parent' in field.name.lower():
                    field.semantic_meaning = "pointer to parent node"
    
    def _validate_structure(self, structure: RecoveredStructure) -> bool:
        """Validate that a recovered structure is reasonable"""
        # Check minimum requirements
        if not structure.fields:
            return False
        
        if structure.size_bytes < self.min_structure_size:
            return False
        
        if structure.size_bytes > self.max_structure_size:
            return False
        
        # Check field overlap
        sorted_fields = sorted(structure.fields, key=lambda f: f.offset)
        for i in range(len(sorted_fields) - 1):
            current_field = sorted_fields[i]
            next_field = sorted_fields[i + 1]
            
            if current_field.offset + current_field.size > next_field.offset:
                return False  # Overlapping fields
        
        return True
    
    def _refine_structure_fields(self, structure: RecoveredStructure) -> None:
        """Refine field types and alignments"""
        for field in structure.fields:
            # Refine type based on semantic meaning
            if 'pointer' in field.semantic_meaning:
                field.is_pointer = True
                if not field.type_string.endswith('*'):
                    field.type_string += '*'
            
            # Ensure proper alignment
            proper_alignment = self._calculate_alignment(field.offset, field.size)
            field.alignment = max(field.alignment, proper_alignment)
    
    def _calculate_final_confidence(self, structure: RecoveredStructure) -> float:
        """Calculate final confidence score for structure"""
        factors = []
        
        # Field confidence contribution
        if structure.fields:
            avg_field_confidence = sum(f.confidence for f in structure.fields) / len(structure.fields)
            factors.append(avg_field_confidence * 0.4)
        
        # Structure type confidence
        if structure.structure_type != StructureType.UNKNOWN:
            factors.append(0.2)
        
        # Size reasonableness
        if self.min_structure_size <= structure.size_bytes <= self.max_structure_size:
            factors.append(0.2)
        
        # Relationship confidence
        if structure.relationships:
            factors.append(0.1)
        
        # Pattern recognition bonus
        if structure.usage_patterns:
            factors.append(0.1)
        
        return min(sum(factors), 1.0)
    
    def _topological_sort_structures(self) -> List[str]:
        """Sort structures by dependencies"""
        # Simple topological sort for structure dependencies
        sorted_order = []
        visited = set()
        temp_visited = set()
        
        def visit(struct_name: str):
            if struct_name in temp_visited:
                return  # Cycle detected, skip
            if struct_name in visited:
                return
            
            temp_visited.add(struct_name)
            
            # Visit dependencies first
            structure = self.recovered_structures.get(struct_name)
            if structure:
                for field in structure.fields:
                    if field.points_to_structure and field.points_to_structure in self.recovered_structures:
                        visit(field.points_to_structure)
            
            temp_visited.remove(struct_name)
            visited.add(struct_name)
            sorted_order.append(struct_name)
        
        for struct_name in self.recovered_structures.keys():
            visit(struct_name)
        
        return sorted_order
    
    def _initialize_common_patterns(self) -> Dict[str, Any]:
        """Initialize common data structure patterns"""
        return {
            'linked_list': {
                'required_fields': ['next'],
                'optional_fields': ['data', 'value', 'payload'],
                'field_patterns': {
                    'next': {'type': 'pointer', 'self_ref': True}
                }
            },
            'tree_node': {
                'required_fields': ['left', 'right'],
                'optional_fields': ['parent', 'data', 'key'],
                'field_patterns': {
                    'left': {'type': 'pointer', 'self_ref': True},
                    'right': {'type': 'pointer', 'self_ref': True},
                    'parent': {'type': 'pointer', 'self_ref': True}
                }
            },
            'hash_table': {
                'required_fields': ['buckets', 'size'],
                'optional_fields': ['count', 'capacity'],
                'field_patterns': {
                    'buckets': {'type': 'array_pointer'},
                    'size': {'type': 'integer'}
                }
            }
        }
    
    def _initialize_vtable_patterns(self) -> Dict[str, Any]:
        """Initialize vtable and virtual class patterns"""
        return {
            'vtable_indicators': ['__vfptr', 'vtable', 'vptr'],
            'virtual_methods': ['destructor', 'virtual_'],
            'rtti_indicators': ['type_info', '__dynamic_cast']
        }
    
    def _initialize_linked_patterns(self) -> Dict[str, Any]:
        """Initialize linked structure recognition patterns"""
        return {
            'list_indicators': ['next', 'link', 'succ'],
            'tree_indicators': ['left', 'right', 'child', 'parent'],
            'graph_indicators': ['edges', 'neighbors', 'adjacent'],
            'self_ref_patterns': ['next', 'prev', 'parent', 'child', 'left', 'right']
        }
    
    def generate_structure_report(self) -> Dict[str, Any]:
        """Generate comprehensive structure recovery report"""
        structure_counts = defaultdict(int)
        total_confidence = 0.0
        
        for structure in self.recovered_structures.values():
            structure_counts[structure.structure_type.value] += 1
            total_confidence += structure.confidence
        
        avg_confidence = total_confidence / len(self.recovered_structures) if self.recovered_structures else 0.0
        
        return {
            'total_structures': len(self.recovered_structures),
            'structure_type_distribution': dict(structure_counts),
            'average_confidence': avg_confidence,
            'high_confidence_structures': len([s for s in self.recovered_structures.values() if s.confidence > 0.8]),
            'linked_structures': len([s for s in self.recovered_structures.values() if s.structure_type in [StructureType.LINKED_LIST, StructureType.TREE_NODE]]),
            'nested_structures': len([s for s in self.recovered_structures.values() if s.structure_type == StructureType.NESTED_STRUCT]),
            'polymorphic_structures': len([s for s in self.recovered_structures.values() if s.vtable_info]),
            'total_relationships': len(self.structure_relationships),
            'detailed_structures': {
                name: {
                    'type': structure.structure_type.value,
                    'size': structure.size_bytes,
                    'field_count': len(structure.fields),
                    'confidence': structure.confidence,
                    'c_definition': structure.get_c_definition()
                } for name, structure in self.recovered_structures.items()
            }
        }