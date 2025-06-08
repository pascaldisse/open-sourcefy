"""
Binary Comparison Engine
Provides semantic similarity analysis and functional equivalence testing for binary files.
Part of Phase 3: Advanced Analysis & Binary Intelligence
"""

import os
import hashlib
import difflib
from typing import Dict, Any, List, Tuple, Optional
import json


class BinaryComparisonEngine:
    """Engine for comparing binary files and analyzing similarity"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
        self.hash_algorithms = ['md5', 'sha1', 'sha256']
    
    def compare_binaries(self, binary1_path: str, binary2_path: str, 
                        context1: Dict[str, Any] = None, 
                        context2: Dict[str, Any] = None) -> Dict[str, Any]:
        """Compare two binary files comprehensively"""
        comparison_result = {
            'binary1': binary1_path,
            'binary2': binary2_path,
            'hash_comparison': {},
            'size_comparison': {},
            'metadata_comparison': {},
            'semantic_similarity': {},
            'functional_equivalence': {},
            'structure_comparison': {},
            'overall_similarity_score': 0.0,
            'analysis_confidence': 0.0
        }
        
        try:
            # Basic file comparisons
            comparison_result['hash_comparison'] = self._compare_hashes(binary1_path, binary2_path)
            comparison_result['size_comparison'] = self._compare_sizes(binary1_path, binary2_path)
            
            # Metadata comparison (if analysis context available)
            if context1 and context2:
                comparison_result['metadata_comparison'] = self._compare_metadata(context1, context2)
                comparison_result['structure_comparison'] = self._compare_structure(context1, context2)
                comparison_result['semantic_similarity'] = self._analyze_semantic_similarity(context1, context2)
                comparison_result['functional_equivalence'] = self._assess_functional_equivalence(context1, context2)
            
            # Calculate overall similarity score
            comparison_result['overall_similarity_score'] = self._calculate_overall_similarity(comparison_result)
            comparison_result['analysis_confidence'] = self._calculate_confidence(comparison_result)
            
        except Exception as e:
            comparison_result['error'] = str(e)
            comparison_result['analysis_confidence'] = 0.0
        
        return comparison_result
    
    def _compare_hashes(self, binary1_path: str, binary2_path: str) -> Dict[str, Any]:
        """Compare cryptographic hashes of binary files"""
        hash_comparison = {
            'identical_files': False,
            'hash_matches': {},
            'hash_values': {'binary1': {}, 'binary2': {}}
        }
        
        try:
            for algorithm in self.hash_algorithms:
                hash1 = self._calculate_hash(binary1_path, algorithm)
                hash2 = self._calculate_hash(binary2_path, algorithm)
                
                hash_comparison['hash_values']['binary1'][algorithm] = hash1
                hash_comparison['hash_values']['binary2'][algorithm] = hash2
                hash_comparison['hash_matches'][algorithm] = (hash1 == hash2)
            
            # Files are identical if all hashes match
            hash_comparison['identical_files'] = all(hash_comparison['hash_matches'].values())
            
        except Exception as e:
            hash_comparison['error'] = str(e)
        
        return hash_comparison
    
    def _calculate_hash(self, file_path: str, algorithm: str) -> str:
        """Calculate hash of a file"""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _compare_sizes(self, binary1_path: str, binary2_path: str) -> Dict[str, Any]:
        """Compare file sizes"""
        try:
            size1 = os.path.getsize(binary1_path)
            size2 = os.path.getsize(binary2_path)
            
            size_diff = abs(size1 - size2)
            size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 1.0
            
            return {
                'size1': size1,
                'size2': size2,
                'size_difference': size_diff,
                'size_ratio': size_ratio,
                'similar_size': size_ratio > 0.9
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _compare_metadata(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metadata from analysis contexts"""
        metadata_comparison = {
            'format_match': False,
            'architecture_match': False,
            'import_similarity': 0.0,
            'export_similarity': 0.0,
            'section_similarity': 0.0,
            'metadata_score': 0.0
        }
        
        try:
            # Get metadata from agent results
            agent15_result1 = self._get_agent_result(context1, 15)
            agent15_result2 = self._get_agent_result(context2, 15)
            
            if agent15_result1 and agent15_result2:
                # Compare file formats
                format1 = agent15_result1.get('file_format', 'unknown')
                format2 = agent15_result2.get('file_format', 'unknown')
                metadata_comparison['format_match'] = (format1 == format2)
                
                # Compare architectures
                arch1 = self._extract_architecture(agent15_result1)
                arch2 = self._extract_architecture(agent15_result2)
                metadata_comparison['architecture_match'] = (arch1 == arch2)
                
                # Compare imports/exports
                imports1 = agent15_result1.get('imports', {})
                imports2 = agent15_result2.get('imports', {})
                metadata_comparison['import_similarity'] = self._compare_imports(imports1, imports2)
                
                exports1 = agent15_result1.get('exports', {})
                exports2 = agent15_result2.get('exports', {})
                metadata_comparison['export_similarity'] = self._compare_exports(exports1, exports2)
                
                # Compare sections
                sections1 = agent15_result1.get('sections', {})
                sections2 = agent15_result2.get('sections', {})
                metadata_comparison['section_similarity'] = self._compare_sections(sections1, sections2)
                
                # Calculate overall metadata score
                scores = [
                    1.0 if metadata_comparison['format_match'] else 0.0,
                    1.0 if metadata_comparison['architecture_match'] else 0.0,
                    metadata_comparison['import_similarity'],
                    metadata_comparison['export_similarity'],
                    metadata_comparison['section_similarity']
                ]
                metadata_comparison['metadata_score'] = sum(scores) / len(scores)
        
        except Exception as e:
            metadata_comparison['error'] = str(e)
        
        return metadata_comparison
    
    def _compare_structure(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare structural aspects of binaries"""
        structure_comparison = {
            'function_count_similarity': 0.0,
            'code_structure_similarity': 0.0,
            'entry_point_similarity': 0.0,
            'structure_score': 0.0
        }
        
        try:
            # Compare function counts
            agent4_result1 = self._get_agent_result(context1, 4)  # Basic decompiler
            agent4_result2 = self._get_agent_result(context2, 4)
            
            if agent4_result1 and agent4_result2:
                analysis1 = agent4_result1.get('analysis_summary', {})
                analysis2 = agent4_result2.get('analysis_summary', {})
                
                func_count1 = analysis1.get('total_functions', 0)
                func_count2 = analysis2.get('total_functions', 0)
                
                if max(func_count1, func_count2) > 0:
                    structure_comparison['function_count_similarity'] = min(func_count1, func_count2) / max(func_count1, func_count2)
                
                # Compare code structure
                functions1 = agent4_result1.get('decompiled_functions', {})
                functions2 = agent4_result2.get('decompiled_functions', {})
                structure_comparison['code_structure_similarity'] = self._compare_code_structure(functions1, functions2)
            
            # Compare entry points
            agent15_result1 = self._get_agent_result(context1, 15)
            agent15_result2 = self._get_agent_result(context2, 15)
            
            if agent15_result1 and agent15_result2:
                entry1 = self._extract_entry_point(agent15_result1)
                entry2 = self._extract_entry_point(agent15_result2)
                structure_comparison['entry_point_similarity'] = 1.0 if entry1 == entry2 else 0.0
            
            # Calculate structure score
            scores = [
                structure_comparison['function_count_similarity'],
                structure_comparison['code_structure_similarity'],
                structure_comparison['entry_point_similarity']
            ]
            structure_comparison['structure_score'] = sum(scores) / len(scores)
        
        except Exception as e:
            structure_comparison['error'] = str(e)
        
        return structure_comparison
    
    def _analyze_semantic_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze semantic similarity between binaries"""
        semantic_analysis = {
            'api_usage_similarity': 0.0,
            'behavioral_similarity': 0.0,
            'string_similarity': 0.0,
            'pattern_similarity': 0.0,
            'semantic_score': 0.0
        }
        
        try:
            # Compare API usage patterns
            agent16_result1 = self._get_agent_result(context1, 16)  # Dynamic analysis
            agent16_result2 = self._get_agent_result(context2, 16)
            
            if agent16_result1 and agent16_result2:
                api_trace1 = agent16_result1.get('api_call_tracing', {})
                api_trace2 = agent16_result2.get('api_call_tracing', {})
                semantic_analysis['api_usage_similarity'] = self._compare_api_usage(api_trace1, api_trace2)
                
                behavior1 = agent16_result1.get('runtime_behavior', {})
                behavior2 = agent16_result2.get('runtime_behavior', {})
                semantic_analysis['behavioral_similarity'] = self._compare_behavior(behavior1, behavior2)
            
            # Compare string content
            agent15_result1 = self._get_agent_result(context1, 15)
            agent15_result2 = self._get_agent_result(context2, 15)
            
            if agent15_result1 and agent15_result2:
                strings1 = agent15_result1.get('strings', {})
                strings2 = agent15_result2.get('strings', {})
                semantic_analysis['string_similarity'] = self._compare_strings(strings1, strings2)
            
            # Compare optimization patterns
            agent6_result1 = self._get_agent_result(context1, 6)  # Optimization matcher
            agent6_result2 = self._get_agent_result(context2, 6)
            
            if agent6_result1 and agent6_result2:
                patterns1 = agent6_result1.get('detected_optimizations', [])
                patterns2 = agent6_result2.get('detected_optimizations', [])
                semantic_analysis['pattern_similarity'] = self._compare_optimization_patterns(patterns1, patterns2)
            
            # Calculate semantic score
            scores = [
                semantic_analysis['api_usage_similarity'],
                semantic_analysis['behavioral_similarity'], 
                semantic_analysis['string_similarity'],
                semantic_analysis['pattern_similarity']
            ]
            valid_scores = [s for s in scores if s > 0]
            semantic_analysis['semantic_score'] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        except Exception as e:
            semantic_analysis['error'] = str(e)
        
        return semantic_analysis
    
    def _assess_functional_equivalence(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> Dict[str, Any]:
        """Assess functional equivalence between binaries"""
        equivalence_analysis = {
            'likely_equivalent': False,
            'equivalence_confidence': 0.0,
            'functional_differences': [],
            'equivalence_indicators': [],
            'equivalence_score': 0.0
        }
        
        try:
            indicators = []
            differences = []
            
            # Check if binaries have similar imported functions
            agent15_result1 = self._get_agent_result(context1, 15)
            agent15_result2 = self._get_agent_result(context2, 15)
            
            if agent15_result1 and agent15_result2:
                imports1 = agent15_result1.get('imports', {})
                imports2 = agent15_result2.get('imports', {})
                
                import_similarity = self._compare_imports(imports1, imports2)
                if import_similarity > 0.8:
                    indicators.append('High import similarity suggests similar functionality')
                elif import_similarity < 0.3:
                    differences.append('Significantly different import tables')
                
                # Check exported functions
                exports1 = agent15_result1.get('exports', {})
                exports2 = agent15_result2.get('exports', {})
                
                export_similarity = self._compare_exports(exports1, exports2)
                if export_similarity > 0.8:
                    indicators.append('Similar export tables suggest functional equivalence')
                elif export_similarity < 0.3:
                    differences.append('Different export interfaces')
            
            # Check behavioral patterns
            agent16_result1 = self._get_agent_result(context1, 16)
            agent16_result2 = self._get_agent_result(context2, 16)
            
            if agent16_result1 and agent16_result2:
                behavior1 = agent16_result1.get('runtime_behavior', {})
                behavior2 = agent16_result2.get('runtime_behavior', {})
                
                if behavior1.get('behavioral_patterns') == behavior2.get('behavioral_patterns'):
                    indicators.append('Identical behavioral patterns detected')
                
            # Calculate equivalence score
            equivalence_indicators = len(indicators)
            functional_differences = len(differences)
            
            equivalence_score = max(0.0, (equivalence_indicators - functional_differences) / max(1, equivalence_indicators + functional_differences))
            
            equivalence_analysis.update({
                'equivalence_indicators': indicators,
                'functional_differences': differences,
                'equivalence_score': equivalence_score,
                'likely_equivalent': equivalence_score > 0.7,
                'equivalence_confidence': min(0.9, equivalence_score)
            })
        
        except Exception as e:
            equivalence_analysis['error'] = str(e)
        
        return equivalence_analysis
    
    def _compare_imports(self, imports1: Dict[str, Any], imports2: Dict[str, Any]) -> float:
        """Compare import tables between binaries"""
        try:
            funcs1 = set()
            funcs2 = set()
            
            imported_functions1 = imports1.get('imported_functions', {})
            imported_functions2 = imports2.get('imported_functions', {})
            
            for dll, functions in imported_functions1.items():
                if isinstance(functions, list):
                    funcs1.update(functions)
            
            for dll, functions in imported_functions2.items():
                if isinstance(functions, list):
                    funcs2.update(functions)
            
            if not funcs1 and not funcs2:
                return 1.0
            
            if not funcs1 or not funcs2:
                return 0.0
            
            intersection = len(funcs1 & funcs2)
            union = len(funcs1 | funcs2)
            
            return intersection / union if union > 0 else 0.0
        
        except:
            return 0.0
    
    def _compare_exports(self, exports1: Dict[str, Any], exports2: Dict[str, Any]) -> float:
        """Compare export tables between binaries"""
        try:
            exported_funcs1 = set(exports1.get('exported_functions', []))
            exported_funcs2 = set(exports2.get('exported_functions', []))
            
            if not exported_funcs1 and not exported_funcs2:
                return 1.0
            
            if not exported_funcs1 or not exported_funcs2:
                return 0.0
            
            intersection = len(exported_funcs1 & exported_funcs2)
            union = len(exported_funcs1 | exported_funcs2)
            
            return intersection / union if union > 0 else 0.0
        
        except:
            return 0.0
    
    def _compare_sections(self, sections1: Dict[str, Any], sections2: Dict[str, Any]) -> float:
        """Compare section structure between binaries"""
        try:
            section_names1 = set(sections1.keys())
            section_names2 = set(sections2.keys())
            
            if not section_names1 and not section_names2:
                return 1.0
            
            if not section_names1 or not section_names2:
                return 0.0
            
            intersection = len(section_names1 & section_names2)
            union = len(section_names1 | section_names2)
            
            return intersection / union if union > 0 else 0.0
        
        except:
            return 0.0
    
    def _compare_api_usage(self, api_trace1: Dict[str, Any], api_trace2: Dict[str, Any]) -> float:
        """Compare API usage patterns"""
        try:
            categories1 = api_trace1.get('api_categories', {})
            categories2 = api_trace2.get('api_categories', {})
            
            if not categories1 and not categories2:
                return 1.0
            
            # Compare category distributions
            all_categories = set(categories1.keys()) | set(categories2.keys())
            if not all_categories:
                return 1.0
            
            similarity_sum = 0.0
            for category in all_categories:
                funcs1 = set(categories1.get(category, []))
                funcs2 = set(categories2.get(category, []))
                
                if funcs1 or funcs2:
                    intersection = len(funcs1 & funcs2)
                    union = len(funcs1 | funcs2)
                    category_similarity = intersection / union if union > 0 else 0.0
                    similarity_sum += category_similarity
            
            return similarity_sum / len(all_categories)
        
        except:
            return 0.0
    
    def _compare_behavior(self, behavior1: Dict[str, Any], behavior2: Dict[str, Any]) -> float:
        """Compare behavioral patterns"""
        try:
            patterns1 = set(behavior1.get('behavioral_patterns', []))
            patterns2 = set(behavior2.get('behavioral_patterns', []))
            
            if not patterns1 and not patterns2:
                return 1.0
            
            if not patterns1 or not patterns2:
                return 0.0
            
            intersection = len(patterns1 & patterns2)
            union = len(patterns1 | patterns2)
            
            return intersection / union if union > 0 else 0.0
        
        except:
            return 0.0
    
    def _compare_strings(self, strings1: Dict[str, Any], strings2: Dict[str, Any]) -> float:
        """Compare string content between binaries"""
        try:
            ascii_strings1 = set(strings1.get('ascii_strings', []))
            ascii_strings2 = set(strings2.get('ascii_strings', []))
            
            if not ascii_strings1 and not ascii_strings2:
                return 1.0
            
            if not ascii_strings1 or not ascii_strings2:
                return 0.0
            
            intersection = len(ascii_strings1 & ascii_strings2)
            union = len(ascii_strings1 | ascii_strings2)
            
            return intersection / union if union > 0 else 0.0
        
        except:
            return 0.0
    
    def _compare_optimization_patterns(self, patterns1: List[Dict[str, Any]], patterns2: List[Dict[str, Any]]) -> float:
        """Compare optimization patterns"""
        try:
            types1 = set(pattern.get('type', '') for pattern in patterns1)
            types2 = set(pattern.get('type', '') for pattern in patterns2)
            
            if not types1 and not types2:
                return 1.0
            
            if not types1 or not types2:
                return 0.0
            
            intersection = len(types1 & types2)
            union = len(types1 | types2)
            
            return intersection / union if union > 0 else 0.0
        
        except:
            return 0.0
    
    def _compare_code_structure(self, functions1: Dict[str, Any], functions2: Dict[str, Any]) -> float:
        """Compare code structure between function sets"""
        try:
            # Compare function complexity distributions
            if not functions1 and not functions2:
                return 1.0
            
            if not functions1 or not functions2:
                return 0.0
            
            # Simple structure comparison based on function count similarity
            func_count1 = len(functions1)
            func_count2 = len(functions2)
            
            if func_count1 == 0 and func_count2 == 0:
                return 1.0
            
            return min(func_count1, func_count2) / max(func_count1, func_count2)
        
        except:
            return 0.0
    
    def _calculate_overall_similarity(self, comparison_result: Dict[str, Any]) -> float:
        """Calculate overall similarity score"""
        try:
            scores = []
            weights = []
            
            # Hash comparison (highest weight if files are identical)
            hash_comp = comparison_result.get('hash_comparison', {})
            if hash_comp.get('identical_files', False):
                return 1.0  # Files are identical
            
            # Size comparison
            size_comp = comparison_result.get('size_comparison', {})
            if 'size_ratio' in size_comp:
                scores.append(size_comp['size_ratio'])
                weights.append(0.1)
            
            # Metadata comparison
            metadata_comp = comparison_result.get('metadata_comparison', {})
            if 'metadata_score' in metadata_comp:
                scores.append(metadata_comp['metadata_score'])
                weights.append(0.3)
            
            # Structure comparison
            structure_comp = comparison_result.get('structure_comparison', {})
            if 'structure_score' in structure_comp:
                scores.append(structure_comp['structure_score'])
                weights.append(0.3)
            
            # Semantic similarity
            semantic_comp = comparison_result.get('semantic_similarity', {})
            if 'semantic_score' in semantic_comp:
                scores.append(semantic_comp['semantic_score'])
                weights.append(0.3)
            
            if not scores:
                return 0.0
            
            # Weighted average
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        except:
            return 0.0
    
    def _calculate_confidence(self, comparison_result: Dict[str, Any]) -> float:
        """Calculate confidence in the comparison analysis"""
        try:
            confidence_factors = []
            
            # Hash comparison confidence
            hash_comp = comparison_result.get('hash_comparison', {})
            if 'error' not in hash_comp:
                confidence_factors.append(0.9)
            
            # Metadata comparison confidence
            metadata_comp = comparison_result.get('metadata_comparison', {})
            if 'error' not in metadata_comp and 'metadata_score' in metadata_comp:
                confidence_factors.append(0.8)
            
            # Structure comparison confidence
            structure_comp = comparison_result.get('structure_comparison', {})
            if 'error' not in structure_comp and 'structure_score' in structure_comp:
                confidence_factors.append(0.7)
            
            # Semantic comparison confidence
            semantic_comp = comparison_result.get('semantic_similarity', {})
            if 'error' not in semantic_comp and 'semantic_score' in semantic_comp:
                confidence_factors.append(0.6)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
        
        except:
            return 0.0
    
    def _get_agent_result(self, context: Dict[str, Any], agent_id: int) -> Optional[Dict[str, Any]]:
        """Get result from specific agent"""
        try:
            agent_results = context.get('agent_results', {})
            agent_result = agent_results.get(agent_id)
            if agent_result and hasattr(agent_result, 'data'):
                return agent_result.data
            return None
        except:
            return None
    
    def _extract_architecture(self, agent15_result: Dict[str, Any]) -> str:
        """Extract architecture information from Agent 15 result"""
        try:
            if 'pe_header' in agent15_result:
                return agent15_result['pe_header'].get('machine_type', 'unknown')
            elif 'elf_header' in agent15_result:
                return agent15_result['elf_header'].get('machine', 'unknown')
            elif 'macho_header' in agent15_result:
                return agent15_result['macho_header'].get('architecture', 'unknown')
            return 'unknown'
        except:
            return 'unknown'
    
    def _extract_entry_point(self, agent15_result: Dict[str, Any]) -> int:
        """Extract entry point from Agent 15 result"""
        try:
            if 'pe_header' in agent15_result:
                return agent15_result['pe_header'].get('entry_point', 0)
            elif 'elf_header' in agent15_result:
                return agent15_result['elf_header'].get('entry_point', 0)
            return 0
        except:
            return 0