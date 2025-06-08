"""
Pattern Database with Learning System
Advanced machine learning system that learns from successful reconstructions.
"""

import json
import sqlite3
import logging
import pickle
import hashlib
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import time


class PatternType(Enum):
    """Types of patterns that can be learned"""
    FUNCTION_PATTERN = "function_pattern"
    OPTIMIZATION_PATTERN = "optimization_pattern"
    CODE_STRUCTURE = "code_structure"
    NAMING_CONVENTION = "naming_convention"
    API_USAGE = "api_usage"
    ERROR_HANDLING = "error_handling"
    MEMORY_MANAGEMENT = "memory_management"
    CONTROL_FLOW = "control_flow"


class LearningSource(Enum):
    """Sources of learning data"""
    SUCCESSFUL_RECONSTRUCTION = "successful_reconstruction"
    USER_FEEDBACK = "user_feedback"
    EXTERNAL_CODEBASE = "external_codebase"
    EXPERT_ANNOTATION = "expert_annotation"
    AUTOMATED_ANALYSIS = "automated_analysis"


@dataclass
class LearningExample:
    """A single learning example"""
    id: str
    pattern_type: PatternType
    source: LearningSource
    input_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    confidence: float
    timestamp: float
    success_rate: float
    usage_count: int
    metadata: Dict[str, Any]


@dataclass
class PatternRule:
    """A learned pattern rule"""
    rule_id: str
    pattern_type: PatternType
    conditions: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    confidence: float
    support_count: int
    success_rate: float
    last_used: float
    created: float


class PatternDatabase:
    """Database for storing and retrieving learned patterns"""
    
    def __init__(self, db_path: str = "patterns.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("PatternDatabase")
        self._init_database()
    
    def _init_database(self):
        """Initialize the pattern database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_examples (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    expected_output TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    usage_count INTEGER NOT NULL,
                    metadata TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_rules (
                    rule_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    support_count INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    last_used REAL NOT NULL,
                    created REAL NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_performance (
                    pattern_id TEXT NOT NULL,
                    applied_timestamp REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    execution_time REAL NOT NULL,
                    context TEXT NOT NULL
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON learning_examples(pattern_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON learning_examples(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_type ON pattern_rules(pattern_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_confidence ON pattern_rules(confidence)')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Pattern database initialized at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pattern database: {e}")
    
    def add_example(self, example: LearningExample):
        """Add a learning example to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_examples 
                (id, pattern_type, source, input_data, expected_output, confidence, 
                 timestamp, success_rate, usage_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                example.id,
                example.pattern_type.value,
                example.source.value,
                json.dumps(example.input_data),
                json.dumps(example.expected_output),
                example.confidence,
                example.timestamp,
                example.success_rate,
                example.usage_count,
                json.dumps(example.metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to add learning example: {e}")
    
    def get_examples(self, pattern_type: PatternType = None, source: LearningSource = None, 
                    min_confidence: float = 0.0, limit: int = 100) -> List[LearningExample]:
        """Retrieve learning examples from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM learning_examples WHERE confidence >= ?'
            params = [min_confidence]
            
            if pattern_type:
                query += ' AND pattern_type = ?'
                params.append(pattern_type.value)
            
            if source:
                query += ' AND source = ?'
                params.append(source.value)
            
            query += ' ORDER BY confidence DESC, timestamp DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            examples = []
            for row in rows:
                example = LearningExample(
                    id=row[0],
                    pattern_type=PatternType(row[1]),
                    source=LearningSource(row[2]),
                    input_data=json.loads(row[3]),
                    expected_output=json.loads(row[4]),
                    confidence=row[5],
                    timestamp=row[6],
                    success_rate=row[7],
                    usage_count=row[8],
                    metadata=json.loads(row[9])
                )
                examples.append(example)
            
            conn.close()
            return examples
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve examples: {e}")
            return []
    
    def add_rule(self, rule: PatternRule):
        """Add a pattern rule to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO pattern_rules 
                (rule_id, pattern_type, conditions, actions, confidence, 
                 support_count, success_rate, last_used, created)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.pattern_type.value,
                json.dumps(rule.conditions),
                json.dumps(rule.actions),
                rule.confidence,
                rule.support_count,
                rule.success_rate,
                rule.last_used,
                rule.created
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to add pattern rule: {e}")
    
    def get_rules(self, pattern_type: PatternType = None, min_confidence: float = 0.5, 
                  limit: int = 50) -> List[PatternRule]:
        """Retrieve pattern rules from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = 'SELECT * FROM pattern_rules WHERE confidence >= ?'
            params = [min_confidence]
            
            if pattern_type:
                query += ' AND pattern_type = ?'
                params.append(pattern_type.value)
            
            query += ' ORDER BY confidence DESC, success_rate DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            rules = []
            for row in rows:
                rule = PatternRule(
                    rule_id=row[0],
                    pattern_type=PatternType(row[1]),
                    conditions=json.loads(row[2]),
                    actions=json.loads(row[3]),
                    confidence=row[4],
                    support_count=row[5],
                    success_rate=row[6],
                    last_used=row[7],
                    created=row[8]
                )
                rules.append(rule)
            
            conn.close()
            return rules
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve rules: {e}")
            return []
    
    def record_usage(self, pattern_id: str, success: bool, execution_time: float, context: Dict[str, Any]):
        """Record pattern usage and performance"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO pattern_performance 
                (pattern_id, applied_timestamp, success, execution_time, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                pattern_id,
                time.time(),
                success,
                execution_time,
                json.dumps(context)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to record pattern usage: {e}")


class PatternMiner:
    """Mines patterns from learning examples"""
    
    def __init__(self, database: PatternDatabase):
        self.database = database
        self.logger = logging.getLogger("PatternMiner")
        self.min_support = 3  # Minimum examples to create a rule
        self.min_confidence = 0.6
    
    def mine_patterns(self, pattern_type: PatternType) -> List[PatternRule]:
        """Mine patterns from learning examples"""
        examples = self.database.get_examples(pattern_type=pattern_type)
        
        if len(examples) < self.min_support:
            return []
        
        rules = []
        
        # Group examples by similar input patterns
        pattern_groups = self._group_similar_examples(examples)
        
        for group in pattern_groups:
            if len(group) >= self.min_support:
                rule = self._create_rule_from_group(group, pattern_type)
                if rule and rule.confidence >= self.min_confidence:
                    rules.append(rule)
        
        return rules
    
    def _group_similar_examples(self, examples: List[LearningExample]) -> List[List[LearningExample]]:
        """Group examples with similar input patterns"""
        groups = []
        
        for example in examples:
            placed = False
            
            for group in groups:
                if self._examples_similar(example, group[0]):
                    group.append(example)
                    placed = True
                    break
            
            if not placed:
                groups.append([example])
        
        return groups
    
    def _examples_similar(self, example1: LearningExample, example2: LearningExample) -> bool:
        """Check if two examples have similar patterns"""
        # Simple similarity based on common keys and values
        input1 = example1.input_data
        input2 = example2.input_data
        
        common_keys = set(input1.keys()) & set(input2.keys())
        if len(common_keys) < max(1, min(len(input1), len(input2)) * 0.5):
            return False
        
        # Check value similarity for common keys
        similar_values = 0
        for key in common_keys:
            if str(input1[key]).lower() == str(input2[key]).lower():
                similar_values += 1
        
        similarity = similar_values / len(common_keys)
        return similarity >= 0.7
    
    def _create_rule_from_group(self, group: List[LearningExample], pattern_type: PatternType) -> Optional[PatternRule]:
        """Create a rule from a group of similar examples"""
        if not group:
            return None
        
        # Extract common conditions
        conditions = self._extract_conditions(group)
        
        # Extract common actions
        actions = self._extract_actions(group)
        
        # Calculate metrics
        confidence = np.mean([ex.confidence for ex in group])
        success_rate = np.mean([ex.success_rate for ex in group])
        support_count = len(group)
        
        rule_id = hashlib.md5(f"{pattern_type.value}_{str(conditions)}_{str(actions)}".encode()).hexdigest()
        
        return PatternRule(
            rule_id=rule_id,
            pattern_type=pattern_type,
            conditions=conditions,
            actions=actions,
            confidence=confidence,
            support_count=support_count,
            success_rate=success_rate,
            last_used=0.0,
            created=time.time()
        )
    
    def _extract_conditions(self, group: List[LearningExample]) -> List[Dict[str, Any]]:
        """Extract common conditions from example group"""
        conditions = []
        
        # Find common input patterns
        all_inputs = [ex.input_data for ex in group]
        common_keys = set.intersection(*[set(inp.keys()) for inp in all_inputs])
        
        for key in common_keys:
            values = [inp[key] for inp in all_inputs]
            
            # If all values are the same, create an equality condition
            if len(set(str(v) for v in values)) == 1:
                conditions.append({
                    'type': 'equals',
                    'field': key,
                    'value': values[0]
                })
            # If values are similar patterns, create a pattern condition
            elif self._is_pattern_field(key, values):
                pattern = self._extract_pattern(values)
                if pattern:
                    conditions.append({
                        'type': 'matches_pattern',
                        'field': key,
                        'pattern': pattern
                    })
        
        return conditions
    
    def _extract_actions(self, group: List[LearningExample]) -> List[Dict[str, Any]]:
        """Extract common actions from example group"""
        actions = []
        
        # Find common output patterns
        all_outputs = [ex.expected_output for ex in group]
        common_keys = set.intersection(*[set(out.keys()) for out in all_outputs])
        
        for key in common_keys:
            values = [out[key] for out in all_outputs]
            
            # Create action based on output pattern
            if len(set(str(v) for v in values)) == 1:
                actions.append({
                    'type': 'set_value',
                    'field': key,
                    'value': values[0]
                })
            else:
                # Multiple values - could be a template or choice
                if self._is_template_field(key, values):
                    template = self._extract_template(values)
                    actions.append({
                        'type': 'apply_template',
                        'field': key,
                        'template': template
                    })
                else:
                    actions.append({
                        'type': 'choose_from',
                        'field': key,
                        'options': list(set(values))
                    })
        
        return actions
    
    def _is_pattern_field(self, field: str, values: List[Any]) -> bool:
        """Check if field contains pattern-like data"""
        pattern_fields = ['instruction', 'function_name', 'variable_name', 'code_pattern']
        return field in pattern_fields and len(values) > 1
    
    def _is_template_field(self, field: str, values: List[Any]) -> bool:
        """Check if field contains template-like data"""
        template_fields = ['suggested_name', 'c_code', 'comment']
        return field in template_fields
    
    def _extract_pattern(self, values: List[Any]) -> Optional[str]:
        """Extract a regex pattern from similar values"""
        # Simple pattern extraction - could be improved
        str_values = [str(v) for v in values]
        
        # Look for common prefixes/suffixes
        if len(str_values) >= 2:
            # Find common prefix
            prefix = ""
            for i in range(min(len(v) for v in str_values)):
                chars = set(v[i] for v in str_values)
                if len(chars) == 1:
                    prefix += list(chars)[0]
                else:
                    break
            
            if len(prefix) >= 2:
                return f"^{prefix}.*"
        
        return None
    
    def _extract_template(self, values: List[Any]) -> str:
        """Extract a template from similar values"""
        # Simple template extraction
        str_values = [str(v) for v in values]
        
        if len(str_values) >= 2:
            # Find common parts and create template
            template = str_values[0]
            
            # Replace varying parts with placeholders
            for i, val in enumerate(str_values[1:], 1):
                # This is a simplified template extraction
                # In practice, would need more sophisticated analysis
                pass
        
        return str_values[0] if str_values else "{value}"


class LearningSystem:
    """Main learning system that coordinates pattern discovery and application"""
    
    def __init__(self, db_path: str = "learning.db"):
        self.database = PatternDatabase(db_path)
        self.miner = PatternMiner(self.database)
        self.logger = logging.getLogger("LearningSystem")
        self.learned_rules = {}
        self._load_existing_rules()
    
    def _load_existing_rules(self):
        """Load existing rules from database"""
        for pattern_type in PatternType:
            rules = self.database.get_rules(pattern_type=pattern_type)
            self.learned_rules[pattern_type] = rules
        
        total_rules = sum(len(rules) for rules in self.learned_rules.values())
        self.logger.info(f"Loaded {total_rules} existing rules")
    
    def learn_from_success(self, reconstruction_data: Dict[str, Any]):
        """Learn from a successful reconstruction"""
        try:
            # Extract learning examples from reconstruction data
            examples = self._extract_learning_examples(reconstruction_data)
            
            # Add examples to database
            for example in examples:
                self.database.add_example(example)
            
            # Mine new patterns
            self._mine_new_patterns()
            
            self.logger.info(f"Learned from reconstruction with {len(examples)} examples")
            
        except Exception as e:
            self.logger.error(f"Failed to learn from success: {e}")
    
    def _extract_learning_examples(self, reconstruction_data: Dict[str, Any]) -> List[LearningExample]:
        """Extract learning examples from reconstruction data"""
        examples = []
        timestamp = time.time()
        
        # Function naming examples
        if 'function_analysis' in reconstruction_data:
            for func_data in reconstruction_data['function_analysis']:
                if 'original_name' in func_data and 'suggested_name' in func_data:
                    example_id = hashlib.md5(f"func_{func_data['original_name']}_{timestamp}".encode()).hexdigest()
                    
                    example = LearningExample(
                        id=example_id,
                        pattern_type=PatternType.NAMING_CONVENTION,
                        source=LearningSource.SUCCESSFUL_RECONSTRUCTION,
                        input_data={
                            'function_characteristics': func_data.get('characteristics', {}),
                            'original_name': func_data['original_name']
                        },
                        expected_output={
                            'suggested_name': func_data['suggested_name'],
                            'confidence': func_data.get('confidence', 0.8)
                        },
                        confidence=func_data.get('confidence', 0.8),
                        timestamp=timestamp,
                        success_rate=1.0,
                        usage_count=1,
                        metadata={
                            'function_type': func_data.get('type', 'unknown'),
                            'complexity': func_data.get('complexity', 'unknown')
                        }
                    )
                    examples.append(example)
        
        # Code pattern examples
        if 'pattern_analysis' in reconstruction_data:
            patterns = reconstruction_data['pattern_analysis']
            for pattern in patterns:
                example_id = hashlib.md5(f"pattern_{pattern.get('type', 'unknown')}_{timestamp}".encode()).hexdigest()
                
                example = LearningExample(
                    id=example_id,
                    pattern_type=PatternType.CODE_STRUCTURE,
                    source=LearningSource.SUCCESSFUL_RECONSTRUCTION,
                    input_data={
                        'assembly_pattern': pattern.get('assembly', ''),
                        'context': pattern.get('context', {})
                    },
                    expected_output={
                        'c_equivalent': pattern.get('c_equivalent', ''),
                        'pattern_type': pattern.get('type', 'unknown')
                    },
                    confidence=pattern.get('confidence', 0.7),
                    timestamp=timestamp,
                    success_rate=1.0,
                    usage_count=1,
                    metadata={
                        'complexity': pattern.get('complexity', 'unknown'),
                        'optimization_level': pattern.get('optimization_level', 'unknown')
                    }
                )
                examples.append(example)
        
        return examples
    
    def _mine_new_patterns(self):
        """Mine new patterns from recent examples"""
        for pattern_type in PatternType:
            new_rules = self.miner.mine_patterns(pattern_type)
            
            # Add new rules to database and memory
            for rule in new_rules:
                self.database.add_rule(rule)
                
                if pattern_type not in self.learned_rules:
                    self.learned_rules[pattern_type] = []
                
                # Replace rule if it exists, otherwise add
                existing_rule_index = None
                for i, existing_rule in enumerate(self.learned_rules[pattern_type]):
                    if existing_rule.rule_id == rule.rule_id:
                        existing_rule_index = i
                        break
                
                if existing_rule_index is not None:
                    self.learned_rules[pattern_type][existing_rule_index] = rule
                else:
                    self.learned_rules[pattern_type].append(rule)
            
            if new_rules:
                self.logger.info(f"Mined {len(new_rules)} new rules for {pattern_type.value}")
    
    def apply_learned_patterns(self, analysis_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned patterns to improve analysis"""
        improvements = {
            'applied_rules': [],
            'suggestions': [],
            'confidence_boosts': {},
            'new_insights': []
        }
        
        try:
            # Apply function naming patterns
            if 'function_analysis' in analysis_data:
                improvements.update(self._apply_naming_patterns(analysis_data['function_analysis']))
            
            # Apply code structure patterns
            if 'code_patterns' in analysis_data:
                improvements.update(self._apply_structure_patterns(analysis_data['code_patterns']))
            
            # Apply optimization patterns
            if 'optimization_analysis' in analysis_data:
                improvements.update(self._apply_optimization_patterns(analysis_data['optimization_analysis']))
            
        except Exception as e:
            self.logger.error(f"Failed to apply learned patterns: {e}")
            improvements['error'] = str(e)
        
        return improvements
    
    def _apply_naming_patterns(self, function_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply learned naming patterns"""
        improvements = {'naming_improvements': []}
        
        naming_rules = self.learned_rules.get(PatternType.NAMING_CONVENTION, [])
        
        for func_data in function_analysis:
            for rule in naming_rules:
                if self._rule_matches(rule, func_data):
                    # Apply rule actions
                    for action in rule.actions:
                        if action['type'] == 'set_value' and action['field'] == 'suggested_name':
                            improvements['naming_improvements'].append({
                                'function': func_data.get('name', 'unknown'),
                                'learned_suggestion': action['value'],
                                'rule_confidence': rule.confidence,
                                'rule_id': rule.rule_id
                            })
        
        return improvements
    
    def _apply_structure_patterns(self, code_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply learned code structure patterns"""
        improvements = {'structure_improvements': []}
        
        structure_rules = self.learned_rules.get(PatternType.CODE_STRUCTURE, [])
        
        for pattern in code_patterns:
            for rule in structure_rules:
                if self._rule_matches(rule, pattern):
                    # Apply rule to improve pattern recognition
                    for action in rule.actions:
                        if action['type'] == 'set_value':
                            improvements['structure_improvements'].append({
                                'pattern_type': pattern.get('type', 'unknown'),
                                'learned_enhancement': action['value'],
                                'rule_confidence': rule.confidence
                            })
        
        return improvements
    
    def _apply_optimization_patterns(self, optimization_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned optimization patterns"""
        improvements = {'optimization_improvements': []}
        
        optimization_rules = self.learned_rules.get(PatternType.OPTIMIZATION_PATTERN, [])
        
        for rule in optimization_rules:
            if self._rule_matches(rule, optimization_analysis):
                improvements['optimization_improvements'].append({
                    'detected_optimization': rule.actions,
                    'confidence': rule.confidence
                })
        
        return improvements
    
    def _rule_matches(self, rule: PatternRule, data: Dict[str, Any]) -> bool:
        """Check if a rule matches the given data"""
        for condition in rule.conditions:
            if condition['type'] == 'equals':
                field = condition['field']
                expected_value = condition['value']
                
                if field not in data or str(data[field]).lower() != str(expected_value).lower():
                    return False
            
            elif condition['type'] == 'matches_pattern':
                field = condition['field']
                pattern = condition['pattern']
                
                if field not in data:
                    return False
                
                import re
                if not re.search(pattern, str(data[field]), re.IGNORECASE):
                    return False
        
        return True
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about the learning system"""
        stats = {
            'total_rules': sum(len(rules) for rules in self.learned_rules.values()),
            'rules_by_type': {ptype.value: len(rules) for ptype, rules in self.learned_rules.items()},
            'total_examples': 0,
            'examples_by_type': {},
            'top_performing_rules': []
        }
        
        # Get example counts
        for pattern_type in PatternType:
            examples = self.database.get_examples(pattern_type=pattern_type, limit=1000)
            stats['examples_by_type'][pattern_type.value] = len(examples)
            stats['total_examples'] += len(examples)
        
        # Get top performing rules
        all_rules = []
        for rules in self.learned_rules.values():
            all_rules.extend(rules)
        
        all_rules.sort(key=lambda r: r.confidence * r.success_rate, reverse=True)
        stats['top_performing_rules'] = [
            {
                'rule_id': rule.rule_id,
                'pattern_type': rule.pattern_type.value,
                'confidence': rule.confidence,
                'success_rate': rule.success_rate,
                'support_count': rule.support_count
            }
            for rule in all_rules[:10]
        ]
        
        return stats
    
    def export_knowledge(self, export_path: str):
        """Export learned knowledge to file"""
        try:
            knowledge = {
                'rules': {},
                'statistics': self.get_learning_stats(),
                'export_timestamp': time.time()
            }
            
            for pattern_type, rules in self.learned_rules.items():
                knowledge['rules'][pattern_type.value] = [asdict(rule) for rule in rules]
            
            with open(export_path, 'w') as f:
                json.dump(knowledge, f, indent=2, default=str)
            
            self.logger.info(f"Knowledge exported to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export knowledge: {e}")
    
    def import_knowledge(self, import_path: str):
        """Import learned knowledge from file"""
        try:
            with open(import_path, 'r') as f:
                knowledge = json.load(f)
            
            # Import rules
            for pattern_type_str, rule_dicts in knowledge['rules'].items():
                pattern_type = PatternType(pattern_type_str)
                
                for rule_dict in rule_dicts:
                    rule = PatternRule(**rule_dict)
                    self.database.add_rule(rule)
            
            # Reload rules from database
            self._load_existing_rules()
            
            self.logger.info(f"Knowledge imported from {import_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import knowledge: {e}")