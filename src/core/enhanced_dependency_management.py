"""
Phase 2 Enhancement: Enhanced Dependency Management System
Advanced dependency resolution, conditional execution, and intelligent optimization.
"""

import logging
import time
from typing import Dict, List, Set, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from .agent_base import BaseAgent, AgentResult, AgentStatus, AGENT_DEPENDENCIES


class DependencyType(Enum):
    """Types of dependencies between agents"""
    HARD = "hard"           # Must complete successfully
    SOFT = "soft"           # Can fail but should run
    CONDITIONAL = "conditional"  # Run based on conditions
    OPTIONAL = "optional"   # Skip if previous failed
    DATA_FLOW = "data_flow" # Requires specific data


class ExecutionCondition(Enum):
    """Conditions for conditional execution"""
    BINARY_TYPE = "binary_type"
    FILE_SIZE = "file_size"
    ARCHITECTURE = "architecture"
    SUCCESS_THRESHOLD = "success_threshold"
    CUSTOM = "custom"


@dataclass
class DependencyRule:
    """Enhanced dependency rule with conditions and constraints"""
    dependent_agent: int
    required_agent: int
    dependency_type: DependencyType = DependencyType.HARD
    condition: Optional[ExecutionCondition] = None
    condition_value: Any = None
    condition_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    required_data_keys: List[str] = field(default_factory=list)
    timeout_override: Optional[float] = None
    retry_if_failed: bool = True


@dataclass
class ExecutionPlan:
    """Optimized execution plan with dependency resolution"""
    batches: List[List[int]]
    skipped_agents: List[int]
    conditional_agents: Dict[int, DependencyRule]
    execution_order: List[int]
    estimated_time: float
    critical_path: List[int]


class EnhancedDependencyResolver:
    """Advanced dependency resolver with intelligent optimization"""
    
    def __init__(self):
        self.dependency_rules: Dict[Tuple[int, int], DependencyRule] = {}
        self.logger = logging.getLogger("EnhancedDependencyResolver")
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default dependency rules with enhanced constraints"""
        # Convert basic dependencies to enhanced rules
        for agent_id, deps in AGENT_DEPENDENCIES.items():
            for dep_id in deps:
                rule = DependencyRule(
                    dependent_agent=agent_id,
                    required_agent=dep_id,
                    dependency_type=DependencyType.HARD
                )
                self.add_dependency_rule(rule)
        
        # Add enhanced conditional dependencies
        self._add_conditional_dependencies()
        self._add_optimization_dependencies()
    
    def _add_conditional_dependencies(self) -> None:
        """Add conditional dependencies based on binary characteristics"""
        
        # Agent 3 (Smart Error Pattern Matching) - only for complex binaries
        self.add_dependency_rule(DependencyRule(
            dependent_agent=3,
            required_agent=1,
            dependency_type=DependencyType.CONDITIONAL,
            condition=ExecutionCondition.FILE_SIZE,
            condition_value=1024 * 1024,  # 1MB threshold
            condition_func=lambda ctx: ctx.get('binary_size', 0) > 1024 * 1024
        ))
        
        # Agent 9 (Advanced Assembly Analyzer) - only for x86/x64
        self.add_dependency_rule(DependencyRule(
            dependent_agent=9,
            required_agent=7,
            dependency_type=DependencyType.CONDITIONAL,
            condition=ExecutionCondition.ARCHITECTURE,
            condition_value=['x86', 'x64', 'x86_64'],
            condition_func=lambda ctx: ctx.get('architecture', '').lower() in ['x86', 'x64', 'x86_64']
        ))
        
        # Agent 10 (Resource Reconstructor) - only if previous agents found resources
        self.add_dependency_rule(DependencyRule(
            dependent_agent=10,
            required_agent=8,
            dependency_type=DependencyType.CONDITIONAL,
            condition=ExecutionCondition.CUSTOM,
            condition_func=lambda ctx: self._has_resource_data(ctx),
            required_data_keys=['resources', 'binary_structure']
        ))
    
    def _add_optimization_dependencies(self) -> None:
        """Add optimization-based dependencies"""
        
        # Soft dependencies for performance optimization
        self.add_dependency_rule(DependencyRule(
            dependent_agent=6,
            required_agent=3,
            dependency_type=DependencyType.SOFT,
            required_data_keys=['error_patterns']
        ))
        
        # Optional dependencies that can be skipped
        self.add_dependency_rule(DependencyRule(
            dependent_agent=11,
            required_agent=10,
            dependency_type=DependencyType.OPTIONAL,
            required_data_keys=['reconstructed_resources']
        ))
    
    def _has_resource_data(self, context: Dict[str, Any]) -> bool:
        """Check if context has resource data for reconstruction"""
        agent_results = context.get('agent_results', {})
        
        # Check if previous agents found any resources
        for agent_id in [5, 8, 9]:  # Structure, Diff, Assembly analyzers
            result = agent_results.get(agent_id)
            if result and result.status == AgentStatus.COMPLETED:
                data = result.data
                if any(key in data for key in ['resources', 'structures', 'assembly_info']):
                    return True
        
        return False
    
    def add_dependency_rule(self, rule: DependencyRule) -> None:
        """Add a dependency rule"""
        key = (rule.dependent_agent, rule.required_agent)
        self.dependency_rules[key] = rule
        self.logger.debug(f"Added dependency rule: Agent {rule.dependent_agent} -> Agent {rule.required_agent} ({rule.dependency_type.value})")
    
    def remove_dependency_rule(self, dependent_agent: int, required_agent: int) -> None:
        """Remove a dependency rule"""
        key = (dependent_agent, required_agent)
        if key in self.dependency_rules:
            del self.dependency_rules[key]
            self.logger.debug(f"Removed dependency rule: Agent {dependent_agent} -> Agent {required_agent}")
    
    def resolve_dependencies(self, target_agents: List[int], 
                           context: Dict[str, Any]) -> ExecutionPlan:
        """Resolve dependencies and create optimized execution plan"""
        
        self.logger.info(f"Resolving dependencies for agents: {target_agents}")
        
        # Evaluate conditions and filter agents
        executable_agents, skipped_agents, conditional_agents = self._evaluate_conditions(
            target_agents, context
        )
        
        # Build dependency graph for executable agents
        dependency_graph = self._build_dependency_graph(executable_agents)
        
        # Detect and resolve circular dependencies
        if self._has_circular_dependencies(dependency_graph):
            dependency_graph = self._resolve_circular_dependencies(dependency_graph)
        
        # Calculate execution batches
        batches = self._calculate_optimal_batches(dependency_graph, context)
        
        # Calculate execution order and critical path
        execution_order = [agent for batch in batches for agent in batch]
        critical_path = self._find_critical_path(dependency_graph, context)
        
        # Estimate total execution time
        estimated_time = self._estimate_execution_time(batches, context)
        
        plan = ExecutionPlan(
            batches=batches,
            skipped_agents=skipped_agents,
            conditional_agents=conditional_agents,
            execution_order=execution_order,
            estimated_time=estimated_time,
            critical_path=critical_path
        )
        
        self.logger.info(f"Execution plan: {len(batches)} batches, {len(skipped_agents)} skipped, estimated time: {estimated_time:.1f}s")
        
        return plan
    
    def _evaluate_conditions(self, target_agents: List[int], 
                           context: Dict[str, Any]) -> Tuple[List[int], List[int], Dict[int, DependencyRule]]:
        """Evaluate conditions and determine which agents should execute"""
        
        executable_agents = []
        skipped_agents = []
        conditional_agents = {}
        
        for agent_id in target_agents:
            should_execute = True
            conditional_rule = None
            
            # Check all dependency rules for this agent
            for (dependent_agent, required_agent), rule in self.dependency_rules.items():
                if dependent_agent == agent_id and rule.condition:
                    # Evaluate condition
                    if rule.condition_func and not rule.condition_func(context):
                        should_execute = False
                        conditional_rule = rule
                        break
                    elif rule.condition == ExecutionCondition.BINARY_TYPE:
                        binary_type = context.get('binary_type', '')
                        if binary_type not in rule.condition_value:
                            should_execute = False
                            conditional_rule = rule
                            break
                    elif rule.condition == ExecutionCondition.FILE_SIZE:
                        file_size = context.get('binary_size', 0)
                        if file_size < rule.condition_value:
                            should_execute = False
                            conditional_rule = rule
                            break
                    elif rule.condition == ExecutionCondition.ARCHITECTURE:
                        arch = context.get('architecture', '').lower()
                        if arch not in [a.lower() for a in rule.condition_value]:
                            should_execute = False
                            conditional_rule = rule
                            break
            
            if should_execute:
                executable_agents.append(agent_id)
            else:
                skipped_agents.append(agent_id)
                if conditional_rule:
                    conditional_agents[agent_id] = conditional_rule
        
        return executable_agents, skipped_agents, conditional_agents
    
    def _build_dependency_graph(self, agents: List[int]) -> Dict[int, Set[int]]:
        """Build dependency graph for given agents"""
        graph = defaultdict(set)
        
        for agent_id in agents:
            graph[agent_id] = set()
            
            # Add dependencies from rules
            for (dependent_agent, required_agent), rule in self.dependency_rules.items():
                if dependent_agent == agent_id and required_agent in agents:
                    if rule.dependency_type in [DependencyType.HARD, DependencyType.SOFT, DependencyType.DATA_FLOW]:
                        graph[agent_id].add(required_agent)
        
        return dict(graph)
    
    def _has_circular_dependencies(self, graph: Dict[int, Set[int]]) -> bool:
        """Check for circular dependencies using DFS"""
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node: WHITE for node in graph}
        
        def dfs(node):
            if colors[node] == GRAY:
                return True  # Back edge found - circular dependency
            if colors[node] == BLACK:
                return False
            
            colors[node] = GRAY
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            colors[node] = BLACK
            return False
        
        for node in graph:
            if colors[node] == WHITE:
                if dfs(node):
                    return True
        
        return False
    
    def _resolve_circular_dependencies(self, graph: Dict[int, Set[int]]) -> Dict[int, Set[int]]:
        """Resolve circular dependencies by converting some to soft dependencies"""
        self.logger.warning("Circular dependencies detected, attempting resolution...")
        
        # Find strongly connected components
        sccs = self._find_strongly_connected_components(graph)
        
        for scc in sccs:
            if len(scc) > 1:  # Circular dependency within this component
                self.logger.warning(f"Circular dependency in agents: {scc}")
                
                # Break the cycle by removing the "weakest" dependency
                # (Remove dependency with highest agent ID to lower agent ID)
                for agent_id in sorted(scc, reverse=True):
                    deps_to_remove = []
                    for dep in graph[agent_id]:
                        if dep in scc and dep < agent_id:
                            deps_to_remove.append(dep)
                    
                    if deps_to_remove:
                        # Remove one dependency to break the cycle
                        removed_dep = deps_to_remove[0]
                        graph[agent_id].remove(removed_dep)
                        self.logger.warning(f"Removed dependency: Agent {agent_id} -> Agent {removed_dep}")
                        break
        
        return graph
    
    def _find_strongly_connected_components(self, graph: Dict[int, Set[int]]) -> List[List[int]]:
        """Find strongly connected components using Tarjan's algorithm"""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        index_map = {}
        sccs = []
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            for successor in graph[node]:
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif on_stack[successor]:
                    lowlinks[node] = min(lowlinks[node], index[successor])
            
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                sccs.append(component)
        
        for node in graph:
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def _calculate_optimal_batches(self, graph: Dict[int, Set[int]], 
                                 context: Dict[str, Any]) -> List[List[int]]:
        """Calculate optimal execution batches using topological sort with optimization"""
        
        # Calculate in-degrees
        in_degree = defaultdict(int)
        for node in graph:
            in_degree[node] = 0
        
        for node in graph:
            for dep in graph[node]:
                in_degree[node] += 1
        
        # Topological sort with batching
        batches = []
        remaining_nodes = set(graph.keys())
        
        while remaining_nodes:
            # Find all nodes with no dependencies (in-degree 0)
            ready_nodes = [node for node in remaining_nodes if in_degree[node] == 0]
            
            if not ready_nodes:
                # Should not happen if no circular dependencies
                self.logger.error("No ready nodes found - possible unresolved circular dependency")
                ready_nodes = list(remaining_nodes)
            
            # Optimize batch by grouping similar agents
            optimized_batch = self._optimize_batch(ready_nodes, context)
            batches.append(optimized_batch)
            
            # Remove processed nodes and update in-degrees
            for node in optimized_batch:
                remaining_nodes.remove(node)
                
                # Update in-degrees for dependent nodes
                for other_node in remaining_nodes:
                    if node in graph[other_node]:
                        in_degree[other_node] -= 1
        
        return batches
    
    def _optimize_batch(self, ready_agents: List[int], context: Dict[str, Any]) -> List[int]:
        """Optimize batch order for better performance"""
        if len(ready_agents) <= 1:
            return ready_agents
        
        # Group agents by type/complexity for better resource utilization
        analysis_agents = [1, 2, 3, 5, 6, 8, 9]  # Analysis-heavy agents
        decompilation_agents = [4, 7]             # Decompilation agents
        reconstruction_agents = [10, 11, 12]      # Reconstruction agents
        validation_agents = [13]                  # Validation agents
        
        # Sort within each group
        def agent_priority(agent_id):
            if agent_id in analysis_agents:
                return (0, agent_id)  # Analysis first
            elif agent_id in decompilation_agents:
                return (1, agent_id)  # Decompilation second
            elif agent_id in reconstruction_agents:
                return (2, agent_id)  # Reconstruction third
            else:
                return (3, agent_id)  # Validation last
        
        return sorted(ready_agents, key=agent_priority)
    
    def _find_critical_path(self, graph: Dict[int, Set[int]], context: Dict[str, Any]) -> List[int]:
        """Find the critical path through the dependency graph"""
        
        # Estimate execution times for each agent
        def estimate_agent_time(agent_id):
            # Base estimates (in seconds)
            time_estimates = {
                1: 5,   # Binary Discovery
                2: 10,  # Architecture Analysis
                3: 15,  # Smart Error Pattern Matching
                4: 30,  # Basic Decompiler
                5: 20,  # Binary Structure Analyzer
                6: 25,  # Optimization Matcher
                7: 60,  # Advanced Decompiler (Ghidra)
                8: 40,  # Binary Diff Analyzer
                9: 35,  # Advanced Assembly Analyzer
                10: 20, # Resource Reconstructor
                11: 25, # Global Reconstructor
                12: 30, # Compilation Orchestrator
                13: 10  # Final Validator
            }
            return time_estimates.get(agent_id, 30)
        
        # Calculate longest path (critical path)
        longest_path = {}
        path_predecessors = {}
        
        # Topologically sort nodes
        topo_order = []
        visited = set()
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for dep in graph[node]:
                dfs(dep)
            topo_order.append(node)
        
        for node in graph:
            dfs(node)
        
        # Calculate longest paths
        for node in reversed(topo_order):
            node_time = estimate_agent_time(node)
            
            if not graph[node]:  # No dependencies
                longest_path[node] = node_time
                path_predecessors[node] = None
            else:
                max_dep_time = 0
                best_predecessor = None
                
                for dep in graph[node]:
                    dep_total_time = longest_path.get(dep, 0)
                    if dep_total_time > max_dep_time:
                        max_dep_time = dep_total_time
                        best_predecessor = dep
                
                longest_path[node] = node_time + max_dep_time
                path_predecessors[node] = best_predecessor
        
        # Find the node with maximum total time (end of critical path)
        if not longest_path:
            return []
        
        critical_end = max(longest_path.keys(), key=lambda x: longest_path[x])
        
        # Reconstruct critical path
        critical_path = []
        current = critical_end
        
        while current is not None:
            critical_path.append(current)
            current = path_predecessors[current]
        
        critical_path.reverse()
        return critical_path
    
    def _estimate_execution_time(self, batches: List[List[int]], context: Dict[str, Any]) -> float:
        """Estimate total execution time for all batches"""
        
        def estimate_agent_time(agent_id):
            # Use same estimates as critical path
            time_estimates = {
                1: 5, 2: 10, 3: 15, 4: 30, 5: 20, 6: 25, 7: 60,
                8: 40, 9: 35, 10: 20, 11: 25, 12: 30, 13: 10
            }
            return time_estimates.get(agent_id, 30)
        
        total_time = 0
        
        for batch in batches:
            if not batch:
                continue
            
            # Time for batch is maximum time of agents in parallel
            batch_time = max(estimate_agent_time(agent_id) for agent_id in batch)
            total_time += batch_time
        
        return total_time
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """Validate the dependency configuration"""
        errors = []
        
        # Check for self-dependencies
        for (dependent, required), rule in self.dependency_rules.items():
            if dependent == required:
                errors.append(f"Agent {dependent} has self-dependency")
        
        # Check for invalid agent IDs
        valid_agents = set(range(1, 14))
        for (dependent, required), rule in self.dependency_rules.items():
            if dependent not in valid_agents:
                errors.append(f"Invalid dependent agent ID: {dependent}")
            if required not in valid_agents:
                errors.append(f"Invalid required agent ID: {required}")
        
        # Check for circular dependencies
        all_agents = set()
        for (dependent, required), rule in self.dependency_rules.items():
            all_agents.add(dependent)
            all_agents.add(required)
        
        if all_agents:
            graph = self._build_dependency_graph(list(all_agents))
            if self._has_circular_dependencies(graph):
                errors.append("Circular dependencies detected in configuration")
        
        return len(errors) == 0, errors
    
    def get_dependency_report(self) -> Dict[str, Any]:
        """Generate comprehensive dependency report"""
        
        is_valid, validation_errors = self.validate_dependencies()
        
        # Count dependencies by type
        type_counts = defaultdict(int)
        for rule in self.dependency_rules.values():
            type_counts[rule.dependency_type.value] += 1
        
        # Find agents with most dependencies
        dependent_counts = defaultdict(int)
        required_counts = defaultdict(int)
        
        for (dependent, required), rule in self.dependency_rules.items():
            dependent_counts[dependent] += 1
            required_counts[required] += 1
        
        most_dependent = max(dependent_counts.items(), key=lambda x: x[1]) if dependent_counts else (None, 0)
        most_required = max(required_counts.items(), key=lambda x: x[1]) if required_counts else (None, 0)
        
        return {
            'total_rules': len(self.dependency_rules),
            'validation': {
                'is_valid': is_valid,
                'errors': validation_errors
            },
            'type_distribution': dict(type_counts),
            'statistics': {
                'most_dependent_agent': most_dependent[0],
                'max_dependencies': most_dependent[1],
                'most_required_agent': most_required[0],
                'max_requirements': most_required[1]
            },
            'conditional_rules': sum(1 for rule in self.dependency_rules.values() if rule.condition),
            'data_flow_rules': sum(1 for rule in self.dependency_rules.values() 
                                 if rule.dependency_type == DependencyType.DATA_FLOW)
        }


# Global instance for easy access
dependency_resolver = EnhancedDependencyResolver()


def get_execution_plan(target_agents: List[int], context: Dict[str, Any]) -> ExecutionPlan:
    """Convenience function to get execution plan"""
    return dependency_resolver.resolve_dependencies(target_agents, context)


def validate_agent_dependencies() -> Tuple[bool, List[str]]:
    """Convenience function to validate dependencies"""
    return dependency_resolver.validate_dependencies()