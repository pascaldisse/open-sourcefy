"""
Agent Factory and Registry System for Open-Sourcefy Matrix Pipeline
Dynamic loading, registration, and management of Matrix agents
"""

import importlib
import importlib.util
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time
from collections import defaultdict

from .matrix_agent_base import MatrixAgentBase, AgentType, AgentResult, AgentStatus
from .config_manager import get_config_manager
from .shared_utils import DataValidator, ErrorHandler, PerformanceMonitor


class AgentLoadingStrategy(Enum):
    """Agent loading strategies"""
    LAZY = "lazy"           # Load on first use
    EAGER = "eager"         # Load all at startup
    ON_DEMAND = "on_demand" # Load when explicitly requested


class AgentRegistrationError(Exception):
    """Agent registration related errors"""
    pass


class AgentLoadingError(Exception):
    """Agent loading related errors"""
    pass


@dataclass
class AgentMetadata:
    """Metadata for a registered agent"""
    agent_id: int
    agent_name: str
    agent_class_name: str
    module_path: str
    matrix_character: str
    agent_type: AgentType
    dependencies: List[int] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    description: str = ""
    is_loaded: bool = False
    load_time: Optional[float] = None
    instance: Optional[MatrixAgentBase] = None
    registration_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'agent_id': self.agent_id,
            'agent_name': self.agent_name,
            'agent_class_name': self.agent_class_name,
            'module_path': self.module_path,
            'matrix_character': self.matrix_character,
            'agent_type': self.agent_type.value,
            'dependencies': self.dependencies,
            'prerequisites': self.prerequisites,
            'version': self.version,
            'description': self.description,
            'is_loaded': self.is_loaded,
            'load_time': self.load_time,
            'registration_time': self.registration_time
        }


@dataclass
class AgentExecutionPlan:
    """Execution plan for agents with dependency resolution"""
    execution_batches: List[List[int]] = field(default_factory=list)
    agent_dependencies: Dict[int, List[int]] = field(default_factory=dict)
    dependency_graph: Dict[int, Set[int]] = field(default_factory=dict)
    total_agents: int = 0
    parallel_potential: int = 1
    estimated_duration: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'execution_batches': self.execution_batches,
            'agent_dependencies': self.agent_dependencies,
            'dependency_graph': {k: list(v) for k, v in self.dependency_graph.items()},
            'total_agents': self.total_agents,
            'parallel_potential': self.parallel_potential,
            'estimated_duration': self.estimated_duration
        }


class AgentRegistry:
    """Registry for managing Matrix agents"""
    
    def __init__(self):
        self.logger = logging.getLogger("AgentRegistry")
        self._agents: Dict[int, AgentMetadata] = {}
        self._name_to_id: Dict[str, int] = {}
        self._type_groups: Dict[AgentType, List[int]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler(self.logger)
    
    def register_agent(self, agent_id: int, agent_class: Type[MatrixAgentBase], 
                      module_path: str, **kwargs) -> bool:
        """Register an agent class"""
        with self._lock:
            try:
                # Validate agent ID
                agent_id = DataValidator.validate_agent_id(agent_id)
                
                # Check if already registered
                if agent_id in self._agents:
                    self.logger.warning(f"Agent {agent_id} already registered, updating...")
                
                # Validate agent class
                if not issubclass(agent_class, MatrixAgentBase):
                    raise AgentRegistrationError(f"Agent class must inherit from MatrixAgentBase")
                
                # Create instance to extract metadata
                try:
                    temp_instance = agent_class()
                    dependencies = temp_instance.get_dependencies()
                    prerequisites = temp_instance.get_prerequisites() if hasattr(temp_instance, 'get_prerequisites') else []
                    description = temp_instance.get_description()
                except Exception as e:
                    self.logger.warning(f"Could not instantiate agent {agent_id} for metadata: {e}")
                    dependencies = []
                    prerequisites = []
                    description = ""
                
                # Create metadata
                metadata = AgentMetadata(
                    agent_id=agent_id,
                    agent_name=agent_class.__name__,
                    agent_class_name=agent_class.__name__,
                    module_path=module_path,
                    matrix_character=getattr(temp_instance, 'matrix_character', f"Agent{agent_id}"),
                    agent_type=getattr(temp_instance, 'agent_type', AgentType.PROGRAM),
                    dependencies=dependencies,
                    prerequisites=prerequisites,
                    description=description,
                    version=kwargs.get('version', '1.0.0')
                )
                
                # Store metadata
                self._agents[agent_id] = metadata
                self._name_to_id[metadata.agent_name] = agent_id
                self._type_groups[metadata.agent_type].append(agent_id)
                
                self.logger.info(f"✅ Registered agent {agent_id}: {metadata.agent_name} ({metadata.matrix_character})")
                return True
                
            except Exception as e:
                error_msg = f"Failed to register agent {agent_id}: {e}"
                self.error_handler.handle_error(e, f"register_agent({agent_id})")
                raise AgentRegistrationError(error_msg)
    
    def unregister_agent(self, agent_id: int):
        """Unregister an agent"""
        with self._lock:
            if agent_id not in self._agents:
                self.logger.warning(f"Agent {agent_id} not registered")
                return
            
            metadata = self._agents[agent_id]
            
            # Remove from all indexes
            del self._agents[agent_id]
            if metadata.agent_name in self._name_to_id:
                del self._name_to_id[metadata.agent_name]
            
            if agent_id in self._type_groups[metadata.agent_type]:
                self._type_groups[metadata.agent_type].remove(agent_id)
            
            self.logger.info(f"Unregistered agent {agent_id}: {metadata.agent_name}")
    
    def is_registered(self, agent_id: int) -> bool:
        """Check if agent is registered"""
        return agent_id in self._agents
    
    def get_agent_metadata(self, agent_id: int) -> Optional[AgentMetadata]:
        """Get agent metadata"""
        return self._agents.get(agent_id)
    
    def get_agent_by_name(self, agent_name: str) -> Optional[AgentMetadata]:
        """Get agent by name"""
        agent_id = self._name_to_id.get(agent_name)
        return self._agents.get(agent_id) if agent_id else None
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentMetadata]:
        """Get all agents of a specific type"""
        agent_ids = self._type_groups.get(agent_type, [])
        return [self._agents[agent_id] for agent_id in agent_ids]
    
    def list_agents(self) -> List[AgentMetadata]:
        """List all registered agents"""
        return list(self._agents.values())
    
    def list_agent_ids(self) -> List[int]:
        """List all registered agent IDs"""
        return sorted(self._agents.keys())
    
    def create_execution_plan(self, agent_ids: Optional[List[int]] = None) -> AgentExecutionPlan:
        """Create execution plan with dependency resolution"""
        with self._lock:
            if agent_ids is None:
                agent_ids = self.list_agent_ids()
            
            # Validate all agents are registered
            for agent_id in agent_ids:
                if agent_id not in self._agents:
                    raise AgentLoadingError(f"Agent {agent_id} not registered")
            
            # Build dependency graph
            dependency_graph = {}
            agent_dependencies = {}
            
            for agent_id in agent_ids:
                metadata = self._agents[agent_id]
                dependencies = [dep for dep in metadata.dependencies if dep in agent_ids]
                dependency_graph[agent_id] = set(dependencies)
                agent_dependencies[agent_id] = dependencies
            
            # Topological sort to create execution batches
            execution_batches = self._topological_sort_batches(dependency_graph)
            
            # Calculate execution metrics
            total_agents = len(agent_ids)
            parallel_potential = max(len(batch) for batch in execution_batches) if execution_batches else 1
            estimated_duration = self._estimate_execution_duration(agent_ids)
            
            return AgentExecutionPlan(
                execution_batches=execution_batches,
                agent_dependencies=agent_dependencies,
                dependency_graph=dependency_graph,
                total_agents=total_agents,
                parallel_potential=parallel_potential,
                estimated_duration=estimated_duration
            )
    
    def _topological_sort_batches(self, dependency_graph: Dict[int, Set[int]]) -> List[List[int]]:
        """Perform topological sort and group into execution batches"""
        in_degree = {node: 0 for node in dependency_graph}
        
        # Calculate in-degrees
        for node in dependency_graph:
            for dependency in dependency_graph[node]:
                if dependency in in_degree:
                    in_degree[dependency] += 1
        
        batches = []
        remaining_nodes = set(dependency_graph.keys())
        
        while remaining_nodes:
            # Find nodes with no dependencies (in-degree 0)
            current_batch = [node for node in remaining_nodes if in_degree[node] == 0]
            
            if not current_batch:
                # Circular dependency detected
                self.logger.error(f"Circular dependency detected in agents: {remaining_nodes}")
                # Add remaining nodes as final batch to prevent infinite loop
                batches.append(list(remaining_nodes))
                break
            
            batches.append(current_batch)
            
            # Remove current batch nodes and update in-degrees
            for node in current_batch:
                remaining_nodes.remove(node)
                for dependent in dependency_graph[node]:
                    if dependent in in_degree:
                        in_degree[dependent] -= 1
        
        return batches
    
    def _estimate_execution_duration(self, agent_ids: List[int]) -> float:
        """Estimate total execution duration"""
        # Base duration estimates (in seconds)
        base_durations = {
            AgentType.MACHINE: 45,
            AgentType.PROGRAM: 120,
            AgentType.FRAGMENT: 90,
            AgentType.AI_COMMANDER: 180,
            AgentType.COLLECTIVE_AI: 60
        }
        
        total_duration = 0.0
        
        for agent_id in agent_ids:
            metadata = self._agents.get(agent_id)
            if metadata:
                base_duration = base_durations.get(metadata.agent_type, 60)
                total_duration += base_duration
        
        return total_duration
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock:
            type_counts = {agent_type.value: len(agent_ids) 
                          for agent_type, agent_ids in self._type_groups.items()}
            
            loaded_count = sum(1 for metadata in self._agents.values() if metadata.is_loaded)
            
            return {
                'total_agents': len(self._agents),
                'loaded_agents': loaded_count,
                'unloaded_agents': len(self._agents) - loaded_count,
                'agents_by_type': type_counts,
                'agent_ids': sorted(self._agents.keys()),
                'performance_metrics': self.performance_monitor.get_metrics_summary()
            }


class AgentFactory:
    """Factory for loading and instantiating Matrix agents"""
    
    def __init__(self, loading_strategy: AgentLoadingStrategy = AgentLoadingStrategy.LAZY):
        self.logger = logging.getLogger("AgentFactory")
        self.loading_strategy = loading_strategy
        self.config_manager = get_config_manager()
        
        # Registry and caching
        self.registry = AgentRegistry()
        self._loaded_classes: Dict[int, Type[MatrixAgentBase]] = {}
        self._instances: Dict[int, MatrixAgentBase] = {}
        self._lock = threading.RLock()
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.error_handler = ErrorHandler(self.logger)
        
        # Auto-discovery settings
        self.agents_package = "src.core.agents"
        self.agents_directory = Path(__file__).parent / "agents"
    
    def auto_discover_agents(self) -> int:
        """Auto-discover and register agents from the agents package"""
        discovered_count = 0
        
        try:
            self.logger.info("🔍 Auto-discovering Matrix agents...")
            
            # Discover from package
            if self.agents_directory.exists():
                discovered_count += self._discover_from_directory(self.agents_directory)
            
            # Try importing from package
            try:
                discovered_count += self._discover_from_package(self.agents_package)
            except ImportError as e:
                self.logger.warning(f"Could not import agents package {self.agents_package}: {e}")
            
            self.logger.info(f"✅ Discovered {discovered_count} Matrix agents")
            
            # Eager loading if configured
            if self.loading_strategy == AgentLoadingStrategy.EAGER:
                self._load_all_agents()
            
            return discovered_count
            
        except Exception as e:
            self.error_handler.handle_error(e, "auto_discover_agents")
            return discovered_count
    
    def _discover_from_directory(self, directory: Path) -> int:
        """Discover agents from directory"""
        discovered_count = 0
        
        for agent_file in directory.glob("agent*.py"):
            if agent_file.name.startswith("__"):
                continue
            
            try:
                agent_id = self._extract_agent_id_from_filename(agent_file.name)
                if agent_id is not None:
                    module_path = f"{self.agents_package}.{agent_file.stem}"
                    if self._register_agent_from_module(agent_id, module_path):
                        discovered_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to discover agent from {agent_file}: {e}")
        
        return discovered_count
    
    def _discover_from_package(self, package_name: str) -> int:
        """Discover agents from package"""
        discovered_count = 0
        
        try:
            package = importlib.import_module(package_name)
            
            for _, module_name, _ in pkgutil.iter_modules(package.__path__, package_name + "."):
                if "agent" in module_name:
                    try:
                        agent_id = self._extract_agent_id_from_module_name(module_name)
                        if agent_id is not None:
                            if self._register_agent_from_module(agent_id, module_name):
                                discovered_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to discover agent from module {module_name}: {e}")
        
        except Exception as e:
            self.logger.warning(f"Failed to discover from package {package_name}: {e}")
        
        return discovered_count
    
    def _extract_agent_id_from_filename(self, filename: str) -> Optional[int]:
        """Extract agent ID from filename like agent01_binary_discovery.py"""
        import re
        match = re.match(r'agent(\d+)', filename)
        return int(match.group(1)) if match else None
    
    def _extract_agent_id_from_module_name(self, module_name: str) -> Optional[int]:
        """Extract agent ID from module name"""
        import re
        match = re.search(r'agent(\d+)', module_name)
        return int(match.group(1)) if match else None
    
    def _register_agent_from_module(self, agent_id: int, module_path: str) -> bool:
        """Register agent from module path"""
        try:
            module = importlib.import_module(module_path)
            
            # Look for agent classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, MatrixAgentBase) and 
                    obj != MatrixAgentBase and 
                    hasattr(obj, '__module__') and 
                    obj.__module__ == module_path):
                    
                    self.registry.register_agent(agent_id, obj, module_path)
                    return True
            
            self.logger.warning(f"No valid agent class found in module {module_path}")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id} from {module_path}: {e}")
            return False
    
    def load_agent(self, agent_id: int) -> Optional[MatrixAgentBase]:
        """Load and instantiate an agent"""
        with self._lock:
            metrics = self.performance_monitor.start_operation(f"load_agent_{agent_id}")
            
            try:
                # Check if already loaded
                if agent_id in self._instances:
                    metadata = self.registry.get_agent_metadata(agent_id)
                    if metadata:
                        metadata.is_loaded = True
                    return self._instances[agent_id]
                
                # Get metadata
                metadata = self.registry.get_agent_metadata(agent_id)
                if not metadata:
                    raise AgentLoadingError(f"Agent {agent_id} not registered")
                
                # Load class if not cached
                if agent_id not in self._loaded_classes:
                    agent_class = self._load_agent_class(metadata.module_path)
                    self._loaded_classes[agent_id] = agent_class
                else:
                    agent_class = self._loaded_classes[agent_id]
                
                # Create instance
                instance = agent_class()
                self._instances[agent_id] = instance
                
                # Update metadata
                metadata.is_loaded = True
                metadata.load_time = time.time()
                metadata.instance = instance
                
                self.logger.info(f"✅ Loaded agent {agent_id}: {metadata.agent_name}")
                return instance
                
            except Exception as e:
                self.error_handler.handle_error(e, f"load_agent({agent_id})")
                raise AgentLoadingError(f"Failed to load agent {agent_id}: {e}")
            
            finally:
                self.performance_monitor.end_operation(metrics)
    
    def _load_agent_class(self, module_path: str) -> Type[MatrixAgentBase]:
        """Load agent class from module"""
        try:
            module = importlib.import_module(module_path)
            
            # Look for agent class
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, MatrixAgentBase) and 
                    obj != MatrixAgentBase and 
                    hasattr(obj, '__module__') and 
                    obj.__module__ == module_path):
                    return obj
            
            raise AgentLoadingError(f"No valid agent class found in {module_path}")
            
        except Exception as e:
            raise AgentLoadingError(f"Failed to load module {module_path}: {e}")
    
    def _load_all_agents(self):
        """Load all registered agents (for eager loading)"""
        agent_ids = self.registry.list_agent_ids()
        
        self.logger.info(f"🚀 Eager loading {len(agent_ids)} agents...")
        
        loaded_count = 0
        for agent_id in agent_ids:
            try:
                self.load_agent(agent_id)
                loaded_count += 1
            except Exception as e:
                self.logger.error(f"Failed to eager load agent {agent_id}: {e}")
        
        self.logger.info(f"✅ Eager loaded {loaded_count}/{len(agent_ids)} agents")
    
    def get_agent(self, agent_id: int) -> Optional[MatrixAgentBase]:
        """Get agent instance (load if necessary)"""
        if self.loading_strategy == AgentLoadingStrategy.LAZY or agent_id not in self._instances:
            return self.load_agent(agent_id)
        
        return self._instances.get(agent_id)
    
    def unload_agent(self, agent_id: int):
        """Unload agent instance"""
        with self._lock:
            if agent_id in self._instances:
                del self._instances[agent_id]
            
            if agent_id in self._loaded_classes:
                del self._loaded_classes[agent_id]
            
            metadata = self.registry.get_agent_metadata(agent_id)
            if metadata:
                metadata.is_loaded = False
                metadata.instance = None
            
            self.logger.info(f"Unloaded agent {agent_id}")
    
    def create_execution_plan(self, agent_ids: Optional[List[int]] = None) -> AgentExecutionPlan:
        """Create execution plan with dependency resolution"""
        return self.registry.create_execution_plan(agent_ids)
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics"""
        registry_stats = self.registry.get_registry_stats()
        
        return {
            'loading_strategy': self.loading_strategy.value,
            'loaded_classes': len(self._loaded_classes),
            'active_instances': len(self._instances),
            'registry_stats': registry_stats,
            'performance_metrics': self.performance_monitor.get_metrics_summary(),
            'error_summary': self.error_handler.get_error_summary()
        }


# Global factory instance
_agent_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get global agent factory instance"""
    global _agent_factory
    if _agent_factory is None:
        loading_strategy = AgentLoadingStrategy.LAZY  # Default strategy
        _agent_factory = AgentFactory(loading_strategy)
    return _agent_factory


def register_agent(agent_id: int, agent_class: Type[MatrixAgentBase], module_path: str = "", **kwargs):
    """Convenience function to register an agent"""
    factory = get_agent_factory()
    if not module_path:
        module_path = agent_class.__module__
    return factory.registry.register_agent(agent_id, agent_class, module_path, **kwargs)


def get_agent(agent_id: int) -> Optional[MatrixAgentBase]:
    """Convenience function to get an agent instance"""
    factory = get_agent_factory()
    return factory.get_agent(agent_id)


def auto_discover_agents() -> int:
    """Convenience function to auto-discover agents"""
    factory = get_agent_factory()
    return factory.auto_discover_agents()


def create_execution_plan(agent_ids: Optional[List[int]] = None) -> AgentExecutionPlan:
    """Convenience function to create execution plan"""
    factory = get_agent_factory()
    return factory.create_execution_plan(agent_ids)