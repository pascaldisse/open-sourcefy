"""
Graceful Degradation Manager for Matrix Pipeline
Allows pipeline to continue with reduced functionality when components fail.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)

class ComponentStatus(Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded" 
    UNAVAILABLE = "unavailable"

class GracefulDegradationManager:
    """Manages graceful degradation of pipeline components"""
    
    def __init__(self):
        self.component_status: Dict[str, ComponentStatus] = {}
        self.fallback_strategies: Dict[str, str] = {}
        self.essential_components: Set[str] = {"binary_analysis", "basic_decompilation"}
        
    def register_component_failure(self, component: str, error: Exception) -> bool:
        """Register component failure and determine if pipeline can continue"""
        
        self.component_status[component] = ComponentStatus.UNAVAILABLE
        logger.warning(f"Component {component} failed: {error}")
        
        # Check if we can degrade gracefully
        if component in self.essential_components:
            logger.critical(f"Essential component {component} failed - pipeline cannot continue")
            return False
        
        # Set up fallback strategy
        fallback = self._get_fallback_strategy(component)
        if fallback:
            self.fallback_strategies[component] = fallback
            self.component_status[component] = ComponentStatus.DEGRADED
            logger.info(f"Using fallback strategy for {component}: {fallback}")
        
        return True
    
    def _get_fallback_strategy(self, component: str) -> Optional[str]:
        """Get fallback strategy for failed component"""
        
        fallback_map = {
            "ghidra_analysis": "basic_disassembly",
            "ai_enhancement": "pattern_matching",
            "advanced_decompilation": "basic_decompilation",
            "quality_analysis": "basic_validation"
        }
        
        return fallback_map.get(component)
    
    def can_continue_pipeline(self) -> bool:
        """Check if pipeline can continue with current component status"""
        
        failed_essential = [comp for comp in self.essential_components 
                          if self.component_status.get(comp) == ComponentStatus.UNAVAILABLE]
        
        return len(failed_essential) == 0
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get degradation status report"""
        
        return {
            "component_status": {k: v.value for k, v in self.component_status.items()},
            "fallback_strategies": self.fallback_strategies,
            "pipeline_viable": self.can_continue_pipeline(),
            "degraded_components": [k for k, v in self.component_status.items() 
                                  if v == ComponentStatus.DEGRADED]
        }
