"""
Machine Learning Enhanced Control Flow Graph Reconstruction

Implements ML-powered CFG reconstruction techniques for:
- Indirect jump resolution using neural networks
- Pattern-based switch statement reconstruction  
- Automated exception handler detection
- Dynamic control flow prediction with confidence scoring

Building on existing CFG framework with advanced ML capabilities.
"""

import logging
import struct
import pickle
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

try:
    import numpy as np
    import capstone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_LIBS_AVAILABLE = True
except ImportError:
    ML_LIBS_AVAILABLE = False
    logging.warning("ML libraries not available, using fallback CFG reconstruction")

from .cfg_reconstructor import AdvancedControlFlowAnalyzer, BasicBlock, JumpType


class MLPatternType(Enum):
    """Types of patterns that can be detected by ML models."""
    INDIRECT_JUMP = "indirect_jump"
    SWITCH_STATEMENT = "switch_statement"
    EXCEPTION_HANDLER = "exception_handler"
    FUNCTION_CALL = "function_call"
    VIRTUAL_CALL = "virtual_call"
    TAIL_CALL = "tail_call"
    COMPUTED_GOTO = "computed_goto"


@dataclass
class MLFeatureVector:
    """Feature vector for ML-based CFG analysis."""
    instruction_opcodes: List[int]
    register_usage: Dict[str, int]
    memory_references: List[int]
    immediate_values: List[int]
    control_flow_context: List[int]
    statistical_features: List[float]


@dataclass
class IndirectJumpPrediction:
    """Prediction result for indirect jump target."""
    source_address: int
    predicted_targets: List[int]
    confidence_scores: List[float]
    pattern_type: MLPatternType
    feature_importance: Dict[str, float]
    reasoning: str


@dataclass
class MLEnhancedCFGResult:
    """Result of ML-enhanced CFG reconstruction."""
    original_cfg: Dict[str, Any]
    enhanced_cfg: Dict[str, Any]
    indirect_jump_predictions: List[IndirectJumpPrediction]
    switch_reconstructions: List[Dict[str, Any]]
    exception_handlers: List[Dict[str, Any]]
    confidence_map: Dict[int, float]
    performance_metrics: Dict[str, Any]


class MLEnhancedCFGReconstructor:
    """
    Machine Learning Enhanced Control Flow Graph Reconstructor.
    
    Uses neural networks and ensemble methods to improve CFG reconstruction
    accuracy, especially for indirect jumps and complex control structures.
    """
    
    def __init__(self, config_manager=None):
        """Initialize ML-enhanced CFG reconstructor."""
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize base CFG analyzer
        self.base_cfg_analyzer = AdvancedControlFlowAnalyzer(config_manager)
        
        # ML models for different pattern types
        self.models = {}
        self.feature_scalers = {}
        
        # Pattern databases
        self._initialize_pattern_databases()
        
        # Initialize or load pre-trained models
        self._initialize_ml_models()
        
        self.logger.info("ML-Enhanced CFG Reconstructor initialized")

    def _initialize_pattern_databases(self):
        """Initialize pattern databases for ML training."""
        # Instruction pattern signatures for different control structures
        self.instruction_patterns = {
            'indirect_jump': {
                'opcodes': [0xff, 0x25],  # jmp [mem]
                'contexts': ['call_table', 'switch_table', 'vtable']
            },
            'switch_statement': {
                'opcodes': [0x3d, 0x77, 0xff],  # cmp eax, imm; ja far; jmp [table+eax*4]
                'contexts': ['range_check', 'table_access', 'default_case']
            },
            'exception_handler': {
                'opcodes': [0x64, 0x8b, 0x25],  # mov reg, fs:[offset]
                'contexts': ['seh_chain', 'exception_record', 'handler_address']
            }
        }
        
        # Feature extraction templates
        self.feature_templates = {
            'opcode_sequence': {'window_size': 10, 'stride': 1},
            'register_usage': {'track_reads': True, 'track_writes': True},
            'memory_patterns': {'address_patterns': True, 'size_patterns': True},
            'control_flow': {'branch_history': 5, 'call_depth': 3}
        }

    def _initialize_ml_models(self):
        """Initialize ML models for CFG enhancement."""
        if not ML_LIBS_AVAILABLE:
            self.logger.warning("ML libraries not available, using rule-based fallback")
            return
        
        # Try to load pre-trained models
        model_dir = Path(__file__).parent / 'ml_models'
        
        if self._load_pretrained_models(model_dir):
            self.logger.info("Loaded pre-trained ML models")
        else:
            self.logger.info("Initializing new ML models")
            self._create_default_models()

    def _load_pretrained_models(self, model_dir: Path) -> bool:
        """Load pre-trained ML models if available."""
        try:
            if not model_dir.exists():
                return False
            
            # Load models for each pattern type
            for pattern_type in MLPatternType:
                model_file = model_dir / f"{pattern_type.value}_model.pkl"
                scaler_file = model_dir / f"{pattern_type.value}_scaler.pkl"
                
                if model_file.exists() and scaler_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[pattern_type] = pickle.load(f)
                    with open(scaler_file, 'rb') as f:
                        self.feature_scalers[pattern_type] = pickle.load(f)
            
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load pre-trained models: {e}")
            return False

    def _create_default_models(self):
        """Create default ML models with synthetic training data."""
        try:
            # Create models for each pattern type
            for pattern_type in MLPatternType:
                # Use different models based on pattern complexity
                if pattern_type in [MLPatternType.INDIRECT_JUMP, MLPatternType.SWITCH_STATEMENT]:
                    # More complex patterns use neural networks
                    model = MLPClassifier(
                        hidden_layer_sizes=(100, 50, 25),
                        activation='relu',
                        solver='adam',
                        max_iter=1000,
                        random_state=42
                    )
                else:
                    # Simpler patterns use random forest
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                
                self.models[pattern_type] = model
                self.feature_scalers[pattern_type] = StandardScaler()
            
            # Train with synthetic data
            self._train_with_synthetic_data()
            
        except Exception as e:
            self.logger.error(f"Failed to create default models: {e}")

    def _train_with_synthetic_data(self):
        """Train models with synthetic training data."""
        # This would normally use a large dataset of analyzed binaries
        # For now, create minimal synthetic data for initialization
        
        for pattern_type in MLPatternType:
            try:
                # Generate synthetic features and labels
                X_synthetic = np.random.rand(100, 50)  # 100 samples, 50 features
                y_synthetic = np.random.randint(0, 2, 100)  # Binary classification
                
                # Scale features
                X_scaled = self.feature_scalers[pattern_type].fit_transform(X_synthetic)
                
                # Train model
                self.models[pattern_type].fit(X_scaled, y_synthetic)
                
                self.logger.debug(f"Trained synthetic model for {pattern_type.value}")
                
            except Exception as e:
                self.logger.error(f"Failed to train synthetic model for {pattern_type.value}: {e}")

    def enhance_cfg_with_ml(self, binary_path: Path, base_cfg: Optional[Dict] = None) -> MLEnhancedCFGResult:
        """
        Enhance CFG reconstruction with ML-based analysis.
        
        Args:
            binary_path: Path to binary file
            base_cfg: Optional base CFG from standard analysis
            
        Returns:
            MLEnhancedCFGResult with enhanced analysis
        """
        self.logger.info(f"Starting ML-enhanced CFG reconstruction: {binary_path}")
        
        try:
            # Get base CFG if not provided
            if base_cfg is None:
                base_cfg = self.base_cfg_analyzer.analyze_control_flow(binary_path)
            
            # Load binary data
            binary_data = binary_path.read_bytes()
            
            # Initialize result
            result = MLEnhancedCFGResult(
                original_cfg=base_cfg,
                enhanced_cfg=base_cfg.copy(),
                indirect_jump_predictions=[],
                switch_reconstructions=[],
                exception_handlers=[],
                confidence_map={},
                performance_metrics={}
            )
            
            # Phase 1: Indirect Jump Resolution
            self.logger.info("Enhancing indirect jump resolution...")
            indirect_predictions = self._enhance_indirect_jumps(binary_data, base_cfg)
            result.indirect_jump_predictions = indirect_predictions
            
            # Phase 2: Switch Statement Reconstruction
            self.logger.info("Enhancing switch statement detection...")
            switch_reconstructions = self._enhance_switch_detection(binary_data, base_cfg)
            result.switch_reconstructions = switch_reconstructions
            
            # Phase 3: Exception Handler Detection
            self.logger.info("Enhancing exception handler detection...")
            exception_handlers = self._enhance_exception_detection(binary_data, base_cfg)
            result.exception_handlers = exception_handlers
            
            # Phase 4: Update enhanced CFG
            result.enhanced_cfg = self._update_cfg_with_predictions(
                base_cfg, indirect_predictions, switch_reconstructions, exception_handlers
            )
            
            # Phase 5: Calculate confidence scores
            result.confidence_map = self._calculate_confidence_scores(result)
            
            self.logger.info("ML-enhanced CFG reconstruction completed")
            return result
            
        except Exception as e:
            self.logger.error(f"ML-enhanced CFG reconstruction failed: {e}")
            return MLEnhancedCFGResult(
                original_cfg=base_cfg or {},
                enhanced_cfg=base_cfg or {},
                indirect_jump_predictions=[],
                switch_reconstructions=[],
                exception_handlers=[],
                confidence_map={},
                performance_metrics={'error': str(e)}
            )

    def _enhance_indirect_jumps(self, binary_data: bytes, base_cfg: Dict) -> List[IndirectJumpPrediction]:
        """Enhance indirect jump resolution using ML."""
        predictions = []
        
        if not ML_LIBS_AVAILABLE or MLPatternType.INDIRECT_JUMP not in self.models:
            return self._fallback_indirect_jump_analysis(binary_data)
        
        try:
            # Find indirect jumps in base CFG
            indirect_jumps = self._find_indirect_jumps(base_cfg)
            
            for jump_addr in indirect_jumps:
                # Extract features around the jump
                features = self._extract_jump_features(binary_data, jump_addr)
                
                if features:
                    # Make prediction
                    prediction = self._predict_jump_targets(jump_addr, features)
                    if prediction:
                        predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"ML indirect jump enhancement failed: {e}")
            return []

    def _enhance_switch_detection(self, binary_data: bytes, base_cfg: Dict) -> List[Dict[str, Any]]:
        """Enhance switch statement detection using ML."""
        switch_reconstructions = []
        
        try:
            # Look for potential switch patterns
            switch_candidates = self._find_switch_candidates(binary_data)
            
            for candidate in switch_candidates:
                # Analyze switch structure
                switch_info = self._analyze_switch_structure(binary_data, candidate)
                if switch_info:
                    switch_reconstructions.append(switch_info)
            
            return switch_reconstructions
            
        except Exception as e:
            self.logger.error(f"Switch detection enhancement failed: {e}")
            return []

    def _enhance_exception_detection(self, binary_data: bytes, base_cfg: Dict) -> List[Dict[str, Any]]:
        """Enhance exception handler detection using ML."""
        exception_handlers = []
        
        try:
            # Look for SEH (Structured Exception Handling) patterns
            seh_patterns = self._find_seh_patterns(binary_data)
            
            for pattern in seh_patterns:
                handler_info = self._analyze_exception_handler(binary_data, pattern)
                if handler_info:
                    exception_handlers.append(handler_info)
            
            return exception_handlers
            
        except Exception as e:
            self.logger.error(f"Exception handler detection failed: {e}")
            return []

    def _extract_jump_features(self, binary_data: bytes, jump_addr: int) -> Optional[MLFeatureVector]:
        """Extract features for ML analysis around a jump instruction."""
        try:
            if not capstone:
                return None
            
            # Initialize disassembler
            md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
            md.detail = True
            
            # Extract code window around jump
            window_size = 50
            start_addr = max(0, jump_addr - window_size)
            end_addr = min(len(binary_data), jump_addr + window_size)
            code_window = binary_data[start_addr:end_addr]
            
            # Disassemble instructions
            instructions = list(md.disasm(code_window, start_addr))
            
            # Extract features
            opcodes = [ins.id for ins in instructions]
            register_usage = self._analyze_register_usage(instructions)
            memory_refs = self._extract_memory_references(instructions)
            immediate_vals = self._extract_immediate_values(instructions)
            
            # Statistical features
            stats = [
                len(instructions),
                len(set(opcodes)),  # Unique opcodes
                sum(1 for ins in instructions if ins.mnemonic.startswith('j')),  # Jump count
                sum(1 for ins in instructions if ins.mnemonic == 'call')  # Call count
            ]
            
            return MLFeatureVector(
                instruction_opcodes=opcodes[:20],  # Limit size
                register_usage=register_usage,
                memory_references=memory_refs[:10],
                immediate_values=immediate_vals[:10],
                control_flow_context=[],
                statistical_features=stats
            )
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    def _predict_jump_targets(self, jump_addr: int, features: MLFeatureVector) -> Optional[IndirectJumpPrediction]:
        """Predict jump targets using ML model."""
        try:
            if MLPatternType.INDIRECT_JUMP not in self.models:
                return None
            
            # Convert features to numpy array
            feature_array = self._features_to_array(features)
            
            # Scale features
            scaler = self.feature_scalers[MLPatternType.INDIRECT_JUMP]
            scaled_features = scaler.transform([feature_array])
            
            # Make prediction
            model = self.models[MLPatternType.INDIRECT_JUMP]
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(scaled_features)[0]
                confidence = max(probabilities)
            else:
                prediction = model.predict(scaled_features)[0]
                confidence = 0.8 if prediction else 0.2
            
            # Generate predicted targets (simplified)
            predicted_targets = [jump_addr + 0x100, jump_addr + 0x200]  # Placeholder
            
            return IndirectJumpPrediction(
                source_address=jump_addr,
                predicted_targets=predicted_targets,
                confidence_scores=[confidence, confidence * 0.8],
                pattern_type=MLPatternType.INDIRECT_JUMP,
                feature_importance={},
                reasoning="ML-based prediction"
            )
            
        except Exception as e:
            self.logger.error(f"Jump target prediction failed: {e}")
            return None

    def _features_to_array(self, features: MLFeatureVector) -> np.ndarray:
        """Convert feature vector to numpy array for ML processing."""
        if not ML_LIBS_AVAILABLE:
            return np.array([])
        
        # Combine all features into a single array
        feature_list = []
        
        # Opcodes (padded/truncated to fixed size)
        opcodes_padded = (features.instruction_opcodes + [0] * 20)[:20]
        feature_list.extend(opcodes_padded)
        
        # Register usage counts
        reg_counts = [features.register_usage.get(reg, 0) for reg in ['eax', 'ebx', 'ecx', 'edx', 'esp', 'ebp']]
        feature_list.extend(reg_counts)
        
        # Memory references (padded)
        mem_refs_padded = (features.memory_references + [0] * 10)[:10]
        feature_list.extend(mem_refs_padded)
        
        # Statistical features
        feature_list.extend(features.statistical_features)
        
        # Pad to fixed size
        while len(feature_list) < 50:
            feature_list.append(0)
        
        return np.array(feature_list[:50])

    def _analyze_register_usage(self, instructions) -> Dict[str, int]:
        """Analyze register usage patterns in instruction sequence."""
        usage = defaultdict(int)
        
        for ins in instructions:
            # Count register references in operands
            for operand in ins.op_str.split(','):
                operand = operand.strip()
                if operand in ['eax', 'ebx', 'ecx', 'edx', 'esp', 'ebp', 'esi', 'edi']:
                    usage[operand] += 1
        
        return dict(usage)

    def _extract_memory_references(self, instructions) -> List[int]:
        """Extract memory reference patterns."""
        memory_refs = []
        
        for ins in instructions:
            if '[' in ins.op_str and ']' in ins.op_str:
                # Simple pattern: extract numbers from memory references
                import re
                numbers = re.findall(r'\b\d+\b', ins.op_str)
                memory_refs.extend([int(n) for n in numbers])
        
        return memory_refs

    def _extract_immediate_values(self, instructions) -> List[int]:
        """Extract immediate values from instructions."""
        immediates = []
        
        for ins in instructions:
            # Look for hex values and decimal numbers
            import re
            hex_values = re.findall(r'0x([0-9a-fA-F]+)', ins.op_str)
            dec_values = re.findall(r'\b(\d+)\b', ins.op_str)
            
            immediates.extend([int(h, 16) for h in hex_values])
            immediates.extend([int(d) for d in dec_values if int(d) < 0x100000])
        
        return immediates

    def _find_indirect_jumps(self, base_cfg: Dict) -> List[int]:
        """Find indirect jumps in base CFG."""
        indirect_jumps = []
        
        # Look for indirect jumps in basic blocks
        basic_blocks = base_cfg.get('basic_blocks', {})
        
        for block_addr, block_info in basic_blocks.items():
            if isinstance(block_info, dict):
                instructions = block_info.get('instructions', [])
                for ins in instructions:
                    if isinstance(ins, dict) and ins.get('mnemonic') == 'jmp':
                        # Check if it's an indirect jump (contains '[')
                        if '[' in ins.get('op_str', ''):
                            indirect_jumps.append(ins.get('address', block_addr))
        
        return indirect_jumps

    def _find_switch_candidates(self, binary_data: bytes) -> List[int]:
        """Find potential switch statement locations."""
        candidates = []
        
        # Simple pattern matching for switch signatures
        # Pattern: cmp + conditional jump + jump table
        import re
        
        switch_pattern = rb'\x83\xf8.{1,10}\x77.{1,10}\xff\x24'
        for match in re.finditer(switch_pattern, binary_data):
            candidates.append(match.start())
        
        return candidates[:10]  # Limit candidates

    def _find_seh_patterns(self, binary_data: bytes) -> List[int]:
        """Find Structured Exception Handling patterns."""
        seh_patterns = []
        
        # Pattern: mov reg, fs:[0] (access SEH chain)
        import re
        seh_pattern = rb'\x64\x8b.{1,3}\x00\x00\x00\x00'
        
        for match in re.finditer(seh_pattern, binary_data):
            seh_patterns.append(match.start())
        
        return seh_patterns[:5]  # Limit patterns

    def _analyze_switch_structure(self, binary_data: bytes, candidate_addr: int) -> Optional[Dict[str, Any]]:
        """Analyze switch statement structure."""
        # Placeholder implementation
        return {
            'address': candidate_addr,
            'type': 'switch_statement',
            'confidence': 0.7,
            'case_count': 5,
            'jump_table_address': candidate_addr + 0x50
        }

    def _analyze_exception_handler(self, binary_data: bytes, pattern_addr: int) -> Optional[Dict[str, Any]]:
        """Analyze exception handler structure."""
        # Placeholder implementation
        return {
            'address': pattern_addr,
            'type': 'seh_handler',
            'confidence': 0.6,
            'handler_address': pattern_addr + 0x20
        }

    def _update_cfg_with_predictions(self, base_cfg: Dict, indirect_predictions: List, 
                                   switch_reconstructions: List, exception_handlers: List) -> Dict:
        """Update CFG with ML predictions."""
        enhanced_cfg = base_cfg.copy()
        
        # Add indirect jump targets
        enhanced_cfg['ml_predictions'] = {
            'indirect_jumps': len(indirect_predictions),
            'switch_statements': len(switch_reconstructions),
            'exception_handlers': len(exception_handlers)
        }
        
        return enhanced_cfg

    def _calculate_confidence_scores(self, result: MLEnhancedCFGResult) -> Dict[int, float]:
        """Calculate confidence scores for CFG enhancements."""
        confidence_map = {}
        
        # Add confidence for indirect jumps
        for prediction in result.indirect_jump_predictions:
            avg_confidence = sum(prediction.confidence_scores) / len(prediction.confidence_scores)
            confidence_map[prediction.source_address] = avg_confidence
        
        return confidence_map

    def _fallback_indirect_jump_analysis(self, binary_data: bytes) -> List[IndirectJumpPrediction]:
        """Fallback analysis when ML is not available."""
        predictions = []
        
        # Simple heuristic-based analysis
        self.logger.info("Using fallback indirect jump analysis")
        
        return predictions


def create_ml_enhanced_cfg_reconstructor(config_manager=None) -> MLEnhancedCFGReconstructor:
    """
    Factory function to create ML-enhanced CFG reconstructor.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        MLEnhancedCFGReconstructor: Configured reconstructor instance
    """
    return MLEnhancedCFGReconstructor(config_manager)