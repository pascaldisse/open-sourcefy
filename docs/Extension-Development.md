# Extension Development

Guide for developing custom extensions, agents, and plugins for the Open-Sourcefy Matrix pipeline.

## Extension Architecture

### Extension Types

Open-Sourcefy supports several types of extensions:

1. **Custom Agents**: New Matrix agents with specialized capabilities
2. **Analysis Plugins**: Specialized analysis modules for specific binary types
3. **Output Generators**: Custom output formats and report generators
4. **Build System Integrations**: Support for additional compilers and build tools
5. **Format Handlers**: Support for new binary formats beyond PE/ELF

### Extension Framework

#### Base Extension Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ExtensionBase(ABC):
    """Base class for all Open-Sourcefy extensions"""
    
    def __init__(self, extension_id: str, name: str, version: str):
        self.extension_id = extension_id
        self.name = name
        self.version = version
        self.dependencies = []
        self.capabilities = {}
        
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the extension with configuration"""
        pass
        
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported binary formats"""
        pass
        
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute extension functionality"""
        pass
        
    def validate_prerequisites(self) -> bool:
        """Validate extension prerequisites"""
        return True
        
    def cleanup(self) -> None:
        """Clean up extension resources"""
        pass
```

## Custom Agent Development

### Creating a New Matrix Agent

#### Step 1: Agent Class Definition

```python
# File: extensions/agents/agent18_custom_character.py

from src.core.shared_components import ReconstructionAgent, MatrixCharacter, AgentStatus
from typing import Dict, Any, List
import time

class Agent18_CustomCharacter(ReconstructionAgent):
    """Agent 18: Custom Character - Specialized analysis capability"""
    
    def __init__(self):
        super().__init__(
            agent_id=18,
            matrix_character=MatrixCharacter.CUSTOM_CHARACTER
        )
        self.dependencies = [1, 5]  # Requires Sentinel and Neo
        self.capabilities = {
            'custom_analysis': True,
            'specialized_detection': True,
            'enhanced_processing': True
        }
        
    def execute_matrix_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom agent analysis"""
        start_time = time.time()
        
        try:
            # Validate prerequisites
            self._validate_prerequisites(context)
            
            # Extract required data
            binary_path = context['binary_path']
            sentinel_data = context['agent_results'][1].data
            neo_data = context['agent_results'][5].data
            
            # Perform custom analysis
            analysis_result = self._perform_custom_analysis(
                binary_path, 
                sentinel_data, 
                neo_data
            )
            
            # Process and validate results
            processed_result = self._process_analysis_results(analysis_result)
            quality_score = self._calculate_quality_score(processed_result)
            
            execution_time = time.time() - start_time
            
            return {
                'agent_id': self.agent_id,
                'status': 'SUCCESS',
                'data': processed_result,
                'execution_time': execution_time,
                'quality_score': quality_score,
                'capabilities_used': self.capabilities
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Agent {self.agent_id} execution failed: {str(e)}")
            
            return {
                'agent_id': self.agent_id,
                'status': 'FAILED',
                'error': str(e),
                'execution_time': execution_time
            }
    
    def _validate_prerequisites(self, context: Dict[str, Any]) -> None:
        """Validate custom agent prerequisites"""
        required_keys = ['binary_path', 'agent_results', 'shared_memory']
        missing_keys = [k for k in required_keys if k not in context]
        if missing_keys:
            raise ValueError(f"Missing required context keys: {missing_keys}")
        
        # Validate dependencies
        agent_results = context.get('agent_results', {})
        for dep_id in self.dependencies:
            if dep_id not in agent_results:
                raise ValueError(f"Required dependency Agent {dep_id} not satisfied")
    
    def _perform_custom_analysis(self, binary_path: str, sentinel_data: Dict, neo_data: Dict) -> Dict[str, Any]:
        """Implement custom analysis logic"""
        # Custom analysis implementation
        return {
            'custom_metrics': {},
            'specialized_results': {},
            'analysis_summary': {}
        }
    
    def _process_analysis_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure analysis results"""
        return {
            'processed_data': raw_results,
            'summary': {},
            'recommendations': []
        }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate quality score for this agent's results"""
        # Implement quality scoring logic
        return 0.85
    
    def get_matrix_description(self) -> str:
        """Agent description for documentation"""
        return "Agent 18: Custom Character - Specialized binary analysis with custom capabilities"
```

#### Step 2: Matrix Character Registration

```python
# File: src/core/matrix_agents.py

class MatrixCharacter(Enum):
    # Existing characters...
    CUSTOM_CHARACTER = "custom_character"
    SPECIALIZED_ANALYZER = "specialized_analyzer"
    ENHANCED_PROCESSOR = "enhanced_processor"
```

#### Step 3: Agent Registration

```python
# File: src/core/matrix_pipeline_orchestrator.py

def _initialize_agent_registry(self) -> Dict[int, ReconstructionAgent]:
    """Initialize all available agents including extensions"""
    agents = {}
    
    # Load core agents (0-16)
    # ... existing agent loading ...
    
    # Load extension agents
    extension_agents = self._load_extension_agents()
    agents.update(extension_agents)
    
    return agents

def _load_extension_agents(self) -> Dict[int, ReconstructionAgent]:
    """Load extension agents from extensions directory"""
    extension_agents = {}
    
    extensions_path = Path("extensions/agents")
    if extensions_path.exists():
        for agent_file in extensions_path.glob("agent*.py"):
            try:
                agent_module = importlib.import_module(f"extensions.agents.{agent_file.stem}")
                agent_class = getattr(agent_module, agent_file.stem.title().replace("_", ""))
                agent_instance = agent_class()
                extension_agents[agent_instance.agent_id] = agent_instance
                self.logger.info(f"Loaded extension agent: {agent_instance.agent_id}")
            except Exception as e:
                self.logger.warning(f"Failed to load extension agent {agent_file}: {e}")
    
    return extension_agents
```

## Analysis Plugin Development

### Binary Format Plugin

```python
# File: extensions/plugins/format_analyzer_macho.py

from extensions.base import ExtensionBase
from typing import Dict, Any, List
import struct

class MachoFormatAnalyzer(ExtensionBase):
    """Mach-O binary format analyzer plugin"""
    
    def __init__(self):
        super().__init__(
            extension_id="format_macho",
            name="Mach-O Format Analyzer",
            version="1.0.0"
        )
        self.magic_numbers = [0xfeedface, 0xfeedfacf, 0xcafebabe, 0xcffaedfe]
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Mach-O analyzer"""
        self.config = config
        return True
        
    def get_supported_formats(self) -> List[str]:
        """Return supported Mach-O formats"""
        return ["MACHO", "MACHO64", "FAT_MACHO"]
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Mach-O binary"""
        binary_path = context['binary_path']
        
        with open(binary_path, 'rb') as f:
            header = f.read(32)
            
        # Parse Mach-O header
        magic = struct.unpack('<I', header[:4])[0]
        
        if magic in self.magic_numbers:
            return self._analyze_macho_structure(binary_path, header)
        else:
            raise ValueError("Not a valid Mach-O binary")
    
    def _analyze_macho_structure(self, binary_path: str, header: bytes) -> Dict[str, Any]:
        """Analyze Mach-O binary structure"""
        # Implement Mach-O analysis logic
        return {
            'format': 'MACHO',
            'architecture': 'x64',
            'segments': [],
            'symbols': [],
            'imports': []
        }
```

### Compiler Detection Plugin

```python
# File: extensions/plugins/compiler_detector_rust.py

from extensions.base import ExtensionBase
from typing import Dict, Any, List
import re

class RustCompilerDetector(ExtensionBase):
    """Rust compiler detection plugin"""
    
    def __init__(self):
        super().__init__(
            extension_id="compiler_rust",
            name="Rust Compiler Detector",
            version="1.0.0"
        )
        self.rust_signatures = [
            b"rustc",
            b"cargo",
            b"rust_panic",
            b"rust_begin_unwind"
        ]
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Rust detector"""
        return True
        
    def get_supported_formats(self) -> List[str]:
        """Return supported binary formats"""
        return ["PE", "ELF", "MACHO"]
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect Rust compiler artifacts"""
        binary_path = context['binary_path']
        
        with open(binary_path, 'rb') as f:
            content = f.read()
        
        rust_indicators = []
        for signature in self.rust_signatures:
            if signature in content:
                rust_indicators.append(signature.decode('utf-8', errors='ignore'))
        
        if rust_indicators:
            version = self._detect_rust_version(content)
            return {
                'compiler': 'rustc',
                'language': 'Rust',
                'version': version,
                'indicators': rust_indicators,
                'confidence': len(rust_indicators) / len(self.rust_signatures)
            }
        
        return {'compiler': None}
    
    def _detect_rust_version(self, content: bytes) -> str:
        """Detect Rust version from binary"""
        # Look for version strings
        version_pattern = rb"rustc (\d+\.\d+\.\d+)"
        match = re.search(version_pattern, content)
        return match.group(1).decode() if match else "unknown"
```

## Output Generator Extensions

### Custom Report Generator

```python
# File: extensions/generators/html_report_generator.py

from extensions.base import ExtensionBase
from typing import Dict, Any, List
from pathlib import Path
import json

class HTMLReportGenerator(ExtensionBase):
    """HTML report generator extension"""
    
    def __init__(self):
        super().__init__(
            extension_id="generator_html",
            name="HTML Report Generator",
            version="1.0.0"
        )
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize HTML generator"""
        self.template_path = Path(config.get('template_path', 'templates/report.html'))
        return True
        
    def get_supported_formats(self) -> List[str]:
        """Return supported output formats"""
        return ["HTML", "HTM"]
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate HTML report"""
        agent_results = context['agent_results']
        pipeline_metrics = context['pipeline_metrics']
        
        html_content = self._generate_html_report(agent_results, pipeline_metrics)
        
        output_path = context['output_paths']['reports'] / 'analysis_report.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            'generator': 'HTML',
            'output_path': str(output_path),
            'file_size': output_path.stat().st_size
        }
    
    def _generate_html_report(self, agent_results: Dict, metrics: Dict) -> str:
        """Generate HTML content"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Open-Sourcefy Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .agent-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .metrics { background-color: #f5f5f5; padding: 10px; }
                .success { color: green; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>Open-Sourcefy Matrix Pipeline Analysis Report</h1>
            
            <div class="metrics">
                <h2>Pipeline Metrics</h2>
                <p>Overall Quality: {overall_quality:.2f}</p>
                <p>Execution Time: {execution_time:.2f}s</p>
                <p>Agents Completed: {agents_completed}/{total_agents}</p>
            </div>
            
            <h2>Agent Results</h2>
            {agent_sections}
        </body>
        </html>
        """
        
        agent_sections = ""
        for agent_id, result in agent_results.items():
            status_class = "success" if result.status == "SUCCESS" else "error"
            agent_sections += f"""
            <div class="agent-section">
                <h3>Agent {agent_id}: {result.agent_name}</h3>
                <p class="{status_class}">Status: {result.status}</p>
                <p>Execution Time: {result.execution_time:.2f}s</p>
                <p>Quality Score: {result.quality_score:.2f}</p>
            </div>
            """
        
        return html_template.format(
            overall_quality=metrics.get('overall_quality', 0.0),
            execution_time=metrics.get('execution_time', 0.0),
            agents_completed=len([r for r in agent_results.values() if r.status == "SUCCESS"]),
            total_agents=len(agent_results),
            agent_sections=agent_sections
        )
```

## Build System Extensions

### Alternative Compiler Support

```python
# File: extensions/build_systems/clang_builder.py

from extensions.base import ExtensionBase
from typing import Dict, Any, List
import subprocess
from pathlib import Path

class ClangBuilder(ExtensionBase):
    """Clang/LLVM build system extension"""
    
    def __init__(self):
        super().__init__(
            extension_id="builder_clang",
            name="Clang Builder",
            version="1.0.0"
        )
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize Clang builder"""
        self.clang_path = config.get('clang_path', 'clang')
        self.clangxx_path = config.get('clangxx_path', 'clang++')
        return self._validate_clang_installation()
        
    def get_supported_formats(self) -> List[str]:
        """Return supported compilation targets"""
        return ["PE", "ELF", "MACHO"]
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compile using Clang"""
        source_files = context['source_files']
        output_path = context['output_path']
        
        compile_result = self._compile_with_clang(source_files, output_path)
        
        return {
            'compiler': 'clang',
            'success': compile_result.returncode == 0,
            'output_path': output_path,
            'compilation_time': compile_result.duration
        }
    
    def _validate_clang_installation(self) -> bool:
        """Validate Clang installation"""
        try:
            result = subprocess.run([self.clang_path, '--version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _compile_with_clang(self, source_files: List[str], output_path: str) -> subprocess.CompletedProcess:
        """Compile source files with Clang"""
        cmd = [
            self.clang_path,
            '-O2',  # Optimization
            '-Wall',  # Warnings
            '-std=c11',  # C standard
            '-o', output_path
        ] + source_files
        
        return subprocess.run(cmd, capture_output=True, text=True)
```

## Extension Configuration

### Extension Configuration File

```yaml
# File: config/extensions.yaml

extensions:
  enabled: true
  auto_load: true
  extension_paths:
    - "extensions/"
    - "~/.openSourcefy/extensions/"
  
  agents:
    custom_character:
      enabled: true
      priority: 100
      dependencies: [1, 5]
      
    specialized_analyzer:
      enabled: false
      priority: 200
      
  plugins:
    format_analyzers:
      macho_analyzer:
        enabled: true
        priority: 10
        
      rust_detector:
        enabled: true
        priority: 20
        
  generators:
    html_report:
      enabled: true
      template_path: "templates/custom_report.html"
      
    json_export:
      enabled: true
      pretty_print: true
      
  build_systems:
    clang_builder:
      enabled: false
      clang_path: "/usr/bin/clang"
      clangxx_path: "/usr/bin/clang++"
```

### Extension Loading System

```python
# File: src/core/extension_manager.py

class ExtensionManager:
    """Manages loading and execution of extensions"""
    
    def __init__(self, config_path: str = "config/extensions.yaml"):
        self.config = self._load_extension_config(config_path)
        self.loaded_extensions = {}
        
    def load_extensions(self) -> None:
        """Load all enabled extensions"""
        if not self.config.get('enabled', False):
            return
            
        extension_paths = self.config.get('extension_paths', [])
        
        for path in extension_paths:
            self._load_extensions_from_path(Path(path))
    
    def _load_extensions_from_path(self, path: Path) -> None:
        """Load extensions from a specific path"""
        if not path.exists():
            return
            
        for ext_type in ['agents', 'plugins', 'generators', 'build_systems']:
            ext_dir = path / ext_type
            if ext_dir.exists():
                self._load_extension_type(ext_dir, ext_type)
    
    def get_extension(self, extension_id: str) -> ExtensionBase:
        """Get loaded extension by ID"""
        return self.loaded_extensions.get(extension_id)
    
    def list_extensions(self) -> List[str]:
        """List all loaded extension IDs"""
        return list(self.loaded_extensions.keys())
```

## Testing Extensions

### Extension Test Framework

```python
# File: tests/extensions/test_custom_agent.py

import unittest
from extensions.agents.agent18_custom_character import Agent18_CustomCharacter
from src.core.matrix_agents import AgentResult, AgentStatus

class TestCustomAgent(unittest.TestCase):
    """Test custom agent extension"""
    
    def setUp(self):
        self.agent = Agent18_CustomCharacter()
        self.test_context = {
            'binary_path': 'tests/fixtures/test_binary.exe',
            'shared_memory': {},
            'output_paths': {},
            'agent_results': {
                1: AgentResult(
                    agent_id=1, status=AgentStatus.SUCCESS,
                    data={'binary_info': {}}, agent_name="Sentinel", 
                    matrix_character="sentinel"
                ),
                5: AgentResult(
                    agent_id=5, status=AgentStatus.SUCCESS,
                    data={'decompilation': {}}, agent_name="Neo", 
                    matrix_character="neo"
                )
            }
        }
    
    def test_agent_initialization(self):
        """Test custom agent initialization"""
        self.assertEqual(self.agent.agent_id, 18)
        self.assertEqual(self.agent.dependencies, [1, 5])
        self.assertTrue(self.agent.capabilities['custom_analysis'])
    
    def test_agent_execution(self):
        """Test custom agent execution"""
        result = self.agent.execute_matrix_task(self.test_context)
        
        self.assertEqual(result['agent_id'], 18)
        self.assertEqual(result['status'], 'SUCCESS')
        self.assertIn('data', result)
        self.assertGreater(result['execution_time'], 0)
    
    def test_prerequisite_validation(self):
        """Test prerequisite validation"""
        # Test with missing dependencies
        incomplete_context = {
            'binary_path': 'test.exe',
            'agent_results': {}
        }
        
        with self.assertRaises(ValueError):
            self.agent._validate_prerequisites(incomplete_context)
```

## Extension Distribution

### Extension Package Structure

```
custom_extension/
├── setup.py
├── README.md
├── requirements.txt
├── extension_config.yaml
├── agents/
│   └── agent18_custom_character.py
├── plugins/
│   └── custom_analyzer.py
├── generators/
│   └── custom_report.py
├── templates/
│   └── report_template.html
└── tests/
    ├── test_agents.py
    ├── test_plugins.py
    └── test_generators.py
```

### Installation Script

```python
# File: setup.py

from setuptools import setup, find_packages

setup(
    name="openSourcefy-custom-extension",
    version="1.0.0",
    description="Custom extension for Open-Sourcefy Matrix pipeline",
    author="Extension Developer",
    packages=find_packages(),
    install_requires=[
        "openSourcefy>=2.0.0",
    ],
    entry_points={
        'openSourcefy.extensions': [
            'custom_agent = agents.agent18_custom_character:Agent18_CustomCharacter',
            'custom_analyzer = plugins.custom_analyzer:CustomAnalyzer',
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
```

---

**Related**: [[Developer Guide|Developer-Guide]] - Core development guide  
**Next**: [[API Reference|API-Reference]] - Programming interface documentation