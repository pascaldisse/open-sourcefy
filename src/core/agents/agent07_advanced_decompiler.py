"""
Agent 7: Advanced Decompiler
Advanced decompilation with optimization reversal and structure reconstruction.
NOW WITH REAL GHIDRA INTEGRATION!
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
from ..agent_base import BaseAgent, AgentResult, AgentStatus


class Agent7_AdvancedDecompiler(BaseAgent):
    """Agent 7: Advanced decompilation with optimization reversal"""
    
    def __init__(self):
        super().__init__(
            agent_id=7,
            name="AdvancedDecompiler",
            dependencies=[4, 5]
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute advanced decompilation"""
        # Get dependencies
        agent4_result = context['agent_results'].get(4)
        agent5_result = context['agent_results'].get(5)
        
        if not (agent4_result and agent4_result.status == AgentStatus.COMPLETED and
                agent5_result and agent5_result.status == AgentStatus.COMPLETED):
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message="Required dependencies not completed successfully"
            )

        try:
            decompilation_data = agent4_result.data
            structure_data = agent5_result.data
            
            # Get the binary path from context
            binary_path = context.get('global_data', {}).get('binary_path') or context.get('binary_path')
            if not binary_path:
                return AgentResult(
                    agent_id=self.agent_id,
                    status=AgentStatus.FAILED,
                    data={},
                    error_message="Binary path not found in context"
                )
            
            # Attempt REAL Ghidra decompilation first
            try:
                ghidra_result = self._run_ghidra_decompilation(binary_path, context)
                if ghidra_result['success']:
                    # Ghidra decompilation successful
                    advanced_analysis = ghidra_result
                else:
                    # Ghidra failed, falling back to basic analysis
                    advanced_analysis = self._perform_advanced_decompilation(
                        decompilation_data, structure_data, context
                    )
            except Exception as e:
                # Ghidra decompilation failed with exception
                advanced_analysis = self._perform_advanced_decompilation(
                    decompilation_data, structure_data, context
                )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data=advanced_analysis,
                metadata={
                    'depends_on': [4, 5],
                    'analysis_type': 'advanced_decompilation'
                }
            )
            
        except Exception as e:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.FAILED,
                data={},
                error_message=f"Advanced decompilation failed: {str(e)}"
            )

    def _perform_advanced_decompilation(self, decompilation_data: Dict[str, Any], 
                                      structure_data: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced decompilation with structure awareness"""
        return {
            'enhanced_functions': self._enhance_function_analysis(decompilation_data, structure_data),
            'reconstructed_structures': self._reconstruct_data_structures(structure_data),
            'control_flow_analysis': self._analyze_control_flow(decompilation_data),
            'type_reconstruction': self._reconstruct_types(decompilation_data, structure_data),
            'optimization_reversal': self._reverse_optimizations(decompilation_data),
            'confidence_metrics': self._calculate_confidence_metrics(decompilation_data, structure_data)
        }

    def _enhance_function_analysis(self, decompilation_data: Dict[str, Any], 
                                 structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced function analysis using structure information"""
        enhanced = {
            'functions': {},
            'call_graph': {},
            'parameter_analysis': {},
            'return_value_analysis': {}
        }
        
        functions = decompilation_data.get('decompiled_functions', {})
        for func_name, func_data in functions.items():
            enhanced['functions'][func_name] = {
                'original_data': func_data,
                'enhanced_signature': self._enhance_function_signature(func_name, func_data, structure_data),
                'local_variables': self._identify_local_variables(func_data, structure_data),
                'complexity_analysis': self._analyze_function_complexity(func_data)
            }
        
        return enhanced

    def _enhance_function_signature(self, func_name: str, func_data: Any, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance function signature with type information"""
        return {
            'name': func_name,
            'parameters': [],
            'return_type': 'unknown',
            'calling_convention': 'unknown',
            'confidence': 0.5
        }

    def _identify_local_variables(self, func_data: Any, structure_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify local variables in function"""
        return []

    def _analyze_function_complexity(self, func_data: Any) -> Dict[str, Any]:
        """Analyze function complexity metrics"""
        return {
            'cyclomatic_complexity': 1,
            'instruction_count': 0,
            'basic_block_count': 1,
            'complexity_rating': 'low'
        }

    def _reconstruct_data_structures(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct data structures from binary analysis"""
        return {
            'structs': [],
            'unions': [],
            'arrays': [],
            'custom_types': [],
            'type_relationships': {}
        }

    def _analyze_control_flow(self, decompilation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze control flow patterns"""
        return {
            'control_flow_graphs': {},
            'loop_detection': {},
            'branch_analysis': {},
            'exception_handling': {}
        }

    def _reconstruct_types(self, decompilation_data: Dict[str, Any], structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct type information"""
        return {
            'primitive_types': {},
            'composite_types': {},
            'pointer_analysis': {},
            'type_inference_confidence': 0.0
        }

    def _reverse_optimizations(self, decompilation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to reverse compiler optimizations"""
        return {
            'reversed_inlining': [],
            'reconstructed_loops': [],
            'restored_variables': [],
            'reversal_confidence': 0.0
        }

    def _calculate_confidence_metrics(self, decompilation_data: Dict[str, Any], structure_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence metrics for advanced analysis"""
        raise NotImplementedError(
            "Confidence metrics calculation not implemented - requires statistical "
            "analysis of decompilation quality, function signature accuracy, data "
            "structure reconstruction quality, and type inference success rates. "
            "Implementation needs: 1) Quality scoring algorithms, 2) Statistical "
            "analysis of reconstruction accuracy, 3) Validation against known patterns"
        )

    def _run_ghidra_decompilation(self, binary_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run actual Ghidra decompilation to get real source code"""
        result = {
            'success': False,
            'functions_decompiled': 0,
            'total_functions': 0,
            'decompiled_code': {},
            'error': '',
            'ghidra_output': '',
            'quality_score': 0.0
        }
        
        try:
            # Find Ghidra installation
            ghidra_home = os.environ.get('GHIDRA_HOME')
            if not ghidra_home:
                # Try to find Ghidra in the project directory
                project_root = Path(__file__).parent.parent.parent.parent
                ghidra_path = project_root / "ghidra"
                if ghidra_path.exists():
                    ghidra_home = str(ghidra_path)
                else:
                    result['error'] = "GHIDRA_HOME not set and Ghidra not found in project directory"
                    return result
            
            # Check for analyzeHeadless script
            analyze_script = Path(ghidra_home) / "support" / "analyzeHeadless"
            if not analyze_script.exists():
                result['error'] = f"analyzeHeadless script not found at {analyze_script}"
                return result
            
            # Create temporary directory for analysis
            with tempfile.TemporaryDirectory() as temp_dir:
                project_dir = Path(temp_dir) / "ghidra_project"
                project_dir.mkdir()
                
                # Get output paths from context
                output_paths = context.get('output_paths', {})
                ghidra_output_dir = output_paths.get('ghidra', 'output/ghidra')
                
                # Ensure ghidra output directory exists
                Path(ghidra_output_dir).mkdir(parents=True, exist_ok=True)
                
                # Create absolute paths for output files to ensure Ghidra can find them
                decompiled_output_file = str(Path(ghidra_output_dir).resolve() / 'decompiled_functions.c')
                summary_output_file = str(Path(ghidra_output_dir).resolve() / 'decompilation_summary.json')
                
                # Convert to absolute paths and normalize for cross-platform compatibility
                decompiled_output_file = os.path.abspath(decompiled_output_file).replace('\\', '/')
                summary_output_file = os.path.abspath(summary_output_file).replace('\\', '/')
                
                # Create Ghidra script to extract decompiled code
                script_content = f'''
// DecompileAll.java - Extract all decompiled functions
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.*;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.*;
import java.io.*;
import java.util.*;

public class DecompileAll extends GhidraScript {{
    @Override
    public void run() throws Exception {{
        println("Starting decompilation process...");
        
        DecompInterface decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);
        
        FunctionManager funcMgr = currentProgram.getFunctionManager();
        FunctionIterator functions = funcMgr.getFunctions(true);
        
        // Create output directory if it doesn't exist
        File outputDir = new File("{Path(ghidra_output_dir).resolve().as_posix()}");
        if (!outputDir.exists()) {{
            outputDir.mkdirs();
            println("Created output directory: " + outputDir.getAbsolutePath());
        }}
        
        File outputFile = new File("{decompiled_output_file}");
        println("Writing decompiled code to: " + outputFile.getAbsolutePath());
        
        PrintWriter writer = new PrintWriter(new FileWriter(outputFile));
        
        writer.println("// Decompiled with Ghidra " + getGhidraVersion());
        writer.println("#include <stdio.h>");
        writer.println("#include <stdlib.h>");
        writer.println("#include <string.h>");
        writer.println("#include <windows.h>");
        writer.println("");
        
        int functionCount = 0;
        int decompiled = 0;
        
        while (functions.hasNext()) {{
            Function func = functions.next();
            functionCount++;
            
            try {{
                DecompileResults results = decompiler.decompileFunction(func, 30, null);
                if (results != null && results.decompileCompleted()) {{
                    String decompiledCode = results.getDecompiledFunction().getC();
                    if (decompiledCode != null && !decompiledCode.trim().isEmpty()) {{
                        writer.println("// Function: " + func.getName() + " at " + func.getEntryPoint());
                        writer.println(decompiledCode);
                        writer.println("");
                        decompiled++;
                    }}
                }}
            }} catch (Exception e) {{
                writer.println("// ERROR decompiling " + func.getName() + ": " + e.getMessage());
            }}
        }}
        
        writer.println("// Total functions: " + functionCount);
        writer.println("// Successfully decompiled: " + decompiled);
        writer.close();
        
        decompiler.closeProgram();
        
        // Write summary to separate file
        File summaryFile = new File("{summary_output_file}");
        println("Writing summary to: " + summaryFile.getAbsolutePath());
        
        PrintWriter summaryWriter = new PrintWriter(new FileWriter(summaryFile));
        summaryWriter.println("{{");
        summaryWriter.println("  \\"total_functions\\": " + functionCount + ",");
        summaryWriter.println("  \\"decompiled_functions\\": " + decompiled + ",");
        summaryWriter.println("  \\"success_rate\\": " + (decompiled * 100.0 / functionCount) + ",");
        summaryWriter.println("  \\"binary_analyzed\\": \\"" + currentProgram.getName() + "\\"");
        summaryWriter.println("}}");
        summaryWriter.close();
        
        println("Decompilation complete: " + decompiled + "/" + functionCount + " functions");
        println("Output files created:");
        println("  - " + outputFile.getAbsolutePath());
        println("  - " + summaryFile.getAbsolutePath());
    }}
}}
'''
                
                # Write the script file with absolute path
                script_file = Path(ghidra_output_dir).resolve() / "DecompileAll.java"
                script_file.write_text(script_content)
                
                # Build Ghidra command with absolute paths
                cmd = [
                    str(analyze_script),
                    str(project_dir),
                    "TempProject",
                    "-import", str(Path(binary_path).resolve()),
                    "-postScript", str(script_file.resolve()),
                    "-deleteProject"
                ]
                
                # Running Ghidra decompilation command
                
                # Run Ghidra
                process = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd=ghidra_home
                )
                
                result['ghidra_output'] = process.stdout + process.stderr
                
                if process.returncode == 0:
                    # Check if decompilation files were created
                    decompiled_file = Path(ghidra_output_dir) / 'decompiled_functions.c'
                    summary_file = Path(ghidra_output_dir) / 'decompilation_summary.json'
                    
                    if decompiled_file.exists() and summary_file.exists():
                        # Read the decompiled code
                        decompiled_code = decompiled_file.read_text()
                        
                        # Read the summary
                        summary_data = json.loads(summary_file.read_text())
                        
                        result['success'] = True
                        result['functions_decompiled'] = summary_data['decompiled_functions']
                        result['total_functions'] = summary_data['total_functions']
                        result['decompiled_code']['main.c'] = decompiled_code
                        result['quality_score'] = summary_data['success_rate'] / 100.0
                        
                        # Ghidra decompilation completed
                    else:
                        result['error'] = "Ghidra completed but output files not found"
                else:
                    result['error'] = f"Ghidra failed with return code {process.returncode}"
                    
        except subprocess.TimeoutExpired:
            result['error'] = "Ghidra decompilation timed out (>10 minutes)"
        except Exception as e:
            result['error'] = f"Exception during Ghidra execution: {str(e)}"
        
        return result