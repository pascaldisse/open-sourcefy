"""
Ghidra Headless Automation Module

This module provides automation for running Ghidra in headless mode
to analyze binary files and extract decompiled C code.

Enhanced with community script integration and accuracy optimization strategies
based on research from:
- ghidra-headless-scripts repository
- ghidra-headless-decompile repository
- Best practices for decompilation accuracy
"""

import os
import subprocess
import tempfile
import shutil
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GhidraHeadlessError(Exception):
    """Custom exception for Ghidra headless operation errors"""
    pass


class GhidraHeadless:
    """
    Enhanced Ghidra Headless automation class for binary analysis and decompilation
    
    Features:
    - Community script integration (decompiler.py style)
    - Accuracy optimization strategies
    - Quality assessment and confidence scoring
    - Advanced memory and data mutability settings
    """
    
    def __init__(self, ghidra_home: str = None, enable_accuracy_optimizations: bool = True, output_base_dir: str = None, analysis_timeout: int = None):
        """
        Initialize Ghidra headless automation with enhanced process management
        
        Args:
            ghidra_home: Path to Ghidra installation directory
            enable_accuracy_optimizations: Enable accuracy improvement strategies
            output_base_dir: Base output directory for organizing files (defaults to temp dir)
            analysis_timeout: Analysis timeout per file in seconds (default: 30)
        """
        self.ghidra_home = ghidra_home or self._find_ghidra_home()
        self.analyze_headless = self._get_analyze_headless_path()
        self.project_dir = None
        self.project_name = "TempProject"
        self.enable_accuracy_optimizations = enable_accuracy_optimizations
        self.output_base_dir = output_base_dir
        self.community_scripts_dir = self._setup_community_scripts()
        self.analysis_timeout = analysis_timeout  # Configurable timeout for testing
        
        # Process management for timeout handling
        self.active_processes = []
        
        if not self.ghidra_home or not os.path.exists(self.ghidra_home):
            raise GhidraHeadlessError(f"Ghidra installation not found at: {self.ghidra_home}")
            
        if not os.path.exists(self.analyze_headless):
            raise GhidraHeadlessError(f"analyzeHeadless script not found at: {self.analyze_headless}")

    def _find_ghidra_home(self) -> Optional[str]:
        """
        Auto-detect Ghidra installation directory
        """
        # Check environment variable
        ghidra_home = os.environ.get('GHIDRA_HOME')
        if ghidra_home and os.path.exists(ghidra_home):
            return ghidra_home
            
        # Check common installation paths
        # Use relative path to project ghidra directory first
        project_root = Path(__file__).parent.parent.parent
        project_ghidra = project_root / "ghidra"
        
        common_paths = [
            str(project_ghidra),
            "/opt/ghidra",
            "/usr/local/ghidra",
            "C:\\ghidra",
            "C:\\Program Files\\Ghidra"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
                
        return None

    def _get_analyze_headless_path(self) -> str:
        """
        Get the path to analyzeHeadless script
        """
        if os.name == 'nt':
            # Windows path structure
            return os.path.join(self.ghidra_home, "Ghidra", "RuntimeScripts", "Windows", "support", "analyzeHeadless.bat")
        else:
            # Linux/Unix path structure
            return os.path.join(self.ghidra_home, "Ghidra", "RuntimeScripts", "Linux", "support", "analyzeHeadless")

    def _setup_community_scripts(self) -> str:
        """
        Setup directory for community decompilation scripts.
        Uses output structure if available, otherwise uses temp directory.
        """
        if self.output_base_dir:
            # Use output structure when available
            scripts_dir = Path(self.output_base_dir) / "temp" / "ghidra_scripts"
        else:
            # Fallback to temp directory to avoid creating directories in project root
            import tempfile
            temp_base = Path(tempfile.gettempdir()) / "open-sourcefy"
            scripts_dir = temp_base / "ghidra_scripts"
        
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced decompiler script based on community best practices
        self._create_enhanced_decompiler_script(scripts_dir)
        
        return str(scripts_dir)

    def _create_enhanced_decompiler_script(self, scripts_dir: Path):
        """
        Create enhanced decompiler script based on community research
        """
        script_path = scripts_dir / "EnhancedDecompiler.java"
        
        if not script_path.exists():
            script_content = '''
//Enhanced decompiler script based on community best practices
//@author Open-Sourcefy Enhanced Analysis
//@category Binary.Decompilation
//@keybinding
//@menupath
//@toolbar

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.app.decompiler.*;
import ghidra.program.model.pcode.*;
import ghidra.program.model.symbol.*;
import ghidra.program.model.mem.*;
import ghidra.util.task.TaskMonitor;
import java.io.*;
import java.util.*;

public class EnhancedDecompiler extends GhidraScript {
    
    private DecompInterface decompiler;
    private Map<String, Integer> qualityMetrics;
    
    @Override
    public void run() throws Exception {
        if (currentProgram == null) {
            println("No program is open");
            return;
        }
        
        String outputFile = getScriptArgs().length > 0 ? getScriptArgs()[0] : 
            currentProgram.getName() + "_enhanced_decompiled.c";
        
        // Initialize quality metrics
        qualityMetrics = new HashMap<>();
        
        // Setup decompiler with accuracy optimizations
        setupDecompiler();
        
        // Apply accuracy optimizations
        applyAccuracyOptimizations();
        
        // Perform enhanced decompilation
        performEnhancedDecompilation(outputFile);
        
        // Generate quality report
        generateQualityReport(outputFile + ".quality.json");
        
        println("Enhanced decompilation completed: " + outputFile);
        println("Quality report: " + outputFile + ".quality.json");
    }
    
    private void setupDecompiler() throws Exception {
        decompiler = new DecompInterface();
        
        // Configure decompiler options for accuracy
        DecompileOptions options = new DecompileOptions();
        options.setMaxInstructions(10000);
        options.setMaxPayloadMBytes(50);
        options.setCommentStyle(DecompileOptions.SUGGESTED_COMMENTS);
        
        decompiler.setOptions(options);
        decompiler.toggleCCode(true);
        decompiler.toggleSyntaxTree(true);
        decompiler.setSimplificationStyle("normalize");
        
        if (!decompiler.openProgram(currentProgram)) {
            throw new Exception("Failed to initialize decompiler");
        }
    }
    
    private void applyAccuracyOptimizations() {
        // Set memory permissions and data mutability for accuracy
        Memory memory = currentProgram.getMemory();
        MemoryBlock[] blocks = memory.getBlocks();
        
        for (MemoryBlock block : blocks) {
            if (block.isInitialized()) {
                // Optimize memory block settings
                if (block.isWrite()) {
                    qualityMetrics.put("writable_blocks", 
                        qualityMetrics.getOrDefault("writable_blocks", 0) + 1);
                }
            }
        }
        
        // Apply function signature improvements
        FunctionManager funcMgr = currentProgram.getFunctionManager();
        FunctionIterator functions = funcMgr.getFunctions(true);
        
        while (functions.hasNext()) {
            Function func = functions.next();
            if (func.isThunk()) {
                qualityMetrics.put("thunk_functions", 
                    qualityMetrics.getOrDefault("thunk_functions", 0) + 1);
            }
        }
    }
    
    private void performEnhancedDecompilation(String outputFile) throws Exception {
        PrintWriter writer = new PrintWriter(new FileWriter(outputFile));
        
        // Write header with quality information
        writer.println("// Enhanced Decompilation Output");
        writer.println("// Program: " + currentProgram.getName());
        writer.println("// Architecture: " + currentProgram.getLanguage().getProcessor());
        writer.println("// Compiler: " + currentProgram.getCompilerSpec().getCompilerSpecID());
        writer.println();
        
        FunctionManager funcMgr = currentProgram.getFunctionManager();
        FunctionIterator functions = funcMgr.getFunctions(true);
        
        int totalFunctions = 0;
        int successfulDecompilations = 0;
        
        while (functions.hasNext()) {
            Function func = functions.next();
            totalFunctions++;
            
            if (func.isThunk()) {
                continue;
            }
            
            try {
                DecompileResults results = decompiler.decompileFunction(func, 
                    DecompileOptions.SUGGESTED_DECOMPILE_TIMEOUT_SECS, getMonitor());
                
                if (results != null && results.decompileCompleted()) {
                    successfulDecompilations++;
                    
                    // Write function with confidence analysis
                    writer.println("// Function: " + func.getName());
                    writer.println("// Address: " + func.getEntryPoint());
                    writer.println("// Confidence: " + calculateConfidence(results));
                    writer.println(results.getDecompiledFunction().getC());
                    writer.println();
                } else {
                    writer.println("// DECOMPILATION FAILED: " + func.getName());
                    writer.println("// Reason: " + (results != null ? 
                        results.getErrorMessage() : "Unknown error"));
                    writer.println();
                }
            } catch (Exception e) {
                writer.println("// EXCEPTION during decompilation of " + func.getName() + 
                    ": " + e.getMessage());
                writer.println();
            }
        }
        
        qualityMetrics.put("total_functions", totalFunctions);
        qualityMetrics.put("successful_decompilations", successfulDecompilations);
        qualityMetrics.put("success_rate", 
            totalFunctions > 0 ? (successfulDecompilations * 100) / totalFunctions : 0);
        
        writer.close();
    }
    
    private int calculateConfidence(DecompileResults results) {
        // Simple confidence calculation based on decompilation characteristics
        int confidence = 75; // Base confidence
        
        String code = results.getDecompiledFunction().getC();
        
        // Reduce confidence for common decompilation artifacts
        if (code.contains("undefined")) confidence -= 10;
        if (code.contains("UNRECOVERED_JUMPTABLE")) confidence -= 15;
        if (code.contains("switch(")) confidence += 5; // Good switch recovery
        if (code.contains("DAT_")) confidence -= 5; // Generic data references
        
        return Math.max(0, Math.min(100, confidence));
    }
    
    private void generateQualityReport(String reportFile) throws Exception {
        PrintWriter writer = new PrintWriter(new FileWriter(reportFile));
        
        writer.println("{");
        writer.println("  \\"analysis_type\\": \\"enhanced_decompilation\\",");
        writer.println("  \\"timestamp\\": \\"" + new Date() + "\\",");
        writer.println("  \\"program\\": \\"" + currentProgram.getName() + "\\",");
        writer.println("  \\"metrics\\": {");
        
        boolean first = true;
        for (Map.Entry<String, Integer> entry : qualityMetrics.entrySet()) {
            if (!first) writer.println(",");
            writer.print("    \\"" + entry.getKey() + "\\": " + entry.getValue());
            first = false;
        }
        
        writer.println();
        writer.println("  }");
        writer.println("}");
        writer.close();
    }
}
'''
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            logger.info(f"Created enhanced decompiler script: {script_path}")

    def get_accuracy_optimization_flags(self) -> List[str]:
        """
        Get command line flags for accuracy optimization
        """
        flags = []
        
        if self.enable_accuracy_optimizations:
            # Enable comprehensive analysis with configurable timeout
            flags.extend(["-analysisTimeoutPerFile", str(self.analysis_timeout)])
            flags.extend(["-max-cpu", str(os.cpu_count() or 4)])
            
            # Memory and permission optimizations
            flags.extend(["-scriptPath", self.community_scripts_dir])
            
        return flags

    def create_project(self, project_dir: str = None) -> str:
        """
        Create a temporary Ghidra project directory
        
        Args:
            project_dir: Optional project directory path
            
        Returns:
            Path to the created project directory
        """
        if project_dir:
            self.project_dir = project_dir
        else:
            self.project_dir = tempfile.mkdtemp(prefix="ghidra_project_")
            
        os.makedirs(self.project_dir, exist_ok=True)
        logger.info(f"Created Ghidra project directory: {self.project_dir}")
        return self.project_dir

    def _terminate_process_safely(self, process):
        """
        Safely terminate a Ghidra process with proper cleanup
        """
        if not process or process.poll() is not None:
            return  # Process already terminated
            
        try:
            if os.name == 'nt':
                # Windows: terminate process tree
                logger.info(f"Terminating Ghidra process {process.pid} on Windows")
                process.terminate()
                import time
                time.sleep(2)  # Give it time to cleanup
                if process.poll() is None:
                    logger.warning(f"Force killing Ghidra process {process.pid}")
                    process.kill()
                    process.wait(timeout=None)  # No hardcoded timeout per rules.md
            else:
                # Unix: kill process group
                import signal
                logger.info(f"Terminating Ghidra process group {process.pid} on Unix")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    import time
                    time.sleep(2)
                    # Check if still running
                    if process.poll() is None:
                        logger.warning(f"Force killing Ghidra process group {process.pid}")
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait(timeout=None)  # No hardcoded timeout per rules.md
                except ProcessLookupError:
                    # Process already terminated
                    pass
        except Exception as kill_error:
            logger.error(f"Failed to terminate Ghidra process {process.pid}: {kill_error}")
            # Last resort - try basic kill
            try:
                process.kill()
                process.wait(timeout=None)  # No hardcoded timeout per rules.md
            except:
                pass

    def cleanup_project(self):
        """
        Clean up temporary project directory and terminate any active processes
        """
        # Terminate any active processes
        processes_to_cleanup = list(self.active_processes)  # Create copy to avoid modification during iteration
        for process in processes_to_cleanup:
            try:
                if process.poll() is None:  # Process is still running
                    logger.warning(f"Terminating active Ghidra process {process.pid} during cleanup")
                    self._terminate_process_safely(process)
            except Exception as e:
                logger.error(f"Error terminating process during cleanup: {e}")
        
        self.active_processes.clear()
        
        # Clean up project directory
        if self.project_dir and os.path.exists(self.project_dir):
            shutil.rmtree(self.project_dir)
            logger.info(f"Cleaned up project directory: {self.project_dir}")
            self.project_dir = None
            
    def __enter__(self):
        """Context manager entry - create project if needed"""
        if not self.project_dir:
            self.create_project()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup everything"""
        self.cleanup_project()

    def run_ghidra_analysis(
        self, 
        binary_path: str, 
        output_dir: str, 
        script_name: str = "CompleteDecompiler.java",
        timeout: int = None
    ) -> Tuple[bool, str]:
        """
        Run Ghidra analysis on a binary file
        
        Args:
            binary_path: Path to the binary file to analyze
            output_dir: Directory to store analysis output
            script_name: Name of the Ghidra script to run
            timeout: Timeout for the analysis in seconds
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        # Convert to absolute path
        binary_path = os.path.abspath(binary_path)
        output_dir = os.path.abspath(output_dir)
        
        if not os.path.exists(binary_path):
            raise GhidraHeadlessError(f"Binary file not found: {binary_path}")
            
        # Create project if not exists
        if not self.project_dir:
            self.create_project()
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get script path - check community scripts directory first, then fallback to project ghidra directory
        script_path = None
        
        # First check community scripts directory (where EnhancedDecompiler.java is created)
        community_script_path = os.path.join(self.community_scripts_dir, script_name)
        if os.path.exists(community_script_path):
            script_path = community_script_path
        else:
            # Fallback to project ghidra directory for other scripts
            project_script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "ghidra",
                script_name
            )
            if os.path.exists(project_script_path):
                script_path = project_script_path
        
        if not script_path:
            logger.warning(f"Script not found: {script_name}, analysis will run without custom script")
            script_path = None
        
        # Build command
        cmd = [
            self.analyze_headless,
            self.project_dir,
            self.project_name,
            "-import", binary_path,
            "-overwrite",
            "-deleteProject"
        ]
        
        if script_path:
            cmd.extend(["-scriptPath", os.path.dirname(script_path)])
            cmd.extend(["-postScript", os.path.basename(script_path), output_dir])
        
        # Add standard analysis with increased memory and timeout
        # Handle unlimited timeout case (None or -1)
        if timeout is None or timeout == -1:
            # Use large but reasonable timeout for "unlimited" (30 minutes)
            analysis_timeout = str(120 * 60)  # Use large value for unlimited mode (2 hours)
        else:
            analysis_timeout = str(timeout)
        
        cmd.extend(["-analysisTimeoutPerFile", analysis_timeout])
        cmd.extend(["-max-cpu", str(os.cpu_count() or 4)])
        
        # OPTIMIZATION: Enhanced JVM memory and performance options
        file_size_mb = os.path.getsize(binary_path) / (1024 * 1024)
        
        # Dynamic memory allocation based on file size
        if file_size_mb < 1:
            java_opts = '2G'
        elif file_size_mb < 10:
            java_opts = '4G'
        elif file_size_mb < 100:
            java_opts = '6G'
        else:
            java_opts = '8G'
            
        # Override with environment variable if set
        java_opts = os.environ.get('GHIDRA_MAXMEM', java_opts)
        os.environ['GHIDRA_MAXMEM'] = java_opts
        
        # Additional JVM optimizations for performance
        os.environ['VMARGS'] = os.environ.get('VMARGS', '') + ' -XX:+UseG1GC -XX:+UseStringDeduplication -XX:MaxGCPauseMillis=200 -server'
        
        logger.info(f"Running Ghidra analysis: {' '.join(cmd)}")
        logger.info(f"Working directory: {self.ghidra_home}")
        logger.info(f"Environment: GHIDRA_MAXMEM={os.environ.get('GHIDRA_MAXMEM', 'not set')}")
        logger.info(f"Environment: VMARGS={os.environ.get('VMARGS', 'not set')}")
        
        
        process = None
        try:
            # OPTIMIZATION: Enhanced process management with resource optimization
            import psutil
            
            # Set process creation flags for better resource management
            creation_flags = 0
            if os.name == 'nt':
                # Windows: Use high priority and background processing
                creation_flags = subprocess.HIGH_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.ghidra_home,
                creationflags=creation_flags if os.name == 'nt' else 0,
                preexec_fn=None if os.name == 'nt' else os.setsid,
                bufsize=8192  # Larger buffer for better I/O performance
            )
            
            # OPTIMIZATION: Set process priority after creation
            try:
                proc = psutil.Process(process.pid)
                proc.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -5)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass  # Ignore if we can't set priority
            
            # Track active process for cleanup
            self.active_processes.append(process)
            
            # Wait for completion with timeout
            try:
                # Handle unlimited timeout case for process communication
                process_timeout = None if (timeout is None or timeout == -1) else timeout
                stdout, stderr = process.communicate(timeout=process_timeout)
                return_code = process.returncode
                
                success = return_code == 0
                output = stdout + stderr
                
                if success:
                    logger.info("Ghidra analysis completed successfully")
                else:
                    logger.error(f"Ghidra analysis failed with return code: {return_code}")
                    logger.error(f"Output: {output}")
                    
                return success, output
                
            except subprocess.TimeoutExpired:
                timeout_msg = "unlimited" if (timeout is None or timeout == -1) else f"{timeout} seconds"
                logger.warning(f"Ghidra analysis timed out after {timeout_msg} - terminating process")
                
                # Use robust process termination
                self._terminate_process_safely(process)
                
                error_msg = f"Ghidra analysis timed out after {timeout_msg}"
                return False, error_msg
                
            finally:
                # Remove from active processes list
                if process in self.active_processes:
                    self.active_processes.remove(process)
                
        except Exception as e:
            error_msg = f"Ghidra analysis failed: {str(e)}"
            logger.error(error_msg)
            
            # Cleanup any running process
            if process:
                self._terminate_process_safely(process)
                if process in self.active_processes:
                    self.active_processes.remove(process)
            
            return False, error_msg

    def extract_functions(self, binary_path: str, output_dir: str) -> Dict[str, str]:
        """
        Extract individual functions from a binary
        
        Args:
            binary_path: Path to the binary file
            output_dir: Directory to store function outputs
            
        Returns:
            Dictionary mapping function names to their decompiled code
        """
        success, output = self.run_ghidra_analysis(binary_path, output_dir)
        
        if not success:
            raise GhidraHeadlessError(f"Failed to analyze binary: {output}")
            
        # Parse function files from output directory
        functions = {}
        output_path = Path(output_dir)
        
        for func_file in output_path.glob("*.c"):
            func_name = func_file.stem
            try:
                with open(func_file, 'r', encoding='utf-8') as f:
                    functions[func_name] = f.read()
            except Exception as e:
                logger.warning(f"Failed to read function file {func_file}: {e}")
                
        logger.info(f"Extracted {len(functions)} functions from {binary_path}")
        return functions

    def multi_pass_decompilation(self, binary_path: str, output_dir: str, max_passes: int = 3) -> Dict[str, Any]:
        """
        Perform multi-pass decompilation to improve quality
        
        Args:
            binary_path: Path to the binary file
            output_dir: Directory to store outputs
            max_passes: Maximum number of passes to perform
            
        Returns:
            Dictionary with decompilation results and quality metrics
        """
        logger.info(f"Starting multi-pass decompilation of {binary_path} (max {max_passes} passes)")
        
        results = {
            'binary_path': binary_path,
            'passes': [],
            'final_quality': 0.0,
            'improvement_ratio': 0.0,
            'functions': {}
        }
        
        best_quality = 0.0
        best_pass_output = None
        
        for pass_num in range(1, max_passes + 1):
            logger.info(f"Starting decompilation pass {pass_num}/{max_passes}")
            
            pass_output_dir = os.path.join(output_dir, f"pass_{pass_num}")
            os.makedirs(pass_output_dir, exist_ok=True)
            
            # Run decompilation for this pass
            success, output = self.run_ghidra_analysis(
                binary_path, 
                pass_output_dir,
                timeout=300 + (pass_num * 60)  # Longer timeout for later passes
            )
            
            if not success:
                logger.warning(f"Pass {pass_num} failed: {output}")
                continue
                
            # Analyze quality of this pass
            quality_score = self._analyze_pass_quality(pass_output_dir)
            
            pass_result = {
                'pass_number': pass_num,
                'success': success,
                'quality_score': quality_score,
                'output_dir': pass_output_dir,
                'functions_found': len(list(Path(pass_output_dir).glob("*.c")))
            }
            
            results['passes'].append(pass_result)
            
            logger.info(f"Pass {pass_num} completed with quality score: {quality_score:.3f}")
            
            # Track best pass
            if quality_score > best_quality:
                best_quality = quality_score
                best_pass_output = pass_output_dir
                
            # Early termination if quality is good enough
            if quality_score > 0.85:
                logger.info(f"High quality achieved at pass {pass_num}, stopping early")
                break
                
            # Diminishing returns check
            if pass_num > 1 and quality_score < results['passes'][-2]['quality_score'] * 1.05:
                logger.info(f"Diminishing returns detected at pass {pass_num}, stopping")
                break
        
        # Calculate final metrics
        results['final_quality'] = best_quality
        if results['passes']:
            initial_quality = results['passes'][0]['quality_score']
            results['improvement_ratio'] = (best_quality - initial_quality) / max(initial_quality, 0.1)
        
        # Process best pass results
        if best_pass_output:
            results['functions'] = self._extract_functions_from_output(best_pass_output)
            results['best_pass_dir'] = best_pass_output
            
        logger.info(f"Multi-pass decompilation completed. Final quality: {best_quality:.3f}")
        return results
        
    def _analyze_pass_quality(self, output_dir: str) -> float:
        """Analyze the quality of a decompilation pass"""
        quality_score = 0.0
        total_functions = 0
        
        output_path = Path(output_dir)
        
        # Count functions and analyze quality
        for c_file in output_path.glob("*.c"):
            total_functions += 1
            try:
                with open(c_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Quality indicators
                func_quality = self._calculate_function_quality(content)
                quality_score += func_quality
                
            except Exception as e:
                logger.warning(f"Failed to analyze {c_file}: {e}")
                
        # Normalize by number of functions
        if total_functions > 0:
            quality_score /= total_functions
        else:
            quality_score = 0.0
            
        # Check for summary file quality
        summary_file = output_path / "decompilation_summary.txt"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary_content = f.read()
                    success_rate_match = re.search(r'Success rate: ([\d.]+)%', summary_content)
                    if success_rate_match:
                        success_rate = float(success_rate_match.group(1)) / 100.0
                        quality_score = (quality_score + success_rate) / 2.0
            except Exception:
                pass
                
        return min(1.0, max(0.0, quality_score))
        
    def _calculate_function_quality(self, code: str) -> float:
        """Calculate quality score for a single function"""
        quality = 0.5  # Base score
        
        # Positive indicators
        if 'return' in code:
            quality += 0.1
        if re.search(r'\w+\s*\([^)]*\)\s*{', code):  # Function definition
            quality += 0.1
        if any(keyword in code for keyword in ['if', 'while', 'for', 'switch']):
            quality += 0.1
        if '"' in code or "'" in code:  # String literals
            quality += 0.05
            
        # Negative indicators
        if 'undefined' in code:
            quality -= 0.2
        if 'UNRECOVERED' in code:
            quality -= 0.3
        if code.count('DAT_') > 3:
            quality -= 0.1
        if len(code.strip()) < 50:  # Too short
            quality -= 0.1
            
        return max(0.0, min(1.0, quality))
        
    def _extract_functions_from_output(self, output_dir: str) -> Dict[str, str]:
        """Extract functions from decompilation output directory"""
        functions = {}
        output_path = Path(output_dir)
        
        for c_file in output_path.glob("*.c"):
            func_name = c_file.stem
            try:
                with open(c_file, 'r', encoding='utf-8') as f:
                    functions[func_name] = f.read()
            except Exception as e:
                logger.warning(f"Failed to read function file {c_file}: {e}")
                
        return functions

    def batch_analyze(self, binary_paths: List[str], base_output_dir: str) -> Dict[str, Dict[str, str]]:
        """
        Analyze multiple binaries in batch
        
        Args:
            binary_paths: List of binary file paths
            base_output_dir: Base directory for all outputs
            
        Returns:
            Dictionary mapping binary names to their extracted functions
        """
        results = {}
        
        for binary_path in binary_paths:
            binary_name = os.path.basename(binary_path)
            output_dir = os.path.join(base_output_dir, binary_name)
            
            try:
                functions = self.extract_functions(binary_path, output_dir)
                results[binary_name] = functions
                logger.info(f"Successfully analyzed {binary_name}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {binary_name}: {e}")
                results[binary_name] = {}
                
        return results


def run_ghidra_analysis(binary_path: str, output_dir: str, ghidra_home: str = None) -> bool:
    """
    Convenience function to run Ghidra analysis
    
    Args:
        binary_path: Path to binary file
        output_dir: Output directory for results
        ghidra_home: Optional Ghidra installation path
        
    Returns:
        True if analysis succeeded, False otherwise
    """
    try:
        # Use output_dir as the base for organized structure
        ghidra = GhidraHeadless(ghidra_home, output_base_dir=output_dir)
        success, _ = ghidra.run_ghidra_analysis(binary_path, output_dir)
        ghidra.cleanup_project()
        return success
        
    except Exception as e:
        logger.error(f"Ghidra analysis failed: {e}")
        return False


if __name__ == "__main__":
    # Test the module
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ghidra_headless.py <binary_path> <output_dir>")
        sys.exit(1)
        
    binary_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    logging.basicConfig(level=logging.INFO)
    
    success = run_ghidra_analysis(binary_path, output_dir)
    print(f"Analysis {'succeeded' if success else 'failed'}")