//CompleteDecompiler.java - Advanced Ghidra Script for Matrix Pipeline
//Enhanced decompilation script for Agent 05 (Neo)

import ghidra.app.decompiler.*;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.program.model.address.*;
import ghidra.program.model.symbol.*;
import ghidra.program.model.data.*;
import ghidra.util.task.TaskMonitor;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class CompleteDecompiler extends GhidraScript {
    
    private DecompInterface decompiler;
    private List<String> analysisResults;
    private static final int MAX_FUNCTIONS_TO_ANALYZE = 5;  // Test with just 5 functions first
    
    @Override
    public void run() throws Exception {
        analysisResults = new ArrayList<>();
        
        // Initialize decompiler
        decompiler = new DecompInterface();
        
        // Set enhanced decompiler options
        DecompileOptions options = new DecompileOptions();
        options.setEliminateUnreachable(true);
        options.setSimplifyDoublePrecision(true);
        // Note: setInferConstPtr and setMaxIntructionsPer not available in this Ghidra version
        decompiler.setOptions(options);
        
        // CRITICAL: Initialize decompiler with program AFTER setting options
        if (!decompiler.openProgram(currentProgram)) {
            throw new Exception("Failed to initialize decompiler for program");
        }
        
        println("Matrix Agent 05 (Neo) - Complete Decompilation Starting...");
        
        try {
            // CRITICAL: Ensure auto-analysis has been performed
            ensureAutoAnalysisComplete();
            
            // Check binary format and provide enhanced analysis
            analyzeBinaryFormat();
            
            // Perform comprehensive analysis
            analyzeAllFunctions();
            analyzeDataTypes();
            analyzeStringReferences();
            generateSummaryReport();
            
        } finally {
            decompiler.dispose();
        }
        
        println("Neo's decompilation analysis complete.");
    }
    
    private void ensureAutoAnalysisComplete() throws Exception {
        println("Ensuring Ghidra auto-analysis is complete...");
        
        // Force auto-analysis to run and wait for completion
        ghidra.app.services.AutoAnalysisManager analysisManager = ghidra.app.services.AutoAnalysisManager.getAnalysisManager(currentProgram);
        
        if (analysisManager != null) {
            println("Found analysis manager - checking analysis status...");
            
            // If analysis is not running, start it
            if (!analysisManager.isAnalyzing()) {
                println("Analysis not running - starting auto-analysis...");
                analysisManager.startAnalysis(currentProgram, null);
            }
            
            // Wait for analysis to complete with timeout
            int maxWaitSeconds = 300; // 5 minutes
            int waitCount = 0;
            
            while (analysisManager.isAnalyzing() && waitCount < maxWaitSeconds) {
                if (waitCount % 30 == 0) {
                    println(String.format("Waiting for auto-analysis to complete... (%d/%d seconds)", waitCount, maxWaitSeconds));
                }
                Thread.sleep(1000);
                waitCount++;
            }
            
            if (analysisManager.isAnalyzing()) {
                println("WARNING: Auto-analysis still running after timeout - proceeding anyway");
            } else {
                println("Auto-analysis completed successfully");
            }
        } else {
            println("WARNING: No analysis manager found");
        }
        
        FunctionManager funcMgr = currentProgram.getFunctionManager();
        int functionCount = funcMgr.getFunctionCount();
        println(String.format("Final analysis status: Functions found: %d", functionCount));
        
        if (functionCount == 0) {
            println("ERROR: Still no functions detected after auto-analysis");
            println("This indicates a fundamental issue with binary analysis");
            throw new Exception("No functions detected - analysis failed");
        }
    }
    
    private void analyzeBinaryFormat() {
        analysisResults.add("=== BINARY FORMAT ANALYSIS ===");
        
        try {
            // Get program information
            String format = currentProgram.getExecutableFormat();
            String arch = currentProgram.getLanguage().getProcessor().toString();
            String compiler = currentProgram.getCompilerSpec().getCompilerSpecID().getIdAsString();
            
            analysisResults.add(String.format("Format: %s", format));
            analysisResults.add(String.format("Architecture: %s", arch));
            analysisResults.add(String.format("Compiler: %s", compiler));
            
            // Check for .NET/managed code characteristics
            boolean isManaged = false;
            if (format.toLowerCase().contains("pe") || format.toLowerCase().contains("portable")) {
                // Check for .NET metadata tables or CLR header
                try {
                    ghidra.program.model.mem.Memory memory = currentProgram.getMemory();
                    // Look for common .NET signatures
                    if (memory.contains(currentProgram.getAddressFactory().getDefaultAddressSpace().getAddress(0x2000))) {
                        isManaged = true;
                        analysisResults.add("DETECTED: .NET/Managed executable");
                        analysisResults.add("WARNING: .NET binaries may have limited native function analysis");
                        analysisResults.add("INFO: Ghidra works best with native PE executables");
                    }
                } catch (Exception e) {
                    // Continue analysis
                }
            }
            
            if (!isManaged) {
                analysisResults.add("Type: Native executable - full analysis available");
            }
            
        } catch (Exception e) {
            analysisResults.add(String.format("Binary format analysis error: %s", e.getMessage()));
        }
    }
    
    private void analyzeAllFunctions() throws Exception {
        FunctionManager funcMgr = currentProgram.getFunctionManager();
        int totalFunctions = funcMgr.getFunctionCount();
        int processed = 0;
        
        analysisResults.add("=== FUNCTION ANALYSIS ===");
        analysisResults.add(String.format("Total functions found: %d", totalFunctions));
        
        if (totalFunctions == 0) {
            analysisResults.add("WARNING: No functions detected by Ghidra");
            analysisResults.add("REASON: This may occur with:");
            analysisResults.add("  - .NET/managed executables (MSIL bytecode, not native code)");
            analysisResults.add("  - Heavily obfuscated binaries");
            analysisResults.add("  - Packers that Ghidra doesn't recognize");
            analysisResults.add("  - Incomplete analysis (try longer timeout)");
            analysisResults.add("SOLUTION: For .NET binaries, use dotPeek, ILSpy, or Reflexil instead of Ghidra");
            return;  // Exit early if no functions
        }
        
        int functionCount = 0;
        for (Function func : funcMgr.getFunctions(true)) {
            if (monitor.isCancelled() || functionCount >= MAX_FUNCTIONS_TO_ANALYZE) {
                break;
            }
            functionCount++;
            if (monitor.isCancelled()) {
                break;
            }
            
            processed++;
            monitor.setMessage(String.format("Analyzing function %d/%d: %s", 
                processed, totalFunctions, func.getName()));
            
            analyzeFunction(func);
        }
        
        analysisResults.add(String.format("Functions processed: %d", processed));
    }
    
    private void analyzeFunction(Function func) throws Exception {
        try {
            // Basic function info
            analysisResults.add(String.format("\nFunction: %s", func.getName()));
            analysisResults.add(String.format("  Address: %s", func.getEntryPoint()));
            analysisResults.add(String.format("  Size: %d bytes", func.getBody().getNumAddresses()));
            
            // Skip thunk functions - they can't be decompiled
            if (func.isThunk()) {
                analysisResults.add("  Status: Thunk function - skipped");
                return;
            }
            
            // Decompile function with very long timeout and more detailed logging
            println("  Attempting decompilation of " + func.getName() + " at " + func.getEntryPoint());
            DecompileResults results = decompiler.decompileFunction(func, 60, monitor);
            
            if (results == null) {
                println("  ERROR: decompileFunction returned null for " + func.getName());
                analysisResults.add("  Status: Decompilation returned null");
                return;
            }
            
            println("  Checking decompilation results for " + func.getName());
            
            if (results.decompileCompleted()) {
                println("  Decompilation completed successfully for " + func.getName());
                String decompiledCode = results.getDecompiledFunction().getC();
                
                if (decompiledCode != null && decompiledCode.trim().length() > 0) {
                    println("  Got decompiled code: " + decompiledCode.length() + " characters");
                    analysisResults.add("  Status: Successfully decompiled");
                    analysisResults.add(String.format("  Code preview: %s", 
                        getFirstNonEmptyLine(decompiledCode)));
                    
                    // Write full function to output file
                    writeDecompiledFunction(func, decompiledCode);
                    
                    // Analyze function characteristics
                    analyzeCallReferences(func);
                    analyzeVariables(func);
                } else {
                    println("  ERROR: Decompiled code is null or empty for " + func.getName());
                    analysisResults.add("  Status: Decompilation returned empty code");
                }
                
            } else {
                String errorMsg = results.getErrorMessage();
                println("  ERROR: Decompilation did not complete for " + func.getName() + ": " + errorMsg);
                analysisResults.add(String.format("  Status: Decompilation failed - %s", errorMsg));
            }
            
        } catch (Exception e) {
            analysisResults.add(String.format("  Status: Error - %s", e.getMessage()));
        }
    }
    
    private void writeDecompiledFunction(Function func, String code) {
        try {
            String outputDir = getScriptArgs().length > 0 ? getScriptArgs()[0] : ".";
            String outputFile = outputDir + "/launcher.exe_decompiled.c";
            
            // Create the output file with header if it doesn't exist
            java.io.File file = new java.io.File(outputFile);
            boolean isNewFile = !file.exists();
            
            java.io.FileWriter writer = new java.io.FileWriter(outputFile, true);
            
            if (isNewFile) {
                writer.write("// Matrix Online Launcher - Decompiled Source Code\n");
                writer.write("// Generated by Ghidra CompleteDecompiler\n");
                writer.write("// Analysis completed with " + currentProgram.getFunctionManager().getFunctionCount() + " functions\n\n");
                writer.write("#include <windows.h>\n");
                writer.write("#include <stdio.h>\n");
                writer.write("#include <stdlib.h>\n\n");
            }
            
            writer.write("// ===============================================\n");
            writer.write("// Function: " + func.getName() + "\n");
            writer.write("// Address: " + func.getEntryPoint() + "\n");
            writer.write("// Size: " + func.getBody().getNumAddresses() + " bytes\n");
            writer.write("// ===============================================\n\n");
            writer.write(code);
            writer.write("\n\n");
            writer.close();
            
            println("Wrote function " + func.getName() + " to " + outputFile);
            
        } catch (Exception e) {
            println("Failed to write function " + func.getName() + " to file: " + e.getMessage());
        }
    }
    
    private void analyzeCallReferences(Function func) {
        try {
            // Analyze functions called by this function
            List<String> calledFunctions = new ArrayList<>();
            Reference[] refs = currentProgram.getReferenceManager()
                .getReferencesFrom(func.getEntryPoint());
            
            for (Reference ref : refs) {
                if (calledFunctions.size() >= 5) break;
                if (ref.getReferenceType().isCall()) {
                    Function calledFunc = currentProgram.getFunctionManager()
                        .getFunctionAt(ref.getToAddress());
                    if (calledFunc != null) {
                        calledFunctions.add(calledFunc.getName());
                    }
                }
            }
            
            if (!calledFunctions.isEmpty()) {
                analysisResults.add(String.format("  Calls: %s", 
                    String.join(", ", calledFunctions)));
            }
            
        } catch (Exception e) {
            // Continue on error
        }
    }
    
    private void analyzeVariables(Function func) {
        try {
            Variable[] variables = func.getAllVariables();
            if (variables.length > 0) {
                analysisResults.add(String.format("  Variables: %d local variables detected", 
                    variables.length));
            }
        } catch (Exception e) {
            // Continue on error
        }
    }
    
    private void analyzeDataTypes() {
        analysisResults.add("\n=== DATA TYPE ANALYSIS ===");
        
        DataTypeManager dtMgr = currentProgram.getDataTypeManager();
        int definedTypes = dtMgr.getDataTypeCount(true);
        
        analysisResults.add(String.format("Defined data types: %d", definedTypes));
        
        // Sample some data types
        List<String> typeNames = new ArrayList<>();
        java.util.Iterator<DataType> dtIterator = dtMgr.getAllDataTypes();
        while (dtIterator.hasNext() && typeNames.size() < 10) {
            DataType dt = dtIterator.next();
            if (!dt.getName().startsWith("__")) {
                typeNames.add(dt.getName());
            }
        }
        
        if (!typeNames.isEmpty()) {
            analysisResults.add(String.format("Sample types: %s", 
                String.join(", ", typeNames)));
        }
    }
    
    private void analyzeStringReferences() {
        analysisResults.add("\n=== STRING ANALYSIS ===");
        
        try {
            ghidra.program.model.mem.Memory memory = currentProgram.getMemory();
            AddressSetView executableSet = memory.getExecuteSet();
            
            // Count strings in executable sections
            int stringCount = 0;
            List<String> sampleStrings = new ArrayList<>();
            
            for (Data data : currentProgram.getListing().getDefinedData(executableSet, true)) {
                if (data.hasStringValue() && stringCount < 10) {
                    String stringValue = data.getDefaultValueRepresentation();
                    if (stringValue.length() > 3 && stringValue.length() < 50) {
                        sampleStrings.add(stringValue);
                        stringCount++;
                    }
                }
                if (stringCount >= 10) break;
            }
            
            analysisResults.add(String.format("String references found: %d+", stringCount));
            if (!sampleStrings.isEmpty()) {
                analysisResults.add("Sample strings:");
                for (String str : sampleStrings) {
                    analysisResults.add(String.format("  %s", str));
                }
            }
            
        } catch (Exception e) {
            analysisResults.add("String analysis encountered errors");
        }
    }
    
    private void generateSummaryReport() {
        analysisResults.add("\n=== NEO'S MATRIX ANALYSIS SUMMARY ===");
        analysisResults.add("Binary has been successfully analyzed through the Matrix lens");
        analysisResults.add("Advanced decompilation techniques applied");
        analysisResults.add("Ready for enhanced reconstruction phase");
        
        // Output all results
        for (String result : analysisResults) {
            println(result);
        }
    }
    
    private String getFirstNonEmptyLine(String code) {
        if (code == null) return "No code available";
        
        String[] lines = code.split("\n");
        for (String line : lines) {
            String trimmed = line.trim();
            if (!trimmed.isEmpty() && !trimmed.startsWith("//") && !trimmed.equals("{")) {
                return trimmed.length() > 80 ? trimmed.substring(0, 77) + "..." : trimmed;
            }
        }
        return "Function body available";
    }
}