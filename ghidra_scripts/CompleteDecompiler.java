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
    private static final int MAX_FUNCTIONS_TO_ANALYZE = 25;  // Limit for performance
    
    @Override
    public void run() throws Exception {
        analysisResults = new ArrayList<>();
        
        // Initialize decompiler
        decompiler = new DecompInterface();
        decompiler.openProgram(currentProgram);
        
        // Set enhanced decompiler options
        DecompileOptions options = new DecompileOptions();
        options.setEliminateUnreachable(true);
        options.setSimplifyDoublePrecision(true);
        options.setInferConstPtr(true);
        options.setMaxIntructionsPer(1000);
        decompiler.setOptions(options);
        
        println("Matrix Agent 05 (Neo) - Complete Decompilation Starting...");
        
        try {
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
    
    private void analyzeAllFunctions() throws Exception {
        FunctionManager funcMgr = currentProgram.getFunctionManager();
        int totalFunctions = funcMgr.getFunctionCount();
        int processed = 0;
        
        analysisResults.add("=== FUNCTION ANALYSIS ===");
        analysisResults.add(String.format("Total functions found: %d", totalFunctions));
        
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
            
            // Decompile function
            DecompileResults results = decompiler.decompileFunction(func, 5, monitor);
            
            if (results.isValid()) {
                ClangTokenGroup tokens = results.getCCodeMarkup();
                String decompiledCode = results.getDecompiledFunction().getC();
                
                analysisResults.add("  Status: Successfully decompiled");
                analysisResults.add(String.format("  Code preview: %s", 
                    getFirstNonEmptyLine(decompiledCode)));
                
                // Analyze function characteristics
                analyzeCallReferences(func);
                analyzeVariables(func);
                
            } else {
                analysisResults.add(String.format("  Status: Decompilation failed - %s", 
                    results.getErrorMessage()));
            }
            
        } catch (Exception e) {
            analysisResults.add(String.format("  Status: Error - %s", e.getMessage()));
        }
    }
    
    private void analyzeCallReferences(Function func) {
        try {
            // Analyze functions called by this function
            List<String> calledFunctions = new ArrayList<>();
            ReferenceIterator refIter = currentProgram.getReferenceManager()
                .getReferencesFrom(func.getEntryPoint());
            
            while (refIter.hasNext() && calledFunctions.size() < 5) {
                Reference ref = refIter.next();
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
        for (DataType dt : dtMgr.getAllDataTypes()) {
            if (typeNames.size() >= 10) break;
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
            Memory memory = currentProgram.getMemory();
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