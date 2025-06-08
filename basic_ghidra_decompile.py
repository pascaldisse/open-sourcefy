#!/usr/bin/env python3
"""
Basic Ghidra Decompilation Script
Direct Ghidra headless analysis without the Matrix agent system
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def find_ghidra():
    """Find Ghidra installation"""
    # Check project ghidra directory
    project_ghidra = Path("ghidra")
    if project_ghidra.exists():
        print(f"✅ Found Ghidra in project: {project_ghidra}")
        return project_ghidra
    
    # Check environment variable
    ghidra_env = os.environ.get('GHIDRA_INSTALL_DIR')
    if ghidra_env and Path(ghidra_env).exists():
        print(f"✅ Found Ghidra from environment: {ghidra_env}")
        return Path(ghidra_env)
    
    # Check common installation paths
    common_paths = [
        Path.home() / "ghidra",
        Path("/opt/ghidra"),
        Path("C:/ghidra"),
        Path("C:/Program Files/ghidra"),
    ]
    
    for path in common_paths:
        if path.exists():
            print(f"✅ Found Ghidra at: {path}")
            return path
    
    print("❌ Ghidra not found in any standard location")
    return None

def run_ghidra_headless(ghidra_path, binary_path, output_dir):
    """Run Ghidra headless analysis"""
    
    # Find analyzeHeadless script
    analyze_script = None
    possible_scripts = [
        ghidra_path / "support" / "analyzeHeadless",
        ghidra_path / "support" / "analyzeHeadless.bat",
        ghidra_path / "analyzeHeadless",
        ghidra_path / "analyzeHeadless.bat"
    ]
    
    for script in possible_scripts:
        if script.exists():
            analyze_script = script
            break
    
    if not analyze_script:
        print("❌ analyzeHeadless script not found")
        return False
    
    print(f"🔧 Using analyzeHeadless: {analyze_script}")
    
    # Create project directory
    project_dir = output_dir / "ghidra_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create decompiled output directory
    decompiled_dir = output_dir / "decompiled"
    decompiled_dir.mkdir(exist_ok=True)
    
    # Build Ghidra command
    cmd = [
        str(analyze_script),
        str(project_dir),           # Project directory
        "TempProject",              # Project name
        "-import", str(binary_path), # Import binary
        "-scriptPath", str(decompiled_dir),  # Script output path
        "-postScript", "DecompileAll.java",  # Run decompile script
        "-deleteProject"            # Clean up project after
    ]
    
    print(f"🚀 Running Ghidra command:")
    print(f"   {' '.join(cmd)}")
    
    # Set up environment
    env = os.environ.copy()
    env['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
    
    try:
        # Run Ghidra headless
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=env
        )
        
        print(f"📊 Ghidra exit code: {result.returncode}")
        
        if result.stdout:
            print("📝 Ghidra output:")
            print(result.stdout[-1000:])  # Last 1000 chars
        
        if result.stderr:
            print("⚠️ Ghidra errors:")
            print(result.stderr[-1000:])  # Last 1000 chars
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Ghidra analysis timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to run Ghidra: {e}")
        return False

def create_decompile_script(output_dir):
    """Create a simple DecompileAll.java script"""
    script_content = '''
//Decompiles all functions and saves them to files
//@author Generated
//@category Analysis
//@keybinding
//@menupath
//@toolbar

import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.*;
import ghidra.app.decompiler.*;
import ghidra.util.task.TaskMonitor;
import java.io.FileWriter;
import java.io.IOException;

public class DecompileAll extends GhidraScript {

    @Override
    public void run() throws Exception {
        
        println("Starting decompilation of all functions...");
        
        // Get the current program
        Program program = getCurrentProgram();
        if (program == null) {
            println("No program loaded");
            return;
        }
        
        // Setup decompiler
        DecompInterface decompInterface = new DecompInterface();
        decompInterface.openProgram(program);
        
        // Get all functions
        FunctionManager functionManager = program.getFunctionManager();
        FunctionIterator functions = functionManager.getFunctions(true);
        
        StringBuilder allCode = new StringBuilder();
        allCode.append("// Decompiled code from " + program.getName() + "\\n\\n");
        allCode.append("#include <stdio.h>\\n");
        allCode.append("#include <stdlib.h>\\n");
        allCode.append("#include <string.h>\\n\\n");
        
        int functionCount = 0;
        
        while (functions.hasNext() && !monitor.isCancelled()) {
            Function function = functions.next();
            
            println("Decompiling function: " + function.getName());
            
            try {
                DecompileResults results = decompInterface.decompileFunction(function, 30, monitor);
                if (results != null && results.decompileCompleted()) {
                    String decompiledCode = results.getDecompiledFunction().getC();
                    allCode.append("// Function: " + function.getName() + "\\n");
                    allCode.append(decompiledCode);
                    allCode.append("\\n\\n");
                    functionCount++;
                } else {
                    allCode.append("// Failed to decompile function: " + function.getName() + "\\n\\n");
                }
            } catch (Exception e) {
                allCode.append("// Error decompiling function " + function.getName() + ": " + e.getMessage() + "\\n\\n");
            }
        }
        
        // Save to file in the same directory as the script
        try (FileWriter writer = new FileWriter("decompiled_code.c")) {
            writer.write(allCode.toString());
            println("Saved decompiled code to: decompiled_code.c");
        } catch (IOException e) {
            println("Failed to save decompiled code: " + e.getMessage());
        }
        
        println("Decompilation complete. Processed " + functionCount + " functions.");
        
        decompInterface.dispose();
    }
}
'''
    
    # Save script to project directory for Ghidra to find
    script_file = output_dir / "DecompileAll.java"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    print(f"📝 Created decompile script: {script_file}")
    return script_file

def main():
    """Main decompilation function"""
    print("🎬 Basic Ghidra Decompilation")
    
    # Check for binary
    binary_path = Path("input/launcher.exe")
    if not binary_path.exists():
        print(f"❌ Binary not found: {binary_path}")
        return False
    
    print(f"📁 Target binary: {binary_path} ({binary_path.stat().st_size} bytes)")
    
    # Find Ghidra
    ghidra_path = find_ghidra()
    if not ghidra_path:
        print("❌ Cannot proceed without Ghidra")
        return False
    
    # Create output directory
    output_dir = Path("output") / f"basic_decompile_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Create decompile script
    script_file = create_decompile_script(output_dir)
    
    # Copy script to decompiled directory for Ghidra to find
    decompiled_dir = output_dir / "decompiled"
    decompiled_dir.mkdir(exist_ok=True)
    import shutil
    shutil.copy2(script_file, decompiled_dir / "DecompileAll.java")
    
    # Run Ghidra decompilation
    print("🚀 Starting Ghidra headless analysis...")
    success = run_ghidra_headless(ghidra_path, binary_path, output_dir)
    
    if success:
        print("🎉 Ghidra decompilation completed!")
        
        # Check for output files
        decompiled_dir = output_dir / "decompiled"
        if decompiled_dir.exists():
            c_files = list(decompiled_dir.glob("*.c"))
            if c_files:
                print(f"📝 Generated C files:")
                for c_file in c_files:
                    size = c_file.stat().st_size
                    print(f"   - {c_file.name} ({size} bytes)")
        
        return True
    else:
        print("💥 Ghidra decompilation failed")
        return False

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        success = main()
        elapsed = time.time() - start_time
        
        if success:
            print(f"\n✅ Decompilation completed in {elapsed:.1f}s")
            sys.exit(0)
        else:
            print(f"\n❌ Decompilation failed after {elapsed:.1f}s")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)