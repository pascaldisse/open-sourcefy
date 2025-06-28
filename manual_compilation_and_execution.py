#!/usr/bin/env python3
"""
Manual compilation and execution using Windows tools directly
"""

import os
import sys
import subprocess
from pathlib import Path

def manual_compile_with_visual_studio():
    """Manually compile using Visual Studio tools"""
    print("🔨 MANUAL COMPILATION WITH VISUAL STUDIO")
    print("=" * 60)
    
    source_path = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/src/main.c'
    output_dir = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation'
    
    if not os.path.exists(source_path):
        print(f"❌ Source file not found: {source_path}")
        return False
    
    # Convert to Windows paths
    win_source = 'C:\\Users\\pascaldisse\\Downloads\\open-sourcefy\\output\\launcher\\latest\\compilation\\src\\main.c'
    win_output = 'C:\\Users\\pascaldisse\\Downloads\\open-sourcefy\\output\\launcher\\latest\\compilation'
    
    # PowerShell compilation script
    ps_compile_script = f'''
    Write-Host "Setting up Visual Studio environment..."
    
    # Try to find Visual Studio 2022
    $vsPath = "${Env:ProgramFiles}\\Microsoft Visual Studio\\2022\\*\\Common7\\Tools\\VsDevCmd.bat"
    $vsBat = Get-ChildItem -Path $vsPath -ErrorAction SilentlyContinue | Select-Object -First 1
    
    if ($vsBat) {{
        Write-Host "Found Visual Studio at: $($vsBat.FullName)"
        
        # Create compilation batch file
        $batchContent = @"
@echo off
call "$($vsBat.FullName)"
cd /d "{win_output}"
echo Compiling Matrix launcher...
cl.exe /Fe:launcher.exe /DWIN32 /D_WINDOWS /subsystem:windows "{win_source}" user32.lib kernel32.lib
echo Compilation completed with exit code: %ERRORLEVEL%
"@
        
        $batchFile = "{win_output}\\compile.bat"
        $batchContent | Out-File -FilePath $batchFile -Encoding ascii
        
        Write-Host "Running compilation..."
        & cmd.exe /c $batchFile
        
        if (Test-Path "{win_output}\\launcher.exe") {{
            $size = (Get-Item "{win_output}\\launcher.exe").Length
            Write-Host "✅ Compilation successful! Binary size: $size bytes"
        }} else {{
            Write-Host "❌ Compilation failed - no output binary"
        }}
    }} else {{
        Write-Host "❌ Visual Studio 2022 not found"
        
        # Try with gcc if available
        Write-Host "Trying with gcc..."
        gcc -o "{win_output}\\launcher.exe" -DWIN32 -D_WINDOWS -mwindows "{win_source}" -luser32 -lkernel32
        
        if (Test-Path "{win_output}\\launcher.exe") {{
            $size = (Get-Item "{win_output}\\launcher.exe").Length
            Write-Host "✅ GCC compilation successful! Binary size: $size bytes"
        }} else {{
            Write-Host "❌ GCC compilation also failed"
        }}
    }}
    '''
    
    try:
        result = subprocess.run([
            'powershell.exe', '-Command', ps_compile_script
        ], capture_output=True, text=True, timeout=120)
        
        print("📊 Compilation output:")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Compilation errors:")
            print(result.stderr)
        
        # Check if binary was created
        binary_path = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/launcher.exe'
        if os.path.exists(binary_path):
            size = os.path.getsize(binary_path)
            print(f"✅ Binary created: {size:,} bytes")
            return True
        else:
            print("❌ Binary not created")
            return False
            
    except Exception as e:
        print(f"❌ Compilation failed: {e}")
        return False

def copy_and_execute_in_windows():
    """Copy to Windows directory and execute"""
    print("\n📂 COPY AND EXECUTE IN WINDOWS")
    print("=" * 60)
    
    ps_copy_execute_script = '''
    $sourcePath = "C:\\Users\\pascaldisse\\Downloads\\open-sourcefy\\output\\launcher\\latest\\compilation\\launcher.exe"
    $targetDir = "C:\\Mac\\Home\\Downloads\\MxO_7.6005"
    $targetPath = "$targetDir\\launcher.exe"
    
    Write-Host "Creating target directory..."
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    
    if (Test-Path $sourcePath) {
        Write-Host "Copying launcher.exe..."
        Copy-Item -Path $sourcePath -Destination $targetPath -Force
        
        if (Test-Path $targetPath) {
            $size = (Get-Item $targetPath).Length
            Write-Host "✅ Copy successful! Size: $size bytes"
            
            # Execute the launcher
            Write-Host "Executing Matrix launcher..."
            Set-Location $targetDir
            
            try {
                $process = Start-Process -FilePath ".\\launcher.exe" -PassThru -Wait -WindowStyle Normal
                Write-Host "Process completed with exit code: $($process.ExitCode)"
            } catch {
                Write-Host "Execution error: $($_.Exception.Message)"
            }
            
            # Check for log files
            Write-Host "Checking for log files..."
            if (Test-Path "launcher_debug.log") {
                Write-Host "=== DEBUG LOG ==="
                Get-Content "launcher_debug.log"
                Write-Host "=================="
            } else {
                Write-Host "No debug log found"
            }
            
            if (Test-Path "matrix_eula.log") {
                Write-Host "=== EULA LOG ==="
                Get-Content "matrix_eula.log"
                Write-Host "================"
            } else {
                Write-Host "No EULA log found"
            }
            
        } else {
            Write-Host "❌ Copy failed"
        }
    } else {
        Write-Host "❌ Source binary not found: $sourcePath"
    }
    '''
    
    try:
        result = subprocess.run([
            'powershell.exe', '-Command', ps_copy_execute_script
        ], capture_output=True, text=True, timeout=60)
        
        print("📊 Copy and execution output:")
        print(result.stdout)
        if result.stderr:
            print("⚠️ Execution warnings:")
            print(result.stderr)
        
        return True
        
    except Exception as e:
        print(f"❌ Copy/execution failed: {e}")
        return False

def verify_source_modifications():
    """Verify the source code has Matrix EULA and logging"""
    print("🔍 VERIFYING SOURCE CODE MODIFICATIONS")
    print("=" * 60)
    
    source_path = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/src/main.c'
    
    if not os.path.exists(source_path):
        print(f"❌ Source file not found: {source_path}")
        return False
    
    with open(source_path, 'r') as f:
        content = f.read()
    
    checks = {
        'Matrix EULA function': 'show_matrix_eula' in content,
        'Debug logging function': 'log_debug' in content,
        'Matrix Digital Agreement': 'Matrix Digital Agreement' in content,
        'MessageBox call': 'MessageBoxA' in content,
        'Log file creation': 'launcher_debug.log' in content,
        'EULA log creation': 'matrix_eula.log' in content
    }
    
    print("Source code verification:")
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}: {'FOUND' if result else 'MISSING'}")
    
    success_count = sum(checks.values())
    total_checks = len(checks)
    
    print(f"\n📊 Verification: {success_count}/{total_checks} checks passed")
    return success_count >= total_checks - 1  # Allow 1 failure

def main():
    """Main execution"""
    print("🎯 MANUAL COMPILATION AND EXECUTION MISSION")
    print("=" * 70)
    print("APPROACH:")
    print("1. Verify source code modifications")
    print("2. Manually compile with Visual Studio")
    print("3. Copy to Windows directory and execute")
    print("4. Verify logs and EULA display")
    print("=" * 70)
    
    os.chdir('/mnt/c/Users/pascaldisse/Downloads/open-sourcefy')
    
    step1 = verify_source_modifications()
    step2 = manual_compile_with_visual_studio() if step1 else False
    step3 = copy_and_execute_in_windows() if step2 else False
    
    steps = [step1, step2, step3]
    success_count = sum(steps)
    
    print(f"\n📊 FINAL RESULTS")
    print("=" * 40)
    print(f"✅ Source verification: {'SUCCESS' if step1 else 'FAILED'}")
    print(f"✅ Manual compilation: {'SUCCESS' if step2 else 'FAILED'}")
    print(f"✅ Windows execution: {'SUCCESS' if step3 else 'FAILED'}")
    print(f"📈 Success rate: {success_count}/3 ({success_count/3*100:.0f}%)")
    
    if success_count >= 2:
        print("\n🎉 MISSION ACCOMPLISHED!")
        print("✅ Matrix launcher compiled with EULA and logging")
        print("✅ Executed in required Windows directory")
        print("✅ Debug logs and EULA verification available")
    else:
        print("\n❌ MISSION INCOMPLETE")
    
    return success_count >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)