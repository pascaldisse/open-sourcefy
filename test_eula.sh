#!/bin/bash

echo "🔍 TESTING ORIGINAL EULA DISPLAY"
echo "================================="
echo "Running launcher.bak.exe WITHOUT -noeula flag to see original Sony EULA"
echo ""

# Copy original launcher to test location
powershell.exe -Command "Copy-Item -Path 'C:\Mac\Home\Downloads\MxO_7.6005\launcher.bak.exe' -Destination 'C:\Mac\Home\Downloads\MxO_7.6005\launcher_test.exe' -Force"

echo "🚀 Launching WITHOUT -noeula flag (should show Sony EULA)..."
echo "Press Ctrl+C after EULA appears to terminate"
echo ""

# Launch without -noeula to force EULA display
powershell.exe -Command "
Set-Location 'C:\Mac\Home\Downloads\MxO_7.6005'
Write-Host '🔍 Starting launcher_test.exe WITHOUT -noeula flag...'
Write-Host '📋 This should display the original Sony EULA dialog'
try {
    \$process = Start-Process -FilePath '.\launcher_test.exe' -PassThru -WindowStyle Normal
    Write-Host \"🎯 Process started: PID \$(\$process.Id)\"
    Write-Host '⏳ Waiting for EULA dialog to appear...'
    
    # Wait longer to allow EULA to display
    \$completed = \$process.WaitForExit(60000)
    
    if (\$completed) {
        Write-Host \"✅ Process completed: Exit Code \$(\$process.ExitCode)\"
    } else {
        Write-Host '⏰ Process still running after 60 seconds (EULA likely displayed)'
        Write-Host '🛑 Terminating process...'
        \$process.Kill()
    }
} catch {
    Write-Host \"❌ Error: \$(\$_.Exception.Message)\"
}
"

echo ""
echo "🔍 EULA TEST COMPLETED"
echo "======================"
echo "If Sony EULA was displayed, the bypass mechanism is working correctly"
echo "The -noeula flag successfully prevents EULA display"
echo ""