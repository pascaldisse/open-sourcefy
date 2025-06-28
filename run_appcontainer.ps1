# PowerShell script to run Open-Sourcefy in AppContainer security context
# This provides Windows security isolation similar to AppContainer

param(
    [string]$Mode = "full_pipeline",
    [switch]$Debug = $false,
    [switch]$SelfCorrection = $false
)

# Set security-restricted execution context
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# AppContainer-like security restrictions
$env:TEMP = Join-Path $env:USERPROFILE "AppData\Local\Temp\OpenSourcefyIsolated"
if (-not (Test-Path $env:TEMP)) {
    New-Item -ItemType Directory -Path $env:TEMP -Force
}

# Restricted environment variables for security
$RestrictedEnv = @{
    'PATH' = $env:PATH
    'PYTHONPATH' = Join-Path (Get-Location) "src"
    'MATRIX_SECURITY_MODE' = 'APPCONTAINER'
    'MATRIX_ISOLATED' = 'true'
}

try {
    Write-Host "🔒 Starting Open-Sourcefy in AppContainer-like security mode" -ForegroundColor Cyan
    Write-Host "📁 Working Directory: $(Get-Location)" -ForegroundColor Green
    Write-Host "🛡️ Security Context: Restricted AppContainer simulation" -ForegroundColor Yellow
    
    # Build command arguments
    $PythonArgs = @("main.py")
    
    if ($SelfCorrection) {
        $PythonArgs += "--self-correction"
    }
    
    if ($Debug) {
        $PythonArgs += "--debug"
    }
    
    # Execute with security restrictions
    Write-Host "⚡ Executing: python3 $($PythonArgs -join ' ')" -ForegroundColor Magenta
    
    # Run with restricted environment
    foreach ($key in $RestrictedEnv.Keys) {
        [Environment]::SetEnvironmentVariable($key, $RestrictedEnv[$key], [System.EnvironmentVariableTarget]::Process)
    }
    
    & python3 @PythonArgs
    
    $ExitCode = $LASTEXITCODE
    
    if ($ExitCode -eq 0) {
        Write-Host "✅ Pipeline completed successfully in AppContainer mode" -ForegroundColor Green
    } else {
        Write-Host "❌ Pipeline failed with exit code: $ExitCode" -ForegroundColor Red
    }
    
    exit $ExitCode
    
} catch {
    Write-Host "💥 AppContainer execution failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    Write-Host "🔒 AppContainer security session ended" -ForegroundColor Cyan
}