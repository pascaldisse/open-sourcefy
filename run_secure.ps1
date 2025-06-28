# AppContainer Security Execution for Open-Sourcefy
Write-Host "🔒 Starting AppContainer Security Mode" -ForegroundColor Cyan

# Set security environment
$env:MATRIX_SECURITY_MODE = 'APPCONTAINER'
$env:MATRIX_ISOLATED = 'true'

Write-Host "⚡ Executing pipeline with security restrictions..." -ForegroundColor Magenta

try {
    python3 main.py
    Write-Host "✅ AppContainer execution completed" -ForegroundColor Green
} catch {
    Write-Host "❌ AppContainer execution failed" -ForegroundColor Red
}