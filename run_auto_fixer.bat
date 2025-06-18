@echo off
REM Automated Pipeline Fixer Launcher for Windows
REM Runs the continuous pipeline fixer with proper environment setup

echo ========================================================================
echo OPEN-SOURCEFY AUTOMATED PIPELINE FIXER
echo ZERO TOLERANCE MODE - WILL NOT STOP UNTIL PIPELINE IS FIXED
echo ========================================================================

REM Get script directory and change to it
cd /d "%~dp0"

REM Check if we're in the right directory
if not exist "main.py" (
    echo ❌ ERROR: main.py not found in current directory
    echo    Make sure you're running this from the open-sourcefy project root
    pause
    exit /b 1
)

if not exist "rules.md" (
    echo ❌ ERROR: rules.md not found in current directory
    echo    Make sure you're running this from the open-sourcefy project root
    pause
    exit /b 1
)

if not exist "CLAUDE.md" (
    echo ❌ ERROR: CLAUDE.md not found in current directory
    echo    Make sure you're running this from the open-sourcefy project root
    pause
    exit /b 1
)

REM Check for virtual environment
set VENV_PATH=
if exist "..\..\venv" (
    set VENV_PATH=..\..\venv
) else if exist "..\venv" (
    set VENV_PATH=..\venv
) else if exist "venv" (
    set VENV_PATH=venv
) else (
    echo ❌ ERROR: Virtual environment not found
    echo    Expected locations: ..\..\venv, ..\venv, or .\venv
    pause
    exit /b 1
)

echo ✅ Found virtual environment at: %VENV_PATH%

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

REM Verify claude-code-sdk is installed
python -c "import claude_code_sdk" 2>nul
if errorlevel 1 (
    echo ❌ ERROR: claude-code-sdk not found in virtual environment
    echo    Please install it with: pip install claude-code-sdk
    pause
    exit /b 1
)

echo ✅ Claude Code SDK found

REM Initialize git repository if needed
if not exist ".git" (
    echo 🔧 Initializing git repository...
    git init
    git add .
    git commit -m "Initial commit for automated pipeline fixing"
)

echo ✅ Git repository ready

REM Check for API key
if "%ANTHROPIC_API_KEY%"=="" (
    echo ⚠️  WARNING: ANTHROPIC_API_KEY not set
    echo    The Claude Code SDK may not function without an API key
    echo    Set it with: set ANTHROPIC_API_KEY=your_key_here
)

REM Create logs directory
if not exist "logs" mkdir logs

echo ========================================================================
echo 🚀 STARTING AUTOMATED PIPELINE FIXER
echo    Working directory: %CD%
echo    Virtual environment: %VENV_PATH%
echo    Git worktree support: ENABLED
echo    Zero tolerance mode: ACTIVE
echo ========================================================================

REM Run the automated pipeline fixer
echo ⚡ Launching auto_pipeline_fixer.py...
echo.

REM Run with python buffering disabled for real-time logs
set PYTHONUNBUFFERED=1
python auto_pipeline_fixer.py "%CD%" 2>&1 | tee "logs\auto_fixer_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log"

echo.
echo ========================================================================
echo AUTOMATED PIPELINE FIXER COMPLETED
echo Check the logs above for results
echo ========================================================================
pause