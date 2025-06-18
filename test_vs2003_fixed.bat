@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio .NET 2003\Common7\Tools\vsvars32.bat"
cd /d "C:\Users\pascaldisse\Downloads\open-sourcefy\output\launcher\20250618-181811\compilation"
echo TEST: VS2003 linking with DRIVER:NO option removed...
echo Step 1: Compiling main.c to main.obj...
cl.exe /nologo /W0 /wd4047 /wd4024 /wd4133 /wd4002 /wd4020 /wd4013 /wd2005 /wd2084 /TC /Zp1 /Od /D_CRT_SECURE_NO_WARNINGS /D__ALLOW_DUPLICATE_FUNCTIONS__ /D__IGNORE_REDEFINITION_ERRORS__ /c src\main.c src\assembly_stubs.c src\assembly_globals.c src\memory_layout.c src\control_flow.c src\winmain_wrapper.c src\exception_handling.c
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to compile main.c
    exit /b %ERRORLEVEL%
)
echo Step 2: Creating assembly_stubs.lib for symbol resolution...
lib.exe /OUT:assembly_stubs.lib assembly_stubs.obj
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create assembly_stubs.lib
    exit /b %ERRORLEVEL%
)
echo Step 3: Linking object files WITHOUT /DRIVER:NO option...
cl.exe /Fe"launcher.exe" main.obj /link /ENTRY:WinMainCRTStartup /SUBSYSTEM:WINDOWS /FORCE:MULTIPLE /IGNORE:4006 /IGNORE:4088 /IGNORE:4217 /SECTION:.data,RW /SECTION:.idata,R /ALIGN:0x1000 /STACK:0x200000 /HEAP:0x200000 /FILEALIGN:0x1000 assembly_stubs.obj assembly_globals.obj memory_layout.obj control_flow.obj winmain_wrapper.obj exception_handling.obj assembly_stubs.lib user32.lib kernel32.lib gdi32.lib advapi32.lib shell32.lib ole32.lib comdlg32.lib wininet.lib version.lib comctl32.lib winspool.lib ws2_32.lib mpr.lib netapi32.lib userenv.lib psapi.lib dbghelp.lib imagehlp.lib rpcrt4.lib setupapi.lib msvcrt.lib oldnames.lib
echo Step 4: Checking if launcher.exe was created...
if exist launcher.exe (
    echo SUCCESS: launcher.exe created successfully!
    dir launcher.exe
) else (
    echo FAILURE: launcher.exe was not created!
)