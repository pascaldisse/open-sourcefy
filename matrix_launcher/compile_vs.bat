@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Preview\Common7\Tools\VsDevCmd.bat" > nul 2>&1
echo Compiling Matrix launcher with Visual Studio...
cl.exe /Fe:matrix_launcher.exe /DWIN32 /D_WINDOWS /subsystem:windows "C:\\Users\\pascaldisse\\Downloads\\open-sourcefy\\matrix_launcher\\matrix_launcher.c" user32.lib kernel32.lib
echo Compilation exit code: %ERRORLEVEL%
