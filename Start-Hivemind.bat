@echo off
setlocal

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Hivemind.ps1" -Install %*

if errorlevel 1 (
    echo.
    echo Hivemind failed to start. Check the messages above.
    pause
)
