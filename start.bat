@echo off
title Polymarket Bot Launcher
echo ============================================
echo   POLYMARKET BOT - Launching...
echo ============================================
echo.

cd /d "%~dp0"

echo [1/2] Starting Dashboard on http://localhost:5000
start "Polymarket Dashboard" cmd /k "cd /d "%~dp0" && python dashboard.py"

timeout /t 2 /nobreak >nul

echo [2/2] Starting Bot in DEMO mode
start "Polymarket Bot" cmd /k "cd /d "%~dp0" && python main.py --mode demo"

echo.
echo ============================================
echo   Both processes launched in separate windows
echo   Dashboard: http://localhost:5000
echo   Close this window or press any key to exit
echo ============================================
pause >nul
