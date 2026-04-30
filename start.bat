@echo off
title Weather Bot v2 Launcher
color 0A

set PYEXE=C:\Python310\python.exe
set BOTDIR=%~dp0

echo.
echo  ============================================================
echo   WEATHER BOT v2 - Ensemble + Confidence
echo   Multi-source forecasts + confidence gating
echo.
echo   Ventana recomendada: 00:00-02:00 y 04:00-07:00 UTC
echo  ============================================================
echo.

if not exist "%PYEXE%" (
    echo [ERROR] Python no encontrado en %PYEXE%
    echo Edita start.bat y ajusta la variable PYEXE.
    pause
    exit /b 1
)

if not exist "%BOTDIR%logs" mkdir "%BOTDIR%logs"
if not exist "%BOTDIR%data" mkdir "%BOTDIR%data"
if not exist "%BOTDIR%reports" mkdir "%BOTDIR%reports"

echo  Comprobando si ya hay una instancia de main.py en ejecucion...
set BOT_RUNNING=0
for /f "tokens=2 delims=," %%A in ('wmic process where "name='python.exe'" get processid^,commandline /format:csv ^| findstr /I "main.py"') do (
    set BOT_RUNNING=1
)
if exist "%BOTDIR%data\bot.lock" (
    echo  [AVISO] data\bot.lock encontrado. Un bot puede estar corriendo.
    echo  El bot comprobara el PID al arrancar y abortara si sigue vivo.
)
if "%BOT_RUNNING%"=="1" (
    echo  [AVISO] Se detecto un main.py ya en ejecucion.
    set /p FORCE="  Continuar de todas formas? (s/N): "
    if /i not "%FORCE%"=="s" goto FIN
)
echo.

echo  Comprobando dependencias...
"%PYEXE%" -c "import psutil, yaml, flask, requests, anthropic" 2>NUL
if errorlevel 1 (
    echo  [AVISO] Faltan dependencias. Ejecuta: pip install -r requirements.txt
    pause
)

if exist "%BOTDIR%.env" (
    echo  .env detectado.
) else (
    echo  [AVISO] No hay .env. El bot funcionara en modo degradado ^(solo fuentes sin key^).
)
echo.

echo  Selecciona el modo:
echo.
echo    [1] DEMO             - Simular apuestas (sin dinero real)
echo    [2] LIVE             - Apuestas reales en Polymarket
echo    [3] Solo Dashboard   - Monitoreo sin bot
echo    [4] Shadow mode      - Todo excepto ejecutar ordenes
echo    [5] Analizar logs    - Reporte markdown (ultimas 24h)
echo    [6] Limpiar datos    - Backup a data\archive y wipe
echo    [7] Salir
echo.
set /p CHOICE="  Tu eleccion (1-7): "

if "%CHOICE%"=="1" goto DEMO
if "%CHOICE%"=="2" goto LIVE_WARN
if "%CHOICE%"=="3" goto SOLO_DASHBOARD
if "%CHOICE%"=="4" goto SHADOW
if "%CHOICE%"=="5" goto ANALYZE
if "%CHOICE%"=="6" goto CLEAN
if "%CHOICE%"=="7" goto FIN
echo  Opcion no valida.
goto FIN

:DEMO
echo.
echo  [1/2] Dashboard en http://localhost:5000 ...
start "Weather Bot Dashboard" cmd /k "cd /d "%BOTDIR%" && "%PYEXE%" "%BOTDIR%dashboard.py""
timeout /t 3 /nobreak >NUL
echo  [2/2] Bot DEMO...
start "Weather Bot [DEMO]" cmd /k "cd /d "%BOTDIR%" && "%PYEXE%" "%BOTDIR%main.py" --mode demo"
goto OPEN_BROWSER

:LIVE_WARN
echo.
echo  ============================================================
echo   ATENCION: MODO LIVE - DINERO REAL EN POLYMARKET
echo  ============================================================
set /p CONFIRM="  Escribe CONFIRMO para continuar: "
if /i not "%CONFIRM%"=="CONFIRMO" (
    echo  Cancelado.
    pause
    goto FIN
)
echo  [1/2] Dashboard...
start "Weather Bot Dashboard" cmd /k "cd /d "%BOTDIR%" && "%PYEXE%" "%BOTDIR%dashboard.py""
timeout /t 3 /nobreak >NUL
echo  [2/2] Bot LIVE...
start "Weather Bot [LIVE]" cmd /k "cd /d "%BOTDIR%" && "%PYEXE%" "%BOTDIR%main.py" --mode live"
goto OPEN_BROWSER

:SOLO_DASHBOARD
echo.
start "Weather Bot Dashboard" cmd /k "cd /d "%BOTDIR%" && "%PYEXE%" "%BOTDIR%dashboard.py""
goto OPEN_BROWSER

:SHADOW
echo.
echo  Shadow mode: todo el pipeline, ordenes NO ejecutadas.
start "Weather Bot Dashboard" cmd /k "cd /d "%BOTDIR%" && "%PYEXE%" "%BOTDIR%dashboard.py""
timeout /t 3 /nobreak >NUL
start "Weather Bot [SHADOW]" cmd /k "cd /d "%BOTDIR%" && set EXECUTE_TRADES=false && "%PYEXE%" "%BOTDIR%main.py" --mode demo --shadow"
goto OPEN_BROWSER

:ANALYZE
echo.
echo  Analizando ultimas 24h...
"%PYEXE%" "%BOTDIR%log_analyzer.py" --last-hours 24 --output "%BOTDIR%reports"
echo.
pause
goto FIN

:CLEAN
echo.
echo  Esto mueve data\bot.db y logs\*.log|*.jsonl a data\archive\<fecha>\.
set /p CONFIRM2="  Continuar? (s/N): "
if /i not "%CONFIRM2%"=="s" goto FIN
for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set DT=%%a
set STAMP=%DT:~0,8%_%DT:~8,4%
set ARCHDIR=%BOTDIR%data\archive\%STAMP%
mkdir "%ARCHDIR%" 2>NUL
if exist "%BOTDIR%data\bot.db" move /Y "%BOTDIR%data\bot.db" "%ARCHDIR%\bot.db"
if exist "%BOTDIR%logs\events.jsonl" move /Y "%BOTDIR%logs\events.jsonl" "%ARCHDIR%\events.jsonl"
for %%f in ("%BOTDIR%logs\*.log") do move /Y "%%f" "%ARCHDIR%\" 2>NUL
echo  Backup en: %ARCHDIR%
pause
goto FIN

:OPEN_BROWSER
timeout /t 4 /nobreak >NUL
start http://localhost:5000

:FIN
echo.
echo  Listo.
pause >nul
