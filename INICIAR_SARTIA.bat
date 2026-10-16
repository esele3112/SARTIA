@echo off
setlocal
title SARTIA - Sistema de Alerta Temprana - CBTA 197
cd /d "%~dp0"

REM ============================================================
REM  SARTIA - Lanzador portable
REM  Doble clic en este archivo y listo: crea el entorno virtual
REM  si no existe, instala dependencias si faltan, libera el
REM  puerto 8501 si quedo ocupado por una ejecucion anterior
REM  y abre la app en el navegador (http://localhost:8501).
REM  No importa desde que carpeta se copie el proyecto.
REM ============================================================

REM 1) Entorno virtual: crearlo solo si no existe
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creando entorno virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: no se pudo crear el entorno virtual.
        echo Asegurate de tener Python 3.12 instalado y agregado al PATH.
        echo Descarga: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

REM 2) Dependencias: instalar solo si faltan (la primera vez)
".venv\Scripts\python.exe" -c "import shap, matplotlib, streamlit, sklearn, pandas" >nul 2>&1
if errorlevel 1 (
    echo [2/4] Instalando dependencias - solo la primera vez, puede tardar...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: fallo la instalacion de dependencias. Revisa tu conexion a internet.
        pause
        exit /b 1
    )
)

REM 3) Liberar el puerto 8501 si quedo ocupado por un proceso anterior
netstat -ano | findstr /R /C:":8501 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [3/4] El puerto 8501 esta ocupado por un proceso anterior. Liberandolo...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8501 .*LISTENING"') do (
        taskkill /F /PID %%a >nul 2>&1
    )
    ping -n 3 127.0.0.1 >nul
)

REM 4) Lanzar la app
echo [4/4] Iniciando SARTIA en http://localhost:8501 ...
echo Cierra esta ventana para detener el sistema.
start "" "http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py

pause
