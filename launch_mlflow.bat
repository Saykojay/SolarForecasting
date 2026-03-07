@echo off
:: =====================================================================
:: PV Forecasting System - MLflow UI Launcher
:: =====================================================================

:: Auto-detect project root (folder dimana .bat ini berada)
set "ROOT_DIR=%~dp0"
set "CONFIG_FILE=%ROOT_DIR%env_config.txt"

:: ============================================================
:: AUTO-DETECT CONDA INSTALLATION
:: ============================================================
set "CONDA_PATH="

:: 1. Cek dari environment variable CONDA_EXE
if defined CONDA_EXE (
    for %%i in ("%CONDA_EXE%") do set "CONDA_PATH=%%~dpi.."
    goto :conda_found
)

:: 2. Cek lokasi umum
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=%USERPROFILE%\anaconda3"
    goto :conda_found
)
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_PATH=%USERPROFILE%\miniconda3"
    goto :conda_found
)
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\ProgramData\anaconda3"
    goto :conda_found
)
if exist "C:\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\anaconda3"
    goto :conda_found
)
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\ProgramData\miniconda3"
    goto :conda_found
)

:: 3. Coba 'where conda'
for /f "tokens=*" %%p in ('where conda 2^>nul') do (
    for %%i in ("%%p") do set "CONDA_PATH=%%~dpi.."
    goto :conda_found
)

echo [ERROR] Anaconda/Miniconda tidak ditemukan secara otomatis!
pause
exit /b

:conda_found
echo [INFO] Conda ditemukan di: %CONDA_PATH%

:: ============================================================
:: BACA ENVIRONMENT
:: ============================================================
setlocal enabledelayedexpansion

if exist "%CONFIG_FILE%" (
    set /p ENV_NAME=<"%CONFIG_FILE%"
    echo [INFO] Menggunakan environment: !ENV_NAME!
) else (
    echo [ERROR] env_config.txt tidak ditemukan.
    echo Jalankan launch_dashboard.bat terlebih dahulu untuk memilih environment.
    pause
    exit /b
)

:: Set ENV_NAME di luar delayed expansion
for /f "delims=" %%e in ("!ENV_NAME!") do (
    endlocal
    set "ENV_NAME=%%e"
)

:: ============================================================
:: RUN MLFLOW
:: ============================================================
echo.
echo [1/2] Mengaktifkan lingkungan Python (%ENV_NAME%)...
if /i "%ENV_NAME%"=="base" (
    call "%CONDA_PATH%\Scripts\activate.bat"
) else (
    call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
)

echo [2/2] Menjalankan MLflow UI di direktori mlruns...
cd /d "%ROOT_DIR%"

:: MLflow default akan mencari folder 'mlruns' di direktori aktif
mlflow ui --port 5000

pause
