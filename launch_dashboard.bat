@echo off
:: =====================================================================
:: PV Forecasting System - Dashboard Launcher
:: =====================================================================
:: Ini adalah script untuk menjalankan dashboard secara otomatis.
:: Anda bisa membuat shortcut dari file ini ke Desktop.
:: =====================================================================

set "CONDA_PATH=C:\Users\Lenovo\miniconda3"
set "ENV_NAME=tf-gpu"
set "ROOT_DIR=c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1"

echo [1/3] Navigating to project directory...
cd /d "%ROOT_DIR%"

echo [2/3] Activating Anaconda environment (%ENV_NAME%)...
call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Gagal mengaktifkan Anaconda. Pastikan path di file ini benar.
    pause
    exit /b
)

echo [3/3] Launching Streamlit Dashboard...
echo Dashboard akan terbuka di browser Anda sesaat lagi.
echo Jangan tutup jendela terminal ini selama menggunakan dashboard.
echo.

streamlit run app.py

pause
