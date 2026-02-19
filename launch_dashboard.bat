@echo off
:: =====================================================================
:: PV Forecasting System - Dashboard Launcher
:: =====================================================================
:: PORTABLE: Auto-detect Conda dan environment yang tersedia.
:: Environment disimpan di env_config.txt (per-device).
:: Untuk ganti environment, hapus env_config.txt atau tekan C saat prompt.
:: =====================================================================

:: Auto-detect project root (folder dimana .bat ini berada)
set "ROOT_DIR=%~dp0"
set "CONFIG_FILE=%ROOT_DIR%env_config.txt"

:: ============================================================
:: AUTO-DETECT CONDA INSTALLATION
:: ============================================================
set "CONDA_PATH="

:: 1. Cek dari environment variable CONDA_EXE (jika conda sudah aktif)
if defined CONDA_EXE (
    for %%i in ("%CONDA_EXE%") do set "CONDA_PATH=%%~dpi.."
    goto :conda_found
)

:: 2. Cek lokasi umum: %USERPROFILE%\anaconda3
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=%USERPROFILE%\anaconda3"
    goto :conda_found
)

:: 3. Cek lokasi umum: %USERPROFILE%\miniconda3
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_PATH=%USERPROFILE%\miniconda3"
    goto :conda_found
)

:: 4. Cek lokasi umum: C:\ProgramData\anaconda3
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\ProgramData\anaconda3"
    goto :conda_found
)

:: 5. Cek lokasi umum: C:\anaconda3
if exist "C:\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\anaconda3"
    goto :conda_found
)

:: 6. Cek lokasi umum: C:\ProgramData\miniconda3
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\ProgramData\miniconda3"
    goto :conda_found
)

:: 7. Coba cari via 'where conda'
for /f "tokens=*" %%p in ('where conda 2^>nul') do (
    for %%i in ("%%p") do set "CONDA_PATH=%%~dpi.."
    goto :conda_found
)

:: Jika tidak ditemukan
echo ============================================================
echo [ERROR] Anaconda/Miniconda tidak ditemukan secara otomatis!
echo.
echo Solusi:
echo   1. Pastikan Anaconda sudah terinstall
echo   2. ATAU edit file ini dan set CONDA_PATH secara manual:
echo      set "CONDA_PATH=C:\path\to\anaconda3"
echo ============================================================
pause
exit /b

:conda_found
echo ============================================================
echo   PV Forecasting System - Dashboard Launcher
echo ============================================================
echo.
echo [INFO] Conda ditemukan di: %CONDA_PATH%
echo [INFO] Project dir: %ROOT_DIR%
echo.

:: ============================================================
:: BACA ATAU DETEKSI ENVIRONMENT
:: ============================================================
:: Cek apakah env_config.txt sudah ada (sudah pernah dipilih)
if exist "%CONFIG_FILE%" (
    set /p ENV_NAME=<"%CONFIG_FILE%"
    echo [INFO] Environment tersimpan: !ENV_NAME!
    
    :: Untuk mengaktifkan delayed expansion agar bisa baca variable dalam if
    setlocal enabledelayedexpansion
    set /p ENV_NAME=<"%CONFIG_FILE%"
    
    :: Validasi: cek apakah environment masih ada
    if exist "%CONDA_PATH%\envs\!ENV_NAME!\python.exe" (
        echo [INFO] Environment '!ENV_NAME!' ditemukan dan valid.
        echo.
        echo Tekan ENTER untuk lanjut, atau ketik C lalu ENTER untuk ganti environment:
        set "USER_CHOICE="
        set /p USER_CHOICE=">> "
        if /i "!USER_CHOICE!"=="C" (
            endlocal
            goto :select_env
        )
        :: Set ENV_NAME di luar delayed expansion
        for /f "delims=" %%e in ("!ENV_NAME!") do (
            endlocal
            set "ENV_NAME=%%e"
        )
        goto :env_ready
    ) else (
        echo [WARNING] Environment '!ENV_NAME!' tidak ditemukan di device ini!
        echo           Mungkin Anda pindah laptop. Mari pilih environment baru.
        endlocal
        goto :select_env
    )
)

:select_env
:: List semua environment yang tersedia
echo.
echo ============================================================
echo   Pilih Conda Environment
echo ============================================================
echo.
echo Environment yang tersedia:
echo.

setlocal enabledelayedexpansion

:: Enumerate environments
set "ENV_COUNT=0"

:: Selalu tambahkan 'base' sebagai opsi pertama
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=base"
echo   [!ENV_COUNT!] base

:: List environments di folder envs
if exist "%CONDA_PATH%\envs" (
    for /d %%d in ("%CONDA_PATH%\envs\*") do (
        if exist "%%d\python.exe" (
            set /a ENV_COUNT+=1
            set "ENV_!ENV_COUNT!=%%~nxd"
            echo   [!ENV_COUNT!] %%~nxd
        )
    )
)

echo.

if !ENV_COUNT! EQU 0 (
    echo [ERROR] Tidak ada environment yang ditemukan!
    pause
    exit /b
)

:: Minta user pilih
set /p "ENV_CHOICE=Pilih nomor environment [1-!ENV_COUNT!]: "

:: Validasi input
set "SELECTED_ENV="
for /l %%i in (1,1,!ENV_COUNT!) do (
    if "!ENV_CHOICE!"=="%%i" (
        set "SELECTED_ENV=!ENV_%%i!"
    )
)

if "!SELECTED_ENV!"=="" (
    echo [ERROR] Pilihan tidak valid! Silakan coba lagi.
    endlocal
    goto :select_env
)

:: Simpan pilihan ke file
echo !SELECTED_ENV!> "%CONFIG_FILE%"
echo.
echo [OK] Environment '!SELECTED_ENV!' dipilih dan disimpan ke env_config.txt
echo     (Hapus env_config.txt jika ingin mengganti environment nanti)

:: Set ENV_NAME di luar delayed expansion
for /f "delims=" %%e in ("!SELECTED_ENV!") do (
    endlocal
    set "ENV_NAME=%%e"
)

:env_ready
echo.
echo [INFO] Menggunakan environment: %ENV_NAME%
echo.

echo [1/3] Menuju direktori project...
cd /d "%ROOT_DIR%"

echo [2/3] Mengaktifkan lingkungan Python (%ENV_NAME%)...
if /i "%ENV_NAME%"=="base" (
    call "%CONDA_PATH%\Scripts\activate.bat"
) else (
    call "%CONDA_PATH%\Scripts\activate.bat" %ENV_NAME%
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Gagal mengaktifkan environment '%ENV_NAME%'.
    echo         Pastikan environment sudah dibuat:
    echo         conda create -n %ENV_NAME% python=3.10
    echo.
    echo [INFO] Menghapus env_config.txt agar bisa memilih ulang...
    del "%CONFIG_FILE%" 2>nul
    pause
    exit /b
)

echo [3/3] Menjalankan Streamlit Dashboard...
echo Dashboard akan segera terbuka di browser Anda.
echo JANGAN TUTUP jendela ini selama aplikasi berjalan.
echo.

streamlit run app.py

pause
