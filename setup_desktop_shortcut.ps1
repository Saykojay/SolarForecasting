# =====================================================================
# PV Forecasting System - Desktop Shortcut Creator
# PORTABLE: Auto-detect Conda & Python paths
# =====================================================================

$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [System.Environment]::GetFolderPath('Desktop')
$ShortcutPath = Join-Path $DesktopPath "Solar Forecasting.lnk"

# Path ke file bat (relatif terhadap script ini)
$BatchPath = Join-Path $PSScriptRoot "launch_dashboard.bat"
$WorkDir = $PSScriptRoot

# ----- AUTO-DETECT CONDA -----
function Find-CondaPath {
    # 1. Cek CONDA_EXE environment variable
    $condaExe = $env:CONDA_EXE
    if ($condaExe -and (Test-Path $condaExe)) {
        return (Split-Path (Split-Path $condaExe))
    }
    
    # 2. Cek lokasi umum
    $commonPaths = @(
        "$env:USERPROFILE\anaconda3",
        "$env:USERPROFILE\miniconda3",
        "C:\ProgramData\anaconda3",
        "C:\ProgramData\miniconda3",
        "C:\anaconda3",
        "C:\miniconda3"
    )
    
    foreach ($p in $commonPaths) {
        if (Test-Path "$p\python.exe") {
            return $p
        }
    }
    
    # 3. Coba where conda
    try {
        $condaCmd = (Get-Command conda -ErrorAction SilentlyContinue).Source
        if ($condaCmd) {
            return (Split-Path (Split-Path $condaCmd))
        }
    }
    catch {}
    
    return $null
}

$CondaBase = Find-CondaPath

# Read environment name from env_config.txt (created by launch_dashboard.bat)
$EnvConfigFile = Join-Path $PSScriptRoot "env_config.txt"
if (Test-Path $EnvConfigFile) {
    $EnvName = (Get-Content $EnvConfigFile -First 1).Trim()
    Write-Host "[INFO] Environment dari env_config.txt: $EnvName" -ForegroundColor Cyan
} else {
    $EnvName = "base"
    Write-Host "[INFO] env_config.txt belum ada. Jalankan launch_dashboard.bat dulu untuk memilih environment." -ForegroundColor Yellow
    Write-Host "[INFO] Menggunakan default: $EnvName" -ForegroundColor Yellow
}

try {
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = $BatchPath
    $Shortcut.WorkingDirectory = $WorkDir
    $Shortcut.Description = "Launch Solar Forecasting Streamlit App (Env: $EnvName)"
    
    # Auto-detect icon path
    $IconSet = $false
    if ($CondaBase) {
        $EnvPython = Join-Path $CondaBase "envs\$EnvName\python.exe"
        $BasePython = Join-Path $CondaBase "python.exe"
        
        if (Test-Path $EnvPython) {
            $Shortcut.IconLocation = "$EnvPython,0"
            $IconSet = $true
        }
        elseif (Test-Path $BasePython) {
            $Shortcut.IconLocation = "$BasePython,0"
            $IconSet = $true
        }
    }
    
    if (-not $IconSet) {
        # Fallback: coba python di PATH
        try {
            $pyPath = (Get-Command python -ErrorAction SilentlyContinue).Source
            if ($pyPath) {
                $Shortcut.IconLocation = "$pyPath,0"
            }
        }
        catch {
            Write-Host "[INFO] Icon default akan digunakan." -ForegroundColor Yellow
        }
    }
    
    $Shortcut.Save()
    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Green
    Write-Host "BERHASIL! Desktop shortcut telah dibuat/diperbarui." -ForegroundColor Green
    Write-Host "Nama: Solar Forecasting" -ForegroundColor Green
    if ($CondaBase) {
        Write-Host "Conda: $CondaBase" -ForegroundColor Cyan
    }
    Write-Host "======================================================" -ForegroundColor Green
}
catch {
    Write-Host "Gagal membuat shortcut: $($_.Exception.Message)" -ForegroundColor Red
}
Write-Host "======================================================" -ForegroundColor Green

