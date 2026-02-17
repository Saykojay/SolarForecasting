$WshShell = New-Object -ComObject WScript.Shell
$DesktopPaths = @(
    [System.Environment]::GetFolderPath('Desktop'),
    "C:\OneDrive\Desktop",
    "C:\Users\Lenovo\Desktop",
    "$env:USERPROFILE\Desktop"
) | Select-Object -Unique

# Filter paths that actually exist
$ExistingPaths = $DesktopPaths | Where-Object { Test-Path $_ }

$BatchPath = "c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1\launch_dashboard.bat"
$WorkDir = "c:\Users\Lenovo\OneDrive\Pretrain GRU\Pre-train model PatchTST\Modular Pipeline v1"

foreach ($path in $ExistingPaths) {
    try {
        $ShortcutPath = Join-Path $path "PV System Dashboard.lnk"
        $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
        $Shortcut.TargetPath = $BatchPath
        $Shortcut.WorkingDirectory = $WorkDir
        $Shortcut.Description = "Launch PV Forecasting ML Dashboard"
        $Shortcut.IconLocation = "C:\Users\Lenovo\miniconda3\envs\tf-gpu\python.exe,0"
        $Shortcut.Save()
        Write-Host "Berhasil membuat shortcut di: $ShortcutPath" -ForegroundColor Cyan
    }
    catch {
        Write-Host "Gagal membuat di $path : $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

Write-Host "`n======================================================" -ForegroundColor Green
Write-Host "Shortcut 'PV System Dashboard' sudah dikirim ke Desktop Anda!" -ForegroundColor Green
Write-Host "Jika masih tidak muncul, coba tekan F5 (Refresh) di Desktop." -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

