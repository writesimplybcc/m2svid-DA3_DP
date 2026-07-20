# StereoFaster WebUI Launcher
# Usage:
#   .\run_webui.ps1                 # Launches on default port 7878
#   .\run_webui.ps1 -Port 7865      # Launches on a custom port
#   .\run_webui.ps1 -Port 7878 -ServerName 0.0.0.0

param(
    [int]$Port = 7878,
    [string]$ServerName = "127.0.0.1"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  StereoFaster WebUI Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Setting PYTHONPATH for DA3 + M2SVid..." -ForegroundColor Yellow

$env:PYTHONPATH = "Depth-Anything-3/src;.;m2svid/third_party/Hi3D-Official;m2svid/third_party/pytorch-msssim;$env:PYTHONPATH"

Write-Host "PYTHONPATH set." -ForegroundColor Green
Write-Host ""
$PythonCmd = "python"
if (Test-Path ".\venv\Scripts\python.exe") {
    Write-Host "Using virtual environment python (.\venv\Scripts\python.exe)..." -ForegroundColor Green
    $PythonCmd = ".\venv\Scripts\python.exe"
} else {
    Write-Host "Virtual environment not found, falling back to system python..." -ForegroundColor Yellow
}

& $PythonCmd webui.py --server-name $ServerName --server-port $Port

