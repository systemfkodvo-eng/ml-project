param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

Write-Host "=== Starting backend (FastAPI) on port $Port ===" -ForegroundColor Cyan

# Go to project root (this script is expected to be in the root folder)
Set-Location $PSScriptRoot

# Paths
$backendPath = Join-Path $PSScriptRoot "backend"
$venvActivate = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"

if (-Not (Test-Path $backendPath)) {
    Write-Host "Backend folder not found at $backendPath" -ForegroundColor Red
    exit 1
}

Set-Location $backendPath

if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venvActivate
} else {
    Write-Host "Virtual env not found, continuing without activation..." -ForegroundColor Yellow
}

Write-Host "Running: python -m uvicorn app.main:app --host 0.0.0.0 --port $Port --reload" -ForegroundColor Green
python -m uvicorn app.main:app --host 0.0.0.0 --port $Port --reload

