$ErrorActionPreference = "Stop"

Write-Host "=== Starting frontend (Next.js) on port 3000 ===" -ForegroundColor Cyan

# Go to project root (this script is expected to be in the root folder)
Set-Location $PSScriptRoot

$frontendPath = Join-Path $PSScriptRoot "frontend"

if (-Not (Test-Path $frontendPath)) {
    Write-Host "Frontend folder not found at $frontendPath" -ForegroundColor Red
    exit 1
}

Set-Location $frontendPath

Write-Host "Running: npm run dev" -ForegroundColor Green
npm run dev

