$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Project virtual environment not found: $python"
}

Set-Location $projectRoot
$logDirectory = Join-Path $projectRoot "logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stdoutLog = Join-Path $logDirectory "backend.log"
$stderrLog = Join-Path $logDirectory "backend.error.log"

Start-Process `
    -FilePath $python `
    -ArgumentList "-c", "import uvicorn; uvicorn.run('src.api.app:app', host='127.0.0.1', port=8000)" `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden | Out-Null