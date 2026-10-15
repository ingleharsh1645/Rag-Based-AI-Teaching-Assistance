Set-Location $PSScriptRoot
Write-Host "Starting FastAPI backend on http://localhost:8000 ..."
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
