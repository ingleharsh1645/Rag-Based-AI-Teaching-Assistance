Set-Location $PSScriptRoot
Write-Host "Starting Streamlit frontend on http://localhost:8501 ..."
python -m streamlit run frontend/app.py
