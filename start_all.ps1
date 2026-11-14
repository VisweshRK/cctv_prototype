Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting Presence B2C AI CCTV Platform (Backend + Frontend)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "."

# Start Backend in new window
Write-Host "`n[1/2] Launching Backend on http://localhost:8000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd e:\cctv; `$env:PYTHONPATH='.'; uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload"

# Start Frontend in new window
Write-Host "[2/2] Launching Vite Frontend on http://localhost:5173 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd e:\cctv\apps\web; npm run dev"

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "Both Frontend and Backend are now running!" -ForegroundColor Green
Write-Host " - Frontend UI: http://localhost:5173" -ForegroundColor White
Write-Host " - Backend API + Built UI: http://localhost:8000" -ForegroundColor White
Write-Host " - API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Green
