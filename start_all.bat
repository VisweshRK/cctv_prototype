@echo off
title Presence B2C AI CCTV Platform Runner
echo ======================================================================
echo Starting Presence B2C AI CCTV Platform (Backend + Frontend)
echo ======================================================================

set PYTHONPATH=.

echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "Presence AI Backend (Port 8000)" cmd /k "cd /d e:\cctv && set PYTHONPATH=. && uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload"

echo [2/2] Starting Vite Frontend Dev Server on http://localhost:5173 ...
start "Presence Frontend Dev Server (Port 5173)" cmd /k "cd /d e:\cctv\apps\web && npm run dev"

echo.
echo ======================================================================
echo Both servers are launching in separate windows!
echo.
echo  - Frontend Dev Server: http://localhost:5173
echo  - Unified App (API + Built Frontend): http://localhost:8000
echo  - Super Admin: http://localhost:8000/admin
echo  - API Swagger Docs: http://localhost:8000/docs
echo.
echo Default Logins:
echo   Customer: demo@presence.ai / demo123
echo   Admin:    admin@presence.ai / admin123
echo ======================================================================
pause
