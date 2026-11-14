@echo off
title Presence B2C AI CCTV Platform Server
cd /d "%~dp0"
echo =========================================================================
echo Starting Presence B2C CCTV Monitoring Platform on http://127.0.0.1:8000
echo =========================================================================
set PYTHONPATH=%~dp0
set USE_INT8_DETECTOR=1

where python >nul 2>nul
if %errorlevel% equ 0 (
    python -m uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload
) else (
    "C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload
)
pause

