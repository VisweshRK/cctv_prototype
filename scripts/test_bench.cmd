@echo off
setlocal
cd /d "%~dp0\..\ai"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at ai\.venv
    exit /b 1
)

echo ===================================================================
echo  Starting CCTV Vision Test Bench (P2 Krithick)
echo  YuNet Face Detection ^| SFace Embeddings ^| RapidOCR Plate Engine
echo ===================================================================
echo.
.venv\Scripts\python test_bench\server.py %*
