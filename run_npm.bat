@echo off
title Presence B2C Frontend (Vite Dev Server)
cd /d "%~dp0\apps\web"
echo Starting Frontend via npm run dev on http://127. 0.0.1:5173 ...
call npm run dev
pause
