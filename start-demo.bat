@echo off
title MedGuardian Launcher
echo ============================================
echo   MedGuardian Medical QC Agent Launcher
echo ============================================
echo.
echo [1/2] Starting backend FastAPI (port 8000)...
start "MedGuardian-API" cmd /k "cd /d %~dp0 && python -m src.main"
echo [2/2] Starting frontend Streamlit (port 8501)...
start "MedGuardian-UI" cmd /k "cd /d %~dp0 && streamlit run src/ui/app.py"
echo.
echo ============================================
echo   Launched! Open your browser at:
echo.
echo     Frontend UI:     http://localhost:8501
echo     Backend docs:    http://localhost:8000/docs
echo.
echo   Close the two black windows to stop.
echo ============================================
echo.
pause
