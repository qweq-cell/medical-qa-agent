@echo off
title RAG Env Setup
echo ==========================================
echo   Create isolated venv for RAG (no impact on main env)
echo ==========================================
echo.
cd /d "%~dp0"

echo [1/3] Creating venv .venv-rag ...
python -m venv .venv-rag

echo [2/3] Installing deps (torch is large, be patient) ...
".venv-rag\Scripts\pip" install -r scripts\requirements-rag.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo [3/3] Done!
echo.
echo Run RAG:
echo   .venv-rag\Scripts\python scripts\rag_demo.py
echo.
pause
