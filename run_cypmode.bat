@echo off
REM Double-click this file to launch the CYPMode screening tool in your browser.
cd /d "%~dp0"

if not exist ".venv" (
  echo No .venv found. Set up the project first, from a terminal:
  echo   python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -e .
  pause
  exit /b 1
)

call .venv\Scripts\activate
streamlit run app.py
