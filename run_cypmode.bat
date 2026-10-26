@echo off
REM Double-click this file to open CYPMode in your browser.
REM   run_cypmode.bat             asks which one to open
REM   run_cypmode.bat app         the screening tool and the results tab of the TDI track (app.py, Streamlit)
REM   run_cypmode.bat explorer    the interactive explorer of the finished results (explorer\cypmode_explorer.py, a marimo notebook)
cd /d "%~dp0"

set MODE=%1
if not "%MODE%"=="" goto run

echo CYPMode
echo   1. the screening tool: paste SMILES, the motif call, the structural check, the results tab of the TDI track
echo   2. the interactive explorer of the finished results
choice /c 12 /n /m "Choose 1 or 2: "
if errorlevel 2 goto explorer
set MODE=app

:run
if /i "%MODE%"=="app" goto app
if /i "%MODE%"=="explorer" goto explorer
echo usage: run_cypmode.bat [app^|explorer]
pause
exit /b 2

:app
if not exist ".venv" (
  echo No .venv found. Set up the project first, from a terminal:
  echo   python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -e .
  pause
  exit /b 1
)
call .venv\Scripts\activate
streamlit run app.py
exit /b 0

:explorer
if not exist ".venv-marimo" (
  echo No .venv-marimo found. Set up the explorer first, from a terminal:
  echo   python -m venv .venv-marimo ^&^& .venv-marimo\Scripts\pip install -r environment\requirements-lock-marimo.txt
  pause
  exit /b 1
)
call .venv-marimo\Scripts\activate
marimo run explorer\cypmode_explorer.py
exit /b 0
