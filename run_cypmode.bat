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
REM The environment is prepared by this file on the first start; scripts\explorer.sh does the same on macOS and Linux.
set "ENVDIR=.venv-marimo"
set "APP=explorer\cypmode_explorer.py"
set "REQ=environment\requirements-lock-marimo.txt"
if exist "explorer\full\app.py" (
  set "APP=explorer\full\app.py"
  set "REQ=explorer\full\requirements.txt"
)
fc /b "%REQ%" "%ENVDIR%\requirements.installed" >nul 2>nul
if not errorlevel 1 if exist "%ENVDIR%\Scripts\marimo.exe" goto start_explorer
echo First start: preparing the environment in %ENVDIR% from %REQ%. This takes a few minutes once; the next start is immediate.
set "PY="
for %%V in (3.13 3.12 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul && set "PY=python"
)
if defined PY goto pip_install
echo Python 3.11 or newer was not found on this machine.
echo This file can download uv (a small installer from https://astral.sh/uv) into .tools; uv then fetches Python and the packages.
if not "%EXPLORER_ASSUME_YES%"=="1" pause
if not exist ".tools\uv.exe" (
  set "UV_INSTALL_DIR=%CD%\.tools"
  set "UV_NO_MODIFY_PATH=1"
  powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
)
".tools\uv.exe" venv --python 3.12 "%ENVDIR%" || goto explorer_failed
".tools\uv.exe" pip install --python "%ENVDIR%\Scripts\python.exe" -r "%REQ%" || goto explorer_failed
goto explorer_ready

:pip_install
%PY% -m venv "%ENVDIR%" || goto explorer_failed
"%ENVDIR%\Scripts\python.exe" -m pip install --quiet --upgrade pip
"%ENVDIR%\Scripts\python.exe" -m pip install -r "%REQ%" || goto explorer_failed

:explorer_ready
copy /y "%REQ%" "%ENVDIR%\requirements.installed" >nul

:start_explorer
call "%ENVDIR%\Scripts\activate"
marimo run %APP%
exit /b 0

:explorer_failed
echo The environment of the explorer could not be prepared. Read the messages above; the requirements are in %REQ%.
pause
exit /b 1
