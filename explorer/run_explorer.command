#!/bin/bash
# Double-click this file in Finder to open the interactive explorer (explorer/cypmode_explorer.py, a marimo notebook) in your browser.
# It starts the same program as `run_cypmode.command explorer`; the environment is .venv-marimo (see the header of scripts/explorer.sh).
cd "$(dirname "$0")/.." || exit 1
exec bash run_cypmode.command explorer
