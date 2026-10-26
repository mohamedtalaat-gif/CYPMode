#!/bin/bash
# Double-click this file in Finder to open CYPMode in your browser.
#   run_cypmode.command             asks which one to open
#   run_cypmode.command app         the screening tool and the results tab of the TDI track (app.py, Streamlit)
#   run_cypmode.command explorer    the interactive explorer of the finished results (explorer/cypmode_explorer.py, a marimo notebook)
cd "$(dirname "$0")"

mode="${1:-}"
if [ -z "$mode" ]; then
  echo "CYPMode"
  echo "  1) the screening tool: paste SMILES, the motif call, the structural check, the results tab of the TDI track"
  echo "  2) the interactive explorer of the finished results"
  read -p "Choose 1 or 2: " answer
  case "$answer" in
    2) mode=explorer ;;
    *) mode=app ;;
  esac
fi

case "$mode" in
  app)
    if [ ! -d ".venv" ]; then
      echo "No .venv found. Set up the project first, from a terminal:"
      echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
      read -p "Press Enter to close..."
      exit 1
    fi
    source .venv/bin/activate
    streamlit run app.py
    ;;
  explorer)
    if [ ! -d ".venv-marimo" ]; then
      echo "No .venv-marimo found. Set up the explorer first, from a terminal:"
      echo "  python3 -m venv .venv-marimo && .venv-marimo/bin/pip install -r environment/requirements-lock-marimo.txt"
      read -p "Press Enter to close..."
      exit 1
    fi
    bash scripts/explorer.sh run
    ;;
  *)
    echo "usage: run_cypmode.command [app|explorer]"
    read -p "Press Enter to close..."
    exit 2
    ;;
esac
