#!/bin/bash
# Double-click this file in Finder to launch the CYPMode screening tool in your browser.
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "No .venv found. Set up the project first, from a terminal:"
  echo "  python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  read -p "Press Enter to close..."
  exit 1
fi

source .venv/bin/activate
streamlit run app.py
