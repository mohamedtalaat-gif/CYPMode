#!/bin/bash
# The interactive explorer of the finished results (explorer/cypmode_explorer.py, a marimo notebook that reads only explorer/public).
#   bash scripts/explorer.sh run            the explorer as an application (read-only)
#   bash scripts/explorer.sh edit           the same notebook in the editor
#   bash scripts/explorer.sh html [DIR]     a static page (no code, the outputs as they stand)
#   bash scripts/explorer.sh ipynb [FILE]   a Jupyter notebook with the outputs
#   bash scripts/explorer.sh wasm [DIR]     a page that runs in a browser without a server (the data files are bundled; the browser loads Pyodide from a CDN)
# The environment is .venv-marimo (python3 -m venv .venv-marimo; .venv-marimo/bin/pip install marimo altair pandas numpy pyyaml uv nbformat). The explorer never starts a nested
# cross-validation, a co-folding run or a queue job: it holds no model and reads the five tables, the excerpts and the out-of-fold predictions of the bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv-marimo/bin:$PATH"
APP=explorer/cypmode_explorer.py
EXPORT=explorer/_export
mode="${1:-run}"
case "$mode" in
  run)   exec marimo run "$APP" ;;
  edit)  exec marimo edit "$APP" ;;
  html)  marimo export html "$APP" -o "${2:-$EXPORT/index.html}" --no-include-code ;;
  ipynb) marimo export ipynb "$APP" -o "${2:-$EXPORT/cypmode_explorer.ipynb}" --include-outputs ;;
  wasm)
    out="${2:-$EXPORT/wasm}"
    marimo export html-wasm "$APP" -o "$out" --mode run   # the folder holds the front-end files of marimo; their interface strings are the library's own
    find "$out" -maxdepth 1 -name '*.md' -delete   # the export places a markdown file of the program's own instructions beside the page; the published folder holds only the page and its data
    ;;
  *) echo "usage: bash scripts/explorer.sh run|edit|html|ipynb|wasm" >&2; exit 2 ;;
esac
