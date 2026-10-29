#!/bin/bash
# The interactive explorer (a marimo notebook). The first start prepares the environment by itself, so that nothing has to be installed or activated by hand.
#   bash scripts/explorer.sh run            the explorer as an application (read-only); also what a double click on explorer/run_explorer.command does
#   bash scripts/explorer.sh edit           the same notebook in the editor
#   bash scripts/explorer.sh html [DIR]     a static page (no code, the outputs as they stand)
#   bash scripts/explorer.sh ipynb [FILE]   a Jupyter notebook with the outputs
#   bash scripts/explorer.sh wasm [DIR]     a page that runs in a browser without a server (the data files are bundled; the browser loads Pyodide from a CDN)
#   bash scripts/explorer.sh plan           say which notebook and which requirements would be used and whether the environment is ready; change nothing
# The notebook is explorer/full/app.py when the complete edition is there, otherwise explorer/cypmode_explorer.py. The environment is .venv-marimo. It is built on the first start (and again when the
# requirements file changes) from Python 3.11 or newer if the machine has one; if not, from uv, which the script downloads into ./.tools after asking, and which then fetches Python itself.
# The explorer never starts a cross-validation, a co-folding run or a queue job: the public explorer reads the tables of its bundle and the complete explorer runs small calculations (a molecule, a formula).
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=.venv-marimo
if [ -f explorer/full/app.py ]; then APP=explorer/full/app.py; REQ=explorer/full/requirements.txt; else APP=explorer/cypmode_explorer.py; REQ=environment/requirements-lock-marimo.txt; fi
EXPORT=explorer/_export

find_python() {
  local c
  for c in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then echo "$c"; return 0; fi
  done
  return 1
}

env_ready() { [ -x "$ENV/bin/marimo" ] && cmp -s "$REQ" "$ENV/requirements.installed"; }

ensure_env() {
  env_ready && return 0
  echo "First start: preparing the environment in $ENV from $REQ. This takes a few minutes once; the next start is immediate."
  local py
  if py=$(find_python); then
    "$py" -m venv "$ENV"
    "$ENV/bin/python" -m pip install --quiet --upgrade pip
    "$ENV/bin/python" -m pip install -r "$REQ"
  else
    echo "Python 3.11 or newer was not found on this machine."
    echo "This script can download uv (a small installer from https://astral.sh/uv) into ./.tools; uv then fetches Python and the packages."
    if [ "${EXPLORER_ASSUME_YES:-}" != "1" ]; then
      if [ -t 0 ]; then read -r -p "Press Enter to continue, or Ctrl+C to stop: " _; else echo "No terminal to ask: set EXPLORER_ASSUME_YES=1 to continue." >&2; exit 1; fi
    fi
    if [ ! -x .tools/uv ]; then
      mkdir -p .tools
      curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh
    fi
    .tools/uv venv --python 3.12 "$ENV"
    .tools/uv pip install --python "$ENV/bin/python" -r "$REQ"
  fi
  cp "$REQ" "$ENV/requirements.installed"
}

mode="${1:-run}"
if [ "$mode" = "plan" ]; then
  echo "notebook: $APP"
  echo "requirements: $REQ"
  if env_ready; then echo "environment: ready ($ENV)"; else echo "environment: to be built on the first start ($ENV)"; fi
  if py=$(find_python); then echo "python: $py"; else echo "python: none of 3.11 or newer; uv would be offered"; fi
  exit 0
fi
ensure_env
export PATH="$PWD/$ENV/bin:$PATH"
case "$mode" in
  run)   exec marimo run "$APP" ;;
  edit)  exec marimo edit "$APP" ;;
  html)  marimo export html "$APP" -o "${2:-$EXPORT/index.html}" --no-include-code ;;
  ipynb) marimo export ipynb "$APP" -o "${2:-$EXPORT/cypmode_explorer.ipynb}" --include-outputs ;;
  wasm)
    out="${2:-$EXPORT/wasm}"
    marimo export html-wasm "$APP" -o "$out" --mode run   # the folder holds the front-end files of marimo; their interface strings are the library's own
    find "$out" -maxdepth 1 -name '*.md' -delete   # the export places a markdown file of the program's own instructions beside the page; the published folder holds only the page and its files
    ;;
  *) echo "usage: bash scripts/explorer.sh run|edit|html|ipynb|wasm|plan" >&2; exit 2 ;;
esac
