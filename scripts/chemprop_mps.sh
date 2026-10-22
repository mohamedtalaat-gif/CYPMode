#!/bin/bash
# Chemprop on the Apple GPU with deterministic algorithms in warn-only mode (see scripts/chemprop_warn_only.py); the arguments are those of the chemprop command line.
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/../.venv-ml/bin/python" "$here/chemprop_warn_only.py" "$@"
