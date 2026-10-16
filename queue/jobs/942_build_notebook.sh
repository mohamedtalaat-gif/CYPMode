#!/bin/bash
source queue/lib.sh
# The notebook (notebooks/PLAN.md): execute every cell against the final files, run the checks (cells, glossary, coverage), write the .ipynb, then the HTML and PDF exports (the build writes them itself when every check passes, so a failed check fails this job).
# Runs after the figures job (940), because the notebook embeds the figures.
need_file notebooks/build_notebook.py
all_lower_done 942_build_notebook || defer "earlier jobs not finished"
python -m notebooks.build_notebook > $L/build_notebook.log 2>&1 || { echo "the notebook build failed; see $L/build_notebook.log"; exit 1; }
