#!/bin/bash
source queue/lib.sh
# The data bundle of the interactive explorer (explorer/public): the ledger, the paired comparisons, the selection rules, the benchmark poses, the log excerpts and the out-of-fold predictions of the
# training compounds, rebuilt from the final logs and caches after the last verdict; the build fails when a table names a blinded compound, mentions a held-back path or has the size of a blinded set.
need_file cypmode/explorer/data.py
python -m cypmode.explorer.data > $L/explorer_bundle.log 2>&1 || exit 1
