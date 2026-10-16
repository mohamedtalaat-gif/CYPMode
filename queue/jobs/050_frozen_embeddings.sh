#!/bin/bash
source queue/lib.sh
# Frozen CheMeleon embeddings: chemprop extraction in .venv-ml (no LightGBM), evaluation in .venv (no torch).
need_file cypmode/challenge/frozen_embeddings.py
[ -f ~/.chemprop/chemeleon_mp.pt ] || defer "CheMeleon weights missing"
.venv-ml/bin/python -m cypmode.challenge.frozen_embeddings --extract > $L/frozen_embeddings_extract.log 2>&1 || exit 1
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.frozen_embeddings --isoform $iso --repeats 3 --run --n-jobs 4 > $L/embeddings_$iso.log 2>&1 || exit 1
done
