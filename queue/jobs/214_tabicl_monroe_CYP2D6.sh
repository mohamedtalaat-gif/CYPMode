#!/bin/bash
source queue/lib.sh
# TabICL on the frozen Monroe embedding, completed for every task of the nested plan (time-boxed chunks, resumable).
set -o pipefail
need_file data/tdi_track/outputs/caches/monroe_embeddings.npz
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP2D6 --groups tabicl_monroe --max-minutes 90 2>&1 | tee -a $L/tabicl_monroe_CYP2D6.log | tail -n 1) || exit 1
chunk_status 214_tabicl_monroe_CYP2D6 "${out##* }"
