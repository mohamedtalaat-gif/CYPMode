#!/bin/bash
source queue/lib.sh
set -o pipefail
# TabPFN on the same frozen CheMeleon representation as job 246/247, completed for every task of the nested plan (time-boxed chunks, resumable; the licence marker of job 221 is needed, see there).
need_file data/tdi_track/outputs/caches/chemeleon_embeddings.npz
need_file data/tdi_track/inputs/tabpfn_licence_accepted.txt
.venv-ml/bin/python -c "import tabpfn" 2>/dev/null || defer "the package tabpfn is not installed in .venv-ml"
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4 --groups tabpfn_chemeleon --max-minutes 90 2>&1 | tee -a $L/tabpfn_chemeleon_CYP3A4.log | tail -n 1) || exit 1
chunk_status 249_tabpfn_chemeleon_CYP3A4 "${out##* }"
