#!/bin/bash
source queue/lib.sh
set -o pipefail
# TabICL candidate completed for every task of the nested plan (time-boxed chunks, resumable; the memory guard stays on).
need_file cypmode/challenge/nested_runners_torch.py
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4 --groups tabicl_svd --max-minutes 90 2>&1 | tee -a $L/tabicl_CYP3A4.log | tail -n 1) || exit 1
chunk_status 201_tabicl_CYP3A4 "${out##* }"
