#!/bin/bash
source queue/lib.sh
# Multitask network with the official screen and Emax readings as auxiliary heads, completed for every task of the nested plan (time-boxed chunks, resumable).
set -o pipefail
need_file cypmode/challenge/nested_runners_torch.py
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4 --groups mlp_multitask_aux --max-minutes 90 2>&1 | tee -a $L/mlp_aux_CYP3A4.log | tail -n 1) || exit 1
chunk_status 208_mlp_aux_CYP3A4 "${out##* }"
