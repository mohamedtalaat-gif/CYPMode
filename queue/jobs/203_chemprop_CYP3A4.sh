#!/bin/bash
source queue/lib.sh
set -o pipefail
# CheMeleon, stage 1 of its staged design (cypmode/challenge/chemeleon_stage.py): the 25 cross-validation tasks of repeat 0 for CYP3A4, time-boxed chunks, resumable. The plain Chemprop cache is complete. Whether it runs, and on
# which device, is decided by job 202 (cypmode/challenge/chemeleon_speed.py): "mps", "cpu" or "not run: cost" (then nothing is run here and the ledger says so).
need_file cypmode/challenge/nested_runners_chemprop.py
cheap_candidates_settled  # the long Chemprop family waits for the cheaper candidates (queue/lib.sh)
need_done 202_chemeleon_speed
decision=$(head -n 1 data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt)
case "$decision" in
  mps|cpu) ;;
  *) echo "stage 1 not run for CYP3A4: $decision"; exit 0 ;;
esac
out=$(PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_chemprop --isoform CYP3A4 --variants chemeleon_mt_shift --max-repeat 0 --max-minutes 90 --accelerator "$decision" 2>&1 | tee -a $L/chemprop_CYP3A4.log | tail -n 1) || exit 1
chunk_status 203_chemprop_CYP3A4 "${out##* }"
