#!/bin/bash
source queue/lib.sh
set -o pipefail
# Stage 2 of CheMeleon (CYP3A4): repeats 1 and 2 and the deployment fits, only if the rule of cypmode/challenge/chemeleon_stage.py asks for it after stage 1; otherwise nothing is run.
need_file cypmode/challenge/chemeleon_stage.py
need_done 236_chemeleon_stage1_screen
[ "$(python -m cypmode.challenge.chemeleon_stage --isoform CYP3A4)" = "CONTINUE" ] || { echo "stage 2 not run for CYP3A4: stage 1 did not meet the rule"; exit 0; }
device=$(head -n 1 data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt)  # "mps" or "cpu" (job 202); stage 2 is only reached after a stage 1 that ran
out=$(PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_chemprop --isoform CYP3A4 --variants chemeleon_mt_shift --max-minutes 90 --accelerator "$device" 2>&1 | tee -a $L/chemprop_CYP3A4.log | tail -n 1) || exit 1
chunk_status 237_chemeleon_stage2_CYP3A4 "${out##* }"
