#!/bin/bash
source queue/lib.sh
set -o pipefail
# Stage 2 of CheMeleon (CYP2D6; the enzyme was dropped from the design on 2026-10-16, so no stage 1 file exists and the rule below always says stop): repeats 1 and 2 and the deployment fits, only if the rule of cypmode/challenge/chemeleon_stage.py asks for it after stage 1; otherwise nothing is run.
need_file cypmode/challenge/chemeleon_stage.py
need_done 236_chemeleon_stage1_screen
[ "$(python -m cypmode.challenge.chemeleon_stage --isoform CYP2D6)" = "CONTINUE" ] || { echo "stage 2 not run for CYP2D6: stage 1 did not meet the rule"; exit 0; }
out=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_chemprop --isoform CYP2D6 --variants chemeleon_mt_shift --max-minutes 90 2>&1 | tee -a $L/chemprop_CYP2D6.log | tail -n 1) || exit 1
chunk_status 238_chemeleon_stage2_CYP2D6 "${out##* }"
