#!/bin/bash
source queue/lib.sh
# Stage 1 of CheMeleon: the stacked comparison on repeat 0 (5 outer folds) for CYP3A4. Written to a .txt file on purpose: a stage 1 screen is not a pre-registered gate and must not enter the ledger.
# When job 202 decided "not run: cost" no comparison is made and the file says so (chemeleon_stage.decide reads a file without a stacked line as STOP). CYP2D6 is not part of the design (decided 2026-10-16).
need_file cypmode/challenge/aux_experiments.py
need_done 203_chemprop_CYP3A4 204_chemprop_CYP2D6
decision=$(head -n 1 data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt 2>/dev/null)
case "$decision" in
  mps|cpu) ;;
  *) echo "CheMeleon stage 1 not run for CYP3A4: ${decision:-no decision}" > $L/stage1_chemeleon_CYP3A4.txt; exit 0 ;;
esac
python3 -m cypmode.challenge.aux_experiments --experiment chemeleon --isoform CYP3A4 --repeats 1 > $L/stage1_chemeleon_CYP3A4.txt 2>&1 || exit 1
