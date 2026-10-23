#!/bin/bash
source queue/lib.sh
# Stage 1 of the OpenADMET pretraining weights: the stacked comparison on repeat 0 (5 outer folds) for CYP3A4, both starting points added to the default stack together. Written to a .txt file on purpose: a stage 1 screen is not a
# pre-registered gate and must not enter the ledger. When job 202 decided not to run, no comparison is made and the file says so (chemeleon_stage.decide reads a file without a stacked line as STOP).
need_file cypmode/challenge/aux_experiments.py
need_done 255_openadmet_weights_stage1_CYP3A4
decision=$(head -n 1 data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt 2>/dev/null)
case "$decision" in
  mps|cpu) ;;
  *) echo "stage 1 of the OpenADMET weights not run for CYP3A4: ${decision:-no decision}" > $L/stage1_chemprop_foundation_CYP3A4.txt; exit 0 ;;
esac
python3 -m cypmode.challenge.aux_experiments --experiment chemprop_foundation --isoform CYP3A4 --repeats 1 --screen > $L/stage1_chemprop_foundation_CYP3A4.txt 2>&1 || exit 1
