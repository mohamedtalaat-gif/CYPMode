#!/bin/bash
source queue/lib.sh
# The OpenADMET pretraining weights added to the default stack (15 folds, the ordinary gate) for CYP3A4 when stage 2 ran; otherwise they stay "screened on 5 folds, not continued" (or "not run" when job 202 decided so).
need_file cypmode/challenge/aux_experiments.py
need_done 257_openadmet_weights_stage2_CYP3A4
[ "$(python -m cypmode.challenge.chemeleon_stage --isoform CYP3A4 --model chemprop_foundation)" = "CONTINUE" ] || { echo "no 15-fold comparison of the OpenADMET weights: stage 1 did not meet the rule"; exit 0; }
python3 -m cypmode.challenge.aux_experiments --experiment chemprop_foundation --isoform CYP3A4 --repeats 3 > $L/aux_chemprop_foundation_CYP3A4.log 2>&1 || exit 1
