#!/bin/bash
source queue/lib.sh
# CYP2D6: the same gated test of the ligand-based hypothesis model, and its permuted-label control (see job 269).
need_file cypmode/challenge/aux_experiments.py
need_done 266_hypothesis_extract
python3 -m cypmode.challenge.aux_experiments --experiment hypothesis --isoform CYP2D6 --repeats 3 --run > $L/aux_hypothesis_CYP2D6.log 2>&1 || exit 1
python3 -m cypmode.challenge.aux_experiments --experiment hypothesis_control --isoform CYP2D6 --repeats 3 --run > $L/exploratory_hypothesis_permuted_CYP2D6.log 2>&1 || exit 1
