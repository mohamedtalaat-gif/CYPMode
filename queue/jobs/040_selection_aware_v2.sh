#!/bin/bash
source queue/lib.sh
# Stronger selection-aware model: cross-isoform hit-profile gate, Heckman-style correction term, gated vs the v1 sets.
need_file cypmode/challenge/selection_aware_v2.py
python -m cypmode.challenge.selection_aware_v2 --isoform CYP2D6 --repeats 3 --run --n-jobs 4 > $L/selection_aware_v2_CYP2D6.log 2>&1
