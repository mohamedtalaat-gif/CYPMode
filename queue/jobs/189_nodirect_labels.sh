#!/bin/bash
source queue/lib.sh
# The 1,249 CYP3A4 compounds without a direct curve, used through what their TDI-arm pIC50 implies (194 certain negatives; soft labels for the rest from a first-stage model on the pool rows of each task), alone and inside the stack.
need_file cypmode/challenge/aux_experiments.py
python3 -m cypmode.challenge.aux_experiments --experiment nodirect --isoform CYP3A4 --repeats 3 --run > $L/aux_nodirect_CYP3A4.log 2>&1
