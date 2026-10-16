#!/bin/bash
source queue/lib.sh
# The neural pair on the 10-outer-fold plan (CYP3A4): the confirmatory comparison against the default stack, declared before any 10-fold result existed.
need_file cypmode/challenge/aux_experiments.py
need_done 232_neural_f10_fill_CYP3A4
python3 -m cypmode.challenge.aux_experiments --experiment neural --isoform CYP3A4 --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 > $L/aux_neural_f10_CYP3A4.log 2>&1 || exit 1
