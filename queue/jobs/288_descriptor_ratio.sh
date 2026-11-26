#!/bin/bash
source queue/lib.sh
# The number of columns against the number of compounds (the practice that Jorgensen 2006 discourages): widths, effective dimension, extra columns without signal for the candidates and for the stack, fewer columns, under the rule of cypmode/challenge/descriptor_ratio.py written before the first run.
need_file cypmode/challenge/descriptor_ratio.py
need_file data/tdi_track/outputs/nested/CYP3A4/plan.npz
need_file data/tdi_track/outputs/nested/CYP2D6/plan.npz
python -m cypmode.challenge.descriptor_ratio --n-jobs 8 > $L/descriptor_ratio.log 2>&1
