#!/bin/bash
source queue/lib.sh
# 10-outer-fold replicate: the default candidate grid on its own plan (outer 10, inner 4, 3 repeats), deployment fits included.
need_file cypmode/challenge/nested_runners.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.nested_runners --isoform $iso --repeats 3 --outer 10 --inner 4 --root data/tdi_track/outputs/nested_f10 --n-jobs 4 > $L/grid_f10_$iso.log 2>&1 || exit 1
done
