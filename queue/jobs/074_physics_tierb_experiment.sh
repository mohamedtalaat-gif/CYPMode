#!/bin/bash
source queue/lib.sh
need_file cypmode/challenge/physics_tierb_experiment.py
need_file data/tdi_track/outputs/caches/physics_b_cache.csv
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.physics_tierb_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/physics_tierb_$iso.log 2>&1 || exit 1
done
