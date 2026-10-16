#!/bin/bash
source queue/lib.sh
# Compact physics: the 20 most useful physics columns chosen inside every task.
need_file cypmode/challenge/physics_compact_experiment.py
need_file data/tdi_track/outputs/caches/physics_cache.csv
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.physics_compact_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/physics_compact_$iso.log 2>&1 || exit 1
done
