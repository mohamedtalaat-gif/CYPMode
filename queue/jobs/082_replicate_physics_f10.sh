#!/bin/bash
source queue/lib.sh
# The physics comparisons again on the 10-outer-fold plan (tier A, compact, tier B).
need_file data/tdi_track/outputs/caches/physics_b_cache.csv
[ -f data/tdi_track/outputs/nested_f10/CYP2D6/plan.npz ] || defer "10-fold grid not built yet"
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.physics_experiment --isoform $iso --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 --run --n-jobs 4 > $L/physics_experiment_f10_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.physics_compact_experiment --isoform $iso --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 --run --n-jobs 4 > $L/physics_compact_f10_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.physics_tierb_experiment --isoform $iso --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 --run --n-jobs 4 > $L/physics_tierb_f10_$iso.log 2>&1 || exit 1
done
