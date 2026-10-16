#!/bin/bash
# Floor-clipped two-arm regressors vs the raw-arm ones, alone and stacked.
source queue/lib.sh
need_file cypmode/challenge/arms_experiment.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.arms_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/arms_experiment_$iso.log 2>&1 || exit 1
done
