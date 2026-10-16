#!/bin/bash
source queue/lib.sh
# Importance-weighted training from the unlabeled test structures; gated alone and stacked.
need_file cypmode/challenge/test_likeness_training.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.test_likeness_training --isoform $iso --repeats 3 --run --n-jobs 4 > $L/test_likeness_$iso.log 2>&1 || exit 1
done
