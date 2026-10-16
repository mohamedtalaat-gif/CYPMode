#!/bin/bash
source queue/lib.sh
# Probabilistic label-process model: noise-aware committee of two-arm regressors -> P(label) and expected improvement.
need_file cypmode/challenge/committee_label_model.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.committee_label_model --isoform $iso --repeats 3 --run --n-jobs 4 > $L/committee_label_model_$iso.log 2>&1 || exit 1
done
