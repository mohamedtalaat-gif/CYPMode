#!/bin/bash
source queue/lib.sh
set -o pipefail
# The all-in-one comparison (cypmode/challenge/all_in_one.py, declared 2026-10-22 before any result): the final stack of each isoform against one stack of every cached candidate, on the experiments' plan, from the prediction caches
# only (nothing is fitted but the blends). Written to a log that no gate parser reads (exploratory_*); it cannot change the final model.
need_file cypmode/challenge/all_in_one.py
for iso in CYP3A4 CYP2D6; do
  PYTHONPATH=. python -m cypmode.challenge.all_in_one --isoform $iso > $L/exploratory_all_in_one_$iso.log 2>&1 || exit 1
done
