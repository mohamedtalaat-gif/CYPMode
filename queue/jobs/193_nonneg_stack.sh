#!/bin/bash
source queue/lib.sh
# Non-negative stacking as an alternative selector, from the cached predictions of the default stack (same folds, same yardstick).
need_file cypmode/challenge/nonneg_stack.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.nonneg_stack --isoform $iso > $L/nonneg_stack_$iso.log 2>&1 || exit 1
done
