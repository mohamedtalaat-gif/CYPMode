#!/bin/bash
source queue/lib.sh
# The softmax-temperature selector (T small: the best model, T large: an equal-weight committee) against the logistic blend, from the cached predictions of the default stack; also the exploration-exploitation curve.
need_file cypmode/challenge/softmax_selector.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.softmax_selector --isoform $iso > $L/softmax_selector_$iso.log 2>&1 || exit 1
done
