#!/bin/bash
source queue/lib.sh
# The neural pair (multitask network and plain Chemprop, caches complete) added to the default stack on the nested plan, both enzymes; the ledger rows of the 15-fold design.
need_file cypmode/challenge/aux_experiments.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment neural --isoform $iso --repeats 3 > $L/aux_neural_$iso.log 2>&1 || exit 1
done
