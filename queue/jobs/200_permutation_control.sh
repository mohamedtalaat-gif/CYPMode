#!/bin/bash
source queue/lib.sh
# Negative control of the whole nested pipeline: the pool labels are permuted at random and the procedure is repeated; the outer AUC must be 0.5.
need_file cypmode/challenge/permutation_control.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.permutation_control --isoform $iso --permutations 3 > $L/permutation_control_$iso.log 2>&1 || exit 1
done
