#!/bin/bash
source queue/lib.sh
# The PXR-induction readings of the same compound library (pEC50 and Emax; training set and the two unblinded test phases of the PXR challenge) as a predicted block, alone and inside the stack.
need_file cypmode/challenge/aux_experiments.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment pxr --isoform $iso --repeats 3 --run > $L/aux_pxr_$iso.log 2>&1 || exit 1
done
