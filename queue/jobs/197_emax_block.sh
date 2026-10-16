#!/bin/bash
source queue/lib.sh
# The official Emax readings of both arms as a predicted block, validated alone and inside the stack.
need_file cypmode/challenge/aux_experiments.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment emax --isoform $iso --repeats 3 --run > $L/aux_emax_$iso.log 2>&1 || exit 1
done
