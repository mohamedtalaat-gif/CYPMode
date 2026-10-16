#!/bin/bash
source queue/lib.sh
# The official single-concentration screen as a predicted block (cross-fitted inside every task); validated alone and inside the stack, both isoforms.
need_file cypmode/challenge/aux_experiments.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment screen --isoform $iso --repeats 3 --run > $L/aux_screen_$iso.log 2>&1 || exit 1
done
