#!/bin/bash
source queue/lib.sh
# Plain Chemprop (cache complete on the nested plan) added to the default stack, both enzymes; CheMeleon has its own staged jobs (203, 204, 236 to 239).
need_file cypmode/challenge/aux_experiments.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment chemprop --isoform $iso --repeats 3 > $L/aux_chemprop_$iso.log 2>&1 || exit 1
done
