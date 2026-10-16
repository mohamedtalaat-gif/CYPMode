#!/bin/bash
source queue/lib.sh
# The multitask network (cache complete) added to the default stack, both isoforms.
need_file cypmode/challenge/aux_experiments.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment mlp --isoform $iso --repeats 3 > $L/aux_mlp_$iso.log 2>&1 || exit 1
done
