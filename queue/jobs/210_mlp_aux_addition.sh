#!/bin/bash
source queue/lib.sh
# The multitask network with auxiliary heads added to the default stack once its cache is complete (jobs 208 and 209).
need_file cypmode/challenge/aux_experiments.py
need_done 208_mlp_aux_CYP3A4 209_mlp_aux_CYP2D6
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment mlp_aux --isoform $iso --repeats 3 > $L/aux_mlp_aux_$iso.log 2>&1 || exit 1
done
