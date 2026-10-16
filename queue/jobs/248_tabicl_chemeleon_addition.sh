#!/bin/bash
source queue/lib.sh
# TabICL on the frozen CheMeleon embedding added to the default stack once its cache is complete (jobs 246 and 247); the pre-registered gate of addition_experiment decides.
need_file cypmode/challenge/aux_experiments.py
need_done 246_tabicl_chemeleon_CYP3A4 247_tabicl_chemeleon_CYP2D6
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment tabicl_chemeleon --isoform $iso --repeats 3 > $L/aux_tabicl_chemeleon_$iso.log 2>&1 || exit 1
done
