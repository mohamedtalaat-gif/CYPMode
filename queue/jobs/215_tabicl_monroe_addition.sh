#!/bin/bash
source queue/lib.sh
# TabICL on Monroe embeddings added to the default stack once its cache is complete (jobs 213 and 214).
need_file cypmode/challenge/aux_experiments.py
need_done 213_tabicl_monroe_CYP3A4 214_tabicl_monroe_CYP2D6
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment tabicl_monroe --isoform $iso --repeats 3 > $L/aux_tabicl_monroe_$iso.log 2>&1 || exit 1
done
