#!/bin/bash
source queue/lib.sh
# TabICL added to the default stack once its cache is complete (jobs 201 and 202).
need_file cypmode/challenge/aux_experiments.py
need_done 201_tabicl_CYP3A4 202_tabicl_CYP2D6
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment tabicl --isoform $iso --repeats 3 > $L/aux_tabicl_$iso.log 2>&1 || exit 1
done
