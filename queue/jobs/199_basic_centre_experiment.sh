#!/bin/bash
source queue/lib.sh
# The basic-centre block added to feature sets A and A+fps, validated alone and inside the stack (it matters most for CYP2D6), both isoforms.
need_file data/tdi_track/outputs/caches/basic_centre.csv
for iso in CYP2D6 CYP3A4; do
  python3 -m cypmode.challenge.aux_experiments --experiment basic_centre --isoform $iso --repeats 3 --run > $L/aux_basic_centre_$iso.log 2>&1 || exit 1
done
