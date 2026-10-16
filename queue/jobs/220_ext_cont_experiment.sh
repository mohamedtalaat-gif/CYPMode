#!/bin/bash
source queue/lib.sh
# The continuous external block, and then the Tox21 block as a separate test, added to the feature sets that carry the binary external block, alone and inside the stack (pre-registered gate of addition_experiment), both isoforms.
need_file cypmode/challenge/aux_experiments.py
need_file data/tdi_track/outputs/caches/ext_continuous.npz
need_file data/tdi_track/outputs/caches/ext_tox21.npz
all_lower_done 220_ext_cont_experiment || defer "earlier jobs not finished"
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment ext_cont --isoform $iso --repeats 3 --run > $L/aux_ext_cont_$iso.log 2>&1 || exit 1
  python3 -m cypmode.challenge.aux_experiments --experiment ext_tox --isoform $iso --repeats 3 --run > $L/aux_ext_tox_$iso.log 2>&1 || exit 1
done
