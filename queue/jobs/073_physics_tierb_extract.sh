#!/bin/bash
source queue/lib.sh
# Tier-B physics extraction (tblite only, own process): Boltzmann conformers and relaxed hydrogen abstraction.
need_file cypmode/challenge/physics_tierb.py
need_file data/tdi_track/outputs/caches/physics_atoms.csv.gz
python -m cypmode.challenge.physics_tierb --extract --n-workers 4 > $L/physics_tierb_extract.log 2>&1
