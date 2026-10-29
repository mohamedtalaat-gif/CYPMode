#!/bin/bash
source queue/lib.sh
# The pocket-level pharmacophore block (declared on 2026-10-23 before any result): the density fields of the CYP3A4 and CYP2D6 pockets from their public crystal ligands in the frame of the heme,
# then the 22 columns of every compound of the training and the test file for each pocket (about 0.8 s per compound, four processes), cached once.
need_file cypmode/challenge/pocket_pharmacophore.py
need_file data/structure_track/inputs/cyp3a4_reference_ligands.json
need_file data/structure_track/inputs/cyp2d6_reference_ligands.json
python3 -m cypmode.challenge.pocket_pharmacophore --extract > $L/pocket_pharmacophore_extract.log 2>&1 || exit 1
need_file data/tdi_track/outputs/caches/pocket_pharmacophore_cache.csv
