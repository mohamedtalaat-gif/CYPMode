#!/bin/bash
# GFN2-xTB atom/site/pharmacophore physics block: tblite-only extraction first, then the LightGBM evaluation in
# separate processes (tblite and LightGBM deadlock in one process).
source queue/lib.sh
need_file cypmode/challenge/physics_block.py
need_file cypmode/challenge/physics_experiment.py
python -m cypmode.challenge.physics_block --extract --n-workers 4 > $L/physics_extract.log 2>&1 || exit 1
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.physics_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/physics_experiment_$iso.log 2>&1 || exit 1
done
