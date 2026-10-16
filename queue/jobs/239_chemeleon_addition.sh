#!/bin/bash
source queue/lib.sh
# CheMeleon added to the default stack (15 folds, the ordinary gate) for each enzyme whose stage 2 ran; the others stay "screened on 5 folds, not continued".
need_file cypmode/challenge/aux_experiments.py
need_done 237_chemeleon_stage2_CYP3A4 238_chemeleon_stage2_CYP2D6
for iso in CYP3A4 CYP2D6; do
  [ "$(python -m cypmode.challenge.chemeleon_stage --isoform $iso)" = "CONTINUE" ] || continue
  python3 -m cypmode.challenge.aux_experiments --experiment chemeleon --isoform $iso --repeats 3 > $L/aux_chemeleon_$iso.log 2>&1 || exit 1
done
