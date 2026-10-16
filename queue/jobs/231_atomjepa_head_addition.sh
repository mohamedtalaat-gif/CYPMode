#!/bin/bash
source queue/lib.sh
# The Atom-JEPA attention readout added to the default stack once its cache is complete (jobs 229 and 230); the pre-registered gate decides.
need_file cypmode/challenge/aux_experiments.py
need_done 229_atomjepa_head_CYP3A4 230_atomjepa_head_CYP2D6
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment atomjepa_head --isoform $iso --repeats 3 > $L/aux_atomjepa_head_$iso.log 2>&1 || exit 1
done
