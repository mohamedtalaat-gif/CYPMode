#!/bin/bash
source queue/lib.sh
# TabPFN on the frozen CheMeleon embedding added to the default stack once its cache is complete (jobs 249 and 250); the pre-registered gate of addition_experiment decides.
need_file cypmode/challenge/aux_experiments.py
need_done 249_tabpfn_chemeleon_CYP3A4 250_tabpfn_chemeleon_CYP2D6
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment tabpfn_chemeleon --isoform $iso --repeats 3 > $L/aux_tabpfn_chemeleon_$iso.log 2>&1 || exit 1
done
