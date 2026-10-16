#!/bin/bash
source queue/lib.sh
# TabPFN added to the default stack once its cache is complete (jobs 221 and 222); the pre-registered gate of addition_experiment decides.
need_file cypmode/challenge/aux_experiments.py
need_done 221_tabpfn_CYP3A4 222_tabpfn_CYP2D6
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment tabpfn --isoform $iso --repeats 3 > $L/aux_tabpfn_$iso.log 2>&1 || exit 1
done
