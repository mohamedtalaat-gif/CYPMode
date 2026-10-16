#!/bin/bash
# SAR made explicit: matched-molecular-pair transformation features, gated against the base stack.
source queue/lib.sh
need_file cypmode/challenge/mmp_transfer.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.mmp_transfer --isoform $iso --repeats 3 --run --n-jobs 4 > $L/mmp_transfer_$iso.log 2>&1 || exit 1
done
