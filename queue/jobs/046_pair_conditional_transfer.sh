#!/bin/bash
# SAR-aware pair-conditional label transfer (descriptor deltas); the physics-delta variant runs in job 075.
source queue/lib.sh
need_file cypmode/challenge/pair_transfer.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.pair_transfer --isoform $iso --repeats 3 --run --n-jobs 4 > $L/pair_transfer_$iso.log 2>&1 || exit 1
done
