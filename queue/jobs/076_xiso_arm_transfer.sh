#!/bin/bash
# Arms of the other isoforms, predicted from structure (cross-fitted), as features of the target-isoform classifier; alone and stacked.
source queue/lib.sh
need_file cypmode/challenge/xiso_transfer.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.xiso_transfer --isoform $iso --repeats 3 --run --n-jobs 4 > $L/xiso_transfer_$iso.log 2>&1 || exit 1
done
