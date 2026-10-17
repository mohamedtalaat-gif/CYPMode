#!/bin/bash
source queue/lib.sh
# The verdict of the 10-outer-fold replicate for both isoforms (replicate_verdict.py), written to replicate_verdict_<isoform>.log.
need_file cypmode/challenge/replicate_verdict.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.replicate_verdict --isoform $iso > $L/replicate_verdict_$iso.log 2>&1 || exit 1
done
