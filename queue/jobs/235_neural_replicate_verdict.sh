#!/bin/bash
source queue/lib.sh
# The replicate verdicts again, now with the neural pair (CYP3A4 only; the physics lines are recomputed from the same logs).
need_file cypmode/challenge/replicate_verdict.py
need_done 233_neural_addition 234_neural_addition_f10
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.replicate_verdict --isoform $iso > $L/replicate_verdict_$iso.log 2>&1 || exit 1
done
