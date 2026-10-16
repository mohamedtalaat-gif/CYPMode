#!/bin/bash
source queue/lib.sh
# The two-part form of the label, P(t > 4.301) x P(t - d > log10 2): the gate reads every compound with a TDI-arm curve. Validated alone and inside the stack.
need_file cypmode/challenge/aux_experiments.py
for iso in CYP3A4 CYP2D6; do
  python3 -m cypmode.challenge.aux_experiments --experiment gate_shift --isoform $iso --repeats 3 --run > $L/aux_gate_shift_$iso.log 2>&1 || exit 1
done
