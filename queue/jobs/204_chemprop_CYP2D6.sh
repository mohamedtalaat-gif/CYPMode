#!/bin/bash
source queue/lib.sh
# CheMeleon stage 1 for CYP2D6 is not run: the enzyme was dropped from the CheMeleon design on 2026-10-16 (plain Chemprop is at chance on CYP2D6, outer AUC 0.54 and 0.53; the frozen CheMeleon block reaches 0.63; see
# cypmode/challenge/chemeleon_speed.py). The four tasks already in the cache stay there and are not used.
echo "stage 1 not run for CYP2D6: dropped from the design (decision of 2026-10-16)"
