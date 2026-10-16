#!/bin/bash
source queue/lib.sh
# Rank blend of hits-only and selection-aware CYP2D6 stacks, judged by the worst case across regimes (cache-only, fast).
need_file cypmode/challenge/hedged_blend.py
python -m cypmode.challenge.hedged_blend --isoform CYP2D6 --repeats 3 > $L/hedged_blend_CYP2D6.log 2>&1
