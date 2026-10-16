#!/bin/bash
source queue/lib.sh
# Label-shift estimate of how hit-like the blinded test structures are (input to the CYP2D6 A/B decision).
need_file cypmode/challenge/test_nonhit_share.py
python -m cypmode.challenge.test_nonhit_share --isoform CYP2D6 > $L/test_nonhit_share_CYP2D6.log 2>&1
