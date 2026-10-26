#!/bin/bash
source queue/lib.sh
# An estimate of the positive rate of the scored set that reads no leaderboard (label shift, cross-fitted soft adjusted classify-and-count), compared with the rate of the final build by a rule fixed before the first run.
need_file cypmode/challenge/label_shift.py
need_file data/tdi_track/outputs/final/manifest.json
python -m cypmode.challenge.label_shift --bootstrap 1000 > $L/label_shift.log 2>&1
