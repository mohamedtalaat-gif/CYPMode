#!/bin/bash
source queue/lib.sh
# The selection-corrected stack on a second, published data set (the direct-inhibition panel of Veith et al. in the benchmark of the Therapeutics Data Commons), under the rule of cypmode/challenge/external_dataset.py written before the first run: the official protocol, the nested design, the negative control, the reliability by similarity, and the calibration.
need_file cypmode/challenge/external_dataset.py
need_file data/tdi_track/inputs/tdc_cache/admet_group/leaderboard_veith.csv
python -m cypmode.challenge.external_dataset --n-jobs 8 > $L/external_dataset.log 2>&1
