#!/bin/bash
source queue/lib.sh
# SAR-aware selection: poses that agree with the poses of the ligand's analogs.
need_file cypmode/validation/series_consensus.py
need_file $OUT/series_pose_validation.csv
python3 -m cypmode.validation.series_consensus > $L/series_consensus.log 2>&1
