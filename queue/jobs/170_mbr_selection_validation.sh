#!/bin/bash
source queue/lib.sh
need_file $OUT/benchmark_engine_pose_validation.csv   # written by job 160 once every benchmark pose is scored
need_file cypmode/validation/selection_validation.py
python3 -m cypmode.validation.selection_validation > $L/selection_validation.log 2>&1
