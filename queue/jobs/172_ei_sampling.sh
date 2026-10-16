#!/bin/bash
source queue/lib.sh
need_file $OUT/benchmark_engine_pose_validation.csv   # written by job 160 once every benchmark pose is scored
# Expected-improvement sampling: extra Boltz samples go to the ligands whose pose pool disagrees most (needs the pooled benchmark poses).
need_file cypmode/validation/ei_sampling.py
all_lower_done 172_ei_sampling || defer "pose pools not finished"
python3 -m cypmode.validation.ei_sampling > $L/ei_sampling.log 2>&1
