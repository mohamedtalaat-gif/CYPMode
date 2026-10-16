#!/bin/bash
source queue/lib.sh
# Physics, molecule and pharmacophore terms of the pose pools, judged on the pooled crystal-anchored benchmark.
need_file cypmode/validation/pose_awareness.py
need_file $OUT/diversity_pilot_poses.csv
python3 -m cypmode.validation.pose_awareness --mode pooled > $L/pose_awareness.log 2>&1 || exit 1
python3 -m cypmode.validation.pose_awareness --mode single --out $OUT/pose_awareness_single.csv > $L/pose_awareness_single.log 2>&1
