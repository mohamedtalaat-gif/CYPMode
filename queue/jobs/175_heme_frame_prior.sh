#!/bin/bash
source queue/lib.sh
# The heme-frame prior on the scored benchmark poses (heme_frame_prior.py).
need_file $OUT/benchmark_engine_pose_validation.csv   # written by job 160 once every benchmark pose is scored
need_file cypmode/validation/heme_frame_prior.py
python3 -m cypmode.validation.heme_frame_prior > $L/heme_frame_prior.log 2>&1
