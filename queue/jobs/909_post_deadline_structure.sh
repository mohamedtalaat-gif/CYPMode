#!/bin/bash
source queue/lib.sh
# The questions of the plan of the analysis after the deadline (docs/post_deadline/PLAN.md) about the structure track: the gate (the scoring of the organisers, run with OpenStructure 2.11.1, reproduces the board), the submitted poses against the published Boltz-2 baseline and the unguided pose, the pool of each compound, and the heme iron. The scores of the poses are kept in data/post_deadline/outputs/structure_poses.csv, so that a second run reads them instead of scoring the poses again.
need_file cypmode/post_deadline/structure.py
need_done 907_post_deadline_released
all_lower_done 909_post_deadline_structure || defer "earlier jobs not finished"
python3 -m cypmode.post_deadline.structure > data/post_deadline/outputs/structure.log 2>&1 || exit 1
