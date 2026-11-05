#!/bin/bash
source queue/lib.sh
# The questions of the plan of the analysis after the deadline (docs/post_deadline/PLAN.md) about the TDI track: the gate (the scoring of the organisers reproduces the board and names the submitted file), the two candidates, the label rule and the scored set, and the threshold. It stops after the gate when the gate is not met.
need_file cypmode/post_deadline/tdi.py
need_done 907_post_deadline_released
all_lower_done 908_post_deadline_tdi || defer "earlier jobs not finished"
python3 -m cypmode.post_deadline.tdi > data/post_deadline/outputs/tdi.log 2>&1 || exit 1
