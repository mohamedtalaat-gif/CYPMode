#!/bin/bash
source queue/lib.sh
# The decision procedure for the final CYP2D6 entry (cypmode/challenge/selection_decision.py, fixed on 2026-10-25 before any upload): the deployment table of the hits-only file, the selection-aware file and their blend with
# bootstrap intervals, the evidence on the share of never-promoted compounds in the scored set, and the operating characteristics of reading the live board as a detector of the regime; it writes the threshold that the rule
# applies to the two uploads. Cache-only, after the hedged blend (job 014) and the label-shift estimate (job 013).
need_file cypmode/challenge/selection_decision.py
need_file $L/hedged_blend_CYP2D6.log
need_file $L/test_nonhit_share_CYP2D6.log
python -m cypmode.challenge.selection_decision --isoform CYP2D6 --repeats 3 > $L/selection_decision_CYP2D6.log 2>&1
