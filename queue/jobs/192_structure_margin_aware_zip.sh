#!/bin/bash
source queue/lib.sh
# Margin-aware pose choice: every pool pose is checked with the organisers' own PoseBusters call, the final recipe keeps only valid poses whose relative
# distance to the receptor is at least 0.80 (the scorer's threshold is 0.75) and rewrites the candidate zip, and the zip is verified the way the scorer reads it.
need_file cypmode/validation/official_posebusters.py
all_lower_done 192_structure_margin_aware_zip || defer "earlier jobs not finished"
python3 -m cypmode.validation.pose_margins > $L/official_posebusters_pools.log 2>&1 || exit 1
python3 -m cypmode.validation.final_recipe > $L/final_recipe_margin.log 2>&1 || exit 1
python3 -m cypmode.validation.official_posebusters data/structure_track/outputs/cyp3a4/final/structures_final.zip > $L/official_posebusters_final.log 2>&1
