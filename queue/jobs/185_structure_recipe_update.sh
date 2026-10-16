#!/bin/bash
source queue/lib.sh
# The final recipe once more after the engine-consensus and selection verdicts exist (jobs 149-170): it reads the verdicts again, adds what they call
# for to the Boltz runs of job 180 (OpenFold3 samples for the official ligands, MBR or rank-term selection) and rewrites the candidate zip.
need_file cypmode/validation/final_recipe.py
all_lower_done 185_structure_recipe_update || defer "earlier jobs not finished"
python3 -m cypmode.validation.final_recipe > $L/final_recipe_update.log 2>&1
