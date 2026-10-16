#!/bin/bash
source queue/lib.sh
# Run the frozen recipe on the 20 official ligands and validate every structure. Needs the recipe module.
need_file cypmode/validation/final_recipe.py
all_lower_done 180_structure_final_recipe_run || defer "structure gates not finished"
python3 -m cypmode.validation.final_recipe > $L/final_recipe.log 2>&1
