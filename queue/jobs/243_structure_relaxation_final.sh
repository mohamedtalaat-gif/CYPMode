#!/bin/bash
source queue/lib.sh
# The force-field relaxation (gate of job 218, MET) applied to the 20 poses of the final recipe under the rule registered in cypmode/validation/relax_final.py, and the final zip rebuilt from the poses that
# pass it; the candidate zip is checked with the portal's rules and the scorer's own PoseBusters call before it replaces anything, and the recipe's own zip is kept as structures_unrelaxed.zip.
need_file cypmode/validation/relax_final.py
need_file data/structure_track/outputs/cyp3a4/final/structures_final.zip
need_file "$HOME/miniforge3/envs/openmm/bin/python"
need_done 192_structure_margin_aware_zip 218_relaxation_gate
python3 -m cypmode.validation.relax_final > $L/relax_final.log 2>&1
