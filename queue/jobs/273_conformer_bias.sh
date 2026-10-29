#!/bin/bash
source queue/lib.sh
# The check of the conformational bias (declared 2026-10-23 before any code). For each of the 148 crystal ligand copies, 300 conformers, the nearest one to the crystal pose and the energy window that the bound state needs, and the coverage
# of the bound state by the ensemble of 12 conformers within 5 kcal/mol that the pocket block uses. No blinded compound is involved. Four processes.
need_file cypmode/validation/conformer_bias.py
need_file data/structure_track/inputs/cyp3a4_reference_ligands.json
python3 -m cypmode.validation.conformer_bias > $L/conformer_bias.log 2>&1 || exit 1
