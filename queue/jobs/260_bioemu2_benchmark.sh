#!/bin/bash
source queue/lib.sh
# BioEmu2 on the 10 benchmark ligands, 3 samples each (seeds 0 to 2, the first 256 rows of the alignment of Boltz-2, the Boltz-2 inputs, no contacts), registered in the checklist item S7 before any
# benchmark ligand was predicted and allowed by the feasibility rule of job 259. It runs alone on the machine, resumes ligand by ligand and fails unless every ligand ends with its three samples.
need_done 259_bioemu2_probe
need_file cypmode/validation/bioemu2.py
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark.csv').Molecule_Name))")
.venv-bioemu2/bin/python -m cypmode.validation.bioemu2 --names $NAMES --samples 3 > $L/bioemu2_benchmark.log 2>&1 || exit 1
for n in $NAMES; do [ "$(ls data/structure_track/outputs/bioemu2/cyp3a4/$n/sample_*.cif 2>/dev/null | wc -l)" -ge 3 ] || { echo "no three samples for $n" >> $L/bioemu2_benchmark.log; exit 1; }; done
