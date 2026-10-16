#!/bin/bash
source queue/lib.sh
# OpenFold3 on the 10 benchmark ligands, 3 samples each, in the memory mode the probe (job 149) found. Ligands that already have structures are
# skipped, a ligand without one is retried, and the job fails unless every ligand ends with a structure.
need_file data/structure_track/inputs/openfold3_memory_mode.txt
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark.csv').Molecule_Name))")
python3 -m cypmode.validation.openfold3 --names $NAMES --samples 3 --strict > $L/bench_of3.log 2>&1
