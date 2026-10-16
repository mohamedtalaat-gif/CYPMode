#!/bin/bash
source queue/lib.sh
# S3: Boltz-2 with the cryo-EM CYP3A4 entries as protein templates, on the three pilot ligands (3 templates each, 1 sample each).
need_file cypmode/validation/cryoem_templates.py
python3 -m cypmode.validation.cryoem_templates fetch > $L/cryoem_fetch.log 2>&1 || exit 1
[ -f $IN/cyp3a4_cryoem_templates.json ] || python3 -m cypmode.validation.cryoem_templates prepare >> $L/cryoem_fetch.log 2>&1 || exit 1
python3 -m cypmode.validation.cryoem_templates build --only bench-A1CIW bench-PK9 bench-TPF --out $IN/structure_benchmark_cryoem_pilot.csv > $L/cryoem_build.log 2>&1 || exit 1
python3 scripts/clear_batch_errors.py $OUT/cryoem_pilot_results.json
python3 -m cypmode.validation.batch $IN/structure_benchmark_cryoem_pilot.csv $OUT/cryoem_pilot_results.json --no-affinity --max-parallel-samples 1 --protein-msa $MSA > $L/cryoem_pilot_boltz.log 2>&1
python3 - <<'PY'
import sys, pandas as pd
from cypmode.validation.pose_consensus import boltz_cif_paths
names = pd.read_csv("data/structure_track/inputs/structure_benchmark_cryoem_pilot.csv").Molecule_Name
sys.exit(1 if [n for n in names if not boltz_cif_paths(n)] else 0)
PY
