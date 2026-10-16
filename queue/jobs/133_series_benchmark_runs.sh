#!/bin/bash
source queue/lib.sh
# Boltz-2 on three crystal-anchored analog pairs (6 ligands, 4 samples each): the benchmark of the series-consensus rule.
need_file cypmode/validation/series.py
python3 -m cypmode.validation.series --build > $L/series_build.log 2>&1 || exit 1
python3 scripts/clear_batch_errors.py $OUT/series_results.json
python3 -m cypmode.validation.batch $IN/structure_benchmark_series.csv $OUT/series_results.json --no-affinity --diffusion-samples 4 --max-parallel-samples 1 --protein-msa $MSA > $L/series_boltz.log 2>&1
# poses exist for every ligand even when post-processing of a distorted pose raises: the check is the written structures
python3 - <<'PY'
import sys, pandas as pd
from cypmode.validation.pose_consensus import boltz_cif_paths
names = pd.read_csv("data/structure_track/inputs/structure_benchmark_series.csv").Molecule_Name
missing = [n for n in names if len(boltz_cif_paths(n)) < 3]
sys.exit(1 if missing else 0)
PY
