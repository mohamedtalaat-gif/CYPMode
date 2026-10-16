#!/bin/bash
source queue/lib.sh
python3 scripts/clear_batch_errors.py $OUT/structure_benchmark_febond_results.json
python3 -m cypmode.validation.batch $IN/structure_benchmark_febond.csv $OUT/structure_benchmark_febond_results.json --no-affinity --diffusion-samples 3 --max-parallel-samples 1 --donor-restraint bond --protein-msa $MSA > $L/bench_boltz_febond.log 2>&1
python3 - <<'PY'
import json,sys
d=json.load(open("data/structure_track/outputs/cyp3a4/structure_benchmark_febond_results.json")); bad=[k for k,v in d.items() if "error" in v]
sys.exit(1 if bad else 0)   # non-zero exit (a retry after clearing errors) when any ligand failed
PY
