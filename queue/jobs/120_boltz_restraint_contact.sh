#!/bin/bash
source queue/lib.sh
python3 scripts/clear_batch_errors.py $OUT/structure_benchmark_fecontact_results.json
python3 -m cypmode.validation.batch $IN/structure_benchmark_fecontact.csv $OUT/structure_benchmark_fecontact_results.json --no-affinity --diffusion-samples 3 --max-parallel-samples 1 --donor-restraint contact --protein-msa $MSA > $L/bench_boltz_fecontact.log 2>&1
python3 - <<'PY'
import json,sys
d=json.load(open("data/structure_track/outputs/cyp3a4/structure_benchmark_fecontact_results.json")); sys.exit(1 if any("error" in v for v in d.values()) else 0)
PY
