#!/bin/bash
source queue/lib.sh
python3 scripts/clear_batch_errors.py $OUT/diversity_pilot_results.json
python3 -m cypmode.validation.batch $IN/structure_benchmark_diversity_pilot.csv $OUT/diversity_pilot_results.json --no-affinity --max-parallel-samples 1 --protein-msa $MSA > $L/pilot_boltz.log 2>&1
python3 - <<'PY'
import json,sys
d=json.load(open("data/structure_track/outputs/cyp3a4/diversity_pilot_results.json")); sys.exit(1 if any("error" in v for v in d.values()) else 0)
PY
