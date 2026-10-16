#!/bin/bash
source queue/lib.sh
# The secondary stratum of the confirmation ligands (31 to 34 heavy atoms): reported next to the primary one, never used for the decision, so it runs after the final recipe.
need_file $L/restraint_expansion_prepare.log
all_lower_done 186_boltz_restraint_expansion_secondary || defer "earlier jobs not finished"
[ -f $IN/restraint_expansion_secondary_unguided.csv ] || { echo "the confirmation is not run (see job 176)"; exit 0; }
python3 scripts/clear_batch_errors.py $OUT/restraint_expansion_secondary_unguided_results.json
python3 -m cypmode.validation.batch $IN/restraint_expansion_secondary_unguided.csv $OUT/restraint_expansion_secondary_unguided_results.json --no-affinity --diffusion-samples 2 --max-parallel-samples 1 --protein-msa $MSA > $L/restraint_expansion_boltz_secondary_unguided.log 2>&1
python3 scripts/clear_batch_errors.py $OUT/restraint_expansion_secondary_febond_results.json
python3 -m cypmode.validation.batch $IN/restraint_expansion_secondary_febond.csv $OUT/restraint_expansion_secondary_febond_results.json --no-affinity --diffusion-samples 2 --max-parallel-samples 1 --donor-restraint bond --protein-msa $MSA > $L/restraint_expansion_boltz_secondary_febond.log 2>&1
python3 - <<'PY'
import json,sys
bad=[]
for stem in ("unguided", "febond"):
    d=json.load(open(f"data/structure_track/outputs/cyp3a4/restraint_expansion_secondary_{stem}_results.json")); bad += [k for k,v in d.items() if "error" in v]
sys.exit(1 if bad else 0)
PY
