#!/bin/bash
source queue/lib.sh
# Boltz-2 on the ligands of the crystal-anchored benchmark without a restraint (5 diffusion samples each, one sample at a time), resuming after the failed entries are cleared.
python3 scripts/clear_batch_errors.py $OUT/structure_benchmark_results.json
python3 -m cypmode.validation.batch $IN/structure_benchmark.csv $OUT/structure_benchmark_results.json --no-affinity --diffusion-samples 5 --max-parallel-samples 1 --protein-msa $MSA > $L/bench_boltz.log 2>&1
