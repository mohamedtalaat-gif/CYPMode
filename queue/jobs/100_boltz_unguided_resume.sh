#!/bin/bash
source queue/lib.sh
python3 scripts/clear_batch_errors.py $OUT/structure_benchmark_results.json
python3 -m cypmode.validation.batch $IN/structure_benchmark.csv $OUT/structure_benchmark_results.json --no-affinity --diffusion-samples 5 --max-parallel-samples 1 --protein-msa $MSA > $L/bench_boltz.log 2>&1
