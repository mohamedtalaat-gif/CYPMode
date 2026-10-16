#!/bin/bash
source queue/lib.sh
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark.csv').Molecule_Name))")
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $NAMES --out $OUT/benchmark_engine_pose_validation.csv > $L/bench_score.log 2>&1
python3 -m cypmode.validation.structure_benchmark_report $OUT/benchmark_engine_pose_validation.csv --out $OUT/benchmark_selection_outcomes.csv > $L/bench_report.log 2>&1
