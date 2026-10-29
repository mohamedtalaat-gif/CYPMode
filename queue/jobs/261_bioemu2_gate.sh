#!/bin/bash
source queue/lib.sh
# Scores the BioEmu2 poses of the benchmark against the crystal poses with the yardstick of the other engines and runs the engine-consensus gate registered on 2026-10-23, with BioEmu2
# as the added engine; the verdict is the STRUCTURE GATE line of the log.
need_done 260_bioemu2_benchmark
need_file cypmode/validation/selection_validation.py
grep -q "bioemu2" cypmode/validation/selection_validation.py || defer "the gate code for BioEmu2 is not written"
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark.csv').Molecule_Name))")
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --engines bioemu2 --names $NAMES --out $OUT/benchmark_engine_pose_validation_bioemu2.csv > $L/bench_score_bioemu2.log 2>&1 || exit 1
python3 -m cypmode.validation.selection_validation --engine bioemu2 > $L/selection_validation_bioemu2.log 2>&1 || exit 1
