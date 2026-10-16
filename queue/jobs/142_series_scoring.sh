#!/bin/bash
source queue/lib.sh
# True ligand RMSD of every series pose against its crystal pose.
need_file $IN/structure_benchmark_series.csv
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark_series.csv').Molecule_Name))")
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $NAMES --out $OUT/series_pose_validation.csv > $L/series_score.log 2>&1
