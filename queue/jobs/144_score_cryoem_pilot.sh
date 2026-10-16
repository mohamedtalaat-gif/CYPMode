#!/bin/bash
source queue/lib.sh
# S3: true ligand RMSD of the cryo-EM template runs, then the pool comparison against the unguided pools (same thresholds as the other pilot levers).
need_file $IN/structure_benchmark_cryoem_pilot.csv
[ -f queue/state/done/140_score_restraint_and_pilot ] || [ -n "$REPRODUCE" ] || defer "unguided pose table not scored yet"
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark_cryoem_pilot.csv').Molecule_Name))")
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $NAMES --out $OUT/diversity_pilot_poses_cryoem.csv > $L/cryoem_pilot_score.log 2>&1 || exit 1
python3 -m cypmode.validation.diversity_pilot report --pilot-poses $OUT/diversity_pilot_poses_cryoem.csv --unguided-poses $OUT/benchmark_engine_pose_validation_unguided_boltz.csv > $L/cryoem_pilot_report.log 2>&1
