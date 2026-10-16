#!/bin/bash
source queue/lib.sh
FE_B=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark_febond.csv').Molecule_Name))")
FE_C=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark_fecontact.csv').Molecule_Name))")
PN=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark_diversity_pilot.csv').Molecule_Name))")
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark.csv').Molecule_Name))")
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $NAMES --out $OUT/benchmark_engine_pose_validation_unguided_boltz.csv > $L/bench_score_unguided.log 2>&1
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $FE_B --out $OUT/benchmark_engine_pose_validation_febond.csv > $L/bench_score_febond.log 2>&1
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $FE_C --out $OUT/benchmark_engine_pose_validation_fecontact.csv > $L/bench_score_fecontact.log 2>&1
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $PN --out $OUT/diversity_pilot_poses.csv > $L/pilot_score.log 2>&1
python3 -m cypmode.validation.structure_benchmark_report --samples 3 --compare unguided=$OUT/benchmark_engine_pose_validation_unguided_boltz.csv febond=$OUT/benchmark_engine_pose_validation_febond.csv fecontact=$OUT/benchmark_engine_pose_validation_fecontact.csv > $L/bench_restraint_comparison.log 2>&1
python3 -m cypmode.validation.diversity_pilot report --pilot-poses $OUT/diversity_pilot_poses.csv --unguided-poses $OUT/benchmark_engine_pose_validation_unguided_boltz.csv > $L/pilot_report.log 2>&1
