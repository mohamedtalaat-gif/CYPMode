#!/bin/bash
# Pose-diversity pilot (cypmode.validation.diversity_pilot): do seeds, crystal-conformer templates or a lower
# sampling step scale make Boltz-2's pose pool differ, and does the difference reach the crystal pose?
# Three ligands whose unguided first pick is far off, one run per (lever, k): ~1.5 h each on this Mac.
#
# It waits until the restraint stages of scripts/run_structure_benchmark.sh are done (that script logs
# "2/4 OpenFold3" and starts the multi-hour OpenFold3 stage), stops that queue, runs the pilot, then restarts
# it -- every stage of the main queue resumes where it stopped. Run with:
#   caffeinate -i -m bash scripts/run_diversity_pilot.sh
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
OUT=data/structure_track/outputs/cyp3a4
IN=data/structure_track/inputs
LOGS=$OUT/logs
mkdir -p "$LOGS"
source scripts/guards.sh

until grep -q "2/4 OpenFold3" "$LOGS/queue_stage.txt" 2>/dev/null; do sleep 120; done
pkill -f run_structure_benchmark.sh; engines -TERM; sleep 10
wait_for_ac
start_guard

stage "PILOT 1/3 Boltz-2 runs: seeds, conformer templates, step scale on 3 ligands"
python3 -m cypmode.validation.batch $IN/structure_benchmark_diversity_pilot.csv $OUT/diversity_pilot_results.json \
  --no-affinity --max-parallel-samples 1 > "$LOGS/pilot_boltz.log" 2>&1

stage "PILOT 2/3 scoring the pilot poses against their crystal structures"
PNAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark_diversity_pilot.csv').Molecule_Name))")
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $PNAMES \
  --out $OUT/diversity_pilot_poses.csv > "$LOGS/pilot_score.log" 2>&1

stage "PILOT 3/3 verdict"
python3 -m cypmode.validation.diversity_pilot report --pilot-poses $OUT/diversity_pilot_poses.csv \
  --unguided-poses $OUT/benchmark_engine_pose_validation_unguided9.csv > "$LOGS/pilot_report.log" 2>&1
stage "PILOT DONE -- restarting the main structure queue"

kill $GUARD_PID 2>/dev/null
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/run_structure_benchmark.sh \
  > "$LOGS/queue.out" 2>&1 < /dev/null &
