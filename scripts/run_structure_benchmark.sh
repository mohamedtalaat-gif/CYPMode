#!/bin/bash
# Crystal-anchored structure benchmark queue, run end to end and resumable (every stage skips finished work):
#   1  Boltz-2, unguided, 5 samples one at a time           (10 ligands)
#   1b Boltz-2 with a bond restraint, donor N -> heme FE    (ligands that have a Type II donor, 3 samples)
#   1c Boltz-2 with a contact restraint at Boltz's 4 A      (same ligands, 3 samples)
#   2  OpenFold3, unguided, 3 samples                       (10 ligands)
#   3  score every pose against its crystal structure
#   4  selection-rule report and the restraint comparison
# Progress goes to $LOGS/queue_stage.txt. Run it with:  caffeinate -i -m bash scripts/run_structure_benchmark.sh
#
# Guards, all from real incidents on this 16 GB Mac:
#  * it waits for AC power before starting and freezes the engines (SIGSTOP) when the battery is under
#    MIN_BATTERY_PCT off AC, resuming when the charger is back -- a flat battery mid-run lost a night;
#  * PyTorch's MPS allocator is trimmed early (a hard 0.9x cap made every Boltz forward pass fail, so the
#    engines do need that headroom) and Boltz samples run one at a time -- five in parallel exhausted swap
#    and the machine took a watchdog kernel panic;
#  * the engine is killed if free memory stays under MIN_FREE_PCT, so one compound fails and is recorded
#    instead of the whole machine going down. For Boltz a failed compound stays in the results JSON and is
#    skipped on re-runs: delete its entry to retry it.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.3
export PYTORCH_MPS_LOW_WATERMARK_RATIO=1.0
MIN_FREE_PCT=6
MIN_BATTERY_PCT=25

OUT=data/structure_track/outputs/cyp3a4
IN=data/structure_track/inputs
LOGS=$OUT/logs
mkdir -p "$LOGS"
NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark.csv').Molecule_Name))")
stage() { echo "$(date '+%m-%d %H:%M:%S') $1 | swap used $(sysctl -n vm.swapusage | awk '{print $6}')" >> "$LOGS/queue_stage.txt"; }
on_ac() { pmset -g batt | head -1 | grep -q "AC Power"; }
battery_pct() { pmset -g batt | grep -o '[0-9]*%' | head -1 | tr -d '%'; }
engines() { pkill "$1" -f "boltz predict"; pkill "$1" -f "run_openfold"; pkill "$1" -f "cypmode.validation"; }

guard() {
  low=0; paused=0
  while true; do
    sleep 5
    free=$(memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%", "", $NF); print $NF}')
    if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE_PCT" ]; then low=$((low + 1)); else low=0; fi
    if [ "$low" -ge 2 ]; then
      stage "GUARD: free memory ${free}% for 10 s -- killing the running engine to protect the machine"
      pkill -f "boltz predict"; pkill -f "run_openfold"; low=0; sleep 20
    fi
    pct=$(battery_pct)
    if ! on_ac && [ -n "$pct" ] && [ "$pct" -lt "$MIN_BATTERY_PCT" ] && [ "$paused" = 0 ]; then
      paused=1; stage "PAUSED: battery ${pct}% and off AC -- engines frozen until the charger is back"; engines -STOP
    elif on_ac && [ "$paused" = 1 ]; then
      paused=0; stage "RESUMED: AC power is back"; engines -CONT
    fi
  done
}

until on_ac; do
  stage "waiting for AC power before starting (battery $(battery_pct)%)"
  sleep 60
done
guard &
GUARD_PID=$!
trap 'kill $GUARD_PID 2>/dev/null' EXIT

stage "1/4 Boltz-2 unguided, 5 samples one at a time, 10 benchmark ligands"
python3 -m cypmode.validation.batch $IN/structure_benchmark.csv $OUT/structure_benchmark_results.json \
  --no-affinity --diffusion-samples 5 --max-parallel-samples 1 > "$LOGS/bench_boltz.log" 2>&1

for mode in bond contact; do
  stage "1/4 Boltz-2 with the donor-to-iron $mode restraint, 3 samples"
  python3 -m cypmode.validation.batch $IN/structure_benchmark_fe$mode.csv $OUT/structure_benchmark_fe${mode}_results.json \
    --no-affinity --diffusion-samples 3 --max-parallel-samples 1 --donor-restraint $mode > "$LOGS/bench_boltz_fe$mode.log" 2>&1
done

stage "2/4 OpenFold3 unguided, 3 samples, 10 benchmark ligands"
python3 -m cypmode.validation.openfold3 --names $NAMES --samples 3 > "$LOGS/bench_of3.log" 2>&1

stage "3/4 scoring every pose against its crystal structure"
python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $NAMES \
  --out $OUT/benchmark_engine_pose_validation.csv > "$LOGS/bench_score.log" 2>&1
for mode in bond contact; do
  RNAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/structure_benchmark_fe$mode.csv').Molecule_Name))")
  python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $RNAMES \
    --out $OUT/benchmark_engine_pose_validation_fe$mode.csv > "$LOGS/bench_score_fe$mode.log" 2>&1
done

stage "4/4 selection-rule report and restraint comparison"
python3 -m cypmode.validation.structure_benchmark_report $OUT/benchmark_engine_pose_validation.csv \
  --out $OUT/benchmark_selection_outcomes.csv > "$LOGS/bench_report.log" 2>&1
python3 -m cypmode.validation.structure_benchmark_report --samples 3 --compare \
  unguided=$OUT/benchmark_engine_pose_validation.csv febond=$OUT/benchmark_engine_pose_validation_febond.csv \
  fecontact=$OUT/benchmark_engine_pose_validation_fecontact.csv > "$LOGS/bench_restraint_comparison.log" 2>&1

stage "DONE"
