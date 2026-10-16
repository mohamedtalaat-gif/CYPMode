#!/bin/bash
# Continuation of scripts/run_tdi_then_structure.sh: the first pass used 5 outer folds (too few to be
# significant), so once the CYP3A4 run of that pass has finished, extend both isoforms to 3 repeats x 5 folds
# (the new repeats reuse the cached predictions of repeat 0), then restart the structure queue and the
# diversity-pilot waiter. Run with:  caffeinate -i -m bash scripts/run_tdi_power_then_structure.sh
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
OUT=data/structure_track/outputs/cyp3a4
LOGS=$OUT/logs
source scripts/guards.sh

while pgrep -f "confidence_experiment --isoform CYP3A4 --repeats 1" > /dev/null; do sleep 10; done
wait_for_ac
start_guard

for iso in CYP2D6 CYP3A4; do
  stage "TDI label-confidence experiment, $iso, 3 repeats x 5 folds (CPU)"
  python3 -m cypmode.challenge.confidence_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > "$LOGS/confidence_${iso}_r3.log" 2>&1
done
stage "TDI power run done -- restarting the structure queue"

kill $GUARD_PID 2>/dev/null
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/run_structure_benchmark.sh > "$LOGS/queue.out" 2>&1 < /dev/null &
sleep 5
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/run_diversity_pilot.sh > "$LOGS/pilot.out" 2>&1 < /dev/null &
