#!/bin/bash
# One CPU window for the TDI experiments, then back to the structure queue (Boltz cannot share memory with them):
#   1. wait for the CYP2D6 15-fold label-confidence run already in progress,
#   2. CYP3A4 15-fold label-confidence run,
#   3. the alert-context block (alone and with confident-row training), 15 folds, both isoforms,
#   4. restart the structure queue and the diversity-pilot waiter.
# Run with:  caffeinate -i -m bash scripts/run_tdi_context_then_structure.sh
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
OUT=data/structure_track/outputs/cyp3a4
LOGS=$OUT/logs
source scripts/guards.sh

while pgrep -f "confidence_experiment --isoform CYP2D6 --repeats 3" > /dev/null; do sleep 10; done
wait_for_ac
start_guard

stage "TDI label-confidence experiment, CYP3A4, 3 repeats x 5 folds (CPU)"
python3 -m cypmode.challenge.confidence_experiment --isoform CYP3A4 --repeats 3 --run --n-jobs 4 > "$LOGS/confidence_CYP3A4_r3.log" 2>&1
for iso in CYP2D6 CYP3A4; do
  stage "TDI alert-context experiment, $iso, 3 repeats x 5 folds (CPU)"
  python3 -m cypmode.challenge.confidence_experiment --isoform $iso --repeats 3 --run --experiment context --n-jobs 4 > "$LOGS/context_${iso}_r3.log" 2>&1
done
stage "TDI experiments done -- restarting the structure queue"

kill $GUARD_PID 2>/dev/null
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/run_structure_benchmark.sh > "$LOGS/queue.out" 2>&1 < /dev/null &
sleep 5
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/run_diversity_pilot.sh > "$LOGS/pilot.out" 2>&1 < /dev/null &
