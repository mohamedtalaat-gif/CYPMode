#!/bin/bash
# The label-confidence experiment (cypmode.challenge.confidence_experiment) needs the whole machine's memory, so
# it cannot share it with Boltz on this 16 GB Mac: stop the structure queue, run the CPU experiment for both
# isoforms, then bring the structure queue and the diversity-pilot waiter back. Every structure stage resumes
# where it stopped (finished ligands are skipped).
# Run with:  caffeinate -i -m bash scripts/run_tdi_then_structure.sh
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
OUT=data/structure_track/outputs/cyp3a4
LOGS=$OUT/logs
mkdir -p "$LOGS"
source scripts/guards.sh

pkill -f run_diversity_pilot.sh; pkill -f run_structure_benchmark.sh; engines -TERM; sleep 10
wait_for_ac
start_guard

for iso in CYP2D6 CYP3A4; do
  stage "TDI label-confidence experiment, $iso (CPU)"
  python3 -m cypmode.challenge.confidence_experiment --isoform $iso --repeats 1 --run --n-jobs 4 > "$LOGS/confidence_$iso.log" 2>&1
done
stage "TDI experiment done -- restarting the structure queue"

kill $GUARD_PID 2>/dev/null
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/run_structure_benchmark.sh > "$LOGS/queue.out" 2>&1 < /dev/null &
sleep 5
nohup perl -e 'use POSIX qw(setsid); setsid(); exec @ARGV' caffeinate -i -m bash scripts/run_diversity_pilot.sh > "$LOGS/pilot.out" 2>&1 < /dev/null &
