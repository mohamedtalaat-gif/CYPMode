#!/bin/bash
source queue/lib.sh
set -o pipefail
# The speed test and the decision on the fine-tuned CheMeleon (cypmode/challenge/chemeleon_speed.py, decided 2026-10-16 before any measurement): one task that is already in the cache from the processor is trained again
# on the Apple GPU, in a scratch directory, and the rule of the module decides whether stage 1 for CYP3A4 runs on the GPU, on the processor, or is not run for cost. It starts once the cheap candidates have settled, so that
# nothing else is heavy while it is timed. CYP2D6 was dropped from the CheMeleon design on the same date.
need_file cypmode/challenge/chemeleon_speed.py
cheap_candidates_settled
[ -f data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt ] && { echo "decision exists: $(cat data/tdi_track/outputs/comparisons/chemeleon_speed_decision.txt)"; exit 0; }
PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.chemeleon_speed > $L/chemeleon_speed.log 2>&1 || exit 1
