#!/bin/bash
source queue/lib.sh
set -o pipefail
# Confirmatory replicate of the neural pair on CYP3A4 (group neural): plain Chemprop and the multitask network on the 10-outer-fold plan (155 tasks each, deployment fits included),
# time-boxed chunks, resumable.
need_file cypmode/challenge/nested_runners_chemprop.py
[ -f data/tdi_track/outputs/nested_f10/CYP3A4/plan.npz ] || defer "10-fold grid not built yet"
a=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_chemprop --isoform CYP3A4 --root data/tdi_track/outputs/nested_f10 --variants chemprop_mt_shift --max-minutes 90 2>&1 | tee -a $L/neural_f10_CYP3A4.log | tail -n 1) || exit 1
b=$(PYTHONPATH=. .venv-ml/bin/python -m cypmode.challenge.nested_runners_torch --isoform CYP3A4 --root data/tdi_track/outputs/nested_f10 --groups mlp_multitask --max-minutes 60 2>&1 | tee -a $L/neural_f10_CYP3A4.log | tail -n 1) || exit 1
chunk_status 232_neural_f10_fill_CYP3A4 "$(( ${a##* } + ${b##* } ))"
