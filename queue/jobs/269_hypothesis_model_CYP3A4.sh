#!/bin/bash
source queue/lib.sh
# CYP3A4: the gated test of the ligand-based hypothesis model (hypothesis, fitted inside every task on the training rows of the task, out of fold for those rows), and its control, the same model on labels permuted
# within the training rows (hypothesis_control), whose log is not a gate and enters no ledger. 3 repeats of 5 outer folds, the default stack as the base.
need_file cypmode/challenge/aux_experiments.py
need_done 266_hypothesis_extract
python3 -m cypmode.challenge.aux_experiments --experiment hypothesis --isoform CYP3A4 --repeats 3 --run > $L/aux_hypothesis_CYP3A4.log 2>&1 || exit 1
python3 -m cypmode.challenge.aux_experiments --experiment hypothesis_control --isoform CYP3A4 --repeats 3 --run > $L/exploratory_hypothesis_permuted_CYP3A4.log 2>&1 || exit 1
