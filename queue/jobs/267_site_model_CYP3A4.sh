#!/bin/bash
source queue/lib.sh
# CYP3A4: the gated test of the structure-based site model of the CYP3A4 pocket (site_cyp3a4, the addition), and the control with the site model of the CYP2D6 pocket (site_cyp2d6 on CYP3A4 data), whose log is not a
# gate and enters no ledger. 3 repeats of 5 outer folds, the default stack as the base. The six columns of the model were written by job 262.
need_file cypmode/challenge/aux_experiments.py
need_done 262_pocket_pharmacophore_extract
python3 -m cypmode.challenge.aux_experiments --experiment site_cyp3a4 --isoform CYP3A4 --repeats 3 --run > $L/aux_site_cyp3a4_CYP3A4.log 2>&1 || exit 1
python3 -m cypmode.challenge.aux_experiments --experiment site_cyp2d6 --isoform CYP3A4 --repeats 3 --run > $L/exploratory_site_control_CYP3A4.log 2>&1 || exit 1
