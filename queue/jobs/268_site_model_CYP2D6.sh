#!/bin/bash
source queue/lib.sh
# CYP2D6: the gated test of the structure-based site model of the CYP2D6 pocket (site_cyp2d6, the addition), and the control with the site model of the CYP3A4 pocket (site_cyp3a4 on CYP2D6 data), whose log is not a gate.
need_file cypmode/challenge/aux_experiments.py
need_done 262_pocket_pharmacophore_extract
python3 -m cypmode.challenge.aux_experiments --experiment site_cyp2d6 --isoform CYP2D6 --repeats 3 --run > $L/aux_site_cyp2d6_CYP2D6.log 2>&1 || exit 1
python3 -m cypmode.challenge.aux_experiments --experiment site_cyp3a4 --isoform CYP2D6 --repeats 3 --run > $L/exploratory_site_control_CYP2D6.log 2>&1 || exit 1
