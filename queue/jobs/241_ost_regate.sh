#!/bin/bash
source queue/lib.sh
# The registered consequence of job 207's verdict: the confirmatory restraint gates repeated with OpenStructure's LDDT-PLI (a few seconds, no model run).
need_file cypmode/validation/ost_regate.py
need_done 207_ost_vs_ours
python3 -m cypmode.validation.ost_regate > $L/ost_regate.log 2>&1
