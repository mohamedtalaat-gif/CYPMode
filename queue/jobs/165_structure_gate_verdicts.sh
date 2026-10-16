#!/bin/bash
source queue/lib.sh
# Machine-readable restraint and pilot verdicts (STRUCTURE GATE lines) from the scored pose tables of job 140.
need_file cypmode/validation/structure_gates.py
[ -f queue/state/done/140_score_restraint_and_pilot ] && [ -f queue/state/done/144_score_cryoem_pilot ] || [ -n "$REPRODUCE" ] || defer "restraint, pilot and cryo-EM pilot runs not scored yet"
python3 -m cypmode.validation.structure_gates > $L/structure_gates.log 2>&1
