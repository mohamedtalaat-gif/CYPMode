#!/bin/bash
source queue/lib.sh
# The confirmation of the restraint on crystal ligands that were not used to decide it (plan registered in data/structure_track/inputs/restraint_expansion_plan.json).
# Run only when minimum Bayes risk did not pass its exploratory LDDT-PLI re-scoring (job 173): the order of the decision was fixed on 2026-10-12.
need_file cypmode/validation/restraint_expansion.py
need_file $L/mbr_lddt_reevaluation.log
all_lower_done 176_restraint_expansion_prepare || defer "earlier jobs not finished"
if grep -q "^STRUCTURE GATE mbr_lddt_exploratory.*: MET" $L/mbr_lddt_reevaluation.log; then echo "minimum Bayes risk passed the exploratory re-scoring: the confirmation of the restraint is not run"; exit 0; fi
python3 -m cypmode.validation.restraint_expansion prepare > $L/restraint_expansion_prepare.log 2>&1
