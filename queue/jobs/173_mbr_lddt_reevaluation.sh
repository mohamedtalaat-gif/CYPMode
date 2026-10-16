#!/bin/bash
source queue/lib.sh
# Minimum Bayes risk and the restraint re-scored with LDDT-PLI against the crystal (exploratory: the registered gate used RMSD). No compute beyond scoring.
need_file cypmode/validation/mbr_lddt_reevaluation.py
all_lower_done 173_mbr_lddt_reevaluation || defer "earlier jobs not finished"
python3 -m cypmode.validation.mbr_lddt_reevaluation > $L/mbr_lddt_reevaluation.log 2>&1
