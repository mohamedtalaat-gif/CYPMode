#!/bin/bash
source queue/lib.sh
# The scoring again with the secondary stratum included (the verdicts of the primary stratum do not change).
need_file $L/restraint_expansion_prepare.log
all_lower_done 187_score_restraint_expansion_secondary || defer "earlier jobs not finished"
[ -f $IN/restraint_expansion_primary_unguided.csv ] || { echo "the confirmation is not run (see job 176)"; exit 0; }
bash queue/jobs/178_score_restraint_expansion.sh
