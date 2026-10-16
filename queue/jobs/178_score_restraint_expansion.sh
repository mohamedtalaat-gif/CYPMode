#!/bin/bash
source queue/lib.sh
# Scores the confirmation runs (RMSD with the existing rescoring harness, LDDT-PLI with lddt_to_crystal) and writes the registered verdicts the final recipe reads.
need_file $L/restraint_expansion_prepare.log
all_lower_done 178_score_restraint_expansion || defer "earlier jobs not finished"
[ -f $IN/restraint_expansion_primary_unguided.csv ] || { echo "the confirmation is not run (see job 176)"; exit 0; }
for stratum in primary secondary; do
  for condition in unguided febond; do
    [ -f $OUT/restraint_expansion_${stratum}_${condition}_results.json ] || continue
    NAMES=$(python3 -c "import pandas as pd;print(' '.join(pd.read_csv('$IN/restraint_expansion_${stratum}_${condition}.csv').Molecule_Name))")
    [ -n "$NAMES" ] || continue
    python3 -m cypmode.validation.pose_rescoring_validation --engine-poses --names $NAMES --out $OUT/restraint_expansion_${stratum}_${condition}_poses.csv > $L/restraint_expansion_rmsd_${stratum}_${condition}.log 2>&1
  done
done
python3 -m cypmode.validation.restraint_expansion score > $L/restraint_expansion.log 2>&1
