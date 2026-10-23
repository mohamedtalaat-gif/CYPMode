#!/bin/bash
source queue/lib.sh
# The replicate rule declared with the two pharmacophore models: when the stacked gate of a group is MET on 3 repeats of 5 outer folds, the same comparison is made on 10 outer folds and the replicate verdicts are
# written again (the final build lets the group in only when both are MET). Groups: site_cyp3a4 (CYP3A4 only), site_cyp2d6 (CYP2D6 only) and hypothesis (both). When a gate is not met nothing is run, and that is written down.
need_file cypmode/challenge/replicate_verdict.py
need_done 267_site_model_CYP3A4 268_site_model_CYP2D6 269_hypothesis_model_CYP3A4 270_hypothesis_model_CYP2D6
ran=0
for pair in CYP3A4:site_cyp3a4 CYP2D6:site_cyp2d6 CYP3A4:hypothesis CYP2D6:hypothesis; do
  iso=${pair%%:*}; exp=${pair##*:}
  if grep -Eq "^$iso GATE \(stacked[^)]*\): MET$" $L/aux_${exp}_$iso.log; then
    [ -f data/tdi_track/outputs/nested_f10/$iso/plan.npz ] || defer "10-fold grid not built yet"
    python3 -m cypmode.challenge.aux_experiments --experiment $exp --isoform $iso --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 --run > $L/aux_${exp}_f10_$iso.log 2>&1 || exit 1
    ran=1
  else
    echo "pharmacophore models, $exp on $iso: the stacked gate is not met on 15 folds; no replicate is run" | tee -a $L/exploratory_pharmacophore_models_replicate.log
  fi
done
if [ $ran = 1 ]; then
  for iso in CYP3A4 CYP2D6; do python3 -m cypmode.challenge.replicate_verdict --isoform $iso > $L/replicate_verdict_$iso.log 2>&1 || exit 1; done
fi
