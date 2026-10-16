#!/bin/bash
source queue/lib.sh
# Every comparison that goes through the stack (StackedLogit) again, after its logistic meta-learner was made deterministic
# (tight tolerance, fixed seed; before, liblinear's random coordinate order changed the stack from process to process). Nothing is refitted:
# the prediction caches are complete, `--run` only evaluates. The verdict lines the final build parses are rewritten, then the replicate
# verdicts and the final build are redone.
need_file cypmode/challenge/nested.py
for iso in CYP3A4 CYP2D6; do
  python -m cypmode.challenge.physics_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/physics_experiment_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.pair_transfer --isoform $iso --repeats 3 --run --n-jobs 4 > $L/pair_transfer_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.pair_transfer --isoform $iso --repeats 3 --run --with-physics --n-jobs 4 > $L/pair_transfer_physics_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.mmp_transfer --isoform $iso --repeats 3 --run --n-jobs 4 > $L/mmp_transfer_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.frozen_embeddings --isoform $iso --repeats 3 --run --n-jobs 4 > $L/embeddings_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.test_likeness_training --isoform $iso --repeats 3 --run --n-jobs 4 > $L/test_likeness_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.arms_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/arms_experiment_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.committee_label_model --isoform $iso --repeats 3 --run --n-jobs 4 > $L/committee_label_model_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.physics_compact_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/physics_compact_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.physics_tierb_experiment --isoform $iso --repeats 3 --run --n-jobs 4 > $L/physics_tierb_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.xiso_transfer --isoform $iso --repeats 3 --run --n-jobs 4 > $L/xiso_transfer_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.physics_experiment --isoform $iso --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 --run --n-jobs 4 > $L/physics_experiment_f10_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.physics_compact_experiment --isoform $iso --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 --run --n-jobs 4 > $L/physics_compact_f10_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.physics_tierb_experiment --isoform $iso --repeats 3 --outer 10 --root data/tdi_track/outputs/nested_f10 --run --n-jobs 4 > $L/physics_tierb_f10_$iso.log 2>&1 || exit 1
  python -m cypmode.challenge.replicate_verdict --isoform $iso > $L/replicate_verdict_$iso.log 2>&1 || exit 1
done
python -m cypmode.challenge.population_experiment --isoform CYP2D6 --repeats 3 --run --n-jobs 4 > $L/population_panel_CYP2D6_r3.log 2>&1 || exit 1
python -m cypmode.challenge.population_experiment --isoform CYP2D6 --repeats 3 --sensitivity > $L/population_sensitivity_panel_CYP2D6_r3.log 2>&1 || exit 1
python -m cypmode.challenge.population_experiment --isoform CYP3A4 --repeats 1 --run --n-jobs 4 > $L/population_panel_CYP3A4.log 2>&1 || exit 1
python -m cypmode.challenge.selection_aware_v2 --isoform CYP2D6 --repeats 3 --run --n-jobs 4 > $L/selection_aware_v2_CYP2D6.log 2>&1 || exit 1
python -m cypmode.challenge.population_physics --isoform CYP2D6 --repeats 3 --run --n-jobs 4 > $L/population_physics_CYP2D6.log 2>&1 || exit 1
python -m cypmode.challenge.hedged_blend --isoform CYP2D6 --repeats 3 > $L/hedged_blend_CYP2D6.log 2>&1 || exit 1
python -m cypmode.challenge.calibration_experiment > $L/calibration.log 2>&1 || exit 1
python -m cypmode.challenge.block_holdout > $L/block_holdout.log 2>&1 || exit 1
python -m cypmode.challenge.final_tdi_build > $L/final_tdi_build.log 2>&1 || exit 1
python -m cypmode.challenge.final_tdi_build --design 10 --out-dir data/tdi_track/outputs/final_f10 > $L/final_tdi_build_f10.log 2>&1
