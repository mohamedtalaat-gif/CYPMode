#!/bin/bash
source queue/lib.sh
# Three official ligands (OCNT-2395415, OCNT-2312554, OCNT-2312966) were first published as racemates and were predicted from the racemic SMILES; the organisers replaced them by the SMILES of the models used for
# scoring (checklist item "Official structure set corrected by the organisers", registered 2026-10-24 before this job ran). The runs made from the old SMILES are set aside unchanged, the base runs and then the runs of
# the recipe (unguided, restrained and extra samples) are made again from the corrected SMILES, and the recipe chooses again under the unchanged rule with one addition: a pose is eligible only when it carries the
# official stereochemistry. About 3.5 hours on the GPU, resumable at every step. HELD BACK: the logs are named stereo_*.log, the set-aside folder is under outputs/out, the pools table is under final/.
need_file cypmode/validation/stereo_rerun.py
need_file cypmode/validation/pose_stereo.py
need_file $OUT/final/consensus.csv
all_lower_done 275_stereo_official_rerun || defer "earlier jobs not finished"
SET_ASIDE=data/structure_track/outputs/out/superseded_racemic
mkdir -p $SET_ASIDE
[ -f $SET_ASIDE/consensus_before.csv ] || cp $OUT/final/consensus.csv $SET_ASIDE/consensus_before.csv   # the picks of the racemic runs, kept for the comparison below (never overwritten on a retry)
python3 -m cypmode.validation.stereo_rerun > $L/stereo_supersede.log 2>&1 || exit 1
python3 -m cypmode.validation.batch $IN/structure_track_test_blinded.csv $OUT/structure_track_results.json --protein-msa $MSA --max-parallel-samples 1 > $L/stereo_base_runs.log 2>&1 || exit 1
python3 -m cypmode.validation.final_recipe > $L/stereo_final_recipe.log 2>&1 || exit 1
python3 -m cypmode.validation.stereo_rerun --report $SET_ASIDE/consensus_before.csv > $L/stereo_picks.log 2>&1 || exit 1
python3 -m cypmode.validation.pose_stereo --zip $OUT/final/structures_final.zip --pools > $L/stereo_check.log 2>&1 || exit 1
python3 -m cypmode.validation.official_posebusters $OUT/final/structures_final.zip > $L/stereo_posebusters_final.log 2>&1
