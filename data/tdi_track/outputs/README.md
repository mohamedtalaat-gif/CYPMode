# TDI track — outputs

Every real result this project has produced for the TDI (time-dependent
inhibition, classification) track: ablations, significance tests,
feature caches, threshold-tuning results, the nested cross-validation
caches and the final candidate files. Everything here is regenerable
from `data/tdi_track/inputs/` by re-running the module listed next to
it — nothing here is hand-edited.

**Where the final model is:** `final/` (the 5-outer-fold design that
decided the model) and `final_f10/` (the 10-outer-fold replicate), both
written by `python -m cypmode.challenge.final_tdi_build`. The files that
would be uploaded are copied from `final/` into `submissions/tdi_track/`
by `python -m cypmode.challenge.submissions_sync`. Everything else in this
tree is the evidence behind that model or an earlier stage of it.

The first stages of the model are drawn in
[`docs/figures/tdi_track/README.md`](../../../docs/figures/tdi_track/README.md);
the final model and its results are described in `README.md` and in
`CYPMode_Architecture_Specification.md` at the repo root. This file
is the reproducibility index: which file came from which command.

## Organized by what it actually is, not flat

**Reorganized 2026-09-29**: this whole tree used to be one flat
directory mixing CYP3A4-only results, CYP2D6-only results, and files
that genuinely compare both isoforms in one table. Splitting *those*
apart would have destroyed the comparison they exist to show, so the
real split is by content, not a blind per-isoform folder pair:

```
data/tdi_track/outputs/
├── cyp3a4/         # CYP3A4-only real results (Tier 3, threshold tuning, KNN, OOF-bridge, ...)
├── cyp2d6/         # CYP2D6-only real results (same set, its own numbers)
├── comparisons/    # genuinely combined isoform-comparison tables (ablations, significance tests,
│                   #   shared-trunk results, baseline reproduction) -- kept together on purpose
├── caches/         # isoform-agnostic feature caches (ChemBERTa/pharmacophore/quantum, the feature
│                   #   store, physics, embeddings), keyed by SMILES, reused by both isoforms
├── nested/         # nested cross-validation: the plan and the cached out-of-fold predictions of
│                   #   every candidate model (5 outer folds x 3 repeats), one folder per experiment
├── nested_f10/     # the same for the 10-outer-fold replicate (10 outer folds x 3 repeats)
├── final/          # the final build of the 5-fold design: candidate files, score tables,
│                   #   manifest and report
├── final_f10/      # the final build of the 10-fold replicate
└── README.md
```

Every module that writes to this tree now points at the right
subfolder by default — nothing here needs a manual `--out-path`
override to land in the right place.

## `cyp3a4/`

| File | Produced by |
|---|---|
| `tier3_structural_results.{json,csv,parquet}` | `python -m cypmode.validation.batch data/tdi_track/outputs/cyp3a4/tier3_selected_compounds.csv data/tdi_track/outputs/cyp3a4/tier3_structural_results.json --no-affinity` |
| `tier3_selected_compounds.csv` | `python -m cypmode.challenge.tier3_selection` |
| `tier3_significance_raw.csv`, `tier3_significance_tukey.csv` | `python -m cypmode.challenge.tier3_features --isoform CYP3A4` |
| `threshold_tuning_cyp3a4_no_amine.csv`, `knn_no_amine_cyp3a4.csv`, `oof_bridge_feature_tuned_cyp3a4.csv`, `cyp3a4_final_combo.csv` | ad hoc re-verification runs behind the numbers in `CYPMode_Architecture_Specification.md`'s "Final architecture" section (the real m3_no_amine_features fix + the OOF-bridge retraction) |

## `cyp2d6/`

| File | Produced by |
|---|---|
| `tier3_structural_results.{json,csv,parquet}` | `python -m cypmode.validation.batch data/tdi_track/outputs/cyp2d6/tier3_selected_compounds.csv data/tdi_track/outputs/cyp2d6/tier3_structural_results.json --target cyp2d6 --protein-sequence <CYP2D6_SEQUENCE> --no-affinity` (the CYP2D6 due-diligence batch — 50 compounds by OOF uncertainty + thioridazine) |
| `tier3_selected_compounds.csv` | `python -m cypmode.challenge.tier3_selection` (CYP2D6 label) |
| `tier3_significance_raw.csv`, `tier3_significance_tukey.csv` | `python -m cypmode.challenge.tier3_features --isoform CYP2D6` |
| `threshold_tuning_cyp2d6_no_amine.csv`, `knn_no_amine_cyp2d6.csv`, `oof_bridge_feature_tuned_cyp2d6.csv` | same re-verification runs as the CYP3A4 set above, CYP2D6 side |
| `final_assembly_predictions.csv` | `python -m cypmode.challenge.final_assembly_cyp2d6` (torch-only process; `cypmode.challenge.final_assembly` reads this cache rather than importing it directly) |
| `cyp2d6_batch_input.csv` | the real 51-compound input list (thioridazine + the 50 Tier3-equivalent picks) fed to the batch run above |

## `comparisons/`

| File | Produced by |
|---|---|
| `baseline_reproduction.csv`, `baseline_fixed_seed.csv` | `python -m cypmode.challenge.tdi_baseline` |
| `ablation_m0_m3.csv` | `python -m cypmode.challenge.tdi_ablation` |
| `knn_features_ablation.csv` | `python -m cypmode.challenge.knn_features` |
| `aid1851_auxiliary_ablation.csv` | `python -m cypmode.challenge.pubchem_auxiliary` |
| `bindingdb_auxiliary_ablation.csv` | `python -m cypmode.challenge.bindingdb_auxiliary` |
| `combined_features_ablation.csv` | `python -m cypmode.challenge.combined_features` |
| `stacking_significance_raw.csv`, `stacking_significance_tukey.csv` | `python -m cypmode.challenge.stacking_significance` (25-fold repeated CV + Tukey HSD, KNN/BindingDB) |
| `quantum_pharmacophore_ablation.csv`, `quantum_pharmacophore_significance_{raw,tukey}.csv` | `python -m cypmode.challenge.quantum_pharmacophore_significance` (tblite/GFN2-xTB; must run in its own process, see the module's own docstring) |
| `pharmacophore_fingerprint_significance_{raw,tukey}.csv` | `python -m cypmode.challenge.pharmacophore_fingerprint --n-workers N` (pmapper) |
| `chemberta_significance_raw.csv`, `chemberta_significance_tukey.csv` | `python -m cypmode.challenge.chemberta_significance` (LightGBM-only process, run *after* `caches/chemberta_embedding_cache.csv` exists — see both modules' docstrings for why these must never share a process) |
| `shared_trunk_m1.csv`, `shared_trunk_m1_threshold_tuning.csv`, `shared_trunk_m1_chemberta_fusion.csv`, `shared_trunk_knn_no_amine.csv`, `shared_trunk_threshold_tuning_no_amine.csv` | `python -m cypmode.challenge.multitask` (own isolated venv, see `.venv-boltz`-style setup in the architecture doc) |
| `threshold_tuning.csv`, `robust_threshold_tuning.csv`, `robust_threshold_stress_test.csv` | `python -m cypmode.challenge.threshold_tuning` |
| `split_validity.csv` | `python -m cypmode.challenge.split_validity` |
| `reliability_correlation.csv` | `python -m cypmode.challenge.reliability` |
| `oof_direct_inhibition.csv` | `python -m cypmode.challenge.oof_bridge` (both isoforms' OOF pIC50 columns side by side) |

## `caches/`

| File | Produced by |
|---|---|
| `chemberta_embedding_cache.csv` | `python -m cypmode.challenge.chemberta_features` (torch-only process) |
| `pharmacophore_fingerprint_cache.csv` | `python -m cypmode.challenge.pharmacophore_fingerprint --n-workers N` |
| `quantum_pharmacophore_cache.csv` | `python -m cypmode.challenge.quantum_pharmacophore --n-workers N` |

Every `*_significance_*` triple was produced with repeated 5×5 stratified
CV (distinct seeds/folds each repeat, not bootstrap resampling of fixed
predictions). The `*_tukey.csv` files treat the repeated paired design
as independent groups and are **superseded**: the `*_paired.csv` files
next to them re-analyse the saved per-fold scores with the corrected
resampled t and Holm adjustment (`python -m cypmode.challenge.recompute_paired_summaries`,
queue job 905), and those are the numbers quoted. The same correction is
used by every nested-CV experiment below.

## `caches/` (added with the nested engine)

| File | Produced by |
|---|---|
| `feature_store.npz` | `cypmode.challenge.feature_store.FeatureStore` — every feature block for the training and the test compounds (fingerprints, descriptors, alerts, motifs, physics...), one array per block, built once and reused by the nested runners |
| `chemeleon_embeddings.npz` | `python -m cypmode.challenge.frozen_embeddings` (frozen CheMeleon embeddings, job 050) |
| `mmp_pairs.npz` | `python -m cypmode.challenge.mmp_transfer` (matched molecular pairs, job 047) |
| `physics_cache.csv`, `physics_atoms.csv.gz` | `python -m cypmode.challenge.physics_block` (GFN2-xTB tier-A physics block and its atom-level table, jobs 070 and 073) |
| `physics_b_cache.csv` | `python -m cypmode.challenge.physics_tierb --extract` (conformer-averaged tier B, job 073) |
| `qm_reactivity_cache.csv` | `python -m cypmode.challenge.qm_reactivity_features` |

## `nested/` and `nested_f10/`

Each isoform folder (`CYP3A4/`, `CYP2D6/`) holds `plan.npz` (the outer and inner splits; the identity of a plan is a hash of its pool, labels, split sizes and seed) and
`cache/<plan id>/<candidate>/r<repeat>_k<outer fold>_m<inner fold>.npy`, the cached predictions of every candidate model on every task (inner fold `m-1` is the refit on the whole
outer-training set, `m-2` the deployment fit on the whole pool). Written by `python -m cypmode.challenge.nested_runners --isoform <iso> --repeats 3 --outer <5 or 10> --inner 4 --root <folder>`
and by the experiment modules of queue jobs 010–082, which add their own candidates to the same caches. The other folders are variants of the plan:
`*_group0.6` (Butina clusters at Tanimoto similarity 0.6 are kept whole inside one fold), `*_pop` and `*_pop_panel` (the selection-aware population experiments, jobs 010–015).
`deploy_scores.csv` is written by `python -m cypmode.challenge.nested_deploy` and `reliability_deploy.csv` by `python -m cypmode.challenge.committee_uncertainty`. `nested_auc.csv` and `nested_ap.csv`
come from the first nested runs and no module in the repository writes them any more; nothing quotes them, and every number quoted anywhere is recomputed from the caches by the final build and the audit. `nested/superseded_candidates/` holds three early candidate submission files (`lgbm_only`, `lgbm_stack`, `v2`) that the final build replaced; none of them is used.

## `final/` and `final_f10/`

| File | What it is |
|---|---|
| `candidate_A_hits_only.csv`, `candidate_B_selection_aware.csv` | the two candidate submission files (750 rows, the portal's schema, validated again after reloading); CYP3A4 is identical in both, CYP2D6 differs (A fitted on measured compounds, B on the panel pool that includes never-measured compounds as negatives) |
| `CYP3A4_hits_only_scores.csv`, `CYP2D6_hits_only_scores.csv`, `CYP2D6_selection_aware_scores.csv` | the stack's score for every test compound, the call, an isotonic-calibrated probability and the committee's disagreement |
| `CYP3A4_stack_choice.csv`, `CYP2D6_stack_choice.csv` | the nested outer-fold AUC and MCC of the stack that was chosen, at both confidence cuts |
| `*_call_rate_choice.csv` | the call rate (a multiple of the prevalence) and the mean outer-fold MCC that chose it |
| `manifest.json`, `final_report.md` | file hashes, environment, seeds, gate verdicts, call rates, the leakage audit, and the same as a readable report |

Written by `python -m cypmode.challenge.final_tdi_build` (and `--design 10 --out-dir data/tdi_track/outputs/final_f10` for the replicate), queue job 090.
