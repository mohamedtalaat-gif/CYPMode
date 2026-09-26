# TDI track — outputs

Every real result this project has produced for the TDI (time-dependent
inhibition, classification) track: ablations, significance tests,
feature caches, and threshold-tuning results. Everything here is
regenerable from `data/tdi_track/inputs/` by re-running the module
listed next to it — nothing here is hand-edited.

For the short, human-readable version of what these numbers mean, see
[`docs/figures/tdi_track/README.md`](../../../docs/figures/tdi_track/README.md)
and `CYPMode_Architecture_Specification.md` at the repo root. This file
is the reproducibility index: which file came from which command.

| File | Produced by |
|---|---|
| `baseline_reproduction.csv`, `baseline_fixed_seed.csv` | `python -m cypmode.challenge.tdi_baseline` |
| `ablation_m0_m3.csv` | `python -m cypmode.challenge.tdi_ablation` |
| `knn_features_ablation.csv` | `python -m cypmode.challenge.knn_features` |
| `aid1851_auxiliary_ablation.csv` | `python -m cypmode.challenge.pubchem_auxiliary` |
| `bindingdb_auxiliary_ablation.csv` | `python -m cypmode.challenge.bindingdb_auxiliary` |
| `combined_features_ablation.csv` | `python -m cypmode.challenge.combined_features` |
| `stacking_significance_raw.csv`, `stacking_significance_tukey.csv` | `python -m cypmode.challenge.stacking_significance` (25-fold repeated CV + Tukey HSD, KNN/BindingDB) |
| `quantum_pharmacophore_cache.csv`, `quantum_pharmacophore_ablation.csv`, `quantum_pharmacophore_significance_{raw,tukey}.csv` | `python -m cypmode.challenge.quantum_pharmacophore --n-workers N` (tblite/GFN2-xTB; must run in its own process, see the module's own docstring) |
| `pharmacophore_fingerprint_cache.csv`, `pharmacophore_fingerprint_significance_{raw,tukey}.csv` | `python -m cypmode.challenge.pharmacophore_fingerprint --n-workers N` (pmapper) |
| `chemberta_embedding_cache.csv` | `python -m cypmode.challenge.chemberta_features` (torch-only process) |
| `chemberta_significance_raw.csv`, `chemberta_significance_tukey.csv` | `python -m cypmode.challenge.chemberta_significance` (LightGBM-only process, run *after* the line above — see both modules' docstrings for why these must never share a process) |
| `shared_trunk_m1.csv`, `shared_trunk_m1_threshold_tuning.csv`, `shared_trunk_m1_chemberta_fusion.csv` | `python -m cypmode.challenge.multitask` (own isolated venv, see `.venv-boltz`-style setup in the architecture doc) |
| `threshold_tuning.csv`, `robust_threshold_tuning.csv`, `robust_threshold_stress_test.csv` | `python -m cypmode.challenge.threshold_tuning` |
| `split_validity.csv` | `python -m cypmode.challenge.split_validity` |
| `reliability_correlation.csv` | `python -m cypmode.challenge.reliability` |
| `oof_direct_inhibition.csv` | `python -m cypmode.challenge.oof_bridge` |
| `tier3_selected_compounds.csv` | `python -m cypmode.challenge.tier3_selection` |
| `tier3_structural_results.{json,csv,parquet}` | `python -m cypmode.validation.batch data/tdi_track/outputs/tier3_selected_compounds.csv data/tdi_track/outputs/tier3_structural_results.json --no-affinity` (real Boltz-2 runs against `data/structure_track/inputs/`-style YAMLs, generated fresh per compound; the raw predicted structures themselves are not kept here, see the Structure track's own outputs/README.md) |

Every `*_significance_*` pair follows the same real methodology:
repeated 5×5 stratified CV (distinct seeds/folds each repeat, not
bootstrap resampling of fixed predictions) + Tukey HSD, the same
protocol independently validated by the OpenADMET-ExpansionRx
retrospective (see `CYPMode_Architecture_Specification.md`).
