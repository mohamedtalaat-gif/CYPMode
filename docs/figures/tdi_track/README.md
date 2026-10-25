# TDI track: first stages

These figures show the first three stages of the TDI model and the first check of a reliability score, drawn from the result files of the first week of the project. They are kept as the record of how the model started. The final model differs from the one drawn here (a stack of many candidates replaced the single LightGBM, and the population and the evaluation were corrected); its results are in the README of the repository (section "Final stack and verdicts"), in `METHOD_REPORT.md` and in the notebook.
The full methodology and every finding, including the ones that did not work, are in [`CYPMode_Architecture_Specification.md`](../../../CYPMode_Architecture_Specification.md) and in the audit report (`audit/AUDIT_REPORT.md`).

## MCC by pipeline stage

![TDI MCC progression](tdi_mcc_progression-light.svg#gh-light-mode-only)
![TDI MCC progression](tdi_mcc_progression-dark.svg#gh-dark-mode-only)

Three stages, all under the same 5-fold cross-validation (`StratifiedKFold(n_splits=5, random_state=42)`, so every bar is comparable with the others), on the population as it was first built: every labelled row of the training file, with the labels as given.

1. **Baseline**: RDKit descriptors only, the approach of the official tutorial (`data/tdi_track/outputs/comparisons/baseline_fixed_seed.csv`).
2. **M3, no amine flag**: the descriptors, the Type II motif flag of CYPMode and a Morgan (ECFP4) fingerprint, with the threshold at 0.5.
3. **M3 with a tuned threshold**: the same features, with the classification threshold chosen for MCC inside each training fold and never on the validation labels, because TDI is an imbalanced label (about 21% positive) and 0.5 is too conservative for it. Stages 2 and 3 are read from `data/tdi_track/outputs/cyp3a4/threshold_tuning_cyp3a4_no_amine.csv` and `data/tdi_track/outputs/cyp2d6/threshold_tuning_cyp2d6_no_amine.csv`.

A trialkylamine flag (`cypmode.metrics.amines`) was left out of the model. The runs of 2026-09-25 that kept it by mistake (`threshold_tuning.csv` and `ablation_m0_m3.csv` in `data/tdi_track/outputs/comparisons/`) are not drawn; the notebook (Part VII) tables them next to the runs without the flag. A later re-test showed that the data neither support nor refute the flag (specification, Tier 2, correction of 2026-10-16); it stays out because no member of the final stack takes it as an input.

CYP2D6 starts and stays far weaker than CYP3A4 in these stages: its baseline MCC is close to zero. The structural evidence of Tier 3, added next, gave no measurable gain (specification, Tier 3).

Two later findings make these bars an early record and not a measure of the final model. The CYP3A4 population mixed two conventions of labelling (1,249 of 3,584 labelled rows have no direct measurement and are all negative), and random cross-validation does not resemble the scored test set (specification, "Corrected TDI population and selection-corrected evaluation", 2026-10-04).

## Does the reliability score track real error?

![Reliability score correlation](tdi_reliability_correlation-light.svg#gh-light-mode-only)
![Reliability score correlation](tdi_reliability_correlation-dark.svg#gh-dark-mode-only)

Two candidate reliability signals were computed for the validation compounds of the first fold of the fixed-seed protocol (CYP3A4, descriptors only): the disagreement of five LightGBM models trained on the same compounds with different seeds, and the distance of each compound to the training set (1 minus the highest ECFP4 Tanimoto similarity). Each was correlated with whether the prediction at the threshold 0.5 was wrong (`data/tdi_track/outputs/comparisons/reliability_correlation.csv`). Neither is called validated uncertainty quantification: the chemical distance tracks error better than the disagreement, and both correlations are weak to modest, on one fold. A later test of query-by-committee uncertainty against the measured errors gave a negative result (specification, "Uncertainty: query by committee", 2026-10-05).

## Reproducing these figures

```bash
pip install matplotlib  # not a project runtime dependency, figures only
python docs/figures/generate_tdi_figures.py
```

The command regenerates the four SVG files from the result files named above; nothing in them is edited by hand.
