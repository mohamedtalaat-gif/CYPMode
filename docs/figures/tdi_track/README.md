# TDI Track, Results

Structural/mechanistic evidence and classical machine learning combined to
predict time-dependent (mechanism-based) CYP3A4/CYP2D6 inhibition for the
[OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge).
Full methodology and every intermediate finding, including the ones that
didn't work, are in
[`CYPMode_DATA_AUDIT.md`](../../../CYPMode_DATA_AUDIT.md) and
[`CYPMode_Architecture_Specification.md`](../../../CYPMode_Architecture_Specification.md).
This page is the short version: what the numbers are, and what they mean.

## MCC by pipeline stage

![TDI MCC progression](tdi_mcc_progression-light.svg#gh-light-mode-only)
![TDI MCC progression](tdi_mcc_progression-dark.svg#gh-dark-mode-only)

Three real, independently measured stages, same 5-fold cross-validation
protocol throughout (`StratifiedKFold(n_splits=5, random_state=42)`, so
every bar is comparable to the others, no stage benefits from an easier
split):

1. **Baseline**: RDKit descriptors only, reproducing the official
   challenge tutorial's own approach.
2. **M3**: adds CYPMode's Type II motif screen and a Morgan (ECFP4)
   fingerprint. A second candidate feature (a trialkylamine flag,
   motivated by a real 6-drug reference panel) was tested here too and
   **dropped**, it measurably hurt CYP2D6 and did nothing for CYP3A4 in
   this multivariate setting. That negative result is documented in the
   data audit, not hidden.
3. **M3 + tuned threshold**: same model, but the classification
   threshold is optimized for MCC inside each training fold (never
   touching validation labels) instead of left at the default 0.5. TDI is
   a ~21%-positive, imbalanced label, so 0.5 was always too conservative.

CYP2D6 starts and stays far weaker than CYP3A4 throughout, this isn't a
gap this project is hiding: CYP2D6's real baseline MCC is close to zero,
and even the improvements here don't change that fundamental difficulty.
It's the direct motivation for adding structural evidence (Tier 3) next.

## Does the reliability score actually track real error?

![Reliability score correlation](tdi_reliability_correlation-light.svg#gh-light-mode-only)
![Reliability score correlation](tdi_reliability_correlation-dark.svg#gh-dark-mode-only)

Two candidate reliability signals, computed from a held-out fold (5
same-data, different-seed models' prediction variance; each compound's
ECFP4 Tanimoto distance to the training set), correlated against whether
that fold's prediction was actually wrong. **Neither component is called
"validated uncertainty quantification" here**, that label is only used
once a correlation is shown to be real and reasonably strong. As it
stands: chemical distance-from-training (OOD) tracks error better
(r≈0.20) than model-ensemble disagreement (r≈0.10), on a single fold.
Both are weak-to-modest, not strong, reported rather than
oversold, and due for a multi-fold re-check before any claim is built on
them.

## Reproducing these figures

```bash
pip install matplotlib  # not a project runtime dependency, figures only
python docs/figures/generate_tdi_figures.py
```

Regenerates all four SVGs from the checked-in result CSVs in
`data/tdi_track/outputs/`, nothing here is hand-edited.
