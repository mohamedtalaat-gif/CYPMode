# CYPMode-TDI: Architecture Specification (v1)

The single reference for the TDI-track architecture, superseding the
various diagrams sketched earlier in planning discussion. See
`CYPMode-TDI_DATA_AUDIT.md` for the evidence each decision below is based
on — this document states the conclusions, that one has the numbers.

## Final architecture

```
                                    SMILES
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
           TIER 1 (all)           TIER 2 (all)         TIER 3 (subset)
        Morgan + RDKit          CYPMode motif flag       Boltz-2, CYP3A4
          descriptors           (amine flag dropped        only — heme
                                 — see Decision Log)      coordination call
                 │                     │                     │
                 └─────────────────────┼─────────────────────┘
                                       │
                                       ▼
                          Feature / Evidence Fusion
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    │      OOF Direct-Inhibition           │
                    │      Bridge (parallel input,          │
                    │      not a downstream step —          │
                    │      see Decision Log)                │
                    │                                       │
                    └──────────────────┬──────────────────┘
                                       ▼
                              Shared Trunk
                     (hypothesis, not assumption —
                      benchmarked against independent
                      per-isoform heads, see Decision Log)
                                       │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
               CYP3A4 Head                            CYP2D6 Head
             (Tier 1+2+3 + bridge)                (Tier 1+2 + bridge only
                                                    — no structural data
                                                    for this isoform)
                     │                                     │
                     ▼                                     ▼
                  P(TDI)                                P(TDI)
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
                          CV-fold-internal threshold
                             tuning, optimizing MCC


─────────────────────── cross-cutting, not in the prediction path ───────────────────────

        2D ensemble disagreement + Tanimoto-to-train OOD distance
                    + Boltz-2 structural confidence
                                  │
                                  ▼
                         Reliability score
                                  │
                                  ▼
        Empirically test correlation with real held-out error
      before calling this "uncertainty quantification" anywhere
```

## Component-by-component

**Tier 1 — universal chemical representation.** Morgan fingerprints (ECFP4, radius 2, 2048 bits) + RDKit descriptors. Computed for every compound, train and test. No missingness.

**Tier 2 — CYPMode mechanistic evidence.** Motif flag only (`cypmode.metrics.motifs`). The trialkylamine flag (`cypmode.metrics.amines`) was tested and dropped — it actively hurt CYP2D6 MCC and did nothing for CYP3A4 in a real multivariate model (M0–M3 ablation). The underlying chemistry (verapamil/diltiazem/troleandomycin's real trialkylamine-mediated TDI) is still real; the specific feature engineering (a standalone binary flag) is what failed at scale.

**Tier 3 — structural augmentation, CYP3A4 only.** Boltz-2 + explicit heme + covalent axial-cysteine bond (`cypmode.validation.pipeline`), run only on a subset selected per CV fold — motif-positive compounds, or compounds where the Tier 1+2 model is uncertain. **Not built for CYP2D6**, deliberately: this project has no validated CYP2D6 structural pipeline, and inventing one to force architectural symmetry would be a real research project on its own, not something to improvise under this timeline. CYP2D6's head simply has no structural branch.

**Selection mask, fold-local.** The uncertainty ranking that decides which compounds get Tier 3 is a model trained on the training fold only, and its decision threshold is frozen from that same training fold. Applied to the validation fold at inference time only — validation labels never influence which compounds get selected, ranking thresholds, or any other Tier 2/3 decision inside a fold.

**OOF Direct-Inhibition Bridge.** A separate regression model, predicting direct-inhibition pIC50 with out-of-fold predictions (trained on the training fold, applied to the held-out fold) to avoid leaking the TDI model's own validation labels through potency information. Uses a Tobit/censored-regression loss (borrowed from `moal`, github.com/OpenADMET/moal) rather than plain regression, because a real fraction of direct-inhibition pIC50 values are left-censored at the assay floor (pIC50 = 4), not exact measurements. Feeds into Feature/Evidence Fusion as a parallel input stream alongside Tier 1/2/3 — not a step applied after them.

**Shared Trunk.** Treated as a hypothesis to test, not an assumed design: benchmarked against fully independent per-isoform models under the same CV protocol once a real multitask implementation exists (see Remaining Work — the M0–M3 ablation used two separate LightGBM models, not a shared trunk, so this comparison hasn't actually been run yet).

**Reliability score.** Cross-cutting, computed from ensemble disagreement + Tanimoto-to-training-set distance + Boltz-2's own confidence score where Tier 3 ran. Deliberately kept outside the prediction path. Not called "uncertainty quantification" in any deliverable until its correlation with real held-out prediction error is measured and reported.

**Stratified evaluation by chemical familiarity.** In addition to overall MCC, validation compounds get split by Tanimoto similarity to the training set (familiar vs. OOD-like), to test the specific, falsifiable claim that structural/mechanistic augmentation helps disproportionately on compounds the 2D model is already uncertain about — a stronger scientific result than an aggregate MCC delta, and the basis of the Innovation Award pitch.

## Decision log (what changed, and why)

| Decision | Status |
|---|---|
| Track: TDI, not Direct Inhibition | Settled |
| Tier 2: motif only, amine dropped | Settled — real M0–M3 ablation evidence (CYP2D6 MCC 0.069→0.038 with the amine flag added) |
| Tier 3: CYP3A4 only, no forced CYP2D6 symmetry | Settled |
| OOF bridge: parallel fusion input, not downstream step | Settled this session |
| Shared trunk: hypothesis requiring its own benchmark | Settled this session — not yet run |
| Reliability score: cross-cutting, empirically validated before being named "uncertainty" | Settled |
| Stratified familiarity evaluation | Added this session |

## Status: done vs. remaining

**Done (real code, real numbers, committed to the repo):**
- Phase 0 data audit — `CYPMode-TDI_DATA_AUDIT.md` (leakage check, label provenance, missingness, chemistry, class balance)
- Baseline reproduction, both tutorial-matching and this project's own fixed-seed CV protocol (`cypmode/challenge/tdi_baseline.py`) — CYP3A4 MCC 0.294, CYP2D6 MCC 0.049
- `cypmode/metrics/amines.py` — trialkylamine detector, verified against the 6-drug reference panel, refined to exclude a real false positive (ketoconazole's N-aryl piperazine)
- M0–M3 ablation (`cypmode/challenge/tdi_ablation.py`) — real result: motif kept, amine dropped, Morgan kept. Current best (M3): CYP3A4 MCC 0.300, CYP2D6 MCC 0.043
- TDI reference panel (`cypmode/validation/tdi_reference.py`) — 6 real drugs, quantitative cross-check against this project's own structural coordination claims
- Structure-track prep running in parallel (`cypmode/validation/batch.py`) — separate from the TDI track itself, uses the same Boltz-2 pipeline

**Not yet built:**
1. OOF Direct-Inhibition Bridge — no code yet; needs the Tobit-loss regression model and the fold-internal OOF prediction logic.
2. Tier 3 wired into an actual TDI classifier — the Boltz-2 pipeline exists (for the structure track), but the fold-local selection-mask logic and feeding structural features into a TDI model haven't been built.
3. Reliability score — not built; no empirical correlation-with-error check has been run yet.
4. Real multitask shared trunk via `openadmet-models` — not installed or integrated. Current results are from two independent single-isoform LightGBM models, not a shared architecture.
5. Threshold optimization — current MCC numbers are all at the naive 0.5 threshold; fold-internal MCC-optimized thresholds haven't been computed.
6. Stratified familiarity evaluation — not run yet (waits on Tier 3 being wired in).
7. Final submission pipeline — no 750-row prediction/formatting/validation code yet.
