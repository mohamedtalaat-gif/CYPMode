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
           TIER 1 (all)           TIER 2 (all)      TIER 3 (fixed N=50)
        Morgan + RDKit          CYPMode motif flag       Boltz-2, CYP3A4
          descriptors           (amine flag dropped     only — top-50 by
                                 — see Decision Log)   OOF uncertainty, no
                                                        motif OR clause
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

**Tier 3 — structural augmentation, CYP3A4 only.** Boltz-2 + explicit heme + covalent axial-cysteine bond (`cypmode.validation.pipeline`), run only on a **fixed budget of 50 compounds** (`cypmode.challenge.tier3_selection.select_fixed_budget_oof`), ranked by out-of-fold classification uncertainty across the whole labeled population. **Not built for CYP2D6**, deliberately: this project has no validated CYP2D6 structural pipeline, and inventing one to force architectural symmetry would be a real research project on its own, not something to improvise under this timeline. CYP2D6's head simply has no structural branch.

**Selection mask — revised from the original "motif OR uncertain" spec.** Checked against the real data first: motif-positive alone covers ~42% of the labeled CYP3A4 population, so the original OR-clause selected 46–72% of ~3,584 compounds regardless of quantile — thousands of compounds against a real observed Boltz-2 cost of 15–30+ minutes each, not tractable. Motif is already a free Tier 2 feature, so spending Boltz-2 on a motif-positive compound the model already "knows about" adds nothing. Revised policy: pure OOF-uncertainty ranking, no motif clause, hard fixed budget (N=50) sized to the real compute budget rather than an open-ended fraction of the dataset. The threshold/ranking model is trained fold-internally (never sees the fold it scores) — see `cypmode/challenge/tier3_selection.py` and its leakage tests.

**OOF Direct-Inhibition Bridge.** A separate regression model (`cypmode.challenge.oof_bridge`), predicting direct-inhibition pIC50 with out-of-fold predictions (trained on the training fold, applied to the held-out fold) to avoid leaking the TDI model's own validation labels through potency information. **Revised from the original Tobit/censored-regression plan**: checked the real data first and found no hard censoring at the pIC50=4 floor (values continue smoothly down to ~1.9, no pile-up) — what's real is heteroscedastic measurement noise, with the reported std column jumping 5–10x below pIC50=4. Tobit assumes the wrong generating process for this data; inverse-variance weighted regression (using the real, measured std per compound) is used instead. Feeds into Feature/Evidence Fusion as a parallel input stream alongside Tier 1/2/3 — not a step applied after them. Real result: CYP3A4 OOF MAE 0.730, CYP2D6 OOF MAE 0.645 (pIC50 spans ~6 log units).

**Shared Trunk.** Treated as a hypothesis to test, not an assumed design. A small, plain-PyTorch shared-trunk MLP (`cypmode.challenge.multitask` — one trunk, one head per isoform, masked BCE loss so a compound missing one isoform's label contributes no gradient to that head) was built instead of installing `openadmet-models`, whose real dependencies (PyTorch, Lightning, ChemProp+CheMeleon, CatBoost, XGBoost, TabPFN) exist only in a separate conda env with no pip-installable `pyproject.toml`.

**Full real comparison, same 5-fold CV protocol, same feature set (descriptors + motif + Morgan) throughout:**

| Model | CYP3A4 MCC | CYP2D6 MCC |
|---|---|---|
| M3, independent heads, 0.5 threshold | 0.300 | 0.043 |
| M3, independent heads, tuned threshold | **0.375** | 0.078 |
| Shared trunk, 0.5 threshold | 0.250 | 0.099 |
| Shared trunk, tuned threshold | 0.266 | **0.113** |

Threshold tuning helps both architectures, but helps the independent-head model more (CYP3A4 +0.075, CYP2D6 +0.035) than the shared trunk (CYP3A4 +0.028, CYP2D6 +0.014) — the gap between the two architectures narrows slightly for CYP3A4 but never closes. **The split is consistent and real, not fold noise from a single run**: independent heads are clearly ahead for CYP3A4 in every variant tested; the shared trunk is clearly ahead for CYP2D6 in every variant tested. Neither architecture dominates the other.

**Decision: per-isoform model choice for the final assembled model, not one global architecture.** CYP3A4 uses the tuned independent-head model (0.375); CYP2D6 uses the tuned shared trunk (0.113). This is a real, evidence-based asymmetry — CYP2D6's weak baseline signal (MCC≈0.05 from the very first baseline) benefits from whatever cross-isoform regularization the shared trunk provides, while CYP3A4 already has enough of its own signal that sharing a trunk with a noisier label dilutes it. Not treated as final until Tier 3's structural features are wired in for CYP3A4, which could change its calculus again.

**Reliability score.** Cross-cutting, computed from ensemble disagreement + Tanimoto-to-training-set distance + Boltz-2's own confidence score where Tier 3 ran. Deliberately kept outside the prediction path. Not called "uncertainty quantification" in any deliverable until its correlation with real held-out prediction error is measured and reported.

**Stratified evaluation by chemical familiarity.** In addition to overall MCC, validation compounds get split by Tanimoto similarity to the training set (familiar vs. OOD-like), to test the specific, falsifiable claim that structural/mechanistic augmentation helps disproportionately on compounds the 2D model is already uncertain about — a stronger scientific result than an aggregate MCC delta, and the basis of the Innovation Award pitch.

## Decision log (what changed, and why)

| Decision | Status |
|---|---|
| Track: TDI, not Direct Inhibition | Settled |
| Tier 2: motif only, amine dropped | Settled — real M0–M3 ablation evidence (CYP2D6 MCC 0.069→0.038 with the amine flag added) |
| Tier 3: CYP3A4 only, no forced CYP2D6 symmetry | Settled |
| OOF bridge: parallel fusion input, not downstream step | Settled this session |
| Shared trunk: hypothesis requiring its own benchmark | Run, with threshold tuning on both sides — real result: independent heads win CYP3A4 in every variant, shared trunk wins CYP2D6 in every variant. Settled as a **per-isoform choice**, not one global architecture (pending Tier 3 for CYP3A4) |
| Reliability score: cross-cutting, empirically validated before being named "uncertainty" | Settled |
| Stratified familiarity evaluation | Added this session |
| OOF bridge: inverse-variance weighted regression, not Tobit | Revised — real data has no hard censoring, just heteroscedastic noise (std jumps 5–10x below pIC50=4) |
| Tier 3 selection: fixed N=50, uncertainty-only, no motif OR clause | Revised — motif-only already covered 42% of the population, made the OR-clause selection intractable (46–72% of ~3,584 compounds) |

## Status: done vs. remaining

**Done (real code, real numbers, committed locally):**
- Phase 0 data audit — `CYPMode-TDI_DATA_AUDIT.md` (leakage check, label provenance, missingness, chemistry, class balance)
- Baseline reproduction, both tutorial-matching and this project's own fixed-seed CV protocol (`cypmode/challenge/tdi_baseline.py`) — CYP3A4 MCC 0.294, CYP2D6 MCC 0.049
- `cypmode/metrics/amines.py` — trialkylamine detector, verified against the 6-drug reference panel, refined to exclude a real false positive (ketoconazole's N-aryl piperazine)
- M0–M3 ablation (`cypmode/challenge/tdi_ablation.py`) — real result: motif kept, amine dropped, Morgan kept. Best single number so far per isoform: CYP3A4 MCC 0.375 (M3 + tuned threshold), CYP2D6 MCC 0.111 (shared trunk, untuned) — no single stage wins both yet, see Decision Log and Shared Trunk section
- TDI reference panel (`cypmode/validation/tdi_reference.py`) — 6 real drugs, quantitative cross-check against this project's own structural coordination claims
- **OOF Direct-Inhibition Bridge** (`cypmode/challenge/oof_bridge.py`) — inverse-variance weighted regression (revised from Tobit, see Decision Log), fold-internal OOF predictions. Real result: CYP3A4 MAE 0.730, CYP2D6 MAE 0.645
- **Tier 3 selection logic** (`cypmode/challenge/tier3_selection.py`) — fixed-budget (N=50) OOF-uncertainty ranking, real 50 CYP3A4 compounds selected and saved (`data/cyp_challenge/tdi_track/tier3_selected_compounds.csv`)
- Structure-track prep + TDI-reference-panel structural runs + Tier 3's own 50-compound structural batch, chained to run automatically in sequence on the one available GPU (`cypmode/validation/batch.py`) — real Boltz-2 predictions for Tier 3 itself are **in progress, not complete** (see below)
- **Threshold optimization** (`cypmode/challenge/threshold_tuning.py`) — fold-internal MCC-optimized threshold (inner K-fold on the training portion only, never touching validation labels), replacing the naive 0.5 default. Real result: CYP3A4 MCC 0.300→0.375, CYP2D6 MCC 0.043→0.078
- **Reliability score** (`cypmode/challenge/reliability.py`) — ensemble disagreement (5 same-data, different-seed models) + Tanimoto OOD distance, checked against real held-out error (`validate_reliability_correlation`). Real result (single fold, n=717): OOD distance r≈0.20, ensemble disagreement r≈0.10 — both weak-to-modest, **not** called "validated uncertainty quantification" per the cross-cutting rule above
- **Shared-trunk multitask model** (`cypmode/challenge/multitask.py`), including its own fold-internal threshold tuning (`cross_validate_shared_trunk_with_threshold_tuning`) — plain PyTorch, built instead of installing `openadmet-models` (see Shared Trunk section above for the real, settled-as-per-isoform result)
- **Submission validator** (`cypmode/challenge/submission.py`) — checks all official format rules (750 rows, exact columns, no dupes/nulls, boolean dtype, correct Molecule_Names) against the real blinded test set

**Not yet built:**
1. Tier 3 features actually joined into a TDI classifier — the 50 compounds are selected and queued for Boltz-2, but no results exist yet (GPU queue: structure-track batch running → TDI-reference batch → Tier 3's 50 compounds, none of the latter two started as of this check), and no code yet merges structural output into the feature matrix once it exists.
2. Stratified familiarity evaluation — not run yet (waits on Tier 3 being wired in).
3. Final assembled model + real 750-row predictions — no code yet that combines Tier 1/2/3 + OOF bridge + the now-settled per-isoform model choice (independent-head for CYP3A4, shared-trunk for CYP2D6) into one final model and produces actual submission predictions. Blocked on Tier 3 (item 1), since that could still change the CYP3A4 choice.
7. Two deliverable notebooks (full project walkthrough + self-contained educational deep-dive) — explicitly deferred until the architecture above is finished.
