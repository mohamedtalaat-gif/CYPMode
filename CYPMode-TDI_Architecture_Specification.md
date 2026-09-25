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

**Split-validity diagnostic** (`cypmode.challenge.split_validity`). A CV split can look far more reliable than it is if the validation compounds are chemically close to training (near-duplicates/analogs) rather than a genuine test of generalization — a known, general risk in cheminformatics benchmarking, not something specific to this challenge. Checked on our own real data: **our own i.i.d. CV folds are, if anything, chemically less similar to their own training portion (median similarity 0.448, 38.3% ≥0.5) than the real 750-compound blinded TDI test set is to our full training set (median 0.569, 81.5% ≥0.5)**. A mildly reassuring sign (not a guarantee) that our CV-measured MCC is unlikely to collapse on the blind set the way it could if the blind set were instead the more chemically-novel side of that comparison. **Since confirmed with a real mechanistic explanation, not just an observed number**: the challenge's own public documentation (`config.py`'s `ABOUT_MD` on the official HF Space) states the blind test set was built by "hit expansion... purchasing the top 10 chemisimilars (ECFP4 Tanimoto similarity) per hit compound" from the training population's own top hits (CYP1A2/CYP2C9/CYP3A4 hits specifically). Our similarity finding isn't a coincidence — it's a direct, documented consequence of how the test set was constructed. Reported honestly as a diagnostic with a now-confirmed mechanism, still not proof of transfer.

**Official scoring harness — verified as far as publicly possible.** The actual metric-computation code runs in a private AWS Lambda (`backend/lambda_handler.py`, referenced but not included in the public HF Space repo) — not independently auditable. What the public repo *does* confirm directly (`config.py`, `submission.py`): the TDI track's primary metric is plain Matthews Correlation Coefficient (no custom variant), macro-averaged (unweighted mean) across exactly `CYP2D6_is_TDI` and `CYP3A4_is_TDI`; the required submission schema (`SMILES`, `Molecule_Name`, `CYP2D6_is_TDI`, `CYP3A4_is_TDI`, 750 rows, no NaN, boolean values) matches `cypmode.challenge.submission`'s validator exactly; and the positive/negative/inferred-positive/assigned-negative label rule (direct-inhibition pIC50 vs. 4, TDI-arm shift vs. log10(2)=0.301) matches `cypmode.validation.tdi_reference.TDI_SHIFT_LOG2_THRESHOLD` exactly. One previously-unknown, real detail: **not every compound's prediction is actually scored** — "only compounds whose label can be assigned with confidence contribute to the score," per the official FAQ. Also newly confirmed: CYP2D6 uses a *different, label-free mass-spectrometry assay* (not the fluorescence-based kit used for the other three isoforms) specifically because "fluorogenic 2D6 probes exhibited poor signal-to-noise ratios" — a real, assay-level reason (not just a chemistry/modeling one) that CYP2D6 is the harder isoform for every team on the leaderboard, not just us.

**Base-rate-shift-robust thresholding** (`cypmode.challenge.threshold_tuning.find_robust_threshold` / `cross_validate_with_robust_threshold_tuning`). A single MCC-optimal threshold implicitly assumes the test base rate matches training's; a real base-rate mismatch can silently cost MCC — a known general risk for any threshold picked by optimizing on one label distribution and applying it to another. Implemented as a minimax choice over an assumed base-rate grid (0.05–0.50), validated with a real base-rate-shift stress test (resampling real validation-fold labels/probabilities to different rates, not simulated). **Real, honest result: the effect is real but asymmetric, not a clean win.** At the natural (unshifted) rate, robust ≈ naive (CYP3A4 0.372 vs 0.375, CYP2D6 0.080 vs 0.078 — negligible either way). Under a shift toward a *rarer* positive class than training (π below ~0.20), the robust threshold clearly helps (e.g. CYP3A4 at π=0.05: 0.232 vs 0.199 naive). Under a shift toward a *more common* positive class (π above ~0.30), it clearly hurts, sometimes substantially (CYP3A4 at π=0.45: 0.415 robust vs 0.481 naive) — the minimax criterion buys protection in one direction at real cost in the other. A "mean" criterion (average MCC across the same grid, instead of worst-case) was also tested and turned out to pick the *exact same threshold* as the naive MCC-optimal one in both isoforms — a wash, not an improvement. **Decision: not adopted as the default threshold** — we have no real evidence the blind test's TDI base rate skews lower than training's (if anything, the split-validity diagnostic above points the other way), so the minimax threshold's asymmetric cost isn't clearly justified. Kept as real, tested, documented infrastructure, usable if external evidence about the blind test's base rate ever surfaces.

**KNN-label (read-across) features** (`cypmode.challenge.knn_features`). For each compound, the known TDI status of its K nearest chemical neighbors (Tanimoto similarity on ECFP4), aggregated into 4 features (fraction positive, similarity-weighted fraction positive, mean similarity to those neighbors, neighbor count) and added to M3. Built fold-internally, not precomputed globally: a training compound's neighbor pool excludes itself (leave-one-out) and is restricted to the rest of that fold's training portion, and a validation compound's neighbor pool is the full training portion only — never the compound itself, never anything from its own validation fold — see the module docstring's leakage-guard tests for why a global precompute would leak. **Real result, same 5-fold CV protocol as M3: CYP3A4 MCC 0.300→0.312, CYP2D6 MCC 0.043→0.075 (+74% relative).** A genuine, positive result on both isoforms, largest relative gain on CYP2D6 — our real bottleneck. Candidate for folding into the assembled feature set alongside M3.

**Auxiliary direct-CYP-activity features, external data** (`cypmode.challenge.pubchem_auxiliary`, `cypmode.challenge.bindingdb_auxiliary`). One LightGBM classifier per isoform, trained on a broader, external compound population and applied to this project's own TDI compounds as an extra feature — no fold-internal OOF treatment needed (both sources are fully external to this project's own TDI-labeled data; see each module's own docstring for the reasoning, which differs from why the OOF Direct-Inhibition Bridge above needs OOF construction).

- **PubChem AID 1851** (NCATS qHTS cytochrome panel, 16,560 real compounds after excluding overlap with our own data). **Real, corrected result: CYP3A4 MCC 0.300→0.308, CYP2D6 MCC 0.043→0.071 (+64% relative).** A first version of this result showed CYP2D6 *dropping* to 0.020 — traced to a real data-handling bug, not a real finding: the raw PubChem file has replicate measurement rows per (compound, isozyme) pair (up to 7 for one pair), and merging without collapsing them first multiplied 16,560 real compounds into 86,721 rows, silently over-weighting whichever compounds happened to have replicates. Fixed with `drop_duplicates` before merging, not after; the corrected number is a genuine, positive result on both isoforms, not the apparent regression the bug produced. Kept in the record as a reminder that a surprising ablation result needs its data pipeline checked before it's trusted, the same discipline already applied to every leakage guard in this project.
- **BindingDB** (literature-curated Ki/IC50/EC50, targeted `getLigandsByUniprots` pulls for CYP3A4/P08684 and CYP2D6/P10635 specifically — confirmed ~3MB/~2MB each, not BindingDB's own 240MB–3GB full dumps). Combined via median affinity across repeated measurements for the same compound (a real quirk here too, but a different one from AID 1851's: independent measurements from different papers, not same-assay replicate rows, so median-combining rather than deduplicating is the right call). **Real result: CYP3A4 MCC 0.300→0.309, CYP2D6 MCC 0.043→0.089 (+107% relative) — the single largest individual CYP2D6 gain found so far**, ahead of KNN-label features (+74%) and AID 1851 (+64%). A real crash was hit and fixed along the way: a small fraction of BindingDB's own SMILES strings (7 of 8,679 for CYP3A4) carry a ChemAxon extended-annotation suffix plain RDKit can't parse — dropped, not passed through, after confirming they're a real but small fraction of the data (<0.1%).

All three auxiliary/read-across additions (KNN, AID 1851, BindingDB) were tested as independent, separate additions to plain M3 at the naive 0.5 threshold — **not yet combined with each other or with the already-validated threshold tuning**. Whether their gains stack, partially overlap, or interact remains real, unmeasured work, not something safe to assume either way.

**Structure standardization for cross-dataset matching** (`cypmode.challenge.structure_standardization`, via `chembl_structure_pipeline` — a real, rdkit-only-dependency package). Both auxiliary modules' exact-overlap leakage checks were matching on plain canonical SMILES, which treats a salt form as a different compound from its free base (confirmed directly: ketoconazole vs. ketoconazole.Cl canonicalize to two different strings) — a real gap that could silently miss a genuine duplicate between an external dataset and this project's own compounds. Both now match on the standardized parent structure instead.

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
| Base-rate-robust threshold: adopted as default? | **Not adopted** — real stress-test evidence shows an asymmetric tradeoff (helps if blind test skews rarer-positive, hurts if it skews more-common-positive), and no evidence points toward the direction that would justify it |

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
- **Split-validity diagnostic** (`cypmode/challenge/split_validity.py`) — Tanimoto-based train/test similarity check, applied to our own real data (see Split-validity diagnostic section above)
- **Base-rate-shift-robust thresholding** (`cypmode/challenge/threshold_tuning.py`'s `find_robust_threshold` + real stress-test validation) — built, tested, and honestly evaluated; **not adopted as the default** (see section above and Decision Log)

**Not yet built:**
1. Tier 3 features actually joined into a TDI classifier — the 50 compounds are selected and queued for Boltz-2, but no results exist yet (GPU queue: structure-track batch complete → TDI-reference batch in progress → Tier 3's 50 compounds not yet started as of this check), and no code yet merges structural output into the feature matrix once it exists.
2. Stratified familiarity evaluation — not run yet (waits on Tier 3 being wired in).
3. Final assembled model + real 750-row predictions — no code yet that combines Tier 1/2/3 + OOF bridge + the now-settled per-isoform model choice (independent-head for CYP3A4, shared-trunk for CYP2D6) into one final model and produces actual submission predictions. Blocked on Tier 3 (item 1), since that could still change the CYP3A4 choice.
4. KNN-label features and PubChem AID 1851 auxiliary data — real, concrete ideas, deliberately deferred: higher implementation effort/risk (KNN-label features are a known real leakage trap if the query compound isn't excluded from its own neighbor set) than items 1–3, and not worth pulling focus from Tier 3 before it's done.
5. Two deliverable notebooks (full project walkthrough + self-contained educational deep-dive) — explicitly deferred until the architecture above is finished.
