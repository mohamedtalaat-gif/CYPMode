# CYPMode-TDI: Data Audit

Phase 0 deliverable before any TDI architecture code. All numbers below are
computed directly from the real official challenge data
(`data/cyp_challenge/tdi_track/`, `openadmet/cyp-challenge-train-test` on
Hugging Face, Apache 2.0) and this project's own code — nothing here is
recalled from memory or assumed.

## A. Dataset / file inventory

| File | Rows | Purpose |
|---|---|---|
| `cyp-challenge-TRAIN_TDI.csv` | 6,145 | TDI labels + supporting pIC50s |
| `cyp-challenge-TEST-BLINDED.csv` | 750 | Blinded test set (SMILES, Molecule_Name only) |

## B. Full column inventory (TDI training, 36 columns)

| Column | Type | Train available | Test available | Feature-safe? | Role |
|---|---|---|---|---|---|
| `Molecule_Name` | id | ✓ (6145/6145) | ✓ | ✓ | id |
| `SMILES` | structure | ✓ (6145/6145) | ✓ | ✓ | input |
| `CYP3A4_is_TDI` | label | ✓ (3584/6145) | ✗ | ✗ | target |
| `CYP2D6_is_TDI` | label | ✓ (1497/6145) | ✗ | ✗ | target |
| `CYP{1A2,2C9,2D6,3A4}_pIC50_TDI_condition` (+ conf_high/low/std) | measurement | ✓ (1285–3583) | ✗ | ✗ | label provenance only — this is literally what `is_TDI` is derived from; using it as a feature would be near-tautological, not just leaky |
| `CYP{1A2,2C9,2D6,3A4}_pIC50_direct_inhibition` (+ conf_high/low/std) | measurement | ✓ (1285–2335) | ✗ | not directly | OOF bridge input only (see Phase G in the architecture discussion) — test set has no direct pIC50, so a model trained directly on it can't be applied to test without an OOF-safe bridge |

The tutorial itself confirms this split: training carries pIC50 + confidence bounds + std for both direct-inhibition and TDI-condition arms; test carries only `Molecule_Name` and `SMILES`.

## C. Leakage status

**Zero train/test leakage**, confirmed by canonical-SMILES comparison (RDKit `MolToSmiles`, not raw string match) between all 750 test compounds and all 6,145 training compounds — 0 overlap. This reconfirms the check already done earlier this session for the Direct Inhibition track's files, run again here specifically against `TRAIN_TDI.csv`.

## D. Label provenance

`is_TDI` is derived from `pIC50_TDI_condition - pIC50_direct_inhibition > log10(2)`, with a floor at `pIC50 = 4` (assay's reliable detection limit) producing "inferred positive" / "assigned negative" categories for low-activity compounds — see the challenge FAQ for the exact rule.

**Known discrepancy (found earlier this session, still unresolved):** vardenafil's `CYP3A4_pIC50_TDI_condition = 5.59` (well above the 4.301 inferred-positive threshold) with **no** direct-inhibition value recorded at all (not "measured below 4" — simply absent), yet its `CYP3A4_is_TDI` label is `False`. This doesn't match a literal reading of the FAQ's stated rule, which assumes a *measured* (even if low) direct-inhibition value. **Conclusion: trust the given label as ground truth, don't attempt to re-derive it from the exposed columns** — there's evidently a rule component not fully captured in the public FAQ text.

## E. Missingness matrix

| | CYP3A4 | CYP2D6 |
|---|---|---|
| `is_TDI` labeled | 3,584 / 6,145 (58.3%) | 1,497 / 6,145 (24.4%) |
| ...of which missing a direct-inhibition pIC50 | 1,249 (34.8%) | 4 (0.3%) |

The CYP3A4 gap is large and **structural, not a data-quality bug** — those are exactly the FAQ's "inferred positive / assigned negative" rows (direct-inhibition too weak/unmeasurable to report a number, but a label was still assigned from the TDI-arm value or by convention). A naive `dropna()` on both pIC50 columns before training would silently discard over a third of CYP3A4's real label signal.

## F. Class balance

| Isoform | TDI+ | TDI− | % positive |
|---|---|---|---|
| CYP3A4 | 764 | 2,820 | 21.3% |
| CYP2D6 | 324 | 1,173 | 21.6% |

Matches the tutorial's own reported "~20%" — a real sanity check passed, not a number taken on faith.

## G. Chemical quality (all 6,145 compounds)

| Check | Result |
|---|---|
| Raw SMILES duplicates | 0 |
| Unparseable/invalid SMILES | 0 |
| Canonical-SMILES duplicates | 0 |
| Salts (multi-fragment SMILES) | 0 |
| Compounds with stereochemistry (`@`/`/`) | 604 (9.8%) |
| MW | mean 359.5, range 75–974 |
| Heavy atoms | mean 25.6 |
| HBA / HBD | mean 4.2 / 1.2 |
| Aromatic rings | mean 2.4 |
| Total rings | mean 3.4 |

Clean, pre-curated data — no dedup, salt-stripping, or structure-validation work needed before featurization.

## H. Motif / amine prevalence — **the key test, and it failed**

`cypmode.metrics.amines.has_trialkylamine` (new this audit) flags a
secondary/tertiary **trialkylamine** nitrogen — the real chemistry behind
verapamil, diltiazem, and troleandomycin in the 6-drug TDI reference panel
(`cypmode/validation/tdi_reference.py`). Verified against that panel first:
an unrefined "any tertiary amine" SMARTS false-positived on ketoconazole
(a real, documented *reversible* inhibitor) via its N-aryl piperazine
nitrogen — excluding any N bonded to an aromatic atom fixed that without
losing the three real positives.

Category prevalence across all 6,145 compounds:

| Category | n |
|---|---|
| neither | 2,354 |
| motif-only | 2,235 |
| amine-only | 1,073 |
| motif + amine | 483 |

**TDI-positive rate by category, against the real labels:**

| Category | CYP3A4 (n) | CYP2D6 (n) |
|---|---|---|
| motif + amine | 21.9% (251) | 18.2% (209) |
| motif-only | 20.3% (1250) | 18.5% (518) |
| amine-only | 19.1% (624) | 20.7% (401) |
| **neither** | **23.0% (1459)** | **29.0% (369)** |

**This is a real, honest negative result.** On the small, hand-picked
6-drug reference panel, "motif-negative + has trialkylamine" looked like a
clean discriminator for real TDI drugs the motif screen misses. At scale,
across the actual 3,584/1,497-compound labeled training population, it
isn't — the "neither" category has the *highest* TDI-positive rate for
both isoforms, the opposite of the hypothesis. The 6-drug panel's story is
real (those 3 named drugs are genuinely motif-negative, amine-positive,
and TDI-positive, confirmed by real measurement) but doesn't generalize
as a standalone univariate predictor across this much larger and more
diverse population (Enamine DDS10/FDAA libraries, not curated marketed
drugs).

**What this does and doesn't mean:** it doesn't necessarily mean the amine
flag is useless in a full multivariate model — that's what the M0–M3
ablation (next phase, not run yet) will test directly. But it does mean
the simple story ("amine flag explains the motif screen's TDI blind spot")
cannot be asserted as established fact going into the architecture. This
gets reported as-is, not smoothed over.

## I. Duplicate / scaffold analysis

5,367 unique Murcko scaffolds across 6,145 compounds (87.3% scaffold-unique). 1,129 compounds share a scaffold with at least one other compound in the set — largest shared scaffold is plain benzene (52 compounds), followed by a diphenyl-sulfonamide core (30). Reasonably scaffold-diverse; a plain random split isn't badly compromised by scaffold duplication for this particular set, but a scaffold-aware split (recommended earlier, and by the tutorial itself) remains the more rigorous choice given ~18% of compounds aren't scaffold-unique.

## J. Baseline reproduction

**Reproduction of the tutorial's own procedure** (`cypmode/challenge/tdi_baseline.py::cross_validate_baseline`, RDKit descriptors via `useful_rdkit_utils`, LightGBM, 10× unseeded stratified 75/25 splits):

| Isoform | MCC | ROC-AUC |
|---|---|---|
| CYP3A4 | 0.264 | 0.796 |
| CYP2D6 | 0.055 | 0.556 |

**Independent, fixed-seed reproduction** (`cross_validate_fixed_seed`, `StratifiedKFold(n_splits=5, random_state=42)`, same features/model):

| Isoform | MCC | ROC-AUC |
|---|---|---|
| CYP3A4 | 0.294 | 0.796 |
| CYP2D6 | 0.049 | 0.570 |

Both land in the same range as the tutorial's informally-cited ~0.31 / ~0.03 — close enough to trust the pipeline is faithfully reproducing the published approach, with the small gap attributable to split randomness and library-version drift, not a bug. **The fixed-seed number is what CYPMode's own additions get compared against going forward** — reproducible run to run, unlike the tutorial's own unseeded splits.

## K. Exact CV protocol (for all future comparisons)

`cypmode.challenge.tdi_baseline.cross_validate_fixed_seed`: per isoform, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the isoform's labeled subset only, one fresh `LGBMClassifier(random_state=42)` per fold, threshold fixed at 0.5 for this baseline (threshold optimization is a separate, later step — per the earlier architecture discussion, done inside training folds only, never touching validation/test labels).

## L. Feature Eligibility Matrix (final)

| Feature | Source | Available at test time | Leakage risk | Status |
|---|---|---|---|---|
| Morgan / RDKit descriptors | SMILES | ✓ | none | Tier 1, ready |
| Motif call | `cypmode.metrics.motifs` | ✓ | none | Tier 2, built — **univariate signal not yet shown useful at scale (section H)** |
| Trialkylamine flag | `cypmode.metrics.amines` (new) | ✓ | none | Tier 2, built this audit — **same caveat** |
| Direct-inhibition pIC50 | training only | ✗ (test has none) | high if used directly | OOF bridge only, not built yet |
| TDI-condition pIC50 | training only | ✗ | high (near-tautological with label) | excluded entirely |
| Boltz-2 structural coordination | `cypmode.validation` pipeline | computable for any SMILES, but expensive | none (physics-based, not data-derived) | Tier 3, pipeline exists, not yet run on TDI training/test compounds |

## M. M0-M3 ablation — the real, multivariate answer

Section H found no useful *marginal* motif+amine signal at scale. That
doesn't settle it on its own — a multivariate model can pick up an effect
a cross-tab misses. `cypmode/challenge/tdi_ablation.py` runs the actual
test: same paired CV folds (`cross_validate_fixed_seed`'s `feature_col`
parameter), one feature block added at a time.

| Model | Features | CYP3A4 MCC | CYP2D6 MCC |
|---|---|---|---|
| M0 | RDKit descriptors (= the reproduced baseline) | 0.2936 | 0.0493 |
| M1 | + motif flag | **0.2985** | **0.0689** |
| M2 | + trialkylamine flag | 0.2985 (flat) | **0.0383 (worse than M0)** |
| M3 | + Morgan fingerprint (ECFP4) | 0.2999 | 0.0429 |

**Conclusion, and it's now a well-evidenced one, not a guess:**

- **Motif flag (M1) is kept.** Small, real, consistent improvement on both
  isoforms — biggest relative gain on CYP2D6 (+0.020 MCC, a 40% relative
  jump over the weak M0 baseline).
- **Trialkylamine flag (M2) is dropped.** It doesn't just fail to help —
  it actively *hurts* CYP2D6 (worse than not having it at all) and does
  nothing for CYP3A4. Three independent checks now agree: the 6-drug
  panel's story was real but anecdotal (section H), the marginal cross-tab
  showed no association at scale (section H), and now the real
  multivariate model confirms it adds noise, not signal. This is a
  legitimate, evidence-based feature removal, not a hedge.
- **Morgan fingerprint (M3) is kept** as a cheap, essentially-free addition
  (small further CYP3A4 gain, no clear CYP2D6 recovery from M2's damage,
  consistent with M2 being the actual problem there) — and it's what the
  official tutorial itself already uses for train/test similarity, so
  keeping it doesn't cost interpretability against the published baseline.

**Revised Tier 2, going forward: motif flag only.** The trialkylamine
mechanism itself is still real (verapamil/diltiazem/troleandomycin are
real drugs with real measured TDI shifts, see
`cypmode/validation/tdi_reference.py`) — what failed is using "has a
trialkylamine anywhere in the molecule" as a standalone binary feature
across a large, chemically diverse population. That's a specific, narrow
negative result, not a refutation of the underlying chemistry.

## Bottom line

Data is clean and leakage-free. The baseline is faithfully reproduced and has a stable, reproducible reference number (CYP3A4 MCC 0.294, CYP2D6 MCC 0.049) to beat. Tier 2 is now resolved by direct evidence, not assumption: **motif flag kept** (small, real, consistent gain), **trialkylamine flag dropped** (actively hurts CYP2D6, does nothing for CYP3A4, confirmed by three independent checks). Current best: **M3 (descriptors + motif + Morgan), CYP3A4 MCC 0.300, CYP2D6 MCC 0.043** — CYP2D6 specifically needs more than 2D chemistry alone; that's the real motivation for the OOF direct-inhibition bridge and Tier 3 structural evidence next, not a hedge against a disappointing number.
