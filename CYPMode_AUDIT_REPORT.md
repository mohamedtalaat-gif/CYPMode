# CYPMode: Full Codebase Audit (2026-09-26)

## Scope and method

Every file under `cypmode/` (38 modules, ~5,000 lines across `challenge/`,
`validation/`, `metrics/`, `data/`), every file under `tests/` (34 files,
~2,900 lines), `app.py`, `pyproject.toml`, `requirements.txt`,
`.github/workflows/tests.yml`, `CONTRIBUTING.md`, `README.md`,
`CYPMode_DATA_AUDIT.md`, `.gitignore`, and the three `docs/figures/*.py`
generation scripts. Read in full, cross-referenced against tests, and
checked against the real filesystem/git history/live HF Space rather than
trusted from docstrings alone. Static analysis (`ruff`, `bandit`,
`vulture`, `radon`, `mypy`, `pip-audit`) run against the real codebase in
an isolated tool venv; the CI workflow's exact install+test steps were
independently reproduced in a clean venv, not just read.

Not covered: `third_party/` (vendored Boltz-2/OpenFold3 clones),
`.venv*`/`.conda-*` environments themselves, the live HF Space's own
private scoring Lambda (not publicly auditable).

## Executive summary

**The most important finding is that CI was not actually testing this
project.** `.github/workflows/tests.yml` installed only the `dev`/`tdc`
extras, but most of the test suite (everything under `cypmode/challenge/`,
19 of 38 real modules) needs `lightgbm`/`pmapper`/`tblite` from the
`challenge` extra. Reproduced directly in a clean venv: pytest aborted
entirely at collection (`ModuleNotFoundError: No module named 'lightgbm'`,
`Interrupted: 1 error during collection`) — meaning a push to `main` would
show red (or in a subtly worse case, if the failing import were guarded,
green while silently running a fraction of the suite), either way with
close to none of the real ~200-test suite actually verified. **Fixed and
re-verified**: 134 passed / 7 skipped / 0 errors with the corrected
workflow.

Two other real, load-bearing bugs were found and fixed: a live path bug
in `pose_validity.py` (reintroduced by the September reorg, not yet
triggered only because nothing had re-run that code since) that would
have silently produced an empty PoseBusters result the next time it ran,
and a dormant tblite+LightGBM same-process deadlock risk in
`quantum_pharmacophore.py`'s own `_main()` — currently masked only because
today's feature cache happens to be fully warm, but would hang silently
(0% CPU, no crash) the moment it runs against any new compound, including
the actual 750-compound blind test this pipeline exists to score.

A real, if narrower, statistical leak was also found and fixed: two of
`multitask.py`'s four cross-validation functions standardized features
using the full dataset's mean/std (not the training fold's), while the
other two in the same file already did this correctly — meaning the
already-published shared-trunk MCC numbers in
`CYPMode_Architecture_Specification.md` were computed with a small,
real, scale-only leak. The fix is in; those numbers should be
re-generated (see Roadmap).

Total: **14 confirmed issues fixed and re-verified this session**
(1 Critical, 2 High, 5 Medium, 6 Low), plus **9 confirmed issues
documented but not fixed** (judgment calls, statistical redesigns, or
repo-structure decisions that are yours to make — see "Confirmed, not
fixed"), and a long list of things specifically checked and found
genuinely correct (see "Verified clean"). All 209 tests pass across every
required process-isolation group after every fix (178 main + 14
multitask/torch + 4 ChemBERTa/torch + 13 quantum-pharmacophore/tblite,
the last two groups also gaining new regression tests this session).

---

## Fixed this session

### 1. [Critical] CI installs the wrong dependency set — most of the suite never ran
**Location:** `.github/workflows/tests.yml`
**Evidence:** Reproduced directly (clean venv, exact CI steps): `pip install -e ".[dev,tdc]"` + `pip install --no-deps "PyTDC==1.1.14"`, then `pytest -v` →
```
collected 49 items / 1 error / 21 skipped
ERROR collecting tests/test_chemberta_significance.py
ModuleNotFoundError: No module named 'lightgbm'
Interrupted: 1 error during collection
21 skipped, 1 error in 91.98s
```
**Problem:** The workflow's own comment ("dev + tdc extras only") describes the project as it was on 2026-09-20, before `cypmode/challenge/` (added 2026-09-24/25) existed. A pytest collection *error* (as opposed to a per-test failure) aborts the entire run — none of the ~178 tests in the main group got a chance to report pass/fail.
**Impact:** Every push/PR since `cypmode/challenge/` was added has had essentially no real CI coverage.
**Fix:** Added the `challenge` extra to the main install step, plus three additional steps (each its own `pip install` + `pytest` invocation) to actually run the three process-isolated test files (`test_multitask.py`, `test_chemberta_features.py`, `test_quantum_pharmacophore.py`), which previously ran nowhere in CI at all.
**Verified:** Re-ran the corrected steps in a clean venv → `134 passed, 7 skipped, 5 warnings in 77.24s`, 0 errors.

### 2. [High] `pose_validity.py`: live, currently-broken path
**Location:** `cypmode/validation/pose_validity.py`, `_main()`
**Evidence:** `out_dir = input_dir / "out"` where `input_dir = Path("data/structure_track/inputs")` → resolves to `data/structure_track/inputs/out`, which does not exist (`ls`: No such file or directory). The real Boltz-2 output lives at `data/structure_track/outputs/out`.
**Problem:** Reintroduced by the September 26 reorg commit (`b5fc67d`); missed by the earlier `grep -rln '"data"'` sweep this session because here the base path is one combined string literal (`Path("data/structure_track/inputs")`) with `"out"` appended via `/`, not three separate literals — a different shape of the same bug class already fixed twice elsewhere today.
**Impact:** The next run of `python -m cypmode.validation.pose_validity` (needed imminently, to refresh the now-stale `posebusters_validity.csv` once the in-flight Structure Track batch finishes) would silently find zero structures and overwrite the real 57-row result with an empty one.
**Fix:** Corrected to `Path("data/structure_track/outputs/out")`.

### 3. [High] `quantum_pharmacophore.py`: dormant tblite+LightGBM same-process deadlock
**Location:** `cypmode/challenge/quantum_pharmacophore.py`, old `_main()`
**Evidence:** `_main()` did `from lightgbm import LGBMClassifier` and, later in the same process, called `add_quantum_features(...)` — which, for any SMILES not already cached, lazily imports `tblite`. The module's own docstring states this exact combination is a confirmed, reproduced deadlock (not a crash — 0% CPU, no progress), the same class of conflict already fixed for torch+LightGBM via `chemberta_features.py`/`chemberta_significance.py`'s process split.
**Impact:** Currently dormant only because `quantum_pharmacophore_cache.csv` already covers all 4,822 current training compounds (confirmed: 0 cache misses on a rerun today) — but the real, intended next use of this pipeline is the 750-compound blind test set, which would hit a 100% cache-miss rate and hang.
**Fix:** Split into an extraction-only `_main()` (never imports lightgbm) and a new module `cypmode/challenge/quantum_pharmacophore_significance.py` (lightgbm-only, never imports tblite) — mirroring the exact pattern `chemberta_features.py`/`chemberta_significance.py` already established for the analogous torch conflict. Added `load_cached_quantum_features()` as the lightgbm-free cache reader the new module uses.
**Verified:** `python -c "import cypmode.challenge.quantum_pharmacophore_significance"` loads with zero `tblite.*` modules in `sys.modules`. Isolated test suite: 13/13 pass (9 original + 4 new).

### 4. [Medium] `multitask.py`: full-dataset (not fold-internal) feature standardization
**Location:** `cypmode/challenge/multitask.py`, `cross_validate_shared_trunk` and `cross_validate_multi_branch_shared_trunk`
**Evidence:** Both functions computed `mean, std = X_raw.mean(axis=0), X_raw.std(axis=0)` over the *entire* dataset before the fold loop, then used those statistics to standardize both the training and validation splits. The other two CV functions in the same file (`cross_validate_shared_trunk_with_threshold_tuning`, `_inner_oof_threshold_shared_trunk`) already compute this from `X_raw[train_idx]` only — the correct pattern exists in the same file, just wasn't applied everywhere.
**Problem:** A real, if scale-only (not label), statistical leak: the validation fold's own feature values contribute to the mean/std used to scale it.
**Impact:** The shared-trunk MCC numbers currently published in `CYPMode_Architecture_Specification.md` (CYP3A4 0.250/0.266, CYP2D6 0.099/0.113) were computed with this leak present. The direction/magnitude of the effect on those specific numbers is not yet known — likely small, since it's a scale leak not a label leak, but not zero.
**Fix:** Moved mean/std computation inside the fold loop, `train_idx`-only, in both functions.
**Verified:** 14/14 multitask tests still pass.
**Follow-up needed:** re-run `python -m cypmode.challenge.multitask` to regenerate the real shared-trunk numbers and update the architecture spec if they move (see Roadmap).

### 5. [Medium] `pharmacophore_fingerprint.py`: failed-embedding rows not excluded from significance test
**Location:** `cypmode/challenge/pharmacophore_fingerprint.py`, `_main()`
**Evidence:** `n_empty` (compounds with an all-zero "failed" fingerprint) was computed and printed but never used to filter `cyp_df`/`aux_features` before `repeated_cv_stacking_comparison`. The sibling `quantum_pharmacophore.py` and `chemberta_significance.py` both explicitly build a validity mask and filter before scoring.
**Impact:** An all-zero fingerprint is a specific false claim ("no pharmacophoric features present"), not a neutral value — feeding it to the significance test as real signal. Current real impact is small (2/4,822 rows are all-zero), but nothing would catch a batch with a higher genuine failure rate (e.g. more awkward blind-test chemotypes) silently contaminating the "does Pharmacophore help" conclusion.
**Fix:** Added the same validity-mask-and-filter step the sibling modules use.

### 6. [Medium] `pyproject.toml`: PyYAML declared under the wrong extra
**Location:** `pyproject.toml`, `validate` extra
**Evidence:** `cypmode/validation/pose_validity.py`'s `_main()` does `import yaml`, but `PyYAML` was declared only under the `tdc` extra. README's documented reproduction path for this exact file is `pip install -e ".[validate]"`.
**Impact:** A contributor following the README's own documented command for this file would hit `ModuleNotFoundError: No module named 'yaml'`. Not caught by tests (`test_pose_validity.py` never exercises `_main()`).
**Fix:** Added `PyYAML` to the `validate` extra.

### 7. [Medium] `octant_benchmark.py`: non-atomic cache download
**Location:** `cypmode/data/octant_benchmark.py`, `load_inhibition_data()`
**Evidence:** `urllib.request.urlretrieve(_PARQUET_URL, cache_path)` wrote directly to the final cache path, with the guard `if not cache_path.exists()`.
**Impact:** An interrupted download (network drop, Ctrl-C, disk full) leaves a partial/corrupt file exactly at `cache_path`; every subsequent call sees "exists," skips re-downloading, and hands the corrupt file to `pd.read_parquet` — a real live-service risk since this hits an actual HuggingFace endpoint at runtime.
**Fix:** Download to a `.part` temp file, then atomically `rename()` into place.
**Verified:** All 5 `test_octant_benchmark.py` tests still pass.

### 8. [Low-Medium] pytest markers declared but not applied
**Location:** `pyproject.toml` markers `torch`/`tblite`; `tests/test_chemberta_features.py`, `tests/test_quantum_pharmacophore.py`
**Evidence:** The `torch` marker's own docstring names both `test_multitask.py` and `test_chemberta_features.py`, but only the former actually carried `pytestmark = pytest.mark.torch`. The `tblite` marker was applied to zero files, including the one its own docstring names.
**Impact:** Currently harmless (nothing in the repo invokes `pytest -m torch`/`-m tblite` — isolation is enforced via `addopts --ignore` + `importorskip`, which works regardless) but misleading metadata that would silently mis-select tests if anyone ever relied on it.
**Fix:** Added the missing `pytestmark` lines to both files.

### 9. [Low] `bisy_rmsd.py`: dead variable, wasted computation
**Location:** `cypmode/validation/bisy_rmsd.py`, `bisy_rmsd()`
**Evidence:** `pred_ca = binding_site_ca_atoms(predicted_structure, ligand_cutoff)` was computed (a real O(n_residues × n_ligand_atoms) distance computation) and never used — the function correctly uses `pred_all_ca` (all residues, not just the predicted structure's own binding site) per its own documented, correct design. Cross-confirmed by both `ruff` (F841) and `vulture`.
**Fix:** Removed the dead line.

### 10. [Low] `pose_comparison.py`: docstring has the mutation direction backwards
**Location:** `cypmode/validation/pose_comparison.py`, `ligand_pose_rmsd()`
**Evidence:** Docstring claimed `GetBestRMS` aligns `mol_b` onto `mol_a`; RDKit's own docs and signature (`GetBestRMS(prbMol, refMol, ...)`) confirm the *first* positional argument (`mol_a`) is the probe molecule that gets mutated/aligned, not `mol_b`. The returned RMSD value is symmetric either way, so this was never a live bug — only a wrong statement about a side effect, in code the module's own docstring says is staged for near-future real use.
**Fix:** Corrected the docstring.

### 11. [Low] `panel.py`: redundant motif computation
**Location:** `cypmode/data/panel.py`, `annotate_mechanism()`
**Evidence:** Called both `detect_motifs(smiles)` and `classify_mechanism(smiles)`, the latter internally re-running `detect_motifs()` on the same SMILES — doubling `Chem.MolFromSmiles` + all 6 SMARTS matches per row, across 5,000+ rows in `load_panel()`.
**Fix:** Derive `mechanism` from the single already-computed `motifs` list instead of calling `classify_mechanism()` a second time.

### 12–13. [Low] Stale comments/docs from the September reorg
- `requirements.txt`'s comment still referenced `data/boltz_test/validation_summary.json` (the actual code was already correct — comment-only drift). Fixed.
- `CYPMode_DATA_AUDIT.md`'s Feature Eligibility Matrix still said Tier 3 structural coordination was "not yet run" — false; real per-compound results exist for the 50-compound selection. Fixed the line.

### 14. Mechanical cleanup
`ruff --fix` applied 47 safe auto-fixes (import sorting, dead `# noqa` comments for a rule that isn't even enabled, `.format()` → f-string, one genuinely-unused import). Manually fixed the 5 remaining findings `--fix` couldn't handle: 3 unused-unpacked-variables in tests (prefixed with `_`), 2 redundant `int(round(...))` casts in `threshold_tuning.py`. The 9 that remain (7 `BLE001` blind-except, 1 `PLW1510`, 1 `S112`) were each individually reviewed and are deliberate, documented design choices (per-compound failure fallbacks, a bounded retry loop, a manually-checked subprocess result with a richer error message than `check=True` would give) — not bugs.

---

## Confirmed, not fixed (your call — judgment/scope beyond a straightforward fix)

### 15. [Medium-High] `amines.py`: trialkylamine SMARTS filter has real false positives beyond its documented fix
**Location:** `cypmode/metrics/amines.py`, `_TRIALKYLAMINE_PATTERNS`
The docstring documents excluding one false-positive chemotype (ketoconazole's N-aryl piperazine, via `!$(N-a)`). Directly verified against RDKit that three *other* chemotypes still false-positive: a sulfonamide nitrogen (`CN(C)S(=O)(=O)c1ccccc1`), an enamine (`CC=CN(C)C`), and a guanidine (`CN(C)C(=N)N`) — none excluded by the current pattern, and sulfonamide is an extremely common drug-like group. Bounded real impact today: `amine_flag` was already tested-and-dropped from the live model in the M0–M3 ablation (it measurably hurt CYP2D6), so this doesn't affect current predictions — but the module itself would mislabel these groups if reused. Needs a real chemistry decision on the correct exclusion SMARTS, not a rushed fix under audit time pressure.

### 16. [Medium] `stacking_significance.py`: Tukey HSD treats a paired design as independent groups
**Location:** `cypmode/challenge/stacking_significance.py`, `tukey_hsd_summary()`
Every variant (M3, M3+KNN, ...) in a given (repeat, fold) row is scored on the *same* train/val split — a repeated-measures design — but `pairwise_tukeyhsd` is called on the raw MCC/variant columns, which assumes independent observations. This doesn't bias the estimated mean differences, but inflates the error term (fold-to-fold difficulty variance shared across all variants in a row isn't netted out), making the test *more conservative* — a real risk specifically for this project's own "is combining features actually non-significant" calls (e.g. the CYP2D6 KNN+BindingDB interaction). Needs a properly blocked/paired statistical redesign (e.g. a mixed model, or blocking by (repeat,fold) before the comparison), which deserves dedicated review rather than a same-session patch.

### 17. [Medium] `batch.py`: resume behavior for error placeholders is real but untested
**Location:** `cypmode/validation/batch.py`, `run_batch()`; `tests/test_batch.py`
`if name in results: continue` treats a pre-existing `{"error": ...}` the same as a success — permanently skipped, never auto-retried (this session's own Tier3 retry chain had to manually delete error entries before a re-run could pick them up). A reasonable design, but the existing resume test only seeds a *successful* prior result — nothing asserts the error-placeholder behavior, so a future refactor could silently change it.

### 18. [Medium] `oof_bridge.py`: leakage test doesn't actually discriminate leakage
**Location:** `tests/test_oof_bridge.py`
The test named `..._predicted_by_a_model_that_never_saw_it` only asserts `corr > 0.3` — a leaking implementation would likely score *higher*, not lower, and would still pass. Real risk is lower than the gap suggests (the actual code uses stock `sklearn.KFold`, not hand-rolled exclusion logic prone to off-by-one leaks), but the test's name oversells what it verifies.

### 19. [Medium] README.md doesn't document `cypmode/challenge/` at all
The architecture walkthrough documents `cypmode/metrics/`, `cypmode/data/`, and most of `cypmode/validation/`, but none of the 19 modules in `cypmode/challenge/` — the single largest package, everything from the M0–M3 ablation through ChemBERTa. Real orientation gap for anyone reading the README to understand the codebase, not a functional bug.

### 20. [Medium] CONTRIBUTING.md describes an earlier, much smaller project
Still says "a motif screen plus a validation harness," lists only `dev`/`tdc`/`validate` extras, and tells a new contributor to just run `pytest`. No mention of `challenge`/`multitask`, and critically, no mention that 3 test files must be run in separate invocations to avoid a real segfault/deadlock. A contributor following this literally would silently never run 27 of 209 real tests and would have no idea why.

### 21. [Medium] TDI figures and the app.py results tab predate 6 real feature experiments
`docs/figures/tdi_track/` (2 figures) and `app.py`'s "TDI Track Results" tab both only cover the original baseline → M3 → tuned-threshold progression. None of KNN, BindingDB, AID 1851, quantum-electronic, pharmacophore-fingerprint, ChemBERTa, or the multi-branch fusion architecture is visualized anywhere, even though real result CSVs for all of them exist in `data/tdi_track/outputs/`. A reader of the figures/app sees a meaningfully smaller, older picture than what the architecture spec documents in prose.

### 22. [Medium-High] Regeneratable caches committed inconsistently with the project's own stated policy
`.gitignore` excludes the Structure Track's raw Boltz-2 output (1.4GB) with a clear rationale ("regenerated, not committed"). But three functionally identical, regeneratable TDI-track feature caches are tracked in git: `chemberta_embedding_cache.csv` (**70MB**), `pharmacophore_fingerprint_cache.csv` (**38MB**), `quantum_pharmacophore_cache.csv` (~1MB) — over 100MB of regeneratable data, with the largest file approaching GitHub's 100MB hard per-file limit. This is a real repo-hygiene inconsistency, but removing it from tracking is a real trade-off (loses "clone and everything's already there" convenience, and — since it's already committed — only stops *future* growth unless you also rewrite history, which is a separate, more invasive decision). Flagged for your call, not changed.

### 23. [Low] `cypmode/validation/constants.py` has no test file
Pure data (a coordination-distance float, a 400+ residue protein sequence string) — low real risk, but a typo'd residue would go undetected. Cheapest possible fix if you want it: one test asserting the sequence's length and maybe a known-residue spot-check.

---

## Verified clean (checked directly, not assumed)

- **Leakage guards**: `knn_features.py` (adversarially tested — two identical-structure compounds, different labels, confirmed the self-exclusion actually catches it), `tier3_selection.py` (test flips validation labels and confirms the selection policy is unchanged), `oof_bridge.py`'s actual fold-internal construction (correct, only its *test* is weak — see #18).
- **Import-chain isolation** across all of `cypmode/challenge/`: only the one real gap found (#3, now fixed) — every other module's lazy vs. top-level imports correctly avoid the torch/lightgbm/tblite conflict, including non-obvious cases like `chemberta_significance.py` safely importing from `chemberta_features.py` without transitively pulling in torch.
- **Path construction** across every file in `cypmode/`: every hardcoded/constructed path checked against the real, current disk layout. Only the one live break found (#2).
- **Motif/amine chemistry** (`motifs.py`): all 6 SMARTS patterns verified correct against 17 real/near-miss heterocycles (correctly matches imidazole/triazole/pyridine/thiazole/benzimidazole, correctly rejects pyrazole/oxazole/isoxazole/tetrazole/pyrimidine/isothiazole/thiophene/furan). `find_coordinating_cysteine`'s residue-index arithmetic confirmed correct against the real CYP3A4 Cys442 position.
- **Geometry/units**: no Å/nm mixing found across `pocket_geometry.py`/`structures.py`/`bisy_rmsd.py`; SASA and pocket-volume both consistently in Å²/ų; RDKit-conformer vs. Biopython-atom coordinate frames confirmed consistent (both built from the same predicted atom positions).
- **`submission.py`**'s boolean-dtype handling: initially suspected a `numpy.bool_`/`bool` mismatch risk, verified empirically in the real environment that it's not an issue.
- **`tdi_reference.py`/`octant_benchmark.py` provenance**: real per-drug threshold/motif calls hand-recomputed and matched against the checked-in reference data; `octant_benchmark.py`'s statistical test (one-sided Mann-Whitney U) matches its own stated hypothesis and inclusion criteria exactly.
- **Test hygiene**: zero vacuous tests found across a full AST scan of all 34 test files; no `conftest.py`, no shared mutable fixtures, no test-order dependency; every cache-writing test correctly uses `tmp_path`, never the real production cache path.
- **CI `addopts` ignore-list**: verified complete and accurate — exactly the 3 files that need process isolation are ignored, no more, no less.
- **`README.md`**'s concrete claims** (CLI args, function names, the validation results table) all verified accurate against real code/data — everything except the `challenge/` omission (#19).
- **Radon complexity**: no function scored worse than C (moderate); average B across the codebase — nothing alarmingly complex.
- **Bandit security scan**: no High-severity findings. The ChemBERTa unpinned `from_pretrained()` revision (Medium) is worth noting for reproducibility but not fixed here (a real, separate judgment call about pinning a specific HF model commit). `subprocess`/`urlopen` usages all reviewed and are safe (list-constructed commands, hardcoded URLs, no untrusted input reaching them).
- **`pip-audit`**: one known CVE (`setuptools` `PYSEC-2026-3447`, fixed in 83.0.0) — but `setuptools<81` is a *deliberate* pin (PyTDC needs the bundled `pkg_resources` modern setuptools removed), so the fix version is incompatible with a real, documented project constraint. Not fixed; a real, currently-accepted risk, low practical severity for a local research CLI with no untrusted-package-install threat model.

---

## Prioritized roadmap

**Critical (done):** CI dependency fix — already applied and verified.

**High (done):** `pose_validity.py` path fix, `quantum_pharmacophore.py` process-isolation fix — both applied and verified.

**Medium — recommended next, in this order:**
1. Re-run `python -m cypmode.challenge.multitask` to regenerate the shared-trunk MCC numbers now that the standardization leak (#4) is fixed, and update `CYPMode_Architecture_Specification.md` if they move.
2. Decide the correct SMARTS fix for `amines.py` (#15) — needs a chemistry call, not a rushed pattern.
3. Redesign `stacking_significance.py`'s Tukey HSD to respect the paired (repeat, fold) design (#16) — affects the confidence of every "not statistically significant" claim already published.
4. Decide on the committed-cache question (#22) — `git rm --cached` + gitignore going forward is straightforward; a full history rewrite to actually shrink the repo is a separate, bigger decision.
5. Update `README.md` (#19) and `CONTRIBUTING.md` (#20) to cover `cypmode/challenge/`, its two extras, and the 3-file test-isolation requirement.
6. Regenerate/extend the TDI figures and `app.py`'s results tab (#21) to include the six post-M3 feature experiments.

**Low — cheap, whenever convenient:**
- Add the missing `run_batch` error-retry test (#17) and tighten the `oof_bridge.py` leakage test (#18) — both are small, well-scoped test additions.
- A one-line test for `constants.py` (#23) if you want the coverage.

**Testing:** all 209 tests (178 main + 14 + 4 + 13 across the three isolated groups) pass as of this report. No test debt was introduced by any fix above; 4 new tests were added (`quantum_pharmacophore.py`'s multi-worker path and its new `load_cached_quantum_features` loader) to close a gap the audit itself found (#3's sibling issue, per Agent 3's finding that this exact untested-multi-worker shape had already caused a real crash once, in `pharmacophore_fingerprint.py`).
