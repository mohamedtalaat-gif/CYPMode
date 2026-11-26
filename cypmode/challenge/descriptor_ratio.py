# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The number of columns against the number of compounds: what each model sees, how many dimensions the data hold, and what extra columns do to the stack.

Why. Jorgensen (J. Chem. Inf. Model. 2006, 46, 937) names "model development featuring high ratios of descriptors to data points" among the practices that the journal discourages: with more
descriptors than compounds a regression can fit chance correlations (Leach, Molecular Modelling, 2nd ed., section 12.13; Topliss and Edwards, J. Med. Chem. 1979, 22, 1238, for the rate at which
chance correlations appear as descriptors are screened). The feature store of the project has 12,468 columns for 6,895 compounds, so the ratio has to be answered with measurements and not with an
argument about tree ensembles. Five measurements, all on the pools and the nested plans of the final build:

1. Widths. For every candidate of the default grid, the number of columns that the candidate reads and the ratio of that number to the rows of the pool (the whole store, 12,468 columns, is read by
   no candidate; the widest fingerprint set reads 5,826 columns).
2. Effective dimension. For the widest set of the grid (``A+fps``) on the pool rows, standardised: the number of non-constant columns, the rank, the number of principal components that hold 90% and
   95% of the variance, and the participation ratio of the spectrum, ``(sum of eigenvalues)^2 / sum of squared eigenvalues`` (the number of components that carry the variance when it is spread evenly).
3. Extra columns that carry no signal. ``lgbm_A_deep`` and ``lgbm_A+fps_deep`` are refitted on every outer fold of the final build (3 repeats of 5 folds) with 0, 1, and 4 times their own number
   of columns added, and scored by the area under the receiver operating characteristic curve (AUC) on the same yardstick as the stack (the weights for similarity, prevalence, and confident rows; z0, the cut on the
   confidence of the label, in {1.0, 1.96}). Two kinds of extra column: ``shuffled`` (the whole design matrix of the
   pool with its rows permuted at random, once per multiple: columns of the same kinds, the same marginal distributions and the same correlations among themselves, that belong to other compounds and
   so cannot carry the label of this one; they are what chance correlation is made of) and ``gaussian`` (independent standard normal columns, a harsher test because a continuous column offers a split
   at every row). A model that fits chance correlations loses AUC as columns that cannot carry a signal are added.
4. The stack. The 16 label-free candidates of the default grid (the ones that do not read the assay arms or a block trained on an external panel) go through the whole nested procedure of the final
   build (inner out-of-fold scores, the logistic blend chosen on them, the outer fold) for the first repeat of 5 outer folds, once as they are and once with 4 times the columns of every candidate that
   reads columns (the LightGBM, extremely randomised trees, and random forest candidates) added as shuffled columns. The neighbour and kernel candidates read the fingerprint only and are unchanged. The
   question is whether the procedure, which can weight a candidate down, keeps its estimate when the candidates that read columns are fed signal-free ones.
5. Fewer columns. ``lgbm_A+fps_deep`` on a random 5%, 10%, 25%, 50% and 100% of its columns (one draw of the columns per fraction, the same on every fold).
6. What the nested design already measured. The outer-fold AUC of each default candidate against its width, and the permutation control of the whole pipeline (outer AUC 0.5 on permuted labels),
   read from its logs.

Rule, fixed before the first run (2026-11-26):

1. Adding 4 times the columns that carry no signal is NOT HARMFUL for a candidate and a kind of column when, at both confidence cuts, the mean paired loss of AUC is below the margin of 0.01 and a
   one-sided test of the corrected resampled t (Nadeau and Bengio, 2003; the variance factor of the final design) rejects a loss of 0.01 or more at p < 0.05. The margin is that of the non-inferiority
   test of the relaxation gate (``relax_final``) and is about the design floor of 0.0099 that the main design can resolve. Otherwise the verdict is HARMFUL OR NOT RESOLVED, and it is reported as it is.
   Both kinds of column are registered and both verdicts are reported; the shuffled columns are the ones that resemble the columns of the store. (The two kinds were registered after a first look at
   3 folds of CYP2D6 with Gaussian columns alone, which lowered the AUC of ``lgbm_A_deep`` from 0.69 to 0.47 and raised the question of what a realistic null column is; that look is not a result of
   this analysis.)
2. The stack is judged by the same rule as a candidate: NOT HARMFUL when, at both cuts, the paired loss of the outer AUC of the stack is below 0.01 and the one-sided corrected t over the 5 outer folds
   of that repeat rejects a loss of 0.01 or more at p < 0.05; with 5 folds the test is conservative (variance factor 1/5 + 1/4), and a result that it cannot resolve is reported as not resolved.
3. Items 1, 2, 5, and 6 carry no verdict. Nothing is tuned after the first run, and no verdict line matches the lines that the ledger of gates reads.

Usage: python -m cypmode.challenge.descriptor_ratio [--n-jobs 8] [--isoform CYP3A4]
Outputs (data/tdi_track/outputs/comparisons): descriptor_ratio_widths.csv, descriptor_ratio_spectrum.csv, descriptor_ratio_noise.csv, descriptor_ratio_stack.csv, descriptor_ratio_fraction.csv, descriptor_ratio_candidates.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cypmode.challenge import final_tdi_build as final
from cypmode.challenge.external_dataset import external_candidates
from cypmode.challenge.feature_store import OPT_IN_BLOCKS, FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes
from cypmode.challenge.nested import (
    OUTER_REFIT,
    NestedPlan,
    PredictionCache,
    StackedLogit,
    _weights,
    nested_evaluate,
    score_metric,
)
from cypmode.challenge.nested_runners import (
    FEATURE_SETS,
    Candidate,
    Context,
    _design,
    _labels_for,
    _seed,
    default_candidates,
)
from cypmode.challenge.validation_matching import SimilarityMatcher

logger = logging.getLogger(__name__)

OUT = Path("data/tdi_track/outputs/comparisons")
LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
Z_CUTS = final.Z_CUTS
NOISE_MULTIPLES = (0, 1, 4)
NOISE_KINDS = ("shuffled", "gaussian")
KINDS_OF = {"lgbm_A_deep": ("shuffled", "gaussian"), "lgbm_A+fps_deep": ("shuffled",)}  # a continuous column at 4 times the width of the larger set takes minutes per fit
STACK_MULTIPLE = 4
WORK = Path("data/tdi_track/outputs/comparisons/descriptor_ratio_work")
NOISE_CANDIDATES = ("lgbm_A_deep", "lgbm_A+fps_deep")
FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
FRACTION_CANDIDATE = "lgbm_A+fps_deep"
MARGIN = 0.01
ALPHA = 0.05
VARIANCE_SHARES = (0.90, 0.95)
SEED = 0


def candidate_width(candidate: Candidate, store: FeatureStore) -> int:
    """Columns that a candidate reads: the blocks of its feature set, or the 2048 fingerprint bits of the Tanimoto kernel for the neighbour and kernel models, plus the 4 read-across columns."""
    if candidate.family in ("knn_tanimoto", "svm_tanimoto"):
        return store.blocks["morgan_bits"].shape[1]
    return sum(store.blocks[b].shape[1] for b in candidate.blocks) + (4 if candidate.knn_label_features else 0)


def width_table(store: FeatureStore, candidates: list[Candidate], rows: dict[str, int]) -> pd.DataFrame:
    out = []
    for c in candidates:
        width = candidate_width(c, store)
        out.append({"candidate": c.name, "family": c.family, "n_columns": width, **{f"ratio_{iso}": width / n for iso, n in rows.items()}})
    return pd.DataFrame(out)


def spectrum(matrix: np.ndarray) -> dict[str, float]:
    """The variance spectrum of the standardised non-constant columns: how many columns, how many dimensions the compounds span, how many components carry the variance."""
    x = np.asarray(matrix, dtype=np.float64)
    sd = x.std(axis=0)
    x = (x[:, sd > 0] - x[:, sd > 0].mean(axis=0)) / sd[sd > 0]
    singular = np.linalg.svd(x, compute_uv=False)
    eigen = singular**2 / (len(x) - 1)
    share = np.cumsum(eigen) / eigen.sum()
    out = {"rows": float(len(x)), "columns_used": float(x.shape[1]), "rank": float(int((singular > singular[0] * 1e-10).sum())), "participation_ratio": float(eigen.sum() ** 2 / (eigen**2).sum())}
    for s in VARIANCE_SHARES:
        out[f"components_for_{int(s * 100)}"] = float(int(np.searchsorted(share, s)) + 1)
    return out


def noninferior(differences: np.ndarray, n_train: int, n_test: int, margin: float = MARGIN) -> tuple[float, float]:
    """(mean difference, one-sided p) for the null hypothesis that the loss is ``margin`` or more: the corrected resampled t of the differences shifted by the margin, halved to one side."""
    shifted = corrected_resampled_t(np.asarray(differences) + margin, n_train, n_test)
    p_one_sided = shifted.p_value / 2 if shifted.mean_difference > 0 else 1 - shifted.p_value / 2
    return float(np.mean(differences)), float(p_one_sided)


# -- the nested design of the final build ---------------------------------------------------------


class Final:
    """The plan, the caches, and the yardstick of the final build of one isoform, rebuilt as ``applicability_domain.final_stack`` does."""

    def __init__(self, iso: str, n_jobs: int):
        self.iso = iso
        blocks = tuple(sorted({b for c in default_candidates() for b in c.blocks}))
        self.ctx, plan = final.build_context_and_plan(final.TRAIN_CSV, final.TEST_CSV, iso, final.STORE, final.REPEATS, final.DESIGN["outer"], 4, 0, n_jobs, blocks)
        saved = NestedPlan.load(Path(final.DESIGN["root"]) / iso / "plan.npz")
        assert saved.plan_id == plan.plan_id, "the plan changed: this analysis must share the caches of the final build"
        self.plan = saved
        self.cache = PredictionCache(Path(final.DESIGN["root"]) / iso / "cache", saved)
        self.store = self.ctx.store
        self.matcher = SimilarityMatcher(self.store.blocks["morgan_bits"], saved.external)
        self.prevalence = float(estimate_from_csv(final.BOARDS[iso])["prevalence"])
        self.weight_fns = {z0: confident_weights(self.matcher, self.ctx.confidence, z0) for z0 in Z_CUTS}

    def outer_auc(self, scores_by_fold: dict[tuple[int, int], np.ndarray]) -> dict[float, list[float]]:
        """The weighted outer AUC of one candidate at each confidence cut, one value per outer fold (same order as ``plan.outer_folds()``)."""
        out: dict[float, list[float]] = {z0: [] for z0 in Z_CUTS}
        for (repeat, k), scores in scores_by_fold.items():
            task = self.plan.task(repeat, k, OUTER_REFIT)
            y = self.plan.labels(task.evaluate)
            for z0 in Z_CUTS:
                w = _weights(y, task.evaluate, task.train, self.weight_fns[z0], self.prevalence)
                out[z0].append(score_metric(y, scores, w, "auc"))
        return out

    def cached_scores(self, name: str) -> dict[tuple[int, int], np.ndarray]:
        return {(r, k): self.cache.get(name, self.plan.task(r, k, OUTER_REFIT)) for r, k in self.plan.outer_folds()}


def _fit_tree_model(candidate: Candidate, ctx: Context, task, train_extra: np.ndarray | None, eval_extra: np.ndarray | None, columns: np.ndarray | None) -> np.ndarray:
    """The ``lgbm_clf``, ``extratrees``, and ``rf`` paths of ``fit_predict`` with extra columns appended to the design or a subset of its columns kept; the labels read are those of the training rows."""
    y = _labels_for(ctx, task)
    train_x, eval_x = _design(candidate, ctx, task, y)
    if columns is not None:
        train_x, eval_x = train_x[:, columns], eval_x[:, columns]
    if train_extra is not None:
        train_x, eval_x = np.hstack([train_x, train_extra]), np.hstack([eval_x, eval_extra])
    seed = _seed(task) % 2**31
    if candidate.family == "lgbm_clf":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(verbose=-1, n_jobs=ctx.n_jobs, random_state=seed, **candidate.params)
    elif candidate.family in ("extratrees", "rf"):
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

        cls = ExtraTreesClassifier if candidate.family == "extratrees" else RandomForestClassifier
        model = cls(n_estimators=400, max_features=0.1, min_samples_leaf=2, n_jobs=ctx.n_jobs, random_state=seed)
    else:
        raise ValueError(f"{candidate.name} does not read columns")
    return model.fit(train_x, y).predict_proba(eval_x)[:, 1]


def extra_columns(kind: str, base: np.ndarray, multiple: int, seed: int) -> np.ndarray:
    """``multiple`` times the width of ``base`` (the design matrix of the pool rows) as columns without signal: row-permuted copies of the design (``shuffled``) or independent standard normals (``gaussian``)."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((len(base), multiple * base.shape[1])).astype(np.float32)
    if kind == "shuffled":
        return np.hstack([base[rng.permutation(len(base))] for _ in range(multiple)]).astype(np.float32)
    raise ValueError(kind)


def noise_experiment(f: Final, candidates: list[Candidate]) -> pd.DataFrame:
    """One row per (candidate, kind of extra column, multiple, outer fold): the outer AUC at each confidence cut."""
    position = {int(r): i for i, r in enumerate(f.plan.pool)}
    rows = []
    for c in candidates:
        base = f.ctx.store.matrix(c.blocks, f.plan.pool)
        for kind in KINDS_OF[c.name]:
            for multiple in NOISE_MULTIPLES:
                if multiple == 0 and kind != KINDS_OF[c.name][0]:
                    continue  # the fit without extra columns is one and the same for both kinds
                noise = extra_columns(kind, base, multiple, SEED + multiple) if multiple else None
                scores = {}
                for repeat, k in f.plan.outer_folds():
                    task = f.plan.task(repeat, k, OUTER_REFIT)
                    train_extra = None if noise is None else noise[[position[int(r)] for r in task.train]]
                    eval_extra = None if noise is None else noise[[position[int(r)] for r in task.evaluate]]
                    scores[(repeat, k)] = _fit_tree_model(c, f.ctx, task, train_extra, eval_extra, None)
                auc = f.outer_auc(scores)
                for i, (repeat, k) in enumerate(f.plan.outer_folds()):
                    rows.append({"isoform": f.iso, "candidate": c.name, "kind": kind if multiple else "none", "multiple": multiple, "extra_columns": multiple * base.shape[1], "repeat": repeat, "outer": k, **{f"auc_z{z0}": auc[z0][i] for z0 in Z_CUTS}})
                logger.info("%s %s %s x%d: AUC %s", f.iso, c.name, kind, multiple, {z0: round(float(np.mean(auc[z0])), 4) for z0 in Z_CUTS})
    return pd.DataFrame(rows)


class RoutedCache(PredictionCache):
    """A cache that holds the predictions of some candidates and borrows those of the others from another cache of the same plan."""

    def __init__(self, root: str | Path, plan: NestedPlan, borrowed: dict[str, PredictionCache]):
        super().__init__(root, plan)
        self.borrowed = borrowed

    def get(self, candidate: str, task) -> np.ndarray:
        return self.borrowed[candidate].get(candidate, task) if candidate in self.borrowed else super().get(candidate, task)


def stack_experiment(f: Final, candidates: list[Candidate], multiple: int = STACK_MULTIPLE) -> pd.DataFrame:
    """The nested procedure for the first repeat, with the stack over ``candidates`` as they are and with ``multiple`` times the columns of every candidate that reads columns added as shuffled columns."""
    plan = NestedPlan(f.plan.pool, f.plan.y, 1, f.plan.outer_splits, f.plan.inner_splits, f.plan.seed, external=f.plan.external, groups=f.plan.groups)
    assert plan.plan_id == f.plan.plan_id, "the first repeat must share the caches of the final build"
    plain = PredictionCache(Path(final.DESIGN["root"]) / f.iso / "cache", plan)
    names = [c.name for c in candidates]
    reading = [c for c in candidates if c.blocks]
    augmented = RoutedCache(WORK / f.iso, plan, {c.name: plain for c in candidates if not c.blocks})
    position = {int(r): i for i, r in enumerate(plan.pool)}
    tasks = [t for t in plan.tasks() if t.repeat == 0]
    for c in reading:
        base = f.ctx.store.matrix(c.blocks, plan.pool)
        noise = extra_columns("shuffled", base, multiple, SEED + multiple)
        for task in tasks:
            if not augmented.has(c.name, task):
                augmented.put(c.name, task, _fit_tree_model(c, f.ctx, task, noise[[position[int(r)] for r in task.train]], noise[[position[int(r)] for r in task.evaluate]], None))
        logger.info("%s stack experiment: %s done", f.iso, c.name)
    rows = []
    for z0 in Z_CUTS:
        for label, cache in (("plain", plain), ("extra columns", augmented)):
            table = nested_evaluate(cache, names, [StackedLogit()], f.weight_fns[z0], f.prevalence, "auc")
            rows += [{"isoform": f.iso, "variant": label, "z0": z0, "repeat": r.repeat, "outer": r.outer, "outer_auc": r.outer_score, "inner_auc": r.inner_score} for r in table.itertuples()]
    return pd.DataFrame(rows)


def summarise_stack(table: pd.DataFrame, f: Final, n_reading: int) -> list[str]:
    n_train, n_test = plan_sizes(NestedPlan(f.plan.pool, f.plan.y, 1, f.plan.outer_splits, f.plan.inner_splits, f.plan.seed, groups=f.plan.groups))
    lines, ok = [], True
    for z0 in Z_CUTS:
        part = table[table.z0 == z0]
        plain = part[part.variant == "plain"].sort_values(["repeat", "outer"])
        extra = part[part.variant == "extra columns"].sort_values(["repeat", "outer"])
        mean, p = noninferior(extra.outer_auc.to_numpy() - plain.outer_auc.to_numpy(), n_train, n_test)
        lines.append(f"  stack of the {len(plain)} outer folds of the first repeat, z0={z0:<4} outer AUC {plain.outer_auc.mean():.4f} as it is, {extra.outer_auc.mean():.4f} with {STACK_MULTIPLE} times the columns of {n_reading} candidates added | change {mean:+.4f}, one-sided p of a loss of {MARGIN} or more: {p:.4f}")
        ok &= bool(mean > -MARGIN and p < ALPHA)
    lines.append(f"CHECK {f.iso} stack with {STACK_MULTIPLE} times the columns of its column-reading candidates added (shuffled): {'NOT HARMFUL' if ok else 'HARMFUL OR NOT RESOLVED'}")
    return lines


def fraction_experiment(f: Final, candidate: Candidate) -> pd.DataFrame:
    rows = []
    total = candidate_width(candidate, f.store)
    for fraction in FRACTIONS:
        columns = None if fraction == 1.0 else np.sort(np.random.default_rng(SEED + int(fraction * 1000)).choice(total, size=max(round(fraction * total), 1), replace=False))
        scores = {}
        for repeat, k in f.plan.outer_folds():
            scores[(repeat, k)] = _fit_tree_model(candidate, f.ctx, f.plan.task(repeat, k, OUTER_REFIT), None, None, columns)
        auc = f.outer_auc(scores)
        for i, (repeat, k) in enumerate(f.plan.outer_folds()):
            rows.append({"isoform": f.iso, "candidate": candidate.name, "fraction": fraction, "n_columns": total if columns is None else len(columns), "repeat": repeat, "outer": k, **{f"auc_z{z0}": auc[z0][i] for z0 in Z_CUTS}})
        logger.info("%s %s fraction %.2f: AUC %s", f.iso, candidate.name, fraction, {z0: round(float(np.mean(auc[z0])), 4) for z0 in Z_CUTS})
    return pd.DataFrame(rows)


def candidate_table(f: Final, candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    for c in candidates:
        if f.cache.missing(c.name, False):
            continue
        auc = f.outer_auc(f.cached_scores(c.name))
        rows.append({"isoform": f.iso, "candidate": c.name, "family": c.family, "n_columns": candidate_width(c, f.store), "rows": len(f.plan.pool), **{f"auc_z{z0}": float(np.mean(auc[z0])) for z0 in Z_CUTS}})
    return pd.DataFrame(rows)


def store_columns() -> dict[str, int]:
    """Columns of every block of the whole feature store (the file holds the opt-in blocks too)."""
    with np.load(final.STORE, allow_pickle=True) as z:
        return {n: int(z[n].shape[1]) for n in z.files if n != "smiles"}


def permutation_lines(iso: str) -> list[str]:
    path = LOGS / f"permutation_control_{iso}.log"
    return [line.strip() for line in path.read_text().splitlines() if "real labels" in line] if path.exists() else []


def spectrum_table(f: Final) -> pd.DataFrame:
    rows = []
    for name in ("A", "A+fps"):
        x = f.store.matrix(FEATURE_SETS[name], f.plan.pool)
        rows.append({"isoform": f.iso, "feature_set": name, "columns_read": x.shape[1], **spectrum(x)})
    return pd.DataFrame(rows)


# -- report -------------------------------------------------------------------------------------


def summarise_noise(table: pd.DataFrame, f: Final) -> list[str]:
    n_train, n_test = plan_sizes(f.plan)
    lines = []
    ordered = ["repeat", "outer"]
    for name in NOISE_CANDIDATES:
        part = table[table.candidate == name]
        if part.empty:
            continue
        for kind in KINDS_OF[name]:
            verdict_ok = True
            for z0 in Z_CUTS:
                col = f"auc_z{z0}"
                reference = part[part.multiple == 0].sort_values(ordered)[col].to_numpy()
                means = [part[part.multiple == 0][col].mean()] + [part[(part.multiple == m) & (part.kind == kind)][col].mean() for m in NOISE_MULTIPLES[1:]]
                d = part[(part.multiple == NOISE_MULTIPLES[-1]) & (part.kind == kind)].sort_values(ordered)[col].to_numpy() - reference
                mean, p = noninferior(d, n_train, n_test)
                lines.append(f"  {name:16s} {kind:8s} z0={z0:<4} AUC with 0 / 1 / 4 times the columns: " + " / ".join(f"{m:.4f}" for m in means) + f" | change at x{NOISE_MULTIPLES[-1]}: {mean:+.4f}, one-sided p of a loss of {MARGIN} or more: {p:.4f}")
                verdict_ok &= bool(mean > -MARGIN and p < ALPHA)
            lines.append(f"CHECK {f.iso} {name} with {NOISE_MULTIPLES[-1]} times its columns added ({kind}): {'NOT HARMFUL' if verdict_ok else 'HARMFUL OR NOT RESOLVED'}")
    return lines


def main() -> None:
    from cypmode import provenance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", choices=["CYP3A4", "CYP2D6"], help="one isoform (both by default)")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    provenance.announce()
    OUT.mkdir(parents=True, exist_ok=True)
    default = default_candidates()
    named = {c.name: c for c in default}
    widths, spectra, noises, fractions, cands, stacks = [], [], [], [], [], []
    store_rows = None
    for iso in [args.isoform] if args.isoform else ["CYP3A4", "CYP2D6"]:
        f = Final(iso, args.n_jobs)
        if store_rows is None:
            columns = store_columns()
            print(f"\nThe feature store: {sum(columns.values())} columns in {len(columns)} blocks for {len(f.ctx.store.smiles)} compounds; {sum(w for n, w in columns.items() if n not in OPT_IN_BLOCKS)} in the blocks that are not opt-in, {max(candidate_width(c, f.store) for c in default)} in the widest candidate")
            store_rows = True
        n = len(f.plan.pool)
        print(f"\n== {iso}: {n} pool compounds, {len(f.plan.outer_folds())} outer folds")
        widths.append(width_table(f.store, default, {iso: n}).assign(isoform=iso, rows=n))
        w = widths[-1]
        print(f"  widest candidate {int(w.n_columns.max())} columns: {w.n_columns.max() / n:.2f} columns per compound; narrowest {int(w.n_columns.min())}: {w.n_columns.min() / n:.2f}; median {w.n_columns.median():.0f}: {w.n_columns.median() / n:.2f}")
        spectra.append(spectrum_table(f))
        for row in spectra[-1].itertuples():
            print(f"  effective dimension, {row.feature_set} ({row.columns_read} columns read, {row.columns_used:.0f} not constant): rank {row.rank:.0f}, {row.components_for_90:.0f} components for 90% and {row.components_for_95:.0f} for 95% of the variance, participation ratio {row.participation_ratio:.0f}")
        cands.append(candidate_table(f, default))
        c = cands[-1]
        rho = {z0: stats.spearmanr(c.n_columns, c[f"auc_z{z0}"]) for z0 in Z_CUTS}
        print("  outer AUC of each default candidate against its width (Spearman): " + "; ".join(f"z0={z0}: rho {r.statistic:+.2f} (p={r.pvalue:.3f}, {len(c)} candidates)" for z0, r in rho.items()))
        for name in ("lgbm_desc_deep", "lgbm_A_deep", "lgbm_A+fps_deep"):
            row = c[c.candidate == name]
            if len(row):
                print(f"    {name}: {int(row.n_columns.iloc[0])} columns, outer AUC {row['auc_z1.0'].iloc[0]:.4f} at z0=1.0 and {row['auc_z1.96'].iloc[0]:.4f} at z0=1.96")
        for line in permutation_lines(iso):
            print(f"  permutation control of the whole pipeline: {line}")
        noises.append(noise_experiment(f, [named[n] for n in NOISE_CANDIDATES]))
        for line in summarise_noise(noises[-1], f):
            print(line)
        stacks.append(stack_experiment(f, external_candidates()))
        for line in summarise_stack(stacks[-1], f, len([c for c in external_candidates() if c.blocks])):
            print(line)
        fractions.append(fraction_experiment(f, named[FRACTION_CANDIDATE]))
        for fraction in FRACTIONS:
            part = fractions[-1][fractions[-1].fraction == fraction]
            print(f"  {FRACTION_CANDIDATE} on {fraction:.0%} of its columns ({int(part.n_columns.iloc[0])}): AUC {part['auc_z1.0'].mean():.4f} at z0=1.0, {part['auc_z1.96'].mean():.4f} at z0=1.96")
    for key, name in ((widths, "widths"), (spectra, "spectrum"), (noises, "noise"), (stacks, "stack"), (fractions, "fraction"), (cands, "candidates")):
        pd.concat(key, ignore_index=True).to_csv(OUT / f"descriptor_ratio_{name}.csv", index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
