# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Selection-honest evaluation: judge the whole *procedure* that picks a model,
not the winner of a contest scored on the same folds that crowned it.

The problem this removes. Choosing among many candidates (feature sets, model
families, hyperparameters, blend weights, a decision threshold) by their
cross-validation score and then quoting that score is optimistic by
construction: the maximum of many noisy estimates is biased upward, and the
more candidates, the worse it gets. Fixing individual constants by hand does
not help -- the selection has to be part of what is evaluated.

How this module does it.

1. A ``NestedPlan`` fixes, once and on disk, every train/validation index set:
   for each repeat and each outer fold, an outer-train/outer-validation split
   and an inner split *of the outer-train rows only*. Nothing about a row in
   an outer-validation set is ever visible to a task in the same outer fold
   other than the outer refit's prediction of it.
2. Any model, in any Python environment (LightGBM, torch, Chemprop, a DeepChem
   env...), fills a ``PredictionCache`` by executing the plan's tasks: "train
   on these rows, predict those rows". The plan is a plain npz and the cache
   plain npy files, so process-isolation constraints (torch vs LightGBM,
   tblite vs LightGBM) do not matter.
3. A ``Selector`` sees only the inner out-of-fold predictions of the
   outer-training rows -- it picks a candidate, a top-k blend or stacking
   weights, and an operating threshold from those alone -- and ``nested_evaluate``
   scores that frozen choice on the outer-validation rows. The gap between the
   selector's own inner score and the outer score is reported as its optimism.

Scoring is similarity- and prior-matched to the blinded test set through
cypmode.challenge.validation_matching (all derived from data, no constants).
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from cypmode import provenance
from cypmode.challenge.validation_matching import prior_weights, weighted_mcc

OUTER_REFIT = -1  # fit on the whole outer-training set, predict the outer-validation set
DEPLOY = -2  # fit on the whole pool, predict the external (blinded test) rows


@dataclass(frozen=True)
class Task:
    """Train a model on ``train``, return one score per row of ``evaluate`` (same order)."""

    repeat: int
    outer: int
    inner: int  # >= 0: inner fold; OUTER_REFIT; DEPLOY
    train: np.ndarray
    evaluate: np.ndarray

    @property
    def key(self) -> str:
        return f"r{self.repeat}_k{self.outer}_m{self.inner}"


class NestedPlan:
    """Repeated K-fold outer splits, each with an M-fold inner split of its training rows.

    A *fold* is one of K disjoint subsets of the rows. In K-fold validation each fold takes a turn as
    the held-out set while the model is fitted on the other K-1 folds, so every row is predicted once
    by a model that never saw it. Here that happens at two levels: an outer fold (1/K of the pool) is
    kept for scoring and never used for any choice, and the remaining rows are themselves cut into M
    inner folds that supply the out-of-fold predictions on which candidates, blend weights and the
    threshold are chosen. A *repeat* redraws the whole partition with a new seed.

    ``pool`` holds the global row ids that carry a usable label, ``y`` their 0/1 labels (aligned).
    Splits are stratified on ``y``. With ``groups`` (one id per pool row, e.g. a chemical cluster or
    scaffold) every group stays whole -- all its members land in the same fold at both levels -- which
    asks the harder question of how a pipeline does on chemical series it has never seen.
    ``external`` (optional) are row ids to predict at deployment time (DEPLOY tasks use the whole pool
    as training data and an inner split of the whole pool for selection).

    The plan identity does NOT include ``repeats``: repeat r's partition depends only on (seed, r), so
    asking for more repeats later reuses every cached prediction of the earlier ones and computes only
    the new tasks.
    """

    def __init__(
        self,
        pool: Sequence[int],
        y: Sequence[int],
        repeats: int,
        outer_splits: int,
        inner_splits: int,
        seed: int,
        external: Sequence[int] | None = None,
        groups: Sequence[int] | None = None,
    ):
        self.pool = np.asarray(pool, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        if len(self.pool) != len(self.y):
            raise ValueError("pool and y must be aligned")
        self.groups = None if groups is None else np.asarray(groups, dtype=np.int64)
        if self.groups is not None and len(self.groups) != len(self.pool):
            raise ValueError("groups must have one entry per pool row")
        self._group = None if self.groups is None else dict(zip(self.pool.tolist(), self.groups.tolist(), strict=True))
        if len(set(self.y.tolist())) != 2:
            raise ValueError("y must contain both classes")
        self.repeats, self.outer_splits, self.inner_splits, self.seed = repeats, outer_splits, inner_splits, seed
        self.external = None if external is None else np.asarray(external, dtype=np.int64)
        self._label = dict(zip(self.pool.tolist(), self.y.tolist(), strict=True))
        self._tasks = self._build()

    # -- identity ---------------------------------------------------------------------------------

    @property
    def plan_id(self) -> str:
        digest = hashlib.sha1()
        digest.update(self.pool.tobytes())
        digest.update(self.y.tobytes())
        digest.update(json.dumps([self.outer_splits, self.inner_splits, self.seed]).encode())  # not `repeats`: see class docstring
        if self.external is not None:
            digest.update(self.external.tobytes())
        if self.groups is not None:
            digest.update(b"groups")
            digest.update(self.groups.tobytes())
        return digest.hexdigest()[:16]

    def labels(self, rows: np.ndarray) -> np.ndarray:
        return np.array([self._label[int(r)] for r in rows], dtype=np.int64)

    # -- construction -----------------------------------------------------------------------------

    def _split(self, rows: np.ndarray, n_splits: int, random_state: int):
        """Index pairs into ``rows``: stratified on the label, and group-preserving when groups are set."""
        labels = self.labels(rows)
        if self._group is None:
            return list(StratifiedKFold(n_splits, shuffle=True, random_state=random_state).split(rows, labels))
        groups = np.array([self._group[int(r)] for r in rows])
        return list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state).split(rows, labels, groups))

    def _inner_folds(self, rows: np.ndarray, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(rows[a], rows[b]) for a, b in self._split(rows, self.inner_splits, random_state)]

    def _build(self) -> list[Task]:
        tasks = []
        for repeat in range(self.repeats):
            for k, (tr, va) in enumerate(self._split(self.pool, self.outer_splits, self.seed + repeat)):
                train_rows, val_rows = self.pool[tr], self.pool[va]
                for m, (itr, iva) in enumerate(self._inner_folds(train_rows, self.seed * 7919 + repeat * 101 + k)):
                    tasks.append(Task(repeat, k, m, itr, iva))
                tasks.append(Task(repeat, k, OUTER_REFIT, train_rows, val_rows))
        if self.external is not None:
            for m, (itr, iva) in enumerate(self._inner_folds(self.pool, self.seed * 7919 + 99991)):
                tasks.append(Task(-1, 0, m, itr, iva))
            tasks.append(Task(-1, 0, DEPLOY, self.pool, self.external))
        return tasks

    # -- access -----------------------------------------------------------------------------------

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def outer_folds(self) -> list[tuple[int, int]]:
        return sorted({(t.repeat, t.outer) for t in self._tasks if t.repeat >= 0})

    def task(self, repeat: int, outer: int, inner: int) -> Task:
        for t in self._tasks:
            if (t.repeat, t.outer, t.inner) == (repeat, outer, inner):
                return t
        raise KeyError((repeat, outer, inner))

    def inner_tasks(self, repeat: int, outer: int) -> list[Task]:
        return [t for t in self._tasks if (t.repeat, t.outer) == (repeat, outer) and t.inner >= 0]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            pool=self.pool,
            y=self.y,
            external=self.external if self.external is not None else np.array([], dtype=np.int64),
            params=np.array([self.repeats, self.outer_splits, self.inner_splits, self.seed]),
            has_external=np.array([self.external is not None]),
            groups=self.groups if self.groups is not None else np.array([], dtype=np.int64),
            has_groups=np.array([self.groups is not None]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NestedPlan":
        z = np.load(path)
        repeats, outer, inner, seed = (int(v) for v in z["params"])
        external = z["external"] if bool(z["has_external"][0]) else None
        groups = z["groups"] if "has_groups" in z.files and bool(z["has_groups"][0]) else None
        return cls(z["pool"], z["y"], repeats, outer, inner, seed, external, groups)


class PredictionCache:
    """One .npy per (plan, candidate, task); the interface every runner writes to."""

    def __init__(self, root: str | Path, plan: NestedPlan):
        self.root, self.plan = Path(root), plan

    def path(self, candidate: str, task: Task) -> Path:
        return self.root / self.plan.plan_id / candidate / f"{task.key}.npy"

    def has(self, candidate: str, task: Task) -> bool:
        return self.path(candidate, task).exists()

    def put(self, candidate: str, task: Task, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != (len(task.evaluate),):
            raise ValueError(f"expected {len(task.evaluate)} scores for {candidate}/{task.key}, got {scores.shape}")
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite scores for {candidate}/{task.key}")
        path = self.path(candidate, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, scores)
        provenance.stamp(self.root)  # the author, the repository and the licence beside the predictions (written once, nothing is rewritten)

    def get(self, candidate: str, task: Task) -> np.ndarray:
        return np.load(self.path(candidate, task))

    def missing(self, candidate: str, include_deploy: bool = True) -> list[Task]:
        return [t for t in self.plan.tasks() if (include_deploy or t.inner != DEPLOY) and not self.has(candidate, t)]

    def candidates(self) -> list[str]:
        base = self.root / self.plan.plan_id
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []


# -- inner out-of-fold matrices ------------------------------------------------------------------


def inner_oof(cache: PredictionCache, candidates: Sequence[str], repeat: int, outer: int) -> tuple[np.ndarray, np.ndarray]:
    """(row ids, rows x candidates matrix) of the inner out-of-fold predictions for one outer
    fold. Every outer-training row appears exactly once (a missing inner task is an error).
    """
    tasks = cache.plan.inner_tasks(repeat, outer)
    rows = np.concatenate([t.evaluate for t in tasks])
    matrix = np.column_stack([np.concatenate([cache.get(c, t) for t in tasks]) for c in candidates])
    if len(set(rows.tolist())) != len(rows):
        raise AssertionError("inner folds overlap")
    return rows, matrix


# -- scoring -------------------------------------------------------------------------------------


def score_metric(y: np.ndarray, score: np.ndarray, weights: np.ndarray | None, metric: str = "auc") -> float:
    if weights is not None and (weights[y == 1].sum() <= 0 or weights[y == 0].sum() <= 0):
        return 0.5 if metric == "auc" else float(np.mean(y))  # a class has no weight: nothing to rank
    if metric == "auc":
        return float(roc_auc_score(y, score, sample_weight=weights))
    if metric == "ap":
        return float(average_precision_score(y, score, sample_weight=weights))
    raise ValueError(f"unknown metric {metric!r}")


def best_threshold(y: np.ndarray, score: np.ndarray, weights: np.ndarray | None, grid: int = 60, n_boot: int = 40, seed: int = 0) -> float:
    """Score threshold maximizing weighted MCC over quantiles of ``score``. The MCC curve of a weak
    ranker is jagged, so the curve is averaged over bootstrap resamples before its maximum is taken
    -- an unsmoothed argmax chases noise (and ``n_boot=0`` restores it).
    """
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, grid)))
    rng = np.random.default_rng(seed)
    samples = [np.arange(len(y))] + [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    curve = np.zeros(len(candidates))
    for idx in samples:
        yy, ss = y[idx], score[idx]
        ww = None if weights is None else weights[idx]
        curve += [weighted_mcc(yy, (ss >= t).astype(int), ww) for t in candidates]
    return float(candidates[int(np.argmax(curve))])


def positive_rate(score: np.ndarray, threshold: float, weights: np.ndarray | None = None) -> float:
    """(Weighted) fraction of rows called positive at ``threshold``."""
    called = (np.asarray(score) >= threshold).astype(float)
    return float(np.average(called, weights=weights))


def call_top_fraction(score: np.ndarray, rate: float) -> np.ndarray:
    """Call the highest-scoring ``rate`` fraction positive. Transferring a RATE instead of a score
    threshold survives a shift in the score distribution (validation compounds that sit closer to
    the training set than the inner folds did get uniformly higher scores).
    """
    k = int(np.clip(round(rate * len(score)), 1, len(score)))
    cutoff = np.sort(score)[::-1][k - 1]
    return np.asarray(score) >= cutoff


class _EmpiricalCdf:
    """Maps a candidate's raw scores onto [0, 1] using the inner out-of-fold scores only, so
    candidates on different scales (probabilities, margins, regressions) can be blended and
    outer-validation scores land on the same scale the selector was fit on.
    """

    def __init__(self, reference: np.ndarray):
        self.sorted = np.sort(reference)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.searchsorted(self.sorted, x, side="right") + 0.5) / (len(self.sorted) + 1.0)


# -- selectors -----------------------------------------------------------------------------------


@dataclass
class Selection:
    """What a selector froze after seeing inner out-of-fold data only."""

    predict: Callable[[np.ndarray], np.ndarray]  # (n x candidates matrix) -> scores
    threshold: float
    inner_score: float  # the selector's own (optimistic) estimate of its quality
    chosen: dict[str, float]  # candidate -> weight, for reporting
    inner_scores: np.ndarray | None = None  # the (out-of-fold) scores the threshold was chosen on
    inner_weights: np.ndarray | None = None

    def rate(self) -> float:
        """The weighted positive rate the operating threshold implies on the inner out-of-fold data."""
        return positive_rate(self.inner_scores, self.threshold, self.inner_weights)


class Selector:
    name = "selector"

    def fit(self, candidates: Sequence[str], oof: np.ndarray, y: np.ndarray, weights: np.ndarray | None, metric: str) -> Selection:
        raise NotImplementedError


class BestSingle(Selector):
    """The classic contest winner: the candidate with the best inner score."""

    name = "best_single"

    def fit(self, candidates, oof, y, weights, metric):
        scores = np.array([score_metric(y, oof[:, j], weights, metric) for j in range(oof.shape[1])])
        j = int(np.argmax(scores))
        return Selection(lambda m, j=j: m[:, j], best_threshold(y, oof[:, j], weights), float(scores[j]), {candidates[j]: 1.0}, oof[:, j], weights)


class OneStandardError(Selector):
    """Among candidates within one bootstrap standard error of the best, the one listed
    earliest in ``candidates`` (callers order candidates from simplest to most complex) --
    a tie-break that resists picking a noise-driven winner.
    """

    name = "one_se"

    def __init__(self, n_boot: int = 100, seed: int = 0):
        self.n_boot, self.seed = n_boot, seed

    def fit(self, candidates, oof, y, weights, metric):
        scores = np.array([score_metric(y, oof[:, j], weights, metric) for j in range(oof.shape[1])])
        best = int(np.argmax(scores))
        rng = np.random.default_rng(self.seed)
        boot = []
        for _ in range(self.n_boot):
            ii = rng.integers(0, len(y), len(y))
            if len(set(y[ii].tolist())) < 2:
                continue
            boot.append(score_metric(y[ii], oof[ii, best], None if weights is None else weights[ii], metric))
        se = float(np.std(boot)) if boot else 0.0
        j = int(next(i for i in range(len(scores)) if scores[i] >= scores[best] - se))
        return Selection(lambda m, j=j: m[:, j], best_threshold(y, oof[:, j], weights), float(scores[j]), {candidates[j]: 1.0}, oof[:, j], weights)


class MeanOfTop(Selector):
    """Average of the empirical-CDF scores of the k best candidates."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"mean_top{k}"

    def fit(self, candidates, oof, y, weights, metric):
        scores = np.array([score_metric(y, oof[:, j], weights, metric) for j in range(oof.shape[1])])
        top = np.argsort(-scores)[: min(self.k, len(scores))]
        cdfs = {int(j): _EmpiricalCdf(oof[:, j]) for j in top}

        def predict(m):
            return np.mean([cdfs[j](m[:, j]) for j in cdfs], axis=0)

        blended = predict(oof)
        return Selection(
            predict,
            best_threshold(y, blended, weights),
            score_metric(y, blended, weights, metric),
            {candidates[j]: 1.0 / len(top) for j in top},
            blended,
            weights,
        )


META_TOL, META_MAX_ITER = 1e-9, 20000  # convergence settings of the stack's logistic meta-learner (see StackedLogit._model)


class StackedLogit(Selector):
    """Logistic blend of every candidate's empirical-CDF score, with the penalty type
    (L1 for a sparse blend, L2 for a smooth one) and its strength chosen by an inner
    cross-validation over the blend itself -- never touching outer rows.

    L1 matters: with dozens of candidates and a few thousand rows, an L2 blend spreads
    weight over candidates that are pure noise and loses to simply picking the best one
    (the unit tests demonstrate this); letting the data pick sparsity removes the need for
    a hand-chosen "top k". The threshold comes from the blend's cross-validated
    predictions rather than its in-sample ones.
    """

    name = "stack_logit"

    def __init__(self, c_grid: Sequence[float] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0), penalties: Sequence[str] = ("l1", "l2"), meta_folds: int = 5, seed: int = 0):
        self.c_grid, self.penalties, self.meta_folds, self.seed = tuple(c_grid), tuple(penalties), meta_folds, seed

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    @staticmethod
    def _model(penalty: str, c: float, seed: int = 0) -> LogisticRegression:
        # liblinear visits the coordinates in a random order. At its default tolerance (1e-4) the stopping point depends on that order, so
        # the stack changed from process to process (CYP2D6 outer AUC 0.7527 or 0.7536, MCC 0.191 or 0.204); the problem is convex, so
        # a tight tolerance gives every seed the same optimum, and the seed is fixed on top of that.
        # scikit-learn 1.8 replaced `penalty` by `l1_ratio` (0 = L2, 1 = L1) and warns on the old spelling
        settings = {"C": c, "solver": "liblinear", "max_iter": META_MAX_ITER, "tol": META_TOL, "random_state": seed}
        if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8):
            return LogisticRegression(l1_ratio=1.0 if penalty == "l1" else 0.0, **settings)
        return LogisticRegression(penalty=penalty, **settings)

    def fit(self, candidates, oof, y, weights, metric):
        cdfs = [_EmpiricalCdf(oof[:, j]) for j in range(oof.shape[1])]

        def features(m):
            return np.column_stack([self._logit(cdfs[j](m[:, j])) for j in range(m.shape[1])])

        z = features(oof)
        folds = list(StratifiedKFold(self.meta_folds, shuffle=True, random_state=self.seed).split(z, y))
        best = None
        for penalty in self.penalties:
            for c in self.c_grid:
                meta = np.zeros(len(y))
                for a, b in folds:
                    lr = self._model(penalty, c, self.seed).fit(z[a], y[a], sample_weight=None if weights is None else weights[a])
                    meta[b] = lr.decision_function(z[b])
                value = score_metric(y, meta, weights, metric)
                if best is None or value > best[0] + 1e-9:  # ties keep the earlier (sparser-first) setting
                    best = (value, penalty, c, meta)
        value, penalty, c, meta = best
        lr = self._model(penalty, c, self.seed).fit(z, y, sample_weight=weights)

        def predict(m):
            return lr.decision_function(features(m))

        coef = lr.coef_[0]
        chosen = {candidates[j]: float(coef[j]) for j in np.argsort(-np.abs(coef))[:5] if coef[j] != 0}
        return Selection(predict, best_threshold(y, meta, weights), value, chosen, meta, weights)


# -- nested evaluation ---------------------------------------------------------------------------


def _weights(y, eval_rows, reference_rows, weight_fn, target_prevalence):
    w = None if weight_fn is None else weight_fn(eval_rows, reference_rows)
    if target_prevalence is not None:
        pw = prior_weights(y, target_prevalence)
        w = pw if w is None else w * pw
    return w


def nested_evaluate(
    cache: PredictionCache,
    candidates: Sequence[str],
    selectors: Sequence[Selector],
    weight_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | None] | None = None,
    target_prevalence: float | None = None,
    metric: str = "auc",
) -> pd.DataFrame:
    """One row per (selector, repeat, outer fold): the selector's inner estimate, the
    score of its frozen choice on the outer-validation rows, and the optimism between them.

    ``weight_fn(rows, reference_rows)`` returns per-row weights for scoring ``rows``
    against a training reference (e.g. similarity weights to the blinded test set);
    ``target_prevalence`` additionally shifts the class prior. Both are optional so the
    module stays usable on synthetic data.
    """
    plan = cache.plan
    out = []
    for repeat, k in plan.outer_folds():
        rows, oof = inner_oof(cache, candidates, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        outer_scores = np.column_stack([cache.get(c, refit) for c in candidates])
        y_in, y_out = plan.labels(rows), plan.labels(refit.evaluate)

        w_in = _weights(y_in, rows, refit.train, weight_fn, target_prevalence)
        w_out = _weights(y_out, refit.evaluate, refit.train, weight_fn, target_prevalence)
        for selector in selectors:
            chosen = selector.fit(list(candidates), oof, y_in, w_in, metric)
            scores = chosen.predict(outer_scores)
            outer_value = score_metric(y_out, scores, w_out, metric)
            out.append(
                {
                    "selector": selector.name,
                    "repeat": repeat,
                    "outer": k,
                    "inner_score": chosen.inner_score,
                    "outer_score": outer_value,
                    "optimism": chosen.inner_score - outer_value,
                    "outer_mcc": weighted_mcc(y_out, (scores >= chosen.threshold).astype(int), w_out),
                    "outer_mcc_rate": weighted_mcc(y_out, call_top_fraction(scores, chosen.rate()).astype(int), w_out),
                    "called_rate_threshold": float(np.mean(scores >= chosen.threshold)),
                    "called_rate_transfer": chosen.rate(),
                    "chosen": json.dumps({a: round(b, 4) for a, b in chosen.chosen.items()}),
                }
            )
    return pd.DataFrame(out)


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby("selector")[["inner_score", "outer_score", "optimism", "outer_mcc", "outer_mcc_rate"]]
        .mean()
        .sort_values("outer_score", ascending=False)
    )


@dataclass
class Deployment:
    rows: np.ndarray  # external row ids, in the order of ``scores``
    scores: np.ndarray
    threshold: float
    selection: Selection

    @property
    def calls_by_threshold(self) -> np.ndarray:
        return self.scores >= self.threshold

    @property
    def calls(self) -> np.ndarray:
        """Rate transfer: the positive rate chosen on the inner data, applied to the external scores."""
        return call_top_fraction(self.scores, self.selection.rate())


def deploy(
    cache: PredictionCache,
    candidates: Sequence[str],
    selector: Selector,
    weight_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | None] | None = None,
    target_prevalence: float | None = None,
    metric: str = "auc",
) -> Deployment:
    """Run the SAME selection procedure that was evaluated by ``nested_evaluate`` on the whole
    pool (its inner out-of-fold predictions) and apply the frozen choice to the external rows.
    The quality to expect is the nested estimate, not the selector's inner score.
    """
    plan = cache.plan
    rows, oof = inner_oof(cache, candidates, -1, 0)
    task = plan.task(-1, 0, DEPLOY)
    y = plan.labels(rows)
    w = _weights(y, rows, task.train, weight_fn, target_prevalence)
    chosen = selector.fit(list(candidates), oof, y, w, metric)
    matrix = np.column_stack([cache.get(c, task) for c in candidates])
    return Deployment(task.evaluate, chosen.predict(matrix), chosen.threshold, chosen)
