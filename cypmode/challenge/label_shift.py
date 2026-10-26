# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""An estimate of the positive rate of the scored test set that does not read a public leaderboard.

The final build takes the positive rate of the scored population from the accuracy, precision and recall that the public boards publish (limitation L13). This module estimates the same
quantity from the training data and from the structures of the blinded file alone, by black-box label-shift estimation (Lipton, Wang and Smola 2018, "Detecting and correcting for label
shift with black box predictors"; the soft version is the probabilistic adjusted classify-and-count of Bella et al. 2010):

    mean score on the test file = pi * m1 + (1 - pi) * m0,    so    pi = (mean test score - m0) / (m1 - m0),

where m1 and m0 are the mean scores of a classifier on held-out positive and negative training compounds. Every fold model scores the held-out compounds and the blinded structures, so
the two sides share one scale. Nothing here reads a board, a label of a blinded compound, or the prevalence that the final build uses; the held-out compounds are weighted only by the
similarity of their nearest-neighbour profile to the blinded set (a function of structures), and the folds keep chemical series whole (Butina clusters at 0.6).

Population. CYP3A4: the compounds with both arms measured, as in the final build. CYP2D6: every compound of the panel, with the labelled hits and the compounds that were never promoted
counted as negatives, because the scored set holds compounds of both kinds.

Scored set. Only confidently labelled compounds are scored. The estimate for all compounds is moved to the scored compounds with the share of confident compounds among positives and among
negatives of the training data (kappa1 and kappa0), a transfer that is itself tested below on held-out chemical series.

Rule, fixed before the first run (2026-10-26):

1. The recovery test comes first. For each isoform the held-out compounds of every fold are split by series into two halves; the first fixes m0, m1, kappa1 and kappa0, and the second is resampled to
   a known positive rate and scored without its labels. The fold results are averaged as in the estimate, over 40 draws. The estimate counts only when, averaged over the synthetic rates, the
   root-mean-square error of the recovered rate of the scored set is at most 0.25 times that rate (RECOVERY_LIMIT).
2. The estimate is the mean of the fold estimates at the primary confidence cut z0 = 1.0 with the first classifier, with a 95% interval from a bootstrap over series (held-out compounds) and over
   blinded compounds.
3. The estimate is informative when the width of its interval is at most 0.5 times the positive rate that the final build uses (INFORMATIVE_WIDTH).
4. The verdict is AGREES when the positive rate of the final build lies inside the interval, DISAGREES when it lies outside, and NOT INFORMATIVE when rule 1 or rule 3 is not met. No verdict
   changes a submitted file; a DISAGREES is reported in the method report and in the limitation that states what depends on the boards.

Usage: python -m cypmode.challenge.label_shift [--bootstrap 1000] [--n-jobs 4]
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
FINAL_MANIFEST = Path("data/tdi_track/outputs/final/manifest.json")

POPULATION = {"CYP3A4": False, "CYP2D6": True}  # True: the whole panel, never-promoted compounds counted as negatives
CLASSIFIERS = ("lgbm_A+fps_deep", "lgbm_A_deep")  # the first is the one of the rule, the second a sensitivity check
CUTS = (1.0, 1.96)
PRIMARY_CUT = 1.0
REPEATS, OUTER, INNER, SEED = 3, 5, 4, 0
GROUP_CUTOFF = 0.6
SYNTHETIC = {"CYP3A4": (0.15, 0.25, 0.35, 0.45), "CYP2D6": (0.03, 0.06, 0.10, 0.20)}
RECOVERY_LIMIT = 0.25
INFORMATIVE_WIDTH = 0.5
MIN_SEPARATION = 0.02  # a fold whose classifier separates the classes by less than this in mean score carries no estimate


@dataclass
class Fold:
    """One held-out fold: scores, labels, weights, confidence and series of its compounds, and the scores of the same fold model on the blinded compounds."""

    score: np.ndarray
    y: np.ndarray
    weight: np.ndarray
    z: np.ndarray
    group: np.ndarray
    test_score: np.ndarray


def class_means(score: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    """Weighted mean score of the negatives and of the positives."""
    out = []
    for cls in (0, 1):
        w = weight[y == cls]
        if w.sum() <= 0:
            return float("nan"), float("nan")
        out.append(float(np.average(score[y == cls], weights=w)))
    return out[0], out[1]


def adjusted_rate(m0: float, m1: float, q: float) -> float:
    """The positive rate that reproduces the mean test score q, clipped to [0, 1]; nan when the classes are not separated."""
    if not np.isfinite(m0 + m1) or m1 - m0 < MIN_SEPARATION:
        return float("nan")
    return float(np.clip((q - m0) / (m1 - m0), 0.0, 1.0))


def confident_shares(y: np.ndarray, weight: np.ndarray, z: np.ndarray, cut: float) -> tuple[float, float]:
    """Weighted share of confident compounds among the positives and among the negatives."""
    shares = []
    for cls in (1, 0):
        w = weight[y == cls]
        shares.append(float(w[z[y == cls] >= cut].sum() / w.sum()) if w.sum() > 0 else float("nan"))
    return shares[0], shares[1]


def to_scored(rate: float, kappa1: float, kappa0: float) -> float:
    """The positive rate among the confident compounds, from the rate among all compounds and the confident share of each class."""
    denominator = rate * kappa1 + (1 - rate) * kappa0
    return float(rate * kappa1 / denominator) if denominator > 0 else float("nan")


def estimates(fold: Fold, test_index: np.ndarray, cut: float, rows: np.ndarray | None = None) -> dict[str, float]:
    """The three estimates of one fold: the rate among all compounds, the rate among the scored compounds by transfer, and by using the confident rows alone."""
    r = np.arange(len(fold.y)) if rows is None else rows
    score, y, w, z = fold.score[r], fold.y[r], fold.weight[r], fold.z[r]
    q = float(fold.test_score[test_index].mean())
    all_rate = adjusted_rate(*class_means(score, y, w), q)
    k1, k0 = confident_shares(y, w, z, cut)
    confident = z >= cut
    direct = adjusted_rate(*class_means(score[confident], y[confident], w[confident]), q) if confident.sum() else float("nan")
    return {"all": all_rate, "scored": to_scored(all_rate, k1, k0) if np.isfinite(all_rate) else float("nan"), "direct": direct}


def summarize(folds: list[Fold], cut: float, bootstrap: int, seed: int = 0) -> dict[str, tuple[float, float, float]]:
    """Mean over the folds of each estimate with a 95% interval from resampling series within every fold and the blinded compounds."""
    rng = np.random.default_rng(seed)
    n_test = len(folds[0].test_score)
    point = [estimates(f, np.arange(n_test), cut) for f in folds]
    index = [{g: np.where(f.group == g)[0] for g in np.unique(f.group)} for f in folds]
    draws: dict[str, list[float]] = {k: [] for k in ("all", "scored", "direct")}
    for _ in range(bootstrap):
        test_index = rng.integers(0, n_test, n_test)
        per_fold = []
        for f, groups in zip(folds, index, strict=True):
            keys = list(groups)
            pick = rng.integers(0, len(keys), len(keys))
            rows = np.concatenate([groups[keys[i]] for i in pick])
            per_fold.append(estimates(f, test_index, cut, rows))
        for k, values in draws.items():
            values.append(float(np.nanmean([e[k] for e in per_fold])))
    out = {}
    for k, values in draws.items():
        lo, hi = np.nanpercentile(values, [2.5, 97.5])
        out[k] = (float(np.nanmean([e[k] for e in point])), float(lo), float(hi))
    return out


def recovery(folds: list[Fold], levels: tuple[float, ...], cut: float, draws: int = 40, seed: int = 1) -> dict[float, dict[str, tuple[float, float]]]:
    """Known rates recovered without labels: per level and route, the bias and the root-mean-square error of the recovered rate among the scored compounds.

    In each draw the held-out compounds of every fold are split by series into two halves. The first fixes the class means and the confident shares. The second is resampled within each class, in proportion to
    its weights, to a known positive rate; the mean score of the resample is the test side. As in the estimate itself, the fold results are averaged; the target is the rate among the confident compounds of the
    resamples.
    """
    rng = np.random.default_rng(seed)
    errors: dict[float, dict[str, list[float]]] = {level: {"scored": [], "direct": []} for level in levels}
    for _ in range(draws):
        for level in levels:
            scored, direct, target = [], [], []
            for f in folds:
                groups = np.unique(f.group)
                rng.shuffle(groups)
                in_a = np.isin(f.group, groups[: len(groups) // 2])
                a, b = np.where(in_a)[0], np.where(~in_a)[0]
                pools = {cls: b[f.y[b] == cls] for cls in (0, 1)}
                if min(len(pools[0]), len(pools[1])) < 5 or len(b) == 0:
                    continue
                m0, m1 = class_means(f.score[a], f.y[a], f.weight[a])
                k1, k0 = confident_shares(f.y[a], f.weight[a], f.z[a], cut)
                confident_a = f.z[a] >= cut
                direct_means = class_means(f.score[a][confident_a], f.y[a][confident_a], f.weight[a][confident_a]) if confident_a.sum() else (float("nan"), float("nan"))
                n1 = max(1, round(level * len(b)))
                rows = np.concatenate([
                    rng.choice(pools[1], n1, p=f.weight[pools[1]] / f.weight[pools[1]].sum()),
                    rng.choice(pools[0], len(b) - n1, p=f.weight[pools[0]] / f.weight[pools[0]].sum()),
                ])
                confident = f.z[rows] >= cut
                if not confident.any():
                    continue
                q = float(f.score[rows].mean())
                rate = adjusted_rate(m0, m1, q)
                scored.append(to_scored(rate, k1, k0) if np.isfinite(rate) else float("nan"))
                direct.append(adjusted_rate(*direct_means, q))
                target.append(float(f.y[rows][confident].mean()))
            if target and np.isfinite(scored).any() and np.isfinite(direct).any():
                errors[level]["scored"].append(float(np.nanmean(scored) - np.mean(target)))
                errors[level]["direct"].append(float(np.nanmean(direct) - np.mean(target)))
    out: dict[float, dict[str, tuple[float, float]]] = {}
    for level, routes in errors.items():
        out[level] = {}
        for route, values in routes.items():
            v = np.array(values, dtype=float)
            out[level][route] = (float(v.mean()), float(np.sqrt((v**2).mean()))) if len(v) else (float("nan"), float("nan"))
    return out


def recovery_ratio(result: dict[float, dict[str, tuple[float, float]]], route: str = "scored") -> float:
    """Mean over the synthetic rates of the root-mean-square error divided by the rate."""
    ratios = [rmse / level for level, routes in result.items() for rmse in [routes[route][1]] if np.isfinite(rmse)]
    return float(np.mean(ratios)) if ratios else float("nan")


def verdict(board: float, interval: tuple[float, float], ratio: float) -> tuple[str, str]:
    """AGREES, DISAGREES or NOT INFORMATIVE, with the reason, by the rule of the module docstring."""
    lo, hi = interval
    if not np.isfinite(ratio) or ratio > RECOVERY_LIMIT:
        return "NOT INFORMATIVE", f"recovery error {ratio:.2f} of the rate is above {RECOVERY_LIMIT}"
    if not np.isfinite(lo + hi) or hi - lo > INFORMATIVE_WIDTH * board:
        return "NOT INFORMATIVE", f"interval width {hi - lo:.3f} is above {INFORMATIVE_WIDTH} times {board:.3f}"
    if lo <= board <= hi:
        return "AGREES", f"{board:.3f} lies inside {lo:.3f} to {hi:.3f}"
    return "DISAGREES", f"{board:.3f} lies outside {lo:.3f} to {hi:.3f}"


def fold_models(isoform: str, n_jobs: int) -> dict[str, list[Fold]]:
    """Fit every classifier on the training part of every outer fold and score the held-out compounds and the blinded compounds with the same model."""
    from cypmode.challenge.nested import OUTER_REFIT, Task
    from cypmode.challenge.nested_runners import (
        FEATURE_SETS,
        build_context_and_plan,
        default_candidates,
        fit_predict,
    )
    from cypmode.challenge.validation_matching import SimilarityMatcher

    blocks = tuple(sorted(set(FEATURE_SETS["A+fps"])))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, isoform, STORE, REPEATS, OUTER, INNER, SEED, n_jobs, blocks, GROUP_CUTOFF, POPULATION[isoform])
    candidates = {c.name: c for c in default_candidates()}
    matcher = SimilarityMatcher(ctx.store.blocks["morgan_bits"], plan.external)
    series = dict(zip(plan.pool.tolist(), plan.groups.tolist(), strict=True))
    folds: dict[str, list[Fold]] = {name: [] for name in CLASSIFIERS}
    for repeat, outer in plan.outer_folds():
        base = plan.task(repeat, outer, OUTER_REFIT)
        task = Task(repeat, outer, OUTER_REFIT, base.train, np.concatenate([base.evaluate, plan.external]))
        held = len(base.evaluate)
        y, z = plan.labels(base.evaluate), ctx.confidence[base.evaluate]
        weight, group = matcher(base.evaluate, base.train), np.array([series[int(r)] for r in base.evaluate])
        for name in CLASSIFIERS:
            scores = fit_predict(candidates[name], ctx, task)
            folds[name].append(Fold(scores[:held], y, weight, z, group, scores[held:]))
            logger.info("%s %s repeat %d fold %d: %d held-out, %d blinded", isoform, name, repeat, outer, held, len(scores) - held)
    return folds


def board_rate(isoform: str) -> float:
    """The positive rate that the final build uses, to be compared with the estimate; it is read here and never enters the estimate."""
    return float(json.loads(FINAL_MANIFEST.read_text())["isoforms"][isoform]["target_prevalence"])


def report(isoform: str, folds: dict[str, list[Fold]], bootstrap: int) -> list[str]:
    board = board_rate(isoform)
    primary = folds[CLASSIFIERS[0]]
    first = primary[0]
    lines = [
        (
            f"LABEL SHIFT {isoform}: {REPEATS * OUTER} held-out folds of series kept whole, {sum(len(f.y) for f in primary)} held-out compounds in all, {len(first.test_score)} blinded compounds, "
            f"classifier {CLASSIFIERS[0]} (sensitivity: {CLASSIFIERS[1]})"
        ),
        f"  training compounds: weighted positive rate {np.average(np.concatenate([f.y for f in primary]), weights=np.concatenate([f.weight for f in primary])):.3f}",
    ]
    outcome, reason = "NOT INFORMATIVE", "the primary cut was not evaluated"
    for cut in CUTS:
        rec = recovery(primary, SYNTHETIC[isoform], cut)
        ratio = recovery_ratio(rec)
        lines.append(f"  z0 = {cut}: recovery of known rates among the scored compounds (bias, root-mean-square error; transfer route | direct route)")
        for level, routes in rec.items():
            lines.append(f"    known rate {level:.2f}: {routes['scored'][0]:+.3f}, {routes['scored'][1]:.3f} | {routes['direct'][0]:+.3f}, {routes['direct'][1]:.3f}")
        lines.append(f"    mean error over the rates, as a share of the rate: {ratio:.2f} (limit {RECOVERY_LIMIT})")
        for name, group in folds.items():
            summary = summarize(group, cut, bootstrap)
            spread = float(np.nanstd([estimates(f, np.arange(len(f.test_score)), cut)["scored"] for f in group]))
            lines.append(
                f"  z0 = {cut} {name}: all compounds {summary['all'][0]:.3f} ({summary['all'][1]:.3f} to {summary['all'][2]:.3f}); scored compounds by transfer {summary['scored'][0]:.3f} "
                f"({summary['scored'][1]:.3f} to {summary['scored'][2]:.3f}), directly {summary['direct'][0]:.3f} ({summary['direct'][1]:.3f} to {summary['direct'][2]:.3f}); spread over folds {spread:.3f}"
            )
            if cut == PRIMARY_CUT and name == CLASSIFIERS[0]:
                outcome, reason = verdict(board, (summary["scored"][1], summary["scored"][2]), ratio)
    lines.append(f"  positive rate used by the final build (read for the comparison only): {board:.3f}")
    lines.append(f"LABEL-SHIFT CHECK {isoform}: {outcome} ({reason})")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--isoform", choices=sorted(POPULATION), nargs="*", default=sorted(POPULATION))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for isoform in args.isoform:
        for line in report(isoform, fold_models(isoform, args.n_jobs), args.bootstrap):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
