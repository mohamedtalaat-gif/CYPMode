"""A softmax-temperature selector between exploiting the best model and exploring a committee, and the pre-registered test of whether it should replace the logistic blend.

Every selector of the nested harness turns the inner out-of-fold scores of the candidates into one score. Two of them sit at the ends of a line: ``BestSingle`` puts all the weight on
the candidate with the best inner score (pure exploitation of what the inner folds say), an equal-weight committee spreads it evenly (pure exploration: no candidate is trusted over another).
``SoftmaxBlend`` fills the line between them with one parameter, the temperature T of a Boltzmann (softmax) distribution over the candidates:

    w_k = exp(s_k / T) / sum_j exp(s_j / T),        blend(x) = sum_k w_k F_k(x_k),

with s_k the inner score of candidate k (the weighted AUC on its inner out-of-fold predictions) and F_k its empirical CDF. T -> 0 is the best single model, T -> infinity the equal-weight
committee. T is chosen from a grid by an inner cross-validation over the blend itself: for each inner fold the scores s_k and the CDFs are computed on the other folds only, so no outer row and no
row of the evaluated fold touches the choice. The threshold comes from the cross-validated blend, as for ``StackedLogit``.

Two outputs. (1) The exploration-exploitation curve: the outer-fold AUC of the blend at every fixed temperature of the grid, written to ``data/tdi_track/outputs/comparisons/softmax_curve_<isoform>.csv``
and drawn by the figure ``ml_exploration_exploitation``. (2) The pre-registered test (``python -m cypmode.challenge.softmax_selector --isoform CYP3A4``): the same cached candidates of the default
stack, the same nested folds and yardstick (similarity x prevalence x confident rows, z0 in {1.0, 1.96}); the paired difference of the outer-fold AUC between this selector and ``StackedLogit`` is judged
with the Nadeau-Bengio corrected resampled t. GATE: the softmax blend beats the logistic blend with corrected p < 0.05 at both z0 (and a positive mean difference). Otherwise the logistic blend stays
the final selector; the curve is reported either way, because it shows how flat the exploration-exploitation trade-off is on these candidates.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes
from cypmode.challenge.nested import NestedPlan, PredictionCache, Selection, Selector, StackedLogit, _EmpiricalCdf, best_threshold, nested_evaluate, score_metric
from cypmode.challenge.nested_runners import build_context_and_plan, default_candidates
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
TEMPERATURES = (0.0005, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, np.inf)  # in units of AUC: the spread of the inner scores of the candidates is a few hundredths
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
NESTED_ROOT = "data/tdi_track/outputs/nested"
CURVE = "data/tdi_track/outputs/comparisons/softmax_curve_{iso}.csv"


def softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    """exp(s / T) normalised; T = 0 gives the best candidate alone and T = infinity the equal weights."""
    scores = np.asarray(scores, dtype=float)
    if np.isinf(temperature):
        return np.full(len(scores), 1.0 / len(scores))
    z = (scores - scores.max()) / max(temperature, 1e-12)
    w = np.exp(z)
    return w / w.sum()


def blend(oof: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted mean of the empirical-CDF scores of the columns of ``oof``, each CDF fitted on the matching column of ``reference``."""
    return sum(w * _EmpiricalCdf(reference[:, j])(oof[:, j]) for j, w in enumerate(weights) if w > 0)


class SoftmaxBlend(Selector):
    """Softmax over the inner scores of the candidates, the temperature chosen by an inner cross-validation (see the module docstring)."""

    name = "softmax"

    def __init__(self, temperatures: Sequence[float] = TEMPERATURES, folds: int = 5, seed: int = 0):
        self.temperatures, self.folds, self.seed = tuple(temperatures), folds, seed

    def _inner_scores(self, oof: np.ndarray, y: np.ndarray, weights: np.ndarray | None, metric: str) -> np.ndarray:
        return np.array([score_metric(y, oof[:, j], weights, metric) for j in range(oof.shape[1])])

    def fit(self, candidates, oof, y, weights, metric):
        folds = list(StratifiedKFold(self.folds, shuffle=True, random_state=self.seed).split(oof, y))
        best = None
        for temperature in self.temperatures:  # ascending: a tie keeps the smaller temperature, the more exploiting setting
            cv = np.zeros(len(y))
            for a, b in folds:
                s = self._inner_scores(oof[a], y[a], None if weights is None else weights[a], metric)
                cv[b] = blend(oof[b], oof[a], softmax_weights(s, temperature))
            value = score_metric(y, cv, weights, metric)
            if best is None or value > best[0] + 1e-9:
                best = (value, temperature, cv)
        value, temperature, cv = best
        w = softmax_weights(self._inner_scores(oof, y, weights, metric), temperature)
        chosen = {candidates[j]: float(w[j]) for j in np.argsort(-w)[:5] if w[j] > 0}
        selection = Selection(lambda m, w=w, ref=oof: blend(m, ref, w), best_threshold(y, cv, weights), value, chosen, cv, weights)
        selection.temperature = float(temperature)
        return selection


class FixedTemperature(SoftmaxBlend):
    """The same blend at one temperature of the grid, for the curve (no inner choice of T)."""

    def __init__(self, temperature: float, folds: int = 5, seed: int = 0):
        super().__init__((temperature,), folds, seed)
        self.name = f"softmax_T{temperature:g}"


def curve(cache: PredictionCache, names: Sequence[str], weight_fn, prevalence: float) -> pd.DataFrame:
    """Outer-fold AUC and MCC at the transferred rate of the blend at every fixed temperature, and of the logistic blend, one row per outer fold."""
    selectors = [FixedTemperature(t) for t in TEMPERATURES] + [SoftmaxBlend(), StackedLogit()]
    return nested_evaluate(cache, list(names), selectors, weight_fn, prevalence, "auc")


def compare(cache: PredictionCache, names: Sequence[str], weight_fn, prevalence: float, n_sizes: tuple[int, int]) -> dict:
    """The nested softmax blend against the logistic blend: outer AUC and MCC of both, the corrected paired test and the temperatures that the inner choice picked."""
    table = nested_evaluate(cache, list(names), [StackedLogit(), SoftmaxBlend()], weight_fn, prevalence, "auc")
    wide = table.pivot_table(index=["repeat", "outer"], columns="selector", values=["outer_score", "outer_mcc_rate"])
    difference = (wide["outer_score"]["softmax"] - wide["outer_score"]["stack_logit"]).to_numpy()
    test = corrected_resampled_t(difference, *n_sizes)
    return {
        "auc_logit": float(wide["outer_score"]["stack_logit"].mean()), "auc_softmax": float(wide["outer_score"]["softmax"].mean()),
        "mcc_logit": float(wide["outer_mcc_rate"]["stack_logit"].mean()), "mcc_softmax": float(wide["outer_mcc_rate"]["softmax"].mean()),
        "difference": test.mean_difference, "p": test.p_value, "better": test.n_better, "folds": test.n_folds, "met": bool(test.p_value < 0.05 and test.mean_difference > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    root = f"{NESTED_ROOT}/{iso}"
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, args.repeats, 5, 4, 0, args.n_jobs, ("morgan_c2", "rdkit_desc"))
    assert NestedPlan.load(f"{root}/plan.npz").plan_id == plan.plan_id, "the plan changed: this test must share the existing caches"
    cache = PredictionCache(f"{root}/cache", plan)
    names = [c.name for c in default_candidates() if not cache.missing(c.name, False)]
    bits = FeatureStore.load_or_build(STORE, list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"]
    sizes = plan_sizes(plan)
    print(f"\n{iso}: {len(plan.outer_folds())} outer folds, {len(names)} candidates of the default stack")
    met, rows = [], []
    for z0 in Z_CUTS:
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        r = compare(cache, names, weight_fn, prevalence, sizes)
        met.append(r["met"])
        print(f"  z0={z0:<4} outer AUC: logistic blend {r['auc_logit']:.4f}, softmax blend {r['auc_softmax']:.4f} ({r['difference']:+.4f}, p={r['p']:.3f}, better {r['better']}/{r['folds']}) | "
              f"MCC at the transferred rate {r['mcc_logit']:.4f} against {r['mcc_softmax']:.4f}")
        table = curve(cache, names, weight_fn, prevalence)
        for selector, part in table.groupby("selector"):
            rows.append({"z0": z0, "selector": selector, "outer_auc": part["outer_score"].mean(), "outer_auc_sd": part["outer_score"].std(ddof=1), "outer_mcc": part["outer_mcc_rate"].mean(), "folds": len(part)})
    out = Path(CURVE.format(iso=iso))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"curve written to {out}")
    print(f"{iso} GATE (selector, corrected p < 0.05 at both z0): {'MET' if all(met) else 'NOT MET'}")


if __name__ == "__main__":
    main()
