# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""A non-negative stacking selector, and the pre-registered test of whether it should replace the logistic blend.

``StackedLogit`` blends the empirical-CDF logit of every candidate with a logistic regression whose penalty (L1 or L2) and strength are chosen by an inner
cross-validation. Nothing stops it from giving a candidate a negative weight; in the final CYP3A4 stack ``knn_tanimoto_k5`` has weight -0.093, a candidate
that is subtracted because it is correlated with better ones, which is a fit to the noise of the inner folds more often than a property of the chemistry.
``NonNegativeStack`` is the same blend with every candidate weight constrained to be at least zero (the intercept is free): an average of experts that can be
switched off but never inverted, the form that representation-diverse stacks in this challenge have used. The constrained problem

    minimise  sum_i w_i log(1 + exp(-y_i (b + z_i . a)))  +  (1 / C) R(a),   a >= 0,

with R the sum (an L1 penalty on a non-negative vector) or the sum of squares, is solved with L-BFGS-B; the penalty and C are chosen by the same inner
cross-validation as in ``StackedLogit``, so the two selectors differ in nothing else.

Test (``python -m cypmode.challenge.nonneg_stack --isoform CYP3A4``): both selectors on the same cached candidates (the default stack the final build uses),
the same nested folds and yardstick (similarity x prevalence x confident rows, z0 in {1.0, 1.96}); the paired difference of the outer-fold AUC is judged with
the Nadeau-Bengio corrected resampled t. GATE: NonNegativeStack beats StackedLogit with corrected p < 0.05 at both z0 (and a positive mean difference). Otherwise
the logistic blend stays the final selector; the mean outer MCC at the transferred rate and the number of negative weights removed are reported either way.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes
from cypmode.challenge.nested import (
    NestedPlan,
    PredictionCache,
    StackedLogit,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import build_context_and_plan, default_candidates
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
NESTED_ROOT = "data/tdi_track/outputs/nested"


class _NonNegativeLogit:
    """Logistic regression with non-negative coefficients and a free intercept (the interface of the scikit-learn estimator the blend needs)."""

    def __init__(self, penalty: str, c: float):
        self.penalty, self.c = penalty, c
        self.coef_ = np.zeros((1, 0))
        self.intercept_ = np.zeros(1)

    def fit(self, z: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> _NonNegativeLogit:
        n, k = z.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w / w.mean()
        sign = 2.0 * np.asarray(y, dtype=float) - 1.0

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            b, a = theta[0], theta[1:]
            margin = sign * (b + z @ a)
            loss = np.sum(w * np.logaddexp(0.0, -margin))
            d = -w * sign / (1.0 + np.exp(np.clip(margin, -50, 50)))
            grad_a = z.T @ d
            if self.penalty == "l1":
                loss += a.sum() / self.c
                grad_a = grad_a + 1.0 / self.c
            else:
                loss += 0.5 * np.dot(a, a) / self.c
                grad_a = grad_a + a / self.c
            return float(loss), np.concatenate([[d.sum()], grad_a])

        start = np.zeros(k + 1)
        bounds = [(None, None)] + [(0.0, None)] * k
        fitted = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
        self.intercept_ = np.array([fitted.x[0]])
        self.coef_ = fitted.x[1:].reshape(1, -1)
        return self

    def decision_function(self, z: np.ndarray) -> np.ndarray:
        return z @ self.coef_[0] + self.intercept_[0]


class NonNegativeStack(StackedLogit):
    """The logistic blend of ``StackedLogit`` with every candidate weight constrained to be non-negative."""

    name = "stack_nonneg"

    @staticmethod
    def _model(penalty: str, c: float, seed: int = 0) -> _NonNegativeLogit:
        return _NonNegativeLogit(penalty, c)


def negative_weight_count(selector_weights: dict[str, float]) -> int:
    return sum(v < 0 for v in selector_weights.values())


def compare_selectors(cache: PredictionCache, names: Sequence[str], weight_fn, prevalence: float, n_sizes: tuple[int, int]) -> dict:
    """Outer-fold AUC and MCC (at the transferred rate) of both selectors on the same candidates, and the corrected paired test of the AUC difference."""
    table = nested_evaluate(cache, list(names), [StackedLogit(), NonNegativeStack()], weight_fn, prevalence, "auc")
    wide = table.pivot_table(index=["repeat", "outer"], columns="selector", values=["outer_score", "outer_mcc_rate"])
    difference = (wide["outer_score"]["stack_nonneg"] - wide["outer_score"]["stack_logit"]).to_numpy()
    test = corrected_resampled_t(difference, *n_sizes)
    return {
        "auc_logit": float(wide["outer_score"]["stack_logit"].mean()),
        "auc_nonneg": float(wide["outer_score"]["stack_nonneg"].mean()),
        "mcc_logit": float(wide["outer_mcc_rate"]["stack_logit"].mean()),
        "mcc_nonneg": float(wide["outer_mcc_rate"]["stack_nonneg"].mean()),
        "difference": test.mean_difference,
        "p": test.p_value,
        "better": test.n_better,
        "folds": test.n_folds,
        "met": bool(test.p_value < 0.05 and test.mean_difference > 0),
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
    met = []
    for z0 in Z_CUTS:
        r = compare_selectors(cache, names, confident_weights(matcher, ctx.confidence, z0), prevalence, sizes)
        met.append(r["met"])
        print(f"  z0={z0:<4} outer AUC: logistic blend {r['auc_logit']:.4f}, non-negative blend {r['auc_nonneg']:.4f} ({r['difference']:+.4f}, p={r['p']:.3f}, better {r['better']}/{r['folds']}) | MCC at the transferred rate {r['mcc_logit']:+.4f} vs {r['mcc_nonneg']:+.4f}")
    print(f"{iso} GATE (selector, corrected p < 0.05 at both z0): {'MET' if all(met) else 'NOT MET'}")


if __name__ == "__main__":
    main()
