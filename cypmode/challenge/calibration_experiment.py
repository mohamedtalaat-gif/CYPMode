# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Does a post-hoc calibration step make the stack's probabilities usable? (job 030, gate fixed in advance)

The audit found the CYP2D6 stack's probabilities almost constant (80% of compounds at 0.49-0.50, calibration slope
-0.02): the regularised logistic blend ranks but does not output probabilities. Any probability-based use of the model
(expected improvement, abstention, the committee) needs calibrated probabilities first.

For every outer fold the calibrator is fitted on the INNER out-of-fold blend scores of that fold (never on outer rows):
  raw       sigmoid of the blend's decision value (what the stack gives today)
  platt     one-feature logistic regression on the inner out-of-fold decision values
  isotonic  monotone regression of the same
All are scored on the outer rows with the same weights as the rest of the evaluation (similarity to the test set x test
prevalence). GATE: weighted ECE <= 0.05 and calibration slope in [0.8, 1.2] on BOTH isoforms for the method that is
adopted; if no method meets it for an isoform that isoform keeps no probability output.

Usage: python -m cypmode.challenge.calibration_experiment [--n-jobs 4]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.nested import (
    OUTER_REFIT,
    NestedPlan,
    PredictionCache,
    StackedLogit,
    _weights,
    inner_oof,
)
from cypmode.challenge.nested_runners import build_context_and_plan, default_candidates
from cypmode.challenge.validation_matching import SimilarityMatcher

TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
ECE_GATE, SLOPE_LO, SLOPE_HI = 0.05, 0.8, 1.2


def fit_calibrator(method: str, decision: np.ndarray, y: np.ndarray, weights: np.ndarray | None):
    """Return a function mapping decision values to probabilities, fitted on out-of-fold decision values only."""
    if method == "raw":
        return lambda d: expit(d)
    if method == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(decision.reshape(-1, 1), y, sample_weight=weights)
        return lambda d: lr.predict_proba(np.asarray(d).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(decision, y, sample_weight=weights)
        return lambda d: iso.predict(d)
    raise ValueError(method)


def calibration_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 10) -> dict:
    w = w / w.sum()
    index = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = index == b
        if m.any():
            ece += float(w[m].sum()) * abs(float(np.average(p[m], weights=w[m])) - float(np.average(y[m], weights=w[m])))
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y, sample_weight=w * len(w))
    return {"ece": ece, "brier": float(np.sum(w * (p - y) ** 2)), "slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]), "mean_pred": float(np.sum(w * p)), "mean_obs": float(np.sum(w * y))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    rows = []
    for iso in ("CYP3A4", "CYP2D6"):
        root = f"data/tdi_track/outputs/nested/{iso}"
        ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, 3, 5, 4, 0, args.n_jobs)
        saved = NestedPlan.load(f"{root}/plan.npz")
        cache = PredictionCache(f"{root}/cache", saved)
        names = [c.name for c in default_candidates() if not cache.missing(c.name, False)]
        wf = SimilarityMatcher(ctx.store.blocks["morgan_bits"], plan.external)
        prev = float(estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"])
        collected = {m: ([], [], []) for m in ("raw", "platt", "isotonic")}
        for repeat, k in plan.outer_folds():
            rws, oof = inner_oof(cache, names, repeat, k)
            refit = saved.task(repeat, k, OUTER_REFIT)
            outer = np.column_stack([cache.get(c, refit) for c in names])
            y_in, y_out = saved.labels(rws), saved.labels(refit.evaluate)
            w_in, w_out = _weights(y_in, rws, refit.train, wf, prev), _weights(y_out, refit.evaluate, refit.train, wf, prev)
            chosen = StackedLogit().fit(names, oof, y_in, w_in, "auc")
            for method, (ys, ps, ws) in collected.items():
                calibrator = fit_calibrator(method, np.asarray(chosen.inner_scores), y_in, w_in)
                ys.append(y_out)
                ps.append(np.clip(calibrator(chosen.predict(outer)), 1e-4, 1 - 1e-4))
                ws.append(w_out)
        for method, (ys, ps, ws) in collected.items():
            m = calibration_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(ws))
            m.update(isoform=iso, method=method, gate_met=bool(m["ece"] <= ECE_GATE and SLOPE_LO <= m["slope"] <= SLOPE_HI))
            rows.append(m)
    table = pd.DataFrame(rows)[["isoform", "method", "ece", "brier", "slope", "intercept", "mean_pred", "mean_obs", "gate_met"]]
    print(table.round(4).to_string(index=False))
    for iso in ("CYP3A4", "CYP2D6"):
        ok = table[(table.isoform == iso) & table.gate_met]
        print(f"{iso}: gate (ECE <= {ECE_GATE}, slope in [{SLOPE_LO}, {SLOPE_HI}]) met by: {', '.join(ok.method) if len(ok) else 'no method'}")


if __name__ == "__main__":
    main()
