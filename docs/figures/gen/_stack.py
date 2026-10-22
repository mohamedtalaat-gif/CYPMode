# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""The default stack rebuilt from the cached nested predictions, with the weights and the prevalence the final build uses.

``outer_folds(isoform)`` returns, per outer fold of the 5-fold design, the labels, the stack's scores and the evaluation weights; ``selector_table(isoform)``
returns the same nested evaluation as the report (inner and outer scores of three ways of choosing). The mean outer AUC is checked against the number the final
build wrote to ``<isoform>_stack_choice.csv``; a figure is never drawn from a rebuild that disagrees with it.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.nested import (
    OUTER_REFIT,
    BestSingle,
    NestedPlan,
    OneStandardError,
    PredictionCache,
    StackedLogit,
    _weights,
    inner_oof,
    score_metric,
)
from cypmode.challenge.nested_runners import default_candidates
from cypmode.challenge.validation_matching import SimilarityMatcher

ROOT = Path(__file__).resolve().parents[3]
NESTED = ROOT / "data/tdi_track/outputs/nested"
STORE = ROOT / "data/tdi_track/outputs/caches/feature_store.npz"
FINAL = ROOT / "data/tdi_track/outputs/final"
BOARDS = {
    "CYP3A4": ROOT / "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP3A4_2026-10-04.csv",
    "CYP2D6": ROOT / "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP2D6_2026-10-04.csv",
}
TRAIN_CSV = ROOT / "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
Z0 = 1.0
TOLERANCE = 1e-6


@cache
def evaluation(isoform: str):
    """(candidate names, selector table, stack folds [(y, scores, weights, weights without the prevalence shift)])."""
    from cypmode.challenge.label_confidence import confidence_z

    root = NESTED / isoform
    plan = NestedPlan.load(root / "plan.npz")
    cache = PredictionCache(root / "cache", plan)
    names = [c.name for c in default_candidates() if not cache.missing(c.name)]

    train = pd.read_csv(TRAIN_CSV)
    z = np.full(int(max(plan.pool.max(), plan.external.max())) + 1, np.nan)
    z[: len(train)] = confidence_z(train, isoform)
    bits = np.load(STORE, allow_pickle=True)["morgan_bits"]  # read-only: the shared feature cache is never rewritten from here
    matcher = SimilarityMatcher(bits, plan.external)
    weight_fn = confident_weights(matcher, z, Z0)
    prevalence = float(estimate_from_csv(BOARDS[isoform])["prevalence"])

    table, folds = [], []
    selectors = (BestSingle(), OneStandardError(), StackedLogit())
    for repeat, k in plan.outer_folds():
        rows, oof = inner_oof(cache, names, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        outer = np.column_stack([cache.get(c, refit) for c in names])
        y_in, y_out = plan.labels(rows), plan.labels(refit.evaluate)
        w_in = _weights(y_in, rows, refit.train, weight_fn, prevalence)
        w_out = _weights(y_out, refit.evaluate, refit.train, weight_fn, prevalence)
        for selector in selectors:
            chosen = selector.fit(names, oof, y_in, w_in, "auc")
            scores = chosen.predict(outer)
            table.append({"selector": selector.name, "repeat": repeat, "outer": k, "inner_score": chosen.inner_score, "outer_score": score_metric(y_out, scores, w_out, "auc")})
            if selector.name == "stack_logit":
                folds.append((y_out, scores, w_out, _weights(y_out, refit.evaluate, refit.train, weight_fn, None)))
    table = pd.DataFrame(table)
    published = pd.read_csv(FINAL / f"{isoform}_stack_choice.csv").query("stack == 'default'")[f"auc_z{Z0}"].iloc[0]
    rebuilt = table.loc[table["selector"] == "stack_logit", "outer_score"].mean()
    if abs(rebuilt - published) > TOLERANCE:
        raise AssertionError(f"{isoform}: the rebuilt stack's mean outer AUC {rebuilt:.6f} differs from the final build's {published:.6f}")
    return names, table, folds
