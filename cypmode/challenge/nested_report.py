# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Honest comparison of every cached candidate and every selection procedure for one isoform.

    python -m cypmode.challenge.nested_report --isoform CYP3A4 --board <per-isoform leaderboard csv>

Prints, per candidate, its own outer-fold score (no selection involved), then per selector the
inner (contest) score, the outer score of its frozen choice and the optimism between them. The
test-set prior comes from the leaderboard export (cypmode.challenge.leaderboard_prior) and the
similarity matching from the test compounds' fingerprints -- nothing is typed in.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.nested import (
    OUTER_REFIT,
    BestSingle,
    MeanOfTop,
    NestedPlan,
    OneStandardError,
    PredictionCache,
    StackedLogit,
    _weights,
    nested_evaluate,
    score_metric,
    summarize,
)
from cypmode.challenge.nested_runners import default_candidates
from cypmode.challenge.validation_matching import SimilarityMatcher


def candidate_table(cache: PredictionCache, names: list[str], weight_fn, prevalence, metric: str) -> pd.DataFrame:
    plan, rows = cache.plan, []
    for name in names:
        values = []
        for repeat, k in plan.outer_folds():
            task = plan.task(repeat, k, OUTER_REFIT)
            y = plan.labels(task.evaluate)
            values.append(score_metric(y, cache.get(name, task), _weights(y, task.evaluate, task.train, weight_fn, prevalence), metric))
        rows.append({"candidate": name, "outer_score": float(np.mean(values)), "sd": float(np.std(values))})
    return pd.DataFrame(rows).sort_values("outer_score", ascending=False)


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isoform", required=True)
    parser.add_argument("--board", help="per-isoform leaderboard csv used to estimate the test positive rate")
    parser.add_argument("--metric", default="auc", choices=["auc", "ap"])
    parser.add_argument("--root", default="data/tdi_track/outputs/nested")
    args = parser.parse_args()

    root = Path(args.root) / args.isoform
    plan = NestedPlan.load(root / "plan.npz")
    cache = PredictionCache(root / "cache", plan)
    store = FeatureStore.load_or_build("data/tdi_track/outputs/caches/feature_store.npz", _smiles())
    weight_fn = SimilarityMatcher(store.blocks["morgan_bits"], plan.external)
    prior = estimate_from_csv(args.board) if args.board else None
    prevalence = prior["prevalence"] if prior else None
    print(f"plan {plan.plan_id}: {len(plan.pool)} pool rows, outer prevalence {plan.y.mean():.3f}, test prevalence {prevalence}")

    order = [c.name for c in default_candidates()]
    names = [n for n in order if not cache.missing(n, include_deploy=False)] + [n for n in cache.candidates() if n not in order and not cache.missing(n, include_deploy=False)]
    print(f"{len(names)} complete candidates\n")
    print(candidate_table(cache, names, weight_fn, prevalence, args.metric).round(4).to_string(index=False))
    selectors = [BestSingle(), OneStandardError(), MeanOfTop(3), MeanOfTop(5), StackedLogit()]
    table = nested_evaluate(cache, names, selectors, weight_fn, prevalence, args.metric)
    print("\nselectors (inner = contest score, outer = honest score):")
    print(summarize(table).round(4).to_string())
    table.to_csv(root / f"nested_{args.metric}.csv", index=False)


def _smiles() -> list[str]:
    train = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv")
    test = pd.read_csv("data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv")
    return list(train["SMILES"]) + list(test["SMILES"])


if __name__ == "__main__":
    _main()
