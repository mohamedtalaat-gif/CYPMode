# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Does training on label confidence help when the evaluation matches the scoring rule?

Pre-registered (before any weighted candidate had been run). The official score only counts compounds
whose TDI label is assigned with confidence (cypmode.challenge.label_confidence), so:

  * evaluation weights = similarity-to-test x test-prevalence x [|z| >= z0], for z0 in {1.0, 1.96};
  * candidates: the deep LightGBM on feature sets A and A+fps trained (a) on every row (the existing
    caches), (b) only on rows with |z| >= 1 ("cdrop10"), (c) on every row weighted by min(1, |z|/2)
    ("csoft");
  * verdict per candidate pair: Nadeau-Bengio corrected resampled t on the paired outer-fold AUC (and
    average precision) differences; an improvement counts only if corrected p < 0.05 at BOTH z0 values
    on the isoform it is claimed for.

Usage: python -m cypmode.challenge.confidence_experiment --isoform CYP2D6 --repeats 1 --run [--experiment context]
"""

import argparse
import logging

import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes
from cypmode.challenge.nested import (
    BestSingle,
    NestedPlan,
    PredictionCache,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import (
    build_context_and_plan,
    context_candidates,
    run,
    weighting_candidates,
)
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
PAIRS = [(f"lgbm_{fs}_deep", f"lgbm_{fs}_deep_{w}") for fs in ("A", "A+fps") for w in ("cdrop10", "csoft")]
# the alert-context block (pre-registered the same way): its effect alone, then combined with confident-row training
CONTEXT_PAIRS = [
    ("lgbm_A_deep", "lgbm_A+ctx_deep"),
    ("lgbm_A+fps_deep", "lgbm_A+fps+ctx_deep"),
    ("lgbm_A_deep_cdrop10", "lgbm_A+ctx_deep_cdrop10"),
    ("lgbm_A+fps_deep_cdrop10", "lgbm_A+fps+ctx_deep_cdrop10"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="fill the prediction cache for the experiment's candidates first")
    parser.add_argument("--experiment", choices=["weighting", "context"], default="weighting")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    root = f"data/tdi_track/outputs/nested/{iso}"
    train_csv, test_csv = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv", "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
    candidates, pairs = (weighting_candidates(), PAIRS) if args.experiment == "weighting" else (context_candidates(), CONTEXT_PAIRS)
    blocks = None if args.experiment == "weighting" else tuple(sorted({b for c in candidates for b in c.blocks}))
    ctx, plan = build_context_and_plan(train_csv, test_csv, iso, "data/tdi_track/outputs/caches/feature_store.npz", args.repeats, 5, 4, 0, args.n_jobs, blocks)
    saved = NestedPlan.load(f"{root}/plan.npz")
    assert saved.plan_id == plan.plan_id, "the plan changed: this experiment must share the existing caches"
    cache = PredictionCache(f"{root}/cache", plan)
    if args.run:
        run(candidates, ctx, plan, cache, include_deploy=False)

    bits = FeatureStore.load_or_build("data/tdi_track/outputs/caches/feature_store.npz", list(pd.read_csv(train_csv).SMILES) + list(pd.read_csv(test_csv).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"]
    n_train, n_test = plan_sizes(plan)
    print(f"\n{iso}: {len(plan.outer_folds())} outer folds; evaluation = similarity x prevalence x confident rows")
    for z0 in Z_CUTS:
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        for plain, weighted in pairs:
            if cache.missing(weighted, False):
                continue
            scores = {}
            for metric in ("auc", "ap"):
                a = nested_evaluate(cache, [weighted], [BestSingle()], weight_fn, prevalence, metric).set_index(["repeat", "outer"])["outer_score"]
                b = nested_evaluate(cache, [plain], [BestSingle()], weight_fn, prevalence, metric).set_index(["repeat", "outer"])["outer_score"]
                scores[metric] = (corrected_resampled_t((a - b).to_numpy(), n_train, n_test), float(a.mean()), float(b.mean()))
            auc, ap = scores["auc"], scores["ap"]
            print(
                f"  z0={z0:<4} {weighted:28s} vs plain: AUC {auc[2]:.4f} -> {auc[1]:.4f} ({auc[0].mean_difference:+.4f}, p={auc[0].p_value:.3f}, better {auc[0].n_better}/{auc[0].n_folds})"
                f" | AP {ap[0].mean_difference:+.4f} (p={ap[0].p_value:.3f})"
            )


if __name__ == "__main__":
    main()
