# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Selection-aware TDI modelling: is the model for the SCORED population better than the model for the hits?

Pre-registered before any population candidate was run. Observation (a cheap probe, not a result): TDI
labels exist only for compounds that were promoted to dose-response on the isoform (screening hits), yet the
scored test set is a full dose-response panel of analogs of hits from OTHER isoforms, so most of its CYP2D6
compounds are non-hits that the challenge scores as "assigned negative". For CYP2D6 the population
prevalence of the training set (5.3% positives among all 6,145 compounds) matches the test prevalence
(about 6-7%), the hits-only pool does not (21.7%); for CYP3A4 the test is analogs of 3A4 hits and the
hits-only pool is the right population, so CYP3A4 is a control where no gain is expected.

  * pool: every training compound in the all-isoform panel block (hits labelled by the rule, screened-but-never-promoted
    compounds labelled 0; the 3A4-focused blocks are excluded, see library_blocks.py), nested
    5-fold CV, plan identical in structure to the hits-only runs;
  * evaluation weights = similarity-to-test x test prevalence x confident rows (|z| >= z0, z0 in {1.0, 1.96};
    non-hits are unambiguous);
  * candidate sets: S0 "hits-only" = {pophits_*} (what the current model does when applied to the
    population), S1 = S0 + gate, S2 = S1 + two-part product, S3 = S2 + end-to-end population classifier;
  * verdict: Nadeau-Bengio corrected resampled t of S_k against S0 on the paired outer-fold AUC and average
    precision; a gain counts only if corrected p < 0.05 at BOTH z0 on CYP2D6. A gain on CYP3A4 would be a
    surprise and is reported, not used.

Usage: python -m cypmode.challenge.population_experiment --isoform CYP2D6 --repeats 1 --run
"""

import argparse
import logging

import numpy as np
import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import compare_candidate_sets
from cypmode.challenge.nested import (
    NestedPlan,
    PredictionCache,
    StackedLogit,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import (
    build_context_and_plan,
    population_candidates,
    run,
)
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
SETS = {
    "S0 hits-only": ["pophits_A_deep", "pophits_A+fps_deep"],
    "S1 +gate": ["pophits_A_deep", "pophits_A+fps_deep", "popgate_A_deep", "popgate_A+fps_deep"],
    "S2 +two-part": ["pophits_A_deep", "pophits_A+fps_deep", "popgate_A_deep", "popgate_A+fps_deep", "poptwo_A_deep", "poptwo_A+fps_deep"],
    "S3 +end-to-end": [f"{t}_{fs}_deep" for t in ("pophits", "popgate", "poptwo", "pop") for fs in ("A", "A+fps")],
}


def scaled_nonhits(weight_fn, hit, factor):
    """Evaluation weights with every never-measured compound's weight multiplied by ``factor``.

    factor 1 is the full training population, 0 is the hits-only population the model was previously judged on;
    values in between mimic a scored set with fewer non-hits than the training set.
    """

    def wrapped(rows, reference_rows):
        return weight_fn(rows, reference_rows) * np.where(hit[rows] == 1, 1.0, factor)

    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run", action="store_true", help="fill the prediction cache first")
    parser.add_argument("--n-jobs", type=int, default=3)
    parser.add_argument("--sensitivity", action="store_true", help="only print the non-hit-share sensitivity table from the existing caches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    root = f"data/tdi_track/outputs/nested/{iso}_pop_panel"
    train_csv, test_csv = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv", "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
    candidates = population_candidates()
    blocks = tuple(sorted({b for c in candidates for b in c.blocks}))
    ctx, plan = build_context_and_plan(train_csv, test_csv, iso, "data/tdi_track/outputs/caches/feature_store.npz", args.repeats, 5, 4, 0, args.n_jobs, blocks, None, True)
    cache = PredictionCache(f"{root}/cache", plan)
    if args.run:
        import pathlib

        pathlib.Path(root).mkdir(parents=True, exist_ok=True)
        plan.save(f"{root}/plan.npz")
        run(candidates, ctx, plan, cache, include_deploy=False)
    else:
        assert NestedPlan.load(f"{root}/plan.npz").plan_id == plan.plan_id

    bits = FeatureStore.load_or_build("data/tdi_track/outputs/caches/feature_store.npz", list(pd.read_csv(train_csv).SMILES) + list(pd.read_csv(test_csv).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"]
    if args.sensitivity:
        print(f"\n{iso}: sensitivity to the share of non-hits in the evaluation population (weight multiplier on never-measured compounds)")
        for z0 in Z_CUTS:
            base_fn = confident_weights(matcher, ctx.confidence, z0)
            for factor in (0.0, 0.1, 0.25, 0.5, 1.0):
                wf = scaled_nonhits(base_fn, ctx.hit, factor)
                line = f"  z0={z0:<4} non-hit weight {factor:<4}"
                for label, names in SETS.items():
                    t = nested_evaluate(cache, names, [StackedLogit()], wf, prevalence, "auc")
                    line += f" | {label.split()[0]} AUC {t['outer_score'].mean():.3f} MCC {t['outer_mcc_rate'].mean():+.3f}"
                print(line)
        return
    print(f"\n{iso}: population pool {len(plan.pool)} compounds, {int(ctx.labels[plan.pool].sum())} positives; {len(plan.outer_folds())} outer folds; test prevalence {prevalence:.3f}")
    for z0 in Z_CUTS:
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        print(f"\n  evaluation: similarity x prevalence x confident rows (z0 = {z0})")
        base = SETS["S0 hits-only"]
        for label, names in SETS.items():
            scored = nested_evaluate(cache, names, [StackedLogit()], weight_fn, prevalence, "auc")
            line = f"    {label:16s} outer AUC {scored['outer_score'].mean():.4f}  outer MCC at transferred rate {scored['outer_mcc_rate'].mean():+.4f}"
            if names is not base:
                for metric in ("auc", "ap"):
                    c = compare_candidate_sets(cache, names, base, StackedLogit(), weight_fn, prevalence, metric)
                    line += f" | vs S0 {metric.upper()} {c.mean_difference:+.4f} (p={c.p_value:.3f}, better {c.n_better}/{c.n_folds})"
            print(line)


if __name__ == "__main__":
    from cypmode import provenance

    provenance.announce()
    main()
