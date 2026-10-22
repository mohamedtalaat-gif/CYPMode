# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Physics inside the selection-aware CYP2D6 model (job 071).

The selection-aware model splits P(label) into P(hit) x P(label | hit) and is the largest effect found (AUC +0.10 to +0.20 against
the hits-only model). Whether the screen promoted a compound on an isoform depends on how reactive and how promiscuous it is, which is
what the physics block measures, so the block is added to all four selection-aware families (gate, hits-only, two-part, end-to-end) on
feature sets A and A+fps, on the same panel-block pool, plan and caches as ``population_experiment``.

  stacked:  the selection-aware stack S3 (eight candidates) + the eight physics candidates vs S3, StackedLogit, weight 1.0 (the
            training share), corrected resampled t on AUC, both z0;
  regimes:  stacks fitted once under the training share and scored under every true non-hit weight of ``hedged_blend`` (0 to 1),
            physics stack minus plain stack, corrected t at the plausible band (0.4, 0.5, 0.68);
  alone:    each family with physics against the same family without (best single), for information.
GATES: stacked (p < 0.05 and a positive mean difference at both z0) and regimes (the same at every band weight at both z0).

Usage: python -m cypmode.challenge.population_physics --isoform CYP2D6 --repeats 3 --run
"""

import argparse
import logging
import pathlib

import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.hedged_blend import (
    BAND,
    GRID,
    SubsetStack,
    evaluate_across_regimes,
)
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import (
    compare_candidate_sets,
    corrected_resampled_t,
    plan_sizes,
)
from cypmode.challenge.nested import (
    BestSingle,
    NestedPlan,
    PredictionCache,
    StackedLogit,
    nested_evaluate,
)
from cypmode.challenge.nested_runners import (
    build_context_and_plan,
    population_candidates,
    population_physics_candidates,
    run,
)
from cypmode.challenge.population_experiment import SETS
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="CYP2D6", choices=["CYP2D6", "CYP3A4"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--fit-weight", type=float, default=1.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    root = f"data/tdi_track/outputs/nested/{iso}_pop_panel"
    candidates = population_physics_candidates()
    blocks = tuple(sorted({b for c in candidates + population_candidates() for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, args.repeats, 5, 4, 0, args.n_jobs, blocks, None, True)
    saved = NestedPlan.load(f"{root}/plan.npz")
    assert saved.plan_id == plan.plan_id, "the population plan changed: this experiment must share the v1 caches"
    cache = PredictionCache(f"{root}/cache", saved)
    if args.run:
        pathlib.Path(root).mkdir(parents=True, exist_ok=True)
        run(candidates, ctx, plan, cache, include_deploy=False)

    bits = FeatureStore.load_or_build(STORE, list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = float(estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"])
    n_train, n_test = plan_sizes(saved)
    base = SETS["S3 +end-to-end"]
    phys = [c.name for c in candidates if not cache.missing(c.name, False)]
    print(f"\n{iso}: population pool {len(plan.pool)}, {len(plan.outer_folds())} outer folds; physics candidates cached: {len(phys)} of {len(candidates)}")
    stacked_ok = bool(phys)
    for z0 in Z_CUTS:
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        for label, names in (("selection-aware stack", base), ("with physics", base + phys)):
            t = nested_evaluate(cache, names, [StackedLogit()], weight_fn, prevalence, "auc")
            print(f"  z0={z0:<4} {label:22s} outer AUC {t['outer_score'].mean():.4f}  MCC at transferred rate {t['outer_mcc_rate'].mean():+.4f}")
        line = f"  z0={z0:<4} stacked with physics vs without:"
        for metric in ("auc", "ap"):
            c = compare_candidate_sets(cache, base + phys, base, StackedLogit(), weight_fn, prevalence, metric)
            line += f" {metric.upper()} {c.mean_difference:+.4f} (p={c.p_value:.3f}, better {c.n_better}/{c.n_folds})"
            if metric == "auc":
                stacked_ok &= c.p_value < 0.05 and c.mean_difference > 0
        print(line)
        for plain in [n for n in base if n.endswith(("_A_deep", "_A+fps_deep"))]:
            with_phys = plain.replace("_A_deep", "_A+phys_deep").replace("_A+fps_deep", "_A+fps+phys_deep")
            if cache.missing(with_phys, False):
                continue
            a = nested_evaluate(cache, [with_phys], [BestSingle()], weight_fn, prevalence, "auc").set_index(["repeat", "outer"])["outer_score"]
            b = nested_evaluate(cache, [plain], [BestSingle()], weight_fn, prevalence, "auc").set_index(["repeat", "outer"])["outer_score"]
            r = corrected_resampled_t((a - b).to_numpy(), n_train, n_test)
            print(f"  z0={z0:<4} alone  {with_phys:28s} vs {plain:26s}: AUC {b.mean():.4f} -> {a.mean():.4f} ({r.mean_difference:+.4f}, p={r.p_value:.3f})")
    print(f"{iso} GATE (population physics stacked, corrected p < 0.05 at both z0): {'MET' if stacked_ok else 'NOT MET'}")

    if phys:
        names = base + phys
        selectors = [SubsetStack(base, "selection_aware"), SubsetStack(base + phys, "selection_aware_physics")]
        table = evaluate_across_regimes(cache, ctx, matcher, prevalence, names, args.fit_weight, selectors)
        mean = table.groupby(["model", "z0", "weight"])[["auc", "mcc"]].mean()
        print(f"\n  regimes: stacks fitted under non-hit weight {args.fit_weight}, scored under each true weight; mean outer AUC | MCC at the transferred rate")
        regime_ok = True
        for z0 in Z_CUTS:
            print(f"\n    z0 = {z0}      " + "  ".join(f"w={w:<5}" for w in GRID))
            for model in ("selection_aware", "selection_aware_physics"):
                print(f"      {model:24s}" + "  ".join(f"{mean.loc[(model, z0, w), 'auc']:.3f}|{mean.loc[(model, z0, w), 'mcc']:+.2f}" for w in GRID))
            line = f"    z0={z0:<4} physics minus plain:"
            for w in BAND:
                a = table[(table.model == "selection_aware_physics") & (table.z0 == z0) & (table.weight == w)].sort_values(["repeat", "outer"]).auc.to_numpy()
                b = table[(table.model == "selection_aware") & (table.z0 == z0) & (table.weight == w)].sort_values(["repeat", "outer"]).auc.to_numpy()
                r = corrected_resampled_t(a - b, n_train, n_test)
                line += f" | w={w}: {r.mean_difference:+.3f} (p={r.p_value:.2f})"
                regime_ok &= r.p_value < 0.05 and r.mean_difference > 0
            print(line)
        print(f"{iso} GATE (population physics over the band, corrected p < 0.05 at every band weight at both z0, fit weight {args.fit_weight}): {'MET' if regime_ok else 'NOT MET'}")


if __name__ == "__main__":
    main()
