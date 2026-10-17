"""Shared pre-registered test for "does adding this candidate group improve the assembled model?".

Used by physics_experiment (xTB block) and pair_transfer (SAR-aware pair model). Everything is on the nested
similarity x prevalence x confident-rows yardstick (z0 in {1.0, 1.96}); a verdict needs Nadeau-Bengio corrected
resampled t with p < 0.05 and a positive mean AUC difference at BOTH z0 values.

  alone    paired (plain, with-addition) candidates, best-single outer-fold AUC / AP; only when the addition is a
           feature block that can be switched on and off;
  stacked  default stack + the added candidates vs the default stack (StackedLogit), because an addition that only
           helps when it replaces the others is not an improvement of the assembled model.
"""

import argparse
import logging

import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
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
    Candidate,
    build_context_and_plan,
    default_candidates,
    run,
)
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
GATE_ALPHA = 0.05
NESTED_ROOT = "data/tdi_track/outputs/nested"


def parse_args(doc: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="fill the prediction cache for the added candidates first")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--outer", type=int, default=5, help="outer folds of the nested plan (the plan under --root must have been built with the same value)")
    parser.add_argument("--root", default=NESTED_ROOT, help="directory holding <isoform>/plan.npz and the prediction caches")
    return parser.parse_args()


def evaluate_addition(
    isoform: str,
    candidates: list[Candidate],
    repeats: int,
    n_jobs: int,
    fill: bool,
    alone_pairs: list[tuple[str, str]] = (),
    diagnostics: tuple[str, ...] = (),
    outer: int = 5,
    root: str = NESTED_ROOT,
) -> dict[str, bool]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = f"{root}/{isoform}"
    blocks = tuple(sorted({b for c in candidates for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, isoform, STORE, repeats, outer, 4, 0, n_jobs, blocks)
    saved = NestedPlan.load(f"{root}/plan.npz")
    assert saved.plan_id == plan.plan_id, "the plan changed: this experiment must share the existing caches"
    cache = PredictionCache(f"{root}/cache", plan)
    if fill:
        run(candidates, ctx, plan, cache, include_deploy=False)

    bits = FeatureStore.load_or_build(STORE, list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{isoform}_2026-10-04.csv")["prevalence"]
    n_train, n_test = plan_sizes(plan)
    base = [c.name for c in default_candidates() if not cache.missing(c.name, False)]
    added = [c.name for c in candidates if not cache.missing(c.name, False)]
    print(f"\n{isoform}: {len(plan.outer_folds())} outer folds; base stack {len(base)} candidates; added candidates cached: {added}")
    verdict: dict[str, dict[float, bool]] = {"alone": {}, "stacked": {}}
    for z0 in Z_CUTS:
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        for name in [n for n in diagnostics if n in added]:
            d = nested_evaluate(cache, [name], [BestSingle()], weight_fn, prevalence, "auc")
            print(f"  z0={z0:<4} diagnostic {name}: outer AUC {d['outer_score'].mean():.4f}")
        if alone_pairs:
            ok_alone = True
            for plain, with_it in alone_pairs:
                if cache.missing(with_it, False):
                    ok_alone = False
                    continue
                res = {}
                for metric in ("auc", "ap"):
                    a = nested_evaluate(cache, [with_it], [BestSingle()], weight_fn, prevalence, metric).set_index(["repeat", "outer"])["outer_score"]
                    b = nested_evaluate(cache, [plain], [BestSingle()], weight_fn, prevalence, metric).set_index(["repeat", "outer"])["outer_score"]
                    res[metric] = (corrected_resampled_t((a - b).to_numpy(), n_train, n_test), float(a.mean()), float(b.mean()))
                auc, ap = res["auc"], res["ap"]
                print(f"  z0={z0:<4} alone  {with_it:24s} vs {plain:16s}: AUC {auc[2]:.4f} -> {auc[1]:.4f} ({auc[0].mean_difference:+.4f}, p={auc[0].p_value:.3f}, better {auc[0].n_better}/{auc[0].n_folds}) | AP {ap[0].mean_difference:+.4f} (p={ap[0].p_value:.3f})")
                ok_alone &= auc[0].p_value < GATE_ALPHA and auc[0].mean_difference > 0
            verdict["alone"][z0] = ok_alone
        if added:
            line = f"  z0={z0:<4} stacked base+added vs base:"
            for metric in ("auc", "ap"):
                c = compare_candidate_sets(cache, base + added, base, StackedLogit(), weight_fn, prevalence, metric)
                line += f" {metric.upper()} {c.mean_difference:+.4f} (p={c.p_value:.3f}, better {c.n_better}/{c.n_folds})"
                if metric == "auc":
                    verdict["stacked"][z0] = c.p_value < GATE_ALPHA and c.mean_difference > 0
            print(line)
    for kind in ("alone", "stacked"):
        if kind == "alone" and not alone_pairs:
            continue
        met = all(verdict[kind].get(z0, False) for z0 in Z_CUTS)
        print(f"{isoform} GATE ({kind}, corrected p < {GATE_ALPHA} at both z0): {'MET' if met else 'NOT MET'}")
    return {kind: all(v.get(z0, False) for z0 in Z_CUTS) for kind, v in verdict.items()}
