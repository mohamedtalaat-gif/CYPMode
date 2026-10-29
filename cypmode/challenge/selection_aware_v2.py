# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Selection-aware v2: a gate that sees the screen's whole promotion pattern (job 040).

v1 (population_experiment.py) multiplies P(label | hit) by a gate P(hit) learned from structure alone. The screen
that produced the training labels promoted compounds to dose-response on four isoforms, and promiscuity is shared
between them: whether a compound was promoted on CYP1A2, CYP2C9 and CYP3A4 says a lot about whether it was promoted
on CYP2D6. v2 therefore
  * trains one first-stage LightGBM per isoform on "was promoted on isoform k" and cross-fits its logits on the
    training rows (out-of-fold, so the second stage never sees in-sample optimism), then fits a logistic second stage
    for the target isoform (``hit_profile_gate``);
  * optionally adds the classical Heckman selection-correction terms (probit score and inverse Mills ratio of the
    gate) to the feature set of the TDI-given-hit model (``sel2_heckman``).
Only the structure of the EVALUATION compound enters its gate; the hit pattern of an evaluation compound is never read.

Candidates: sel2gate_A (gate only), sel2two_A (gate x TDI-given-hit), sel2heck_A (Heckman terms). Comparison on the
same panel-block population pool, nested CV and yardstick as v1 (similarity x prevalence x confident rows,
z0 in {1.0, 1.96}):
  V1 = hits-only + v1 gate + v1 two-part   vs   V2 = V1 + the three v2 candidates     (StackedLogit)
  gate head to head: sel2gate_A vs popgate_A_deep (best single).
GATE: V2 beats V1 with corrected p < 0.05 at both z0 (AUC), and is not worse than hits-only when non-hits are rare
(non-hit weight 0.1 in the sensitivity table). Otherwise v1 stays the selection-aware candidate.

Usage: python -m cypmode.challenge.selection_aware_v2 --isoform CYP2D6 --repeats 3 --run
"""

import argparse
import logging
import pathlib

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
    build_context_and_plan,
    population_candidates,
    run,
    selection_v2_candidates,
)
from cypmode.challenge.population_experiment import SETS, scaled_nonhits
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
V1 = SETS["S2 +two-part"]
V2 = V1 + ["sel2gate_A", "sel2two_A", "sel2heck_A"]
HITS_ONLY = SETS["S0 hits-only"]
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true", help="fill the prediction cache for the v2 candidates first")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    root = f"data/tdi_track/outputs/nested/{iso}_pop_panel"
    candidates = selection_v2_candidates()
    blocks = tuple(sorted({b for c in candidates + population_candidates() for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, args.repeats, 5, 4, 0, args.n_jobs, blocks, None, True)
    cache = PredictionCache(f"{root}/cache", plan)
    if args.run:
        pathlib.Path(root).mkdir(parents=True, exist_ok=True)
        if not pathlib.Path(f"{root}/plan.npz").exists():
            plan.save(f"{root}/plan.npz")
        assert NestedPlan.load(f"{root}/plan.npz").plan_id == plan.plan_id, "the population plan changed: v2 must share the v1 caches"
        run(candidates, ctx, plan, cache, include_deploy=False)

    bits = FeatureStore.load_or_build(STORE, list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"]
    n_train, n_test = plan_sizes(plan)
    have_v2 = [n for n in V2 if not cache.missing(n, False)]
    print(f"\n{iso}: population pool {len(plan.pool)}, {len(plan.outer_folds())} outer folds; v2 candidates cached: {[n for n in have_v2 if n.startswith('sel2')]}")
    met = {}
    for z0 in Z_CUTS:
        base_fn = confident_weights(matcher, ctx.confidence, z0)
        print(f"\n  z0 = {z0}")
        for label, names in (("hits-only", HITS_ONLY), ("V1 (v1 gate + two-part)", V1), ("V2 (V1 + v2)", have_v2)):
            t = nested_evaluate(cache, names, [StackedLogit()], base_fn, prevalence, "auc")
            print(f"    {label:26s} outer AUC {t['outer_score'].mean():.4f}  MCC at transferred rate {t['outer_mcc_rate'].mean():+.4f}")
        if len(have_v2) > len(V1):
            line = "    V2 vs V1:"
            for metric in ("auc", "ap"):
                c = compare_candidate_sets(cache, have_v2, V1, StackedLogit(), base_fn, prevalence, metric)
                line += f" {metric.upper()} {c.mean_difference:+.4f} (p={c.p_value:.3f}, better {c.n_better}/{c.n_folds})"
                if metric == "auc":
                    met[z0] = c.p_value < 0.05 and c.mean_difference > 0
            print(line)
        if not cache.missing("sel2gate_A", False):
            a = nested_evaluate(cache, ["sel2gate_A"], [BestSingle()], base_fn, prevalence, "auc").set_index(["repeat", "outer"])["outer_score"]
            b = nested_evaluate(cache, ["popgate_A_deep"], [BestSingle()], base_fn, prevalence, "auc").set_index(["repeat", "outer"])["outer_score"]
            r = corrected_resampled_t((a - b).to_numpy(), n_train, n_test)
            print(f"    gate head to head: sel2gate_A {a.mean():.4f} vs popgate_A_deep {b.mean():.4f} ({r.mean_difference:+.4f}, p={r.p_value:.3f}, better {r.n_better}/{r.n_folds})")
    print("\n  sensitivity to the share of non-hits in the scored set (weight multiplier on never-measured compounds)")
    for z0 in Z_CUTS:
        base_fn = confident_weights(matcher, ctx.confidence, z0)
        for factor in (0.0, 0.1, 0.25, 0.5, 1.0):
            wf = scaled_nonhits(base_fn, ctx.hit, factor)
            line = f"    z0={z0:<4} non-hit weight {factor:<4}"
            for label, names in (("hits-only", HITS_ONLY), ("V1", V1), ("V2", have_v2)):
                t = nested_evaluate(cache, names, [StackedLogit()], wf, prevalence, "auc")
                line += f" | {label} AUC {t['outer_score'].mean():.3f} MCC {t['outer_mcc_rate'].mean():+.3f}"
            print(line)
    print(f"\n{iso} GATE (V2 beats V1, corrected p < 0.05 at both z0, AUC): {'MET' if all(met.get(z, False) for z in Z_CUTS) else 'NOT MET'}")


if __name__ == "__main__":
    main()
