"""Hedged CYP2D6 model: a rank blend of the hits-only stack and the selection-aware stack, judged across regimes.

The selection-aware model wins only when the scored set holds enough never-promoted compounds (job 012: crossover near half
the training share), and that share is not identifiable (job 013 narrows it to a weight of about 0.4-0.7 on the job-012
axis, with a stated bias). A decision that does not depend on guessing the regime is the one that does best in the worst
plausible case. This module measures exactly that:

  * the stacks are FITTED once under one assumed regime (non-hit weight ``--fit-weight``, default 0.5, the job-013 point
    estimate) because at deployment the regime is unknown; they are EVALUATED under every true regime in the grid
    (weights 0, 0.1, 0.25, 0.4, 0.5, 0.68, 1.0) and at both confidence cut-offs z0 in {1.0, 1.96};
  * three models: hits-only stack (S0), selection-aware stack (S3, the four selection-aware families plus the hits-only
    ones), and BLEND = average of the two stacks' empirical-CDF ranks (each mapped through its own inner out-of-fold scores);
  * pre-registered rule: the blend is adopted if, over the plausible band of true weights (0.4, 0.5, 0.68) at BOTH z0, its
    worst-case outer AUC is at least the best of the two pure models' worst cases, and over the whole grid its worst case
    is better than the worst case of the hits-only model. Otherwise the blend is dropped.

The same table answers the more basic question that job 012 could not, because there the stack was fitted under the same regime
it was scored under (which is not a choice available at deployment): fitted under a fixed assumed regime, is the selection-aware
stack better than the hits-only stack across the whole plausible band? Rule (fixed in advance): selection-aware replaces hits-only
only if, for the chosen fit weight, its mean AUC is higher at every band weight at both z0 AND the corrected paired p < 0.05 at
every band weight at both z0. The fit weight itself is reported for 0.25, 0.5 and 1.0 (a design choice we control at
deployment) and the one with the best worst-case selection-aware AUC over the band is the one the final build uses.

Usage: python -m cypmode.challenge.hedged_blend --isoform CYP2D6 --repeats 3
"""

import argparse
import logging

import numpy as np
import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes
from cypmode.challenge.nested import (
    OUTER_REFIT,
    NestedPlan,
    PredictionCache,
    Selection,
    Selector,
    StackedLogit,
    _EmpiricalCdf,
    _weights,
    best_threshold,
    call_top_fraction,
    inner_oof,
    score_metric,
)
from cypmode.challenge.nested_runners import build_context_and_plan, population_candidates
from cypmode.challenge.population_experiment import SETS, scaled_nonhits
from cypmode.challenge.validation_matching import SimilarityMatcher, weighted_mcc

Z_CUTS = (1.0, 1.96)
GRID = (0.0, 0.1, 0.25, 0.4, 0.5, 0.68, 1.0)
BAND = (0.4, 0.5, 0.68)
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"


class SubsetStack(Selector):
    """StackedLogit restricted to the candidates named in ``subset``."""

    def __init__(self, subset: list[str], label: str):
        self.subset, self.name = list(subset), label
        self.inner = StackedLogit()

    def _columns(self, candidates) -> list[int]:
        return [i for i, c in enumerate(candidates) if c in self.subset]

    def fit(self, candidates, oof, y, weights, metric):
        cols = self._columns(candidates)
        sel = self.inner.fit([candidates[i] for i in cols], oof[:, cols], y, weights, metric)
        return Selection(lambda m: sel.predict(m[:, cols]), sel.threshold, sel.inner_score, sel.chosen, sel.inner_scores, sel.inner_weights)


class BlendOfStacks(Selector):
    """Average of the empirical-CDF ranks of two stacks, each mapped through its own inner out-of-fold scores."""

    name = "blend"

    def __init__(self, subset_a: list[str], subset_b: list[str]):
        self.a, self.b = SubsetStack(subset_a, "a"), SubsetStack(subset_b, "b")

    def fit(self, candidates, oof, y, weights, metric):
        sa, sb = self.a.fit(candidates, oof, y, weights, metric), self.b.fit(candidates, oof, y, weights, metric)
        cdf_a, cdf_b = _EmpiricalCdf(sa.inner_scores), _EmpiricalCdf(sb.inner_scores)

        def predict(m):
            return 0.5 * (cdf_a(sa.predict(m)) + cdf_b(sb.predict(m)))

        inner = 0.5 * (cdf_a(sa.inner_scores) + cdf_b(sb.inner_scores))
        return Selection(predict, best_threshold(y, inner, weights), score_metric(y, inner, weights, metric), {**sa.chosen, **sb.chosen}, inner, weights)


def evaluate_across_regimes(cache, ctx, matcher, prevalence, names, fit_weight, selectors) -> pd.DataFrame:
    """Fit every selector under one assumed regime, score it under every true regime and z0."""
    plan = cache.plan
    rows_out = []
    for repeat, k in plan.outer_folds():
        rows, oof = inner_oof(cache, names, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        outer = np.column_stack([cache.get(c, refit) for c in names])
        y_in, y_out = plan.labels(rows), plan.labels(refit.evaluate)
        fit_fn = scaled_nonhits(confident_weights(matcher, ctx.confidence, 1.0), ctx.hit, fit_weight)
        w_in = _weights(y_in, rows, refit.train, fit_fn, prevalence)
        for selector in selectors:
            chosen = selector.fit(list(names), oof, y_in, w_in, "auc")
            scores = chosen.predict(outer)
            called = call_top_fraction(scores, chosen.rate()).astype(int)
            for z0 in Z_CUTS:
                for factor in GRID:
                    wf = scaled_nonhits(confident_weights(matcher, ctx.confidence, z0), ctx.hit, factor)
                    w_out = _weights(y_out, refit.evaluate, refit.train, wf, prevalence)
                    rows_out.append({"model": selector.name, "repeat": repeat, "outer": k, "z0": z0, "weight": factor, "auc": score_metric(y_out, scores, w_out, "auc"), "mcc": weighted_mcc(y_out, called, w_out)})
    return pd.DataFrame(rows_out)


def decide(table: pd.DataFrame) -> dict:
    mean = table.groupby(["model", "z0", "weight"])[["auc", "mcc"]].mean().reset_index()

    def worst(model: str, weights) -> float:
        sub = mean[(mean.model == model) & mean.weight.isin(weights)]
        return float(sub.groupby("z0").auc.min().min())  # worst over the weights, at the worse of the two z0

    models = ("hits_only", "selection_aware")
    band_blend, grid_blend = worst("blend", BAND), worst("blend", GRID)
    return {
        "band_worst": {m: worst(m, BAND) for m in (*models, "blend")},
        "grid_worst": {m: worst(m, GRID) for m in (*models, "blend")},
        "adopt_blend": bool(band_blend >= max(worst(m, BAND) for m in models) and grid_blend > worst("hits_only", GRID)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="CYP2D6", choices=["CYP2D6", "CYP3A4"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fit-weight", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    root = f"data/tdi_track/outputs/nested/{iso}_pop_panel"
    blocks = tuple(sorted({b for c in population_candidates() for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, args.repeats, 5, 4, 0, args.n_jobs, blocks, None, True)
    saved = NestedPlan.load(f"{root}/plan.npz")
    assert saved.plan_id == plan.plan_id
    cache = PredictionCache(f"{root}/cache", saved)
    names = SETS["S3 +end-to-end"]
    bits = FeatureStore.load_or_build(STORE, list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = float(estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"])
    selectors = [SubsetStack(SETS["S0 hits-only"], "hits_only"), SubsetStack(names, "selection_aware"), BlendOfStacks(SETS["S0 hits-only"], names)]
    n_train, n_test = plan_sizes(saved)
    best = None
    for fit_weight in sorted({0.25, args.fit_weight, 1.0}):
        table = evaluate_across_regimes(cache, ctx, matcher, prevalence, names, fit_weight, selectors)
        mean = table.groupby(["model", "z0", "weight"])[["auc", "mcc"]].mean()
        print(f"\n{iso}: stacks fitted under non-hit weight {fit_weight}, scored under each true weight (columns); mean outer AUC | MCC at the transferred rate")
        for z0 in Z_CUTS:
            print(f"\n  z0 = {z0}      " + "  ".join(f"w={w:<5}" for w in GRID))
            for model in ("hits_only", "selection_aware", "blend"):
                print(f"    {model:16s}" + "  ".join(f"{mean.loc[(model, z0, w), 'auc']:.3f}|{mean.loc[(model, z0, w), 'mcc']:+.2f}" for w in GRID))
        print("\n  paired differences over outer folds, corrected resampled t on AUC (mean difference, p):")
        all_significant, all_positive = True, True
        for z0 in Z_CUTS:
            for first, second in (("selection_aware", "hits_only"), ("blend", "hits_only"), ("blend", "selection_aware")):
                line = f"    z0={z0:<4} {first} - {second:16s}"
                for w in BAND:
                    a = table[(table.model == first) & (table.z0 == z0) & (table.weight == w)].sort_values(["repeat", "outer"]).auc.to_numpy()
                    b = table[(table.model == second) & (table.z0 == z0) & (table.weight == w)].sort_values(["repeat", "outer"]).auc.to_numpy()
                    r = corrected_resampled_t(a - b, n_train, n_test)
                    line += f" | w={w}: {r.mean_difference:+.3f} (p={r.p_value:.2f})"
                    if (first, second) == ("selection_aware", "hits_only"):
                        all_significant &= r.p_value < 0.05
                        all_positive &= r.mean_difference > 0
                print(line)
        verdict = decide(table)
        print(f"\n  worst-case AUC over the plausible band {BAND}: " + ", ".join(f"{m} {v:.3f}" for m, v in verdict["band_worst"].items()))
        print("  worst-case AUC over the whole grid:           " + ", ".join(f"{m} {v:.3f}" for m, v in verdict["grid_worst"].items()))
        print(f"{iso} GATE (hedged blend adopted by the minimax rule, fit weight {fit_weight}): {'MET' if verdict['adopt_blend'] else 'NOT MET'}")
        print(f"{iso} GATE (selection-aware replaces hits-only: higher mean AUC and corrected p < 0.05 at every band weight at both z0, fit weight {fit_weight}): {'MET' if all_positive and all_significant else 'NOT MET'}")
        if best is None or verdict["band_worst"]["selection_aware"] > best[1]:
            best = (fit_weight, verdict["band_worst"]["selection_aware"])
    print(f"\n{iso} fit weight with the best worst-case selection-aware AUC over the band: {best[0]} (worst case {best[1]:.3f})")

if __name__ == "__main__":
    main()
