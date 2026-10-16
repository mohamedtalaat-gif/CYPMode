"""Label-permutation control of the whole nested pipeline (a randomisation test: Leach, Molecular Modelling, 2nd ed., section 12.13.1).

A regression or a classifier can look predictive by chance, above all when many descriptors are tried on few compounds; the standard check is to reassign the activities at random
and repeat the whole analysis: the real model must be clearly better than every randomised one, and the randomised ones must sit where chance puts them. Here the labels of the pool
are permuted and the nested procedure (inner folds, selector, outer folds, the similarity x prevalence x confident-rows yardstick) is run on the classification candidates; the
outer-fold AUC must be 0.5 and the matthews correlation at the transferred call rate 0 within the spread of the permutations. A pipeline that leaks a label, or a selector that
overfits its inner folds, shows up as a permuted-label AUC above 0.5.

Only candidates that read nothing but the labels of ``Context.labels`` are used (the regression and arm candidates read the measured pIC50 values, which a label permutation does not
touch). Caches live in a temporary directory and are discarded.

Usage: python -m cypmode.challenge.permutation_control --isoform CYP3A4 --permutations 3
GATE (negative control): the mean permuted-label outer AUC of both z0 is within 0.04 of 0.5 and below the real-label AUC by at least 0.1, and the matthews correlation is within 0.08 of 0. The tolerances
are two sampling standard errors of one run (about 0.02 for the AUC and 0.04 for the MCC at the size of the pools and the share of positives), not a tuned value.
"""

import argparse
import logging
import tempfile

import numpy as np
import pandas as pd

from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.nested import NestedPlan, PredictionCache, StackedLogit, nested_evaluate
from cypmode.challenge.nested_runners import build_context_and_plan, default_candidates, run
from cypmode.challenge.validation_matching import SimilarityMatcher

Z_CUTS = (1.0, 1.96)
CANDIDATES = ("knn_tanimoto_k15", "svm_tanimoto_C1", "lgbm_A_deep", "lgbm_A+fps_deep", "extratrees_A", "rf_A")
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"


def evaluate_once(ctx, plan, matcher, prevalence, candidates) -> dict[float, tuple[float, float]]:
    """Fit the candidates on ``plan`` (whose labels may be permuted) and return {z0: (mean outer AUC, mean MCC at the transferred rate)}."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = PredictionCache(tmp, plan)
        run(candidates, ctx, plan, cache, include_deploy=False)
        out = {}
        for z0 in Z_CUTS:
            table = nested_evaluate(cache, [c.name for c in candidates], [StackedLogit()], confident_weights(matcher, ctx.confidence, z0), prevalence, "auc")
            out[z0] = (float(table["outer_score"].mean()), float(table["outer_mcc_rate"].mean()))
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--permutations", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    candidates = [c for c in default_candidates() if c.name in CANDIDATES]
    blocks = tuple(sorted({b for c in candidates for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, 1, 5, 4, 0, args.n_jobs, blocks)
    bits = FeatureStore.load_or_build(STORE, list(pd.read_csv(TRAIN_CSV).SMILES) + list(pd.read_csv(TEST_CSV).SMILES)).blocks["morgan_bits"]
    matcher = SimilarityMatcher(bits, plan.external)
    prevalence = float(estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"])
    real = evaluate_once(ctx, plan, matcher, prevalence, candidates)
    permuted = []
    for k in range(args.permutations):
        y = np.random.default_rng(1000 + k).permutation(plan.y)
        labels = ctx.labels.copy()
        labels[plan.pool] = y
        shuffled_ctx = type(ctx)(**{**ctx.__dict__, "labels": labels})
        shuffled_plan = NestedPlan(plan.pool, y, 1, 5, 4, 0, external=plan.external)
        permuted.append(evaluate_once(shuffled_ctx, shuffled_plan, matcher, prevalence, candidates))
    print(f"\n{iso}: {len(candidates)} classification candidates, 1 repeat of 5 outer folds, {args.permutations} permutations of the {len(plan.pool)} pool labels")
    ok = True
    for z0 in Z_CUTS:
        aucs, mccs = [p[z0][0] for p in permuted], [p[z0][1] for p in permuted]
        print(f"  z0={z0:<4} real labels: AUC {real[z0][0]:.4f}, MCC {real[z0][1]:+.4f} | permuted: AUC {np.mean(aucs):.4f} (range {min(aucs):.4f} to {max(aucs):.4f}), MCC {np.mean(mccs):+.4f} (range {min(mccs):+.4f} to {max(mccs):+.4f})")
        ok &= abs(np.mean(aucs) - 0.5) <= 0.04 and real[z0][0] - np.mean(aucs) >= 0.1 and abs(np.mean(mccs)) <= 0.08
    print(f"{iso} GATE (negative control, permuted labels give chance): {'MET' if ok else 'NOT MET'}")


if __name__ == "__main__":
    main()
