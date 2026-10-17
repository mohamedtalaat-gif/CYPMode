"""Test-like holdout: the ids 2,395,316-2,395,799 block of the training file.

Why this block. The training ids fall into blocks that were screened differently (see ``library_blocks``). The highest
block is an analog-expansion round, built like the blinded test set: its median nearest-neighbour similarity to the
rest of the training set is 0.54 (the test set's is 0.59, the diversity panel's 0.31) and its CYP3A4 positive rate is
0.298 (the test set's, from the leaderboards, is 0.29-0.31). Holding the whole block out and training on everything
else therefore measures the stack in a regime that matches the test far better than any random or similarity-weighted
cross-validation fold. It exists only for CYP3A4 (the block was not measured on CYP2D6).

Output: for every candidate, its AUC / average precision / MCC on the block, next to the same candidate's
similarity-weighted cross-validation AUC on the panel pool, so the agreement between the two yardsticks is itself a
result. The candidate set and the "best" candidate are chosen from cross-validation only, never from the block.

Usage: python -m cypmode.challenge.block_holdout [--n-jobs 4]
"""

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cypmode.challenge.library_blocks import compound_numbers
from cypmode.challenge.nested import (
    OUTER_REFIT,
    NestedPlan,
    PredictionCache,
    Task,
    call_top_fraction,
    score_metric,
)
from cypmode.challenge.nested_runners import (
    build_context_and_plan,
    default_candidates,
    fit_predict,
)
from cypmode.challenge.validation_matching import weighted_mcc

ISO = "CYP3A4"
BLOCK_MIN_ID = 2_395_316
TRAIN_CSV = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST_CSV = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"
N_BOOT = 400


def holdout_split(ctx, plan, train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = compound_numbers(train_frame["Molecule_Name"])
    in_block = np.zeros(len(ctx.labels), dtype=bool)
    in_block[: len(ids)] = ids >= BLOCK_MIN_ID
    pool = plan.pool
    return pool[~in_block[pool]], pool[in_block[pool]]


def auc_interval(y: np.ndarray, score: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            values.append(roc_auc_score(y[i], score[i]))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    train_frame = pd.read_csv(TRAIN_CSV)
    candidates = [c for c in default_candidates() if c.family != "lgbm_reg_arms" and not c.knn_label_features]
    blocks = tuple(sorted({b for c in candidates for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, ISO, STORE, 3, 5, 4, 0, args.n_jobs, blocks)
    train_rows, hold_rows = holdout_split(ctx, plan, train_frame)
    y_hold = ctx.labels[hold_rows]
    logging.info("train rows %d, held-out block rows %d (positive rate %.3f)", len(train_rows), len(hold_rows), y_hold.mean())

    saved = NestedPlan.load(f"data/tdi_track/outputs/nested/{ISO}/plan.npz")
    cache = PredictionCache(f"data/tdi_track/outputs/nested/{ISO}/cache", saved)
    task = Task(repeat=0, outer=0, inner=OUTER_REFIT, train=train_rows, evaluate=hold_rows)
    rows, scores = [], {}
    for c in candidates:
        if cache.missing(c.name, False):
            continue
        s = fit_predict(c, ctx, task)
        scores[c.name] = s
        cv = np.mean([score_metric(saved.labels(saved.task(r, k, OUTER_REFIT).evaluate), cache.get(c.name, saved.task(r, k, OUTER_REFIT)), None, "auc") for r, k in saved.outer_folds()])
        called = call_top_fraction(s, float(y_hold.mean())).astype(int)
        lo, hi = auc_interval(y_hold, s)
        rows.append({"candidate": c.name, "cv_auc_unweighted": cv, "block_auc": roc_auc_score(y_hold, s), "block_auc_lo": lo, "block_auc_hi": hi, "block_ap": average_precision_score(y_hold, s), "block_mcc_at_prevalence": weighted_mcc(y_hold, called)})
        logging.info("%s: block AUC %.3f", c.name, rows[-1]["block_auc"])
    table = pd.DataFrame(rows).sort_values("cv_auc_unweighted", ascending=False)
    ranks = pd.DataFrame({k: pd.Series(v).rank(pct=True) for k, v in scores.items()})
    committee = ranks.mean(axis=1).to_numpy()
    called = call_top_fraction(committee, float(y_hold.mean())).astype(int)
    print(f"\n{ISO} test-like block holdout: {len(hold_rows)} compounds, positive rate {y_hold.mean():.3f}; trained on {len(train_rows)} other compounds\n")
    print(table.round(4).to_string(index=False))
    best = table.iloc[0]
    print(f"\nbest candidate by cross-validation alone: {best['candidate']} (cv AUC {best['cv_auc_unweighted']:.4f}) -> block AUC {best['block_auc']:.4f} [{best['block_auc_lo']:.3f}, {best['block_auc_hi']:.3f}]")
    lo, hi = auc_interval(y_hold, committee)
    print(f"equal-weight rank committee of all {len(scores)} candidates: block AUC {roc_auc_score(y_hold, committee):.4f} [{lo:.3f}, {hi:.3f}], AP {average_precision_score(y_hold, committee):.4f}, MCC at the block prevalence {weighted_mcc(y_hold, called):.4f}")
    print(f"agreement of the two yardsticks across candidates: Spearman rho between cv AUC and block AUC = {table['cv_auc_unweighted'].corr(table['block_auc'], method='spearman'):.3f}")


if __name__ == "__main__":
    main()
