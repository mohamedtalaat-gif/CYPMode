"""How hit-like is the blinded test set? A label-shift estimate from structures alone (support for the CYP2D6 A/B decision).

The selection-aware model beats the hits-only one only when the scored set holds enough compounds that the screen would
never have promoted on CYP2D6 (job 012: crossover near half the training share of non-hits). The share cannot be read
from the test file (no labels), but the gate model that predicts "was promoted on this isoform" can be run on the 750
test structures and its score distribution compared with the one it produces out-of-fold on the training panel:

    probabilistic adjusted count   share_hit(test) = (mean P(test) - mean P(panel non-hits)) / (mean P(panel hits) - mean P(panel non-hits))

and the classify-and-count version at three thresholds. The equivalent non-hit weight is the ratio of non-hit to hit odds
in the test set over the same ratio in the panel (1.0 = the training share, the weight axis of job 012). Assumption:
the class-conditional structure distributions are the same in test and panel, which analog expansion violates, so this is
an estimate with a stated bias, not a measurement; the nearest-neighbour hit fraction is reported as an independent
sanity check. Nothing here reads a test label (none exists).

Usage: python -m cypmode.challenge.test_nonhit_share --isoform CYP2D6
"""

import argparse

import numpy as np

TRAIN = "data/tdi_track/inputs/cyp-challenge-TRAIN_TDI.csv"
TEST = "data/tdi_track/inputs/cyp-challenge-TEST-BLINDED.csv"
STORE = "data/tdi_track/outputs/caches/feature_store.npz"


def pac_share(p_test: np.ndarray, p_hits: np.ndarray, p_nonhits: np.ndarray) -> float:
    return float((p_test.mean() - p_nonhits.mean()) / (p_hits.mean() - p_nonhits.mean()))


def equivalent_nonhit_weight(share_test: float, share_panel: float) -> float:
    """Odds of non-hit to hit in the test over the same odds in the panel."""
    return float(((1 - share_test) / share_test) / ((1 - share_panel) / share_panel))


def main() -> None:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    from cypmode.challenge.nested_runners import (
        FEATURE_SETS,
        LGBM_PRESETS,
        build_context_and_plan,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="CYP2D6", choices=["CYP2D6", "CYP3A4"])
    parser.add_argument("--n-jobs", type=int, default=3)
    args = parser.parse_args()
    ctx, plan = build_context_and_plan(TRAIN, TEST, args.isoform, STORE, 3, 5, 4, 0, args.n_jobs, tuple(FEATURE_SETS["A"]), None, True)
    pool, test = plan.pool, plan.external
    hit = ctx.hit[pool].astype(int)
    x, xt = ctx.store.matrix(FEATURE_SETS["A"], pool), ctx.store.matrix(FEATURE_SETS["A"], test)
    params = dict(LGBM_PRESETS["deep"])
    oof = np.zeros(len(pool))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(x, hit):
        oof[b] = LGBMClassifier(verbose=-1, n_jobs=args.n_jobs, random_state=0, **params).fit(x[a], hit[a]).predict_proba(x[b])[:, 1]
    p_test = LGBMClassifier(verbose=-1, n_jobs=args.n_jobs, random_state=0, **params).fit(x, hit).predict_proba(xt)[:, 1]
    panel_share = float(hit.mean())
    print(f"{args.isoform}: panel pool {len(pool)} compounds, hit share {panel_share:.3f}; {len(test)} test structures; gate out-of-fold AUC {roc_auc_score(hit, oof):.3f}")
    print(f"mean P(hit): panel hits {oof[hit == 1].mean():.3f} | panel non-hits {oof[hit == 0].mean():.3f} | test {p_test.mean():.3f}")
    share = pac_share(p_test, oof[hit == 1], oof[hit == 0])
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(500):
        i, j = rng.integers(0, len(pool), len(pool)), rng.integers(0, len(p_test), len(p_test))
        h = hit[i]
        if 5 < h.sum() < len(h) - 5:
            boot.append(pac_share(p_test[j], oof[i][h == 1], oof[i][h == 0]))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    print(f"probabilistic adjusted count: hit share of the test set {share:.3f} (95% bootstrap {lo:.3f}-{hi:.3f}); panel {panel_share:.3f}")
    print(f"equivalent non-hit weight on the job-012 axis: {equivalent_nonhit_weight(share, panel_share):.2f} (95%: {equivalent_nonhit_weight(hi, panel_share):.2f}-{equivalent_nonhit_weight(lo, panel_share):.2f})")
    for threshold in (0.3, 0.4, 0.5):
        tpr, fpr, rate = (oof[hit == 1] >= threshold).mean(), (oof[hit == 0] >= threshold).mean(), (p_test >= threshold).mean()
        print(f"classify and count at {threshold}: TPR {tpr:.2f} FPR {fpr:.2f} test called-hit rate {rate:.3f} -> hit share {(rate - fpr) / (tpr - fpr):.3f}")
    nn_test = np.argsort(-ctx.similarity[np.ix_(test, pool)], axis=1)[:, :5]
    within = ctx.similarity[np.ix_(pool, pool)].copy()
    np.fill_diagonal(within, -1.0)
    nn_panel = np.argsort(-within, axis=1)[:, :5]
    print(f"sanity check, hit fraction among the 5 nearest panel neighbours: test {hit[nn_test].mean():.3f} vs panel leave-one-out {hit[nn_panel].mean():.3f}")


if __name__ == "__main__":
    main()
