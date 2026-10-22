# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Probabilistic label-process model: a committee of two-arm regressors that outputs P(label) and expected improvement.

The label is a deterministic threshold of two noisy measurements, positive <=> s = t - max(d, 4) > log10 2 (t: TDI
condition arm, d: direct-inhibition arm, each with a posterior standard deviation in the training file). A classifier
sees only the thresholded shadow. This model keeps the structure of the process:

  * every committee member fits BOTH arms by gradient boosting on floor-clipped arms (clipping at 4 leaves every label
    unchanged), weighting each training row by the inverse of its own measurement variance (noise-aware);
  * a member's predicted shift is t_hat - max(d_hat, 4); the committee mean m is the point estimate and the spread of
    the members is the epistemic uncertainty sd_e (members differ in feature set, bagging subset and seed);
  * the out-of-bag error of the members (each member is scored on the rows it never saw) gives an honest scale
    sigma_base, so the predictive distribution of the shift is N(m, sigma_base^2 + sd_e^2) and
        P(label) = Phi((m - log10 2) / sigma),        EI = sigma * (phi(z) + z * Phi(z)),   z = (m - log10 2) / sigma
    where EI is the expected improvement of the shift over the threshold, the quantity used to decide where an extra
    measurement or sample is worth most.

Candidates: committee_m (point estimate only, to isolate what the uncertainty adds), committee_p, committee_ei.
Only the training rows' arms, standard deviations and the evaluation rows' structures are read; the evaluation rows'
measurements never are.

Gates (fixed in advance): (1) the stack that contains the three committee candidates beats the default stack with
corrected p < 0.05 at both z0 (addition_experiment.py); (2) calibration of committee_p as it comes out, with no
post-hoc recalibration: weighted ECE <= 0.05 and slope in [0.8, 1.2] (same bar as calibration_experiment.py).

Usage: python -m cypmode.challenge.committee_label_model --isoform CYP3A4 --repeats 3 --run
"""

import argparse
import hashlib

import numpy as np

from cypmode.challenge.tdi_targets import ASSAY_FLOOR, SHIFT_THRESHOLD

OUTPUTS = ("m", "p", "ei")
SUBSET_FRACTION = 0.7


def measurement_variance(ctx, rows: np.ndarray) -> np.ndarray:
    """sd(s)^2 = sd_t^2 + [d >= 4] sd_d^2 for the given rows; missing standard deviations take the median."""
    if "tdi_sd" not in ctx.targets or "direct_sd" not in ctx.targets:
        raise ValueError("Context.targets needs 'tdi_sd' and 'direct_sd' for noise-aware weighting")
    sd_t, sd_d = ctx.targets["tdi_sd"][rows], ctx.targets["direct_sd"][rows]
    d = ctx.targets["direct"][rows]
    var = sd_t**2 + np.where(d >= ASSAY_FLOOR, np.nan_to_num(sd_d) ** 2, 0.0)
    return np.where(np.isfinite(var) & (var > 0), var, np.nanmedian(var[var > 0]) if (var > 0).any() else 0.25)


_MEMBER_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}  # the three outputs share one set of fits


def _member_fits(ctx, task, member_sets, n_seeds: int, lgbm: dict, seed: int, noise_weights: bool):
    """Member shift predictions on the evaluation rows, out-of-bag sums and counts on the training rows, training shifts."""
    from lightgbm import LGBMRegressor

    digest = hashlib.md5(task.train.tobytes() + b"|" + task.evaluate.tobytes() + repr((member_sets, n_seeds, sorted(lgbm.items()), seed, noise_weights, ctx.isoform)).encode())
    for name in sorted(ctx.targets):  # the training rows' measurements are part of the fit's identity
        digest.update(name.encode() + np.ascontiguousarray(ctx.targets[name][task.train]).tobytes())
    key = digest.hexdigest()
    if key in _MEMBER_CACHE:
        return _MEMBER_CACHE[key]
    rng = np.random.default_rng(seed)
    n = len(task.train)
    t = np.maximum(ctx.targets["tdi"][task.train], ASSAY_FLOOR)
    d = np.maximum(ctx.targets["direct"][task.train], ASSAY_FLOOR)
    if noise_weights:
        var = measurement_variance(ctx, task.train)
        w = 1.0 / (var + np.median(var))
        w = w / w.mean()
    else:
        w = np.ones(n)
    shifts, oob_sum, oob_n = [], np.zeros(n), np.zeros(n)
    k = 0
    for blocks in member_sets:
        train_x, eval_x = ctx.store.matrix(blocks, task.train), ctx.store.matrix(blocks, task.evaluate)
        for _ in range(n_seeds):
            subset = rng.random(n) < SUBSET_FRACTION
            fits = []
            for target in (t, d):
                model = LGBMRegressor(verbose=-1, n_jobs=ctx.n_jobs, random_state=(seed + k) % 2**31, **lgbm)
                fits.append(model.fit(train_x[subset], target[subset], sample_weight=w[subset]))
            shifts.append(fits[0].predict(eval_x) - np.maximum(fits[1].predict(eval_x), ASSAY_FLOOR))
            out = ~subset
            if out.any():
                oob_sum[out] += fits[0].predict(train_x[out]) - np.maximum(fits[1].predict(train_x[out]), ASSAY_FLOOR)
                oob_n[out] += 1
            k += 1
    if len(_MEMBER_CACHE) > 400:
        _MEMBER_CACHE.clear()
    _MEMBER_CACHE[key] = (np.array(shifts), oob_sum, oob_n, t - d)
    return _MEMBER_CACHE[key]


def committee_predict(ctx, task, member_sets: list[tuple[str, ...]], n_seeds: int, lgbm: dict, output: str, seed: int, noise_weights: bool = True) -> np.ndarray:
    from scipy.stats import norm

    if output not in OUTPUTS:
        raise ValueError(output)
    shifts, oob_sum, oob_n, s_train = _member_fits(ctx, task, member_sets, n_seeds, lgbm, seed, noise_weights)
    mean = shifts.mean(axis=0)
    if output == "m":
        return mean
    seen = oob_n > 0
    mse = float(np.mean((oob_sum[seen] / oob_n[seen] - s_train[seen]) ** 2)) if seen.any() else float(np.var(s_train))
    sigma = np.sqrt(mse + shifts.std(axis=0, ddof=1) ** 2)
    z = (mean - SHIFT_THRESHOLD) / sigma
    if output == "p":
        return norm.cdf(z)
    return sigma * (norm.pdf(z) + z * norm.cdf(z))


def main() -> None:
    from cypmode.challenge.addition_experiment import (
        STORE,
        TEST_CSV,
        TRAIN_CSV,
        evaluate_addition,
    )
    from cypmode.challenge.calibration_experiment import (
        ECE_GATE,
        SLOPE_HI,
        SLOPE_LO,
        calibration_metrics,
    )
    from cypmode.challenge.label_confidence import confident_weights
    from cypmode.challenge.leaderboard_prior import estimate_from_csv
    from cypmode.challenge.nested import (
        OUTER_REFIT,
        NestedPlan,
        PredictionCache,
        _weights,
    )
    from cypmode.challenge.nested_runners import (
        build_context_and_plan,
        committee_candidates,
    )
    from cypmode.challenge.validation_matching import SimilarityMatcher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", required=True, choices=["CYP3A4", "CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    candidates = committee_candidates()
    evaluate_addition(args.isoform, candidates, args.repeats, args.n_jobs, args.run, diagnostics=tuple(c.name for c in candidates))

    iso = args.isoform
    root = f"data/tdi_track/outputs/nested/{iso}"
    blocks = tuple(sorted({b for c in candidates for b in c.blocks}))
    ctx, plan = build_context_and_plan(TRAIN_CSV, TEST_CSV, iso, STORE, args.repeats, 5, 4, 0, args.n_jobs, blocks)
    saved = NestedPlan.load(f"{root}/plan.npz")
    cache = PredictionCache(f"{root}/cache", saved)
    matcher = SimilarityMatcher(ctx.store.blocks["morgan_bits"], plan.external)
    prevalence = float(estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"])
    name = "committee_p_A+fps"
    if cache.missing(name, False):
        return
    print(f"\n{iso}: calibration of {name} exactly as it comes out (no recalibration); gate ECE <= {ECE_GATE}, slope in [{SLOPE_LO}, {SLOPE_HI}]")
    for z0 in (1.0, 1.96):
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        ys, ps, ws = [], [], []
        for repeat, k in saved.outer_folds():
            refit = saved.task(repeat, k, OUTER_REFIT)
            y = saved.labels(refit.evaluate)
            ys.append(y)
            ps.append(np.clip(cache.get(name, refit), 1e-4, 1 - 1e-4))
            ws.append(_weights(y, refit.evaluate, refit.train, matcher, prevalence) * weight_fn(refit.evaluate, refit.train))
        m = calibration_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(ws))
        met = m["ece"] <= ECE_GATE and SLOPE_LO <= m["slope"] <= SLOPE_HI
        print(f"  z0={z0:<4} ECE {m['ece']:.4f}  Brier {m['brier']:.4f}  slope {m['slope']:.3f}  mean predicted {m['mean_pred']:.3f} vs observed {m['mean_obs']:.3f}  -> calibration gate {'MET' if met else 'NOT MET'}")


if __name__ == "__main__":
    main()
