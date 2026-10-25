# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Which CYP2D6 file is the final entry: the hits-only file A or the selection-aware file B? A decision procedure that is fixed before any upload.

The two files differ only in CYP2D6 (CYP3A4 is identical) and, on the 750 blinded compounds, in 40 of the CYP2D6 calls. Whether B is better depends on one
number that no label identifies: the weight w of the compounds that the screen never promoted in the scored set (1.0 is the share of the all-isoform panel).
The deployment table of ``hedged_blend`` (the stacks fitted once under the weight of the final build, 1.0, and scored under every true w) says what each
file is worth in each regime. The table that job 012 printed fits the stack under the weight it is scored under, which no deployment can do, and is not used here.

The operating point matters. The experiments of jobs 010 to 014 call the compounds above the threshold that is best on the inner folds, which calls about a third of the pool;
the final build calls the top share of the scored set that equals the estimated prevalence times a multiplier (5.9 percent for CYP2D6 in both files). A first version of this
module scored the calls of the experiments; it was changed to the rule of the final build before any output of it was committed or used, and the log prints both MCC values.
The evidence on w is of two kinds and neither settles it: the label-shift estimate of job 013 (w about 0.5, from structures alone, assuming that the structure
distribution within each class is the same in the test set and the panel) and the board fit of ``population_share`` (w and the TDI rate among hits trade off, and the plausible
region reaches from 0 to 1.5). This module therefore does three things.

  1. The deployment table at fit weight 1.0 with percentile intervals from a bootstrap over the 15 outer units (3 repeats x 5 folds): the mean MCC when the top ``RATE``
     share of the scored population is called (the weighted share, as the final build calls the top share of the test set), the MCC at the rate of the experiments, and the
     AUC of A, B and the rank blend of the two, under w in {0, 0.1, 0.25, 0.4, 0.5, 0.68, 1.0} and both confidence cut-offs z0.
  2. The summary of the evidence on w, read from the stored outputs of jobs 013 and the board fit (nothing is recomputed).
  3. The operating characteristics of reading the live board as a detector of the regime. The live board scores half of the test set. For each w and z0, a half of
     ``HALF`` compounds is drawn from the out-of-fold pool with probability proportional to the evaluation weight (the test-like population of that regime), each file calls
     the top ``RATE`` share of that half by its own score, the calls are scored with the MCC, and the difference MCC(B) - MCC(A) is recorded, ``RESAMPLES`` times per repeat. The distribution of that
     difference under each w is what an upload of A and then B would show.

Procedure, fixed in advance (written 2026-10-25 after the tables of jobs 012, 013 and 014 had been read; the bootstrap intervals, the distribution of the live
difference and the threshold had not been computed):

  * LOW = {0.0} (almost no never-promoted compound in the scored set; B loses), TIE = {0.1}, HIGH = {0.25, 0.4, 0.5, 0.68, 1.0} (B wins). Both z0 count equally.
  * delta_low is the mean over z0 of MCC(A) - MCC(B) at w = 0; delta_high is the mean over z0 and HIGH of MCC(B) - MCC(A).
  * For a threshold t on the live difference, B is chosen when MCC(B) - MCC(A) >= t. error_low(t) is the probability of choosing B when w = 0, error_high(t) the
    mean over HIGH of the probability of choosing A. The risk of t is max(error_low * delta_low, error_high * delta_high); t* is the threshold of the smallest risk
    on the grid -0.2 to +0.2 in steps of 0.005 (ties go to the larger t, that is to A).
  * The detector is usable if error_low(t*) and error_high(t*) are both at most 0.25. If it is not usable, no upload of B is made to decide, and the entry is the file
    with the smaller largest regret over the whole grid of w in the deployment table (``fallback``).
  * Reading the board: the CYP2D6 MCC of the upload of B minus that of the upload of A, on the same board. B is the final entry if the difference is at least t*, A
    otherwise (``python -m cypmode.challenge.selection_decision --live-delta X``).

The result is written to ``data/tdi_track/outputs/comparisons/selection_decision.json``; the method report prints the log and, once the scores of the uploads are in
``docs/leaderboard_scores.csv``, the table of them and the entry that the rule gives.

Usage: python -m cypmode.challenge.selection_decision --isoform CYP2D6 --repeats 3
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cypmode.challenge import hedged_blend as hb

LOGS = Path("data/structure_track/outputs/cyp3a4/logs")
RESULT = Path("data/tdi_track/outputs/comparisons/selection_decision.json")
BOARD_BAND = Path("data/tdi_track/outputs/comparisons/population_share_band.csv")
FIT_WEIGHT = 1.0  # the weight under which the selection-aware stack of the final build is fitted
MULTIPLIER = 1.0  # the call-rate multiplier of CYP2D6 in both files of the final build (``manifest.json``: called rate 0.0587 against a prevalence of 0.0591)
LOW, TIE, HIGH = (0.0,), (0.1,), (0.25, 0.4, 0.5, 0.68, 1.0)
HALF = 375  # compounds in the half of the test set that the live board scores
RESAMPLES = 2000  # per repeat, z0 and w
THRESHOLDS = np.round(np.arange(-0.2, 0.2001, 0.005), 3)
MAX_ERROR = 0.25
MODELS = ("hits_only", "selection_aware", "blend")


def mcc_of(y: np.ndarray, call: np.ndarray) -> np.ndarray:
    """Matthews correlation coefficient along the last axis of boolean arrays (a zero denominator gives 0)."""
    tp, fp = (y & call).sum(-1).astype(float), (~y & call).sum(-1).astype(float)
    fn, tn = (y & ~call).sum(-1).astype(float), (~y & ~call).sum(-1).astype(float)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def top_share_calls(scores: np.ndarray, rate: float) -> np.ndarray:
    """Boolean calls of the ``round(rate * n)`` highest scores along the last axis of ``scores`` (at least one)."""
    k = max(1, round(rate * scores.shape[-1]))
    calls = np.zeros(scores.shape, dtype=bool)
    top = np.argpartition(-scores, k - 1, axis=-1)[..., :k]
    np.put_along_axis(calls, top, True, axis=-1)
    return calls


def weighted_top_calls(scores: np.ndarray, weights: np.ndarray, rate: float) -> np.ndarray:
    """Calls of the highest scores until their weight reaches ``rate`` of the total weight: the call rule of the final build applied to a weighted population."""
    order = np.argsort(-scores, kind="stable")
    reached = np.cumsum(weights[order])
    keep = reached - weights[order] < rate * weights.sum()  # the compound that crosses the share is called
    calls = np.zeros(len(scores), dtype=bool)
    calls[order[keep]] = True
    return calls


def live_differences(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, weights: np.ndarray, n: int, resamples: int, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """MCC(B) - MCC(A) on ``resamples`` halves of ``n`` compounds drawn with probability proportional to ``weights``, each file calling the top ``rate`` share of the half by its own score,
    and the mean number of calls that differ in a half."""
    index = rng.choice(len(y), size=(resamples, n), p=weights / weights.sum())
    sampled, call_a, call_b = y[index], top_share_calls(scores_a[index], rate), top_share_calls(scores_b[index], rate)
    return mcc_of(sampled, call_b) - mcc_of(sampled, call_a), float((call_a != call_b).sum(-1).mean())


def choose_threshold(low: np.ndarray, high: list[np.ndarray], delta_low: float, delta_high: float, grid: np.ndarray = THRESHOLDS) -> dict:
    """The threshold of the smallest risk, with the two error rates there (ties go to the larger threshold)."""
    best = None
    for t in grid:
        error_low = float((low >= t).mean())
        error_high = float(np.mean([(h < t).mean() for h in high]))
        risk = max(error_low * delta_low, error_high * delta_high)
        if best is None or risk < best["risk"] - 1e-12 or abs(risk - best["risk"]) <= 1e-12:
            best = {"threshold": float(t), "error_low": error_low, "error_high": error_high, "risk": float(risk)}
    return best


def apply_rule(live_delta: float, result: dict) -> str:
    """The final CYP2D6 file by the rule: B if the live difference reaches the threshold, A otherwise; the fallback when the detector is not usable."""
    if not result["usable"]:
        return result["fallback"]
    return "B" if live_delta >= result["threshold"] else "A"


def bootstrap_interval(a: np.ndarray, b: np.ndarray, resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of the paired differences a - b over units and the 95% percentile interval of a bootstrap over the units."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pac_estimate(log_text: str) -> dict:
    """The label-shift estimate of job 013 from its log: w and its 95% interval."""
    found = re.search(r"equivalent non-hit weight on the job-012 axis: ([0-9.]+) \(95%: ([0-9.]+)-([0-9.]+)\)", log_text)
    if not found:
        raise ValueError("the log of job 013 has no estimate of the non-hit weight")
    return {"w": float(found.group(1)), "low": float(found.group(2)), "high": float(found.group(3))}


def board_band(frame: pd.DataFrame) -> dict:
    """The best fit and the extent of the plausible region of the board fit (w and the rate among hits trade off)."""
    best, plausible = frame.loc[frame["chi2"].idxmin()], frame[frame["plausible"]]
    return {"best_w": float(best["w"]), "best_c": float(best["c"]), "w_low": float(plausible["w"].min()), "w_high": float(plausible["w"].max())}


def fallback_choice(scores: pd.DataFrame) -> tuple[str, dict]:
    """The file with the smaller largest regret over the whole grid (mean MCC over z0 and units), and the regrets."""
    table = scores.groupby(["model", "weight"])["mcc"].mean().unstack(0)
    regret = {m: float((table[["hits_only", "selection_aware"]].max(axis=1) - table[m]).max()) for m in ("hits_only", "selection_aware")}
    return ("A" if regret["hits_only"] <= regret["selection_aware"] else "B"), regret


def collect_outer(cache, ctx, matcher, prevalence, names, fit_weight, selectors) -> list[dict]:
    """For every outer unit: the labels, the scores and calls of every selector, and the evaluation weights under every z0 and w (the same objects as ``hedged_blend``)."""
    plan = cache.plan
    units = []
    for repeat, k in plan.outer_folds():
        rows, oof = hb.inner_oof(cache, names, repeat, k)
        refit = plan.task(repeat, k, hb.OUTER_REFIT)
        outer = np.column_stack([cache.get(c, refit) for c in names])
        y_in, y_out = plan.labels(rows), plan.labels(refit.evaluate)
        fit_fn = hb.scaled_nonhits(hb.confident_weights(matcher, ctx.confidence, 1.0), ctx.hit, fit_weight)
        w_in = hb._weights(y_in, rows, refit.train, fit_fn, prevalence)
        unit = {"repeat": repeat, "outer": k, "y": np.asarray(y_out).astype(bool), "models": {}, "weights": {}}
        for selector in selectors:
            chosen = selector.fit(list(names), oof, y_in, w_in, "auc")
            scores = chosen.predict(outer)
            unit["models"][selector.name] = {"scores": scores, "called": hb.call_top_fraction(scores, chosen.rate()).astype(bool)}
        for z0 in hb.Z_CUTS:
            for factor in hb.GRID:
                wf = hb.scaled_nonhits(hb.confident_weights(matcher, ctx.confidence, z0), ctx.hit, factor)
                unit["weights"][(z0, factor)] = np.asarray(hb._weights(y_out, refit.evaluate, refit.train, wf, prevalence), dtype=float)
        units.append(unit)
    return units


def unit_metrics(units: list[dict], rate: float) -> pd.DataFrame:
    """One row per unit, model, z0 and w: the AUC, the MCC when the top ``rate`` of the weighted population is called (``mcc``) and the MCC at the rate of the experiments (``mcc_cv``)."""
    rows = []
    for unit in units:
        y = unit["y"].astype(int)
        for model, made in unit["models"].items():
            for (z0, factor), w in unit["weights"].items():
                rows.append({"model": model, "repeat": unit["repeat"], "outer": unit["outer"], "z0": z0, "weight": factor, "auc": hb.score_metric(y, made["scores"], w, "auc"),
                             "mcc": hb.weighted_mcc(y, weighted_top_calls(made["scores"], w, rate).astype(int), w), "mcc_cv": hb.weighted_mcc(y, made["called"].astype(int), w)})
    return pd.DataFrame(rows)


def pooled_by_repeat(units: list[dict]) -> dict[int, dict]:
    pooled = {}
    for repeat in sorted({u["repeat"] for u in units}):
        part = [u for u in units if u["repeat"] == repeat]
        pooled[repeat] = {
            "y": np.concatenate([u["y"] for u in part]),
            "scores": {m: np.concatenate([u["models"][m]["scores"] for u in part]) for m in MODELS},
            "weights": {key: np.concatenate([u["weights"][key] for u in part]) for key in part[0]["weights"]},
        }
    return pooled


def live_distributions(pooled: dict[int, dict], rate: float, seed: int = 0) -> tuple[dict, float]:
    """``(z0, w) -> samples of MCC(B) - MCC(A)`` over all repeats, and the mean number of differing calls in a half."""
    rng = np.random.default_rng(seed)
    out, differing = {}, []
    for (z0, factor) in pooled[next(iter(pooled))]["weights"]:
        parts = []
        for p in pooled.values():
            samples, n_diff = live_differences(p["y"], p["scores"]["hits_only"], p["scores"]["selection_aware"], p["weights"][(z0, factor)], HALF, RESAMPLES, rate, rng)
            parts.append(samples)
            differing.append(n_diff)
        out[(z0, factor)] = np.concatenate(parts)
    return out, float(np.mean(differing))


def decide(table: pd.DataFrame, distributions: dict) -> dict:
    """Apply the procedure of the module docstring to the deployment table and the distributions of the live difference."""
    mean = table.groupby(["model", "z0", "weight"])["mcc"].mean()
    z_cuts = sorted({z for z, _ in distributions})
    delta_low = float(np.mean([mean[("hits_only", z, w)] - mean[("selection_aware", z, w)] for z in z_cuts for w in LOW]))
    delta_high = float(np.mean([mean[("selection_aware", z, w)] - mean[("hits_only", z, w)] for z in z_cuts for w in HIGH]))
    low = np.concatenate([distributions[(z, w)] for z in z_cuts for w in LOW])
    high = [distributions[(z, w)] for z in z_cuts for w in HIGH]
    best = choose_threshold(low, high, delta_low, delta_high)
    fallback, regret = fallback_choice(table[["model", "weight", "mcc"]])
    usable = best["error_low"] <= MAX_ERROR and best["error_high"] <= MAX_ERROR
    return {**best, "delta_low": delta_low, "delta_high": delta_high, "usable": bool(usable), "fallback": fallback, "regret": regret}


def unit_values(table: pd.DataFrame, model: str, z0: float, weight: float, name: str) -> np.ndarray:
    """The value of ``name`` of one model under one z0 and w, one entry per outer unit in the order repeat, outer."""
    return table[(table.model == model) & (table.z0 == z0) & (table.weight == weight)].sort_values(["repeat", "outer"])[name].to_numpy()


def rate_of(evidence: dict) -> float:
    return float(evidence["rate"])


def print_tables(table: pd.DataFrame, evidence: dict, distributions: dict, n_diff: float, decision: dict) -> None:
    models = ("hits_only", "selection_aware", "blend")
    units = table[(table.model == "hits_only") & (table.z0 == hb.Z_CUTS[0]) & (table.weight == hb.GRID[0])].shape[0]
    print(f"\nCYP2D6: deployment table, stacks fitted under non-hit weight {FIT_WEIGHT}, scored under each true weight w; {units} outer units; mean AUC | MCC when the top {rate_of(evidence):.4f} of the scored population is called (the rule of the final build);")
    print("difference of that MCC to the hits-only file with the 95% interval of a bootstrap over the units (the MCC at the rate of the experiments is given for comparison)")
    for z0 in hb.Z_CUTS:
        print(f"\n  z0 = {z0}")
        for w in hb.GRID:
            cells = "  ".join(f"{m[:5]} {unit_values(table, m, z0, w, 'auc').mean():.3f}|{unit_values(table, m, z0, w, 'mcc').mean():+.3f}" for m in models)
            diffs = []
            for model in ("selection_aware", "blend"):
                mean, lo, hi = bootstrap_interval(unit_values(table, model, z0, w, "mcc"), unit_values(table, "hits_only", z0, w, "mcc"))
                diffs.append(f"{model[:5]} - hits {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            cv = unit_values(table, "selection_aware", z0, w, "mcc_cv").mean() - unit_values(table, "hits_only", z0, w, "mcc_cv").mean()
            print(f"    w={w:<4} {cells}   MCC: {'; '.join(diffs)}   (at the rate of the experiments: selec - hits {cv:+.3f})")
    pac, band = evidence["pac"], evidence["band"]
    print(f"\nCYP2D6: evidence on w. Label shift from structures (job 013): w = {pac['w']:.2f} (95% {pac['low']:.2f}-{pac['high']:.2f}), assuming the same structure distribution within each class in the test set and the panel.")
    print(f"Board fit of the official baseline (w and the TDI rate among hits trade off): best fit w = {band['best_w']:.2f} with c = {band['best_c']:.2f}; the plausible region reaches w = {band['w_low']:.2f} to {band['w_high']:.2f}, so the board does not identify w.")
    print(f"\nCYP2D6: live board read as a detector of the regime ({HALF} compounds per half, each file calling its top {evidence['rate']:.4f}; {RESAMPLES} draws per repeat, z0 and w; {n_diff:.1f} calls differ in a half on average, about 20 are expected from the 40 of 750 that differ in the final files):")
    for z0 in hb.Z_CUTS:
        for w in hb.GRID:
            d = distributions[(z0, w)]
            print(f"    z0={z0:<4} w={w:<4} MCC(B) - MCC(A): mean {d.mean():+.3f}  sd {d.std():.3f}  P(>= {decision['threshold']:+.3f}) {float((d >= decision['threshold']).mean()):.2f}")
    print(f"\nCYP2D6 loss if the regime is LOW (w = 0, B loses): {decision['delta_low']:.3f}; if it is HIGH (w >= 0.25, B wins): {decision['delta_high']:.3f}")
    print(f"CYP2D6 threshold on the live difference MCC(B) - MCC(A): {decision['threshold']:+.3f}; error if w = 0: {decision['error_low']:.2f}; mean error if w >= 0.25: {decision['error_high']:.2f}; risk {decision['risk']:.4f}")
    print(f"CYP2D6 largest regret over the whole grid: hits-only {decision['regret']['hits_only']:.3f}, selection-aware {decision['regret']['selection_aware']:.3f}; the fallback is file {decision['fallback']}")
    print(f"CYP2D6 A/B DETECTOR: {'USABLE' if decision['usable'] else 'NOT USABLE'} (both errors at most {MAX_ERROR})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isoform", default="CYP2D6", choices=["CYP2D6"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--live-delta", type=float, help="apply the stored rule to the CYP2D6 MCC of the upload of B minus that of the upload of A")
    args = parser.parse_args()
    if args.live_delta is not None:
        print(f"final CYP2D6 entry by the rule: {apply_rule(args.live_delta, json.loads(RESULT.read_text()))}")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    iso = args.isoform
    root = f"data/tdi_track/outputs/nested/{iso}_pop_panel"
    blocks = tuple(sorted({b for c in hb.population_candidates() for b in c.blocks}))
    ctx, plan = hb.build_context_and_plan(hb.TRAIN_CSV, hb.TEST_CSV, iso, hb.STORE, args.repeats, 5, 4, 0, args.n_jobs, blocks, None, True)
    saved = hb.NestedPlan.load(f"{root}/plan.npz")
    assert saved.plan_id == plan.plan_id
    cache = hb.PredictionCache(f"{root}/cache", saved)
    names = hb.SETS["S3 +end-to-end"]
    bits = hb.FeatureStore.load_or_build(hb.STORE, list(pd.read_csv(hb.TRAIN_CSV).SMILES) + list(pd.read_csv(hb.TEST_CSV).SMILES)).blocks["morgan_bits"]
    matcher = hb.SimilarityMatcher(bits, plan.external)
    prevalence = float(hb.estimate_from_csv(f"data/tdi_track/inputs/leaderboard_snapshots/interim_{iso}_2026-10-04.csv")["prevalence"])
    selectors = [hb.SubsetStack(hb.SETS["S0 hits-only"], "hits_only"), hb.SubsetStack(names, "selection_aware"), hb.BlendOfStacks(hb.SETS["S0 hits-only"], names)]
    units = collect_outer(cache, ctx, matcher, prevalence, names, FIT_WEIGHT, selectors)
    rate = prevalence * MULTIPLIER
    table = unit_metrics(units, rate)
    distributions, n_diff = live_distributions(pooled_by_repeat(units), rate)
    evidence = {"pac": pac_estimate((LOGS / "test_nonhit_share_CYP2D6.log").read_text()), "band": board_band(pd.read_csv(BOARD_BAND)), "rate": rate}
    decision = decide(table, distributions)
    print_tables(table, evidence, distributions, n_diff, decision)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps({**decision, "fit_weight": FIT_WEIGHT, "half": HALF, "resamples": RESAMPLES, "evidence": evidence, "differing_calls_in_a_half": n_diff}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
