"""Why an idea that did not join the final stack added nothing: diagnostics generated from the cached predictions.

A verdict "NOT MET" says that an idea did not beat the final stack under the pre-registered rule; it does not say why. For every candidate group of ``final_tdi_build.GROUPS`` and each isoform this
module measures four things with the machinery of the gates themselves (same nested outer folds, same stack selector, same similarity, prevalence and confidence weights, same Nadeau-Bengio corrected t):

  * the paired change in outer AUC when the group's candidates are added to the default stack, its standard error and corrected p value (the quantity of the gate);
  * the redundancy: the mean and the largest absolute Spearman correlation between a candidate's predictions and the default stack's predictions on the held-out outer rows, within each fold;
  * the weight the stack gave the group when it was allowed to use it: the share of the absolute logistic coefficients that falls on the group's candidates, averaged over the outer folds;
  * the smallest effect the design could detect, ``(t(0.95, J-1) + t(0.80, J-1)) x SE``: a one-sided test at the 5% level with 80% power, from the corrected standard error of the same paired
    differences. A change smaller than it cannot be told from zero by this design, whatever its sign.

Five more measures come from the treatment of inferential statistics in Janczyk and Pfister, Understanding Inferential Statistics (Springer, 2023), and are descriptive, not gates:

  * the 90% confidence interval of the paired change (the corrected standard error and the t quantile of the J - 1 degrees of freedom; section 6.3.2) and, from it, the equivalence test (two
    one-sided tests, "TOST", section 7.5 and the equivalence tests of Lakens 2017 that the book cites): a non-significant result does not show that an idea adds nothing, only an interval that
    lies inside +-``EQUIVALENCE_MARGIN`` AUC does; an interval wider than the margin is "inconclusive";
  * Cohen's d_z of the paired differences, the mean difference over the standard deviation of the differences (section 7.2);
  * Shapiro-Wilk p value of the paired differences, the check of the normality assumption of the t test (section 5.1.4); with 15 differences it can only flag a gross violation;
  * the design floor: the corrected standard error contains the term n_test / n_train that no number of repeats removes, so even infinitely many repeats leave a smallest detectable effect of
    (t(0.95) + t(0.80)) x sqrt(n_test / n_train x variance); a change below it can only be resolved by a design with more folds (the 10-fold replicate);
  * the partial correlation of the group's predictions with the label given the stack's score, r(x, y . z) = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2)) on ranks (section 11.4.2): the
    information the group carries about the label that the stack does not already carry.

A reason is then assigned by fixed rules, in this order, and every clause carries its number: no weight, redundant, lowered the AUC, smaller than the detectable effect. A group whose change is positive,
larger than the detectable effect and still not significant is reported as "no reason found in these four measures". The verdict of the equivalence test is a separate column.

Over the whole ledger the module also reports how the corrected p values are distributed: if no idea carried a real effect, about 5% of them would fall below 0.05 and they would otherwise be spread
evenly (the p value is uniform under the null hypothesis, section 12.4); the corrected test is conservative, so they pile up towards 1 instead. The count is a description of the ledger, not a test of it (the
rows are not independent).

Usage: python -m cypmode.challenge.why_no_gain [--out-dir data/tdi_track/outputs/final]
Outputs: why_no_gain.csv and why_no_gain.md (a group already in the final stack is listed as such and not diagnosed).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cypmode.challenge import final_tdi_build as final
from cypmode.challenge.feature_store import FeatureStore
from cypmode.challenge.label_confidence import confident_weights
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes
from cypmode.challenge.nested import (
    OUTER_REFIT,
    BestSingle,
    NestedPlan,
    PredictionCache,
    StackedLogit,
    _weights,
    inner_oof,
    nested_evaluate,
)
from cypmode.challenge.submission import load_test_set
from cypmode.challenge.validation_matching import SimilarityMatcher

NO_WEIGHT = 0.05  # a group that receives less than this share of the stack's coefficients got "almost no weight"
REDUNDANT = 0.80  # a mean absolute rank correlation with the stack at or above this is "largely redundant"
ALPHA, POWER = 0.05, 0.80
EQUIVALENCE_MARGIN = 0.01  # AUC: the smallest change that would matter for the final stack (the corrected design cannot resolve less than about 0.01 at 5 outer folds)


def detectable_effect(standard_error: float, n_differences: int) -> float:
    """Smallest true difference a one-sided test at ALPHA detects with probability POWER, given the corrected standard error of the paired differences."""
    df = n_differences - 1
    return float((stats.t.ppf(1 - ALPHA, df) + stats.t.ppf(POWER, df)) * standard_error)


def design_floor(standard_error: float, n_differences: int, n_train: int, n_test: int) -> float:
    """Smallest detectable effect with infinitely many repeats: the term 1 / J of the corrected variance vanishes, n_test / n_train remains."""
    variance = standard_error**2 / (1.0 / n_differences + n_test / n_train)
    return float((stats.norm.ppf(1 - ALPHA) + stats.norm.ppf(POWER)) * np.sqrt(n_test / n_train * variance))


def equivalence(delta: float, standard_error: float, n_differences: int, margin: float = EQUIVALENCE_MARGIN) -> tuple[float, float, str]:
    """90% interval of the paired change and the verdict of the two one-sided tests against +-margin: ``equivalent``, ``inconclusive`` or ``worse`` (the whole interval below -margin)
    or ``better`` (the whole interval above +margin)."""
    half = stats.t.ppf(1 - ALPHA, n_differences - 1) * standard_error
    low, high = delta - half, delta + half
    if low > margin:
        return low, high, "better"
    if high < -margin:
        return low, high, "worse"
    return low, high, "equivalent" if (low > -margin and high < margin) else "inconclusive"


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial correlation of x and y with z removed, on ranks for x and z (scores are not linear in each other) and as is for the 0/1 label y."""
    rx, rz = stats.rankdata(x), stats.rankdata(z)
    rxy, rxz, ryz = np.corrcoef(rx, y)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(y, rz)[0, 1]
    denominator = np.sqrt(max(1.0 - rxz**2, 1e-12) * max(1.0 - ryz**2, 1e-12))
    return float((rxy - rxz * ryz) / denominator)


def partial_signal(cache: PredictionCache, names: list[str], reference: dict[tuple[int, int], np.ndarray]) -> float:
    """Mean over the candidates and the outer folds of the partial correlation of a candidate's score with the label given the default stack's score."""
    plan, values = cache.plan, []
    for repeat, k in plan.outer_folds():
        refit = plan.task(repeat, k, OUTER_REFIT)
        y = plan.labels(refit.evaluate)
        values += [partial_correlation(cache.get(n, refit), y, reference[(repeat, k)]) for n in names]
    return float(np.mean(values))


def default_stack_scores(cache: PredictionCache, base: list[str], weight_fn, prevalence: float) -> dict[tuple[int, int], np.ndarray]:
    """The default stack's scores on the held-out outer rows of every (repeat, outer fold), fitted on the inner out-of-fold predictions exactly as ``nested_evaluate`` does."""
    plan, out = cache.plan, {}
    for repeat, k in plan.outer_folds():
        rows, oof = inner_oof(cache, base, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        y_in = plan.labels(rows)
        chosen = StackedLogit().fit(base, oof, y_in, _weights(y_in, rows, refit.train, weight_fn, prevalence), "auc")
        out[(repeat, k)] = chosen.predict(np.column_stack([cache.get(c, refit) for c in base]))
    return out


def redundancy(cache: PredictionCache, names: list[str], reference: dict[tuple[int, int], np.ndarray]) -> tuple[float, float]:
    """Mean and largest absolute Spearman correlation between each candidate and the default stack on the held-out outer rows (per fold, then averaged)."""
    plan, per_candidate = cache.plan, {n: [] for n in names}
    for repeat, k in plan.outer_folds():
        refit = plan.task(repeat, k, OUTER_REFIT)
        for n in names:
            rho = stats.spearmanr(cache.get(n, refit), reference[(repeat, k)]).statistic
            per_candidate[n].append(abs(float(rho)) if np.isfinite(rho) else 0.0)
    means = np.array([np.mean(v) for v in per_candidate.values()])
    return float(means.mean()), float(means.max())


def weight_share(cache: PredictionCache, base: list[str], names: list[str], weight_fn, prevalence: float) -> float:
    """Share of the absolute coefficients of the stack over base + group that falls on the group's candidates, averaged over the outer folds."""
    plan, shares, everyone = cache.plan, [], base + names
    for repeat, k in plan.outer_folds():
        rows, oof = inner_oof(cache, everyone, repeat, k)
        refit = plan.task(repeat, k, OUTER_REFIT)
        y_in = plan.labels(rows)
        chosen = StackedLogit().fit(everyone, oof, y_in, _weights(y_in, rows, refit.train, weight_fn, prevalence), "auc").chosen
        total = sum(abs(v) for v in chosen.values())
        shares.append(sum(abs(chosen.get(n, 0.0)) for n in names) / total if total > 0 else 0.0)
    return float(np.mean(shares))


def reason(row: pd.Series) -> str:
    parts = []
    if row.weight_share < NO_WEIGHT:
        parts.append(f"the stack gave it almost no weight ({row.weight_share:.1%} of the coefficients)")
    if row.redundancy_mean >= REDUNDANT:
        parts.append(f"its predictions are largely redundant with the stack (mean absolute rho {row.redundancy_mean:.2f}, largest {row.redundancy_max:.2f})")
    if row.delta_auc < 0:
        parts.append(f"it lowered the AUC by {abs(row.delta_auc):.4f}")
    if abs(row.delta_auc) < row.detectable_effect:
        parts.append(f"the change ({row.delta_auc:+.4f}) is smaller than the smallest effect this design can detect ({row.detectable_effect:.4f})")
    return "; ".join(parts) if parts else "no reason found in these four measures"


def diagnose(iso: str, gates: dict, store: FeatureStore, only: set[str] | None = None) -> pd.DataFrame:
    root = Path(final.DESIGN["root"]) / iso
    ctx, plan = final.build_context_and_plan(final.TRAIN_CSV, final.TEST_CSV, iso, final.STORE, final.REPEATS, final.DESIGN["outer"], 4, 0, 4, None)
    saved = NestedPlan.load(root / "plan.npz")
    assert saved.plan_id == plan.plan_id, "the plan changed: the diagnostics must share the experiment caches"
    cache = PredictionCache(root / "cache", saved)
    matcher = SimilarityMatcher(store.blocks["morgan_bits"], plan.external)
    prevalence = float(estimate_from_csv(final.BOARDS[iso])["prevalence"])
    base = [c.name for c in final.default_candidates() if not cache.missing(c.name, include_deploy=False)]
    in_final = set(final.survivors(gates, iso))
    n_train, n_test = plan_sizes(plan)
    shared = {}  # what depends on the default stack and the confidence cut only: computed once per cut, not once per group
    for z0 in final.Z_CUTS:
        weight_fn = confident_weights(matcher, ctx.confidence, z0)
        shared[z0] = (weight_fn, nested_evaluate(cache, base, [StackedLogit()], weight_fn, prevalence, "auc").set_index(["repeat", "outer"]), default_stack_scores(cache, base, weight_fn, prevalence))
    rows = []
    for group, (builder, _) in final.GROUPS.items():
        if only and group not in only:
            continue
        names = [c.name for c in builder() if c.name and not cache.missing(c.name, include_deploy=False)]
        if not names:
            logging.info("%s %s: no cached predictions, skipped", iso, group)
            continue
        for z0 in final.Z_CUTS:
            weight_fn, b, reference = shared[z0]
            a = nested_evaluate(cache, base + names, [StackedLogit()], weight_fn, prevalence, "auc").set_index(["repeat", "outer"])
            alone = nested_evaluate(cache, names, [BestSingle()], weight_fn, prevalence, "auc")
            diff = (a["outer_score"] - b["outer_score"]).to_numpy()
            comparison = corrected_resampled_t(diff, n_train, n_test)
            rho_mean, rho_max = redundancy(cache, names, reference)
            low, high, verdict_equivalence = equivalence(comparison.mean_difference, comparison.standard_error, comparison.n_folds)
            rows.append({
                "isoform": iso, "group": group, "candidates": len(names), "z0": z0, "in_final_stack": group in in_final, "default_auc": float(b["outer_score"].mean()), "with_group_auc": float(a["outer_score"].mean()),
                "delta_auc": comparison.mean_difference, "standard_error": comparison.standard_error, "corrected_p": comparison.p_value, "detectable_effect": detectable_effect(comparison.standard_error, comparison.n_folds),
                "group_alone_auc": float(alone["outer_score"].mean()), "redundancy_mean": rho_mean, "redundancy_max": rho_max, "weight_share": weight_share(cache, base, names, weight_fn, prevalence),
                "ci90_low": low, "ci90_high": high, "equivalence": verdict_equivalence, "cohens_dz": float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0,
                "shapiro_p": float(stats.shapiro(diff).pvalue) if len(diff) >= 3 and diff.std() > 0 else float("nan"),
                "design_floor": design_floor(comparison.standard_error, comparison.n_folds, n_train, n_test), "partial_r_label": partial_signal(cache, names, reference),
            })
        logging.info("%s %s done", iso, group)
    table = pd.DataFrame(rows)
    table["reason"] = table.apply(lambda r: "joined the final stack" if r.in_final_stack else reason(r), axis=1)
    return table


def p_value_summary(table: pd.DataFrame) -> str:
    """How the corrected p values of the ideas that did not join are distributed (a description of the ledger; the rows are not independent)."""
    p = table[~table.in_final_stack].corrected_p.to_numpy()
    if len(p) == 0:
        return ""
    below = int((p < ALPHA).sum())
    deciles = np.histogram(p, bins=np.linspace(0, 1, 6))[0]
    return (f"Corrected p values of the {len(p)} comparisons of ideas that did not join: {below} below {ALPHA} (about {ALPHA * len(p):.1f} would be expected if none carried an effect and the test were exact); counts in the five "
            f"equal bins of [0, 1]: {', '.join(str(int(c)) for c in deciles)}. The corrected test is conservative, so the counts rise towards 1.")


def markdown(table: pd.DataFrame) -> str:
    lines = ["# Why the ideas that did not join added nothing (generated by `cypmode/challenge/why_no_gain.py`)", "",
             "Every row compares the default stack with the default stack plus one group of candidates, on the nested outer folds, with the weights and the corrected resampled t of the gates. "
             f"The detectable effect is the smallest true AUC difference that a one-sided test at {ALPHA:.0%} detects with {POWER:.0%} power at the corrected standard error; the design floor is the same quantity with "
             f"infinitely many repeats. Redundancy is the mean absolute Spearman correlation between the group's predictions and the default stack's on held-out rows. Weight is the share of the stack's absolute "
             f"coefficients that falls on the group. The equivalence verdict compares the 90% interval of the change with +-{EQUIVALENCE_MARGIN} AUC: *equivalent* means the idea adds nothing that matters, "
             "*inconclusive* means the design cannot tell.", ""]
    for iso in sorted(table.isoform.unique()):
        lines += [f"## {iso}", "", "| group | z0 | candidates | change in AUC | 90% interval | corrected p | equivalence | d_z | detectable effect | design floor | partial r | redundancy | weight | reason |",
                  "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for r in table[table.isoform == iso].itertuples():
            lines.append(f"| {r.group} | {r.z0} | {r.candidates} | {r.delta_auc:+.4f} | {r.ci90_low:+.4f} to {r.ci90_high:+.4f} | {r.corrected_p:.3f} | {r.equivalence} | {r.cohens_dz:+.2f} | {r.detectable_effect:.4f} | "
                         f"{r.design_floor:.4f} | {r.partial_r_label:+.3f} | {r.redundancy_mean:.2f} | {r.weight_share:.1%} | {r.reason} |")
        lines += ["", p_value_summary(table[table.isoform == iso]), ""]
    shapiro = table.shapiro_p.dropna()
    if len(shapiro):
        lines += [f"Shapiro-Wilk p of the paired differences is below {ALPHA} in {int((shapiro < ALPHA).sum())} of {len(shapiro)} comparisons; the t test assumes normally distributed differences.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/tdi_track/outputs/final")
    parser.add_argument("--groups", help="comma-separated groups to diagnose (default: all)")
    parser.add_argument("--isoforms", default="CYP3A4,CYP2D6")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_dir = Path(args.out_dir)
    train, test = pd.read_csv(final.TRAIN_CSV), load_test_set(final.TEST_CSV)
    store = FeatureStore.load_or_build(final.STORE, list(train["SMILES"]) + list(test["SMILES"]))
    gates = final.parse_gates()
    only = set(args.groups.split(",")) if args.groups else None
    table = pd.concat([diagnose(iso, gates, store, only) for iso in args.isoforms.split(",")], ignore_index=True)
    table.to_csv(out_dir / "why_no_gain.csv", index=False)
    (out_dir / "why_no_gain.md").write_text(markdown(table))
    print((out_dir / "why_no_gain.md").read_text())


if __name__ == "__main__":
    main()
