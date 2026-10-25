# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VIII (second part): turning scores into calls (8.4), calibration (8.5), the candidate files and their checks (8.6), the results with their uncertainty (8.7)."""

from notebooks.notebook_builder import code, figure, md

CALL_RULES = '''
# Two ways of turning the stack's scores into calls. The stack gives every compound a score. The classical way freezes a score threshold on the training part (the one that maximises the weighted MCC of the inner
# out-of-fold scores, smoothed over bootstrap resamples) and calls the validation compounds above it. The project's way freezes the *share* of compounds that threshold calls on the inner scores and calls that share of the
# validation compounds, the highest-scoring ones (call_top_fraction). On the 15 outer folds of the CYP3A4 plan the table gives what share of the validation compounds each rule calls, and the weighted MCC each earns
# (similarity x positive rate x confident rows at z0 = 1.0, section 6.4).
from cypmode.challenge.nested import BestSingle, best_threshold, call_top_fraction

stack_table = nested_evaluate(cache8, names8, [StackedLogit()], weights8(1.0), prevalence8, "auc")
call_summary = pd.DataFrame([{"rule": "score threshold frozen on the inner scores", "share of validation compounds called, lowest": stack_table["called_rate_threshold"].min(), "highest": stack_table["called_rate_threshold"].max(),
                              "sd over the folds": stack_table["called_rate_threshold"].std(), "mean weighted MCC": stack_table["outer_mcc"].mean()},
                             {"rule": "share of calls frozen on the inner scores (the project's)", "share of validation compounds called, lowest": stack_table["called_rate_transfer"].min(), "highest": stack_table["called_rate_transfer"].max(),
                              "sd over the folds": stack_table["called_rate_transfer"].std(), "mean weighted MCC": stack_table["outer_mcc_rate"].mean()}])
show_table(call_summary, floatfmt=".3f")
threshold_rule, rate_rule = call_summary.iloc[0], call_summary.iloc[1]
rate_vs_threshold = stack_table["outer_mcc_rate"] - stack_table["outer_mcc"]
n_rate_better = int((rate_vs_threshold > 0).sum())
'''

SMOOTH = '''
# Smoothing the threshold. The weighted MCC of a ranker, as a function of the threshold, is made of a few thousand compounds and can be jagged; best_threshold (cypmode/challenge/nested.py) averages the curve over 40
# bootstrap resamples of the compounds before it takes the maximum. For all 15 outer folds of both enzymes the cell computes the curve of the inner out-of-fold stack scores, raw and averaged, the threshold at the maximum of each,
# and the share of compounds each threshold calls. The figure draws the CYP2D6 fold in which the two thresholds differ most, the weak ranker for which the argument is made.
from cypmode.challenge.nested import _weights
from cypmode.challenge.validation_matching import weighted_mcc

def threshold_curves(plan_i, cache_i, matcher_i, prevalence_i, z_i, repeat, k):
    rows_i, oof_i = inner_oof(cache_i, names8, repeat, k)
    refit_i = plan_i.task(repeat, k, OUTER_REFIT)
    y_i = plan_i.labels(rows_i)
    w_i = _weights(y_i, rows_i, refit_i.train, confident_weights(matcher_i, z_i, 1.0), prevalence_i)
    meta = np.asarray(StackedLogit().fit(names8, oof_i, y_i, w_i, "auc").inner_scores)
    grid = np.unique(np.quantile(meta, np.linspace(0.02, 0.98, 60)))
    rng = np.random.default_rng(0)
    raw = np.array([weighted_mcc(y_i, (meta >= t).astype(int), w_i) for t in grid])
    boots = [np.array([weighted_mcc(y_i[ii], (meta[ii] >= t).astype(int), w_i[ii]) for t in grid]) for ii in (rng.integers(0, len(y_i), len(y_i)) for _ in range(40))]
    return meta, grid, raw, (raw + np.sum(boots, axis=0)) / (1 + len(boots))

smooth_rows, kept = [], {}
for iso, (plan_i, cache_i, matcher_i, prevalence_i, z_i) in setups8.items():
    for repeat, k in plan_i.outer_folds():
        meta, grid, raw, smooth = threshold_curves(plan_i, cache_i, matcher_i, prevalence_i, z_i, repeat, k)
        share_raw, share_smooth = float((meta >= grid[raw.argmax()]).mean()), float((meta >= grid[smooth.argmax()]).mean())
        roughness = float(np.abs(np.diff(raw)).mean() / max(np.abs(np.diff(smooth)).mean(), 1e-9))
        smooth_rows.append({"enzyme": iso, "fold": f"{repeat}-{k}", "share called at the raw maximum": share_raw, "share called at the smoothed maximum": share_smooth, "roughness ratio": roughness})
        kept[(iso, f"{repeat}-{k}")] = (grid, raw, smooth)
smooth_table = pd.DataFrame(smooth_rows)
smooth_table["difference"] = (smooth_table["share called at the raw maximum"] - smooth_table["share called at the smoothed maximum"]).abs()
smooth_summary = smooth_table.groupby("enzyme").agg(**{"folds in which the two maxima differ": ("difference", lambda v: int((v > 0).sum())), "mean change of the share called": ("difference", "mean"),
                                                         "largest change of the share called": ("difference", "max")}).reset_index()
show_table(smooth_summary, floatfmt=".3f")
worst = smooth_table[smooth_table["enzyme"] == "CYP2D6"].sort_values("difference", ascending=False).iloc[0]
grid_w, raw_w, smooth_w = kept[("CYP2D6", worst["fold"])]
fig, axis = plt.subplots(figsize=(7.4, 3.6))
axis.plot(grid_w, raw_w, color=theme["muted"], lw=1.8, label="weighted MCC, one sample")
axis.plot(grid_w, smooth_w, color=theme["blue"], lw=2.6, label="average over 40 bootstrap resamples")
axis.axvline(grid_w[raw_w.argmax()], color=theme["muted"], ls=":", lw=1.2)
axis.axvline(grid_w[smooth_w.argmax()], color=theme["blue"], ls="--", lw=1.2)
axis.set(xlabel="threshold on the stack's score (inner out-of-fold)", ylabel="weighted MCC", title=f"CYP2D6, fold {worst['fold']}: the threshold curve")
axis.legend(frameon=False, fontsize=8)
tidy(axis, theme)
fig.tight_layout()
smooth_3a4, smooth_2d6 = (smooth_summary[smooth_summary["enzyme"] == e].iloc[0] for e in ("CYP3A4", "CYP2D6"))
n_folds_total = len(smooth_table[smooth_table["enzyme"] == "CYP3A4"])
'''

MULTIPLIER = '''
# The call rate of the final model. The final build (cypmode/challenge/final_tdi_build.py) calls a share of the 750 blinded compounds equal to a multiple of the scored set's positive rate, the multiple chosen from
# {1.0, 1.1, 1.2, 1.3} by the highest mean outer-fold weighted MCC over both confidence cut-offs. The tables it wrote are read here (per-fold scores of the stack refitted inside every outer fold), with the choice it made.
import json as json_module

from cypmode.challenge.tdi_baseline import TDI_ISOFORMS

multiplier_rows, chosen_multiplier = [], {}
manifest_path = ROOT / "data/tdi_track/outputs/final/manifest.json"
manifest = json_module.loads(manifest_path.read_text())
for iso in TDI_ISOFORMS:
    table = pd.read_csv(ROOT / f"data/tdi_track/outputs/final/{iso}_call_rate_choice.csv")
    by_multiplier = table.groupby("multiplier")["mean_outer_mcc"].mean()
    for multiplier, value in by_multiplier.items():
        multiplier_rows.append({"enzyme": iso, "call rate as a multiple of the positive rate": multiplier, "mean outer MCC over both cut-offs": float(value),
                                **{f"MCC at z0 {z0}": float(table[(table["multiplier"] == multiplier) & (table["z0"] == z0)]["mean_outer_mcc"].iloc[0]) for z0 in (1.0, 1.96)}})
    chosen_multiplier[iso] = float(manifest["isoforms"][iso]["call_rate_multiplier"])
multiplier_table = pd.DataFrame(multiplier_rows)
show_table(multiplier_table, floatfmt=".4f")
prevalence_final = {iso: float(manifest["isoforms"][iso]["target_prevalence"]) for iso in TDI_ISOFORMS}
called_final = {iso: float(manifest["isoforms"][iso]["called_rate"]) for iso in TDI_ISOFORMS}
threshold_rate_final = {iso: float(manifest["isoforms"][iso]["rate_before_cap_rule"]) for iso in TDI_ISOFORMS}
multiplier_gain = {iso: float(multiplier_table[(multiplier_table["enzyme"] == iso)]["mean outer MCC over both cut-offs"].max() - multiplier_table[(multiplier_table["enzyme"] == iso) & (multiplier_table["call rate as a multiple of the positive rate"] == 1.0)]["mean outer MCC over both cut-offs"].iloc[0]) for iso in TDI_ISOFORMS}
'''

CALIBRATION = '''
# Calibration, for both enzymes. For every outer fold the stack is refitted on the inner scores; three maps from its decision value to a probability are fitted on the *inner* out-of-fold decision values only and applied to
# the outer-validation compounds (cypmode/challenge/calibration_experiment.py): "raw" (the logistic function of the decision value, what the stack gives), "platt" (a one-feature logistic regression on the inner decision values) and
# "isotonic" (a monotone step function fitted on them). The pooled outer predictions of the 15 folds are scored with the same weights as the rest of the evaluation: the expected calibration error (ECE, ten equal-width bins),
# the Brier score and the calibration slope.
from cypmode.challenge.calibration_experiment import calibration_metrics, fit_calibrator
from cypmode.challenge.nested import _weights

calibration_rows, reliability_points = [], {}
for iso, (plan_i, cache_i, matcher_i, prevalence_i, z_i) in setups8.items():
    collected = {m: ([], [], []) for m in ("raw", "platt", "isotonic")}
    weight_fn = confident_weights(matcher_i, z_i, 1.0)
    for repeat, k in plan_i.outer_folds():
        rows_i, oof_i = inner_oof(cache_i, names8, repeat, k)
        refit_i = plan_i.task(repeat, k, OUTER_REFIT)
        outer_i = np.column_stack([cache_i.get(n, refit_i) for n in names8])
        y_in_i, y_out_i = plan_i.labels(rows_i), plan_i.labels(refit_i.evaluate)
        w_in_i, w_out_i = _weights(y_in_i, rows_i, refit_i.train, weight_fn, prevalence_i), _weights(y_out_i, refit_i.evaluate, refit_i.train, weight_fn, prevalence_i)
        chosen_i = StackedLogit().fit(names8, oof_i, y_in_i, w_in_i, "auc")
        for method, (ys, ps, ws) in collected.items():
            calibrator = fit_calibrator(method, np.asarray(chosen_i.inner_scores), y_in_i, w_in_i)
            ys.append(y_out_i); ps.append(np.clip(calibrator(chosen_i.predict(outer_i)), 1e-4, 1 - 1e-4)); ws.append(w_out_i)
    for method, (ys, ps, ws) in collected.items():
        y_all, p_all, w_all = np.concatenate(ys), np.concatenate(ps), np.concatenate(ws)
        metrics = calibration_metrics(y_all, p_all, w_all)
        calibration_rows.append({"enzyme": iso, "map": method, **{k: metrics[k] for k in ("ece", "brier", "slope", "mean_pred", "mean_obs")}, "gate met": bool(metrics["ece"] <= 0.05 and 0.8 <= metrics["slope"] <= 1.2)})
        reliability_points[(iso, method)] = (y_all, p_all, w_all)
calibration_table = pd.DataFrame(calibration_rows)
show_table(calibration_table, floatfmt=".4f")
fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)
for axis, iso in zip(axes, ("CYP3A4", "CYP2D6")):
    for method, colour in (("raw", theme["red"]), ("platt", theme["orange"]), ("isotonic", theme["blue"])):
        y_all, p_all, w_all = reliability_points[(iso, method)]
        index = np.minimum((p_all * 10).astype(int), 9)
        xs, ys_ = zip(*[(np.average(p_all[index == b], weights=w_all[index == b]), np.average(y_all[index == b], weights=w_all[index == b])) for b in range(10) if (index == b).any()])
        axis.plot(xs, ys_, "o-", color=colour, lw=1.8, ms=4, label=method)
    axis.plot([0, 1], [0, 1], color=theme["muted"], ls=":", lw=1.2)
    axis.set(xlabel="predicted probability (mean in the bin)", title=iso)
axes[0].set_ylabel("observed share of positives (weighted)")
axes[0].legend(frameon=False, fontsize=8)
for axis in axes:
    tidy(axis, theme)
fig.tight_layout()
def calibration_value(iso, method, column):
    return float(calibration_table[(calibration_table["enzyme"] == iso) & (calibration_table["map"] == method)][column].iloc[0])
'''

PAV = '''
# Isotonic regression by hand. The pool-adjacent-violators algorithm (PAV) fits the monotone non-decreasing step function that is closest, in squared error, to a set of labels ordered by score: walk along the
# compounds from the lowest score up, keep a block for each run of compounds, and whenever a block's mean is higher than the next block's, merge the two into one block with the weighted mean of both. What remains is a
# staircase that never goes down. The cell codes it in a dozen lines and checks it against scikit-learn's IsotonicRegression on the inner out-of-fold decision values of the first outer fold of CYP3A4.
def pool_adjacent_violators(y, weights):
    blocks = []                                   # each block: [weighted sum of y, total weight, number of compounds]
    for value, weight in zip(y, weights):
        blocks.append([value * weight, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            last = blocks.pop()
            blocks[-1] = [blocks[-1][0] + last[0], blocks[-1][1] + last[1], blocks[-1][2] + last[2]]
    return np.concatenate([np.full(count, total / weight) for total, weight, count in blocks])

from sklearn.isotonic import IsotonicRegression

repeat_p, outer_p = plan8.outer_folds()[0]
rows_p, oof_p = inner_oof(cache8, names8, repeat_p, outer_p)
y_p = plan8.labels(rows_p)
meta_p = np.asarray(StackedLogit().fit(names8, oof_p, y_p, None, "auc").inner_scores)
order = np.argsort(meta_p, kind="stable")
by_hand = pool_adjacent_violators(y_p[order].astype(float), np.ones(len(order)))
library = IsotonicRegression().fit(meta_p[order], y_p[order].astype(float)).predict(meta_p[order])
pav_difference = float(np.abs(by_hand - library).max())
pav_blocks = len(np.unique(np.round(by_hand, 12)))
print(f"largest difference between the algorithm written here and the library: {pav_difference:.1e}; the staircase has {pav_blocks} steps for {len(order):,} compounds")
'''

MANIFEST_CHECK = '''
# The candidate files and what is checked about them (cypmode/challenge/final_tdi_build.py, written to data/tdi_track/outputs/final/). For each file: the hash recorded in the manifest against the hash of the file on disk now,
# the portal's own schema check as the project reimplements it (cypmode/challenge/submission.py: the file has the four columns, 750 rows, binary calls, and both enzymes have both classes), and the positive rates.
# The per-compound columns of the files are not shown: they are not public before the challenge closes.
import hashlib

from cypmode.challenge.submission import portal_validate_classification, validate_submission

file_rows = []
for entry in manifest["files"]:
    path = ROOT / "data/tdi_track/outputs/final" / entry["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    ok, message = portal_validate_classification(path)
    frame = pd.read_csv(path)
    file_rows.append({"file": entry["file"], "rows": len(frame), "sha256 matches the manifest": digest == entry["sha256"], "portal check": "passes" if ok else f"fails: {message}", "CYP3A4 positive rate": float(frame["CYP3A4_is_TDI"].mean()),
                      "CYP2D6 positive rate": float(frame["CYP2D6_is_TDI"].mean()), "sha256 (first 12)": digest[:12]})
file_table = pd.DataFrame(file_rows)
show_table(file_table, floatfmt=".4f")
audit_rows = [{"enzyme": iso, "check": check, "passes": bool(passed)} for iso in TDI_ISOFORMS for check, passed in manifest["isoforms"][iso]["leakage_audit"].items()]
show_table(pd.DataFrame(audit_rows))
environment_line = ", ".join(f"{k} {v}" for k, v in manifest["environment"].items() if k in ("python", "numpy", "scikit-learn", "lightgbm", "rdkit"))
all_hashes_match, all_portal_ok, all_audit_ok = bool(file_table["sha256 matches the manifest"].all()), bool((file_table["portal check"] == "passes").all()), bool(pd.DataFrame(audit_rows)["passes"].all())
'''

RESULTS8 = '''
# The nested estimate of the final stack, with its spread. For each enzyme and each cut-off z0, the 15 outer-fold scores of the stack that the final build uses (the stack named in the manifest, evaluated with the weights of 6.4):
# the AUC and the MCC at the transferred call rate. The mean, the standard deviation over the folds, and the 95 % interval from the corrected resampled t (section 6.5), whose standard error is s * sqrt(1/J + n_test/n_train)
# and is therefore wider than the naive one. The numbers of the manifest (the build that wrote the candidate files) are set beside them.
from scipy import stats

from cypmode.challenge.model_comparison import corrected_resampled_t, plan_sizes

result_rows = []
for iso, (plan_i, cache_i, matcher_i, prevalence_i, z_i) in setups8.items():
    stack_names = [c.name for c in default_candidates() if not cache_i.missing(c.name, False)]
    n_train_i, n_test_i = plan_sizes(plan_i)
    for z0 in (1.0, 1.96):
        table = nested_evaluate(cache_i, stack_names, [StackedLogit()], confident_weights(matcher_i, z_i, z0), prevalence_i, "auc")
        for metric_name, column in (("AUC", "outer_score"), ("MCC at the transferred rate", "outer_mcc_rate")):
            values = table[column].to_numpy()
            standard_error = float(values.std(ddof=1) * np.sqrt(1 / len(values) + n_test_i / n_train_i))
            half = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
            result_rows.append({"enzyme": iso, "z0": z0, "score": metric_name, "mean over 15 folds": float(values.mean()), "sd over the folds": float(values.std(ddof=1)), "95 % interval": f"{values.mean() - half:.3f} to {values.mean() + half:.3f}", "members": len(stack_names)})
result_table = pd.DataFrame(result_rows)
show_table(result_table, floatfmt=".3f")
# Is the blend better than the alternatives, in the same folds? The paired difference of the outer AUC, fold by fold, between the stack and (a) the member that the BestSingle selector picks on the inner scores, (b) the
# plain LightGBM on the descriptors alone, the closest member to the organisers' baseline; the corrected resampled t of 6.5 with the sizes of the plan.
comparison_rows = []
for iso, (plan_i, cache_i, matcher_i, prevalence_i, z_i) in setups8.items():
    stack_names = [c.name for c in default_candidates() if not cache_i.missing(c.name, False)]
    n_train_i, n_test_i = plan_sizes(plan_i)
    for z0 in (1.0, 1.96):
        weight_fn = confident_weights(matcher_i, z_i, z0)
        both = nested_evaluate(cache_i, stack_names, [StackedLogit(), BestSingle()], weight_fn, prevalence_i, "auc")
        stack_auc = both[both["selector"] == "stack_logit"].set_index(["repeat", "outer"])["outer_score"]
        best_auc = both[both["selector"] == "best_single"].set_index(["repeat", "outer"])["outer_score"]
        desc = nested_evaluate(cache_i, ["lgbm_desc_deep"], [BestSingle()], weight_fn, prevalence_i, "auc").set_index(["repeat", "outer"])["outer_score"]
        for label, other in (("the member picked on the inner scores", best_auc), ("LightGBM on the descriptors alone", desc)):
            result = corrected_resampled_t((stack_auc - other).to_numpy(), n_train_i, n_test_i)
            comparison_rows.append({"enzyme": iso, "z0": z0, "stack against": label, "AUC of the stack": float(stack_auc.mean()), "AUC of the other": float(other.mean()), "gain": result.mean_difference, "folds in which the stack is better": f"{result.n_better} of {result.n_folds}", "p (corrected t)": result.p_value})
comparison_table = pd.DataFrame(comparison_rows)
show_table(comparison_table, floatfmt=".3f")
n_best_significant = int(((comparison_table["stack against"] == "the member picked on the inner scores") & (comparison_table["p (corrected t)"] < 0.05)).sum())
def result_value(iso, z0, score, column="mean over 15 folds"):
    return float(result_table[(result_table["enzyme"] == iso) & (result_table["z0"] == z0) & (result_table["score"] == score)][column].iloc[0])
final_stack_names = {iso: manifest["isoforms"][iso]["stack"] for iso in TDI_ISOFORMS}
n_members_final = {iso: int(manifest["isoforms"][iso]["n_candidates"]) for iso in TDI_ISOFORMS}
'''

APPLICABILITY = r"""
# The applicability domain of the final stack (cypmode/challenge/applicability_domain.py, queue job 096): where the model interpolates, measured on the outer folds of the nested validation. Each held-out compound
# gets its nearest-training similarity (ECFP4 Tanimoto) to the rows its fold was trained on; the held-out compounds are cut into similarity bins, each bin gets an AUC with a 95 percent interval whose bootstrap
# resamples whole chemical series (Butina clusters, so that series are not counted as independent observations), and each of the 750 blinded compounds is given the tier of the bin it falls in. No model is refitted.
# The cell reads the two tables the module wrote and counts the tiers of the blinded compounds.
applicability_compounds = pd.read_csv(ROOT / "data/tdi_track/outputs/final/applicability_domain_compounds.csv")
applicability_bins = pd.read_csv(ROOT / "data/tdi_track/outputs/final/applicability_domain_bins.csv")
applicability_rows = []
for iso in TDI_ISOFORMS:
    applicability_tiers = applicability_compounds[f"{iso}_domain_tier"].value_counts()
    applicability_similarity = applicability_compounds[f"{iso}_nearest_training_similarity"]
    applicability_bin_rows = applicability_bins[(applicability_bins["isoform"] == iso) & (applicability_bins["z0"] == 1.0)].dropna(subset=["auc"])
    applicability_rows.append({"enzyme": iso, "supported": int(applicability_tiers.get("supported", 0)), "weak": int(applicability_tiers.get("weak", 0)), "unsupported": int(applicability_tiers.get("unsupported", 0)),
                               "median nearest-training similarity": float(applicability_similarity.median()), "share at or above 0.5": float((applicability_similarity >= 0.5).mean()),
                               "lowest bin AUC": float(applicability_bin_rows["auc"].min()), "highest bin AUC": float(applicability_bin_rows["auc"].max()),
                               "lowest bin interval, lower end": float(applicability_bin_rows["auc_low"].min())})
applicability_table = pd.DataFrame(applicability_rows)
show_table(applicability_table, floatfmt=".3f")
n_blinded_compounds = len(applicability_compounds)
n_supported = {row["enzyme"]: row["supported"] for row in applicability_rows}
n_weak = {row["enzyme"]: row["weak"] for row in applicability_rows}
n_unsupported = {row["enzyme"]: row["unsupported"] for row in applicability_rows}
median_similarity = {row["enzyme"]: row["median nearest-training similarity"] for row in applicability_rows}
share_above_half = {row["enzyme"]: row["share at or above 0.5"] for row in applicability_rows}
lowest_auc = {row["enzyme"]: row["lowest bin AUC"] for row in applicability_rows}
highest_auc = {row["enzyme"]: row["highest bin AUC"] for row in applicability_rows}
n_bins = {iso: int(len(applicability_bins[(applicability_bins["isoform"] == iso) & (applicability_bins["z0"] == 1.0)])) for iso in TDI_ISOFORMS}
n_supported_bins = {iso: int((applicability_bins[(applicability_bins["isoform"] == iso) & (applicability_bins["z0"] == 1.0)]["tier"] == "supported").sum()) for iso in TDI_ISOFORMS}
assert not applicability_table.isna().any().any()
assert sum(n_supported.values()) + sum(n_weak.values()) + sum(n_unsupported.values()) == 2 * n_blinded_compounds
"""

CELLS = [
    md("""
## 8.4 Turning scores into calls: a smoothed threshold, a share of calls, and a multiple

> **Field:** Statistics · Machine learning

*Where this section sits:* the stack of 8.2 ranks the compounds, and the submission is a yes or no for each (the challenge scores the MCC of the calls, section 4.10). A ranking becomes calls once it is decided how many compounds, the highest-scoring ones, are called positive. Section 7.5 showed that the decision must be made without the labels of the compounds it is scored on, and that the MCC of a rule depends on the positive rate. This section gives the three decisions that the final model takes and tests each one.

**First decision: a threshold from a jagged curve.** The natural rule is to call positive every compound whose score exceeds a threshold, and to choose the threshold that maximises the weighted MCC (6.4) of the inner out-of-fold scores. The curve of the weighted MCC against the threshold is made of a few thousand compounds, a fifth of them positive, and for a weak ranker it is jagged: its maximum sits where the noise happens to be highest. `best_threshold` averages the curve over 40 bootstrap resamples of the compounds (the bootstrap of section 6.5, drawing compounds with replacement) before it takes the maximum, so that the threshold is the top of the average curve and not of the one sample. The cell draws both curves for the first outer fold.
"""),
    code(SMOOTH),
    md("""
Over the {{=n_folds_total}} outer folds of each enzyme the two maxima coincide in most folds. They differ in {{=smooth_3a4['folds in which the two maxima differ']}} fold{{='s' if smooth_3a4['folds in which the two maxima differ'] != 1 else ''}} of CYP3A4 and in {{=smooth_2d6['folds in which the two maxima differ']}} of CYP2D6, and when they differ they differ a lot: the share of compounds called moves by up to {{=smooth_3a4['largest change of the share called']:.1%}} for CYP3A4 and {{=smooth_2d6['largest change of the share called']:.1%}} for CYP2D6, since the curve then has two separate peaks and one sample's maximum can sit on either of them (the mean change over all folds is {{=smooth_3a4['mean change of the share called']:.1%}} and {{=smooth_2d6['mean change of the share called']:.1%}}). The figure shows the CYP2D6 fold with the largest change. The average of the bootstrap curves picks the peak that holds up in most resamples. Smoothing cannot raise the MCC, since the averaged curve is lower than the raw maximum by construction; what it buys is a threshold, and so a call rate, that is less of an accident of the sample.

**Second decision: transfer a share of calls, not a score.** A threshold on the score is frozen on the inner scores and applied to the outer-validation compounds. The alternative freezes the *share* of compounds that the threshold calls on the inner scores and calls that share of the outer-validation compounds, the highest-scoring ones (`call_top_fraction`). The two rules would be the same if the outer scores were distributed like the inner ones, and the reason to prefer the second is that they are not, in general: validation compounds that sit closer to the training set than the inner folds did get uniformly higher scores (6.8), and a fixed threshold then calls too many. The cell tests the rules on the 15 outer folds of CYP3A4, with the yardstick of 6.4 at $z_0 = 1.0$.
"""),
    code(CALL_RULES),
    md("""
The frozen score threshold calls between {{=threshold_rule['share of validation compounds called, lowest']:.1%}} and {{=threshold_rule['highest']:.1%}} of the validation compounds from fold to fold (standard deviation {{=threshold_rule['sd over the folds']:.3f}}), and the frozen share between {{=rate_rule['share of validation compounds called, lowest']:.1%}} and {{=rate_rule['highest']:.1%}} ({{=rate_rule['sd over the folds']:.3f}}). The weighted MCC is {{=threshold_rule['mean weighted MCC']:.3f}} for the threshold and {{=rate_rule['mean weighted MCC']:.3f}} for the share, and the share is better in {{=n_rate_better}} of the 15 folds. The measured result is that *inside the folds the two rules are equivalent*, which is what the argument predicts: the outer-validation compounds are drawn from the same pool as the inner ones, and their scores have the same distribution. The case for the share is the blinded set, whose compounds are known to sit closer to the training file than a fold's do.

**Third decision: how many to call.** A share of calls equal to the positive rate is not necessarily the best. If the ranking is weak, calling a few more compounds than there are positives recovers positives at a cost in false calls that the MCC tolerates. The project fixed in advance a family of rates, 1.0, 1.1, 1.2 and 1.3 times the positive rate of the scored set, and chose the member of the family with the highest mean outer-fold weighted MCC over both confidence cut-offs, the stack being refitted inside every outer fold (`choose_call_rate` in `cypmode/challenge/final_tdi_build.py`). The cell reads the tables the build wrote, with its choice.
"""),
    code(MULTIPLIER),
    md("""
For CYP3A4 the mean MCC is nearly flat over the family, from {{=multiplier_table[(multiplier_table['enzyme'] == 'CYP3A4')]['mean outer MCC over both cut-offs'].min():.4f}} to {{=multiplier_table[(multiplier_table['enzyme'] == 'CYP3A4')]['mean outer MCC over both cut-offs'].max():.4f}}, and the rule picks the multiple {{=chosen_multiplier['CYP3A4']:.1f}}, which is worth {{=multiplier_gain['CYP3A4']:+.4f}} MCC over calling the positive rate itself. For CYP2D6 the MCC falls steadily with the multiple, from {{=multiplier_table[(multiplier_table['enzyme'] == 'CYP2D6')]['mean outer MCC over both cut-offs'].max():.4f}} at 1.0 to {{=multiplier_table[(multiplier_table['enzyme'] == 'CYP2D6')]['mean outer MCC over both cut-offs'].min():.4f}} at 1.3, and the rule picks {{=chosen_multiplier['CYP2D6']:.1f}}. The differences are small beside the spread of the folds (0.035 for CYP3A4 and 0.13 for CYP2D6, 8.7), so the choice should be read as a pre-declared convention that is not hurt by the data and not as a finding about the optimum. It has the property that matters: it was fixed before the numbers, it is the same procedure for both enzymes, and it can be reproduced from the two tables.

The final build also shows what the second decision buys on the real thing. For CYP3A4 the rate that the inner threshold implies would have called {{=threshold_rate_final['CYP3A4']:.1%}} of the 750 blinded compounds, and for CYP2D6 {{=threshold_rate_final['CYP2D6']:.1%}}, while the share that the build calls is {{=called_final['CYP3A4']:.1%}} and {{=called_final['CYP2D6']:.1%}}. A share of calls is a quantity the project can reason about against the scored set's positive rate ({{=prevalence_final['CYP3A4']:.1%}} and {{=prevalence_final['CYP2D6']:.1%}}, section 2.10); a score threshold is not.

*On what actually ran.* `best_threshold`, `call_top_fraction` and `Selection.rate` are in `cypmode/challenge/nested.py`; `choose_call_rate`, `RATE_MULTIPLIERS` and the construction of the calls from the deployment scores are in `cypmode/challenge/final_tdi_build.py`, which writes the tables `CYP3A4_call_rate_choice.csv` and `CYP2D6_call_rate_choice.csv` into `data/tdi_track/outputs/final/`.
"""),
    md("""
## 8.5 Calibration: from a score to a probability

> **Field:** Statistics

*Where this section sits:* the calls of 8.4 need only the order of the scores. A different use of the model, abstaining on the uncertain compounds, choosing which compound to measure next, the expected-improvement selection of Part XI, needs the score to be a *probability*: when the model says 0.3, three compounds in ten should be positive. The stack does not give that, and this section is about the step that makes it so and the test that decides whether it worked. The step was declared with its gate in advance (`cypmode/challenge/calibration_experiment.py`).

**What a calibrated probability is, and how it is measured.** Put the compounds in ten equal-width bins of predicted probability. In bin $b$, let $\\omega_b$ be the share of the weight (the weights of 6.4) that falls in it, $\\bar p_b$ the mean predicted probability and $\\bar y_b$ the weighted share of positives. A calibrated model has $\\bar p_b \\approx \\bar y_b$ in every bin, and the **expected calibration error** is the weighted gap,

$$\\mathrm{ECE} \\;=\\; \\sum_{b=1}^{10} \\omega_b\\,\\big|\\,\\bar p_b - \\bar y_b\\,\\big|.$$

The *reliability diagram* of the cell below plots $\\bar y_b$ against $\\bar p_b$: a calibrated model lies on the diagonal. Two further numbers are used. The *Brier score* is the mean squared difference between the probability and the label, $\\tfrac1n\\sum_i (p_i - y_i)^2$, lower being better. The *calibration slope* $s$ is the coefficient of a logistic regression of the label on the log-odds of the probability, $\\mathrm{logit}\\,P(y=1) = a + s\\,\\mathrm{logit}(p)$; a calibrated model has $s = 1$ and $a = 0$, a slope below 1 means that the predictions are too extreme (over-confident), above 1 that they are too timid. The project's gate: the adopted map must reach an ECE of at most 0.05 and a slope between 0.8 and 1.2 on both enzymes; an enzyme for which no map does keeps no probability output.

**The three maps.** For every outer fold the stack is refitted, and three maps from its decision value $d$ to a probability are fitted on the *inner* out-of-fold decision values only, and then applied to the outer-validation compounds. *Raw* is the logistic function $\\sigma(d)$, what the stack gives today. **Platt scaling** (Platt, 1999) fits the logistic function to the inner values, $P(y=1) = \\sigma(A\\,d + B)$, two numbers that stretch and shift the decision value. **Isotonic regression** (Zadrozny and Elkan, 2002) fits the monotone non-decreasing step function that is closest in squared error to the labels ordered by score, by the **pool-adjacent-violators algorithm**: walk along the compounds from the lowest score upwards, keep a block for each run, and whenever a block's mean exceeds the next block's, merge the two into one block with the weighted mean of both; what remains is a staircase that never goes down, and it is the nearest monotone fit. It has no shape to impose and needs enough data to fill its steps.
"""),
    code(CALIBRATION),
    md("""
For CYP3A4 the raw probability is close but fails the gate on this cache (ECE {{=calibration_value('CYP3A4', 'raw', 'ece'):.4f}}, slope {{=calibration_value('CYP3A4', 'raw', 'slope'):.2f}}; the record of the day, on an earlier state of the cache, had it just inside), and the mean predicted probability, {{=calibration_value('CYP3A4', 'raw', 'mean_pred'):.3f}}, overshoots the observed share, {{=calibration_value('CYP3A4', 'raw', 'mean_obs'):.3f}}. Platt scaling and isotonic regression both pass: ECE {{=calibration_value('CYP3A4', 'platt', 'ece'):.4f}} and {{=calibration_value('CYP3A4', 'isotonic', 'ece'):.4f}}, slopes {{=calibration_value('CYP3A4', 'platt', 'slope'):.2f}} and {{=calibration_value('CYP3A4', 'isotonic', 'slope'):.2f}}. For CYP2D6 the raw output is not a probability at all: the mean predicted probability is {{=calibration_value('CYP2D6', 'raw', 'mean_pred'):.3f}} where {{=calibration_value('CYP2D6', 'raw', 'mean_obs'):.3f}} of the compounds are positive, the ECE is {{=calibration_value('CYP2D6', 'raw', 'ece'):.3f}} and the slope {{=calibration_value('CYP2D6', 'raw', 'slope'):.2f}}, a line that does not rise (the audit found it: for 80 percent of the compounds the stack's probability was between 0.49 and 0.50; a heavily penalised blend ranks without producing probabilities). Platt scaling brings the ECE to {{=calibration_value('CYP2D6', 'platt', 'ece'):.4f}} but with a slope of {{=calibration_value('CYP2D6', 'platt', 'slope'):.2f}}, outside the gate, because its straight line in the log-odds cannot follow a relation that bends; isotonic regression reaches an ECE of {{=calibration_value('CYP2D6', 'isotonic', 'ece'):.4f}} and a slope of {{=calibration_value('CYP2D6', 'isotonic', 'slope'):.2f}} and is the only map that passes for both enzymes. It is the one the build uses.

The step-function map is simple enough to write out. The cell below codes the pool-adjacent-violators algorithm in a dozen lines and checks it against scikit-learn's `IsotonicRegression` on the inner out-of-fold decision values of the first outer fold of CYP3A4.
"""),
    code(PAV),
    md("""
The algorithm written here and the library agree to {{=pav_difference:.0e}}. The staircase has {{=pav_blocks}} steps for {{=len(order):,}} compounds: the data allow only that many distinct levels, which is also the limit of the map, a probability resolved no finer than its steps. The final build writes the isotonic probability next to the score and the call of each blinded compound in its private score tables (`probability_isotonic`), where a probability-based use can read it; a boolean submission cannot carry it.

*On what actually ran.* `fit_calibrator` and `calibration_metrics` in `cypmode/challenge/calibration_experiment.py` (the run of the module is stored in the log `calibration.log`, which the notebook does not read: the numbers above are recomputed), `calibrated_probabilities` in `final_tdi_build.py` (the isotonic map fitted on the inner out-of-fold stack scores of the whole pool, applied to the blinded compounds' scores), and `sklearn.isotonic.IsotonicRegression` for the library check.
"""),
    figure("ml_calibration_methods"),
    md("""
## 8.6 The candidate files, the checks they pass, and the manifest

> **Field:** Software engineering · Data science

*Where this section sits:* a model is useful to the challenge only as a file of 750 rows. The final build turns the stack and the call rule into files and records everything about how they were made, in a form that a reader can check. It never uploads anything. This section lists what it writes, what is checked, and what the record contains.

**What is written** (to `data/tdi_track/outputs/final/`). There are two candidate files for the A/B plan of the project (section 13.4 gives the rule that chooses the final one). Candidate A is the stack of this part for both enzymes (the "hits-only" stack: trained on the compounds that were promoted to dose-response). Candidate B is the same for CYP3A4 and, for CYP2D6, the selection-aware stack on the panel pool that section 9.1 describes. Beside the candidates, one table per enzyme holds the scores, the calls, the isotonic probabilities and the committee disagreement of each blinded compound (private until the challenge closes), a table with the evidence for the stack choice (`*_stack_choice.csv`), the call-rate table of 8.4, `manifest.json` and a report. The manifest is a machine-readable record: the environment (the versions of Python and of every library whose version could change a number), the seeds and the design, the verdict of every gate that was read, the nested numbers, the call rates, the weights of the stack, the audit of leakage, and, for each file, its number of rows and its **SHA-256**, a fingerprint of the file's bytes (explained below).

**What a hash is, and why it is there.** The SHA-256 of a file is a number of 256 bits, written as 64 hexadecimal digits, computed from the bytes of the file in such a way that changing any byte changes the number in an unpredictable way and no one can construct a different file with the same number. A hash recorded in the manifest when the file was written, and recomputed from the file as it lies on the disk now, proves that the file submitted is the file the build wrote. The cell recomputes the hashes, applies the portal's schema check as the project reimplements it (`portal_validate_classification` in `cypmode/challenge/submission.py`: the file is a CSV or a parquet, has exactly 750 rows and the identifier columns and both endpoint columns, no empty values, only binary calls, and each endpoint has both classes, since a constant prediction cannot be scored), and shows the three checks of the leakage audit.
"""),
    code(MANIFEST_CHECK),
    md("""
The hashes of the {{=len(file_table)}} files match the manifest ({{=all_hashes_match}}); the portal check passes for every file ({{=all_portal_ok}}); the leakage audit passes for both enzymes ({{=all_audit_ok}}). The audit is three assertions that the build makes before it fits anything and that stop the build if one fails: the external rows are exactly the rows of the test file, no test row is in the labelled pool, and no label is stored for any test row. They guard the one way in which this kind of project can fool itself without noticing, a blinded compound whose label, or something computed from its label, reached the fit. The positive rates of the candidates are {{=file_table['CYP3A4 positive rate'].iloc[0]:.1%}} for CYP3A4 and {{=file_table['CYP2D6 positive rate'].iloc[0]:.1%}} for CYP2D6, the calls of 8.4. The environment recorded is {{=environment_line}}. The files are written by `cypmode/challenge/final_tdi_build.py`, which also writes the report `final_report.md`; the files that would be uploaded are copied from there to `submissions/tdi_track/` by `cypmode/challenge/submissions_sync.py`, and the pre-submission check of the audit (section 12.5) compares them with the final outputs byte for byte. The audit report itself is generated twice and compared (job 900, section 12.2), which is how the defect of 8.3 was found.

*On what actually ran.* `cypmode/challenge/final_tdi_build.py` (the build; `parse_gates` reads the verdicts, `choose_stack` applies the eligibility rule of its docstring, `leakage_audit`, `file_hash`, `environment`), `cypmode/challenge/submission.py` (`validate_submission`, `portal_validate_classification`, `build_submission`), and `cypmode/challenge/submissions_sync.py`. The per-compound columns of the files are not read or shown by the notebook.
"""),
    md("""
## 8.7 Results with their uncertainty

> **Field:** Statistics

*Where this section sits:* this is the number the project can give for its model before the challenge closes. It is an estimate made on the training data, with the nested procedure of Part VI, under the yardstick of 6.4. It is not a forecast of the score on the board. The cell computes, for each enzyme and each confidence cut-off, the AUC and the MCC of the stack on the 15 outer folds, with their spread and the 95 percent interval from the corrected resampled t of 6.5, whose standard error $s\\sqrt{1/J + n_{\\text{test}}/n_{\\text{train}}}$ is wider than the naive one, and then asks whether the stack is better than two simpler alternatives in the same folds: the member that a user would have picked on the inner scores, and the plain LightGBM on the descriptors, the member closest to the organisers' baseline.
"""),
    code(RESULTS8),
    code(APPLICABILITY),
    md("""
For CYP3A4 the stack earns an AUC of {{=result_value('CYP3A4', 1.0, 'AUC'):.3f}} at $z_0 = 1.0$ (95 percent interval {{=result_table[(result_table['enzyme'] == 'CYP3A4') & (result_table['z0'] == 1.0) & (result_table['score'] == 'AUC')]['95 % interval'].iloc[0]}}) and {{=result_value('CYP3A4', 1.96, 'AUC'):.3f}} at $z_0 = 1.96$, and an MCC of {{=result_value('CYP3A4', 1.0, 'MCC at the transferred rate'):.3f}} and {{=result_value('CYP3A4', 1.96, 'MCC at the transferred rate'):.3f}} (intervals {{=result_table[(result_table['enzyme'] == 'CYP3A4') & (result_table['z0'] == 1.0) & (result_table['score'] == 'MCC at the transferred rate')]['95 % interval'].iloc[0]}} and {{=result_table[(result_table['enzyme'] == 'CYP3A4') & (result_table['z0'] == 1.96) & (result_table['score'] == 'MCC at the transferred rate')]['95 % interval'].iloc[0]}}). For CYP2D6 the AUC is {{=result_value('CYP2D6', 1.0, 'AUC'):.3f}} and {{=result_value('CYP2D6', 1.96, 'AUC'):.3f}} and the MCC {{=result_value('CYP2D6', 1.0, 'MCC at the transferred rate'):.3f}} and {{=result_value('CYP2D6', 1.96, 'MCC at the transferred rate'):.3f}}, with intervals ({{=result_table[(result_table['enzyme'] == 'CYP2D6') & (result_table['z0'] == 1.0) & (result_table['score'] == 'MCC at the transferred rate')]['95 % interval'].iloc[0]}} for the first) that are as wide as the effect: the standard deviation of the CYP2D6 MCC over the folds is {{=result_value('CYP2D6', 1.0, 'MCC at the transferred rate', 'sd over the folds'):.2f}}, {{=result_value('CYP2D6', 1.0, 'MCC at the transferred rate', 'sd over the folds') / result_value('CYP3A4', 1.0, 'MCC at the transferred rate', 'sd over the folds'):.1f}} times that of CYP3A4. The enzyme remains the hard one of the challenge; the model separates its time-dependent inhibitors from the rest by an AUC of about 0.73 to 0.75, and a call rate of {{=called_final['CYP2D6']:.1%}} of the compounds turns that into an MCC whose interval comes close to zero. The two cut-offs differ in the way 6.4 predicted: the stricter one removes the compounds whose label is a coin flip and every number rises, because the yardstick is cleaner and not because the model is better.

Is the blend better than the simpler things? The second table answers, fold by fold. Over the 15 folds the stack beats LightGBM on the descriptors alone by {{=comparison_table[(comparison_table['enzyme'] == 'CYP3A4') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'LightGBM on the descriptors alone')]['gain'].iloc[0]:+.3f}} AUC on CYP3A4 (corrected p = {{=comparison_table[(comparison_table['enzyme'] == 'CYP3A4') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'LightGBM on the descriptors alone')]['p (corrected t)'].iloc[0]:.4f}}) and by {{=comparison_table[(comparison_table['enzyme'] == 'CYP2D6') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'LightGBM on the descriptors alone')]['gain'].iloc[0]:+.3f}} on CYP2D6 (p = {{=comparison_table[(comparison_table['enzyme'] == 'CYP2D6') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'LightGBM on the descriptors alone')]['p (corrected t)'].iloc[0]:.3f}}). Against the member that a user would have picked on the inner scores the gain is {{=comparison_table[(comparison_table['enzyme'] == 'CYP3A4') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'the member picked on the inner scores')]['gain'].iloc[0]:+.3f}} for CYP3A4 (p = {{=comparison_table[(comparison_table['enzyme'] == 'CYP3A4') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'the member picked on the inner scores')]['p (corrected t)'].iloc[0]:.3f}}) and {{=comparison_table[(comparison_table['enzyme'] == 'CYP2D6') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'the member picked on the inner scores')]['gain'].iloc[0]:+.3f}} for CYP2D6 (p = {{=comparison_table[(comparison_table['enzyme'] == 'CYP2D6') & (comparison_table['z0'] == 1.0) & (comparison_table['stack against'] == 'the member picked on the inner scores')]['p (corrected t)'].iloc[0]:.3f}}), at $z_0 = 1.0$; the table gives the other cut-off. The stack is clearly better than the baseline-like member for CYP3A4 and better by a similar amount, but with an interval that includes zero, for CYP2D6; against the member picked on the inner scores its gain is of the size that the folds cannot confirm (the gain over the member picked on the inner scores reaches p < 0.05 in {{=n_best_significant}} of the 4 combinations of enzyme and cut-off). The blend is therefore not shown to beat the best single member. Its case is the one of 6.2: that member is itself the result of a choice made on the scores of the folds, so its own estimate is flattered, and the stack, which chooses by a cross-validation of the blend inside the training part, pays that cost in its estimate and is at least as good. These are the comparisons that the first models of Part VII could not make, and they are made with the design of Part VI.

What the number is, and is not. It is the expected quality of the procedure on compounds like the pool, weighted to look like the blinded set and scored where the labels are not coin flips. It is not the board: the baseline of 7.1 earned 0.065 and 0.097 MCC more on the board than the cross-validation said, for reasons that Part VI listed and that this yardstick narrows and does not close. The final stack of the build named in the manifest is `{{=final_stack_names['CYP3A4']}}` for CYP3A4 and `{{=final_stack_names['CYP2D6']}}` for CYP2D6, with {{=n_members_final['CYP3A4']}} and {{=n_members_final['CYP2D6']}} members; if a group of Part IX or X passes its gate, the build says so here, and the numbers above move to those of the new stack. At the time the notebook was built they are the numbers of the default stack evaluated in the cell, which are also the numbers of the build's own report.

**Where the number can be trusted.** An average over the pool is not a promise for every compound: a model is most reliable where it interpolates between compounds it has seen. The final build is therefore accompanied by a measurement of that, `cypmode/challenge/applicability_domain.py` (queue job 096), which refits nothing: it rebuilds the final stack of each enzyme from the cached predictions exactly as the build does, applies it to the held-out rows of every outer fold, gives each held-out compound its nearest-training similarity to the rows its fold was trained on (ECFP4 **Tanimoto similarity**), cuts the held-out compounds into similarity bins and gives each bin an AUC with a 95 percent interval whose bootstrap resamples whole Butina clusters rather than single compounds, because the compounds of one chemical series are not independent observations. The bin that a blinded compound falls in gives it a tier: supported when the lower end of the bin's interval reaches the threshold the module declares, weak when the bin holds enough compounds but a lower end below it, and unsupported when the bin holds too few held-out compounds to be trusted. Of the {{=n_bins['CYP3A4']}} similarity bins at the first confidence cut-off {{=n_supported_bins['CYP3A4']}} are supported for CYP3A4 and {{=n_supported_bins['CYP2D6']}} for CYP2D6; of the {{=n_blinded_compounds}} blinded compounds, {{=n_supported['CYP3A4']}} fall in a supported bin for CYP3A4 and {{=n_supported['CYP2D6']}} for CYP2D6 ({{=n_weak['CYP2D6']}} weak and {{=n_unsupported['CYP2D6']}} unsupported). The median nearest-training similarity of the blinded compounds is {{=median_similarity['CYP3A4']:.2f}} for CYP3A4 and {{=median_similarity['CYP2D6']:.2f}} for CYP2D6, and {{=share_above_half['CYP3A4']:.0%}} and {{=share_above_half['CYP2D6']:.0%}} of them are at a similarity of 0.5 or more. The AUC of a held-out bin at the first cut-off runs from {{=lowest_auc['CYP3A4']:.2f}} to {{=highest_auc['CYP3A4']:.2f}} over the bins of CYP3A4 and from {{=lowest_auc['CYP2D6']:.2f}} to {{=highest_auc['CYP2D6']:.2f}} over those of CYP2D6. The module's own reading, in its docstring, is that absolute cross-validation numbers are not forecasts of the leaderboard and the table is read by its shape across bins; the shape here is that the final CYP3A4 stack is supported on every bin and every blinded compound, and that the final CYP2D6 stack, the weaker ranker, is useful where the nearest training compound is at least moderately similar and is not supported anywhere at the first cut-off.

*On what actually ran.* `nested_evaluate` with `StackedLogit` and `BestSingle` (`cypmode/challenge/nested.py`), the weights of `confident_weights` and `SimilarityMatcher`, `corrected_resampled_t` and `plan_sizes` (`model_comparison.py`); the report of the build, `final_report.md`, holds the same AUC and MCC. The last cell reads `applicability_domain_compounds.csv` and `applicability_domain_bins.csv` from `data/tdi_track/outputs/final/`, written by `cypmode/challenge/applicability_domain.py` in queue job 096.
"""),
    figure("arch_final_results"),
]
