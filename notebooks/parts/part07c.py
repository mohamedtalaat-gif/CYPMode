"""Part VII (third part): the decision threshold and the reliability of a prediction (7.5), query by committee (7.6)."""

from notebooks.notebook_builder import code, md

THRESHOLD_RULES = '''
# Four ways of choosing the threshold, on the same probabilities. The model is the LightGBM of 7.1 on the descriptors; five outer folds; for each outer fold the training part gets an inner four-fold cross-validation to produce
# out-of-fold probabilities that never touch the outer validation fold. The rules: (1) the fixed 0.5; (2) the threshold that is best on the outer validation fold itself, which uses the labels it is then scored on;
# (3) the project's rule (cypmode/challenge/threshold_tuning.py): the threshold that maximises the MCC on the inner out-of-fold probabilities of the training part, frozen and applied to the validation fold; (4) the rate rule:
# call as positive the same share of the validation compounds as the positive rate of the training part, whatever the probabilities are.
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from cypmode.challenge.tdi_baseline import TDI_ISOFORMS
from cypmode.challenge.threshold_tuning import DEFAULT_THRESHOLD_GRID, find_optimal_threshold

def top_share(score, share):
    k = int(round(share * len(score)))
    return (score >= np.sort(score)[::-1][k - 1]).astype(int)

rule_rows, chosen_rows = [], []
for iso in TDI_ISOFORMS:
    column = f"{iso}_is_TDI"
    sub = described.dropna(subset=[column]).reset_index(drop=True)
    y = sub[column].astype(int).to_numpy()
    X = np.stack(sub["descriptors"])
    for fold, (fit_rows, held) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
        inner = np.full(len(fit_rows), np.nan)
        for inner_fit, inner_held in StratifiedKFold(4, shuffle=True, random_state=42).split(X[fit_rows], y[fit_rows]):
            inner[inner_held] = LGBMClassifier(verbose=-1, random_state=42).fit(X[fit_rows][inner_fit], y[fit_rows][inner_fit]).predict_proba(X[fit_rows][inner_held])[:, 1]
        tuned, _ = find_optimal_threshold(y[fit_rows], inner)
        p = LGBMClassifier(verbose=-1, random_state=42).fit(X[fit_rows], y[fit_rows]).predict_proba(X[held])[:, 1]
        best_on_validation, _ = find_optimal_threshold(y[held], p)
        calls = {"fixed 0.5": (p >= 0.5).astype(int), "best on the validation fold itself (uses its labels)": (p >= best_on_validation).astype(int),
                 "tuned on the inner out-of-fold probabilities (the project rule)": (p >= tuned).astype(int), "call the training positive rate": top_share(p, y[fit_rows].mean())}
        for rule, called in calls.items():
            rule_rows.append({"enzyme": iso, "rule": rule, "fold": fold, "MCC": matthews_corrcoef(y[held], called)})
        chosen_rows.append({"enzyme": iso, "fold": fold, "threshold tuned inside the training part": tuned, "threshold best on the validation fold": best_on_validation})
rule_scores = pd.DataFrame(rule_rows)
rule_table = rule_scores.groupby(["enzyme", "rule"], sort=False)["MCC"].agg(["mean", "std"]).reset_index().rename(columns={"mean": "mean MCC over the 5 folds", "std": "sd over the folds"})
show_table(rule_table, floatfmt=".3f")
chosen_table = pd.DataFrame(chosen_rows)
show_table(chosen_table.groupby("enzyme")[["threshold tuned inside the training part", "threshold best on the validation fold"]].agg(["min", "max"]).reset_index().round(2), floatfmt=".2f")
def rule_value(iso, rule, column="mean MCC over the 5 folds"):
    return float(rule_table[(rule_table["enzyme"] == iso) & (rule_table["rule"] == rule)][column].iloc[0])
'''

BASE_RATE = '''
# The MCC of a fixed decision rule is not a fixed number: it depends on how common the positives are. Let a rule have a true-positive rate TPR and a false-positive rate FPR, and let the share of positives be pi. The expected
# counts per compound are TP = pi TPR, FN = pi (1 - TPR), FP = (1 - pi) FPR, TN = (1 - pi)(1 - FPR). Put into the definition of the MCC (section 4.10) they give MCC = sqrt(pi (1 - pi)) (TPR - FPR) /
# sqrt((pi TPR + (1 - pi) FPR)(pi (1 - TPR) + (1 - pi)(1 - FPR))). The formula is checked against the MCC computed from counts, then drawn for the operating point of the baseline of 7.1 at its best threshold, and the
# project's stored stress test of the minimax threshold (robust_threshold_stress_test.csv, which resamples real validation folds to other positive rates) is read beside it.
from scipy import stats

def mcc_at_rate(tpr, fpr, pi):
    numerator = np.sqrt(pi * (1 - pi)) * (tpr - fpr)
    denominator = np.sqrt((pi * tpr + (1 - pi) * fpr) * (pi * (1 - tpr) + (1 - pi) * (1 - fpr)))
    return numerator / denominator

def mcc_from_counts(tp, fn, fp, tn):
    return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

rate_grid = np.linspace(0.03, 0.6, 58)
operating_points, check_errors = {}, []
for iso in TDI_ISOFORMS:
    column = f"{iso}_is_TDI"
    sub = described.dropna(subset=[column]).reset_index(drop=True)
    y = sub[column].astype(int).to_numpy()
    X = np.stack(sub["descriptors"])
    p = np.zeros(len(y))
    for fit_rows, held in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        p[held] = LGBMClassifier(verbose=-1, random_state=42).fit(X[fit_rows], y[fit_rows]).predict_proba(X[held])[:, 1]
    best_threshold, _ = find_optimal_threshold(y, p)                 # the pooled out-of-fold probabilities; the best threshold on them is used only to fix one operating point to draw
    called = p >= best_threshold
    tpr, fpr = float(called[y == 1].mean()), float(called[y == 0].mean())
    operating_points[iso] = (tpr, fpr)
    for pi in (0.05, 0.2, 0.45):
        check_errors.append(abs(mcc_at_rate(tpr, fpr, pi) - mcc_from_counts(pi * tpr, pi * (1 - tpr), (1 - pi) * fpr, (1 - pi) * (1 - fpr))))
formula_error = float(max(check_errors))
stored_stress = pd.read_csv(COMPARISONS / "robust_threshold_stress_test.csv")
stress_by_rate = stored_stress.groupby(["isoform", "target_base_rate"])[["mcc_naive_threshold", "mcc_robust_threshold"]].mean().reset_index()
stored_robust = pd.read_csv(COMPARISONS / "robust_threshold_tuning.csv")
robust_natural = stored_robust.groupby("isoform")[["mcc_naive_threshold", "mcc_robust_threshold"]].mean()
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.7))
for iso, colour in zip(TDI_ISOFORMS, (theme["blue"], theme["orange"])):
    left.plot(rate_grid, mcc_at_rate(*operating_points[iso], rate_grid), color=colour, lw=2.4, label=f"{iso}: TPR {operating_points[iso][0]:.2f}, FPR {operating_points[iso][1]:.2f}")
left.axvline(0.213, color=theme["muted"], ls=":", lw=1.2)
left.set(xlabel="share of positives pi", ylabel="MCC of the same rule", title="one decision rule, other positive rates")
left.legend(frameon=False, fontsize=8)
cyp3a4_stress = stress_by_rate[stress_by_rate["isoform"] == "CYP3A4"]
right.plot(cyp3a4_stress["target_base_rate"], cyp3a4_stress["mcc_naive_threshold"], color=theme["blue"], lw=2.4, label="threshold tuned for the natural rate")
right.plot(cyp3a4_stress["target_base_rate"], cyp3a4_stress["mcc_robust_threshold"], color=theme["red"], lw=2.4, label="minimax threshold (robust)")
right.set(xlabel="positive rate of the resampled validation folds", ylabel="mean MCC", title="CYP3A4: the stored stress test")
right.legend(frameon=False, fontsize=8)
for axis in (left, right):
    tidy(axis, theme)
fig.tight_layout()
print(f"largest difference between the closed formula and the MCC from counts: {formula_error:.1e}")
mcc_low_rate = {iso: float(mcc_at_rate(*operating_points[iso], 0.05)) for iso in TDI_ISOFORMS}
mcc_high_rate = {iso: float(mcc_at_rate(*operating_points[iso], 0.45)) for iso in TDI_ISOFORMS}
stress_gain = {}
for iso in TDI_ISOFORMS:
    part = stress_by_rate[stress_by_rate["isoform"] == iso]
    low, high = part[part["target_base_rate"] <= 0.15], part[part["target_base_rate"] >= 0.3]
    stress_gain[iso] = (float((low["mcc_robust_threshold"] - low["mcc_naive_threshold"]).mean()), float((high["mcc_robust_threshold"] - high["mcc_naive_threshold"]).mean()))
'''

RELIABILITY = '''
# The first reliability score of the project (cypmode/challenge/reliability.py): two numbers per compound, computed for the validation compounds of the first fold of the fixed-seed five-fold protocol (CYP3A4, the
# descriptors), and checked against the thing a reliability score is for, the model's actual mistakes at the threshold 0.5. The first is the disagreement of an ensemble of five LightGBM models trained on the same
# compounds with different random seeds; the second is the distance to the training set, 1 minus the highest Tanimoto similarity to a training compound (section 3.2). The correlations with the mistake are the stored ones
# (reliability_correlation.csv) and the recomputed ones.
from scipy.stats import pointbiserialr, spearmanr

from cypmode.challenge.reliability import ensemble_disagreement, tanimoto_ood_distance

cyp3a4_sub = described.dropna(subset=["CYP3A4_is_TDI"]).reset_index(drop=True)
y_rel = cyp3a4_sub["CYP3A4_is_TDI"].astype(int).to_numpy()
X_rel = np.stack(cyp3a4_sub["descriptors"])
fit_rel, held_rel = next(iter(StratifiedKFold(5, shuffle=True, random_state=42).split(X_rel, y_rel)))
disagreement = ensemble_disagreement(X_rel[fit_rel], y_rel[fit_rel], X_rel[held_rel])
distance = tanimoto_ood_distance(cyp3a4_sub.loc[fit_rel, "SMILES"], cyp3a4_sub.loc[held_rel, "SMILES"])
probabilities = LGBMClassifier(verbose=-1, random_state=42).fit(X_rel[fit_rel], y_rel[fit_rel]).predict_proba(X_rel[held_rel])[:, 1]
mistake = ((probabilities >= 0.5).astype(int) != y_rel[held_rel]).astype(int)
stored_reliability = pd.read_csv(COMPARISONS / "reliability_correlation.csv").set_index("component")
reliability_rows = []
for name, values in (("ensemble_disagreement", disagreement), ("ood_distance", distance)):
    reliability_rows.append({"component": name, "point-biserial r with a mistake": float(pointbiserialr(mistake, values)[0]), "Spearman r": float(spearmanr(mistake, values)[0]),
                             "stored point-biserial r": float(stored_reliability.loc[name, "point_biserial_r"]), "stored Spearman r": float(stored_reliability.loc[name, "spearman_r"])})
reliability_table = pd.DataFrame(reliability_rows)
show_table(reliability_table, floatfmt=".3f")
disagreement_summary = {"largest value": float(disagreement.max()), "share exactly zero": float((disagreement == 0).mean()), "compounds": len(disagreement)}
n_mistakes, mistake_rate = int(mistake.sum()), float(mistake.mean())
distance_quartile_rates = [float(mistake[(distance >= lo) & (distance <= hi)].mean()) for lo, hi in zip(np.quantile(distance, [0, 0.25, 0.5, 0.75]), np.quantile(distance, [0.25, 0.5, 0.75, 1.0]))]
print(f"disagreement of the five seeds: largest {disagreement_summary['largest value']:.1e}, exactly zero for {disagreement_summary['share exactly zero']:.1%} of {disagreement_summary['compounds']} compounds; mistakes {n_mistakes} ({mistake_rate:.1%}); mistake rate by distance quartile {[round(r, 3) for r in distance_quartile_rates]}")
'''

COMMITTEE_TOY = '''
# The two committee signals, on a toy. Five members each vote "positive" or "negative" for three compounds, and each also gives a score on a common 0-to-1 scale (the project puts the members on one scale with
# their own empirical cumulative distribution functions; section 8.2). Vote entropy is the binary entropy of the share of positive votes, -(q ln q + (1 - q) ln(1 - q)) for the share q: 0 when the members agree and
# 0.693 (ln 2) when they split as evenly as possible. Spread is the standard deviation of the five scores.
from cypmode.challenge.committee_uncertainty import committee_spread, vote_entropy

toy_votes = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
toy_scores = np.array([[0.91, 0.88, 0.95, 0.9, 0.93], [0.7, 0.6, 0.65, 0.4, 0.35], [0.62, 0.58, 0.41, 0.35, 0.3]])
toy_table = pd.DataFrame({"compound": ["A", "B", "C"], "votes for positive (of 5)": toy_votes.sum(axis=1), "vote entropy": vote_entropy(toy_votes), "spread of the scores": committee_spread(toy_scores)})
show_table(toy_table, floatfmt=".3f")
toy_entropies = toy_table["vote entropy"].tolist()
'''

COMMITTEE_LIVE = '''
# The committee on real compounds. The 23 default members of the stack (section 8.1) have their predictions for every compound in the outer-validation folds of the nested plan of 6.3, made by models that never saw it.
# For each compound: the stack's call (cypmode.challenge.nested), whether the call is wrong, the margin (minus the distance of the stack's score from its own threshold, the simplest signal of "close to the line") and the committee signals.
# Each is judged by cypmode/challenge/committee_uncertainty.py: the AUROC for flagging the stack's mistakes, the gain in that AUROC when the signal is added to the margin in a cross-validated error model, with a p-value from
# a permutation null (the signal's values are permuted across compounds 200 times; section 6.6 gave the reason for not bootstrapping fixed predictions), and then whether acting on it, by falling back to a negative call
# for the most uncertain share of the compounds, raises the MCC. One repeat is used (2,334 compounds, each once) so that the compounds are independent.
from cypmode.challenge.committee_uncertainty import BASELINE, COMMITTEE_SIGNALS, collect_signals, evaluate_components, selective_fallback
from cypmode.challenge.nested import NestedPlan, PredictionCache
from cypmode.challenge.nested_runners import default_candidates

nested_root = ROOT / "data/tdi_track/outputs/nested/CYP3A4"
committee_plan = NestedPlan.load(nested_root / "plan.npz")
committee_cache = PredictionCache(nested_root / "cache", committee_plan)
committee_names = [c.name for c in default_candidates()]
morgan_bits = np.load(ROOT / "data/tdi_track/outputs/caches/feature_store.npz", allow_pickle=True)["morgan_bits"]
signals_all = collect_signals(committee_cache, committee_names, morgan_bits)
signals = signals_all[signals_all["fold"].str.startswith("0-")].reset_index(drop=True)
evaluation = evaluate_components(signals)
fallback = pd.concat([selective_fallback(signals, s) for s in (*COMMITTEE_SIGNALS, BASELINE)], ignore_index=True)
show_table(evaluation[["component", "auroc", "auroc_lo", "auroc_hi", "gain_over_margin", "gain_p", "verdict"]], floatfmt=".3f")
best_fallback = fallback.groupby("signal")["mcc"].agg(["first", "max"]).rename(columns={"first": "MCC with no fallback", "max": "best MCC over the fallback shares"}).reset_index()
show_table(best_fallback, floatfmt=".3f")
n_committee, n_members, stack_error = len(signals), len(committee_names), float(signals["wrong"].mean())
best_committee_gain = float(evaluation[evaluation["component"].isin(COMMITTEE_SIGNALS)]["gain_over_margin"].max())
# How collective are the mistakes? For every compound of the first repeat, the share of the 23 members whose own call (each calling the stack's share of compounds positive) is wrong, split by whether the stack is wrong.
from cypmode.challenge.nested import OUTER_REFIT, StackedLogit, call_top_fraction, inner_oof

share_rows = []
for repeat, k in committee_plan.outer_folds():
    if repeat != 0:
        continue
    rows_in, oof = inner_oof(committee_cache, committee_names, repeat, k)
    refit = committee_plan.task(repeat, k, OUTER_REFIT)
    y_in, y_out = committee_plan.labels(rows_in), committee_plan.labels(refit.evaluate)
    outer = np.column_stack([committee_cache.get(n, refit) for n in committee_names])
    chosen = StackedLogit().fit(list(committee_names), oof, y_in, None, "auc")
    member_votes = np.column_stack([call_top_fraction(outer[:, j], chosen.rate()) for j in range(outer.shape[1])])
    share_rows.append(pd.DataFrame({"stack wrong": call_top_fraction(chosen.predict(outer), chosen.rate()) != y_out, "share of members wrong": (member_votes != y_out[:, None]).mean(axis=1)}))
collective = pd.concat(share_rows)
share_wrong_when_stack_wrong = float(collective[collective["stack wrong"]]["share of members wrong"].mean())
share_wrong_when_stack_right = float(collective[~collective["stack wrong"]]["share of members wrong"].mean())
share_majority_wrong = float((collective[collective["stack wrong"]]["share of members wrong"] > 0.5).mean())
def evaluation_value(component, column):
    return float(evaluation[evaluation["component"] == component][column].iloc[0])
adds_information = [row.component for row in evaluation.itertuples() if row.verdict == "adds_information"]
fallback_helps = {r["signal"]: bool(r["best MCC over the fallback shares"] > r["MCC with no fallback"] + 0.01) for _, r in best_fallback.iterrows()}
'''

COMMITTEE_STORED = '''
# The project's record, for both enzymes, as written when the experiment was run (committee_uncertainty_decision_<enzyme>.json): per component the AUROC, the gain over the margin and its permutation p-value, the verdict, and
# the two conditions for letting a component alter a prediction.
import json

record_rows = []
for iso in TDI_ISOFORMS:
    record = json.loads((COMPARISONS / f"committee_uncertainty_decision_{iso}.json").read_text())
    for component, values in record["components"].items():
        record_rows.append({"enzyme": iso, "component": component, "AUROC": values["auroc"], "gain over the margin": values["gain_over_margin"], "p": values["gain_p"], "verdict": values["verdict"],
                            "acting on it improves the MCC": values["acting_on_it_improves_mcc"], "used to alter predictions": values["used_to_alter_predictions"]})
record_table = pd.DataFrame(record_rows)
show_table(record_table, floatfmt=".3f")
n_used = int(record_table["used to alter predictions"].sum())
record_2d6 = record_table[record_table["enzyme"] == "CYP2D6"].set_index("component")
'''

CELLS = [
    md("""
## 7.5 The decision threshold, the positive rate, and a first reliability score

> **Field:** Statistics · Machine learning

*Where this section sits:* 7.1 showed that the baseline ranks well and decides badly, because 0.5 is the wrong place to cut. A threshold is a choice, and Part VI taught that a choice must be made without the labels it will be scored on. This section applies the lesson to the threshold, shows why the MCC of one decision rule changes with the positive rate and what the project made of that, and ends with the first attempt at saying how far a single prediction can be trusted, which turned out to be instructive for a reason nobody planned.

**Choosing the threshold without cheating.** The module `cypmode/challenge/threshold_tuning.py` sweeps a grid of thresholds and keeps the one that maximises the MCC. The question is on which probabilities the sweep is run. The cell compares four answers on the same model and the same five outer folds, with an inner four-fold cross-validation inside each training part (the descriptors and LightGBM of 7.1, to keep the cell quick): the fixed 0.5; the best threshold *for the validation fold itself*, which looks at the labels the fold is scored on; the project's rule, the threshold that maximises the MCC on the inner out-of-fold probabilities of the training part, frozen, and applied to the validation fold it has never seen; and a naive rate rule, calling positive the same share of the validation compounds as the positive rate of the training part.
"""),
    code(THRESHOLD_RULES),
    md("""
For CYP3A4 the fixed 0.5 gives {{=rule_value('CYP3A4', 'fixed 0.5'):.3f}}; the threshold chosen on the validation fold itself gives {{=rule_value('CYP3A4', 'best on the validation fold itself (uses its labels)'):.3f}}, which no real user can have, and the project's rule {{=rule_value('CYP3A4', 'tuned on the inner out-of-fold probabilities (the project rule)'):.3f}}: nearly all of the gain with no label of the validation fold used. The naive rate rule gives {{=rule_value('CYP3A4', 'call the training positive rate'):.3f}} with a larger spread over the folds. For CYP2D6 the four numbers are {{=rule_value('CYP2D6', 'fixed 0.5'):.3f}}, {{=rule_value('CYP2D6', 'best on the validation fold itself (uses its labels)'):.3f}}, {{=rule_value('CYP2D6', 'tuned on the inner out-of-fold probabilities (the project rule)'):.3f}} and {{=rule_value('CYP2D6', 'call the training positive rate'):.3f}}. The gap between the second and the third column, the optimism of choosing the threshold on the fold that is scored, is {{=rule_value('CYP3A4', 'best on the validation fold itself (uses its labels)') - rule_value('CYP3A4', 'tuned on the inner out-of-fold probabilities (the project rule)'):.3f}} MCC for CYP3A4 and {{=rule_value('CYP2D6', 'best on the validation fold itself (uses its labels)') - rule_value('CYP2D6', 'tuned on the inner out-of-fold probabilities (the project rule)'):.3f}} for CYP2D6, where it is larger because the signal is weaker and the best threshold moves from fold to fold (the second table: the threshold that is best on a CYP2D6 fold ranges from {{=chosen_table[chosen_table['enzyme'] == 'CYP2D6']['threshold best on the validation fold'].min():.2f}} to {{=chosen_table[chosen_table['enzyme'] == 'CYP2D6']['threshold best on the validation fold'].max():.2f}}, while the one tuned inside the training part stays between {{=chosen_table[chosen_table['enzyme'] == 'CYP2D6']['threshold tuned inside the training part'].min():.2f}} and {{=chosen_table[chosen_table['enzyme'] == 'CYP2D6']['threshold tuned inside the training part'].max():.2f}}). Of the two rules that use no label of the validation fold, the project's is the first, and it is the one whose descendant, the transfer of a call *rate* tuned inside the folds (section 8.4), the final model uses.

**Why a rate and not a threshold.** The MCC of a fixed decision rule is not a fixed number; it depends on how common the positives are among the compounds it is applied to. Write $\\pi$ for the share of positives, $\\mathrm{TPR}$ for the share of positives that the rule calls positive (the true-positive rate, the recall of section 4.10) and $\\mathrm{FPR}$ for the share of negatives that it calls positive (the false-positive rate). Per compound, the expected shares of the four cells of the confusion table are $TP = \\pi\\,\\mathrm{TPR}$, $FN = \\pi\\,(1-\\mathrm{TPR})$, $FP = (1-\\pi)\\,\\mathrm{FPR}$ and $TN = (1-\\pi)(1-\\mathrm{FPR})$. Put into the definition of the MCC, $(TP\\cdot TN - FP\\cdot FN)/\\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}$, the numerator becomes $\\pi(1-\\pi)(\\mathrm{TPR}-\\mathrm{FPR})$ and the factors under the root become $(\\pi\\,\\mathrm{TPR} + (1-\\pi)\\,\\mathrm{FPR})$, $\\pi$, $(1-\\pi)$ and $(\\pi(1-\\mathrm{TPR}) + (1-\\pi)(1-\\mathrm{FPR}))$, so that

$$\\mathrm{MCC}(\\pi) \\;=\\; \\frac{\\sqrt{\\pi(1-\\pi)}\\;(\\mathrm{TPR}-\\mathrm{FPR})}{\\sqrt{\\bigl(\\pi\\,\\mathrm{TPR} + (1-\\pi)\\,\\mathrm{FPR}\\bigr)\\bigl(\\pi\\,(1-\\mathrm{TPR}) + (1-\\pi)(1-\\mathrm{FPR})\\bigr)}}.$$

A rule is described by its two rates, and the same two rates give different MCC values at different $\\pi$. The cell checks the formula against the MCC computed from counts, draws it for the best operating point of the baseline of each enzyme, and reads the project's stored stress test of a *minimax* threshold: a threshold chosen to have the best worst-case MCC over an assumed range of positive rates (5 to 50 percent), tested on real validation folds resampled to other positive rates.
"""),
    code(BASE_RATE),
    md("""
The formula equals the MCC computed from counts to {{=formula_error:.0e}}. For the operating point drawn in the left panel, the MCC of CYP3A4 runs from {{=mcc_low_rate['CYP3A4']:.2f}} at a positive rate of 5 percent to {{=mcc_high_rate['CYP3A4']:.2f}} at 45 percent, and that of CYP2D6 from {{=mcc_low_rate['CYP2D6']:.2f}} to {{=mcc_high_rate['CYP2D6']:.2f}}: the same rule, scored on populations that differ only in the share of positives. The minimax threshold buys protection in one direction and pays for it in the other. On the stored stress test it raises the CYP3A4 MCC by {{=stress_gain['CYP3A4'][0]:+.3f}} on average when the positive rate is 15 percent or less, and lowers it by {{=-stress_gain['CYP3A4'][1]:.3f}} when it is 30 percent or more; at the natural rate it is a wash ({{=robust_natural.loc['CYP3A4', 'mcc_robust_threshold']:.3f}} against {{=robust_natural.loc['CYP3A4', 'mcc_naive_threshold']:.3f}}). The project of the day did not adopt it, because nothing said which way the blinded set's positive rate would differ from the training file's.

Something did say, later. The positive rates of the scored sets were recovered from the public boards (section 2.10): about 0.31 for CYP3A4 and about 0.06 for CYP2D6, against 0.21 for both enzymes in the training file. The shifts are in *opposite* directions. A threshold tuned on the training file's rate is therefore too eager for CYP2D6 and too timid for CYP3A4, and no single minimax rule serves both. Two conclusions followed, and the final model of Part VIII acts on both. The yardstick of 6.4 re-weights the validation compounds to the estimated rate of the scored set, so that the MCC of a candidate is read at the rate it will be scored at. And the operating point is transferred as a *rate of calls*, the share of compounds called positive, chosen on the inner out-of-fold scores, because a probability threshold means little when the positive rate moves and a rate of calls means the same thing in both populations (section 8.4).

**A first reliability score.** The last component of the first TDI models was a *reliability score*, a number per compound meant to say how far its prediction can be trusted, kept outside the prediction path from the start (`cypmode/challenge/reliability.py`). It was to be called uncertainty quantification only if its correlation with the real mistakes of the model had been measured. It had two components. The first was the disagreement of an ensemble: five LightGBM models trained on the same compounds with different random seeds, the standard deviation of their predicted probabilities for a compound. The second was the distance of the compound to the training set, $1 - \\max_j s_{ij}$, with $s_{ij}$ the Tanimoto similarity (section 3.2) of the compound $i$ to a training compound $j$. The cell recomputes them for the validation compounds of the first fold of the fixed-seed protocol, for CYP3A4, and the correlation of each with a mistake of the model at 0.5 (point-biserial, the Pearson correlation of a number with a yes/no variable, and Spearman), next to the stored values.
"""),
    code(RELIABILITY),
    md("""
The stored correlations are reproduced: {{=reliability_table.iloc[0]['point-biserial r with a mistake']:.3f}} for the ensemble and {{=reliability_table.iloc[1]['point-biserial r with a mistake']:.3f}} for the distance. The first number is a correlation with nothing. The five models differ only by their random seed, and a LightGBM model with its default settings has no random element: it uses every row and every column at every tree, so five seeds give five identical models. The disagreement of the ensemble is therefore zero up to the rounding of floating-point numbers: its largest value over the {{=disagreement_summary['compounds']}} compounds is {{=disagreement_summary['largest value']:.1e}}, and it is exactly zero for {{=disagreement_summary['share exactly zero']:.1%}} of them. A correlation computed from such values is a correlation with rounding error. The distance is a real signal: the model errs on {{=distance_quartile_rates[0]:.1%}} of the compounds in the quartile closest to the training set and on {{=distance_quartile_rates[2]:.1%}} and {{=distance_quartile_rates[3]:.1%}} in the two farthest quartiles, and the correlation of {{=reliability_table.iloc[1]['point-biserial r with a mistake']:.2f}} summarises a relation that is steep at the near end and flat beyond it.

The lesson is a check that costs one line and was not made: before a correlation is computed, look at the spread of the numbers. The disagreement score was dropped from the plan, the distance stayed, and the replacement for the first component is the subject of 7.6.

*On what actually ran.* `cypmode/challenge/threshold_tuning.py` holds `find_optimal_threshold` (the sweep over the grid 0.05 to 0.95), `cross_validate_with_threshold_tuning` (the inner-fold rule), `find_robust_threshold` (the minimax rule, which reconstructs the MCC at each assumed positive rate from the TPR and FPR curve with the formula above) and `base_rate_shift_stress_test`; the stored tables are `threshold_tuning.csv`, `robust_threshold_tuning.csv` and `robust_threshold_stress_test.csv`. `cypmode/challenge/reliability.py` holds `ensemble_disagreement`, `tanimoto_ood_distance`, `one_class_svm_ood_score` and `validate_reliability_correlation`; the stored table is `reliability_correlation.csv`. The cell above uses the descriptors only, where the stored experiments used the M3 columns for the thresholds. All of this is on the mixed population of section 2.4.
"""),
    md("""
## 7.6 Uncertainty by committee, tested against real mistakes

> **Field:** Machine learning · Statistics

*Where this section sits:* 7.5 ended with a reliability score whose first component measured nothing. This section builds a better candidate for it and then applies the strictest test the project could think of: the candidate must tell something that the simplest possible signal does not. It is one of the clearest cases in the project of a measured negative result, kept as such on purpose.

**What uncertainty is for.** A classifier that says 0.7 for a compound is right or wrong, and an *uncertainty* is a number, computed with the prediction, that is high when the prediction is likely to be wrong. It would be useful: a prediction with a flag could be set aside, or the compound sent for a measurement or a structure prediction. Three families of such numbers are common. The first uses the random passes of one model, dropout kept on at prediction time and the spread of the outputs (Gal and Ghahramani, 2016). The second uses the spread of one algorithm over random seeds, which is what 7.5 tried and found empty for gradient boosting. The third is **query by committee** (Seung, Opper and Sompolinsky, 1992; the technique comes from active learning, the choice of which compound to measure next): train a committee of *different* models and take their disagreement as the uncertainty. The nested design of Part VI already holds the out-of-fold predictions of 23 different models for every compound (6.3), so the committee costs nothing to build.

**The signals and the bar.** Two disagreements are used. For compound $i$ let $q_i$ be the share of the members that call it positive, each member calling positive the same share of compounds as the stack does. The *vote entropy* is $-\\bigl(q_i \\ln q_i + (1-q_i)\\ln(1-q_i)\\bigr)$, the binary entropy of the vote: 0 when the members agree, $\\ln 2 = 0.693$ when they split evenly. The *spread* is the standard deviation across members of their scores, each put on a common 0-to-1 scale. Two variants restrict the committee to its five best members and to one family of models. The bar is not "is it correlated with the stack's mistakes" but "does it tell something that the **margin** does not": the margin is the distance of the stack's score from its own decision threshold, so that a compound near the line is uncertain by construction. The module (`cypmode/challenge/committee_uncertainty.py`) asks three questions of every signal, on compounds that no member and no selector has seen: does it flag the stack's mistakes better than chance (the AUROC for "the stack is wrong", with a bootstrap interval); does it add information beyond the margin (the gain in that AUROC when the signal is added to the margin in a cross-validated error model, against a null made by permuting the signal's values across compounds, since bootstrapping fixed predictions produced a "discovery" from pure-noise committees in the module's own tests, the point of 6.6); and does acting on it improve the decisions (the most uncertain share of the compounds is given a negative call and the MCC is recomputed). A component may alter a prediction only if it adds information beyond the margin *and* acting on it raises the MCC by at least 0.01. First the signals on a toy.
"""),
    code(COMMITTEE_TOY),
    md("""
Compound A, on which the five members agree, has entropy {{=toy_entropies[0]:.3f}}; compound B, with three votes against two, has {{=toy_entropies[1]:.3f}}, close to the maximum of $\\ln 2 = 0.693$; compound C, with two votes against three, has the same entropy as B, since the entropy does not know which side is the majority. The spread is a different measure, large when the members' scores are far apart even if their votes are the same. The next cell computes the signals for the 2,334 compounds of the first repeat of the CYP3A4 plan and judges them with the module's own functions.
"""),
    code(COMMITTEE_LIVE),
    md("""
The stack of the {{=n_members}} members errs on {{=stack_error:.1%}} of the {{=n_committee:,}} compounds. The margin flags its mistakes with an AUROC of {{=evaluation_value('boundary_margin', 'auroc'):.3f}}, as it must: it is the signal built from the same score that makes the call. The committee does worse: {{=evaluation_value('qbc_vote_entropy', 'auroc'):.3f}} for the vote entropy and {{=evaluation_value('qbc_spread', 'auroc'):.3f}} for the spread, and, what decides, the gain over the margin is {{=evaluation_value('qbc_vote_entropy', 'gain_over_margin'):+.3f}} and {{=evaluation_value('qbc_spread', 'gain_over_margin'):+.3f}}, with permutation p-values of {{=evaluation_value('qbc_vote_entropy', 'gain_p'):.2f}} and {{=evaluation_value('qbc_spread', 'gain_p'):.2f}}: nothing that the margin does not already say. The components that pass the second question in this run are {{=', '.join(adds_information) if adds_information else 'none'}}. A component that adds information still has to pass the third: the table of the best MCC over the fallback shares shows that acting on a signal never raises the MCC of the stack by as much as 0.01 (it does for {{=sum(fallback_helps.values())}} of the {{=len(fallback_helps)}} signals, the margin included), so that, whatever the second question says, no component may alter a prediction. The restricted committees do no better, and the committee of five best members has an AUROC of {{=evaluation_value('qbc_vote_entropy_top5', 'auroc'):.3f}}, below the 0.5 of chance: a vote among the best members is *more* often unanimous when the stack is wrong.

The project's record of the day, written on 2026-10-05 with the members and the plan of that day and stored for both enzymes, is read in the next cell. The numbers differ in detail from the run above (the margin's AUROC was 0.687, the members were not the same 23) and agree in substance, with one difference worth keeping in view: in the record of the day no committee signal passes the second question for CYP3A4, while in this run {{=len(adds_information)}} of them does, with a gain of {{=best_committee_gain:.3f}} AUROC. A gain that small is statistically detectable with 2,334 compounds and of no use, which is exactly why the rule has two conditions.
"""),
    code(COMMITTEE_STORED),
    md("""
For CYP2D6 the record has two committee signals ({{=', '.join(c for c in record_2d6.index if record_2d6.loc[c, 'verdict'] == 'adds_information') or 'none'}}) that add a small but detectable amount beyond the margin, and the spread's relationship is inverted, high disagreement meaning *fewer* mistakes. Acting on them lowers the MCC, so that, over both enzymes and every component, {{=n_used}} may alter a prediction. The reason lies in the nature of the mistakes. The stack's mistakes are mostly *collective*: on the compounds where the stack is wrong, {{=share_wrong_when_stack_wrong:.0%}} of the members are wrong too on average, against {{=share_wrong_when_stack_right:.0%}} where it is right, and in {{=share_majority_wrong:.0%}} of its mistakes more than half of the members are wrong. The members are trained on the same descriptors and fingerprints, and a compound that one member misjudges is mostly misjudged by many, so that they agree on the wrong answer and the committee sees nothing. A committee can detect only the mistakes that its members disagree about.

The result is kept on purpose. The committee signals and the margin are part of the reliability table that the final model writes for the blinded compounds (`reliability_deploy.csv`), and they are *reported*, not used; the decision record (`committee_uncertainty_decision_<enzyme>.json`) says for each component whether it adds information and whether acting on it helps, and today the answer to "may it change a prediction" is no for every component. A competitor's accuracy gain can be reproduced by anyone who has the code; an uncertainty claim that was never checked against the simplest baseline cannot, and the project prefers the first kind of claim.

*On what actually ran.* `cypmode/challenge/committee_uncertainty.py` holds `vote_entropy`, `committee_spread`, `member_signals` (the members put on a common scale with their own inner out-of-fold distributions and each voting its top share), `collect_signals`, `evaluate_components` (300 bootstrap resamples for the intervals, 200 permutations for the gain, the Bonferroni level 0.05 divided by the number of tested components), `selective_fallback`, `deploy_signals` and `decision_record`. The stored tables are `committee_uncertainty_<enzyme>.csv`, `committee_uncertainty_fallback_<enzyme>.csv` and the decision JSON. This section is on the nested design of Part VI, that is, on the corrected population; the shares of members that are wrong are computed in the cell with the same functions that the module uses.
"""),
]
