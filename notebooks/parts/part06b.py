# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Part VI (second part): making the validation resemble the test (6.4) and comparing two models (6.5)."""

from notebooks.notebook_builder import code, figure, md

SIMILARITY = '''
# How close a validation compound is to the training compounds, against how close a test compound is. For the first outer fold of the plan: the highest Tanimoto similarity (section 3.2) of every outer-validation
# compound to the outer-training compounds, and of every blinded test compound to the same training compounds (a random 80 % of them, as the validation compounds' own training set has no more than that).
from cypmode.challenge.leaderboard_prior import estimate_from_csv
from cypmode.challenge.validation_matching import DEFAULT_BINS, SimilarityMatcher, nearest_neighbor_similarity, prior_weights, similarity_weights, weighted_auc, weighted_mcc

bits_all = feature_store["morgan_bits"].astype(np.float32)               # train rows first, then the 750 blinded test rows, as in the plan
matcher = SimilarityMatcher(bits_all, plan.external)
refit_0 = plan.task(0, 0, OUTER_REFIT)
validation_nn = nearest_neighbor_similarity(bits_all[refit_0.evaluate], bits_all[refit_0.train])
rng_sim = np.random.default_rng(0)
subset = refit_0.train[rng_sim.random(len(refit_0.train)) < 0.8]
test_nn = nearest_neighbor_similarity(bits_all[plan.external], bits_all[subset])
fold_weights = similarity_weights(validation_nn, test_nn)
bin_rows = []
for lo, hi in zip(DEFAULT_BINS[:-1], DEFAULT_BINS[1:]):
    in_validation, in_test = (validation_nn >= lo) & (validation_nn < hi), (test_nn >= lo) & (test_nn < hi)
    bin_rows.append({"nearest-training similarity": f"{lo:.1f} to {min(hi, 1.0):.1f}", "share of validation compounds": in_validation.mean(), "share of test compounds": in_test.mean(),
                     "weight": float(in_test.mean() / in_validation.mean()) if in_validation.any() else 0.0})
show_table(pd.DataFrame(bin_rows), floatfmt=".3f")
kish_ess = float(fold_weights.sum() ** 2 / (fold_weights**2).sum())
print(f"median nearest-training similarity: validation {np.median(validation_nn):.3f}, test {np.median(test_nn):.3f}; share at or above 0.5: {np.mean(validation_nn >= 0.5):.2f} and {np.mean(test_nn >= 0.5):.2f}")
print(f"effective sample size of the weights of this fold: {kish_ess:.0f} of {len(fold_weights)} validation compounds")
median_validation, median_test = float(np.median(validation_nn)), float(np.median(test_nn))
share_validation_close, share_test_close = float(np.mean(validation_nn >= 0.5)), float(np.mean(test_nn >= 0.5))
n_validation_fold = len(fold_weights)

fig, axis = plt.subplots(figsize=(6.6, 3.5))
axis.hist(validation_nn, bins=DEFAULT_BINS[:-1].tolist() + [1.0], density=True, alpha=0.75, color=theme["blue"], label="validation compounds of the fold")
axis.hist(test_nn, bins=DEFAULT_BINS[:-1].tolist() + [1.0], density=True, alpha=0.6, color=theme["orange"], label="blinded test compounds")
axis.set(xlabel="highest Tanimoto similarity to the training compounds", ylabel="density")
axis.legend(frameon=False)
tidy(axis, theme)
fig.tight_layout()
'''

WEIGHTED_AUC = '''
# The weighted AUC written out and checked on one fold: the probability that a positive is ranked above a negative, with every compound counted in proportion to its weight. Then the weighted MCC by its
# counts. Both are checked against the library.
y_fold = plan.labels(refit_0.evaluate)
scores_fold = cache.get("lgbm_A+fps_deep", refit_0)
positive, negative = y_fold == 1, y_fold == 0
pair_weight = fold_weights[positive][:, None] * fold_weights[negative][None, :]
wins = (scores_fold[positive][:, None] > scores_fold[negative][None, :]) + 0.5 * (scores_fold[positive][:, None] == scores_fold[negative][None, :])
auc_by_hand = float((pair_weight * wins).sum() / (fold_weights[positive].sum() * fold_weights[negative].sum()))
auc_library = float(roc_auc_score(y_fold, scores_fold, sample_weight=fold_weights))
called = scores_fold >= np.quantile(scores_fold, 0.7)
tp, fp = (fold_weights * (called & positive)).sum(), (fold_weights * (called & negative)).sum()
fn, tn = (fold_weights * (~called & positive)).sum(), (fold_weights * (~called & negative)).sum()
mcc_by_hand = float((tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
mcc_library = weighted_mcc(y_fold, called.astype(int), fold_weights)
print(f"weighted AUC by hand {auc_by_hand:.5f}, by the library {auc_library:.5f}; unweighted {roc_auc_score(y_fold, scores_fold):.5f}")
print(f"weighted MCC by the counts {mcc_by_hand:.5f}, by the project's function {mcc_library:.5f}")
'''

LADDER = '''
# The yardstick, built up one correction at a time, for one member of the stack (the best single member of 6.2) over all 15 outer folds. Each row adds a correction to the one above: similarity to the test
# compounds; the positive rate of the scored set (estimated from the public leaderboard, section 2.10); and the confident labels only, at the two cut-offs the project uses (section 2.5).
from cypmode.challenge.label_confidence import confidence_z

target_prevalence = estimate_from_csv(ROOT / "data/tdi_track/inputs/leaderboard_snapshots/interim_CYP3A4_2026-10-04.csv")["prevalence"]
z_train = confidence_z(train_file, "CYP3A4")
z_all = np.concatenate([z_train, np.full(len(plan.external), np.nan)])             # indexed like the plan's rows; the test rows have no label and no z
from cypmode.challenge.nested import call_top_fraction

ladder_cells = {"no correction": [], "similar to the test": [], "and the scored positive rate": [], "and confident labels, z0 = 1.0": [], "and confident labels, z0 = 1.96": []}
ladder_mcc = {name: [] for name in ladder_cells}                       # the same rungs for the MCC of calling the top 31 % positive (the scored rate)
n_kept = {1.0: [], 1.96: []}
for repeat, outer in plan.outer_folds():
    refit = plan.task(repeat, outer, OUTER_REFIT)
    y_out, s_out = plan.labels(refit.evaluate), cache.get("lgbm_A+fps_deep", refit)
    w_sim = matcher(refit.evaluate, refit.train)
    w_prior = w_sim * prior_weights(y_out, target_prevalence)
    called_out = call_top_fraction(s_out, target_prevalence).astype(int)
    weights_by_rung = {"no correction": None, "similar to the test": w_sim, "and the scored positive rate": w_prior}
    for z0, label in ((1.0, "and confident labels, z0 = 1.0"), (1.96, "and confident labels, z0 = 1.96")):
        keep = np.nan_to_num(z_all[refit.evaluate], nan=0.0) >= z0
        n_kept[z0].append(keep.mean())
        weights_by_rung[label] = w_prior * keep
    for label, weights in weights_by_rung.items():
        ladder_cells[label].append(roc_auc_score(y_out, s_out, sample_weight=weights))
        ladder_mcc[label].append(weighted_mcc(y_out, called_out, weights))
ladder = pd.DataFrame({"yardstick": list(ladder_cells), "mean AUC over the 15 outer folds": [float(np.mean(v)) for v in ladder_cells.values()],
                       "spread of the AUC over the folds": [float(np.std(v, ddof=1)) for v in ladder_cells.values()], "mean MCC at the scored rate": [float(np.mean(v)) for v in ladder_mcc.values()]})
show_table(ladder, floatfmt=".4f")
print(f"target positive rate (leaderboard estimate): {target_prevalence:.4f}; share of validation compounds with a confident label: z0=1.0 {np.mean(n_kept[1.0]):.2f}, z0=1.96 {np.mean(n_kept[1.96]):.2f}")
ladder_values = dict(zip(ladder["yardstick"], ladder["mean AUC over the 15 outer folds"]))
ladder_mcc_values = dict(zip(ladder["yardstick"], ladder["mean MCC at the scored rate"]))
share_kept_1, share_kept_196 = float(np.mean(n_kept[1.0])), float(np.mean(n_kept[1.96]))
'''

NULL_SIMULATION = '''
# Two classifiers that are equal in truth, compared by cross-validation on many simulated data sets. Each data set has 600 compounds with 20 features of the same strength; model A uses 10 of the features and model B
# the other 10, so that the two have the same expected AUC and any difference between them is chance. Each is scored by 5-fold cross-validation repeated 3 times (J = 15 folds); the difference of the AUCs is tested
# with the naive paired t-test and with the corrected resampled t-test of the project (cypmode/challenge/model_comparison.py). A test that works rejects in about 5 % of the data sets.
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cypmode.challenge.model_comparison import corrected_resampled_t, naive_paired_t

def simulate_differences(rng, n=600, strength_a=0.35, strength_b=0.35, folds=5, repeats=3):
    X = rng.normal(size=(n, 20))
    logit = X[:, :10].sum(axis=1) * strength_a + X[:, 10:].sum(axis=1) * strength_b
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    differences = []
    for repeat in range(repeats):
        for fit_rows, held in StratifiedKFold(folds, shuffle=True, random_state=repeat).split(X, y):
            auc = [roc_auc_score(y[held], LogisticRegression(max_iter=300).fit(X[fit_rows][:, cols], y[fit_rows]).decision_function(X[held][:, cols])) for cols in (slice(0, 10), slice(10, 20))]
            differences.append(auc[0] - auc[1])
    return np.array(differences)

rng_null = np.random.default_rng(11)
n_data_sets = 300
null_differences = np.array([simulate_differences(rng_null) for _ in range(n_data_sets)])
n_train_sim, n_test_sim = 480, 120
rejections = {"naive paired t": 0, "corrected resampled t": 0}
for d in null_differences:
    rejections["naive paired t"] += naive_paired_t(d).p_value < 0.05
    rejections["corrected resampled t"] += corrected_resampled_t(d, n_train_sim, n_test_sim).p_value < 0.05
rejection_rates = {k: v / n_data_sets for k, v in rejections.items()}

# the same comparison when model A really is better (its features are stronger)
rng_alt = np.random.default_rng(12)
alt_differences = np.array([simulate_differences(rng_alt, strength_a=0.35, strength_b=0.26) for _ in range(n_data_sets)])
alt_rates = {"naive paired t": float(np.mean([naive_paired_t(d).p_value < 0.05 for d in alt_differences])), "corrected resampled t": float(np.mean([corrected_resampled_t(d, n_train_sim, n_test_sim).p_value < 0.05 for d in alt_differences]))}
rates_table = pd.DataFrame([{"test": name, "rejects when the models are equal": rejection_rates[name], "rejects when A is truly better": alt_rates[name]} for name in rejection_rates])
show_table(rates_table, floatfmt=".3f")
print(f"mean true AUC advantage of A in the second setting: {alt_differences.mean():+.4f}")
'''

VARIANCE_CHECK = '''
# Why the naive test fails, measured. The folds are not independent observations: their training sets overlap. Over the simulated data sets (the equal-models setting), three quantities: the variance of the mean difference
# actually seen from one data set to the next; the average of the naive estimate of that variance, s^2 / J, with s^2 the sample variance of the J differences; and the average of the corrected estimate,
# s^2 (1/J + n_test/n_train). The correlation between the differences of two folds of the same data set is measured too.
J = null_differences.shape[1]
true_variance_of_mean = float(np.var(null_differences.mean(axis=1), ddof=1))
s_squared = null_differences.var(axis=1, ddof=1)
naive_estimate = float(np.mean(s_squared / J))
corrected_estimate = float(np.mean(s_squared * (1 / J + n_test_sim / n_train_sim)))
correlations = np.corrcoef(null_differences.T)
mean_correlation = float(correlations[np.triu_indices(J, 1)].mean())
rho_expected = n_test_sim / (n_test_sim + n_train_sim)
variance_rows = [{"quantity": "variance of the mean difference, seen over the data sets", "value": true_variance_of_mean},
                 {"quantity": "naive estimate of it, s^2 / J", "value": naive_estimate},
                 {"quantity": "corrected estimate, s^2 (1/J + n_test/n_train)", "value": corrected_estimate}]
show_table(pd.DataFrame(variance_rows), floatfmt=".2e")
print(f"average correlation between the differences of two folds: {mean_correlation:.3f}; n_test/(n_test+n_train) = {rho_expected:.3f}")
naive_ratio, corrected_ratio = naive_estimate / true_variance_of_mean, corrected_estimate / true_variance_of_mean
'''

HOLM = '''
# A family of comparisons, from the project's own logs: the stacked comparison of every addition tested on CYP3A4 (the default stack plus the addition against the default stack, section 6.7). The gate asks for
# p < 0.05 at both confidence cut-offs, so the p-value of an addition is the larger of the two. Holm's step-down adjustment, written out and checked against the library.
import re

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

LOG_DIR = ROOT / "data/structure_track/outputs/cyp3a4/logs"
stacked_line = re.compile(r"z0=(1\\.0|1\\.96)\\s+stacked base\\+added vs base: AUC ([+-]\\d\\.\\d+) \\(p=([\\d.]+)")
family = []
for path in sorted(LOG_DIR.glob("*CYP3A4.log")):
    found = {z0: (float(gain), float(p)) for z0, gain, p in stacked_line.findall(path.read_text(errors="ignore"))}
    if "1.0" in found and "1.96" in found:
        family.append({"addition": path.stem.removesuffix("_CYP3A4").removeprefix("aux_"), "gain at z0 1.0": found["1.0"][0], "gain at z0 1.96": found["1.96"][0], "p (larger of the two)": max(found["1.0"][1], found["1.96"][1])})
family = pd.DataFrame(family).sort_values("p (larger of the two)").reset_index(drop=True)

def holm(p_values):
    """Holm's adjusted p-values: the smallest is multiplied by m, the next by m - 1, and so on, with the running maximum taken so that the order of the adjusted values never reverses."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted

family["Holm-adjusted p"] = holm(family["p (larger of the two)"].to_numpy())
library_holm = multipletests(family["p (larger of the two)"].to_numpy(), method="holm")[1]
holm_agrees = bool(np.allclose(family["Holm-adjusted p"], library_holm))
show_table(family.head(8), floatfmt=".3f")
n_family = len(family)
smallest_raw, smallest_adjusted = float(family["p (larger of the two)"].min()), float(family["Holm-adjusted p"].min())
n_raw_significant = int((family["p (larger of the two)"] < 0.05).sum())
n_adjusted_significant = int((family["Holm-adjusted p"] < 0.05).sum())
print(f"{n_family} additions; raw p below 0.05: {n_raw_significant}; Holm-adjusted p below 0.05: {n_adjusted_significant}; Holm by hand equals the library: {holm_agrees}")
'''

CELLS = [
    md("""
## 6.4 Making validation resemble the test: similarity, prevalence and confident labels

> **Field:** Statistics · Cheminformatics

*Where this section sits:* the nested construction of 6.3 keeps account of *whether* a choice sees its test rows. It says nothing about *which* rows are the test. The score of the challenge is computed on 750 blinded compounds that are not a random sample of the training file: section 2.7 showed how they differ, and section 2.10 how many of them are positives. This section turns those differences into a **yardstick**, a way of weighting the outer-validation rows so that the score measures what the challenge will measure. Every outer score of Parts VII to XI is computed with it.

**Why the validation rows are weighted.** Suppose a model is better on compounds close to the training set and worse on compounds far from it, as read-across models are (section 7.3). A random validation fold has some mixture of close and far compounds, the blinded test set has another, and the score of the model on the fold is the wrong average for the test. The remedy is **importance weighting**: give every validation compound a weight $w$ so that the weighted validation set has the same mixture as the test set. For a single property that is split into bins, with $p_b$ the share of the validation compounds in bin $b$ and $q_b$ the share of the test compounds in it, the weight of a compound in bin $b$ is

$$w(b) \\;=\\; \\frac{q_b}{p_b},$$

the density ratio of section 2.7: a bin that is over-represented in the validation set is turned down, one that is under-represented is turned up. The property the project uses is the one that matters most for a model that reads across analogues: the highest Tanimoto similarity (section 3.2) of a compound to the compounds the model was fitted on. It is a single number per compound, so the weights are well conditioned, unlike weights that try to match the whole fingerprint (a classifier trained to tell the training compounds from the test compounds does so almost perfectly, and the weights it implies put nearly all the weight on a handful of compounds; `validation_matching` records the figures it found). The first cell computes the profile for the first outer fold: the nearest-training similarity of the
validation compounds, against that of the blinded test compounds measured against the same training rows (against a random 80 percent of them, since a validation compound's training set holds about that share of the pool).
"""),
    code(SIMILARITY),
    md("""
For this fold the validation compounds have a median nearest-training similarity of {{=median_validation:.3f}} and the test compounds of {{=median_test:.3f}}; {{=share_validation_close:.0%}} of the validation compounds and {{=share_test_close:.0%}} of the test compounds have a nearest neighbour at 0.5 or more. The profiles differ in shape more than in centre: the test compounds crowd the band from 0.3 to 0.4 (the table: {{=bin_rows[1]['share of test compounds']:.0%}} against {{=bin_rows[1]['share of validation compounds']:.0%}}) and are thin in the band from 0.4 to 0.5
({{=bin_rows[2]['share of test compounds']:.0%}} against {{=bin_rows[2]['share of validation compounds']:.0%}}), and the weights, between 0 and about {{=max(r['weight'] for r in bin_rows):.1f}}, undo that. Weighting costs information: a weighted sample is worth fewer independent compounds than it contains. Kish's **effective sample size**, $n_{\\text{eff}} = (\\sum_i w_i)^2 / \\sum_i w_i^2$ with $w_i$ the weight of the $i$-th compound (section 2.7), is {{=kish_ess:.0f}} for the {{=n_validation_fold}} validation compounds of this fold, a loss the project accepts because the alternative is to measure the wrong thing precisely.
(The reference of this fold is the pool of {{=len(plan.pool):,}} compounds with both arms measured; against the whole training file, which the test compounds were designed as analogues of, the picture is different, and the module `validation_matching` states the numbers it found there.)

**The weighted metrics.** With a weight $w_i$ for each compound, a metric counts each compound in proportion to it. The weighted AUC is the probability that a positive is ranked above a negative with each pair counted in proportion to the product of the two weights, ties half,

$$\\mathrm{AUC}_w \\;=\\; \\frac{\\sum_{i \\in +}\\sum_{j \\in -} w_i\\, w_j \\left( \\mathbb{1}[s_i > s_j] + \\tfrac12\\, \\mathbb{1}[s_i = s_j] \\right)}{\\left(\\sum_{i \\in +} w_i\\right)\\left(\\sum_{j \\in -} w_j\\right)},$$

where $s_i$ is the score of compound $i$, $j$ runs over the negatives and $i$ over the positives, and $+$ and $-$ stand for the two classes. The weighted MCC is the formula of section 4.10 with the four counts of the confusion table replaced by sums of weights. The next cell writes both out for the first fold, for the best single member of the stack, and checks them against the library.
"""),
    code(WEIGHTED_AUC),
    md("""
The weighted AUC by hand is {{=auc_by_hand:.5f}} and the library's is {{=auc_library:.5f}}; the weighted MCC by the counts is {{=mcc_by_hand:.5f}} and the project's function gives {{=mcc_library:.5f}}. The two pairs agree to every printed digit, so nothing in the weighting is a black box.

**Two more corrections, and the ladder.** The similarity weights are the first of three corrections. The second is the **positive rate**. The scored positive rate is known approximately (section 2.10: about 0.31 for CYP3A4) and is not the rate in the training file, so the classes are re-weighted: positives by $\\pi_{\\text{target}}/\\pi_{\\text{observed}}$ and negatives by $(1-\\pi_{\\text{target}})/(1-\\pi_{\\text{observed}})$, with $\\pi$ the positive rate. The AUC does not depend on the positive rate (section 4.10), so this correction cannot change it and matters only for metrics that do, the MCC and the precision. The third is the **confident labels**: the challenge scores only the compounds whose label can be assigned with confidence (section 2.5), whose exact rule is unknown, so the project
keeps the compounds with $|z| \\ge z_0$, where $z = (s - \\log_{10} 2)/\\mathrm{sd}(s)$ is the distance of the shift from the labelling threshold in units of its measurement noise, and repeats every analysis at two cut-offs, $z_0 = 1.0$ and $z_0 = 1.96$. The combined weight of a compound is the product of its three factors. The cell builds the yardstick one correction at a time for the best single member and shows what each does to the measured AUC and to the MCC of calling the top 31 percent positive.
"""),
    code(LADDER),
    md("""
The table is the lesson of the section. The similarity correction moves the AUC from {{=ladder_values['no correction']:.4f}} to {{=ladder_values['similar to the test']:.4f}}, almost nothing for this member and this pool, and the positive rate leaves it exactly where it was ({{=ladder_values['and the scored positive rate']:.4f}}), as it must, while it moves the MCC from {{=ladder_mcc_values['similar to the test']:.3f}} to {{=ladder_mcc_values['and the scored positive rate']:.3f}}. The confident labels change the measured level a great deal: {{=ladder_values['and confident labels, z0 = 1.0']:.4f}} at $z_0 = 1.0$, where {{=share_kept_1:.0%}} of the compounds are kept, and
{{=ladder_values['and confident labels, z0 = 1.96']:.4f}} at $z_0 = 1.96$, where {{=share_kept_196:.0%}} are. The model did not improve; the yardstick became cleaner, because the compounds whose label is a coin flip are no longer counted against it. That is the reason the same analysis is repeated at both cut-offs, and the reason an AUC is meaningful only together with the yardstick that produced it.

What the yardstick does *not* do is as important. It cannot turn a cross-validation number into a forecast of the leaderboard: even with all three corrections the official baseline's cross-validated MCC stays below its score on the leaderboard (the project measured the gap on the baseline, `validation_matching` records it), because the test set differs from any training fold in ways that no one-dimensional weight captures. So an absolute number from this yardstick is never quoted as what a submission will score; what is meaningful is the *ordering* of candidates under the same conditions, and the size of the difference between them with its uncertainty, which is the subject of the next section.

*On what actually ran.* `similarity_weights` and `SimilarityMatcher` (in `cypmode/challenge/validation_matching.py`) bin the nearest-training similarity (the bins `DEFAULT_BINS`, eight of them, from 0 to 1) and give each validation compound the ratio of the test share to the validation share of its bin, a bin that the validation rows never populate carrying no weight; `prior_weights` and `weighted_mcc` do the class correction and the weighted metric; `confident_weights` in `label_confidence.py` multiplies a weight function by the indicator that $|z| \\ge z_0$. `nested_evaluate` applies the product to the outer-validation rows and, for the selectors' own inner scores, to the inner rows. No label and no hand-set number enters the similarity weights; the target positive rate comes from the leaderboard estimate of section 2.10.
"""),
    figure("ml_evaluation_weights"),
    md("""
## 6.5 Comparing two models: why folds are correlated, and the corrected resampled t

> **Field:** Statistics

*Where this section sits:* 6.3 gave a score for one procedure and 6.4 made it fit the challenge. The questions of the project are comparisons: does adding a block of features help, does one model beat another. A comparison needs more than two scores, it needs a statement of how far their difference can be trusted, and the obvious tools give a wrong answer. This section shows the failure, derives the correction the project uses, and applies the correction's close relative, **Holm's adjustment** for running many tests at once, to a family of tests. It is the section that the gate of 6.7 and every verdict of Parts VII, IX and X rest on.

**The setting.** Two procedures A and B are scored on the same $J$ outer folds (folds and repeats together, $J = KR$, section 6.3), giving $J$ pairs of scores and $J$ differences $d_1, \\dots, d_J$ with $d_j = S_{A,j} - S_{B,j}$. Let $\\bar d$ be their mean, $s^2$ their sample variance (section 2.5), and $n_{\\text{train}}$ and $n_{\\text{test}}$ the sizes of the training and the validation part of one fold. The textbook test of "is the mean difference zero?" is the paired t statistic $t = \\bar d / \\sqrt{s^2/J}$, compared with a Student's t distribution with $J-1$ degrees of freedom. It rests on the $d_j$ being independent draws, and they are not. The training sets of two folds share most of their compounds (with five folds, any two share a fraction $(K-2)/(K-1)$ of their rows), so a compound that is hard to predict makes the score of
every fold that contains it in its validation part lower together and, more important, two folds fitted on nearly the same compounds produce nearly the same models, so their errors move together. Positively correlated differences carry less information than $J$ independent ones, and $s^2/J$ is too small an estimate of the variance of $\\bar d$.

**Seeing it fail.** The cell makes the claim testable. It simulates 300 data sets of 600 compounds with 20 features of equal strength; model A uses ten of the features and model B the other ten, so that the two are equal in expected quality and any difference is chance. Each is scored by five-fold cross-validation repeated three times ($J = 15$), and the difference is tested two ways: the naive paired t, and the corrected resampled t of Nadeau and Bengio (2003) that the project uses and that is derived below. A test that works rejects in about 5 percent of these data sets. The simulation is repeated with a second setting in which A truly is better, to see what the correction costs.
"""),
    code(NULL_SIMULATION),
    md("""
When the two models are equal, the naive test declares a difference significant at the 5 percent level in {{=rejection_rates['naive paired t']:.0%}} of the data sets, {{=rejection_rates['naive paired t'] / 0.05:.0f}} times too often, and the corrected test in {{=rejection_rates['corrected resampled t']:.0%}}. In the second setting, with a true AUC advantage of {{=alt_differences.mean():.3f}} for A, the naive test finds it in {{=alt_rates['naive paired t']:.0%}} of the data sets and the corrected test in {{=alt_rates['corrected resampled t']:.0%}}. The correction has a price: it is conservative, and with this design it misses the real difference in {{=1 - alt_rates['corrected resampled t']:.0%}} of the data sets. The naive test's higher **power** (the chance of finding a difference that is really there) is not an advantage, since it is bought by rejecting equal models a third of the time.
(The price is the subject of section 6.7: a design with more folds buys it back.)

**Deriving the correction.** Let the $J$ differences have a common variance $\\sigma^2$ and a common correlation $\\rho$ between any two of them. The variance of their mean is then $\\sigma^2\\,(1/J + (1 - 1/J)\\,\\rho)$, the first term what independence would give and the second what the correlation adds. The sample variance $s^2$ is computed around the mean of the same $J$ values, and its expected value is $\\sigma^2 (1-\\rho)$, which is smaller than $\\sigma^2$ for positive $\\rho$. To estimate the variance of the mean from $s^2$, divide by $(1-\\rho)$ and multiply by the factor above:

$$\\mathrm{Var}_{\\text{est}}(\\bar d) \\;=\\; s^2\\,\\frac{1/J + (1 - 1/J)\\,\\rho}{1 - \\rho} \\;=\\; s^2\\left( \\frac{1}{J} + \\frac{\\rho}{1-\\rho} \\right).$$

What remains is $\\rho$. Nadeau and Bengio (2003) argued that two resampled scores share, in the sense of correlation, about the fraction of data that is in common between their test parts and the whole, which gives $\\rho \\approx n_{\\text{test}}/(n_{\\text{test}} + n_{\\text{train}})$, and then $\\rho/(1-\\rho) = n_{\\text{test}}/n_{\\text{train}}$. So

$$t_{\\text{corrected}} \\;=\\; \\frac{\\bar d}{\\sqrt{\\left(\\dfrac{1}{J} + \\dfrac{n_{\\text{test}}}{n_{\\text{train}}}\\right) s^2}},$$

compared with a Student's t distribution with $J - 1$ degrees of freedom. The only change from the naive statistic is the term $n_{\\text{test}}/n_{\\text{train}}$ added to $1/J$; for the project's five outer folds it is $1/4$, and the factor in brackets is $1/15 + 1/4 = 0.317$, nearly five times the naive $1/15$. The next cell checks the derivation instead of trusting it, on the simulation of the equal models: the real variance of the mean difference seen from one data set to the next, the naive estimate of it and the corrected estimate, and the correlation between the differences of two folds.
"""),
    code(VARIANCE_CHECK),
    md("""
The variance of the mean difference that the data sets actually show is {{=true_variance_of_mean:.2e}}. The naive estimate is {{=naive_estimate:.2e}}, which is {{=naive_ratio:.2f}} of the truth: it is about {{=1 / naive_ratio:.1f}} times too small, and that is why the naive test rejects so often. The corrected estimate is {{=corrected_estimate:.2e}}, {{=corrected_ratio:.2f}} of the truth, a little conservative, and the correlation between the differences of two folds is {{=mean_correlation:.3f}}, against the {{=rho_expected:.3f}} that the argument predicts. The derivation holds, and the small excess is the conservatism seen in the rates above. The project uses this test for every comparison of two procedures on the same folds (`corrected_resampled_t` in `cypmode/challenge/model_comparison.py`), and keeps the naive one in the module only so that the optimism it adds can be demonstrated, as it was here.

**Many comparisons.** A test with a false-positive rate of 5 percent run on 27 comparisons is expected to flag one or two by chance alone. When a family of comparisons is run and any can be reported, the p-values have to be adjusted for the number of tests. The simplest adjustment, Bonferroni's, multiplies every p-value by the number of tests $m$. **Holm's** step-down adjustment (Holm, 1979) is never less powerful and equally safe: sort the p-values from the smallest, $p_{(1)} \\le \\dots \\le p_{(m)}$, multiply the smallest by $m$, the next by $m-1$, and so on, so that $p_{(k)}$ is multiplied by $m-k+1$, cap each at 1, and then take the running maximum so that the adjusted values never fall as the raw ones rise. The cell writes this out and applies it to a real family: the stacked comparison of every addition that the project tested on CYP3A4, in the logs of its experiments (the default stack with the addition against the default stack, at both cut-offs, the p-value of an addition being the larger of the two, since the gate asks for both).
"""),
    code(HOLM),
    md("""
The logs hold {{=n_family}} such comparisons for CYP3A4 (the physics blocks appear twice, in the five-fold design and in the ten-fold replicate of 6.7). The smallest p-value is {{=smallest_raw:.3f}}. The number of comparisons below 0.05 is {{=n_raw_significant}} before any adjustment and {{=n_adjusted_significant}} after Holm's, which multiplies the smallest p-value by {{=n_family}} and turns it into {{=smallest_adjusted:.2f}}. The hand-written adjustment equals the library's: {{=holm_agrees}}. The gain of the strongest additions is of the order of a few thousandths of AUC, positive in almost every row: an
addition that adds a little almost always adds a little, and the design that scores it cannot tell that little from chance. Section 6.7 asks how small a difference the design could have recognised at all.

*On what actually ran.* `corrected_resampled_t` returns the mean difference, the standard error, t, the two-sided p-value and the number of folds in which A beat B, a count that is a check on the pattern of signs and not an inference. `naive_paired_t` is the uncorrected test, kept for the demonstration. `compare_candidate_sets` runs two sets of candidates through the same plan, folds and selector and tests the difference of the outer scores of the *whole procedures*, so the test is on what a user of the procedure would get. The simulation, the check of the variance and the Holm adjustment of this section are computed in the cells of the notebook; the repository's own Holm adjustment is the one `statsmodels` supplies, used in `stacking_significance.py`.
"""),
]
